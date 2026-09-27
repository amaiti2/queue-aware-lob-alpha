#==================================================================================================================================
#this cell is for importation
#==================================================================================================================================
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import zip_longest
from pathlib import Path

import argparse
import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



#==================================================================================================================================
## 2. Configuration

#`Config` centralizes the book depth, latency, holding horizon, transaction-cost assumptions, notional size, flow windows, and validation constraints. The dataclass is frozen so a particular experiment cannot be mutated accidentally after creation.
#==================================================================================================================================
@dataclass(frozen=True)
class Config:
    level: int = 5
    event_stride: int = 1
    latency_rows: int = 10
    horizon_rows: int = 100
    commission_bps_per_side: float = 0.00
    slippage_ticks_per_side: float = 1.0
    tick_size_dollars: float = 0.01
    notional_dollars: float = 10_000.0
    minimum_validation_trades: int = 5
    flow_windows: tuple[int, ...] = (20, 100, 500)
    random_state: int = 42
    chunk_size: int = 500_000

#==================================================================================================================================
## 3. LOBSTER schema helpers

#Create the expected order-book column names for the first `level` price levels. Each level contains ask price/size and bid price/size.
#==================================================================================================================================

def orderbook_columns(level: int) -> list[str]:
    columns=[]
    for level in range(1, level+1):
        columns.extend([f"ask_price_{level}", f"ask_size_{level}", f"bid_price_{level}", f"bid_size_{level}"])
    return columns

#==================================================================================================================================
## 4. Load aligned order-book and message data
##Read the LOBSTER order-book and message CSV files in matching chunks, verify row alignment, apply event_stride, and merge each order-book snapshot with its message metadata.
#==================================================================================================================================
def read_lobster(orederbook_path: Path, message_path: Path, config: Config) -> pd.DataFrame:
    columns= orderbook_columns(config.level)
    message_columns= ["seconds", "event_type", "order_id", "message_size", "message_price", "direction" ]
    parts=[]
    row_offset= 0
    orderbook_reader = pd.read_csv(orderbook_path, header=None, usecols=range(4 * config.level),
        chunksize=config.chunk_size,)

    message_reader = pd.read_csv(message_path, header=None, usecols=range(6),chunksize=config.chunk_size,)

    for book_chunk, message_chunk in zip_longest(orderbook_reader,message_reader,):
        if book_chunk is None or message_chunk is None:
            raise ValueError("The order-book and message files have different row counts.")

        if len(book_chunk) != len(message_chunk):
            raise ValueError("The order-book and message chunks are not aligned.")

        book_chunk.columns = columns
        message_chunk.columns = message_columns

        global_rows = row_offset + np.arange(len(book_chunk))
        keep = global_rows % config.event_stride == 0
        selected = np.flatnonzero(keep)

        sampled_book = book_chunk.iloc[selected].copy()
        sampled_message = message_chunk.iloc[selected]

        sampled_book.insert(0, "event_number", global_rows[selected])
        sampled_book.insert(1, "seconds", sampled_message["seconds"].to_numpy())
        for column in message_columns[1:]:
            sampled_book[column] = (sampled_message[column].to_numpy())

        parts.append(sampled_book)

        row_offset += len(book_chunk)

    if not parts:
        raise ValueError("No data were read.")
    data = pd.concat(parts, ignore_index=True)

    logging.info(
        "Loaded %d sampled snapshots from %d original events.",
        len(data),
        row_offset,
    )

    return data


#==================================================================================================================================
#5. Depth-aware market-order execution
#depth_vwap walks the displayed book from the best price outward until the requested share quantity is filled. It returns the volume-weighted execution price and can apply an extra slippage amount. This is used for conservative taker execution.
#==================================================================================================================================
def depth_vwap(
    price_frame: pd.DataFrame,
    size_frame: pd.DataFrame,
    shares: pd.Series,
    is_buy: bool,
    extra_slippage_dollars: float,
) -> pd.Series:
    """
    Execute a fixed number of shares against displayed book depth.

    For a buy, consume ask levels from best to worst.
    For a sell, consume bid levels from best to worst.

    Returns NaN if the displayed levels cannot fill the order.
    """

    prices = price_frame.to_numpy(dtype="float64")
    sizes = size_frame.to_numpy(dtype="float64")
    requested = shares.to_numpy(dtype="float64")

    remaining = requested.copy()
    filled = np.zeros(len(price_frame), dtype="float64")
    dollar_value = np.zeros(len(price_frame), dtype="float64")

    for level in range(prices.shape[1]):
        available = np.maximum(sizes[:, level], 0.0)
        quantity = np.minimum(available, remaining)

        dollar_value += quantity * prices[:, level]
        filled += quantity
        remaining -= quantity

    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = dollar_value / filled

    completely_filled = (
        np.isfinite(requested)
        & (requested > 0)
        & (remaining <= 1e-9)
        & (filled > 0)
    )

    vwap[~completely_filled] = np.nan

    if is_buy:
        vwap = vwap + extra_slippage_dollars
    else:
        vwap = vwap - extra_slippage_dollars

    return pd.Series(vwap, index=price_frame.index)

#==================================================================================================================================
#6. Build the research frame
# This is the main feature/target construction block. It creates spread, microprice, depth imbalance, order-flow, volatility, event-rate, and OFI features. It also creates four execution-cost scenarios and the original long/short executable-return targets.
# The passive-entry diagnostic inside this function is intentionally simple: after the configured latency, it posts at the contemporaneous best bid/ask and records the first visible execution at that same price within 100 events. NaN is preserved during the search so the first fill time can be recorded correctly; only afterward are unfilled rows encoded as 101. This is an optimistic price-level fill proxy, not yet a queue-position model.
#==================================================================================================================================

def build_research_frame(raw: pd.DataFrame,config: Config,) -> tuple[pd.DataFrame, list[str]]:
    data = raw.copy()
    price_columns = [column for column in data.columns if column.startswith("ask_price_") or column.startswith("bid_price") ]

    # LOBSTER equity prices are stored in units of 1/10,000 dollars.
    for column in price_columns:
        data[column] = data[column].astype("float64") / 10_000.0

    data["message_price"] = (data["message_price"].astype("float64") / 10_000.0)
    
    # Remove locked, crossed, missing, or sentinel-value snapshots.
    valid = ((data["bid_price_1"] > 0) & (data["ask_price_1"] > data["bid_price_1"])
        & (data["ask_price_1"] < 900_000))

    invalid_fraction = 1.0 - valid.mean()

    if invalid_fraction > 0.01:
        logging.warning( "%.3f%% of snapshots failed book validation.",100 * invalid_fraction,)

    data = data.loc[valid].reset_index(drop=True)

    mid = (data["ask_price_1"] + data["bid_price_1"]) / 2.0

    microprice = (data["ask_price_1"] * data["bid_size_1"]+ data["bid_price_1"] * data["ask_size_1"]) / (data["bid_size_1"] + data["ask_size_1"]).replace(0, np.nan)

    features = pd.DataFrame(index=data.index)

    features["spread_bps"] = (10_000 * (data["ask_price_1"] - data["bid_price_1"]) / mid)

    features["microprice_deviation_bps"] = (10_000 * (microprice - mid) / mid)

    # Cumulative imbalance at multiple depths.
    depths = sorted(set([1, min(3, config.level), config.level]))

    for depth in depths:
        bid_depth = sum(data[f"bid_size_{level}"]for level in range(1, depth + 1))
        ask_depth = sum(data[f"ask_size_{level}"]for level in range(1, depth + 1))

        denominator = (bid_depth + ask_depth).replace(0, np.nan)

        features[f"imbalance_{depth}"] = (bid_depth - ask_depth) / denominator

        features[f"log_bid_depth_{depth}"] = np.log1p(bid_depth)
        features[f"log_ask_depth_{depth}"] = np.log1p(ask_depth)

    # Give more importance to queues near the touch.
    weights = {level: 1.0 / level for level in range(1, config.level + 1)}

    weighted_bid = sum(weights[level] * data[f"bid_size_{level}"]
        for level in range(1, config.level + 1))

    weighted_ask = sum(weights[level] * data[f"ask_size_{level}"]
        for level in range(1, config.level + 1))

    features["weighted_imbalance"] = (weighted_bid - weighted_ask)/ (weighted_bid + weighted_ask).replace(0, np.nan)

    # Price distance of each book level from the midpoint.
    for level in range(1, config.level + 1):
        features[f"ask_gap_{level}_bps"] = (10_000 * (data[f"ask_price_{level}"] - mid) / mid)

        features[f"bid_gap_{level}_bps"] = (10_000 * (mid - data[f"bid_price_{level}"]) / mid)

    log_mid = np.log(mid)
    one_event_return = 10_000 * log_mid.diff()

    features["mid_return_1"] = one_event_return
    features["mid_return_5"] = 10_000 * log_mid.diff(5)
    features["mid_return_20"] = 10_000 * log_mid.diff(20)

    features["realized_volatility_20"] = (one_event_return.rolling(20).std())
    
    features["realized_volatility_100"] = (one_event_return.rolling(100).std())

    features["imbalance_change_1"] = (features["imbalance_1"].diff())

    features["imbalance_change_10"] = (features["imbalance_1"].diff(10))

    features["imbalance_change_10"] = (features["imbalance_1"].diff(10))

    event_time_difference_ms = (1_000 * data["seconds"].diff().clip(lower=0, upper=5))

    features["log_event_time_ms"] = np.log1p(event_time_difference_ms)

    # Message_book flow

    event_type = data["event_type"].astype(int)
    direction = data["direction"].astype(float).fillna(0.0)
    message_size = (data["message_size"].astype(float).fillna(0.0) )

    is_submission = event_type.eq(1)
    is_cancellation = event_type.isin([2, 3])
    is_execution = event_type.isin([4, 5])

    signed_trade_flow = pd.Series(np.where(is_execution,-direction * message_size,0.0,),index=data.index,)

    absolute_trade_flow = pd.Series(np.where(is_execution, message_size, 0.0,), index=data.index,)

    signed_submission_flow = pd.Series(np.where(is_submission, direction * message_size, 0.0,),index=data.index,)

    absolute_submission_flow = pd.Series(np.where(is_submission,message_size,0.0,),index=data.index,)

    signed_cancellation_pressure = pd.Series(np.where(is_cancellation,-direction * message_size,0.0,),index=data.index,)

    absolute_cancellation_flow = pd.Series(np.where(is_cancellation,message_size,0.0,),index=data.index,)

    signed_total_pressure = (signed_trade_flow + signed_submission_flow + signed_cancellation_pressure)

    absolute_total_flow = (absolute_trade_flow + absolute_submission_flow + absolute_cancellation_flow)

    def rolling_normalized_flow(
        signed_flow: pd.Series,
        absolute_flow: pd.Series,
        window: int,
    ) -> pd.Series:
        numerator = signed_flow.rolling(window, min_periods=window,).sum()

        denominator = absolute_flow.rolling(window, min_periods=window,).sum()

        result = numerator / denominator.where(denominator > 0)

        # A complete window with no corresponding messages
        # has neutral imbalance rather than missing data.
        result = result.mask(denominator.eq(0),0.0,)

        return result

    for window in config.flow_windows:
        features[f"trade_imbalance_{window}"] = (
            rolling_normalized_flow(signed_trade_flow, absolute_trade_flow, window,))

        features[f"submission_imbalance_{window}"] = (
            rolling_normalized_flow(signed_submission_flow, absolute_submission_flow, window,))

        features[f"cancellation_pressure_{window}"] = (
            rolling_normalized_flow(signed_cancellation_pressure, absolute_cancellation_flow, window,))

        features[f"net_message_pressure_{window}"] = (
            rolling_normalized_flow(signed_total_pressure, absolute_total_flow,window,))

        features[f"execution_fraction_{window}"] = (
            is_execution.astype(float).rolling(window,min_periods=window,).mean())

        features[f"cancellation_fraction_{window}"] = (
            is_cancellation.astype(float).rolling(window,min_periods=window,).mean())

        elapsed_seconds = (data["seconds"] - data["seconds"].shift(window)).clip(lower=1e-6)

        features[f"event_rate_{window}"] = (window / elapsed_seconds)
    


    #best-level order flow imbalance
    bid_price = data["bid_price_1"]
    ask_price = data["ask_price_1"]
    bid_size = data["bid_size_1"].astype(float)
    ask_size = data["ask_size_1"].astype(float)

    previous_bid_price = bid_price.shift(1)
    previous_ask_price = ask_price.shift(1)
    previous_bid_size = bid_size.shift(1)
    previous_ask_size = ask_size.shift(1)

    ofi_increment = (
        bid_price.ge(previous_bid_price).astype(float)
        * bid_size
        - bid_price.le(previous_bid_price).astype(float)
        * previous_bid_size
        - ask_price.le(previous_ask_price).astype(float)
        * ask_size
        + ask_price.ge(previous_ask_price).astype(float)
        * previous_ask_size
    )

    top_depth = bid_size + ask_size

    for window in config.flow_windows:
        rolling_ofi = ofi_increment.rolling(window, min_periods=window,).sum()

        average_top_depth = top_depth.rolling(window, min_periods=window,).mean()

        features[f"normalized_ofi_{window}"] = (rolling_ofi / average_top_depth.replace(0, np.nan))

    #latency-horizon flow
    latency = config.latency_rows
    horizon = config.horizon_rows
    exit_offset= latency+horizon
    entry_mid = mid.shift(-latency)
    shares = np.floor(
        config.notional_dollars / entry_mid
    ).clip(lower=1)

    extra_slippage = (
        config.slippage_ticks_per_side
        * config.tick_size_dollars
    )

    ask_price_columns = [
        f"ask_price_{level}"
        for level in range(1, config.level + 1)
    ]

    ask_size_columns = [
        f"ask_size_{level}"
        for level in range(1, config.level + 1)
    ]

    bid_price_columns = [
        f"bid_price_{level}"
        for level in range(1, config.level + 1)
    ]

    bid_size_columns = [
        f"bid_size_{level}"
        for level in range(1, config.level + 1)
    ]

    entry_buy_vwap = depth_vwap(
        price_frame=data[ask_price_columns].shift(-latency),
        size_frame=data[ask_size_columns].shift(-latency),
        shares=shares,
        is_buy=True,
        extra_slippage_dollars=extra_slippage,
     )

    entry_sell_vwap = depth_vwap(
        price_frame=data[bid_price_columns].shift(-latency),
        size_frame=data[bid_size_columns].shift(-latency),
        shares=shares,
        is_buy=False,
        extra_slippage_dollars=extra_slippage,
    )

    exit_sell_vwap = depth_vwap(
        price_frame=data[bid_price_columns].shift(-exit_offset),
        size_frame=data[bid_size_columns].shift(-exit_offset),
        shares=shares,
        is_buy=False,
        extra_slippage_dollars=extra_slippage,
    )

    exit_buy_vwap = depth_vwap(
        price_frame=data[ask_price_columns].shift(-exit_offset),
        size_frame=data[ask_size_columns].shift(-exit_offset),
        shares=shares,
        is_buy=True,
        extra_slippage_dollars=extra_slippage,
    )



    # ============================================================
    # PASSIVE ENTRY FILL DIAGNOSTIC
    # ============================================================
    # This is an optimistic price-level fill proxy. We post at the
    # best bid/ask after latency and ask whether a visible execution
    # occurs at that same price within the next `fill_horizon` events.
    # Queue position is not modeled yet.

    fill_horizon = 100
    passive_buy_price = data["bid_price_1"].shift(-latency).to_numpy()
    passive_sell_price = data["ask_price_1"].shift(-latency).to_numpy()

    # NaN means "not filled yet". Do NOT replace NaN before the loop.
    passive_buy_fill_events = np.full(len(data), np.nan, dtype="float64")
    passive_sell_fill_events = np.full(len(data), np.nan, dtype="float64")

    for k in range(1, fill_horizon + 1):
        future_event_type = data["event_type"].shift(-(latency + k)).to_numpy()
        future_direction = data["direction"].shift(-(latency + k)).to_numpy()
        future_trade_price = data["message_price"].shift(-(latency + k)).to_numpy()

        # LOBSTER event type 4 = visible execution.
        is_visible_execution = future_event_type == 4

        buy_fill_now = (
            np.isnan(passive_buy_fill_events)
            & is_visible_execution
            & (future_direction == 1)
            & np.isclose(future_trade_price, passive_buy_price, rtol=0.0, atol=1e-12)
        )
        passive_buy_fill_events[buy_fill_now] = k

        sell_fill_now = (
            np.isnan(passive_sell_fill_events)
            & is_visible_execution
            & (future_direction == -1)
            & np.isclose(future_trade_price, passive_sell_price, rtol=0.0, atol=1e-12)
        )
        passive_sell_fill_events[sell_fill_now] = k

    # Determine fill status BEFORE replacing missing fill times.
    passive_buy_filled = ~np.isnan(passive_buy_fill_events)
    passive_sell_filled = ~np.isnan(passive_sell_fill_events)

    passive_buy_filled = pd.Series(passive_buy_filled.astype(int), index=data.index)
    passive_sell_filled = pd.Series(passive_sell_filled.astype(int), index=data.index)

    passive_buy_fill_events = pd.Series(passive_buy_fill_events, index=data.index)
    passive_sell_fill_events = pd.Series(passive_sell_fill_events, index=data.index)

    # Use fill_horizon + 1 as a sentinel so frame.dropna() keeps unfilled rows.
    passive_buy_fill_events = passive_buy_fill_events.fillna(fill_horizon + 1)
    passive_sell_fill_events = passive_sell_fill_events.fillna(fill_horizon + 1)


    #Queue-aware passive-fill approximation
    #The previous passive-fill test only asked whether our quoted price traded. That is an optimistic upper bound because other orders may already be ahead of us in the queue.
    #Here we assume our new limit order joins the back of the displayed best-price queue. The displayed size at arrival is treated as queue_ahead. We accumulate visible executions at exactly our quoted price and declare a full fill only once those executions consume both the initial queue ahead and our own order quantity.
    #Cancellations are ignored in this first version, making the test conservative. Later, an order-ID-aware model could distinguish cancellations ahead of us from orders added behind us.

    # ============================================================
    # QUEUE-AWARE PASSIVE ENTRY
    # ============================================================
    # Conservative approximation:
    # Full fill occurs only when cumulative visible executions
    # at our price consume:
    # initial queue ahead + our own shares
    # We IGNORE cancellations for now.
    # Therefore this tends to UNDERESTIMATE fill probability.
    # ============================================================

    n_rows = len(data)

    # ------------------------------------------------------------
    # Queue already resting ahead of us when our order arrives
    # ------------------------------------------------------------

    queue_ahead_buy = (data["bid_size_1"].shift(-latency).to_numpy(dtype=float))

    queue_ahead_sell = (data["ask_size_1"].shift(-latency).to_numpy(dtype=float))

    our_shares = shares.to_numpy(dtype=float)

    passive_buy_price_np = np.asarray(passive_buy_price,dtype=float)

    passive_sell_price_np = np.asarray(passive_sell_price,dtype=float)


    # ------------------------------------------------------------
    # Running executed volume at our quoted price
    # ------------------------------------------------------------

    queue_buy_executed_volume = np.zeros(n_rows,dtype=float,)

    queue_sell_executed_volume = np.zeros(n_rows,dtype=float,)


    # First event at which OUR ENTIRE order would be filled
    queue_buy_fill_events = np.full(n_rows,np.nan,)

    queue_sell_fill_events = np.full(n_rows,np.nan,)


# ------------------------------------------------------------
# Search forward through the same passive fill window
# ------------------------------------------------------------

    for k in range(1, fill_horizon + 1):
        future_event_type = (data["event_type"].shift(-(latency + k)).to_numpy())

        future_direction = (data["direction"].shift(-(latency + k)).to_numpy())

        future_trade_price = (data["message_price"].shift(-(latency + k)).to_numpy(dtype=float))

        future_trade_size = (data["message_size"].shift(-(latency + k)).to_numpy(dtype=float))
                

    # LOBSTER event type 4:
    # visible limit-order execution
        visible_execution = (future_event_type == 4)

         # ========================================================
    # PASSIVE BUY
    # ========================================================
    #
    # direction +1 means resting BUY liquidity was executed.
    #
    # Count execution only if it occurs at our exact bid price.
    # ========================================================
        
        buy_execution_at_our_price = (visible_execution & (future_direction == 1)
        & np.isclose(future_trade_price,passive_buy_price_np, equal_nan=False,))

        queue_buy_executed_volume += np.where(buy_execution_at_our_price,
        np.nan_to_num(future_trade_size,nan=0.0,),0.0,)


        buy_required_volume = (queue_ahead_buy+ our_shares)

        buy_fills_now = (np.isnan(queue_buy_fill_events) & np.isfinite(buy_required_volume)
        & (queue_buy_executed_volume>= buy_required_volume))

        queue_buy_fill_events[buy_fills_now] = k


    # ========================================================
    # PASSIVE SELL
    # ========================================================
    #
    # direction -1 means resting SELL liquidity was executed.
    # ========================================================

        sell_execution_at_our_price = (visible_execution & (future_direction == -1)
        & np.isclose(future_trade_price, passive_sell_price_np, equal_nan=False,))
        queue_sell_executed_volume += np.where(sell_execution_at_our_price,
        np.nan_to_num(future_trade_size,nan=0.0,),0.0,)


        sell_required_volume = (queue_ahead_sell+ our_shares)

        sell_fills_now = (np.isnan(queue_sell_fill_events) & np.isfinite(sell_required_volume)
        & (queue_sell_executed_volume >= sell_required_volume))

        queue_sell_fill_events[sell_fills_now] = k


# ------------------------------------------------------------
# Record whether each order fully filled
# ------------------------------------------------------------

    queue_buy_filled = (~np.isnan(queue_buy_fill_events))

    queue_sell_filled = (~np.isnan(queue_sell_fill_events))


# Keep copies with NaN because we need the true fill time
# for the adverse-selection calculation below.
    queue_buy_fill_events_raw = (queue_buy_fill_events.copy())

    queue_sell_fill_events_raw = (queue_sell_fill_events.copy())


     # ============================================================
    # POST-FILL ADVERSE SELECTION
    # ============================================================
    # After a queue-aware fill, measure the signed midpoint move:
    # BUY:
    #     positive = midpoint rises after we buy
    #     negative = midpoint falls after we buy
    # SELL:
    #     positive = midpoint falls after we sell
    #     negative = midpoint rises after we sell
    # Negative values therefore indicate adverse selection.
    # ============================================================

    post_fill_horizons = [10,25,50,100,250,]

    mid_np = mid.to_numpy(dtype=float)

    row_numbers = np.arange(n_rows)


    def post_fill_mid_return(fill_events,side,post_horizon,):
        result = np.full(n_rows,np.nan,)

        filled_rows = np.flatnonzero(np.isfinite(fill_events))

        if len(filled_rows) == 0:
            return result

        fill_k = (fill_events[filled_rows].astype(int))

    # Signal row
    # + latency
    # + events until our order fills
        fill_indices = (filled_rows+ latency + fill_k )

        future_indices = (fill_indices + post_horizon)

        valid_future = (future_indices < n_rows)

        rows_valid = filled_rows[valid_future]

        fill_indices = fill_indices[valid_future]

        future_indices = future_indices[valid_future]


        raw_move = (10_000* (mid_np[future_indices]/ mid_np[fill_indices]- 1.0))


        if side == 1:
            # Long: rising midpoint is favorable
            signed_move = raw_move

        else:
            # Short: falling midpoint is favorable
            signed_move = -raw_move


        result[rows_valid] = signed_move

        return result


    queue_buy_postfill = {}
    queue_sell_postfill = {}


    for h in post_fill_horizons:
        queue_buy_postfill[h] = (post_fill_mid_return(queue_buy_fill_events_raw,side=1,
            post_horizon=h,))

        queue_sell_postfill[h] = (post_fill_mid_return(queue_sell_fill_events_raw, side=-1,
            post_horizon=h,))

    queue_buy_fill_events = pd.Series(queue_buy_fill_events_raw,index=data.index,).fillna(
    fill_horizon + 1)

    queue_sell_fill_events = pd.Series(queue_sell_fill_events_raw,index=data.index,).fillna(
    fill_horizon + 1)


    queue_buy_filled = pd.Series(queue_buy_filled.astype(int),index=data.index,)

    queue_sell_filled = pd.Series(queue_sell_filled.astype(int),index=data.index,)


    queue_ahead_buy = pd.Series(queue_ahead_buy,index=data.index,)

    queue_ahead_sell = pd.Series(queue_ahead_sell,index=data.index,)


    queue_buy_executed_volume = pd.Series(queue_buy_executed_volume,index=data.index,)

    queue_sell_executed_volume = pd.Series(queue_sell_executed_volume,index=data.index,)

    for h in post_fill_horizons:
        queue_buy_postfill[h] = pd.Series(queue_buy_postfill[h],index=data.index,).fillna(0.0)

        queue_sell_postfill[h] = pd.Series(queue_sell_postfill[h],index=data.index,).fillna(0.0)


    # ============================================================
    # COST ABLATION
    # ============================================================

    # ------------------------------------------------------------
    # A. BEST-TOUCH EXECUTION
    # Cross spread, but:
    #   - no depth walking
    #   - no extra slippage
    # ------------------------------------------------------------

    entry_best_ask = data["ask_price_1"].shift(-latency)
    entry_best_bid = data["bid_price_1"].shift(-latency)

    exit_best_bid = data["bid_price_1"].shift(-exit_offset)
    exit_best_ask = data["ask_price_1"].shift(-exit_offset)


    touch_long_return_bps = (10_000* (
        exit_best_bid - entry_best_ask)/ entry_best_ask)

    touch_short_return_bps = (10_000* (
        entry_best_bid - exit_best_ask)/ entry_best_bid)


    # ------------------------------------------------------------
    # B. DEPTH VWAP, BUT ZERO EXTRA SLIPPAGE
    # ------------------------------------------------------------

    entry_buy_vwap_no_slip = depth_vwap(price_frame=data[ask_price_columns].shift(-latency),
    size_frame=data[ask_size_columns].shift(-latency),shares=shares,is_buy=True,
    extra_slippage_dollars=0.0,)


    entry_sell_vwap_no_slip = depth_vwap(price_frame=data[bid_price_columns].shift(-latency),
    size_frame=data[bid_size_columns].shift(-latency),shares=shares,is_buy=False,
    extra_slippage_dollars=0.0,)


    exit_sell_vwap_no_slip = depth_vwap(price_frame=data[bid_price_columns].shift(-exit_offset),
    size_frame=data[bid_size_columns].shift(-exit_offset),shares=shares,is_buy=False,
    extra_slippage_dollars=0.0,)


    exit_buy_vwap_no_slip = depth_vwap(price_frame=data[ask_price_columns].shift(-exit_offset),
    size_frame=data[ask_size_columns].shift(-exit_offset),shares=shares,is_buy=True,
    extra_slippage_dollars=0.0,)


    depth_no_slip_long_bps = (10_000* (exit_sell_vwap_no_slip- entry_buy_vwap_no_slip)/ entry_buy_vwap_no_slip)


    depth_no_slip_short_bps = (10_000* (entry_sell_vwap_no_slip- exit_buy_vwap_no_slip)/ entry_sell_vwap_no_slip)
    
    round_trip_commission_bps = (2 * config.commission_bps_per_side)

    long_net_return_bps = (10_000* (exit_sell_vwap - entry_buy_vwap)/ entry_buy_vwap
    - round_trip_commission_bps)

    short_net_return_bps = (10_000* (entry_sell_vwap - exit_buy_vwap)/ entry_sell_vwap
    - round_trip_commission_bps)

    exit_mid = mid.shift(-exit_offset)

    target_mid_bps = (10_000 * (exit_mid / entry_mid - 1.0))


    execution = pd.DataFrame(
        {
            "event_number": data["event_number"],
            "seconds": data["seconds"],
            "mid": mid,
            "shares": shares,
            "target_mid_bps": target_mid_bps,
            "long_net_return_bps": long_net_return_bps,
            "short_net_return_bps": short_net_return_bps,
            "entry_buy_vwap": entry_buy_vwap,
            "entry_sell_vwap": entry_sell_vwap,
            "exit_sell_vwap": exit_sell_vwap,
            "exit_buy_vwap": exit_buy_vwap,
            "mid_long_return_bps":target_mid_bps,
            "mid_short_return_bps":-target_mid_bps,
            "touch_long_return_bps":touch_long_return_bps,
            "touch_short_return_bps":touch_short_return_bps,
            "depth_no_slip_long_bps":depth_no_slip_long_bps,
            "depth_no_slip_short_bps":depth_no_slip_short_bps,
            "passive_buy_price":passive_buy_price,
            "passive_sell_price":passive_sell_price,
            "passive_buy_filled": passive_buy_filled,
            "passive_sell_filled": passive_sell_filled,
            "passive_buy_fill_events":passive_buy_fill_events,
            "passive_sell_fill_events":passive_sell_fill_events,
            "queue_ahead_buy":queue_ahead_buy,
            "queue_ahead_sell":queue_ahead_sell,
            "queue_buy_filled":queue_buy_filled,
            "queue_sell_filled":queue_sell_filled,
            "queue_buy_fill_events":queue_buy_fill_events,
            "queue_sell_fill_events":queue_sell_fill_events,
            "queue_buy_executed_volume":queue_buy_executed_volume,
            "queue_sell_executed_volume":queue_sell_executed_volume,
            "queue_buy_postfill_10_bps":queue_buy_postfill[10],
            "queue_buy_postfill_25_bps":queue_buy_postfill[25],
            "queue_buy_postfill_50_bps":queue_buy_postfill[50],
            "queue_buy_postfill_100_bps":queue_buy_postfill[100],
            "queue_buy_postfill_250_bps":queue_buy_postfill[250],
            "queue_sell_postfill_10_bps":queue_sell_postfill[10],
            "queue_sell_postfill_25_bps":queue_sell_postfill[25],
            "queue_sell_postfill_50_bps":queue_sell_postfill[50],
            "queue_sell_postfill_100_bps":queue_sell_postfill[100],
            "queue_sell_postfill_250_bps":queue_sell_postfill[250],
        }
    )

    frame = pd.concat(
        [execution, features],
        axis=1,
    )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    frame = frame.reset_index(drop=True)

    feature_names = list(features.columns)

    if len(frame) < 10_000:
        raise ValueError(
            "Too few valid observations for chronological evaluation."
        )

    return frame, feature_names


#==================================================================================================================================
## 8. Purged chronological split

# Split observations chronologically into train/validation/test blocks and purge rows at the train/validation boundaries so future target windows do not overlap an earlier split.

#==================================================================================================================================

def chronological_splits(
    number_of_rows: int,
    purge_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_boundary = int(0.60 * number_of_rows)
    validation_boundary = int(0.80 * number_of_rows)

    train = np.arange(0, train_boundary - purge_rows)

    validation = np.arange(
        train_boundary,
        validation_boundary - purge_rows,
    )

    test = np.arange(validation_boundary, number_of_rows)

    return train, validation, test

#==================================================================================================================================
## 9. Trading simulator and metrics

# Convert model predictions into non-overlapping long/short trades. Realized executable returns already contain spread, displayed-depth impact, commissions, and configured extra slippage.
#==================================================================================================================================

def trading_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "number_of_trades": 0,
            "net_pnl_dollars": 0.0,
            "average_trade_bps": None,
            "win_rate": None,
            "profit_factor": None,
            "maximum_drawdown_dollars": None,
        }

    returns = trades["net_return_bps"]
    pnl = trades["pnl_dollars"]

    positive_pnl = pnl[pnl > 0].sum()
    negative_pnl = -pnl[pnl < 0].sum()

    cumulative_pnl = pnl.cumsum()
    running_maximum = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_maximum

    return {
        "number_of_trades": int(len(trades)),
        "net_pnl_dollars": float(pnl.sum()),
        "average_trade_bps": float(returns.mean()),
        "median_trade_bps": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": (
            float(positive_pnl / negative_pnl)
            if negative_pnl > 0
            else None
        ),
        "maximum_drawdown_dollars": float(drawdown.min()),
    }

#==================================================================================================================================

## 22. Classification candidates

# Compare multinomial logistic regression, histogram gradient boosting, and random forest. Class balancing is used where supported because the three classes are not equally frequent.
#==================================================================================================================================


def make_classifier_candidates(config):

    return {

        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=config.random_state,
            ),
        ),

        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                l2_regularization=1.0,
                random_state=config.random_state,
            )
        ),

        "random_forest": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=100,
                max_features=0.70,
                class_weight="balanced",
                n_jobs=-1,
                random_state=config.random_state,
            )
        ),
    }


#==================================================================================================================================
## 23. Queue-aware non-overlapping classifier backtest

# The classifier predicts one of three economic states:
# - `+1`: profitable queue-aware LONG,
# - `-1`: profitable queue-aware SHORT,
# - `0`: NO TRADE.
# A predicted LONG or SHORT does not automatically become a trade.
# The strategy first submits a passive limit order at the best quote. The order must then satisfy the queue-aware fill condition developed earlier.
# If the order does not fill within the 100-event fill window, no trade occurs and the order is assumed cancelled.
# If it fills, entry occurs passively and exit occurs using the existing aggressive depth-VWAP execution model at the H = 1000 horizon.
# This produces a backtest in which model confidence, fill probability, queue position, execution costs and non-overlapping positions are all respected.

#==================================================================================================================================

def simulate_classifier_trades(
    block,
    probabilities,
    classes,
    probability_threshold,
    config,
    fill_horizon=100,
):

    trades = []

    next_available_event = -np.inf


    class_to_column = {
        int(label): i
        for i, label in enumerate(classes)
    }

    p_short = probabilities[:, class_to_column[-1]]
    p_no_trade = probabilities[:, class_to_column[0]]
    p_long = probabilities[:, class_to_column[1]]


    for row_number, (_, row) in enumerate(
        block.iterrows()
    ):

        event_number = int(
            row["event_number"]
        )

        if event_number < next_available_event:
            continue


        long_probability = float(
            p_long[row_number]
        )

        no_trade_probability = float(
            p_no_trade[row_number]
        )

        short_probability = float(
            p_short[row_number]
        )

        # ====================================================
        # MODEL DECISION
        # ====================================================

        if (
            long_probability >= probability_threshold
            and long_probability > no_trade_probability
            and long_probability >= short_probability
        ):

            side = 1

            selected_probability = (
                long_probability
            )

            filled = (
                row["queue_buy_filled"] == 1
            )

            fill_events = (
                row["queue_buy_fill_events"]
            )

            realized_return = float(
                row[
                    "queue_passive_long_return_bps"
                ]
            )

            entry_price = float(
                row["passive_buy_price"]
            )

            exit_price = float(
                row["exit_sell_vwap"]
            )


        elif (
            short_probability >= probability_threshold
            and short_probability > no_trade_probability
            and short_probability > long_probability
        ):

            side = -1

            selected_probability = (
                short_probability
            )

            filled = (
                row["queue_sell_filled"] == 1
            )

            fill_events = (
                row["queue_sell_fill_events"]
            )

            realized_return = float(
                row[
                    "queue_passive_short_return_bps"
                ]
            )

            entry_price = float(
                row["passive_sell_price"]
            )

            exit_price = float(
                row["exit_buy_vwap"]
            )


        else:
            continue


        # ====================================================
        # ORDER DID NOT FILL
        # ====================================================

        if not filled:

            # Order remains live until the end of the
            # passive fill window, then is cancelled.

            next_available_event = (
                event_number
                + (
                    config.latency_rows
                    + fill_horizon
                )
                * config.event_stride
            )

            continue

        # ====================================================
        # ORDER FILLED
        # ====================================================

        trades.append(
            {
                "signal_event_number":
                    event_number,

                "signal_event_time":
                    row["seconds"],

                "side":
                    side,

                "predicted_probability":
                    selected_probability,

                "prob_long":
                    long_probability,

                "prob_short":
                    short_probability,

                "prob_no_trade":
                    no_trade_probability,

                "fill_events":
                    fill_events,

                "entry_price":
                    entry_price,

                "exit_price":
                    exit_price,

                "realized_net_return_bps":
                    realized_return,
                
                "net_return_bps":
                    realized_return,

                "pnl_dollars":
                    (
                        config.notional_dollars
                        * realized_return
                        / 10_000
                    ),
            }
        )


        # Capital occupied until H=1000 exit
        next_available_event = (
            event_number
            + (
                config.latency_rows
                + config.horizon_rows
            )
            * config.event_stride
        )


    return pd.DataFrame(trades)

        
#==================================================================================================================================
#24. Classifier threshold selection
# Search the high-confidence region where the earlier diagnostics suggested better realized economics. Threshold selection uses validation PnL only.

#==================================================================================================================================

# ============================================================
# PROBABILITY THRESHOLD SELECTION
# VALIDATION DATA ONLY
# ============================================================

def choose_classifier_threshold(
    validation_frame,
    probabilities,
    classes,
    config,
):

    thresholds = np.array([ 0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,])

    rows = []

    for threshold in thresholds:

        trades = simulate_classifier_trades(
            block=validation_frame,
            probabilities=probabilities,
            classes=classes,
            probability_threshold=threshold,
            config=config,
        )

        metrics = trading_metrics(
            trades
        )

        rows.append(
            {
                "probability_threshold":
                    threshold,

                **metrics,
            }
        )


    sweep = pd.DataFrame(rows)

    # Prefer thresholds giving a reasonable
    # number of validation trades
    eligible = sweep[
        sweep["number_of_trades"]
        >= config.minimum_validation_trades
    ]


    if eligible.empty:
        eligible = sweep


    best = eligible.sort_values(
        [
            "net_pnl_dollars",
            "average_trade_bps",
        ],
        ascending=False,
    ).iloc[0]


    return (
        float(
            best["probability_threshold"]
        ),
        sweep,
    )    