# Execution-Aware Limit Order Book Alpha Research — Stage 1

This repository studies whether short-horizon limit-order-book signals remain profitable after increasingly realistic execution assumptions are applied to LOBSTER data.

## Research question

The project starts with a standard HFT-style prediction problem: use five-level order-book state and message-flow features to predict future executable long and short returns. The key lesson from Stage 1 is that **statistical predictability is not the same as tradable profitability**. The research methodology is changed whenever diagnostics show that an assumption is too optimistic.

## Data

Stage 1 uses the LOBSTER AAPL sample for 2012-06-21. Order-book snapshots are aligned event-by-event with the corresponding message file. The notebook uses the first five displayed levels.

## Main features

The feature set includes:

- bid-ask spread and microprice displacement;
- depth imbalance at levels 1, 3, and 5;
- weighted near-touch imbalance;
- log displayed depth and price gaps;
- short-horizon midprice returns and realized volatility;
- trade, submission, cancellation, and net message-flow imbalance;
- execution/cancellation fractions and event rate;
- normalized best-level order-flow imbalance (OFI).

For example, depth imbalance at depth `d` is

```math
I_t(d)=\frac{Q_t^b(d)-Q_t^a(d)}{Q_t^b(d)+Q_t^a(d)}.
```

## Execution model

Signals are generated at event `t`, with a latency of 10 events. Market orders walk through the five displayed book levels to compute VWAP. For a market buy of `q` shares,

```math
P_t^{buy}(q)=\frac{\sum_{\ell=1}^5 q_\ell A_{t,\ell}}{\sum_{\ell=1}^5 q_\ell}.
```

The initial research target uses aggressive entry and aggressive exit, including spread, depth, and one additional tick of slippage per side.

## How the methodology changed

### 1. H=100 regression baseline

Ridge, Histogram Gradient Boosting, and Random Forest regressors were trained separately for long and short executable returns.

At `H=100`, validation mean executable returns were approximately:

- long: **-2.79 bps**;
- short: **-2.72 bps**.

Directional accuracy was around **97%**, but this was misleading because almost every executable target was negative. The model was largely correct about the dominant negative sign while finding no positive validation edge.

### 2. Horizon diagnostic

Horizons from 10 to 1000 events were compared. The first horizon where the row-wise oracle best executable return became positive on average was `H=1000`:

- mean absolute mid move: **3.185 bps**;
- oracle best executable return: **+0.446 bps**.

This motivated moving execution research to `H=1000`.

### 3. Cost ablation

At `H=1000`, the row-wise oracle mean changed approximately as follows:

| Execution assumption | Oracle best mean |
|---|---:|
| Midpoint / zero cost | 3.185 bps |
| Best bid-ask / spread only | 0.861 bps |
| Depth VWAP / no extra slippage | 0.790 bps |
| Depth VWAP / +1 tick | 0.446 bps |

The dominant cost was **crossing the spread**, not displayed-depth impact.

### 4. Passive-entry diagnostic

An optimistic passive fill was first defined as any visible execution at the quoted price within 100 events.

- buy fill rate: **32.98%**;
- sell fill rate: **33.06%**.

Passive entry improved returns but did not make them positive on average.

### 5. Queue-aware passive fills

The optimistic fill assumption was replaced by a conservative queue approximation. A new order is assumed to join behind the displayed best-price queue and fills only after cumulative visible executions at that price consume

```math
Q_t^{ahead}+q_t.
```

Fill rates fell sharply:

- buy: **6.61%**;
- sell: **5.87%**.

Queue-aware passive-entry/aggressive-exit mean returns were still negative:

- long: **-1.549 bps**;
- short: **-1.556 bps**.

### 6. Adverse selection

After queue-aware fills, the midpoint systematically moved against the position. At 100 events after fill, signed midpoint movement averaged approximately:

- buy: **-0.532 bps**;
- sell: **-0.633 bps**.

This shows that fill probability is endogenous: a passive order often fills because aggressive order flow is already moving against it.

### 7. Queue-aware trade-state target

The final target is based on the actual intended mechanism: **queue-aware passive entry + aggressive exit**.

```math
Y_t\in\{-1,0,+1\}
```

where `+1` is a profitable queue-aware long, `-1` a profitable queue-aware short, and `0` no profitable executable trade.

Validation class distribution:

- SHORT: **2.21%**;
- NO TRADE: **95.47%**;
- LONG: **2.32%**.

The profitable classes had mean realized returns of about **2.40 bps long** and **2.14 bps short**.

## Classifier results

Three queue-aware classifiers were compared.

| Model | Accuracy | Balanced accuracy | Selected threshold | Validation PnL |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.9547 | 0.3342 | 0.30 | +$5.50 (1 trade) |
| Logistic Regression | 0.3517 | 0.4599 | 0.70 | -$5.15 (9 trades) |
| Random Forest | 0.9285 | 0.3554 | 0.50 | -$5.84 (7 trades) |

The high raw accuracy of HistGradientBoosting should **not** be interpreted as strong predictive performance because the NO-TRADE base rate itself is about 95.47%. Balanced accuracy is much more informative here.

## Final untouched test

The validation-selected HistGradientBoosting model with threshold `0.30` was frozen before test evaluation.

- test accuracy: **0.9091**;
- test balanced accuracy: **0.3342**;
- filled trades: **1**;
- test PnL: **-$1.21**;
- realized return: **-1.2121 bps**.

Stage 1 therefore does **not** claim a profitable strategy. Its main result is a realistic research pipeline that explains why apparent alpha disappears after execution and queue effects.

## Statistical interpretation

The main statistical issues are:

- severe final class imbalance;
- high raw accuracy caused by the dominant NO-TRADE class;
- execution costs shifting the whole return distribution below zero even when ranking information exists;
- endogenous fills and adverse selection;
- threshold overfitting when only a few trades remain;
- strong serial dependence between adjacent event-level observations;
- very small effective sample size after non-overlap and queue-aware fill constraints.

## Production handoff

The notebook serializes a production bundle containing:

- the frozen classifier;
- model name;
- feature schema;
- probability threshold;
- class ordering;
- configuration;
- fill horizon.

The bundle is saved as:

```text
artifacts/stage1_execution_aware/production/queue_aware_alpha_model.joblib
```

This project stops before live order routing. The next production step would be a deterministic streaming feature pipeline, schema/version checks, model-health monitoring, shadow replay, latency monitoring, and explicit risk controls.

## Limitations

- one AAPL trading day only;
- simplified queue model;
- no maker rebates / exchange fee schedule;
- fixed event-count latency rather than measured wall-clock latency;
- only five displayed levels;
- fixed $10,000 notional;
- no inventory or market-impact optimization;
- only one filled trade in the final test;
- no probability-calibration study;
- no multi-day walk-forward validation.

## Next research stage

The next stage should preserve the same queue-aware evaluation and test whether stronger alpha features improve out-of-sample PnL. Natural extensions include:

- multi-day walk-forward testing;
- order-ID-aware queue reconstruction;
- fill-probability and adverse-selection modeling;
- fair-price / martingale residual features;
- mean-reverting stochastic alpha models;
- point-process or Hawkes models for submissions, cancellations, executions, and queue depletion;
- calibrated rare-event classification.

The acceptance criterion should remain **incremental out-of-sample queue-aware PnL**, not model complexity alone.

## Files

- `exploration_stage1_clean.ipynb` — research notebook
- `stage1_methodology.tex` — full mathematical methodology report
- `stage1_methodology.pdf` — compiled report
