# Real Abilene Traffic Forecasting Results

This repository now includes a verified one-step-ahead traffic forecasting experiment using the real Abilene/Internet2 traffic dataset.

## Data preparation

The prepared dataset initially contained 48,348 rows and 167 columns. During validation, one extreme corrupted observation was found in week `X24` at `2004-09-08 16:25:00` with `total_traffic = 5.329427e13`, while the 99.9th percentile was about `4.103442e9`.

The cleaning pipeline therefore:

- removes only extremely large corrupted observations using a conservative threshold of 10x the 99.9th percentile,
- rebuilds lag and rolling features after cleaning,
- rebuilds `target_next_traffic` within each Abilene week using `groupby("week").shift(-1)`,
- removes week-boundary rows so the final sample in one week is never paired with the first sample of the next week,
- verifies target consistency after preprocessing.

After cleaning:

- rows: 48,323
- columns: 167
- minimum traffic: 541,930,978
- median traffic: 1,196,747,220
- maximum traffic: 4,823,911,267
- target consistency errors: 0

## Evaluation setup

- forecasting task: predict `target_next_traffic`
- horizon: one step / 5 minutes ahead
- split: chronological 80/20 holdout
- models: Extra Trees and HistGradientBoosting
- target transform for ML models: `log1p`, inverted with `expm1`
- persistence baseline: predict the next traffic value as the current `total_traffic`
- metrics: MAE, RMSE, MAPE, and R2

## Verified results

| Model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Persistence Baseline | **41,897,294** | **76,768,189** | **3.366%** | **0.9764** |
| HistGradientBoosting | 67,280,095 | 191,095,658 | 4.099% | 0.8536 |
| ExtraTrees | 77,467,191 | 199,857,639 | 4.773% | 0.8398 |

HistGradientBoosting was the strongest machine-learning model, but the persistence baseline was better for this very short one-step forecasting horizon.

Relative to persistence:

- HistGradientBoosting MAE: 60.58% worse
- HistGradientBoosting RMSE: 148.93% worse
- ExtraTrees MAE: 84.90% worse
- ExtraTrees RMSE: 160.34% worse

## Interpretation

The result shows that Abilene traffic has strong short-term temporal persistence. For a 5-minute-ahead forecast, simply using the current traffic level provides a very strong prediction. The more complex ML models still achieved useful predictive performance, but they did not add value over persistence at this horizon.

This is an important result rather than a failed experiment: it demonstrates the value of benchmarking against a simple baseline before claiming an ML improvement.

## Reproduce

Clean the prepared Abilene dataset:

```bash
python src/clean_abilene_training.py
```

Then train and evaluate:

```bash
python src/train_abilene_real.py --data data/abilene_traffic_clean.csv --target target_next_traffic
```

The detailed metrics are stored in:

```text
results/abilene_model_metrics.csv
```

A useful next extension is multi-horizon forecasting at 15, 30, and 60 minutes ahead to study whether ML becomes more competitive as persistence weakens.
