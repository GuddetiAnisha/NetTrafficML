# NetTrafficML — Statistical ML for Network Traffic

It focuses on supervised learning, imbalance handling, feature reduction, probabilistic prediction, regional generalization, storage constraints, and real network-traffic forecasting.

## What it covers

- synthetic 100 ms multi-region traffic observations
- probabilistic congestion classification
- traffic-volume regression
- SMOTE for imbalanced classes
- Logistic Regression and tuned Random Forest
- RandomizedSearchCV hyperparameter tuning
- mutual-information feature selection
- PCA dimensionality analysis
- K-Means traffic-regime clustering
- quantile regression (10th/50th/90th percentiles)
- leave-one-region-out geographical generalization
- CSV vs Parquet I/O/storage benchmarking
- persisted models and batch inference
- Streamlit dashboard
- automated tests
- generated experiment report
- real Abilene/Internet2 one-step-ahead traffic forecasting
- persistence-baseline comparison against Extra Trees and HistGradientBoosting

## Architecture

Synthetic traffic generator -> CSV/Parquet -> preprocessing -> classification/regression -> evaluation/diagnostics -> saved models + dashboard + report.

Real Abilene data -> preparation -> data validation/cleaning -> chronological forecasting split -> persistence baseline + ML models -> verified metrics.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py
python src/train.py
python src/benchmark_io.py
python src/report.py
pytest -q
streamlit run app.py
```

## Inference

```bash
python src/inference.py --input data/network_traffic.parquet
```

The prediction output includes congestion probability, point traffic prediction, and a 10th–90th percentile interval.

## Evaluation outputs

`outputs/metrics.json` includes classification metrics, tuned parameters, feature ranking, PCA results, clustering scores, point-regression metrics, quantile coverage, and regional holdout performance.

`outputs/experiment_report.md` provides a human-readable summary.

`outputs/io_benchmark.json` compares CSV and Parquet storage/read performance and demonstrates chunked CSV processing.

## Real Abilene traffic forecasting

A real-data extension evaluates one-step-ahead network traffic forecasting on the Abilene/Internet2 traffic dataset.

The preparation and validation workflow identified one extreme corrupted observation in week `X24` (`5.329427e13` total traffic). The cleaning pipeline removes that malformed sample, rebuilds lag/rolling features, creates `target_next_traffic` inside each weekly sequence, and prevents artificial cross-week targets.

After cleaning, the dataset contains **48,323 rows** and **167 columns**. Evaluation uses a chronological 80/20 holdout and compares a persistence baseline with Extra Trees and HistGradientBoosting.

### Verified 5-minute-ahead results

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Persistence Baseline | **41,897,294** | **76,768,189** | **3.366%** | **0.9764** |
| HistGradientBoosting | 67,280,095 | 191,095,658 | 4.099% | 0.8536 |
| ExtraTrees | 77,467,191 | 199,857,639 | 4.773% | 0.8398 |

The persistence baseline is strongest for this very short horizon. HistGradientBoosting is the strongest ML model, but its MAE is about **60.58% worse** than persistence. This indicates strong short-term temporal persistence in Abilene traffic and shows why a simple baseline is essential before claiming ML improvement.

Run the cleaned real-data experiment with:

```bash
python src/clean_abilene_training.py
python src/train_abilene_real.py --data data/abilene_traffic_clean.csv --target target_next_traffic
```

Detailed verified results are stored in:

```text
results/abilene_model_metrics.csv
results/ABILENE_RESULTS.md
```

A useful next step is multi-horizon forecasting at 15, 30, and 60 minutes ahead to test when ML starts to become more competitive than persistence.

## Repository structure

```text
NetTrafficML/
├── app.py
├── requirements.txt
├── src/
│   ├── generate_data.py
│   ├── train.py
│   ├── inference.py
│   ├── benchmark_io.py
│   ├── report.py
│   └── clean_abilene_training.py
├── tests/
│   └── test_pipeline.py
├── data/       # generated/prepared locally
├── models/     # generated locally
├── results/    # verified Abilene benchmark outputs
└── outputs/    # generated locally
```

## CV-safe description

**NetTrafficML — Statistical ML for Network Traffic**  
Python, Scikit-learn, Pandas, imbalanced-learn, Streamlit

- Built a reproducible ML pipeline for synthetic multi-region network traffic using supervised probabilistic classification and regression.
- Addressed class imbalance with SMOTE, tuned Random Forest models, and evaluated classification with ROC-AUC, PR-AUC, precision, recall, and F1.
- Added mutual-information feature selection, PCA, clustering, quantile regression, and leave-one-region-out generalization experiments.
- Extended the project with real Abilene/Internet2 traffic forecasting, chronological evaluation, robust data cleaning, and persistence-baseline benchmarking.
- Benchmarked CSV/Parquet storage and implemented saved-model inference, automated tests, and an interactive evaluation dashboard.

## Traffic-aware RL resource optimisation extension

A software-only reinforcement-learning extension has been added to study traffic-driven resource control for containerized/network workloads.

### Scope

- generates a reproducible 24-hour synthetic traffic curve with morning and evening peaks
- models four abstract compute/resource levels
- estimates software-side latency and energy cost for each traffic/resource combination
- includes a fixed-resource baseline
- includes a deterministic rule-based scaling baseline
- trains a lightweight PPO agent using Stable-Baselines3
- penalizes SLA latency violations, overload, excessive energy use and unnecessary resource switching
- compares policies using total estimated energy, mean latency, SLA violations, overload steps, resource changes and cumulative reward
- stores per-step traces and a policy comparison CSV for analysis
- includes automated tests for traffic generation, latency/resource behaviour, energy trade-offs and environment execution

This extension is intentionally **software-only**. It does not manipulate real CPU frequency, DVFS, CPU pinning, production RAN nodes, or Ericsson hardware. Energy values are simulation estimates for controlled experimentation rather than physical measurements.

### Run the RL experiment

```bash
python src/rl_energy_control.py
```

Generated outputs include:

```text
outputs/rl_energy_comparison.csv
outputs/rl_static_trace.csv
outputs/rl_rule_trace.csv
outputs/rl_ppo_trace.csv
models/ppo_traffic_energy.zip
```

### CV-safe extension description

- Extended NetTrafficML with a software-only PPO resource-optimisation experiment that reacts to synthetic daily traffic demand and selects abstract compute levels in real time.
- Compared static, rule-based and PPO control using estimated energy consumption, latency, SLA violations, overload events, resource switching and cumulative reward.
- Implemented the experiment with Gymnasium and Stable-Baselines3 and added reproducible traces and automated tests.

Do not describe the extension as real DVFS, CPU pinning, physical power measurement, or production RAN control.
