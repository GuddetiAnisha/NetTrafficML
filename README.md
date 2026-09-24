# NetTrafficML — Statistical ML for Network Traffic

A complete portfolio/research prototype inspired by Ericsson Req ID 791323. It focuses on supervised learning, imbalance handling, feature reduction, probabilistic prediction, regional generalization, and storage constraints for network-traffic modeling.

> **Data note:** all traffic is synthetic and generated locally. This repository does not contain Ericsson data or claim access to real 5G production traces.

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

## Architecture

Synthetic traffic generator -> CSV/Parquet -> preprocessing -> classification/regression -> evaluation/diagnostics -> saved models + dashboard + report.

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
│   └── report.py
├── tests/
│   └── test_pipeline.py
├── data/       # generated locally
├── models/     # generated locally
└── outputs/    # generated locally
```

## CV-safe description

**NetTrafficML — Statistical ML for Network Traffic**  
Python, Scikit-learn, Pandas, imbalanced-learn, Streamlit

- Built a reproducible ML pipeline for synthetic multi-region network traffic using supervised probabilistic classification and regression.
- Addressed class imbalance with SMOTE, tuned Random Forest models, and evaluated classification with ROC-AUC, PR-AUC, precision, recall, and F1.
- Added mutual-information feature selection, PCA, clustering, quantile regression, and leave-one-region-out generalization experiments.
- Benchmarked CSV/Parquet storage and implemented saved-model inference, automated tests, and an interactive evaluation dashboard.

Do not quote numerical model results in a CV until the project has been run and the metrics have been verified.
