# NetTrafficML — Statistical ML for Network Traffic

Portfolio prototype aligned with the ML topics in Ericsson Req ID 791323.

## Demonstrates
- supervised probabilistic classification and regression
- imbalanced classes and SMOTE
- Logistic Regression and Random Forest baselines
- feature preprocessing and PCA dimensionality analysis
- ROC-AUC, PR-AUC, precision, recall, F1
- MAE, RMSE and R² regression evaluation
- leave-one-region-out geographical generalization
- reproducible synthetic multi-region traffic data
- saved models, inference and a Streamlit dashboard

**Important:** the data is synthetic. This project does not claim access to Ericsson or real 5G production traffic.

## Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
streamlit run app.py
```

Inference:
```bash
python src/inference.py --csv data/network_traffic.csv
```

## CV wording after running and verifying
**NetTrafficML — Statistical ML for Network Traffic** — Python, Scikit-learn, Pandas, imbalanced-learn, Streamlit
- Built a reproducible ML pipeline for synthetic multi-region network traffic using supervised classification and regression.
- Addressed class imbalance with SMOTE and evaluated probabilistic congestion classification using ROC-AUC, PR-AUC, precision, recall and F1.
- Evaluated traffic regression with MAE, RMSE and R² and tested geographical generalization by holding out entire regions.
- Added PCA-based dimensionality analysis, model persistence, inference and an interactive dashboard.
