"""Train and evaluate classification/regression models for NetTrafficML."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score,
    r2_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "network_traffic.parquet"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

FEATURES = [
    "region", "hour", "day_of_week", "active_users", "packet_rate", "latency_ms",
    "retransmission_rate", "queue_depth", "signal_quality_dbm",
]
CAT = ["region"]
NUM = [c for c in FEATURES if c not in CAT]

def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])

def classification_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "report": classification_report(y_true, pred, output_dict=True, zero_division=0),
    }

def main():
    df = pd.read_parquet(DATA)
    X, y_cls, y_reg = df[FEATURES], df["congested"], df["downlink_mbps"]
    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "positive_rate": float(y_cls.mean()),
            "regions": sorted(df["region"].unique().tolist())
        },
        "classification": {},
        "regression": {},
        "generalization": {},
        "feature_selection": {},
        "dimensionality_reduction": {},
        "clustering": {},
    }

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )

    logistic = ImbPipeline([
        ("prep", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    logistic.fit(Xtr, ytr)
    metrics["classification"]["logistic_smote"] = classification_metrics(
        yte, logistic.predict_proba(Xte)[:, 1]
    )

    rf_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("clf", RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=-1
        )),
    ])
    search = RandomizedSearchCV(
        rf_pipe,
        {
            "clf__n_estimators": [150, 250, 350],
            "clf__max_depth": [None, 8, 14, 20],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.6, 0.9],
        },
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(Xtr, ytr)
    rf = search.best_estimator_
    metrics["classification"]["random_forest_tuned"] = classification_metrics(
        yte, rf.predict_proba(Xte)[:, 1]
    )
    metrics["classification"]["random_forest_tuned"]["best_params"] = search.best_params_

    candidates = {"logistic_smote": logistic, "random_forest_tuned": rf}
    best_name = max(candidates, key=lambda n: metrics["classification"][n]["pr_auc"])
    metrics["best_classifier"] = best_name
    joblib.dump(candidates[best_name], MODELS / "congestion_classifier.joblib")

    Xn = StandardScaler().fit_transform(df[NUM])
    selector = SelectKBest(mutual_info_classif, k="all").fit(Xn, y_cls)
    ranking = sorted(zip(NUM, selector.scores_), key=lambda x: x[1], reverse=True)
    metrics["feature_selection"]["mutual_information"] = [
        {"feature": name, "score": float(score)} for name, score in ranking
    ]

    pca = PCA().fit(Xn)
    cum = np.cumsum(pca.explained_variance_ratio_)
    metrics["dimensionality_reduction"] = {
        "components_for_95pct_variance": int(np.argmax(cum >= 0.95) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    sample = Xn[: min(12000, len(Xn))]
    cluster_scores = {}
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(sample)
        cluster_scores[str(k)] = float(silhouette_score(sample, labels))
    metrics["clustering"] = {
        "silhouette_by_k": cluster_scores,
        "best_k": int(max(cluster_scores, key=cluster_scores.get)),
    }

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )
    mean_reg = Pipeline([
        ("prep", make_preprocessor()),
        ("reg", HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=220, random_state=42
        )),
    ])
    mean_reg.fit(Xtr_r, ytr_r)
    pred = mean_reg.predict(Xte_r)
    metrics["regression"]["point"] = {
        "mae": float(mean_absolute_error(yte_r, pred)),
        "rmse": float(mean_squared_error(yte_r, pred) ** 0.5),
        "r2": float(r2_score(yte_r, pred)),
    }
    joblib.dump(mean_reg, MODELS / "traffic_regressor.joblib")

    q_preds = {}
    for q in [0.10, 0.50, 0.90]:
        model = Pipeline([
            ("prep", make_preprocessor()),
            ("reg", HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=180, random_state=42
            )),
        ])
        model.fit(Xtr_r, ytr_r)
        q_preds[q] = model.predict(Xte_r)
        joblib.dump(model, MODELS / f"traffic_q{int(q * 100):02d}.joblib")

    coverage = np.mean(
        (yte_r.to_numpy() >= q_preds[0.10]) &
        (yte_r.to_numpy() <= q_preds[0.90])
    )
    metrics["regression"]["probabilistic"] = {
        "interval": "10th-90th percentile",
        "empirical_coverage": float(coverage),
        "median_mae": float(mean_absolute_error(yte_r, q_preds[0.50])),
    }

    for region in sorted(df["region"].unique()):
        tr = df[df["region"] != region]
        te = df[df["region"] == region]
        model = Pipeline([
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=220,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])
        model.fit(tr[FEATURES], tr["congested"])
        prob = model.predict_proba(te[FEATURES])[:, 1]
        metrics["generalization"][region] = {
            "samples": int(len(te)),
            "roc_auc": float(roc_auc_score(te["congested"], prob)),
            "pr_auc": float(average_precision_score(te["congested"], prob)),
        }

    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
