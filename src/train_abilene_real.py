from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    column_map = {str(column).lower().strip(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]
    return None


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "R2": float(r2),
    }


def improvement_percentage(baseline_value: float, model_value: float) -> float:
    if baseline_value == 0:
        return float("nan")
    return float((baseline_value - model_value) / baseline_value * 100.0)


def prepare_dataframe(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, str | None]:
    df = df.copy().dropna(axis=1, how="all")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist. Available columns: {list(df.columns)}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if (df[target_col] < 0).any():
        print("Warning: negative target values found. Clipping them to zero.")
        df[target_col] = df[target_col].clip(lower=0)

    time_col = find_column(df, ["timestamp", "datetime", "date_time", "date", "time"])
    if time_col is not None:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce")
        if parsed_time.notna().mean() > 0.80:
            df[time_col] = parsed_time
            df = df.sort_values(time_col).reset_index(drop=True)
        else:
            time_col = None

    return df, time_col


def add_time_features(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    df = df.copy()
    if time_col is None or not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return df

    if "hour" not in df.columns:
        df["hour"] = df[time_col].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df[time_col].dt.dayofweek
    if "minute" not in df.columns:
        df["minute"] = df[time_col].dt.minute
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def build_features(df: pd.DataFrame, target_col: str, time_col: str | None) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    if time_col is not None and time_col in X.columns:
        X = X.drop(columns=[time_col])

    for column in X.select_dtypes(include=["bool"]).columns:
        X[column] = X[column].astype(int)

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False, dtype=float)

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    empty_columns = [column for column in X.columns if X[column].notna().sum() == 0]
    if empty_columns:
        X = X.drop(columns=empty_columns)

    return X, y


def create_models() -> dict[str, Pipeline]:
    return {
        "ExtraTrees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        max_features=1.0,
                        min_samples_leaf=1,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.07,
                        max_iter=300,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models on real Abilene traffic data.")
    parser.add_argument("--data", type=str, default="data/abilene_traffic_clean.csv")
    parser.add_argument("--target", type=str, default="target_next_traffic")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    model_dir = project_root / "models"
    results_dir = project_root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    print("\n" + "=" * 70)
    print("NETTRAFFICML - REAL ABILENE TRAINING")
    print("=" * 70)
    print(f"\nDataset:\n{data_path}")

    df = pd.read_csv(data_path)
    print(f"\nOriginal dataset shape: {df.shape}")

    target_col = args.target
    print(f"\nTarget column: {target_col}")

    df, time_col = prepare_dataframe(df, target_col)
    print(f"Time column: {time_col if time_col is not None else 'not detected'}")
    df = add_time_features(df, time_col)

    before_rows = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    print(f"\nRows removed because target was missing: {before_rows - len(df)}")
    print(f"Usable rows: {len(df)}")

    X, y = build_features(df, target_col, time_col)
    if "total_traffic" not in X.columns:
        raise ValueError("The prepared dataset must contain 'total_traffic' for the persistence baseline.")

    print(f"Number of model features: {X.shape[1]}")

    split_index = int(len(X) * (1.0 - args.test_size))
    X_train, X_test = X.iloc[:split_index].copy(), X.iloc[split_index:].copy()
    y_train, y_test = y.iloc[:split_index].copy(), y.iloc[split_index:].copy()

    print("\n" + "-" * 70)
    print("CHRONOLOGICAL HOLDOUT")
    print("-" * 70)
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Train share   : {100 * len(X_train) / len(X):.2f}%")
    print(f"Test share    : {100 * len(X_test) / len(X):.2f}%")

    baseline_predictions = X_test["total_traffic"].to_numpy(dtype=float)
    baseline_metrics = calculate_metrics(y_test.to_numpy(), baseline_predictions)

    print("\n" + "=" * 70)
    print("PERSISTENCE BASELINE")
    print("=" * 70)
    print("Prediction rule: next traffic = current observed traffic")
    print(f"MAE   : {baseline_metrics['MAE']:.6f}")
    print(f"RMSE  : {baseline_metrics['RMSE']:.6f}")
    print(f"MAPE  : {baseline_metrics['MAPE_percent']:.3f}%")
    print(f"R²    : {baseline_metrics['R2']:.6f}")

    y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))
    print("\nTarget transformation:")
    print("  Training target = log1p(target_next_traffic)")
    print("  Prediction output = expm1(prediction)")

    all_results = [
        {
            "model": "Persistence Baseline",
            **baseline_metrics,
            "MAE_improvement_percent": 0.0,
            "RMSE_improvement_percent": 0.0,
        }
    ]

    prediction_output = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "current_total_traffic": X_test["total_traffic"].to_numpy(),
            "Persistence_Baseline": baseline_predictions,
        }
    )
    trained_model_paths = {}

    for model_name, model in create_models().items():
        print("\n" + "=" * 70)
        print(f"TRAINING: {model_name}")
        print("=" * 70)

        model.fit(X_train, y_train_log)
        predictions = np.maximum(np.expm1(model.predict(X_test)), 0.0)
        metrics = calculate_metrics(y_test.to_numpy(), predictions)

        mae_improvement = improvement_percentage(baseline_metrics["MAE"], metrics["MAE"])
        rmse_improvement = improvement_percentage(baseline_metrics["RMSE"], metrics["RMSE"])

        all_results.append(
            {
                "model": model_name,
                **metrics,
                "MAE_improvement_percent": mae_improvement,
                "RMSE_improvement_percent": rmse_improvement,
            }
        )
        prediction_output[model_name] = predictions

        print(f"MAE   : {metrics['MAE']:.6f}")
        print(f"RMSE  : {metrics['RMSE']:.6f}")
        print(f"MAPE  : {metrics['MAPE_percent']:.3f}%")
        print(f"R²    : {metrics['R2']:.6f}")
        print("\nImprovement over persistence:")
        print(f"  MAE improvement  : {mae_improvement:.2f}%")
        print(f"  RMSE improvement : {rmse_improvement:.2f}%")

        safe_name = model_name.lower().replace(" ", "_")
        model_path = model_dir / f"abilene_{safe_name}.joblib"
        joblib.dump(
            {
                "model": model,
                "model_name": model_name,
                "target_column": target_col,
                "feature_columns": X.columns.tolist(),
                "baseline_column": "total_traffic",
                "uses_log_target": True,
                "target_transform": "log1p",
                "inverse_transform": "expm1",
                "test_size": args.test_size,
                "random_state": RANDOM_STATE,
            },
            model_path,
        )
        trained_model_paths[model_name] = str(model_path)
        print(f"\nSaved model:\n{model_path}")

    results_df = pd.DataFrame(all_results).sort_values("MAE").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    ml_results = results_df[results_df["model"] != "Persistence Baseline"].copy()
    best_row = ml_results.sort_values("MAE").iloc[0]
    best_model_name = best_row["model"]

    print("\n" + "=" * 70)
    print("BEST MACHINE-LEARNING MODEL")
    print("=" * 70)
    print(f"Model              : {best_model_name}")
    print(f"MAE                : {best_row['MAE']:.6f}")
    print(f"RMSE               : {best_row['RMSE']:.6f}")
    print(f"MAPE               : {best_row['MAPE_percent']:.3f}%")
    print(f"R²                 : {best_row['R2']:.6f}")
    print(f"MAE vs persistence : {best_row['MAE_improvement_percent']:.2f}%")
    print(f"RMSE vs persistence: {best_row['RMSE_improvement_percent']:.2f}%")

    if best_row["MAE_improvement_percent"] > 0:
        print("\nResult: the ML model BEATS the persistence baseline.")
    elif best_row["MAE_improvement_percent"] < 0:
        print("\nResult: the persistence baseline is BETTER than the ML model.")
    else:
        print("\nResult: the ML model and persistence baseline are equal on MAE.")

    metrics_path = results_dir / "abilene_model_metrics.csv"
    predictions_path = results_dir / "abilene_test_predictions.csv"
    summary_path = results_dir / "abilene_training_summary.json"

    results_df.to_csv(metrics_path, index=False)
    prediction_output.to_csv(predictions_path, index=False)

    summary = {
        "dataset": str(data_path),
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": target_col,
        "time_column": time_col,
        "log_target": True,
        "baseline": {
            "name": "Persistence Baseline",
            "prediction_rule": "target_next_traffic = current total_traffic",
            "column": "total_traffic",
            "metrics": baseline_metrics,
        },
        "best_ml_model": best_model_name,
        "best_model_mae": float(best_row["MAE"]),
        "best_model_rmse": float(best_row["RMSE"]),
        "best_model_mape_percent": float(best_row["MAPE_percent"]),
        "best_model_r2": float(best_row["R2"]),
        "best_model_mae_improvement_percent": float(best_row["MAE_improvement_percent"]),
        "best_model_rmse_improvement_percent": float(best_row["RMSE_improvement_percent"]),
        "models": trained_model_paths,
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print(f"\nMetrics:\n{metrics_path}")
    print(f"\nPredictions:\n{predictions_path}")
    print(f"\nTraining summary:\n{summary_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
