"""Run model inference on CSV or Parquet traffic data."""
import argparse
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "region", "hour", "day_of_week", "active_users", "packet_rate",
    "latency_ms", "retransmission_rate", "queue_depth", "signal_quality_dbm",
]

def load_data(path: Path):
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or Parquet input file")
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()

    df = load_data(Path(args.input))
    clf = joblib.load(ROOT / "models" / "congestion_classifier.joblib")
    reg = joblib.load(ROOT / "models" / "traffic_regressor.joblib")
    q10 = joblib.load(ROOT / "models" / "traffic_q10.joblib")
    q90 = joblib.load(ROOT / "models" / "traffic_q90.joblib")

    df["congestion_probability"] = clf.predict_proba(df[FEATURES])[:, 1]
    df["predicted_downlink_mbps"] = reg.predict(df[FEATURES])
    df["downlink_p10"] = q10.predict(df[FEATURES])
    df["downlink_p90"] = q90.predict(df[FEATURES])

    out = ROOT / args.output
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved predictions to {out}")
    print(df[
        ["congestion_probability", "predicted_downlink_mbps",
         "downlink_p10", "downlink_p90"]
    ].head(10))

if __name__ == "__main__":
    main()
