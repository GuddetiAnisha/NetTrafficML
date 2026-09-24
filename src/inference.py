import argparse,joblib,pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parents[1]; a=argparse.ArgumentParser(); a.add_argument("--csv",required=True); x=a.parse_args()
df=pd.read_csv(x.csv); F=["region","hour","active_users","packet_rate","latency_ms","retransmission_rate","queue_depth","signal_quality_dbm"]
c=joblib.load(R/"models"/"congestion_classifier.joblib"); r=joblib.load(R/"models"/"traffic_regressor.joblib")
df["congestion_probability"]=c.predict_proba(df[F])[:,1]; df["predicted_downlink_mbps"]=r.predict(df[F])
print(df[["congestion_probability","predicted_downlink_mbps"]].head(20).to_string(index=False))
