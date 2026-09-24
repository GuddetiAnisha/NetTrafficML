import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="NetTrafficML", layout="wide")
st.title("NetTrafficML — Statistical ML for Network Traffic")
st.caption(
    "Synthetic-data research prototype: classification, regression, "
    "imbalance handling and regional generalization."
)

data_path = ROOT / "data" / "network_traffic.parquet"
if not data_path.exists():
    st.warning("Run python src/generate_data.py first.")
    st.stop()

df = pd.read_parquet(data_path)
metrics_path = ROOT / "outputs" / "metrics.json"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df):,}")
c2.metric("Regions", df["region"].nunique())
c3.metric("Congestion rate", f"{100 * df['congested'].mean():.1f}%")
c4.metric("Average downlink", f"{df['downlink_mbps'].mean():.1f} Mbps")

region = st.selectbox("Region", ["All"] + sorted(df["region"].unique().tolist()))
view = df if region == "All" else df[df["region"] == region]

left, right = st.columns(2)
with left:
    st.subheader("Mean traffic by hour")
    st.line_chart(view.groupby("hour")["downlink_mbps"].mean())
with right:
    st.subheader("Congestion by hour")
    st.line_chart(view.groupby("hour")["congested"].mean())

left, right = st.columns(2)
with left:
    st.subheader("Class imbalance")
    st.bar_chart(
        df["congested"].value_counts().rename(index={0: "Normal", 1: "Congested"})
    )
with right:
    st.subheader("Regional mean downlink")
    st.bar_chart(df.groupby("region")["downlink_mbps"].mean())

if metrics_path.exists():
    m = json.loads(metrics_path.read_text(encoding="utf-8"))

    st.subheader("Classification model comparison")
    rows = []
    for name, v in m["classification"].items():
        rows.append({
            "model": name,
            "ROC-AUC": v["roc_auc"],
            "PR-AUC": v["pr_auc"],
            "precision": v["precision"],
            "recall": v["recall"],
            "F1": v["f1"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Geographical generalization")
    st.dataframe(pd.DataFrame(m["generalization"]).T, use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Feature relevance")
        fi = pd.DataFrame(m["feature_selection"]["mutual_information"])
        st.bar_chart(fi.set_index("feature")["score"])
    with b:
        st.subheader("Clustering diagnostic")
        st.bar_chart(
            pd.Series(m["clustering"]["silhouette_by_k"], name="silhouette")
        )

    st.subheader("Regression")
    st.json(m["regression"])
else:
    st.info("Run python src/train.py to display model evaluation results.")
