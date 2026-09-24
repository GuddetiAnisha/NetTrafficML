import json,pandas as pd,streamlit as st
from pathlib import Path
R=Path(__file__).resolve().parent; st.set_page_config(page_title="NetTrafficML",layout="wide"); st.title("NetTrafficML — Network Traffic Modeling")
p=R/"data"/"network_traffic.csv"
if not p.exists(): st.warning("Run python src/generate_data.py first"); st.stop()
df=pd.read_csv(p); a,b,c,d=st.columns(4); a.metric("Samples",f"{len(df):,}"); b.metric("Regions",df.region.nunique()); c.metric("Congestion",f"{df.congested.mean()*100:.1f}%"); d.metric("Avg downlink",f"{df.downlink_mbps.mean():.1f} Mbps")
st.subheader("Traffic by hour"); st.line_chart(df.groupby("hour").downlink_mbps.mean())
st.subheader("Class imbalance"); st.bar_chart(df.congested.value_counts())
m=R/"outputs"/"metrics.json"
if m.exists():
 x=json.loads(m.read_text()); st.subheader("Geographical generalization"); st.dataframe(pd.DataFrame(x["generalization"]).T); st.write("Regression",x["regression"])
