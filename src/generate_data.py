import numpy as np, pandas as pd
from pathlib import Path
rng=np.random.default_rng(42); n=30000
regions=np.array(["Stockholm","Gothenburg","Malmo","Uppsala"])
region=rng.choice(regions,n,p=[.42,.27,.21,.10]); hour=rng.integers(0,24,n)
users=rng.integers(20,2500,n); packets=rng.gamma(3,450,n)
lat=np.clip(rng.normal(18,8,n)+users/600,1,None)
ret=np.clip(rng.beta(1.5,25,n)+lat/1500,0,1); queue=np.maximum(0,rng.poisson(users/220))
signal=np.clip(rng.normal(-78,10,n),-120,-45)
down=np.maximum(0,.055*users+.018*packets-1.5*queue+.6*(signal+100)+rng.normal(0,18,n))
score=.0022*users+.0015*packets+.12*queue+7*ret+.035*lat
cong=(score>np.quantile(score,.88)).astype(int)
df=pd.DataFrame({"region":region,"hour":hour,"active_users":users,"packet_rate":packets,
"latency_ms":lat,"retransmission_rate":ret,"queue_depth":queue,
"signal_quality_dbm":signal,"downlink_mbps":down,"congested":cong})
out=Path(__file__).resolve().parents[1]/"data"/"network_traffic.csv"
df.to_csv(out,index=False); print("saved",len(df),"rows to",out); print(df.congested.value_counts(normalize=True))
