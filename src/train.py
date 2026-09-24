import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,average_precision_score,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
R=Path(__file__).resolve().parents[1]; (R/"models").mkdir(exist_ok=True); (R/"outputs").mkdir(exist_ok=True)
df=pd.read_csv(R/"data"/"network_traffic.csv")
F=["region","hour","active_users","packet_rate","latency_ms","retransmission_rate","queue_depth","signal_quality_dbm"]
num=[x for x in F if x!="region"]
prep=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["region"])])
Xtr,Xte,ytr,yte=train_test_split(df[F],df.congested,test_size=.25,random_state=42,stratify=df.congested)
models={"logistic_smote":ImbPipeline([("prep",prep),("smote",SMOTE(random_state=42)),("clf",LogisticRegression(max_iter=1500,class_weight="balanced"))]),
"random_forest":Pipeline([("prep",prep),("clf",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced",n_jobs=-1))])}
M={"classification":{},"generalization":{}}
best=(-1,None,None)
for name,m in models.items():
 m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int); ap=average_precision_score(yte,p)
 M["classification"][name]={"roc_auc":float(roc_auc_score(yte,p)),"pr_auc":float(ap),"report":classification_report(yte,pred,output_dict=True)}
 if ap>best[0]: best=(ap,name,m)
joblib.dump(best[2],R/"models"/"congestion_classifier.joblib"); M["best_classifier"]=best[1]
Xtr,Xte,ytr,yte=train_test_split(df[F],df.downlink_mbps,test_size=.25,random_state=42)
reg=Pipeline([("prep",prep),("reg",HistGradientBoostingRegressor(random_state=42))]); reg.fit(Xtr,ytr); p=reg.predict(Xte)
M["regression"]={"mae":float(mean_absolute_error(yte,p)),"rmse":float(mean_squared_error(yte,p)**.5),"r2":float(r2_score(yte,p))}
joblib.dump(reg,R/"models"/"traffic_regressor.joblib")
for region in sorted(df.region.unique()):
 tr=df[df.region!=region]; te=df[df.region==region]
 m=Pipeline([("prep",prep),("clf",RandomForestClassifier(n_estimators=180,random_state=42,class_weight="balanced",n_jobs=-1))]); m.fit(tr[F],tr.congested); p=m.predict_proba(te[F])[:,1]
 M["generalization"][region]={"samples":int(len(te)),"roc_auc":float(roc_auc_score(te.congested,p)),"pr_auc":float(average_precision_score(te.congested,p))}
z=StandardScaler().fit_transform(df[num]); pc=PCA().fit(z); M["pca_components_for_95pct_variance"]=int(np.argmax(np.cumsum(pc.explained_variance_ratio_)>=.95)+1)
(R/"outputs"/"metrics.json").write_text(json.dumps(M,indent=2)); print(json.dumps(M,indent=2))
