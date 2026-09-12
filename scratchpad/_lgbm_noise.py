import os, sys, time
sys.path.insert(0,'src'); sys.path.insert(0,'tests'); sys.path.insert(0,'tests/experiments')
os.environ['KIRBY_PRUNE_CAP']='0'
import numpy as np, lightgbm
from experiments._shared import load_data
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
print('lightgbm version', lightgbm.__version__)
d=load_data('buchwald_hartwig', rep_names=['diff_mhg_gnn'])
y=np.asarray(d['labels'],float); g=np.asarray(d['groups'])
X=np.nan_to_num(np.asarray(d['reps']['diff_mhg_gnn'],np.float64))
print('groups', len(np.unique(g)), 'X', X.shape, 'y mean', y.mean().round(3))
from lightgbm import LGBMRegressor
for jobs in [1,4,-1]:
    v=np.full(len(y),np.nan)
    for tr,va in GroupKFold(3).split(np.zeros((len(y),1)),y,g):
        s=StandardScaler(); Xtr=s.fit_transform(X[tr]); Xva=s.transform(X[va])
        m=LGBMRegressor(n_estimators=150,num_leaves=31,learning_rate=0.05,subsample=0.8,
                        colsample_bytree=0.8,random_state=42,n_jobs=jobs,verbosity=-1)
        m.fit(Xtr,y[tr]); v[va]=m.predict(Xva)
    print(f'n_jobs={jobs}: rmse={np.sqrt(np.mean((y-v)**2)):.3f} rho={spearmanr(y,v).correlation:.3f}', flush=True)
