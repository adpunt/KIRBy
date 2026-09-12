import os, sys, time
sys.path.insert(0,'src'); sys.path.insert(0,'tests'); sys.path.insert(0,'tests/experiments')
os.environ['KIRBY_PRUNE_CAP']='0'
import numpy as np
from rep_prune_model_check import score_rep, cap_cols
from experiments._shared import load_data
d=load_data('buchwald_hartwig', rep_names=['reactant_grover'])
y=np.asarray(d['labels'],float); g=np.asarray(d['groups'])
X=np.nan_to_num(np.asarray(d['reps']['reactant_grover'],np.float64))
print('dim',X.shape)
for m in ['rf','lightgbm','svm']:
    t=time.time(); s=score_rep(X,y,g,False,None,m); print(m, round(time.time()-t,1),'s', s, flush=True)
