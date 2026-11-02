"""svm_reaction.py DATASET : plain svm, label-scaled svm, rf, lightgbm, xgboost on the first three general representations, 5 group folds."""
import sys, time, json, os
sys.path.insert(0, '/Users/apunt/repos/KIRBy/src'); sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests')
os.environ.setdefault('KIRBY_SKIP_INSTALLS', '1')
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from kirby.hybrid import SVRScaledTarget
from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments import _shared as S
ds = sys.argv[1]
reps = DATASET_REPS[ds]['general'][:3]
rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
M = {'svm_plain': lambda: SVR(), 'svm_fixed': lambda: SVRScaledTarget(),
     'rf': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2),
     'lightgbm': lambda: LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=2),
     'xgboost': lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=2)}
out = {}
for rep in reps:
    t = time.time()
    try:
        d = S.load_data(ds, rep_names=[rep])
        y = np.asarray(d['labels'], float); g = np.asarray(d['groups']); X = np.asarray(list(d['reps'].values())[0], float)
        sc = {k: [] for k in M}
        for tr, va in GroupKFold(n_splits=5).split(X, y, g):
            s = StandardScaler().fit(X[tr]); Xt, Xv = s.transform(X[tr]), s.transform(X[va])
            for k, f in M.items(): sc[k].append(rmse(f().fit(Xt, y[tr]).predict(Xv), y[va]))
        out[rep] = {k: float(np.mean(v)) for k, v in sc.items()}; out[rep]['_width'] = int(X.shape[1])
        print(f'DONE {ds} {rep}', json.dumps({k: round(v, 2) for k, v in out[rep].items()}), f'{time.time()-t:.0f}s', flush=True)
    except Exception as e:
        out[rep] = {'error': f'{type(e).__name__}: {str(e)[:150]}'}; print('FAIL', ds, rep, out[rep], flush=True)
    json.dump(out, open(f'/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/svm_screen/reaction_{ds}.json', 'w'))
