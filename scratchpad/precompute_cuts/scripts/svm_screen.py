"""Quick screen: does the label-scaled svm change the picture on this dataset?
usage: python svm_screen.py DATASET [max_seconds]
One representation (first in the dataset's general pool), the dataset's 5 group folds (GroupKFold, as the search uses for regression),
features standardised on the training fold. Fits plain SVR, SVRScaledTarget, and the dataset's two chosen models where they have a default
(rf, lightgbm, xgboost; dnn and gp are not fitted here). Prints one JSON line."""
import sys, time, json, os
sys.path.insert(0, '/Users/apunt/repos/KIRBy/src'); sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests')
sys.path.insert(0, '/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/scripts')
os.environ.setdefault('KIRBY_SKIP_INSTALLS', '1')
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from kirby.hybrid import SVRScaledTarget
from axes import decided
from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments import _shared as S

ds = sys.argv[1]; budget_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1500
pair = decided()[0][ds]
t0 = time.time()
rep = DATASET_REPS[ds]['general'][0]
d = S.load_data(ds, rep_names=[rep])
y = np.asarray(d['labels'], float); g = np.asarray(d['groups']); X = np.asarray(list(d['reps'].values())[0], float)
rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
rho = lambda a, b: float(spearmanr(a, b)[0]) if np.std(a) > 0 else float('nan')
def make(name):
    if name == 'rf': return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2)
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor; return LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=2)
    if name == 'xgboost':
        from xgboost import XGBRegressor; return XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=2)
    return None
models = {'svm_plain': lambda: SVR(), 'svm_scaled': lambda: SVRScaledTarget()}
for m in pair:
    if make(m) is not None: models[m] = (lambda m=m: make(m))
res = {k: {'rmse': [], 'spearman': []} for k in models}
folds_done = 0
for tr, va in GroupKFold(n_splits=5).split(X, y, g):
    if time.time() - t0 > budget_s: break
    sc = StandardScaler().fit(X[tr]); Xt, Xv = sc.transform(X[tr]), sc.transform(X[va])
    for k, f in models.items():
        p = f().fit(Xt, y[tr]).predict(Xv)
        res[k]['rmse'].append(rmse(p, y[va])); res[k]['spearman'].append(rho(p, y[va]))
    folds_done += 1
out = {'dataset': ds, 'rep': rep, 'n': int(len(y)), 'n_features': int(X.shape[1]), 'label_mean': float(y.mean()), 'label_sd': float(y.std()),
       'pair': pair, 'not_fitted': [m for m in pair if make(m) is None], 'folds_done': folds_done, 'seconds': round(time.time() - t0),
       'mean': {k: {m: float(np.nanmean(v)) if v else None for m, v in r.items()} for k, r in res.items()}}
os.makedirs('/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/svm_screen', exist_ok=True)
json.dump(out, open(f'/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/svm_screen/{ds}.json', 'w'))
print('RESULT ' + json.dumps(out))
