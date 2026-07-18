"""Step-1 validity gate — does a dataset's best base rep give meaningful signal?

Per dataset: load its general reps, score each under the dataset's group-aware CV
with lightgbm (+ ridge for NUCLEOTIDE only, per user), take the best. Metric:
Spearman for regression, MCC for classification (both have a ~0 trivial baseline,
so "meaningful" = clearly above 0). Saves $KIRBY_CACHE_DIR/validity/<ds>.json.

Cache location comes from KIRBY_CACHE_DIR (setup.sh sets it on the server). NO
hardcoded local path.
"""
import os, sys, json, time
import numpy as np

# repo-relative imports
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'experiments'))

from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from lightgbm import LGBMRegressor, LGBMClassifier

from experiments._shared import load_data

CACHE = os.environ.get('KIRBY_CACHE_DIR', os.path.expanduser('~/.cache/kirby'))
OUT = os.path.join(CACHE, 'validity')
NF = 5
PASS_MIN = 0.10   # suggested threshold; the USER makes the final keep/cut call


def is_nucleotide(ds):
    return ds.startswith('nt_') or ds.startswith('gb_')


def _model(name, is_clf):
    if name == 'lightgbm':
        return (LGBMClassifier if is_clf else LGBMRegressor)(
            n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)
    if name == 'ridge':   # nucleotide only
        if is_clf:
            from sklearn.linear_model import RidgeClassifier
            return RidgeClassifier(alpha=1.0)
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    raise ValueError(name)


def score_rep(X, y, groups, is_clf, model_name):
    n = len(y); k = min(NF, len(np.unique(groups)))
    if k < 2:
        return float('nan')
    pred = np.full(n, np.nan)
    sp = (StratifiedGroupKFold(k, shuffle=True, random_state=42).split(X, y, groups)
          if is_clf else GroupKFold(k).split(X, y, groups))
    for tr, va in sp:
        s = StandardScaler(with_mean=True); Xtr = s.fit_transform(X[tr]); Xva = s.transform(X[va])
        m = _model(model_name, is_clf)
        m.fit(Xtr, y[tr])
        pred[va] = m.predict(Xva)
    mask = ~np.isnan(pred)
    if is_clf:
        try:
            return float(matthews_corrcoef(y[mask].astype(int), np.rint(pred[mask]).astype(int)))
        except Exception:
            return float('nan')
    return float(spearmanr(y[mask], pred[mask]).correlation)


def run(ds):
    t0 = time.time()
    print(f"=== {ds} ===", flush=True)
    data = load_data(ds)
    reps = data['reps']; y = np.asarray(data['labels'])
    groups = np.asarray(data['groups'])
    is_clf = data.get('task_type') == 'classification'
    metric = 'mcc' if is_clf else 'spearman'
    models = ['lightgbm'] + (['ridge'] if is_nucleotide(ds) else [])
    scores = {}
    for rep in sorted(reps):
        X = np.nan_to_num(np.asarray(reps[rep], dtype=np.float64), nan=0.0)
        best = None
        for mn in models:
            try:
                sc = score_rep(X, (y.astype(int) if is_clf else y.astype(float)), groups, is_clf, mn)
            except Exception as e:
                print(f"  {rep}/{mn} FAILED: {e}", flush=True); continue
            if sc is not None and np.isfinite(sc) and (best is None or sc > best[1]):
                best = (mn, sc)
        if best:
            scores[rep] = {'model': best[0], 'score': round(best[1], 4), 'dim': int(X.shape[1])}
            print(f"  {rep:22s} {metric}={best[1]:+.3f} ({best[0]}, dim={X.shape[1]})", flush=True)
    ranked = dict(sorted(scores.items(), key=lambda kv: -kv[1]['score']))
    best_rep = next(iter(ranked), None)
    best_score = ranked[best_rep]['score'] if best_rep else float('nan')
    verdict = 'PASS' if best_score >= PASS_MIN else ('BORDERLINE' if best_score >= 0.05 else 'FAIL')
    os.makedirs(OUT, exist_ok=True)
    json.dump({'dataset': ds, 'metric': metric, 'is_clf': is_clf, 'models': models,
               'best_rep': best_rep, 'best_score': best_score, 'verdict': verdict,
               'reps': ranked}, open(os.path.join(OUT, f'{ds}.json'), 'w'), indent=2)
    print(f"  => best {best_rep} {metric}={best_score:+.3f}  VERDICT={verdict}  "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    for ds in sys.argv[1:]:
        try:
            run(ds)
        except Exception:
            import traceback; traceback.print_exc()
            print(f"DATASET {ds} FAILED", flush=True)
