"""Step 2/3 evidence — PAIRWISE concat numbers, no thresholds, no auto-cut.

Never concatenates more than 2 reps at once (user constraint — a growing greedy
is a compute blowup). Produces, for every rep pair (A,B):
  - prediction correlation (Spearman of OOF vectors)  [from step-1/2 artifacts]
  - solo_A, solo_B                                     [capped lightgbm CV]
  - concat(A,B)                                        [capped lightgbm CV, 2 reps]
  - gain = concat - max(solo_A, solo_B)                [does pairing add signal?]

The user reads these and decides which reps are redundant and what the <=10 pool
is. NOTHING is dropped here. Per-rep feature cap keeps each concat to <=2*CAP cols.

Output: prediction_vectors/<ds>/pairwise.json + a printed table sorted by corr desc.
"""
import os, sys, json, time, itertools
import numpy as np
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests'); sys.path.insert(0, 'tests/experiments')
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMRegressor, LGBMClassifier
from experiments._shared import load_data

OUT = '/Volumes/seagate/kirby_cache/prediction_vectors'
DATASETS = sys.argv[1:] or ['herg_ki']
NF, NEST, CAP = 3, 120, 512
print(f"[pairwise] datasets={DATASETS} folds={NF} trees={NEST} cap={CAP} (max 2 reps concatenated)",
      flush=True)


def cap_cols(X):
    if X.shape[1] <= CAP:
        return X
    idx = np.argsort(X.var(0))[::-1][:CAP]
    return X[:, np.sort(idx)]


def cv_score(X, y, groups, is_clf):
    n = len(y); oof = np.full(n, np.nan)
    k = min(NF, len(np.unique(groups)))
    sp = (StratifiedGroupKFold(k, shuffle=True, random_state=42).split(X, y, groups)
          if is_clf else GroupKFold(k).split(X, y, groups))
    for tr, va in sp:
        s = StandardScaler(); Xtr = s.fit_transform(X[tr]); Xva = s.transform(X[va])
        m = (LGBMClassifier if is_clf else LGBMRegressor)(
            n_estimators=NEST, num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xtr, y[tr])
        oof[va] = m.predict_proba(Xva)[:, 1] if is_clf else m.predict(Xva)
    mask = ~np.isnan(oof)
    return float(roc_auc_score(y[mask], oof[mask])) if is_clf \
        else float(spearmanr(y[mask], oof[mask]).correlation)


def run(ds):
    t0 = time.time()
    print(f"\n{'='*70}\n{ds}\n{'='*70}", flush=True)
    data = load_data(ds)
    reps = data['reps']; y = np.asarray(data['labels'], float)
    groups = np.asarray(data['groups']); is_clf = data.get('task_type') == 'classification'
    metric = 'auroc' if is_clf else 'spearman'
    S = json.load(open(f'{OUT}/{ds}/scores.json'))['reps']
    npz = np.load(f'{OUT}/{ds}/preds.npz')
    vec = {k[5:]: npz[k] for k in npz.files if k.startswith('rep__')}
    names = list(S)  # best-first by step-1 spearman

    Xc = {r: cap_cols(np.nan_to_num(np.asarray(reps[r], float), nan=0.0)) for r in names}
    solo = {r: cv_score(Xc[r], y, groups, is_clf) for r in names}
    print("  solo (capped) scores:", {r: round(solo[r], 3) for r in names}, flush=True)

    rows = []
    for a, b in itertools.combinations(names, 2):
        corr = float(spearmanr(vec[a], vec[b]).correlation)
        concat = cv_score(np.hstack([Xc[a], Xc[b]]), y, groups, is_clf)
        gain = concat - max(solo[a], solo[b])
        rows.append({'A': a, 'B': b, 'corr': round(corr, 3),
                     'solo_A': round(solo[a], 3), 'solo_B': round(solo[b], 3),
                     'concat': round(concat, 3), 'gain': round(gain, 4)})
    rows.sort(key=lambda r: -r['corr'])

    json.dump({'dataset': ds, 'metric': metric,
               'solo': {r: round(solo[r], 4) for r in names},
               'pairs': rows}, open(f'{OUT}/{ds}/pairwise.json', 'w'), indent=2)

    print(f"\n  most-correlated pairs (redundancy candidates) — does concatenating help?")
    print(f"  {'A':16s} {'B':16s} corr  soloA  soloB concat   gain")
    for r in rows[:20]:
        print(f"  {r['A']:16s} {r['B']:16s} {r['corr']:+.2f} {r['solo_A']:+.3f} "
              f"{r['solo_B']:+.3f} {r['concat']:+.3f} {r['gain']:+.4f}", flush=True)
    print(f"  -> {len(rows)} pairs, saved pairwise.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    for ds in DATASETS:
        try:
            run(ds)
        except Exception:
            import traceback; traceback.print_exc()
