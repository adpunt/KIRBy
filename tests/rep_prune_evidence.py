"""Rep-pruning EVIDENCE — greedy correlation elimination, swept over thresholds.

Produces, per dataset, the numbers the USER reads to pick the final <=10 reps. It
does NOT pick the pool: it shows, for a sweep of correlation thresholds, what a
greedy "drop reps too correlated with a better-scoring rep" filter would keep, and
why each drop happened. Different thresholds -> different pools; the user compares
and decides per dataset.

Method (per dataset):
  A. per-rep out-of-fold (OOF) prediction vectors under the dataset's group CV
     (lightgbm) + solo score.                                    [n CVs]
  B. full rep x rep prediction-correlation matrix (Spearman of OOF vectors). [free]
  C. GREEDY ELIMINATION, best-solo-first, for each threshold T: keep a rep only if
     its correlation to every already-kept rep is < T; otherwise drop it and record
     the kept rep it was redundant with. Dropped reps are never used as anchors
     again (user: "stop using it once filtered").                [free, uses B]
  D. one bounded concat check per rep: concat(rep, nearest higher-solo partner) -
     solo(partner) -> was dropping that rep actually free?        [n-1 CVs]

Compute is ~2n CVs per dataset (NOT the O(n^2) all-pairs concat). lightgbm only.
Cache path from KIRBY_CACHE_DIR; no hardcoded local path.

Tunables via env (light defaults are for a LOCAL smoke test; the server run should
override to the real values):
  KIRBY_PRUNE_FOLDS=3            group-CV folds
  KIRBY_PRUNE_CAP=512           per-rep top-variance feature cap (0 = no cap)
  KIRBY_PRUNE_THRESHOLDS=0.75,0.80,0.85,0.90,0.95   correlation sweep

Writes $KIRBY_CACHE_DIR/rep_prune/<ds>/:
  preds.npz     : {rep__<name>: oof_vector} + 'labels'
  scores.json   : {rep: {solo, rmse, n, dim}} best-first
  pred_corr.csv : rep x rep prediction-correlation matrix
  greedy.json   : {thresholds: {T: {kept, n_kept, dropped:[{rep,redundant_with,
                  corr,solo,anchor_solo}]}}, nearest:[{rep,partner,corr,concat,gain}]}

Run one dataset locally (import/paths verified; the real sweep is a server job):
  python tests/rep_prune_evidence.py herg_ki
"""
import os, sys, json, time, csv, itertools
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'experiments'))

from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMRegressor, LGBMClassifier

from experiments._shared import load_data

CACHE = os.environ.get('KIRBY_CACHE_DIR', os.path.expanduser('~/.cache/kirby'))
OUT_ROOT = os.path.join(CACHE, 'rep_prune')

N_FOLDS = int(os.environ.get('KIRBY_PRUNE_FOLDS', '3'))
CAP = int(os.environ.get('KIRBY_PRUNE_CAP', '512'))          # 0 disables the cap
THRESHOLDS = [float(t) for t in
              os.environ.get('KIRBY_PRUNE_THRESHOLDS', '0.75,0.80,0.85,0.90,0.95').split(',')]
# The concat check (does adding a rep to a stronger one gain signal?) is part of
# the COMBINATION work, not rep-pool cleanup — off by default. Set =1 to include it.
RUN_CONCAT = os.environ.get('KIRBY_PRUNE_CONCAT', '0') == '1'
N_EST = 150


def _model(is_clf):
    return (LGBMClassifier if is_clf else LGBMRegressor)(
        n_estimators=N_EST, num_leaves=31, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)


def cap_cols(X):
    if not CAP or X.shape[1] <= CAP:
        return X
    idx = np.argsort(X.var(0))[::-1][:CAP]
    return X[:, np.sort(idx)]


def oof(X, y, groups, is_clf):
    """OOF predictions under the group CV -> (vector, score). clf: proba + AUROC;
    reg: prediction + Spearman."""
    n = len(y); v = np.full(n, np.nan)
    k = min(N_FOLDS, len(np.unique(groups)))
    if k < 2:
        return None, None
    sp = (StratifiedGroupKFold(k, shuffle=True, random_state=42).split(X, y, groups)
          if is_clf else GroupKFold(k).split(X, y, groups))
    for tr, va in sp:
        s = StandardScaler(); Xtr = s.fit_transform(X[tr]); Xva = s.transform(X[va])
        m = _model(is_clf); m.fit(Xtr, y[tr])
        v[va] = m.predict_proba(Xva)[:, 1] if is_clf else m.predict(Xva)
    mask = ~np.isnan(v)
    sc = (float(roc_auc_score(y[mask], v[mask])) if is_clf
          else float(spearmanr(y[mask], v[mask]).correlation))
    return v, sc


def greedy_eliminate(names, solo, corr, T):
    """Walk reps best-solo-first; keep a rep only if its correlation to every
    already-kept rep is < T, else drop it (recording the anchor it duplicates).
    Dropped reps never anchor a later comparison."""
    kept, dropped = [], []
    for r in names:                     # names already sorted best-solo-first
        anchor, cmax = None, -2.0
        for k in kept:
            c = corr[r][k]
            if c > cmax:
                cmax, anchor = c, k
        if anchor is not None and cmax >= T:
            dropped.append({'rep': r, 'redundant_with': anchor,
                            'corr': round(cmax, 3), 'solo': round(solo[r], 3),
                            'anchor_solo': round(solo[anchor], 3)})
        else:
            kept.append(r)
    return kept, dropped


def run(ds):
    t0 = time.time()
    print(f"\n{'='*70}\n{ds}\n{'='*70}", flush=True)
    data = load_data(ds)
    reps = data['reps']; y = np.asarray(data['labels'], float)
    groups = np.asarray(data['groups'])
    is_clf = data.get('task_type') == 'classification'
    primary = 'auroc' if is_clf else 'spearman'
    out_dir = os.path.join(OUT_ROOT, ds); os.makedirs(out_dir, exist_ok=True)

    # ---- A: OOF vectors + solo scores --------------------------------------
    Xc, vec, solo, meta = {}, {}, {}, {}
    for rep in sorted(reps):
        X = cap_cols(np.nan_to_num(np.asarray(reps[rep], np.float64), nan=0.0))
        try:
            v, sc = oof(X, y, groups, is_clf)
        except Exception as e:
            print(f"  {rep:24s} FAILED: {e}", flush=True); continue
        if v is None:
            print(f"  {rep:24s} skipped (too few groups)", flush=True); continue
        m = ~np.isnan(v)
        Xc[rep] = X; vec[rep] = v; solo[rep] = sc
        meta[rep] = {'solo': round(sc, 4),
                     'rmse': float(np.sqrt(np.mean((y[m] - v[m]) ** 2))),
                     'n': int(m.sum()), 'dim': int(X.shape[1])}
        print(f"  {rep:24s} dim={X.shape[1]:5d}  {primary}={sc:+.4f}", flush=True)

    names = sorted(solo, key=lambda r: -solo[r])   # best-solo-first
    if len(names) < 2:
        print(f"  only {len(names)} scorable rep(s); nothing to prune", flush=True)
        return
    np.savez_compressed(os.path.join(out_dir, 'preds.npz'),
                        labels=y, **{f'rep__{k}': vec[k] for k in names})
    json.dump({'dataset': ds, 'is_clf': is_clf, 'primary': primary,
               'reps': {r: meta[r] for r in names}},
              open(os.path.join(out_dir, 'scores.json'), 'w'), indent=2)

    # ---- B: prediction-correlation matrix (free) ---------------------------
    corr = {a: {} for a in names}
    for a, b in itertools.combinations(names, 2):
        c = float(spearmanr(vec[a], vec[b]).correlation)
        corr[a][b] = corr[b][a] = c
    for a in names:
        corr[a][a] = 1.0
    with open(os.path.join(out_dir, 'pred_corr.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow([''] + names)
        for a in names:
            w.writerow([a] + [f"{corr[a][b]:.3f}" for b in names])

    # ---- C: greedy elimination over the threshold sweep --------------------
    thresholds = {}
    for T in THRESHOLDS:
        kept, dropped = greedy_eliminate(names, solo, corr, T)
        thresholds[f"{T:.2f}"] = {'kept': kept, 'n_kept': len(kept), 'dropped': dropped}
        print(f"  T={T:.2f}: keep {len(kept):2d} / {len(names)}  "
              f"(drop {len(dropped)})", flush=True)

    # ---- D: OPTIONAL concat check (combination work, off by default) --------
    nearest = []
    if RUN_CONCAT:
        for r in names[1:]:             # top-solo rep has no higher partner
            better = [p for p in names if solo[p] >= solo[r] and p != r]
            if not better:
                continue
            p = max(better, key=lambda p: corr[r][p])  # most-correlated anchor
            try:
                _, concat = oof(np.hstack([Xc[r], Xc[p]]), y, groups, is_clf)
            except Exception as e:
                print(f"    concat {r}+{p} FAILED: {e}", flush=True); continue
            nearest.append({'rep': r, 'partner': p, 'corr': round(corr[r][p], 3),
                            'concat': round(concat, 3),
                            'gain': round(concat - solo[p], 4)})

    json.dump({'dataset': ds, 'primary': primary, 'n_reps': len(names),
               'solo': {r: round(solo[r], 4) for r in names},
               'thresholds': thresholds, 'nearest': nearest},
              open(os.path.join(out_dir, 'greedy.json'), 'w'), indent=2)
    print(f"  -> saved rep_prune/{ds}/  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    print(f"[rep_prune] EVIDENCE ONLY — selects nothing. folds={N_FOLDS} cap={CAP} "
          f"thresholds={THRESHOLDS} concat={'on' if RUN_CONCAT else 'off'}", flush=True)
    for ds in sys.argv[1:]:
        try:
            run(ds)
        except Exception:
            import traceback; traceback.print_exc()
            print(f"DATASET {ds} FAILED", flush=True)
