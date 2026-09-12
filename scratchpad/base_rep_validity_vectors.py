"""Phase 2.1 — base-rep validity gate + per-fold PREDICTION VECTORS.

For each dataset, for each base rep in its DATASET_REPS['general'] pool:
  - group-aware 5-fold CV (StratifiedGroupKFold for clf, GroupKFold for reg),
    StandardScaler fit per train fold, lightgbm as the scoring model,
  - collect the OUT-OF-FOLD prediction vector (one prediction per sample),
  - score it (rmse + spearman for regression).

Saves durably to /Volumes/seagate/kirby_cache/prediction_vectors/<ds>/:
  - preds.npz   : {rep_name: oof_vector}, plus 'labels' and 'fold_id'
  - scores.json : {rep: {rmse, spearman, n, dim}}, sorted best-first
  - pred_corr.csv : rep x rep PREDICTION-correlation matrix (Spearman of OOF
    prediction vectors) — the decorrelation input Chat 2 needs.

lightgbm only (the decided internal evaluator) — this is the light validity/
decorrelation pass, NOT the full rep x model target (Phase 2.2).
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests'); sys.path.insert(0, 'tests/experiments')

from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier

from experiments._shared import load_data

OUT_ROOT = '/Volumes/seagate/kirby_cache/prediction_vectors'
DATASETS = sys.argv[1:] or ['flab_thermo_tresanco', 'herg_ki', 'buchwald_hartwig', 'logd']
N_FOLDS = 3          # light validity/decorrelation pass, not the full target
N_ESTIMATORS = 150
print(f"[harness] datasets={DATASETS} folds={N_FOLDS} trees={N_ESTIMATORS}", flush=True)


def oof_predictions(X, y, groups, is_clf):
    n = len(y)
    oof = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    ng = len(np.unique(groups))
    k = min(N_FOLDS, ng)
    if k < 2:
        return None, None
    if is_clf:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups=groups)
    else:
        splits = GroupKFold(n_splits=k).split(X, y, groups=groups)
    for fi, (tr, va) in enumerate(splits):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        m = (LGBMClassifier if is_clf else LGBMRegressor)(
            n_estimators=N_ESTIMATORS, max_depth=-1, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xtr, y[tr])
        pred = m.predict_proba(Xva)[:, 1] if is_clf else m.predict(Xva)
        oof[va] = pred; fold_id[va] = fi
    return oof, fold_id


def run_dataset(ds):
    t0 = time.time()
    print(f"\n{'='*60}\n{ds}\n{'='*60}", flush=True)
    data = load_data(ds)
    reps = data['reps']; y = np.asarray(data['labels'], dtype=float)
    groups = np.asarray(data['groups'])
    is_clf = data.get('task_type') == 'classification'
    out_dir = os.path.join(OUT_ROOT, ds); os.makedirs(out_dir, exist_ok=True)

    vectors, scores = {}, {}
    for rep in sorted(reps):
        X = np.nan_to_num(np.asarray(reps[rep], dtype=np.float64), nan=0.0)
        try:
            oof, fold_id = oof_predictions(X, y, groups, is_clf)
        except Exception as e:
            print(f"  {rep:24s} FAILED: {e}", flush=True); continue
        if oof is None:
            print(f"  {rep:24s} skipped (too few groups)", flush=True); continue
        mask = ~np.isnan(oof)
        rmse = float(np.sqrt(np.mean((y[mask] - oof[mask]) ** 2)))
        rho = float(spearmanr(y[mask], oof[mask]).correlation)
        vectors[rep] = oof
        scores[rep] = {'rmse': rmse, 'spearman': rho, 'n': int(mask.sum()),
                       'dim': int(X.shape[1])}
        print(f"  {rep:24s} dim={X.shape[1]:5d}  rmse={rmse:.4f}  rho={rho:+.4f}", flush=True)

    # save vectors + labels + fold ids
    np.savez_compressed(os.path.join(out_dir, 'preds.npz'),
                        labels=y, **{f'rep__{k}': v for k, v in vectors.items()})
    ranked = dict(sorted(scores.items(), key=lambda kv: -kv[1]['spearman']))
    with open(os.path.join(out_dir, 'scores.json'), 'w') as f:
        json.dump({'dataset': ds, 'is_clf': is_clf, 'metric_primary':
                   'spearman', 'reps': ranked}, f, indent=2)

    # prediction-correlation matrix (Spearman of OOF vectors) — decorrelation input
    names = list(vectors)
    if len(names) >= 2:
        M = np.vstack([vectors[n_] for n_ in names])
        common = ~np.isnan(M).any(axis=0)
        C = np.corrcoef(np.apply_along_axis(
            lambda r: r.argsort().argsort(), 1, M[:, common]))
        import csv
        with open(os.path.join(out_dir, 'pred_corr.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow([''] + names)
            for i, n_ in enumerate(names):
                w.writerow([n_] + [f"{C[i, j]:.3f}" for j in range(len(names))])
    print(f"  -> saved {len(vectors)} rep vectors to {out_dir}  ({time.time()-t0:.0f}s)", flush=True)
    return ranked


if __name__ == '__main__':
    for ds in DATASETS:
        try:
            run_dataset(ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"DATASET {ds} FAILED: {e}", flush=True)
