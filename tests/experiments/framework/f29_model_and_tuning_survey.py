#!/usr/bin/env python3
"""f29 — base-rep model survey + hyperparameter-tuning arms.

Two questions:

  1. Which models show signal on the BASE representations? (no hybrid, no
     feature allocation — each of a dataset's 10 general reps is fed to each
     model whole.)
  2. Does hyperparameter tuning pay, and is the right unit per-rep or
     per-model?

Three arms per (dataset, rep, model):

  default          DEFAULT_MODELS parameters, untouched.
  tuned_per_rep    RandomizedSearchCV inside each outer-TRAIN fold, on that
                   rep's own features. One parameter set per (rep, model).
  tuned_per_model  RandomizedSearchCV inside each outer-TRAIN fold, on the
                   concatenation of all 10 reps. One parameter set per model,
                   reused for every rep of that dataset.

Tuning always runs on the outer-train fold only, so the reported outer-fold
score never sees hyperparameters chosen on it.

Search spaces come from kirby.find_best_hybrid._HP_SEARCH_SPACES (plus a ridge
alpha grid defined here — the engine has no ridge entry). dnn and cnn have no
search space in the engine; they carry their internal early stopping into every
arm, and are reported as untuned.

Every fit is timed. Seconds-per-fit and seconds-per-tune are written next to
every score.

Usage:
    python tests/experiments/framework/f29_model_and_tuning_survey.py --probe
    python tests/experiments/framework/f29_model_and_tuning_survey.py \
        --datasets caco2 --arms default
"""

import os
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')   # torch + LightGBM in one process
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import sys
import json
import time
import argparse
import warnings
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset -> primary metric (user-specified, 2026-09-02).
DATASETS = {
    'buchwald_hartwig':       'rmse',
    'caco2':                  'rmse',
    'flab_expression_koenig': 'spearman',
    'tcga_brca_pam50':        'f1_weighted',
    'logd':                   'rmse',
}

MODELS = ['ridge', 'rf', 'lightgbm', 'catboost', 'xgboost',
          'dnn', 'mlp', 'gp', 'knn', 'cnn']

N_OUTER_FOLDS = 5
N_TUNE_FOLDS = 3
N_TUNE_ITER = 20
SEED = 42

# The engine defines no ridge search space (ridge is nucleotide-only there).
# alpha is the only thing worth sweeping on a linear baseline.
RIDGE_SEARCH_SPACE = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}


# =============================================================================
# HELPERS
# =============================================================================

def _metric_fn(name):
    from kirby.find_best_hybrid import BUILTIN_METRICS
    fn, lower_is_better = BUILTIN_METRICS[name]
    return fn, lower_is_better


def _sklearn_scorer(metric_name, task_type):
    """Scorer for RandomizedSearchCV — higher is better, matching the
    dataset's own metric so tuning optimises what is reported."""
    from sklearn.metrics import make_scorer, f1_score, mean_squared_error
    from scipy.stats import spearmanr
    if metric_name == 'rmse':
        return make_scorer(lambda y, p: -np.sqrt(mean_squared_error(y, p)))
    if metric_name == 'spearman':
        return make_scorer(lambda y, p: float(spearmanr(y, p)[0]))
    if metric_name == 'f1_weighted':
        return make_scorer(lambda y, p: f1_score(y, p, average='weighted',
                                                 zero_division=0))
    raise ValueError(f'no scorer for metric {metric_name!r}')


def _is_binary(X):
    if X.dtype == np.uint8:
        return True
    u = np.unique(X)
    return len(u) <= 2 and set(u.tolist()).issubset({0, 0.0, 1, 1.0})


def _prep(X_train, X_val):
    """Match _evaluate_single: scale continuous reps, leave binary raw."""
    from kirby.find_best_hybrid import _impute_nan
    if _is_binary(X_train):
        a = X_train.astype(np.float32)
        b = X_val.astype(np.float32)
    else:
        sc = StandardScaler()
        a = sc.fit_transform(X_train)
        b = sc.transform(X_val)
    a, b, _ = _impute_nan(a, b)
    return a, b


def _make_model(name, task_type, n_samples):
    """Fresh instance, or None when the engine's size guard rejects it."""
    from kirby.find_best_hybrid import get_model
    m = get_model(name, task_type=task_type, n_samples=n_samples)
    if m is not None and hasattr(m, 'n_neighbors'):
        # Same adaptive k as _evaluate_single.
        m.set_params(n_neighbors=min(5, max(1, int(np.sqrt(n_samples)))))
    return m


def _search_space(model_name, model):
    from kirby.find_best_hybrid import _HP_SEARCH_SPACES
    grid = RIDGE_SEARCH_SPACE if model_name == 'ridge' else \
        _HP_SEARCH_SPACES.get(model_name)
    if grid is None:
        return None
    valid = set(model.get_params().keys()) if hasattr(model, 'get_params') else set()
    grid = {k: v for k, v in grid.items() if k in valid}
    return grid or None


def _tune(model_name, model, X, y, groups, task_type, metric_name):
    """RandomizedSearchCV on (X, y) — the caller passes outer-TRAIN only.

    Returns (tuned_model, seconds, best_params or None). Falls back to the
    untouched model on any failure, and says so.
    """
    from sklearn.model_selection import (RandomizedSearchCV, GroupKFold,
                                         StratifiedGroupKFold)
    from sklearn.pipeline import Pipeline
    grid = _search_space(model_name, model)
    if grid is None:
        return deepcopy(model), 0.0, None

    n_groups = len(np.unique(groups))
    folds = min(N_TUNE_FOLDS, n_groups)
    if folds < 2:
        return deepcopy(model), 0.0, None
    if task_type == 'classification':
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    else:
        cv = GroupKFold(n_splits=folds)

    n_iter = min(N_TUNE_ITER, int(np.prod([len(v) for v in grid.values()])))
    # The search must see the SAME features the outer fit sees. _prep leaves a
    # binary representation raw (matching _evaluate_single), so the search must
    # leave it raw too — otherwise the parameters are chosen for a standardised
    # ECFP and then applied to an unstandardised one.
    steps = [] if _is_binary(X) else [('scaler', StandardScaler())]
    pipe = Pipeline(steps + [('model', deepcopy(model))])
    search = RandomizedSearchCV(
        pipe, {f'model__{k}': v for k, v in grid.items()},
        n_iter=n_iter, cv=cv, scoring=_sklearn_scorer(metric_name, task_type),
        random_state=SEED, n_jobs=1, error_score=np.nan)

    t0 = time.time()
    try:
        search.fit(X, y, groups=groups)
    except Exception as e:
        return deepcopy(model), time.time() - t0, {'_failed': str(e)[:200]}
    secs = time.time() - t0
    best = search.best_estimator_.named_steps['model']
    if 'LGBM' in type(best).__name__:
        best.set_params(verbose=-1)
    elif hasattr(best, 'verbose'):
        try:
            best.set_params(verbose=0)
        except Exception:
            pass
    params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    return best, secs, params


def _fit_score(model, X_train, y_train, X_val, y_val, metric_fn):
    """One fit + one score. Returns (score, fit_seconds, error or None)."""
    t0 = time.time()
    try:
        m = deepcopy(model)
        m.fit(X_train, y_train)
        secs = time.time() - t0
        score = float(metric_fn(y_val, m.predict(X_val)))
        return score, secs, None
    except Exception as e:
        return float('nan'), time.time() - t0, f'{type(e).__name__}: {e}'[:200]


# =============================================================================
# RUN
# =============================================================================

def run_dataset(ds, metric_name, arms, model_names, n_folds, out_dir,
                probe=False, fold=None, tag=''):
    from experiments._shared import load_data, cv_split

    print(f"\n{'='*78}\n{ds}  (metric={metric_name})\n{'='*78}", flush=True)
    data = load_data(ds)
    reps = data['reps']
    y = np.asarray(data['labels'])
    groups = np.asarray(data['groups'])
    task_type = data['task_type']
    n = len(y)
    metric_fn, lower_is_better = _metric_fn(metric_name)

    rep_names = sorted(reps.keys())
    if probe:
        # Narrowest and widest rep only — enough to time the width scaling.
        widths = sorted(rep_names, key=lambda r: reps[r].shape[1])
        rep_names = [widths[0], widths[-1]]
        n_folds = 1

    splits = list(cv_split(groups, n_folds, next(iter(reps.values())), y,
                           seed=SEED, task_type=task_type))
    if fold is not None:
        if fold >= len(splits):
            raise SystemExit(f"--fold {fold} but only {len(splits)} folds exist")
        splits = [splits[fold]]
    print(f"  n={n}  reps={len(rep_names)}  folds={len(splits)}  "
          f"groups={len(np.unique(groups))}  task={task_type}", flush=True)

    X_all = np.hstack([reps[r] for r in sorted(reps.keys())])  # per-model tuning input
    rows = []

    for model_name in model_names:
        probe_model = _make_model(model_name, task_type, n)
        if probe_model is None:
            print(f"  {model_name}: SKIPPED by engine size guard "
                  f"(n={n} over _max_samples)", flush=True)
            rows.append({'dataset': ds, 'model': model_name, 'rep': None,
                         'arm': None, 'skipped': 'engine _max_samples guard'})
            continue

        # tuned_per_model: one search per (model, outer fold) on all reps glued.
        per_model_params = {}
        if 'tuned_per_model' in arms:
            for fi, (tr, _) in enumerate(splits):
                tm, secs, params = _tune(model_name, probe_model,
                                         X_all[tr], y[tr], groups[tr],
                                         task_type, metric_name)
                per_model_params[fi] = (tm, secs, params)
                print(f"  [tune per_model] {model_name} fold{fi}: "
                      f"{secs:.1f}s  {params}", flush=True)

        for rep in rep_names:
            X = reps[rep]
            for arm in arms:
                fold_scores, fit_times, tune_times, errs, chosen = [], [], [], [], []
                for fi, (tr, va) in enumerate(splits):
                    Xtr, Xva = _prep(X[tr], X[va])
                    if arm == 'default':
                        model, tsec, params = deepcopy(probe_model), 0.0, None
                    elif arm == 'tuned_per_rep':
                        model, tsec, params = _tune(
                            model_name, probe_model, X[tr], y[tr], groups[tr],
                            task_type, metric_name)
                    else:
                        model, tsec, params = per_model_params[fi]
                        tsec = tsec / max(1, len(rep_names))  # search shared across reps
                    s, fsec, err = _fit_score(model, Xtr, y[tr], Xva, y[va],
                                              metric_fn)
                    fold_scores.append(s)
                    fit_times.append(fsec)
                    tune_times.append(tsec)
                    chosen.append(params)
                    if err:
                        errs.append(err)
                row = {
                    'dataset': ds, 'model': model_name, 'rep': rep,
                    'rep_dim': int(X.shape[1]), 'arm': arm,
                    'metric': metric_name, 'lower_is_better': bool(lower_is_better),
                    'fold_scores': fold_scores,
                    'score_mean': float(np.nanmean(fold_scores)),
                    'score_std': float(np.nanstd(fold_scores)),
                    'sec_per_fit': float(np.mean(fit_times)),
                    'sec_per_tune': float(np.mean(tune_times)),
                    'best_params': chosen,
                    'n_failed_folds': int(np.sum(np.isnan(fold_scores))),
                    'errors': errs[:3],
                }
                rows.append(row)
                print(f"  {model_name:<9} {rep:<20} {X.shape[1]:>5}D "
                      f"{arm:<16} {row['score_mean']:>8.4f} "
                      f"fit={row['sec_per_fit']:>7.2f}s "
                      f"tune={row['sec_per_tune']:>8.2f}s"
                      + (f"  ERR {errs[0]}" if errs else ''), flush=True)

                out_dir.mkdir(parents=True, exist_ok=True)
                stem = 'probe' if probe else 'full'
                # The model set is ALWAYS in the filename. Without it a later
                # run over a different model list silently overwrites an
                # earlier one's table (it did, once: a dnn-only stage rewrote
                # all five eight-model files).
                suffix = f'__{tag}' if tag else '__' + '-'.join(model_names)
                with open(out_dir / f'f29_{ds}_{stem}{suffix}.json', 'w') as f:
                    json.dump({'dataset': ds, 'metric': metric_name,
                               'n_samples': int(n), 'task_type': task_type,
                               'n_outer_folds': len(splits),
                               'fold_index': fold,
                               'written': datetime.now().isoformat(),
                               'rows': rows}, f, indent=2, default=str)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--arms', nargs='+',
                    default=['default', 'tuned_per_rep', 'tuned_per_model'])
    ap.add_argument('--folds', type=int, default=N_OUTER_FOLDS)
    ap.add_argument('--probe', action='store_true',
                    help='1 fold, narrowest + widest rep — timing only')
    ap.add_argument('--fold', type=int, default=None,
                    help='run only this outer fold (0-based) — one SLURM task')
    ap.add_argument('--tag', default='',
                    help='suffix for the output filename, so array tasks do '
                         'not overwrite each other')
    ap.add_argument('--output', default=str(_ROOT / 'tests' / 'results' / 'framework'))
    args = ap.parse_args()

    out_dir = Path(args.output)
    for ds in args.datasets:
        try:
            run_dataset(ds, DATASETS[ds], args.arms, args.models, args.folds,
                        out_dir, probe=args.probe, fold=args.fold,
                        tag=args.tag)
        except Exception:
            print(f"DATASET FAILED: {ds}", flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    main()
