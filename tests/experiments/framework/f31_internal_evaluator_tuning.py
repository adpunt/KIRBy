#!/usr/bin/env python3
"""f31 — internal-evaluator settings sweep.

The internal evaluator scores each representation individually inside
allocate_performance_weighted; those scores are what the quota allocators
divide the budget by. Its hyperparameters are literals in
src/kirby/hybrid.py::_evaluate_allocation and nothing tunes them —
find_best_hybrid(tune=True) tunes the DOWNSTREAM model only, and
_HP_SEARCH_SPACES has no dnn entry at all.

This sweeps NAMED settings per evaluator (user decision 2026-09-04). Each
setting is its own arm, so the winner is a setting that can be written into
hybrid.py, not a per-fold draw.

Settings are injected through the callable branch of _evaluate_allocation
(`elif callable(internal_evaluator): m = internal_evaluator()`), which is the
same extension point f07 used. The shipped 500-row fit cap for svm/mlp is
reproduced by _CappedFit with the same seed and RandomState draw as
hybrid._subsample_for_scorer, so the *_default arms are bit-identical to the
shipped evaluator. --verify-defaults asserts that.

Fixed by the user: budget 100, importance lightgbm_gain, three quota
allocations, downstream lightgbm + rf, the 10 post-pruning representations per
dataset. Every metric is higher-is-better; each dataset is ranked on its own
primary and NEVER pooled across datasets.

Usage:
    python tests/experiments/framework/f31_internal_evaluator_tuning.py --smoke
    python tests/experiments/framework/f31_internal_evaluator_tuning.py --verify-defaults
    python tests/experiments/framework/f31_internal_evaluator_tuning.py
"""

import os
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')   # torch + LightGBM in one process
# MEASURED 2026-09-04: a LightGBM fit followed by a torch fit in the same
# process segfaults this machine outright (dnn.py:113, no traceback) — the
# reverse of the crash hybrid.lgbm_n_jobs documents, and it kills f31 because
# importance_method='lightgbm_gain' runs before every dnn arm's first fit.
# OMP_NUM_THREADS=1 survives it; KMP_DUPLICATE_LIB_OK=TRUE does not. Fitting
# torch FIRST also survives, but f29 recorded a silent 22-minute deadlock from
# that ordering, so this takes the crash-free one. Must be set before any
# import that pulls in libomp.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import faulthandler
faulthandler.enable()   # the segfault above prints nothing without this

import sys
import json
import time
import argparse
import warnings
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset -> primary metric. Every one is higher-is-better (user, 2026-09-04:
# "make sure everything's going in the same direction ... not RMSE"). The
# pipeline metric for buchwald_hartwig and caco2 is rmse; r2 is the same
# quantity with the sign and scale fixed, on the same predictions.
DATASETS = {
    'buchwald_hartwig':       'r2',
    'caco2':                  'r2',
    'flab_expression_koenig': 'spearman',
}

# rank_exp / quota, power_2 / quota, linear / quota.
ALLOCATIONS = ['performance_weighted', 'power_2', 'rank_exp_d0.3']

DOWNSTREAM_MODELS = ['lightgbm', 'rf']
BUDGET = 100
IMPORTANCE_METHOD = 'lightgbm_gain'
N_OUTER_FOLDS = 3          # reduced from 5 on user instruction, 2026-09-04
SEED = 42

# The shipped fit cap for svm/mlp (hybrid._SCORER_FIT_MAX_SAMPLES).
SHIPPED_FIT_CAP = 500


# =============================================================================
# EVALUATOR SETTINGS
# =============================================================================

class _CappedFit(BaseEstimator, RegressorMixin):
    """Fit on at most `max_rows` rows, predict on everything.

    Reproduces hybrid._subsample_for_scorer exactly: same seed, same
    RandomState.choice call, so a *_default arm carrying the shipped cap
    selects the same features as the shipped evaluator.
    """

    def __init__(self, factory=None, max_rows=None, seed=42):
        self.factory = factory
        self.max_rows = max_rows
        self.seed = seed

    def fit(self, X, y):
        if self.max_rows is not None and X.shape[0] > self.max_rows:
            idx = np.random.RandomState(self.seed).choice(
                X.shape[0], self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.model_ = self.factory()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _rf(**kw):
    from sklearn.ensemble import RandomForestRegressor
    # n_jobs is joblib threads, not OpenMP, so OMP_NUM_THREADS=1 does not cap
    # it. KIRBY_F31_RF_JOBS exists so the three datasets can run as three
    # concurrent processes without each one claiming every core.
    base = dict(random_state=42,
                n_jobs=int(os.environ.get('KIRBY_F31_RF_JOBS', '-1')))
    base.update(kw)
    return RandomForestRegressor(**base)


def _lgbm(**kw):
    from lightgbm import LGBMRegressor
    from kirby.hybrid import lgbm_n_jobs
    base = dict(random_state=42, verbosity=-1, n_jobs=lgbm_n_jobs())
    base.update(kw)
    return LGBMRegressor(**base)


def _svr(**kw):
    from sklearn.svm import SVR
    base = dict(kernel='rbf')
    base.update(kw)
    return SVR(**base)


def _mlp(**kw):
    from sklearn.neural_network import MLPRegressor
    base = dict(random_state=42, early_stopping=True, max_iter=200)
    base.update(kw)
    return MLPRegressor(**base)


def _dnn(**kw):
    from kirby.models.dnn import DNNRegressor
    # hybrid._evaluate_allocation dropped its epochs=50/patience=10 override on
    # 2026-09-04 and now builds a plain DNNRegressor, so the default arm below
    # takes the class defaults too — otherwise 'dnn_default' stops naming what
    # the framework actually runs.
    base = dict(random_state=42)
    base.update(kw)
    return DNNRegressor(**base)


# name -> (family, factory, fit-row cap). The *_default arm of every family
# reproduces the literal currently in hybrid._evaluate_allocation.
EVALUATOR_SETTINGS = {
    # rf — shipped: n_estimators=50, max_depth=10, no cap.
    'rf_default':            ('rf', lambda: _rf(n_estimators=50, max_depth=10), None),
    'rf_200_d10':            ('rf', lambda: _rf(n_estimators=200, max_depth=10), None),
    'rf_200_dNone':          ('rf', lambda: _rf(n_estimators=200, max_depth=None), None),
    'rf_200_dNone_sqrt':     ('rf', lambda: _rf(n_estimators=200, max_depth=None,
                                                max_features='sqrt'), None),

    # lightgbm — shipped: n_estimators=50, max_depth=4, no cap.
    'lgbm_default':          ('lightgbm', lambda: _lgbm(n_estimators=50, max_depth=4), None),
    'lgbm_200_dfree':        ('lightgbm', lambda: _lgbm(n_estimators=200, max_depth=-1), None),
    'lgbm_200_d10_lr0.05':   ('lightgbm', lambda: _lgbm(n_estimators=200, max_depth=10,
                                                        learning_rate=0.05), None),
    'lgbm_500_dfree_lr0.01': ('lightgbm', lambda: _lgbm(n_estimators=500, max_depth=-1,
                                                        learning_rate=0.01), None),

    # svm — shipped: C=1.0, gamma='scale', fit capped at 500 rows.
    'svm_default':           ('svm', lambda: _svr(C=1.0), SHIPPED_FIT_CAP),
    'svm_default_allrows':   ('svm', lambda: _svr(C=1.0), None),
    'svm_C10_allrows':       ('svm', lambda: _svr(C=10.0), None),
    'svm_C100_gauto_allrows': ('svm', lambda: _svr(C=100.0, gamma='auto'), None),

    # mlp — shipped: (64, 32), max_iter=200, early_stopping, capped at 500 rows.
    'mlp_default':           ('mlp', lambda: _mlp(hidden_layer_sizes=(64, 32)), SHIPPED_FIT_CAP),
    'mlp_default_allrows':   ('mlp', lambda: _mlp(hidden_layer_sizes=(64, 32)), None),
    'mlp_256_128_allrows':   ('mlp', lambda: _mlp(hidden_layer_sizes=(256, 128)), None),
    'mlp_512_allrows':       ('mlp', lambda: _mlp(hidden_layer_sizes=(512,)), None),
    'mlp_256_128_a0.01':     ('mlp', lambda: _mlp(hidden_layer_sizes=(256, 128),
                                                  alpha=0.01), None),
    'mlp_256_128_lr0.01':    ('mlp', lambda: _mlp(hidden_layer_sizes=(256, 128),
                                                  learning_rate_init=0.01), None),

    # dnn — shipped: epochs=50, patience=10, hidden (256,128,64), dropout 0.2,
    # lr 1e-3, no fit cap. Reachable in hybrid.py but in no default list, so
    # nothing has run it.
    'dnn_default':           ('dnn', lambda: _dnn(), None),
    # Was 'dnn_e200_p20'. 200/20 became the shipped default on 2026-09-04, so
    # that arm and dnn_default are now the same settings — degenerate. Re-anchored
    # onto the OLD default so the axis keeps its width and still answers the same
    # question, just from the other side: is the longer schedule worth it?
    'dnn_e50_p10':           ('dnn', lambda: _dnn(epochs=50, patience=10), None),
    'dnn_small_128_64':      ('dnn', lambda: _dnn(hidden_sizes=(128, 64)), None),
    'dnn_wide_512_256_128':  ('dnn', lambda: _dnn(hidden_sizes=(512, 256, 128)), None),
    'dnn_do0.5_lr3e-4_e100': ('dnn', lambda: _dnn(dropout=0.5, lr=3e-4, epochs=100), None),
}

# CUT 2026-09-05 (user). Not on a score — on cost. Slowest build per fold, in
# seconds, buchwald / caco2 / koenig:
#     rf_200_dNone           12786 /  5274 / 16793
#     lgbm_500_dfree_lr0.01   9751 /  7713 /  7698
#     rf_200_d10              6284 /  2308 / 13074
# against rf_200_dNone_sqrt at 215 / 122 / 121 and lgbm_default at 127 / 73 /
# 301. The point of the sweep is a setting the hybrid search can afford to run,
# and these three were 13.6 of buchwald's 21.0 hours and 15.0 of koenig's 20.4.
# rf_200_dNone_sqrt already answers the RF question: capping max_features
# scores 0.4750 vs 0.4727 (caco2), 0.6856 vs 0.6875 (buchwald), 0.7179 vs
# 0.7201 (koenig) for a fortieth to a hundred-and-fortieth of the time.
# They stay DEFINED so f31_analyse can still label the buchwald and caco2 rows
# already on disk; they are just out of the default sweep.
DROPPED_SETTINGS = {
    'rf_200_dNone',
    'rf_200_d10',
    'lgbm_500_dfree_lr0.01',
}

DEFAULT_SETTINGS = [s for s in EVALUATOR_SETTINGS if s not in DROPPED_SETTINGS]


def setting_spec(name):
    """The parameters a setting actually resolves to, as a string.

    Written into every result file. On 2026-09-04 the shipped DNN moved from
    epochs=50/patience=10 to the class defaults, which changed what
    'dnn_default' names while runs were in flight. A name is not enough to say
    what was fitted; this is.
    """
    _family, factory, cap = EVALUATOR_SETTINGS[name]
    try:
        m = factory()
        # get_params, NOT repr: sklearn's repr prints only what differs from the
        # class defaults, so DNNRegressor() rendered as "DNNRegressor()" and
        # recorded none of the settings that actually moved.
        params = m.get_params() if hasattr(m, 'get_params') else {}
        spec = (f'{type(m).__name__}(' +
                ', '.join(f'{k}={v!r}' for k, v in sorted(params.items())) + ')')
    except Exception as e:
        spec = f'<uninstantiable: {type(e).__name__}: {e}>'
    spec = ' '.join(spec.split())
    return spec if cap is None else f'{spec} [fit capped at {cap} rows]'


# The shipped evaluator name each *_default arm is supposed to reproduce.
DEFAULT_ARM_OF = {
    'rf': 'rf_default', 'lightgbm': 'lgbm_default', 'svm': 'svm_default',
    'mlp': 'mlp_default', 'dnn': 'dnn_default',
}


def make_evaluator(setting_name):
    """Return the zero-arg callable _evaluate_allocation's callable branch wants."""
    if setting_name in DROPPED_SETTINGS:
        raise ValueError(
            f'{setting_name!r} was cut on 2026-09-05 for cost (see '
            f'DROPPED_SETTINGS) and cannot be run. It stays defined only so '
            f'f31_analyse can label rows already on disk.')
    _family, factory, cap = EVALUATOR_SETTINGS[setting_name]
    if cap is None:
        return factory
    return lambda: _CappedFit(factory=factory, max_rows=cap, seed=42)


# =============================================================================
# SCORING
# =============================================================================

def _is_binary(X):
    if X.dtype == np.uint8:
        return True
    u = np.unique(X)
    return len(u) <= 2 and set(u.tolist()).issubset({0, 0.0, 1, 1.0})


def _prep(X_train, X_val):
    """Match _evaluate_single: scale continuous, leave binary raw, impute nan."""
    from kirby.find_best_hybrid import _impute_nan
    if _is_binary(X_train):
        a, b = X_train.astype(np.float32), X_val.astype(np.float32)
    else:
        sc = StandardScaler()
        a, b = sc.fit_transform(X_train), sc.transform(X_val)
    a, b, _ = _impute_nan(a, b)
    return a, b


def _score_both(y_true, y_pred):
    """r2 and spearman on the same predictions. Both higher-is-better."""
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    out = {}
    try:
        out['r2'] = float(r2_score(y_true, y_pred))
    except Exception:
        out['r2'] = float('nan')
    try:
        rho = spearmanr(y_true, y_pred)[0]
        out['spearman'] = float(rho) if rho is not None and np.isfinite(rho) else float('nan')
    except Exception:
        out['spearman'] = float('nan')
    return out


def _downstream(model_name, task_type, n_samples):
    from kirby.find_best_hybrid import get_model
    return get_model(model_name, task_type=task_type, n_samples=n_samples)


# =============================================================================
# RUN
# =============================================================================

def run_dataset(ds, settings, allocations, n_folds, out_dir, verbose=True):
    from experiments._shared import load_data, cv_split
    from kirby.hybrid import (create_hybrid, apply_feature_selection,
                              compute_feature_importance,
                              internal_holdout_split)

    cut = [s for s in settings if s in DROPPED_SETTINGS]
    if cut:
        raise ValueError(f'cut settings requested: {cut}. See DROPPED_SETTINGS.')
    primary = DATASETS[ds]
    data = load_data(ds)
    reps, y, groups = data['reps'], data['labels'], np.asarray(data['groups'])
    task_type = data['task_type']
    if task_type != 'regression':
        raise ValueError(f'{ds}: task_type={task_type!r}, f31 assumes regression')
    X0 = next(iter(reps.values()))
    rows = []

    for fold_idx, (tr, te) in enumerate(cv_split(groups, n_folds, X0, y, SEED)):
        train_reps = {k: v[tr] for k, v in reps.items()}
        val_reps = {k: v[te] for k, v in reps.items()}
        y_tr, y_te, g_tr = y[tr], y[te], groups[tr]

        # Importance does not depend on the evaluator, the allocation or the
        # budget, so it is computed once per fold and handed to every build.
        #
        # It must be computed on the SAME rows create_hybrid would use. Until
        # commit c74905b (2026-09-04, "Importance from the training part only")
        # that was every row handed to create_hybrid; it is now the internal
        # training part only, carved by hybrid.internal_holdout_split. Calling
        # that same function here is what keeps the precompute bit-identical to
        # the recomputing path — --verify-defaults asserts it, and that
        # assertion caught this change.
        t0 = time.time()
        inner_tr, _inner_val = internal_holdout_split(len(y_tr), y_tr, g_tr)
        imp = compute_feature_importance(
            {k: v[inner_tr] for k, v in train_reps.items()},
            y_tr[inner_tr], method=IMPORTANCE_METHOD)
        t_imp = time.time() - t0
        if verbose:
            print(f'[{ds} fold {fold_idx}] importance {t_imp:.1f}s '
                  f'(train {len(tr)} rows, test {len(te)})', flush=True)

        for setting in settings:
            family = EVALUATOR_SETTINGS[setting][0]
            ev = make_evaluator(setting)
            alloc_scores = None   # filled by the first allocation, reused after

            for alloc in allocations:
                t0 = time.time()
                try:
                    X_tr_h, info = create_hybrid(
                        train_reps, y_tr,
                        allocation_method=alloc,
                        budget=BUDGET,
                        importance_method=IMPORTANCE_METHOD,
                        internal_evaluator=ev,
                        precomputed_importance=imp,
                        precomputed_alloc_scores=alloc_scores,
                        groups=g_tr,
                    )
                except Exception as e:
                    rows.append({'dataset': ds, 'fold': fold_idx,
                                 'setting': setting, 'family': family,
                                 'allocation': alloc, 'error':
                                 f'{type(e).__name__}: {e}'})
                    if verbose:
                        print(f'  {setting}/{alloc}: BUILD FAILED '
                              f'{type(e).__name__}: {e}', flush=True)
                    continue
                t_build = time.time() - t0

                # The per-rep scores this evaluator produced. They do not depend
                # on the allocation method, so the first build pays for them and
                # the rest are handed them back RAW (hybrid.py returns them raw
                # for exactly this).
                if alloc_scores is None:
                    alloc_scores = info.get('allocation_baseline_scores')

                X_te_h = apply_feature_selection(val_reps, info)
                a, b = _prep(X_tr_h, X_te_h)

                for mname in DOWNSTREAM_MODELS:
                    t1 = time.time()
                    m = _downstream(mname, task_type, len(y_tr))
                    try:
                        m.fit(a, y_tr)
                        scores = _score_both(y_te, m.predict(b))
                        err = None
                    except Exception as e:
                        scores = {'r2': float('nan'), 'spearman': float('nan')}
                        err = f'{type(e).__name__}: {e}'
                    rows.append({
                        'dataset': ds, 'fold': fold_idx, 'setting': setting,
                        'family': family, 'allocation': alloc,
                        'downstream': mname, 'primary_metric': primary,
                        'primary': scores[primary],
                        'r2': scores['r2'], 'spearman': scores['spearman'],
                        'n_features': int(X_tr_h.shape[1]),
                        'allocation_dict': {k: int(v) for k, v in
                                            (info.get('allocation') or {}).items()},
                        'evaluator_rep_scores': (
                            {k: float(v) for k, v in alloc_scores.items()}
                            if alloc_scores else None),
                        'build_seconds': round(t_build, 2),
                        'fit_seconds': round(time.time() - t1, 2),
                        'error': err,
                    })
                if verbose:
                    last = [r for r in rows[-len(DOWNSTREAM_MODELS):]]
                    got = ' '.join(f"{r['downstream']}={r['primary']:.4f}"
                                   for r in last if r.get('primary') is not None)
                    print(f'  {setting}/{alloc}: build {t_build:.1f}s  {got}',
                          flush=True)

            # write after every setting so a killed run is still readable
            out = out_dir / f'f31_{ds}.json'
            out.write_text(json.dumps(
                {'dataset': ds, 'primary_metric': primary, 'budget': BUDGET,
                 'importance_method': IMPORTANCE_METHOD,
                 'settings_spec': {s: setting_spec(s) for s in settings},
                 'allocations': allocations, 'downstream': DOWNSTREAM_MODELS,
                 'n_outer_folds': n_folds, 'reps': sorted(reps),
                 'n_samples': int(len(y)),
                 'written': datetime.now().isoformat(timespec='seconds'),
                 'rows': rows}, indent=1))
    return rows


def verify_defaults(ds='caco2'):
    """Two checks, both on real data, before any grid is trusted.

    1. precomputed_importance must give the identical allocation and the
       identical selected features as letting create_hybrid compute it.
    2. Each *_default arm must give the identical allocation as the shipped
       evaluator name it reproduces.
    """
    from experiments._shared import load_data, cv_split
    from kirby.hybrid import (create_hybrid, compute_feature_importance,
                              internal_holdout_split)

    data = load_data(ds)
    reps, y, groups = data['reps'], data['labels'], np.asarray(data['groups'])
    X0 = next(iter(reps.values()))
    tr, _te = next(iter(cv_split(groups, N_OUTER_FOLDS, X0, y, SEED)))
    train_reps = {k: v[tr] for k, v in reps.items()}
    y_tr, g_tr = y[tr], groups[tr]
    inner_tr, _ = internal_holdout_split(len(y_tr), y_tr, g_tr)
    imp = compute_feature_importance(
        {k: v[inner_tr] for k, v in train_reps.items()},
        y_tr[inner_tr], method=IMPORTANCE_METHOD)

    def build(ev, use_imp):
        X, info = create_hybrid(
            train_reps, y_tr, allocation_method='performance_weighted',
            budget=BUDGET, importance_method=IMPORTANCE_METHOD,
            internal_evaluator=ev,
            precomputed_importance=(imp if use_imp else None),
            groups=g_tr)
        sel = {rn: np.asarray(v['selected_indices']).tolist()
               for rn, v in info.items()
               if isinstance(v, dict) and 'selected_indices' in v}
        return info.get('allocation'), sel

    ok = True
    a1, s1 = build('rf', False)
    a2, s2 = build('rf', True)
    same = (a1 == a2) and (s1 == s2)
    print(f'precomputed_importance identical: {same}')
    if not same:
        ok = False
        print(f'  alloc recompute={a1}\n  alloc precomputed={a2}')

    for shipped, arm in DEFAULT_ARM_OF.items():
        try:
            a_ship, s_ship = build(shipped, True)
            a_arm, s_arm = build(make_evaluator(arm), True)
        except Exception as e:
            print(f'{arm} vs shipped {shipped!r}: ERROR {type(e).__name__}: {e}')
            ok = False
            continue
        same = (a_ship == a_arm) and (s_ship == s_arm)
        print(f'{arm} reproduces shipped {shipped!r}: {same}')
        if not same:
            ok = False
            print(f'  shipped={a_ship}\n  arm    ={a_arm}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=sorted(DATASETS))
    ap.add_argument('--settings', nargs='+', default=sorted(DEFAULT_SETTINGS))
    ap.add_argument('--allocations', nargs='+', default=ALLOCATIONS)
    ap.add_argument('--folds', type=int, default=N_OUTER_FOLDS)
    ap.add_argument('--smoke', action='store_true',
                    help='caco2, fold 0, one setting per family, one allocation')
    ap.add_argument('--verify-defaults', action='store_true')
    ap.add_argument('--output', default=str(_ROOT / 'tests' / 'results' / 'framework'))
    args = ap.parse_args()

    cut = [s for s in args.settings if s in DROPPED_SETTINGS]
    if cut:
        sys.exit(f'refusing: {cut} were cut on 2026-09-05 for cost. '
                 f'They cannot be run. See DROPPED_SETTINGS in this file.')
    unknown = [s for s in args.settings if s not in EVALUATOR_SETTINGS]
    if unknown:
        sys.exit(f'unknown settings: {unknown}')

    if args.verify_defaults:
        sys.exit(0 if verify_defaults() else 1)

    if args.smoke:
        args.datasets = ['caco2']
        args.folds = 1
        args.settings = list(DEFAULT_ARM_OF.values())
        args.allocations = ['performance_weighted']

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        print(f'\n=== {ds} (primary {DATASETS[ds]}) ===', flush=True)
        t0 = time.time()
        try:
            run_dataset(ds, args.settings, args.allocations, args.folds, out_dir)
        except Exception:
            traceback.print_exc()
            continue
        print(f'=== {ds} done in {(time.time()-t0)/60:.1f} min ===', flush=True)


if __name__ == '__main__':
    main()
