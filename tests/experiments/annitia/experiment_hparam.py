"""Round 5 — Optuna hyperparameter tuning for the strongest model families.

Target: improve C-index of the strongest individual models so the fusion
pool has stronger building blocks. Tunes RSF / CatBoost-Cox / CatBoost-AFT /
GBSA / NGBoost / Coxnet-stack on the best view set found by BE.

Usage:
    python tests/experiments/annitia/experiment_hparam.py
    python tests/experiments/annitia/experiment_hparam.py --endpoint death
    python tests/experiments/annitia/experiment_hparam.py --models rsf,catboost_cox
    python tests/experiments/annitia/experiment_hparam.py --n-trials 50

Outputs to tests/results/annitia/hparam/:
    best_<model>_<endpoint>.json      best hparams + CV C-index
    oof_<endpoint>_hparam-<model>.npy  OOF predictions from refit-best model
    val_<endpoint>_hparam-<model>.npy  val_test predictions
    cv_summary.json                    per-condition records (matches the
                                       schema used by other stages so
                                       experiment_full.py picks them up
                                       automatically when fusing).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds       # noqa: E402
from experiments.annitia.views import build_views                          # noqa: E402
from experiments.annitia.survival import (                                 # noqa: E402
    SurvivalCoxPH, SurvivalCoxnet, SurvivalRSF, SurvivalGBSA,
    SurvivalNGBoost, SurvivalLightGBMAft, SurvivalCatBoostCox,
    SurvivalCatBoostAft, concordance,
)
SurvivalCatBoostAFT = SurvivalCatBoostAft
SurvivalLightGBMAFT = SurvivalLightGBMAft
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold        # noqa: E402


def _read_be_top_view_set(be_dir: Path, endpoint: str):
    """Return the highest-CV view set for the given endpoint (list of strings)."""
    cv = json.loads((be_dir / 'cv_summary.json').read_text())
    recs = [r for r in cv['per_condition'].get(endpoint, [])
            if np.isfinite(r.get('mean_cindex', float('nan')))]
    if not recs:
        raise RuntimeError(f'No BE results for {endpoint}')
    recs.sort(key=lambda r: -r['mean_cindex'])
    # Top recs grouped by view_set; pick the view that wins most often.
    # NOTE: BE writes records under key 'views'; older code used 'view_set'.
    # Support both for forward-compat.
    by_views = {}
    for r in recs[:20]:
        key = tuple(r.get('views', r.get('view_set', [])))
        if not key:
            continue
        by_views.setdefault(key, []).append(r['mean_cindex'])
    if not by_views:
        raise RuntimeError(
            f'No BE records had a usable view list for {endpoint}; '
            f'expected key "views" or "view_set" on each record.')
    best_views = max(by_views, key=lambda k: max(by_views[k]))
    return list(best_views)


# ----------------------------------------------------------------------
# Per-model search spaces (Optuna trial → estimator factory)
# ----------------------------------------------------------------------
def _space_rsf(trial):
    # SurvivalRSF wrapper only accepts these constructor args (no max_depth).
    return SurvivalRSF(
        n_estimators=trial.suggest_categorical('n_estimators', [300, 500, 1000, 2000]),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 30),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
    )


def _space_catboost_cox(trial):
    return SurvivalCatBoostCox(
        iterations=trial.suggest_categorical('iterations', [500, 1000, 2000]),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        depth=trial.suggest_int('depth', 4, 10),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
    )


def _space_catboost_aft(trial):
    return SurvivalCatBoostAFT(
        iterations=trial.suggest_categorical('iterations', [500, 1000, 2000]),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        depth=trial.suggest_int('depth', 4, 10),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
    )


def _space_gbsa(trial):
    return SurvivalGBSA(
        n_estimators=trial.suggest_categorical('n_estimators', [200, 500, 1000, 2000]),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        max_depth=trial.suggest_int('max_depth', 2, 8),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
    )


def _space_ngboost(trial):
    # SurvivalNGBoost wrapper only accepts n_estimators / learning_rate / dist.
    return SurvivalNGBoost(
        n_estimators=trial.suggest_categorical('n_estimators', [200, 500, 1000]),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        dist=trial.suggest_categorical('dist', ['logNormal', 'exponential']),
    )


def _space_lgbm_aft(trial):
    # SurvivalLightGBMAft wrapper accepts these only (no min_data_in_leaf).
    return SurvivalLightGBMAFT(
        n_estimators=trial.suggest_categorical('n_estimators', [200, 500, 1000, 2000]),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        num_leaves=trial.suggest_int('num_leaves', 15, 255),
        sigma=trial.suggest_float('sigma', 0.5, 2.0),
    )


def _space_coxnet(trial):
    return SurvivalCoxnet(
        l1_ratio=trial.suggest_float('l1_ratio', 0.05, 0.95),
        alpha_min_ratio=trial.suggest_float('alpha_min_ratio', 1e-4, 0.1, log=True),
    )


SPACES = {
    'rsf':            _space_rsf,
    'catboost_cox':   _space_catboost_cox,
    'catboost_aft':   _space_catboost_aft,
    'gbsa':           _space_gbsa,
    'ngboost':        _space_ngboost,
    'lgbm_aft':       _space_lgbm_aft,
    'coxnet':         _space_coxnet,
}


def _objective(trial, X, y, valid, folds, factory):
    fold_c = []
    for tr, va in folds:
        try:
            est = factory(trial)
            est.fit(X[tr], y[tr])
            risk = est.predict_risk(X[va])
            c = concordance(y[va], risk)
            if not np.isfinite(c):
                raise RuntimeError('non-finite concordance')
            fold_c.append(c)
        except Exception as e:
            # Any fit failure → bad trial
            return 0.0
    return float(np.mean(fold_c))


def _refit_with_best(best_params, factory_template, X, y, valid, folds, n_train):
    """Build estimator from a fake trial that just returns suggested values,
    fit on all CV folds for OOF + refit on full valid for val_test."""
    class _ConstTrial:
        def __init__(self, params): self._p = params
        def suggest_categorical(self, name, *_): return self._p[name]
        def suggest_int(self, name, *_): return self._p[name]
        def suggest_float(self, name, *_, **__): return self._p[name]

    oof = init_oof_buffer(n_train)
    fold_c = []
    for tr, va in folds:
        est = factory_template(_ConstTrial(best_params))
        est.fit(X[tr], y[tr])
        risk = est.predict_risk(X[va])
        fold_c.append(concordance(y[va], risk))
        fill_oof_fold(oof, va, risk)
    vidx = np.where(valid)[0]
    refit = factory_template(_ConstTrial(best_params))
    refit.fit(X[vidx], y[vidx])
    val_risk = None  # filled by caller (needs X_va_full)
    return oof, np.array(fold_c), refit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--endpoint', default='both', choices=('hepatic', 'death', 'both'))
    ap.add_argument('--models', default='rsf,catboost_cox,catboost_aft,gbsa,ngboost,lgbm_aft,coxnet',
                    help='Comma-separated subset of: rsf,catboost_cox,catboost_aft,'
                         'gbsa,ngboost,lgbm_aft,coxnet')
    ap.add_argument('--n-trials', type=int, default=30,
                    help='Optuna trials per (model, endpoint) — default 30.')
    ap.add_argument('--n-folds', type=int, default=3,
                    help='Folds per trial during search (default 3 for speed); '
                         'refit-best uses n_folds_refit folds.')
    ap.add_argument('--n-folds-refit', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--be-dir', default='tests/results/annitia/be')
    ap.add_argument('--output-dir', default='tests/results/annitia/hparam')
    args = ap.parse_args()

    try:
        import optuna
    except ImportError as e:
        print(f'optuna not installed — pip install optuna ({e})')
        sys.exit(1)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # Mirror the cv_summary.json schema used by other stages so
    # experiment_full.py automatically pools these results.
    summary_path = out / 'cv_summary.json'
    summary = (json.loads(summary_path.read_text())
               if summary_path.exists()
               else {'per_condition': {'hepatic': [], 'death': []}})
    done = {ep: {r['config_id']: r for r in summary['per_condition'].get(ep, [])}
            for ep in ('hepatic', 'death')}

    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)
    n_train = len(train['patient_ids'])
    eps = ('hepatic', 'death') if args.endpoint == 'both' else (args.endpoint,)
    models = [m.strip() for m in args.models.split(',') if m.strip() in SPACES]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_start = time.time()
    for endpoint in eps:
        primary = _read_be_top_view_set(Path(args.be_dir), endpoint)
        print(f'\n=== {endpoint}: top view set = {primary} ===')
        views_train = build_views(train, view_names=primary)
        views_val   = build_views(val,   view_names=primary)
        X_tr_full = np.concatenate([views_train[v][0] for v in primary], axis=1)
        X_va_full = np.concatenate([views_val[v][0]   for v in primary], axis=1)
        y_full = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        folds_search = get_cv_folds(train, endpoint=endpoint,
                                    n_folds=args.n_folds, seed=args.seed)
        folds_refit = get_cv_folds(train, endpoint=endpoint,
                                   n_folds=args.n_folds_refit, seed=args.seed)

        for model in models:
            cid = f'hparam-{model}__{"+".join(primary)}'
            existing = done[endpoint].get(cid)
            oof_path = out / f'oof_{endpoint}_{cid}.npy'
            val_path = out / f'val_{endpoint}_{cid}.npy'
            if (existing is not None and oof_path.exists() and val_path.exists()
                    and np.isfinite(existing.get('mean_cindex', float('nan')))):
                print(f'  {model:15s} CACHED  C={existing["mean_cindex"]:.4f}')
                continue
            print(f'  {model:15s} searching ({args.n_trials} trials × '
                  f'{args.n_folds} folds)...')
            t_search = time.time()
            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=args.seed))
            factory = SPACES[model]
            try:
                study.optimize(
                    lambda trial: _objective(trial, X_tr_full, y_full, valid,
                                             folds_search, factory),
                    n_trials=args.n_trials, show_progress_bar=False,
                    catch=(Exception,))
            except Exception as e:
                print(f'    ERR during search: {e}; skipping {model}')
                continue
            elapsed_search = time.time() - t_search
            best = study.best_params
            best_c = study.best_value
            print(f'    best CV-{args.n_folds} C={best_c:.4f}  '
                  f'({elapsed_search:.0f}s, params={best})')
            # Save best params
            (out / f'best_{model}_{endpoint}.json').write_text(
                json.dumps({'best_params': best,
                            'best_cv_c': float(best_c),
                            'n_trials': args.n_trials,
                            'view_set': primary,
                            'elapsed_search_sec': elapsed_search}, indent=2))

            # Refit on full 5-fold for OOF + val_test prediction
            try:
                oof, fold_c, refit = _refit_with_best(
                    best, factory, X_tr_full, y_full, valid, folds_refit, n_train)
                val_risk = np.asarray(refit.predict_risk(X_va_full),
                                      dtype=np.float64)
                np.save(oof_path, oof); np.save(val_path, val_risk)
                rec = {
                    'config_id': cid,
                    'label': f'hparam-{model}',
                    'view_set': primary,
                    'mean_cindex': float(np.nanmean(fold_c)),
                    'std_cindex':  float(np.nanstd(fold_c)),
                    'fold_cindex': fold_c.tolist(),
                    'best_params': best,
                    'elapsed_sec': time.time() - t_search,
                }
                summary['per_condition'][endpoint] = [
                    r for r in summary['per_condition'][endpoint]
                    if r.get('config_id') != cid]
                summary['per_condition'][endpoint].append(rec)
                done[endpoint][cid] = rec
                summary_path.write_text(json.dumps(summary, indent=2))
                print(f'  {model:15s} refit-{args.n_folds_refit} '
                      f'C={rec["mean_cindex"]:.4f}±{rec["std_cindex"]:.4f}')
            except Exception as e:
                print(f'    refit failed: {type(e).__name__}: {e}')

    print(f'\nDONE in {(time.time() - t_start)/60:.1f} min')
    print(f'  cv_summary: {summary_path}')


if __name__ == '__main__':
    main()
