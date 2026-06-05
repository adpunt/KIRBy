"""Aggressive hep-only Optuna tuning, 60-min Mac CPU budget.

Tunes RSF / CatBoost-Cox / GBSA / Coxnet / LightGBM-AFT with much wider
search spaces than experiment_hparam.py, across 3 hand-picked view sets.
Saves OOFs as oof_hepatic_aggressive_<algo>_<viewset>.npy under
tests/results/annitia/branches/.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))
warnings.filterwarnings('ignore')

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds  # noqa
from experiments.annitia.views import build_views                    # noqa
from experiments.annitia.survival import (                            # noqa
    SurvivalRSF, SurvivalCatBoostCox, SurvivalGBSA,
    SurvivalCoxnet, SurvivalLightGBMAft, concordance,
)
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold   # noqa


VIEW_SETS = {
    'cur':  ['snapshot', 'clinical_pro', 'fpca', 'gp_smooth'],
    'vd':   ['snapshot', 'clinical_pro', 'fpca', 'visit_density', 'clinical_slope'],
    'meld': ['snapshot', 'clinical_pro', 'meld_na', 'visit_density'],
    'all6': ['snapshot', 'clinical_pro', 'fpca', 'gp_smooth', 'visit_density', 'clinical_slope'],
}


def _space_rsf(trial):
    md = trial.suggest_categorical('max_depth', [4, 8, 12, 20, None])
    mf = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.5])
    return SurvivalRSF(
        n_estimators=trial.suggest_categorical('n_estimators', [300, 500, 1000]),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 12),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 8),
        max_features=mf,
        # SurvivalRSF doesn't accept max_depth. We monkey-patch via subclass below.
    ), {'max_depth': md}


def _make_rsf(params):
    """Build a deep-tree RSF using sksurv directly (max_depth is critical)."""
    from sksurv.ensemble import RandomSurvivalForest

    class _RSFDeep:
        def __init__(self, p):
            self.p = p
            self._m = None
        def fit(self, X, y):
            mf = self.p['max_features']
            self._m = RandomSurvivalForest(
                n_estimators=self.p['n_estimators'],
                min_samples_split=self.p['min_samples_split'],
                min_samples_leaf=self.p['min_samples_leaf'],
                max_features=mf,
                max_depth=self.p['max_depth'],
                n_jobs=-1, random_state=42)
            self._m.fit(np.asarray(X, np.float64), y)
            return self
        def predict_risk(self, X):
            return np.asarray(self._m.predict(np.asarray(X, np.float64)), np.float64)
    return _RSFDeep(params)


def _space_catboost_cox(trial):
    return SurvivalCatBoostCox(
        iterations=trial.suggest_categorical('iterations', [500, 1000, 2000]),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        depth=trial.suggest_int('depth', 3, 10),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 20.0, log=True),
    ), {}


def _space_gbsa(trial):
    return SurvivalGBSA(
        n_estimators=trial.suggest_categorical('n_estimators', [300, 600, 1000]),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
    ), {}


def _space_coxnet(trial):
    return SurvivalCoxnet(
        l1_ratio=trial.suggest_float('l1_ratio', 0.1, 0.99),
        alpha_min_ratio=trial.suggest_float('alpha_min_ratio', 1e-4, 0.1, log=True),
    ), {}


def _space_lgbm_aft(trial):
    return SurvivalLightGBMAft(
        n_estimators=trial.suggest_categorical('n_estimators', [300, 600, 1000, 2000]),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        num_leaves=trial.suggest_int('num_leaves', 15, 127),
        sigma=trial.suggest_float('sigma', 0.5, 2.0),
    ), {}


SPACES = {
    'rsf':           ('rsf_deep', _space_rsf),
    'catboost_cox':  ('std',      _space_catboost_cox),
    'gbsa':          ('std',      _space_gbsa),
    'coxnet':        ('std',      _space_coxnet),
    'lgbm_aft':      ('std',      _space_lgbm_aft),
}


def _objective(trial, X, y, folds, kind, space_fn):
    est_proto, extra = space_fn(trial)
    fold_c = []
    for tr, va in folds:
        try:
            if kind == 'rsf_deep':
                params = {**est_proto.__dict__, **extra}
                est = _make_rsf(params)
            else:
                # rebuild a fresh instance per fold
                est = est_proto.__class__(**{k: v for k, v in est_proto.__dict__.items()
                                             if not k.startswith('_')})
            est.fit(X[tr], y[tr])
            r = est.predict_risk(X[va])
            c = concordance(y[va], r)
            if not np.isfinite(c):
                return 0.0
            fold_c.append(c)
        except Exception:
            return 0.0
    return float(np.mean(fold_c))


def _refit_best(best_trial, X, y, folds, kind, space_fn, n_train):
    """Use the best trial directly to rebuild estimator (preserves all params)."""
    est_proto, extra = space_fn(best_trial)
    oof = init_oof_buffer(n_train)
    fold_c = []
    for tr, va in folds:
        if kind == 'rsf_deep':
            params = {**est_proto.__dict__, **extra}
            est = _make_rsf(params)
        else:
            est = est_proto.__class__(**{k: v for k, v in est_proto.__dict__.items()
                                         if not k.startswith('_')})
        est.fit(X[tr], y[tr])
        r = est.predict_risk(X[va])
        fold_c.append(concordance(y[va], r))
        fill_oof_fold(oof, va, r)
    return oof, np.array(fold_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='rsf,catboost_cox,gbsa,coxnet,lgbm_aft')
    ap.add_argument('--view-sets', default='cur,vd,meld')
    ap.add_argument('--n-trials', type=int, default=25)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--time-budget-min', type=float, default=55.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='tests/results/annitia/branches')
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = _REPO / 'AUDIT_HEP_AGGRESSIVE.md'

    train = load_annitia()
    n_train = len(train['patient_ids'])
    y = train['hepatic']['y_struct']
    folds = get_cv_folds(train, endpoint='hepatic', n_folds=args.n_folds, seed=args.seed)

    models = [m.strip() for m in args.models.split(',') if m.strip() in SPACES]
    view_sets = [v.strip() for v in args.view_sets.split(',') if v.strip() in VIEW_SETS]

    # Pre-build views for each view-set
    view_X = {}
    for vk in view_sets:
        v = build_views(train, view_names=VIEW_SETS[vk])
        X = np.concatenate([v[n][0] for n in VIEW_SETS[vk]], axis=1)
        view_X[vk] = X
        print(f'  view-set {vk}: shape={X.shape}')

    t0 = time.time()
    budget_s = args.time_budget_min * 60.0
    results = []

    # Order: prioritise rsf+cur (proven lift), then sweep
    order = []
    for m in ['rsf', 'gbsa', 'catboost_cox', 'coxnet', 'lgbm_aft']:
        if m not in models: continue
        for vk in ['cur', 'vd', 'meld', 'all6']:
            if vk in view_sets:
                order.append((m, vk))

    for model, vk in order:
        elapsed = time.time() - t0
        if elapsed > budget_s:
            print(f'\nBUDGET EXCEEDED ({elapsed/60:.1f} min); stopping.')
            break
        remaining = budget_s - elapsed
        # cap per-cell time at ~15% of remaining budget if many cells left
        remaining_cells = max(1, len(order) - len(results))
        cell_budget = max(120.0, remaining * 1.0 / remaining_cells)
        print(f'\n[{elapsed/60:5.1f}/{budget_s/60:.0f}m] {model:14s} × {vk:5s} '
              f'(cell budget ~{cell_budget/60:.1f}m, n_trials={args.n_trials})')

        kind, space_fn = SPACES[model]
        X = view_X[vk]
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        try:
            study.optimize(
                lambda tr: _objective(tr, X, y, folds, kind, space_fn),
                n_trials=args.n_trials,
                timeout=cell_budget,
                catch=(Exception,),
                show_progress_bar=False,
            )
        except Exception as e:
            print(f'  search failed: {e}')
            continue
        if not study.best_trial or not np.isfinite(study.best_value):
            print(f'  no valid trials')
            continue

        best_c = study.best_value
        best_p = study.best_params
        print(f'  best CV C={best_c:.4f}  params={best_p}')

        # Refit + save OOF
        try:
            oof, fold_c = _refit_best(study.best_trial, X, y, folds, kind, space_fn, n_train)
            cid = f'aggressive_{model}_{vk}'
            oof_path = out_dir / f'oof_hepatic_{cid}.npy'
            np.save(oof_path, oof)

            # Recompute concordance from the refit OOF (same folds, will match best_c closely)
            valid = train['hepatic']['valid_mask']
            m = valid & np.isfinite(oof)
            from sksurv.metrics import concordance_index_censored
            ev = y['event'][m].astype(bool); tt = y['time'][m]
            c_oof, *_ = concordance_index_censored(ev, tt, oof[m])
            print(f'  OOF C={c_oof:.4f}  saved → {oof_path.name}')
            results.append({
                'cid': cid, 'model': model, 'view_set': vk,
                'cv_c': float(best_c), 'oof_c': float(c_oof),
                'fold_c': fold_c.tolist(),
                'best_params': {k: (v if not (isinstance(v, float) and not np.isfinite(v)) else None)
                                for k, v in best_p.items()},
                'n_trials': len(study.trials),
            })
        except Exception as e:
            print(f'  refit failed: {type(e).__name__}: {e}')

    # Aggregate (rank-mean) across new aggressive OOFs
    valid = train['hepatic']['valid_mask']
    new_oofs = []
    for r in results:
        oof = np.load(out_dir / f"oof_hepatic_{r['cid']}.npy")
        new_oofs.append((r['cid'], r['oof_c'], oof))

    agg_c = float('nan')
    if new_oofs:
        from scipy.stats import rankdata
        from sksurv.metrics import concordance_index_censored
        ranks = []
        for _, _, oof in new_oofs:
            r = np.full(len(oof), np.nan)
            m = np.isfinite(oof)
            r[m] = rankdata(oof[m])
            ranks.append(r)
        R = np.nanmean(np.stack(ranks, axis=0), axis=0)
        m = valid & np.isfinite(R)
        ev = y['event'][m].astype(bool); tt = y['time'][m]
        agg_c, *_ = concordance_index_censored(ev, tt, R[m])

    # Write audit
    lines = ['# Hep Aggressive Tuning — Audit', '']
    lines.append(f'Total wall time: {(time.time()-t0)/60:.1f} min')
    lines.append(f'Cells run: {len(results)} / planned {len(order)}')
    lines.append(f'Baseline (synaug-none-rsf-500-n0 OOF): 0.8681')
    lines.append(f'Prior best hparam RSF refit: 0.8815')
    lines.append('')
    lines.append('## Per-cell results (sorted by OOF C)')
    lines.append('| algo | viewset | CV-5 C | OOF C | n_trials |')
    lines.append('|------|---------|--------|-------|----------|')
    for r in sorted(results, key=lambda x: -x['oof_c']):
        lines.append(f"| {r['model']} | {r['view_set']} | "
                     f"{r['cv_c']:.4f} | {r['oof_c']:.4f} | {r['n_trials']} |")
    lines.append('')
    lines.append(f'## Rank-mean aggregate of new aggressive OOFs')
    lines.append(f'C-index = **{agg_c:.4f}**')
    lines.append(f'(target: beat 0.8815)')
    lines.append('')
    lines.append('## Best params per algo (best across view-sets)')
    by_algo = {}
    for r in results:
        if r['model'] not in by_algo or r['oof_c'] > by_algo[r['model']]['oof_c']:
            by_algo[r['model']] = r
    for m, r in sorted(by_algo.items(), key=lambda kv: -kv[1]['oof_c']):
        lines.append(f"### {m} (viewset={r['view_set']}, OOF C={r['oof_c']:.4f})")
        lines.append('```json')
        lines.append(json.dumps(r['best_params'], indent=2))
        lines.append('```')
    audit_path.write_text('\n'.join(lines) + '\n')
    print(f'\nAudit → {audit_path}')

    # Build sub_master_aggressive_hep.csv
    sub_dir = _REPO / 'tests/results/annitia/full'
    if results:
        # Average new aggressive hep OOFs (rank-mean across patients)
        # For test set we need a fitted model. Approach: refit the BEST (top OOF)
        # on full data and predict on val_test. To keep this fast and aligned
        # with existing behavior, we average the val_test predictions of the
        # newly-fit models.
        # Simpler: write a sub that uses BEST_BET death + the rank-mean of
        # aggressive hep OOFs PROJECTED via val_test. But val_test predictions
        # weren't saved here. Fall back: train on all train data once per
        # (model, vk), predict on val_test.
        try:
            val = load_annitia(
                data_dir=str(Path('/Volumes/seagate/kirby/annitia')),
                filename='val_test.csv', has_outcomes=False)
            preds = []
            for r in results:
                vk = r['view_set']
                vviews = build_views(val, view_names=VIEW_SETS[vk])
                X_va = np.concatenate([vviews[n][0] for n in VIEW_SETS[vk]], axis=1)
                X_tr = view_X[vk]
                # rebuild best trial -> est, fit on full, predict
                kind, space_fn = SPACES[r['model']]
                # Reconstruct via FixedTrial
                ft = optuna.trial.FixedTrial(r['best_params'])
                est_proto, extra = space_fn(ft)
                if kind == 'rsf_deep':
                    params = {**est_proto.__dict__, **extra}
                    est = _make_rsf(params)
                else:
                    est = est_proto.__class__(**{k: v for k, v in est_proto.__dict__.items()
                                                 if not k.startswith('_')})
                vmask = train['hepatic']['valid_mask']
                est.fit(X_tr[vmask], y[vmask])
                p = est.predict_risk(X_va)
                preds.append(p.astype(np.float64))
            P = np.stack(preds, axis=0)
            # rank-mean across models for val_test
            from scipy.stats import rankdata
            R = np.stack([rankdata(p) for p in P], axis=0)
            hep_va = R.mean(axis=0)
            hep_va = (hep_va - hep_va.min()) / (hep_va.max() - hep_va.min() + 1e-12)

            # Anchor death from BEST_BET
            import pandas as pd
            anchor = pd.read_csv(sub_dir / 'sub_master_BEST_BET.csv')
            sub = pd.DataFrame({
                'trustii_id': anchor['trustii_id'],
                'risk_hepatic_event': hep_va,
                'risk_death': anchor['risk_death'],
            })
            out_csv = sub_dir / 'sub_master_aggressive_hep.csv'
            sub.to_csv(out_csv, index=False)
            print(f'Submission → {out_csv}')
        except Exception as e:
            print(f'submission build failed: {type(e).__name__}: {e}')

    print(f'\nDONE in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
