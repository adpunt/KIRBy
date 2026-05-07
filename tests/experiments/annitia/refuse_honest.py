"""Honest re-fusion: nested-CV evaluation of every (K, mode) combination.

Round 3's submission scored Trustii test C=0.7181 vs reported OOF C=0.98.
The gap is in the fusion stage:

    experiment_full.py fits stack_meta Coxnet on (oof_list, y) USING ALL
    TRAINING ROWS, then predicts back on the same oof_list and computes
    in-sample concordance. With ~138 base models as features the
    meta-learner overfits dramatically; in-sample C is wildly optimistic.

    Worse, fusion.py picks coefs[:, -1] along the alpha path, which is
    the SMALLEST alpha = LEAST regularised = MOST overfit Coxnet
    solution.

This script re-evaluates the K x mode sweep with TRUE OOF estimates:

    For learned-weight modes (stack_meta, stack_meta_xe):
        Run a 5-fold OUTER CV over the training rows. For each outer
        fold: fit Coxnet on the 4 training folds' per-branch OOFs +
        labels, predict on the held-out fold. Concatenate. Concordance
        on these out-of-fold predictions is honest.

    For non-learned modes (top1, equal, cv_weighted, single, rrf):
        OOF predictions are already clean (each row is from a base
        model that didn't see it). Concordance is honest as-is.

Also clamps the Coxnet alpha to a less-aggressive index (the median of
the path, not the last column).

Outputs a new submission to:
    tests/results/annitia/full/sub_master_honest.csv

Usage:
    python tests/experiments/annitia/refuse_honest.py
    python tests/experiments/annitia/refuse_honest.py --outer-folds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds       # noqa: E402
from experiments.annitia.fusion import (                                   # noqa: E402
    fuse_equal, fuse_cv_weighted, fuse_single, fuse_rrf,
    rank_matrix, _ordinal_rank,
)
from experiments.annitia.survival import concordance                       # noqa: E402


K_SWEEP = [3, 5, 10, 20, 40, 'all']
MODES_NO_STACK = ('top1', 'equal', 'cv_weighted', 'single', 'rrf')
ALPHA_INDEX_FRAC = 0.5  # mid of path → moderate regularisation


def _collect_runs(pool_dirs, endpoint):
    runs = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists():
            continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r.get('config_id', '')
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc):
                continue
            oof = d / f'oof_{endpoint}_{cid}.npy'
            val = d / f'val_{endpoint}_{cid}.npy'
            if not (oof.exists() and val.exists()):
                continue
            runs.append({
                'label': f'{d.name}__{cid}',
                'cv': float(mc),
                'oof_path': str(oof), 'val_path': str(val),
                'source': d.name,
            })
    runs.sort(key=lambda r: -r['cv'])
    return runs


def _select(recs, k):
    if k == 'all':
        return recs
    return recs[:k]


def _coxnet_fit_predict(X_train, y_train, X_test, l1_ratio=0.5,
                        alpha_min_ratio=0.01, alpha_index_frac=0.5,
                        max_iter=2000):
    """Fit Coxnet on (X_train, y_train), predict risk on X_test using a
    moderately-regularised alpha (default median of the alpha path).

    Returns predictions, or NaN array on failure.
    """
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        # Drop rows with non-finite features or labels (defensive).
        finite = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train['time'])
        Xfit = X_train[finite]; yfit = y_train[finite]
        if Xfit.shape[0] < X_train.shape[1] + 5 or not yfit['event'].any():
            return np.full(X_test.shape[0], np.nan)
        model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio,
            max_iter=max_iter, fit_baseline_model=False)
        model.fit(Xfit, yfit)
        coefs = np.asarray(model.coef_)
        if coefs.ndim == 2:
            idx = max(0, min(coefs.shape[1] - 1,
                             int(round((coefs.shape[1] - 1) * alpha_index_frac))))
            w = coefs[:, idx]
        else:
            w = coefs
        if not np.any(np.abs(w) > 1e-12):
            # All-zero coefs at this alpha → fall back to uniform.
            w = np.ones(X_train.shape[1]) / X_train.shape[1]
        # Replace any non-finite test rows with column means
        Xtest = X_test.copy()
        col_means = np.nanmean(Xfit, axis=0)
        for j in range(Xtest.shape[1]):
            mask = ~np.isfinite(Xtest[:, j])
            if mask.any():
                Xtest[mask, j] = col_means[j]
        return Xtest @ w.astype(np.float64)
    except Exception as e:
        print(f'    Coxnet fit failed: {type(e).__name__}: {e}')
        return np.full(X_test.shape[0], np.nan)


def _nested_oof_stack(oof_self, oof_cross, y, valid, n_folds=5, seed=42):
    """Honest OOF prediction for a Coxnet meta-learner.

    Splits training rows into n_folds (StratifiedKFold on event), fits
    Coxnet on training subset, predicts on held-out subset.

    Returns (oof_pred, n_used_rows).
    """
    n = len(y)
    R_self = rank_matrix(oof_self)
    if oof_cross:
        R_cross = rank_matrix(oof_cross)
        R = np.concatenate([R_self, R_cross], axis=1)
    else:
        R = R_self
    finite_rows = np.all(np.isfinite(R), axis=1) & np.isfinite(y['time'])
    keep = finite_rows & valid
    if keep.sum() < R.shape[1] + 10:
        return np.full(n, np.nan), 0
    keep_idx = np.where(keep)[0]
    R_k = R[keep_idx]; y_k = y[keep_idx]
    # Build folds on keep_idx
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan)
    for tr_local, va_local in skf.split(keep_idx, y_k['event'].astype(int)):
        Xtr = R_k[tr_local]; ytr = y_k[tr_local]
        Xva = R_k[va_local]
        pred = _coxnet_fit_predict(Xtr, ytr, Xva)
        oof[keep_idx[va_local]] = pred
    return oof, int(keep.sum())


def _evaluate_endpoint(train, val, endpoint, runs_self, runs_cross,
                       outer_folds=5, seed=42, val_trustii_id=None):
    y = train[endpoint]['y_struct']
    valid = train[endpoint]['valid_mask']

    # For each (K, mode) build trial list
    trials = []
    print(f'\n=== {endpoint}: pool {len(runs_self)} (self), '
          f'{len(runs_cross)} (cross) ===')

    for K in K_SWEEP:
        recs = _select(runs_self, K)
        if not recs:
            continue
        oof_list = [np.load(r['oof_path']) for r in recs]
        val_list = [np.load(r['val_path']) for r in recs]
        cv_list = [r['cv'] for r in recs]
        weights = np.maximum(np.asarray(cv_list) - 0.5, 0.0)
        if weights.sum() == 0:
            weights = np.ones(len(recs))

        # ---- Non-learned modes: existing OOFs are already honest ----
        for mode_name, oof_combo, val_combo in (
            ('top1',
                oof_list[int(np.argmax(cv_list))],
                val_list[int(np.argmax(cv_list))]),
            ('equal',        fuse_equal(oof_list),        fuse_equal(val_list)),
            ('cv_weighted',
                fuse_cv_weighted(oof_list, weights),
                fuse_cv_weighted(val_list, weights)),
            ('single',
                fuse_single(oof_list, cv_list),
                fuse_single(val_list, cv_list)),
            ('rrf',          fuse_rrf(oof_list, k=60.0),  fuse_rrf(val_list, k=60.0)),
        ):
            try:
                fmask = valid & np.isfinite(oof_combo) & np.isfinite(y['time'])
                c = concordance(y[fmask], oof_combo[fmask])
            except Exception:
                c = float('nan')
            trials.append({'K': K, 'mode': mode_name, 'oof_c': float(c),
                           'val': val_combo})

        # ---- Learned modes: nested-CV on training, refit-best on full ----
        for mode_name, cross_oof, cross_val in (
            ('stack_meta', None, None),
            ('stack_meta_xe',
                [np.load(r['oof_path']) for r in _select(runs_cross, K)],
                [np.load(r['val_path']) for r in _select(runs_cross, K)]),
        ):
            if mode_name == 'stack_meta_xe' and not runs_cross:
                continue
            oof_pred, n_used = _nested_oof_stack(
                oof_list, cross_oof, y, valid, n_folds=outer_folds, seed=seed)
            try:
                fmask = valid & np.isfinite(oof_pred) & np.isfinite(y['time'])
                c = concordance(y[fmask], oof_pred[fmask])
            except Exception:
                c = float('nan')
            # Refit on full training and predict val
            R_self_full = rank_matrix(oof_list)
            R_self_val  = rank_matrix(val_list)
            if cross_oof is not None:
                R_cross_full = rank_matrix(cross_oof)
                R_cross_val  = rank_matrix(cross_val)
                X_full = np.concatenate([R_self_full, R_cross_full], axis=1)
                X_val  = np.concatenate([R_self_val,  R_cross_val],  axis=1)
            else:
                X_full = R_self_full; X_val = R_self_val
            keep_full = np.all(np.isfinite(X_full), axis=1) & valid
            val_pred = _coxnet_fit_predict(X_full[keep_full], y[keep_full], X_val)
            trials.append({'K': K, 'mode': mode_name, 'oof_c': float(c),
                           'val': val_pred, 'n_used': n_used})
            print(f'    K={str(K):>3} mode={mode_name:14s} '
                  f'TRUE-OOF C={c:.4f}  (used {n_used} rows)')

        # Print non-learned summary line
        non_stack_for_K = [t for t in trials if t['K'] == K
                           and t['mode'] in MODES_NO_STACK
                           and np.isfinite(t['oof_c'])]
        if non_stack_for_K:
            best_ns = max(non_stack_for_K, key=lambda t: t['oof_c'])
            print(f'    K={str(K):>3} best non-stack: {best_ns["mode"]:14s} '
                  f'C={best_ns["oof_c"]:.4f}')

    finite = [t for t in trials if np.isfinite(t['oof_c'])]
    if not finite:
        return None, []
    best = max(finite, key=lambda t: t['oof_c'])
    no_stack = [t for t in finite if t['mode'] in MODES_NO_STACK]
    best_ns = max(no_stack, key=lambda t: t['oof_c']) if no_stack else None
    print(f'\n  BEST overall : K={best["K"]} mode={best["mode"]} '
          f'TRUE-OOF C={best["oof_c"]:.4f}')
    if best_ns:
        print(f'  BEST non-stack: K={best_ns["K"]} mode={best_ns["mode"]} '
              f'TRUE-OOF C={best_ns["oof_c"]:.4f}')
    return best, trials


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--outer-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='tests/results/annitia/full/sub_master_honest.csv')
    args = ap.parse_args()

    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = [res_root / p for p in
                 ('be', 'cd', 'hep_boost', 'branches', 'external')
                 if (res_root / p).exists()]
    print(f'Pooling from: {[d.name for d in pool_dirs]}')

    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)

    runs_hep = _collect_runs(pool_dirs, 'hepatic')
    runs_dea = _collect_runs(pool_dirs, 'death')
    print(f'  hepatic pool: {len(runs_hep)} models')
    print(f'  death  pool: {len(runs_dea)} models')

    master = pd.DataFrame({'trustii_id': val['trustii_id']})
    summary = {}
    for endpoint, col, runs_self, runs_cross in (
        ('hepatic', 'risk_hepatic_event', runs_hep, runs_dea),
        ('death',   'risk_death',         runs_dea, runs_hep),
    ):
        best, trials = _evaluate_endpoint(
            train, val, endpoint, runs_self, runs_cross,
            outer_folds=args.outer_folds, seed=args.seed)
        if best is None:
            print(f'  {endpoint}: no usable trials, filling 0')
            master[col] = 0.0
            continue
        master[col] = best['val']
        summary[endpoint] = {
            'K': best['K'], 'mode': best['mode'], 'oof_c': best['oof_c'],
            'all_trials': [{'K': t['K'], 'mode': t['mode'], 'oof_c': t['oof_c']}
                           for t in trials],
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)
    summary_path = out_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\nDONE')
    print(f'  CSV:     {out_path}')
    print(f'  summary: {summary_path}')


if __name__ == '__main__':
    main()
