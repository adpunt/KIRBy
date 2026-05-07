#!/usr/bin/env python3
"""ANNITIA hepatic-focused boost runner.

Run AFTER experiment_be.py + experiment_cd.py have produced their OOFs.
This script adds heavyweight branches just for the hepatic endpoint where
the ~39 events leave us most exposed.

What runs (hepatic only, 5-fold CV):

    * RSF n_estimators=1000 on the four strongest BE view-sets.
    * GBSA n_estimators=500, learning_rate=0.03 on those view-sets.
    * CatBoost-Cox + CatBoost-AFT on those view-sets (catboost now installed).
    * WeightedRSF × {linear, sqrt, rank, softmax_T(T=2)}
        with importance method = permutation, on the strongest view-set.
    * WeightedRSF × ensemble importance (permutation + lightgbm_gain + treeshap)
        × {linear, rank}, on the strongest view-set.
    * BaggedRSF B=200 on the strongest view-set.
    * Snapshot-RSF N=5 on the strongest view-set.
    * TabPFN / TabICL landmark with horizon = median event time, on the
        strongest view-set (deps were silently skipped during CD; we retry
        in a fresh process).

Then a hepatic-only fusion across this script's OOFs + BE + CD pooled top-K.

The "strongest view-set" is auto-detected from BE's cv_summary.json — the
hep best by mean CV. The "four strongest" are the top-4 view-sets among
all (view-set, model) configurations.

Outputs:
    tests/results/annitia/hep_boost/
        cv_summary.json
        oof_hepatic_<config_id>.npy
        val_hepatic_<config_id>.npy
        sub_<mode>_hepatic.csv     — per-fusion-mode submissions

Usage:
    python tests/experiments/annitia/experiment_hep_boost.py
    python tests/experiments/annitia/experiment_hep_boost.py --views snapshot,clinical_pro,fpca
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds
from experiments.annitia.views import build_views
from experiments.annitia.survival import (
    SurvivalCoxPH, SurvivalRSF, SurvivalGBSA,
    SurvivalCatBoostCox, SurvivalCatBoostAft,
    concordance)
from experiments.annitia.feature_weighting import WeightedSurvivalModel
from experiments.annitia.bagging import BaggedSurvivalModel
from experiments.annitia.snapshot_ensemble import SnapshotEnsembleSurvival
from experiments.annitia.tabpfn import TabPFNSurvivalLandmark
from experiments.annitia.tabicl import TabICLSurvivalLandmark
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold
from experiments.annitia.fusion import (
    fuse_equal, fuse_cv_weighted, fuse_single, fuse_rrf, StackMetaFusion)


def _convert(o):
    if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_convert(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o


class _AutoHorizonLandmark:
    """Wrapper for TabPFN/TabICL landmark survival — computes horizon from y."""
    def __init__(self, base_cls, **kwargs):
        self.base_cls = base_cls; self.kwargs = kwargs; self._inner = None
    def fit(self, X, y):
        time = np.asarray(y['time'], dtype=np.float64)
        event = np.asarray(y['event'], dtype=bool)
        ev = time[event & np.isfinite(time)]
        if ev.size == 0: ev = time[np.isfinite(time)]
        h = float(np.median(ev))
        self._inner = self.base_cls(horizon=h, **self.kwargs)
        self._inner.fit(X, y)
        return self
    def predict_risk(self, X): return self._inner.predict_risk(X)


def _read_be_top_view_sets(be_dir: Path, k: int = 4):
    """Read BE's cv_summary.json, return the top-k view-sets by hep CV."""
    cv_path = be_dir / 'cv_summary.json'
    if not cv_path.exists():
        # Fallback: hand-picked strong view-sets from the smoke runs.
        return [['snapshot', 'clinical_pro', 'fpca', 'gp_smooth'],
                ['snapshot', 'clinical_pro', 'fpca'],
                ['snapshot', 'clinical_pro', 'fpca', 'visit_density', 'clinical_slope'],
                ['snapshot', 'clinical_pro', 'meld_na', 'visit_density']]
    sm = json.loads(cv_path.read_text())
    recs = [r for r in sm['per_condition'].get('hepatic', [])
            if np.isfinite(r['mean_cindex'])]
    recs.sort(key=lambda r: -r['mean_cindex'])
    seen, out = set(), []
    for r in recs:
        vs = tuple(r['views'])
        if vs not in seen:
            seen.add(vs)
            out.append(list(vs))
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--be-dir', default='tests/results/annitia/be')
    ap.add_argument('--cd-dir', default='tests/results/annitia/cd')
    ap.add_argument('--output-dir', default='tests/results/annitia/hep_boost')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--views', default=None,
                    help='Override the view-set used for trick conditions. '
                         'Comma-separated, e.g. snapshot,clinical_pro,fpca,gp_smooth.')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('ANNITIA_CACHE_DIR', '/tmp/annitia_hb_cache')
    t_start = time.time()

    print('Loading train.csv...')
    train = load_annitia()
    n_train = len(train['patient_ids'])
    val_path = Path(args.validation_csv)
    if not val_path.exists():
        sys.exit(f'val_test missing: {val_path}')
    val = load_annitia(data_dir=str(val_path.parent), filename=val_path.name,
                       has_outcomes=False)
    n_val = len(val['patient_ids'])

    # Pick view-sets
    be_dir = Path(args.be_dir)
    top_sets = _read_be_top_view_sets(be_dir, k=4)
    if args.views:
        # User override: single set used across all conditions
        primary = args.views.split(',')
        top_sets = [primary] + [vs for vs in top_sets if vs != primary]
    primary = top_sets[0]
    print(f'  primary view-set: {"+".join(primary)}')
    print(f'  top-4 view-sets:')
    for vs in top_sets:
        print(f'    {"+".join(vs)}')

    # Pre-build all views once
    all_views_needed = sorted(set(v for vs in top_sets for v in vs))
    print(f'  building views: {all_views_needed}')
    views_train = build_views(train, view_names=all_views_needed)
    views_val = build_views(val,   view_names=all_views_needed)

    def X_for(vs):
        return (np.concatenate([views_train[v][0] for v in vs], axis=1),
                np.concatenate([views_val[v][0]   for v in vs], axis=1))

    X_primary_tr, X_primary_va = X_for(primary)
    print(f'  primary feature dim: {X_primary_tr.shape[1]}')

    endpoint = 'hepatic'
    y = train[endpoint]['y_struct']
    valid = train[endpoint]['valid_mask']
    folds = get_cv_folds(train, endpoint=endpoint,
                         n_folds=args.n_folds, seed=args.seed)
    print(f'\n=== ENDPOINT: {endpoint} (5-fold) ===')

    summary_path = out / 'cv_summary.json'
    summary = {'view_sets': top_sets, 'primary': primary,
               'per_condition': {'hepatic': []},
               'n_folds': args.n_folds}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            summary['per_condition']['hepatic'] = existing.get(
                'per_condition', {}).get('hepatic', [])
            n_resume = len(summary['per_condition']['hepatic'])
            if n_resume:
                print(f'  RESUME: loaded {n_resume} prior conditions from {summary_path}')
        except Exception as e:
            print(f'  RESUME: failed to load {summary_path} ({e}); starting fresh')

    done_cids_hb = {r['config_id']: r for r in summary['per_condition']['hepatic']
                    if 'config_id' in r}

    # Per-view-set heavyweight branches
    heavy_per_set = [
        ('rsf-1000',     lambda: SurvivalRSF(n_estimators=1000)),
        ('gbsa-500',     lambda: SurvivalGBSA(n_estimators=500, learning_rate=0.03)),
        ('catboost-cox', lambda: SurvivalCatBoostCox(iterations=600, learning_rate=0.03, depth=4)),
        ('catboost-aft', lambda: SurvivalCatBoostAft(iterations=600, learning_rate=0.03, depth=4)),
    ]
    # Primary-set-only trick branches
    trick_branches = [
        ('weighted-rsf-perm-linear', lambda: WeightedSurvivalModel(
            SurvivalRSF, importance_method='permutation', transform='linear',
            base_kwargs={'n_estimators': 500})),
        ('weighted-rsf-perm-sqrt', lambda: WeightedSurvivalModel(
            SurvivalRSF, importance_method='permutation', transform='sqrt',
            base_kwargs={'n_estimators': 500})),
        ('weighted-rsf-perm-rank', lambda: WeightedSurvivalModel(
            SurvivalRSF, importance_method='permutation', transform='rank',
            base_kwargs={'n_estimators': 500})),
        ('weighted-rsf-perm-soft', lambda: WeightedSurvivalModel(
            SurvivalRSF, importance_method='permutation', transform='softmax_T',
            temperature=2.0, base_kwargs={'n_estimators': 500})),
        ('weighted-rsf-ens-linear', lambda: WeightedSurvivalModel(
            SurvivalRSF, transform='linear',
            ensemble_methods=('permutation', 'lightgbm_gain', 'treeshap'),
            base_kwargs={'n_estimators': 500})),
        ('weighted-rsf-ens-rank', lambda: WeightedSurvivalModel(
            SurvivalRSF, transform='rank',
            ensemble_methods=('permutation', 'lightgbm_gain', 'treeshap'),
            base_kwargs={'n_estimators': 500})),
        ('bagged-rsf-B200', lambda: BaggedSurvivalModel(
            SurvivalRSF, B=200, base_kwargs={'n_estimators': 200}, random_state=0)),
        ('snapshot-rsf-N5', lambda: SnapshotEnsembleSurvival(
            SurvivalRSF, n_snapshots=5, base_kwargs={'n_estimators': 500})),
        ('tabpfn-landmark', lambda: _AutoHorizonLandmark(TabPFNSurvivalLandmark)),
        ('tabicl-landmark', lambda: _AutoHorizonLandmark(TabICLSurvivalLandmark)),
    ]

    def _run_condition(label, ctor, X_tr, X_va, view_set):
        cid = f'{label}__{"+".join(view_set)}'
        oof_path = out / f'oof_hepatic_{cid}.npy'
        val_path = out / f'val_hepatic_{cid}.npy'
        existing = done_cids_hb.get(cid)
        if (existing is not None and oof_path.exists() and val_path.exists()
                and np.isfinite(existing.get('mean_cindex', float('nan')))):
            print(f'  {label:30s} {"+".join(view_set)[:40]:40s} CACHED  '
                  f'C={existing["mean_cindex"]:.4f}±{existing["std_cindex"]:.4f}')
            return existing
        t0 = time.time()
        oof = init_oof_buffer(n_train)
        fold_c = []
        try:
            for k, (tr, va) in enumerate(folds):
                est = ctor()
                est.fit(X_tr[tr], y[tr])
                risk = est.predict_risk(X_tr[va])
                fold_c.append(concordance(y[va], risk))
                fill_oof_fold(oof, va, risk)
            mean_c = float(np.nanmean(fold_c))
            std_c  = float(np.nanstd(fold_c))
            valid_idx = np.where(valid)[0]
            refit = ctor()
            refit.fit(X_tr[valid_idx], y[valid_idx])
            val_risk = np.asarray(refit.predict_risk(X_va), dtype=np.float64)
        except Exception as e:
            print(f'  {label:30s} ERR: {type(e).__name__}: {str(e)[:120]}')
            return None
        np.save(oof_path, oof)
        np.save(val_path, val_risk)
        rec = {'config_id': cid, 'label': label, 'view_set': view_set,
               'fold_cindex': fold_c, 'mean_cindex': mean_c, 'std_cindex': std_c,
               'elapsed_sec': time.time() - t0}
        summary['per_condition']['hepatic'] = [
            r for r in summary['per_condition']['hepatic']
            if r.get('config_id') != cid
        ]
        summary['per_condition']['hepatic'].append(rec)
        done_cids_hb[cid] = rec
        summary_path.write_text(json.dumps(_convert(summary), indent=2))
        print(f'  {label:30s} {"+".join(view_set)[:40]:40s} '
              f'C={mean_c:.4f}±{std_c:.4f}  ({time.time()-t0:.1f}s)')
        return rec

    print('\n--- per-view-set heavyweights (4 sets × 4 algos) ---')
    for vs in top_sets:
        X_tr, X_va = X_for(vs)
        for label, ctor in heavy_per_set:
            _run_condition(label, ctor, X_tr, X_va, vs)

    print('\n--- primary-set-only tricks ---')
    for label, ctor in trick_branches:
        _run_condition(label, ctor, X_primary_tr, X_primary_va, primary)

    # ----- Hepatic-only fusion across THIS pool + BE + CD top-K -----
    print('\n=== Hepatic fusion: this script ∪ BE ∪ CD ===')
    pool = []
    for r in summary['per_condition']['hepatic']:
        if not np.isfinite(r['mean_cindex']):
            continue
        pool.append({
            'label': f'hb__{r["label"]}__{"+".join(r["view_set"])}',
            'mean_cindex': r['mean_cindex'],
            'oof_path': str(out / f'oof_hepatic_{r["config_id"]}.npy'),
            'val_path': str(out / f'val_hepatic_{r["config_id"]}.npy'),
        })
    # Pull BE + CD pools
    for src_dir in (Path(args.be_dir), Path(args.cd_dir)):
        cv_path = src_dir / 'cv_summary.json'
        if not cv_path.exists():
            continue
        sm = json.loads(cv_path.read_text())
        for r in sm['per_condition'].get('hepatic', []):
            if not np.isfinite(r.get('mean_cindex', float('nan'))):
                continue
            cid = r['config_id']
            oof_p = src_dir / f'oof_hepatic_{cid}.npy'
            val_p = src_dir / f'val_hepatic_{cid}.npy'
            if oof_p.exists() and val_p.exists():
                label = r.get('label') or f'{cid}'
                pool.append({
                    'label': f'{src_dir.name}__{label}',
                    'mean_cindex': r['mean_cindex'],
                    'oof_path': str(oof_p),
                    'val_path': str(val_p),
                })

    pool.sort(key=lambda r: -r['mean_cindex'])
    print(f'  pooled {len(pool)} hepatic conditions')
    top_k = 8
    survivors = pool[:top_k]
    print(f'  top-{top_k} for fusion:')
    for r in survivors:
        print(f'    {r["label"][:60]:60s}  C={r["mean_cindex"]:.4f}')

    if survivors:
        oof_list = [np.load(r['oof_path']) for r in survivors]
        val_list = [np.load(r['val_path']) for r in survivors]
        cv_list  = [r['mean_cindex'] for r in survivors]
        weights = np.maximum(np.asarray(cv_list) - 0.5, 0.0)
        if weights.sum() == 0: weights = np.ones(len(survivors))

        modes = {
            'top1':        val_list[int(np.argmax(cv_list))],
            'equal':       fuse_equal(val_list),
            'cv_weighted': fuse_cv_weighted(val_list, weights),
            'single':      fuse_single(val_list, cv_list),
            'rrf':         fuse_rrf(val_list, k=60.0),
        }
        sm_obj = None
        try:
            sm_obj = StackMetaFusion(l1_ratio=0.5)
            sm_obj.fit(oof_list, y, valid_mask=valid)
            modes['stack_meta'] = sm_obj.transform(val_list)
        except Exception as e:
            print(f'  stack_meta failed: {e}')

        # Score each mode by OOF-train C-index against y[valid]
        for mode, val_vec in modes.items():
            try:
                if mode == 'top1':
                    ov = oof_list[int(np.argmax(cv_list))]
                elif mode == 'equal':
                    ov = fuse_equal(oof_list)
                elif mode == 'cv_weighted':
                    ov = fuse_cv_weighted(oof_list, weights)
                elif mode == 'single':
                    ov = fuse_single(oof_list, cv_list)
                elif mode == 'rrf':
                    ov = fuse_rrf(oof_list, k=60.0)
                elif mode == 'stack_meta':
                    ov = sm_obj.transform(oof_list)
                c = concordance(y[valid], ov[valid])
            except Exception:
                c = float('nan')
            print(f'    mode={mode:12s} OOF-train C={c:.4f}')
            pd.DataFrame({'trustii_id': val['trustii_id'],
                          'risk_hepatic_event': val_vec}).to_csv(
                out / f'sub_{mode}_hepatic.csv', index=False)

    summary['elapsed_sec'] = time.time() - t_start
    (out / 'cv_summary.json').write_text(json.dumps(_convert(summary), indent=2))
    print(f'\nDONE in {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
