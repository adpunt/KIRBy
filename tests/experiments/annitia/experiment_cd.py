#!/usr/bin/env python3
"""ANNITIA experiment C+D — Phase C target framings + Phase D training tricks.

Estimated runtime LOCALLY: 30-60 min (n=1253, 3-fold CV, both endpoints).

What this does:
    1. Loads train.csv + val_test.csv.
    2. Uses one fixed view subset (snapshot+clinical_pro+fpca by default —
       a strong baseline from our smoke runs).
    3. Sweeps Phase C framings × algorithms:
           pseudo_obs × {LightGBM, MLP}
           landmark    × {MLP-drop, MLP-ipcw}
       (catboost/tabpfn paths skipped if optional dep absent.)
    4. Sweeps Phase D tricks on top of a Cox baseline:
           weighted-cox × {linear, rank}
           bagged-cox B=50
           robust-cox trim=0.05
           snapshot-rsf N=3
           self-train cox
           xgb-pairwise-ranker
       (Importance methods are cached — second run uses cache.)
    5. For each condition: 3-fold CV on training → OOF + refit on full
       valid → val_test prediction.
    6. Stack_meta over the survivors (CV ≥ 0.55) per endpoint.

Outputs:
    tests/results/annitia/cd/
        cv_summary.json
        oof_<endpoint>_<config_id>.npy
        val_<endpoint>_<config_id>.npy
        sub_<mode>_<endpoint>.csv
        sub_master.csv

Usage:
    python tests/experiments/annitia/experiment_cd.py
    python tests/experiments/annitia/experiment_cd.py --views snapshot,clinical_pro
    python tests/experiments/annitia/experiment_cd.py --skip-d  # only Phase C

This script is designed to RUN AFTER experiment_be.py: combine its OOFs +
this script's OOFs in a single stack_meta pool for the strongest fusion.
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
    MLPRegressorSurvival, MLPClassifierSurvival,
    RegressorLightGBM, ClassifierCatBoost,
    concordance)
from experiments.annitia.pseudo_obs import PseudoObservationsSurvival
from experiments.annitia.landmark import LandmarkBinarySurvival
from experiments.annitia.feature_weighting import WeightedSurvivalModel
from experiments.annitia.bagging import BaggedSurvivalModel
from experiments.annitia.robust_cox import RobustCox
from experiments.annitia.snapshot_ensemble import SnapshotEnsembleSurvival
from experiments.annitia.self_train import SelfTrainSurvival
from experiments.annitia.ranking_loss import XGBoostPairwiseRanker
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


def _build_conditions(skip_c: bool, skip_d: bool):
    """Return list of (label, framing, ctor, requires_unlab) tuples.

    requires_unlab: bool — True for SelfTrain (needs X_unlab at fit time).
    """
    out = []
    if not skip_c:
        out += [
            ('C-baseline-cox',     'cox',         lambda: SurvivalCoxPH(),                                                    False),
            ('C-baseline-rsf',     'cox',         lambda: SurvivalRSF(n_estimators=200),                                      False),
            ('C-pseudo-LGBM',      'pseudo_obs',  lambda: PseudoObservationsSurvival(regressor_cls=RegressorLightGBM),         False),
            ('C-pseudo-MLP',       'pseudo_obs',  lambda: PseudoObservationsSurvival(regressor_cls=MLPRegressorSurvival),      False),
            ('C-landmark-MLP-drop','landmark',    lambda: LandmarkBinarySurvival(classifier_cls=MLPClassifierSurvival, mode='drop'), False),
            ('C-landmark-MLP-ipcw','landmark',    lambda: LandmarkBinarySurvival(classifier_cls=MLPClassifierSurvival, mode='ipcw'), False),
            ('C-landmark-Cat-drop','landmark',    lambda: LandmarkBinarySurvival(classifier_cls=ClassifierCatBoost, mode='drop'), False),
        ]
    if not skip_d:
        out += [
            ('D-weighted-cox-lin',  'cox', lambda: WeightedSurvivalModel(SurvivalCoxPH, importance_method='permutation', transform='linear'),  False),
            ('D-weighted-cox-rank', 'cox', lambda: WeightedSurvivalModel(SurvivalCoxPH, importance_method='permutation', transform='rank'),    False),
            ('D-weighted-rsf-lin',  'cox', lambda: WeightedSurvivalModel(SurvivalRSF, importance_method='permutation', transform='linear', base_kwargs={'n_estimators': 200}), False),
            ('D-bagged-cox-B50',    'cox', lambda: BaggedSurvivalModel(SurvivalCoxPH, B=50),                                                    False),
            ('D-robust-cox-trim5',  'cox', lambda: RobustCox(trim_frac=0.05),                                                                  False),
            ('D-snapshot-rsf-N3',   'cox', lambda: SnapshotEnsembleSurvival(SurvivalRSF, n_snapshots=3, base_kwargs={'n_estimators': 200}),     False),
            ('D-selftrain-cox',     'cox', lambda: SelfTrainSurvival(SurvivalCoxPH, top_frac=0.2, pseudo_weight=0.1),                          True),
            ('D-xgb-pairwise',      'landmark', lambda: XGBoostPairwiseRanker(n_estimators=200, learning_rate=0.05),                            False),
        ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--views', default='snapshot,clinical_pro,fpca')
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-c', action='store_true')
    ap.add_argument('--skip-d', action='store_true')
    ap.add_argument('--output-dir', default='tests/results/annitia/cd')
    ap.add_argument('--cv-floor', type=float, default=0.55,
                    help='Drop conditions with mean CV below this from fusion.')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('ANNITIA_CACHE_DIR', '/tmp/annitia_cd_cache')
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
    if val['trustii_id'] is None:
        sys.exit('val_test.csv has no trustii_id column')

    view_names = tuple(args.views.split(','))
    print(f'  view subset: {"+".join(view_names)}')
    views_train = build_views(train, view_names=list(view_names))
    views_val = build_views(val,   view_names=list(view_names))
    X_tr_full = np.concatenate([views_train[v][0] for v in view_names], axis=1)
    X_va_full = np.concatenate([views_val[v][0]   for v in view_names], axis=1)
    print(f'  feature dim: {X_tr_full.shape[1]}')

    conditions = _build_conditions(args.skip_c, args.skip_d)
    print(f'  conditions: {len(conditions)}')

    summary = {'view_set': list(view_names),
               'n_features': X_tr_full.shape[1],
               'per_condition': {'hepatic': [], 'death': []},
               'n_folds': args.n_folds}

    for endpoint in ('hepatic', 'death'):
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        folds = get_cv_folds(train, endpoint=endpoint,
                             n_folds=args.n_folds, seed=args.seed)
        n_ev = int((y['event'] & valid).sum())
        print(f'\n=== ENDPOINT: {endpoint} (valid={int(valid.sum())}, events={n_ev}) ===')

        for label, framing, ctor, needs_unlab in conditions:
            t0 = time.time()
            fold_c = []
            oof = init_oof_buffer(n_train)
            try:
                for k, (tr, va) in enumerate(folds):
                    est = ctor()
                    if needs_unlab:
                        est.fit(X_tr_full[tr], y[tr], X_unlab=X_tr_full[va])
                    else:
                        est.fit(X_tr_full[tr], y[tr])
                    risk = est.predict_risk(X_tr_full[va])
                    fold_c.append(concordance(y[va], risk))
                    fill_oof_fold(oof, va, risk)
                mean_c = float(np.nanmean(fold_c))
                std_c  = float(np.nanstd(fold_c))
                # Refit on full valid + predict on val_test
                valid_idx = np.where(valid)[0]
                refit = ctor()
                if needs_unlab:
                    refit.fit(X_tr_full[valid_idx], y[valid_idx], X_unlab=X_va_full)
                else:
                    refit.fit(X_tr_full[valid_idx], y[valid_idx])
                val_risk = np.asarray(refit.predict_risk(X_va_full), dtype=np.float64)
            except ImportError as e:
                # Optional dep missing; record graceful skip and move on.
                print(f'  {label:25s} SKIP: {str(e)[:60]}')
                continue
            except Exception as e:
                print(f'  {label:25s} ERR: {type(e).__name__}: {str(e)[:80]}')
                continue
            cid = label
            np.save(out / f'oof_{endpoint}_{cid}.npy', oof)
            np.save(out / f'val_{endpoint}_{cid}.npy', val_risk)
            summary['per_condition'][endpoint].append({
                'config_id': cid, 'label': label, 'framing': framing,
                'fold_cindex': fold_c, 'mean_cindex': mean_c, 'std_cindex': std_c,
                'elapsed_sec': time.time() - t0,
            })
            print(f'  {label:25s} framing={framing:<11s} '
                  f'C={mean_c:.4f}±{std_c:.4f}  ({time.time()-t0:.1f}s)')
            (out / 'cv_summary.json').write_text(json.dumps(_convert(summary), indent=2))

    # ----- Phase E fusion -----
    fusion_records = {}
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death',   'risk_death')):
        recs = summary['per_condition'][endpoint]
        finite = [r for r in recs
                  if np.isfinite(r['mean_cindex']) and r['mean_cindex'] >= args.cv_floor]
        finite.sort(key=lambda r: -r['mean_cindex'])
        if not finite:
            print(f'\n  {endpoint}: no conditions ≥ {args.cv_floor} — picking best regardless')
            finite = sorted([r for r in recs if np.isfinite(r['mean_cindex'])],
                            key=lambda r: -r['mean_cindex'])[:3]
        if not finite:
            continue
        print(f'\n=== {endpoint}: {len(finite)} survivors for fusion ===')
        for r in finite:
            print(f'    {r["label"]:25s} C={r["mean_cindex"]:.4f}')
        oof_list = [np.load(out / f'oof_{endpoint}_{r["config_id"]}.npy') for r in finite]
        val_list = [np.load(out / f'val_{endpoint}_{r["config_id"]}.npy') for r in finite]
        cv_list  = [r['mean_cindex'] for r in finite]
        weights = np.maximum(np.asarray(cv_list) - 0.5, 0.0)
        if weights.sum() == 0: weights = np.ones(len(finite))

        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        modes = {
            'top1':       val_list[int(np.argmax(cv_list))],
            'equal':      fuse_equal(val_list),
            'cv_weighted':fuse_cv_weighted(val_list, weights),
            'single':     fuse_single(val_list, cv_list),
            'rrf':        fuse_rrf(val_list, k=60.0),
        }
        sm_obj = None
        try:
            sm_obj = StackMetaFusion(l1_ratio=0.5)
            sm_obj.fit(oof_list, y, valid_mask=valid)
            modes['stack_meta'] = sm_obj.transform(val_list)
        except Exception as e:
            print(f'  stack_meta failed: {e}')

        fusion_records[endpoint] = {}
        for mode, val_vec in modes.items():
            sub_path = out / f'sub_{mode}_{endpoint}.csv'
            df = pd.DataFrame({'trustii_id': val['trustii_id'], col: val_vec})
            df.to_csv(sub_path, index=False)
            # OOF train C-index per mode
            oof_modes = {
                'top1':       oof_list[int(np.argmax(cv_list))],
                'equal':      fuse_equal(oof_list),
                'cv_weighted':fuse_cv_weighted(oof_list, weights),
                'single':     fuse_single(oof_list, cv_list),
                'rrf':        fuse_rrf(oof_list, k=60.0),
                'stack_meta': sm_obj.transform(oof_list) if sm_obj is not None else None,
            }
            ov = oof_modes.get(mode)
            train_c = (concordance(y[valid], ov[valid])
                       if ov is not None and np.all(np.isfinite(ov[valid])) else float('nan'))
            fusion_records[endpoint][mode] = {
                'submission': str(sub_path),
                'oof_train_cindex': train_c,
            }
            print(f'    mode={mode:12s} OOF-train C={train_c:.4f}')

    # Master submission
    master = pd.DataFrame({'trustii_id': val['trustii_id']})
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death',   'risk_death')):
        if endpoint in fusion_records:
            best_mode = max(fusion_records[endpoint].items(),
                            key=lambda kv: (kv[1]['oof_train_cindex']
                                            if not np.isnan(kv[1]['oof_train_cindex'])
                                            else -np.inf))[0]
            df = pd.read_csv(fusion_records[endpoint][best_mode]['submission'])
            master[col] = df[col].to_numpy()
            print(f'  picked {endpoint} → mode={best_mode} '
                  f'C={fusion_records[endpoint][best_mode]["oof_train_cindex"]:.4f}')
        else:
            master[col] = 0.0
    (out / 'sub_master.csv').write_text(master.to_csv(index=False))

    summary['fusion'] = fusion_records
    summary['elapsed_sec'] = time.time() - t_start
    (out / 'cv_summary.json').write_text(json.dumps(_convert(summary), indent=2))
    print(f'\nDONE in {(time.time() - t_start)/60:.1f} min')
    print(f'  master: {out / "sub_master.csv"}')


if __name__ == '__main__':
    main()
