#!/usr/bin/env python3
"""ANNITIA experiment B+E — Phase B (extended algorithms + views) + Phase E
(fusion across branches).

Estimated runtime LOCALLY: 30-60 min (n=1253, 5-fold CV, both endpoints).

What this does:
    1. Loads train.csv (n=1253) and val_test.csv (n=423).
    2. Builds 6 hand-selected view sets (single + small concat) from
       Phases A's 14 view builders.
    3. For each (view-set, algorithm) trains a 5-fold CV survival model:
           algorithms: cox, coxnet, rsf, gbsa, lgbm_aft, ngboost
       Records: per-fold C, OOF risk vector on training, refit-on-full
       prediction on val_test.
    4. Picks the top-3 (view-set, algo) combos per endpoint.
    5. Fuses the top-3 with all 5 modes:
           equal, cv_weighted, single, rrf, stack_meta
    6. Writes one submission CSV per fusion mode + a JSON summary of CV.

Outputs:
    tests/results/annitia/be/
        cv_summary.json                — per-condition CV table
        oof_<endpoint>_<config_id>.npy — OOF risk vectors (for stack_meta)
        sub_top1_<endpoint>.csv        — best single config submission
        sub_<mode>.csv                 — per-fusion-mode master submission

Usage:
    python tests/experiments/annitia/experiment_be.py
    python tests/experiments/annitia/experiment_be.py --max-subset-size 2
    python tests/experiments/annitia/experiment_be.py --models cox,rsf,gbsa  # subset

Pick the submission with the highest CV C-index from cv_summary.json's
`fusion_cv` field (computed by leave-one-fold-out OOF on the meta model
when applicable; for non-stack modes we report the mean of the within-mode
fold C-indices). For final submission, ship the highest CV among the
fusion modes — see "Selection rule" below.

Selection rule (matches the roadmap):
    Replace incumbent only if candidate's CV exceeds by ≥0.5 SE.
    Among ties, prefer fewer features / fewer tricks.
"""

import argparse
import json
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
    SURVIVAL_MODELS, concordance)
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold
from experiments.annitia.fusion import (
    fuse_equal, fuse_cv_weighted, fuse_single, fuse_rrf,
    StackMetaFusion)


VIEW_SETS = [
    ('snapshot',),
    ('snapshot', 'clinical_pro'),
    ('snapshot', 'clinical_pro', 'fpca'),
    ('snapshot', 'clinical_pro', 'meld_na', 'visit_density'),
    ('snapshot', 'clinical_pro', 'fpca', 'gp_smooth'),
    ('snapshot', 'clinical_pro', 'fpca', 'visit_density', 'clinical_slope'),
    # Added 2026-05-08: summary view captures max/min/mean per lab — fibroscan_MAX
    # alone gives C_hep=0.815 vs fibroscan_last 0.764. Pipeline never trained on
    # summary or nit views; this fills that gap.
    ('snapshot', 'summary'),
    ('snapshot', 'clinical_pro', 'summary'),
    ('snapshot', 'clinical_pro', 'fpca', 'gp_smooth', 'summary'),
    ('snapshot', 'clinical_pro', 'summary', 'nit'),
]


def _convert(o):
    if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_convert(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv',
                    help='Path to val_test.csv (Trustii test patients).')
    ap.add_argument('--models',
                    default='cox,coxnet,rsf,gbsa,lgbm_aft,ngboost,catboost_cox,catboost_aft',
                    help='Comma-separated algorithm names.')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', default='tests/results/annitia/be')
    ap.add_argument('--top-k', type=int, default=3,
                    help='Top-K configs to fuse with E modes (per endpoint).')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print('Loading train.csv...')
    train = load_annitia()
    n_train = len(train['patient_ids'])
    print(f'  train n={n_train}')

    print(f'Loading {args.validation_csv}...')
    val_path = Path(args.validation_csv)
    if not val_path.exists():
        sys.exit(f'val_test missing: {val_path}')
    val = load_annitia(data_dir=str(val_path.parent), filename=val_path.name,
                       has_outcomes=False)
    n_val = len(val['patient_ids'])
    if val['trustii_id'] is None:
        sys.exit('val_test.csv has no trustii_id column')
    print(f'  val n={n_val}')

    print(f'Building all 14 views on train + val...')
    views_train = build_views(train)
    views_val = build_views(val)

    models = tuple(args.models.split(','))
    for m in models:
        if m not in SURVIVAL_MODELS:
            sys.exit(f'unknown model: {m}')

    # Resume: load existing cv_summary.json so we can skip already-evaluated
    # conditions whose OOF + val .npy files are still on disk.
    summary_path = out / 'cv_summary.json'
    summary = {'view_sets': [list(s) for s in VIEW_SETS],
               'models': list(models),
               'per_condition': {'hepatic': [], 'death': []},
               'n_folds': args.n_folds}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            for ep in ('hepatic', 'death'):
                summary['per_condition'][ep] = existing.get('per_condition', {}).get(ep, [])
            n_resume = sum(len(summary['per_condition'][e]) for e in ('hepatic','death'))
            if n_resume:
                print(f'  RESUME: loaded {n_resume} prior conditions from {summary_path}')
        except Exception as e:
            print(f'  RESUME: failed to load {summary_path} ({e}); starting fresh')

    # Per-endpoint training loop
    for endpoint in ('hepatic', 'death'):
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        folds = get_cv_folds(train, endpoint=endpoint,
                             n_folds=args.n_folds, seed=args.seed)
        print(f'\n=== ENDPOINT: {endpoint} '
              f'(valid={int(valid.sum())}, events={int((y["event"] & valid).sum())}) ===')

        # Build a cid→record index over existing rows for O(1) hit check.
        done_cids = {r['config_id']: r for r in summary['per_condition'][endpoint]
                     if 'config_id' in r}

        for vs in VIEW_SETS:
            X_tr = np.concatenate([views_train[v][0] for v in vs], axis=1)
            X_va = np.concatenate([views_val[v][0]   for v in vs], axis=1)
            for mname in models:
                cls = SURVIVAL_MODELS[mname]
                cid = f'{"+".join(vs)}__{mname}'
                oof_path = out / f'oof_{endpoint}_{cid}.npy'
                val_path = out / f'val_{endpoint}_{cid}.npy'
                if (cid in done_cids and oof_path.exists() and val_path.exists()
                        and np.isfinite(done_cids[cid].get('mean_cindex', float('nan')))):
                    rec = done_cids[cid]
                    print(f'  {"+".join(vs):44s} {mname:12s} CACHED  '
                          f'C={rec["mean_cindex"]:.4f}±{rec["std_cindex"]:.4f}')
                    continue

                t0 = time.time()
                fold_c = []
                oof = init_oof_buffer(n_train)
                for k, (tr, va) in enumerate(folds):
                    try:
                        est = cls()
                        est.fit(X_tr[tr], y[tr])
                        risk = est.predict_risk(X_tr[va])
                        fold_c.append(concordance(y[va], risk))
                        fill_oof_fold(oof, va, risk)
                    except Exception as e:
                        fold_c.append(float('nan'))
                        if k == 0:
                            print(f'  [{"+".join(vs):40s} {mname:12s}] fold0 ERR: {e!r}')
                            break  # don't waste cycles
                mean_c = float(np.nanmean(fold_c))
                std_c  = float(np.nanstd(fold_c))
                # Refit on all valid + predict on val_test
                try:
                    valid_idx = np.where(valid)[0]
                    refit = cls()
                    refit.fit(X_tr[valid_idx], y[valid_idx])
                    val_risk = np.asarray(refit.predict_risk(X_va), dtype=np.float64)
                except Exception as e:
                    val_risk = np.full(n_val, np.nan)
                np.save(oof_path, oof)
                np.save(val_path, val_risk)
                rec = {
                    'config_id': cid, 'views': list(vs), 'model': mname,
                    'fold_cindex': fold_c, 'mean_cindex': mean_c, 'std_cindex': std_c,
                    'n_features': int(X_tr.shape[1]),
                    'elapsed_sec': time.time() - t0,
                }
                # If this cid already had a stale entry, replace it; otherwise append.
                summary['per_condition'][endpoint] = [
                    r for r in summary['per_condition'][endpoint]
                    if r.get('config_id') != cid
                ]
                summary['per_condition'][endpoint].append(rec)
                done_cids[cid] = rec
                print(f'  {"+".join(vs):44s} {mname:12s} d={X_tr.shape[1]:4d} '
                      f'C={mean_c:.4f}±{std_c:.4f}  ({time.time()-t0:.1f}s)')
                # Checkpoint after EACH condition so kills don't cost cycles.
                summary_path.write_text(json.dumps(_convert(summary), indent=2))

        # Save partial after each endpoint so a Ctrl+C still has progress
        summary_path.write_text(json.dumps(_convert(summary), indent=2))

    # ----- Phase E fusion across top-K per endpoint -----
    final_subs = pd.DataFrame({'trustii_id': val['trustii_id']})
    fusion_records = {}
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death',   'risk_death')):
        recs = summary['per_condition'][endpoint]
        finite = [r for r in recs if np.isfinite(r['mean_cindex'])]
        finite.sort(key=lambda r: -r['mean_cindex'])
        top = finite[:args.top_k]
        if not top:
            print(f'  {endpoint}: no finite conditions — skipping fusion')
            continue
        print(f'\n=== {endpoint}: top-{len(top)} by CV ===')
        for r in top:
            print(f'    {"+".join(r["views"]):40s} {r["model"]:12s}  C={r["mean_cindex"]:.4f}')
        oof_list = [np.load(out / f'oof_{endpoint}_{r["config_id"]}.npy') for r in top]
        val_list = [np.load(out / f'val_{endpoint}_{r["config_id"]}.npy') for r in top]
        cv_list = [r['mean_cindex'] for r in top]
        weights = np.maximum(np.asarray(cv_list) - 0.5, 0.0)
        if weights.sum() == 0: weights = np.ones(len(top))

        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        modes = {
            'top1':       val_list[int(np.argmax(cv_list))],
            'equal':      fuse_equal(val_list),
            'cv_weighted':fuse_cv_weighted(val_list, weights),
            'single':     fuse_single(val_list, cv_list),
            'rrf':        fuse_rrf(val_list, k=60.0),
        }
        # stack_meta: fit on training OOFs, apply to val
        try:
            sm = StackMetaFusion(l1_ratio=0.5)
            sm.fit(oof_list, y, valid_mask=valid)
            modes['stack_meta'] = sm.transform(val_list)
        except Exception as e:
            print(f'  stack_meta failed: {e}')

        # Score each mode by OOF C-index against training labels (the only
        # labelled set we have locally) — picked-by-CV is a separate concern.
        # NOTE: this isn't the test C-index (we don't have test labels);
        # it tells us how the meta-model fits the training set.
        oof_modes = {
            'equal':      fuse_equal(oof_list),
            'cv_weighted':fuse_cv_weighted(oof_list, weights),
            'single':     fuse_single(oof_list, cv_list),
            'rrf':        fuse_rrf(oof_list, k=60.0),
            'stack_meta': sm.transform(oof_list) if 'stack_meta' in modes else None,
        }
        fusion_records[endpoint] = {}
        for mode, val_vec in modes.items():
            sub_path = out / f'sub_{mode}_{endpoint}.csv'
            df = pd.DataFrame({'trustii_id': val['trustii_id'], col: val_vec})
            df.to_csv(sub_path, index=False)
            oof_vec = oof_modes.get(mode)
            train_c = (concordance(y[valid], oof_vec[valid])
                       if oof_vec is not None and np.all(np.isfinite(oof_vec[valid]))
                       else float('nan'))
            fusion_records[endpoint][mode] = {
                'submission': str(sub_path),
                'oof_train_cindex': train_c,
            }
            print(f'    mode={mode:12s} OOF-train C={train_c:.4f}')

        # Combine into one master CSV per mode
        for mode, val_vec in modes.items():
            final_subs[f'{col}__{mode}'] = val_vec

    # Final master submission CSV (uses the highest-CV mode per endpoint).
    master = pd.DataFrame({'trustii_id': val['trustii_id']})
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death',   'risk_death')):
        if endpoint in fusion_records:
            best_mode = max(fusion_records[endpoint].items(),
                            key=lambda kv: (kv[1]['oof_train_cindex']
                                            if not np.isnan(kv[1]['oof_train_cindex'])
                                            else -np.inf))[0]
            master[col] = final_subs[f'{col}__{best_mode}']
            print(f'  picked {endpoint} → mode={best_mode}')
        else:
            master[col] = 0.0
    master_path = out / 'sub_master.csv'
    master.to_csv(master_path, index=False)
    print(f'\nFINAL MASTER: {master_path}')

    summary['fusion'] = fusion_records
    summary['elapsed_sec'] = time.time() - t_start
    (out / 'cv_summary.json').write_text(json.dumps(_convert(summary), indent=2))
    print(f'\nDONE in {(time.time() - t_start)/60:.1f} min')
    print(f'  cv_summary.json: {out / "cv_summary.json"}')
    print(f'  per-mode submissions: {out}/sub_*.csv')


if __name__ == '__main__':
    main()
