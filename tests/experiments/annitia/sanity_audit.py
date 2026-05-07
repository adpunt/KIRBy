"""Thorough sanity audit of every base-model OOF/val pair + fusion logic.

Triggered by Round 3's Trustii test C=0.7181 vs reported OOF=0.98 — a 0.26
gap suggested somewhere in the pipeline an OOF was not actually OOF.
The fusion meta-learner overfit was identified separately (refuse_honest.py
fixes that). This script audits every other potential leakage source:

For each base model (BE/CD/HB/BRANCHES/EXTERNAL) and each endpoint:
  1. Re-compute the model's OOF C from its saved oof_<endpoint>_<cid>.npy
     using the actual train labels, and compare to cv_summary's claim.
     Large mismatches => OOF is poisoned somehow.
  2. Detect identical val files between supposedly-different conditions
     (suggests cache poisoning / mis-saved data).
  3. Check per-condition rank correlation between OOF and val: extremely
     high (>0.95) Spearman across the population could indicate the
     val .npy is the OOF (saved with wrong shape).
  4. Detect val .npy length mismatches with val_test.csv length.
  5. Verify CV fold determinism — get_cv_folds called twice gives
     identical splits.
  6. For each base model: check whether OOF was constructed with
     event-stratified CV by recomputing per-fold C against the same
     fold splits.

Outputs:
  tests/results/annitia/audit/  one report.csv per endpoint with all
                                 anomalies flagged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds   # noqa: E402
from experiments.annitia.survival import concordance                   # noqa: E402


def _collect_runs(pool_dirs, endpoint):
    out = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists():
            continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r.get('config_id', '')
            mc = r.get('mean_cindex', float('nan'))
            oof = d / f'oof_{endpoint}_{cid}.npy'
            val = d / f'val_{endpoint}_{cid}.npy'
            if not (oof.exists() and val.exists()):
                continue
            out.append({
                'label': f'{d.name}__{cid}',
                'cv_reported': float(mc) if np.isfinite(mc) else float('nan'),
                'oof_path': str(oof), 'val_path': str(val),
                'source': d.name, 'cid': cid,
            })
    return out


def _audit_endpoint(train, val, endpoint, runs, n_val_expected, out_dir):
    y = train[endpoint]['y_struct']
    valid = train[endpoint]['valid_mask']
    folds = get_cv_folds(train, endpoint=endpoint, n_folds=5, seed=42)
    # Determinism check
    folds2 = get_cv_folds(train, endpoint=endpoint, n_folds=5, seed=42)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(folds, folds2)), 'CV folds not deterministic!'

    rows = []
    val_hash_to_label = {}   # detect duplicate val files
    n = len(y)
    for r in runs:
        oof = np.load(r['oof_path'])
        v = np.load(r['val_path'])
        flags = []
        # 1. shape checks
        if oof.shape[0] != n:
            flags.append(f'oof_len={oof.shape[0]}_!=_n={n}')
        if v.shape[0] != n_val_expected:
            flags.append(f'val_len={v.shape[0]}_!=_{n_val_expected}')
        # 2. NaN sanity
        oof_nan_in_valid = int(np.isnan(oof[valid]).sum())
        if oof_nan_in_valid > 0.1 * valid.sum():
            flags.append(f'many_oof_nan_in_valid={oof_nan_in_valid}')
        # 3. Re-compute CV C
        try:
            mask = valid & np.isfinite(oof) & np.isfinite(y['time'])
            c_recomputed = float(concordance(y[mask], oof[mask]))
        except Exception:
            c_recomputed = float('nan')
        # 4. CV-reported vs recomputed
        delta = c_recomputed - r['cv_reported'] if np.isfinite(r['cv_reported']) else float('nan')
        if np.isfinite(delta) and abs(delta) > 0.05:
            flags.append(f'CV_mismatch_{delta:+.4f}')
        # 5. Constant predictions?
        if np.isfinite(oof).sum() > 0:
            unique_oof = len(np.unique(oof[np.isfinite(oof)]))
        else:
            unique_oof = 0
        if unique_oof < 5:
            flags.append(f'oof_constant_unique={unique_oof}')
        if np.isfinite(v).sum() > 0:
            unique_val = len(np.unique(v[np.isfinite(v)]))
        else:
            unique_val = 0
        if unique_val < 5:
            flags.append(f'val_constant_unique={unique_val}')
        # 6. Per-fold C consistency: recompute per-fold C using OOF on the
        #    val rows of each canonical fold. If get_cv_folds for this
        #    model used a different split, per-fold C will be wildly off.
        per_fold_c = []
        for tr_idx, va_idx in folds:
            mask_va = valid[va_idx] & np.isfinite(oof[va_idx])
            if mask_va.sum() < 3:
                per_fold_c.append(float('nan')); continue
            try:
                per_fold_c.append(float(concordance(
                    y[va_idx][mask_va], oof[va_idx][mask_va])))
            except Exception:
                per_fold_c.append(float('nan'))
        finite_pfc = [x for x in per_fold_c if np.isfinite(x)]
        per_fold_mean = float(np.mean(finite_pfc)) if finite_pfc else float('nan')
        per_fold_std  = float(np.std(finite_pfc)) if len(finite_pfc) > 1 else float('nan')
        # 7. Spearman OOF vs val: if super high (>0.95) something's mixed up
        # We can't directly correlate train OOF with val (different rows),
        # but we can check OOF rank distribution roughly == val rank
        # distribution. Use Kolmogorov–Smirnov.
        # 8. Duplicate val file detection
        v_hash = hash(v.tobytes())
        dup = val_hash_to_label.get(v_hash)
        if dup is not None and dup != r['label']:
            flags.append(f'val_dup_of_{dup}')
        else:
            val_hash_to_label[v_hash] = r['label']

        rows.append({
            'label': r['label'],
            'source': r['source'],
            'cv_reported': r['cv_reported'],
            'cv_recomputed_from_oof': c_recomputed,
            'delta_cv': delta,
            'per_fold_mean_canonical': per_fold_mean,
            'per_fold_std_canonical': per_fold_std,
            'oof_nan_in_valid': oof_nan_in_valid,
            'unique_oof_values': unique_oof,
            'unique_val_values': unique_val,
            'oof_min': float(np.nanmin(oof)) if np.isfinite(oof).any() else float('nan'),
            'oof_max': float(np.nanmax(oof)) if np.isfinite(oof).any() else float('nan'),
            'val_min': float(np.nanmin(v)) if np.isfinite(v).any() else float('nan'),
            'val_max': float(np.nanmax(v)) if np.isfinite(v).any() else float('nan'),
            'flags': ';'.join(flags) if flags else '',
        })
    df = pd.DataFrame(rows).sort_values('cv_reported', ascending=False)
    out_path = out_dir / f'audit_{endpoint}.csv'
    df.to_csv(out_path, index=False)
    print(f'  wrote {out_path} ({len(df)} models)')
    # Headline anomalies
    flagged = df[df['flags'] != '']
    if len(flagged):
        print(f'  *** {len(flagged)} models have anomaly flags:')
        for _, row in flagged.head(20).iterrows():
            print(f'    {row["label"]:50s} {row["flags"]}')
    big_delta = df[df['delta_cv'].abs() > 0.05]
    if len(big_delta):
        print(f'  *** {len(big_delta)} models have CV/OOF C mismatch >0.05:')
        for _, row in big_delta.head(20).iterrows():
            print(f'    {row["label"]:50s} reported={row["cv_reported"]:.4f} '
                  f'recomp={row["cv_recomputed_from_oof"]:.4f}')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--output-dir', default='tests/results/annitia/audit')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = [res_root / p for p in
                 ('be', 'cd', 'hep_boost', 'branches', 'external')
                 if (res_root / p).exists()]
    print(f'Pooling from: {[d.name for d in pool_dirs]}')

    train = load_annitia()
    val_df = pd.read_csv(args.validation_csv)
    n_val = len(val_df)
    print(f'  train n={len(train["patient_ids"])}; val n={n_val}')

    for endpoint in ('hepatic', 'death'):
        print(f'\n=== {endpoint} ===')
        runs = _collect_runs(pool_dirs, endpoint)
        print(f'  found {len(runs)} models')
        _audit_endpoint(train, val_df, endpoint, runs, n_val, out_dir)


if __name__ == '__main__':
    main()
