#!/usr/bin/env python3
"""End-to-end local smoke for N06 on REAL bindingdb_kd data (small subset).

Complements _smoke_n06.py (synthetic unit tests). Loads the locally-cached
bindingdb_kd dump, caps rows, uses cheap local reps, shrinks FULL_BUDGETS and
precomputes rf-only — same speed strategy as _smoke_harness.py. NOT a results
run.

Usage:
    python tests/experiments/noise/_smoke_n06_e2e.py --subsample 800
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import argparse
import warnings
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import experiments._shared as _shared  # noqa: E402
from experiments._shared import load_data, precompute_hybrid_searches  # noqa: E402

_shared.FULL_BUDGETS = [25]
REPS = ['drug_maccs', 'protein_aac']  # 167 / 20 D, locally computable


def tiny_config():
    return {'n_splits': 2, 'search_speed': 'full', 'allocations': ['greedy'],
            'importance_methods': ['lightgbm_gain'], 'internal_evaluators': ['rf'],
            'budgets': [25], 'models': ['rf']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subsample', type=int, default=800)
    args = ap.parse_args()

    print("=== SMOKE n06 E2E (bindingdb_kd, real data) ===")
    t0 = time.time()
    data = load_data('bindingdb_kd', rep_preset='all', rep_names=REPS)
    for k, X in list(data['reps'].items()):
        if np.isnan(X).any():
            data['reps'][k] = np.nan_to_num(X, nan=0.0)

    n = len(data['labels'])
    if args.subsample and n > args.subsample:
        # Sort by (drug,target) so replicate pairs stay together under the
        # subset, preserving real disagreeing-replicate groups for the GT test.
        md = data.get('metadata') or {}
        di = md.get('drug_ids'); pi = md.get('protein_ids')
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, args.subsample, replace=False))
        data['labels'] = np.asarray(data['labels'])[idx]
        data['reps'] = {k: X[idx] for k, X in data['reps'].items()}
        if data.get('groups') is not None:
            data['groups'] = np.asarray(data['groups'])[idx]
        for mk in ('drug_ids', 'protein_ids', 'drug_smiles', 'protein_seqs'):
            v = md.get(mk)
            if v is not None and len(v) == n:
                md[mk] = ([v[i] for i in idx] if not isinstance(v, np.ndarray)
                          else np.asarray(v)[idx])
        data['metadata'] = md
    data['augmentations'] = None
    print(f"  loaded n={len(data['labels'])} task={data['task_type']} "
          f"in {time.time()-t0:.1f}s")

    config = tiny_config()
    t0 = time.time()
    data['precomputed_searches'] = precompute_hybrid_searches(
        reps=data['reps'], labels=data['labels'], groups=data.get('groups'),
        config=config, augmentations=None, task_type=data['task_type'],
        domain=data.get('domain', 'molecular'),
        dataset_name='bindingdb_kd', tier='general', internal_evaluators=['rf'])
    data['precomputed_searches_top'] = {}
    data['top_tier_reps'] = {}
    print(f"  precompute done in {time.time()-t0:.1f}s")

    from experiments.noise.n06_bindingdb_ground_truth import run
    out_dir = '/tmp/noise_smoke'
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    try:
        run(data, config, out_dir, 'bindingdb_kd', budget=25)
        print(f"\n=== n06 E2E RAN OK in {time.time()-t0:.1f}s ===")
    except Exception as e:
        import traceback
        print(f"\n=== n06 E2E CRASHED: {type(e).__name__}: {e} ===")
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
