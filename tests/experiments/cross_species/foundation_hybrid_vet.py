#!/usr/bin/env python3
"""foundation_hybrid_vet — KIRBy hybrid search on small vet ChEMBL targets.

(Previously the PT-ε / `x_pt_epsilon_foundation` experiment, renamed 2026-05-13.)

Tests whether `find_best_hybrid` on KIRBy's standard molecular rep set beats
from-scratch baselines on the four small vet eval targets, with NO new
cross-species pretraining step. This is the cheap anchor for the cross-species
pretraining track — if the existing pipeline already handles small vet data,
the heavier pretraining variants (same_target_transfer, rodent_to_vet, etc.)
only need to clear it.

Per-target evaluation:
  - 5-fold scaffold GroupKFold (3-fold fallback for n < 150)
  - find_best_hybrid on one of three rep presets:
      --preset minimal -> REPS_TOP_TIER_UNIVERSAL (4 reps: mhg_gnn, mordred,
                           pdv, grover; Top Tier in every surveyed molecular
                           dataset per PIPELINE_READINESS_TODO.md)
      --preset mid     -> REPS_TOP_TIER_ANY (11 reps; Top Tier on ≥1 dataset)
      --preset full    -> REPS_GENERAL_TIER (18 reps; union of General Tier)
  - Baselines per (rep × target): ridge (RidgeCV) and dnn_scratch

Datasets (locked in docs/cross_species_vet_eval_targets.json):
  - cow_CA4   (CHEMBL281,  n≈1,541) - bovine Carbonic anhydrase 4
  - cow_DRD2  (CHEMBL3998, n≈406)   - bovine D(2) dopamine receptor
  - dog_PTGS2 (CHEMBL4033, n≈110)   - canine Cyclooxygenase-2
  - cow_PTGS2 (CHEMBL3331, n≈28)    - bovine Prostaglandin G/H synthase 2

Usage:
    # Minimal preset (4 reps, fastest sanity-check pass)
    python tests/experiments/cross_species/foundation_hybrid_vet.py \\
        --preset minimal

    # Mid preset (11 reps, Top Tier ≥1 dataset)
    python tests/experiments/cross_species/foundation_hybrid_vet.py \\
        --preset mid

    # Full preset (18 reps, default — main experiment)
    python tests/experiments/cross_species/foundation_hybrid_vet.py

    # Custom rep list (overrides --preset)
    python tests/experiments/cross_species/foundation_hybrid_vet.py \\
        --keys cow_CA4 --reps ecfp4,mordred,molformer --n-folds 3
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    get_representations, assign_scaffold_groups,
)
from run_baseline_survey import _make_cv, _score, _maybe_scale
from experiments._shared import convert_numpy, FULL_BUDGETS
from experiments.cross_species.vet_eval_data import (
    load_vet_dataset, list_vet_keys, DEFAULT_CACHE_DIR as VET_DEFAULT_CACHE_DIR,
)


# Audit-grounded molecular rep sets, sourced from PIPELINE_READINESS_TODO.md
# tier tables across the 8 surveyed molecular datasets (caco2, herg_ki, herg,
# logd, qm9, adme_hlm, adme_rlm, adme_mdr1).
#
# REPS_TOP_TIER_UNIVERSAL: Top Tier in ALL 8 surveyed datasets.
# Fastest sanity-check pass.
REPS_TOP_TIER_UNIVERSAL = [
    'mhg_gnn', 'mordred', 'pdv', 'grover',
]

# REPS_TOP_TIER_ANY: Top Tier on at least one surveyed dataset.
# Drops the 7 reps that were General-Tier-consistent but never Top
# (chemberta, chembert, selformer, graphmvp, top_torsion, rdkfp, maccs).
REPS_TOP_TIER_ANY = [
    'mhg_gnn', 'mordred', 'pdv', 'grover',          # 8/8 (universal)
    'avalon',                                        # 6/8
    'ecfp4',                                         # 5/8
    'mol2vec',                                       # 4/8
    'smited', 'molformer', 'graph_kernel',           # 3/8
    'atom_pair',                                     # 1/8
]

# REPS_GENERAL_TIER: union of General Tier across all 7 surveyed datasets.
# Full hybrid-search rep set; diverse enough for find_best_hybrid to find
# cross-family combinations.
#
# Excluded by design: mole, gp_molformer, minimol — wired in code but zero
# measured numbers in tests/results/ as of 2026-05-13. Re-add once benchmarked.
REPS_GENERAL_TIER = [
    # 6 fingerprints
    'ecfp4', 'maccs', 'avalon', 'atom_pair', 'top_torsion', 'rdkfp',
    # 2 descriptors
    'pdv', 'mordred',
    # 5 GNN / learned-graph
    'mol2vec', 'mhg_gnn', 'grover', 'graphmvp', 'graph_kernel',
    # 5 SMILES LM
    'chemberta', 'chembert', 'molformer', 'smited', 'selformer',
]

REP_PRESETS = {
    'minimal': REPS_TOP_TIER_UNIVERSAL,  # 4 reps — Top Tier in every dataset
    'mid':     REPS_TOP_TIER_ANY,        # 11 reps — Top Tier on ≥1 dataset
    'full':    REPS_GENERAL_TIER,        # 18 reps — full audit-grounded set
}

# Small-n datasets get fewer folds so each fold has enough rows
SMALL_N_THRESHOLD = 150
SMALL_N_FOLDS = 3
DEFAULT_N_FOLDS = 5


def _resolve_folds(n: int, requested: int) -> int:
    if n < SMALL_N_THRESHOLD:
        return min(requested, SMALL_N_FOLDS)
    return requested


def evaluate_one_rep_model(X, y, groups, model, n_folds):
    """Run scaffold GroupKFold for one (rep, model) and return mean metric dict."""
    from sklearn.base import clone

    cv, cv_kwargs = _make_cv(groups, n_folds, y)
    splits = list(cv.split(X, y, **cv_kwargs))

    fold_metrics = []
    for tr, va in splits:
        Xtr, Xva = _maybe_scale(X[tr], X[va])
        try:
            m = clone(model)
            m.fit(Xtr, y[tr])
            p = m.predict(Xva)
            if not np.all(np.isfinite(p)):
                continue
            fold_metrics.append(_score(y[va], p, 'regression'))
        except Exception as e:
            print(f"        [err: {e}]")
            continue

    if not fold_metrics:
        return {'mean': {}, 'std': {}, 'n_folds_completed': 0}

    keys = fold_metrics[0].keys()
    return {
        'mean': {k: float(np.mean([f[k] for f in fold_metrics if k in f]))
                 for k in keys},
        'std':  {k: float(np.std([f[k] for f in fold_metrics if k in f]))
                 for k in keys},
        'n_folds_completed': len(fold_metrics),
    }


def build_baseline_models():
    """Two baselines per (rep × target): RidgeCV and from-scratch DNN."""
    from sklearn.linear_model import RidgeCV
    from kirby.models.dnn import DNNRegressor

    return {
        'ridge': RidgeCV(alphas=np.logspace(-4, 4, 50)),
        'dnn_scratch': DNNRegressor(
            hidden_sizes=(256, 128, 64), dropout=0.2, lr=0.001,
            epochs=200, batch_size=256, patience=20, random_state=42,
        ),
    }


def evaluate_target(key: str, rep_names, n_folds_requested: int,
                    vet_cache_dir: str = VET_DEFAULT_CACHE_DIR):
    print(f"\n{'='*70}")
    print(f"Vet target: {key}")
    print('='*70)

    smiles, y, task_type = load_vet_dataset(key, cache_dir=vet_cache_dir)
    n = len(smiles)
    print(f"  {n} compounds, pActivity {y.min():.2f}..{y.max():.2f}")
    if n < 10:
        print(f"  SKIPPED: only {n} compounds")
        return {'n_samples': n, 'skipped': True}

    groups, n_scaffolds = assign_scaffold_groups(smiles)
    n_folds = _resolve_folds(n, n_folds_requested)
    print(f"  {n_scaffolds} scaffolds, {n_folds}-fold CV")

    # Build all reps once (only requested)
    reps = get_representations(smiles, ','.join(rep_names))

    ds_results = {
        'n_samples': n,
        'n_scaffolds': int(n_scaffolds),
        'n_folds': n_folds,
        'task_type': task_type,
        'baselines_per_rep': {},
        'hybrid': {},
    }

    # --- Per-rep ridge + dnn_scratch baselines ---
    baselines = build_baseline_models()
    for rep_name in rep_names:
        if rep_name not in reps:
            print(f"  {rep_name}: SKIPPED (rep failed to build)")
            continue
        X = np.asarray(reps[rep_name], dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != n:
            print(f"  {rep_name}: SKIPPED (shape mismatch)")
            continue

        rep_block = {'dim': int(X.shape[1]), 'conditions': {}}
        print(f"  --- {rep_name} (dim={X.shape[1]}) ---")
        for cond_name, model in baselines.items():
            t0 = time.time()
            res = evaluate_one_rep_model(X, y, groups, model, n_folds)
            res['time_s'] = time.time() - t0
            rmse = res['mean'].get('rmse', float('nan'))
            rho = res['mean'].get('spearman', float('nan'))
            print(f"    {cond_name:12s} RMSE={rmse:.4f} ρ={rho:.4f} "
                  f"({res['time_s']:.1f}s, folds={res['n_folds_completed']})")
            rep_block['conditions'][cond_name] = res
        ds_results['baselines_per_rep'][rep_name] = rep_block

    # --- KIRBy hybrid search ---
    print(f"\n  --- find_best_hybrid (General-Tier rep set) ---")
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        t0 = time.time()
        search = find_best_hybrid(
            base_reps=reps,
            labels=y,
            groups=groups,
            budgets=FULL_BUDGETS,
            speed='full',
            return_fitted=True,
            verbose=False,
            cache=True,
            dataset_name=f"vet_{key}",
            tier='general',
            metric='rmse',
            task_type=task_type,
            augmentation_strategy='none',
            adaptive_pruning=0.25,
        )
        elapsed = time.time() - t0
        # Pull the best result per model
        per_model = {}
        for model_name, fitted in (search.fitted_results or {}).items():
            best = getattr(fitted, 'best_metric', None) or \
                   getattr(fitted, 'best_score', None)
            per_model[model_name] = {
                'best_rmse': float(best) if best is not None else None,
                'selected_reps': list(getattr(fitted, 'selected_indices', {}).keys()),
            }
        ds_results['hybrid'] = {
            'time_s': elapsed,
            'per_model': per_model,
        }
        for m, r in per_model.items():
            best = r.get('best_rmse')
            print(f"    {m:12s} best RMSE={best:.4f}" if best is not None
                  else f"    {m:12s} (no result)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        ds_results['hybrid'] = {'error': str(e)[:200]}

    return ds_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--keys', type=str, default=None,
                    help='Comma-separated vet target keys '
                         '(default: all in cross_species_vet_eval_targets.json)')
    ap.add_argument('--reps', type=str, default=None,
                    help='Comma-separated rep names. Overrides --preset.')
    ap.add_argument('--preset', type=str, default='full',
                    choices=sorted(REP_PRESETS),
                    help='Named rep preset: '
                         '"minimal" (4 reps, Top Tier in every surveyed dataset), '
                         '"mid" (11 reps, Top Tier on ≥1 dataset), '
                         'or "full" (18 reps, General-Tier union). '
                         'Default: full.')
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument('--vet-cache-dir', type=str, default=VET_DEFAULT_CACHE_DIR,
                    help=f'ChEMBL activity CSV cache dir for vet targets '
                         f'(default: {VET_DEFAULT_CACHE_DIR}). '
                         f'Use a server-local path on ARC.')
    ap.add_argument('--output', type=str, default=None,
                    help='Output JSON path (default: '
                         'tests/results/cross_species/foundation_hybrid_vet.json)')
    args = ap.parse_args()

    keys = args.keys.split(',') if args.keys else list_vet_keys()
    if args.reps:
        reps = args.reps.split(',')
        preset_label = 'custom'
    else:
        reps = REP_PRESETS[args.preset]
        preset_label = args.preset

    out_default = (f'tests/results/cross_species/'
                   f'foundation_hybrid_vet_{preset_label}.json')

    print("=" * 70)
    print("foundation_hybrid_vet  (T-PT.2)")
    print(f"  Vet targets: {keys}")
    print(f"  Rep preset:  {preset_label} ({len(reps)} reps)")
    print(f"  Reps:        {reps}")
    print(f"  Folds:       {args.n_folds} (3-fold fallback at n<{SMALL_N_THRESHOLD})")
    print(f"  Cache dir:   {args.vet_cache_dir}")
    print("=" * 70)

    results = {}
    for key in keys:
        try:
            results[key] = evaluate_target(
                key, reps, args.n_folds, vet_cache_dir=args.vet_cache_dir,
            )
        except Exception as e:
            print(f"  FAILED on {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = {'error': str(e)[:200]}

    out = Path(args.output or out_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(results), indent=2))
    print(f"\nResults written: {out}")


if __name__ == '__main__':
    main()
