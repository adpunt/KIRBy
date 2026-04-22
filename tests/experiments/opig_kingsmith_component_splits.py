#!/usr/bin/env python3
"""
Component-Wise Extrapolation Splits for Buchwald-Hartwig (OPIG C6)

Tests whether KIRBy hybrids improve generalization under increasingly strict
component-wise leave-one-out splits on Buchwald-Hartwig C-N coupling datasets.

Scientific questions:
  1. How does hybrid performance degrade across component split difficulty?
  2. Do hybrids maintain advantage over base reps on unseen components?
  3. Which component axis is hardest to extrapolate (aryl halide > ligand > base > additive)?

Motivation:
  King-Smith et al. (Nature Chemistry, 2024) "Probing the Chemical Reactome"
  showed that standard random splits are overly optimistic for HTE reaction
  datasets. Component-wise leave-one-out splits (unseen aryl halide, unseen
  ligand, etc.) test true extrapolation ability.

  Saebi et al. (JCIM, 2024) "When Yield Prediction Does Not Yield Prediction"
  demonstrated that models often memorize component identities rather than
  learning generalizable reactivity patterns.

  Bradshaw et al. (ACS Central Science, 2025) "Challenging Reaction Prediction
  Models to Generalize to Novel Chemistry" confirmed random-split R2 >> OOS R2.

  This experiment tests whether hybrid representations — by combining
  complementary chemical information — improve OOS generalization compared
  to individual reps on each component split axis.

Split conditions (all GroupKFold):
  - additive:    leave-one-additive-out (current default, 23 groups)
  - aryl_halide: leave-one-aryl-halide-out (15 groups for original BH)
  - ligand:      leave-one-ligand-out (4 groups — hardest, fewest groups)
  - base:        leave-one-base-out (3 groups — very few, extreme OOS)

Datasets:
  - buchwald_hartwig (Ahneman/Dreher/Doyle, Science 2018, ~3955 reactions)
  - bh_doyle_plates (Doyle lab HTE, ~4599 reactions)

Paper baselines:
  - Ahneman et al. 2018: RF + DFT descriptors, R2=0.92 (random split)
  - Schwaller/yield-BERT 2021: R2=0.951 (random), drops to ~0.7 OOS
  - King-Smith et al. 2024: HiTEA framework, 39K+ HTE reactions
  - Saebi et al. 2024: showed leave-one-component-out drops vs random

Usage:
    # Smoke test (~5 min, 1 dataset, 2 splits)
    python tests/experiments/opig_kingsmith_component_splits.py \\
        --datasets buchwald_hartwig --splits additive,aryl_halide --n-folds 2

    # Full evaluation
    python tests/experiments/opig_kingsmith_component_splits.py

Reference:
    King-Smith et al. (2024). Nature Chemistry.
    Saebi et al. (2024). JCIM. DOI: 10.1021/acs.jcim.3c01524
    Bradshaw et al. (2025). ACS Central Science. DOI: 10.1021/acscentsci.5c00055
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_DATASETS = ['buchwald_hartwig', 'bh_doyle_plates']

# Component keys to use as group axes for leave-one-out splits.
# amine is constant across BH datasets (4-methylaniline), so excluded.
ALL_SPLIT_KEYS = ['additive', 'aryl_halide', 'ligand', 'base']

MODELS_LIST = DOWNSTREAM_MODELS


def _is_binary_rep(X):
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_reaction_data(dataset_name):
    """Load a BH reaction dataset with all reps from DATASET_REPS."""
    from kirby.datasets.reaction_yields import (
        load_buchwald_hartwig, load_bh_doyle_plates,
    )
    from kirby.representations.reaction import (
        create_drfp, create_concat_ecfp4, create_concat_maccs,
        create_concat_pdv, create_product_ecfp4, create_reactant_ecfp4,
        create_product_maccs, create_reactant_maccs,
        create_product_pdv, create_reactant_pdv,
        create_rxnfp, create_onehot_components,
        create_diff_ecfp4, create_diff_maccs,
        create_component_descriptors, create_ligand_descriptors,
        create_reactiont5, create_reaction_rep_variant,
        create_cgr_morgan, create_chytorch_graphormer,
        create_cgr_smiles_lm,
    )

    if dataset_name == 'buchwald_hartwig':
        data = load_buchwald_hartwig()
    elif dataset_name == 'bh_doyle_plates':
        data = load_bh_doyle_plates()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    rxn_smiles = data['rxn_smiles']
    components = data['components']

    # Build only the reps listed in DATASET_REPS for this dataset
    ds_config = DATASET_REPS.get(dataset_name, {})
    wanted = set(ds_config.get('general', []))
    wanted.update(ds_config.get('augmentation', []))
    wanted.update(ds_config.get('top_tier', []))

    # Generator registry (same as _shared.py but local)
    _generators = {
        'drfp_2048': lambda: create_drfp(rxn_smiles),
        'concat_ecfp4': lambda: create_concat_ecfp4(components),
        'concat_maccs': lambda: create_concat_maccs(components),
        'concat_pdv': lambda: create_concat_pdv(components),
        'product_ecfp4': lambda: create_product_ecfp4(rxn_smiles),
        'reactant_ecfp4': lambda: create_reactant_ecfp4(rxn_smiles),
        'product_maccs': lambda: create_product_maccs(rxn_smiles),
        'reactant_maccs': lambda: create_reactant_maccs(rxn_smiles),
        'product_pdv': lambda: create_product_pdv(rxn_smiles),
        'reactant_pdv': lambda: create_reactant_pdv(rxn_smiles),
        'rxnfp': lambda: create_rxnfp(rxn_smiles),
        'onehot_components': lambda: create_onehot_components(components),
        'diff_ecfp4': lambda: create_diff_ecfp4(rxn_smiles),
        'diff_maccs': lambda: create_diff_maccs(rxn_smiles),
        'component_descriptors': lambda: create_component_descriptors(components),
        'ligand_descriptors': lambda: create_ligand_descriptors(components),
        'reactiont5': lambda: create_reactiont5(rxn_smiles),
        'cgr_morgan': lambda: create_cgr_morgan(rxn_smiles),
        'chytorch_graphormer': lambda: create_chytorch_graphormer(rxn_smiles),
        'cgr_smiles_lm': lambda: create_cgr_smiles_lm(rxn_smiles),
    }

    # Add molecular variant generators dynamically
    from kirby.representations.reaction import _MOL_REP_VARIANTS
    for mol_name in _MOL_REP_VARIANTS:
        for role in ('product', 'reactant', 'diff'):
            rep_name = f'{role}_{mol_name}'
            _generators[rep_name] = (
                lambda rr=rxn_smiles, mn=mol_name, r=role:
                    create_reaction_rep_variant(rr, mn, r)
            )

    reps = {}
    for name in wanted:
        if name not in _generators:
            print(f"  WARNING: unknown rep '{name}', skipping")
            continue
        try:
            reps[name] = _generators[name]()
            print(f"  {name}: {reps[name].shape}")
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")

    return {
        'labels': np.array(data['labels'], dtype=np.float64),
        'components': components,
        'reps': reps,
        'metadata': data.get('metadata', {}),
    }


def assign_component_groups(components, split_key):
    """Assign GroupKFold groups based on a reaction component key.

    Args:
        components: List[Dict] with keys like 'aryl_halide', 'ligand', etc.
        split_key: Which component to group by.

    Returns:
        (groups, n_groups, group_names) tuple.
    """
    values = []
    for comp in components:
        val = comp.get(split_key, '')
        # Normalize: strip whitespace, lowercase
        val = str(val).strip().lower() if val else ''
        values.append(val)

    unique_vals = sorted(set(values))
    val_to_id = {v: i for i, v in enumerate(unique_vals)}
    groups = np.array([val_to_id[v] for v in values])
    return groups, len(unique_vals), unique_vals


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep(X, labels, groups, model, n_folds):
    """Evaluate one rep x model via GroupKFold, return R2/RMSE/Spearman."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    n_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_groups)
    if actual_folds < 2:
        return None

    cv = GroupKFold(n_splits=actual_folds)
    splits = list(cv.split(X, labels, groups=groups))

    is_binary = _is_binary_rep(X)
    r2_scores, rmse_scores, spearman_scores = [], [], []

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if not is_binary:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        try:
            m = clone(model)
            m.fit(X_train, y_train)
        except Exception:
            continue

        preds = m.predict(X_val)

        ss_res = np.sum((y_val - preds) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        rmse_scores.append(np.sqrt(np.mean((y_val - preds) ** 2)))

        if len(y_val) > 2:
            rho, _ = spearmanr(y_val, preds)
            spearman_scores.append(rho if not np.isnan(rho) else 0.0)

    if not r2_scores:
        return None

    return {
        'r2': {'mean': float(np.mean(r2_scores)), 'std': float(np.std(r2_scores))},
        'rmse': {'mean': float(np.mean(rmse_scores)), 'std': float(np.std(rmse_scores))},
        'spearman': {
            'mean': float(np.mean(spearman_scores)) if spearman_scores else 0.0,
            'std': float(np.std(spearman_scores)) if spearman_scores else 0.0,
        },
        'n_folds_completed': len(r2_scores),
    }


def evaluate_split(dataset_name, data, split_key, n_folds):
    """Evaluate all reps + hybrid under one component split.

    Returns dict with per-rep per-model metrics and hybrid comparison.
    """
    reps = data['reps']
    labels = data['labels']
    components = data['components']

    groups, n_groups, group_names = assign_component_groups(components, split_key)
    actual_folds = min(n_folds, n_groups)

    print(f"\n  --- Split: leave-one-{split_key}-out "
          f"({n_groups} groups, {actual_folds}-fold) ---")

    if actual_folds < 2:
        print(f"  SKIPPED: only {n_groups} groups, need >= 2 for CV")
        return {'skipped': True, 'reason': f'only {n_groups} groups',
                'n_groups': n_groups, 'group_names': group_names}

    # Build hybrid under this split's groups
    aug_reps = set(DATASET_REPS.get(dataset_name, {}).get('augmentation', []))
    base_reps = {k: v for k, v in reps.items() if k not in aug_reps}
    aug_dict = {k: v for k, v in reps.items() if k in aug_reps}

    try:
        from kirby.find_best_hybrid import find_best_hybrid
        from kirby.hybrid import apply_feature_selection

        search = find_best_hybrid(
            base_reps=base_reps if base_reps else reps,
            labels=labels, groups=groups,
            metric='rmse',
            task_type='regression',
            budgets=FULL_BUDGETS,
            augmentations=aug_dict if aug_dict else None,
            augmentation_strategy='uncapped' if aug_dict else 'none',
            internal_evaluators=INTERNAL_EVALUATORS,
            speed='full', return_fitted=True, verbose=False, cache=True,
            dataset_name=dataset_name, tier='general',
            condition_suffix=f'loo_{split_key}',
            adaptive_pruning=0.25)

        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps: {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

        fitted = search.fitted_results.get('rf')
        if fitted is not None:
            all_reps = {**reps}
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
            reps_eval = dict(reps)
            reps_eval['kirby_hybrid'] = apply_feature_selection(all_reps, fi)
            print(f"  kirby_hybrid: {reps_eval['kirby_hybrid'].shape[1]}D")
        else:
            reps_eval = dict(reps)
    except Exception as e:
        print(f"  Hybrid build failed: {e}")
        reps_eval = dict(reps)

    # Evaluate all reps x models
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=len(labels))
    n = len(labels)

    print(f"  Evaluating {len(reps_eval)} reps x {len(models)} models "
          f"x {actual_folds}-fold CV...")

    results = {}
    for rep_name in sorted(reps_eval.keys()):
        X = reps_eval[rep_name]
        try:
            X = np.asarray(X, dtype=np.float64)
        except Exception:
            continue
        if X.ndim != 2 or X.shape[0] != n:
            continue

        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            res = evaluate_rep(X, labels, groups, factory(), actual_folds)
            elapsed = time.time() - t0

            if res is None:
                continue

            rep_results[model_name] = res
            r2 = res['r2']['mean']
            tag = ' ***' if rep_name == 'kirby_hybrid' else ''
            print(f"    {rep_name:25s} {model_name:8s}  "
                  f"R2={r2:.4f}  RMSE={res['rmse']['mean']:.2f}  ({elapsed:.1f}s){tag}")

        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'split_key': split_key,
        'n_groups': n_groups,
        'n_folds': actual_folds,
        'group_names': group_names,
        'results': results,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print cross-split comparison: how does performance degrade?"""
    print(f"\n{'=' * 100}")
    print(f"COMPONENT-WISE EXTRAPOLATION ANALYSIS (King-Smith et al. 2024)")
    print(f"{'=' * 100}")

    for ds_name, ds_results in all_results.items():
        print(f"\n  Dataset: {ds_name} ({ds_results.get('n_reactions', '?')} reactions)")
        print(f"  {'Split key':15s} {'#Groups':>7s} {'Best base':25s} "
              f"{'R2':>8s} {'Hybrid R2':>10s} {'Delta':>8s}")
        print(f"  {'-'*15} {'-'*7} {'-'*25} {'-'*8} {'-'*10} {'-'*8}")

        for split_key in ALL_SPLIT_KEYS:
            sr = ds_results.get('splits', {}).get(split_key)
            if sr is None or sr.get('skipped'):
                reason = sr.get('reason', 'not run') if sr else 'not run'
                print(f"  {split_key:15s} {'—':>7s} {'SKIPPED':25s} "
                      f"{'—':>8s} {'—':>10s} {'—':>8s}  ({reason})")
                continue

            results = sr.get('results', {})
            if not results:
                print(f"  {split_key:15s} {sr['n_groups']:>7d} {'NO RESULTS':25s}")
                continue

            # Find best base rep (excluding hybrid) and hybrid performance
            best_base_name = None
            best_base_r2 = -float('inf')
            hybrid_r2 = float('nan')

            for rep_name, rep_data in results.items():
                for model_name, model_res in rep_data['models'].items():
                    r2 = model_res['r2']['mean']
                    if rep_name == 'kirby_hybrid':
                        if np.isnan(hybrid_r2) or r2 > hybrid_r2:
                            hybrid_r2 = r2
                    else:
                        if r2 > best_base_r2:
                            best_base_r2 = r2
                            best_base_name = rep_name

            delta = hybrid_r2 - best_base_r2 if not np.isnan(hybrid_r2) else float('nan')
            h_s = f"{hybrid_r2:.4f}" if not np.isnan(hybrid_r2) else "---"
            d_s = f"{delta:+.4f}" if not np.isnan(delta) else "---"

            print(f"  {split_key:15s} {sr['n_groups']:>7d} "
                  f"{best_base_name or '---':25s} {best_base_r2:8.4f} "
                  f"{h_s:>10s} {d_s:>8s}")

    # Key insight summary
    print(f"\n  Key question: does the hybrid maintain its advantage as split")
    print(f"  difficulty increases (additive > aryl_halide > ligand > base)?")
    print(f"  If hybrid delta stays positive or grows, it suggests the hybrid")
    print(f"  captures complementary chemical info that aids extrapolation.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Component-wise extrapolation splits for Buchwald-Hartwig. '
                    'Tests hybrid generalization under leave-one-component-out.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated datasets (default: buchwald_hartwig,bh_doyle_plates)')
    parser.add_argument(
        '--splits', type=str, default=None,
        help='Comma-separated component keys to split by '
             '(default: additive,aryl_halide,ligand,base)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Max folds for GroupKFold (capped by #groups, default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path (default: tests/results/comparisons/'
             'opig_kingsmith_component_splits.json)')

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets else ALL_DATASETS)
    split_keys = (args.splits.split(',') if args.splits else ALL_SPLIT_KEYS)

    print("=" * 70)
    print("COMPONENT-WISE EXTRAPOLATION EXPERIMENT")
    print(f"  King-Smith et al. (2024), Saebi et al. (2024)")
    print(f"  Datasets: {datasets}")
    print(f"  Split axes: {split_keys}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Max folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}
    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        try:
            data = load_reaction_data(ds_name)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            import traceback
            traceback.print_exc()
            continue

        n = len(data['labels'])
        ds_result = {'n_reactions': n, 'splits': {}}

        for split_key in split_keys:
            try:
                split_result = evaluate_split(
                    ds_name, data, split_key, args.n_folds)
                ds_result['splits'][split_key] = split_result
            except Exception as e:
                print(f"  FAILED split={split_key}: {e}")
                import traceback
                traceback.print_exc()

        all_results[ds_name] = ds_result

    # Analysis
    if all_results:
        print_analysis(all_results)

    # Save
    output_path = (args.output or
                   'tests/results/comparisons/opig_kingsmith_component_splits.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
