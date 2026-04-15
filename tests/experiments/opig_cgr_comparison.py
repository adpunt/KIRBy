#!/usr/bin/env python3
"""
CGR vs SMILES Representation Comparison (Phase 3 of CGR_EMBEDDING_RESEARCH.md)

Tests whether CGR (Condensed Graph of Reaction) representations add unique
signal beyond SMILES-based representations for reaction yield prediction.

Scientific questions:
  1. Do CGR reps outperform their SMILES counterparts individually?
  2. Does adding CGR reps to the hybrid pool improve hybrid performance?
  3. Which CGR features (if any) are selected by the hybrid allocator?
  4. How correlated are CGR and SMILES reps (Linear CKA)?

Two comparison families:
  Fingerprint: cgr_morgan vs diff_ecfp4 vs drfp_2048
  Neural:      chytorch_graphormer vs rxnfp vs reactiont5

Datasets:
  - buchwald_hartwig (Ahneman/Dreher/Doyle, Science 2018, ~3955 reactions)
  - bh_doyle_plates (Doyle lab HTE, ~4599 reactions)
  - suzuki (Perera et al., Science 2018, ~5760 reactions)

Usage:
    # Smoke test (~10 min, 1 dataset, 2 folds, no hybrid)
    python tests/experiments/opig_cgr_comparison.py \\
        --datasets buchwald_hartwig --n-folds 2 --skip-hybrid

    # Full evaluation
    python tests/experiments/opig_cgr_comparison.py

Reference:
    CGR_EMBEDDING_RESEARCH.md — Phase 3: Comparison Experiment
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
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
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS,
    INTERNAL_EVALUATORS, FULL_BUDGETS, extract_top3_base_reps,
    _cache_rep, _fitted_to_feature_info,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_DATASETS = ['buchwald_hartwig', 'bh_doyle_plates', 'suzuki']

CGR_REPS = {'cgr_morgan', 'chytorch_graphormer'}

FAMILIES = {
    'fingerprint': {
        'cgr': ['cgr_morgan'],
        'smiles': ['diff_ecfp4', 'drfp_2048'],
    },
    'neural': {
        'cgr': ['chytorch_graphormer'],
        'smiles': ['rxnfp', 'reactiont5'],
    },
}

# All family reps for CKA analysis
FAMILY_REPS = ['cgr_morgan', 'diff_ecfp4', 'drfp_2048',
               'chytorch_graphormer', 'rxnfp', 'reactiont5']

MODELS_LIST = DOWNSTREAM_MODELS


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cgr_comparison_data(dataset_name):
    """Load a reaction dataset with all reps including CGR."""
    from kirby.datasets.reaction_yields import (
        load_buchwald_hartwig, load_bh_doyle_plates, load_suzuki,
    )
    from kirby.representations.reaction import (
        create_drfp, create_concat_ecfp4, create_concat_maccs,
        create_concat_pdv, create_product_ecfp4, create_reactant_ecfp4,
        create_product_maccs, create_reactant_maccs,
        create_product_pdv, create_reactant_pdv,
        create_rxnfp, create_onehot_components,
        create_diff_ecfp4, create_diff_maccs,
        create_component_descriptors, create_ligand_descriptors,
        create_reactiont5, create_cgr_morgan, create_chytorch_graphormer,
        create_reaction_rep_variant, _MOL_REP_VARIANTS,
    )

    if dataset_name == 'buchwald_hartwig':
        data = load_buchwald_hartwig()
    elif dataset_name == 'bh_doyle_plates':
        data = load_bh_doyle_plates()
    elif dataset_name == 'suzuki':
        data = load_suzuki()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    groups = data.get('groups')
    if groups is None:
        print(f"  WARNING: {dataset_name} has no groups, skipping")
        return None

    rxn_smiles = data['rxn_smiles']
    components = data['components']

    # Full generator registry — all reaction reps
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
    }

    # Add molecular variant generators
    for mol_name in _MOL_REP_VARIANTS:
        for role in ('product', 'reactant', 'diff'):
            rep_name = f'{role}_{mol_name}'
            _generators[rep_name] = (
                lambda rr=rxn_smiles, mn=mol_name, r=role:
                    create_reaction_rep_variant(rr, mn, r)
            )

    # Generate all reps with caching
    reps = {}
    for name, gen_fn in _generators.items():
        try:
            reps[name] = _cache_rep(dataset_name, name, gen_fn)
        except ImportError as e:
            print(f"  WARNING: {name} skipped (missing dep): {e}")
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")

    print(f"  Generated {len(reps)} reps successfully")

    return {
        'labels': np.array(data['labels'], dtype=np.float64),
        'groups': np.array(groups),
        'reps': reps,
        'components': components,
        'metadata': data.get('metadata', {}),
    }


# =============================================================================
# REP POOL PARTITIONING
# =============================================================================

def partition_rep_pools(reps):
    """Split reps into SMILES-only and SMILES+CGR pools.

    Returns:
        (smiles_only, smiles_plus_cgr, cgr_available) dicts.
    """
    cgr_available = {k: v for k, v in reps.items() if k in CGR_REPS}
    smiles_only = {k: v for k, v in reps.items() if k not in CGR_REPS}
    smiles_plus_cgr = dict(reps)
    return smiles_only, smiles_plus_cgr, cgr_available


# =============================================================================
# EVALUATION
# =============================================================================

def _is_binary_rep(X):
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


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


# =============================================================================
# HYBRID BUILDING
# =============================================================================

def build_and_evaluate_hybrid(pool_reps, all_reps, labels, groups, n_folds,
                              pool_name='pool', dataset_name=None):
    """Build hybrid from a rep pool and evaluate it.

    Args:
        pool_reps: Dict of reps to use as the hybrid pool.
        all_reps: Full rep dict (for apply_feature_selection).
        labels: Target values.
        groups: GroupKFold groups.
        n_folds: Max CV folds.
        pool_name: Label for printing.

    Returns:
        Dict with search results, hybrid evaluation, top3 base reps,
        and selected rep info.
    """
    from kirby.find_best_hybrid import find_best_hybrid
    from kirby.hybrid import apply_feature_selection

    print(f"\n  Building hybrid ({pool_name}, {len(pool_reps)} reps)...")

    search = find_best_hybrid(
        base_reps=pool_reps,
        labels=labels,
        groups=groups,
        metric='rmse',
        task_type='regression',
        budgets=FULL_BUDGETS,
        internal_evaluators=INTERNAL_EVALUATORS,
        speed='full',
        return_fitted=True,
        verbose=False,
        cache=True,
        dataset_name=f"{dataset_name}_{pool_name}",
        tier='general',
        adaptive_pruning=0.25,
    )

    top3 = extract_top3_base_reps(search, model='rf')
    if top3:
        print(f"    Top-3 base: {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

    # Extract hybrid features from RF fitted result
    fitted = (search.fitted_results or {}).get('rf')
    hybrid_eval = {}
    selected_reps = {}

    if fitted is not None:
        fi = _fitted_to_feature_info(fitted)
        X_hybrid = apply_feature_selection(all_reps, fi)
        selected_reps = {
            rep_name: len(indices)
            for rep_name, indices in fitted.selected_indices.items()
        }
        print(f"    Hybrid: {X_hybrid.shape[1]}D from {list(selected_reps.keys())}")

        # Evaluate hybrid with all models
        n_groups = len(np.unique(groups))
        actual_folds = min(n_folds, n_groups)
        models = get_model_factories(MODELS_LIST, 'regression',
                                     n_samples=len(labels))

        for model_name, factory in models.items():
            res = evaluate_rep(X_hybrid, labels, groups, factory(),
                               actual_folds)
            if res is not None:
                hybrid_eval[model_name] = res
                print(f"    hybrid × {model_name:8s}  "
                      f"RMSE={res['rmse']['mean']:.2f}  "
                      f"R2={res['r2']['mean']:.4f}")
    else:
        print(f"    WARNING: no fitted RF result for {pool_name}")

    return {
        'n_reps': len(pool_reps),
        'rep_names': sorted(pool_reps.keys()),
        'top3_base': [[n, s] for n, s in top3],
        'hybrid_eval': hybrid_eval,
        'selected_reps': selected_reps,
        'baselines': {
            rep: {m: float(s) for m, s in scores.items()}
            for rep, scores in (getattr(search, 'baselines', {}) or {}).items()
            if not rep.startswith('_')
        },
    }


def extract_cgr_selection_info(hybrid_result):
    """Check if CGR reps were selected in the hybrid."""
    selected = hybrid_result.get('selected_reps', {})
    info = {}
    for cgr_rep in CGR_REPS:
        n_features = selected.get(cgr_rep, 0)
        info[cgr_rep] = {
            'selected': cgr_rep in selected,
            'n_features': n_features,
        }
    return info


# =============================================================================
# CORRELATION ANALYSIS (LINEAR CKA)
# =============================================================================

def linear_cka(X, Y):
    """Linear CKA between two representation matrices.

    Kornblith et al. (2019). Similarity of Neural Network Representations
    Revisited. ICML.

    Args:
        X: (n, d1) array.
        Y: (n, d2) array.

    Returns:
        float in [0, 1]. Higher = more similar.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX_norm_sq = np.linalg.norm(X.T @ X, 'fro') ** 2
    YtY_norm_sq = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    YtX_norm_sq = np.linalg.norm(Y.T @ X, 'fro') ** 2

    denom = np.sqrt(XtX_norm_sq * YtY_norm_sq)
    if denom < 1e-10:
        return 0.0
    return float(YtX_norm_sq / denom)


def compute_correlation_matrix(reps):
    """Pairwise Linear CKA for family reps.

    Returns:
        (rep_names, cka_matrix) — only includes reps that were generated.
    """
    available = [r for r in FAMILY_REPS if r in reps]
    n = len(available)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            cka = linear_cka(reps[available[i]], reps[available[j]])
            matrix[i, j] = cka
            matrix[j, i] = cka

    return available, matrix


# =============================================================================
# FAMILY ANALYSIS
# =============================================================================

def analyze_families(individual_results, primary_model='rf'):
    """Compare CGR vs SMILES within each family.

    Returns:
        {family: {best_cgr: {rep, rmse}, best_smiles: {rep, rmse}, cgr_wins: bool}}
    """
    analysis = {}
    for family_name, family_def in FAMILIES.items():
        cgr_reps = family_def['cgr']
        smiles_reps = family_def['smiles']

        best_cgr = _best_in_group(individual_results, cgr_reps, primary_model)
        best_smiles = _best_in_group(individual_results, smiles_reps,
                                     primary_model)

        cgr_wins = False
        if best_cgr and best_smiles:
            cgr_wins = best_cgr['rmse'] < best_smiles['rmse']

        analysis[family_name] = {
            'best_cgr': best_cgr,
            'best_smiles': best_smiles,
            'cgr_wins': cgr_wins,
        }
    return analysis


def _best_in_group(individual_results, rep_names, model):
    """Find best rep by RMSE within a group."""
    best = None
    for rn in rep_names:
        res = individual_results.get(rn, {}).get('models', {}).get(model)
        if res is None:
            continue
        rmse = res['rmse']['mean']
        if best is None or rmse < best['rmse']:
            best = {'rep': rn, 'rmse': rmse, 'r2': res['r2']['mean'],
                    'spearman': res['spearman']['mean']}
    return best


# =============================================================================
# MAIN PER-DATASET EVALUATION
# =============================================================================

def evaluate_dataset(dataset_name, n_folds, skip_hybrid=False,
                     skip_correlation=False):
    """Run full CGR comparison for one dataset."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    data = load_cgr_comparison_data(dataset_name)
    if data is None:
        return None

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n = len(labels)
    n_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_groups)

    print(f"  {n} reactions, {n_groups} groups, {actual_folds}-fold CV")

    # Check which CGR reps are available
    cgr_available = {k for k in CGR_REPS if k in reps}
    if not cgr_available:
        print("  WARNING: no CGR reps available — comparison not possible")

    # --- Phase A: Individual rep evaluation ---
    print(f"\n  --- Individual Rep Evaluation ---")
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n)
    individual_results = {}

    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        try:
            X = np.asarray(X, dtype=np.float64)
        except Exception:
            continue
        if X.ndim != 2 or X.shape[0] != n:
            continue

        rep_type = 'cgr' if rep_name in CGR_REPS else 'smiles'
        family = _get_family(rep_name)

        rep_results = {}
        for model_name, factory in models.items():
            res = evaluate_rep(X, labels, groups, factory(), actual_folds)
            if res is not None:
                rep_results[model_name] = res

        if rep_results:
            rf_rmse = rep_results.get('rf', {}).get('rmse', {}).get('mean')
            tag = ' [CGR]' if rep_type == 'cgr' else ''
            if rf_rmse is not None:
                print(f"    {rep_name:30s} {X.shape[1]:>5d}D  "
                      f"RF RMSE={rf_rmse:.2f}{tag}")

            individual_results[rep_name] = {
                'dim': X.shape[1],
                'family': family,
                'type': rep_type,
                'models': rep_results,
            }

    # --- Phase B: Hybrid comparison ---
    hybrid_comparison = {}
    if not skip_hybrid:
        smiles_only, smiles_plus_cgr, _ = partition_rep_pools(reps)

        hybrid_comparison['smiles_only'] = build_and_evaluate_hybrid(
            smiles_only, reps, labels, groups, n_folds, 'SMILES-only',
            dataset_name=dataset_name)

        if cgr_available:
            hybrid_comparison['smiles_plus_cgr'] = build_and_evaluate_hybrid(
                smiles_plus_cgr, reps, labels, groups, n_folds, 'SMILES+CGR',
                dataset_name=dataset_name)

            # CGR selection info
            hybrid_comparison['smiles_plus_cgr']['cgr_selection'] = \
                extract_cgr_selection_info(
                    hybrid_comparison['smiles_plus_cgr'])

    # --- Phase C: Compute deltas ---
    delta_by_model = {}
    if ('smiles_only' in hybrid_comparison
            and 'smiles_plus_cgr' in hybrid_comparison):
        so = hybrid_comparison['smiles_only'].get('hybrid_eval', {})
        sc = hybrid_comparison['smiles_plus_cgr'].get('hybrid_eval', {})
        for model_name in so:
            if model_name in sc:
                so_rmse = so[model_name]['rmse']['mean']
                sc_rmse = sc[model_name]['rmse']['mean']
                delta = sc_rmse - so_rmse  # negative = CGR helped
                pct = (delta / so_rmse * 100) if so_rmse > 0 else 0.0
                delta_by_model[model_name] = {
                    'rmse_delta': float(delta),
                    'pct_change': float(pct),
                }

    # --- Phase D: Family analysis ---
    family_analysis = analyze_families(individual_results)

    # --- Phase E: CKA correlation ---
    correlation = {}
    if not skip_correlation:
        names, matrix = compute_correlation_matrix(reps)
        if len(names) >= 2:
            correlation = {
                'rep_names': names,
                'cka_matrix': matrix.tolist(),
            }

    return {
        'n_reactions': n,
        'n_groups': n_groups,
        'n_folds': actual_folds,
        'group_key': data['metadata'].get('group_key', 'unknown'),
        'cgr_reps_available': sorted(cgr_available),
        'individual_results': individual_results,
        'hybrid_comparison': hybrid_comparison,
        'delta_by_model': delta_by_model,
        'family_analysis': family_analysis,
        'correlation_matrix': correlation,
    }


def _get_family(rep_name):
    """Return family name for a rep, or 'other'."""
    for fam_name, fam_def in FAMILIES.items():
        if rep_name in fam_def['cgr'] or rep_name in fam_def['smiles']:
            return fam_name
    return 'other'


# =============================================================================
# ANALYSIS PRINTING
# =============================================================================

def print_analysis(all_results):
    """Print cross-dataset CGR comparison analysis."""
    print(f"\n{'=' * 100}")
    print(f"CGR vs SMILES COMPARISON ANALYSIS")
    print(f"{'=' * 100}")

    for ds_name, ds in all_results.items():
        if ds is None:
            continue
        print(f"\n  Dataset: {ds_name} ({ds['n_reactions']} reactions, "
              f"{ds['n_groups']} groups, {ds['n_folds']}-fold)")

        # Table 1: Family comparison
        print(f"\n  --- Family Head-to-Head (RF RMSE, lower is better) ---")
        print(f"  {'Family':12s} {'Best CGR':25s} {'RMSE':>8s}   "
              f"{'Best SMILES':25s} {'RMSE':>8s}   {'Winner':>8s}")
        print(f"  {'-'*12} {'-'*25} {'-'*8}   {'-'*25} {'-'*8}   {'-'*8}")

        for fam_name, fam in ds.get('family_analysis', {}).items():
            cgr = fam.get('best_cgr')
            smi = fam.get('best_smiles')
            cgr_s = f"{cgr['rep']}" if cgr else '---'
            cgr_r = f"{cgr['rmse']:.2f}" if cgr else '---'
            smi_s = f"{smi['rep']}" if smi else '---'
            smi_r = f"{smi['rmse']:.2f}" if smi else '---'
            winner = 'CGR' if fam.get('cgr_wins') else 'SMILES'
            if not cgr or not smi:
                winner = '---'
            print(f"  {fam_name:12s} {cgr_s:25s} {cgr_r:>8s}   "
                  f"{smi_s:25s} {smi_r:>8s}   {winner:>8s}")

        # Table 2: Hybrid comparison
        hc = ds.get('hybrid_comparison', {})
        if 'smiles_only' in hc and 'smiles_plus_cgr' in hc:
            print(f"\n  --- Hybrid Pool Comparison ---")
            print(f"  {'Pool':15s} {'#Reps':>5s}  ", end='')
            model_names = sorted(
                set(hc['smiles_only'].get('hybrid_eval', {}).keys())
                & set(hc['smiles_plus_cgr'].get('hybrid_eval', {}).keys()))
            for mn in model_names:
                print(f"  {mn:>10s}", end='')
            print()
            print(f"  {'-'*15} {'-'*5}  " + '  '.join('-'*10 for _ in model_names))

            for pool_name, label in [('smiles_only', 'SMILES-only'),
                                     ('smiles_plus_cgr', 'SMILES+CGR')]:
                pool = hc[pool_name]
                print(f"  {label:15s} {pool['n_reps']:>5d}  ", end='')
                for mn in model_names:
                    ev = pool.get('hybrid_eval', {}).get(mn)
                    if ev:
                        print(f"  {ev['rmse']['mean']:>10.2f}", end='')
                    else:
                        print(f"  {'---':>10s}", end='')
                print()

            # Delta row
            delta = ds.get('delta_by_model', {})
            if delta:
                print(f"  {'Delta':15s} {'':>5s}  ", end='')
                for mn in model_names:
                    d = delta.get(mn)
                    if d:
                        print(f"  {d['rmse_delta']:>+10.2f}", end='')
                    else:
                        print(f"  {'---':>10s}", end='')
                print("   (negative = CGR helped)")

        # Table 3: CGR selection
        cgr_sel = hc.get('smiles_plus_cgr', {}).get('cgr_selection', {})
        if cgr_sel:
            print(f"\n  --- CGR Feature Selection (in SMILES+CGR hybrid) ---")
            for rep_name, info in cgr_sel.items():
                status = (f"YES ({info['n_features']} features)"
                          if info['selected'] else "NO")
                print(f"    {rep_name:25s} selected: {status}")

        # Table 4: CKA correlation
        corr = ds.get('correlation_matrix', {})
        if corr.get('rep_names'):
            names = corr['rep_names']
            matrix = corr['cka_matrix']
            print(f"\n  --- Linear CKA Correlation ---")
            header = f"  {'':25s}" + ''.join(f"  {n[:10]:>10s}" for n in names)
            print(header)
            for i, n in enumerate(names):
                row = f"  {n:25s}"
                for j in range(len(names)):
                    row += f"  {matrix[i][j]:>10.3f}"
                print(row)

        # Verdict
        fa = ds.get('family_analysis', {})
        fp_wins = fa.get('fingerprint', {}).get('cgr_wins', False)
        nn_wins = fa.get('neural', {}).get('cgr_wins', False)
        delta = ds.get('delta_by_model', {})
        hybrid_helps = any(d['rmse_delta'] < 0 for d in delta.values()) if delta else False

        print(f"\n  Verdict: CGR fingerprint {'beats' if fp_wins else 'loses to'} "
              f"SMILES FPs, CGR neural {'beats' if nn_wins else 'loses to'} "
              f"SMILES neural. "
              f"Hybrid pool {'improves' if hybrid_helps else 'does not improve'} "
              f"with CGR.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CGR vs SMILES representation comparison for reaction '
                    'yield prediction. Tests whether CGR reps add unique '
                    'signal beyond SMILES-based representations.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated datasets (default: all 3)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Max folds for GroupKFold (capped by #groups, default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path (default: tests/results/comparisons/'
             'opig_cgr_comparison.json)')
    parser.add_argument(
        '--skip-hybrid', action='store_true',
        help='Skip hybrid building (individual rep eval only)')
    parser.add_argument(
        '--skip-correlation', action='store_true',
        help='Skip CKA correlation analysis')

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets else ALL_DATASETS)

    print("=" * 70)
    print("CGR vs SMILES REPRESENTATION COMPARISON")
    print(f"  Phase 3 of CGR_EMBEDDING_RESEARCH.md")
    print(f"  Datasets: {datasets}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Max folds: {args.n_folds}")
    print(f"  Hybrid: {'skip' if args.skip_hybrid else 'yes'}")
    print(f"  CKA: {'skip' if args.skip_correlation else 'yes'}")
    print("=" * 70)

    all_results = {}
    for ds_name in datasets:
        try:
            result = evaluate_dataset(
                ds_name, args.n_folds,
                skip_hybrid=args.skip_hybrid,
                skip_correlation=args.skip_correlation)
            all_results[ds_name] = result
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Analysis
    if all_results:
        print_analysis(all_results)

    # Save
    output_path = (args.output or
                   'tests/results/comparisons/opig_cgr_comparison.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
