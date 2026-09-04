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

Split conditions (true leave-one-group-out GroupKFold; n_folds == n_groups):
  - additive:    leave-one-additive-out (23 groups → 23 folds)
  - aryl_halide: leave-one-aryl-halide-out (15 groups → 15 folds)
  - ligand:      leave-one-ligand-out (4 groups → 4 folds — hardest)
  - base:        leave-one-base-out (3 groups → 3 folds — very few, extreme OOS)

Datasets:
  - buchwald_hartwig (Ahneman/Dreher/Doyle, Science 2018, ~3955 reactions)
  - bh_doyle_plates (Doyle lab HTE, ~4599 reactions)

Paper baselines:
  - Ahneman et al. 2018: RF + DFT descriptors, R2=0.92 (random split)
  - Schwaller/yield-BERT 2021: R2=0.951 (random), drops to ~0.7 OOS
  - King-Smith et al. 2024: HiTEA framework, 39K+ HTE reactions
  - Saebi et al. 2024: showed leave-one-component-out drops vs random

Usage:
    # Smoke test (1 dataset, 2 splits; n_folds == n_groups always)
    python tests/experiments/opig_kingsmith_component_splits.py \\
        --datasets buchwald_hartwig --splits additive,aryl_halide

    # Full evaluation
    python tests/experiments/opig_kingsmith_component_splits.py

Reference:
    King-Smith et al. (2024). Nature Chemistry.
    Saebi et al. (2024). JCIM. DOI: 10.1021/acs.jcim.3c01524
    Bradshaw et al. (2025). ACS Central Science. DOI: 10.1021/acscentsci.5c00055
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load xgboost/lightgbm native OpenMP runtimes BEFORE rdkit (imported during
# rep generation) to avoid a duplicate-libomp segfault on macOS when these
# models and threaded sklearn estimators run in sequence. Optional deps —
# guard so environments without them are unaffected.
try:
    import xgboost as _xgboost  # noqa: F401
except Exception:
    pass
try:
    import lightgbm as _lightgbm  # noqa: F401
except Exception:
    pass

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
# Two warnings must survive that blanket. The first says the inner search had
# fewer groups than folds and quietly ran 2- or 3-fold instead of 5; the second
# says two cache entries collided. Both change what a number means.
warnings.filterwarnings('always', message=r'.*Reducing to .*-fold GroupKFold.*')
warnings.filterwarnings('always', message=r'.*[Cc]ollision.*')

from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    honest_rescore,
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
    labels = list(data['labels'])

    # Test-only row cap BEFORE rep generation (no-op unless KIRBY_SMOKE_N set).
    # Capping here keeps the slow rxnmapper/neural rep generators off the full
    # dataset and keeps components aligned for the per-row component->group map.
    _smoke = os.environ.get('KIRBY_SMOKE_N', '').strip()
    if _smoke.isdigit() and 0 < int(_smoke) < len(labels):
        n_keep = int(_smoke)
        n = len(labels)
        _idx = np.unique(np.linspace(0, n - 1, n_keep).round().astype(int))
        rxn_smiles = [rxn_smiles[i] for i in _idx]
        components = [components[i] for i in _idx]
        labels = [labels[i] for i in _idx]
        print(f"  [SMOKE] KIRBY_SMOKE_N={n_keep}: capped {n} -> {len(_idx)} reactions before rep gen")
    labels = np.asarray(labels, dtype=np.float64)

    # Build only the reps listed in DATASET_REPS for this dataset
    ds_config = DATASET_REPS.get(dataset_name, {})
    wanted = set(ds_config.get('general', []))
    wanted.update(ds_config.get('augmentation', []))

    # Test-only: under KIRBY_SMOKE_N drop the heavy rep generators that dominate
    # local wall time — neural reps (transformer/graph-net CPU forward passes)
    # and the mordred reaction variants (~1613-D each; can hang). Fingerprint /
    # descriptor reps (incl cgr_morgan) remain, so the LOO pipeline is still
    # exercised. No-op when the env var is unset, so server runs build the full
    # configured rep set.
    if os.environ.get('KIRBY_SMOKE_N', '').strip():
        _smoke_drop = {'reactiont5', 'chytorch_graphormer', 'cgr_smiles_lm'}
        _smoke_drop |= {r for r in wanted if 'mordred' in r}
        if _smoke_drop & wanted:
            print(f"  [SMOKE] dropping heavy reps: "
                  f"{sorted(_smoke_drop & wanted)}")
        wanted -= _smoke_drop

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
        'labels': labels,
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
    # Collect raw values + their lowercased forms in parallel.
    raw_values = []
    norm_values = []
    for comp in components:
        val = comp.get(split_key, '')
        raw = str(val).strip() if val else ''
        raw_values.append(raw)
        norm_values.append(raw.lower())

    # P1-12 warning: case-insensitive grouping merges entries that differed
    # only in capitalisation. If your BH CSV uses case to distinguish two
    # distinct compounds this will silently collapse them.
    raw_unique = set(raw_values)
    norm_unique = set(norm_values)
    if len(raw_unique) != len(norm_unique):
        # Find the names that collapsed.
        from collections import defaultdict
        per_norm = defaultdict(set)
        for raw, norm in zip(raw_values, norm_values):
            per_norm[norm].add(raw)
        collisions = {n: sorted(r) for n, r in per_norm.items() if len(r) > 1}
        warnings.warn(
            f"assign_component_groups[{split_key}]: case-insensitive grouping "
            f"merged {len(raw_unique)} raw names into {len(norm_unique)} "
            f"groups. Collisions: {collisions}. If these names refer to "
            f"distinct compounds, drop the .lower() step.",
            RuntimeWarning, stacklevel=2,
        )

    unique_vals = sorted(norm_unique)
    val_to_id = {v: i for i, v in enumerate(unique_vals)}
    groups = np.array([val_to_id[v] for v in norm_values])
    return groups, len(unique_vals), unique_vals


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep(X, labels, groups, model, n_folds):
    """Evaluate one rep x model via GroupKFold, return R2/RMSE/Spearman.

    Every list is FOLD-INDEX ALIGNED: a fold whose fit raised gets np.nan in
    each list rather than being dropped. Alignment is what lets the summary
    subtract this representation's score on one held-out group from the
    hybrid's score on the SAME held-out group.
    """
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
        except Exception as e:
            # Log fold-fit failures (OPIG-18) instead of silently dropping.
            print(f"      [fold fit failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            r2_scores.append(np.nan)
            rmse_scores.append(np.nan)
            spearman_scores.append(np.nan)
            continue

        preds = m.predict(X_val)

        ss_res = np.sum((y_val - preds) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        # LOO folds frequently have ss_tot ≈ 0 (single-group held out): treat
        # as undefined R2 (np.nan) and aggregate with nanmean below. The old
        # 0.0 fallback silently dragged the mean toward zero.
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        rmse_scores.append(np.sqrt(np.mean((y_val - preds) ** 2)))

        if len(y_val) > 2:
            rho, _ = spearmanr(y_val, preds)
            spearman_scores.append(rho if not np.isnan(rho) else np.nan)
        else:
            spearman_scores.append(np.nan)

    if not r2_scores or all(np.isnan(v) for v in r2_scores):
        return None

    return _summarize_scores(r2_scores, rmse_scores, spearman_scores)


def _summarize_scores(r2_scores, rmse_scores, spearman_scores):
    """Aggregate per-fold metrics, ignoring NaNs (LOO ss_tot=0 folds)."""
    r2_arr = np.asarray(r2_scores, dtype=np.float64)
    rmse_arr = np.asarray(rmse_scores, dtype=np.float64)
    sp_arr = np.asarray(spearman_scores, dtype=np.float64) if spearman_scores else None
    return {
        'r2': {
            'mean': float(np.nanmean(r2_arr)) if np.any(~np.isnan(r2_arr)) else float('nan'),
            'std': float(np.nanstd(r2_arr)) if np.any(~np.isnan(r2_arr)) else float('nan'),
        },
        'rmse': {
            'mean': float(np.nanmean(rmse_arr)) if np.any(~np.isnan(rmse_arr)) else float('nan'),
            'std': float(np.nanstd(rmse_arr)) if np.any(~np.isnan(rmse_arr)) else float('nan'),
        },
        'spearman': {
            'mean': (float(np.nanmean(sp_arr))
                     if sp_arr is not None and np.any(~np.isnan(sp_arr)) else float('nan')),
            'std': (float(np.nanstd(sp_arr))
                    if sp_arr is not None and np.any(~np.isnan(sp_arr)) else float('nan')),
        },
        'n_folds_completed': int(np.sum(~np.isnan(rmse_arr))),
        # Fold-index-aligned held-out scores. Kept so the summary can pair the
        # hybrid against a base representation on the same held-out group
        # instead of comparing two maxima.
        'r2_per_fold': [None if np.isnan(v) else float(v) for v in r2_arr],
        'rmse_per_fold': [None if np.isnan(v) else float(v) for v in rmse_arr],
    }


def _evaluate_hybrid_per_fold(
    dataset_name, split_key, base_reps, aug_dict, labels, groups,
    fold_splits, models, pruning_mode='default_pruning',
):
    """Build the KIRBy hybrid PER FOLD and evaluate downstream models.

    Each fold:
      1. Runs find_best_hybrid on train indices only (no peeking at val).
      2. Pulls FittedResult (selected_indices + augmentation_info) for 'rf'.
      3. Applies apply_feature_selection on train_reps -> train_hybrid.
         Applies it on val_reps -> val_hybrid.
      4. If augmentation_info present, applies apply_augmentation_selection on
         train/val augs and hstack-s to the hybrid blocks.
      5. Trains each downstream model on (scaled) train_hybrid, predicts on
         val_hybrid, accumulates R2/RMSE/Spearman.

    Returns:
        (rep_results_dict_for_kirby_hybrid_per_model, dim_estimate, top3_first_fold)
        or (None, 0, None) if every fold failed.
    """
    from kirby.find_best_hybrid import find_best_hybrid
    from kirby.hybrid import apply_feature_selection, apply_augmentation_selection
    from experiments._shared import resolve_pruning_kwargs

    per_model_scores = {m: {'r2': [], 'rmse': [], 'spearman': []}
                        for m in models}
    dims = []
    top3_first_fold = None

    src_reps = base_reps if base_reps else {}
    has_aug = bool(aug_dict)

    def _pad_fold():
        """Record a missing held-out score for every model on this fold.

        Called on any path that abandons a fold. Without it the per-model
        lists lose fold alignment, and the paired hybrid-minus-base
        subtraction in the summary would compare two different held-out
        groups.
        """
        for _scores in per_model_scores.values():
            _scores['r2'].append(np.nan)
            _scores['rmse'].append(np.nan)
            _scores['spearman'].append(np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        # ----- slice everything to train indices -----
        train_reps = {r: arr[train_idx] for r, arr in src_reps.items()}
        train_aug = ({r: arr[train_idx] for r, arr in aug_dict.items()}
                     if has_aug else None)
        train_groups = groups[train_idx]
        train_labels = labels[train_idx]

        # ----- build hybrid on train only -----
        try:
            _pkw = resolve_pruning_kwargs(
                pruning_mode, len(train_labels),
                dataset_label=f'{dataset_name}/loo_{split_key}/fold{fold_idx}',
            )
            # Test-only: shrink the per-fold search under KIRBY_SMOKE_N (tiny
            # budget, rf-only evaluator, cheap lightgbm_gain importance) so the
            # per-fold search finishes quickly on tiny local subsets. No-op
            # (full budgets/tiers) when the env var is unset, so server runs and
            # the precompute cache key are unchanged.
            _is_smoke = bool(os.environ.get('KIRBY_SMOKE_N', '').strip())
            _smoke_budgets = [25] if _is_smoke else FULL_BUDGETS
            _eval_list = ['rf'] if _is_smoke else INTERNAL_EVALUATORS
            _smoke_imp = ({'importance_methods': ['lightgbm_gain']}
                          if _is_smoke else {})
            fold_search = find_best_hybrid(
                train_reps, train_labels,
                groups=train_groups,
                augmentations=train_aug,
                augmentation_strategy=('uncapped' if has_aug else 'none'),
                budgets=_smoke_budgets,
                task_type='regression',
                internal_evaluators=_eval_list,
                return_fitted=True,
                cache=True,
                dataset_name=dataset_name,
                # find_best_hybrid(cache=True) requires a tier. These BH datasets
                # have no shared precompute, so hardcode 'general' (same
                # convention as opig_blueprint_transfer); the per-fold
                # condition_suffix keeps each fold's cache file distinct.
                tier='general',
                condition_suffix=f'loo_{split_key}_fold{fold_idx}',
                adaptive_pruning=0.25,
                **_smoke_imp,
                **_pkw,
            )
        except Exception as e:
            print(f"    fold {fold_idx}: find_best_hybrid failed: {e}")
            _pad_fold()
            continue

        if fold_idx == 0 and fold_search is not None:
            top3_first_fold = extract_top3_base_reps(fold_search)

        fitted = (fold_search.fitted_results.get('rf')
                  if fold_search is not None else None)
        if fitted is None:
            print(f"    fold {fold_idx}: no fitted 'rf' result; skipping")
            _pad_fold()
            continue

        # ----- assemble train/val hybrid using the fold's selection -----
        try:
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
            train_hybrid = apply_feature_selection(train_reps, fi)
            val_reps = {r: arr[val_idx] for r, arr in src_reps.items()}
            val_hybrid = apply_feature_selection(val_reps, fi)

            if fitted.augmentation_info and has_aug:
                val_aug = {r: arr[val_idx] for r, arr in aug_dict.items()}
                train_aug_block = apply_augmentation_selection(
                    train_aug, fitted.augmentation_info)
                val_aug_block = apply_augmentation_selection(
                    val_aug, fitted.augmentation_info)
                if train_aug_block is not None and val_aug_block is not None:
                    train_hybrid = np.hstack([train_hybrid, train_aug_block])
                    val_hybrid = np.hstack([val_hybrid, val_aug_block])
        except Exception as e:
            print(f"    fold {fold_idx}: hybrid assembly failed: {e}")
            _pad_fold()
            continue

        dims.append(train_hybrid.shape[1])

        train_hybrid = np.asarray(train_hybrid, dtype=np.float64)
        val_hybrid = np.asarray(val_hybrid, dtype=np.float64)
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Scale unless binary
        if not _is_binary_rep(train_hybrid):
            scaler = StandardScaler()
            train_hybrid_s = scaler.fit_transform(train_hybrid)
            val_hybrid_s = scaler.transform(val_hybrid)
        else:
            train_hybrid_s, val_hybrid_s = train_hybrid, val_hybrid

        # ----- train each downstream model -----
        for model_name, factory in models.items():
            try:
                m = clone(factory())
                m.fit(train_hybrid_s, y_train)
                preds = m.predict(val_hybrid_s)
            except Exception:
                per_model_scores[model_name]['r2'].append(np.nan)
                per_model_scores[model_name]['rmse'].append(np.nan)
                per_model_scores[model_name]['spearman'].append(np.nan)
                continue

            ss_res = np.sum((y_val - preds) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            per_model_scores[model_name]['r2'].append(
                1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
            per_model_scores[model_name]['rmse'].append(
                np.sqrt(np.mean((y_val - preds) ** 2)))
            if len(y_val) > 2:
                rho, _ = spearmanr(y_val, preds)
                per_model_scores[model_name]['spearman'].append(
                    rho if not np.isnan(rho) else np.nan)
            else:
                per_model_scores[model_name]['spearman'].append(np.nan)

    if not dims:
        return None, 0, top3_first_fold

    # Aggregate per model
    rep_results = {}
    for model_name, scores in per_model_scores.items():
        if not scores['rmse'] or all(np.isnan(v) for v in scores['rmse']):
            continue
        rep_results[model_name] = _summarize_scores(
            scores['r2'], scores['rmse'], scores['spearman'])

    return rep_results, int(np.mean(dims)), top3_first_fold


def _paired_hybrid_delta(results):
    """Per-model hybrid-minus-base-representation on the same held-out groups.

    `results` is {rep_name: {'models': {model: summary}}} where each summary
    carries 'r2_per_fold', fold-index aligned. Returns honest_rescore's output
    keyed by model: the hybrid is the only candidate, and the base
    representation is re-picked on every held-out group from the other groups.

    Empty when no summary carries per-fold scores — a result file written
    before 2026-09-04 has none, and there is nothing to pair.
    """
    hybrid = results.get('kirby_hybrid', {}).get('models', {})
    candidates_by_model = {}
    baselines_by_model = {}
    for model_name, summary in hybrid.items():
        folds = summary.get('r2_per_fold')
        if not folds:
            continue
        candidates_by_model[model_name] = [
            {'name': 'kirby_hybrid', 'fold_scores': list(folds)}]

    for rep_name, rep_data in results.items():
        if rep_name == 'kirby_hybrid':
            continue
        for model_name, summary in rep_data.get('models', {}).items():
            folds = summary.get('r2_per_fold')
            if not folds:
                continue
            baselines_by_model.setdefault(model_name, {})[rep_name] = list(folds)

    if not candidates_by_model or not baselines_by_model:
        return {}
    # R2, higher is better. metric_name is left off: 'r2' as computed here is a
    # per-fold coefficient of determination, not a BUILTIN_METRICS entry.
    return honest_rescore(candidates_by_model, baselines_by_model,
                          lower_is_better=False)


def evaluate_split(dataset_name, data, split_key, n_folds, pruning_mode='default_pruning'):
    """Evaluate all reps + hybrid under one component split.

    Note: ``n_folds`` is IGNORED — this experiment runs TRUE leave-one-group-out
    (n_folds == n_groups) so the docstring/cache_suffix and the CV setup agree.
    The parameter is kept for backward-compatible CLI but unused.

    Returns dict with per-rep per-model metrics and hybrid comparison.
    """
    del n_folds  # unused; we always run true LOO
    reps = data['reps']
    labels = data['labels']
    components = data['components']

    groups, n_groups, group_names = assign_component_groups(components, split_key)
    actual_folds = n_groups  # TRUE leave-one-group-out

    print(f"\n  --- Split: leave-one-{split_key}-out "
          f"({n_groups} groups, {actual_folds}-fold TRUE LOO) ---")

    if actual_folds < 2:
        print(f"  SKIPPED: only {n_groups} groups, need >= 2 for CV")
        return {'skipped': True, 'reason': f'only {n_groups} groups',
                'n_groups': n_groups, 'group_names': group_names}

    # Partition reps into base + augmentation according to DATASET_REPS
    aug_reps = set(DATASET_REPS.get(dataset_name, {}).get('augmentation', []))
    base_reps = {k: v for k, v in reps.items() if k not in aug_reps}
    aug_dict = {k: v for k, v in reps.items() if k in aug_reps}

    # Build fold splits ONCE so the hybrid is built and evaluated on identical
    # train/val partitions (avoids any drift between baseline eval and hybrid).
    sample_X = next(iter(reps.values()))
    cv = GroupKFold(n_splits=actual_folds)
    fold_splits = list(cv.split(sample_X, labels, groups=groups))

    # ===== Per-fold hybrid build + evaluate =====
    n = len(labels)
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n)

    print(f"  Building KIRBy hybrid per fold ({actual_folds} folds, train-only) ...")
    hybrid_results, hybrid_dim, top3 = _evaluate_hybrid_per_fold(
        dataset_name=dataset_name,
        split_key=split_key,
        base_reps=base_reps if base_reps else reps,
        aug_dict=aug_dict,
        labels=labels, groups=groups,
        fold_splits=fold_splits,
        models=models,
        pruning_mode=pruning_mode,
    )

    if top3:
        print(f"  Top-3 base reps (fold 0): "
              f"{', '.join(f'{nm}={sc:.4f}' for nm, sc in top3)}")

    # ===== Evaluate base reps (same fold_splits, no leakage) =====
    print(f"  Evaluating {len(reps)} base reps x {len(models)} models "
          f"x {actual_folds}-fold LOO...")

    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
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
            print(f"    {rep_name:25s} {model_name:8s}  "
                  f"R2={r2:.4f}  RMSE={res['rmse']['mean']:.2f}  ({elapsed:.1f}s)")

        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    # Insert hybrid results
    if hybrid_results:
        results['kirby_hybrid'] = {'dim': hybrid_dim, 'models': hybrid_results}
        for model_name, res in hybrid_results.items():
            print(f"    {'kirby_hybrid':25s} {model_name:8s}  "
                  f"R2={res['r2']['mean']:.4f}  RMSE={res['rmse']['mean']:.2f}  ***")

    return {
        'split_key': split_key,
        'n_groups': n_groups,
        'n_folds': actual_folds,
        'group_names': group_names,
        'results': results,
        # Per-model, per-held-out-group hybrid-minus-base-representation. The
        # base representation is re-picked on each held-out group from the
        # other groups only, so it is not the same-group maximum over 70 draws
        # the Delta column used to report.
        'paired_hybrid_delta': _paired_hybrid_delta(results),
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print cross-split comparison: how does performance degrade?"""
    print(f"\n{'=' * 100}")
    print(f"COMPONENT-WISE EXTRAPOLATION ANALYSIS (King-Smith et al. 2024)")
    print(f"{'=' * 100}")

    # The Delta column is now computed PER DOWNSTREAM MODEL and PER HELD-OUT
    # GROUP. It used to be one maximum minus another: the largest R2 over every
    # representation-and-model pair on the base side (about 70 draws) against
    # the largest over the 5 hybrid models. Both maxima were taken on the same
    # held-out groups they were measured on, and the base side drew from a much
    # larger set, so the two were not comparable. Changed 2026-09-04.
    #
    # For each model, the base representation is re-picked on every held-out
    # group from the other groups' scores only, and its score on that group is
    # subtracted from the hybrid's score on the SAME group. This is the
    # arithmetic honest_rescore already runs; both arms pay the same tax.
    for ds_name, ds_results in all_results.items():
        print(f"\n  Dataset: {ds_name} ({ds_results.get('n_reactions', '?')} reactions)")
        print(f"  {'Split key':15s} {'#Groups':>7s} {'Model':10s} "
              f"{'Hybrid R2':>10s} {'BestBase R2':>12s} {'Delta':>9s} "
              f"{'Folds':>6s}")
        print(f"  {'-'*15} {'-'*7} {'-'*10} {'-'*10} {'-'*12} {'-'*9} {'-'*6}")

        for split_key in ALL_SPLIT_KEYS:
            sr = ds_results.get('splits', {}).get(split_key)
            if sr is None or sr.get('skipped'):
                reason = sr.get('reason', 'not run') if sr else 'not run'
                print(f"  {split_key:15s} {'—':>7s} {'SKIPPED':10s} "
                      f"{'—':>10s} {'—':>12s} {'—':>9s} {'—':>6s}  ({reason})")
                continue

            results = sr.get('results', {})
            if not results:
                print(f"  {split_key:15s} {sr['n_groups']:>7d} {'NO RESULTS':10s}")
                continue

            paired = _paired_hybrid_delta(results)
            if not paired:
                print(f"  {split_key:15s} {sr['n_groups']:>7d} "
                      f"{'NO PER-FOLD SCORES':10s}  (re-run: this result file "
                      f"predates r2_per_fold)")
                continue

            first = True
            for model_name in sorted(paired):
                r = paired[model_name]
                h, b = r['hybrid_honest'], r['best_single_honest']
                d = r['delta_mean_of_folds']
                h_s = f"{h:.4f}" if not np.isnan(h) else "---"
                b_s = f"{b:.4f}" if not np.isnan(b) else "---"
                d_s = f"{d:+.4f}" if not np.isnan(d) else "---"
                key_s = split_key if first else ''
                grp_s = f"{sr['n_groups']:>7d}" if first else f"{'':>7s}"
                first = False
                print(f"  {key_s:15s} {grp_s} {model_name:10s} "
                      f"{h_s:>10s} {b_s:>12s} {d_s:>9s} "
                      f"{r['n_paired_folds']:>6d}")

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
        '--n-folds', type=int, default=0,
        help='IGNORED. True leave-one-group-out is always used '
             '(n_folds == n_groups per split).')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path (default: tests/results/comparisons/'
             'opig_kingsmith_component_splits.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets else ALL_DATASETS)
    split_keys = (args.splits.split(',') if args.splits else ALL_SPLIT_KEYS)

    print("=" * 70)
    print("COMPONENT-WISE EXTRAPOLATION EXPERIMENT")
    print(f"  King-Smith et al. (2024), Saebi et al. (2024)")
    print(f"  Datasets: {datasets}")
    print(f"  Split axes: {split_keys}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: TRUE LOO (n_folds == n_groups per split)")
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
                    ds_name, data, split_key, args.n_folds,
                    pruning_mode=args.pruning_mode)
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
