#!/usr/bin/env python3
"""
⛔ CUT 2026-09-23 — TOO FEW HELD-OUT COMPONENTS TO MEASURE ANYTHING.
Do not submit. Leave-one-component-out gives 4 held-out ligands and 3
held-out bases. Measured on the ligand axis 2026-09-23: fold spreads of 8.1
to 15.9 RMSE in yield units against hybrid-minus-baseline deltas of 1.2 to
7.2 -- the delta sits inside the spread on every one of the 5 downstream
models, and with 3 or 4 folds no grid size makes it readable. The additive
axis has 23 components and would be measurable, but this file runs all four
axes together, so it goes as a whole. Removed from submit_everything.sh.

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

from experiments._shared import _resolve_dataset_metric
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    fold_reselection,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
    RULE_N_GENERAL, RULE_N_DISTANCE, build_fold_hybrid, top_hybrids,
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

# The owner's set of hybrids, 2026-10-02: for each downstream model, the top 2
# hybrids from the general search plus the top 1 from the distance search, each
# built and scored on every held-out component. 'general1' keeps the existing
# 'kirby_hybrid' name.
RULE_SLOTS = ['general1', 'general2', 'distance1']


def _slot_key(slot):
    """Result key for one slot of the owner's set. general1 keeps the old name."""
    return 'kirby_hybrid' if slot == 'general1' else f'kirby_hybrid__{slot}'


def _rule_entries(gen_search, dist_search, model_name, context):
    """Top 2 general + top 1 distance hybrids for one downstream model, read off
    this fold's two searches (owner's set, 2026-10-02).

    Returns ({slot: (tier, top_hybrids entry)}, [reason per missing slot]).
    """
    out, missing = {}, []
    for lineage, tier, search, n in (
            ('general', 'general', gen_search, RULE_N_GENERAL),
            ('distance', 'general_distance', dist_search, RULE_N_DISTANCE)):
        if search is None:
            missing.append(f'{lineage} for {model_name}: the {tier} search '
                           f'did not run on this fold')
            continue
        try:
            th = top_hybrids(search, model_name, n, require_full=False,
                             context=f'{context}/{tier}/{model_name}')
        except Exception as e:
            missing.append(f'{lineage} for {model_name}: '
                           f'{type(e).__name__}: {e}')
            continue
        for e in th['selected']:
            out[f"{lineage}{e['rank']}"] = (tier, e)
        if not th['complete']:
            missing.append(f'{lineage} for {model_name}: '
                           f'{th["shortfall_reason"]}')
    return out, missing


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
    selection_out=None,
):
    """Build the KIRBy hybrid PER FOLD and evaluate downstream models.

    Each fold:
      1. Runs find_best_hybrid on train indices only (no peeking at val), over
         the same downstream models this function trains.
      2. Per downstream model, pulls the FittedResult stored FOR THAT MODEL
         (selected_indices + augmentation_info). A model the search stored no
         recipe for gets no score on the fold and is named; no other model's
         recipe is used in its place.
      3. Applies apply_feature_selection on train_reps -> train_hybrid.
         Applies it on val_reps -> val_hybrid.
      4. If augmentation_info present, applies apply_augmentation_selection on
         train/val augs and hstack-s to the hybrid blocks.
      5. Trains that downstream model on (scaled) train_hybrid, predicts on
         val_hybrid, accumulates R2/RMSE/Spearman.

    Returns:
        (rep_results_dict_for_kirby_hybrid_per_model, dim_estimate, top3_first_fold)
        or (None, 0, None) if every fold failed.

    `selection_out`, when a dict is passed, is FILLED with fold 0's top-2 rows
    and its two hybrid deltas. It is an out-parameter and not a fourth return
    value so that the 3-tuple this function has always returned is unchanged.
    """
    from kirby.find_best_hybrid import find_best_hybrid
    from kirby.hybrid import apply_feature_selection, apply_augmentation_selection
    from experiments._shared import resolve_pruning_kwargs

    # One score list per (slot of the owner's set, downstream model), owner's
    # set 2026-10-02. 'general1' is what this function always scored.
    per_slot = {s: {m: {'r2': [], 'rmse': [], 'spearman': []} for m in models}
                for s in RULE_SLOTS}
    per_model_scores = per_slot['general1']
    rule_recipes = []
    rule_missing = []
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
        for _by_m in per_slot.values():
            for _scores in _by_m.values():
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
            from experiments._shared import smoke_budgets
            _smoke_budgets = smoke_budgets() if _is_smoke else FULL_BUDGETS
            _eval_list = ['rf'] if _is_smoke else INTERNAL_EVALUATORS
            _smoke_imp = ({'importance_methods': ['lightgbm_gain']}
                          if _is_smoke else {})
            fold_search = find_best_hybrid(
                train_reps, train_labels,
                groups=train_groups,
                augmentations=train_aug,
                augmentation_strategy=('uncapped' if has_aug else 'none'),
                budgets=_smoke_budgets,
                # metric= is REQUIRED since 2026-09-10. Without it this call
                # raised on every fold and the raise was swallowed by the
                # `except Exception` below, so the run finished and wrote a
                # results file with no hybrid rows in it. Resolved from the
                # registry here, on the experiment side -- src/ does not look a
                # dataset up. (ginkgo_hic is spearman, not rmse, so a hardcoded
                # metric would be wrong for these.)
                metric=_resolve_dataset_metric(dataset_name)[0],
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
                # ⛔ speed='fast' SINCE 2026-09-18, OWNER'S DECISION.
                #
                # Nothing precomputes this. The comment above says why: these
                # Buchwald datasets have no shared precompute, and the per-fold
                # condition_suffix makes every fold its own search identity. So
                # every one of these is built live inside the job and there is
                # no stored full grid to miss by asking for a shorter one.
                #
                # The multiplier is the fold count. A leave-one-component-out
                # split is one fold per component: 23 additives, 15 aryl
                # halides, 4 ligands and 3 bases, so 45 folds per dataset where
                # an ordinary run has 5, across 2 datasets. At the full grid
                # that was about 44 million model fits.
                #
                # Full is 5 allocation methods x 3 importance methods x 4
                # representation counts x 8 budgets x 17 (model, evaluator)
                # pairs = 8,160 configurations. Fast is 3 x 1 x 2 x 8 x 9 = 432,
                # so about 44 million model fits becomes about 2.3 million.
                #
                # ⛔ THE MODEL AXIS IS NO LONGER PART OF THE CUT, CHANGED
                # 2026-09-28. This call used to pass no models= argument. With
                # models=None and speed='fast', find_best_hybrid keeps only the
                # models on its 'fast' tier (find_best_hybrid.py:8639-8646), so
                # the search scored rf, lightgbm and dnn. The loop below then
                # trained all five names in MODELS_LIST
                # (opig_kingsmith_component_splits.py:128) on those features. Two
                # of the five downstream models the table reports were never
                # scored by the search at all, and nothing said so.
                #
                # models= is now the same five names the loop below trains. With
                # a list given, the fast filter on the model axis does not run
                # (find_best_hybrid.py:8650-8660), so fast now cuts only the
                # allocation methods, the importance methods and the
                # representation counts.
                #
                # Cost, from the axis counts written above: 3 allocation methods
                # x 1 importance method x 2 representation counts x 8 budgets x
                # 17 (model, evaluator) pairs = 816 configurations per fold,
                # where the model-axis cut made it 432. So roughly double the
                # fits per fold, over the same 45 folds per dataset.
                models=list(models.keys()),
                #
                # ⛔ internal_evaluators=_eval_list above is explicit, so fast
                # does not shorten THAT axis. The cut comes from the allocation
                # methods, the importance methods, the representation counts and
                # the model list.
                #
                # ⛔ IT OVERLAPS WITH adaptive_pruning=0.25 BELOW. Pruning votes
                # on the allocation axis, and fast searches 3 allocation methods
                # instead of 5, so there is less for the vote to act on. Neither
                # is broken by the other; they are two levers on one axis and
                # this is recorded so a later reader does not add a third.
                speed='fast',
                adaptive_pruning=0.25,
                **_smoke_imp,
                **_pkw,
            )
        except Exception as e:
            print(f"    fold {fold_idx}: find_best_hybrid failed: {e}")
            _pad_fold()
            continue

        # The distance search on the same training rows: the third hybrid of
        # the owner's set (top 2 general + top 1 distance per model,
        # 2026-10-02). The distance precompute's settings -- tier
        # 'general_distance', DISTANCE_EVALUATORS, DISTANCE_BUDGETS -- over the
        # downstream models this function trains, with the same speed='fast'
        # and pruning as the general search above. Under KIRBY_SMOKE_N it takes
        # the smoke budgets and the knn evaluator alone.
        dist_search = None
        try:
            dist_search = find_best_hybrid(
                train_reps, train_labels,
                groups=train_groups,
                augmentations=train_aug,
                augmentation_strategy=('uncapped' if has_aug else 'none'),
                budgets=(([b for b in _smoke_budgets
                           if b <= max(DISTANCE_BUDGETS)]
                          or [min(DISTANCE_BUDGETS)]) if _is_smoke
                         else DISTANCE_BUDGETS),
                metric=_resolve_dataset_metric(dataset_name)[0],
                task_type='regression',
                internal_evaluators=(['knn'] if _is_smoke
                                     else DISTANCE_EVALUATORS),
                return_fitted=False,
                cache=True,
                dataset_name=dataset_name,
                tier='general_distance',
                condition_suffix=f'loo_{split_key}_fold{fold_idx}',
                models=list(models.keys()),
                speed='fast',
                adaptive_pruning=0.25,
                **_smoke_imp,
                **_pkw,
            )
        except Exception as e:
            print(f"    fold {fold_idx}: distance find_best_hybrid failed: {e}")
            rule_missing.append({'fold': fold_idx, 'reason':
                                 f'distance search raised '
                                 f'{type(e).__name__}: {e}'})

        if fold_idx == 0 and fold_search is not None:
            top3_first_fold = extract_top3_base_reps(fold_search)

            # Top 2 beside that top 3, from the SAME fold-0 search, and kept
            # instead of printed. Every search here is built on one fold's
            # training components only, so these rows are fold 0's ranking and
            # the keys say so. Until now the only ranking this experiment
            # produced was the 3 representation names printed on the line above
            # and thrown away; the saved JSON held per-representation scores but
            # no ranking and no hybrid configuration other than the one that was
            # rebuilt and scored. Rows already stored by this search: 0 new
            # model fits, on top of the fits the fold already paid for.
            if selection_out is not None:
                try:
                    from experiments._shared import (select_model_rep_pairs,
                                                      top_hybrids,
                                                      hybrid_deltas,
                                                      resolve_comparison_models)
                    _ctx = f'{dataset_name}/loo_{split_key}/fold0'
                    # WHICH DOWNSTREAM MODELS THE TWO KEYS BELOW COVER, resolved
                    # 2026-09-28. `models` is the factory dict this fold built.
                    # Fold 0's search scored whatever it scored on top of that:
                    # svm drops itself above 10,000 rows, and a dataset's skip
                    # policy can drop a name. The two disagree whenever that
                    # happened, and the loop below then asked for a model with no
                    # stored rows.
                    #
                    # `models` stays what this fold ASKS for, so the set of models
                    # reported does not change: it goes in as `requested`, not
                    # None. A name the search never scored is dropped and named,
                    # never swapped for another model's rows.
                    #
                    # Both lists are SAVED, not only printed. Until now every
                    # reason below went to the terminal alone.
                    _cmp_models, _cmp_dropped = resolve_comparison_models(
                        fold_search, list(models.keys()))
                    selection_out['top2_models_compared'] = \
                        list(_cmp_models)
                    selection_out['top2_models_dropped'] = {
                        _n: _w for _n, _w in _cmp_dropped}
                    for _n, _w in _cmp_dropped:
                        print(f"    fold 0: downstream model {_n} not "
                              f"compared: {_w}")
                    _pairs = select_model_rep_pairs(
                        fold_search, models=_cmp_models,
                        n_base=2, n_hybrids=2,
                        # require_full=False: a downstream model this search
                        # never scored yields a short list and a line in
                        # 'shortfalls'. The default raises, and the raise would
                        # be caught by the fold loop and cost the fold.
                        require_full=False, context=_ctx)

                    # The hybrid against two reference numbers, per downstream
                    # model: the mean of the base representations that actually
                    # fed it, and the best single base representation. A hybrid
                    # can beat the first and lose to the second.
                    _deltas = {}
                    for _m in _cmp_models:
                        try:
                            # hybrid= is passed by hand. hybrid_deltas(search,
                            # m) with no hybrid reads
                            # _picked['selected'][0]['row'] (_shared.py:7432)
                            # and top_hybrids' entries carry no 'row' key --
                            # score_mean and blueprint sit at the top level of
                            # the entry -- so the defaulted call raises
                            # KeyError('row'). Checked against those two
                            # functions on 2026-09-18.
                            _t1 = top_hybrids(fold_search, _m, 1,
                                              require_full=False, context=_ctx)
                            if not _t1['selected']:
                                print(f"    fold 0: no hybrid configuration "
                                      f"stored for downstream model {_m}")
                                selection_out[
                                    'top2_models_dropped'][_m] = (
                                    'no hybrid delta: %s'
                                    % _t1['shortfall_reason'])
                                continue
                            _deltas[_m] = hybrid_deltas(
                                fold_search, _m, hybrid=_t1['selected'][0],
                                context=_ctx)
                        except Exception as _e:
                            print(f"    fold 0: hybrid_deltas[{_m}] failed: {_e}")
                            selection_out['top2_models_dropped'][_m] = (
                                'no hybrid delta: %s: %s'
                                % (type(_e).__name__, _e))

                    # The Blueprint object off each selected hybrid: it is a
                    # recipe object, not JSON, and every field of it reported
                    # here (allocation method, importance method, budget,
                    # internal evaluator, step size, representation count) is
                    # already a key of the same entry. Dropped AFTER
                    # hybrid_deltas, which reads Blueprint.allocation_per_fold
                    # to find the representations that fed the hybrid.
                    for _sel in _pairs.get('per_model', {}).values():
                        for _e in _sel['hybrid']['selected']:
                            _e.pop('blueprint', None)

                    selection_out['top2_pairs_by_model_fold0'] = _pairs
                    # The distance search's top 1 per model beside the general
                    # top 2 (owner's set, 2026-10-02). Fold 0's training rows.
                    if dist_search is not None:
                        _d1 = {}
                        for _m in _cmp_models:
                            _t = top_hybrids(dist_search, _m, RULE_N_DISTANCE,
                                             require_full=False, context=_ctx)
                            for _e in _t['selected']:
                                _e.pop('blueprint', None)
                            _d1[_m] = _t
                        selection_out['distance_top1_by_model_fold0'] = _d1
                    if _deltas:
                        selection_out['hybrid_deltas_by_model_fold0'] = _deltas
                except Exception as _e:
                    # A failure here leaves both keys out of the split's saved
                    # dict. Every number this file reported before stays where
                    # it was, and the fold continues.
                    print(f"    fold 0: top-2 selection failed: {_e}")
                    selection_out['top2_pairs_by_model_fold0_absent_reason'] = \
                        f"{type(_e).__name__}: {_e}"

        _fitted_all = (getattr(fold_search, 'fitted_results', None) or {}
                       if fold_search is not None else {})
        if not _fitted_all:
            # general1 gets a missing score for every model below; general2 and
            # distance1 are still built from the stored rankings.
            print(f"    fold {fold_idx}: the search stored no fitted recipe "
                  f"for any downstream model; general1 not scored this fold")

        val_reps = {r: arr[val_idx] for r, arr in src_reps.items()}
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        val_aug = ({r: arr[val_idx] for r, arr in aug_dict.items()}
                   if has_aug else None)

        def _score_slot(train_hybrid, val_hybrid, factory):
            """(r2, rmse, rho) on the held-out component, or None if the fit
            failed. The same scaling every hybrid arm gets."""
            train_hybrid = np.asarray(train_hybrid, dtype=np.float64)
            val_hybrid = np.asarray(val_hybrid, dtype=np.float64)
            if not _is_binary_rep(train_hybrid):
                scaler = StandardScaler()
                train_hybrid_s = scaler.fit_transform(train_hybrid)
                val_hybrid_s = scaler.transform(val_hybrid)
            else:
                train_hybrid_s, val_hybrid_s = train_hybrid, val_hybrid
            try:
                m = clone(factory())
                m.fit(train_hybrid_s, y_train)
                preds = m.predict(val_hybrid_s)
            except Exception:
                return None
            ss_res = np.sum((y_val - preds) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            if len(y_val) > 2:
                rho, _ = spearmanr(y_val, preds)
                rho = rho if not np.isnan(rho) else np.nan
            else:
                rho = np.nan
            return r2, rmse, rho

        # The 2nd general and the 1st distance hybrid of the owner's set, per
        # model, rebuilt on this fold's training rows from their recipe and
        # scored on the held-out component (2026-10-02). general1 is built
        # below from the search's own fitted recipe, as before.
        for model_name, factory in models.items():
            _entries, _miss = _rule_entries(
                fold_search, dist_search, model_name,
                f'{dataset_name}/loo_{split_key}/fold{fold_idx}')
            for _r in _miss:
                rule_missing.append({'fold': fold_idx, 'model': model_name,
                                     'reason': _r})
            for _slot, (_tier, _e) in sorted(_entries.items()):
                rule_recipes.append({
                    'fold': fold_idx, 'model': model_name, 'slot': _slot,
                    'tier': _tier, 'recipe': _e.get('label'),
                    'budget': _e.get('budget'),
                    'score_train_rows': _e.get('score_mean')})
            for _slot in ('general2', 'distance1'):
                _res = None
                if _slot in _entries:
                    _bp = _entries[_slot][1].get('blueprint')
                    try:
                        _use_aug = (has_aug and getattr(
                            _bp, 'augmentation_strategy', 'none') != 'none')
                        _trh, _vah, _ = build_fold_hybrid(
                            train_reps, val_reps, y_train, _bp,
                            augmentations_train=(train_aug if _use_aug
                                                 else None),
                            augmentations_val=(val_aug if _use_aug
                                               else None),
                            groups=train_groups)
                        _res = _score_slot(_trh, _vah, factory)
                    except Exception as e:
                        print(f"    fold {fold_idx} {model_name} {_slot}: "
                              f"build failed: {e}")
                        rule_missing.append({
                            'fold': fold_idx, 'model': model_name,
                            'slot': _slot,
                            'reason': f'build raised {type(e).__name__}: {e}'})
                if _res is None:
                    _res = (np.nan, np.nan, np.nan)
                _s = per_slot[_slot][model_name]
                _s['r2'].append(_res[0])
                _s['rmse'].append(_res[1])
                _s['spearman'].append(_res[2])

        # ----- train each downstream model on ITS OWN selected features -----
        #
        # ⛔ What used to be here, changed 2026-09-28: one recipe was read from
        # `fold_search.fitted_results['rf']`, one set of features was assembled
        # from it, and all five downstream models were trained on that set. The
        # search stores one recipe per downstream model it scored -- which
        # representations, how many features from each, how to rank them -- and
        # four of the five rows in the table were the random forest's recipe
        # scored under another model's name. So a row differed from the random
        # forest's row in one way only, the model, while the name said the
        # comparison was over both the model and its own selected features.
        #
        # Now each downstream model uses the recipe stored for it. A model with
        # no recipe of its own gets a missing held-out score on this fold, named
        # on the line printed below, and nothing is substituted for it.
        for model_name, factory in models.items():
            fitted = _fitted_all.get(model_name)
            if fitted is None:
                print(f"    fold {fold_idx} {model_name}: no recipe of its own "
                      f"-- the search stored recipes for "
                      f"{sorted(_fitted_all)} on this fold; no score")
                per_model_scores[model_name]['r2'].append(np.nan)
                per_model_scores[model_name]['rmse'].append(np.nan)
                per_model_scores[model_name]['spearman'].append(np.nan)
                continue

            try:
                fi = {r: {'selected_indices': idx}
                      for r, idx in fitted.selected_indices.items()}
                train_hybrid = apply_feature_selection(train_reps, fi)
                val_hybrid = apply_feature_selection(val_reps, fi)

                if fitted.augmentation_info and has_aug:
                    train_aug_block = apply_augmentation_selection(
                        train_aug, fitted.augmentation_info)
                    val_aug_block = apply_augmentation_selection(
                        val_aug, fitted.augmentation_info)
                    if (train_aug_block is not None
                            and val_aug_block is not None):
                        train_hybrid = np.hstack([train_hybrid,
                                                  train_aug_block])
                        val_hybrid = np.hstack([val_hybrid, val_aug_block])
            except Exception as e:
                print(f"    fold {fold_idx} {model_name}: hybrid assembly "
                      f"failed: {e}")
                per_model_scores[model_name]['r2'].append(np.nan)
                per_model_scores[model_name]['rmse'].append(np.nan)
                per_model_scores[model_name]['spearman'].append(np.nan)
                continue

            train_hybrid = np.asarray(train_hybrid, dtype=np.float64)
            val_hybrid = np.asarray(val_hybrid, dtype=np.float64)
            # One width per (fold, downstream model) now, because each model
            # selects its own features. The returned dim estimate is the mean
            # over all of those, where it used to be one width per fold.
            dims.append(train_hybrid.shape[1])

            # Scale unless binary
            if not _is_binary_rep(train_hybrid):
                scaler = StandardScaler()
                train_hybrid_s = scaler.fit_transform(train_hybrid)
                val_hybrid_s = scaler.transform(val_hybrid)
            else:
                train_hybrid_s, val_hybrid_s = train_hybrid, val_hybrid

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

    # The other two slots of the owner's set, in the same summary shape, plus
    # the recipe behind each slot on each fold and every slot left empty.
    if selection_out is not None:
        _by_slot = {}
        for _slot in ('general2', 'distance1'):
            _r = {}
            for model_name, scores in per_slot[_slot].items():
                if not scores['rmse'] or all(np.isnan(v)
                                             for v in scores['rmse']):
                    continue
                _r[model_name] = _summarize_scores(
                    scores['r2'], scores['rmse'], scores['spearman'])
            _by_slot[_slot] = _r
        selection_out['_rule_by_slot'] = _by_slot
        selection_out['rule_recipes'] = rule_recipes
        selection_out['rule_missing'] = rule_missing

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


def _paired_hybrid_delta(results, per_fold_key, lower_is_better,
                         metric_name=None, hybrid_key='kirby_hybrid'):
    """Per-model hybrid-minus-base-representation on the same held-out groups.

    `results` is {rep_name: {'models': {model: summary}}} where each summary
    carries fold-index-aligned held-out scores under `per_fold_key` — either
    'rmse_per_fold' or 'r2_per_fold'. Returns fold_reselection's output keyed by
    model: the hybrid is the only candidate, and the base representation is
    re-picked on every held-out group from the other groups.

    Empty when no summary carries per-fold scores — a result file written
    before 2026-09-04 has none, and there is nothing to pair.
    """
    hybrid = results.get(hybrid_key, {}).get('models', {})
    candidates_by_model = {}
    baselines_by_model = {}
    for model_name, summary in hybrid.items():
        folds = summary.get(per_fold_key)
        if not folds:
            continue
        candidates_by_model[model_name] = [
            {'name': hybrid_key, 'fold_scores': list(folds)}]

    for rep_name, rep_data in results.items():
        # Every slot of the owner's set ('kirby_hybrid',
        # 'kirby_hybrid__general2', 'kirby_hybrid__distance1') is a hybrid,
        # never a single representation.
        if rep_name.startswith('kirby_hybrid'):
            continue
        for model_name, summary in rep_data.get('models', {}).items():
            folds = summary.get(per_fold_key)
            if not folds:
                continue
            baselines_by_model.setdefault(model_name, {})[rep_name] = list(folds)

    if not candidates_by_model or not baselines_by_model:
        return {}
    # No exclude_folds: one candidate per model, the hybrid rebuilt by its own
    # search inside each fold. There is no shared recipe list for fold 0 to have
    # filtered, so no fold needs dropping from either arm.
    return fold_reselection(candidates_by_model, baselines_by_model,
                          lower_is_better=lower_is_better,
                          metric_name=metric_name)


def _paired_hybrid_deltas(results, dataset_name):
    """Both pairings for one split: RMSE, and R2 beside it.

    RMSE is the metric the registry reports these reaction datasets in, and it
    is defined on every held-out component. R2 is not: a component whose yields
    are near-constant has ss_tot == 0, evaluate_rep writes None for that
    component, and fold_reselection drops any component where either arm has no
    score. So the R2 pairing covers fewer held-out components than the RMSE
    table does, and both n_paired_folds are kept so the two coverages are
    visible side by side.

    Returns (registry_metric_name, paired_on_rmse, paired_on_r2).
    """
    registry_metric, _, _ = _resolve_dataset_metric(
        dataset_name, task_type='regression')
    # rmse is a BUILTIN_METRICS entry, so metric_name= is passed and the
    # direction is cross-checked there. 'r2' is not, so its pairing passes the
    # flag alone.
    paired_rmse = _paired_hybrid_delta(
        results, 'rmse_per_fold', True, metric_name='rmse')
    paired_r2 = _paired_hybrid_delta(results, 'r2_per_fold', False)
    return registry_metric, paired_rmse, paired_r2


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

    # How many inner folds each outer fold's recipe search runs on.
    # The find_best_hybrid call below passes no n_folds, so it asks for
    # DEFAULT_N_FOLDS folds, and _make_cv_splitter (find_best_hybrid.py:
    # 3842-3859) returns GroupKFold over however many distinct components the
    # training rows hold whenever that is the smaller number. The docstring at
    # the top of this file records 4 ligands and 3 bases, so on those two axes
    # the training rows hold 3 and 2 components and the recipe is picked on 3
    # or 2 inner folds, not 5. Nothing written to the split's JSON said which,
    # and the warning that says so is swallowed by the blanket
    # filterwarnings('ignore') at line 92 unless the 'always' filter at line 96
    # keeps it. Recorded 2026-09-18, split audit section D,
    # opig_kingsmith_component_splits.py:422. One count per outer fold, in fold
    # order; None where the training rows hold a single component, because
    # _make_cv_splitter raises there and no search runs. Bookkeeping only: this
    # fits no model and moves no reported score.
    from kirby.find_best_hybrid import DEFAULT_N_FOLDS
    inner_n_folds_per_outer_fold = []
    for _tr_idx, _ in fold_splits:
        _n_train_components = len(np.unique(groups[_tr_idx]))
        inner_n_folds_per_outer_fold.append(
            int(min(DEFAULT_N_FOLDS, _n_train_components))
            if _n_train_components >= 2 else None)

    # ===== Per-fold hybrid build + evaluate =====
    n = len(labels)
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n)

    print(f"  Building KIRBy hybrid per fold ({actual_folds} folds, train-only) ...")
    # Filled by fold 0 of the call below, and empty when that fold's search
    # failed or was not reached. Passed in rather than returned so the 3-tuple
    # this call has always unpacked is unchanged.
    fold0_selection = {}
    hybrid_results, hybrid_dim, top3 = _evaluate_hybrid_per_fold(
        dataset_name=dataset_name,
        split_key=split_key,
        base_reps=base_reps if base_reps else reps,
        aug_dict=aug_dict,
        labels=labels, groups=groups,
        fold_splits=fold_splits,
        models=models,
        pruning_mode=pruning_mode,
        selection_out=fold0_selection,
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

    # The 2nd general and the 1st distance hybrid of the owner's set
    # (2026-10-02), filed beside 'kirby_hybrid' with the slot in the name, and
    # each paired against the best single representation the same way.
    _rule_by_slot = fold0_selection.pop('_rule_by_slot', {})
    paired_by_slot = {}
    for _slot, _res in _rule_by_slot.items():
        if not _res:
            continue
        results[_slot_key(_slot)] = {'dim': None, 'models': _res,
                                     'slot': _slot}
        for model_name, res in _res.items():
            print(f"    {_slot_key(_slot):25s} {model_name:8s}  "
                  f"R2={res['r2']['mean']:.4f}  "
                  f"RMSE={res['rmse']['mean']:.2f}  ***")
        paired_by_slot[_slot] = {
            'rmse': _paired_hybrid_delta(results, 'rmse_per_fold', True,
                                         metric_name='rmse',
                                         hybrid_key=_slot_key(_slot)),
            'r2': _paired_hybrid_delta(results, 'r2_per_fold', False,
                                       hybrid_key=_slot_key(_slot)),
        }

    registry_metric, paired_rmse, paired_r2 = _paired_hybrid_deltas(
        results, dataset_name)

    # Additive, and NOT put inside `results`: that dict is keyed by
    # representation name and read as {rep: {'models': ...}} by
    # _paired_hybrid_delta (line 679), so a key that is not a representation
    # would be handed to .get('models', {}) there. These go on
    # the split's own dict, whose keys print_analysis reads by name.
    split_result = {
        'split_key': split_key,
        'n_groups': n_groups,
        'n_folds': actual_folds,
        # The outer fold count above is the component count. This is the INNER
        # count, one entry per outer fold: how many folds the recipe search for
        # that outer fold ran on, or None where no search could run. See where
        # it is computed, just after fold_splits is built.
        'inner_search_n_folds_per_fold': inner_n_folds_per_outer_fold,
        'inner_search_n_folds_requested': int(DEFAULT_N_FOLDS),
        'group_names': group_names,
        'results': results,
        # Per-model, per-held-out-group hybrid-minus-base-representation, in
        # RMSE — the metric the registry reports this dataset in, and the one
        # that is defined on every held-out component. The base representation
        # is re-picked on each held-out group from the other groups only, so it
        # is not the same-group maximum over 70 draws the Delta column used to
        # report. Positive means the hybrid's RMSE is the smaller one.
        'paired_hybrid_delta': paired_rmse,
        'paired_hybrid_delta_metric': 'rmse',
        'registry_metric': registry_metric,
        # The same subtraction in R2, kept beside it. Its n_paired_folds is the
        # number of held-out components where R2 was defined for both arms, and
        # can be smaller than the RMSE one.
        'paired_hybrid_delta_r2': paired_r2,
        # Owner's set (top 2 general + top 1 distance per model, 2026-10-02):
        # the same pairing for general2 and distance1. 'paired_hybrid_delta'
        # above is general1.
        'paired_hybrid_delta_by_slot': paired_by_slot,
    }
    # The 2 best base representations and the 2 best hybrid configurations per
    # downstream model, and the two hybrid deltas per downstream model, both
    # from fold 0's train-only search. They do not replace 'paired_hybrid_delta'
    # above: that one is held-out, over every component, and this one is fold 0's
    # own ranking of what its search stored.
    split_result.update(fold0_selection)
    return split_result


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
    # arithmetic fold_reselection already runs; both arms pay the same tax.
    for ds_name, ds_results in all_results.items():
        print(f"\n  Dataset: {ds_name} ({ds_results.get('n_reactions', '?')} reactions)")
        # Both deltas are hybrid minus best single representation on the SAME
        # held-out component. The RMSE columns are in the yield units the labels
        # are in, and positive means the hybrid's error is the smaller one. The
        # R2 column is the same subtraction in R2, positive means the hybrid
        # explains more. Each has its own count of held-out components, because
        # R2 is undefined on a component whose yields are near-constant and that
        # component drops out of the R2 pairing while the RMSE pairing keeps it.
        print(f"  {'Split key':15s} {'#Groups':>7s} {'Model':10s} "
              f"{'Hybrid RMSE':>12s} {'BestBase RMSE':>14s} "
              f"{'Delta RMSE':>11s} {'Comps':>6s} "
              f"{'Delta R2':>10s} {'Comps':>6s}")
        print(f"  {'-'*15} {'-'*7} {'-'*10} {'-'*12} {'-'*14} "
              f"{'-'*11} {'-'*6} {'-'*10} {'-'*6}")

        for split_key in ALL_SPLIT_KEYS:
            sr = ds_results.get('splits', {}).get(split_key)
            if sr is None or sr.get('skipped'):
                reason = sr.get('reason', 'not run') if sr else 'not run'
                print(f"  {split_key:15s} {'—':>7s} {'SKIPPED':10s} "
                      f"{'—':>12s} {'—':>14s} {'—':>11s} {'—':>6s} "
                      f"{'—':>10s} {'—':>6s}  ({reason})")
                continue

            results = sr.get('results', {})
            if not results:
                print(f"  {split_key:15s} {sr['n_groups']:>7d} {'NO RESULTS':10s}")
                continue

            registry_metric, paired, paired_r2 = _paired_hybrid_deltas(
                results, ds_name)
            if not paired:
                print(f"  {split_key:15s} {sr['n_groups']:>7d} "
                      f"{'NO PER-FOLD SCORES':10s}  (re-run: this result file "
                      f"predates rmse_per_fold)")
                continue
            if registry_metric != 'rmse':
                print(f"  [{ds_name}: the registry reports this dataset in "
                      f"{registry_metric}, but only rmse and r2 are stored per "
                      f"held-out component, so the pairing below is in rmse]")

            first = True
            for model_name in sorted(paired):
                r = paired[model_name]
                h, b = r['hybrid_reselected'], r['best_single_reselected']
                d = r['delta_mean_of_folds']
                h_s = f"{h:.4f}" if not np.isnan(h) else "---"
                b_s = f"{b:.4f}" if not np.isnan(b) else "---"
                d_s = f"{d:+.4f}" if not np.isnan(d) else "---"
                r2r = paired_r2.get(model_name)
                d2 = r2r['delta_mean_of_folds'] if r2r else float('nan')
                d2_s = f"{d2:+.4f}" if not np.isnan(d2) else "---"
                n2_s = f"{r2r['n_paired_folds']:>6d}" if r2r else f"{'—':>6s}"
                key_s = split_key if first else ''
                grp_s = f"{sr['n_groups']:>7d}" if first else f"{'':>7s}"
                first = False
                print(f"  {key_s:15s} {grp_s} {model_name:10s} "
                      f"{h_s:>12s} {b_s:>14s} {d_s:>11s} "
                      f"{r['n_paired_folds']:>6d} {d2_s:>10s} {n2_s}")

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

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard what an earlier attempt finished '
             '(<output>.partial.jsonl) and recompute every item.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()


    apply_distance_metric_arg(args)
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

    output_path = (args.output or
                   'tests/results/comparisons/opig_kingsmith_component_splits.json')

    # Each (dataset, split axis) result is written to <output>.partial.jsonl
    # as soon as it finishes and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). Before, a failure or a kill lost every
    # dataset, because the one JSON below was written after the last. When
    # every split axis of a dataset is already stored the dataset is not
    # loaded again. A split that raised or returned {'error'}, and a dataset
    # that failed to load, is not stored, is retried next time, and makes the
    # script exit 1 after the JSON is written.
    from experiments._checkpoint import (
        script_checkpoint, is_failed_item, exit_if_failed)
    _ck = script_checkpoint('opig_kingsmith_component_splits', output_path,
                            args, exclude=('datasets', 'splits', 'output'))
    _failed = []

    all_results = {}
    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        if split_keys and all(_ck.done(dataset=ds_name, split=sk)
                              for sk in split_keys):
            _saved = [_ck.get(dataset=ds_name, split=sk) for sk in split_keys]
            ds_result = {'n_reactions': _saved[0]['n_reactions'], 'splits': {}}
            for sk, sv in zip(split_keys, _saved):
                ds_result['splits'][sk] = sv['split']
            print(f"  all {len(split_keys)} split axes finished in an earlier "
                  f"attempt, read back")
            all_results[ds_name] = ds_result
            continue

        try:
            data = load_reaction_data(ds_name)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(ds_name)
            continue

        n = len(data['labels'])
        ds_result = {'n_reactions': n, 'splits': {}}

        for split_key in split_keys:
            try:
                if _ck.done(dataset=ds_name, split=split_key):
                    split_result = _ck.get(dataset=ds_name,
                                           split=split_key)['split']
                else:
                    split_result = evaluate_split(
                        ds_name, data, split_key, args.n_folds,
                        pruning_mode=args.pruning_mode)
                    if not is_failed_item(split_result):
                        _ck.put({'n_reactions': n, 'split': split_result},
                                dataset=ds_name, split=split_key)
                if is_failed_item(split_result):
                    _failed.append(f'{ds_name}/{split_key}')
                ds_result['splits'][split_key] = split_result
            except Exception as e:
                print(f"  FAILED split={split_key}: {e}")
                import traceback
                traceback.print_exc()
                _failed.append(f'{ds_name}/{split_key}')

        all_results[ds_name] = ds_result

    # Analysis
    if all_results:
        print_analysis(all_results)

    # Save
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")
    exit_if_failed(_failed, 'dataset or split')


if __name__ == '__main__':
    main()
