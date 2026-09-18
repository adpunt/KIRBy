#!/usr/bin/env python3
"""
Ginkgo Antibody Developability Experiment (OPIG B5)

Replicates van Niekerk et al. 2026 "2025 Ginkgo Datapoints Antibody
Developability Competition" — evaluates KIRBy reps across all 9 biophysical
properties using the competition's pre-computed hierarchical cluster splits.

Scientific questions:
  1. Which antibody reps best predict each developability property?
  2. Does hybrid augmentation help on these small (n=94-242) datasets?
  3. Across the 5 configured properties, how often does hybrid win?
  4. Which properties are predictable and which are noise?

Properties (GDPa1 v1.2, 242 antibodies):
  HIC (retention), Tm2 (thermal stability), Titer (expression),
  PR_CHO (protein recovery), AC-SINS (self-interaction),
  Purity, SEC %Monomer (aggregation), SMAC (colloidal), HAC (heparin)

Paper baselines (van Niekerk et al. 2026):
  Competition had 100+ team submissions. Best reported Spearman rho:
    HIC ~0.65-0.70, Tm2 ~0.30-0.45, Titer ~0.35-0.50,
    AC-SINS ~0.55-0.65, Heparin ~0.60-0.75 (varies by property)
  Pre-computed 5-fold hierarchical cluster splits (IgG-stratified).

Usage:
    # Smoke test (1 property, fast reps, ~2 min)
    python tests/experiments/opig_ginkgo_developability.py \\
        --properties ginkgo_hic --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation (all 9 properties)
    python tests/experiments/opig_ginkgo_developability.py

Reference:
    van Niekerk et al. (2026). mAbs.
    https://github.com/ginkgobioworks/abdev-benchmark
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
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from run_baseline_survey import evaluate_rep as _evaluate_rep
from run_baseline_survey import _filter_high_dim_per_fold
from experiments._shared import _resolve_dataset_metric
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS,
    extract_top3_base_reps, resolve_pruning_kwargs,
    FULL_BUDGETS, INTERNAL_EVALUATORS,
    fold_reselection, hr_direction, hr_exclude_folds,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# 6 of the competition's 9 properties. ginkgo_tm2, ginkgo_aggregation and
# ginkgo_purity were CUT on 2026-08-20 (DATASET_STATUS.md "Cut datasets"):
# near-dead targets whose best single representation reaches spearman +0.092,
# +0.179 and +0.151 with most representations scoring negative. That is the
# validity gate in the pruning procedure — no representation is meaningful, so
# the dataset goes. Their pruning runs are on disk if the call is ever revisited.
# 5 of the competition's 9 properties. The other 4 were cut, each with a reason
# on record:
#   ginkgo_tm2, ginkgo_aggregation, ginkgo_purity  cut 2026-08-20 as near-dead
#     targets — best single representation reaches spearman +0.092, +0.179 and
#     +0.151, with most representations scoring negative (DATASET_STATUS.md).
#   ginkgo_heparin  cut 2026-07-09 at n=94, too small
#     (DATASET_EXPANSION_2026_07_09.md, via antibody_baseline_survey.py:133).
ALL_PROPERTIES = [
    'ginkgo_hic', 'ginkgo_titer', 'ginkgo_pr_cho',
    'ginkgo_acsins', 'ginkgo_colloidal',
]

# Representations come from DATASET_REPS, resolved PER PROPERTY — every ginkgo
# property has its own approved pool and they differ, so one list cannot serve
# all of them. The hand-written list this replaced asked for blosum, onehot,
# esm2 and antiberty, which are in no ginkgo pool and which this experiment does
# nothing special with; it also skipped 4 to 5 of the 10 general reps per
# property, and it loaded only 'paratope' of the two approved augmentation reps,
# so AUG_REPS never found 'structural'. Both now come back with the pool.
#
# A property with no DATASET_REPS entry is skipped and recorded in '_skipped'
# rather than run against a hand-written list.
def _default_reps(prop_name):
    from experiments._shared import approved_rep_pool
    return approved_rep_pool(prop_name)

MODELS_LIST = [m for m in DOWNSTREAM_MODELS if m != 'gp']


def _smoke_search_kwargs():
    """Smoke-only hybrid-search shrink, gated on KIRBY_SMOKE_N.

    Caps the find_best_hybrid pool to one small budget + fast tier for local
    smoke runs (default is minutes of compute).
    No-op when KIRBY_SMOKE_N is unset.
    """
    if not os.environ.get('KIRBY_SMOKE_N', '').strip():
        return {}
    # speed is now accepted but ignored (the former tier dict was removed; defaults
    # live in the DEFAULT_* module constants). Smoke shrink is via tiny budgets
    # only. internal_evaluators=['rf'] cuts the search ~5x (DEFAULT_INTERNAL_EVALUATORS
    # is match/rf/svm/lightgbm/mlp); rf alone still exercises the hybrid pipeline.
    # Also drop importance to lightgbm_gain (cheap): DEFAULT_IMPORTANCE_METHODS runs
    # deeplift+treeshap, which compute SHAP over the (here 5600-dim) features
    # per fold and dominate the smoke wall time. lightgbm_gain still produces a
    # real per-rep importance ranking so the hybrid pipeline is exercised.
    return {'speed': 'full', 'budgets': [25], 'internal_evaluators': ['rf'],
            'importance_methods': ['lightgbm_gain']}

# Competition baselines (van Niekerk et al. 2026, 113 teams)
# test_top = best team Spearman rho on held-out test set (80 antibodies)
# cv_top = best team cross-validation Spearman rho (246 training antibodies)
# Note: competition only evaluated 5 of 9 properties
PAPER_BASELINES = {
    'ginkgo_hic':         {'test_top': 0.708, 'cv_top': 0.758},
    'ginkgo_tm2':         {'test_top': 0.392, 'cv_top': 0.140},
    'ginkgo_titer':       {'test_top': 0.310, 'cv_top': 0.640},
    'ginkgo_pr_cho':      {'test_top': 0.337, 'cv_top': 0.559},
    'ginkgo_acsins':      {'test_top': 0.356, 'cv_top': 0.653},
    # These 4 were NOT in the competition — no published baselines
    'ginkgo_purity':      {},
    'ginkgo_aggregation': {},
    'ginkgo_colloidal':   {},
    'ginkgo_heparin':     {},
}

# =============================================================================
# EVALUATION
# =============================================================================

def _fold_assignments_to_splits(fold_assignments, n_folds):
    """Convert Ginkgo's pre-computed fold IDs to (train_idx, val_idx) list."""
    unique_folds = sorted(set(fold_assignments))
    actual_folds = min(n_folds, len(unique_folds))
    if actual_folds < n_folds:
        print(f"  WARNING: requested {n_folds} folds but only "
              f"{len(unique_folds)} pre-computed folds available, "
              f"using {actual_folds}")
    splits = []
    fa = np.asarray(fold_assignments)
    for fold_id in unique_folds[:actual_folds]:
        val_idx = np.where(fa == fold_id)[0]
        train_idx = np.where(fa != fold_id)[0]
        splits.append((train_idx, val_idx))
    return splits


def _recall_at_k(y_true, y_pred, frac=0.1):
    """Top-K% recall: overlap between true top-K and predicted top-K.

    Matches ginkgobioworks/abdev-benchmark evaluation_metrics.recall_at_k.
    """
    top_k = max(1, int(len(y_true) * frac))
    true_top = set(np.argsort(y_true)[-top_k:].tolist())
    pred_top = set(np.argsort(y_pred)[-top_k:].tolist())
    return len(true_top & pred_top) / top_k


# Properties where LOWER values are better (sign-flip for top-K recall).
# From abdev-benchmark constants.py ASSAY_HIGHER_IS_BETTER:
#   HIC=False, PR_CHO=False, AC-SINS=False (i.e. lower is better for these)
LOWER_IS_BETTER = {'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_acsins'}


def evaluate_rep_reg(X, labels, groups, prebuilt_splits, model,
                     prop_name='', return_fold_scores=False):
    """Evaluate one (rep, model) for regression on fixed (train, val) splits.

    Scores on the SAME ``prebuilt_splits`` used for the per-fold hybrid so the
    baseline and hybrid arms share identical held-out folds (no leakage).
    Delegates to run_baseline_survey.evaluate_rep for CV, scaling, and metrics
    AND fold-level predictions (single pass). Computes Top-10% recall from
    those predictions without retraining.

    Returns dict with r2, spearman, pearson, rmse, top10_recall as
    {metric: {mean, std}}. When ``return_fold_scores`` is True, returns
    ``(result, fold_spearman)`` where ``fold_spearman`` is the per-fold primary
    metric (Spearman rho) in split order — the per-fold baseline scores the
    fold_reselection helper re-selects over. Higher rho is always better, so the
    honest block's ``lower_is_better`` is False regardless of property polarity.
    """
    mean_m, std_m, fold_preds = _evaluate_rep(
        X, labels, groups, model, len(prebuilt_splits), 'regression',
        prebuilt_splits=prebuilt_splits, also_return_predictions=True)
    if not mean_m:
        return (None, []) if return_fold_scores else None
    result = {k: {'mean': mean_m[k], 'std': std_m.get(k, 0.0)} for k in mean_m}

    # Top-10% recall per fold (benchmark's secondary metric) AND per-fold
    # Spearman rho — both computed from the predictions already returned by
    # _evaluate_rep (no retraining), in split order.
    flip = prop_name in LOWER_IS_BETTER
    recall_scores = []
    fold_spearman = []
    for _val_idx, y_true, y_pred in fold_preds:
        y_t = -y_true if flip else y_true
        y_p = -y_pred if flip else y_pred
        recall_scores.append(_recall_at_k(y_t, y_p, frac=0.1))
        if len(y_true) > 2:
            rho, _ = spearmanr(y_true, y_pred)
            fold_spearman.append(float(rho) if not np.isnan(rho) else float('nan'))
        else:
            fold_spearman.append(float('nan'))
    if recall_scores:
        result['top10_recall'] = {
            'mean': float(np.mean(recall_scores)),
            'std': float(np.std(recall_scores)),
        }

    if return_fold_scores:
        return result, fold_spearman
    return result


def _is_binary_hybrid(X):
    """True if X is a 0/1 (or uint8) matrix — skip StandardScaler if so."""
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def _summarize_hybrid_scores(r2_scores, rmse_scores, spearman_scores):
    """Aggregate per-fold hybrid metrics into the legacy {mean, std} shape.

    Mirrors opig_kingsmith_component_splits._summarize_scores so the legacy
    results['kirby_hybrid'] entry keeps the same structure as the base reps.
    NaN-tolerant (folds with too-few val points contribute no Spearman).
    """
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
    }


def _top2_selection_from_search(search, model_names, context):
    """Top-2 rows and the two hybrid deltas off one fold's search. 0 new fits.

    Additive, added 2026-09-18. The baseline arm below still scores every
    representation in `reps` against every model in MODELS_LIST, and the
    per-fold hybrid still runs on every fold. Nothing here feeds
    results[...], print_analysis or fold_reselection.

    All four selectors read rows the search already stored, so this costs 0
    new model fits.

    What was absent before it: extract_top3_base_reps names 3 base
    representations by one downstream model, random forest, on fold 0, and the
    line is printed and dropped. No ranking was recorded for the other 4
    downstream models, and none at all for the hybrid configurations the search
    scored.

    `search` is fold 0's search, built on fold 0's TRAIN rows only, so every
    number here is a training-fold number and not a held-out one. The held-out
    hybrid-versus-best-single comparison stays where it is, in
    reselected_comparison.

    require_full=False everywhere. A downstream model this search never scored
    comes back as a short list carrying its own shortfall_reason, instead of
    raising and taking the fold loop with it. Every call is wrapped: this file
    ran before these keys existed and has to keep running if they fail.

    Returns None when nothing could be computed.
    """
    block = {}
    try:
        # 'rf' is the downstream model this file pulls its FittedResult from
        # (fold_search.fitted_results.get('rf')) and the model
        # extract_top3_base_reps ranks by, so the two are the same quantity at
        # two depths.
        block['top2_base_reps'] = top_base_reps(
            search, 'rf', 2, require_full=False, context=context)
    except Exception as _e:
        print(f"    top2_base_reps not added: {type(_e).__name__}: {_e}")
    try:
        # budget=None: the best 2 configurations over every budget the search
        # ran, not within one budget in features. The find_best_hybrid call
        # above passes the whole budget list, so cutting per budget here would
        # name a pair this experiment never used.
        block['top2_hybrids'] = top_hybrids(
            search, 'rf', 2, require_full=False, context=context)
    except Exception as _e:
        print(f"    top2_hybrids not added: {type(_e).__name__}: {_e}")
    try:
        # models=list(model_names) is the models dict this file actually built
        # factories for, which get_model_factories can shorten at small sample
        # counts - these properties are 94 to 242 antibodies. Passing it rather
        # than the default keeps the models axis at the width this file scores
        # at. MODELS_LIST is 5 downstream model names (DOWNSTREAM_MODELS holds
        # rf, xgboost, svm, dnn, lightgbm and none of them is the 'gp' the
        # comprehension drops), so 5 * (2 representations + 2 hybrids) = 20
        # rows when every model survives get_model_factories.
        block['top2_pairs_by_model'] = select_model_rep_pairs(
            search, models=list(model_names), n_base=2, n_hybrids=2,
            require_full=False, context=context)
    except Exception as _e:
        print(f"    top2_pairs_by_model not added: {type(_e).__name__}: {_e}")

    # Two reference numbers for the rank-1 hybrid per downstream model, not
    # one:
    #   delta_vs_source_mean - against the MEAN of the base representations the
    #     budget split actually gave features to, read off
    #     Blueprint.allocation_per_fold, so a representation given 0 features
    #     on every fold is out of that mean.
    #   delta_vs_best_rep - against the single best base representation for
    #     that same downstream model.
    # Positive means the hybrid is better, on either metric direction.
    #
    # hybrid= is passed rather than left to default. hybrid_deltas with
    # hybrid=None reads _picked['selected'][0]['row'] at _shared.py:7432, and
    # the entry top_hybrids builds (_shared.py:7043-7066) carries no 'row' key
    # - both read in this session. The rank-1 entry passed in below carries
    # 'score_mean' and 'blueprint', the only two keys hybrid_deltas reads off
    # it.
    #
    # One try per downstream model, so a model with no stored hybrid
    # configuration does not cost the others their entry. The keys are
    # downstream model names; this dict is new and nothing iterates it.
    deltas = {}
    for m in model_names:
        try:
            top1 = top_hybrids(search, m, 1, require_full=False,
                               context=f'{context}/deltas/{m}')
            if not top1['selected']:
                print(f"    hybrid_deltas for downstream model {m} not added: "
                      f"{top1['shortfall_reason']}")
                continue
            deltas[m] = hybrid_deltas(search, m, hybrid=top1['selected'][0],
                                      context=f'{context}/deltas/{m}')
        except Exception as _e:
            print(f"    hybrid_deltas for downstream model {m} not added: "
                  f"{type(_e).__name__}: {_e}")
    if deltas:
        block['hybrid_deltas_by_model'] = deltas
        # delta_vs_best_rep is None when no base representation carries a score
        # for that downstream model, so it is formatted only after the None
        # test - a format spec on None raises TypeError, and this print sits
        # outside the try above.
        for m, d in sorted(deltas.items()):
            dsm = d['delta_vs_source_mean']
            dbr = d['delta_vs_best_rep']
            print(f"    deltas x {m:8s}  hybrid={d['hybrid_score']:.4f}"
                  f"  minus the mean of the {d['n_source_reps']} "
                  f"representations that fed it: "
                  f"{'-' if dsm is None else format(dsm, '+.4f')}"
                  f"  minus the best representation ({d['best_rep']}): "
                  f"{'-' if dbr is None else format(dbr, '+.4f')}")
    return block or None


def _evaluate_hybrid_per_fold(prop_name, base_reps, aug_dict, labels, groups,
                              fold_splits, models, pruning_mode='default_pruning'):
    """Build the KIRBy hybrid PER FOLD (train-only) and score the held-out fold.

    Mirrors opig_kingsmith_component_splits._evaluate_hybrid_per_fold. For each
    fold: run find_best_hybrid on the TRAIN indices only (no peeking at val),
    pull the 'rf' FittedResult (selected_indices + augmentation_info), apply the
    SAME selection to train and val reps, train each downstream model on the
    train hybrid and score the held-out fold. This replaces the old full-data
    hybrid build (which leaked selection into every fold's score).

    Returns (rep_results, dim_estimate, top3_first_fold, fold_spearman_by_model,
    top2_first_fold) where fold_spearman_by_model = {model: [per_fold_held_out_rho,
    ...]} feeds the fold_reselection candidate arm. Returns
    (None, 0, None, {}, top2_first_fold) if every fold failed.

    top2_first_fold was added 2026-09-18 beside top3_first_fold, off the same
    fold-0 search and costing 0 new model fits. It is a 5th return value rather
    than an entry in rep_results: rep_results is keyed by downstream model name
    and evaluate_property writes it straight into results['kirby_hybrid']
    ['models'], which print_analysis reads with max() over every value's
    ['spearman']['mean'].
    """
    from kirby.find_best_hybrid import find_best_hybrid
    from kirby.hybrid import (apply_feature_selection,
                              apply_augmentation_selection)

    per_model = {m: {'r2': [], 'rmse': [], 'spearman': []} for m in models}
    dims = []
    top3_first_fold = None
    # Filled from fold 0's search, the same search top3_first_fold reads.
    # Stays None when fold 0's find_best_hybrid raised.
    top2_first_fold = None
    src_reps = base_reps if base_reps else {}
    has_aug = bool(aug_dict)

    # Smoke shrink (KIRBY_SMOKE_N) reuses the file's helper; otherwise fall back
    # to the FULL budgets/evaluators the server run uses.
    _smoke = _smoke_search_kwargs()
    _budgets = _smoke.get('budgets', FULL_BUDGETS)
    _evals = _smoke.get('internal_evaluators', INTERNAL_EVALUATORS)
    _smoke_imp = ({'importance_methods': _smoke['importance_methods']}
                  if 'importance_methods' in _smoke else {})

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_reps = {r: arr[train_idx] for r, arr in src_reps.items()}
        train_aug = ({r: arr[train_idx] for r, arr in aug_dict.items()}
                     if has_aug else None)
        train_groups = groups[train_idx]
        train_labels = labels[train_idx]

        # ----- build hybrid on train only -----
        try:
            _pkw = resolve_pruning_kwargs(
                pruning_mode, len(train_labels),
                dataset_label=f'{prop_name}/fold{fold_idx}')
            # Per-fold train subsets do NOT align with the full-data precompute
            # cache, so use a distinct 'general'-tier namespace per fold
            # (condition_suffix), matching the kingsmith template.
            fold_search = find_best_hybrid(
                train_reps, train_labels,
                groups=train_groups,
                augmentations=train_aug,
                augmentation_strategy=('uncapped' if has_aug else 'none'),
                budgets=_budgets,
                # metric= is REQUIRED since 2026-09-10. Without it this call
                # raised on every fold and the raise was swallowed by the
                # `except Exception` below, so the run finished and wrote a
                # results file with no hybrid rows in it. Resolved from the
                # registry here, on the experiment side -- src/ does not look a
                # dataset up. (ginkgo_hic is spearman, not rmse, so a hardcoded
                # metric would be wrong for these.)
                metric=_resolve_dataset_metric(prop_name)[0],
                task_type='regression',
                internal_evaluators=_evals,
                return_fitted=True,
                cache=True,
                dataset_name=prop_name,
                tier='general',
                condition_suffix=f'ginkgo_hold_fold{fold_idx}',
                adaptive_pruning=0.25,
                **_smoke_imp,
                **_pkw,
            )
        except Exception as e:
            print(f"    fold {fold_idx}: find_best_hybrid failed: {e}")
            continue

        if fold_idx == 0 and fold_search is not None:
            top3_first_fold = extract_top3_base_reps(fold_search)
            top2_first_fold = _top2_selection_from_search(
                fold_search, list(models), f'{prop_name}/fold{fold_idx}')

        fitted = (fold_search.fitted_results.get('rf')
                  if fold_search is not None else None)
        if fitted is None:
            print(f"    fold {fold_idx}: no fitted 'rf' result; skipping")
            continue

        # ----- assemble train/val hybrid using this fold's selection -----
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
            continue

        dims.append(train_hybrid.shape[1])
        train_hybrid = np.asarray(train_hybrid, dtype=np.float64)
        val_hybrid = np.asarray(val_hybrid, dtype=np.float64)
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Same feature cut the baseline arm gets. run_baseline_survey.evaluate_rep
        # runs _filter_high_dim_per_fold on every baseline representation, so any
        # representation wider than 500 features is scored on 200 of them there
        # (a 1280-feature language-model representation loses 1080). Without this
        # the hybrid keeps every feature it selected, and part of the reported
        # difference between the two arms is that preprocessing difference rather
        # than a representation difference. Fit on this fold's training rows only,
        # before scaling, in the same order evaluate_rep uses. `dims` above keeps
        # the width the hybrid SELECTED, matching the baseline rows, which record
        # each representation's own width and not its post-cut width.
        train_hybrid, val_hybrid, _ = _filter_high_dim_per_fold(
            train_hybrid, val_hybrid, y_train, 'regression')

        if not _is_binary_hybrid(train_hybrid):
            scaler = StandardScaler()
            train_hybrid_s = scaler.fit_transform(train_hybrid)
            val_hybrid_s = scaler.transform(val_hybrid)
        else:
            train_hybrid_s, val_hybrid_s = train_hybrid, val_hybrid

        # ----- train each downstream model, score held-out fold -----
        for model_name, factory in models.items():
            try:
                m = clone(factory())
                m.fit(train_hybrid_s, y_train)
                preds = m.predict(val_hybrid_s)
            except Exception:
                continue
            ss_res = np.sum((y_val - preds) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            per_model[model_name]['r2'].append(
                1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
            per_model[model_name]['rmse'].append(
                float(np.sqrt(np.mean((y_val - preds) ** 2))))
            if len(y_val) > 2:
                rho, _ = spearmanr(y_val, preds)
                per_model[model_name]['spearman'].append(
                    float(rho) if not np.isnan(rho) else np.nan)
            else:
                per_model[model_name]['spearman'].append(np.nan)

    if not dims:
        return None, 0, top3_first_fold, {}, top2_first_fold

    rep_results = {}
    fold_spearman_by_model = {}
    for model_name, s in per_model.items():
        if not s['rmse']:
            continue
        rep_results[model_name] = _summarize_hybrid_scores(
            s['r2'], s['rmse'], s['spearman'])
        fold_spearman_by_model[model_name] = s['spearman']

    return (rep_results, int(np.mean(dims)), top3_first_fold,
            fold_spearman_by_model, top2_first_fold)


def evaluate_property(prop_name, rep_names, n_folds, pruning_mode='default_pruning'):
    """Evaluate all reps × models on one Ginkgo property.

    Returns dict with per-rep results.
    """
    print(f"\n  --- {prop_name} ---")

    data = load_dataset(prop_name)
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    groups = data.get('groups')
    fold_assignments = data.get('fold_assignments')
    n = len(labels)
    # _smoke_subsample subsets 'groups' but not 'fold_assignments' (separate
    # per-row key). For Ginkgo the two are identical (groups == fold IDs), so
    # under KIRBY_SMOKE_N realign fold_assignments to the subsampled groups
    # when lengths disagree, otherwise the fold-index splits run out of bounds.
    if fold_assignments is not None and len(fold_assignments) != n:
        fold_assignments = groups
    print(f"  {n} samples, target={data.get('property_name', prop_name)}")

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag=prop_name,
    )

    if not reps:
        print(f"  No reps available, skipping")
        return None

    # Group label for CV. Prefer the hierarchical fold assignments so the
    # internal evaluator's CV does not degenerate to LeaveOneOut (P1-3).
    if groups is not None:
        grp = np.asarray(groups)
    elif fold_assignments is not None:
        grp = np.asarray(fold_assignments)
    else:
        grp = np.arange(n)

    # Augmentation config matches ginkgo_* in DATASET_REPS:
    # augmentation = ['paratope', 'structural']
    AUG_REPS = {'paratope', 'structural'}
    base = {k: v for k, v in reps.items() if k not in AUG_REPS}
    aug_dict = {k: v for k, v in reps.items() if k in AUG_REPS}
    if not base:
        print("  WARNING: no base reps after stripping augmentations; "
              "using all reps as base.")
        base_for_search = reps
    else:
        base_for_search = base

    # Build ONE set of (train, val) splits shared by BOTH arms so the hybrid
    # (built train-only per fold) and every baseline rep are scored on the
    # identical held-out folds — no selection leakage, no fold drift.
    if fold_assignments is not None:
        fold_splits = _fold_assignments_to_splits(fold_assignments, n_folds)
    else:
        from sklearn.model_selection import GroupKFold
        actual = max(2, min(n_folds, len(set(grp.tolist()))))
        sample_X = next(iter(reps.values()))
        fold_splits = list(GroupKFold(n_splits=actual).split(
            sample_X, labels, groups=grp))

    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n)

    # ===== Hybrid arm: build PER FOLD (train-only), score held-out =====
    # Replaces the old full-data hybrid build (which selected features on all
    # rows, then scored the same rows — selection leakage in every fold).
    hybrid_results, hybrid_dim, top3, hybrid_fold_scores, top2_block = (
        _evaluate_hybrid_per_fold(
            prop_name, base_for_search, aug_dict, labels, grp,
            fold_splits, models, pruning_mode=pruning_mode))
    if top3:
        print(f"  Top-3 base reps (fold 0, by RF): "
              f"{', '.join(f'{nm}={sc:.4f}' for nm, sc in top3)}")

    # ===== Baseline arm: each rep × model on the SAME held-out folds =====
    results = {}
    baselines_by_model = {}  # {model: {rep: [per_fold_rho, ...]}}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            res, fold_scores = evaluate_rep_reg(
                X, labels, grp, fold_splits, factory(),
                prop_name=prop_name, return_fold_scores=True)
            if res is not None:
                rep_results[model_name] = res
                if fold_scores:
                    baselines_by_model.setdefault(model_name, {})[rep_name] = \
                        fold_scores

        if rep_results:
            # Find best model by Spearman
            best_model = max(rep_results.items(),
                             key=lambda kv: kv[1]['spearman']['mean'])
            best_rho = best_model[1]['spearman']['mean']
            best_r2 = best_model[1]['r2']['mean']
            print(f"    {rep_name:25s} best={best_model[0]:6s}  "
                  f"rho={best_rho:.4f}  R2={best_r2:.4f}")
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    # Legacy kirby_hybrid entry — now the honest per-fold aggregate (same shape
    # as the base reps: {'dim', 'models': {model: {metric: {mean, std}}}}).
    if hybrid_results:
        results['kirby_hybrid'] = {'dim': hybrid_dim, 'models': hybrid_results}
        for model_name, res in hybrid_results.items():
            print(f"    {'kirby_hybrid':25s} {model_name:8s}  "
                  f"rho={res['spearman']['mean']:.4f}  "
                  f"R2={res['r2']['mean']:.4f}  ***")

    # ===== fold_reselection block (CONTRACT shape) =====
    # One candidate per model (the per-fold-rebuilt hybrid); fold_reselection's
    # leave-one-fold-out then averages its held-out scores for the hybrid arm
    # and re-selects the best baseline rep per held-out fold for the base arm.
    # Spearman rho is higher-is-better regardless of property polarity, so
    # lower_is_better is False (the LOWER_IS_BETTER set only flips top-K recall).
    candidates_by_model = {}
    for model_name, fs in (hybrid_fold_scores or {}).items():
        if fs:
            candidates_by_model[model_name] = [
                {'name': 'kirby_hybrid', 'fold_scores': list(fs)}]

    reselection_block = {
        # Both arms' fold scores are Spearman rho (evaluate_rep_reg's
        # fold_spearman and _evaluate_hybrid_per_fold's per-fold rho), whatever
        # metric the search itself ran under. Naming it lets hr_direction check
        # the direction against BUILTIN_METRICS instead of trusting the flag.
        'metric_name': 'spearman',
        'lower_is_better': False,
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': candidates_by_model,
        'baselines_by_model': baselines_by_model,
    }

    # The COMPUTED held-out comparison, not just its inputs. Before this the
    # block was stored and never scored, so the only hybrid-versus-baseline
    # number this file produced was print_analysis's Delta, which takes a
    # maximum over five models on the hybrid side against a maximum over ten
    # representations and five models on the baseline side. Pure arithmetic on
    # per-fold scores already computed — no model fits.
    reselected_comparison = fold_reselection(
        candidates_by_model,
        baselines_by_model,
        hr_direction(reselection_block),
        metric_name=reselection_block.get('metric_name'),
        exclude_folds=hr_exclude_folds(reselection_block),
    )

    return {
        'n_samples': n,
        'property': data.get('property_name', prop_name),
        'results': results,
        # Per-property fold_reselection block (per-fold scores for BOTH arms).
        # A reader feeds this straight to experiments._shared.fold_reselection.
        'fold_reselection': reselection_block,
        'reselected_comparison': reselected_comparison,
        # Beside 'results', never inside it: print_analysis loops
        # results.items() and reads rep_data['models'] for every value, so an
        # entry there with no 'models' key raises KeyError and takes the whole
        # cross-property table with it. print_analysis and
        # print_reselection_analysis read this dict by name only -
        # 'n_samples', 'results', 'fold_reselection',
        # 'reselected_comparison' - so this key is printed by nothing and
        # saved to the JSON. None when fold 0's search did not run.
        'top2_selection': top2_block,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print cross-property summary table."""
    print(f"\n{'=' * 100}")
    print(f"GINKGO DEVELOPABILITY CROSS-PROPERTY SUMMARY (OPIG B5)")
    print(f"{'=' * 100}")

    # Header
    print(f"\n  {'Property':20s} {'N':>5s} {'Comp.test':>10s} {'Comp.CV':>10s} "
          f"{'Best indiv':>25s} {'rho':>8s} {'Hybrid rho':>10s} {'Delta':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*10} {'-'*25} {'-'*8} {'-'*10} {'-'*8}")

    hybrid_wins = 0
    total = 0

    for prop_name in ALL_PROPERTIES:
        if prop_name not in all_results:
            continue

        ds = all_results[prop_name]
        n = ds['n_samples']
        results = ds['results']

        # Best individual rep (by Spearman)
        best_indiv_name = None
        best_indiv_rho = float('-inf')
        hybrid_rho = float('nan')

        for rep_name, rep_data in results.items():
            best_model_res = max(rep_data['models'].values(),
                                  key=lambda v: v['spearman']['mean'])
            rho = best_model_res['spearman']['mean']
            if rep_name == 'kirby_hybrid':
                hybrid_rho = rho
            elif rho > best_indiv_rho:
                best_indiv_rho = rho
                best_indiv_name = rep_name

        delta = hybrid_rho - best_indiv_rho if not np.isnan(hybrid_rho) else float('nan')
        if not np.isnan(delta) and delta > 0:
            hybrid_wins += 1
        total += 1

        baselines = PAPER_BASELINES.get(prop_name, {})
        test_s = f"{baselines['test_top']:.3f}" if 'test_top' in baselines else "—"
        cv_s = f"{baselines['cv_top']:.3f}" if 'cv_top' in baselines else "—"
        hybrid_s = f"{hybrid_rho:.4f}" if not np.isnan(hybrid_rho) else "—"
        delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "—"
        indiv_label = f"{best_indiv_name}" if best_indiv_name else "—"

        print(f"  {prop_name:20s} {n:5d} {test_s:>10s} {cv_s:>10s} "
              f"{indiv_label:>25s} {best_indiv_rho:8.4f} {hybrid_s:>10s} {delta_s:>8s}")

    print(f"\n  Hybrid wins: {hybrid_wins}/{total} properties")


def print_reselection_analysis(all_results):
    """Per-model, held-out hybrid-versus-best-single table, one block per property.

    The Delta in print_analysis above takes a maximum over five models on the
    hybrid side and a maximum over ten representations and five models on the
    baseline side, both read off the same folds. Here each downstream model is
    its own row: on every held-out fold the best single representation is
    re-picked from the OTHER folds' mean and the hybrid's score on that fold is
    read, and the two are subtracted fold by fold. 'Delta' is
    delta_mean_of_folds, the mean of those per-fold differences, and 'Folds' is
    how many folds both arms scored. Pure arithmetic on per-fold Spearman rho
    already computed — no model fits.
    """
    print(f"\n{'=' * 100}")
    print(f"HONEST PER-MODEL HELD-OUT COMPARISON (Spearman rho; higher better)")
    print(f"{'=' * 100}")
    for prop_name in ALL_PROPERTIES:
        ds = all_results.get(prop_name)
        if not ds:
            continue
        res = ds.get('reselected_comparison')
        if res is None:
            block = ds.get('fold_reselection')
            if not block:
                print(f"\n  {prop_name}: no fold_reselection block")
                continue
            res = fold_reselection(
                block.get('candidates_by_model', {}),
                block.get('baselines_by_model', {}),
                hr_direction(block),
                metric_name=block.get('metric_name'),
                exclude_folds=hr_exclude_folds(block),
            )
        if not res:
            print(f"\n  {prop_name}: fold_reselection produced no models")
            continue
        print(f"\n  {prop_name}:")
        print(f"    {'Model':>8s} {'Hybrid':>10s} {'BestSingle':>11s} "
              f"{'Delta':>9s} {'Folds':>6s}")
        for model_name in sorted(res):
            r = res[model_name]
            h = r['hybrid_reselected']
            b = r['best_single_reselected']
            d = r['delta_mean_of_folds']
            hs = f"{h:.4f}" if not np.isnan(h) else "—"
            bs = f"{b:.4f}" if not np.isnan(b) else "—"
            ds_s = f"{d:+.4f}" if not np.isnan(d) else "—"
            flag = ' (single-fold: optimistic, NOT honest)' if r.get('single_fold') else ''
            print(f"    {model_name:>8s} {hs:>10s} {bs:>11s} {ds_s:>9s} "
                  f"{r.get('n_paired_folds', 0):>6d}{flag}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ginkgo developability experiment (OPIG B5). '
                    'Evaluates all 9 properties with KIRBy antibody reps.')
    parser.add_argument(
        '--properties', type=str, default=None,
        help='Comma-separated property names (default: all 9)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 10)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of CV folds (default: 5, uses pre-computed splits)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_ginkgo_developability.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()


    apply_distance_metric_arg(args)
    properties = (args.properties.split(',') if args.properties
                  else ALL_PROPERTIES)
    rep_override = (args.reps.split(',') if args.reps else None)

    print("=" * 70)
    print("GINKGO DEVELOPABILITY EXPERIMENT (OPIG B5)")
    print(f"  Paper: van Niekerk et al. 2026, mAbs")
    print(f"  Properties: {len(properties)}")
    print(f"  Reps: {rep_override if rep_override else 'per-property approved pool'}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}
    skipped = {}
    for prop_name in properties:
        if prop_name not in DATASET_REGISTRY:
            print(f"WARNING: Unknown property '{prop_name}', skipping")
            continue
        try:
            from experiments._shared import NoApprovedRepPool
            try:
                rep_names = (rep_override if rep_override
                             else _default_reps(prop_name))
            except NoApprovedRepPool as e:
                print(f"\n  SKIPPED {prop_name}: {e}")
                skipped[prop_name] = str(e)
                continue
            prop_results = evaluate_property(prop_name, rep_names, args.n_folds,
                                              pruning_mode=args.pruning_mode)
            if prop_results is not None:
                all_results[prop_name] = prop_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Cross-property analysis
    if all_results:
        print_analysis(all_results)
        # Per-model held-out table — the leak-free counterpart to the
        # max-over-models Delta in the table above.
        print_reselection_analysis(all_results)
    if skipped:
        print(f"\n  {len(skipped)} of {len(properties)} properties skipped "
              f"for want of an approved rep pool: {sorted(skipped)}")

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_ginkgo_developability.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if skipped:
        all_results['_skipped'] = skipped

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
