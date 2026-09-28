#!/usr/bin/env python3
"""
Humanness Classification Experiment (OPIG B3)

Extends Prihoda et al. 2022 "BioPhi" (mAbs) — a score discrimination study
comparing 8 humanness scoring methods on IMGT/mAb-DB therapeutic antibodies.
BioPhi does NOT train classifiers; it computes ROC AUC via threshold sweep
on the full dataset (no train/test split, no CV). KIRBy instead trains
supervised classifiers on high-dimensional reps with grouped, stratified CV.
The classifiers are whatever DOWNSTREAM_MODELS holds when this file is
imported (rf, xgboost, svm, dnn, lightgbm as of 2026-09-28, and a shorter
list under a local grid). The names 'Ridge, RF' stood here until 2026-09-28
and were never what the file ran.

NOTE: Previous versions cited Marks et al. 2021 (Hu-mAb) as primary reference.
Marks 2021 is one of the 8 methods BioPhi evaluates, not the data source.
Data source = BioPhi's IMGT/mAb-DB curation (Prihoda et al. 2022).

Scientific questions:
  1. Which antibody rep is best for humanness classification?
  2. Does hybrid augmentation improve over the best individual rep?
  3. How do sequence-only LM reps compare to classical encodings (BLOSUM, onehot)?

Datasets:
  - humanness_therapeutic: human vs non-human real therapeutics (BioPhi IMGT,
    about 565 rows). The label is 1 when the IMGT species field reads Human and
    0 for Humanized, Chimeric, Mouse and the rest, so one class holds three
    kinds of antibody.
  - A species preset exists in src/kirby/datasets/humanness.py but is not a key
    of DATASET_REGISTRY, so this file cannot run it.

Paper baselines (Prihoda et al. 2022, BioPhi):
  - 8 methods scored on IMGT/mAb-DB therapeutics (VH+VL averaged):
    Hu-mAb:    AUROC=97.7%  Accuracy=93.3%  (best overall)
    OASis:     AUROC=97.2%  Accuracy=94.4%  (relaxed threshold, their method)
    Germline:  AUROC=96.8%  Accuracy=92.8%
  - No train/test split — full-dataset threshold sweep, NOT comparable to CV.
  - KIRBy uses VH-only; BioPhi averages VH+VL scores.

Usage:
    # Three representations only, for a quick check
    python tests/experiments/opig_humanness.py --datasets humanness_therapeutic \
        --n-folds 3 --reps blosum,onehot,aac

    # The approved representation pool for the dataset
    python tests/experiments/opig_humanness.py --datasets humanness_therapeutic

    # 'species' and 'therapeutic' are NOT accepted: DATASET_REGISTRY holds
    # humanness_therapeutic and nothing else here, and an unknown name makes
    # main() print ERROR and return 1 before any work.

Reference:
    Prihoda et al. (2022). BioPhi: A platform for antibody design,
    humanization and humanness evaluation. mAbs 14(1):2020203.
    Marks et al. (2021). J Mol Biol 433(11):166928. (Hu-mAb, one of 8 methods)
    Ramon et al. (2024). Bioinformatics 40(6):btae332. (AbNatiV, species data)
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY, _identity_groups
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    fold_reselection, build_fold_reselection_input,
    hr_exclude_folds, hr_direction,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# humanness_species is not a key of DATASET_REGISTRY (antibody_baseline_survey.py
# registers humanness_therapeutic only), so naming it here made main's validation
# loop print ERROR and return 1 before any work whenever --datasets was left off.
# The cluster submitter passes --datasets humanness_therapeutic and escaped that;
# a bare local run did not.
DEFAULT_DATASETS = ['humanness_therapeutic']

# Representations come from DATASET_REPS for each dataset — the pool the
# rep-pruning step approved and precompute builds. The hand-written list this
# replaced asked for aac, esm2, antiberty and paratope, none of which are in
# humanness_therapeutic's pool and none of which this experiment does anything
# special with; it also skipped 5 of the 10 that are.
#
# 'onehot' is added back: scientific question 3 in the docstring compares the LM
# reps against the classical encodings and names BLOSUM and onehot. BLOSUM is in
# the pool; onehot is not, and dropping it would leave that question with one
# classical encoding instead of the two it asks about.
#
# NOTE: LLR reps (esmc_llr, antiberty_llr) stay out — they require mutation
# lists, but the humanness loader sets mutations=[[] for _ in range(n)]
# (see src/kirby/datasets/humanness.py:244,322), which makes
# src/kirby/representations/mutation.py:490-494 emit all-zero 3D vectors per sample.
REP_EXCEPTIONS = ['onehot']


def _default_reps(dataset_name):
    from experiments._shared import approved_rep_pool
    return approved_rep_pool(dataset_name, extra=REP_EXCEPTIONS)

MODELS_LIST = DOWNSTREAM_MODELS


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
    from experiments._shared import smoke_budgets
    return {'speed': 'full', 'budgets': smoke_budgets(),
            'internal_evaluators': ['rf'],
            'importance_methods': ['lightgbm_gain']}


# Published reference baselines
# Prihoda et al. 2022 (BioPhi): threshold-sweep on full therapeutic dataset,
# NO train/test split. Numbers from notebook 03_oasis_humanness.ipynb Cell 49.
# Note: BioPhi uses VH+VL averaged. KIRBy uses VH-only.
PAPER_BASELINES = {
    # No 'species' entry — Marks 2021 reports near-perfect AUROC on OAS
    # species-level but doesn't publish numbers, and BioPhi doesn't either.
    # Comparison block is skipped when this key is missing.
    'therapeutic': {
        # Human vs non-human therapeutics (BioPhi Cell 49, VH+VL averaged)
        'Hu-mAb (Marks)':      {'auroc': 0.977, 'accuracy': 0.933},
        'OASis relaxed':       {'auroc': 0.972, 'accuracy': 0.944},
        'Germline content':    {'auroc': 0.968, 'accuracy': 0.928},
        'IgReconstruct':       {'auroc': 0.967, 'accuracy': 0.926},
        'MG Score':            {'auroc': 0.959, 'accuracy': 0.918},
        'AbLSTM':              {'auroc': 0.937, 'accuracy': 0.878},
        'T20':                 {'auroc': 0.896, 'accuracy': 0.839},
        'Z-score':             {'auroc': 0.837, 'accuracy': 0.765},
    },
}


# =============================================================================
# EVALUATION
# =============================================================================

def _is_binary_rep(X):
    """Check if feature matrix is binary (0/1)."""
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def evaluate_rep_clf(X, labels, groups, model, n_folds):
    """Evaluate one (rep, model) pair for classification via stratified CV.

    Returns dict with accuracy, f1_weighted, auroc, balanced_accuracy (mean/std).
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    # A non-finite entry is not a score problem, it is a representation
    # problem, and it reaches the mean silently: StandardScaler propagates a
    # NaN (and raises on an inf, which the caller prints as SKIPPED), LightGBM
    # and RandomForest (scikit-learn 1.6.1, checked 2026-09-28) accept a NaN as
    # a missing value and still return a number, while SVC raises and loses
    # the fold. Said out loud here so a
    # degraded row cannot be read as a weak representation.
    _n_bad = int((~np.isfinite(X)).sum())
    if _n_bad:
        print(f"      [{_n_bad} non-finite feature values out of "
              f"{X.shape[0] * X.shape[1]}]", flush=True)

    if groups is None or len(np.unique(groups)) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds}-fold group split: "
            f"groups={'None' if groups is None else len(np.unique(groups))} unique")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(cv.split(X, labels, groups=groups))

    is_binary = _is_binary_rep(X)
    acc_scores, f1_scores, auroc_scores, bacc_scores = [], [], [], []

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
            # Don't swallow fold-fit failures silently (OPIG-18): a dropped
            # fold biases the mean. Log the model + exception type and skip.
            print(f"      [fold fit failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            continue

        try:
            preds = m.predict(X_val)
        except Exception as e:
            # predict sat outside every try. SVC refuses a NaN in the held-out
            # rows, so the ValueError left this function and the caller printed
            # the whole representation-and-model pair as SKIPPED, throwing away
            # the folds that had scored; any other exception type from predict
            # reached main() and failed the whole dataset. Now it loses this
            # fold only, like a failed fit, and n_folds_scored says so.
            print(f"      [fold predict failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            continue
        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds, average='weighted'))
        bacc_scores.append(balanced_accuracy_score(y_val, preds))

        # AUROC (binary only, need decision function or probabilities)
        try:
            if hasattr(m, 'decision_function'):
                scores = m.decision_function(X_val)
            elif hasattr(m, 'predict_proba'):
                scores = m.predict_proba(X_val)[:, 1]
            else:
                scores = preds.astype(float)
            auroc_scores.append(roc_auc_score(y_val, scores))
        except Exception as e:
            # Was a bare pass. A held-out fold carrying one class makes
            # roc_auc_score raise, so the AUROC mean is then taken over fewer
            # folds than the accuracy mean beside it, with nothing on the page
            # to say so. Both the reason and the two fold counts are recorded
            # below.
            print(f"      [AUROC not scored on one fold: "
                  f"{type(e).__name__}: {e}]", flush=True)

    if not acc_scores:
        return None

    # n_folds_requested is what was asked for, n_folds_scored how many folds
    # produced a number. They differ when a fit raised, and a mean over 2 folds
    # otherwise reads exactly like a mean over 3.
    result = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
        'f1_weighted': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
        'balanced_accuracy': {'mean': np.mean(bacc_scores), 'std': np.std(bacc_scores)},
        'n_folds_requested': int(n_folds),
        'n_folds_scored': len(acc_scores),
    }
    if auroc_scores:
        result['auroc'] = {'mean': np.mean(auroc_scores), 'std': np.std(auroc_scores)}
        result['n_folds_scored_auroc'] = len(auroc_scores)
    return result


def evaluate_dataset(dataset_name, rep_names, n_folds, pruning_mode='default_pruning'):
    """Evaluate all reps × models on one humanness dataset.

    Returns dict of {rep_name: {model_name: {metric: {mean, std}}}}.
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    data = load_dataset(dataset_name)
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_pos = int((labels == 1).sum())
    print(f"  {n} samples, {n_pos} human ({100*n_pos/n:.1f}%), "
          f"{n - n_pos} non-human ({100*(n-n_pos)/n:.1f}%)")

    # Identity groups via NEAR-duplicate clustering (ISS-29/30 leakage fix, §B).
    # Delegates to the SAME helper the main antibody survey uses
    # (antibody_baseline_survey._identity_groups -> sequence_identity_clusters:
    # mandatory Levenshtein, single-linkage at 95% identity, canonical cluster IDs).
    # This REPLACES the old exact-match grouping, which placed single-indel
    # near-duplicate VH sequences in DIFFERENT groups -> they leaked across the
    # StratifiedGroupKFold boundary, the exact leakage §B exists to prevent.
    # Effect by preset: species (~10K near-unique OAS VH) stays ~one-group-per-sample;
    # therapeutic (INN entries sharing near-identical VH) now co-clusters those.
    groups, _n_groups = _identity_groups(heavy_seqs, light_seqs)

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag=dataset_name,
        mutations=data.get('mutations'),
    )

    if not reps:
        print("  No reps available, skipping dataset")
        return None

    # Run the hybrid search. NO 'kirby_hybrid' representation is added to
    # `reps`: it used to be built by refitting the search's winning blueprint on
    # every antibody's label, so each fold's own labels had chosen both the
    # blueprint and the feature indices it was then scored on. Those rows
    # (rf AUROC 0.9229 in the stored result file) are gone as of 2026-09-04,
    # split audit. The honest comparison below is the only hybrid number here.
    # Augmentation config matches humanness_therapeutic in DATASET_REPS:
    # augmentation = [] (no augmentation reps for antibody classification).
    #
    # HONEST COMPARISON (the correct arm): the SearchResult returned here already
    # carries PER-FOLD scores for every candidate hybrid recipe AND every single
    # baseline rep, per downstream model, with each hybrid built PER FOLD (no
    # full-data selection leak). We hand those straight to
    # build_fold_reselection_input() to produce the standard "fold_reselection" JSON
    # block (contract json_convention). A reader feeds that block into
    # fold_reselection(), which re-selects the winner per held-out fold (same tax
    # on both arms) and reports a per-model, held-out hybrid-vs-best-single delta.
    # The shared helper fits this experiment exactly (classification, primary
    # metric = auc), so NO local re-score helper is needed here.
    reselection_block = None
    # Filled from the same SearchResult as reselection_block, below. Stay None
    # when the hybrid search is unavailable, and the two keys they feed are then
    # absent from the saved dict rather than present and empty.
    top2_block = None
    hybrid_delta_block = None
    # One entry per downstream model that has NO hybrid-versus-reference numbers,
    # and the reason. Written so a table of three models cannot be read as a
    # table of five.
    hybrid_delta_unscored = {}
    try:
        from experiments._shared import (resolve_pruning_kwargs,
                                          get_best_search_for_downstream)
        _pkw = resolve_pruning_kwargs(pruning_mode, len(labels),
                                       dataset_label=dataset_name)
        _smoke = _smoke_search_kwargs()
        search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            # augmentations=None -> pool iterates only aug_condition='none',
            # so no cache-key collision is possible (helper sets aug_strat
            # per iteration). See project_cache_collision_fix.
            augmentations=None,
            metric='auc',
            task_type='classification',
            dataset_name=dataset_name,
            return_fitted=True,
            adaptive_pruning=0.25,
            **{k: v for k, v in _smoke.items() if k != 'speed'},
            **_pkw,
        )
        if search is not None:
            top3 = extract_top3_base_reps(search)
            if top3:
                print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            # Per-model, held-out per-fold reselection input (both arms carry per-fold
            # scores). Model set is derived by the helper from what the search
            # actually evaluated, so gates are respected automatically.
            reselection_block = build_fold_reselection_input(search)

            # Top 2 beside the top 3 printed above, and saved rather than
            # printed. The line above names 3 representations under rf only and
            # keeps nothing; until now the JSON held no ranking at all, so a
            # reader could not see the second-best representation, or any
            # hybrid configuration other than the winner, for any of the 5
            # downstream models. Both halves read rows this search already
            # stored: 0 new model fits.
            try:
                from experiments._shared import (select_model_rep_pairs,
                                                  top_hybrids, hybrid_deltas,
                                                  resolve_comparison_models)
                top2_block = select_model_rep_pairs(
                    search, models=MODELS_LIST, n_base=2, n_hybrids=2,
                    # require_full=False: a downstream model this search never
                    # scored yields a short list and a line in 'shortfalls'.
                    # The default raises, which would lose the whole dataset
                    # over a missing model.
                    require_full=False, context=f'{dataset_name}/top2')

                # The hybrid against two reference numbers, per downstream
                # model: the mean of the base representations that actually fed
                # it, and the best single base representation. A hybrid can beat
                # the first and lose to the second. 0 new model fits.
                #
                # MODELS_LIST is the five names written down in this file
                # (opig_humanness.py:113). The search scores whatever list it
                # was given, which can be shorter: a dataset's skip policy drops
                # svm, and svm drops itself above 10,000 molecules. Asking for a
                # model the search never scored used to give a bare failure line
                # with no reason. resolve_comparison_models keeps the names the
                # search has scores for and names every one it dropped. It never
                # substitutes another model's rows.
                hybrid_delta_block = {}
                _delta_models, _delta_dropped = resolve_comparison_models(
                    search, MODELS_LIST)
                for _name, _why in _delta_dropped:
                    hybrid_delta_unscored[_name] = _why
                    print(f"  hybrid_deltas[{_name}]: not computed -- {_why}")
                for _m in _delta_models:
                    try:
                        # hybrid= is passed by hand. The defaulted call used to
                        # read _picked['selected'][0]['row'], a key top_hybrids
                        # never writes, and raised KeyError('row'); _shared.py
                        # was corrected on 2026-09-28 to read the entry itself.
                        # Passing the entry here does the same thing either
                        # way, so this call does not depend on which version of
                        # that helper is on disk.
                        _top1 = top_hybrids(search, _m, 1, require_full=False,
                                            context=f'{dataset_name}/{_m}')
                        if not _top1['selected']:
                            hybrid_delta_unscored[_m] = (
                                _top1.get('shortfall_reason')
                                or 'the search stored no hybrid configuration '
                                   'for this downstream model')
                            print(f"  hybrid_deltas[{_m}]: "
                                  f"{hybrid_delta_unscored[_m]}")
                            continue
                        hybrid_delta_block[_m] = hybrid_deltas(
                            search, _m, hybrid=_top1['selected'][0],
                            context=f'{dataset_name}/{_m}')
                    except Exception as _e:
                        hybrid_delta_unscored[_m] = (f'{type(_e).__name__}: {_e}')
                        print(f"  hybrid_deltas[{_m}] failed: {_e}")

                # The Blueprint object off each selected hybrid: it is a recipe
                # object, not JSON, and every field of it this file reports
                # (allocation method, importance method, budget, internal
                # evaluator, step size, representation count) is already a key
                # of the same entry. Dropped AFTER hybrid_deltas, which reads
                # Blueprint.allocation_per_fold to find the representations that
                # fed the hybrid.
                for _sel in top2_block.get('per_model', {}).values():
                    for _e in _sel['hybrid']['selected']:
                        _e.pop('blueprint', None)
            except Exception as e:
                # A failure here leaves the two new keys out of the saved dict.
                # Every number this file reported before stays where it was.
                print(f"  Top-2 selection failed: {e}")
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Evaluate all reps × models
    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    print(f"\n  Evaluating {len(reps)} reps × {len(models)} models × {n_folds}-fold CV...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = evaluate_rep_clf(X, labels, groups, factory(), n_folds)
            except ValueError as e:
                print(f"    {rep_name:25s} {model_name:8s}  SKIPPED: {e}")
                continue
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                acc = res['accuracy']['mean']
                auroc = res.get('auroc', {}).get('mean', float('nan'))
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.1f}s)")
        if rep_results:
            results[rep_name] = {
                'dim': X.shape[1],
                'models': rep_results,
            }

    out = {
        'n_samples': n,
        'n_human': n_pos,
        'n_nonhuman': n - n_pos,
        'results': results,
        # Which representations were asked for and which ones actually built.
        # compute_or_load_reps prints FAILED and drops a representation whose
        # builder raises -- esmc needs the esm package, which is not installed
        # everywhere -- so a table of 9 representations could be read as the
        # approved pool of 10. The two lists say which it is.
        'reps_requested': list(rep_names),
        'reps_built': sorted(reps.keys()),
        'reps_not_built': sorted(set(rep_names) - set(reps.keys())),
    }
    # Additive: the honest, per-model, held-out comparison block (contract
    # json_convention). Absent only when the hybrid search was unavailable.
    if reselection_block is not None:
        out['fold_reselection'] = reselection_block
        # Also store the COMPUTED comparison, not just its inputs. Before
        # 2026-09-04 the per-model hybrid-vs-best-single numbers were printed
        # and then thrown away, so the only hybrid rows in this file's JSON
        # were the leaked ones. A reader now gets the held-out numbers without
        # re-running fold_reselection.
        out['reselected_comparison'] = fold_reselection(
            reselection_block.get('candidates_by_model', {}),
            reselection_block.get('baselines_by_model', {}),
            # hr_direction, not a defaulted False: a block with neither a
            # metric name nor a stored flag is unscoreable, and guessing
            # higher-is-better inverts every rmse delta. Fixed 2026-09-10.
            hr_direction(reselection_block),
            metric_name=reselection_block.get('metric_name'),
            exclude_folds=hr_exclude_folds(reselection_block),
        )
    # Additive, and named so they cannot be read as the existing blocks: the
    # 2 best base representations and the 2 best hybrid configurations per
    # downstream model, and the two hybrid deltas per downstream model. Nothing
    # above is recomputed. Both are keyed by name, and nothing iterates this
    # dict as models or as representations -- print_analysis reads 'results',
    # 'n_samples', 'n_human', 'n_nonhuman' and 'fold_reselection' by name, and
    # main() only writes it to JSON.
    if top2_block is not None:
        out['top2_pairs_by_model'] = top2_block
    if hybrid_delta_block:
        out['hybrid_deltas_by_model'] = hybrid_delta_block
    if hybrid_delta_unscored:
        out['hybrid_deltas_unscored_models'] = hybrid_delta_unscored
    return out


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(dataset_name, ds_results):
    """Print comparison table with paper baselines."""
    if ds_results is None:
        return

    results = ds_results['results']
    baselines = PAPER_BASELINES.get(dataset_name.replace('humanness_', ''), {})

    print(f"\n{'=' * 90}")
    print(f"HUMANNESS EXPERIMENT RESULTS: {dataset_name}")
    print(f"  {ds_results['n_samples']} samples "
          f"({ds_results['n_human']} human, {ds_results['n_nonhuman']} non-human)")
    print(f"{'=' * 90}")

    # Header
    print(f"\n  {'Rep':25s} {'Model':8s} {'Acc':>10s} {'AUROC':>10s} {'F1w':>10s} {'BalAcc':>10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Paper baselines first
    for method, metrics in baselines.items():
        acc = metrics.get('accuracy', float('nan'))
        auroc = metrics.get('auroc', float('nan'))
        acc_s = f"{acc:.4f}" if not np.isnan(acc) else "—"
        auroc_s = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        print(f"  {method:25s} {'paper':8s} {acc_s:>10s} {auroc_s:>10s} {'—':>10s} {'—':>10s}")

    if baselines:
        print(f"  {'. . .':25s}")

    # KIRBy results: find best model per rep, sort by AUROC then accuracy
    rows = []
    for rep_name, rep_data in results.items():
        for model_name, metrics in rep_data['models'].items():
            acc = metrics['accuracy']['mean']
            auroc = metrics.get('auroc', {}).get('mean', float('nan'))
            f1w = metrics['f1_weighted']['mean']
            bacc = metrics['balanced_accuracy']['mean']
            rows.append((rep_name, model_name, acc, auroc, f1w, bacc))

    # Sort by AUROC descending (fallback to accuracy)
    rows.sort(key=lambda r: (r[3] if not np.isnan(r[3]) else 0, r[2]), reverse=True)

    for rep_name, model_name, acc, auroc, f1w, bacc in rows:
        auroc_s = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        print(f"  {rep_name:25s} {model_name:8s} {acc:10.4f} {auroc_s:>10s} "
              f"{f1w:10.4f} {bacc:10.4f}")

    # Summary: best individual rep. There is no hybrid row here — the hybrid's
    # only number in this experiment is the held-out one printed below.
    #
    # ONE METRIC, NOT WHICHEVER IS PRESENT. This loop used to take a row's
    # AUROC when it had one and its accuracy when it did not, then compare the
    # winner against the published AUROC below. A row scored on accuracy can
    # win that maximum and be subtracted from an AUROC, which is two different
    # quantities either side of the minus sign. AUROC is used when any row
    # carries one, and rows without it are left out of the maximum.
    _have_auroc = [r for r in rows if not np.isnan(r[3])]
    _summary_metric = 'AUROC' if _have_auroc else 'accuracy'
    _summary_rows = _have_auroc if _have_auroc else rows
    _value_idx = 3 if _have_auroc else 2

    best_individual = None
    for row in _summary_rows:
        metric_val = row[_value_idx]
        if best_individual is None or metric_val > best_individual[2]:
            best_individual = (row[0], row[1], metric_val)

    print(f"\n  Summary:")
    if best_individual:
        print(f"    Best individual representation by {_summary_metric}: "
              f"{best_individual[0]} ({best_individual[1]}) "
              f"= {best_individual[2]:.4f}")
        if _have_auroc and len(_have_auroc) < len(rows):
            print(f"    {len(rows) - len(_have_auroc)} of {len(rows)} "
                  f"representation-and-model rows scored no AUROC and are not "
                  f"in that maximum.")

    if baselines and best_individual:
        # The deltas below are printed because the published numbers are the
        # reason this experiment exists, and they are not a like-for-like
        # comparison. Four differences, all in the published method's favour
        # or against it in ways this line cannot net out.
        print(f"    The published numbers were measured differently, so read "
              f"each delta as a landmark and not as a win or a loss.")
        print(f"    Those methods are scored by sweeping a threshold over "
              f"every antibody at once, with no training and no held-out set.")
        print(f"    They average a heavy-chain and a light-chain score, while "
              f"every number above is heavy chain only.")
        print(f"    The KIRBy value is the best of "
              f"{len(_summary_rows)} representation-and-model rows, picked on "
              f"the same folds it is reported on, so it is the optimistic end "
              f"of what one prespecified choice would give.")

    for method, metrics in baselines.items():
        ref_val = metrics.get('auroc', metrics.get('accuracy', float('nan')))
        _ref_metric = 'AUROC' if 'auroc' in metrics else 'accuracy'
        if np.isnan(ref_val) or not best_individual:
            continue
        if _ref_metric != _summary_metric:
            print(f"    vs {method}: not printed -- the published number is "
                  f"{_ref_metric} and the best row here is {_summary_metric}.")
            continue
        delta = best_individual[2] - ref_val
        print(f"    vs {method}: {delta:+.4f} {_summary_metric}")

    # -------------------------------------------------------------------------
    # HONEST per-model, held-out comparison (supersedes the max-vs-max summary
    # above, which selects hybrid and best-single on the SAME folds and scores a
    # full-data-selected hybrid rep). Pure arithmetic on per-fold scores already
    # in the search result — no model fitting, no re-run. The winner (hybrid
    # recipe / baseline rep) is re-picked per held-out fold from the OTHER folds'
    # mean, and the identical held-out tax is applied to both arms.
    # -------------------------------------------------------------------------
    reselection_block = ds_results.get('fold_reselection')
    if reselection_block:
        res = fold_reselection(
            reselection_block.get('candidates_by_model', {}),
            reselection_block.get('baselines_by_model', {}),
            # The console path was the one left on a defaulted False when the
            # saved path above was corrected. Fixed 2026-09-10.
            hr_direction(reselection_block),
            metric_name=reselection_block.get('metric_name'),
            exclude_folds=hr_exclude_folds(reselection_block),
        )
        print(f"\n  Honest per-model comparison (leave-one-fold-out, held-out "
              f"selection; primary metric = auc):")
        print(f"    Delta = delta_mean_of_folds, the mean over folds of "
              f"(hybrid AUROC - best-single AUROC) on that held-out fold. "
              f"Folds = how many folds both arms scored.")
        print(f"    {'Model':10s} {'Hybrid':>10s} {'BestSingle':>12s} "
              f"{'Delta':>10s} {'Folds':>6s}")
        print(f"    {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*6}")
        for model_name in sorted(res.keys()):
            r = res[model_name]
            h = r['hybrid_reselected']
            b = r['best_single_reselected']
            # delta_mean_of_folds, not delta: 'delta' subtracts the two arms'
            # MEAN scores, which equals the paired quantity only when both arms
            # scored the same fold set (_shared.py:1552-1568). The paired
            # per-fold difference is the house convention.
            d = r['delta_mean_of_folds']
            h_s = f"{h:.4f}" if not np.isnan(h) else "—"
            b_s = f"{b:.4f}" if not np.isnan(b) else "—"
            d_s = f"{d:+.4f}" if not np.isnan(d) else "—"
            flag = ' (single-fold: optimistic, NOT honest)' if r['single_fold'] else ''
            print(f"    {model_name:10s} {h_s:>10s} {b_s:>12s} {d_s:>10s} "
                  f"{r['n_paired_folds']:>6d}{flag}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Humanness classification experiment (OPIG B3). '
                    'Extends Prihoda et al. 2022 (BioPhi) with KIRBy antibody reps.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated dataset names '
             '(default: humanness_therapeutic)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: the approved pool for the '
             'dataset, plus onehot)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of stratified CV folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_humanness.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default: 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard what an earlier attempt finished '
             '(<output>.partial.jsonl) and recompute every item.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()


    apply_distance_metric_arg(args)
    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    rep_override = (args.reps.split(',') if args.reps else None)

    # Validate datasets
    for ds in datasets:
        if ds not in DATASET_REGISTRY:
            print(f"ERROR: Unknown dataset '{ds}'")
            return 1

    print("=" * 70)
    print("HUMANNESS CLASSIFICATION EXPERIMENT (OPIG B3)")
    print(f"  Paper: Prihoda et al. 2022, mAbs 14(1):2020203 (BioPhi)")
    print(f"  Datasets: {datasets}")
    print(f"  Reps: {rep_override if rep_override else 'per-dataset approved pool'}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    output_path = (args.output or
                   'tests/results/comparisons/opig_humanness.json')

    # Each dataset is written to <output>.partial.jsonl as soon as it finishes
    # and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). Before, a failure or a kill lost every
    # dataset, because the one JSON below was written after the last. A dataset
    # that raised or returned {'error'} is not stored, is retried next time,
    # and makes the script exit 1 after the JSON is written.
    from experiments._checkpoint import (
        script_checkpoint, is_failed_item, exit_if_failed)
    _ck = script_checkpoint('opig_humanness', output_path, args,
                            exclude=('datasets', 'output'))
    _failed = []

    all_results = {}
    skipped = {}
    for ds_name in datasets:
        try:
            from experiments._shared import NoApprovedRepPool
            try:
                rep_names = (rep_override if rep_override
                             else _default_reps(ds_name))
            except NoApprovedRepPool as e:
                # No approved representation list for this dataset. Refuse rather
                # than fall back to a hand-written one; --reps overrides.
                print(f"\n  SKIPPED {ds_name}: {e}")
                skipped[ds_name] = str(e)
                continue
            if _ck.done(dataset=ds_name, rep_names=rep_names):
                ds_results = _ck.get(dataset=ds_name, rep_names=rep_names)
                print(f"  {ds_name}: finished in an earlier attempt, read back")
            else:
                ds_results = evaluate_dataset(ds_name, rep_names, args.n_folds,
                                              pruning_mode=args.pruning_mode)
                _ck.put(ds_results, dataset=ds_name, rep_names=rep_names)
            if is_failed_item(ds_results):
                _failed.append(ds_name)
            if ds_results is not None:
                all_results[ds_name] = ds_results
                print_analysis(ds_name, ds_results)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(ds_name)

    # Save results
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if skipped:
        all_results['_skipped'] = skipped

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")
    exit_if_failed(_failed, 'dataset')


if __name__ == '__main__':
    main()
