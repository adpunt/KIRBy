#!/usr/bin/env python3
"""S09: Cross-Dataset Generalization Summary.

Paper: Similarity
Conclusion: Results hold across 4 diverse datasets.
Dependencies: s02, s04, s05, s06, s08 (reads their JSON outputs).

Aggregates metrics across datasets: SAC deltas, cliff reductions, kNN deltas,
AD deltas, FP reductions, dim ratios. Runs Wilcoxon test if enough datasets.

Each quantity is reported next to the arm it belongs to (see _ARM_* below) and
the input key it was read from. Where an experiment stores both a
selection-optimistic and a held-out version of the same comparison, BOTH are
aggregated, under separate names; where it stores only one, the reason the
other is absent is recorded in place of a gap. No existing key changes value.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from scipy.stats import wilcoxon

from experiments._shared import (
    save_experiment_results, load_experiment_results, fold_reselection,
    hr_exclude_folds, hr_direction,
)
from experiments.similarity._summary_keys import (
    ACCEPTED_INPUT_KEYS, MIN_DATASETS_FOR_WILCOXON,
    selection_rule_check, get_input as _get_input_for_rule,
    accepted_names, get_input, print_refusal, refusal,
)

# Metrics where lower scores are better (RMSE, MAE).
_LOWER_IS_BETTER_METRICS = {'rmse', 'mae'}


def _metric_lower_is_better(metric_name):
    """Infer metric direction from name.  Default: higher is better."""
    return metric_name.lower() in _LOWER_IS_BETTER_METRICS if metric_name else False


# Map from experiment IDs used internally to the result directory names.
# The 'expN_*' spellings written by tests/similarity_experiments.py are read too
# (see _summary_keys.ACCEPTED_INPUT_KEYS).
EXPERIMENT_IDS = list(ACCEPTED_INPUT_KEYS)

# (aggregate metric, the per-dataset summary field carrying it, its input).
# Used only to explain a refusal — never to decide what gets computed.
METRIC_FIELDS = [
    ('SAC_delta', 'sac_delta', 's02_hybrid_similarity'),
    ('RMSE_cliff_reduction_%', 'cliff_rmse_reduction_pct', 's04_activity_cliffs'),
    ('kNN_delta', 'knn_delta', 's05_knn_prediction'),
    ('AD_corr_delta', 'ad_corr_delta', 's06_applicability_domain'),
    ('optimal_dim_ratio', 'optimal_dim_ratio', 's08_optimal_dimension'),
]

# The counterpart quantities, read from the same six inputs. Listed for the same
# purpose as METRIC_FIELDS: so a refusal names what each dataset lacked. A run
# in which only these reached the aggregate is a result, not a refusal.
HELDOUT_METRIC_FIELDS = [
    ('SAC_delta_heldout', 'sac_delta_heldout', 's02_hybrid_similarity'),
    ('RMSE_cliff_reduction_%_vs_weighted_rep',
     'cliff_rmse_reduction_pct_vs_weighted_rep', 's04_activity_cliffs'),
    ('cliff_overall_delta_heldout', 'cliff_overall_delta_heldout',
     's04_activity_cliffs'),
    ('kNN_delta_fixed_ref', 'knn_delta_fixed_ref', 's05_knn_prediction'),
    ('kNN_delta_heldout', 'knn_delta_heldout', 's05_knn_prediction'),
    ('SAC_delta_heldout_s08', 'sac_delta_heldout_s08', 's08_optimal_dimension'),
]

# ── Arm labels ───────────────────────────────────────────────────────────────
# Every number stored and printed below carries one of these, plus the input key
# it was read from, so which arm a number came from is never left to inference.
#   selection-optimistic: the winning cell (best rep, best budget, best k,
#       best weight method) is picked on the same scores that are reported.
#   held-out: the winner is re-picked on the other folds and the reported score
#       is read off the held-out fold (experiments._shared.fold_reselection).
#   pre-registered: a fixed cell fixed before the grid was scored — no selection
#       over the grid, but the score is still in-sample.
#   no winner selected: a descriptive statistic with no cell to pick.
_ARM_OPTIMISTIC = 'selection-optimistic'
_ARM_HELD_OUT = 'held-out'
_ARM_PREREGISTERED = 'pre-registered cell, scored in-sample'
_ARM_NO_SELECTION = 'no winner selected'

# Print / store order of the aggregated series: each optimistic quantity is
# followed by the counterpart arms read from the same experiment.
_SERIES_ORDER = (
    'SAC_delta',
    'SAC_delta_heldout',
    'RMSE_cliff_reduction_%',
    'RMSE_cliff_reduction_%_vs_weighted_rep',
    'cliff_overall_delta_heldout',
    'kNN_delta',
    'kNN_delta_fixed_ref',
    'kNN_delta_heldout',
    'AD_corr_delta',
    'optimal_dim_ratio',
    'SAC_delta_heldout_s08',
    'SAC_delta_heldout_s08_unweighted',
)


def _is_number(value):
    """True for a real, non-NaN number."""
    try:
        return value is not None and not np.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _record(ds_summary, field, value, arm, source):
    """Store one number with its arm label and the input key it was read from."""
    ds_summary[field] = value
    ds_summary[f'{field}_arm'] = arm
    ds_summary[f'{field}_source'] = source


def _record_absent(ds_summary, field, source, reason):
    """Record why a quantity could not be read, instead of leaving a gap."""
    ds_summary[f'{field}_status'] = {'computed': False, 'source': source,
                                     'reason': reason}


def _heldout_delta(hr_result, model_key):
    """Paired held-out delta and arm label from a stored fold_reselection result.

    Accepts the {model: entry} shape (s02/s05) and the already-unwrapped
    single entry (s08 'sac_reselected'). Reads 'delta_mean_of_folds': the mean over
    folds where BOTH arms scored, which fold_reselection documents as the paired
    quantity to report. Returns (value, arm, None) or (None, None, reason).
    """
    if not isinstance(hr_result, dict):
        return None, None, 'key absent'
    if 'error' in hr_result:
        return None, None, f"stored an error: {hr_result['error']}"
    if 'delta_mean_of_folds' in hr_result or 'delta' in hr_result:
        entry = hr_result           # already one model's entry
    else:
        entry = hr_result.get(model_key)
        if not isinstance(entry, dict):
            return None, None, (f"no '{model_key}' entry (models present: "
                                f"{sorted(hr_result) or 'none'})")
        if 'error' in entry:
            return None, None, (f"'{model_key}' entry stored an error: "
                                f"{entry['error']}")
    value = entry.get('delta_mean_of_folds')
    if not _is_number(value):
        return None, None, (f"delta_mean_of_folds={value!r}, "
                            f"n_paired_folds={entry.get('n_paired_folds')}")
    # single_fold=True means fold_reselection had one fold, so no held-out tax was
    # applied; the flag rides in the arm label rather than being dropped.
    arm = (_ARM_HELD_OUT + ' (single fold: no held-out tax applied)'
           if entry.get('single_fold') else _ARM_HELD_OUT)
    # The recipe — allocation method, budget, importance method, evaluator —
    # can have been picked over every row even when the features were refit per
    # fold. Then only half the number is held out, and the label says so
    # instead of the reader having to know which experiment did which.
    if entry.get('recipe_selection') == 'whole_dataset':
        arm += ' (recipe picked on all rows; only the features are refit)'
    elif entry.get('recipe_selection') == 'other_folds_of_pool_search':
        # s04 and s05: the recipe was ranked on the pool search's other folds,
        # but that search ran over every row. The fold's scores are held out
        # of the ranking; its rows are not. (dirty-local 2026-09-28)
        arm += (' (recipe ranked on the pool search\'s other folds; that '
                'search saw every row)')
    # Fold 0 chose which recipes survived, so its term was dropped from both
    # arms. Say how many folds the mean actually covers.
    if entry.get('excluded_folds'):
        arm += (f" (folds {sorted(entry['excluded_folds'])} dropped: the "
                f"search filtered its recipes on them)")
    return float(value), arm, None


def _rescore_block(block):
    """Apply the shared leave-one-fold-out arithmetic to a stored contract block.

    s04 persists 'fold_reselection' (per-fold scores for both arms) but no
    re-scored result, so the result is produced here. Pure arithmetic over
    stored numbers — fold_reselection fits no model and reads no data.
    Returns (result, None) or (None, reason).
    """
    if not isinstance(block, dict):
        return None, 'key absent'
    if 'error' in block:
        return None, f"stored an error: {block['error']}"
    candidates = block.get('candidates_by_model') or {}
    baselines = block.get('baselines_by_model') or {}
    if not candidates or not baselines:
        return None, (f"{len(candidates)} candidate model(s) and "
                      f"{len(baselines)} baseline model(s); both arms required")
    # The block says which folds the search filtered its own recipes on; those
    # folds come out of both arms. s02, s05, s12 and s13 all read it the same
    # way, so s04's block is re-scored on the same folds they would use.
    # Direction through hr_direction, not bool(block.get('lower_is_better')):
    # a missing flag read as False is higher-is-better, which inverts every
    # RMSE delta. hr_direction checks the stored metric name against
    # BUILTIN_METRICS and raises when the block carries no direction at all.
    # The recipe label s04 writes is passed through so the arm label says it.
    # (dirty-local 2026-09-28)
    try:
        _lib = hr_direction(block)
        return fold_reselection(candidates, baselines, _lib,
                              metric_name=block.get('metric_name'),
                              exclude_folds=hr_exclude_folds(block),
                              recipe_selection=block.get('recipe_selection')), None
    except Exception as e:
        return None, f'fold_reselection raised {type(e).__name__}: {e}'


def run(all_dataset_results):
    """Consolidate findings across datasets. Show improvement is consistent."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S09: CROSS-DATASET GENERALIZATION SUMMARY")
    print("=" * 70)

    # 'arm_legend' spells out the labels attached to every number below, so a
    # stored file explains itself without this script.
    results = {
        'per_dataset': {},
        'summary': {},
        'arm_legend': {
            _ARM_OPTIMISTIC: ('the winning cell is picked on the same scores '
                              'that are then reported'),
            _ARM_HELD_OUT: ('the winner is re-picked on the other folds and '
                            'the score is read off the held-out fold '
                            '(experiments._shared.fold_reselection)'),
            _ARM_PREREGISTERED: ('the cell was fixed before the grid was '
                                 'scored; no selection over the grid, score '
                                 'still in-sample'),
            _ARM_NO_SELECTION: 'a descriptive statistic with no cell to pick',
        },
    }

    cliff_n_skipped_clf = 0

    # One entry per aggregated series: its values, the dataset each value came
    # from (the lists are appended to conditionally, so position alone does not
    # identify the dataset), and the arm and input key that apply to all of them.
    series = {}

    def _series(name, ds_name, value, arm, source):
        """Append one dataset's value to a series, carrying the series labels."""
        entry = series.setdefault(name, {'values': [], 'datasets': [],
                                         'arm': arm, 'source': source})
        entry['values'].append(float(value))
        entry['datasets'].append(ds_name)

    # ⛔ DO THE FIVE INPUTS AGREE ON HOW THEIR NUMBERS WERE SELECTED?
    # Checked 2026-09-18, owner's decision. This function takes one number from
    # each of s02, s04, s05, s06 and s08 and runs a signed-rank test down the
    # column. If those files were written under different selection rules the
    # test compares quantities that are not the same, and it does not raise —
    # it returns a p-value. Each of the five now stamps the rule it ran under
    # (_summary_keys.SELECTION_RULE); a file written before the stamp existed
    # reads 'unstamped', which is a value, not a wildcard.
    #
    # A disagreement does NOT stop the run. Everything per dataset is still
    # computed and printed. What it stops is the signed-rank test, because that
    # is the only part that pools the five.
    _rule_checks = {}
    for _ds_name, _ds_results in all_dataset_results.items():
        _rule_checks[_ds_name] = selection_rule_check({
            _cid: _get_input_for_rule(_ds_results, _cid)
            for _cid in ACCEPTED_INPUT_KEYS})
    _mixed = {d: c for d, c in _rule_checks.items() if not c['ok']}
    results['selection_rule_by_dataset'] = {
        d: c['rules'] for d, c in _rule_checks.items()}
    results['selection_rule_mixed'] = {d: c['reason'] for d, c in _mixed.items()}
    if _mixed:
        print("\n  ⛔ SELECTION RULES DISAGREE — the signed-rank test is held back.")
        for _d, _c in sorted(_mixed.items()):
            print(f"     {_d}: {_c['reason']}")

    for ds_name, ds_results in all_dataset_results.items():
        print(f"\n  {ds_name}:")
        ds_summary = {}

        # Exp 2: SAC improvement vs best individual rep
        exp2 = get_input(ds_results, 's02_hybrid_similarity')
        if exp2:
            best_rep_info = exp2.get('best_individual_rep', {})
            best_rep_sac = best_rep_info.get('sac_rho', float('nan'))
            best_rep_name = best_rep_info.get('name', '?')
            best_hybrid_sac = exp2.get('best_sac', float('nan'))
            delta_sac = best_hybrid_sac - best_rep_sac
            _src2 = ('s02 best_sac - best_individual_rep.sac_rho '
                     '(both scored on the whole dataset; best_sac is a max '
                     'over weight methods)')
            # Either side can be missing from the stored file, and the
            # subtraction then gives NaN. One NaN in the list makes the whole
            # series' Wilcoxon p NaN, so the dataset is recorded as absent
            # instead of entering the series.
            if not _is_number(delta_sac):
                _record_absent(
                    ds_summary, 'sac_delta', _src2,
                    f"best_sac={best_hybrid_sac!r}, "
                    f"best_individual_rep.sac_rho={best_rep_sac!r}")
                print(f"    SAC delta [{_ARM_OPTIMISTIC}]: not read "
                      f"(best_sac={best_hybrid_sac!r}, "
                      f"best_rep sac_rho={best_rep_sac!r})")
            else:
                delta_sac = float(delta_sac)
                _record(ds_summary, 'sac_delta', delta_sac, _ARM_OPTIMISTIC,
                        _src2)
                _series('SAC_delta', ds_name, delta_sac, _ARM_OPTIMISTIC, _src2)
                print(f"    SAC delta [{_ARM_OPTIMISTIC}]: {delta_sac:+.4f} "
                      f"(best_rep {best_rep_name}={best_rep_sac:.4f} -> "
                      f"hybrid={best_hybrid_sac:.4f})")
            ds_summary['best_rep'] = best_rep_name

            # Counterpart arm: s02 also stores the held-out CV SAC, winner
            # re-picked per fold, under 'fold_reselection_result'.
            _src2h = 's02 fold_reselection_result.knn.delta_mean_of_folds'
            _v2, _arm2, _why2 = _heldout_delta(
                exp2.get('fold_reselection_result'), 'knn')
            if _v2 is None:
                if exp2.get('fold_reselection_error'):
                    _why2 = (f"s02 fold_reselection_error: "
                             f"{exp2['fold_reselection_error']}")
                _record_absent(ds_summary, 'sac_delta_heldout', _src2h, _why2)
                print(f"    SAC delta [{_ARM_HELD_OUT}]: not read ({_why2})")
            else:
                _record(ds_summary, 'sac_delta_heldout', _v2, _arm2, _src2h)
                _series('SAC_delta_heldout', ds_name, _v2, _arm2, _src2h)
                print(f"    SAC delta [{_arm2}]: {_v2:+.4f}")

        # Exp 4: RMSE_cliff improvement (hybrid vs best per-rep kNN)
        # Classification datasets are silently skipped (no rmse_cliff key).
        exp4 = get_input(ds_results, 's04_activity_cliffs')
        if exp4 and 'per_rep' in exp4 and 'hybrid' in exp4:
            per_rep_rmse = [v.get('rmse_cliff', float('nan'))
                           for v in exp4['per_rep'].values()]
            valid_rmse = [r for r in per_rep_rmse if not np.isnan(r)]
            best_rep_rmse = min(valid_rmse) if valid_rmse else float('nan')
            hybrid_w = exp4['hybrid'].get('weighted', {})
            hybrid_rmse = hybrid_w.get('rmse_cliff', float('nan'))
            if not np.isnan(best_rep_rmse) and not np.isnan(hybrid_rmse) and best_rep_rmse > 0:
                reduction = (best_rep_rmse - hybrid_rmse) / best_rep_rmse * 100
                _src4 = ('s04 min over per_rep[*].rmse_cliff vs '
                         'hybrid.weighted.rmse_cliff')
                _record(ds_summary, 'cliff_rmse_reduction_pct', reduction,
                        _ARM_OPTIMISTIC, _src4)
                # The two sides of this ratio differ in TWO ways: the hybrid
                # mixes reps AND weights its distance by importance, the
                # baseline does neither. The '_vs_weighted_rep' quantity below
                # holds the like-for-like version.
                ds_summary['cliff_rmse_reduction_pct_contrast'] = (
                    'importance-weighted hybrid vs UNWEIGHTED single reps '
                    '(rep mixing and importance weighting together)')
                _series('RMSE_cliff_reduction_%', ds_name, reduction,
                        _ARM_OPTIMISTIC, _src4)
                print(f"    RMSE_cliff reduction [{_ARM_OPTIMISTIC}, "
                      f"unweighted baseline]: {reduction:.1f}% "
                      f"(best_rep={best_rep_rmse:.4f} -> hybrid={hybrid_rmse:.4f})")
            else:
                # rmse_cliff absent (e.g., classification dataset)
                cliff_n_skipped_clf += 1

            # Like-for-like baseline: s04 also scores each single rep with the
            # same importance-weighted kNN distance the hybrid arm uses, under
            # per_rep[*].rmse_cliff_weighted. Both sides then differ in rep
            # mixing only.
            _src4w = ('s04 min over per_rep[*].rmse_cliff_weighted vs '
                      'hybrid.weighted.rmse_cliff')
            _rmse_w = [v.get('rmse_cliff_weighted')
                       for v in exp4['per_rep'].values()]
            _valid_w = [r for r in _rmse_w if _is_number(r)]
            _best_rep_rmse_w = min(_valid_w) if _valid_w else float('nan')
            if (_valid_w and _is_number(hybrid_rmse) and _best_rep_rmse_w > 0):
                reduction_w = ((_best_rep_rmse_w - hybrid_rmse)
                               / _best_rep_rmse_w * 100)
                _record(ds_summary, 'cliff_rmse_reduction_pct_vs_weighted_rep',
                        reduction_w, _ARM_OPTIMISTIC, _src4w)
                ds_summary['cliff_rmse_reduction_pct_vs_weighted_rep_contrast'] = (
                    'importance-weighted hybrid vs importance-weighted single '
                    'reps (rep mixing only)')
                _series('RMSE_cliff_reduction_%_vs_weighted_rep', ds_name,
                        reduction_w, _ARM_OPTIMISTIC, _src4w)
                print(f"    RMSE_cliff reduction [{_ARM_OPTIMISTIC}, "
                      f"weighted baseline]: {reduction_w:.1f}% "
                      f"(best_rep_weighted={_best_rep_rmse_w:.4f} -> "
                      f"hybrid={hybrid_rmse:.4f})")
            else:
                _why4w = (f"{len(_valid_w)} rep(s) carry rmse_cliff_weighted, "
                          f"hybrid.weighted.rmse_cliff={hybrid_rmse!r}")
                _record_absent(ds_summary,
                               'cliff_rmse_reduction_pct_vs_weighted_rep',
                               _src4w, _why4w)
                print(f"    RMSE_cliff reduction [{_ARM_OPTIMISTIC}, weighted "
                      f"baseline]: not read ({_why4w})")

            ds_summary['n_cliff_compounds'] = exp4.get('cliff_definition', {}).get('n_cliff_compounds', 0)

        # Counterpart arm, outside the guard above: s04 stores per-fold scores
        # for both arms under 'fold_reselection' but no re-scored result, so it is
        # computed here. Which rows those fold scores cover is read from the
        # block: s04 files written since the cliff-rows change store the
        # dataset's own metric over each fold's CLIFF rows
        # (fold_score_stratum='cliff_rows_only'); older files stored the metric
        # over every molecule in the fold. The key name 'cliff_overall_...' is
        # kept so existing readers still find it; the source string says which.
        # Either way it is the dataset's metric, not rmse_cliff, so this delta
        # is a different statistic from the two reductions above.
        # (dirty-local 2026-09-28: the label said OVERALL for both.)
        if exp4:
            _stratum4 = ((exp4.get('fold_reselection') or {})
                         .get('fold_score_stratum') or 'all_rows_in_fold')
            _metric4 = ((exp4.get('fold_reselection') or {})
                        .get('metric_name') or 'the dataset metric')
            _src4h = ('s04 fold_reselection -> knn.delta_mean_of_folds '
                      f'(per-fold {_metric4} over {_stratum4}, not rmse_cliff)')
            ds_summary['cliff_overall_delta_heldout_stratum'] = _stratum4
            _hr4, _why4h = _rescore_block(exp4.get('fold_reselection'))
            if _hr4 is None:
                _record_absent(ds_summary, 'cliff_overall_delta_heldout',
                               _src4h, _why4h)
                print(f"    cliff overall delta [{_ARM_HELD_OUT}]: "
                      f"not read ({_why4h})")
            else:
                _v4, _arm4, _why4 = _heldout_delta(_hr4, 'knn')
                if _v4 is None:
                    _record_absent(ds_summary, 'cliff_overall_delta_heldout',
                                   _src4h, _why4)
                    print(f"    cliff overall delta [{_ARM_HELD_OUT}]: "
                          f"not read ({_why4})")
                else:
                    _record(ds_summary, 'cliff_overall_delta_heldout', _v4,
                            _arm4, _src4h)
                    _series('cliff_overall_delta_heldout', ds_name, _v4,
                            _arm4, _src4h)
                    print(f"    cliff overall delta [{_arm4}]: {_v4:+.4f}")

        # Exp 5: kNN improvement vs best individual rep
        exp5 = get_input(ds_results, 's05_knn_prediction')
        if exp5 and 'per_budget' in exp5:
            # Infer metric direction from per-rep baselines. All reps must
            # report the same metric -- mixing directions silently would
            # invert the delta sign for some datasets.
            per_rep = exp5.get('per_rep_baselines', {})
            _metrics_seen = {_prv.get('metric') for _prv in per_rep.values()
                             if _prv.get('metric') is not None}
            if len(_metrics_seen) > 1:
                raise ValueError(
                    f"s05 per_rep_baselines for {ds_name} mix metrics "
                    f"{_metrics_seen}; cannot aggregate kNN delta."
                )
            _exp5_metric = next(iter(_metrics_seen)) if _metrics_seen else None
            _exp5_lib = _metric_lower_is_better(_exp5_metric)
            _best = min if _exp5_lib else max

            # Collect hybrid_weighted scores across all budgets
            hw_scores = []
            for b_res in exp5['per_budget'].values():
                for k, v in b_res.items():
                    if 'hybrid_weighted' in k:
                        hw_scores.append(v['score_mean'])
            # Compare against best individual rep kNN baseline
            if hw_scores and per_rep:
                best_rep_knn = _best(v['score_mean'] for v in per_rep.values())
                best_hybrid_knn = _best(hw_scores)
                # Delta should always be positive = improvement
                if _exp5_lib:  # lower is better
                    delta_knn = best_rep_knn - best_hybrid_knn
                else:
                    delta_knn = best_hybrid_knn - best_rep_knn
                _src5 = ('s05 per_budget[*][*hybrid_weighted*].score_mean, MAX '
                         'over budgets x conditions x k, vs the best '
                         'per_rep_baselines score')
                _record(ds_summary, 'knn_delta', delta_knn, _ARM_OPTIMISTIC,
                        _src5)
                _series('kNN_delta', ds_name, delta_knn, _ARM_OPTIMISTIC, _src5)
                _label = _exp5_metric or 'score'
                print(f"    kNN delta {_label} (vs best rep) "
                      f"[{_ARM_OPTIMISTIC}]: {delta_knn:+.4f}")

        # The other s05 arms live outside the 'per_budget' guard above: a result
        # file can carry them without carrying the budget grid.
        if exp5:
            # s05's own MAX over the grid, read by name. It ranges over every
            # condition (weighted and unweighted), so it is not the same cell as
            # 'knn_delta' above, which ranges over hybrid_weighted only. Stored
            # per dataset, not aggregated: it is a second read of the same arm.
            _sum5 = exp5.get('summary', {})
            _src5p = 's05 summary.posthoc_max.delta_vs_best_rep'
            _ph = _sum5.get('posthoc_max', {})
            if _is_number(_ph.get('delta_vs_best_rep')):
                _record(ds_summary, 'knn_delta_posthoc_max',
                        float(_ph['delta_vs_best_rep']), _ARM_OPTIMISTIC, _src5p)
                ds_summary['knn_delta_posthoc_max_note'] = _ph.get('note')
                ds_summary['knn_delta_posthoc_max_cell'] = {
                    'budget': _ph.get('budget'), 'condition': _ph.get('condition'),
                    'k': _ph.get('k')}
            else:
                _record_absent(ds_summary, 'knn_delta_posthoc_max', _src5p,
                               'delta_vs_best_rep absent or NaN')

            # Counterpart arm 1: the pre-registered cell s05 fixes before the
            # grid is scored. No selection over the grid; still scored in-sample.
            _src5f = 's05 summary.fixed_ref.delta_vs_best_rep'
            _fr = _sum5.get('fixed_ref', {})
            if _is_number(_fr.get('delta_vs_best_rep')):
                _v5f = float(_fr['delta_vs_best_rep'])
                _record(ds_summary, 'knn_delta_fixed_ref', _v5f,
                        _ARM_PREREGISTERED, _src5f)
                ds_summary['knn_delta_fixed_ref_cell'] = {
                    'budget': _fr.get('budget'), 'condition': _fr.get('condition'),
                    'k': _fr.get('k')}
                _series('kNN_delta_fixed_ref', ds_name, _v5f,
                        _ARM_PREREGISTERED, _src5f)
                print(f"    kNN delta [{_ARM_PREREGISTERED}]: {_v5f:+.4f} "
                      f"(budget={_fr.get('budget')}, "
                      f"condition={_fr.get('condition')}, k={_fr.get('k')})")
            else:
                _why5f = (_sum5.get('fixed_ref_status', {}).get('reason')
                          or 'summary.fixed_ref absent')
                _record_absent(ds_summary, 'knn_delta_fixed_ref', _src5f, _why5f)
                print(f"    kNN delta [{_ARM_PREREGISTERED}]: "
                      f"not read ({_why5f})")

            # Counterpart arm 2: leave-one-fold-out re-score, winner re-picked
            # per held-out fold.
            _src5h = 's05 fold_reselection_result.knn.delta_mean_of_folds'
            _v5, _arm5, _why5 = _heldout_delta(
                exp5.get('fold_reselection_result'), 'knn')
            if _v5 is None:
                _st5 = exp5.get('fold_reselection_status', {})
                if not _st5.get('computed', True) and _st5.get('reason'):
                    _why5 = f"s05 fold_reselection_status: {_st5['reason']}"
                _record_absent(ds_summary, 'knn_delta_heldout', _src5h, _why5)
                print(f"    kNN delta [{_ARM_HELD_OUT}]: not read ({_why5})")
            else:
                _record(ds_summary, 'knn_delta_heldout', _v5, _arm5, _src5h)
                _series('kNN_delta_heldout', ds_name, _v5, _arm5, _src5h)
                print(f"    kNN delta [{_arm5}]: {_v5:+.4f}")

        # Exp 6: AD correlation (hybrid_weighted vs concat baseline).
        # Average across all k_keys instead of picking whichever happens to
        # come first in dict-insertion order (silently mixed across datasets).
        exp6 = get_input(ds_results, 's06_applicability_domain')
        if exp6 and 'distance_error_correlation' in exp6:
            per_k_deltas = []
            per_k_hw = []
            per_k_baseline = []
            # s06 stopped using 'concat' as its baseline on 2026-09-04 — the
            # arm is now the best single base representation and its name is
            # recorded at exp6['comparator_rep']. Older result files carry
            # neither that key nor a per-rep arm, so 'concat' remains the
            # fallback and a file written under either convention still reads.
            baseline_label = exp6.get('comparator_rep') or 'concat'
            _exp6_missing_concat = False
            # s06 labels how it built each correlation; passed straight through
            # so the stored delta says which construction it came from.
            _exp6_fold_handling = set()
            _exp6_error_source = set()
            for k_key, corrs in exp6['distance_error_correlation'].items():
                hw = corrs.get('hybrid_weighted')
                baseline = corrs.get(baseline_label)
                if corrs.get('fold_handling'):
                    _exp6_fold_handling.add(corrs['fold_handling'])
                if corrs.get('error_source'):
                    _exp6_error_source.add(corrs['error_source'])
                # A stored NaN passes an 'is not None' test and then poisons
                # the whole series' Wilcoxon p, so both sides are checked as
                # numbers, not for presence.
                if not _is_number(baseline):
                    _exp6_missing_concat = True
                    continue
                if _is_number(hw):
                    per_k_deltas.append(float(hw) - float(baseline))
                    per_k_hw.append(float(hw))
                    per_k_baseline.append(float(baseline))
            if _exp6_missing_concat and not per_k_deltas:
                print(f"    WARNING: s06 for {ds_name} has no "
                      f"{baseline_label!r} baseline; skipping AD delta.")
            if per_k_deltas:
                delta_ad = float(np.mean(per_k_deltas))
                hw_mean = float(np.mean(per_k_hw))
                base_mean = float(np.mean(per_k_baseline))
                _src6 = ('s06 mean over k of '
                         'distance_error_correlation.k<K>.hybrid_weighted '
                         f'minus .{baseline_label}')
                _record(ds_summary, 'ad_corr_delta', delta_ad,
                        _ARM_NO_SELECTION, _src6)
                ds_summary['ad_corr_weighted'] = hw_mean
                ds_summary['ad_corr_baseline'] = base_mean
                # Which arm the delta was taken against: the name of the single
                # base representation s06 compared to, or 'concat' on a file
                # written before s06 stopped concatenating.
                ds_summary['ad_corr_baseline_rep'] = baseline_label
                ds_summary['ad_corr_n_k'] = len(per_k_deltas)
                ds_summary['ad_corr_fold_handling'] = sorted(_exp6_fold_handling)
                ds_summary['ad_corr_error_source'] = sorted(_exp6_error_source)
                # s06 picks no winner over a grid and stores no leave-one-fold-out
                # block, so there is no second arm of this comparison to read.
                _record_absent(
                    ds_summary, 'ad_corr_delta_heldout',
                    's06 (no key)',
                    's06 selects no cell and stores no fold_reselection block; it '
                    'has no held-out counterpart to this correlation delta')
                _series('AD_corr_delta', ds_name, delta_ad, _ARM_NO_SELECTION,
                        _src6)
                print(f"    AD corr (mean over {len(per_k_deltas)} k) "
                      f"[{_ARM_NO_SELECTION}]: "
                      f"hybrid_weighted={hw_mean:.4f}, "
                      f"{baseline_label}={base_mean:.4f} "
                      f"(delta={delta_ad:+.4f})")
                print(f"      no held-out counterpart in s06 "
                      f"(fold_handling={sorted(_exp6_fold_handling) or 'unlabelled'})")

        # Exp 8: optimal dim ratio
        exp8 = get_input(ds_results, 's08_optimal_dimension')
        if exp8 and 'optimal' in exp8:
            opt = exp8['optimal']
            # s08 may write None (via .get('knn')) when no budget was found;
            # treat None and missing the same.
            knn_opt = opt.get('best_knn_budget') or 0
            rf_opt = opt.get('best_rf_budget') or 0
            if rf_opt and knn_opt and rf_opt > 0:
                ratio = knn_opt / rf_opt
                _src8 = ('s08 optimal.best_knn_budget / optimal.best_rf_budget '
                         '(each the budget with the best cross-validation mean, '
                         'picked over the same budgets it is read from)')
                _record(ds_summary, 'optimal_dim_ratio', ratio,
                        _ARM_OPTIMISTIC, _src8)
                # s08 holds out folds for the SAC arms below, but the budget
                # ratio itself is a pair of argmax picks with no held-out
                # counterpart stored.
                _record_absent(
                    ds_summary, 'optimal_dim_ratio_heldout', 's08 (no key)',
                    's08 stores no held-out re-pick of best_knn_budget or '
                    'best_rf_budget; the ratio has no held-out counterpart')
                _series('optimal_dim_ratio', ds_name, ratio, _ARM_OPTIMISTIC,
                        _src8)
                print(f"    Optimal dim ratio (kNN/RF) [{_ARM_OPTIMISTIC}]: "
                      f"{ratio:.2f}")

        # s08's held-out SAC arms, read whether or not the ratio above was
        # computed. Two arms, differing in the hybrid-side distance only:
        # 'sac_reselected' scores the hybrid with importance-weighted distances
        # against unweighted single reps; 'sac_reselected_unweighted' scores both
        # sides unweighted.
        if exp8:
            for _field, _key, _agg, _contrast in (
                    ('sac_delta_heldout_s08', 'sac_reselected',
                     'SAC_delta_heldout_s08',
                     'hybrid importance-weighted vs single reps unweighted'),
                    ('sac_delta_heldout_s08_unweighted',
                     'sac_reselected_unweighted',
                     'SAC_delta_heldout_s08_unweighted',
                     'both arms unweighted')):
                _src8h = f's08 {_key}.delta_mean_of_folds ({_contrast})'
                _v8, _arm8, _why8 = _heldout_delta(exp8.get(_key), 'sac')
                if _v8 is None:
                    _record_absent(ds_summary, _field, _src8h, _why8)
                    print(f"    SAC delta [{_ARM_HELD_OUT}, {_contrast}]: "
                          f"not read ({_why8})")
                else:
                    _record(ds_summary, _field, _v8, _arm8, _src8h)
                    ds_summary[f'{_field}_contrast'] = _contrast
                    _series(_agg, ds_name, _v8, _arm8, _src8h)
                    print(f"    SAC delta [{_arm8}, {_contrast}]: {_v8:+.4f}")

        results['per_dataset'][ds_name] = ds_summary

    # Refuse an aggregate built from nothing. A run launched with only s09 in
    # its experiment list leaves every lookup above returning None, which used
    # to produce {'per_dataset': {ds: {} ...}, 'summary': {}} and still be
    # written out as a result (see similarity_full_20260216_022715.json).
    _refusal = refusal(all_dataset_results, results['per_dataset'],
                       METRIC_FIELDS + HELDOUT_METRIC_FIELDS)
    if _refusal is not None:
        print_refusal(_refusal)
        return _refusal

    # Summary statistics. Each series is printed and stored with the arm it
    # belongs to and the input key it was read from; series named in
    # _SERIES_ORDER come first, in that order, so an optimistic quantity and its
    # counterpart sit next to each other. A series not named there is still
    # reported, after them, so a new one cannot vanish.
    print("\n  Aggregate summary:")
    _ordered = ([n for n in _SERIES_ORDER if n in series]
                + [n for n in series if n not in _SERIES_ORDER])
    for name in _ordered:
        values = series[name]['values']
        _datasets = series[name]['datasets']
        _arm = series[name]['arm']
        _source = series[name]['source']
        if values:
            arr = np.array(values)
            # NO MEAN ACROSS DATASETS. Averaging a delta over datasets that
            # differ in size, domain, task and metric produces a headline
            # number that describes none of them; the project bans it. The
            # per-dataset values are the result. The paired Wilcoxon below
            # stays: it is a sign test over the per-dataset values, not a
            # summary statistic, and the roadmap asks for it by name.
            results['summary'][name] = {
                'n_datasets': len(arr),
                'arm': _arm,
                'source': _source,
                'per_dataset': {ds: float(v) for ds, v
                                in zip(_datasets, values)},
                'values': [float(v) for v in arr],
            }
            print(f"    {name} [{_arm}] "
                  f"(per dataset, never averaged, n={len(arr)}):")
            print(f"      read from: {_source}")
            for _ds, _v in zip(_datasets, values):
                print(f"      {_ds:<28} {_v:+.4f}")

            # Wilcoxon test if enough samples. Ratios (centered at 1) must
            # be log-transformed so the test is against 0 in log-space
            # (equivalent to ratio=1); the default wilcoxon tests symmetry
            # around 0, which is meaningless for raw ratios.
            # Every path that does NOT produce a p-value records why, so an
            # absent wilcoxon_p is never mistaken for a test that ran.
            if _mixed:
                _why = (f'held back: the stored inputs of '
                        f'{", ".join(sorted(_mixed))} were written under more '
                        f'than one selection rule, so this column does not '
                        f'hold one quantity. See selection_rule_by_dataset.')
                results['summary'][name]['wilcoxon_skipped'] = _why
                print(f"      Wilcoxon {_why}")
            elif len(arr) >= MIN_DATASETS_FOR_WILCOXON:
                try:
                    if name == 'optimal_dim_ratio':
                        with np.errstate(divide='ignore', invalid='ignore'):
                            log_arr = np.log(arr[arr > 0])
                        if len(log_arr) >= MIN_DATASETS_FOR_WILCOXON:
                            stat, p = wilcoxon(log_arr)
                            results['summary'][name]['wilcoxon_p_logratio'] = float(p)
                            print(f"      Wilcoxon p (log-ratio vs 1)={p:.4f}")
                        else:
                            _why = (f'only {len(log_arr)} positive ratio(s) after '
                                    f'log transform, {MIN_DATASETS_FOR_WILCOXON} '
                                    f'required')
                            results['summary'][name]['wilcoxon_skipped'] = _why
                            print(f"      Wilcoxon skipped: {_why}")
                    else:
                        stat, p = wilcoxon(arr)
                        results['summary'][name]['wilcoxon_p'] = float(p)
                        print(f"      Wilcoxon p={p:.4f}")
                except Exception as e:
                    _why = f'wilcoxon raised {type(e).__name__}: {e}'
                    results['summary'][name]['wilcoxon_skipped'] = _why
                    print(f"      Wilcoxon skipped: {_why}")
            else:
                _why = (f'{len(arr)} dataset(s) contributed, '
                        f'{MIN_DATASETS_FOR_WILCOXON} required')
                results['summary'][name]['wilcoxon_skipped'] = _why
                print(f"      Wilcoxon skipped: {_why}")

    if cliff_n_skipped_clf:
        print(f"\n  Note: {cliff_n_skipped_clf} dataset(s) skipped in "
              f"cliff_reductions (no rmse_cliff, likely classification).")
        results['summary']['cliff_n_skipped_clf'] = cliff_n_skipped_clf

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S09: Cross-dataset summary')
    parser.add_argument('--datasets', nargs='+', default=['logd', 'caco2', 'herg_ki', 'qm9'],
                        help='Datasets to aggregate (default: logd caco2 herg_ki qm9)')
    parser.add_argument('--results-dir', type=str, default='tests/results',
                        help='Directory containing experiment result subdirs')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for summary JSON')
    args = parser.parse_args()

    print(f"Datasets: {args.datasets}")
    print(f"Results dir: {args.results_dir}")

    # Load results from each experiment for each dataset. A result directory
    # written under either naming scheme is accepted; it is stored under the
    # canonical sNN id.
    all_dataset_results = {}
    for ds in args.datasets:
        ds_results = {}
        for exp_id in EXPERIMENT_IDS:
            for dir_name in accepted_names(exp_id):
                r = load_experiment_results(args.results_dir, dir_name, ds)
                if r:
                    ds_results[exp_id] = r
                    print(f"  Loaded {dir_name}/{ds}.json")
                    break
            else:
                print(f"  Missing {exp_id}/{ds}.json "
                      f"(also tried {', '.join(accepted_names(exp_id)[1:])})")
        if ds_results:
            all_dataset_results[ds] = ds_results
        else:
            print(f"  WARNING: no results found for {ds}, skipping")

    # ── What each file was computed under ──
    # s09 puts one number per dataset in a column and runs a signed-rank test
    # down it. Every file it reads records the distance coefficient it was
    # computed under, its molecule count, its label hash and its split count,
    # and none of that was read. So a dataset re-run under cosine and a dataset
    # left on manhattan went into one column, and the summary was stamped with
    # whatever coefficient the s09 process itself happened to default to.
    #
    # Collected here and written into the result file. A disagreement is named
    # and the summary's own stamp says 'mixed' rather than asserting one.
    provenance = {}
    for ds, ds_results in all_dataset_results.items():
        for exp_id, r in ds_results.items():
            fp = (r or {}).get('input_fingerprint') or {}
            provenance.setdefault(ds, {})[exp_id] = {
                'distance_metric': (r or {}).get('distance_metric')
                                   or fp.get('distance_metric'),
                'n_samples': fp.get('n_samples'),
                'labels_hash': fp.get('labels_hash'),
                'n_splits': fp.get('n_splits'),
            }
    _coeffs = {v['distance_metric']
               for per_exp in provenance.values()
               for v in per_exp.values()}
    _named = sorted(c for c in _coeffs if c)
    _unstamped = sum(1 for per_exp in provenance.values()
                     for v in per_exp.values() if not v['distance_metric'])
    if len(_named) > 1:
        print(f"  !! the files being pooled were computed under more than one "
              f"distance coefficient: {_named}. The summary records 'mixed' "
              f"rather than claiming one of them.")
    if _unstamped:
        print(f"  !! of the files being pooled, {_unstamped} record no "
              f"distance coefficient at all — they predate the stamp. The "
              f"summary cannot say what they were computed under.")
    # Within one dataset the six files must agree about the molecules.
    for ds, per_exp in provenance.items():
        _hashes = {v['labels_hash'] for v in per_exp.values()
                   if v['labels_hash'] is not None}
        _counts = {v['n_samples'] for v in per_exp.values()
                   if v['n_samples'] is not None}
        if len(_hashes) > 1 or len(_counts) > 1:
            print(f"  !! {ds}: the files being pooled disagree about the "
                  f"molecules — {len(_counts)} different molecule counts and "
                  f"{len(_hashes)} different label hashes. They were not all "
                  f"computed on the same data.")

    if not all_dataset_results:
        print("No results found for any dataset, exiting.")
        sys.exit(1)

    results = run(all_dataset_results)
    if isinstance(results, dict):
        results['input_provenance'] = provenance
        results['input_provenance_schema'] = (
            "One entry per (dataset, experiment): the distance coefficient, "
            "molecule count, label hash and split count the file it came from "
            "records. s09 pools numbers across datasets, so these have to "
            "agree before the pooled number means anything. Written since "
            "2026-09-11; before that nothing was read and the summary was "
            "stamped with the s09 process's own default coefficient.")
        results['distance_metric_of_inputs'] = (
            _named[0] if len(_named) == 1 and not _unstamped
            else ('mixed:' + ','.join(_named) if _named else 'unknown'))
    if results.get('error'):
        # Nothing aggregated: write no result file, fail the job.
        print("\nNo result file written.")
        sys.exit(1)
    save_experiment_results(results, args.output, 's09_cross_dataset_summary')
