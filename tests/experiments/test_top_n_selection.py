"""Tests for the shared top-N selectors in tests/experiments/_shared.py.

Every SearchResult here is built by hand. Nothing loads a dataset and nothing
fits a model: the point of these four selectors is that they cost 0 new model
fits, so a test that fitted anything would not be testing them.

Run: python -m pytest tests/experiments/test_top_n_selection.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'src'))

from kirby.find_best_hybrid import SearchResult, Blueprint, PRUNED_FOLD0_SUBSET
from _shared import (DOWNSTREAM_MODELS, top_base_reps, top_hybrids,
                     rank_downstream_models_by_base_reps,
                     select_model_rep_pairs, selection_diff)


# =============================================================================
# Synthetic stored results. Scores are made up; only their ORDER is under test.
# =============================================================================

def _hybrid_row(model, score, *, alloc='proportional', imp='lightgbm_gain',
                budget=500, ev='match', step=50, rep_count=3, tune=False,
                rep_subset=None, n_folds=5, fold_scores=None, **extra):
    """One all_results-shaped row (find_best_hybrid.py:5144-5177)."""
    row = {
        'blueprint': Blueprint(reps=['a', 'b'], allocation_method=alloc,
                               importance_method=imp, internal_evaluator=ev,
                               budget=budget, step_size=step,
                               rep_count=rep_count),
        'model': model,
        'score_mean': score,
        'score_std': 0.01,
        'fold_scores': (fold_scores if fold_scores is not None
                        else [score] * n_folds),
        'n_folds_scored': n_folds,
        'n_folds_asked': n_folds,
        'metrics': {},
        'n_features': budget,
        'time_seconds': 1.0,
        'allocation_method': alloc,
        'importance_method': imp,
        'internal_evaluator': ev,
        'budget': budget,
        'step_size': step,
        'had_imputation': False,
        'tune': tune,
        'rep_count': rep_count,
    }
    if rep_subset is not None:
        row['rep_subset'] = rep_subset
    row.update(extra)
    return row


def make_rmse_search():
    """5 representations, 5 downstream models, metric rmse (lower is better).

    svm is absent from every baselines[rep] on purpose: that is what a model
    over its max-samples limit or in a skip policy looks like
    (find_best_hybrid.py:2969-2978).
    """
    baselines = {
        # rep       rf      xgboost  dnn     lightgbm
        'ecfp4':    {'rf': 0.60, 'xgboost': 0.70, 'dnn': 0.80, 'lightgbm': 0.55,
                     'dimension': 2048},
        'mordred':  {'rf': 0.62, 'xgboost': 0.72, 'dnn': 0.82, 'lightgbm': 0.58,
                     'dimension': 1613},
        'maccs':    {'rf': 0.64, 'xgboost': 0.74, 'dnn': 0.84, 'lightgbm': 0.61,
                     'dimension': 167},
        'rdkit':    {'rf': 0.66, 'xgboost': 0.76, 'dnn': 0.86, 'lightgbm': 0.64,
                     'dimension': 200},
        # a present-but-NaN score for rf, and no key at all for lightgbm
        'esm3':     {'rf': float('nan'), 'xgboost': 0.90, 'dnn': 0.99,
                     'dimension': 5120, 'dimension_scored': 500},
        '_secondary': {'rf': 0.01},
    }
    all_results = [
        _hybrid_row('rf', 0.59, alloc='proportional'),
        _hybrid_row('rf', 0.57, alloc='weighted_global'),
        _hybrid_row('rf', 0.61, alloc='equal'),
        _hybrid_row('rf', float('nan'), alloc='fixed'),
        _hybrid_row('lightgbm', 0.54, alloc='proportional'),
        _hybrid_row('lightgbm', 0.53, alloc='weighted_global'),
        _hybrid_row('xgboost', 0.69, alloc='proportional'),
        _hybrid_row('xgboost', 0.68, alloc='equal'),
        _hybrid_row('dnn', 0.79, alloc='proportional'),
        _hybrid_row('dnn', 0.78, alloc='equal'),
        _hybrid_row('svm', 0.90, alloc='proportional'),
        _hybrid_row('svm', 0.91, alloc='equal'),
    ]
    return SearchResult(
        best_by_model={}, best_by_dimension={}, performance_matrix={},
        baselines=baselines, all_results=all_results,
        metric_name='rmse', lower_is_better=True, dataset_name='synthetic_rmse')


def make_spearman_search():
    """Same representations, metric spearman (higher is better)."""
    baselines = {
        'ecfp4':   {'rf': 0.60, 'dimension': 2048},
        'mordred': {'rf': 0.62, 'dimension': 1613},
        'maccs':   {'rf': 0.64, 'dimension': 167},
        'rdkit':   {'rf': 0.66, 'dimension': 200},
    }
    all_results = [
        _hybrid_row('rf', 0.70, alloc='proportional'),
        _hybrid_row('rf', 0.75, alloc='weighted_global'),
        _hybrid_row('rf', 0.65, alloc='equal'),
    ]
    return SearchResult(
        best_by_model={}, best_by_dimension={}, performance_matrix={},
        baselines=baselines, all_results=all_results,
        metric_name='spearman', lower_is_better=False,
        dataset_name='synthetic_spearman')


# =============================================================================
# Direction
# =============================================================================

def test_lower_is_better_ranks_smallest_first():
    s = make_rmse_search()
    sel = top_base_reps(s, 'rf', 2)
    assert [e['rep'] for e in sel['selected']] == ['ecfp4', 'mordred']
    assert sel['direction'] == {'lower_is_better': True,
                                'source': 'metric_name',
                                'disagrees_with_stored_flag': False}
    assert sel['complete'] is True


def test_higher_is_better_ranks_largest_first():
    s = make_spearman_search()
    sel = top_base_reps(s, 'rf', 2)
    assert [e['rep'] for e in sel['selected']] == ['rdkit', 'maccs']
    assert sel['direction']['lower_is_better'] is False


def test_hybrids_follow_the_stored_flag_not_the_metric_name():
    """A blend search: metric_name stays 'rmse', lower_is_better is flipped to
    False at find_best_hybrid.py:8726 because the winner is scored on a
    correlation. The hybrid scores in all_results are then higher-is-better.
    """
    s = make_rmse_search()
    s.lower_is_better = False              # what :8726 does under a blend
    hyb = top_hybrids(s, 'rf', 2)
    # Highest first, because the stored flag says so.
    assert [e['score_mean'] for e in hyb['selected']] == [0.61, 0.59]
    assert hyb['direction']['source'] == 'stored_flag'
    assert hyb['direction']['disagrees_with_metric_name'] is True
    # The base half still reads rmse from the metric name, so it disagrees.
    base = top_base_reps(s, 'rf', 2)
    assert [e['rep'] for e in base['selected']] == ['ecfp4', 'mordred']
    assert base['direction']['disagrees_with_stored_flag'] is True


# =============================================================================
# Missing and NaN
# =============================================================================

def test_nan_and_missing_never_reach_the_top_n():
    s = make_rmse_search()
    # esm3 has a NaN under rf; it must not be selected and must be counted.
    sel = top_base_reps(s, 'rf', 4)
    assert 'esm3' not in [e['rep'] for e in sel['selected']]
    assert sel['dropped']['nan'] == ['esm3']
    assert sel['dropped']['underscore_key'] == ['_secondary']
    assert sel['n_reps_scored_for_model'] == 4
    # esm3 has no lightgbm key at all: that is a different fact from NaN.
    sel_l = top_base_reps(s, 'lightgbm', 4)
    assert sel_l['dropped']['no_score_for_model'] == ['esm3']
    assert sel_l['dropped']['nan'] == []


def test_feature_counts_cannot_be_ranked_as_scores():
    s = make_rmse_search()
    with pytest.raises(ValueError, match='not a downstream model'):
        top_base_reps(s, 'dimension', 2)


def test_dimension_scored_is_reported_on_the_entry():
    s = make_rmse_search()
    sel = top_base_reps(s, 'xgboost', 5, require_full=False)
    by_rep = {e['rep']: e for e in sel['selected']}
    assert by_rep['ecfp4']['dimension'] == 2048
    assert by_rep['ecfp4']['scored_at_full_width'] is True
    assert by_rep['esm3']['dimension_scored'] == 500
    assert by_rep['esm3']['scored_at_full_width'] is False


def test_nan_hybrid_row_is_dropped_and_counted():
    s = make_rmse_search()
    hyb = top_hybrids(s, 'rf', 3)
    assert [e['score_mean'] for e in hyb['selected']] == [0.57, 0.59, 0.61]
    assert hyb['ranked_over']['n_rows_dropped_nan'] == 1


def test_pruned_fold0_rows_are_dropped():
    s = make_rmse_search()
    s.all_results.append(_hybrid_row('rf', 0.01, alloc='cut_after_fold0',
                                     rep_subset=PRUNED_FOLD0_SUBSET + 'abc',
                                     n_folds=1))
    hyb = top_hybrids(s, 'rf', 2)
    assert 'cut_after_fold0' not in [e['allocation_method']
                                     for e in hyb['selected']]
    assert hyb['ranked_over']['n_rows_dropped_pruned_fold0'] == 1


# =============================================================================
# Shortfall is reported, never silently truncated
# =============================================================================

def test_shortfall_raises_by_default():
    s = make_rmse_search()
    # svm has no key under any representation: 0 of 5 representations scored.
    with pytest.raises(ValueError) as e:
        top_base_reps(s, 'svm', 2)
    assert 'asked for 2 base representations' in str(e.value)
    assert 'downstream model svm' in str(e.value)


def test_shortfall_reported_when_require_full_false():
    s = make_rmse_search()
    sel = top_base_reps(s, 'svm', 2, require_full=False)
    assert sel['n_returned'] == 0
    assert sel['complete'] is False
    assert sel['n_reps_scored_for_model'] == 0
    assert 'svm' in sel['shortfall_reason']


def test_hybrid_shortfall_names_both_counts_and_never_falls_back():
    s = make_rmse_search()
    s.all_results = [r for r in s.all_results if r['model'] != 'rf'
                     or r['allocation_method'] == 'proportional']
    hyb = top_hybrids(s, 'rf', 2, require_full=False)
    assert hyb['n_returned'] == 1
    assert hyb['ranked_over']['n_rows_for_model'] == 1
    assert hyb['ranked_over']['n_rows_other_model'] == 8
    # Not one entry belongs to another downstream model.
    assert all(e['model'] == 'rf' for e in hyb['selected'])


def test_context_string_reaches_the_error():
    s = make_rmse_search()
    with pytest.raises(ValueError, match=r'herg_ki / c02'):
        top_base_reps(s, 'svm', 2, context='herg_ki / c02')


# =============================================================================
# Ties
# =============================================================================

def test_ties_break_by_name_and_are_reported():
    s = make_rmse_search()
    s.baselines['mordred']['rf'] = 0.60     # exact tie with ecfp4
    sel = top_base_reps(s, 'rf', 1)
    assert [e['rep'] for e in sel['selected']] == ['ecfp4']
    assert sel['selected'][0]['tied_with'] == ['mordred']
    assert sel['tie_at_cut'] == {'score': 0.60, 'kept': ['ecfp4'],
                                 'dropped': ['mordred']}


def test_tie_order_does_not_depend_on_dict_insertion_order():
    """Same stored numbers, representations inserted in the opposite order."""
    s1 = make_rmse_search()
    s1.baselines['mordred']['rf'] = 0.60
    s2 = make_rmse_search()
    s2.baselines['mordred']['rf'] = 0.60
    s2.baselines = dict(reversed(list(s2.baselines.items())))
    a = [e['rep'] for e in top_base_reps(s1, 'rf', 3)['selected']]
    b = [e['rep'] for e in top_base_reps(s2, 'rf', 3)['selected']]
    assert a == b == ['ecfp4', 'mordred', 'maccs']


def test_same_rows_give_the_same_order_twice():
    s = make_rmse_search()
    first = [e['label'] for e in top_hybrids(s, 'rf', 3)['selected']]
    second = [e['label'] for e in top_hybrids(s, 'rf', 3)['selected']]
    assert first == second


def test_duplicate_identities_collapse_before_the_cut():
    """A refinement row and a checkpoint-reload row for the same configuration
    must not fill both hybrid slots. Two hybrids means two configurations."""
    s = make_rmse_search()
    s.all_results = [
        _hybrid_row('rf', 0.50, alloc='proportional'),
        _hybrid_row('rf', 0.51, alloc='proportional'),   # same identity
        _hybrid_row('rf', 0.60, alloc='equal'),
    ]
    hyb = top_hybrids(s, 'rf', 2)
    assert hyb['ranked_over']['n_duplicate_identities_collapsed'] == 1
    assert [e['allocation_method'] for e in hyb['selected']] == \
        ['proportional', 'equal']
    assert hyb['selected'][0]['score_mean'] == 0.50     # the better of the two


def test_tune_is_part_of_the_hybrid_identity():
    s = make_rmse_search()
    s.all_results = [
        _hybrid_row('rf', 0.50, alloc='proportional', tune=False),
        _hybrid_row('rf', 0.52, alloc='proportional', tune=True),
    ]
    hyb = top_hybrids(s, 'rf', 2)
    assert hyb['ranked_over']['n_duplicate_identities_collapsed'] == 0
    assert [e['tune'] for e in hyb['selected']] == [False, True]


def test_mixed_tune_provenance_raises():
    """_db_row_to_result does not restore 'tune', so a database-read row and a
    freshly computed row for the same configuration would key differently."""
    s = make_rmse_search()
    fresh = _hybrid_row('rf', 0.50, alloc='proportional', tune=False)
    from_db = _hybrid_row('rf', 0.52, alloc='proportional')
    del from_db['tune']
    s.all_results = [fresh, from_db]
    with pytest.raises(ValueError, match="mixes rows that carry 'tune'"):
        top_hybrids(s, 'rf', 2)


def test_all_database_rows_report_rather_than_raise():
    s = make_rmse_search()
    rows = []
    for alloc, sc in (('proportional', 0.50), ('equal', 0.60)):
        r = _hybrid_row('rf', sc, alloc=alloc)
        del r['tune']
        del r['rep_count']       # also absent from a database-read row
        rows.append(r)
    s.all_results = rows
    hyb = top_hybrids(s, 'rf', 2)
    assert hyb['ranked_over']['n_rows_missing_tune'] == 2
    assert any('result database' in n for n in hyb['notes'])
    # rep_count recovered from the pickled Blueprint, and the source said.
    assert [e['rep_count_source'] for e in hyb['selected']] == \
        ['blueprint', 'blueprint']
    assert [e['rep_count'] for e in hyb['selected']] == [3, 3]


# =============================================================================
# The owner's grid
# =============================================================================

def test_grid_is_at_most_20_pairs():
    s = make_rmse_search()
    grid = select_model_rep_pairs(s, models=['rf', 'xgboost', 'dnn', 'lightgbm'],
                                  require_full=False)
    assert grid['n_pairs_possible'] == 16
    assert grid['n_pairs'] == 16
    assert grid['complete'] is True
    # 5 downstream models * (2 + 2) = 20 is the shape with the full list; svm
    # has no scores here, so asking for it is short and says so.
    full = select_model_rep_pairs(s, require_full=False)
    assert full['n_pairs_possible'] == 20
    assert full['n_pairs'] == 18          # svm contributes 0 base + 2 hybrids
    assert full['n_missing'] == 2
    assert any('downstream model svm' in t for t in full['shortfalls'])


def test_grid_collapses_a_representation_two_models_both_picked():
    s = make_rmse_search()
    grid = select_model_rep_pairs(s, models=['rf', 'lightgbm'])
    assert grid['n_pairs'] == 8
    # ecfp4 is top for rf and for lightgbm, so 8 rows are fewer than 8 units.
    assert grid['n_distinct_units'] < grid['n_pairs']
    collapsed_labels = {c['label'] for c in grid['collapsed']}
    assert 'ecfp4' in collapsed_labels
    chosen_by = {c['label']: [m for m, _k, _r in c['chosen_by']]
                 for c in grid['collapsed']}
    assert sorted(chosen_by['ecfp4']) == ['lightgbm', 'rf']


def test_grid_raises_when_a_row_is_missing_by_default():
    s = make_rmse_search()
    with pytest.raises(ValueError, match='model-and-representation pairs'):
        select_model_rep_pairs(s)          # svm is short


# =============================================================================
# The second selector: top downstream models
# =============================================================================

def test_downstream_models_ranked_by_mean_of_top_3():
    s = make_rmse_search()
    keep = rank_downstream_models_by_base_reps(s, n_models=2, top_k=3)
    names = [e['downstream_model'] for e in keep['selected']]
    assert names == ['lightgbm', 'rf']
    # lightgbm's top 3 rmse are 0.55, 0.58, 0.61 -> mean 0.58.
    assert keep['selected'][0]['mean_score'] == pytest.approx(0.58)
    # rf's top 3 are 0.60, 0.62, 0.64 -> 0.62, NOT the mean of all four (0.63).
    assert keep['selected'][1]['mean_score'] == pytest.approx(0.62)
    assert all(e['n_reps_used'] == 3 for e in keep['selected'])
    # The discarded means are on the page, with their size.
    assert [e['downstream_model'] for e in keep['not_kept']] == \
        ['xgboost', 'dnn']
    # svm never ran: it is ineligible, not ranked last.
    assert [e['downstream_model'] for e in keep['ineligible']] == ['svm']
    assert keep['ineligible'][0]['n_reps_scored'] == 0


def test_the_mean_is_over_top_k_not_over_every_representation():
    s = make_rmse_search()
    k3 = rank_downstream_models_by_base_reps(s, n_models=1, top_k=3)
    k4 = rank_downstream_models_by_base_reps(s, n_models=1, top_k=4)
    rf3 = k3['per_model']['rf']
    assert len(rf3['selected']) == 3
    assert k4['per_model']['rf']['selected'][3]['rep'] == 'rdkit'
    # rf: mean of 3 is 0.62, mean of 4 is 0.63. Different quantities.
    rf_mean_3 = np.mean([e['score'] for e in rf3['selected']])
    rf_mean_4 = np.mean([e['score']
                         for e in k4['per_model']['rf']['selected']])
    assert rf_mean_3 == pytest.approx(0.62)
    assert rf_mean_4 == pytest.approx(0.63)


def test_top_k_1_reproduces_n07s_rule_and_differs_from_the_owners():
    """n07_remediation.py:487-488 ranks a downstream model by its single best
    representation. Built so the two rules name different models."""
    baselines = {
        'a': {'rf': 0.10, 'lightgbm': 0.20},
        'b': {'rf': 0.90, 'lightgbm': 0.21},
        'c': {'rf': 0.91, 'lightgbm': 0.22},
    }
    s = SearchResult(best_by_model={}, best_by_dimension={},
                     performance_matrix={}, baselines=baselines,
                     all_results=[], metric_name='rmse', lower_is_better=True,
                     dataset_name='synthetic_split')
    top1 = rank_downstream_models_by_base_reps(s, n_models=1, top_k=1,
                                               min_reps=1)
    top3 = rank_downstream_models_by_base_reps(s, n_models=1, top_k=3)
    assert top1['selected'][0]['downstream_model'] == 'rf'        # 0.10 wins
    assert top3['selected'][0]['downstream_model'] == 'lightgbm'  # 0.21 mean


def test_geometry_style_identical_lists_are_flagged():
    """Under a geometry objective one score per representation is written under
    every downstream model key (find_best_hybrid.py:5465-5489)."""
    baselines = {'a': {m: 0.5 for m in DOWNSTREAM_MODELS},
                 'b': {m: 0.6 for m in DOWNSTREAM_MODELS},
                 'c': {m: 0.7 for m in DOWNSTREAM_MODELS}}
    s = SearchResult(best_by_model={}, best_by_dimension={},
                     performance_matrix={}, baselines=baselines,
                     all_results=[], metric_name='rmse', lower_is_better=True)
    keep = rank_downstream_models_by_base_reps(s, n_models=2)
    assert keep['all_models_identical'] is True


# =============================================================================
# other_folds rule, max_features bar, diff
# =============================================================================

def test_other_folds_rule_excludes_the_held_out_fold():
    s = make_rmse_search()
    s.baseline_fold_scores = {
        'ecfp4':   {'rf': [0.90, 0.10, 0.10, 0.10, 0.10]},
        'mordred': {'rf': [0.10, 0.50, 0.50, 0.50, 0.50]},
    }
    # The whole-dataset rule reads baselines[rep]['rf']: ecfp4 0.60,
    # mordred 0.62 -- NOT the fold scores above. ecfp4 first.
    whole = top_base_reps(s, 'rf', 1, require_full=False)
    assert whole['selected'][0]['rep'] == 'ecfp4'
    # The other-folds rule means the fold scores, fold 0 left out:
    # ecfp4 0.10, mordred 0.50 -- ecfp4 first, on different numbers.
    other = top_base_reps(s, 'rf', 2, rule='other_folds', held_out_fold=0)
    assert [e['rep'] for e in other['selected']] == ['ecfp4', 'mordred']
    assert other['selected'][0]['score'] == pytest.approx(0.10)
    assert other['selected'][0]['n_folds_used'] == 4
    assert other['score_basis'] == 'other_folds_mean(held_out=0)'
    assert whole['score_basis'] == 'whole_dataset_mean'


def test_other_folds_needs_the_held_out_fold():
    s = make_rmse_search()
    with pytest.raises(ValueError, match='held_out_fold'):
        top_base_reps(s, 'rf', 2, rule='other_folds')


def test_max_features_bars_a_wide_representation_but_reports_it():
    s = make_rmse_search()
    sel = top_base_reps(s, 'xgboost', 2, max_features=1000)
    assert [e['rep'] for e in sel['selected']] == ['maccs', 'rdkit']
    assert ('ecfp4', 2048) in sel['dropped']['too_wide']
    assert ('esm3', 5120) in sel['dropped']['too_wide']
    # It was still measured: it counts as scored.
    assert sel['n_reps_scored_for_model'] == 5


def test_only_filter_keeps_the_ranking_to_named_representations():
    s = make_rmse_search()
    sel = top_base_reps(s, 'rf', 2, only=['maccs', 'rdkit'])
    assert [e['rep'] for e in sel['selected']] == ['maccs', 'rdkit']
    assert sorted(sel['dropped']['not_in_only']) == ['ecfp4', 'mordred']


def test_selection_diff_names_what_moved():
    s = make_rmse_search()
    sel = top_base_reps(s, 'rf', 2)
    line = selection_diff(sel, ['mordred', 'maccs'])
    assert 'ecfp4' in line and 'maccs' in line
    same = selection_diff(sel, ['ecfp4', 'mordred'])
    assert same.startswith('same 2 selected')


def test_budget_filter_on_hybrids():
    s = make_rmse_search()
    s.all_results = [
        _hybrid_row('rf', 0.50, alloc='proportional', budget=200),
        _hybrid_row('rf', 0.60, alloc='proportional', budget=500),
        _hybrid_row('rf', 0.61, alloc='equal', budget=500),
    ]
    hyb = top_hybrids(s, 'rf', 2, budget=500)
    assert [e['budget'] for e in hyb['selected']] == [500, 500]
    assert hyb['ranked_over']['n_rows_dropped_wrong_budget'] == 1


def test_require_n_folds_drops_a_shorter_mean():
    s = make_rmse_search()
    s.all_results = [
        _hybrid_row('rf', 0.50, alloc='proportional', n_folds=3),
        _hybrid_row('rf', 0.60, alloc='equal', n_folds=5),
        _hybrid_row('rf', 0.61, alloc='weighted_global', n_folds=5),
    ]
    hyb = top_hybrids(s, 'rf', 2, require_n_folds=5)
    assert [e['allocation_method'] for e in hyb['selected']] == \
        ['equal', 'weighted_global']
    assert hyb['ranked_over']['n_rows_dropped_wrong_fold_count'] == 1
