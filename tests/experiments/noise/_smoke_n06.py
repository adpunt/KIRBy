#!/usr/bin/env python3
"""Fast smoke test for n06 — synthetic data whose answer is known by hand.

No downloads, no torch, no server, no cluster. Every number asserted here was
worked out on paper first; a test whose expected value came out of the code it
is testing proves nothing.

What it checks:
  1. build_replicate_groups puts rows sharing a drug and a protein in one
     group, and an explicit grouping wins over the metadata.
  2. The drug-target provider takes the exact population standard deviation per
     group, counts the groups whose repeats agree exactly, and refuses below
     the floor with the count it reached.
  3. reduce_to_groups gives ONE point per group, not one per row.
  4. agreement() recovers a perfect Spearman when the scores are monotone in
     the group's spread, and a perfect AUROC when they are monotone in a
     two-curator flip.
  5. The GP cap drops whole groups, never single rows.
  6. Estimator names from n06 and from n11 resolve to the same class.
  7. run() end to end: one score per molecule from its own fold, the hybrid
     minus each of rf's top 2 single representations (base_1, base_2), a
     transfer entry that refuses with a reason when there is no stored
     ranking, per-row scores written to disk, and NO best_estimator anywhere.
  8. A dataset with no repeat structure writes a skip carrying its count,
     instead of raising or writing an empty result.
  9. With no rf ranking -- no stored search, or a similarity search with no rf
     scores -- the single-representation spaces are skipped and the reason is
     written to base_reps_missing. No fallback list (owner's decision
     2026-09-25).

Run:  python tests/experiments/noise/_smoke_n06.py
"""
import sys
import os
import json
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import n06_bindingdb_ground_truth as n06  # noqa: E402
import _ground_truth_providers as gtp  # noqa: E402

# Where run() writes during this test. Under tests/results, like every other
# output in the suite, and removed afterwards — three complexity files were
# dead for weeks because they read from a scratch directory, and nothing under
# tests/results should be left holding numbers made up by a test.
SMOKE_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results',
    '_smoke', '_n06_unit_smoke')


def test_build_replicate_groups():
    data = {
        'labels': np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0]),
        'metadata': {
            'drug_ids':    ['d1', 'd1', 'd1', 'd2', 'd2', 'd3'],
            'protein_ids': ['p1', 'p1', 'p1', 'p9', 'p9', 'p3'],
        },
    }
    g = n06.build_replicate_groups(data)
    assert g[0] == g[1] == g[2], g
    assert g[3] == g[4], g
    assert len({g[0], g[3], g[5]}) == 3, "three distinct pairs expected"

    explicit = np.array([0, 0, 1])
    d2 = {'labels': np.zeros(3), 'replicate_groups': explicit,
          'metadata': {'drug_ids': ['x', 'y', 'z'],
                       'protein_ids': ['a', 'b', 'c']}}
    assert np.array_equal(n06.build_replicate_groups(d2), explicit)
    print("PASS build_replicate_groups: groups =", g.tolist(),
          "and an explicit grouping wins")


def test_provider_arithmetic_and_floor():
    # Worked by hand:
    #   group 0: {5.0, 7.0}      mean 6.0, deviations -1 and +1, spread 1.0
    #   group 1: {2.0, 2.0, 8.0} mean 4.0, squared deviations 4, 4, 16,
    #            mean 8, spread sqrt(8) = 2.8284271...
    #   group 2: {3.0}           measured once, no spread
    #   group 3: {4.0, 4.0}      the repeats agree exactly, spread 0
    labels = np.array([5.0, 7.0, 2.0, 2.0, 8.0, 3.0, 4.0, 4.0])
    groups = np.array([0,   0,   1,   1,   1,   2,   3,   3])
    data = {'labels': labels, 'replicate_groups': groups}

    saved = gtp.MIN_GROUND_TRUTH_GROUPS
    try:
        gtp.MIN_GROUND_TRUTH_GROUPS = 2
        gt = gtp.drug_target_repeats(data, 'smoke')
    finally:
        gtp.MIN_GROUND_TRUTH_GROUPS = saved

    assert abs(gt.group_value[0] - 1.0) < 1e-12, gt.group_value
    assert abs(gt.group_value[1] - np.sqrt(8.0)) < 1e-12, gt.group_value
    assert 2 not in gt.group_value, "a molecule measured once has no spread"
    assert 3 not in gt.group_value, "repeats that agree carry no ranking"
    assert gt.n_groups == 2, gt.n_groups
    assert gt.counts['n_groups_whose_repeats_agree_exactly'] == 1
    assert gt.n_rows_covered == 5, gt.n_rows_covered

    # The floor refuses, and says what it counted.
    try:
        gtp.drug_target_repeats(data, 'smoke')
    except gtp.ProviderUnavailable as exc:
        assert exc.counts['n_groups'] == 2, exc.counts
        assert str(gtp.MIN_GROUND_TRUTH_GROUPS) in exc.reason
    else:
        raise AssertionError("the floor did not refuse two groups")

    print(f"PASS provider arithmetic: spreads {gt.group_value[0]:.6f} "
          f"(expect 1.0) and {gt.group_value[1]:.6f} "
          f"(expect {np.sqrt(8.0):.6f}); 1 group's repeats agree exactly; "
          f"the floor refuses 2 groups and reports the count")


def test_one_row_per_group():
    labels = np.array([5.0, 7.0, 2.0, 2.0, 8.0])
    groups = np.array([0, 0, 1, 1, 1])
    saved = gtp.MIN_GROUND_TRUTH_GROUPS
    try:
        gtp.MIN_GROUND_TRUTH_GROUPS = 2
        gt = gtp.drug_target_repeats(
            {'labels': labels, 'replicate_groups': groups}, 'smoke')
    finally:
        gtp.MIN_GROUND_TRUTH_GROUPS = saved

    scores = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    gids, values, means, sizes = gtp.reduce_to_groups(gt, scores)
    assert len(gids) == 2, gids
    assert list(sizes) == [2, 3], sizes
    assert abs(means[0] - 2.0) < 1e-12, means
    assert abs(means[1] - 20.0) < 1e-12, means
    print(f"PASS one row per group: 5 rows reduce to {len(gids)} points, "
          f"mean scores {means.tolist()}, rows behind them {sizes.tolist()}")


def test_agreement_spearman_and_auroc():
    # Six groups of two. Group g has labels {10g, 10g + (g+1)}, so its spread
    # is (g+1)/2 — strictly increasing in g. A score equal to the group index
    # is therefore monotone in the spread and Spearman must be exactly 1.
    labels, groups = [], []
    for g in range(6):
        labels += [10.0 * g, 10.0 * g + (g + 1)]
        groups += [g, g]
    labels = np.array(labels)
    groups = np.array(groups)
    saved = gtp.MIN_GROUND_TRUTH_GROUPS
    try:
        gtp.MIN_GROUND_TRUTH_GROUPS = 6
        gt = gtp.drug_target_repeats(
            {'labels': labels, 'replicate_groups': groups}, 'smoke')
        scores = np.array([float(g) for g in groups])
        out = gtp.agreement(gt, scores)
    finally:
        gtp.MIN_GROUND_TRUTH_GROUPS = saved
    assert out['statistic'] == 'spearman'
    assert abs(out['value'] - 1.0) < 1e-9, out
    assert out['n_replicate_groups'] == 6, out
    assert out['n_rows_behind_them'] == 12, out

    # Four published collections: six molecules, three they split over. A
    # score that is higher on exactly those three separates them perfectly,
    # AUROC 1.0.
    flags = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0}
    gt2 = gtp.GroundTruth(
        provider='cross_curator', statistic='auroc',
        units='two or more of the four collections called this molecule '
              'differently: 1 or 0',
        group_ids=np.arange(6), group_value=flags)
    scores2 = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    saved = gtp.MIN_GROUND_TRUTH_GROUPS
    try:
        gtp.MIN_GROUND_TRUTH_GROUPS = 6
        out2 = gtp.agreement(gt2, scores2)
    finally:
        gtp.MIN_GROUND_TRUTH_GROUPS = saved
    assert out2['statistic'] == 'auroc'
    assert abs(out2['value'] - 1.0) < 1e-9, out2
    assert out2['n_positive'] == 3, out2
    print(f"PASS agreement: spearman {out['value']:.6f} over "
          f"{out['n_replicate_groups']} groups (expect 1.0), auroc "
          f"{out2['value']:.6f} over {out2['n_positive']} flipped molecules "
          f"(expect 1.0)")


def test_gp_cap_drops_whole_groups():
    # 10 groups of 3 rows. A cap of 10 rows can hold 3 whole groups.
    exclusion = np.repeat(np.arange(10), 3)
    rows = np.arange(30)
    keep, dropped, n_repeat = n06._cap_whole_groups(
        rows, exclusion, 10, np.random.RandomState(0))
    kept_groups = np.unique(exclusion[keep])
    assert len(keep) == 3 * len(kept_groups), (len(keep), kept_groups)
    assert len(keep) <= 10, len(keep)
    assert n_repeat == 10, n_repeat
    assert dropped == 10 - len(kept_groups), (dropped, kept_groups)
    for g in kept_groups:
        assert (exclusion[keep] == g).sum() == 3, "a group was split"

    # The repeat groups are taken before the rows measured once. 20 groups of
    # 3 among 900 lone rows, cap 30: filling from every row on equal terms
    # would keep almost no groups.
    excl2 = np.concatenate([np.repeat(np.arange(20), 3),
                            np.arange(100, 1000)])
    keep2, dropped2, n_repeat2 = n06._cap_whole_groups(
        np.arange(len(excl2)), excl2, 30, np.random.RandomState(0))
    kept2 = [g for g in range(20) if (excl2[keep2] == g).sum() == 3]
    assert n_repeat2 == 20, n_repeat2
    assert len(kept2) == 10, (len(kept2), len(keep2))
    assert dropped2 == 10, dropped2
    assert not (excl2[keep2] >= 100).any(), \
        "a row measured once took a place a repeat group needed"
    # herg_ki's shape: every ground-truth group is ONE row. 1,412 lone rows,
    # 13 of them held out and gradable, cap 1,000: all 13 must survive.
    excl3 = np.arange(1412)
    prio3 = np.zeros(1412, dtype=bool)
    prio3[[5, 80, 300, 301, 700, 900, 1000, 1100, 1200, 1300, 1400, 1410,
           1411]] = True
    keep3, dropped3, n_first3 = n06._cap_whole_groups(
        np.arange(1412), excl3, 1000, np.random.RandomState(0),
        priority=prio3)
    assert len(keep3) == 1000, len(keep3)
    assert prio3[keep3].sum() == 13 and dropped3 == 0 and n_first3 == 13, \
        (prio3[keep3].sum(), dropped3, n_first3)
    print(f"PASS gaussian-process cap: {len(kept_groups)} whole groups kept "
          f"({len(keep)} rows, cap 10), {dropped} of {n_repeat} dropped, none "
          f"split; with 900 lone rows beside 20 groups and a cap of 30, "
          f"{len(kept2)} whole groups kept and no lone row taken")


def test_estimator_names_join():
    assert n06._canonical('distance_weighted_residual') == \
        n06._canonical('distance_weighted')
    assert n06._canonical('cross_val_residual') == n06._canonical('cross_val')
    assert n06._canonical('ensemble_disagreement') == n06._canonical('ensemble')
    assert n06._canonical('hybrid_noise') == n06._canonical('hybrid')
    stem, metric = n06._strip_metric_suffix('neighbor_residual_cosine')
    assert (stem, metric) == ('neighbor_residual', 'cosine'), (stem, metric)
    stem2, metric2 = n06._strip_metric_suffix('conformal')
    assert (stem2, metric2) == ('conformal', None)
    # n06 runs n11's list, imported rather than copied.
    assert set(n06.BASE_ESTIMATORS) <= set(n06.resolve_estimators())
    print(f"PASS estimator names: the four alias pairs resolve to one class "
          f"each, '_cosine' strips to '{stem}', and n06 runs n11's "
          f"{len(n06.BASE_ESTIMATORS)} base estimators")


class _SpreadEstimator:
    """A planted estimator whose score is monotone in the group's spread.

    The group's spread is constant inside a group, so a perfectly-ranking
    estimator must be constant inside a group too. Rows here come in
    consecutive pairs with increasing labels, so the pair's first label is
    constant within the group and increasing across groups.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def estimate(self, X, y):
        y = np.asarray(y, dtype=float)
        return np.asarray(X[:, 0], dtype=float)

    def fit(self, X, y):
        self.noise_scores_ = self.estimate(X, y)
        return self

    def get_weights(self, strength=1.0):
        w = 1.0 / (1.0 + strength * self.noise_scores_)
        return w / w.mean()


def _fake_search(rf_scores):
    """A stored search holding only single-representation scores.

    rf_scores: {rep: rf score}. None for a rep stands for a search that fitted
    no rf, the way a similarity-objective search fits no model.
    """
    return SimpleNamespace(
        metric_name='spearman', lower_is_better=False,
        baselines={r: ({'rf': v} if v is not None else {})
                   for r, v in rf_scores.items()},
        baseline_fold_scores={}, all_results=[])


def _synthetic_dataset(n_groups=60):
    """n_groups pairs, each pair a repeat measurement of one (drug, protein).

    Group g gets labels {10g, 10g + (g+1)}, so its spread is (g+1)/2 and rises
    with g. The single feature is g itself, so a planted estimator reading that
    feature ranks the groups exactly right.
    """
    labels, drug_ids, protein_ids, feat, cv_groups = [], [], [], [], []
    for g in range(n_groups):
        labels += [10.0 * g, 10.0 * g + (g + 1)]
        drug_ids += [f'd{g}', f'd{g}']
        protein_ids += [f'p{g}', f'p{g}']
        feat += [float(g), float(g)]
        # The cold-target CV grouping is NOT the replicate grouping: several
        # pairs share a protein group, the way real proteins carry many drugs.
        cv_groups += [g % 12, g % 12]
    return {
        'labels': np.array(labels),
        # feat2 is feat doubled: the same ranking of the groups, so the
        # planted estimator scores it exactly as it scores feat.
        'reps': {'feat': np.array(feat).reshape(-1, 1),
                 'feat2': 2.0 * np.array(feat).reshape(-1, 1)},
        'groups': np.array(cv_groups),
        'task_type': 'regression',
        'domain': 'molecular',
        'augmentations': None,
        'metadata': {'drug_ids': drug_ids, 'protein_ids': protein_ids},
    }


def test_run_end_to_end():
    data = _synthetic_dataset()
    original_get = n06.get_noise_estimator
    original_resolve = n06.resolve_estimators
    original_bp = n06._blueprint_for
    original_pair = n06._fold_hybrid_pair
    n06.get_noise_estimator = lambda name, **kw: _SpreadEstimator(**kw)
    n06.resolve_estimators = lambda is_clf=False: ['neighbor_residual']

    # A stubbed hybrid arm, so the subtraction between the two arms is
    # actually computed. Its feature is the comparator's feature NEGATED, so
    # the planted estimator ranks the groups backwards there: Spearman -1
    # against +1, and the difference must be exactly -2.
    class _Blueprint:
        budget = 10
        allocation_method = 'stub'
        importance_method = 'stub'

    # rf ranks feat first and feat2 second, so base_1 is feat, base_2 feat2.
    search = _fake_search({'feat': 0.60, 'feat2': 0.50})
    n06._blueprint_for = lambda *a, **kw: (
        _Blueprint(), 'none', {'tier': 'stub'}, search)
    n06._fold_hybrid_pair = lambda tr, va, reps, labels, groups, bp, aug, augs: (
        -np.asarray(reps['feat'])[tr], -np.asarray(reps['feat'])[va])

    out_dir = os.path.join(SMOKE_OUTPUT, 'end_to_end')
    try:
        res = n06.run(data, {}, out_dir, 'smoke_dataset',
                      similarity_arm=False)
    finally:
        n06.get_noise_estimator = original_get
        n06.resolve_estimators = original_resolve
        n06._blueprint_for = original_bp
        n06._fold_hybrid_pair = original_pair

    assert res.get('skipped') is not True, res.get('reason')
    assert res['metric'] and res['task_type'] == 'regression', res
    assert 'best_estimator' not in json.dumps(res), \
        "n06 must not report a maximum over the table any more"

    prov = res['providers']['drug_target_repeats']
    assert prov['used'] is True, prov
    assert prov['counts']['n_groups'] == 60, prov['counts']

    assert res['base_reps_by_model'] == {'rf': ['feat', 'feat2']}, \
        res['base_reps_by_model']
    assert res['comparator_spaces'] == ['base_1', 'base_2'], \
        res['comparator_spaces']
    comparator = 'base_1'
    block = res['ground_truth']['drug_target_repeats']
    for sp in ('base_1', 'base_2'):
        e = block['per_space'][sp]['neighbor_residual']
        assert abs(e['value'] - 1.0) < 1e-9, (sp, e)
        assert e['n_replicate_groups'] == 60, (sp, e)
        assert e['n_rows_behind_them'] == 120, (sp, e)
    rho = block['per_space'][comparator]['neighbor_residual']['value']
    n_groups = block['per_space'][comparator]['neighbor_residual'][
        'n_replicate_groups']

    # The second question is subtracted, not merely written twice, once per
    # single representation.
    minus = block['minus_comparator']
    assert sorted(minus) == ['base_1', 'base_2'], sorted(minus)
    for sp, rn in (('base_1', 'feat'), ('base_2', 'feat2')):
        assert minus[sp]['comparator_space'] == sp, minus[sp]
        assert minus[sp]['comparator_representation'] == rn, minus[sp]
        d = minus[sp]['per_space']['hybrid_predictive']['neighbor_residual']
        assert abs(d - (-2.0)) < 1e-9, (
            sp, d, "the stubbed hybrid ranks the groups backwards, so the "
                   "hybrid minus the comparator must be -1 - (+1) = -2")
        assert 'base_1' not in minus[sp]['per_space'] and \
            'base_2' not in minus[sp]['per_space'], minus[sp]
    delta = minus['base_1']['per_space']['hybrid_predictive'][
        'neighbor_residual']
    hybrid_rho = block['per_space']['hybrid_predictive'][
        'neighbor_residual']['value']
    assert abs(hybrid_rho - (-1.0)) < 1e-9, hybrid_rho

    # No stored n11 or n13 ranking exists under this temporary directory, so
    # the transfer statistic must refuse and say so.
    for label in ('n11', 'n13'):
        assert label in res['transfer'], res['transfer']
        assert 'refused' in res['transfer'][label], res['transfer'][label]

    # Per-molecule scores are on disk, under tests/results-style output, never
    # a scratch directory.
    path = res['per_row_scores_file']
    assert path.startswith(out_dir), path
    arrays = np.load(path)
    key = f'score__{comparator}__neighbor_residual'
    assert key in arrays, list(arrays)
    assert f'score__hybrid_predictive__neighbor_residual' in arrays, \
        list(arrays)
    assert np.isfinite(arrays[key]).sum() == 120, \
        "every molecule should carry a score from its own fold"
    assert 'score__base_2__neighbor_residual' in arrays, list(arrays)
    assert len(res['per_group']['ground_truth']) == 60

    print(f"PASS run(): base_1 feat and base_2 feat2 by rf's ranking, "
          f"hybrid minus each is -2; "
          f"spearman {rho:.6f} over {n_groups} groups "
          f"(expect 1.0 over 60) in {comparator}, {hybrid_rho:.6f} in the "
          f"stubbed hybrid (expect -1.0), the two subtracted give "
          f"{delta:.6f} (expect -2.0); 120 molecules scored, transfer "
          f"refused with a reason for both n11 and n13, per-row scores at "
          f"{path}")


def test_run_on_a_classification_dataset():
    """The classification branch, which no real local run reaches quickly.

    Three things are specific to it and all three are checked here: the folds
    are stratified as well as group-disjoint, only the estimators with a
    class-aware branch run, and the transfer statistic says up front that it
    cannot be computed because too few estimators are left.

    The ground truth is the two-curator flag, supplied by a stand-in for the
    published collections so the test needs no data files. A magnitude-valued
    provider is degenerate here for a separate reason: on 0/1 labels every
    disagreeing pair has the same spread, so there is nothing to rank.
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    smiles = ['CCO', 'CCC', 'c1ccccc1', 'CCN', 'CCCl', 'CCBr', 'CC=O', 'CCF',
              'CC(C)O', 'CCOC', 'CCS', 'C1CCCCC1', 'CC(=O)O', 'CCCO', 'CCCN']
    smiles = smiles * 8                      # 120 molecules, 15 distinct
    keys = [InchiToInchiKey(MolToInchi(Chem.MolFromSmiles(s))) for s in smiles]

    # Six of the fifteen distinct molecules are ones the collections split
    # over. Two collections call every one of them, so every molecule is a
    # group.
    distinct = sorted(set(keys))
    split = set(distinct[:6])
    published = {
        k: ({'Doddareddy': {'Active'}, 'Sun': {'Inactive'}} if k in split
            else {'Doddareddy': {'Active'}, 'Sun': {'Active'}})
        for k in distinct}

    n = len(smiles)
    labels = np.array([i % 2 for i in range(n)], dtype=float)
    feature = np.array([1.0 if k in split else 0.0 for k in keys])
    data = {
        'labels': labels,
        'smiles': smiles,
        'reps': {'feat': feature.reshape(-1, 1)},
        'groups': np.arange(n) % 10,
        'task_type': 'classification',
        'domain': 'molecular',
        'augmentations': None,
    }

    saved_floor = gtp.MIN_GROUND_TRUTH_GROUPS
    saved_published = gtp._published_source_calls
    saved_get = n06.get_noise_estimator
    saved_resolve = n06.resolve_estimators
    saved_bp = n06._blueprint_for
    # No hybrid recipe, but a stored search that ranks the one representation,
    # so base_1 runs and base_2 does not exist.
    n06._blueprint_for = lambda *a, **kw: (
        None, None, 'stub: no recipe', _fake_search({'feat': 0.7}))
    gtp.MIN_GROUND_TRUTH_GROUPS = 5
    gtp._published_source_calls = lambda *a, **kw: (published, {'stand_in': True})
    n06.get_noise_estimator = lambda name, **kw: _SpreadEstimator(**kw)
    n06.resolve_estimators = (
        lambda is_clf=False: [e for e in n06.BASE_ESTIMATORS
                              if e in n06.IS_CLF_AWARE_ESTIMATORS])
    out_dir = os.path.join(SMOKE_OUTPUT, 'classification')
    try:
        res = n06.run(data, {}, out_dir, 'herg', similarity_arm=False)
    finally:
        gtp.MIN_GROUND_TRUTH_GROUPS = saved_floor
        gtp._published_source_calls = saved_published
        n06.get_noise_estimator = saved_get
        n06.resolve_estimators = saved_resolve
        n06._blueprint_for = saved_bp

    assert res.get('skipped') is not True, res.get('reason')
    assert res['task_type'] == 'classification'
    assert res['comparator_spaces'] == ['base_1'], res['comparator_spaces']
    # Only the class-aware estimators, and too few of them for the headline.
    assert 'transfer_cannot_be_computed' in res, sorted(res)
    ran = {n06._canonical(e) for e in res['estimators_run']}
    assert ran <= {n06._canonical(e) for e in n06.IS_CLF_AWARE_ESTIMATORS}, ran

    block = res['ground_truth']['cross_curator']
    assert block['statistic'] == 'auroc', block['statistic']
    assert block['n_groups'] == 15, block['n_groups']
    entry = block['per_space']['base_1']
    est = sorted(entry)[0]
    # The planted estimator scores exactly the molecules the collections split
    # over, so it separates them perfectly.
    assert abs(entry[est]['value'] - 1.0) < 1e-9, entry[est]
    print(f"PASS classification: {len(ran)} class-aware estimators ran, the "
          f"two-curator AUROC is {entry[est]['value']:.6f} over "
          f"{entry[est]['n_replicate_groups']} molecules (expect 1.0 over 15), "
          f"and the transfer statistic refuses up front — "
          f"{res['transfer_cannot_be_computed'][:60]}...")


def test_run_with_no_repeat_structure():
    n = 40
    data = {
        'labels': np.arange(n, dtype=float),
        'reps': {'feat': np.arange(n, dtype=float).reshape(-1, 1)},
        'groups': np.arange(n) % 8,
        'task_type': 'regression',
        'domain': 'molecular',
        'augmentations': None,
        'metadata': {'drug_ids': [f'd{i}' for i in range(n)],
                     'protein_ids': [f'p{i}' for i in range(n)]},
    }
    original = n06.resolve_estimators
    n06.resolve_estimators = lambda is_clf=False: ['neighbor_residual']
    out_dir = os.path.join(SMOKE_OUTPUT, 'no_repeats')
    try:
        res = n06.run(data, {}, out_dir, 'smoke_no_repeats',
                      similarity_arm=False)
    finally:
        n06.resolve_estimators = original

    assert res['skipped'] is True, res
    assert 'reason' in res and res['policy'] == 'no_ground_truth_provider'
    counts = res['providers']['drug_target_repeats']['counts']
    assert counts['n_groups'] == 0, counts
    assert counts['n_pairs_measured_more_than_once'] == 0, counts
    assert res['metric'] and res['task_type'], res
    print(f"PASS no repeat structure: a skip carrying "
          f"n_pairs_measured_more_than_once="
          f"{counts['n_pairs_measured_more_than_once']}, not a crash and not "
          f"an empty result")


def test_no_rf_ranking_skips_the_single_representations():
    """No fallback list: without an rf ranking there is no base_1 or base_2.

    Two ways to have no rf ranking. The pool returns no search at all, or it
    returns a similarity-objective search whose single representations carry
    no rf score. Either way the single-representation spaces are skipped,
    base_reps_missing says why, and a run with no hybrid either is a skip with
    a reason, not a crash.
    """
    data = _synthetic_dataset()
    saved_resolve = n06.resolve_estimators
    saved_bp = n06._blueprint_for
    saved_get = n06.get_noise_estimator
    n06.resolve_estimators = lambda is_clf=False: ['neighbor_residual']
    n06.get_noise_estimator = lambda name, **kw: _SpreadEstimator(**kw)
    got = {}
    try:
        for label, search in (('no search', None),
                              ('similarity search, no rf scores',
                               _fake_search({'feat': None, 'feat2': None}))):
            n06._blueprint_for = lambda *a, _s=search, **kw: (
                None, None, 'stub: no recipe', _s)
            got[label] = n06.run(data, {}, os.path.join(SMOKE_OUTPUT, 'no_rf'),
                                 'smoke_no_rf', similarity_arm=False)
    finally:
        n06.resolve_estimators = saved_resolve
        n06._blueprint_for = saved_bp
        n06.get_noise_estimator = saved_get

    for label, res in got.items():
        assert res['base_reps_by_model'] == {}, (label,
                                                 res['base_reps_by_model'])
        assert 'rf' in res['base_reps_missing'], (label,
                                                  res['base_reps_missing'])
        assert res['comparator_spaces'] == [], (label,
                                                res['comparator_spaces'])
        assert (res['skipped'] is True
                and res['policy'] == 'no_feature_space'), (label,
                                                          res.get('policy'))
    print("PASS no rf ranking: with no search, and with a similarity search "
          "holding no rf scores, base_1 and base_2 are skipped and "
          "base_reps_missing gives the reason -- "
          + '; '.join(f"{k}: {v['base_reps_missing']['rf'][:60]}"
                      for k, v in got.items()))


def test_published_source_calls():
    """The four collections' own calls, read off Lhasa's test file.

    Skipped where the file is not on this machine. Every number here was
    counted from the file on 2026-09-12, not produced by the code under test.
    """
    try:
        published, counts = gtp._published_source_calls()
    except gtp.ProviderUnavailable as exc:
        print(f"SKIP published-source calls: {exc.reason}")
        return

    multi = {k: v for k, v in published.items() if len(v) >= 2}
    disagree = sum(1 for v in multi.values()
                   if len(set().union(*v.values())) > 1)
    # Counted off the file by hand on 2026-09-12, grouping its 5,770 rows into
    # 5,713 molecules and unioning each collection's calls across the rows of
    # one molecule. A molecule listed on two rows, one carrying Doddareddy and
    # the other Sun, was called by two collections.
    assert len(published) == 5713, len(published)
    assert len(multi) == 798, len(multi)
    assert disagree == 137, disagree
    assert set(next(iter(published.values()))) <= set(gtp.CROSS_CURATOR_SOURCES)
    print(f"PASS published-source calls: {len(published)} molecules carry a "
          f"call, {len(multi)} were called by two or more of "
          f"{', '.join(gtp.CROSS_CURATOR_SOURCES)} (expect 798), and those "
          f"collections disagree on {disagree} of them (expect 137)")


def test_stored_ranking_readers():
    """The n11 and n13 readers, on the two result files that exist on disk."""
    repo = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    n11_path = os.path.join(
        repo, 'tests/results/noise/n11_estimation_remediation_grid/'
              'buchwald_hartwig.json')
    n13_path = os.path.join(
        repo, 'tests/results/noise/n13_noise_distributions/'
              'buchwald_hartwig.json')
    if not os.path.exists(n11_path):
        print("SKIP stored-ranking readers: no n11 result on this machine")
        return
    ranking, cells, stamp, extra = n06.n11_ranking(n11_path)
    assert 'neighbor_residual' in ranking, sorted(ranking)
    assert 'distance_weighted' in ranking, sorted(ranking)
    assert 'distance_weighted_residual' not in ranking, \
        "the alias spelling must fold onto the class"
    assert all(c > 0 for c in cells.values()), cells
    r13, c13, _, x13 = n06.n13_ranking(n13_path)
    # The stored n13 file predates the 2026-06 fix that stopped writing a cell
    # for a model whose fit ignores sample weights; such a cell records the
    # noisy score unchanged, so the reader drops it and counts it.
    assert x13['dropped']['cells_where_the_cleanup_changed_nothing'] == 90, \
        x13['dropped']
    assert x13['dropped']['cells_whose_noise_did_not_hurt_the_model'] == 12, \
        x13['dropped']
    assert extra['n_feature_spaces_pooled'] == 9, extra
    assert 'local_variance_ratio' in r13, sorted(r13)
    print(f"PASS stored-ranking readers: n11 gives {len(ranking)} estimators "
          f"from {sum(cells.values())} cells over "
          f"{extra['n_feature_spaces_pooled']} feature spaces (written "
          f"{stamp}); n13 gives {len(r13)} estimators from "
          f"{sum(c13.values())} cells after dropping 90 where the cleanup "
          f"changed nothing and 12 where the noise did not hurt the model — "
          f"{len(r13)} is below the {n06.MIN_ESTIMATORS_FOR_TRANSFER} a rank "
          f"correlation needs")


if __name__ == '__main__':
    shutil.rmtree(SMOKE_OUTPUT, ignore_errors=True)
    test_build_replicate_groups()
    test_provider_arithmetic_and_floor()
    test_one_row_per_group()
    test_agreement_spearman_and_auroc()
    test_gp_cap_drops_whole_groups()
    test_estimator_names_join()
    test_published_source_calls()
    test_stored_ranking_readers()
    test_run_end_to_end()
    test_run_on_a_classification_dataset()
    test_run_with_no_repeat_structure()
    test_no_rf_ranking_skips_the_single_representations()
    shutil.rmtree(SMOKE_OUTPUT, ignore_errors=True)
    print("\nALL N06 SMOKE TESTS PASSED")
