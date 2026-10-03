#!/usr/bin/env python3
"""Smoke for n14 / n15 / n16 — the three noise experiments with no check.

WHY THIS FILE EXISTS. Every other noise experiment has a smoke. n14, n15 and
n16 had none, so nothing caught a change to the quantity each one reports.

WHAT EACH ONE REPORTS, and so what is checked here.

  n16  how much the top of a feature-importance ranking moves as the labels get
       noisier. Its `_turnover` carries a defect its own docstring records: the
       plain `top_k` set reads as "nothing moved" whenever the scorer left more
       features at exactly zero than the top set has room for, because
       np.argsort orders the tied zeros by index identically on both sides.
       `top_k_effective` caps k at the number of features the reference side
       actually scored. Test 1 reproduces the buchwald_hartwig shape from that
       docstring — 39 of 167 features scored — and asserts the two figures
       disagree, which is the whole reason the second one exists.

  n15  whether searching on noisy labels picks a different recipe than
       searching on clean ones. Its `_agreement` divides by the UNION of the
       two representation sets, so an arm that selected a superset does not
       score 1.0.

  n14  the cliff graph. `_cliff_mask_from_pairs` and
       `_cliff_mask_from_similarity` build the adjacency two ways and
       `_geometry` reads it. Both thresholds are STRICT: a gap exactly equal to
       the threshold is not a cliff.

These are CHEAP by design: pure functions on in-memory arrays. No
find_best_hybrid, no result store, no server data, no foundation-model
representations. Nothing here needs a rebuilt result database, so this smoke
runs while the stores are stale.

Run:  python tests/experiments/noise/_smoke_n14_n15_n16.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# 2026-09-25: the noise experiments sweep every similarity setting by default
# (KIRBY_SIMILARITY_SETTINGS unset = all 58, plan section 1). This smoke checks
# what each run() computed before that axis existed, so it runs today's
# setting alone, and it writes its caches under a temporary directory, never
# the real KIRBY_CACHE_DIR.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
import tempfile as _tempfile
os.environ['KIRBY_CACHE_DIR'] = _tempfile.mkdtemp(prefix='kirby_smoke_cache_')

import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np

import experiments.noise.n14_activity_cliff_landscape as n14
import experiments.noise.n15_clean_vs_noisy_hybrid as n15
import experiments.noise.n16_importance_under_noise as n16

_FAILS = []


def _check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ''))
    if not cond:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# n16 — the turnover measure
# ---------------------------------------------------------------------------

def test_n16_turnover_effective_sees_what_plain_top_k_misses():
    """THE HEADLINE. 167 features wide, 39 of them scored, the rest exactly zero.

    The shape is the one n16's own docstring measured on buchwald_hartwig:
    lightgbm_gain scores 39 of diff_maccs's 167 features. A top-50 set is then
    those 39 plus 11 features sitting at exactly zero, which np.argsort orders
    by index — so BOTH sides carry the same 11, and they count as retained
    however much the scoring moved. The plain figure is inflated by them.

    Here the clean side scores features 0 to 38 and the noisy side scores 10 to
    48, so 29 scored features are shared. The effective figure divides that by
    the 39 the reference side scored. The plain figure adds the 11 shared zeros
    to both the top and the bottom and comes out higher.
    """
    width, k, n_scored = 167, 50, 39
    a = np.zeros(width)
    a[:n_scored] = np.linspace(1.0, 2.0, n_scored)
    b = np.zeros(width)
    b[10:10 + n_scored] = np.linspace(1.0, 2.0, n_scored)

    out = n16._turnover(a, b, k)
    _check('n16 turnover reports both top_k sizes',
           {'top_k', 'top_k_effective', 'n_scored_reference'} <= set(out))
    _check('n16 top_k is the size asked for, capped at the width',
           out['top_k'] == k, f"got {out['top_k']}")
    _check('n16 top_k_effective caps at the features the reference side scored',
           out['top_k_effective'] == n_scored, f"got {out['top_k_effective']}")
    _check('n16 n_scored_reference counts the reference side',
           out['n_scored_reference'] == n_scored,
           f"got {out['n_scored_reference']}")
    _check('n16 the effective retained share counts only scored features',
           abs(out['retained_fraction_effective'] - 29.0 / 39.0) < 1e-12,
           f"got {out['retained_fraction_effective']:.4f}, "
           f"expected {29.0 / 39.0:.4f}")
    _check('n16 the plain retained share is inflated by the zeros both sides carry',
           out['retained_fraction'] > out['retained_fraction_effective'],
           f"plain {out['retained_fraction']:.4f} against effective "
           f"{out['retained_fraction_effective']:.4f}")


def test_n16_turnover_agrees_when_every_feature_is_scored():
    """Where the scorer touches everything the two figures are the same."""
    rng = np.random.RandomState(1)
    a = rng.rand(40) + 0.5          # no zeros
    b = rng.rand(40) + 0.5
    out = n16._turnover(a, b, 10)
    _check('n16 both top_k sizes equal when nothing is left at zero',
           out['top_k'] == out['top_k_effective'] == 10,
           f"{out['top_k']} vs {out['top_k_effective']}")
    _check('n16 both retained fractions equal when nothing is left at zero',
           out['retained_fraction'] == out['retained_fraction_effective'],
           f"{out['retained_fraction']:.4f} vs "
           f"{out['retained_fraction_effective']:.4f}")


def test_n16_turnover_counts_and_caps():
    """n_left and n_entered are the two directions, and k cannot exceed width."""
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # exactly reversed
    out = n16._turnover(a, b, 2)
    # a's top 2 are features 0 and 1; b's are 3 and 4. Disjoint.
    _check('n16 n_left counts features that fell out of the top set',
           out['n_left'] == 2, f"got {out['n_left']}")
    _check('n16 n_entered counts features that arrived in it',
           out['n_entered'] == 2, f"got {out['n_entered']}")
    _check('n16 retained_fraction is the shared count over the top set size',
           out['retained_fraction'] == 0.0,
           f"got {out['retained_fraction']}")
    wide = n16._turnover(a, b, 99)
    _check('n16 k above the width caps to the width',
           wide['top_k'] == 5, f"got {wide['top_k']}")


# ---------------------------------------------------------------------------
# n15 — recipe agreement between the clean-searched and noise-searched arms
# ---------------------------------------------------------------------------

class _BP:
    """Enough of a blueprint for _bp_fingerprint to read."""
    def __init__(self, reps, allocation_method='performance_weighted',
                 importance_method='lightgbm_gain', budget=50,
                 internal_evaluator='match'):
        self.reps = reps
        self.allocation_method = allocation_method
        self.importance_method = importance_method
        self.budget = budget
        self.internal_evaluator = internal_evaluator


def test_n15_agreement_identical_and_disjoint():
    fp = n15._bp_fingerprint(_BP(['ecfp4', 'avalon']))
    same = n15._agreement(fp, fp)
    _check('n15 identical recipes agree completely',
           same['identical'] and same['reps_identical']
           and same['rep_jaccard'] == 1.0)

    other = n15._bp_fingerprint(_BP(['maccs', 'rdkit_desc']))
    none_shared = n15._agreement(fp, other)
    _check('n15 recipes sharing no representation score zero overlap',
           none_shared['rep_jaccard'] == 0.0
           and not none_shared['reps_identical'],
           f"got {none_shared['rep_jaccard']}")


def test_n15_agreement_divides_by_the_union():
    """A superset is not agreement. Two shared of three between them is 2/3."""
    clean = n15._bp_fingerprint(_BP(['ecfp4', 'avalon']))
    noisy = n15._bp_fingerprint(_BP(['ecfp4', 'avalon', 'maccs']))
    ag = n15._agreement(clean, noisy)
    _check('n15 the overlap denominator is both recipes together, not either one',
           abs(ag['rep_jaccard'] - 2.0 / 3.0) < 1e-12,
           f"got {ag['rep_jaccard']:.6f}, expected 0.666667")
    _check('n15 names the representation only the noise-searched arm took',
           ag['reps_noisy_only'] == ['maccs'] and ag['reps_clean_only'] == [],
           f"clean_only={ag['reps_clean_only']} noisy_only={ag['reps_noisy_only']}")


def test_n15_agreement_separates_the_non_rep_settings():
    """Same representations, different budget, must not read as identical."""
    clean = n15._bp_fingerprint(_BP(['ecfp4'], budget=50))
    noisy = n15._bp_fingerprint(_BP(['ecfp4'], budget=250))
    ag = n15._agreement(clean, noisy)
    _check('n15 same representations with a different budget is not identical',
           ag['reps_identical'] and not ag['same_budget']
           and not ag['identical'])

    alloc = n15._bp_fingerprint(_BP(['ecfp4'], allocation_method='fixed'))
    ag2 = n15._agreement(clean, alloc)
    _check('n15 a different allocation method is reported on its own',
           not ag2['same_allocation'] and ag2['same_budget'])


def test_n15_agreement_handles_a_missing_arm():
    fp = n15._bp_fingerprint(_BP(['ecfp4']))
    _check('n15 a missing blueprint gives no fingerprint',
           n15._bp_fingerprint(None) is None)
    _check('n15 a missing arm gives no agreement rather than a false one',
           n15._agreement(fp, None) is None
           and n15._agreement(None, fp) is None)


# ---------------------------------------------------------------------------
# n14 — the cliff graph
# ---------------------------------------------------------------------------

def test_n14_pairs_mask_is_symmetric_and_strict():
    # Four molecules. One pair over the 2.0 threshold, one exactly at it.
    pairs = [(0, 1, 2.5), (2, 3, 2.0)]
    adj, considered = n14._cliff_mask_from_pairs(4, pairs, 2.0)
    _check('n14 every pair handed in is counted as considered',
           considered == 2, f"got {considered}")
    _check('n14 a pair over the potency threshold becomes a cliff',
           adj[0, 1] and adj[1, 0])
    _check('n14 the potency threshold is strict, so a gap exactly at it is not a cliff',
           not adj[2, 3] and not adj[3, 2])
    _check('n14 no molecule is its own cliff partner',
           not adj.diagonal().any())


def test_n14_pairs_mask_ignores_out_of_range_indices():
    pairs = [(0, 1, 5.0), (0, 99, 5.0), (-1, 2, 5.0)]
    adj, considered = n14._cliff_mask_from_pairs(3, pairs, 2.0)
    _check('n14 a pair naming a molecule outside the set is not counted',
           considered == 1, f"got {considered}")
    _check('n14 and it adds no edge', int(adj.sum()) == 2)


def test_n14_similarity_mask_absolute_and_percentile():
    labels = np.array([0.0, 5.0, 0.1, 5.2])
    sim = np.array([
        [1.00, 0.90, 0.20, 0.10],
        [0.90, 1.00, 0.15, 0.30],
        [0.20, 0.15, 1.00, 0.95],
        [0.10, 0.30, 0.95, 1.00],
    ])
    adj, cutoff = n14._cliff_mask_from_similarity(sim, labels, 0.55, 2.0)
    _check('n14 the absolute path uses the similarity threshold it was given',
           cutoff == 0.55, f"got {cutoff}")
    _check('n14 a similar pair with a large activity gap is a cliff',
           adj[0, 1] and adj[2, 3])
    _check('n14 a dissimilar pair with a large activity gap is not',
           not adj[0, 3] and not adj[1, 2])
    _check('n14 no molecule is its own cliff partner',
           not adj.diagonal().any())

    # The percentile path reads the off-diagonal pairs only. Asking for the top
    # 25% of them must give a cutoff above the smallest off-diagonal value.
    adj_p, cutoff_p = n14._cliff_mask_from_similarity(
        sim, labels, 0.55, 2.0, similar_percentile=25.0)
    off = sim[~np.eye(4, dtype=bool)]
    _check('n14 the percentile path sets its cutoff from the off-diagonal pairs',
           off.min() < cutoff_p <= off.max(),
           f"cutoff {cutoff_p:.4f} against off-diagonal range "
           f"{off.min():.4f} to {off.max():.4f}")


def test_n14_geometry_counts_edges_not_ends():
    # Two molecules joined, two alone.
    adj = np.zeros((4, 4), dtype=bool)
    adj[0, 1] = adj[1, 0] = True
    g, cliff_mask = n14._geometry(adj)
    _check('n14 geometry counts every molecule in the set',
           g['n_compounds'] == 4, f"got {g['n_compounds']}")
    _check('n14 both ends of a cliff are cliff molecules',
           g['n_cliff_compounds'] == 2, f"got {g['n_cliff_compounds']}")
    _check('n14 one cliff is one pair, not two',
           g['n_cliff_pairs'] == 1, f"got {g['n_cliff_pairs']}")
    _check('n14 the share of molecules on a cliff is out of all of them',
           abs(g['pct_cliff_compounds'] - 50.0) < 1e-9,
           f"got {g['pct_cliff_compounds']}")
    _check('n14 an isolated pair gives each end one cliff neighbour',
           abs(g['mean_cliff_degree'] - 1.0) < 1e-9,
           f"got {g['mean_cliff_degree']}")
    _check('n14 the returned mask marks exactly the molecules on a cliff',
           cliff_mask.tolist() == [True, True, False, False],
           f"got {cliff_mask.tolist()}")
    _check('n14 two molecules joined to each other and nothing else is one '
           'isolated pair',
           g['n_cliff_components'] == 1 and g['n_singleton_cliff_pairs'] == 1
           and g['largest_component_size'] == 2)
    _check('n14 a landscape of isolated pairs is not called clustered',
           g['geometry_lean'] == 'isolated', f"got {g['geometry_lean']}")


def test_n14_geometry_on_an_empty_graph():
    g, cliff_mask = n14._geometry(np.zeros((5, 5), dtype=bool))
    _check('n14 a graph with no cliffs reports none rather than failing',
           g['n_cliff_compounds'] == 0 and g['n_cliff_pairs'] == 0
           and g['pct_cliff_compounds'] == 0.0
           and g['mean_cliff_degree'] == 0.0
           and g['geometry_lean'] == 'no_cliffs'
           and not cliff_mask.any())


def main():
    print("=== SMOKE: n14 / n15 / n16 ===\n")
    print("n16 — how far the top of an importance ranking moves:")
    test_n16_turnover_effective_sees_what_plain_top_k_misses()
    test_n16_turnover_agrees_when_every_feature_is_scored()
    test_n16_turnover_counts_and_caps()
    print("\nn15 — do the clean-searched and noise-searched recipes agree:")
    test_n15_agreement_identical_and_disjoint()
    test_n15_agreement_divides_by_the_union()
    test_n15_agreement_separates_the_non_rep_settings()
    test_n15_agreement_handles_a_missing_arm()
    print("\nn14 — the cliff graph:")
    test_n14_pairs_mask_is_symmetric_and_strict()
    test_n14_pairs_mask_ignores_out_of_range_indices()
    test_n14_similarity_mask_absolute_and_percentile()
    test_n14_geometry_counts_edges_not_ends()
    test_n14_geometry_on_an_empty_graph()

    print(f"\n=== {'ALL GREEN' if not _FAILS else f'{len(_FAILS)} FAILED: {_FAILS}'} ===")
    sys.exit(1 if _FAILS else 0)


if __name__ == '__main__':
    main()
