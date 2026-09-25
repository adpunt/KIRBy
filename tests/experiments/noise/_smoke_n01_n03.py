#!/usr/bin/env python3
"""Fast, self-contained unit smoke tests for n01/n02 cliff-threshold fixes.

These do NOT run find_best_hybrid (that's what _smoke_harness.py is for). They
exercise the pure cliff/zone-detection logic that the 2026-06-03 noise audit
fixed, on synthetic standardised high-dimensional data — the exact regime where
the old absolute thresholds collapsed to ~0 cliffs on antibody / reaction / TCGA
data (ginkgo_hic all-zero, tcga stratified too_few_cliff_compounds).

Run:  python tests/experiments/noise/_smoke_n01_n03.py
Exit 0 = all green.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# 2026-09-25: the noise experiments sweep every similarity setting by default
# (KIRBY_SIMILARITY_SETTINGS unset = all 66, plan section 1). This smoke checks
# what each run() computed before that axis existed, so it runs today's
# setting alone, and it writes its caches under a temporary directory, never
# the real KIRBY_CACHE_DIR.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
import tempfile as _tempfile
os.environ['KIRBY_CACHE_DIR'] = _tempfile.mkdtemp(prefix='kirby_smoke_cache_')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.noise import n01_activity_cliff_framework as n01
from experiments.noise import n02_sample_weighting as n02
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


def _high_dim_euclidean_sim(n=120, d=300, seed=0):
    rng = np.random.RandomState(seed)
    X = StandardScaler().fit_transform(rng.randn(n, d))
    D = compute_distance_matrix(X, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    return sim


def test_n01_percentile_finds_cliffs_where_absolute_fails():
    """The core audit fix: on standardised high-D reps the absolute 0.55
    Euclidean cutoff finds 0 cliffs; the distribution-relative percentile path
    finds a healthy non-degenerate set."""
    n = 120
    rng = np.random.RandomState(1)
    sim = _high_dim_euclidean_sim(n=n)
    y = rng.randn(n) * 5.0

    _, info_abs = n01.identify_cliff_pairs(sim, y, 0.55, 2.0,
                                           is_clf=False, similar_percentile=None)
    _, info_pct = n01.identify_cliff_pairs(sim, y, 0.55, 2.0,
                                           is_clf=False,
                                           similar_percentile=n01.SIMILAR_PERCENTILE)

    max_off = sim[~np.eye(n, dtype=bool)].max()
    assert max_off < 0.55, f"test premise broken: max off-diag sim {max_off} >= 0.55"
    assert info_abs['n_cliff_pairs'] == 0, \
        f"expected degenerate 0 cliffs at abs 0.55, got {info_abs['n_cliff_pairs']}"
    assert info_pct['n_cliff_pairs'] > 0, \
        f"percentile path still degenerate: {info_pct['n_cliff_pairs']} cliffs"
    # cutoff should be distribution-relative (well below 0.55, ~the 98th pct)
    assert info_pct['sim_cutoff'] < 0.55
    print(f"  n01 reg: abs={info_abs['n_cliff_pairs']} cliffs, "
          f"pct={info_pct['n_cliff_pairs']} cliffs (cutoff={info_pct['sim_cutoff']:.4f}) OK")


def test_n01_classification_branch_nonzero():
    """is_clf path (different-class cliffs) is non-degenerate under percentile."""
    n = 120
    rng = np.random.RandomState(2)
    sim = _high_dim_euclidean_sim(n=n, seed=2)
    y = rng.randint(0, 2, n).astype(float)
    _, info = n01.identify_cliff_pairs(sim, y, 0.55, 2.0,
                                       is_clf=True,
                                       similar_percentile=n01.SIMILAR_PERCENTILE)
    assert info['n_cliff_pairs'] > 0, "clf percentile path degenerate"
    assert info['activity_threshold'] == 0.5, "clf must use 0.5 class-diff threshold"
    print(f"  n01 clf: {info['n_cliff_pairs']} cliffs OK")


def test_n01_editdist_percentile_threaded():
    """The antibody stratified branch must pass similar_percentile (not the old
    absolute max_edits=5) so therapeutic panels with high mutual edit distance
    don't collapse to too_few_cliff_compounds."""
    from experiments.similarity.similarity_helpers import identify_cliff_compounds_editdist
    rng = np.random.RandomState(3)
    # 40 sequences len 60, a few near-duplicates with large activity gaps
    base = rng.randint(0, 20, (40, 60))
    seqs = [''.join('ACDEFGHIKLMNPQRSTVWY'[c] for c in row) for row in base]
    y = rng.randn(40)
    mask_abs, info_abs = identify_cliff_compounds_editdist(
        seqs, y, max_edits=5, activity_k=2.0, is_clf=False)
    mask_pct, info_pct = identify_cliff_compounds_editdist(
        seqs, y, activity_k=2.0, is_clf=False,
        similar_percentile=n01.SIMILAR_PERCENTILE)
    # On random high-divergence sequences max_edits=5 flags ~nothing.
    assert info_abs['n_cliff_pairs'] <= info_pct['n_cliff_pairs'] or \
        info_pct['edit_cutoff'] >= 5, "percentile should relax the similar set"
    assert info_pct['similar_percentile'] == n01.SIMILAR_PERCENTILE
    print(f"  n01 editdist: abs(max5)={info_abs['n_cliff_pairs']} "
          f"pct={info_pct['n_cliff_pairs']} (cutoff={info_pct['edit_cutoff']:.1f}) OK")


def test_n02_cliff_aware_threads_is_clf():
    """compute_cliff_aware_weights must run the classification noise path when
    is_clf=True (estimator default is regression, meaningless on class labels)."""
    rng = np.random.RandomState(4)
    X = StandardScaler().fit_transform(rng.randn(80, 30))
    y_clf = rng.randint(0, 2, 80).astype(float)
    w_clf = n02.compute_cliff_aware_weights(X, y_clf, k=10, is_clf=True)
    w_reg = n02.compute_cliff_aware_weights(X, y_clf, k=10, is_clf=False)
    assert w_clf.shape == (80,)
    assert np.isfinite(w_clf).all()
    # The two paths compute different scores -> generally different weights.
    assert not np.allclose(w_clf, w_reg), \
        "is_clf flag had no effect — clf path not wired"
    print(f"  n02 cliff-aware clf-vs-reg weights differ "
          f"(mean clf={w_clf.mean():.3f}) OK")


if __name__ == '__main__':
    tests = [
        test_n01_percentile_finds_cliffs_where_absolute_fails,
        test_n01_classification_branch_nonzero,
        test_n01_editdist_percentile_threaded,
        test_n02_cliff_aware_threads_is_clf,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'ALL GREEN' if failed == 0 else f'{failed} FAILED'} "
          f"({len(tests) - failed}/{len(tests)} passed)")
    sys.exit(1 if failed else 0)
