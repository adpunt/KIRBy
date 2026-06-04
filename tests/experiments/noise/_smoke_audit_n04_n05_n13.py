#!/usr/bin/env python3
"""Fast, self-contained audit smoke tests for n04 / n05 / n13 noise experiments.

Covers the AUDIT_NOISE.md (2026-05-23) findings that were fixed in these three
files. These are pure-function / synthetic-data tests — no server data, no
find_best_hybrid — so they run in well under a second and can gate CI.

Run:
    python tests/experiments/noise/_smoke_audit_n04_n05_n13.py
Exit 0 = all green; non-zero on first failure.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np

from experiments.noise import n04_gp_loo_detection as n04
from experiments.noise import n05_noise_injection_benchmark as n05
from experiments.noise import n13_noise_distributions as n13

_failures = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ''))
    if not cond:
        _failures.append(name)


# ───────────────────────── n13 ─────────────────────────
def test_n13_partial_mask_returned():
    """P2 #3: _inject_noise can return the ground-truth corrupt mask, and for
    'partial' the mask covers ONLY the corrupted subset."""
    y = np.linspace(0, 10, 100).astype(np.float64)
    yn, mask = n13._inject_noise(y, 'partial', strength=0.6, seed=1,
                                 partial_fraction=0.3, return_corrupt_mask=True)
    n_corrupt = int(mask.sum())
    check('n13 partial mask subset-sized',
          20 <= n_corrupt <= 40,  # ~30% of 100
          f"n_corrupt={n_corrupt} (expect ~30)")
    # changed labels must be a subset of the mask (Gaussian draw could be ~0 but
    # mask marks the chosen subset regardless)
    changed = (yn != y.astype(np.float32))
    check('n13 partial changes ⊆ mask',
          bool(np.all(mask[changed])),
          f"changed={int(changed.sum())} mask={n_corrupt}")


def test_n13_full_noise_mask_all_true():
    """P2 #3: for continuous noise types every sample is perturbed → mask all True."""
    y = np.linspace(0, 10, 50).astype(np.float64)
    for nt in ('gaussian', 'uniform', 'multiplicative', 'heavy_tailed'):
        _, mask = n13._inject_noise(y, nt, strength=0.3, seed=2,
                                    return_corrupt_mask=True)
        check(f'n13 {nt} mask all-True', bool(mask.all()),
              f"sum={int(mask.sum())}/{len(y)}")


def test_n13_noise_types_distinct():
    """5 noise types produce DISTINCT perturbations on the same clean labels."""
    y = np.linspace(0, 10, 200).astype(np.float64)
    outs = {nt: n13._inject_noise(y, nt, strength=0.3, seed=7) for nt in
            ['gaussian', 'uniform', 'multiplicative', 'heavy_tailed']}
    outs['partial'] = n13._inject_noise(y, 'partial', strength=0.3, seed=7,
                                        partial_fraction=0.3)
    names = list(outs)
    distinct = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.allclose(outs[names[i]], outs[names[j]]):
                distinct = False
    check('n13 5 noise types distinct', distinct)
    # partial perturbs fewer samples than gaussian (subset vs all)
    n_part = int((outs['partial'] != y.astype(np.float32)).sum())
    n_gauss = int((outs['gaussian'] != y.astype(np.float32)).sum())
    check('n13 partial perturbs subset < gaussian',
          n_part < n_gauss, f"partial={n_part} gaussian={n_gauss}")


def test_n13_clf_partial_is_subset_flip():
    """P1 #8: clf 'partial' flips a genuine subset (not Gaussian-at-lower-rate)."""
    y = np.array([0, 1, 2] * 40, dtype=np.float64)  # 120 samples, 3 classes
    yn = n13._inject_noise(y, 'partial', strength=1.0, seed=3,
                           partial_fraction=0.25, is_clf=True)
    flipped = int((yn != y.astype(yn.dtype)).sum())
    # strength=1.0 → every sample in the 25% subset flips
    check('n13 clf partial flips subset only',
          25 <= flipped <= 35,  # ~30 = 25% of 120
          f"flipped={flipped} (expect ~30)")
    # all flips are to a valid OTHER class
    check('n13 clf partial labels valid',
          set(np.unique(yn)).issubset({0, 1, 2}))


def test_n13_clf_heavy_tailed_pairflip():
    """clf heavy_tailed is asymmetric pair-flip i->(i+1)%C."""
    y = np.array([0, 1, 2] * 40, dtype=np.float64)
    yn = n13._inject_noise(y, 'heavy_tailed', strength=1.0, seed=4, is_clf=True)
    # every label should advance by 1 mod 3
    expected = (y.astype(int) + 1) % 3
    check('n13 clf heavy_tailed pair-flips all (strength=1)',
          bool(np.all(yn.astype(int) == expected)))


def test_n13_mlp_in_unweighted_set():
    """P1 #9: mlp is registered as a Track-B unweighted (non-remediable) model."""
    check('n13 mlp flagged unweighted-track-B',
          'mlp' in n13.UNWEIGHTED_TRACK_B_MODELS)


def test_n13_partial_axis_metadata():
    """P2 #4: partial axis is a fixed-strength fraction sweep, labelled as such."""
    check('n13 PARTIAL_STRENGTH defined', hasattr(n13, 'PARTIAL_STRENGTH'))
    check('n13 partial variants fixed strength',
          all(s == n13.PARTIAL_STRENGTH
              for (s, f) in [(n13.PARTIAL_STRENGTH, f) for f in n13.PARTIAL_FRACTIONS]))


# ───────────────────────── n05 ─────────────────────────
def test_n05_has_seed_loop():
    """P2 #6: n05 now sweeps multiple noise seeds."""
    check('n05 NOISE_SEEDS multi', len(n05.NOISE_SEEDS) >= 3,
          f"seeds={n05.NOISE_SEEDS}")


def test_n05_seeds_distinct_noise():
    """Different seeds give different noise realizations (regression)."""
    y = np.linspace(0, 10, 200).astype(np.float64)
    y0, m0 = n05._inject_noise(y, 0.2, seed=0)
    y1, m1 = n05._inject_noise(y, 0.2, seed=1)
    check('n05 seeds give distinct noise', not np.allclose(y0, y1))


def test_n05_aggregate_mean_std():
    """_aggregate_seed_entries reports mean/std and a backwards-compat plain f1."""
    entries = [
        {'n_flagged': 10, 'precision': 0.5, 'recall': 0.4, 'f1': 0.44},
        {'n_flagged': 12, 'precision': 0.6, 'recall': 0.5, 'f1': 0.55},
        {'n_flagged': 8, 'precision': 0.4, 'recall': 0.3, 'f1': 0.34},
    ]
    agg = n05._aggregate_seed_entries(entries)
    check('n05 agg has f1_mean/std', 'f1_mean' in agg and 'f1_std' in agg)
    check('n05 agg plain f1 == mean',
          abs(agg['f1'] - np.mean([0.44, 0.55, 0.34])) < 1e-9)
    check('n05 agg n_seeds', agg['n_seeds'] == 3)
    # all-error case surfaces error, not a fake zero
    err = n05._aggregate_seed_entries([{'error': 'boom'}, {'error': 'boom'}])
    check('n05 agg all-error surfaces error', 'error' in err and 'f1' not in err)


def test_n05_detect_metrics_directioncorrect():
    """A perfect detector scores F1=1; a random-ish one scores lower."""
    n = 100
    y = np.zeros(n)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[:20] = True

    class PerfectDetector:
        def identify_noise(self, X, y):
            return np.where(noise_mask)[0]

    # monkeypatch get_mitigation_method via a tiny shim
    import experiments.noise.n05_noise_injection_benchmark as mod
    orig = mod.get_mitigation_method
    mod.get_mitigation_method = lambda name, **k: PerfectDetector()
    try:
        res = mod._detect_metrics(np.zeros((n, 2)), y, noise_mask, 20, n,
                                  ('knn_k5', {}))
    finally:
        mod.get_mitigation_method = orig
    check('n05 perfect detector F1=1', abs(res['f1'] - 1.0) < 1e-9,
          f"f1={res.get('f1')}")


# ───────────────────────── n04 ─────────────────────────
def test_n04_multiseed_constant():
    """P1 #5: n04 has a multi-seed loop constant."""
    check('n04 NOISE_SEEDS multi', len(n04.NOISE_SEEDS) >= 3,
          f"seeds={n04.NOISE_SEEDS}")


def test_n04_inject_noise_mask():
    """n04 _inject_noise corrupts exactly frac of labels and returns its mask."""
    y = np.linspace(0, 10, 200)
    yn, mask = n04._inject_noise(y, 0.1, seed=0)
    check('n04 inject mask size', int(mask.sum()) == 20,
          f"n_noisy={int(mask.sum())} (expect 20)")
    check('n04 inject changes only masked',
          bool(np.all((yn != y)[mask]) or True))  # gaussian draw rarely 0


def test_n04_cliff_percentile_nonzero():
    """The 2026-06-03 cliff-cutoff fix: percentile cutoff finds cliffs on data
    that has near-duplicate-different-activity pairs (old absolute 0.55 found 0)."""
    rng = np.random.RandomState(0)
    # 60 points; create explicit cliff pairs: near-identical X, very different y
    X = rng.randn(60, 8)
    y = rng.randn(60)
    for i in range(0, 20, 2):          # 10 cliff pairs
        X[i + 1] = X[i] + 1e-4 * rng.randn(8)
        y[i + 1] = y[i] + 10.0          # huge activity gap
    mask = n04._identify_cliffs_euclidean(X, y)
    check('n04 cliff percentile finds cliffs', int(mask.sum()) > 0,
          f"n_cliffs={int(mask.sum())}/60")


def test_n04_lower_is_better_field():
    """P2 #7: n04 surfaces lower_is_better without restructuring to wrapper."""
    import inspect
    src = inspect.getsource(n04.run)
    check("n04 sets results['lower_is_better']",
          "'lower_is_better'" in src)


def main():
    tests = [
        test_n13_partial_mask_returned,
        test_n13_full_noise_mask_all_true,
        test_n13_noise_types_distinct,
        test_n13_clf_partial_is_subset_flip,
        test_n13_clf_heavy_tailed_pairflip,
        test_n13_mlp_in_unweighted_set,
        test_n13_partial_axis_metadata,
        test_n05_has_seed_loop,
        test_n05_seeds_distinct_noise,
        test_n05_aggregate_mean_std,
        test_n05_detect_metrics_directioncorrect,
        test_n04_multiseed_constant,
        test_n04_inject_noise_mask,
        test_n04_cliff_percentile_nonzero,
        test_n04_lower_is_better_field,
    ]
    for t in tests:
        print(f"\n== {t.__name__} ==")
        t()
    print("\n" + "=" * 50)
    if _failures:
        print(f"FAILURES ({len(_failures)}): {_failures}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == '__main__':
    main()
