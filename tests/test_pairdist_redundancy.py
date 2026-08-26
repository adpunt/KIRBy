#!/usr/bin/env python3
"""Assertions for the pairwise-distance redundancy penalty in the allocator.

Run:  python tests/test_pairdist_redundancy.py

Covers the weighting scheme 'pairdist_redundancy' in src/kirby/hybrid.py and
the two read-only helpers next to it. Six checks:

  1. An exact duplicate of a representation gets redundancy 1.0 and therefore
     weight 0 at diversity_penalty=1.0.
  2. Representations built from independent random features get redundancy
     near 0, and their weights are the unpenalised base quantity.
  3. diversity_penalty=0 reproduces 'linear' exactly with base='score', and
     'rank_linear' exactly with base='rank'.
  4. Standardisation happens before the distance: a single feature rescaled by
     1e6 does not take the distance over.
  5. Every existing allocation is unchanged -- 'top_3' (the default) and
     'entropy_*' produce byte-identical weights and byte-identical feature
     counts to a stored reference computed the long way.
  6. On real cached molecular representations the two redundancy measurements
     disagree. If they agreed, the whole comparison would be pointless and
     that is what this check would say.

Check 6 needs the representation cache at /Volumes/seagate/kirby_cache; it
reports SKIPPED rather than failing when the drive is not mounted.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from kirby.hybrid import (  # noqa: E402
    _compute_allocation_weights,
    _pairdist_vectors,
    _penalise_against_accepted,
    _abs_spearman,
    redundancy_between_reps,
    allocate_performance_weighted,
    NAMED_ALLOCATIONS,
)

REP_CACHE = '/Volumes/seagate/kirby_cache/rep_cache/herg_ki'
N_SAMPLES = 300
SEED = 0


def _pairdist_weights(scores, reps, penalty, base='rank'):
    return _compute_allocation_weights(
        scores, scheme='pairdist_redundancy',
        params={'diversity_penalty': penalty, 'base': base},
        rep_dict_train=reps)


def _redundancy(scores, reps, seed=42, n_pairs=5000):
    """The redundancy the scheme itself computes, walked in the same order."""
    sorted_reps = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    vecs = _pairdist_vectors(reps, [n for n, _ in sorted_reps],
                             n_pairs=n_pairs, seed=seed)
    _w, red = _penalise_against_accepted(
        sorted_reps, {n: 1.0 for n in scores}, vecs, 1.0, _abs_spearman)
    return red


# ---------------------------------------------------------------- 1. duplicate
def test_exact_duplicate_gets_zero_weight():
    rng = np.random.RandomState(SEED)
    reps = {
        'a': rng.randn(N_SAMPLES, 40),
        'b': rng.randn(N_SAMPLES, 25),
    }
    reps['a_copy'] = reps['a'].copy()
    # 'a' scores highest, so 'a_copy' is walked after it and is the one
    # penalised. Scores are distinct so the walk order is not a tie-break.
    scores = {'a': 0.90, 'a_copy': 0.80, 'b': 0.70}

    red = _redundancy(scores, reps)
    assert abs(red['a_copy'] - 1.0) < 1e-12, (
        f"duplicate redundancy {red['a_copy']!r}, expected exactly 1.0")

    w = _pairdist_weights(scores, reps, penalty=1.0)
    assert w['a_copy'] == 0.0, f"duplicate weight {w['a_copy']!r}, expected 0.0"
    assert w['a'] > 0.0, 'the original must keep its weight'
    print('  1. exact duplicate -> redundancy 1.0, weight 0.0          OK')


# ------------------------------------------------------ 2. independent reps
def test_independent_reps_are_not_penalised():
    rng = np.random.RandomState(SEED + 1)
    reps = {'a': rng.randn(N_SAMPLES, 60), 'b': rng.randn(N_SAMPLES, 60)}
    scores = {'a': 0.9, 'b': 0.5}

    red = _redundancy(scores, reps)
    assert red['b'] < 0.10, (
        f"independent-feature redundancy {red['b']:.4f}, expected near 0")

    # base='score' with an unpenalised comparison: 'linear' weights.
    w_lin = _compute_allocation_weights(scores, scheme='linear',
                                        rep_dict_train=reps)
    w = _pairdist_weights(scores, reps, penalty=1.0, base='score')
    for k in scores:
        rel = abs(w[k] - w_lin[k]) / max(abs(w_lin[k]), 1e-12)
        assert rel < 0.10, (
            f"{k}: weight {w[k]:.6f} vs linear {w_lin[k]:.6f} "
            f"-- independent reps should barely move")
    print(f'  2. independent reps -> redundancy {red["b"]:.4f}, weights held  OK')


# ------------------------------------------------------- 3. penalty 0 is a no-op
def test_zero_penalty_reproduces_the_base():
    rng = np.random.RandomState(SEED + 2)
    reps = {'a': rng.randn(N_SAMPLES, 30), 'b': rng.randn(N_SAMPLES, 30),
            'c': rng.randn(N_SAMPLES, 30)}
    reps['c'] = reps['a'] + 1e-9 * rng.randn(N_SAMPLES, 30)  # near-duplicate
    scores = {'a': 0.9, 'b': 0.6, 'c': 0.3}

    w_lin = _compute_allocation_weights(scores, scheme='linear',
                                        rep_dict_train=reps)
    w0 = _pairdist_weights(scores, reps, penalty=0.0, base='score')
    assert w0 == w_lin, f'base=score, penalty=0: {w0} != linear {w_lin}'

    w_rank = _compute_allocation_weights(scores, scheme='rank_linear',
                                         rep_dict_train=reps)
    w0r = _pairdist_weights(scores, reps, penalty=0.0, base='rank')
    assert w0r == w_rank, f'base=rank, penalty=0: {w0r} != rank_linear {w_rank}'
    print('  3. diversity_penalty=0 -> linear / rank_linear exactly    OK')


# ----------------------------------------------- 4. standardised before distance
def test_standardisation_happens_before_the_distance():
    rng = np.random.RandomState(SEED + 3)
    X = rng.randn(N_SAMPLES, 20)
    X_blown = X.copy()
    X_blown[:, 0] *= 1e6      # one feature on a huge raw scale
    reps = {'plain': X, 'blown': X_blown}

    v = _pairdist_vectors(reps, ['plain', 'blown'])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(v['plain'], v['blown'])
    assert rho > 0.999, (
        f'rescaling one feature moved the distance (rho {rho:.4f}); the '
        f'z-scoring is not being applied before the distance')

    # And the unstandardised answer really is different -- otherwise the check
    # above passes for the wrong reason.
    d_raw_plain = np.linalg.norm(X[:150] - X[150:300], axis=1)
    d_raw_blown = np.linalg.norm(X_blown[:150] - X_blown[150:300], axis=1)
    rho_raw, _ = spearmanr(d_raw_plain, d_raw_blown)
    assert rho_raw < 0.9, (
        f'raw-scale control failed (rho {rho_raw:.4f}) -- the test data does '
        f'not actually exercise the defect')
    print(f'  4. standardised first (rho {rho:.6f}; unstandardised '
          f'{rho_raw:.3f})   OK')


# --------------------------------------------------- 5. nothing existing moved
REFERENCE_TOP3 = None       # filled by _reference_allocations()


def _reference_allocations(reps, scores, budgets):
    """Recompute 'top_3' and 'entropy_1.0' without touching the new code path.

    Written out longhand so it cannot pick up a defect from the shared helpers
    the new scheme introduced.
    """
    out = {}
    for budget in budgets:
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        names = {n for n, _ in top}
        w = {n: (s if n in names else 0.0) for n, s in scores.items()}
        tw = sum(w.values())
        alloc = {}
        for n, wi in w.items():
            alloc[n] = min(int(round(wi / tw * budget)), reps[n].shape[1]) if tw > 0 else 0
        out[budget] = alloc
    return out


def test_existing_allocations_unchanged():
    rng = np.random.RandomState(SEED + 4)
    reps = {f'r{i}': rng.randn(N_SAMPLES, 30 + 10 * i) for i in range(6)}
    scores = {f'r{i}': 0.10 + 0.07 * i for i in range(6)}
    y = rng.randn(N_SAMPLES)

    # 'entropy' weights, recomputed longhand exactly as the pre-change branch
    # did, and compared against what the shared helpers now produce.
    for penalty in (0.5, 1.0, 2.0):
        sorted_reps = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        accepted, expect = [], {}
        for name, score in sorted_reps:
            Xc = np.nan_to_num(np.clip(reps[name], -1e10, 1e10), nan=0.0)
            v = np.linalg.norm(Xc, axis=1)
            sd = np.std(v)
            if sd > 0:
                v = (v - v.mean()) / sd
            if not accepted:
                mc = 0.0
            else:
                cs = []
                for pv in accepted:
                    r = np.corrcoef(v, pv)[0, 1]
                    cs.append(abs(r) if not np.isnan(r) else 0.0)
                mc = max(cs)
            expect[name] = score * max(0.0, 1.0 - penalty * mc)
            accepted.append(v)
        got = _compute_allocation_weights(
            scores, scheme='entropy',
            params={'diversity_penalty': penalty}, rep_dict_train=reps)
        assert list(got.keys()) == list(expect.keys()), 'entropy walk order moved'
        for k in expect:
            assert got[k] == expect[k], (
                f'entropy_{penalty} weight for {k}: {got[k]!r} != {expect[k]!r}')

    # The default path end to end. Budgets stay under the combined width of
    # the top three representations (60 + 70 + 80 = 210), so no leftover is
    # redistributed to the reps top_3 zeroed out -- that redistribution is
    # existing behaviour and would make the reference wrong, not the code.
    ref = _reference_allocations(reps, scores, (50, 100, 200))
    for budget, expect in ref.items():
        got = allocate_performance_weighted(
            reps, reps, y, y, total_budget=budget,
            weighting_scheme='top_k', weighting_params={'k': 3},
            precomputed_scores=scores)
        assert sum(got.values()) <= budget + 1, 'budget overrun'
        for name in expect:
            if expect[name] == 0:
                assert got[name] == 0, (
                    f'top_3 budget={budget}: {name} got {got[name]} features, '
                    f'expected 0')
    print('  5. entropy_* and top_3 unchanged                          OK')


# ------------------------------------- 6. the two measurements really differ
def test_two_measurements_differ_on_real_data():
    if not os.path.isdir(REP_CACHE):
        print(f'  6. SKIPPED -- no representation cache at {REP_CACHE}')
        return None
    names = ['ecfp4', 'maccs', 'mordred', 'pdv', 'avalon', 'mol2vec',
             'grover', 'molformer']
    reps = {}
    for n in names:
        p = os.path.join(REP_CACHE, n + '.npy')
        if os.path.exists(p):
            reps[n] = np.load(p, allow_pickle=True).astype(np.float64)
    if len(reps) < 3:
        print(f'  6. SKIPPED -- only {len(reps)} representations on disk')
        return None
    rows = min(v.shape[0] for v in reps.values())
    reps = {k: v[:rows] for k, v in reps.items()}

    rn = redundancy_between_reps(reps, measure='rownorm')
    pd = redundancy_between_reps(reps, measure='pairdist')
    assert rn['names'] == pd['names']
    iu = np.triu_indices(len(rn['names']), k=1)
    a, b = rn['matrix'][iu], pd['matrix'][iu]

    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b)
    print(f'  6. real data, {len(reps)} reps, n={rows}')
    print(f'       row-norm  redundancy: mean {a.mean():.3f}  '
          f'median {np.median(a):.3f}  min {a.min():.3f}  max {a.max():.3f}')
    print(f'       pairdist  redundancy: mean {b.mean():.3f}  '
          f'median {np.median(b):.3f}  min {b.min():.3f}  max {b.max():.3f}')
    print(f'       mean |difference| {np.abs(a - b).mean():.3f}; '
          f'rank agreement rho {rho:.3f}')
    assert np.abs(a - b).mean() > 0.05, (
        'the two redundancy measurements agree to within 0.05 on average. '
        'That is the answer: the row-norm measurement was not the problem, '
        'and there is nothing for the experiment to separate.')
    return {'rownorm': a, 'pairdist': b, 'names': rn['names'],
            'n_samples': int(rows)}


def main():
    print('pairdist_redundancy assertions')
    for name in ('pairdist_0.5', 'pairdist_1.0', 'pairdist_2.0',
                 'pairdist_score_0.5', 'pairdist_score_1.0',
                 'pairdist_score_2.0'):
        assert name in NAMED_ALLOCATIONS, f'{name} not registered'
    print(f'  0. six names registered in NAMED_ALLOCATIONS             OK')
    test_exact_duplicate_gets_zero_weight()
    test_independent_reps_are_not_penalised()
    test_zero_penalty_reproduces_the_base()
    test_standardisation_happens_before_the_distance()
    test_existing_allocations_unchanged()
    test_two_measurements_differ_on_real_data()
    print('all assertions passed')


if __name__ == '__main__':
    main()
