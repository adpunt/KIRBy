#!/usr/bin/env python3
"""n12 end to end, twice: with both search families in the pool and with one.

The other two smokes check the resolution step and the build call. This one
runs a whole experiment and reads its result file, which is the only place the
three things the change has to satisfy can be checked together:

  1. With both families in the pool, every condition of BOTH hybrid spaces is
     populated, and the scores are in the dataset's own metric.
  2. The two spaces are not the same numbers. A geometry recipe that silently
     resolved to the predictive one is the failure this change exists to fix,
     and it would leave every other check passing.
  3. With the similarity family absent, every number the file reported before
     is the same to 1e-12 as it is with the family present, and the missing
     space carries a written reason. Not the bit, because the roster's random
     forest is not bit-reproducible in one process -- NOISE_AUDIT_2026_09_10.md
     section 17 measures that floor at about 1e-15 and sets 1e-12 as the
     tolerance every same-versus-same check in the suite uses.

The pool is stubbed: the lookup on a cold cache runs a full search, which is
hours to days locally. Everything after the lookup is the real file --
create_hybrid, the estimators, the folds and the scoring.

n12 is the one run end to end because it is the cheapest of the six on
synthetic rows and because its conditions are the most tangled: eight of them,
three per hybrid space, sharing one base arm.

    python tests/smoke/smoke_n12_two_spaces.py     (about 8 minutes)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments',
                                'noise'))

from kirby.find_best_hybrid import Blueprint  # noqa: E402
import experiments.noise.n12_importance_weighted_distance as n12  # noqa: E402

FAILURES = []

ORIGINAL_CONDITIONS = ('base_unweighted', 'base_imp_weighted',
                       'hybrid_unweighted', 'hybrid_imp_base',
                       'hybrid_imp_own')
SIM_CONDITIONS = ('hybrid_similarity_unweighted', 'hybrid_similarity_imp_base',
                  'hybrid_similarity_imp_own')


def check(name, cond, saw):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}: {saw}")
    if not cond:
        FAILURES.append(name)


def _bp(alloc, budget, objective, metric, lower_is_better):
    return Blueprint(
        reps=['a', 'b'], allocation_method=alloc,
        importance_method='lightgbm_gain', budget=budget,
        internal_evaluator='rf', metric=metric,
        lower_is_better=lower_is_better, objective=objective,
        augmentation_strategy='none')


BP_PREDICTIVE = _bp('performance_weighted', 10, 'predictive', 'rmse', True)
BP_SIMILARITY = _bp('weighted_global', 20, 'similarity', 'readacross', False)


def _stub_pool(with_similarity):
    """Stands in for get_best_search_for_downstream.

    The winner is the predictive family whichever way this is called, which is
    what the pool returns under 'predictive_and_similarity' on every dataset
    that holds a predictive search -- that is the whole reason the geometry
    family was never built.
    """
    by_obj = {'predictive': {
        'tier': 'general', 'aug_condition': 'none', 'score_mean': 0.71,
        'metric_name': 'rmse', 'lower_is_better': True,
        'best_config': {'blueprint': BP_PREDICTIVE}}}
    if with_similarity:
        by_obj['similarity'] = {
            'tier': 'general', 'aug_condition': 'none', 'score_mean': 0.38,
            'metric_name': 'readacross', 'lower_is_better': False,
            'best_config': {'blueprint': BP_SIMILARITY}}

    def _fake(*_a, **_k):
        return None, {
            'objective': 'predictive', 'objective_choice': 'subset_order',
            'pool_subset': 'predictive_and_similarity',
            'tier': 'general', 'aug_condition': 'none', 'score_mean': 0.71,
            'metric_name': 'rmse', 'lower_is_better': True,
            'best_config': {'blueprint': BP_PREDICTIVE},
            'by_objective': by_obj,
        }
    return _fake


def _data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    reps = {'a': rng.normal(size=(n, 14)), 'b': rng.normal(size=(n, 18))}
    y = reps['a'][:, 0] * 2 + reps['b'][:, 1] + rng.normal(scale=0.3, size=n)
    return {'reps': reps, 'labels': y,
            'groups': np.repeat(np.arange(n // 10), 10),
            'task_type': 'regression', 'domain': 'molecular',
            'dataset_name': 'logd', 'augmentations': None}


def _run(with_similarity):
    n12.get_best_search_for_downstream = _stub_pool(with_similarity)
    return n12.run(_data(), {'n_splits': 2, 'n_folds': 2, 'mode': 'fast'},
                   None, 'logd', models=['rf'], sigmas=[0.3])


def _denoised(res, condition):
    """{sigma|importance method|estimator key: denoised score} for one condition."""
    out = {}
    for sigma, sv in (res.get('per_sigma') or {}).items():
        for imp, imd in (sv.get('per_importance_method') or {}).items():
            for ek, rv in (imd.get(condition) or {}).items():
                if isinstance(rv, dict) and 'denoised' in rv:
                    out[f'{sigma}|{imp}|{ek}'] = rv['denoised']
    return out


def main():
    print("1. both families in the pool\n")
    both = _run(True)

    spaces = sorted(both.get('hybrid_spaces') or {})
    check('one space per family', spaces == ['hybrid', 'hybrid_similarity'],
          spaces)
    check('each space built its own recipe',
          both.get('n_hybrid_builds_by_space', {}).get('hybrid')
          == both.get('n_hybrid_builds_by_space', {}).get('hybrid_similarity')
          and bool(both.get('n_hybrid_builds_by_space')),
          both.get('n_hybrid_builds_by_space'))
    check('the geometry space says what chose its features',
          'five nearest neighbours' in str(
              (both.get('hybrid_spaces') or {})
              .get('hybrid_similarity', {}).get('features_chosen_by', '')),
          (both.get('hybrid_spaces') or {})
          .get('hybrid_similarity', {}).get('features_chosen_by'))

    _cond_seen = set()
    for sv in (both.get('per_sigma') or {}).values():
        for imd in (sv.get('per_importance_method') or {}).values():
            _cond_seen |= set(imd)
    check('all eight conditions are populated',
          set(ORIGINAL_CONDITIONS) | set(SIM_CONDITIONS) <= _cond_seen,
          sorted(_cond_seen))

    # The metric every denoised score is in. n12 measures it itself with
    # metric_fn, so it is the dataset's own metric in both spaces, never the
    # 'readacross' correlation the geometry search scored its candidates by.
    check("the scores are in the dataset's own metric",
          both.get('metric') == 'rmse' and both.get('lower_is_better') is True,
          f"{both.get('metric')}, lower_is_better={both.get('lower_is_better')}")

    for pred_c, sim_c in zip(ORIGINAL_CONDITIONS[2:], SIM_CONDITIONS):
        a, b = _denoised(both, pred_c), _denoised(both, sim_c)
        shared = sorted(set(a) & set(b))
        check(f'{pred_c} and {sim_c} are not the same numbers',
              bool(shared) and any(a[k] != b[k] for k in shared),
              f"{len(shared)} cells compared, "
              f"{sum(1 for k in shared if a[k] != b[k])} differ")

    print("\n2. the similarity family absent from the pool\n")
    one = _run(False)

    check('only the predictive space is built',
          sorted(one.get('hybrid_spaces') or {}) == ['hybrid'],
          sorted(one.get('hybrid_spaces') or {}))
    _why = (one.get('hybrid_spaces_unavailable') or {}).get(
        'hybrid_similarity', '')
    check('the missing space carries a written reason',
          'objective=similarity' in _why, _why)

    _cond_seen_one = set()
    for sv in (one.get('per_sigma') or {}).values():
        for imd in (sv.get('per_importance_method') or {}).values():
            _cond_seen_one |= set(imd)
    check('no geometry condition is reported',
          not (set(SIM_CONDITIONS) & _cond_seen_one),
          sorted(_cond_seen_one))

    # The point of the whole change: nothing already reported moves.
    #
    # ⛔ NOT AN EXACT EQUALITY, and section 17 of NOISE_AUDIT_2026_09_10.md is
    # why. The roster's random forest sums each tree's prediction into one
    # shared array from worker threads, the summation order is the thread
    # completion order, and floating-point addition is not associative -- so
    # two identical fits in one process agree to about 1e-15 and not to the
    # bit. That audit set 1e-12 as the tolerance for every same-versus-same
    # comparison in the suite; this uses the same one. A real change of
    # behaviour is orders of magnitude above it: the geometry conditions in
    # part 1 differ from their predictive twins by far more than this.
    TOL = 1e-12
    for cond in ORIGINAL_CONDITIONS:
        a, b = _denoised(both, cond), _denoised(one, cond)
        _diffs = [abs(a[k] - b[k]) for k in a if k in b]
        _worst = max(_diffs) if _diffs else float('inf')
        check(f'{cond} does not move when the second space is added',
              bool(a) and set(a) == set(b) and _worst <= TOL,
              f"{len(a)} cells, largest move {_worst:.3g} "
              f"(tolerance {TOL:g}; the thread-order floor is ~1e-15)")

    for _k in ('weighting_helps_base', 'weighting_helps_hybrid',
               'hybrid_already_weighted', 'best_k', 'best_importance_method'):
        _a = (both.get('summary') or {}).get('sigma_0.3', {}).get(_k)
        _b = (one.get('summary') or {}).get('sigma_0.3', {}).get(_k)
        check(f"summary '{_k}' is unchanged", _a == _b, f"{_a} vs {_b}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
