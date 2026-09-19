#!/usr/bin/env python3
"""Does every experiment that asks the pool for two search families BUILD both?

Nine experiments asked the pool for `pool_subset='predictive_and_similarity'`
and built one feature space from it. POOL_SUBSETS names the two families in
PREFERENCE order and the pool returns the first that has an entry, so a
predictive recipe -- which always exists -- was built and the other family was
written into the result file as a score nothing had measured in. Fixed
2026-09-19.

WHAT THIS CHECKS, per file:

  1. The file names a second space, and the first one still carries the name it
     carried before, so no key already reported is renamed.
  2. With both families in the pool, the file's own resolution step produces one
     recipe per family, and the two recipes are different objects with
     different settings.
  3. With the similarity family absent from the pool, the predictive recipe is
     the SAME object it was in case 2, and the missing space carries a written
     reason naming why.
  4. The two recipes, built through the same call the file builds them with,
     give feature matrices that are NOT byte-identical. A geometry recipe that
     silently resolved to the predictive one is the failure this whole change
     exists to fix, and it would leave every other check passing.

WHAT THIS DOES NOT CHECK. It runs no experiment end to end and reads no real
dataset: the pool lookup on a cold cache runs a full search, which is hours to
days locally (see the note in the noise suite). The end-to-end runs belong on
the cluster.

    python tests/smoke/smoke_similarity_hybrid_spaces.py
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
from experiments._shared import (  # noqa: E402
    pool_recipes_by_family, POOL_SUBSETS, build_fold_hybrid,
)
from experiments.noise.noise_helpers import hybrid_for_fold  # noqa: E402

FAILURES = []


def check(name, cond, saw):
    print(f"    {'ok  ' if cond else 'FAIL'} {name}: {saw}")
    if not cond:
        FAILURES.append(name)


# ── the two recipes ───────────────────────────────────────────────────────────
# Different in four ways, so a matrix built from one can never be mistaken for a
# matrix built from the other: the number of features, how the budget is split,
# the metric the search scored in and its direction.
def _predictive_bp():
    return Blueprint(
        reps=['a', 'b'], allocation_method='performance_weighted',
        importance_method='lightgbm_gain', budget=10, internal_evaluator='rf',
        metric='rmse', lower_is_better=True, objective='predictive',
        augmentation_strategy='none')


def _similarity_bp():
    return Blueprint(
        reps=['a', 'b'], allocation_method='weighted_global',
        importance_method='lightgbm_gain', budget=20, internal_evaluator='rf',
        metric='readacross', lower_is_better=False, objective='similarity',
        augmentation_strategy='none')


def _meta(bp_pred, bp_sim=None):
    """Metadata shaped like get_best_search_for_downstream's.

    The winner is the predictive family, which is what the pool returns under
    'predictive_and_similarity' on every dataset that has one.
    """
    by_obj = {'predictive': {
        'tier': 'general', 'aug_condition': 'none', 'score_mean': 0.71,
        'metric_name': 'rmse', 'lower_is_better': True,
        'best_config': {'blueprint': bp_pred}}}
    if bp_sim is not None:
        by_obj['similarity'] = {
            'tier': 'general', 'aug_condition': 'none', 'score_mean': 0.38,
            'metric_name': 'readacross', 'lower_is_better': False,
            'best_config': {'blueprint': bp_sim}}
    return {
        'objective': 'predictive', 'objective_choice': 'subset_order',
        'pool_subset': 'predictive_and_similarity',
        'tier': 'general', 'aug_condition': 'none',
        'score_mean': 0.71, 'metric_name': 'rmse', 'lower_is_better': True,
        'best_config': {'blueprint': bp_pred},
        'by_objective': by_obj,
    }


# ── synthetic molecules ───────────────────────────────────────────────────────
def _fixture(seed=0, n=120):
    rng = np.random.default_rng(seed)
    reps = {'a': rng.normal(size=(n, 14)), 'b': rng.normal(size=(n, 18))}
    y = reps['a'][:, 0] * 2 + reps['b'][:, 1] + rng.normal(scale=0.3, size=n)
    groups = np.repeat(np.arange(n // 10), 10)
    train = np.arange(0, int(n * 0.75))
    val = np.arange(int(n * 0.75), n)
    return reps, y, groups, train, val


def _rf():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=20, random_state=0)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true)
                                  - np.asarray(y_pred)) ** 2)))


def _matrices_through(builder, bp_pred, bp_sim):
    """(X from the predictive recipe, X from the geometry one)."""
    reps, y, groups, tr, va = _fixture()
    out = []
    for bp in (bp_pred, bp_sim):
        out.append(builder(reps, y, groups, tr, va, bp))
    return out


def _build_via_hybrid_for_fold(reps, y, groups, tr, va, bp):
    """What n17, n19 and n23 call."""
    X_tr, _ = hybrid_for_fold(
        {k: v[tr] for k, v in reps.items()},
        {k: v[va] for k, v in reps.items()},
        y[tr], bp, groups[tr], _rf(), _rmse, True)
    return X_tr


def _build_via_build_fold_hybrid(reps, y, groups, tr, va, bp):
    """What n02, n07, n12, s06, s08 and s10 call, geometry-guarded."""
    from kirby.find_best_hybrid import _is_geometry_objective
    geometry = _is_geometry_objective(getattr(bp, 'objective', 'predictive'))
    extra = ({} if geometry
             else {'model': _rf(), 'metric_fn': _rmse, 'lower_is_better': True})
    X_tr, _X_va, _fi = build_fold_hybrid(
        {k: v[tr] for k, v in reps.items()},
        {k: v[va] for k, v in reps.items()},
        y[tr], bp, groups=groups[tr], **extra)
    return X_tr


# ── the files, and where each one's second space is named ────────────────────
# (module path, what the module calls its space map, the build call it uses)
FILES = [
    ('experiments.noise.n02_sample_weighting', 'SPACE_FOR_FAMILY',
     _build_via_build_fold_hybrid),
    ('experiments.noise.n07_remediation', 'SPACE_FOR_FAMILY',
     _build_via_build_fold_hybrid),
    ('experiments.noise.n12_importance_weighted_distance', 'SPACE_FOR_FAMILY',
     _build_via_build_fold_hybrid),
    ('experiments.noise.n17_replicate_removal', 'SPACE_FOR_FAMILY',
     _build_via_hybrid_for_fold),
    ('experiments.noise.n19_measurement_count', 'SPACE_FOR_FAMILY',
     _build_via_hybrid_for_fold),
    ('experiments.noise.n23_uncertainty_cleaning', 'SPACE_FOR_FAMILY',
     _build_via_hybrid_for_fold),
    ('experiments.similarity.s10_importance_downstream', '_SPACE_FOR_FAMILY',
     _build_via_build_fold_hybrid),
]

# The two similarity files that name their second thing differently: s06 adds
# two ARMS beside its existing two, and s08 scores one CELL per family rather
# than naming a space at all.
ARM_FILES = [
    ('experiments.similarity.s06_applicability_domain',
     ('SIM_ARM', 'SIM_ARM_W'), ('hybrid_similarity',
                                'hybrid_similarity_weighted')),
]


def check_space_map(mod_path, attr):
    import importlib
    mod = importlib.import_module(mod_path)
    spaces = getattr(mod, attr, None)
    check('names a space for each family',
          isinstance(spaces, dict)
          and set(spaces) == {'predictive', 'similarity'}, spaces)
    if not isinstance(spaces, dict):
        return None
    check("the predictive space is still called 'hybrid'",
          spaces.get('predictive') == 'hybrid', spaces.get('predictive'))
    check('the two spaces have different names',
          spaces.get('predictive') != spaces.get('similarity'),
          f"{spaces.get('predictive')} vs {spaces.get('similarity')}")
    return spaces


def check_resolution(spaces):
    bp_p, bp_s = _predictive_bp(), _similarity_bp()
    both = POOL_SUBSETS['predictive_and_similarity']

    recipes, unavailable = pool_recipes_by_family(
        _meta(bp_p, bp_s), spaces, eligible_families=both)
    check('both families give a recipe',
          [r['space'] for r in recipes]
          == [spaces['predictive'], spaces['similarity']],
          [r['space'] for r in recipes])
    check('nothing is left unexplained', unavailable == {}, unavailable)
    if len(recipes) == 2:
        check('the two recipes are different objects',
              recipes[0]['blueprint'] is not recipes[1]['blueprint'],
              'different' if recipes[0]['blueprint']
              is not recipes[1]['blueprint'] else 'THE SAME OBJECT')
        check('they differ in features and in budget split',
              (recipes[0]['budget'], recipes[0]['allocation_method'])
              != (recipes[1]['budget'], recipes[1]['allocation_method']),
              f"{recipes[0]['budget']} features/"
              f"{recipes[0]['allocation_method']} vs "
              f"{recipes[1]['budget']} features/"
              f"{recipes[1]['allocation_method']}")

    recipes_one, unavailable_one = pool_recipes_by_family(
        _meta(bp_p), spaces, eligible_families=both)
    check('without the similarity family only the predictive space is built',
          [r['space'] for r in recipes_one] == [spaces['predictive']],
          [r['space'] for r in recipes_one])
    check('the predictive recipe is unchanged',
          recipes_one and recipes_one[0]['blueprint'] is bp_p,
          'same object' if (recipes_one
                            and recipes_one[0]['blueprint'] is bp_p) else 'NO')
    _why = unavailable_one.get(spaces['similarity'], '')
    check('the missing space carries a written reason',
          'objective=similarity' in _why, _why)


def check_matrices(builder):
    bp_p, bp_s = _predictive_bp(), _similarity_bp()
    X_p, X_s = _matrices_through(builder, bp_p, bp_s)
    check('the predictive recipe builds', X_p is not None,
          None if X_p is None else X_p.shape)
    check('the geometry recipe builds', X_s is not None,
          None if X_s is None else X_s.shape)
    if X_p is None or X_s is None:
        return
    identical = (X_p.shape == X_s.shape and np.allclose(X_p, X_s))
    check('the two spaces are NOT byte-identical', not identical,
          f"{X_p.shape} vs {X_s.shape}, "
          f"{'IDENTICAL VALUES' if identical else 'different'}")


def main():
    print("The nine files that asked for two search families and built one.\n")

    for mod_path, attr, builder in FILES:
        print(f"  {mod_path.rsplit('.', 1)[-1]}")
        spaces = check_space_map(mod_path, attr)
        if spaces:
            check_resolution(spaces)
        print()

    # The build path is shared, so it is exercised once per distinct call
    # rather than once per file.
    print("  the per-fold build, through hybrid_for_fold (n17, n19, n23)")
    check_matrices(_build_via_hybrid_for_fold)
    print()
    print("  the per-fold build, through build_fold_hybrid "
          "(n02, n07, n12, s06, s08, s10)")
    check_matrices(_build_via_build_fold_hybrid)
    print()

    for mod_path, attrs, expected in ARM_FILES:
        import importlib
        print(f"  {mod_path.rsplit('.', 1)[-1]}")
        mod = importlib.import_module(mod_path)
        got = tuple(getattr(mod, a, None) for a in attrs)
        check('names the two arms it adds', got == expected, got)
        print()

    print("  s08_optimal_dimension")
    import importlib
    s08 = importlib.import_module(
        'experiments.similarity.s08_optimal_dimension')
    check('asks for both families',
          tuple(POOL_SUBSETS[s08.POOL_SUBSET])
          == ('predictive', 'similarity'),
          POOL_SUBSETS[s08.POOL_SUBSET])
    check("results['budgets'] still holds the predictive family",
          s08.PRIMARY_FAMILY == 'predictive', s08.PRIMARY_FAMILY)
    print()

    # The five the handoff listed that already build both families, so a later
    # change cannot quietly undo them without this failing.
    print("  already building both families before 2026-09-19")
    from experiments.noise import n06_bindingdb_ground_truth as n06
    check('n06 maps both families to a space name',
          set(n06.SPACE_FOR_FAMILY) == {'predictive', 'similarity'},
          n06.SPACE_FOR_FAMILY)
    import inspect
    src = inspect.getsource(n06.run)
    check('n06 loops over by_objective and fills a space per family',
          "for o, d in (meta.get('by_objective') or {}).items():" in src
          and "spaces[name] = ('hybrid', o_bp, o_aug)" in src,
          'both lines present' if "spaces[name] = ('hybrid', o_bp, o_aug)"
          in src else 'MISSING')
    for _m, _needle in (
            ('experiments.similarity.s02_hybrid_similarity',
             "['predictive', 'similarity']"),
            ('experiments.similarity.s04_activity_cliffs',
             "h['objective'], h['rank']"),
            ('experiments.similarity.s05_knn_prediction',
             "(h['objective'], h['rank'])"),
            ('experiments.similarity.s13_clustering_retrieval',
             "f\"hybrid_{h['objective']}_r{h['rank']}\"")):
        _src = inspect.getsource(importlib.import_module(_m))
        check(f"{_m.rsplit('.', 1)[-1]} still keys its arms by family",
              _needle in _src, 'yes' if _needle in _src else 'GONE')
    print()

    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
