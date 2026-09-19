#!/usr/bin/env python3
"""Does the family loop give one space per family, and a reason for the rest?

`pool_recipes_by_family` is the loop thirteen experiments call to turn one pool
lookup into one feature space per search family. This checks the four cases it
has to get right, on hand-built metadata dicts — no dataset, no search, no fits.

    python tests/smoke/smoke_pool_recipes_by_family.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments._shared import (  # noqa: E402
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)


class _BP:
    """Stands in for a Blueprint: the fields the loop reads off one."""

    def __init__(self, objective, budget, alloc):
        self.objective = objective
        self.budget = budget
        self.allocation_method = alloc
        self.importance_method = 'lightgbm_gain'
        self.internal_evaluator = 'rf'
        self.augmentation_strategy = None


SPACES = {'predictive': 'hybrid', 'similarity': 'hybrid_similarity'}
BOTH = POOL_SUBSETS['predictive_and_similarity']


def _meta(families):
    """A metadata dict shaped like get_best_search_for_downstream's."""
    return {
        'objective': 'predictive',
        'tier': 'general', 'aug_condition': 'none',
        'score_mean': 0.63, 'metric_name': 'rmse', 'lower_is_better': True,
        'best_config': {'blueprint': families.get('predictive')},
        'by_objective': {
            o: {'tier': 'general', 'aug_condition': 'none',
                'score_mean': 0.63 if o == 'predictive' else 0.41,
                'metric_name': 'rmse' if o == 'predictive' else 'readacross',
                'lower_is_better': o == 'predictive',
                'best_config': {'blueprint': bp}}
            for o, bp in families.items()},
    }


def main():
    failures = []

    def check(name, cond, saw):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}: {saw}")
        if not cond:
            failures.append(name)

    print("1. both families in the pool")
    bp_p = _BP('predictive', 100, 'performance_weighted')
    bp_s = _BP('similarity', 250, 'equal')
    recipes, unavailable = pool_recipes_by_family(
        _meta({'predictive': bp_p, 'similarity': bp_s}), SPACES,
        eligible_families=BOTH)
    names = [r['space'] for r in recipes]
    check('two spaces', names == ['hybrid', 'hybrid_similarity'], names)
    check('nothing unavailable', unavailable == {}, unavailable)
    check('each recipe keeps its own family',
          [r['objective'] for r in recipes] == ['predictive', 'similarity'],
          [r['objective'] for r in recipes])
    # The failure this whole change exists to catch: a geometry space that
    # silently resolved to the predictive recipe. Different budget, different
    # allocation method, different object.
    check('the two recipes are not the same object',
          recipes[0]['blueprint'] is not recipes[1]['blueprint'],
          f"{id(recipes[0]['blueprint'])} vs {id(recipes[1]['blueprint'])}")
    check('the two recipes differ in budget and allocation method',
          (recipes[0]['budget'], recipes[0]['allocation_method']) !=
          (recipes[1]['budget'], recipes[1]['allocation_method']),
          f"{recipes[0]['budget']}/{recipes[0]['allocation_method']} vs "
          f"{recipes[1]['budget']}/{recipes[1]['allocation_method']}")
    check('the pool winner is named',
          [r['is_pool_winner'] for r in recipes] == [True, False],
          [r['is_pool_winner'] for r in recipes])
    check('each score keeps its own metric name',
          [r['search_metric_name'] for r in recipes] == ['rmse', 'readacross'],
          [r['search_metric_name'] for r in recipes])

    print("2. the similarity family is absent from the pool")
    recipes2, unavailable2 = pool_recipes_by_family(
        _meta({'predictive': bp_p}), SPACES, eligible_families=BOTH)
    check('one space', [r['space'] for r in recipes2] == ['hybrid'],
          [r['space'] for r in recipes2])
    check('the predictive recipe is the same one',
          recipes2[0]['blueprint'] is bp_p, recipes2[0]['blueprint'].budget)
    check('the missing space carries a reason',
          'hybrid_similarity' in unavailable2
          and 'no search under objective=similarity' in
              unavailable2['hybrid_similarity'],
          unavailable2.get('hybrid_similarity'))

    print("3. the similarity family is in the pool with no recipe on it")
    m3 = _meta({'predictive': bp_p, 'similarity': bp_s})
    m3['by_objective']['similarity']['best_config'] = {'blueprint': None}
    recipes3, unavailable3 = pool_recipes_by_family(
        m3, SPACES, eligible_families=BOTH)
    check('one space', [r['space'] for r in recipes3] == ['hybrid'],
          [r['space'] for r in recipes3])
    check('the reason says the family was there',
          'carried no blueprint' in unavailable3.get('hybrid_similarity', ''),
          unavailable3.get('hybrid_similarity'))

    print("4. the run asked for the predictive family alone")
    recipes4, unavailable4 = pool_recipes_by_family(
        _meta({'predictive': bp_p}), SPACES,
        eligible_families=POOL_SUBSETS['predictive_only'])
    check('one space', [r['space'] for r in recipes4] == ['hybrid'],
          [r['space'] for r in recipes4])
    check('the reason says it was never asked for',
          'not in the pool subset' in unavailable4.get('hybrid_similarity', ''),
          unavailable4.get('hybrid_similarity'))

    print("5. the pool returned nothing at all")
    recipes5, unavailable5 = pool_recipes_by_family(
        None, SPACES, eligible_families=BOTH)
    check('no spaces', recipes5 == [], recipes5)
    check('both spaces carry a reason',
          set(unavailable5) == {'hybrid', 'hybrid_similarity'},
          sorted(unavailable5))

    print("6. metadata from before by_objective existed")
    m6 = _meta({'predictive': bp_p})
    m6.pop('by_objective')
    recipes6, unavailable6 = pool_recipes_by_family(m6, SPACES,
                                                    eligible_families=BOTH)
    check('the one family it holds still builds',
          [r['space'] for r in recipes6] == ['hybrid'],
          [r['space'] for r in recipes6])

    print("7. the sentence each family is described by")
    check('similarity says what is computed',
          'five nearest neighbours' in describe_pool_family('similarity'),
          describe_pool_family('similarity'))

    print()
    if failures:
        print(f"FAILED: {len(failures)} of the checks above: {failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
