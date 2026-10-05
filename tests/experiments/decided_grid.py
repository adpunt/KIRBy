"""The owner's decided search grid per dataset (DECISIONS.md, 2026-10-03).

One question, answered in one place: for this dataset and this search
('general' or 'general_distance'), which downstream models, budgets,
importance methods, allocation methods, internal evaluators, representation
counts and augmentation conditions does a search run?

The answer comes from decided_grid.json, which build_decided_grid.py writes
from scratchpad/precompute_cuts/DECISIONS.md (uncommitted, so not on the
server). Every find_best_hybrid call an experiment makes goes through
apply_decided_grid(), via _shared._apply_dataset_search_speed, so the cuts
cannot differ between experiments or between the general and distance search.

A dataset or search the record does not cover raises UndecidedSearch. It does
not fall back to the engine defaults: those are the five-model, eight-budget
grid the record replaced.
"""
import json
from functools import lru_cache
from pathlib import Path

GRID_FILE = Path(__file__).resolve().parent / 'decided_grid.json'

# The axes find_best_hybrid takes by name, in the order they are printed.
AXES = ('models', 'budgets', 'importance_methods', 'allocation_methods',
        'internal_evaluators', 'rep_counts')


class UndecidedSearch(KeyError):
    """The record has no decision for this dataset or this search."""


@lru_cache(maxsize=1)
def _table():
    return json.loads(GRID_FILE.read_text())['datasets']


def search_kind(tier) -> str:
    """'general_distance' for any distance-lineage tier, else 'general'.

    Any tier name containing 'distance' counts: experiment-local distance
    tiers (c15c_fast_distance, top_distance, ...) do not start with
    'general_distance' and would otherwise get the general grid.
    """
    return 'general_distance' if 'distance' in str(tier or '') else 'general'


def decided_search(dataset_name, tier='general') -> dict:
    """The decided grid for one dataset and one search, as a fresh dict.

    Keys: models, budgets, importance_methods, allocation_methods,
    internal_evaluators, rep_counts (None = the whole pool), aug_conditions,
    speed.
    """
    kind = search_kind(tier)
    entry = _table().get(dataset_name)
    if entry is None:
        raise UndecidedSearch(
            f'{dataset_name!r} has no row in DECISIONS.md (Step 1). Its two '
            f'models and its cuts are undecided, so no search is run for it.')
    if kind not in entry:
        raise UndecidedSearch(
            f'{dataset_name!r}: DECISIONS.md Step 7 has no row for the distance '
            f'search, so its distance search is undecided.')
    return json.loads(json.dumps(entry[kind]))


def decided_models(dataset_name, tier='general'):
    return decided_search(dataset_name, tier)['models']


def decided_aug_conditions(dataset_name, tier='general'):
    return decided_search(dataset_name, tier)['aug_conditions']


def is_decided(dataset_name, tier='general') -> bool:
    try:
        decided_search(dataset_name, tier)
        return True
    except UndecidedSearch:
        return False


def apply_decided_grid(kwargs, dataset_name, verbose=True):
    """Fill or narrow a find_best_hybrid kwargs dict to the decided grid.

    An axis the caller left as None gets the decided list. An axis the caller
    named is narrowed to the decided members it contains; what was dropped is
    printed. A named axis with no decided member left raises ValueError, so a
    caller asking for something the record cut fails at once instead of
    running a different search.

    speed is set to the dataset's class (fast for the ten fast-tier datasets).
    With every axis named it only labels the search; it changes no list.
    """
    grid = decided_search(dataset_name, kwargs.get('tier'))
    for axis in AXES:
        want = grid[axis]
        have = kwargs.get(axis)
        if have is None:
            kwargs[axis] = list(want)
            continue
        if isinstance(have, dict):
            # A {name: estimator} models dict: narrow by name, keep the objects.
            kept_d = {k: v for k, v in have.items() if k in want}
            dropped = [k for k in have if k not in want]
            if not kept_d:
                raise ValueError(
                    f'{dataset_name}: models={list(have)} has nothing '
                    f'DECISIONS.md keeps (decided: {want}).')
            if dropped and verbose:
                print(f'  [decided grid] {dataset_name}: models {dropped} not '
                      f'in DECISIONS.md; searching {list(kept_d)}')
            kwargs[axis] = kept_d
            continue
        have = list(have)
        kept = [x for x in have if x in want]
        dropped = [x for x in have if x not in want]
        if not kept:
            raise ValueError(
                f'{dataset_name} {search_kind(kwargs.get("tier"))}: {axis}={have} '
                f'has nothing DECISIONS.md keeps (decided: {want}).')
        if dropped and verbose:
            print(f'  [decided grid] {dataset_name}: {axis} {dropped} not in '
                  f'DECISIONS.md; searching {kept}')
        kwargs[axis] = kept
    kwargs['speed'] = grid['speed']
    return kwargs


def fast_model_search_kwargs(dataset_name, model, tier='general'):
    """find_best_hybrid axes for ONE extra model on the fast tier, inside the
    dataset's decided cuts.

    Adelaide, 2026-10-05, for n09: its methods train neural networks, so dnn is
    fixed by the question. On a dataset whose two models do not include dnn,
    n09 runs a per-fold dnn search on "the fast tier with whatever cuts are
    already in place for that dataset". Each axis is the fast list intersected
    with the decided list; where that leaves nothing, the decided list is used
    and the axis is named in 'fast_fell_back'. Budgets are the decided ones.
    The model list is [model] only, and speed is 'fast'.

    Pass the result to find_best_hybrid directly (or through
    search_on_fold(..., fixed_grid=...)); do NOT run it through
    apply_decided_grid, which refuses a model outside the decided two.
    """
    from kirby.find_best_hybrid import (
        FAST_ALLOCATION_METHODS, FAST_IMPORTANCE_METHODS,
        FAST_INTERNAL_EVALUATORS, FAST_REP_COUNTS)
    grid = decided_search(dataset_name, tier)
    fast = {'allocation_methods': FAST_ALLOCATION_METHODS,
            'importance_methods': FAST_IMPORTANCE_METHODS,
            'internal_evaluators': FAST_INTERNAL_EVALUATORS,
            'rep_counts': FAST_REP_COUNTS}
    kw, fell_back = {}, []
    for axis, fast_list in fast.items():
        kept = [x for x in grid[axis] if x in fast_list]
        if not kept:
            kept = list(grid[axis])
            fell_back.append(axis)
        kw[axis] = kept
    kw.update(models=[model], budgets=list(grid['budgets']), speed='fast',
              tier=tier)
    return kw, fell_back

