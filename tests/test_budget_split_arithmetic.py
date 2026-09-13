"""What the budget split must do, whatever the weighting scheme decides.

These are the three properties an allocation has to have for a weighting knob to
mean anything, and two of them were false until 2026-09-10:

  1. A representation the weights gave a non-zero share to gets at least ONE
     feature. Zero features means "not in the hybrid", which is a statement about
     the representation — arriving there by rounding is an accident of the budget.
  2. The counts sum to EXACTLY the budget.
  3. Adding a WORSE representation to the pool DOES change what the others get,
     and that is a decision, not a defect — measured by f32 on 9 datasets and
     rejected on 2026-09-13 because the alternative predicted held-out molecules
     worse. The third test asserts today's behaviour so nobody changes it by
     accident.

Properties 1 and 2 were real defects. Property 1 was broken by the rounding
repair: it took a representation's whole allocation with no floor. Property 2 was
met, but by a rule that overrode the weighting — the shortfall went to the single
best-scoring representation. Both are now largest-remainder with a one-feature
floor.

Synthetic, no cache, no cluster, runs in about a second.

Run: python -m pytest tests/test_budget_split_arithmetic.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kirby.hybrid import allocate_performance_weighted


def _pool(widths, n_rows=60, seed=0):
    """A representation dict of the given feature widths."""
    rng = np.random.RandomState(seed)
    return {f'r{i}': rng.normal(size=(n_rows, w)).astype(np.float32)
            for i, w in enumerate(widths)}


def _split(scores, widths, budget, **kw):
    """The allocation these scores produce, with the scoring loop bypassed.

    precomputed_scores is the path a real search takes: find_best_hybrid scores
    every representation once per fold and hands the vector in.
    """
    reps = _pool(widths)
    names = list(reps)
    rng = np.random.RandomState(1)
    y = rng.normal(size=60)
    return allocate_performance_weighted(
        reps, reps, y, y, total_budget=budget,
        precomputed_scores={n: s for n, s in zip(names, scores)},
        **kw)


# -- Property 2: the counts sum to the budget -----------------------------------

@pytest.mark.parametrize('budget', [10, 25, 50, 100, 250, 500, 1000])
def test_allocation_sums_to_budget(budget):
    scores = [-0.30, -0.31, -0.32, -0.33, -0.34, -0.35, -0.36, -0.37, -0.38, -0.39]
    alloc = _split(scores, [512] * 10, budget)
    assert sum(alloc.values()) == budget, alloc


def test_allocation_sums_to_budget_when_reps_are_narrow():
    """Narrow representations cannot hold their share; the rest must absorb it."""
    scores = [-0.30, -0.31, -0.32, -0.33, -0.34, -0.35, -0.36, -0.37, -0.38, -0.39]
    alloc = _split(scores, [32, 32, 512, 512, 512, 512, 512, 512, 512, 512], 200)
    assert sum(alloc.values()) == 200, alloc
    for name, n in alloc.items():
        assert n <= 512
    assert alloc['r0'] <= 32 and alloc['r1'] <= 32


# -- Property 1: a non-zero weight keeps at least one feature -------------------

def test_no_rep_is_rounded_out_of_the_hybrid():
    """400 random draws. Before 2026-09-10 this zeroed a rep in 39 of them."""
    rng = np.random.RandomState(0)
    for draw in range(400):
        scores = list(-rng.uniform(0.2, 0.9, size=10))
        alloc = _split(scores, [512] * 10, 100)
        assert sum(alloc.values()) == 100, (draw, alloc)
        zeroed = [n for n, v in alloc.items() if v == 0]
        # The WORST representation legitimately reaches zero: the shift puts it
        # at exactly zero weight. Nothing else may.
        worst = f'r{int(np.argmin(scores))}'
        assert zeroed in ([], [worst]), (draw, scores, alloc)


def test_a_tiny_share_survives_a_small_budget():
    """One clearly better rep and nine close together, at the smallest budget."""
    scores = [-0.10] + [-0.80 - 0.001 * i for i in range(9)]
    alloc = _split(scores, [512] * 10, 10)
    assert sum(alloc.values()) == 10, alloc
    assert alloc['r0'] >= 1
    non_zero_weighted = [n for n, v in alloc.items() if v > 0]
    assert len(non_zero_weighted) >= 2, alloc


# -- Property 3: DELIBERATELY NOT TRUE -----------------------------------------

def test_adding_a_worse_rep_DOES_resize_the_others_and_that_is_the_decision():
    """A worse representation joining the pool resizes everyone else. On purpose.

    `_sanitize_baseline_scores` shifts the score vector up by its own minimum, so
    on a metric whose scores are all negative each share is proportional to the
    gap to the WORST representation in the pool, and that one always lands on
    zero features. Adding a worse representation therefore changes what everyone
    else gets.

    THIS WAS MEASURED AND KEPT. 2026-09-13, owner's decision. The alternative was
    to convert each score to a skill score first -- how much better than
    predicting the training mean -- so that zero means no signal rather than
    "worst in this pool". f32 ran both on 9 datasets and 45 folds
    (tests/results/framework/f32_*__fold*.json). On the 226 cells where the two
    rules produced a different split, the skill score predicted held-out
    molecules BETTER in 85 and WORSE in 133. Every metric, every weighting
    scheme and both budgets came out worse; five of six datasets came out worse.
    Even where today's rule starves two or more representations -- the failure
    the skill score exists to fix -- it is a coin flip, not a win.

    So this test asserts today's behaviour, and it exists to stop someone
    "fixing" it without re-running f32. If you are here because this test
    failed: you have changed the budget split's normalisation, and the question
    is whether you have evidence that beats f32's.
    """
    scores = [-0.62, -0.66, -0.70, -0.74, -0.78, -0.82, -0.86, -0.90]
    before = _split(scores, [512] * 8, 1000)
    after = _split(scores + [-0.95], [512] * 9, 1000)

    # The worst representation in each pool gets nothing.
    assert before['r7'] == 0, before
    assert after['r8'] == 0, after

    # And everyone else moved, because the origin moved with the new worst one.
    moved = [i for i in range(len(scores)) if before[f'r{i}'] != after[f'r{i}']]
    assert moved, (
        "the shares did NOT move — the normalisation has been changed. See this "
        "test's docstring: that change was measured and rejected on 2026-09-13.")
