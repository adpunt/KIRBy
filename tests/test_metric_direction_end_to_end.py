"""Phase 0.2 gate: does a genuine win come out POSITIVE, whichever way the metric points?

`delta_mean_of_folds` is the number KIRBy reports. It is a paired per-fold
difference, sign-corrected by `lower_is_better` — so one wrong direction flag
anywhere between `_resolve_metric` and the aggregation turns a win into a loss
and nothing raises.

The 2026-07-15 rebuild roadmap called this a blocking gate and it was never
built. `test_metric_direction_persisted.py` checks that a saved result FILE
records its direction; it does not run a search, so it cannot see a sign flip
inside one. This does.

It found nothing on the predictive paths. It DOES cover the case that was
broken on 2026-09-03: under a blend containing prediction, `lower_is_better` is
flipped to False because the blend is a mean of correlations, while
`SearchResult.baselines` still holds the dataset's own metric — so the
comparator was picked with `max()` over a set of RMSEs, i.e. the WORST
representation.

Data is synthetic and tiny; the whole file runs in well under a minute with no
cache and no cluster.

Run: python -m pytest tests/test_metric_direction_end_to_end.py -q
"""
import numpy as np
import pytest

from kirby.find_best_hybrid import nested_cv_evaluate


def _planted(seed=0, n=300, n_groups=30):
    """One representation carries the label, two are noise.

    A hybrid that can pick features has something real to find, and the best
    single representation is unambiguously the planted one — so a genuine win
    is expected and its SIGN is the thing under test.
    """
    rng = np.random.RandomState(seed)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    y = rng.randn(n)
    reps = {
        'signal': np.column_stack([y + rng.randn(n) * 0.35, rng.randn(n, 9)]),
        'noise_a': rng.randn(n, 12),
        'weak': np.column_stack([y * 0.4 + rng.randn(n), rng.randn(n, 9)]),
    }
    return reps, y, groups


SEARCH = dict(
    budgets=[8], models=['rf'], task_type='regression', n_folds=3,
    allocation_methods=['performance_weighted'],
    importance_methods=['lightgbm_gain'], internal_evaluators=['rf'],
    cache=False, verbose=False,
)


def _run(metric, objective=None, seed=0):
    reps, y, groups = _planted(seed)
    kw = dict(SEARCH)
    if objective is not None:
        kw['objective'] = objective
    res = nested_cv_evaluate(reps, y, groups, n_outer_folds=3,
                             metric=metric, **kw)
    return res.by_model['rf']


def _sign_is_consistent(entry):
    """The reported delta must agree in sign with the per-fold differences.

    Recomputed here from the raw fold scores, so a flip introduced anywhere in
    the aggregation shows up as a disagreement rather than as a plausible number.
    """
    per_fold = [d for d in entry['per_fold_delta'] if d is not None]
    assert per_fold, 'no fold produced a paired difference'
    recomputed = float(np.mean(per_fold))
    assert np.isclose(recomputed, entry['delta_mean_of_folds']), (
        f"delta_mean_of_folds={entry['delta_mean_of_folds']} but the mean of "
        f"the per-fold differences is {recomputed}")
    return recomputed


@pytest.mark.parametrize('metric', ['rmse', 'spearman'])
def test_delta_sign_matches_the_raw_fold_scores(metric):
    """Positive delta must mean the hybrid actually scored better, both ways.

    rmse is lower-is-better, spearman is higher-is-better. The delta is
    sign-corrected, so in BOTH cases positive means the hybrid won — and that
    has to agree with the raw hybrid and comparator scores it was built from.
    """
    e = _run(metric)
    _sign_is_consistent(e)

    lower_is_better = (metric == 'rmse')
    for f, (h, s) in enumerate(zip(e['hybrid_fold_scores'],
                                   e['best_single_fold_scores'])):
        if h is None or s is None:
            continue
        hybrid_won = (h < s) if lower_is_better else (h > s)
        delta = e['per_fold_delta'][f]
        assert delta is not None
        assert (delta > 0) == hybrid_won, (
            f"[{metric}, fold {f}] hybrid={h}, best single={s}, "
            f"hybrid_won={hybrid_won}, but delta={delta}")


@pytest.mark.parametrize('metric', ['rmse', 'spearman'])
def test_comparator_is_the_best_representation_not_the_worst(metric):
    """The hybrid is compared against the BEST single representation.

    The comparator is chosen from the inner search's own baselines, which are in
    the dataset's metric — so the pick has to follow THAT metric's direction. It
    is the one place a direction flag is read against a quantity it does not
    describe, and it is where the blend case broke.
    """
    e = _run(metric)
    picks = [p for p in e['best_single_reps_per_fold'] if p is not None]
    assert picks, 'no comparator was picked on any fold'
    assert all(p == 'signal' for p in picks), (
        f"[{metric}] the comparator should be the planted representation on "
        f"every fold; got {picks}. Picking 'noise_a' means the direction was "
        f"read backwards.")


def test_blend_comparator_follows_the_baselines_metric():
    """A blend flips the delta's direction but NOT the baselines'.

    Under a blend containing prediction the winner is a mean of correlations, so
    `lower_is_better` becomes False. The baselines it picks the comparator from
    are still RMSE. Using the delta's direction there took the max of a set of
    RMSEs — the worst representation — and the hybrid was measured against it.
    """
    e = _run('rmse', objective={'predictive': 0.5, 'similarity': 0.5})
    picks = [p for p in e['best_single_reps_per_fold'] if p is not None]
    assert picks, 'no comparator was picked on any fold'
    assert all(p == 'signal' for p in picks), (
        f"under a blend on an rmse dataset the comparator should still be the "
        f"planted representation; got {picks}. 'noise_a' means the RMSE "
        f"baselines were maximised instead of minimised.")
    _sign_is_consistent(e)


def test_geometry_delta_is_higher_is_better():
    """A geometry objective has no dataset metric — it is always higher-better.

    The metric passed in is rmse, and it must NOT reach the delta's direction:
    `similarity` is a Spearman correlation and a bigger one is better.
    """
    e = _run('rmse', objective='similarity')
    _sign_is_consistent(e)
    for f, (h, s) in enumerate(zip(e['hybrid_fold_scores'],
                                   e['best_single_fold_scores'])):
        if h is None or s is None:
            continue
        assert (e['per_fold_delta'][f] > 0) == (h > s), (
            f"[similarity, fold {f}] hybrid={h}, best single={s}, "
            f"delta={e['per_fold_delta'][f]} — a geometry score is "
            f"higher-is-better whatever `metric` says.")


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
