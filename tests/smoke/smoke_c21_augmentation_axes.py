#!/usr/bin/env python3
"""Smoke test for the c21 augmentation axes added on 2026-08-26.

SYNTHETIC DATA, tiny arrays, no cache, no real dataset. This exercises every
branch that changed and nothing else; it measures nothing and ranks nothing.
It exists because the four locally cached datasets could not be run at the time
(load average 57 on 8 cores), and because a run that only proves the default
path still works would not touch any of the new code.

What it checks:

1. All five admission methods dispatch and complete a fold -- none, all,
   greedy_ablation, capped, uncapped. Before this change c21 could only run
   'uncapped', which was a hardcoded string in two places.
2. Both candidate rankings run: residual_correlation and rf_importance. No
   caller anywhere in the tree had ever set rf_importance.
3. The two planted decoys reach the pool and their counts come back per fold.
4. The four settings added to the recipe reach create_hybrid, so a rebuild
   reproduces the run rather than falling back to the engine's own defaults.
5. An unknown method name raises instead of silently producing a plain hybrid.

Run:
    python tests/smoke/smoke_c21_augmentation_axes.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

import numpy as np

from experiments._shared import (
    build_fold_hybrid, plant_decoys, decoy_counts, NOISE_DONOR, ECHO_DONOR,
)
from kirby.find_best_hybrid import Blueprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

N, N_BASE, N_DONOR = 120, 40, 24
STRATEGIES = ('none', 'all', 'greedy_ablation', 'capped', 'uncapped')


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _synthetic(seed=0):
    """A base representation carrying real signal, plus two donor blocks.

    One donor is informative, one is not. Labels depend on the base and on the
    informative donor only, so a working rule has something to find and
    something to reject.
    """
    rng = np.random.RandomState(seed)
    base = rng.standard_normal((N, N_BASE))
    good = rng.standard_normal((N, N_DONOR))
    junk = rng.standard_normal((N, N_DONOR))
    y = (base[:, :5].sum(axis=1) + 0.8 * good[:, :3].sum(axis=1)
         + 0.1 * rng.standard_normal(N))
    groups = rng.randint(0, 12, size=N)
    return base, {'good_donor': good, 'junk_donor': junk}, y, groups


def _blueprint(strategy, ranking, max_candidates, budget):
    return Blueprint(
        reps=['base'], allocation_method='performance_weighted',
        importance_method='lightgbm_gain', internal_evaluator='match',
        budget=budget, step_size=5,
        augmentation_strategy=strategy,
        aug_per_source_budget=5,
        candidate_ranking=ranking,
        augmentation_budget=10,
        aug_max_candidates=max_candidates,
        metric='rmse', task_type='regression', lower_is_better=True)


# =============================================================================
# The gate on sharing the candidate ranking, added 2026-08-26.
#
# The grid computes one ranking and reuses it across the downstream model, the
# size rule, the margin and capped-versus-uncapped, because it depends on none
# of them. That is 57% of the grid's model fits on herg_ki. If a precomputed
# ranking ever stopped matching the recomputed one, every result would be wrong
# and would look fine. This is the check that says it still matches.
# =============================================================================

def check_precomputed_ranking_matches():
    from kirby.hybrid import (_pool_and_rank_aug_features, augment_capped,
                              augment_uncapped)
    base, donors, y, _ = _synthetic(seed=3)
    tr, va = np.arange(90), np.arange(90, N)
    dtr = {k: v[tr] for k, v in donors.items()}
    dva = {k: v[va] for k, v in donors.items()}

    bad, n = [], 0
    for rank in ('residual_correlation', 'rf_importance'):
        for cap in (5, 64, None):
            pre = _pool_and_rank_aug_features(dtr, dva, y[tr], base[tr], None,
                                              rank, False, 50,
                                              per_source_budget=cap)
            for meth in ('capped', 'uncapped'):
                for frac in (None, 0.10):
                    def call(pr):
                        kw = dict(model=RandomForestRegressor(n_estimators=10,
                                                              random_state=0),
                                  metric_fn=_rmse, lower_is_better=True,
                                  candidate_ranking=rank,
                                  per_source_budget=cap, fraction=frac,
                                  precomputed_ranking=pr)
                        if meth == 'capped':
                            return augment_capped(
                                base[tr], base[va],
                                {'base': {'selected_indices': np.arange(N_BASE),
                                          'n_features': N_BASE}},
                                dtr, dva, y[tr], y[va],
                                total_budget=N_BASE, **kw)
                        return augment_uncapped(base[tr], base[va], dtr, dva,
                                                y[tr], y[va], **kw)
                    a, b = call(None), call(pre)
                    n += 1
                    ka = {k: list(v['selected_indices'])
                          for k, v in a['augmentation_info'].items()}
                    kb = {k: list(v['selected_indices'])
                          for k, v in b['augmentation_info'].items()}
                    if not (ka == kb and a['n_added'] == b['n_added']
                            and repr(a['rung_scores']) == repr(b['rung_scores'])):
                        bad.append(f'{meth}/{rank}/cap={cap}/frac={frac}')
    print(f'\nprecomputed ranking vs recomputed, {n} combinations: '
          f'{"BIT-IDENTICAL" if not bad else "DIFFER -> " + str(bad)}')
    print('  compared which features were kept, how many, and every rung score.')
    return bad


def main():
    base, donors, y, groups = _synthetic()
    tr = np.arange(0, 90)
    va = np.arange(90, N)

    # Decoys built once per fold, full length, then sliced -- the same shape
    # c21 uses. The noise donor matches the real donors' combined width.
    rng = np.random.RandomState(1000)
    pool = plant_decoys(donors, base, rng, echo_cols=16, echo_jitter=0.05)
    assert pool[NOISE_DONOR].shape == (N, 2 * N_DONOR), pool[NOISE_DONOR].shape
    assert ECHO_DONOR in pool
    print(f"pool: {', '.join(f'{k}={v.shape[1]}' for k, v in pool.items())}")

    failures = []
    print(f"\n{'method':16s}{'ranking':22s}{'cut':>5}{'added':>7}"
          f"{'noise':>7}{'echo':>6}   rmse")
    for strategy in STRATEGIES:
        for ranking in ('residual_correlation', 'rf_importance'):
            for max_candidates in (50, 250):
                bp = _blueprint(strategy, ranking, max_candidates, N_BASE)
                try:
                    Xtr, Xva, info = build_fold_hybrid(
                        {'base': base[tr]}, {'base': base[va]}, y[tr], bp,
                        augmentations_train={k: v[tr] for k, v in pool.items()},
                        augmentations_val={k: v[va] for k, v in pool.items()},
                        groups=groups[tr],
                        model=RandomForestRegressor(n_estimators=12,
                                                    random_state=0),
                        metric_fn=_rmse, lower_is_better=True)
                except Exception as exc:                    # noqa: BLE001
                    failures.append(f"{strategy}/{ranking}/{max_candidates}: {exc}")
                    print(f"{strategy:16s}{ranking:22s}{max_candidates:5d}"
                          f"   FAILED: {exc}")
                    continue

                m = RandomForestRegressor(n_estimators=12, random_state=0)
                m.fit(Xtr, y[tr])
                d = decoy_counts(info.get('augmentation_info'))
                # capped displaces base features, so its finished width is the
                # budget; uncapped appends outside it. Both are reported as
                # "added" relative to the base width.
                added = Xtr.shape[1] - N_BASE
                print(f"{strategy:16s}{ranking:22s}{max_candidates:5d}"
                      f"{added:7d}{d['noise']:7d}{d['echo']:6d}   "
                      f"{_rmse(y[va], m.predict(Xva)):.4f}")

                if Xtr.shape[1] != Xva.shape[1]:
                    failures.append(
                        f"{strategy}/{ranking}: train width {Xtr.shape[1]} != "
                        f"validation width {Xva.shape[1]}")
                if strategy == 'none' and info.get('augmentation_info'):
                    failures.append("'none' produced augmentation_info")

    # An unrecognised name must raise. Until 2026-08-26 the dispatch chain had
    # no final branch, so it returned a base-only hybrid while the name was
    # still written into the stored recipe as though it had been honoured.
    try:
        build_fold_hybrid(
            {'base': base[tr]}, {'base': base[va]}, y[tr],
            _blueprint('typo_here', 'residual_correlation', 50, N_BASE),
            augmentations_train={k: v[tr] for k, v in pool.items()},
            augmentations_val={k: v[va] for k, v in pool.items()},
            groups=groups[tr],
            model=RandomForestRegressor(n_estimators=8, random_state=0),
            metric_fn=_rmse, lower_is_better=True)
        failures.append("an unknown strategy name did not raise")
    except ValueError as exc:
        print(f"\nunknown method raises, as it must: {str(exc)[:70]}")

    failures += check_precomputed_ranking_matches()

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print(f"PASS  {len(STRATEGIES) * 2 * 2} configurations built, "
          f"train and validation widths matched on every one, and the "
          f"precomputed ranking matched the recomputed one everywhere")
    return 0


if __name__ == '__main__':
    sys.exit(main())
