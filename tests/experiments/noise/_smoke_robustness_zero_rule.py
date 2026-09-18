#!/usr/bin/env python3
"""The zero rule in robustness_metrics, and c10's noise-level guard.

Owner's decision 2026-09-18: drop any curve whose clean score is at or below
zero, and keep no other floor. Retention divides by the clean score. A negative
one flips the sign of the whole curve, so a representation that gets BETTER
under noise reads as a clean decay; a zero one makes the division undefined.

Measured on ginkgo_hic, where `aac` is in the signed-off pool and scores -0.154
spearman on clean labels. Before this rule that curve came back with an
auc_norm of 0.4346 and nothing warned.

Also checks c10's own guard on the number of noise levels. It said three until
2026-09-18 while robustness_metrics needs four, so a three-level run passed the
check and published a table of nan.

Run: python3 tests/experiments/noise/_smoke_robustness_zero_rule.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from kirby.noise_robustness import robustness_metrics

FAILURES = []


def check(label, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ''))
    if not ok:
        FAILURES.append(label)


LEVELS = np.array([0.0, 0.15, 0.3, 0.5, 0.75, 1.0])


def test_negative_clean_score_is_refused():
    """The ginkgo_hic shape: starts below zero, climbs toward it under noise."""
    print("\nnegative clean score, higher is better")
    scores = np.array([-0.154, -0.120, -0.090, -0.060, -0.030, -0.005])
    r = robustness_metrics(LEVELS, scores, baseline_threshold=None,
                           lower_is_better=False)
    check('a clean score of -0.154 spearman gives auc_norm nan',
          np.isnan(r['auc_norm']), f"got {r['auc_norm']}")
    check('and no Weibull shape either',
          np.isnan(r['weibull_tau']) and np.isnan(r['weibull_beta']),
          f"tau={r['weibull_tau']} beta={r['weibull_beta']}")

    # The number it used to report, so the record says what changed.
    ret = scores / scores[0]
    check('the retention it used to build ran 1.000 down to 0.032 while the '
          'score rose from -0.154 to -0.005',
          np.isclose(ret[0], 1.0) and np.isclose(ret[-1], 0.0325, atol=1e-3),
          f"{np.round(ret, 3).tolist()}")


def test_zero_clean_score_is_refused():
    print("\nclean score of exactly zero")
    scores = np.array([0.0, 0.10, 0.08, 0.05, 0.02, 0.01])
    r = robustness_metrics(LEVELS, scores, baseline_threshold=None,
                           lower_is_better=False)
    check('a clean score of 0.0 gives auc_norm nan',
          np.isnan(r['auc_norm']), f"got {r['auc_norm']}")


def test_small_positive_clean_score_is_kept():
    """No floor above zero. abagym_EGFR `aac` scores 0.1014 clean and stays in."""
    print("\nsmall positive clean score, no other floor")
    for base, name in ((0.1014, 'abagym_EGFR aac'),
                       (0.2715, 'abcov_kd blosum'),
                       (0.2895, 'ddg_combined onehot')):
        scores = base * np.array([1.0, 0.9, 0.8, 0.65, 0.5, 0.4])
        r = robustness_metrics(LEVELS, scores, baseline_threshold=None,
                               lower_is_better=False)
        check(f'a clean score of {base:+.4f} spearman ({name}) is kept',
              np.isfinite(r['auc_norm']), f"auc_norm={r['auc_norm']:.4f}")


def test_lower_is_better_side_unchanged():
    print("\nerror metrics, lower is better")
    errs = np.array([0.50, 0.55, 0.62, 0.70, 0.80, 0.95])
    r = robustness_metrics(LEVELS, errs, baseline_threshold=None,
                           lower_is_better=True)
    check('an rmse of 0.50 clean, rising to 0.95, keeps its auc_norm',
          np.isfinite(r['auc_norm']), f"auc_norm={r['auc_norm']:.4f}")
    check('and retention stays at or below 1.0', r['auc_norm'] <= 1.0,
          f"{r['auc_norm']:.4f}")
    r0 = robustness_metrics(LEVELS, np.array([0.0, .1, .2, .3, .4, .5]),
                            baseline_threshold=None, lower_is_better=True)
    check('an rmse of exactly 0 clean is still refused',
          np.isnan(r0['auc_norm']), f"got {r0['auc_norm']}")


def test_explicit_threshold_still_works_for_callers_that_pass_one():
    """phase4_analysis passes its own accuracy and AUROC floors. Those stay."""
    print("\nthe opt-in floor, for callers that ask for one")
    scores = 0.25 * np.array([1.0, 0.9, 0.8, 0.65, 0.5, 0.4])
    kept = robustness_metrics(LEVELS, scores, baseline_threshold=None,
                              lower_is_better=False)
    gated = robustness_metrics(LEVELS, scores, baseline_threshold=0.3,
                               lower_is_better=False)
    check('a clean score of 0.25 is kept when no floor is asked for',
          np.isfinite(kept['auc_norm']), f"auc_norm={kept['auc_norm']:.4f}")
    check('and refused when a caller asks for a floor of 0.3',
          np.isnan(gated['auc_norm']), f"got {gated['auc_norm']}")
    gated_err = robustness_metrics(LEVELS, np.array([0.25, .3, .35, .4, .45, .5]),
                                   baseline_threshold=0.3, lower_is_better=True)
    check('a floor of 0.3 does NOT reach the error side',
          np.isfinite(gated_err['auc_norm']),
          f"auc_norm={gated_err['auc_norm']:.4f}")


def test_c10_refuses_three_levels():
    print("\nc10's guard on the number of noise levels")
    import importlib
    c10 = importlib.import_module('experiments.noise.c10_noise_robustness')
    import inspect
    src = inspect.getsource(c10)
    check('c10 refuses fewer than 4 levels', 'if len(lv) < 4:' in src)
    check('robustness_metrics needs 4 points, and three of them give nan',
          np.isnan(robustness_metrics(
              np.array([0.0, 0.5, 1.0]), np.array([0.6, 0.5, 0.4]),
              baseline_threshold=None)['auc_norm']))


def test_the_finished_run_is_unchanged():
    """The 171 curves c10 wrote for flab_thermo_tresanco all had positive clean
    scores, so the zero rule must leave every one of them exactly as reported."""
    print("\nthe flab_thermo_tresanco run already on disk")
    import json
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                        'c10_noise_robustness', 'flab_thermo_tresanco.json')
    if not os.path.exists(path):
        print("  [skip] no results file at tests/results/c10_noise_robustness/")
        return
    res = json.load(open(path))
    curves = []
    for byc in res['individual_reps'].values():
        curves += [c for c in byc.values() if not c.get('skipped')]
    curves += [c for c in res['best_single'].values() if not c.get('skipped')]
    for byc in res['hybrids'].values():
        curves += [c for c in byc.values() if not c.get('skipped')]
    moved, lowest = 0, 1.0
    for c in curves:
        sbl = c['scores_by_level']
        lv = np.array([float(k) for k in sbl], float)
        sc = np.array(list(sbl.values()), float)
        r = robustness_metrics(lv, sc, baseline_threshold=None,
                               lower_is_better=False)
        lowest = min(lowest, float(sc[np.isclose(lv, 0.0)][0]))
        if not np.isclose(r['auc_norm'], c['auc_norm']):
            moved += 1
    check(f'all {len(curves)} curves keep the auc_norm c10 reported',
          moved == 0, f"{moved} moved")
    check('and the lowest clean score among them is above zero',
          lowest > 0, f"lowest {lowest:+.4f} spearman")


if __name__ == '__main__':
    print("=" * 66)
    print("SMOKE: the zero rule in robustness_metrics")
    print("=" * 66)
    test_negative_clean_score_is_refused()
    test_zero_clean_score_is_refused()
    test_small_positive_clean_score_is_kept()
    test_lower_is_better_side_unchanged()
    test_explicit_threshold_still_works_for_callers_that_pass_one()
    test_c10_refuses_three_levels()
    test_the_finished_run_is_unchanged()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        sys.exit(1)
    print("ALL GREEN")
