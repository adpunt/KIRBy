#!/usr/bin/env python3
"""Fast audit-regression tests for n02 / n07 noise experiments.

Covers the AUDIT_NOISE.md items re-verified 2026-06-03/04:

  P0#1  n07 per-rep evaluation must thread is_clf (classification path).
  P0#3  n02 reliability scoring must thread is_clf at every call site.
  P2    n02 top3_base_reps must come from the search pool, not the last
        per-model loop variable.

These are deliberately CHEAP: pure-function behaviour + source-level call-site
checks + a tiny in-memory _evaluate_fold round-trip. No find_best_hybrid, no
server data, no foundation-model reps (honours the no-heavy-local-compute rule).

Run:  python tests/experiments/noise/_smoke_audit_n02_n03_n07.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# 2026-09-25: the noise experiments sweep every similarity setting by default
# (KIRBY_SIMILARITY_SETTINGS unset = all 58, plan section 1). This smoke checks
# what each run() computed before that axis existed, so it runs today's
# setting alone, and it writes its caches under a temporary directory, never
# the real KIRBY_CACHE_DIR.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
import tempfile as _tempfile
os.environ['KIRBY_CACHE_DIR'] = _tempfile.mkdtemp(prefix='kirby_smoke_cache_')

import sys
import re
import inspect
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np

import experiments.noise.n02_sample_weighting as n02
import experiments.noise.n07_remediation as n07

_HERE = os.path.dirname(__file__)
_FAILS = []


def _check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ''))
    if not cond:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# P0#1 — n07 per-rep evaluation threads is_clf (classification path)
# ---------------------------------------------------------------------------
def test_n07_inject_noise_clf_flips_classes():
    """is_clf=True must FLIP class labels (not add continuous noise).

    n07's private injector is gone; it corrupts through the shared one now, as the
    Outlier condition -- a random few records are simply wrong. Selection stays
    random and is not keyed to the label value, because a mistyped unit is a
    property of the record.
    """
    from experiments.noise.noise_helpers import inject_train_labels
    _check('n07 has no private _inject_noise left',
           not hasattr(n07, '_inject_noise'))

    y = np.array([0, 1] * 5)
    from kirby.noise_spec import NoiseSpec
    y_noisy, mask, _ = inject_train_labels(
        y, NoiseSpec(strategy='uniform', target_effective_noise=0.5, noise_seed=7),
        train_idx=np.arange(len(y)), is_clf=True)
    # Every output value is still a valid class label (0 or 1) — no floats.
    _check('n07 clf noise keeps labels in {0,1}',
           set(np.unique(y_noisy)).issubset({0, 1}),
           f"unique={sorted(set(np.unique(y_noisy)))}")
    # Flagged positions actually changed class.
    _check('n07 clf noise flips exactly the flagged labels',
           bool(np.all(y_noisy[mask] != y[mask])) and int(mask.sum()) > 0)
    # Regression path on the SAME integer labels adds continuous noise.
    spec_reg = n07._noise_spec('herg_ki', 0.5, 0.5, noise_seed=0)
    y_noisy_reg, _, prov = inject_train_labels(y.astype(float), spec_reg,
                                               train_idx=np.arange(len(y)),
                                               is_clf=False)
    _check('n07 regression noise is continuous (contrast)',
           not set(np.unique(y_noisy_reg)).issubset({0.0, 1.0}))
    _check('n07 regression noise records what it delivered',
           prov.get('realised_dose_label_units') is not None)


def test_n07_per_rep_call_sites_pass_is_clf():
    """Source check: the PER-REP block's _evaluate_fold + _inject_noise calls
    must pass is_clf. This is the exact P0#1 regression."""
    # _run_one_condition, not run(): since 2026-09-23 run() is the condition
    # sweep (noise_helpers.sweep_conditions) and the per-rep block lives in
    # the single-condition body, so this check failed at HEAD with "marker
    # not found" before it reached a call site. Fixed 2026-09-25.
    src = inspect.getsource(n07._run_one_condition)
    # Isolate the per-rep block (after the 'Per-rep conditions' marker).
    marker = 'Per-rep conditions'
    assert marker in src, "per-rep block marker not found"
    per_rep_src = src[src.index(marker):]
    # Every _evaluate_fold( in the per-rep block must pass is_clf=is_clf.
    eval_calls = re.findall(r'_evaluate_fold\((.*?)\)', per_rep_src, re.DOTALL)
    _check('n07 per-rep block has _evaluate_fold calls', len(eval_calls) >= 2,
           f"found {len(eval_calls)}")
    all_pass_isclf = all('is_clf=is_clf' in c for c in eval_calls)
    _check('n07 per-rep _evaluate_fold calls all pass is_clf=is_clf',
           all_pass_isclf)
    # n07 injects through _inject_or_recorded, which passes is_clf on to
    # inject_train_labels (n07_remediation.py, _inject_or_recorded). The
    # per-rep block has called that wrapper rather than inject_train_labels
    # directly since the AS_THEY_COME condition was added.
    inject_calls = re.findall(r'(?:inject_train_labels|_inject_or_recorded)\((.*?)\)',
                              per_rep_src, re.DOTALL)
    all_inject_isclf = all('is_clf=is_clf' in c for c in inject_calls)
    _check('n07 per-rep injection calls all pass is_clf=is_clf',
           all_inject_isclf and len(inject_calls) >= 1,
           f"found {len(inject_calls)}")


def test_n07_compute_noise_scores_threads_is_clf():
    """_compute_noise_scores must accept and forward is_clf, and all three
    call sites in n07 must pass it (so clf datasets use the estimator's
    class-disagreement branch, not the regression neighbor-mean path)."""
    sig = inspect.signature(n07._compute_noise_scores)
    _check('n07 _compute_noise_scores has is_clf param',
           'is_clf' in sig.parameters)
    src = inspect.getsource(n07)
    # Exclude the `def _compute_noise_scores(...)` line — only match invocations.
    calls = re.findall(r'(?<!def )_compute_noise_scores\((.*?)\)', src, re.DOTALL)
    call_sites = [c for c in calls if 'X_tr_s' in c]
    _check('n07 has >=3 _compute_noise_scores call sites',
           len(call_sites) >= 3, f"found {len(call_sites)}")
    _check('n07 all _compute_noise_scores call sites pass is_clf=is_clf',
           all('is_clf=is_clf' in c for c in call_sites))
    # Behavioural: clf vs reg noise scores differ on multiclass data.
    rng = np.random.RandomState(1)
    X = rng.randn(50, 6)
    y = rng.randint(0, 3, size=50)
    s_clf = n07._compute_noise_scores(X, y, is_clf=True)
    s_reg = n07._compute_noise_scores(X, y.astype(float), is_clf=False)
    _check('n07 _compute_noise_scores clf path returns scores',
           s_clf is not None and len(s_clf) == 50)
    _check('n07 _compute_noise_scores clf differs from reg (3-class)',
           s_clf is not None and s_reg is not None
           and not np.allclose(s_clf, s_reg))


def test_n07_evaluate_fold_clf_roundtrip():
    """End-to-end (tiny, in-memory): _evaluate_fold on clf data with a real RF
    classifier produces a finite AUC — proving the clf branch is live."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(0)
    n = 60
    X = rng.randn(n, 8)
    y = (X[:, 0] + 0.3 * rng.randn(n) > 0).astype(int)
    tr, va = np.arange(40), np.arange(40, n)
    mf = lambda: RandomForestClassifier(n_estimators=20, random_state=0)
    # y_all/train_idx: the no-cache path draws over the WHOLE label column and
    # indexes the training rows out, so a molecule's corruption does not depend
    # on its fold (noise_helpers.inject_train_labels, 2026-09-12).
    score = n07._evaluate_fold(
        X[tr], X[va], y[tr], y[va], 'noisy_baseline', 0.1, mf,
        roc_auc_score, is_clf=True, metric_name='auc', noise_seed=0,
        y_all=y, train_idx=tr)
    _check('n07 _evaluate_fold clf noisy_baseline returns finite AUC',
           score is not None and np.isfinite(score), f"score={score}")
    # knn_smooth on clf must use the class-mode branch and still score.
    scores = n07._compute_noise_scores(X[tr], y[tr])
    score2 = n07._evaluate_fold(
        X[tr], X[va], y[tr], y[va], 'knn_smooth', 0.1, mf,
        roc_auc_score, is_clf=True, metric_name='auc', noise_seed=0,
        precomputed_y_tr=y[tr], noise_scores=scores)
    _check('n07 _evaluate_fold clf knn_smooth returns finite AUC',
           score2 is not None and np.isfinite(score2), f"score={score2}")


# ---------------------------------------------------------------------------
# P0#3 — n02 reliability scoring threads is_clf at every call site
# ---------------------------------------------------------------------------
def test_n02_reliability_clf_vs_reg_differ():
    """compute_reliability_scores must take a class-agreement path when
    is_clf=True, distinct from the regression neighbor-mean path."""
    rng = np.random.RandomState(0)
    n = 60
    X = rng.randn(n, 6)
    # MULTICLASS (3 classes). For binary {0,1} with range=1 the clf
    # (fraction-same-class) and reg (1-|y-neighbor_mean|/range) formulas are
    # algebraically identical, so equivalence there is correct, not a bug.
    # The clf branch only DIVERGES — and matters — for >=3 classes, where the
    # reg path's |y - mean| / range is meaningless on nominal labels.
    y = rng.randint(0, 3, size=n)
    imp = np.ones(6)
    rel_clf = n02.compute_reliability_scores(X, y, imp, k=5, is_clf=True)
    rel_reg = n02.compute_reliability_scores(X, y.astype(float), imp, k=5,
                                             is_clf=False)
    _check('n02 reliability clf path returns [0.1,1] array',
           rel_clf.shape == (n,) and rel_clf.min() >= 0.1 - 1e-9
           and rel_clf.max() <= 1.0 + 1e-9)
    # On >=3 classes the two paths MUST differ (proves is_clf actually branches
    # and isn't a no-op alias of the regression path).
    _check('n02 reliability clf path differs from reg path (3-class)',
           not np.allclose(rel_clf, rel_reg),
           f"max|diff|={np.max(np.abs(rel_clf - rel_reg)):.3f}")


def test_n02_reliability_call_sites_pass_is_clf():
    """Source check: all three compute_reliability_scores / cliff-aware call
    sites in n02.run thread is_clf."""
    src = inspect.getsource(n02.run)
    rel_calls = re.findall(r'compute_reliability_scores\((.*?)\)', src, re.DOTALL)
    _check('n02 has compute_reliability_scores calls', len(rel_calls) >= 2,
           f"found {len(rel_calls)}")
    _check('n02 all compute_reliability_scores calls pass is_clf=is_clf',
           all('is_clf=is_clf' in c for c in rel_calls))
    cliff_calls = re.findall(r'compute_cliff_aware_weights\((.*?)\)', src, re.DOTALL)
    _check('n02 cliff-aware call passes is_clf=is_clf',
           all('is_clf=is_clf' in c for c in cliff_calls) and cliff_calls,
           f"found {len(cliff_calls)}")


# ---------------------------------------------------------------------------
# P2 — n02 top3_base_reps comes from the search pool, not the loop variable
# ---------------------------------------------------------------------------
def test_n02_top3_from_search_not_loop_var():
    """results['top3_base_reps'] must be assigned from a search-derived var,
    not the per-model-loop `top3` 4-tuple (rep, uw, w, d)."""
    src = inspect.getsource(n02.run)
    m = re.search(r"results\['top3_base_reps'\]\s*=\s*(\w+)", src)
    _check("n02 assigns results['top3_base_reps']", m is not None)
    if m:
        assigned_var = m.group(1)
        _check("n02 top3_base_reps assigned from search-derived var "
               "(not loop 'top3')",
               assigned_var != 'top3', f"assigned from '{assigned_var}'")
        # The assigned var must be built from extract_top3_base_reps(search).
        build = re.search(
            rf"{assigned_var}\s*=\s*\[\(n,\s*float\(s\)\)\s*for n, s in "
            r"extract_top3_base_reps\(search\)\]", src)
        _check("n02 top3 var built from extract_top3_base_reps(search)",
               build is not None)


def main():
    print("=== AUDIT REGRESSION TESTS: n02 / n03 / n07 ===\n")
    print("P0#1 — n07 per-rep is_clf:")
    test_n07_inject_noise_clf_flips_classes()
    test_n07_per_rep_call_sites_pass_is_clf()
    test_n07_compute_noise_scores_threads_is_clf()
    test_n07_evaluate_fold_clf_roundtrip()
    print("\nP0#3 — n02 reliability is_clf:")
    test_n02_reliability_clf_vs_reg_differ()
    test_n02_reliability_call_sites_pass_is_clf()
    print("\nP2 — n02 top3 source:")
    test_n02_top3_from_search_not_loop_var()

    print(f"\n=== {'ALL GREEN' if not _FAILS else f'{len(_FAILS)} FAILED: {_FAILS}'} ===")
    sys.exit(1 if _FAILS else 0)


if __name__ == '__main__':
    main()
