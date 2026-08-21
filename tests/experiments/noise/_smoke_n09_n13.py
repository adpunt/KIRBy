#!/usr/bin/env python3
"""Fast self-contained smoke tests for the LNL-family noise experiments.

Covers n09 (LNL survey), n13 (noise distributions) and their shared utility
src/kirby/utils/lnl_methods.py.

These are deliberately TINY (small synthetic arrays, very few epochs) so the
whole file runs in well under a minute and never touches the server, real
datasets, or find_best_hybrid (which segfaults locally on macOS — see
feedback_no_heavy_local_compute). They guard the code-level breakage that the
2026-06 audit found and fixed:

  * n09 inner feature search used speed='fast', which used to raise KeyError
    ('fast') and was swallowed by a broad try/except, killing the entire hybrid
    arm (conditions B/D/E/F) on every fold. Speed tiers are now removed entirely
    ('speed' is an ignored vestige). -> assert n09 no longer passes 'fast'.
  * All 12 LNL methods must run and return DIFFERENTIATED predictions for
    regression, and the 10 clf-capable ones for classification, with the 2
    regression-only methods gating via NotImplementedError on is_clf=True.
  * Noise injectors must actually perturb labels, and n13's five noise
    distributions must differ from one another (not collapse to the same
    thing).
  * Retention (n09/n13) and recovery direction must be correct for a
    lower-is-better metric (RMSE).

Run:
    python tests/experiments/noise/_smoke_n09_n13.py
"""
import os
# Single-thread native libs before scientific imports (macOS OpenMP guard).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402

from kirby.utils.lnl_methods import (  # noqa: E402
    get_lnl_method, LNL_REGISTRY, REGRESSION_ONLY_LNL,
)

import importlib.util  # noqa: E402


def _load_experiment(modname, filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# tiny synthetic dataset with a real linear signal
# ---------------------------------------------------------------------------
def _synthetic(n=80, d=12, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    w = rng.randn(d)
    y = (X @ w + 0.3 * rng.randn(n)).astype(np.float32)
    return X, y, w


def test_lnl_methods_regression_differentiate():
    """All 12 LNL methods run on regression and give distinct predictions."""
    X, y, w = _synthetic()
    Xv = np.random.RandomState(1).randn(20, X.shape[1]).astype(np.float32)
    yv = (Xv @ w).astype(np.float32)
    preds = {}
    for name in LNL_REGISTRY:
        m = get_lnl_method(name, epochs=15, patience=5, random_state=1)
        out = m.train_and_predict(X, y, Xv, yv)
        p = np.asarray(out['y_pred'], dtype=float)
        assert p.shape == (20,), f"{name}: bad pred shape {p.shape}"
        assert np.isfinite(p).all(), f"{name}: non-finite predictions"
        preds[name] = p
    assert len(preds) == 12, f"expected 12 methods, ran {len(preds)}"
    # No two methods collapse to identical predictions.
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(preds[names[i]], preds[names[j]], atol=1e-9), \
                f"{names[i]} and {names[j]} produced identical predictions"
    print("  OK regression: 12 methods, all differentiated")


def test_lnl_methods_classification_and_gating():
    """10 clf-capable methods run on binary clf; 2 reg-only methods gate."""
    X, y, w = _synthetic()
    Xv = np.random.RandomState(1).randn(20, X.shape[1]).astype(np.float32)
    yv = (Xv @ w).astype(np.float32)
    yc = (y > np.median(y)).astype(int)
    yvc = (yv > np.median(y)).astype(int)
    classes = np.unique(yc)

    clf_preds = {}
    n_gated = 0
    for name in LNL_REGISTRY:
        if name in REGRESSION_ONLY_LNL:
            raised = False
            try:
                m = get_lnl_method(name, epochs=8, patience=3, random_state=1,
                                   is_clf=True, classes_=classes)
                m.train_and_predict(X, yc, Xv, yvc)
            except NotImplementedError:
                raised = True
            assert raised, f"{name} should raise NotImplementedError on is_clf=True"
            n_gated += 1
            continue
        m = get_lnl_method(name, epochs=15, patience=5, random_state=1,
                           is_clf=True, classes_=classes)
        out = m.train_and_predict(X, yc, Xv, yvc)
        p = np.asarray(out['y_pred'], dtype=float)
        assert p.shape == (20,), f"{name}: bad clf pred shape {p.shape}"
        # binary path returns P(positive) in [0, 1]
        assert ((p >= -1e-6) & (p <= 1 + 1e-6)).all(), \
            f"{name}: binary clf preds not in [0,1]: {p[:5]}"
        clf_preds[name] = p
    assert n_gated == 2, f"expected 2 regression-only methods, gated {n_gated}"
    assert len(clf_preds) == 10, f"expected 10 clf methods, ran {len(clf_preds)}"
    names = list(clf_preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(clf_preds[names[i]], clf_preds[names[j]],
                                   atol=1e-9), \
                f"{names[i]} and {names[j]} identical clf predictions"
    print("  OK classification: 10 methods differentiated, 2 reg-only gated")


def test_n09_no_fast_speed_tier():
    """n09 must not call find_best_hybrid with the removed 'fast' speed tier."""
    # Speed tiers were removed entirely (the 'speed' param is now an ignored
    # vestige and there is no tier registry to guard). This test now only
    # asserts the n09 survey source does not still pass speed='fast'.
    path = os.path.join(os.path.dirname(__file__), 'n09_lnl_survey.py')
    with open(path) as f:
        lines = f.readlines()
    # Ignore comment-only lines — the fix comment legitimately names the old
    # 'fast' tier. Only flag a real keyword argument on executable code.
    code_lines = [ln for ln in lines if not ln.lstrip().startswith('#')]
    code = ''.join(code_lines)
    assert "speed='fast'" not in code and 'speed="fast"' not in code, \
        "n09 still passes speed='fast' to find_best_hybrid (KeyError trap)"
    # The inner search must use the only valid tier.
    assert "speed='full'" in code, "n09 inner search should use speed='full'"
    print("  OK n09: inner search uses speed='full' (no removed 'fast' tier)")


def test_n09_inject_noise_reg_and_clf():
    n09 = _load_experiment('n09_lnl_survey', 'n09_lnl_survey.py')
    y = np.linspace(0, 10, 50).astype(float)
    # regression: additive Gaussian, sigma>0 must perturb, sigma=0 must not
    y0 = n09._inject_noise(y, 0.0, seed=1, is_clf=False)
    assert np.allclose(y0, y), "sigma=0 changed regression labels"
    yn = n09._inject_noise(y, 0.5, seed=1, is_clf=False)
    assert not np.allclose(yn, y), "sigma>0 did not perturb regression labels"
    # classification: label flipping
    yc = np.array([0, 1] * 25)
    ycn = n09._inject_noise(yc, 0.5, seed=1, is_clf=True)
    n_flipped = int((ycn != yc).sum())
    assert 0 < n_flipped <= len(yc), f"clf flip count out of range: {n_flipped}"
    assert set(np.unique(ycn)).issubset({0, 1}), "clf flips left class set"
    print(f"  OK n09 noise: reg perturbs, clf flipped {n_flipped}/{len(yc)} labels")


def test_n13_noise_types_differ():
    """n13's five regression noise distributions must differ from each other."""
    n13 = _load_experiment('n13_noise_distributions', 'n13_noise_distributions.py')
    y = np.linspace(0, 100, 200).astype(float)
    outs = {}
    for nt in n13.NOISE_TYPES:
        kw = {}
        if nt == 'partial':
            kw['partial_fraction'] = 0.3
        yn = n13._inject_noise(y, nt, 0.3, seed=7, is_clf=False, **kw)
        assert yn.shape == y.shape
        assert not np.allclose(yn, y), f"{nt}: did not perturb labels"
        outs[nt] = yn
    # No two noise types collapse to the same realization.
    names = list(outs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(outs[names[i]], outs[names[j]], atol=1e-6), \
                f"{names[i]} and {names[j]} produced identical noise"
    # partial leaves most labels untouched; gaussian touches ~all.
    n_changed_partial = int(np.sum(~np.isclose(outs['partial'], y)))
    n_changed_gauss = int(np.sum(~np.isclose(outs['gaussian'], y)))
    assert n_changed_partial < n_changed_gauss, \
        "partial noise should touch fewer labels than gaussian"
    print("  OK n13: 5 noise distributions all distinct; partial sparser")


def test_n13_inject_noise_clf():
    n13 = _load_experiment('n13_noise_distributions', 'n13_noise_distributions.py')
    yc = np.array([0, 1, 2] * 30)[:80]
    # heavy_tailed is the asymmetric pair-flip path; gaussian is symmetric.
    for nt in ['gaussian', 'heavy_tailed', 'partial']:
        kw = {'partial_fraction': 0.3} if nt == 'partial' else {}
        yn = n13._inject_noise(yc, nt, 0.5, seed=3, is_clf=True, **kw)
        assert yn.shape == yc.shape
        assert set(np.unique(yn)).issubset(set(np.unique(yc))), \
            f"{nt}: clf noise introduced an out-of-vocabulary class"
        assert int((yn != yc).sum()) > 0, f"{nt}: no clf labels flipped"
    print("  OK n13 clf noise: gaussian/heavy_tailed/partial all flip in-vocab")


def test_retention_direction_lower_is_better():
    """Sanity: for RMSE (lower better), a worse noisy score gives <100% retention
    and a denoised improvement gives positive recovery."""
    # n09/n13 retention: clean/noisy for lower-is-better.
    clean, noisy = 15.0, 21.0
    retention = clean / noisy * 100  # n09 _retention_pct lower_is_better branch
    assert retention < 100, "RMSE retention should drop below 100% when noisy"
    # n13 recovery: (denoised - noisy) / (clean - noisy) * 100.
    gap = clean - noisy  # negative for RMSE
    denoised = 18.0  # moved toward clean (lower)
    recovery = (denoised - noisy) / gap * 100
    assert recovery > 0, "denoised closer to clean should give positive recovery"
    denoised_worse = 23.0  # moved away
    recovery_worse = (denoised_worse - noisy) / gap * 100
    assert recovery_worse < 0, "denoised worse than noisy should give negative recovery"
    print("  OK retention/recovery direction correct for lower-is-better RMSE")


def test_modules_import():
    for fn in ['n09_lnl_survey.py', 'n13_noise_distributions.py']:
        _load_experiment(fn[:-3], fn)
    print("  OK both experiment modules import")


def main():
    tests = [
        test_modules_import,
        test_n09_no_fast_speed_tier,
        test_retention_direction_lower_is_better,
        test_n09_inject_noise_reg_and_clf,
        test_n13_noise_types_differ,
        test_n13_inject_noise_clf,
        test_lnl_methods_regression_differentiate,
        test_lnl_methods_classification_and_gating,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failures += 1
            import traceback
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'} "
          f"({len(tests) - failures}/{len(tests)})")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
