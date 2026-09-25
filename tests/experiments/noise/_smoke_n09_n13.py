#!/usr/bin/env python3
"""Fast self-contained smoke tests for the LNL-family noise experiments.

Covers n09 (LNL survey), n13 (noise distributions) and their shared utility
src/kirby/utils/lnl_methods.py.

These are deliberately TINY (small synthetic arrays, very few epochs) so the
whole file runs in well under a minute and never touches the server, real
datasets, or find_best_hybrid (which segfaults locally on macOS — see
feedback_no_heavy_local_compute). They guard the code-level breakage that the
2026-06 audit found and fixed:

  * n09's inner feature search once passed speed='fast' when that value raised
    KeyError, and a broad try/except swallowed it, killing the whole hybrid arm
    (conditions B/D/E/F) on every fold. The guard against it was dropped
    2026-09-09: speed='fast' is a supported value again and selects the short
    allocation / importance / evaluator lists, so passing it is no longer a
    defect to test for.
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


def test_n09_inject_noise_reg_and_clf():
    """n09 no longer carries its own injector; it uses the shared one."""
    n09 = _load_experiment('n09_lnl_survey', 'n09_lnl_survey.py')
    assert not hasattr(n09, '_inject_noise'), \
        "n09 still has a private _inject_noise; it must use noise_helpers"
    from experiments.noise.noise_helpers import inject_train_labels, noise_spec_for

    y = np.linspace(0, 10, 50).astype(float)
    # Level 0 must deliver EXACTLY zero -- the negative control, not a small number.
    y0, _, _ = inject_train_labels(y, noise_spec_for('qm9', 'gaussian', 0.0), train_idx=np.arange(len(y)))
    assert np.array_equal(y0, y), "level 0 changed regression labels"
    yn, _, prov = inject_train_labels(y, noise_spec_for('qm9', 'gaussian', 0.5), train_idx=np.arange(len(y)))
    assert not np.allclose(yn, y), "level > 0 did not perturb regression labels"
    # And it must deliver the amount it was asked for.
    target = 0.5 * float(np.std(y))
    assert abs(prov['realised_dose_label_units'] / target - 1) < 0.25, \
        f"delivered {prov['realised_dose_label_units']:.4f}, asked for {target:.4f}"
    # classification: label flipping
    yc = np.array([0, 1] * 25)
    ycn, mask, _ = inject_train_labels(yc, noise_spec_for('qm9', 'uniform', 0.5),
                                       is_clf=True, train_idx=np.arange(len(yc)))
    n_flipped = int((ycn != yc).sum())
    assert 0 < n_flipped <= len(yc), f"clf flip count out of range: {n_flipped}"
    assert set(np.unique(ycn)).issubset({0, 1}), "clf flips left class set"
    assert int(mask.sum()) == n_flipped, "clf mask does not match the flips"
    print(f"  OK n09 noise: level 0 is exactly zero, dose lands, "
          f"clf flipped {n_flipped}/{len(yc)} labels")


def test_n13_noise_types_differ():
    """n13's conditions must be distinct realisations AND deliver the same dose.

    Distinct is not enough on its own. The whole point of the redesign is that they
    differ in SHAPE at a matched amount: before it, six conditions at one nominal
    setting delivered between 0.49x and 2.00x the Gaussian amount, and their apparent
    severity ordering was entirely explained by that.
    """
    n13 = _load_experiment('n13_noise_distributions', 'n13_noise_distributions.py')
    assert not hasattr(n13, '_inject_noise'), \
        "n13 still has a private _inject_noise; it must use noise_helpers"
    from experiments.noise.noise_helpers import inject_train_labels, noise_spec_for
    from kirby.noise_spec import DOSE_MATCHED_STRATEGIES, GROUPED_STRATEGIES

    rs = np.random.RandomState(0)
    y = rs.normal(50.0, 20.0, 2000)
    groups = np.arange(len(y)) // 4
    outs, doses, flagged = {}, {}, {}
    for nt in DOSE_MATCHED_STRATEGIES:
        g = groups if nt in GROUPED_STRATEGIES else None
        yn, mask, prov = inject_train_labels(
            y, noise_spec_for('qm9', nt, 0.3),
            train_idx=np.arange(len(y)), groups=g)
        assert yn.shape == y.shape
        assert not np.allclose(yn, y), f"{nt}: did not perturb labels"
        outs[nt], doses[nt] = yn, prov['realised_dose_fraction_of_spread']
        flagged[nt] = float(mask.mean())

    names = list(outs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(outs[names[i]], outs[names[j]], atol=1e-6), \
                f"{names[i]} and {names[j]} produced identical noise"

    # The dose is flat across every condition. Student-t at nu = 3 has an infinite
    # fourth moment, so its per-run realised dose is unstable by construction and it
    # gets the wider band the design quotes for it.
    for nt, d in doses.items():
        tol = 0.35 if nt == 'student_t_nu3' else 0.15
        assert abs(d / 0.3 - 1) < tol, \
            f"{nt}: delivered {d:.3f} of the label spread, asked for 0.300"

    # The outlier conditions hit a sparse subset harder; the shape-only ones hit
    # everything equally.
    assert flagged['outlier_p01'] < flagged['outlier_p10'] < flagged['gaussian']
    assert flagged['gaussian'] == 1.0
    print(f"  OK n13: {len(outs)} conditions distinct, dose flat to "
          f"{max(doses.values()) / min(doses.values()) - 1:.1%}, outliers sparser")


def test_n13_inject_noise_clf():
    """Classification takes the flip models; a flipped label has no magnitude."""
    from experiments.noise.noise_helpers import inject_train_labels, noise_spec_for
    yc = np.array([0, 1, 2] * 30)[:80]
    for nt in ['uniform', 'class_dependent', 'confusion_directed']:
        yn, mask, _ = inject_train_labels(
            yc, noise_spec_for('qm9', nt, 0.5),
            train_idx=np.arange(len(yc)), is_clf=True)
        assert yn.shape == yc.shape
        assert set(np.unique(yn)).issubset(set(np.unique(yc))), \
            f"{nt}: clf noise introduced an out-of-vocabulary class"
        assert int((yn != yc).sum()) > 0, f"{nt}: no clf labels flipped"
        assert int(mask.sum()) == int((yn != yc).sum()), f"{nt}: mask mismatch"
    print("  OK n13 clf noise: three flip models all flip in-vocab, masks agree")


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


def test_n09_base_reps_are_top_two_for_the_dnn():
    """n09's single-representation comparison is the DNN's top 2. 2026-09-25.

    The stored search is a stand-in whose baselines rank three representations
    differently for rf and for the DNN, so the test fails if n09 reads the rf
    ranking. The inner feature search is replaced by one that raises, so the
    hybrid arms drop out and only conditions A and C run -- the part that
    changed -- without calling find_best_hybrid locally.
    """
    from types import SimpleNamespace
    n09 = _load_experiment('n09_lnl_survey', 'n09_lnl_survey.py')
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 6))
    y = X @ rng.normal(size=6) + rng.normal(0, 0.1, n)
    data = {'labels': y,
            'reps': {'aaa': X, 'bbb': X[:, ::-1] + rng.normal(0, 0.5, X.shape),
                     'ccc': rng.normal(size=(n, 6))},
            'groups': np.arange(n), 'task_type': 'regression', 'metric': 'rmse'}
    search = SimpleNamespace(
        metric_name='rmse', lower_is_better=True,
        baselines={'aaa': {'rf': 0.9, 'dnn': 0.3},
                   'bbb': {'rf': 0.2, 'dnn': 0.5},
                   'ccc': {'rf': 0.1, 'dnn': 0.9}},
        baseline_fold_scores={})
    meta = {'best_config': {'blueprint': SimpleNamespace(budget=4)},
            'aug_condition': 'none'}

    def _lookup(*a, **k):
        return search, meta

    def _no_search(*a, **k):
        raise RuntimeError('inner search stubbed out in the smoke')

    n09.get_best_search_for_downstream = _lookup
    n09.find_best_hybrid = _no_search
    r = n09._run_one_condition(
        data, {'n_splits': 2}, '/tmp', 'qm9', noise_condition='gaussian',
        sigmas=[0.0, 0.5], noise_seeds=[0], methods=['small_loss'])
    assert r['base_reps_by_model'] == {'dnn': ['aaa', 'bbb']}, \
        f"expected the DNN's top 2 (aaa, bbb), got {r['base_reps_by_model']}"
    assert r['base_reps_ranked_by'] == 'dnn'
    assert r['base_reps_search_model'] == 'dnn'
    assert 'base_rep' not in r
    for lvl in ('0.0', '0.5'):
        blk = r['per_sigma'][lvl]
        for bk in ('base_1_no_lnl', 'base_2_no_lnl'):
            assert blk[bk]['n_folds'] == 2, (lvl, bk, blk[bk])
        assert 'base_no_lnl' not in blk
    hi = r['per_sigma']['0.5']
    for bk in ('base_1_lnl', 'base_2_lnl'):
        assert hi[bk]['small_loss']['n_folds'] == 2, (bk, hi[bk])
    assert (hi['base_1_no_lnl']['fold_scores']
            != hi['base_2_no_lnl']['fold_scores']), \
        "the two ranks produced identical scores -- same features twice?"
    s = r['summary']
    assert set(s['hybrid_vs_base']) == {'base_1', 'base_2'}
    assert s['hybrid_vs_base']['base_2']['representation'] == 'bbb'
    assert 'clean_base_1' in s and 'clean_base_2' in s
    assert set(r['retention']['0.5']) >= {'base_1_no_lnl', 'base_2_no_lnl',
                                          'base_1_lnl', 'base_2_lnl'}

    # A caller-given list overrides the ranking.
    r2 = n09._run_one_condition(
        data, {'n_splits': 2}, '/tmp', 'qm9', noise_condition='gaussian',
        sigmas=[0.0, 0.5], noise_seeds=[0], methods=['small_loss'],
        base_reps={'dnn': ['ccc']})
    assert r2['base_reps_by_model'] == {'dnn': ['ccc']}
    assert 'base_1_no_lnl' in r2['per_sigma']['0.0']
    assert 'base_2_no_lnl' not in r2['per_sigma']['0.0']
    print("  OK n09: conditions A and C run in the DNN's rank-1 (aaa) and "
          "rank-2 (bbb) representations; a caller list overrides the ranking")


def test_n13_single_reps_per_model_and_track_a_for_the_dnn():
    """n13 end to end, the single representations. 2026-09-25.

    The stored search ranks aaa then bbb for rf, and ccc then bbb for the DNN,
    so Track A (the DNN) and Track B (rf) use different pairs. The hybrid
    builds are stubbed to raise, so only the single-representation spaces run.
    """
    from types import SimpleNamespace
    n13 = _load_experiment('n13_noise_distributions', 'n13_noise_distributions.py')
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 6))
    y = X @ rng.normal(size=6) + rng.normal(0, 0.1, n)
    data = {'labels': y,
            'reps': {'aaa': X, 'bbb': X[:, ::-1] + rng.normal(0, 0.5, X.shape),
                     'ccc': rng.normal(size=(n, 6))},
            'groups': np.arange(n), 'task_type': 'regression', 'metric': 'rmse'}
    search = SimpleNamespace(
        metric_name='rmse', lower_is_better=True,
        baselines={'aaa': {'rf': 0.3, 'dnn': 0.5},
                   'bbb': {'rf': 0.5, 'dnn': 0.3},
                   'ccc': {'rf': 0.9, 'dnn': 0.1}},
        baseline_fold_scores={}, all_results=[])
    meta = {'best_config': {'blueprint': SimpleNamespace(
                budget=4, augmentation_strategy='none')},
            'objective': 'predictive', 'objective_choice': 'only_family',
            'pool_subset': 'predictive_and_similarity', 'metric_name': 'rmse',
            'aug_condition': 'none', 'by_objective': {}}

    def _no_build(*_a, **_k):
        raise RuntimeError('hybrid build stubbed out in the smoke')

    n13.get_best_search_for_downstream = lambda *a, **k: (search, meta)
    n13.build_fold_hybrid = _no_build
    n13.LNL_METHODS = ['small_loss']
    n13.ESTIMATORS = ['local_variance_ratio']
    n13.LNL_EPOCHS = 5
    n13.NOISE_SEEDS = [0]
    r = n13.run(data, {'n_splits': 2, 'models': ['rf']}, '/tmp', 'qm9',
                noise_types=['gaussian'], strengths=[0.5])
    assert 'error' not in r, r.get('error')
    assert r['base_reps_by_model'] == {'rf': ['aaa', 'bbb'],
                                       'dnn': ['ccc', 'bbb']}, \
        r['base_reps_by_model']
    st = r['per_noise_type']['gaussian']['per_strength']['0.5']
    assert set(st['track_a']) == {'base_1_ccc', 'base_2_bbb'}, set(st['track_a'])
    row = st['track_a']['base_1_ccc']['small_loss']
    assert set(row['retention']) == {'base_1', 'base_2'}, row
    assert set(st['clean_score_dnn']) == {'base_1', 'base_2'}
    assert set(st['track_b']) == {'base_1_aaa', 'base_2_bbb'}, set(st['track_b'])
    for sp in ('base_1_aaa', 'base_2_bbb'):
        units = set(st['track_b'][sp]['local_variance_ratio'])
        assert units == {'rf@base_1', 'rf@base_2'}, (sp, units)
    assert set(st['noisy_score_by_model']) == {'rf@base_1', 'rf@base_2'}
    assert np.isfinite(st['noisy_score'])
    blk = st['per_fold_scores_single_recipe']
    assert set(blk['baselines_by_model']['rf']) == {'aaa', 'bbb'}, blk
    s = r['summary']
    assert set(s['hybrid_advantage_persists']) == {'base_1', 'base_2'}
    print("  OK n13: Track A in the DNN's top 2 (ccc, bbb), Track B detects in "
          "rf's top 2 (aaa, bbb) and fits rf on both, baselines list both")


def test_modules_import():
    for fn in ['n09_lnl_survey.py', 'n13_noise_distributions.py']:
        _load_experiment(fn[:-3], fn)
    print("  OK both experiment modules import")


def main():
    tests = [
        test_modules_import,
        test_retention_direction_lower_is_better,
        test_n09_inject_noise_reg_and_clf,
        test_n09_base_reps_are_top_two_for_the_dnn,
        test_n13_noise_types_differ,
        test_n13_inject_noise_clf,
        test_n13_single_reps_per_model_and_track_a_for_the_dnn,
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
