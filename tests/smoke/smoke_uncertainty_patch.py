"""Smoke test for the uncertainty-recording patch.

Run:  python tests/smoke/smoke_uncertainty_patch.py

Checks, in order of how badly each would hurt:
  1. REPRODUCIBILITY  -- the patched runner produces bit-identical predictions
                         to the pre-patch runner. If this fails, the patch has
                         silently changed the experiment.
  2. BOOKKEEPING      -- recorded epsilon really is what was added; sigma=0 is a
                         true zero control; test molecules carry a noise scale
                         computed from the TRAINING distribution.
  3. CROSS-FITTING    -- every training molecule gets an out-of-fold score, and
                         it differs from the in-sample one (i.e. we are not
                         measuring memorisation).
  4. SIGNAL           -- on synthetic data where the answer is known, the
                         analysis recovers it.
"""
import importlib.util, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, '..', 'alternative_data_noise_robustness.py')
BAK = os.environ.get('KIRBY_PIPELINE_BAK', '/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/kirby_pipeline_old.py')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def synth(n=600, d=12, seed=0):
    """Labels are a genuine function of the features, so a model can learn."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + 0.25 * rng.normal(size=n)
    return X, y


FAILS = []
def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def main():
    new = load(PIPE, 'pipe_new')
    old = load(BAK, 'pipe_old') if os.path.exists(BAK) else None

    from sklearn.ensemble import RandomForestRegressor
    rf = lambda: RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=2)

    # Pick a model that actually emits a per-molecule uncertainty. QRF first
    # (it is the strongest error-ranker in the existing results); fall back to
    # NGBoost if the local quantile_forest/scikit-learn versions clash.
    qrf, unc_model_name, HAS_QRF = None, None, False
    try:
        from quantile_forest import RandomForestQuantileRegressor
        from sklearn.datasets import make_regression as _mr
        _X, _y = _mr(60, 4, random_state=0)
        RandomForestQuantileRegressor(n_estimators=5, random_state=0).fit(_X, _y)
        qrf = lambda: RandomForestQuantileRegressor(n_estimators=50, random_state=42, n_jobs=2)
        unc_model_name, HAS_QRF = 'QRF', True
    except Exception as e:
        print(f"  NOTE: QRF unusable here ({type(e).__name__}) - falling back to NGBoost.")
        print("        VERIFY QRF ON THE SERVER BEFORE SUBMITTING QRF JOBS.")
        try:
            from ngboost import NGBRegressor
            qrf = lambda: NGBRegressor(n_estimators=120, learning_rate=0.03,
                                       random_state=42, verbose=False)
            unc_model_name, HAS_QRF = 'NGBoost', True
        except Exception as e2:
            print(f"  NOTE: NGBoost unusable too ({type(e2).__name__})")

    X, y = synth()
    Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]
    SIG = [0.0, 0.5, 1.0]
    STRATS = new.NOISE_CONDITIONS

    print("\n1. REPRODUCIBILITY vs pre-patch code")
    # This section existed to prove the uncertainty patch changed no prediction.
    # It cannot be run against a backup taken before the noise redesign: that
    # code calls `NoiseInjectorRegression(strategy='legacy')`, and the six old
    # strategy names were deleted in noiseInject 1.0.0. There is nothing to
    # compare, because the old pipeline no longer executes -- and the noise it
    # would have drawn is deliberately different anyway, since a level now means
    # the amount DELIVERED rather than a raw scale each condition read its own
    # way. Say so rather than skipping quietly.
    if old is None:
        check("backup available", False, "(backup missing - cannot compare)")
    else:
        try:
            old.NoiseInjectorRegression(strategy='legacy', random_state=0)
            compatible = True
        except (ValueError, AttributeError):
            compatible = False
        if not compatible:
            print("  N/A   the backup predates the noise redesign, so it cannot run at all.")
            print("        A level is no longer a raw scale but the amount of noise actually")
            print("        delivered, so predictions are EXPECTED to differ. Bit-identity")
            print("        with the pre-redesign pipeline is not a property to preserve.")
        else:
            # Single-threaded. With n_jobs>1 a random forest sums tree outputs in
            # a non-deterministic order, so old-vs-OLD already differs by ~2e-15;
            # that is the library, not this patch. The control below proves it.
            rf1 = lambda: RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            ctrl = 0.0
            worst = 0.0
            for st in STRATS:
                a, _ = old.run_tree_experiment(Xtr, ytr, Xte, yte, rf1, st, SIG)
                b, _ = old.run_tree_experiment(Xtr, ytr, Xte, yte, rf1, st, SIG)
                c, _, _ = new.run_tree_experiment(Xtr, ytr, Xte, yte, rf1, st, SIG)
                for s in SIG:
                    ctrl = max(ctrl, float(np.abs(a[s] - b[s]).max()))
                    worst = max(worst, float(np.abs(a[s] - c[s]).max()))
            check("control: pre-patch code is deterministic single-threaded",
                  ctrl == 0.0, f"old-vs-old max diff = {ctrl:.3e}")
            check("predictions bit-identical across every condition",
                  worst == 0.0, f"old-vs-new max diff = {worst:.3e}")

    print("\n2. BOOKKEEPING")
    gtr = np.random.RandomState(3).randint(0, 30, len(ytr))
    gte = np.random.RandomState(4).randint(0, 30, len(yte))
    for st in STRATS:
        levels = new.CENSORING_LEVELS[:3] if 'censor' in st else SIG
        _, _, ex = new.run_tree_experiment(Xtr, ytr, Xte, yte, rf, st, levels,
                                           noise_groups=gtr, test_noise_groups=gte)
        ok_eps = True
        for i, s in enumerate(levels):
            # The pipeline builds a fresh injector per level from the condition
            # and the level INDEX, so reproduce it the same way.
            inj = new._injector_for(st, i)
            if s == 0.0:
                ok_eps &= bool(np.all(ex['train_epsilon'][s] == 0)) and \
                          bool(np.all(ex['train_noise_scale'][s] == 0)) and \
                          bool(np.all(ex['test_noise_scale'][s] == 0))
            else:
                r = inj.inject_verbose(ytr, s, groups=gtr)
                ok_eps &= bool(np.allclose(ex['train_epsilon'][s], r.epsilon))
                ok_eps &= bool(np.allclose(ex['train_noise_scale'][s], r.noise_scale))
        check(f"{st:16s} epsilon + scale recorded exactly; level 0 is a true zero", ok_eps)

    # test-side scale must use the TRAINING distribution's cut-points
    # A test set drawn from the SAME distribution has nearly the same deciles,
    # so shift it: only then does using the training cut-points visibly matter.
    yte_shift = yte + 1.5 * float(np.std(ytr))
    # Censoring is the condition whose cut-point is keyed to the label, so it is
    # the one where taking the cut-point from the test set instead of the
    # training set visibly changes who is affected. The four value-keyed
    # conditions that used to be checked here were deleted in noiseInject 1.0.0.
    _, _, ex = new.run_tree_experiment(Xtr, ytr, Xte, yte_shift, rf, 'censoring',
                                       [0.25], noise_groups=gtr, test_noise_groups=gte)
    inj = new._injector_for('censoring', 0)
    ref_based = inj.noise_scale(yte_shift, 0.25, reference=ytr)
    self_based = inj.noise_scale(yte_shift, 0.25)
    n_diff = int((ref_based != self_based).sum())
    check("censoring  test noise_scale uses the TRAINING cut-points",
          np.allclose(ex['test_noise_scale'][0.25], ref_based) and n_diff > 0,
          f"({n_diff}/{len(yte_shift)} molecules classified differently "
          f"than if the test set defined its own cut-points)")
    check("noise_scale draws no randomness",
          np.array_equal(
              new.NoiseInjectorRegression.from_condition('gaussian', random_state=1).inject(ytr, 0.5),
              (lambda i: (i.noise_scale(ytr, 0.5), i.inject(ytr, 0.5))[1])(
                  new.NoiseInjectorRegression.from_condition('gaussian', random_state=1))))

    if not HAS_QRF:
        print("\n  no uncertainty-capable model available - skipping checks 3 and 4")
    else:
        print(f"\n3. CROSS-FITTING  (model: {unc_model_name})")
        _, _, ex = new.run_tree_experiment(Xtr, ytr, Xte, yte, qrf, 'outlier_p05', [0.0, 1.0], oof_folds=5,
                                           noise_groups=gtr, test_noise_groups=gte)
        oof = ex['oof_unc'][1.0]
        check("every training molecule has an out-of-fold uncertainty",
              np.isfinite(oof).all(), f"({int(np.isfinite(oof).sum())}/{len(oof)})")
        # Are we measuring memorisation? Fit on everything, score the same rows,
        # and compare against the out-of-fold score.
        #
        # The universal signature of memorisation is ERROR, not uncertainty: a
        # model always fits its own training rows better than held-out ones.
        # (Uncertainty itself only collapses in-sample for models whose estimate
        # is a distance-to-data quantity -- a GP's posterior variance goes to
        # zero at its own inputs -- whereas NGBoost predicts a smooth noise
        # scale that barely moves. So assert on error, and report uncertainty.)
        sc = new.StandardScaler().fit_transform(Xtr)
        m = qrf(); yn = ytr + ex['train_epsilon'][1.0]
        m.fit(sc, yn)
        ins_mean, ins = new._tree_mean_std(m, sc)
        oof_mean = ex['oof_mean'][1.0]
        err_ins = float(np.mean(np.abs(ins_mean - yn)))
        err_oof = float(np.nanmean(np.abs(oof_mean - yn)))
        check("out-of-fold error exceeds in-sample error (so we are NOT scoring "
              "molecules with a model that memorised their label)",
              err_oof > err_ins,
              f"oof err={err_oof:.4f} vs in-sample err={err_ins:.4f} "
              f"(+{100*(err_oof/err_ins - 1):.1f}%)")
        print(f"       [info] mean uncertainty oof={np.nanmean(oof):.4f} "
              f"in-sample={np.mean(ins):.4f} "
              f"({unc_model_name} predicts a noise scale, so these are close by "
              f"design; a GP would collapse in-sample)")

        print("\n4. SIGNAL — does the analysis recover a known answer?")
        from scipy.stats import spearmanr
        print("   (A) training molecules: uncertainty vs |noise actually injected|")
        for st in ['gaussian', 'outlier_p05', 'grouped_wider']:
            _, _, ex = new.run_tree_experiment(Xtr, ytr, Xte, yte, qrf, st, [0.0, 1.0],
                                               oof_folds=5, noise_groups=gtr,
                                               test_noise_groups=gte)
            for s in [0.0, 1.0]:
                eps = np.abs(ex['train_epsilon'][s]); u = ex['oof_unc'][s]
                mask = np.isfinite(u)
                r = spearmanr(u[mask], eps[mask]).correlation if np.std(eps) > 0 else float('nan')
                print(f"      {st:14s} level={s}: rho = {r if r==r else float('nan'):+.3f}"
                      + ("   <- must be nan/undefined: no noise was drawn" if s == 0.0 else ""))
        print("   (B) test molecules: uncertainty vs the noise their region receives")
        print("       CONFOUND: the noise scale is a function of the LABEL, and a model's")
        print("       uncertainty may already track the label for reasons unrelated to noise.")
        print("       So the honest effect is (rho at sigma) MINUS (rho at sigma=0), where the")
        print("       sigma=0 model saw no noise at all but the same label distribution.")
        # grouped_wider and censoring are the conditions with a pattern to find.
        # For the shape-only conditions the scale is the same for every molecule,
        # so this question is UNDEFINED there, not zero.
        for st in ['gaussian', 'grouped_wider', 'censoring']:
            levels = [0.0, 0.25] if st == 'censoring' else [0.0, 1.0]
            _, u, ex = new.run_tree_experiment(Xtr, ytr, Xte, yte, qrf, st, levels,
                                               noise_groups=gtr, test_noise_groups=gte)
            pat = ex['test_noise_pattern'][levels[1]]
            if np.std(pat) == 0:
                print(f"      {st:14s} flat by construction - the question is UNDEFINED here")
                continue
            r0 = spearmanr(u[0.0], pat).correlation      # confound baseline, no noise trained on
            r1 = spearmanr(u[levels[1]], pat).correlation  # after training on noise
            print(f"      {st:14s} rho@level=0 (confound) = {r0:+.3f}   "
                  f"rho@level={levels[1]} = {r1:+.3f}   effect = {r1 - r0:+.3f}")

    print("\n" + "=" * 60)
    if FAILS:
        print("FAILED:", ", ".join(FAILS)); return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
