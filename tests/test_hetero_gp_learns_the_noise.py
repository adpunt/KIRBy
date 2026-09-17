#!/usr/bin/env python3
"""Does the heteroscedastic Gaussian process actually learn which labels are noisy?

This is the guard for one line in `HeteroscedasticGPEstimator`: the noise
network is trained against the residual of the FITTED process, not of the prior.

Why it needs a guard. gpytorch returns the PRIOR at the training inputs while
the model is in train mode, so `y - mean` is the spread of the labels themselves
rather than anything about which ones are wrong. A network trained on that comes
out flat, and nothing else in the run says so -- the fit succeeds, every number
is finite, and the column reads as a per-molecule noise estimate. Measured in
~/repos/qsar_qm_models on 300 molecules whose loud half carries 100x the noise
variance of the quiet half: prior residuals 6.44 against 5.92, a ratio of 1.09;
fitted residuals 0.88 against 0.033, a ratio of 27.08. Trained on the first, the
rank correlation with the true corruption sat between -0.12 and +0.09 at every
learning rate and every length tried.

So the test is theirs, rebuilt here: make half the molecules much noisier than
the other half, and ask whether the predicted noise separates the two halves.

    python tests/test_hetero_gp_learns_the_noise.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Two OpenMP runtimes in one process; both switches are needed and the crash
# runs in both orders (project_fold_hook_and_lgbm_segfault_2026_09_02).
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from kirby.utils.noise_estimation import get_noise_estimator

FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_split_noise(n=300, d=8, ratio=100.0, seed=0, feature_dependent=True):
    """Half the molecules get `ratio` times the noise variance of the other half.

    ⛔ WHICH HALF IS LOUD HAS TO BE DECIDED BY THE FEATURES, or this model cannot
    answer at all. It predicts a molecule's noise FROM ITS FEATURES, and it is
    scored out of fold, so it can only rank a held-out molecule when that
    molecule's own features say how noisy it is. Draw the loud half at random
    instead and there is nothing in the features to generalise from, whatever
    the model.

    MEASURED on this fixture, 2026-09-18, rank correlation against the true noise
    width, out of fold over three folds:

        which molecules are loud      hetero_gp   ngboost   activity_cliff_aware
        drawn at random                  +0.149    -0.027                 +0.057
        decided by the features          +0.383    +0.511                 +0.078

    That is not a property of these two models. It is what the condition is.
    `outlier_p01`, `outlier_p05` and `outlier_p10` pick the molecules that get
    the wider error AT RANDOM, with no reference to the features
    (NOISE_AUDIT_2026_09_10.md 6.7), so under those three a feature-based noise
    model is being asked something undefined. `grouped_wider` keys the wider
    error to whole scaffold families, which the representation encodes, and it
    is the one dose-matched condition where the question has an answer.

    `feature_dependent=False` reproduces the random draw, and the second test
    below uses it to show the collapse rather than describe it.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y_true = X @ rng.normal(size=d)
    if feature_dependent:
        loud = X[:, 0] > np.median(X[:, 0])
    else:
        loud = np.zeros(n, dtype=bool)
        loud[rng.choice(n, size=n // 2, replace=False)] = True
    quiet_sd = 0.10
    sd = np.where(loud, quiet_sd * np.sqrt(ratio), quiet_sd)
    return X, y_true + rng.normal(0, sd), loud, sd


def test_it_separates_the_two_halves():
    print("does the predicted noise tell the loud half from the quiet half:")
    X, y, loud, sd = make_split_noise(seed=1)
    est = get_noise_estimator('hetero_gp', n_folds=3, epochs=120)
    scores = np.asarray(est.estimate(X, y), dtype=float)

    check('it returns one finite number per molecule',
          scores.shape == (len(y),) and bool(np.all(np.isfinite(scores))))

    loud_mean = float(scores[loud].mean())
    quiet_mean = float(scores[~loud].mean())
    ratio = loud_mean / max(quiet_mean, 1e-12)
    print(f"        predicted noise: loud half {loud_mean:.4f}, "
          f"quiet half {quiet_mean:.4f}, ratio {ratio:.2f}")
    # The injected ratio is 10 in standard deviations. A network that learned
    # nothing gives 1.0. This is the line the fitted-residual fix moves.
    check('the loud half is predicted noisier than the quiet half',
          ratio > 1.5, f'ratio {ratio:.2f}, and 1.0 is a flat output')

    from scipy.stats import spearmanr
    rho = float(spearmanr(scores, sd).correlation)
    print(f"        rank correlation against the true noise width: {rho:+.3f}")
    check('it ranks molecules by how noisy they really are', rho > 0.2,
          f'spearman {rho:+.3f}')


def test_it_cannot_answer_when_the_noise_is_drawn_at_random():
    """The same model, the same amount of noise, chosen without the features.

    This is not a bug to fix. It is the line between the conditions this model
    can be read under and the ones it cannot, and it is here so that a
    chance-level number under `outlier_p10` reads as the condition rather than
    as the method failing.
    """
    print("\nthe same model where the loud molecules were drawn at random:")
    from scipy.stats import spearmanr
    X, y, loud, sd = make_split_noise(seed=1, feature_dependent=False)
    est = get_noise_estimator('hetero_gp', n_folds=3, epochs=120)
    scores = np.asarray(est.estimate(X, y), dtype=float)
    rho_random = float(spearmanr(scores, sd).correlation)

    X2, y2, _, sd2 = make_split_noise(seed=1, feature_dependent=True)
    est2 = get_noise_estimator('hetero_gp', n_folds=3, epochs=120)
    rho_feat = float(spearmanr(
        np.asarray(est2.estimate(X2, y2), dtype=float), sd2).correlation)

    print(f"        drawn at random         {rho_random:+.3f}")
    print(f"        decided by the features {rho_feat:+.3f}")
    check('it does much better when the features decide which molecules are noisy',
          rho_feat > rho_random + 0.15,
          f'{rho_feat:+.3f} against {rho_random:+.3f}; if these are close, '
          f'something other than the features is carrying the signal')


def test_the_two_halves_of_the_uncertainty_are_not_one_signal():
    """The label-noise term and the model-unsure term must not be the same thing.

    This is the check that failed for the random forest in the sister study --
    its two terms correlated with the injected noise at +0.84 and +0.81, the same
    signal reported twice (RERUN_PLAN.md 2.32, separation 1.1, recorded as FAIL).
    Here the noise term must track the injected width and the latent variance
    must not track it nearly as well.
    """
    print("\nare the two halves of the uncertainty separate quantities:")
    from scipy.stats import spearmanr
    X, y, loud, sd = make_split_noise(seed=2)
    est = get_noise_estimator('hetero_gp', n_folds=3, epochs=120)
    noise = np.asarray(est.estimate(X, y), dtype=float)
    latent = np.asarray(est.latent_variance_, dtype=float)

    rho_noise = float(spearmanr(noise, sd).correlation)
    rho_latent = float(spearmanr(latent, sd).correlation)
    print(f"        the noise term tracks the true width at {rho_noise:+.3f}")
    print(f"        the latent variance tracks it at      {rho_latent:+.3f}")
    check('the latent variance is kept beside the noise term',
          latent.shape == noise.shape and bool(np.all(np.isfinite(latent))))
    check('the noise term tracks the injected width better than the latent one',
          rho_noise > rho_latent,
          f'{rho_noise:+.3f} against {rho_latent:+.3f}; if these are equal the '
          f'two are one signal reported twice')


def test_it_gives_a_denoised_value_to_move_toward():
    print("\ndoes it hand back a denoised value:")
    X, y, loud, sd = make_split_noise(n=200, seed=3)
    est = get_noise_estimator('hetero_gp', n_folds=3, epochs=100)
    est.estimate(X, y)
    pred = np.asarray(est.predictions_, dtype=float)
    check('one out-of-fold prediction per molecule',
          pred.shape == (len(y),) and bool(np.all(np.isfinite(pred))))
    # It has to be a real fit, not the prior repeated. A collapsed process
    # returns one flat number, which is the failure the lengthscale start exists
    # to prevent.
    check('the prediction is not one number repeated',
          float(pred.std()) > 1e-6, f'spread {float(pred.std()):.4g}')


if __name__ == '__main__':
    test_it_separates_the_two_halves()
    test_it_cannot_answer_when_the_noise_is_drawn_at_random()
    test_the_two_halves_of_the_uncertainty_are_not_one_signal()
    test_it_gives_a_denoised_value_to_move_toward()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
