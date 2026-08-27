"""Executable checks for what the experimental models are fitted ON.

    python tests/smoke/smoke_kirby_target_scaling.py

Three properties, all run rather than read (RERUN_PLAN.md §2.13 in
qsar_qm_models):

  1. the target is z-scored for EVERY model, not just the neural ones. It used
     to be raw log units on the tree/kernel path while QM9 fits every model on a
     standardised label -- and the shared spec is written for a unit-variance
     target: SVR's epsilon 0.1 is a tenth of a standard deviation on QM9 and a
     large dead zone on a label whose spread is a few tenths of a log unit.
  2. the scaler is fitted on the CLEAN training labels. It used to be fitted on
     the noisy ones, so the divisor was sqrt(sd_clean^2 + dose^2) and GREW with
     the level: the same nominal level posed a different optimisation problem at
     every point on the grid.
  3. predictions and uncertainties come back in RAW label units, so R2 and RMSE
     stay comparable with the labels and across the two paths.

Each check carries a CONTROL showing what the old behaviour gave.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

_argv, sys.argv = sys.argv, ['smoke_kirby_target_scaling.py']
_spec = importlib.util.spec_from_file_location(
    'runner', ROOT / 'tests' / 'alternative_data_noise_robustness.py')
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
sys.argv = _argv

T0 = time.time()
FAILS = []
NCHECK = [0]


def check(name, ok, detail=''):
    NCHECK[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILS.append(name)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


rng = np.random.default_rng(0)
N = 240
X = rng.normal(size=(N, 6))
# A label on a scale nothing like unit variance -- the situation on LogD,
# Caco-2 and hERG, where the spread is a few tenths of a log unit.
Y = 7.0 + 0.45 * (X[:, 0] + 0.5 * X[:, 1]) + 0.05 * rng.normal(size=N)
GROUPS = np.arange(N) // 6

SEEN = []


class Recorder(LinearRegression):
    """A model that remembers the labels it was handed."""

    def fit(self, Xf, yf):
        SEEN.append(np.asarray(yf, dtype=float).copy())
        return super().fit(Xf, yf)


section("the target the models are fitted on")

SEEN.clear()
preds, uncs, extras = P.run_tree_experiment(
    X[:180], Y[:180], X[180:], Y[180:], Recorder, 'gaussian', [0.0, 0.6],
    noise_groups=GROUPS[:180], test_noise_groups=GROUPS[180:], rep_name='PDV')

fitted_clean = SEEN[0]
fitted_noisy = SEEN[1]
check("the tree/kernel path fits on a z-scored target, not raw log units",
      abs(fitted_clean.mean()) < 1e-8 and abs(fitted_clean.std() - 1.0) < 1e-8,
      f"level 0.0: mean {fitted_clean.mean():+.2e}, sd {fitted_clean.std():.6f} "
      f"(raw label sd was {Y[:180].std():.4f})")

# CONTROL: fitted on the CLEAN spread, so the noisy target's spread GROWS with
# the level rather than being renormalised back to 1.
check("the scaler is fitted on the CLEAN labels, so the noisy target is wider "
      "at a higher level",
      fitted_noisy.std() > 1.05,
      f"level 0.6 target sd = {fitted_noisy.std():.4f} "
      f"(a scaler refitted on the noisy labels would give 1.0000)")

section("what comes back out")

check("predictions are in raw label units",
      abs(np.mean(preds[0.0]) - Y[180:].mean()) < 0.5 * Y[180:].std(),
      f"predicted mean {np.mean(preds[0.0]):.4f} against a test label mean of "
      f"{Y[180:].mean():.4f}")
check("a prediction is NOT on the model's internal scale",
      abs(np.mean(preds[0.0])) > 1.0,
      f"|mean| = {abs(np.mean(preds[0.0])):.4f}, which a z-scored prediction "
      f"could not be for this label")

section("the GP fit method and the collapse flag are recorded per level")

check("gp_fit_method is a mapping from level, not one scalar",
      isinstance(extras['gp_fit_method'], dict))
check("gp_collapsed is recorded at every level that ran",
      set(extras['gp_collapsed']) == {0.0, 0.6},
      f"levels: {sorted(extras['gp_collapsed'])}")

section("the level-free pattern describes the injection that happens")

# `noise_pattern` is the shape at a fixed reference level, and the zero-level
# subtraction that removes the label-magnitude confound rests on it being the
# SAME column at every level. It was not: the injector's seed carries the level,
# and WHO gets hit was drawn from that seed, so the affected set moved with the
# level for the two selection-based conditions (RERUN_PLAN.md §2.13).
from scipy.stats import spearmanr  # noqa: E402

_y = np.linspace(0, 10, 120)
_g = np.arange(120) // 6


def _top(v, frac=0.15):
    k = max(1, int(len(v) * frac))
    return set(np.argsort(-v)[:k].tolist())


for _cond in ('outlier_p10', 'grouped_wider'):
    _pat = P._injector_for(_cond, 0.0).noise_scale(
        _y, 1.0, reference=_y, groups=_g, reference_groups=_g)
    for _level in (0.3, 0.6):
        _act = P._injector_for(_cond, _level).noise_scale(
            _y, _level, reference=_y, groups=_g, reference_groups=_g)
        _rho = spearmanr(_pat, _act).statistic
        _overlap = len(_top(_pat) & _top(_act))
        check(f"{_cond} at level {_level}: the pattern ranks the molecules the "
              f"injection does",
              _rho > 0.999 and _overlap == len(_top(_pat)),
              f"spearman {_rho:.3f}, top-15% overlap {_overlap}/{len(_top(_pat))}")

# CONTROL: with the selection seeded from the level, as it used to be, the two
# disagree. Built here rather than asserted from memory.
from noiseInject import NoiseInjectorRegression  # noqa: E402

_old_pat = NoiseInjectorRegression.from_condition('outlier_p10', random_state=11)
_old_act = NoiseInjectorRegression.from_condition('outlier_p10', random_state=22)
_p = _old_pat.noise_scale(_y, 1.0, reference=_y, groups=_g, reference_groups=_g)
_a = _old_act.noise_scale(_y, 0.6, reference=_y, groups=_g, reference_groups=_g)
check("CONTROL: two seeds DO pick different molecules, so the check above means "
      "something",
      len(_top(_p) & _top(_a)) < len(_top(_p)),
      f"top-15% overlap {len(_top(_p) & _top(_a))}/{len(_top(_p))}")

section("a level is a fraction of the clean label spread")

# The same numeric level must mean the same RELATIVE corruption on every
# dataset, which is what makes auc_norm comparable without rescaling
# (RERUN_PLAN.md 2.12). Censoring is exempt: its level is the fraction of labels
# clipped, already dimensionless, and scaling it by a spread puts it outside
# [0, 1] and raises.
_lab = np.linspace(0, 10, 200)
_grp = np.arange(200) // 10
for _spread_factor in (1.0, 4.0):
    _y = _lab * _spread_factor
    # The TRAINING spread, which is what the dose is a fraction of -- not the
    # spread of the whole array.
    _sd = float(np.std(_y[:150]))
    _, _, _ex = P.run_tree_experiment(
        _y[:150].reshape(-1, 1).repeat(4, axis=1), _y[:150],
        _y[150:].reshape(-1, 1).repeat(4, axis=1), _y[150:],
        Recorder, 'gaussian', [0.0, 1.0],
        noise_groups=_grp[:150], test_noise_groups=_grp[150:], rep_name='PDV')
    _delivered = float(np.std(_ex['train_epsilon'][1.0]))
    check(f"at level 1.0 the noise is one label spread (spread {_sd:.2f})",
          abs(_delivered / _sd - 1.0) < 0.15,
          f"delivered {_delivered:.3f} against a spread of {_sd:.3f} "
          f"= {_delivered / _sd:.3f}")

# CONTROL: under the retired grids the level WAS the dose in raw units, so the
# same level meant a different fraction of the spread on every dataset.
check("CONTROL: a raw-unit level of 1.0 is not one spread when the spread is 4x",
      abs(1.0 / float(np.std(_lab * 4.0)) - 1.0) > 0.5,
      f"1.0 raw units is {1.0 / float(np.std(_lab * 4.0)):.3f} of that spread")

section("a representation that failed to build stops the job")

try:
    P.generate_representations(['CCO', 'CCC'], rep_filter=['NotARepresentation'])
except RuntimeError as exc:
    check("a requested representation that produced nothing is refused by name",
          'partial roster' in str(exc), str(exc)[:70])
except Exception as exc:  # noqa: BLE001
    check("a requested representation that produced nothing is refused by name",
          False, f"{type(exc).__name__}: {exc}")
else:
    check("a requested representation that produced nothing is refused by name",
          False, "it returned quietly with the representation missing")

print("\n" + "=" * 70)
print(f"{NCHECK[0]} checks in {time.time() - T0:.1f}s")
if FAILS:
    print("FAILED:", "; ".join(FAILS))
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
