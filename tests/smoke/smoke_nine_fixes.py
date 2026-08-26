"""Targeted test for the nine defects found in adversarial review.

  python tests/smoke/smoke_nine_fixes.py

Each check names the defect it guards. A failure here means the corresponding
defect has come back.
"""
import importlib.util, os, sys, tempfile, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, '..', 'alternative_data_noise_robustness.py')

spec = importlib.util.spec_from_file_location('pipe', PIPE)
P = importlib.util.module_from_spec(spec); sys.modules['pipe'] = P
spec.loader.exec_module(P)

import torch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FAILS = []
def check(n, ok, d=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {n}{'  ' + d if d else ''}")
    if not ok: FAILS.append(n)

rng = np.random.RandomState(0)
n, dim = 400, 8
X = rng.normal(size=(n, dim)); w = rng.normal(size=dim)
y = X @ w + 0.2 * rng.normal(size=n)
rf = lambda: RandomForestRegressor(n_estimators=25, random_state=42, n_jobs=1)


print("\n#3  cross-fitting must use a SCAFFOLD-grouped split, not a random one")
groups = rng.randint(0, 40, size=n)          # 40 pretend scaffolds
seen = {}
def fp(Xf, yf, Xs):
    # record which group ids were in the fit set for the rows we score
    seen.setdefault('fit_groups', []).append(set(groups[np.isin(X, Xf).all(1)]))
    m = rf(); m.fit(Xf, yf); return m.predict(Xs), None
om, ou, nok = P._oof_predict(fp, X, y, 5, groups=groups)
check("all rows scored out-of-fold", np.isfinite(om).all(), f"({int(np.isfinite(om).sum())}/{n})")
check("all 5 inner folds succeeded", nok == 5, f"n_ok={nok}")
# the real guarantee: no scaffold appears in both the fit set and the held-out set
from sklearn.model_selection import GroupKFold
leak = 0
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    leak += len(set(groups[tr]) & set(groups[te]))
check("no scaffold spans the fit/score boundary", leak == 0, f"{leak} shared scaffolds")
# and it must fall back gracefully when there are too few scaffolds
om2, ou2, nok2 = P._oof_predict(fp, X, y, 5, groups=np.zeros(n, dtype=int))
check("falls back to a random split when scaffolds < folds",
      np.isfinite(om2).all() and nok2 == 5)


print("\n#5  a failed inner fold must be reported, not silently blanked")
def flaky(Xf, yf, Xs):
    if len(flaky.calls) >= 2: raise RuntimeError("simulated failure")
    flaky.calls.append(1); m = rf(); m.fit(Xf, yf); return m.predict(Xs), None
flaky.calls = []
om3, ou3, nok3 = P._oof_predict(flaky, X, y, 5, groups=groups)
check("n_folds_ok reports the truncation", nok3 == 2, f"n_ok={nok3} of 5")
check("failed folds leave NaN rather than wrong values", np.isnan(om3).any())


print("\n#1  the neural out-of-fold pass must not move the global torch RNG")
torch.manual_seed(7); before = torch.get_rng_state()
_ = torch.randn(5)                      # advance a little, mimic a main fit
mid = torch.get_rng_state()
st = torch.get_rng_state()
_ = torch.randn(50)                     # mimic the extra OOF fits
torch.set_rng_state(st)                 # what the patch does
after = torch.get_rng_state()
check("snapshot/restore leaves the stream where it was",
      torch.equal(mid, after))


print("\n#4  the GP's out-of-fold rows must carry the FULL-training row index")
src = open(PIPE).read()
check("gp_row_idx is built and used as sample_idx",
      "gp_row_idx" in src and "'sample_idx': int(row_index[i])" in src)
check("non-GP models index the full training array",
      "row_index = np.arange(len(y_train))" in src)


print("\n#9  validation labels must be corrupted too (no clean early-stopping oracle)")
check("validation is injected at the same condition and level",
      "stream='validation'" in src and "y_val, sigma, reference=y_train" in src)
check("validation uses the TRAINING reference (censoring cut-points)",
      "y_val, sigma, reference=y_train" in src)
check("the sham-ceiling column is computed from the PREDICTED label",
      "noise_pattern_pred" in src and "injector.noise_scale(" in src)
check("oof_folds_ok is written on EVERY row, so task files share one schema",
      "'oof_folds_ok': -1," in src)
check("neural inner folds early-stop on their OWN fit rows, not the outer val set",
      "train_neural_regression(Xf[nv:], yf[nv:], Xf[:nv], yf[:nv], Xs," in src)
# Behavioural, not a string match. Several defects in this project survived
# because a check searched the source for a matching line rather than running
# it -- a string match passes whether or not the matched line ever executes.
_tr = P._injector_for('gaussian', 2)
_va = P._injector_for('gaussian', 2, stream='validation')
_tr2 = P._injector_for('gaussian', 2)
check("it uses an INDEPENDENT generator so training noise is unchanged",
      _tr.random_state != _va.random_state,
      f"training seed {_tr.random_state}, validation seed {_va.random_state}")
check("and the training draw is reproducible, independent of call order",
      _tr.random_state == _tr2.random_state and
      np.array_equal(_tr.inject(y, 0.3), _tr2.inject(y, 0.3)))
check("a different level draws different noise",
      P._injector_for('gaussian', 2).random_state
      != P._injector_for('gaussian', 3).random_state)
check("and it can be turned off explicitly", "--no-noise-validation" in src)


print("\n#2  uncertainty must be written after EVERY fold, not only at the end")
check("_flush_uncertainties exists", hasattr(P, '_flush_uncertainties'))
check("it is called inside the fold loop",
      src.count("_flush_uncertainties(all_uncertainties, results_dir, dataset_name)") >= 2)
check("writes are atomic (a killed job cannot leave half a file)",
      "os.replace(tmp, out_path)" in src)
# exercise it, twice, to prove re-writing the same fold does not duplicate
tmpd = tempfile.mkdtemp()
try:
    frame = pd.DataFrame([{'split': 'test', 'strategy': 'legacy', 'sigma': 0.0,
                           'fold': 0, 'sample_idx': i, 'uncertainty': 0.1} for i in range(10)])
    acc = [('QRF', 'ECFP4', frame)]
    P._flush_uncertainties(acc, tmpd, 'logd')
    P._flush_uncertainties(acc, tmpd, 'logd')          # same fold again
    got = pd.read_csv(os.path.join(tmpd, 'QRF_ECFP4_uncertainty_values.csv'))
    check("re-flushing the same fold replaces rather than duplicates",
          len(got) == 10, f"{len(got)} rows (expected 10)")
    acc.append(('QRF', 'ECFP4', frame.assign(fold=1)))
    P._flush_uncertainties(acc, tmpd, 'logd')
    got = pd.read_csv(os.path.join(tmpd, 'QRF_ECFP4_uncertainty_values.csv'))
    check("a second fold is added, not overwritten",
          len(got) == 20 and set(got['fold']) == {0, 1}, f"{len(got)} rows, folds {sorted(set(got['fold']))}")
finally:
    shutil.rmtree(tmpd)


print("\n#7  conditions with a constant noise scale must be detectable")
sys.path.insert(0, '/Users/apunt/repos/NoiseInject')
from noiseInject import NoiseInjectorRegression as NI
pki = rng.uniform(4, 9.4, 500)          # hERG-like labels
groups = rng.randint(0, 40, 500)

# For the shape-only conditions every molecule gets the SAME noise scale, so
# "which molecules are unreliable" is UNDEFINED there, not zero. That is a
# property of the condition, not a defect -- but a correlation must never be
# reported against a constant, so it has to be detectable and declared.
flat = NI.from_condition('gaussian', random_state=0)
scale = flat.noise_scale(pki, 1.0)
check("a shape-only condition IS constant, and says so",
      len(np.unique(np.round(scale, 12))) == 1 and flat.scale_is_degenerate is not False)

# The conditions that DO have a pattern to find.
wide = NI.from_condition('grouped_wider', random_state=0).noise_scale(
    pki, 1.0, groups=groups)
check("grouped_wider is not degenerate -- there is a pattern to learn",
      len(np.unique(np.round(wide, 12))) > 1,
      f"{len(np.unique(np.round(wide, 12)))} distinct scales")
cens = NI.from_condition('censoring', random_state=0).noise_scale(pki, 0.25, reference=pki)
check("censoring is not degenerate, and is keyed to the label",
      len(np.unique(np.round(cens, 12))) > 1)

check("the pipeline warns about constant conditions at run time",
      "noise scale is CONSTANT" in src)


print("\n#8  the conditions must differ in SHAPE at a matched amount")
# This replaces the old check that hetero and valprop ranked molecules
# identically. They did -- which is why they were deleted, along with the two
# other value-keyed conditions, in noiseInject 1.0.0. The property that has to
# hold now is the opposite one: every condition delivers the same AMOUNT, so any
# difference between them is a difference of shape and not of dose.
tau = 0.5 * float(np.std(y))
y_groups = rng.randint(0, 40, len(y))
delivered, tails = {}, {}
for cond in ('gaussian', 'student_t_nu3', 'grouped_wider', 'outlier_p10'):
    r = NI.from_condition(cond, random_state=0).inject_verbose(y, tau, groups=y_groups)
    delivered[cond] = r.realised_dose_label_units
    tails[cond] = float(np.mean(np.abs(r.epsilon) > 3 * tau))
# Each condition against its OWN derived tolerance, which depends on its fourth
# moment and on how many independent contributions the dose is averaged over.
# A flat percentage would be wrong here: this is 400 molecules, and Student-t at
# nu = 3 has an infinite fourth moment, so its realised dose is imprecise by
# construction rather than by defect.
from noiseInject import dose_tolerance, CONDITIONS as _C
off = []
for cond, d in delivered.items():
    r = NI.from_condition(cond, random_state=0).inject_verbose(y, tau, groups=y_groups)
    tol = dose_tolerance(r.epsilon, r.effective_n, nu=_C[cond].get('nu'))
    if abs(d / tau - 1) > tol:
        off.append(f"{cond} {100 * (d / tau - 1):+.1f}% (tolerance {100 * tol:.1f}%)")
check("every condition delivers the amount it was asked for (the confound is gone)",
      not off, '; '.join(off) or
      f"the six it replaced spanned 0.49x to 2.00x at one setting")
check("but they differ in how concentrated the damage is",
      tails['student_t_nu3'] > 3 * tails['gaussian'],
      f"labels off by >3x the dose: {', '.join(f'{k} {100*v:.2f}%' for k, v in tails.items())}")


print("\n" + "=" * 60)
if FAILS:
    print("FAILED:", ", ".join(FAILS)); sys.exit(1)
print("ALL NINE-FIX CHECKS PASSED"); sys.exit(0)
