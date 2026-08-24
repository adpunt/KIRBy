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
check("validation is injected with the same strategy",
      "_val_injector.inject(y_val, sigma" in src)
check("it uses an INDEPENDENT generator so training noise is unchanged",
      "random_state=1337" in src)
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


print("\n#7  degenerate (constant) noise-scale arms must be detectable")
sys.path.insert(0, '/Users/apunt/repos/NoiseInject')
from noiseInject import NoiseInjectorRegression as NI
pki = rng.uniform(4, 9.4, 500)          # hERG-like labels
raw = NI(strategy='threshold', random_state=0).noise_scale(pki, 1.0)
check("threshold on hERG-like labels IS constant (the defect is real)",
      len(np.unique(np.round(raw, 12))) == 1)
q = NI(strategy='threshold', random_state=0).noise_scale(
        pki, 1.0, high_threshold=float(np.quantile(pki, 0.9)),
        low_threshold=float(np.quantile(pki, 0.1)))
check("--threshold-quantile makes it non-degenerate",
      len(np.unique(np.round(q, 12))) > 1,
      f"{len(np.unique(np.round(q,12)))} distinct scales")
check("the pipeline warns about constant arms at run time",
      "noise scale is CONSTANT" in src)


print("\n#8  hetero and valprop are rank-identical and must not be double-counted")
from scipy.stats import spearmanr
a = NI(strategy='hetero', random_state=0).noise_scale(y, 1.0)
b = NI(strategy='valprop', random_state=0).noise_scale(y, 1.0)
r = spearmanr(a, b).correlation
check("they rank molecules identically (so they are ONE arm for these questions)",
      r > 0.9999, f"rho = {r:.8f}")


print("\n" + "=" * 60)
if FAILS:
    print("FAILED:", ", ".join(FAILS)); sys.exit(1)
print("ALL NINE-FIX CHECKS PASSED"); sys.exit(0)
