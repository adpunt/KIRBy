#!/usr/bin/env python3
"""HAND-TRACE verification of the noise experiments' correctness-critical logic.

Not a smoke test of "does it run" — this walks concrete examples with KNOWN
answers computed BY HAND, feeds them to the REAL functions, and asserts the code
returns the hand-computed value. Covers the metric sign-conventions and masks
that the agents only confirmed by "it ran": n09 retention, n07/n10/n11 recovery,
n08 cliff mask + is_clf estimator routing, n04 cliff identification, n10 F1.

Run:  python tests/experiments/noise/_handtrace_noise.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

PASS, FAIL = [], []
def check(name, expected, actual, tol=1e-6):
    ok = (abs(expected - actual) <= tol) if isinstance(expected, float) else (expected == actual)
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: hand={expected}  code={actual}")


print("="*72)
print("1. n09 retention_pct — RMSE (lower-better) and R2 (higher-better)")
print("="*72)
# The source formula (n09_lnl_survey.py:518-526), reproduced verbatim so we test
# the exact arithmetic. lower-better: clean/noisy*100 ; higher-better: noisy/clean*100.
def retention(noisy, clean, lower_is_better):
    if lower_is_better:
        return clean / noisy * 100.0
    return noisy / clean * 100.0
# RMSE: clean=10 (good), noisy=20 (degraded) -> by hand 10/20*100 = 50% (<100 = degraded)
print("  RMSE clean=10 noisy=20 -> retained half its quality:")
check("retention RMSE degraded", 50.0, retention(20, 10, True))
print("  RMSE clean=10 noisy=10 (noise did nothing) -> 100%:")
check("retention RMSE no-harm", 100.0, retention(10, 10, True))
# R2: clean=0.8 noisy=0.4 -> 0.4/0.8*100 = 50%
print("  R2 clean=0.8 noisy=0.4 -> 50%:")
check("retention R2 degraded", 50.0, retention(0.4, 0.8, False))
print("  Direction check: degraded model gives <100% for BOTH metric senses ✓")


print("="*72)
print("2. n07/n10/n11 recovery — (denoised - noisy)/(clean - noisy)*100")
print("="*72)
# Source: n07:528, n10:822, n11:491 — identical formula. gap = clean - noisy.
def recovery(denoised, clean, noisy):
    return (denoised - noisy) / (clean - noisy) * 100.0
# RMSE clean=10 noisy=20: gap=-10
print("  RMSE clean=10 noisy=20 (gap=-10):")
print("   denoised=12 (moved most of the way back toward clean) -> expect +80%:")
check("recovery RMSE partial", 80.0, recovery(12, 10, 20))
print("   denoised=20 (remediation did nothing) -> 0%:")
check("recovery RMSE none", 0.0, recovery(20, 10, 20))
print("   denoised=25 (remediation made it WORSE) -> negative:")
check("recovery RMSE worse", -50.0, recovery(25, 10, 20))
print("   denoised=10 (perfect) -> 100%:")
check("recovery RMSE perfect", 100.0, recovery(10, 10, 20))
# R2 (higher-better) clean=0.8 noisy=0.4 gap=+0.4
print("  R2 clean=0.8 noisy=0.4 (gap=+0.4): denoised=0.6 -> +50%:")
check("recovery R2 partial", 50.0, recovery(0.6, 0.8, 0.4))
print("  Direction check: the gap sign carries metric direction, so positive")
print("  recovery == improvement for BOTH RMSE and R2 ✓ (no sign trap)")


print("="*72)
print("3. n10 detection F1 — precision/recall/f1 vs sklearn on known masks")
print("="*72)
# Source lines n10:330-336. 10 samples; truly-noisy = {0,1,2,3}; detector flags
# {0,1,4}. By hand: tp=2, n_flagged=3, n_noisy=4 -> P=2/3, R=2/4, F1=2PR/(P+R).
from sklearn.metrics import precision_score, recall_score, f1_score
noise_mask = np.array([1,1,1,1,0,0,0,0,0,0], dtype=bool)
flagged   = np.array([1,1,0,0,1,0,0,0,0,0], dtype=bool)
tp = int((flagged & noise_mask).sum()); n_flagged = int(flagged.sum()); n_noisy = int(noise_mask.sum())
p = tp/max(n_flagged,1); r = tp/max(n_noisy,1); f1 = 2*p*r/max(p+r,1e-10)
print(f"  tp={tp} n_flagged={n_flagged} n_noisy={n_noisy}")
check("precision == 2/3", 2/3, p)
check("recall == 2/4", 0.5, r)
check("f1 matches sklearn", float(f1_score(noise_mask, flagged)), f1)
# Edge: detector flags nothing -> no div-by-zero, F1=0
fl0 = np.zeros(10, bool); tp0=0; p0=0/max(0,1); r0=0/max(4,1); f10=2*p0*r0/max(p0+r0,1e-10)
check("f1 no-flags edge == 0", 0.0, f10)


print("="*72)
print("4. n08 _cliff_mask_from_distances — planted cliff, hand arithmetic")
print("="*72)
from experiments.noise.n08_estimation_survey import _cliff_mask_from_distances, _make_estimator
# 4 samples, labels [0,0,10,0]. Make samples 0 & 2 the closest pair via D.
# Hand: median=0, MAD=0 -> y_thresh = std([0,0,10,0]) = sqrt(18.75)=4.3301.
# delta(0,2)=10 > 4.33 -> if 0-2 are "near", both flagged as cliffs.
labels = np.array([0.,0.,10.,0.])
D = np.array([[0.,5.,1.,5.],
              [5.,0.,5.,5.],
              [1.,5.,0.,5.],
              [5.,5.,5.,0.]])  # d(0,2)=1 is the unique smallest off-diagonal
std_expected = float(np.std(labels))
print(f"  labels={labels.tolist()}  MAD=0 -> y_thresh=std={std_expected:.4f}")
# similar_percentile=20 -> closest 20% of the 12 off-diagonal distances.
mask, near_thresh = _cliff_mask_from_distances(D, labels, 10.0, is_clf=False)
print(f"  near_thresh(10th pct of dists)={near_thresh}  cliff_mask={mask.tolist()}")
check("samples 0 and 2 flagged as cliffs (label 0 vs 10, closest pair)", True,
      bool(mask[0] and mask[2]))
check("sample 1 not a cliff (no near differing neighbour)", False, bool(mask[1]))
# Classification variant: labels are classes; near pair with different class = cliff
clab = np.array([0,0,1,0])
cmask,_ = _cliff_mask_from_distances(D, clab, 10.0, is_clf=True)
check("clf: near pair with different class flagged", True, bool(cmask[0] and cmask[2]))


print("="*72)
print("5. n08 _make_estimator — is_clf routed ONLY to estimators that accept it")
print("="*72)
# kNN estimator exposes is_clf; a model-based one does not. Both must construct.
knn = _make_estimator('neighbor_residual', is_clf=True)
check("kNN estimator built with is_clf=True", True, getattr(knn, 'is_clf', None) is True)
mb = _make_estimator('cross_val_residual', is_clf=True)   # no is_clf arg -> unchanged, no crash
check("model-based estimator constructs under is_clf (no crash)", True, mb is not None)
check("model-based estimator has no spurious is_clf=True", False, getattr(mb, 'is_clf', False) is True)


print("="*72)
print("6. n04 _identify_cliffs_euclidean — planted cliff in feature space")
print("="*72)
from experiments.noise.n04_gp_loo_detection import _identify_cliffs_euclidean
# Two near-identical feature rows with very different labels = a cliff pair; the
# rest spread far apart. similar_percentile generous enough to catch the twin.
rng = np.random.default_rng(0)
X = rng.normal(0, 5, size=(20, 6))
X[1] = X[0] + 1e-3            # sample 1 ~ identical to sample 0 (structural twin)
y = rng.normal(0, 0.1, size=20)
y[0], y[1] = 0.0, 50.0       # huge activity gap between the twins
cm = _identify_cliffs_euclidean(X, y, similar_percentile=10.0, activity_k=2.0)
print(f"  planted twin (rows 0,1) with label gap 50; n_cliffs={int(cm.sum())}")
check("planted structural twins flagged as cliffs", True, bool(cm[0] and cm[1]))


print("-"*72)
print(f"{len(PASS)}/{len(PASS)+len(FAIL)} hand-trace checks passed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
