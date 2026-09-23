"""An empty silo used to end the run. Prove it no longer does."""
import sys, os
sys.path.insert(0, 'tests/experiments')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from federated.federated_helpers import (
    predict_fluid_similarity_ensemble_regression,
    predict_fluid_similarity_ensemble_classification)

rng = np.random.RandomState(0)
n_test, d, k = 12, 6, 3
Xte = rng.rand(n_test, d)

# silo 1 and 2 have rows; silo 3 has a fitted model but ZERO training rows,
# which is what a cluster holding one class leaves behind after a rebuild.
def build(reg=True):
    models, train, test = [None], [None], [None]
    for t in range(1, k + 1):
        n = 0 if t == k else 20
        Xtr = rng.rand(max(n, 2), d)
        ytr = (rng.rand(max(n, 2)) if reg else (rng.rand(max(n, 2)) > .5).astype(int))
        if not reg and len(set(ytr)) < 2:
            ytr[0], ytr[1] = 0, 1
        m = (RandomForestRegressor(n_estimators=5, random_state=0) if reg
             else RandomForestClassifier(n_estimators=5, random_state=0))
        m.fit(Xtr, ytr)
        models.append((m,))
        train.append(np.empty((0, d)) if n == 0 else Xtr)
        test.append(Xte)
    return models, train, test

ok = True
m, tr, te = build(reg=True)
try:
    preds, w = predict_fluid_similarity_ensemble_regression(m, tr, te, k)
    print(f"  regression   OK  -> {len(preds)} predictions for {n_test} test rows")
except Exception as e:
    ok = False; print(f"  regression   RAISED {type(e).__name__}: {e}")

m, tr, te = build(reg=False)
try:
    cls, prob, w = predict_fluid_similarity_ensemble_classification(m, tr, te, k)
    print(f"  classification OK -> {len(cls)} predictions for {n_test} test rows")
except Exception as e:
    ok = False; print(f"  classification RAISED {type(e).__name__}: {e}")

print("EMPTY-SILO TEST:", "PASS" if ok else "FAIL")
