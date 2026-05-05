"""Meta-stacker survival model.

Trains an outer Coxnet on out-of-fold (OOF) predictions of base survival models,
then refits each base on the full training set for inference.

The OOF construction prevents leakage: each base model never sees the data it
predicts during stacking-target generation. At predict time we use the
full-data refits to score validation patients, then pass through the outer
Coxnet trained on OOF stacks.

This wraps the (potentially heterogeneous) base models behind the same
.fit(X, y_struct) / .predict_risk(X) interface as the other survival models,
so the existing search.py + run.py + submit.py machinery handles it natively.
"""

from typing import List, Optional, Tuple

import numpy as np

from experiments.annitia.survival import (
    SurvivalCoxPH, SurvivalCoxnet, SurvivalRSF, SurvivalGBSA,
)


def _oof_predict_risk(model_factory, X, y_struct, n_inner_folds: int = 5,
                       random_state: int = 42) -> Tuple[np.ndarray, list]:
    """Generate out-of-fold risk predictions via stratified K-fold on event flag.

    Returns (oof_pred, fold_models) where fold_models is the list of fitted
    inner-fold models (so the caller can also keep them for full-data refit).
    """
    from sklearn.model_selection import StratifiedKFold
    n = X.shape[0]
    oof = np.zeros(n, dtype=np.float64)
    inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True,
                            random_state=random_state)
    fold_models = []
    for tr, va in inner.split(np.arange(n), y_struct['event'].astype(int)):
        m = model_factory()
        try:
            m.fit(X[tr], y_struct[tr])
            oof[va] = m.predict_risk(X[va])
        except Exception:
            oof[va] = 0.0
        fold_models.append(m)
    return oof, fold_models


class SurvivalStacker:
    """Stack of Cox/Coxnet/RSF/GBSA + outer Coxnet meta-learner.

    Hyper-parameters:
        bases:           sequence of survival-model factories. Each must
                          implement .fit(X, y) and .predict_risk(X).
        n_inner_folds:   K-fold for OOF generation (default 5).
        outer_l1_ratio:  Coxnet l1_ratio for the meta-model.
    """

    def __init__(self,
                 bases: Optional[List[type]] = None,
                 n_inner_folds: int = 5,
                 outer_l1_ratio: float = 0.5,
                 random_state: int = 42):
        if bases is None:
            bases = [SurvivalCoxPH, SurvivalCoxnet, SurvivalRSF, SurvivalGBSA]
        self.bases = bases
        self.n_inner_folds = n_inner_folds
        self.outer_l1_ratio = outer_l1_ratio
        self.random_state = random_state
        self._fitted_bases = None        # base models fit on FULL training data
        self._meta = None                # outer Coxnet
        self._meta_scaler = None         # standardise the OOF stack

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sklearn.preprocessing import StandardScaler
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        X = np.asarray(X, dtype=np.float64)
        oof_cols = []
        for base in self.bases:
            oof, _ = _oof_predict_risk(
                base, X, y_struct,
                n_inner_folds=self.n_inner_folds,
                random_state=self.random_state)
            oof_cols.append(oof)
        S = np.column_stack(oof_cols)

        # Standardise OOF stack (different base models live on very different scales)
        self._meta_scaler = StandardScaler().fit(S)
        S_s = self._meta_scaler.transform(S)

        # Fit outer Coxnet
        self._meta = CoxnetSurvivalAnalysis(
            l1_ratio=self.outer_l1_ratio,
            alpha_min_ratio=0.01, max_iter=1000,
            fit_baseline_model=True)
        try:
            self._meta.fit(S_s, y_struct)
        except Exception:
            self._meta = CoxnetSurvivalAnalysis(
                l1_ratio=0.1, alpha_min_ratio=0.1,
                max_iter=1000, fit_baseline_model=True)
            self._meta.fit(S_s, y_struct)

        # Refit each base on the FULL training data for inference.
        self._fitted_bases = []
        for base in self.bases:
            m = base()
            m.fit(X, y_struct)
            self._fitted_bases.append(m)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        cols = [m.predict_risk(X) for m in self._fitted_bases]
        S = np.column_stack(cols)
        S_s = self._meta_scaler.transform(S)
        pred = np.asarray(self._meta.predict(S_s), dtype=np.float64)
        if pred.ndim == 2:
            pred = pred[:, -1]
        return pred
