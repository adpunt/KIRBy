"""KIRBy-style hybrid search adapted for survival outcomes.

Mirrors the find_best_hybrid pattern (enumerate rep combinations × downstream
models, CV-evaluate, return the best) but stripped down for survival:

  * "Reps" here are the 6 feature views from `_annitia_views` rather than
    learned embeddings.
  * Downstream models are sksurv estimators (Cox, RSF), not regressors.
  * Score is censored C-index, not RMSE/R2.

Subset enumeration: all 2^V - 1 non-empty subsets of the V views (V=6 → 63).
For each (subset, model), runs n_folds CV using the loader's pre-computed
event-stratified folds and reports per-fold C-index.

The winning (subset, model) is refit on all valid patients to produce the
final estimator returned to the caller.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.annitia.survival import SURVIVAL_MODELS, concordance


@dataclass
class SurvivalSearchResult:
    endpoint: str
    per_condition: List[Dict] = field(default_factory=list)  # all results
    best: Optional[Dict] = None                              # winning condition
    final_model: object = None                               # refit on full data
    final_view_names: Tuple[str, ...] = ()
    final_model_name: str = ''


def _concat_views(views: Dict[str, Tuple[np.ndarray, list]],
                  view_names: Tuple[str, ...]) -> np.ndarray:
    return np.concatenate([views[v][0] for v in view_names], axis=1)


def _enumerate_subsets(view_names: List[str],
                       max_size: Optional[int] = None) -> List[Tuple[str, ...]]:
    out = []
    cap = max_size if max_size is not None else len(view_names)
    for k in range(1, cap + 1):
        for combo in combinations(view_names, k):
            out.append(combo)
    return out


def search_survival_hybrid(views: Dict[str, Tuple[np.ndarray, list]],
                           y_struct: np.ndarray,
                           valid_mask: np.ndarray,
                           folds: List[Tuple[np.ndarray, np.ndarray]],
                           endpoint: str,
                           model_names: Tuple[str, ...] = ('cox', 'rsf'),
                           max_subset_size: Optional[int] = None,
                           verbose: bool = True) -> SurvivalSearchResult:
    """Run the survival hybrid search.

    Args:
        views:        dict view_name -> (X[n_full], feature_names). X is
                      already imputed/scrubbed; rows align to data['patient_ids'].
        y_struct:     sksurv structured array (n_full,). Invalid rows have NaN time.
        valid_mask:   bool[n_full]. Only rows where True are used.
        folds:        list of (train_idx, val_idx) into the FULL n_full array,
                      already restricted to valid rows.
        endpoint:     'hepatic' or 'death'. Used only for reporting.
        model_names:  which sksurv models to try.
        max_subset_size: cap on number of views to combine (default: all).
    """
    view_names = list(views.keys())
    subsets = _enumerate_subsets(view_names, max_size=max_subset_size)
    results = []

    for subset in subsets:
        X_full = _concat_views(views, subset)
        for mname in model_names:
            ModelCls = SURVIVAL_MODELS[mname]
            fold_c = []
            for k, (tr, va) in enumerate(folds):
                Xtr, Xva = X_full[tr], X_full[va]
                ytr, yva = y_struct[tr], y_struct[va]
                try:
                    est = ModelCls()
                    est.fit(Xtr, ytr)
                    risk = est.predict_risk(Xva)
                    c = concordance(yva, risk)
                except Exception as exc:
                    if verbose:
                        print(f"   [{endpoint}] {'+'.join(subset):40s} "
                              f"{mname:4s} fold {k}: ERR {exc}")
                    c = float('nan')
                fold_c.append(c)
            mean_c = float(np.nanmean(fold_c)) if any(np.isfinite(fold_c)) else float('nan')
            std_c = float(np.nanstd(fold_c)) if any(np.isfinite(fold_c)) else float('nan')
            results.append({
                'views': subset,
                'model': mname,
                'fold_cindex': fold_c,
                'mean_cindex': mean_c,
                'std_cindex': std_c,
                'n_features': int(X_full.shape[1]),
            })
            if verbose:
                print(f"   [{endpoint}] {'+'.join(subset):40s} "
                      f"{mname:4s} d={X_full.shape[1]:4d} "
                      f"C={mean_c:.4f}±{std_c:.4f}")

    finite = [r for r in results if np.isfinite(r['mean_cindex'])]
    if not finite:
        return SurvivalSearchResult(endpoint=endpoint, per_condition=results)
    best = max(finite, key=lambda r: r['mean_cindex'])

    # Refit winner on all valid rows
    X_full = _concat_views(views, best['views'])
    valid_idx = np.where(valid_mask)[0]
    final_model = SURVIVAL_MODELS[best['model']]()
    final_model.fit(X_full[valid_idx], y_struct[valid_idx])

    return SurvivalSearchResult(
        endpoint=endpoint,
        per_condition=results,
        best=best,
        final_model=final_model,
        final_view_names=best['views'],
        final_model_name=best['model'],
    )


def predict_risk_for_subset(final_model,
                            views: Dict[str, Tuple[np.ndarray, list]],
                            view_names: Tuple[str, ...]) -> np.ndarray:
    """Predict risk scores using a fitted final_model + the same view subset."""
    X = _concat_views(views, view_names)
    return final_model.predict_risk(X)
