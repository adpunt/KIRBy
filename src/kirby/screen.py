"""Pre-screen individual representations before running find_best_hybrid.

Why: when you bring a new dataset, the "top reps" list that worked for other
datasets may not transfer. Combining reps that perform poorly on the new task
just dilutes the hybrid with noise. Run `screen_reps()` first, look at the
ranked table, then pass only the reps worth keeping to `find_best_hybrid`.

Example:
    from kirby.screen import screen_reps, print_screen_table, top_n_reps

    rows = screen_reps(reps, labels, groups, task_type='regression')
    print_screen_table(rows)
    kept = top_n_reps(rows, n=8)              # keep top 8
    # or: kept = top_n_reps(rows, threshold=0.2)   # keep all above ρ=0.2

    result = find_best_hybrid(
        base_reps={k: reps[k] for k in kept},
        ...
    )
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def _default_models(task_type: str) -> Dict[str, object]:
    if task_type == 'classification':
        return {
            'ridge': RidgeClassifier(),
            'rf': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1),
        }
    return {
        'ridge': Ridge(),
        'rf': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1),
    }


def _score_one(X, y, groups, model, n_folds, task_type, metric):
    """Single (rep, model) CV. Returns mean score across folds."""
    is_clf = (task_type == 'classification')
    n_unique_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique_groups)
    if actual_folds < 2:
        return float('nan')

    if is_clf and n_unique_groups >= actual_folds:
        cv = StratifiedGroupKFold(
            n_splits=actual_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, y, groups=groups))
    else:
        cv = GroupKFold(n_splits=actual_folds)
        splits = list(cv.split(X, y, groups=groups))

    fold_scores = []
    for train_idx, val_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        m = type(model)(**model.get_params())
        m.fit(X_tr, y[train_idx])
        if metric == 'auroc' and is_clf and hasattr(m, 'decision_function'):
            yp = m.decision_function(X_va)
            try:
                s = roc_auc_score(y[val_idx], yp)
            except ValueError:
                s = float('nan')
        else:
            yp = m.predict(X_va)
            if is_clf:
                s = accuracy_score(y[val_idx], yp)
            else:
                rho, _ = spearmanr(y[val_idx], yp)
                s = float(rho) if np.isfinite(rho) else 0.0
        fold_scores.append(s)
    return float(np.nanmean(fold_scores))


def screen_reps(
    reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    task_type: str = 'regression',
    n_folds: int = 3,
    models: Optional[Dict[str, object]] = None,
    metric: Optional[str] = None,
    impute_nan: bool = True,
) -> List[dict]:
    """Evaluate each representation individually via group-aware CV.

    Args:
        reps: {rep_name: (n_samples, n_features) array}
        labels, groups: (n_samples,) arrays
        task_type: 'regression' or 'classification'
        n_folds: CV folds (default 3 for fast screening)
        models: {model_name: sklearn-like estimator}. Default: ridge + RF.
        metric: For classification, 'accuracy' (default) or 'auroc'.
        impute_nan: Replace NaN with 0 before fitting (default True).

    Returns:
        List of dicts, sorted by best score (descending):
            [{'rep': str, 'dim': int, 'score': float, 'best_model': str,
              'per_model': {model_name: score}}, ...]
    """
    if models is None:
        models = _default_models(task_type)
    if metric is None:
        metric = 'accuracy' if task_type == 'classification' else 'spearman'

    labels = np.asarray(labels)
    groups = np.asarray(groups)

    rows = []
    for rep_name, X in reps.items():
        X = np.asarray(X, dtype=np.float64)
        if impute_nan and np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        per_model = {}
        for mname, model in models.items():
            try:
                per_model[mname] = _score_one(
                    X, labels, groups, model, n_folds, task_type, metric)
            except Exception as e:
                per_model[mname] = float('nan')
        # Best across models (higher is better for ρ, accuracy, auroc)
        valid = {k: v for k, v in per_model.items() if np.isfinite(v)}
        if valid:
            best_model = max(valid, key=valid.get)
            best_score = valid[best_model]
        else:
            best_model = next(iter(per_model)) if per_model else ''
            best_score = float('nan')
        rows.append({
            'rep': rep_name,
            'dim': X.shape[1],
            'score': best_score,
            'best_model': best_model,
            'per_model': per_model,
        })
    rows.sort(key=lambda r: (-r['score']
                              if np.isfinite(r['score']) else float('inf')))
    return rows


def print_screen_table(
    rows: List[dict],
    title: Optional[str] = None,
    metric_label: Optional[str] = None,
) -> None:
    """Pretty-print the screen_reps output as a ranked table."""
    if title is not None:
        print(title)
    if not rows:
        print('  (no reps)')
        return
    model_names = sorted({m for r in rows for m in r['per_model']})
    metric_label = metric_label or 'score'
    header = f"{'#':>3}  {'Rep':<25s}  {'Dim':>6s}  {metric_label:>8s}  " + \
        "  ".join(f"{m:>10s}" for m in model_names) + "  Best"
    print(header)
    print('-' * len(header))
    for i, r in enumerate(rows, 1):
        scores = "  ".join(f"{r['per_model'].get(m, float('nan')):>10.3f}"
                            for m in model_names)
        s = r['score']
        s_str = f"{s:>8.3f}" if np.isfinite(s) else f"{'nan':>8s}"
        print(f"{i:>3}  {r['rep']:<25s}  {r['dim']:>6d}  "
              f"{s_str}  {scores}  {r['best_model']}")


def top_n_reps(
    rows: List[dict],
    n: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[str]:
    """Filter screen_reps output to the reps worth keeping.

    Pass either `n` (top-N by score) or `threshold` (all reps with score >=
    threshold). If both are given, both filters apply.

    Returns a list of rep names suitable for `{k: reps[k] for k in kept}`.
    """
    kept = list(rows)
    if threshold is not None:
        kept = [r for r in kept
                if np.isfinite(r['score']) and r['score'] >= threshold]
    if n is not None:
        kept = kept[:n]
    return [r['rep'] for r in kept]
