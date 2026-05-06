"""KIRBy-style hybrid search adapted for survival outcomes.

Mirrors find_best_hybrid: enumerate (rep subset × downstream model × prune
level), CV-evaluate by censored C-index, return the best. "Reps" are the
handcrafted feature views from views.py.

Subset enumeration: all 2^V - 1 non-empty subsets of the V views.
Importance prune (optional): for each (subset, model) candidate, also try a
variant where features are reduced to the top-K most important by RandomSurvivalForest
feature_importances_ (importances are computed once per fold on the training fold).

The winner is refit on all valid patients to produce the final estimator.
The top-K candidates per endpoint are also retained for ensemble at predict time.

Phase X (2026-05-06): every condition is now stamped with the standard
ConditionRecord schema (see `_schema.py`), and per-condition OOF risk vectors
are written to <output_dir>/oof/<endpoint>/<config_id>.npy when an
`output_dir` is provided. Backwards compatible: legacy field names
(`views`, `model`, `prune`, `mean_cindex`, ...) are still emitted, so any
older consumers (build_submission_zip, combine_branches, submit) keep working.
"""

import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.annitia.survival import SURVIVAL_MODELS, concordance
from experiments.annitia._schema import (
    SCHEMA_VERSION,
    condition_key,
    config_id as _config_id,
    make_record,
    upgrade_record,
)
from experiments.annitia._oof import (
    fill_oof_fold,
    init_oof_buffer,
    save_oof,
)


@dataclass
class SurvivalSearchResult:
    endpoint: str
    per_condition: List[Dict] = field(default_factory=list)
    best: Optional[Dict] = None
    final_model: object = None
    final_view_names: Tuple[str, ...] = ()
    final_model_name: str = ''
    final_prune: Optional[int] = None
    final_feature_indices: Optional[np.ndarray] = None
    # Top-K candidates kept for ensemble at predict time. Each entry is a
    # dict {views, model_name, prune, feature_indices, fitted_model}.
    top_k_models: List[Dict] = field(default_factory=list)


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


def _rsf_importance_indices(X: np.ndarray, y_struct: np.ndarray,
                            top_k: int, random_state: int = 42) -> np.ndarray:
    """Fit a quick GradientBoostingSurvivalAnalysis on (X, y_struct), return
    indices of the top-k features by feature_importances_. (RSF in sksurv does
    not expose feature_importances_; GBSA does, and is fast enough at
    n_estimators=100.) If X has <= top_k columns, returns all indices."""
    if X.shape[1] <= top_k:
        return np.arange(X.shape[1])
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    est = GradientBoostingSurvivalAnalysis(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        subsample=0.8, random_state=random_state)
    est.fit(X, y_struct)
    imp = est.feature_importances_
    return np.argsort(imp)[::-1][:top_k]


def search_survival_hybrid(views: Dict[str, Tuple[np.ndarray, list]],
                           y_struct: np.ndarray,
                           valid_mask: np.ndarray,
                           folds: List[Tuple[np.ndarray, np.ndarray]],
                           endpoint: str,
                           model_names: Tuple[str, ...] = ('cox', 'coxnet', 'rsf'),
                           max_subset_size: Optional[int] = None,
                           prune_levels: Tuple[Optional[int], ...] = (None,),
                           top_k_keep: int = 3,
                           verbose: bool = True,
                           checkpoint_callback=None,
                           prior_results: Optional[List[Dict]] = None,
                           output_dir: Optional[str] = None,
                           framing: str = 'cox',
                           tricks: Optional[List[str]] = None,
                           ) -> SurvivalSearchResult:
    """Run the survival hybrid search.

    Args:
        prune_levels: tuple of (None | int). Each int is a "keep top-K features
            by RSF importance" prune level evaluated alongside the no-prune
            baseline. Pruning is skipped silently when n_features <= K.
        top_k_keep: number of top-scoring (subset, model, prune) configurations
            to retain (refitted on all valid rows) for ensemble at predict time.
        checkpoint_callback: optional callable(results_list) invoked after each
            condition's CV is complete. Used by run.py to write the JSON
            incrementally so SLURM timeouts don't lose work.
        prior_results: optional list of already-evaluated condition dicts
            (e.g. loaded from a checkpoint JSON). Conditions matching a prior
            (subset, model, prune, framing, tricks) tuple are skipped — their
            entry is reused verbatim (with on-the-fly upgrade to the v1 schema).
        output_dir: if provided, OOF risk vectors are written to
            <output_dir>/oof/<endpoint>/<config_id>.npy and the path is stamped
            on each condition record. Required by Layer 5 stack_meta.
        framing: Layer 2 label (default 'cox'). Currently this search loop only
            evaluates the Cox framing; later phases will pass other strings.
        tricks: Layer 4 trick labels for this batch (default []).
    """
    view_names = list(views.keys())
    subsets = _enumerate_subsets(view_names, max_size=max_subset_size)

    # Lift any prior (legacy or v1) records into the v1 schema before resume
    raw_prior = list(prior_results) if prior_results else []
    results: List[Dict] = [upgrade_record(r, endpoint=endpoint) for r in raw_prior]
    completed_keys = {condition_key(r) for r in results}
    if completed_keys and verbose:
        print(f"   [{endpoint}] resuming with {len(completed_keys)} already-evaluated conditions")

    n_total = len(valid_mask)
    tricks = list(tricks or [])

    for subset in subsets:
        X_full = _concat_views(views, subset)
        for mname in model_names:
            ModelCls = SURVIVAL_MODELS[mname]
            for prune in prune_levels:
                if prune is not None and X_full.shape[1] <= prune:
                    # Equivalent to no-prune; skip duplicate.
                    continue
                # Build the condition key on (subset, model, prune, framing, tricks)
                cid = _config_id(subset, prune, framing, mname, tricks)
                key = (tuple(subset), mname, prune, framing, tuple(sorted(tricks)))
                if key in completed_keys:
                    continue  # already done in a prior run

                t0 = time.time()
                fold_c = []
                oof_buf = init_oof_buffer(n_total) if output_dir else None
                for k, (tr, va) in enumerate(folds):
                    Xtr, Xva = X_full[tr], X_full[va]
                    ytr, yva = y_struct[tr], y_struct[va]
                    try:
                        if prune is not None:
                            keep = _rsf_importance_indices(Xtr, ytr, prune)
                            Xtr = Xtr[:, keep]
                            Xva = Xva[:, keep]
                        est = ModelCls()
                        est.fit(Xtr, ytr)
                        risk = est.predict_risk(Xva)
                        c = concordance(yva, risk)
                        if oof_buf is not None:
                            fill_oof_fold(oof_buf, va, risk)
                    except Exception as exc:
                        if verbose:
                            err = repr(exc) if str(exc) == '' else str(exc)
                            print(f"   [{endpoint}] {'+'.join(subset):40s} "
                                  f"{mname:6s} prune={prune} fold {k}: ERR {err}")
                        c = float('nan')
                    fold_c.append(c)

                mean_c = float(np.nanmean(fold_c)) if any(np.isfinite(fold_c)) else float('nan')
                std_c = float(np.nanstd(fold_c)) if any(np.isfinite(fold_c)) else float('nan')
                elapsed = time.time() - t0

                oof_path: Optional[str] = None
                if output_dir is not None and oof_buf is not None:
                    oof_path = save_oof(oof_buf, output_dir, endpoint, cid)

                rec = make_record(
                    endpoint=endpoint,
                    views=subset,
                    model=mname,
                    prune=prune,
                    fold_cindex=fold_c,
                    mean_cindex=mean_c,
                    std_cindex=std_c,
                    n_features=int(X_full.shape[1]),
                    framing=framing,
                    tricks=tricks,
                    oof_path=oof_path,
                    elapsed_sec=elapsed,
                )
                results.append(rec)
                if checkpoint_callback is not None:
                    try:
                        checkpoint_callback(results)
                    except Exception as cb_exc:
                        if verbose:
                            print(f"   [{endpoint}] checkpoint callback error: {cb_exc}")
                if verbose:
                    p_label = f'p{prune}' if prune else 'all'
                    print(f"   [{endpoint}] {'+'.join(subset):40s} "
                          f"{mname:6s} d={X_full.shape[1]:4d} {p_label:>5s} "
                          f"C={mean_c:.4f}±{std_c:.4f}")

    finite = [r for r in results if np.isfinite(r['mean_cindex'])]
    if not finite:
        return SurvivalSearchResult(endpoint=endpoint, per_condition=results)
    finite_sorted = sorted(finite, key=lambda r: -r['mean_cindex'])
    best = finite_sorted[0]

    valid_idx = np.where(valid_mask)[0]
    y_valid = y_struct[valid_idx]

    def _refit(cfg: Dict):
        X_full = _concat_views(views, tuple(cfg['views']))[valid_idx]
        if cfg['prune'] is not None and X_full.shape[1] > cfg['prune']:
            keep = _rsf_importance_indices(X_full, y_valid, cfg['prune'])
        else:
            keep = np.arange(X_full.shape[1])
        Xp = X_full[:, keep]
        est = SURVIVAL_MODELS[cfg['model']]()
        est.fit(Xp, y_valid)
        return est, keep

    final_model, final_keep = _refit(best)

    # Top-K refits for ensemble. Keep entries even if scores collide.
    top_k_models: List[Dict] = []
    for cfg in finite_sorted[:top_k_keep]:
        est, keep = _refit(cfg)
        top_k_models.append({
            'views': tuple(cfg['views']),
            'model_name': cfg['model'],
            'prune': cfg['prune'],
            'feature_indices': keep,
            'fitted_model': est,
            'mean_cindex': cfg['mean_cindex'],
            'config_id': cfg.get('config_id'),
        })

    return SurvivalSearchResult(
        endpoint=endpoint,
        per_condition=results,
        best=best,
        final_model=final_model,
        final_view_names=tuple(best['views']),
        final_model_name=best['model'],
        final_prune=best['prune'],
        final_feature_indices=final_keep,
        top_k_models=top_k_models,
    )


def predict_risk_for_subset(final_model,
                            views: Dict[str, Tuple[np.ndarray, list]],
                            view_names: Tuple[str, ...],
                            feature_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Predict risk scores using a fitted final_model + view subset (+ optional
    pruned column indices)."""
    X = _concat_views(views, view_names)
    if feature_indices is not None:
        X = X[:, feature_indices]
    return final_model.predict_risk(X)


def rank_ensemble(top_k_models: List[Dict],
                  views: Dict[str, Tuple[np.ndarray, list]]) -> np.ndarray:
    """Rank-average risk predictions from a list of fitted models. Each model
    in `top_k_models` has its own (views, prune, feature_indices) configuration.

    Output: a single risk vector (higher = worse) whose values are the average
    of within-model rank percentiles across all models in the ensemble.
    Rank-averaging avoids issues with predictors on different scales (e.g., Cox
    linear predictor vs RSF cumulative hazard).
    """
    if not top_k_models:
        raise ValueError('top_k_models is empty')
    n = len(views[next(iter(views))][0])
    rank_sum = np.zeros(n)
    for cfg in top_k_models:
        risk = predict_risk_for_subset(
            cfg['fitted_model'], views, cfg['views'],
            feature_indices=cfg.get('feature_indices'))
        # argsort gives ranks where ties get arbitrary but stable ordering;
        # using rankdata-style average ranks would be slightly cleaner but
        # plain argsort is sufficient for C-index purposes.
        ranks = np.empty(n, dtype=np.float64)
        ranks[np.argsort(risk)] = np.arange(n)
        rank_sum += ranks
    return rank_sum / len(top_k_models)
