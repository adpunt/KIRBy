#!/usr/bin/env python3
"""C11: Model-level fusion (ensemble) vs feature-level fusion (hybrid).

Paper: Complexity
Core question: an ensemble of one model per representation aggregates K sets of
predictions and needs all K featurisers + K models at inference forever. The
hybrid pays a ONE-OFF search, then deploys a SINGLE model on a fused feature
vector. Does the hybrid recover the ensemble's accuracy at a fraction of the
standing cost?

Arms (all scored on the SAME folds, so every comparison is paired):
  1. best_single    - one model per base rep, report the best rep
  2. pca_matched    - concat all reps, PCA to the budget dim, one model
  3. hybrid         - one model on the searched fused representation
  4. ensemble_s1    - STYLE 1 (per model): fix one downstream model, train it on
                      every rep in a rep set, aggregate the PREDICTIONS
  5. ensemble_s2    - STYLE 2 (optimal): let each rep pick its own best model,
                      then aggregate those predictions
  6. oracle_single  - best (model x rep) chosen on the held-out fold itself.
                      NOT a fair arm; reported only as an upper bound.

Rep sets (the "(1) top / (2) general" split):
  top   - the top-3 base reps
  all   - every rep eligible as a standalone baseline (tier >= 2)

Aggregators swept per (model, rep set): mean, median, 20% trimmed mean,
IQR-filtered mean, MAD-filtered mean, score-weighted mean, top-k mean, and
(classification only) majority vote.

-----------------------------------------------------------------------------
HISTORY / DIRECTIVE RECONCILIATION
-----------------------------------------------------------------------------
The prediction-averaging arm was removed on 2026-05-12 (commit 5d72d3c) under
the "no averaged baselines" directive. It is reinstated here because that
directive and this experiment are about different objects:

  * BANNED, and still banned: averaging rep SCORES into a composite "mean
    baseline" number. It hides which reps are good. Nothing here does that -
    baselines are still reported per rep, individually.
  * REINSTATED: averaging model PREDICTIONS. That is an ensemble - a real
    method and the strongest competitor the hybrid has to beat. It is scored
    as a METHOD arm, never as a baseline.

-----------------------------------------------------------------------------
SELECTION OPTIMISM (why this is not the s05 bug again)
-----------------------------------------------------------------------------
Sweeping aggregators is a SEARCH. Reporting the best aggregator picked on the
folds it was scored on would be exactly the post-hoc-max optimism fixed in s05.
So:

  * Every (rep set x aggregator) rule is a pre-registered CANDIDATE with
    per-fold scores. `honest_rescore` re-picks the winner leave-one-fold-out.
  * Rep-set MEMBERSHIP is chosen per held-out fold from the OTHER folds' scores
    (a fixed rule, not a peek). Aggregator WEIGHTS likewise.
  * The same held-out tax lands on the hybrid, the ensemble and the baselines.
  * The one arm that does peek, `oracle_single`, is labelled as such and is an
    upper bound only.

-----------------------------------------------------------------------------
COST ACCOUNTING
-----------------------------------------------------------------------------
Per Dehghani et al., "The Efficiency Misnomer" (ICLR 2022), no single cost
number is trustworthy; several are reported per arm:

  n_featurisers            distinct base reps that must be COMPUTED at inference
  featuriser_dim           summed width of those reps (upstream encoder cost)
  n_models_deployed        models to store and run
  model_input_dim          width of the downstream model's input
  model_bytes              pickled size of the deployed model(s)
  storage_bytes_per_sample bytes that must be PERSISTED per sample
  fit_time_s / predict_time_s  measured wall-clock, summed over members

The honest caveat, stated up front because it is the first thing a reviewer
attacks: the hybrid does NOT escape the upstream featurisers. It still has to
run every rep its blueprint draws a column from. Its saving is real only insofar
as the blueprint touches FEWER reps than the ensemble does - so `n_featurisers`
is measured for the hybrid from `feature_info['allocation']`, not assumed.
PCA is the worst offender and is measured, not excused: it needs ALL reps at
inference plus a dense (budget x total_dim) projection matrix.

Search cost is amortised: `break_even` reports how many inferences it takes for
the hybrid's one-off search to be repaid by its per-inference saving, in model
fits (hardware-independent) and in seconds.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import pickle

import numpy as np
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
    get_model_factories, cv_split,
    get_domain_metric, cap_dimensions,
    build_fold_hybrid, extract_top3_base_reps,
    get_best_search_for_downstream,
    honest_rescore, classify_rep_tier,
)

# Rep sets swept for both ensemble styles.
TOP_K = 3
# Members kept by the 'topk_mean' aggregator.
AGG_TOP_K = 3


# =========================================================================
# PREDICTION CAPTURE
# =========================================================================

def _fit_capture(model_factory, X_train, y_train, X_val, task_type):
    """Fit once and capture everything the ensemble arms need.

    Returns (model, agg_pred, hard_pred, fit_time, predict_time).

    `agg_pred` is what the aggregators average - it must live in a CONTINUOUS
    space or averaging is meaningless:
      regression      -> the predicted value
      classification  -> the predict_proba matrix (n_val, n_classes)
    `hard_pred` is the raw predict() output, kept for majority voting.
    """
    model = model_factory()
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    hard_pred = model.predict(X_val)
    predict_time = time.time() - t0

    agg_pred = hard_pred
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        try:
            agg_pred = model.predict_proba(X_val)
        except Exception:
            agg_pred = hard_pred
    return model, agg_pred, hard_pred, fit_time, predict_time


def _to_metric_space(agg, task_type, metric_needs_proba, classes):
    """Map an aggregated prediction back into the space the metric expects.

    Averaging happens in probability space; the metric may want scores (AUC) or
    hard labels (accuracy/f1/mcc). Getting this wrong silently scores garbage,
    e.g. handing 0.4 to accuracy_score.
    """
    agg = np.asarray(agg)
    if task_type != 'classification':
        return agg

    if agg.ndim == 1:
        # Members had no predict_proba, so these are hard labels. A mean of
        # hard labels is a fractional non-class (0/1 -> 0.4) and would be
        # silently garbage to accuracy/f1/mcc: snap it back to the nearest
        # real class. A majority vote is already a class, so this is a no-op
        # for that path. AUC, by contrast, wants the continuous score.
        if metric_needs_proba:
            return agg
        cls = np.asarray(classes)
        return cls[np.argmin(np.abs(cls[None, :] - agg[:, None]), axis=1)]

    if metric_needs_proba:
        return agg[:, 1] if agg.shape[1] == 2 else agg
    return np.asarray(classes)[np.argmax(agg, axis=1)]


def _model_bytes(model):
    """Pickled size of a fitted model, in bytes."""
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0


# =========================================================================
# AGGREGATORS
# =========================================================================
# Each takes a stack P of member predictions with members on axis 0:
#   regression      P.shape == (M, n_val)
#   classification  P.shape == (M, n_val, n_classes)
# and per-member weights w (M,), and returns the aggregate WITHOUT axis 0.
# Every one of them is a fixed rule - none inspects the held-out fold.

def _agg_mean(P, w):
    return P.mean(axis=0)


def _agg_median(P, w):
    return np.median(P, axis=0)


def _agg_trimmed(P, w):
    """20% trimmed mean - drops the most extreme member at each tail."""
    if P.shape[0] < 5:
        return np.median(P, axis=0)
    return sp_stats.trim_mean(P, 0.2, axis=0)


def _mask_mean(P, keep):
    """Mean of P over axis 0 using a boolean keep-mask, broadcast-safe.

    Falls back to the median wherever a sample rejected every member.
    """
    kf = keep.astype(float)
    while kf.ndim < P.ndim:
        kf = kf[..., None]
    denom = kf.sum(axis=0)
    out = (P * kf).sum(axis=0) / np.where(denom == 0, 1.0, denom)
    med = np.median(P, axis=0)
    return np.where(denom == 0, med, out)


def _agg_iqr_filtered(P, w):
    """Per-sample Tukey fence: drop members outside [Q1-1.5*IQR, Q3+1.5*IQR]."""
    if P.shape[0] < 4:
        return P.mean(axis=0)
    q1 = np.percentile(P, 25, axis=0)
    q3 = np.percentile(P, 75, axis=0)
    iqr = q3 - q1
    keep = (P >= (q1 - 1.5 * iqr)) & (P <= (q3 + 1.5 * iqr))
    return _mask_mean(P, keep)


def _agg_mad_filtered(P, w):
    """Per-sample MAD fence: drop members >3 scaled-MAD from the median."""
    if P.shape[0] < 4:
        return P.mean(axis=0)
    med = np.median(P, axis=0)
    mad = np.median(np.abs(P - med), axis=0) * 1.4826
    mad = np.where(mad == 0, 1e-12, mad)
    keep = np.abs(P - med) <= (3.0 * mad)
    return _mask_mean(P, keep)


def _agg_weighted(P, w):
    """Mean weighted by each member's score on the OTHER folds."""
    if w is None or not np.isfinite(w).any() or w.sum() <= 0:
        return P.mean(axis=0)
    ww = np.asarray(w, dtype=float)
    while ww.ndim < P.ndim:
        ww = ww[..., None]
    return (P * ww).sum(axis=0) / ww.sum(axis=0)


def _agg_topk(P, w):
    """Mean of the AGG_TOP_K members that scored best on the OTHER folds."""
    if w is None or P.shape[0] <= AGG_TOP_K:
        return P.mean(axis=0)
    idx = np.argsort(-np.asarray(w, dtype=float))[:AGG_TOP_K]
    return P[idx].mean(axis=0)


def _agg_majority(P, w):
    """Hard majority vote. Classification only; P is (M, n_val) hard labels."""
    m = sp_stats.mode(P, axis=0, keepdims=False)
    return np.asarray(m.mode)


CONTINUOUS_AGGREGATORS = {
    'mean': _agg_mean,
    'median': _agg_median,
    'trimmed20': _agg_trimmed,
    'iqr_filtered': _agg_iqr_filtered,
    'mad_filtered': _agg_mad_filtered,
    'weighted': _agg_weighted,
    f'top{AGG_TOP_K}_mean': _agg_topk,
}


def _rank_weights(scores, lower_is_better):
    """Turn other-fold scores into non-negative weights.

    Rank-based, so a single blown-up member (a rep whose model diverged) cannot
    dominate the weighting the way raw-score weights would.
    """
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    if not finite.any():
        return None
    filler = (np.nanmax(s[finite]) if lower_is_better else np.nanmin(s[finite]))
    s = np.where(finite, s, filler)
    order = np.argsort(s) if lower_is_better else np.argsort(-s)
    w = np.empty(len(s), dtype=float)
    w[order] = np.arange(len(s), 0, -1)  # best member gets the largest weight
    return w


def _mean_excluding(fold_scores, f):
    """Mean score over every fold EXCEPT f. The honest ranking signal."""
    other = [s for i, s in enumerate(fold_scores)
             if i != f and s is not None and np.isfinite(s)]
    return float(np.mean(other)) if other else float('nan')


def _rank_members(names, per_rep_folds, f, lower_is_better):
    """Rank members for held-out fold f using only the other folds' scores."""
    scored = []
    for n in names:
        m = _mean_excluding(per_rep_folds[n], f)
        if np.isfinite(m):
            scored.append((n, m))
    scored.sort(key=lambda x: x[1], reverse=not lower_is_better)
    return scored


# =========================================================================
# COST MODEL
# =========================================================================

def _cost_row(n_featurisers, featuriser_dim, n_models, model_input_dim,
              model_bytes, fit_s, predict_s, storage_dim):
    """One arm's standing (post-search) cost. Several metrics, never one."""
    return {
        'n_featurisers': int(n_featurisers),
        'featuriser_dim': int(featuriser_dim),
        'n_models_deployed': int(n_models),
        'model_input_dim': int(model_input_dim),
        'model_bytes': int(model_bytes),
        'storage_bytes_per_sample': int(4 * storage_dim),  # float32
        'total_fit_time_s': float(fit_s),
        'total_predict_time_s': float(predict_s),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run ensemble (model-level fusion) vs hybrid (feature-level fusion)."""
    print("\n" + "=" * 78)
    print("EXPERIMENT C11: ENSEMBLE (model-level) VS HYBRID (feature-level)")
    print("=" * 78)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']

    from kirby.find_best_hybrid import BUILTIN_METRICS
    metric_needs_proba = (metric_fn is BUILTIN_METRICS.get('auc', (None,))[0])

    models = get_model_factories(config['models'],
                                 task_type=task_type,
                                 n_samples=len(data['labels']))
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    rep_names = sorted(reps.keys())
    rep_dims = {k: int(v.shape[1]) for k, v in reps.items()}
    n_reps = len(rep_names)
    total_dim = sum(rep_dims.values())

    # "General reps" = every rep legal as a STANDALONE baseline. Tier 1 (<50D)
    # is augmentation-only and must never stand alone, so it cannot be an
    # ensemble member either.
    general_reps = [r for r in rep_names if classify_rep_tier(rep_dims[r], r) >= 2]
    if not general_reps:
        general_reps = list(rep_names)

    classes = np.unique(labels) if task_type == 'classification' else None

    print(f"  {n_reps} base reps ({len(general_reps)} tier>=2 / ensemble-eligible), "
          f"total dim={total_dim}, {len(models)} models, {n_folds} folds")
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})\n")

    sample_X = next(iter(reps.values()))
    n_samples = len(labels)
    from experiments._shared import FULL_BUDGETS
    budgets = [b for b in FULL_BUDGETS if b <= total_dim and b <= n_samples]

    # Materialise the folds ONCE. Every arm must see byte-identical splits or
    # the ensemble cannot aggregate across reps and the comparison is unpaired.
    folds = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))
    n_folds_actual = len(folds)

    augmentations = data.get('augmentations')
    last_search = None
    # One-off search size, in model fits (hardware-independent currency).
    search_fits = 0

    def _blueprints_for(model_name):
        """Best hybrid blueprint per budget, optimised FOR THIS downstream model."""
        nonlocal last_search, search_fits
        bps = {}
        for budget in budgets:
            try:
                search, meta = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=model_name, budget=budget,
                    augmentations=augmentations,
                    speed=config.get('search_speed', 'full'),
                    dataset_name=dataset_name, task_type=task_type,
                    data=data,
                )
            except Exception as e:
                print(f"    Blueprint lookup failed ({model_name}, b={budget}): {e}")
                continue
            if meta:
                bps[budget] = meta['best_config']['blueprint']
                last_search = search
                n_cfg = len(getattr(search, 'all_results', None) or [])
                search_fits = max(search_fits, n_cfg * n_folds_actual)
        return bps

    results = []
    honest_candidates = {}          # hybrid recipes  -> honest_rescore
    honest_baselines = {}           # single reps     -> honest_rescore
    honest_ens_candidates = {}      # ensemble rules  -> honest_rescore
    cost_rows = []

    # Per-model prediction store, filled by the single-rep arm and reused (for
    # free) by both ensemble styles. member_agg[model][rep][fold] = predictions.
    member_agg = {}
    member_hard = {}
    member_folds = {}   # model -> rep -> [per-fold score]
    member_cost = {}    # model -> rep -> (fit_s, predict_s, bytes)

    y_val_by_fold = [labels[v] for _, v in folds]

    for model_name, model_factory in models.items():
        print(f"\n  --- {model_name} ---")

        blueprints = _blueprints_for(model_name)
        if not blueprints:
            print(f"    No blueprint for {model_name} - hybrid arm empty.")

        member_agg[model_name] = {}
        member_hard[model_name] = {}
        member_folds[model_name] = {}
        member_cost[model_name] = {}

        # =================================================================
        # 1. Best single base rep  (also trains every ensemble member)
        # =================================================================
        best_single_score = np.inf if lower_is_better else -np.inf
        best_single_rep = None
        single_rep_results = {}
        model_baseline_folds = {}
        model_candidate_folds = []

        for rep_name in rep_names:
            X_rep = reps[rep_name]
            fold_scores, fold_agg, fold_hard = [], [], []
            ft_tot = pt_tot = 0.0
            mbytes = 0

            for train_idx, val_idx in folds:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_rep[train_idx])
                X_val = scaler.transform(X_rep[val_idx])
                X_train, X_val = cap_dimensions(X_train, X_val)
                y_train, y_val = labels[train_idx], labels[val_idx]

                try:
                    model, agg, hard, ft, pt = _fit_capture(
                        model_factory, X_train, y_train, X_val, task_type)
                except Exception:
                    fold_scores.append(float('nan'))
                    fold_agg.append(None)
                    fold_hard.append(None)
                    continue

                scored = _to_metric_space(agg, task_type, metric_needs_proba, classes)
                fold_scores.append(float(metric_fn(y_val, scored)))
                fold_agg.append(np.asarray(agg))
                fold_hard.append(np.asarray(hard))
                ft_tot += ft
                pt_tot += pt
                mbytes = max(mbytes, _model_bytes(model))

            fold_scores_f = [float(s) for s in fold_scores]
            model_baseline_folds[rep_name] = fold_scores_f
            member_agg[model_name][rep_name] = fold_agg
            member_hard[model_name][rep_name] = fold_hard
            member_folds[model_name][rep_name] = fold_scores_f
            member_cost[model_name][rep_name] = (
                ft_tot / max(n_folds_actual, 1),
                pt_tot / max(n_folds_actual, 1),
                mbytes)

            valid = [s for s in fold_scores_f if np.isfinite(s)]
            score_mean = float(np.mean(valid)) if valid else float('nan')
            single_rep_results[rep_name] = {
                'score_mean': score_mean,
                'score_std': float(np.std(valid)) if valid else float('nan'),
                'dim': rep_dims[rep_name],
                'fit_time_s': ft_tot / max(n_folds_actual, 1),
                'predict_time_s': pt_tot / max(n_folds_actual, 1),
                'model_bytes': mbytes,
                'fold_scores': fold_scores_f,
            }
            if np.isfinite(score_mean) and (
                    (lower_is_better and score_mean < best_single_score) or
                    (not lower_is_better and score_mean > best_single_score)):
                best_single_score = score_mean
                best_single_rep = rep_name

        if best_single_rep is None:
            print("    Best single: NONE (all reps failed)")
            honest_baselines[model_name] = model_baseline_folds
            continue

        bs = single_rep_results[best_single_rep]
        print(f"    Best single: {best_single_rep} "
              f"{metric_name}={bs['score_mean']:.4f} dim={bs['dim']}")

        results.append({
            'model': model_name, 'method': 'best_single', 'augmented': False,
            'score_mean': bs['score_mean'], 'score_std': bs['score_std'],
            'total_dim': bs['dim'], 'n_models_trained': 1,
            'total_fit_time_s': bs['fit_time_s'],
            'total_predict_time_s': bs['predict_time_s'],
            'fold_scores': bs['fold_scores'],
            'detail': {'best_rep': best_single_rep, 'all_reps': single_rep_results},
            'cost': _cost_row(1, bs['dim'], 1, bs['dim'], bs['model_bytes'],
                              bs['fit_time_s'], bs['predict_time_s'], bs['dim']),
        })
        cost_rows.append((model_name, 'best_single', results[-1]['cost'],
                          bs['score_mean']))

        # =================================================================
        # 2. PCA-matched control
        # =================================================================
        for budget in budgets:
            f_scores, f_fit, f_pred = [], [], []
            n_comp = 0
            pca_bytes = 0
            for train_idx, val_idx in folds:
                y_train, y_val = labels[train_idx], labels[val_idx]
                X_tr_cat = np.hstack([reps[r][train_idx] for r in rep_names])
                X_va_cat = np.hstack([reps[r][val_idx] for r in rep_names])

                n_comp = min(budget, X_tr_cat.shape[1], len(y_train) - 1)
                if n_comp < 2:
                    continue

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_cat)
                X_va_s = scaler.transform(X_va_cat)

                try:
                    pca = PCA(n_components=n_comp, random_state=42)
                    X_tr_p = pca.fit_transform(X_tr_s)
                    X_va_p = pca.transform(X_va_s)
                    model, agg, hard, ft, pt = _fit_capture(
                        model_factory, X_tr_p, y_train, X_va_p, task_type)
                except Exception:
                    continue

                scored = _to_metric_space(agg, task_type, metric_needs_proba, classes)
                f_scores.append(float(metric_fn(y_val, scored)))
                f_fit.append(ft)
                f_pred.append(pt)
                # PCA's own projection matrix is part of what you must deploy.
                pca_bytes = max(pca_bytes,
                                _model_bytes(model) + pca.components_.nbytes)

            if not f_scores:
                continue
            results.append({
                'model': model_name, 'method': 'pca_matched', 'budget': budget,
                'score_mean': float(np.mean(f_scores)),
                'score_std': float(np.std(f_scores)),
                'total_dim': n_comp, 'n_models_trained': 1,
                'total_fit_time_s': float(np.mean(f_fit)),
                'total_predict_time_s': float(np.mean(f_pred)),
                'augmented': False,
                'fold_scores': [float(s) for s in f_scores],
                # PCA needs EVERY rep at inference plus a dense projection.
                'cost': _cost_row(n_reps, total_dim, 1, n_comp, pca_bytes,
                                  float(np.mean(f_fit)), float(np.mean(f_pred)),
                                  total_dim),
            })
            cost_rows.append((model_name, f'pca_b{budget}', results[-1]['cost'],
                              results[-1]['score_mean']))

        # =================================================================
        # 3. Hybrid: one model on the fused representation
        # =================================================================
        for budget in budgets:
            if budget not in blueprints:
                continue

            f_scores, f_fit, f_pred = [], [], []
            n_feat = 0
            hyb_bytes = 0
            touched = set()

            for train_idx, val_idx in folds:
                tr_reps = {k: v[train_idx] for k, v in reps.items()}
                va_reps = {k: v[val_idx] for k, v in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]

                augs = augmentations or {}
                tr_aug = {k: v[train_idx] for k, v in augs.items()} if augs else None
                va_aug = {k: v[val_idx] for k, v in augs.items()} if augs else None

                try:
                    t0 = time.time()
                    X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                        tr_reps, va_reps, y_train, blueprints[budget],
                        augmentations_train=tr_aug, augmentations_val=va_aug)
                    build_s = time.time() - t0
                except Exception:
                    continue

                # THE reviewer's question: how many featurisers does the hybrid
                # still need? Measured from the blueprint, never assumed.
                alloc = feat_info.get('allocation') or {}
                touched |= {r for r, n in alloc.items()
                            if isinstance(n, (int, float)) and n > 0}
                aug_info = feat_info.get('augmentation_info') or {}
                if isinstance(aug_info, dict):
                    touched |= {k for k in aug_info.keys() if k in reps}

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_h)
                X_va_s = scaler.transform(X_va_h)
                n_feat = X_tr_h.shape[1]

                try:
                    model, agg, hard, ft, pt = _fit_capture(
                        model_factory, X_tr_s, y_train, X_va_s, task_type)
                except Exception:
                    continue

                scored = _to_metric_space(agg, task_type, metric_needs_proba, classes)
                f_scores.append(float(metric_fn(y_val, scored)))
                f_fit.append(ft + build_s)
                f_pred.append(pt)
                hyb_bytes = max(hyb_bytes, _model_bytes(model))

            if not f_scores:
                continue

            b_label = 'full' if budget == total_dim else str(budget)
            score = float(np.mean(f_scores))
            n_touch = len(touched) if touched else n_reps
            touch_dim = sum(rep_dims.get(r, 0) for r in touched) or total_dim
            print(f"    Hybrid b={b_label:<5}: {metric_name}={score:.4f} "
                  f"dim={n_feat} featurisers={n_touch}/{n_reps}")

            f_scores_f = [float(s) for s in f_scores]
            model_candidate_folds.append({'name': f'hybrid_b{b_label}',
                                          'fold_scores': f_scores_f})
            results.append({
                'model': model_name, 'method': 'hybrid', 'augmented': False,
                'budget': budget, 'score_mean': score,
                'score_std': float(np.std(f_scores)),
                'total_dim': n_feat, 'n_models_trained': 1,
                'total_fit_time_s': float(np.mean(f_fit)),
                'total_predict_time_s': float(np.mean(f_pred)),
                'fold_scores': f_scores_f,
                'reps_touched': sorted(touched),
                'cost': _cost_row(n_touch, touch_dim, 1, n_feat, hyb_bytes,
                                  float(np.mean(f_fit)), float(np.mean(f_pred)),
                                  n_feat),
            })
            cost_rows.append((model_name, f'hybrid_b{b_label}',
                              results[-1]['cost'], score))

        honest_baselines[model_name] = model_baseline_folds
        honest_candidates[model_name] = model_candidate_folds

    # =====================================================================
    # 4/5. ENSEMBLE ARMS - pure arithmetic on the predictions already stored.
    #      No model is fitted here. That is the point: the aggregator sweep is
    #      free, so the ONLY thing separating the two styles is cost at deploy.
    # =====================================================================
    print("\n" + "=" * 78)
    print("  ENSEMBLE ARMS (no new model fits - reusing stored OOF predictions)")
    print("=" * 78)

    def _stack(members, f, use_hard=False):
        """Stack fold-f predictions for `members`; drop members that failed."""
        src = member_hard if use_hard else member_agg
        rows, keep = [], []
        for (mdl, rep) in members:
            p = src[mdl][rep][f]
            if p is None:
                continue
            rows.append(p)
            keep.append((mdl, rep))
        if not rows:
            return None, []
        try:
            return np.stack(rows, axis=0), keep
        except ValueError:
            return None, []

    def _score_rule(rule_members_fn, use_hard=False, agg_fn=None):
        """Score one pre-registered ensemble rule across every held-out fold.

        rule_members_fn(f) -> ([(model, rep), ...], weights) using ONLY the
        other folds. Returns the per-fold score list.
        """
        out = []
        for f in range(n_folds_actual):
            members, w = rule_members_fn(f)
            if not members or len(members) < 2:
                out.append(float('nan'))
                continue
            P, kept = _stack(members, f, use_hard=use_hard)
            if P is None or len(kept) < 2:
                out.append(float('nan'))
                continue
            if w is not None and len(w) == len(members):
                wmap = dict(zip(members, w))
                w_kept = np.array([wmap[k] for k in kept], dtype=float)
            else:
                w_kept = None
            try:
                agg = agg_fn(P, w_kept)
                scored = _to_metric_space(agg, task_type,
                                          metric_needs_proba, classes)
                out.append(float(metric_fn(y_val_by_fold[f], scored)))
            except Exception:
                out.append(float('nan'))
        return out

    rep_sets = {'top': TOP_K, 'all': None}

    for model_name in models:
        if model_name not in member_folds or not member_folds[model_name]:
            continue
        honest_ens_candidates.setdefault(model_name, [])
        per_rep = member_folds[model_name]
        eligible = [r for r in general_reps if r in per_rep]

        print(f"\n  {model_name}:")

        # -------- STYLE 1: fixed model, one member per rep ----------------
        for set_name, k in rep_sets.items():
            def _members_s1(f, _k=k):
                ranked = _rank_members(eligible, per_rep, f, lower_is_better)
                if _k is not None:
                    ranked = ranked[:_k]
                names = [n for n, _ in ranked]
                w = _rank_weights([s for _, s in ranked], lower_is_better)
                return [(model_name, n) for n in names], w

            aggs = dict(CONTINUOUS_AGGREGATORS)
            for agg_name, agg_fn in aggs.items():
                fs = _score_rule(_members_s1, agg_fn=agg_fn)
                if not any(np.isfinite(s) for s in fs):
                    continue
                _emit_ensemble(
                    results, cost_rows, honest_ens_candidates,
                    model_name, 'ensemble_s1', set_name, agg_name, fs,
                    _members_s1, member_cost, rep_dims, metric_name,
                    n_folds_actual)

            # Majority vote only makes sense for hard-label metrics. Under AUC a
            # 0/1 vote throws away the ranking the metric is built on, so skip it.
            if task_type == 'classification' and not metric_needs_proba:
                fs = _score_rule(_members_s1, use_hard=True, agg_fn=_agg_majority)
                if any(np.isfinite(s) for s in fs):
                    _emit_ensemble(
                        results, cost_rows, honest_ens_candidates,
                        model_name, 'ensemble_s1', set_name, 'majority_vote', fs,
                        _members_s1, member_cost, rep_dims, metric_name,
                        n_folds_actual)

    # -------- STYLE 2: each rep contributes its OWN best model ------------
    # Cross-model, so it lives outside the per-model loop. Keyed under the
    # pseudo-model 'optimal' so honest_rescore treats it as its own arm.
    all_models = [m for m in models if m in member_folds and member_folds[m]]
    if len(all_models) >= 1:
        honest_ens_candidates.setdefault('optimal', [])
        print(f"\n  optimal (best model per rep):")

        def _best_model_for(rep, f):
            """Pick rep's best model from the OTHER folds only."""
            best, best_s = None, (np.inf if lower_is_better else -np.inf)
            for m in all_models:
                if rep not in member_folds[m]:
                    continue
                s = _mean_excluding(member_folds[m][rep], f)
                if not np.isfinite(s):
                    continue
                if (lower_is_better and s < best_s) or \
                   (not lower_is_better and s > best_s):
                    best, best_s = m, s
            return best, best_s

        for set_name, k in rep_sets.items():
            def _members_s2(f, _k=k):
                scored = []
                for rep in general_reps:
                    m, s = _best_model_for(rep, f)
                    if m is not None and np.isfinite(s):
                        scored.append(((m, rep), s))
                scored.sort(key=lambda x: x[1], reverse=not lower_is_better)
                if _k is not None:
                    scored = scored[:_k]
                members = [mr for mr, _ in scored]
                w = _rank_weights([s for _, s in scored], lower_is_better)
                return members, w

            for agg_name, agg_fn in CONTINUOUS_AGGREGATORS.items():
                fs = _score_rule(_members_s2, agg_fn=agg_fn)
                if not any(np.isfinite(s) for s in fs):
                    continue
                _emit_ensemble(
                    results, cost_rows, honest_ens_candidates,
                    'optimal', 'ensemble_s2', set_name, agg_name, fs,
                    _members_s2, member_cost, rep_dims, metric_name,
                    n_folds_actual)

            # Majority vote only makes sense for hard-label metrics. Under AUC a
            # 0/1 vote throws away the ranking the metric is built on, so skip it.
            if task_type == 'classification' and not metric_needs_proba:
                fs = _score_rule(_members_s2, use_hard=True, agg_fn=_agg_majority)
                if any(np.isfinite(s) for s in fs):
                    _emit_ensemble(
                        results, cost_rows, honest_ens_candidates,
                        'optimal', 'ensemble_s2', set_name, 'majority_vote', fs,
                        _members_s2, member_cost, rep_dims, metric_name,
                        n_folds_actual)

        # -------- ORACLE: best (model x rep) picked ON the held-out fold ---
        # Deliberately optimistic. NOT a fair arm - an upper bound, so that the
        # gap between it and the honest arms is visible rather than implied.
        oracle_folds = []
        for f in range(n_folds_actual):
            best = np.nan
            for m in all_models:
                for rep in general_reps:
                    if rep not in member_folds[m]:
                        continue
                    s = member_folds[m][rep][f]
                    if not np.isfinite(s):
                        continue
                    if not np.isfinite(best) or \
                       (lower_is_better and s < best) or \
                       (not lower_is_better and s > best):
                        best = s
            oracle_folds.append(float(best))
        if any(np.isfinite(s) for s in oracle_folds):
            results.append({
                'model': 'optimal', 'method': 'oracle_single', 'augmented': False,
                'rep_set': 'all', 'aggregator': 'none',
                'score_mean': float(np.nanmean(oracle_folds)),
                'score_std': float(np.nanstd(oracle_folds)),
                'fold_scores': oracle_folds,
                'n_models_trained': 1,
                'is_oracle': True,
                'total_dim': 0, 'total_fit_time_s': 0.0,
                'total_predict_time_s': 0.0,
            })
            print(f"    oracle_single (UPPER BOUND, peeks at the held-out fold): "
                  f"{metric_name}={np.nanmean(oracle_folds):.4f}")

    # =====================================================================
    # HONEST RE-SCORE + HEAD-TO-HEAD
    # =====================================================================
    honest_hybrid = honest_rescore(honest_candidates, honest_baselines,
                                   lower_is_better)
    honest_ens = honest_rescore(honest_ens_candidates, honest_baselines,
                                lower_is_better)

    def _delta(better, worse):
        """Direction-corrected: positive means `better` really is better."""
        if not (np.isfinite(better) and np.isfinite(worse)):
            return float('nan')
        return (worse - better) if lower_is_better else (better - worse)

    head_to_head = {}
    for m in sorted(set(honest_hybrid) | set(honest_ens)):
        h = honest_hybrid.get(m, {}).get('hybrid_honest', float('nan'))
        e = honest_ens.get(m, {}).get('hybrid_honest', float('nan'))
        b = honest_hybrid.get(m, {}).get('best_single_honest', float('nan'))
        head_to_head[m] = {
            'hybrid_honest': h,
            'ensemble_honest': e,
            'best_single_honest': b,
            'ensemble_minus_hybrid': _delta(e, h),
            'hybrid_minus_single': _delta(h, b),
            'ensemble_minus_single': _delta(e, b),
            'ensemble_rule_per_fold':
                honest_ens.get(m, {}).get('hybrid_winners_per_fold', []),
            'hybrid_recipe_per_fold':
                honest_hybrid.get(m, {}).get('hybrid_winners_per_fold', []),
            'single_fold': honest_hybrid.get(m, {}).get('single_fold',
                           honest_ens.get(m, {}).get('single_fold', False)),
        }

    print("\n" + "=" * 78)
    print("  HONEST HEAD-TO-HEAD (leave-one-fold-out; same tax on every arm)")
    print("=" * 78)
    print(f"    {'Model':<10} {'Hybrid':>9} {'Ensemble':>9} {'Single':>9} "
          f"{'Ens-Hyb':>9} {'Hyb-Sgl':>9}")
    print("    " + "-" * 62)
    for m, hh in head_to_head.items():
        print(f"    {m:<10} {hh['hybrid_honest']:>9.4f} "
              f"{hh['ensemble_honest']:>9.4f} {hh['best_single_honest']:>9.4f} "
              f"{hh['ensemble_minus_hybrid']:>+9.4f} "
              f"{hh['hybrid_minus_single']:>+9.4f}")
    print("\n    Ens-Hyb > 0 => the ensemble is more accurate. The cost table")
    print("    below says what that accuracy costs, forever, per inference.")

    # =====================================================================
    # COST + BREAK-EVEN
    # =====================================================================
    print("\n" + "=" * 78)
    print("  STANDING COST PER ARM (after any one-off search)")
    print("=" * 78)
    print(f"    {'Model':<9} {'Arm':<26} {'Score':>8} {'Feat':>5} {'FeatDim':>8} "
          f"{'Mdls':>5} {'InDim':>7} {'MB':>8} {'Pred(s)':>9}")
    print("    " + "-" * 90)
    for mdl, arm, c, score in cost_rows:
        sc = score if np.isfinite(score) else float('nan')
        print(f"    {mdl:<9} {arm:<26} {sc:>8.4f} {c['n_featurisers']:>5} "
              f"{c['featuriser_dim']:>8} {c['n_models_deployed']:>5} "
              f"{c['model_input_dim']:>7} {c['model_bytes'] / 1e6:>8.2f} "
              f"{c['total_predict_time_s']:>9.4f}")

    # Break-even: the hybrid's one-off search vs the ensemble's per-inference tax.
    break_even = {}
    for model_name in models:
        hybs = [r for r in results if r['model'] == model_name
                and r['method'] == 'hybrid' and 'cost' in r]
        enss = [r for r in results if r['method'] in ('ensemble_s1', 'ensemble_s2')
                and r.get('rep_set') == 'all' and 'cost' in r
                and r['model'] == model_name]
        if not hybs or not enss:
            continue
        _best = min if lower_is_better else max
        h = _best(hybs, key=lambda x: x['score_mean'])
        e = _best(enss, key=lambda x: x['score_mean'])

        dt = e['cost']['total_predict_time_s'] - h['cost']['total_predict_time_s']
        n_fit_saved = e['cost']['n_models_deployed'] - h['cost']['n_models_deployed']
        break_even[model_name] = {
            'search_cost_model_fits': int(search_fits),
            'ensemble_models_deployed': e['cost']['n_models_deployed'],
            'hybrid_models_deployed': h['cost']['n_models_deployed'],
            'ensemble_featurisers': e['cost']['n_featurisers'],
            'hybrid_featurisers': h['cost']['n_featurisers'],
            'featuriser_reduction': (
                e['cost']['n_featurisers'] / h['cost']['n_featurisers']
                if h['cost']['n_featurisers'] else float('nan')),
            'storage_reduction': (
                e['cost']['storage_bytes_per_sample'] /
                h['cost']['storage_bytes_per_sample']
                if h['cost']['storage_bytes_per_sample'] else float('nan')),
            'predict_time_saved_per_fold_s': float(dt),
            'model_fits_saved_per_retrain': int(n_fit_saved),
            # How many retrains until the one-off search is repaid, counted in
            # model fits - hardware-independent, unlike wall-clock.
            'break_even_retrains': (
                float(search_fits) / n_fit_saved if n_fit_saved > 0
                else float('inf')),
            'accuracy_delta_ensemble_minus_hybrid': _delta(
                e['score_mean'], h['score_mean']),

            # THE load-bearing number. Xu et al. (Cost-Sensitive Tree of
            # Classifiers, ICML 2013): test-time cost is DOMINATED by feature
            # extraction, and a featuriser is paid for the first time any of its
            # columns is touched. So if the blueprint draws even one column from
            # every rep, the hybrid runs every encoder the ensemble runs and the
            # compute saving collapses to the downstream forward pass - a
            # rounding error next to a protein language model. Flag it loudly
            # rather than letting a favourable-looking break-even hide it.
            'hybrid_rep_sparsity': (
                h['cost']['n_featurisers'] / n_reps if n_reps else float('nan')),
            'featurisation_saving_exists': bool(
                h['cost']['n_featurisers'] < e['cost']['n_featurisers']),
        }

    if break_even:
        print("\n" + "=" * 78)
        print("  BREAK-EVEN: what the hybrid's one-off search buys")
        print("=" * 78)
        for m, be in break_even.items():
            print(f"    {m}:")
            print(f"      featurisers  {be['ensemble_featurisers']:>3} (ens) -> "
                  f"{be['hybrid_featurisers']:>3} (hyb)   "
                  f"{be['featuriser_reduction']:.1f}x fewer")
            print(f"      models       {be['ensemble_models_deployed']:>3} (ens) -> "
                  f"{be['hybrid_models_deployed']:>3} (hyb)")
            print(f"      storage/sample                     "
                  f"{be['storage_reduction']:.1f}x smaller")
            # Spell the direction out. A bare signed delta next to a
            # lower-is-better metric is the exact sign trap that produced the
            # bogus "hybrid hurts" reading once already.
            d = be['accuracy_delta_ensemble_minus_hybrid']
            verdict = ('ensemble is better' if d > 0 else
                       'HYBRID is better' if d < 0 else 'tied')
            print(f"      accuracy: ensemble - hybrid        "
                  f"{d:+.4f} {metric_name}  ({verdict})")
            print(f"      search repaid after                "
                  f"{be['break_even_retrains']:.1f} retrains")
            if not be['featurisation_saving_exists']:
                print("      *** WARNING: the blueprint touches EVERY rep the "
                      "ensemble needs.")
                print("      *** Featurisation cost is IDENTICAL; the only "
                      "saving is the downstream")
                print("      *** forward pass. Do NOT claim a compute win for "
                      "this dataset.")

        print("\n    Deployment regime matters and the two answers differ:")
        print("      online  - featurisers run per query  -> the win is compute,")
        print("                and it is real only where rep_sparsity < 1.")
        print("      batch   - reps precomputed and cached -> featurisation is")
        print("                paid once per sample, so the honest axis is")
        print("                STORAGE (bytes/sample), not FLOPs.")
        print("    This pipeline precomputes reps, i.e. the BATCH regime.")

    top3 = extract_top3_base_reps(last_search) if last_search is not None else []
    return {
        'results': results,
        'n_reps': n_reps,
        'total_dim': total_dim,
        'rep_names': rep_names,
        'general_reps': general_reps,
        'rep_dims': rep_dims,
        'metric_name': metric_name,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # Existing contract - hybrid vs best single rep. UNCHANGED.
        'honest_rescore': {
            'lower_is_better': bool(lower_is_better),
            'candidates_by_model': honest_candidates,
            'baselines_by_model': honest_baselines,
        },
        # Same shape, ensemble rules as the candidates. NOTE: the key
        # 'hybrid_honest' returned by honest_rescore() means "best ENSEMBLE
        # rule, honestly selected" when fed this block.
        'honest_rescore_ensemble': {
            'lower_is_better': bool(lower_is_better),
            'candidates_by_model': honest_ens_candidates,
            'baselines_by_model': honest_baselines,
        },
        'head_to_head': head_to_head,
        'break_even': break_even,
        'search_cost_model_fits': int(search_fits),
    }


def _emit_ensemble(results, cost_rows, honest_ens_candidates,
                   model_key, method, set_name, agg_name, fold_scores,
                   members_fn, member_cost, rep_dims, metric_name,
                   n_folds_actual):
    """Record one ensemble rule: score, honest candidate, and standing cost."""
    valid = [s for s in fold_scores if np.isfinite(s)]
    if not valid:
        return
    score = float(np.mean(valid))

    # Cost is what this rule deploys. Membership can vary by fold, so take the
    # union across folds - you must ship every featuriser the rule can ask for.
    union = set()
    for f in range(n_folds_actual):
        m, _ = members_fn(f)
        union |= set(m)
    if not union:
        return

    reps_needed = {rep for _, rep in union}
    fit_s = pred_s = 0.0
    mbytes = 0
    for (mdl, rep) in union:
        c = member_cost.get(mdl, {}).get(rep)
        if not c:
            continue
        fit_s += c[0]
        pred_s += c[1]
        mbytes += c[2]
    feat_dim = sum(rep_dims.get(r, 0) for r in reps_needed)

    name = f'{method}|{set_name}|{agg_name}'
    honest_ens_candidates.setdefault(model_key, []).append(
        {'name': name, 'fold_scores': [float(s) for s in fold_scores]})

    row = {
        'model': model_key,
        'method': method,
        'rep_set': set_name,
        'aggregator': agg_name,
        'augmented': False,
        'score_mean': score,
        'score_std': float(np.std(valid)),
        'fold_scores': [float(s) for s in fold_scores],
        'n_models_trained': len(union),
        'total_dim': feat_dim,
        'members': sorted(f'{m}:{r}' for m, r in union),
        'total_fit_time_s': fit_s,
        'total_predict_time_s': pred_s,
        # An ensemble must run EVERY member's featuriser and EVERY member's
        # model, at every inference, forever.
        'cost': _cost_row(len(reps_needed), feat_dim, len(union), feat_dim,
                          mbytes, fit_s, pred_s, feat_dim),
    }
    results.append(row)
    cost_rows.append((model_key, f'{method[-2:]}|{set_name}|{agg_name}',
                      row['cost'], score))
    print(f"    {method}/{set_name:<4}/{agg_name:<14} "
          f"{metric_name}={score:>8.4f}  members={len(union):>2} "
          f"featurisers={len(reps_needed):>2}")


if __name__ == '__main__':
    parser = create_experiment_parser('C11: Ensemble vs Hybrid',
                                      has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c11_ensemble_vs_hybrid',
                            args.dataset)
