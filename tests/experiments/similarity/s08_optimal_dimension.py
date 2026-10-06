#!/usr/bin/env python3
"""S08: Optimal Dimension for Similarity vs Prediction.

Paper: Similarity
Hypothesis: Optimal dimensionality may differ for similarity vs prediction.
Dependencies: None (standalone).

The question is whether the number of features at which distances in a hybrid
best follow activity is the same number at which a model best predicts it. The
budget sweep IS the question, so every budget of the dataset's decided grid is
swept, on both sides.

DECIDED 2026-10-06 (Adelaide delegated): budgets unchanged. Prediction sweeps
the decided general budgets, distance the decided distance budgets, so the two
optima can only coincide at a budget in both grids.

PER-FOLD REWRITE (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT.md Part 2, chat 4).
Every hybrid is chosen per outer fold from searches on that fold's TRAINING
rows (similarity_helpers.fold_similarity_selection and, per budget,
_shared.fold_standard_selection(budget=b)), rebuilt on the training rows with
the fold search's own representation scores (build_fold_entry), and scored on
the held-out rows. Nothing is built or scored on every row any more.

What one outer fold measures:
- Prediction side, at every decided GENERAL budget: each of the dataset's two
  decided models (decided_models(ds, 'general')) on its own top 2 hybrids at
  that budget from the per-fold general search, scored on the held-out rows
  in the dataset's own metric (get_domain_metric, direction included).
- Distance side, at every decided DISTANCE budget: knn's top hybrid at that
  budget from the per-fold distance search (tier general_distance). Until this
  date the distance side swept the old DISTANCE_BUDGETS list and read
  recipes whose budgets came from the general lineage; it now reads
  decided_search(ds, 'general_distance')['budgets'].
- Similarity family, at every budget the per-fold similarity-objective search
  ran (that search is a general-tier search, so its budgets are the general
  budgets): the top hybrid at that budget over every model it scored.
- On every hybrid built above and on knn's 2 best single representations of
  the fold (fold_similarity_selection's knn_base; Part 1 step 5): the
  neighbour numbers. Per distance in knn_axes(ds, n)['metrics']: the held-out
  Spearman between pair distance and activity difference (unweighted, and
  importance-weighted where the space carries importance), the Spearman
  between the space's held-out distances and the full concatenation's, and
  the violation rate at each neighbour count, and the Levina-Bickel intrinsic
  dimension of the training rows (10 nearest neighbours, self excluded by
  index; mle_id_by_distance). Per (distance, neighbour count,
  counting scheme): kNN prediction of each held-out molecule from its nearest
  training molecules (knn_sweep's settings), in the dataset's metric.
- Each decided model on its own 2 best single representations of the fold.

Every scaler is kirby.scaling.FeatureScaler fitted on the training rows
(similarity_helpers.scale_fold): 0/1 features untouched, continuous
standardised, held-out clipped to the training range. kNN distances, neighbour
counts and counting schemes are the decided ones (knn_axes); no sklearn
default Euclidean anywhere in this file, and a molecule's own scaffold group is
never excluded from its neighbours.

The importance-weighted distance arms keep the weighting code they had
(flatten_fold_weights on the hybrid, lightgbm_gain on a single
representation, the power compute_distance_matrix applies by default); how a
similarity experiment weights its own distance is still open (handoff 2.10).
The importance for a single representation is now computed on the fold's
training rows; it was computed on every row.

Removed with this rewrite, because each was a whole-dataset search or a
statistic on every row (R2): get_rule_hybrids, get_best_search_for_downstream,
get_fitted_hybrid, extract_top3_base_reps, the 'general1'/'general2'/
'distance1' slots of the old owner's set (replaced by the per-fold 2 + 2 and
the per-fold distance and similarity hybrids), the full-data sac_rho,
violation_rate, dist_corr_with_full and mle_id per budget, the top-2 rows read
off a stored search, and DISTANCE_MODELS (rf, xgboost, svm, dnn, lightgbm,
knn fitted on every dataset; now the two decided models, and knn only through
the distance search). fold_reselection is kept: it re-picks a budget per
held-out fold from the other folds' held-out numbers of hybrids built inside
each fold, which the handoff allows.

Kept keys that s09 reads: results['optimal']['best_rf_budget'] /
['best_knn_budget'] and results['sac_reselected'] /
results['sac_reselected_unweighted'] (the block for model 'sac'). Their
meaning under the per-fold rewrite is in the comments where they are written.

Older history of this file (whole-dataset pool reads, 2026-06 to 2026-10-02)
is in git; none of those code paths survive.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments.similarity.similarity_helpers import _CODE_FAULTS
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    flatten_fold_weights,
    get_domain_metric, get_model_factories, fold_reselection,
    fold_standard_selection, fold_searches, top_hybrids_across_models,
)
from experiments.decided_grid import decided_search, decided_models, is_decided
from experiments.similarity.similarity_helpers import (
    sim_outer_folds, fold_spread_note, knn_axes, scale_fold, knn_lists,
    knn_from_lists, score_prediction, fold_similarity_selection,
    build_fold_entry, entry_record, sac_pair_sample, SIM_FOLD_SEED,
)
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.utils.noise_mitigation_methods import compute_distance_matrix
# Scores for the metric (2026-10-05), as s10 does. On a binary dataset scored
# by auc this returns the predicted probability of class 1, which is how the
# searches score auc (find_best_hybrid.py); every other metric gets
# model.predict.
from kirby.find_best_hybrid import _predict_for_metric
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)


# How many distinct budgets one side must have a usable score at before
# 'optima_differ' is called trustworthy. The optimum is an argmax over the
# scored budgets; with 1 there is no choice and with 2 it is one comparison,
# so 3 is the smallest count at which the argmax picks among more than two.
# 3 is the number recorded for rf on 2026-10-05 (results audit, S08e). It now
# applies to each decided model on the prediction side and to the distance
# side, since the distance side's decided grid holds 3 budgets on most
# datasets (100, 250, 500 features); the earlier 4-distinct-widths rule would
# suppress the comparison on every one of them.
MIN_SCORED_WIDTHS = 3

# Pairs of training molecules whose activity difference sets the violation
# threshold (the median). The same count violation_rate used.
VIOLATION_PAIRS = 50000

# The importance method of the weighted single-representation distance arm, as
# before this rewrite (it now runs on the fold's training rows).
SINGLE_REP_IMPORTANCE = 'lightgbm_gain'

# Version of what one fold unit holds. Bumped whenever the unit's meaning
# changes, so a partial file written under the old meaning is not read back.
UNIT_VERSION = 'per_fold_selection_mle_zero_distance_excluded_2026_10_06'


# ─────────────────────────────────────────────────────────────────────────────
# Measurements on one space (pure functions)
# ─────────────────────────────────────────────────────────────────────────────

def _finite(v):
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def _pair_delta(y, pi, pj, is_clf):
    if is_clf:
        return (y[pi] != y[pj]).astype(float)
    return np.abs(y[pi].astype(float) - y[pj].astype(float))


def _sac_on_pairs(D, y, pi, pj, is_clf):
    """Spearman between the pair distance and the pair activity difference,
    over the sampled held-out pairs (similarity_activity_correlation on fixed
    pairs, so every space in a fold is scored on the same pairs)."""
    if len(pi) < 3:
        return float('nan')
    rho, _ = spearmanr(D[pi, pj], _pair_delta(y, pi, pj, is_clf))
    return float(rho)


def _violation(idx, y_tr, y_va, k, median_delta, is_clf):
    """Share of each held-out molecule's k nearest training molecules whose
    activity differs from it by more than the training-pair median
    (classification: whose class differs). Every neighbour counts once."""
    nb = np.asarray(y_tr)[np.asarray(idx)[:, :k]]
    yq = np.asarray(y_va)[:, None]
    if is_clf:
        bad = nb != yq
    else:
        bad = np.abs(nb.astype(float) - yq.astype(float)) > median_delta
    return float(bad.mean()) if bad.size else float('nan')


# Neighbours per molecule in the intrinsic-dimension estimate (Levina-Bickel),
# the k the earlier estimate_intrinsic_dim_mle call used.
MLE_K = 10


def _mle(A, metrics):
    """Levina-Bickel intrinsic dimension of the training rows A, per distance.

    The formula is estimate_intrinsic_dim_mle's (_shared.py): for each
    molecule, (k - 1) / sum_j log(d_k / d_j) over its k nearest other training
    molecules, averaged over molecules with d_k > 0 and a positive sum. It is
    computed here, not by that function, because that function reads the
    process-wide KIRBY_DISTANCE_METRIC and drops the first neighbour column to
    remove self (dirty-run bug 7). Here the neighbours come from knn_lists
    under each decided distance in knn_axes(ds, n)['metrics'], with self
    excluded by index.

    DECIDED 2026-10-06 (Adelaide delegated): computed here with the decided
    distances and self excluded by index, not by estimate_intrinsic_dim_mle.
    A molecule with another training molecule at distance 0 (identical
    features) has log(d_k / 0) = infinity, so its estimate is 0 (under cosine,
    a distance near 1e-16 gives the same near-0 estimate). The earlier
    code averaged those zeros in, which pulls the mean down by the share of
    such molecules. They are now left out and counted.

    Returns {distance: {'estimate': float or None, 'n_molecules_used': int,
    'n_zero_distance_neighbour': int}} or {distance: None} when it could not
    be computed.
    """
    out = {}
    for metric in metrics:
        if len(A) <= MLE_K + 1:
            out[metric] = None
            continue
        try:
            _, dist = knn_lists(A, metric=metric, k_max=MLE_K)
            k = dist.shape[1]
            d = np.asarray(dist, dtype=np.float64)
            # Nearest-first, so d[:, 0] is the closest other molecule. Cosine
            # gives identical rows a distance near 1e-16 rather than 0, so a
            # nearest distance under 1e-9 of the molecule's k-th counts as 0.
            zero = ~(d[:, 0] > 1e-9 * d[:, -1]) | ~(d[:, -1] > 0)
            ok = ~zero & np.all(np.isfinite(d), axis=1)
            s = np.log(d[ok, -1:] / d[ok, :-1]).sum(1)
            s = s[s > 0]
            out[metric] = {
                'estimate': float(np.mean((k - 1) / s)) if s.size else None,
                'n_molecules_used': int(s.size),
                'n_zero_distance_neighbour': int(zero.sum())}
        except Exception:
            out[metric] = None
    return out


def _neighbour_block(A, B, y_tr, y_va, *, axes, pairs, D_full, median_delta,
                     is_clf, metric_name, metric_fn, weights=None):
    """Every neighbour number for one space on one fold.

    A, B: the space's training and held-out rows, already scaled by
    scale_fold. weights: importance per feature for the weighted distance arm,
    or None for a space with no weighted arm.

    Returns {'by_distance': {distance: {...}}, 'knn_rows': [...]} where every
    kNN number carries its distance, neighbour count and counting scheme.
    """
    pi, pj = pairs
    out = {'by_distance': {}, 'knn_rows': []}
    w = None
    if weights is not None:
        w = np.maximum(np.asarray(weights, dtype=np.float64), 0)
        if w.shape[0] != B.shape[1] or not np.any(w > 0):
            out['weighted_unavailable'] = (
                f'{w.shape[0]} importance values for {B.shape[1]} features'
                if w.shape[0] != B.shape[1] else 'every importance value is 0')
            w = None
    for metric in axes['metrics']:
        row = {}
        try:
            D = compute_distance_matrix(B, metric=metric, standardize=False)
            row['sac_unweighted'] = _sac_on_pairs(D, y_va, pi, pj, is_clf)
            Df = D_full.get(metric)
            row['dist_corr_with_full'] = (
                float(spearmanr(D[pi, pj], Df[pi, pj])[0])
                if Df is not None and len(pi) >= 3 else float('nan'))
            if w is not None:
                Dw = compute_distance_matrix(B, metric=metric,
                                             feature_weights=w,
                                             standardize=False)
                row['sac_weighted'] = _sac_on_pairs(Dw, y_va, pi, pj, is_clf)
                row['dist_corr_with_full_weighted'] = (
                    float(spearmanr(Dw[pi, pj], Df[pi, pj])[0])
                    if Df is not None and len(pi) >= 3 else float('nan'))
        except Exception as e:
            row['distance_error'] = f'{type(e).__name__}: {e}'
        try:
            idx, dist = knn_lists(A, B, metric=metric, k_max=axes['k_max'])
            row['violation_rate'] = {}
            for k in axes['ks']:
                if k > idx.shape[1]:
                    row['violation_rate'][str(k)] = None
                    out['knn_rows'].extend(
                        {'distance': metric, 'k': int(k), 'scheme': s,
                         'score': None,
                         'note': f'only {idx.shape[1]} training rows'}
                        for s in axes['schemes'])
                    continue
                row['violation_rate'][str(k)] = _violation(
                    idx, y_tr, y_va, k, median_delta, is_clf)
                for scheme in axes['schemes']:
                    pred, proba, classes = knn_from_lists(
                        y_tr, idx, dist, k=k, scheme=scheme, is_clf=is_clf)
                    out['knn_rows'].append({
                        'distance': metric, 'k': int(k), 'scheme': scheme,
                        'score': score_prediction(y_va, pred, proba, classes,
                                                  metric_name, metric_fn)})
        except Exception as e:
            row['knn_error'] = f'{type(e).__name__}: {e}'
        out['by_distance'][metric] = row
    return out


def _fit_score(factory, A, B, y_tr, y_va, metric_fn):
    m = factory()
    m.fit(A, y_tr)
    return float(metric_fn(y_va, _predict_for_metric(m, B, metric_fn)))


# ─────────────────────────────────────────────────────────────────────────────
# One outer fold
# ─────────────────────────────────────────────────────────────────────────────

def _run_fold(data, dataset_name, *, fold_i, n_folds, train_idx, val_idx,
              models, factories, knn_factory, gen_budgets, dist_budgets, axes,
              metric_name, metric_fn, lower_is_better, is_clf, n_sample,
              full_reps):
    reps = data['reps']
    labels = np.asarray(data['labels'])
    tr, va = np.asarray(train_idx), np.asarray(val_idx)
    y_tr, y_va = labels[tr], labels[va]
    task_type = data.get('task_type', 'auto')
    t0 = time.time()

    rec = {'fold': int(fold_i), 'n_outer_folds': int(n_folds),
           'n_train': int(len(tr)), 'n_held_out': int(len(va)),
           'selection_scope': 'fold_train_rows',
           'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'unavailable': {}}

    # ── Selection on this fold's training rows. One call; a search that
    # failed is recorded here once and not asked again at any budget (the
    # dirty run retried the failing similarity search at every budget). ──
    try:
        sel = fold_similarity_selection(
            data, dataset_name, train_idx=tr, fold_i=fold_i,
            n_outer_folds=n_folds, neighbour_spaces=True)
    except _CODE_FAULTS:
        raise  # a fault in the code, not a failed search
    except Exception as e:
        rec['error'] = (f'fold_similarity_selection raised '
                        f'{type(e).__name__}: {e}')
        print(f"    fold {fold_i}: {rec['error']}")
        return rec
    rec['unavailable'].update(sel.get('unavailable') or {})
    rec['selection'] = {
        'models': {m: (None if s is None else {
            'base': list(s.get('base') or []),
            'base_train_rows_score': [
                e.get('score') for e in
                ((s.get('base_detail') or {}).get('selected') or [])],
            'hybrids': [entry_record(h) for h in s.get('hybrids') or []],
            'hybrids_skipped_identical_features': len(
                s.get('hybrids_skipped_identical_features') or [])})
            for m, s in (sel.get('models') or {}).items()},
        'knn_base': list(sel.get('knn_base') or []),
        'knn_base_train_rows_score': [
            e.get('score') for e in
            ((sel.get('knn_base_detail') or {}).get('selected') or [])],
        'distance_hybrid': entry_record(sel.get('distance_hybrid')),
        'similarity_hybrid': entry_record(sel.get('similarity_hybrid')),
    }

    # ── What every space in this fold is measured against ──
    pairs = sac_pair_sample(len(va), n_sample)
    rec['n_held_out_pairs'] = int(len(pairs[0]))
    tri, trj = sac_pair_sample(len(tr), VIOLATION_PAIRS)
    median_delta = (float(np.median(np.abs(y_tr[tri].astype(float)
                                           - y_tr[trj].astype(float))))
                    if not is_clf and len(tri) else None)
    rec['violation_threshold'] = median_delta
    # The full concatenation of the dataset's general representations, scaled
    # per representation on the training rows (FeatureScaler is per feature,
    # so this equals scaling the concatenation), held-out rows only.
    D_full = {}
    try:
        B_full = np.hstack([scale_fold(np.asarray(reps[r])[tr],
                                       np.asarray(reps[r])[va])[1]
                            for r in full_reps])
        for metric in axes['metrics']:
            D_full[metric] = compute_distance_matrix(B_full, metric=metric,
                                                     standardize=False)
        rec['full_concat_width'] = int(B_full.shape[1])
        rec['full_concat_reps'] = list(full_reps)
        del B_full
    except Exception as e:
        rec['unavailable']['full_concat'] = f'{type(e).__name__}: {e}'
    nb_kw = dict(axes=axes, pairs=pairs, D_full=D_full,
                 median_delta=median_delta, is_clf=is_clf,
                 metric_name=metric_name, metric_fn=metric_fn)

    def _hybrid_space(entry, *, model_factory=None, budget):
        """Build one selected hybrid on the training rows; neighbour numbers
        on it. Returns (cell, A, B)."""
        cell = {'hybrid': entry_record(entry), 'budget_asked': int(budget)}
        Xtr, Xva, fi = build_fold_entry(
            data, entry, train_idx=tr, val_idx=va,
            model_factory=model_factory, metric_fn=metric_fn,
            lower_is_better=lower_is_better)
        A, B = scale_fold(Xtr, Xva)
        cell['width_built'] = int(A.shape[1])
        cell['mle_id_by_distance'] = _mle(A, axes['metrics'])
        w = flatten_fold_weights(fi) if fi else None
        cell['neighbour'] = _neighbour_block(A, B, y_tr, y_va, weights=w,
                                             **nb_kw)
        return cell, A, B

    # ── Prediction side: each decided model's top 2 hybrids at each general
    # budget, from the same per-fold general search (memoised). ──
    rec['prediction'] = {m: {} for m in models}
    live_models = [m for m in models
                   if (sel.get('models') or {}).get(m) is not None]
    for m in models:
        if m not in live_models:
            rec['prediction'][m] = {
                'unavailable': rec['unavailable'].get(
                    f'general/{m}', 'the per-fold general search failed')}
    common = dict(train_idx=tr, fold_i=fold_i, n_outer_folds=n_folds,
                  dataset_name=dataset_name,
                  augmentations=data.get('augmentations'),
                  task_type=task_type, seed=SIM_FOLD_SEED)
    for b in gen_budgets if live_models else []:
        try:
            sel_b = fold_standard_selection(
                reps, labels, data.get('groups'), models=live_models,
                budget=b, **common)
        except Exception as e:
            for m in live_models:
                rec['prediction'][m][str(b)] = {
                    'unavailable': f'{type(e).__name__}: {e}'}
            continue
        for m in live_models:
            s = sel_b.get(m)
            hyb = (s or {}).get('hybrids') or []
            cells = []
            for rank, entry in enumerate(hyb, start=1):
                try:
                    cell, A, B = _hybrid_space(entry, model_factory=factories[m],
                                               budget=b)
                    cell['rank'] = rank
                    cell['held_out_score'] = _fit_score(
                        factories[m], A, B, y_tr, y_va, metric_fn)
                except Exception as e:
                    cell = {'rank': rank, 'hybrid': entry_record(entry),
                            'budget_asked': int(b),
                            'failed': f'{type(e).__name__}: {e}'}
                cells.append(cell)
            rec['prediction'][m][str(b)] = {
                'hybrids': cells,
                'hybrids_skipped_identical_features': len(
                    (s or {}).get('hybrids_skipped_identical_features') or []),
                'shortfall': (None if len(hyb) >= 2 else
                              f'the per-fold search holds {len(hyb)} hybrid(s) '
                              f'for {m} at {b} features'),
            }

    # Each decided model on its own 2 best single representations.
    rec['prediction_base'] = {}
    for m in live_models:
        rows = []
        for rank, rep in enumerate(sel['models'][m]['base'], start=1):
            try:
                X = np.asarray(reps[rep])
                A, B = scale_fold(X[tr], X[va])
                rows.append({'rank': rank, 'representation': rep,
                             'width_built': int(A.shape[1]),
                             'held_out_score': _fit_score(
                                 factories[m], A, B, y_tr, y_va, metric_fn)})
            except Exception as e:
                rows.append({'rank': rank, 'representation': rep,
                             'failed': f'{type(e).__name__}: {e}'})
        rec['prediction_base'][m] = rows

    # ── Distance side: knn's top hybrid at each decided distance budget, from
    # the per-fold distance search. Skipped once, with the reason, when that
    # search failed or is undecided. ──
    rec['distance_family'] = {}
    if 'distance_search' in rec['unavailable'] or not dist_budgets:
        rec['distance_family']['unavailable'] = rec['unavailable'].get(
            'distance_search', 'no decided distance budgets')
    else:
        for b in dist_budgets:
            try:
                # n_hybrids=1: the top hybrid only, so there is no second one
                # to compare features with.
                sd = fold_standard_selection(
                    reps, labels, data.get('groups'), distance=True,
                    models=['knn'], n_hybrids=1, budget=b,
                    distinct_features=False, **common)
                hyb = ((sd.get('knn') or {}).get('hybrids') or [])
                if not hyb:
                    rec['distance_family'][str(b)] = {
                        'unavailable': f'the per-fold distance search holds no '
                                       f'knn hybrid at {b} features'}
                    continue
                cell, _, _ = _hybrid_space(hyb[0], model_factory=knn_factory,
                                           budget=b)
                rec['distance_family'][str(b)] = cell
            except Exception as e:
                rec['distance_family'][str(b)] = {
                    'failed': f'{type(e).__name__}: {e}'}

    # ── Similarity family: the top hybrid at each budget the per-fold
    # similarity-objective search ran. Asked once; when the selection above
    # found it undefined or failed, every budget is skipped with that one
    # reason. ──
    rec['similarity_family'] = {}
    if sel.get('similarity_hybrid') is None:
        rec['similarity_family']['unavailable'] = rec['unavailable'].get(
            'similarity_hybrid', 'the per-fold similarity search scored no '
                                 'hybrid')
    else:
        srch = None
        try:
            srch = fold_searches(reps, labels, data.get('groups'),
                                 tier='general', objective='similarity',
                                 **common).get('none')
        except Exception as e:
            rec['similarity_family']['unavailable'] = (
                f'{type(e).__name__}: {e}')
        sim_budgets = sorted({int(r.get('budget')) for r in
                              (getattr(srch, 'all_results', None) or [])
                              if isinstance(r, dict)
                              and r.get('budget') is not None})
        rec['similarity_family_budgets'] = sim_budgets
        for b in sim_budgets if srch is not None else []:
            try:
                th = top_hybrids_across_models(
                    srch, n=1, budget=b, require_full=False,
                    context=f'{dataset_name} similarity fold {fold_i} b{b}')
                if not th['selected']:
                    rec['similarity_family'][str(b)] = {
                        'unavailable': f'no similarity hybrid at {b} features'}
                    continue
                entry = dict(th['selected'][0], aug_condition='none')
                cell, _, _ = _hybrid_space(entry, budget=b)
                cell['recipe_chosen_by'] = entry.get('recipe_chosen_by')
                rec['similarity_family'][str(b)] = cell
            except Exception as e:
                rec['similarity_family'][str(b)] = {
                    'failed': f'{type(e).__name__}: {e}'}

    # ── knn's 2 best single representations of this fold (Part 1 step 5):
    # the single-representation neighbour arms. ──
    rec['neighbour_base'] = []
    from kirby.hybrid import compute_feature_importance as _cfi
    for rank, rep in enumerate(rec['selection']['knn_base'], start=1):
        row = {'rank': rank, 'representation': rep}
        try:
            X = np.asarray(reps[rep])
            A, B = scale_fold(X[tr], X[va])
            row['width_built'] = int(A.shape[1])
            row['mle_id_by_distance'] = _mle(A, axes['metrics'])
            try:
                w = _cfi({rep: X[tr]}, y_tr, method=SINGLE_REP_IMPORTANCE,
                         task_type=task_type)[rep]
                row['importance_method'] = SINGLE_REP_IMPORTANCE
            except Exception as e:
                w = None
                row['weighted_unavailable'] = f'{type(e).__name__}: {e}'
            row['neighbour'] = _neighbour_block(A, B, y_tr, y_va, weights=w,
                                                **nb_kw)
        except Exception as e:
            row['failed'] = f'{type(e).__name__}: {e}'
        rec['neighbour_base'].append(row)

    rec['time_seconds'] = float(time.time() - t0)
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Summaries over folds
# ─────────────────────────────────────────────────────────────────────────────

def _spread(vals):
    """Per-fold values, their mean, min and max. No interval, no test."""
    ok = [float(v) for v in vals if _finite(v)]
    return {'fold_values': [float(v) if _finite(v) else None for v in vals],
            'mean': float(np.mean(ok)) if ok else float('nan'),
            'min': float(min(ok)) if ok else float('nan'),
            'max': float(max(ok)) if ok else float('nan'),
            'n_folds_scored': len(ok),
            'spread_note': fold_spread_note(ok)}


def _best_budget(by_budget, lower_is_better):
    """(budget, mean) with the best fold mean, NaN means dropped; None when no
    budget has a finite mean. Also the budgets dropped."""
    scored = [(int(b), d['mean']) for b, d in by_budget.items()
              if _finite(d.get('mean'))]
    dropped = sorted(int(b) for b, d in by_budget.items()
                     if not _finite(d.get('mean')))
    if not scored:
        return None, None, dropped
    pick = (min if lower_is_better else max)(scored, key=lambda t: (
        t[1], t[0]) if lower_is_better else (t[1], -t[0]))
    return pick[0], pick[1], dropped


def _family_cells(fold, family):
    block = fold.get(family) or {}
    return {b: c for b, c in block.items()
            if b != 'unavailable' and isinstance(c, dict)}


def _summarise(results, folds, *, models, axes, lower_is_better, gen_budgets,
               dist_budgets):
    n = len(folds)

    # Prediction: per model, per general budget, held-out score of the rank-1
    # and rank-2 hybrid, per fold.
    pred = {}
    for m in models:
        pred[m] = {}
        for b in gen_budgets:
            by_rank = {1: [None] * n, 2: [None] * n}
            widths = [None] * n
            for i, f in enumerate(folds):
                cells = ((f.get('prediction') or {}).get(m) or {}).get(str(b))
                for c in (cells or {}).get('hybrids') or []:
                    if c.get('rank') in by_rank:
                        by_rank[c['rank']][i] = c.get('held_out_score')
                    if c.get('rank') == 1:
                        widths[i] = c.get('width_built')
            pred[m][str(b)] = {'rank1': _spread(by_rank[1]),
                               'rank2': _spread(by_rank[2]),
                               'rank1_width_built_per_fold': widths}
    results['prediction_by_budget'] = pred
    results['prediction_base_by_rank'] = {
        m: {str(r): _spread([next((x.get('held_out_score') for x in
                                    (f.get('prediction_base') or {}).get(m, [])
                                    if x.get('rank') == r), None)
                              for f in folds])
            for r in (1, 2)} for m in models}

    # Neighbour numbers: per family, per budget, per distance; kNN per
    # (distance, neighbour count, counting scheme).
    def _nb_summary(get_block):
        out = {'by_distance': {}, 'knn': {}}
        for metric in axes['metrics']:
            d = {}
            for key in ('sac_unweighted', 'sac_weighted',
                        'dist_corr_with_full', 'dist_corr_with_full_weighted'):
                d[key] = _spread([((get_block(f) or {}).get('by_distance', {})
                                   .get(metric, {}).get(key)) for f in folds])
            d['violation_rate'] = {
                str(k): _spread([((get_block(f) or {}).get('by_distance', {})
                                  .get(metric, {}).get('violation_rate')
                                  or {}).get(str(k)) for f in folds])
                for k in axes['ks']}
            out['by_distance'][metric] = d
        for metric in axes['metrics']:
            for k in axes['ks']:
                for s in axes['schemes']:
                    vals = []
                    for f in folds:
                        rows = (get_block(f) or {}).get('knn_rows') or []
                        vals.append(next((r.get('score') for r in rows
                                          if r['distance'] == metric
                                          and r['k'] == k
                                          and r['scheme'] == s), None))
                    out['knn'][f'{metric}|k{k}|{s}'] = dict(
                        _spread(vals), distance=metric, k=int(k), scheme=s)
        return out

    nb = {'distance_family': {}, 'similarity_family': {},
          'general_hybrids': {}, 'single_representations': {}}
    for fam in ('distance_family', 'similarity_family'):
        budgets = sorted({int(b) for f in folds
                          for b in _family_cells(f, fam)})
        for b in budgets:
            nb[fam][str(b)] = _nb_summary(
                lambda f, fam=fam, b=b: (_family_cells(f, fam).get(str(b))
                                         or {}).get('neighbour'))
    for m in models:
        nb['general_hybrids'][m] = {}
        for b in gen_budgets:
            def _g(f, m=m, b=b):
                cells = (((f.get('prediction') or {}).get(m) or {})
                         .get(str(b)) or {}).get('hybrids') or []
                c = next((c for c in cells if c.get('rank') == 1), None)
                return (c or {}).get('neighbour')
            nb['general_hybrids'][m][str(b)] = _nb_summary(_g)
    for r in (1, 2):
        nb['single_representations'][f'knn_base_rank{r}'] = _nb_summary(
            lambda f, r=r: next((x.get('neighbour') for x in
                                 f.get('neighbour_base') or []
                                 if x.get('rank') == r), None))
    results['neighbour_by_budget'] = nb

    # ── Optimal budgets, NaN-scored budgets dropped, None when none scored ──
    opt = {'prediction': {}, 'distance_family_sac': {},
           'similarity_family_sac': {}, 'distance_family_knn': {}}
    for m in models:
        b, v, dropped = _best_budget(
            {bb: d['rank1'] for bb, d in pred[m].items()}, lower_is_better)
        opt['prediction'][m] = {'budget': b, 'mean': v,
                                'budgets_without_score': dropped,
                                'n_scored_budgets': len(gen_budgets) - len(dropped)}
    for fam in ('distance_family', 'similarity_family'):
        for metric in axes['metrics']:
            for key in ('sac_weighted', 'sac_unweighted'):
                b, v, dropped = _best_budget(
                    {bb: d['by_distance'][metric][key]
                     for bb, d in nb[fam].items()}, False)
                opt[f'{fam}_sac'].setdefault(metric, {})[key] = {
                    'budget': b, 'mean': v, 'budgets_without_score': dropped,
                    'n_scored_budgets': len(nb[fam]) - len(dropped)}
    for setting in (next(iter(nb['distance_family'].values()), {})
                    .get('knn') or {}):
        b, v, dropped = _best_budget(
            {bb: d['knn'][setting] for bb, d in
             nb['distance_family'].items()}, lower_is_better)
        opt['distance_family_knn'][setting] = {
            'budget': b, 'mean': v, 'budgets_without_score': dropped}
    # Which (budget, kNN setting) has the best fold mean over all of them.
    # Picked on the same held-out numbers it reports, so optimistic.
    _knn_all = [(int(bb), s, d['knn'][s]['mean'])
                for bb, d in nb['distance_family'].items()
                for s in d['knn'] if _finite(d['knn'][s]['mean'])]
    best_knn = ((min if lower_is_better else max)(_knn_all, key=lambda t: t[2])
                if _knn_all else None)
    opt['best_knn_budget'] = best_knn[0] if best_knn else None
    opt['best_knn_setting'] = best_knn[1] if best_knn else None
    opt['best_knn_note'] = (
        "the distance budget and kNN setting with the best fold mean over "
        "every (budget, distance, neighbour count, counting scheme); picked on "
        "the numbers it reports, so optimistic")
    opt['best_model_budgets'] = {m: opt['prediction'][m]['budget']
                                 for m in models}
    # s09 reads best_rf_budget. It is rf's prediction-optimal budget when rf
    # is one of the dataset's two decided models, None otherwise; the same
    # number for each decided model is in best_model_budgets.
    opt['best_rf_budget'] = opt['best_model_budgets'].get('rf')
    results['optimal'] = opt

    # ── Do the two optima differ? Per decided model, per distance: the
    # prediction-optimal general budget against the budget at which knn's
    # distance hybrid's held-out SAC is highest. ──
    cmp_ = {}
    for m in models:
        pm = opt['prediction'][m]
        for metric in axes['metrics']:
            for key in ('sac_weighted', 'sac_unweighted'):
                dm = opt['distance_family_sac'].get(metric, {}).get(key, {})
                trust = (pm['budget'] is not None
                         and dm.get('budget') is not None
                         and pm['n_scored_budgets'] >= MIN_SCORED_WIDTHS
                         and dm.get('n_scored_budgets', 0) >= MIN_SCORED_WIDTHS)
                cmp_.setdefault(m, {}).setdefault(metric, {})[key] = {
                    'prediction_budget': pm['budget'],
                    'distance_budget': dm.get('budget'),
                    'optima_differ': (None if not trust
                                      else pm['budget'] != dm['budget']),
                    'optima_differ_trustworthy': bool(trust),
                    'budget_gap': (abs(pm['budget'] - dm['budget'])
                                   if trust else None),
                    'min_scored_widths': MIN_SCORED_WIDTHS,
                }
    results['optima_compare'] = cmp_
    results['budget_grids'] = {
        'general': list(gen_budgets), 'distance': list(dist_budgets),
        'both': sorted(set(gen_budgets) & set(dist_budgets)),
        'note': ('the prediction side sweeps the decided general budgets and '
                 'the distance side the decided distance budgets; the two '
                 'optima can only coincide at a budget in both')}

    # ── Same features, two quantities: per decided model, at each general
    # budget, its rank-1 hybrid's held-out score against the held-out SAC of
    # the same features. Pareto front and the loss of each at the other's
    # optimum. ──
    pareto = {}
    for m in models:
        for metric in axes['metrics']:
            pts = []
            for b in gen_budgets:
                sc = pred[m][str(b)]['rank1']['mean']
                sac = (nb['general_hybrids'][m][str(b)]['by_distance'][metric]
                       ['sac_weighted']['mean'])
                if _finite(sc) and _finite(sac):
                    pts.append((int(b), float(sac), float(sc)))
            front, best_seen = [], float('-inf')
            # Highest SAC first; on equal SAC the better score first, so a
            # point dominated at the same SAC never enters the front.
            for b, sac, sc in sorted(pts, key=lambda t: (
                    -t[1], t[2] if lower_is_better else -t[2])):
                gain = -sc if lower_is_better else sc
                if gain > best_seen:
                    front.append({'budget': b, 'sac_weighted': sac,
                                  'held_out_score': sc})
                    best_seen = gain
            entry = {'points': [{'budget': b, 'sac_weighted': s,
                                 'held_out_score': c} for b, s, c in pts],
                     'frontier': front}
            if len(pts) >= 1:
                b_sc = (min if lower_is_better else max)(
                    pts, key=lambda t: t[2])
                b_sac = max(pts, key=lambda t: t[1])
                entry['score_optimum_budget'] = b_sc[0]
                entry['sac_optimum_budget'] = b_sac[0]
                entry['sac_loss_at_score_optimum'] = b_sac[1] - b_sc[1]
                entry['score_loss_at_sac_optimum'] = (
                    b_sac[2] - b_sc[2] if lower_is_better
                    else b_sc[2] - b_sac[2])
                entry['trustworthy'] = len(pts) >= MIN_SCORED_WIDTHS
            pareto.setdefault(m, {})[metric] = entry
    results['pareto_same_features'] = pareto

    # ── Held-out SAC with the budget re-picked per held-out fold from the
    # other folds (fold_reselection). Candidates: knn's distance hybrid and
    # the similarity hybrid at each budget; baselines: knn's rank-1 and rank-2
    # single representations of each fold. Every hybrid was chosen and built
    # on that fold's training rows. SAC is higher-is-better whatever the
    # dataset's metric. ──
    resel = {}
    for metric in axes['metrics']:
        for key, base_key in (('sac_weighted', 'sac_unweighted'),
                              ('sac_unweighted', 'sac_unweighted')):
            cands = []
            for fam, tag in (('distance_family', 'distance'),
                             ('similarity_family', 'similarity')):
                for b, d in nb[fam].items():
                    fv = d['by_distance'][metric][key]['fold_values']
                    if any(v is not None for v in fv):
                        cands.append({'name': f'{tag}_b{b}',
                                      'fold_scores': [float('nan') if v is None
                                                      else v for v in fv]})
            bases = {}
            for r in (1, 2):
                fv = (nb['single_representations'][f'knn_base_rank{r}']
                      ['by_distance'][metric][base_key]['fold_values'])
                if any(v is not None for v in fv):
                    bases[f'knn_base_rank{r}'] = [float('nan') if v is None
                                                  else v for v in fv]
            try:
                blk = fold_reselection({'sac': cands}, {'sac': bases},
                                       lower_is_better=False,
                                       recipe_selection='fold_train_rows')
                resel.setdefault(metric, {})[key] = {
                    'block': blk.get('sac', {}),
                    'candidates_by_model': {'sac': cands},
                    'baselines_by_model': {'sac': bases},
                    'hybrid_distance': key, 'baseline_distance': base_key}
            except Exception as e:
                resel.setdefault(metric, {})[key] = {
                    'error': f'{type(e).__name__}: {e}'}
    results['fold_reselection_by_distance'] = resel
    # s09 reads these two. They are the block for the first decided distance
    # (manhattan on every dataset group), the hybrid side importance-weighted
    # and unweighted respectively, single representations unweighted.
    first = axes['metrics'][0]
    results['sac_reselected'] = (resel.get(first, {}).get('sac_weighted', {})
                                 .get('block', {}))
    results['sac_reselected_unweighted'] = (
        resel.get(first, {}).get('sac_unweighted', {}).get('block', {}))
    results['sac_reselected_distance'] = first


# ─────────────────────────────────────────────────────────────────────────────

def run(data, config, output_dir, dataset_name, **kwargs):
    """The optimal hybrid dimension may differ for prediction vs similarity.
    Similarity may peak at lower dim than prediction (curse of dimensionality)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S08: OPTIMAL DIMENSION FOR SIMILARITY VS PREDICTION")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    n_mol = len(labels)
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    n_sample = int(kwargs.get('n_sample', 5000))
    if kwargs.get('budgets') is not None:
        print("  --budgets ignored: s08 sweeps the dataset's decided general "
              "and distance budgets (decided_grid).")

    # The dataset's two decided models; knn only through the distance search.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=n_mol)
    models = [m for m in models if m in factories]
    gen_budgets = sorted(int(b) for b in
                         decided_search(dataset_name, 'general')['budgets'])
    dist_budgets = (sorted(int(b) for b in decided_search(
        dataset_name, 'general_distance')['budgets'])
        if is_decided(dataset_name, 'general_distance') else [])
    # knn's factory is used only to rebuild knn's distance-search recipe the
    # way the search scored it; no kNN number in the result comes from it.
    knn_factory = get_model_factories(['knn'], task_type=task_type,
                                      n_samples=n_mol).get('knn')
    axes = knn_axes(dataset_name, n_mol)
    full_reps = [r for r in (DATASET_REPS.get(dataset_name, {}).get('general')
                             or sorted(data['reps']))
                 if r in data['reps']]

    folds = sim_outer_folds(data, config)
    n_folds = len(folds)
    print(f"  Models: {models}; general budgets {gen_budgets}; distance "
          f"budgets {dist_budgets}; {n_folds} outer fold(s); kNN group "
          f"{axes['group']}: distances {axes['metrics']}, neighbour counts "
          f"{axes['ks']}, counting {axes['schemes']}")

    results = {
        'selection_rule': _SELECTION_RULE,
        'selection_scope': 'fold_train_rows',
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'models': models, 'n_outer_folds': n_folds,
        'knn_axes': axes, 'general_budgets': gen_budgets,
        'distance_budgets': dist_budgets, 'n_sample_pairs': n_sample,
        'single_rep_importance': SINGLE_REP_IMPORTANCE,
        'unit_version': UNIT_VERSION, 'folds': [],
    }

    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's08_optimal_dimension', dataset_name),
        run_identity('s08_optimal_dimension', dataset_name, data, config,
                     n_sample=n_sample, models=models,
                     general_budgets=gen_budgets, distance_budgets=dist_budgets,
                     knn_axes=axes, unit_version=UNIT_VERSION))

    for fold_i, (tr, va) in enumerate(folds):
        print(f"\n  Fold {fold_i + 1}/{n_folds}: {len(tr)} training, "
              f"{len(va)} held-out")
        if _ck.done(fold=fold_i):
            rec = _ck.get(fold=fold_i)
        else:
            rec = convert_numpy(_run_fold(
                data, dataset_name, fold_i=fold_i, n_folds=n_folds,
                train_idx=tr, val_idx=va, models=models, factories=factories,
                knn_factory=knn_factory, gen_budgets=gen_budgets,
                dist_budgets=dist_budgets, axes=axes, metric_name=metric_name,
                metric_fn=metric_fn, lower_is_better=lower_is_better,
                is_clf=is_clf, n_sample=n_sample, full_reps=full_reps))
            # A fold whose selection raised carries 'error' and is not saved,
            # so the next attempt retries it.
            _ck.put(rec, fold=fold_i)
        results['folds'].append(rec)
        for m in models:
            for b in gen_budgets:
                cells = (((rec.get('prediction') or {}).get(m) or {})
                         .get(str(b)) or {}).get('hybrids') or []
                c = next((c for c in cells if c.get('rank') == 1), None)
                if c and _finite(c.get('held_out_score')):
                    print(f"    {m} @ {b} features (built "
                          f"{c.get('width_built')}): {metric_name} "
                          f"{c['held_out_score']:.4f}")
        for b, c in sorted(_family_cells(rec, 'distance_family').items(),
                           key=lambda t: int(t[0])):
            _s = ((c.get('neighbour') or {}).get('by_distance', {})
                  .get(axes['metrics'][0], {}).get('sac_unweighted'))
            if _finite(_s):
                print(f"    knn distance hybrid @ {b} features (built "
                      f"{c.get('width_built')}): held-out SAC "
                      f"[{axes['metrics'][0]}, unweighted] {_s:.4f}")
        if rec.get('unavailable'):
            print(f"    unavailable: {rec['unavailable']}")

    _summarise(results, results['folds'], models=models, axes=axes,
               lower_is_better=lower_is_better, gen_budgets=gen_budgets,
               dist_budgets=dist_budgets)

    print("\n  Optimal budgets (best fold mean; None = no budget scored):")
    for m in models:
        print(f"    {m}: {results['optimal']['prediction'][m]['budget']}")
    for metric, d in results['optimal']['distance_family_sac'].items():
        print(f"    knn distance hybrid, SAC [{metric}]: weighted "
              f"{d['sac_weighted']['budget']}, unweighted "
              f"{d['sac_unweighted']['budget']}")
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S08: Optimal Dimension for Similarity vs Prediction',
                                     has_delta=True, has_augment=True)
    # create_experiment_parser already defines --budgets (comma-separated). It
    # is ignored: the budgets are the dataset's decided ones.
    parser.add_argument('--n-sample', type=int, default=5000,
                        help='Number of held-out pairs sampled for SAC (default: 5000)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  budgets=(args.budgets if getattr(args, 'budgets', None)
                           else None),
                  n_sample=args.n_sample)
    save_experiment_results(results, args.output, 's08_optimal_dimension', args.dataset)
