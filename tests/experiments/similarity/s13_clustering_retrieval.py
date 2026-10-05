#!/usr/bin/env python3
"""S13: Clustering Quality & Retrieval Precision.

Paper: Similarity
If hybrid distance is a better metric, it should produce better clusters
(members have similar labels) and better retrieval (query -> similar-activity
neighbors). Tests this across single-representation, PCA-matched, and hybrid
spaces.

Two sub-analyses:
  13a. Clustering: k-means label coherence (regression: 1 - within-cluster
       std(y)/std(y); classification: mean cluster majority-class purity).
       Higher = better.
  13b. Retrieval: for each held-out molecule, the label gap to its nearest
       training molecules, each neighbour counted by the counting scheme
       (regression: weighted mean |y_query - y_neighbour|; classification:
       1 - weighted share of neighbours in the query's class). Lower = better
       (a retrieval COST).

HONEST-COMPARISON REDESIGN (2026-07-11, user-approved decision B). Both
statistics are computed UNDER CROSS-VALIDATION so no space gets a look at the
points it is scored on: k-means and the neighbour index are fitted on the
training rows of an outer fold and scored on its held-out rows (cluster
assignment for held-out points; queries answered by training neighbours, so
there is no self-neighbour). PCA and every scaler are fitted on the training
rows only.

ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md Part 2, chat 4).
  * folds: sim_outer_folds, count from config['n_splits'] (any count,
    including 1, which is one held-out split).
  * selection, per outer fold, from searches on that fold's TRAINING rows on
    the dataset's decided grid (fold_similarity_selection, neighbour_spaces=
    True). Until this date the hybrids were read from the stored
    whole-dataset searches (get_rule_hybrids), ranked on the other folds'
    stored scores: that held out a fold's scores, not its rows (R2). The
    spaces are now:
      rep_knn1, rep_knn2      knn's 2 best single representations, ranked by
                              the per-fold distance search (Part 1 step 5).
                              Until this date every loaded representation
                              was a space.
      hybrid_<model>_general1/2
                              each decided model's 2 best hybrids of the
                              per-fold general search (R1, R4).
      hybrid_knn_distance1    knn's top hybrid of the per-fold distance search.
      hybrid_similarity       the top hybrid of the per-fold similarity-
                              objective search.
      pca_matched, pca_for_<hybrid slot>
                              PCA controls of the dataset's representations,
                              at 100 features and at each hybrid's own width.
    A slot's representation or recipe can differ from fold to fold; which one
    each fold held is in results['folds'][i]['spaces'].
    A hybrid whose selected features equal an earlier slot's on the same fold
    is not scored again; it is listed with 'same_features_as' (2.3 item 8).
  * every hybrid is rebuilt on the fold's training rows with the fold search's
    own representation scores (build_fold_entry; dirty-run bug 1).
  * scaling: FeatureScaler (0/1 features unscaled, the rest standardised),
    fitted on the training rows, held-out values cut to the training range
    (scale_fold; 2.3 item 17, dirty-run bug 3).
  * PCA controls are no longer standardised a second time after PCA (dirty-run
    s13 bug). Standardising the components gave every component unit
    variance, so a 250-component control weighed its last, near-noise
    components as heavily as its first and scored worse than the
    100-component one on every dataset of the dirty run.
  * retrieval at every decided kNN setting (knn_axes, KNN_SEARCH_SPACE_RECORD
    axes 2, 4, 5, 6): each distance of the dataset's group, 5, 10 and 20
    neighbours, rank and inverse_square counting, no scaffold group excluded.
    Until this date one distance (KIRBY_DISTANCE_METRIC), one neighbour count
    (KIRBY_S13_K_RETRIEVAL, default 5) and equal counting.
  * the importance-weighted retrieval of each hybrid is kept as it was
    (importance ** KIRBY_IMPORTANCE_EXPONENT, rescaled to sum 1): the
    weighting inside a similarity experiment's own distance is still open
    (handoff 2.10).
  * no pool read, so no fold is excluded for fold-0 pruning: per-fold
    searches run with pruning off.

The two axes have OPPOSITE metric directions (coherence higher-better,
retrieval cost lower-better), so each has its own fold_reselection block. A
reader feeds each block straight to experiments._shared.fold_reselection.

Usage:
    python s13_clustering_retrieval.py --dataset logd --mode intermediary
    python s13_clustering_retrieval.py --dataset flab_thermo_tresanco --mode intermediary
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments.similarity.similarity_helpers import _CODE_FAULTS
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, kmeans_plusplus
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    compute_feature_importance, get_domain_metric, get_model_factories,
    fold_reselection, hr_exclude_folds,
    importance_feature_weights, hybrid_feature_signature,
)
from experiments.similarity.similarity_helpers import (
    DATASET_REPS, sim_outer_folds, fold_spread_note, knn_axes, scale_fold,
    knn_lists, fold_similarity_selection, build_fold_entry, entry_record,
)

# The selection rule the per-fold spaces come from. Part of the checkpoint
# identity, so partial files written under the pool read are not read back.
from experiments.similarity._summary_keys import SELECTION_RULE as _SELECTION_RULE

# How many restarts the clusterer gets. k-means has used 10 since the file was
# written; the coefficient-matched clusterer below is given the same number so
# the two differ in the distance and in nothing else.
_N_INIT = int(os.environ.get('KIRBY_S13_N_INIT', '').strip() or '10')
_MAX_ITER = int(os.environ.get('KIRBY_S13_MAX_ITER', '').strip() or '25')

# Cluster counts. 3, 5 and 10 were written into the signature and never swept;
# KIRBY_S13_CLUSTERS sweeps them without editing the file. Coherence is a mean
# of within-cluster standard deviations, and more clusters means smaller
# clusters, so the statistic is NOT comparable across counts — see the note on
# _coherence_from_labels.
_ENV_CLUSTERS = os.environ.get('KIRBY_S13_CLUSTERS')
DEFAULT_N_CLUSTERS = ([int(v) for v in _ENV_CLUSTERS.split(',') if v.strip()]
                      if _ENV_CLUSTERS else [3, 5, 10])
# Whether to also score every space with the clusterer that matches each
# decided distance. Off by default: it is a second full set of cluster fits
# per distance.
_MATCHED_ON = os.environ.get('KIRBY_S13_MATCHED_CLUSTERING', '') not in ('', '0')
# Which cluster counts the matched arm is scored at. Unset means every count in
# the sweep; a shorter list keeps the second set of cluster fits affordable.
_ENV_MATCHED_CLUSTERS = os.environ.get('KIRBY_S13_MATCHED_CLUSTERS')
MATCHED_N_CLUSTERS = ([int(v) for v in _ENV_MATCHED_CLUSTERS.split(',')
                       if v.strip()] if _ENV_MATCHED_CLUSTERS else None)

# Width in features of the fixed PCA control. The per-hybrid controls are
# fitted at each hybrid's own width; this one is kept beside them as it was.
FIXED_PCA_WIDTH = 100

# Feature weighting of a retrieval: the plain scaled features, or the hybrid's
# features multiplied by their training-row importance (hybrid slots only).
RET_PLAIN = 'features_unweighted'
RET_IMPORTANCE = 'features_importance_weighted'


# =========================================================================
# LOCAL HELD-OUT STATISTICS (unsupervised — the shared fold_reselection helper
# does the honest ARITHMETIC on per-fold scores, but these two statistics are
# not model-prediction metrics, so the held-out computation lives here.)
# =========================================================================

def _has_finite(scores):
    """True if the list has at least one finite (non-NaN/None) entry."""
    return any(s is not None and np.isfinite(s) for s in scores)


def _nanmean(scores):
    """Mean over finite entries; NaN if none are finite."""
    arr = np.asarray([s for s in scores if s is not None], dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float('nan')
    return float(np.nanmean(arr))


def _fold_spread(scores):
    """Per-fold scores and their range; 'one fold, no spread' under 2 folds."""
    vals = [float(s) for s in scores if s is not None and np.isfinite(s)]
    note = fold_spread_note(vals)
    return {'fold_scores': [None if (s is None or not np.isfinite(s))
                            else float(s) for s in scores],
            'fold_min': (min(vals) if vals and note is None else None),
            'fold_max': (max(vals) if vals and note is None else None),
            'spread_note': note}


def setting_key(distance, k, scheme):
    """Name of one kNN setting: distance, neighbour count, counting scheme."""
    return f'{distance}|k{int(k)}|{scheme}'


def matched_clusterer_name(coefficient):
    """Which clusterer matches distance `coefficient`, or None when none does.

    k-means minimises the sum of SQUARED EUCLIDEAN distances to a centroid and
    takes no metric argument. The retrieval half of this file measures every
    distance of the dataset's kNN group (knn_axes), none of which is squared
    euclidean.

    Two of the three decided distances have an exact same-shape algorithm,
    each one 'assign to the nearest centre, move the centre to the point that
    minimises summed distance inside its cluster':

      manhattan        k-medians. The centre is the per-feature median, which
                       is the point minimising summed L1 distance.
      cosine           k-means on L2-normalised rows. On unit vectors cosine
                       distance is 1 - x.y and squared Euclidean is 2 - 2 x.y,
                       so the two order every pair identically.
      minkowski_p0.5   none. No closed form for the point minimising summed
                       L_0.5 distance, and the medoid alternative needs every
                       within-cluster pair. Returns None; the caller writes no
                       matched block for it and says so.
      euclidean        k-means itself (not a decided distance; kept so the
                       name resolves).
    """
    return {'euclidean': 'kmeans',
            'manhattan': 'kmedians',
            'cosine': 'spherical_kmeans'}.get(coefficient)


def _fit_assign(X_tr_s, X_va_s, n_clusters, method):
    """Fit `method` on TRAIN rows, return the HELD-OUT rows' cluster labels.

    `method` is one of the names matched_clusterer_name returns. Every one gets
    _N_INIT restarts from k-means++ seeds, so the arms differ in the distance
    and not in how many times they were restarted.
    """
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=_N_INIT)
        km.fit(X_tr_s)
        return km.predict(X_va_s)
    if method == 'spherical_kmeans':
        def _unit(A):
            n = np.linalg.norm(A, axis=1, keepdims=True)
            return A / np.maximum(n, 1e-12)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=_N_INIT)
        km.fit(_unit(X_tr_s))
        return km.predict(_unit(X_va_s))
    if method != 'kmedians':
        raise ValueError(f"no clusterer named {method!r}")
    best_C, best_cost = None, np.inf
    for s in range(_N_INIT):
        C, _ = kmeans_plusplus(X_tr_s, n_clusters, random_state=42 + s)
        C = np.asarray(C, dtype=np.float64)
        for _ in range(_MAX_ITER):
            lab = cdist(X_tr_s, C, 'cityblock').argmin(axis=1)
            newC = C.copy()
            for c in range(n_clusters):
                m = lab == c
                if m.any():
                    newC[c] = np.median(X_tr_s[m], axis=0)
            if np.array_equal(newC, C):
                break
            C = newC
        D = cdist(X_tr_s, C, 'cityblock')
        cost = float(D.min(axis=1).sum())
        if cost < best_cost:
            best_cost, best_C = cost, C
    return cdist(X_va_s, best_C, 'cityblock').argmin(axis=1)


def _coherence_from_labels(va_lab, y_va, n_clusters, is_clf):
    """Score held-out label coherence from the held-out rows' cluster labels.

    Regression: 1 - mean(within-cluster std(y_val)) / std(y_val_global).
    Classification: mean majority-class fraction among held-out points per
    cluster. Higher = the train-learned cluster geometry organises unseen
    labels well. Returns NaN when the fold cannot support the statistic
    (degenerate labels, no cluster with >=2 held-out points).

    Both branches need at least 2 held-out rows in a cluster. A cluster holding
    one row has purity 1.0 by construction, so admitting singletons rewarded a
    space for scattering held-out points into clusters of one while the
    regression branch, which always required 2, did not.

    THE MEAN IS OVER CLUSTERS, AND CLUSTERS BELOW 2 HELD-OUT ROWS ARE DROPPED,
    so two spaces at the same cluster count can be scored on different numbers
    of terms and a space that scatters held-out points is scored only on its
    dense clusters. `pooled` is the same quantity with each cluster weighted by
    how many held-out rows it holds: 1 - (pooled within-cluster standard
    deviation) / std(y_val_global), regression only. It is NaN for
    classification.

    THE STATISTIC IS NOT COMPARABLE ACROSS CLUSTER COUNTS. More clusters means
    fewer rows per cluster and a smaller within-cluster standard deviation, so
    coherence rises with the count for any space at all. Taking each arm's best
    over a list of counts compares two different granularities.

    Returns (score, n_clusters_dropped, pooled).
    """
    if is_clf:
        if (y_va < 0).any():
            return float('nan'), 0, float('nan')
        purities = []
        n_dropped = 0
        for c in range(n_clusters):
            mask = va_lab == c
            if mask.sum() >= 2:
                counts = np.bincount(y_va[mask].astype(int))
                purities.append(counts.max() / mask.sum())
            else:
                n_dropped += 1
        if not purities:
            return float('nan'), n_dropped, float('nan')
        return float(np.mean(purities)), n_dropped, float('nan')

    y_std = np.std(y_va)
    if y_std < 1e-10:
        return float('nan'), 0, float('nan')
    stds = []
    ss, n_in = 0.0, 0
    n_dropped = 0
    for c in range(n_clusters):
        mask = va_lab == c
        if mask.sum() >= 2:
            yc = y_va[mask]
            stds.append(np.std(yc))
            ss += float(np.sum((yc - yc.mean()) ** 2))
            n_in += int(mask.sum())
        else:
            n_dropped += 1
    if not stds:
        return float('nan'), n_dropped, float('nan')
    pooled = (float(1.0 - np.sqrt(ss / n_in) / y_std) if n_in > 0
              else float('nan'))
    return float(1.0 - np.mean(stds) / y_std), n_dropped, pooled


def _heldout_coherence(X_tr_s, y_tr, X_va_s, y_va, n_clusters, is_clf,
                       method='kmeans'):
    """Cluster the TRAIN rows, assign the HELD-OUT rows, score label coherence.

    `method` names the clusterer: 'kmeans' is what this file has always used and
    minimises squared Euclidean distance; the others come from
    matched_clusterer_name and match one decided distance.

    Returns (score, n_clusters_dropped, pooled). NaN score when the fold cannot
    support the statistic (too few train rows for n_clusters, degenerate labels,
    no cluster with >=2 held-out rows) — the fold_reselection helper skips NaN
    folds. n_clusters_dropped is how many of the n_clusters received fewer than
    2 held-out rows and so contributed no term.
    """
    if n_clusters >= len(y_tr):
        return float('nan'), 0, float('nan')
    va_lab = _fit_assign(X_tr_s, X_va_s, n_clusters, method)
    return _coherence_from_labels(va_lab, y_va, n_clusters, is_clf)


def _retrieval_cost(y_tr, y_va, idx, dist, k, scheme, is_clf):
    """Retrieval COST of the held-out rows from their k nearest training rows.

    idx/dist are each held-out row's neighbour list (knn_lists, nearest first).
    Each neighbour counts by `scheme` (rank: 1, 1/2, 1/3, ...; inverse_square:
    one over the distance squared; KNN_SEARCH_SPACE_RECORD axis 5).
    Regression: the weighted mean |y_query - y_neighbour|, averaged over the
    held-out rows. Classification: 1 - the weighted share of neighbours in the
    query's class, averaged over the held-out rows. LOWER = better.
    NaN when the list holds fewer than k neighbours.
    """
    from experiments.noise.similarity_space import neighbour_weights
    idx = np.asarray(idx)
    if k > idx.shape[1]:
        return float('nan')
    i = idx[:, :k]
    w = np.asarray(neighbour_weights(np.asarray(dist)[:, :k], scheme),
                   dtype=np.float64)
    tot = w.sum(axis=1)
    nb = np.asarray(y_tr)[i]
    yq = np.asarray(y_va)[:, None]
    if is_clf:
        per_row = (w * (nb == yq)).sum(axis=1)
        share = np.divide(per_row, tot, out=np.full_like(tot, np.nan),
                          where=tot > 0)
        return float(1.0 - np.nanmean(share))
    per_row = (w * np.abs(yq - nb)).sum(axis=1)
    gap = np.divide(per_row, tot, out=np.full_like(tot, np.nan),
                    where=tot > 0)
    return float(np.nanmean(gap))


def _retrieval_sweep(X_tr_s, y_tr, X_va_s, y_va, axes, is_clf):
    """Retrieval cost at every decided setting: {setting_key: cost}.

    One neighbour list per distance (k_max = the largest decided count) serves
    every count and scheme. The query rows are the held-out rows, so no row is
    its own neighbour; no scaffold group is excluded (axis 6).
    """
    out = {}
    for metric in axes['metrics']:
        idx, dist = knn_lists(X_tr_s, X_va_s, metric=metric,
                              k_max=axes['k_max'])
        for k in axes['ks']:
            for scheme in axes['schemes']:
                out[setting_key(metric, k, scheme)] = _retrieval_cost(
                    y_tr, y_va, idx, dist, k, scheme, is_clf)
    return out


def _plain(obj):
    """A JSON-safe copy of a selection record (blueprints and searches out)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()
                if k not in ('blueprint', 'search', 'feature_signature')}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def _pca_input_reps(dataset_name, reps):
    """The representations the PCA controls are fitted on, and where the list
    came from: DATASET_REPS[ds]['general'] (handoff 2.4 step 9), restricted to
    those loaded. Falls back to every loaded representation, and says so, when
    the dataset has no DATASET_REPS row."""
    general = (DATASET_REPS.get(dataset_name) or {}).get('general')
    if general:
        names = [r for r in general if r in reps]
        missing = [r for r in general if r not in reps]
        return sorted(names), ("DATASET_REPS['general']"
                               + (f'; not loaded: {missing}' if missing else ''))
    return sorted(reps), 'every loaded representation (no DATASET_REPS row)'


def _pca_uses_augmentation(dataset_name):
    """True when a decided search of this dataset keeps an augmentation
    condition (Step 8: only ginkgo_colloidal and ginkgo_hic, distance search).
    Elsewhere the PCA controls see no augmentation features, as no hybrid
    does."""
    from experiments.decided_grid import decided_aug_conditions
    for tier in ('general', 'general_distance'):
        try:
            if any(c != 'none' for c in decided_aug_conditions(dataset_name,
                                                               tier)):
                return True
        except Exception:
            continue
    return False


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S13: Clustering quality & retrieval precision (held-out, under CV)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S13: CLUSTERING & RETRIEVAL")
    print("=" * 70)

    from experiments.decided_grid import is_decided, decided_models

    reps = data['reps']
    labels = np.asarray(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    augmentations = data.get('augmentations')
    # Width of the fixed PCA control; not a hybrid budget.
    budget = int(kwargs.get('budget') or FIXED_PCA_WIDTH)
    n_clusters_list = kwargs.get('n_clusters_list') or DEFAULT_N_CLUSTERS
    axes = knn_axes(dataset_name, len(labels))

    if not is_decided(dataset_name, 'general'):
        why = (f'{dataset_name} has no DECISIONS.md row, so no per-fold '
               f'search is run for it')
        print(f"  s13 skipped: {why}")
        return {'skipped': why, 'knn_axes': axes}

    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models + ['knn'], task_type=task_type,
                                    n_samples=len(labels))

    # The clusterer matching each decided distance, for the optional matched
    # arm. k-means itself minimises squared euclidean, which is none of them.
    matched_methods = {d: matched_clusterer_name(d) for d in axes['metrics']}
    matched_on = bool(_MATCHED_ON)
    matched_clusters = [nc for nc in (MATCHED_N_CLUSTERS or n_clusters_list)
                        if nc in n_clusters_list]
    print(f"  kNN settings ({axes['group']}): distances {axes['metrics']}, "
          f"neighbour counts {axes['ks']}, counting {axes['schemes']}, "
          f"scaffold group not excluded")
    print(f"  clustering: k-means (squared euclidean), cluster counts "
          f"{n_clusters_list}, {_N_INIT} restarts; matched clusterers "
          f"{matched_methods}; matched arm "
          f"{('ON at cluster counts ' + str(matched_clusters)) if matched_on else 'off'}")
    print(f"  models {models}; training-row scores in {metric_name} "
          f"({'lower' if lower_is_better else 'higher'} is better)")

    pca_reps, pca_reps_source = _pca_input_reps(dataset_name, reps)
    pca_augs = (sorted(augmentations) if (augmentations
                                          and _pca_uses_augmentation(dataset_name))
                else [])

    try:
        folds = list(sim_outer_folds(data, config))
    except Exception as e:
        # No random-split fallback (feedback_no_random_splits.md). Bail out
        # cleanly rather than crash the batch.
        print(f"  CV split failed: {type(e).__name__}: {e}")
        return {'config': {'n_clusters_list': n_clusters_list,
                           'n_folds': int(config['n_splits']),
                           'knn_axes': axes},
                'cv_error': f"{type(e).__name__}: {e}"}
    n_folds = len(folds)

    # Each fold's record is written to
    # s13_clustering_retrieval/<dataset>.partial.jsonl when the fold finishes
    # and read back on the next attempt (_checkpoint.py). A fold in which a
    # selection, build, scale, PCA or retrieval raised is not stored, so it is
    # retried. The identity names the selection rule and the kNN settings, so
    # a partial file from the pool read (before 2026-10-05) is not read back.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's13_clustering_retrieval', dataset_name),
        run_identity('s13_clustering_retrieval', dataset_name, data, config,
                     kwargs=kwargs, budget=budget,
                     n_clusters_list=list(n_clusters_list),
                     matched_on=matched_on,
                     matched_clusters=list(matched_clusters),
                     n_folds=n_folds, selection_rule=_SELECTION_RULE,
                     knn_axes={k: axes[k] for k in
                               ('metrics', 'ks', 'schemes')},
                     pca_reps=pca_reps, pca_augs=pca_augs))

    fold_records = []
    print(f"\n  Held-out CV ({n_folds} fold(s), group-aware)")
    for fold_i, (tr, va) in enumerate(folds):
        if _ck.done(fold=fold_i):
            fold_records.append(_ck.get(fold=fold_i))
            print(f"  fold {fold_i}: read back from the partial file")
            continue
        t_fold = time.time()
        rec, threw = _run_fold(
            data, dataset_name, fold_i, n_folds, np.asarray(tr),
            np.asarray(va), labels=labels, augmentations=augmentations,
            models=models, factories=factories, metric_fn=metric_fn,
            lower_is_better=lower_is_better, is_clf=is_clf, axes=axes,
            n_clusters_list=n_clusters_list, budget=budget,
            pca_reps=pca_reps, pca_augs=pca_augs, matched_on=matched_on,
            matched_methods=matched_methods,
            matched_clusters=matched_clusters)
        rec['seconds'] = round(time.time() - t_fold, 1)
        print(f"  fold {fold_i}: {rec['seconds']} s"
              + (f", {len(threw)} step(s) raised, not checkpointed"
                 if threw else ''))
        if not threw:
            _ck.put(rec, fold=fold_i)
        fold_records.append(rec)

    return _aggregate(fold_records, n_folds=n_folds, axes=axes,
                      n_clusters_list=n_clusters_list, budget=budget,
                      metric_name=metric_name,
                      lower_is_better=lower_is_better, models=models,
                      matched_on=matched_on, matched_methods=matched_methods,
                      matched_clusters=matched_clusters,
                      pca_reps=pca_reps, pca_reps_source=pca_reps_source,
                      pca_augs=pca_augs)


def _run_fold(data, dataset_name, fold_i, n_folds, tr, va, *, labels,
              augmentations, models, factories, metric_fn, lower_is_better,
              is_clf, axes, n_clusters_list, budget, pca_reps, pca_augs,
              matched_on, matched_methods, matched_clusters):
    """Select, build and score every space of one outer fold.

    Returns (record, threw): the fold's record (selection, spaces and every
    held-out score, keyed by space) and the list of steps that raised.
    """
    reps = data['reps']
    y_tr, y_va = labels[tr], labels[va]
    threw = []
    rec = {'fold': int(fold_i), 'n_train': int(len(tr)),
           'n_held_out': int(len(va)), 'selection': None, 'spaces': {},
           'clustering': {}, 'clustering_pooled': {}, 'clusters_dropped': {},
           'clustering_matched': {}, 'retrieval': {}}

    # ── Selection on this fold's training rows (R2, R4, 2.3 item 1) ──
    # The FULL data['reps'] goes in; the hybrid pool is never cut.
    try:
        sel = fold_similarity_selection(
            data, dataset_name, train_idx=tr, fold_i=fold_i,
            n_outer_folds=n_folds, neighbour_spaces=True)
    except _CODE_FAULTS:
        raise  # a fault in the code, not a failed search
    except Exception as e:
        threw.append('selection')
        print(f"    fold {fold_i}: selection raised {type(e).__name__}: {e}")
        rec['selection'] = {'error': f'{type(e).__name__}: {e}',
                            'selection_scope': 'fold_train_rows'}
        sel = None

    # (space name, kind, train matrix, held-out matrix, already scaled?)
    spaces = []
    if sel is not None:
        rec['selection'] = {
            'selection_scope': sel['selection_scope'],
            'models': {m: (None if ms is None else {
                'base': list(ms.get('base') or []),
                'hybrids': [_plain(entry_record(e))
                            for e in ms.get('hybrids') or []],
                'hybrids_skipped_identical_features': _plain(
                    ms.get('hybrids_skipped_identical_features') or []),
            }) for m, ms in sel['models'].items()},
            'knn_base': list(sel['knn_base']),
            'knn_base_detail': _plain(sel.get('knn_base_detail')),
            'distance_hybrid': _plain(entry_record(sel.get('distance_hybrid'))),
            'similarity_hybrid': _plain(
                entry_record(sel.get('similarity_hybrid'))),
            'unavailable': dict(sel.get('unavailable') or {}),
        }

        # Single-representation neighbour arms: knn's 2 best from the per-fold
        # distance search (Part 1 step 5). Named by rank, so a slot is one
        # series across folds even when its representation changes.
        for r_i, rname in enumerate(sel['knn_base'][:2], start=1):
            sname = f'rep_knn{r_i}'
            rec['spaces'][sname] = {'kind': 'single', 'representation': rname,
                                    'ranked_by': 'knn, per-fold distance '
                                                 'search, training rows'}
            spaces.append((sname, 'single', reps[rname][tr], reps[rname][va],
                           False))

        # Hybrid arms: each decided model's 2 general hybrids, knn's top
        # distance hybrid, the top similarity hybrid. (slot, entry, model the
        # hybrid was ranked for; None for a similarity recipe, which fits none)
        hyb = []
        for m in models:
            ms = sel['models'].get(m)
            if ms is None:
                continue
            for h_i, e in enumerate((ms.get('hybrids') or [])[:2], start=1):
                hyb.append((f'hybrid_{m}_general{h_i}', e, m))
        if sel.get('distance_hybrid') is not None:
            hyb.append(('hybrid_knn_distance1', sel['distance_hybrid'], 'knn'))
        if sel.get('similarity_hybrid') is not None:
            hyb.append(('hybrid_similarity', sel['similarity_hybrid'], None))

        seen_sigs = {}
        for sname, e, model in hyb:
            info = {'kind': 'hybrid', 'model': model,
                    'record': _plain(entry_record(e))}
            rec['spaces'][sname] = info
            try:
                Xh_tr, Xh_va, fi = build_fold_entry(
                    data, e, train_idx=tr, val_idx=va,
                    model_factory=(factories.get(model) if model else None),
                    metric_fn=metric_fn, lower_is_better=lower_is_better)
            except Exception as ex:
                threw.append(f'build {sname}')
                info['build_failed'] = f'{type(ex).__name__}: {ex}'
                print(f"    fold {fold_i}: hybrid build failed ({sname}): "
                      f"{type(ex).__name__}: {str(ex)[:120]}")
                continue
            info['width_built'] = int(np.asarray(Xh_tr).shape[1])
            try:
                sig = hybrid_feature_signature(fi)
            except Exception:
                sig = None
            if sig and sig in seen_sigs:
                # Same selected features as a slot already scored on this
                # fold: its scores would be identical (2.3 item 8).
                info['same_features_as'] = seen_sigs[sig]
                print(f"    fold {fold_i}: {sname} selects the same features "
                      f"as {seen_sigs[sig]}; not scored twice")
                continue
            if sig:
                seen_sigs[sig] = sname
            spaces.append((sname, 'hybrid', Xh_tr, Xh_va, False))

    # ── PCA controls, fitted on the training rows only ──
    # Input: the dataset's general representations (plus augmentation
    # features only where a decided search keeps them), scaled with
    # FeatureScaler on the training rows. The components are NOT scaled again
    # (dirty-run s13 bug): rescaling gave every component unit variance.
    pca_cache = {}
    Zc_tr = Zc_va = None
    try:
        Xc_tr = np.hstack([np.asarray(reps[r])[tr] for r in pca_reps]
                          + [np.asarray(augmentations[a])[tr] for a in pca_augs])
        Xc_va = np.hstack([np.asarray(reps[r])[va] for r in pca_reps]
                          + [np.asarray(augmentations[a])[va] for a in pca_augs])
        Zc_tr, Zc_va = scale_fold(Xc_tr, Xc_va)
    except Exception as e:
        threw.append('pca input')
        print(f"    fold {fold_i}: PCA input failed: {type(e).__name__}: {e}")

    def _pca_space(n_features):
        if Zc_tr is None:
            return None, None
        dim = min(int(n_features), Zc_tr.shape[1], len(tr) - 1)
        if dim < 2:
            return None, None
        if dim not in pca_cache:
            try:
                pca = PCA(n_components=dim, random_state=42)
                pca_cache[dim] = (pca.fit_transform(Zc_tr), pca.transform(Zc_va))
            except Exception as e:
                threw.append(f'pca {dim}')
                print(f"    fold {fold_i}: PCA failed ({dim} components): "
                      f"{type(e).__name__}: {e}")
                return None, None
        return pca_cache[dim], dim

    pair, dim = _pca_space(budget)
    if pair is not None:
        rec['spaces']['pca_matched'] = {'kind': 'pca_control',
                                        'components': int(dim),
                                        'controls_for': None,
                                        'requested_width': int(budget)}
        spaces.append(('pca_matched', 'pca_control', pair[0], pair[1], True))
    for sname, kind, Xtr, _, _ in list(spaces):
        if kind != 'hybrid':
            continue
        w = int(np.asarray(Xtr).shape[1])
        pair, dim = _pca_space(w)
        if pair is None:
            continue
        cname = f'pca_for_{sname}'
        rec['spaces'][cname] = {'kind': 'pca_control', 'components': int(dim),
                                'controls_for': sname, 'slot_width': w}
        spaces.append((cname, 'pca_control', pair[0], pair[1], True))

    # ── Score every space on the held-out rows ──
    for sname, kind, Xtr, Xva, prescaled in spaces:
        try:
            if prescaled:
                Xtr_s = np.asarray(Xtr, dtype=np.float64)
                Xva_s = np.asarray(Xva, dtype=np.float64)
            else:
                Xtr_s, Xva_s = scale_fold(Xtr, Xva)
        except Exception as e:
            threw.append(f'scale {sname}')
            print(f"    fold {fold_i}: scale failed ({sname}): "
                  f"{type(e).__name__}: {e}")
            continue

        for nc in n_clusters_list:
            coh, dropped, pooled = _heldout_coherence(
                Xtr_s, y_tr, Xva_s, y_va, nc, is_clf, method='kmeans')
            rec['clustering'].setdefault(sname, {})[int(nc)] = coh
            rec['clustering_pooled'].setdefault(sname, {})[int(nc)] = pooled
            rec['clusters_dropped'].setdefault(sname, {})[int(nc)] = dropped
            if matched_on and nc in matched_clusters:
                for dname, method in matched_methods.items():
                    if method is None:
                        continue
                    try:
                        cm, _, _ = _heldout_coherence(
                            Xtr_s, y_tr, Xva_s, y_va, nc, is_clf,
                            method=method)
                    except Exception as e:
                        threw.append(f'{method} {sname}')
                        print(f"    fold {fold_i}: {method} failed ({sname}, "
                              f"nc={nc}): {type(e).__name__}: {e}")
                        cm = float('nan')
                    rec['clustering_matched'].setdefault(sname, {}).setdefault(
                        dname, {})[int(nc)] = cm

        try:
            rec['retrieval'].setdefault(sname, {})[RET_PLAIN] = (
                _retrieval_sweep(Xtr_s, y_tr, Xva_s, y_va, axes, is_clf))
        except Exception as e:
            threw.append(f'retrieval {sname}')
            print(f"    fold {fold_i}: retrieval failed ({sname}): "
                  f"{type(e).__name__}: {e}")

        # Importance-weighted retrieval — hybrid slots only, weights from the
        # training rows, each with the importance method its own recipe was
        # built with. The power (KIRBY_IMPORTANCE_EXPONENT, 0.5 when unset)
        # and the rescaling to sum 1 are unchanged: both are still open
        # (handoff 2.10).
        if kind == 'hybrid':
            bp_imp = ((rec['spaces'][sname].get('record') or {})
                      .get('importance') or 'lightgbm_gain')
            try:
                imp = compute_feature_importance(
                    {'hybrid': Xtr_s}, y_tr, method=bp_imp).get(
                    'hybrid', np.ones(Xtr_s.shape[1]))
                w = importance_feature_weights(imp)
                w = w / (w.sum() + 1e-10)
                rec['retrieval'][sname][RET_IMPORTANCE] = _retrieval_sweep(
                    Xtr_s * w, y_tr, Xva_s * w, y_va, axes, is_clf)
            except Exception as e:
                threw.append(f'weighted retrieval {sname}')
                print(f"    fold {fold_i}: weighted retrieval failed "
                      f"({sname}): {type(e).__name__}: {e}")
    return rec, threw


def _aggregate(fold_records, *, n_folds, axes, n_clusters_list, budget,
               metric_name, lower_is_better, models, matched_on,
               matched_methods, matched_clusters, pca_reps, pca_reps_source,
               pca_augs):
    """Fold-aligned series, means, re-pick blocks and the summary."""
    settings = [setting_key(d, k, s) for d in axes['metrics']
                for k in axes['ks'] for s in axes['schemes']]

    def _series(get):
        """One value per fold index, NaN where that fold has none."""
        out = []
        for i in range(n_folds):
            r = next((f for f in fold_records if f.get('fold') == i), None)
            v = get(r) if r is not None else None
            out.append(float(v) if (v is not None and np.isfinite(v))
                       else float('nan'))
        return out

    spaces = sorted({s for r in fold_records
                     for s in list(r.get('clustering', {}))
                     + list(r.get('retrieval', {}))})
    hybrid_spaces = [s for s in spaces if s.startswith('hybrid_')]

    # clustering_fold[space][nc] = per-fold coherence
    clustering_fold = {s: {nc: _series(lambda r, s=s, nc=nc: r.get(
        'clustering', {}).get(s, {}).get(nc)) for nc in n_clusters_list}
        for s in spaces}
    pooled_fold = {s: {nc: _series(lambda r, s=s, nc=nc: r.get(
        'clustering_pooled', {}).get(s, {}).get(nc)) for nc in n_clusters_list}
        for s in spaces}
    # retrieval_fold[space][cond][setting] = per-fold cost
    retrieval_fold = {}
    for s in spaces:
        for cond in (RET_PLAIN, RET_IMPORTANCE):
            if not any(cond in (r.get('retrieval', {}).get(s) or {})
                       for r in fold_records):
                continue
            retrieval_fold.setdefault(s, {})[cond] = {
                st: _series(lambda r, s=s, c=cond, st=st: (
                    (r.get('retrieval', {}).get(s) or {}).get(c) or {}).get(st))
                for st in settings}

    clustering_results = {}
    for s, ncmap in clustering_fold.items():
        entry = {}
        for nc, fs in ncmap.items():
            drops = [r.get('clusters_dropped', {}).get(s, {}).get(nc)
                     for r in fold_records]
            drops = [d for d in drops if d is not None]
            entry[nc] = dict(_fold_spread(fs), **{
                'label_coherence': _nanmean(fs),
                'n_folds_used': int(sum(1 for v in fs if np.isfinite(v))),
                # Each cluster weighted by its held-out row count instead of
                # every cluster counting once.
                'label_coherence_pooled': _nanmean(pooled_fold[s][nc]),
                # Clusters that carried >=2 held-out rows and so contributed a
                # term, averaged over the folds that scored.
                'mean_clusters_scored': (float(nc - np.mean(drops)) if drops
                                         else float(nc)),
            })
        clustering_results[s] = entry

    retrieval_results = {}
    for s, conds in retrieval_fold.items():
        retrieval_results[s] = {
            cond: {st: dict(_fold_spread(fs), mean_cost=_nanmean(fs))
                   for st, fs in by_st.items()}
            for cond, by_st in conds.items()}

    results = {
        'clustering': convert_numpy(clustering_results),
        'retrieval': convert_numpy(retrieval_results),
        'config': {
            'n_clusters_list': list(n_clusters_list),
            'n_folds': n_folds,
            'fold_seed': 42,
            'spaces': spaces,
            'selection_rule': _SELECTION_RULE,
            'selection_scope': 'fold_train_rows',
            'models': list(models),
            'clustering_distance': 'squared_euclidean',
            'matched_clusterers': matched_methods,
            'matched_clustering_scored': matched_on,
            'n_init': _N_INIT,
            'fixed_pca_width': budget,
            'pca_input_representations': pca_reps,
            'pca_input_source': pca_reps_source,
            'pca_input_augmentations': pca_augs,
            'pca_components_rescaled': False,
            'scaler': 'FeatureScaler on training rows (0/1 features '
                      'unscaled), held-out cut to the training range',
            'importance_weighting': 'importance ** KIRBY_IMPORTANCE_EXPONENT, '
                                    'rescaled to sum 1 (open, handoff 2.10)',
        },
        'knn_axes': axes,
        'retrieval_settings': {st: {'distance': st.split('|')[0],
                                    'k': int(st.split('|')[1][1:]),
                                    'scheme': st.split('|')[2]}
                               for st in settings},
        # What each number's direction is. The training-row scores in the
        # selection records are in the dataset's metric; s13's own held-out
        # statistics have their own fixed directions.
        'metric': {'name': metric_name, 'lower_is_better': lower_is_better,
                   'applies_to': 'train_rows_score in each selection record'},
        'heldout_directions': {'coherence': 'higher_is_better',
                               'retrieval_cost': 'lower_is_better'},
        'spread_note': fold_spread_note([1.0] * n_folds),
        # Per fold: what was selected (models, base representations, hybrids
        # with their training-row scores), which representation or recipe
        # each slot held, and every held-out score.
        'folds': convert_numpy(sorted(fold_records,
                                      key=lambda r: r.get('fold', 0))),
    }

    # ── Legacy summary, on held-out means ──
    # Best single slot and best hybrid slot per axis, each read off the same
    # folds it is compared on (same-fold optimism); the re-pick blocks below
    # choose per held-out fold instead.
    def _best_coh(s):
        vals = [(v['label_coherence'], nc)
                for nc, v in clustering_results.get(s, {}).items()
                if np.isfinite(v['label_coherence'])]
        return max(vals) if vals else None

    def _best_ret(s):
        vals = [(v['mean_cost'], st) for st, v in
                (retrieval_results.get(s, {}).get(RET_PLAIN) or {}).items()
                if v['mean_cost'] is not None and np.isfinite(v['mean_cost'])]
        return min(vals) if vals else None

    rep_spaces = [s for s in spaces if s.startswith('rep_')]
    rep_coh = {s: _best_coh(s) for s in rep_spaces if _best_coh(s)}
    rep_ret = {s: _best_ret(s) for s in rep_spaces if _best_ret(s)}
    hyb_coh = {s: _best_coh(s) for s in hybrid_spaces if _best_coh(s)}
    hyb_ret = {s: _best_ret(s) for s in hybrid_spaces if _best_ret(s)}
    results['top_clustering'] = sorted(
        [(s, v[0], v[1]) for s, v in rep_coh.items()], key=lambda x: -x[1])
    results['top_retrieval'] = sorted(
        [(s, v[0], v[1]) for s, v in rep_ret.items()], key=lambda x: x[1])
    if hyb_coh or hyb_ret:
        hc = max(hyb_coh, key=lambda s: hyb_coh[s][0]) if hyb_coh else None
        hr_ = min(hyb_ret, key=lambda s: hyb_ret[s][0]) if hyb_ret else None
        bc = max(rep_coh, key=lambda s: rep_coh[s][0]) if rep_coh else None
        br = min(rep_ret, key=lambda s: rep_ret[s][0]) if rep_ret else None
        h_coh = hyb_coh[hc][0] if hc else float('nan')
        h_ret = hyb_ret[hr_][0] if hr_ else float('nan')
        b_coh = rep_coh[bc][0] if bc else float('nan')
        b_ret = rep_ret[br][0] if br else float('nan')
        results['summary'] = {
            'hybrid_coherence': h_coh,
            'hybrid_retrieval': h_ret,
            'hybrid_coherence_space': hc,
            'hybrid_coherence_n_clusters': hyb_coh[hc][1] if hc else None,
            'hybrid_retrieval_space': hr_,
            'hybrid_retrieval_setting': hyb_ret[hr_][1] if hr_ else None,
            'hybrid_spaces': hybrid_spaces,
            'best_rep_coherence': (bc, b_coh, rep_coh[bc][1] if bc else None),
            'best_rep_retrieval': (br, b_ret, rep_ret[br][1] if br else None),
            'hybrid_clustering_wins': bool(h_coh > b_coh),
            'hybrid_retrieval_wins': bool(h_ret < b_ret),
            'metrics_agree': bool((h_coh > b_coh) == (h_ret < b_ret)),
        }
        print(f"\n  Best hybrid slot (held-out): coherence={h_coh:.4f} ({hc}) "
              f"vs best single={b_coh:.4f} ({bc}); retrieval cost="
              f"{h_ret:.4f} ({hr_}) vs best single={b_ret:.4f} ({br})")

    # ── Re-pick blocks (CONTRACT shape) ──
    # Every hybrid slot x cell is a candidate, every single slot x cell a
    # baseline, every PCA control a third arm. Per-fold searches run with
    # pruning off, so no fold is excluded. recipe_selection says the recipes
    # were chosen on each fold's own training rows.
    def _cells(series_by_space, prefix, cell_of):
        out = {}
        for s, by in series_by_space.items():
            if not s.startswith(prefix):
                continue
            for key, fs in by.items():
                if fs and _has_finite(fs):
                    out[f'{s}|{cell_of(key)}'] = [float(v) for v in fs]
        return out

    def _block(cands, bases, ctrls, lower, model_key, **extra):
        return dict({
            'lower_is_better': lower,
            'recipe_selection': 'fold_train_rows',
            'exclude_folds': [],
            'fold0_pruned': False,
            'candidates_by_model': ({model_key: [
                {'name': n, 'fold_scores': v} for n, v in cands.items()]}
                if cands else {}),
            'baselines_by_model': ({model_key: bases} if bases else {}),
            'controls_by_model': ({model_key: ctrls} if ctrls else {}),
        }, **extra)

    _nc = lambda nc: f'nc{nc}'
    results['fold_reselection_clustering'] = _block(
        _cells(clustering_fold, 'hybrid_', _nc),
        _cells(clustering_fold, 'rep_', _nc),
        _cells(clustering_fold, 'pca_', _nc), False, 'kmeans',
        clusterer='kmeans', clusterer_distance='squared_euclidean')

    def _ret_series(cond, only_setting=None):
        return {s: {st: fs for st, fs in conds.get(cond, {}).items()
                    if only_setting is None or st == only_setting}
                for s, conds in retrieval_fold.items()}

    def _ret_cells(prefix, only_setting=None, conds=(RET_PLAIN,)):
        out = {}
        for cond in conds:
            tag = '' if cond == RET_PLAIN else 'importance_weighted|'
            out.update(_cells(_ret_series(cond, only_setting), prefix,
                              lambda st, tag=tag: f'{tag}{st}'))
        return out

    # Hybrids with plain features against single representations, every
    # setting a cell (each side free to pick its setting per fold).
    results['fold_reselection_retrieval_unweighted'] = _block(
        _ret_cells('hybrid_'), _ret_cells('rep_'), _ret_cells('pca_'),
        True, 'knn', hybrid_retrieval_conditions=[RET_PLAIN])
    # Secondary: hybrids also with importance-weighted features, which no
    # single representation gets, so the two sides differ in the weighting as
    # well as the features.
    results['fold_reselection_retrieval'] = _block(
        _ret_cells('hybrid_', conds=(RET_PLAIN, RET_IMPORTANCE)),
        _ret_cells('rep_'), _ret_cells('pca_'), True, 'knn',
        hybrid_retrieval_conditions=[RET_PLAIN, RET_IMPORTANCE])

    if matched_on:
        matched = {}
        for dname, method in matched_methods.items():
            if method is None:
                matched[dname] = {'skipped': (
                    f'no clusterer minimises summed {dname} distance to a '
                    f'centre')}
                continue
            mfold = {s: {nc: _series(lambda r, s=s, d=dname, nc=nc: (
                r.get('clustering_matched', {}).get(s, {}).get(d) or {}).get(nc))
                for nc in matched_clusters} for s in spaces}
            matched[dname] = _block(
                _cells(mfold, 'hybrid_', _nc), _cells(mfold, 'rep_', _nc),
                _cells(mfold, 'pca_', _nc), False, method,
                clusterer=method, clusterer_distance=dname, n_init=_N_INIT)
        results['fold_reselection_clustering_matched'] = matched
    else:
        results['clustering_matched_note'] = (
            'matched arm off; set KIRBY_S13_MATCHED_CLUSTERING=1 to also '
            'cluster every space with the clusterer matching each decided '
            f'distance {matched_methods}')

    def _repick(block, label):
        """fold_reselection against the singles and against the PCA controls."""
        out = {}
        if not block.get('candidates_by_model'):
            return out
        for tag, other in (('', block['baselines_by_model']),
                           ('_vs_pca', block.get('controls_by_model') or {})):
            if not other:
                continue
            try:
                out[f'result{tag}'] = fold_reselection(
                    block['candidates_by_model'], other,
                    block['lower_is_better'],
                    exclude_folds=hr_exclude_folds(block),
                    recipe_selection=block['recipe_selection'])
            except Exception as e:
                print(f"  [warn] fold_reselection ({label}{tag}) failed: "
                      f"{type(e).__name__}: {e}")
        return out

    blocks = [('clustering', results['fold_reselection_clustering']),
              ('retrieval_unweighted',
               results['fold_reselection_retrieval_unweighted']),
              ('retrieval', results['fold_reselection_retrieval'])]
    for dname, b in (results.get('fold_reselection_clustering_matched')
                     or {}).items():
        if 'candidates_by_model' in b:
            blocks.append((f'clustering_matched_{dname}', b))
    for axis, block in blocks:
        rr = _repick(block, axis)
        if 'result' in rr:
            results[f'fold_reselection_{axis}_result'] = rr['result']
        if 'result_vs_pca' in rr:
            results[f'fold_reselection_{axis}_vs_pca_result'] = rr['result_vs_pca']

    # ── Re-picks with the cell held fixed on both sides ──
    # Clustering per cluster count; retrieval per kNN setting (distance,
    # neighbour count, counting scheme). Both arms are read at the same
    # granularity / the same setting.
    def _fixed(block, suffix, label):
        mk = next(iter(block.get('candidates_by_model') or {}), None)
        if mk is None:
            return None
        c = [x for x in block['candidates_by_model'][mk]
             if x['name'].endswith(suffix)]
        b = {k: v for k, v in (block['baselines_by_model'].get(mk) or {}).items()
             if k.endswith(suffix)}
        p = {k: v for k, v in
             ((block.get('controls_by_model') or {}).get(mk) or {}).items()
             if k.endswith(suffix)}
        if not c or not b:
            return None
        try:
            hr = fold_reselection({mk: c}, {mk: b}, block['lower_is_better'],
                                  exclude_folds=hr_exclude_folds(block),
                                  recipe_selection=block['recipe_selection'])
            r = hr.get(mk)
            if r is None:
                return None
            out = {k: r.get(k) for k in (
                'delta', 'hybrid_reselected', 'best_single_reselected',
                'n_paired_folds', 'per_fold_delta', 'hybrid_fold_scores',
                'baseline_fold_scores', 'hybrid_winners_per_fold',
                'baseline_winners_per_fold')}
            if p:
                hp = fold_reselection({mk: c}, {mk: p}, block['lower_is_better'],
                                      exclude_folds=hr_exclude_folds(block),
                                      recipe_selection=block['recipe_selection'])
                rp = hp.get(mk)
                if rp is not None:
                    out['vs_pca'] = {k: rp.get(k) for k in (
                        'delta', 'hybrid_reselected', 'best_single_reselected',
                        'n_paired_folds', 'per_fold_delta')}
            return out
        except Exception as e:
            print(f"  [warn] fixed-cell re-pick ({label}) failed: "
                  f"{type(e).__name__}: {e}")
            return None

    per_count = {}
    clu_blocks = [('clustering', results['fold_reselection_clustering'])]
    for dname, b in (results.get('fold_reselection_clustering_matched')
                     or {}).items():
        if 'candidates_by_model' in b:
            clu_blocks.append((f'clustering_matched_{dname}', b))
    for axis, block in clu_blocks:
        out = {}
        for nc in n_clusters_list:
            r = _fixed(block, f'|nc{nc}', f'{axis} nc={nc}')
            if r is not None:
                out[int(nc)] = r
        if out:
            per_count[axis] = out
            print(f"  Held-out coherence delta by cluster count ({axis}; "
                  f"higher is better; positive = hybrid won):")
            for nc in sorted(out):
                e = out[nc]
                print(f"    {nc:>3} clusters: delta={e['delta']:+.4f} "
                      f"(hybrid {e['hybrid_reselected']:.4f}, best single "
                      f"representation {e['best_single_reselected']:.4f}, "
                      f"{e['n_paired_folds']} paired folds)")
    if per_count:
        results['clustering_delta_by_cluster_count'] = per_count

    per_setting = {}
    blk = results['fold_reselection_retrieval_unweighted']
    for st in settings:
        r = _fixed(blk, f'|{st}', f'retrieval {st}')
        if r is not None:
            per_setting[st] = dict(r, **results['retrieval_settings'][st])
    if per_setting:
        results['retrieval_delta_by_setting'] = per_setting
        print("  Held-out retrieval-cost delta by kNN setting (distance, "
              "neighbours, counting; positive = hybrid's neighbours closer in "
              "label):")
        for st, e in per_setting.items():
            print(f"    {st:<32} delta={e['delta']:+.4f} (hybrid "
                  f"{e['hybrid_reselected']:.4f}, best single representation "
                  f"{e['best_single_reselected']:.4f}, {e['n_paired_folds']} "
                  f"paired folds)")

    # Which numbers are the headline. Clustering (owner delegated,
    # 2026-10-05): hybrid minus best single coherence at each cluster count
    # separately. Retrieval: hybrid (plain features) minus best single
    # representation at each kNN setting separately, both sides at the same
    # distance, neighbour count and counting scheme (extends the delegated
    # unweighted-only headline to the swept settings, 2026-10-05). The blocks
    # where each side picks its own cell per fold and the summary stay as
    # secondary numbers.
    results['headline'] = {
        'clustering': 'clustering_delta_by_cluster_count',
        'retrieval': 'retrieval_delta_by_setting',
        'secondary': ['fold_reselection_clustering_result',
                      'fold_reselection_retrieval_unweighted_result',
                      'fold_reselection_retrieval_result', 'summary'],
        'set_by': 'owner delegated, 2026-10-05; per kNN setting from the '
                  'chat-4 alignment, 2026-10-05',
    }
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S13: Clustering & Retrieval')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's13_clustering_retrieval', args.dataset)
