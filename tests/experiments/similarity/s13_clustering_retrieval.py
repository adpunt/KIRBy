#!/usr/bin/env python3
"""S13: Clustering Quality & Retrieval Precision.

Paper: Similarity
If hybrid distance is a better metric, it should produce better clusters
(members have similar labels) and better retrieval (query -> similar-activity
neighbors). Tests this across per-rep, PCA-matched, and hybrid spaces.

Two sub-analyses:
  13a. Clustering: k-means label coherence (regression: 1 - within-cluster
       std(y)/std(y); classification: mean cluster majority-class purity).
       Higher = better.
  13b. Retrieval: mean |y_query - y_neighbor| (regression) or 1 - precision@k
       (classification) for k-NN retrieval. Lower = better (a retrieval COST).

HONEST-COMPARISON REDESIGN (2026-07-11, user-approved decision B).
The previous version computed both statistics on the FULL dataset and, for the
hybrid space, selected its features using ALL labels (a self-noted tautology).
Both are now computed UNDER CROSS-VALIDATION so the hybrid gets no free look at
the points it is scored on:
  * folds are group-aware (nested_cv_outer_folds, no random splits);
  * the hybrid RECIPES — allocation method, importance method, internal
    evaluator, budget, augmentation strategy — are READ FROM THE POOL, ranked
    once per held-out fold (2026-09-12 decision). s13 no longer calls
    get_fold_best_search and runs no search of its own. Per fold it takes the
    best single representation and the top three hybrids for its downstream
    model ('knn', distance lineage), ranked on the OTHER folds' stored scores.
    Hybrids built under objective='similarity' are in its subset alongside the
    predictive ones — s13 is the clearest case for them because it fits no
    model at all. The two families' stored scores are in different units and
    are never compared with each other here: each returned recipe becomes its
    own arm and s13 scores every arm itself, with the one statistic each axis
    reports. What the per-fold ranking holds out is that fold's SCORES, not its
    rows — the pool searches ran over every row. (2026-09-04 to 2026-09-12 the
    recipe was instead searched inside each fold on that fold's training rows;
    before that, one recipe was searched over every row by a kNN score close to
    the retrieval cost reported here.);
  * the hybrid's feature selection is done on the TRAIN fold only
    (build_fold_hybrid from that fold's blueprint) and PCA is fit on train
    only;
  * KMeans / the k-NN retrieval index are FIT ON TRAIN and the coherence /
    retrieval statistic is measured on the HELD-OUT fold's points
    (train cluster assignment for held-out points; queries answered by train
    neighbours, so there is no self-neighbour);
  * per-fold held-out scores are averaged across folds.
Both axes ALSO persist the per-fold held-out scores for BOTH arms (hybrid
candidate cells and single-rep baseline cells) in the CONTRACT-shaped blocks
`fold_reselection_clustering` and `fold_reselection_retrieval`. The two axes have
OPPOSITE metric directions (coherence higher-better, retrieval-cost
lower-better), so they cannot share one `fold_reselection` block (that helper
takes a single `lower_is_better`); hence two separate, identically-structured
blocks. A reader feeds each block straight to experiments._shared.fold_reselection.

WHAT CHANGED IN THE RESULT FILE ON 2026-09-12. There is no single 'hybrid' space
any more. `clustering` and `retrieval` hold one space per pool slot, named
`hybrid_<family>_r<rank>` — up to three ranks in each of the predictive and
similarity families, six hybrid spaces at most. Every one of them is a candidate
cell in the honest block. `fold_recipes[fold]` is now a dict of slot -> recipe
instead of one recipe, and it carries each slot's pool score and the metric that
score is in. `summary` keeps its keys and adds which slot supplied its number.
Both honest blocks now say `recipe_selection='other_folds_of_pool_search'` and
set `exclude_folds` to [0] when any pool member filtered its own candidate list
on fold 0's scores.

Empirically (pre-redesign, 2026-06-03) this was a MIXED finding: the hybrid
tended to WIN retrieval but LOSE clustering coherence on the powered antibody
sets. `summary.metrics_agree` flags that divergence.

Usage:
    python s13_clustering_retrieval.py --dataset logd --mode intermediary
    python s13_clustering_retrieval.py --dataset flab_thermo_tresanco --mode intermediary
2026-10-02, the owner's set: the three hybrids per family are now kNN's top 2
from the general searches and its top 1 from the distance searches
(get_rule_hybrids), not the top 3 from the distance searches. Spaces are
'hybrid_<family>_<slot>' (slot general1, general2, distance1). Every slot a
fold's read could not fill is in results['pool_read']['rule_missing'] with the
reason.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, kmeans_plusplus
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    compute_feature_importance, get_domain_metric,
    build_fold_hybrid, get_rule_hybrids,
    fold_reselection,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
    hr_exclude_folds,
    importance_feature_weights,
)
from kirby.similarity.distance_config import get_distance_metric
from kirby.find_best_hybrid import nested_cv_outer_folds

# The seed the folds are built with. The stored pool searches cross-validated
# over the same partition (5 folds, seed 42, group-aware), so this file's fold i
# and the pool's fold i are the same rows.
_FOLD_SEARCH_SEED = 42

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
# Whether to also score every space with the clusterer that honours the
# process-wide distance coefficient. Off by default: it is a second full set of
# cluster fits.
_MATCHED_ON = os.environ.get('KIRBY_S13_MATCHED_CLUSTERING', '') not in ('', '0')
# Which cluster counts the matched arm is scored at. Unset means every count in
# the sweep; a shorter list keeps the second set of cluster fits affordable.
_ENV_MATCHED_CLUSTERS = os.environ.get('KIRBY_S13_MATCHED_CLUSTERS')
MATCHED_N_CLUSTERS = ([int(v) for v in _ENV_MATCHED_CLUSTERS.split(',')
                       if v.strip()] if _ENV_MATCHED_CLUSTERS else None)

# 2026-09-12 decision: s13 stops calling get_fold_best_search and reads the pool
# — best single representation plus the top three hybrids for its downstream
# model, chosen per fold from the other folds' scores. Similarity hybrids are in
# its subset.
POOL_SUBSET = 'predictive_and_similarity'
N_HYBRIDS_PER_FAMILY = 3
# THE OWNER'S SET (top 2 general + top 1 distance per model per family,
# 2026-10-02): the three per family are kNN's top 2 recipes from the general
# searches and its top 1 from the distance searches (get_rule_hybrids). They
# used to be the top 3 from the distance searches. Space names are
# 'hybrid_<family>_<slot>', slot general1, general2 or distance1.
RULE_SLOTS = ('general1', 'general2', 'distance1')


# =========================================================================
# LOCAL HELD-OUT STATISTICS (unsupervised — the shared fold_reselection helper
# does the honest ARITHMETIC on per-fold scores, but these two statistics are
# not model-prediction metrics, so the held-out computation lives here.)
# =========================================================================

def _scale_train_val(X_tr, X_va):
    """Fit StandardScaler on TRAIN, transform both. No peeking at val."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_va)


def _has_finite(scores):
    """True if the list has at least one finite (non-NaN/None) entry."""
    return any(s is not None and np.isfinite(s) for s in scores)


def _nanmean(scores):
    """Mean over finite entries; NaN if none are finite."""
    arr = np.asarray([s for s in scores if s is not None], dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float('nan')
    return float(np.nanmean(arr))


def matched_clusterer_name(coefficient=None):
    """Which clusterer honours `coefficient`, or None when none does.

    k-means minimises the sum of SQUARED EUCLIDEAN distances to a centroid and
    takes no metric argument. The retrieval half of this file reads the
    process-wide coefficient, manhattan by default since 2026-09-04. So under
    the shipped setting the two halves of this file measure distance two
    different ways, and `summary.metrics_agree` compares their win booleans
    against each other.

    Three of the four coefficients the suite sweeps have an exact same-shape
    algorithm, each one 'assign to the nearest centre, move the centre to the
    point that minimises summed distance inside its cluster':

      euclidean        k-means itself. The centre is the mean.
      manhattan        k-medians. The centre is the per-feature median, which
                       is the point minimising summed L1 distance.
      cosine           k-means on L2-normalised rows. On unit vectors cosine
                       distance is 1 - x.y and squared Euclidean is 2 - 2 x.y,
                       so the two order every pair identically.
      minkowski p=0.5  none. No closed form for the point minimising summed
                       L_0.5 distance, and the medoid alternative needs every
                       within-cluster pair. Returns None; the caller writes no
                       matched block and says so.
    """
    coef = coefficient or get_distance_metric()
    return {'euclidean': 'kmeans',
            'manhattan': 'kmedians',
            'cosine': 'spherical_kmeans'}.get(coef)


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
    matched_clusterer_name and honour the process-wide distance coefficient.

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


def _heldout_retrieval(X_tr_s, y_tr, X_va_s, y_va, k, is_clf):
    """Fit a k-NN index on TRAIN, query HELD-OUT points, score retrieval COST.

    Regression: mean |y_query - y_neighbor|. Classification: 1 - precision@k
    (fraction of retrieved neighbours whose class differs from the query).
    LOWER = better in both cases. Neighbours come from the TRAIN index only, so
    the query point can never retrieve itself (no self-neighbour to exclude).
    """
    k_actual = min(k, len(y_tr))
    if k_actual < 1:
        return float('nan')
    nn = make_nearest_neighbors(n_neighbors=k_actual)
    nn.fit(X_tr_s)
    _, idx = nn.kneighbors(X_va_s)
    nbr_y = y_tr[idx]
    if is_clf:
        same = (y_va[:, None] == nbr_y).astype(float)
        return float(1.0 - np.mean(same))
    return float(np.mean(np.abs(y_va[:, None] - nbr_y)))


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S13: Clustering quality & retrieval precision (held-out, under CV)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S13: CLUSTERING & RETRIEVAL")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    budget = kwargs.get('budget', 100)
    # How many nearest molecules a retrieval is scored over. Hardcoded at 5
    # until 2026-09-23, so every s13 number on disk is at five neighbours and
    # the axis has never been swept. s05 and s10 both found the neighbour count
    # worth 0.007 to 0.072 in their own statistics, and s06's sweep found it the
    # largest of its four levers, peaking at 150 on caco2 and 15 on adme_mdr1.
    # KIRBY_S13_K_RETRIEVAL sets it; unset still gives 5, so every stored number
    # reproduces.
    k_retrieval = kwargs.get('k_retrieval')
    if k_retrieval is None:
        # .strip() before int(): an exported-but-empty KIRBY_S13_K_RETRIEVAL is
        # a set variable whose value is '', and int('') raises.
        _kr = os.environ.get('KIRBY_S13_K_RETRIEVAL', '').strip()
        k_retrieval = int(_kr) if _kr else 5
    n_clusters_list = kwargs.get('n_clusters_list') or DEFAULT_N_CLUSTERS
    augmentations = data.get('augmentations')
    # The coefficient the retrieval half reads, and the clusterer that would
    # honour it. When the two names differ the clustering half is measuring
    # squared Euclidean distance while the retrieval half is not.
    coefficient = get_distance_metric()
    matched_method = matched_clusterer_name(coefficient)
    matched_on = bool(_MATCHED_ON and matched_method
                      and matched_method != 'kmeans')
    matched_clusters = [nc for nc in (MATCHED_N_CLUSTERS or n_clusters_list)
                        if nc in n_clusters_list]
    print(f"  distance coefficient={coefficient}; clustering half uses k-means "
          f"(squared euclidean); coefficient-matched clusterer="
          f"{matched_method or 'none (undefined for this coefficient)'}; "
          f"matched arm "
          f"{('ON at cluster counts ' + str(matched_clusters)) if matched_on else 'off'}")
    print(f"  cluster counts: {n_clusters_list}; clusterer restarts: {_N_INIT}")

    # ── Group-aware CV folds (shared by every space + both axes) ──
    # nested_cv_outer_folds is the same group-aware split make_cv_splitter built
    # before. The stored pool searches cross-validated over this same partition
    # (5 folds, seed 42), so this file's fold i and the pool's fold i cover the
    # same rows; get_fold_pool_recipes reports 'n_folds_in_pool' and warns when
    # the two counts disagree.
    sample_X = next(iter(reps.values()))
    hybrid_error = None
    top3 = None
    try:
        folds = nested_cv_outer_folds(reps, labels, groups, n_folds,
                                      _FOLD_SEARCH_SEED, task_type)
    except Exception as e:
        # No random-split fallback (feedback_no_random_splits.md). Bail out
        # cleanly rather than crash the batch.
        print(f"  CV split failed: {type(e).__name__}: {e}")
        return {'config': {'budget': budget, 'k_retrieval': k_retrieval,
                           'n_clusters_list': n_clusters_list,
                           'n_folds': n_folds},
                'cv_error': f"{type(e).__name__}: {e}"}

    # ── Per-fold pool read (2026-09-12 decision) ──
    # s13 stops calling get_fold_best_search. It reads the pool: the best single
    # representation plus the top three hybrids for its downstream model ('knn',
    # distance lineage), chosen for each fold from the OTHER folds' stored
    # scores. Similarity hybrids are in its subset. Nothing is searched here —
    # the searches already on disk are read and their stored per-fold scores
    # ranked, so fold i's recipes were chosen without fold i's scores.
    print(f"  Reading the pool per fold "
          f"(subset={POOL_SUBSET}, top 2 general + top 1 distance per "
          f"family, model=knn)...")
    _pool_cache = {}
    _pool_warnings_seen = set()
    # True as soon as any pool member filtered its own candidate list on fold 0's
    # scores. It decides which folds the per-fold reselection drops from BOTH arms.
    pool_fold0_pruned = None
    pool_read = {
        'pool_subset': POOL_SUBSET,
        'n_hybrids_per_family': N_HYBRIDS_PER_FAMILY,
        'downstream_model': 'knn',
        'budget': None,
        'selection_scope': None,
        'selection_note': None,
        'n_folds_in_pool': None,
        'objectives': None,
        'warnings': [],
        'errors': {},
        # Every slot of the owner's set a fold's read could not fill, and why.
        'rule_missing': {},
        'rule': 'top 2 general + top 1 distance per family',
    }

    def _fold_pool(fold_i):
        """The pool for one held-out fold, or None when the read threw.

        budget=None: every budget the stored searches scored for kNN is
        eligible, which is what this file asked for before the change.
        """
        nonlocal pool_fold0_pruned
        if fold_i in _pool_cache:
            return _pool_cache[fold_i]
        try:
            # The owner's set (2026-10-02), ranked on the folds other than
            # fold_i.
            pool = get_rule_hybrids(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(folds),
                downstream_model='knn', budget=None,
                dataset_name=dataset_name, augmentations=augmentations,
                data=data,
                metric=metric_name, task_type=task_type, domain=domain,
                pool_subset=POOL_SUBSET)
        except Exception as e:
            print(f"  [warn] pool read failed (fold {fold_i}): "
                  f"{type(e).__name__}: {e}")
            pool_read['errors'][f"fold{fold_i}"] = f"{type(e).__name__}: {e}"
            pool_read['rule_missing'][f"fold{fold_i}"] = [
                f"the read raised {type(e).__name__}: {e}"]
            _pool_cache[fold_i] = None
            return None
        # Whether the stored search filtered its candidate list on fold 0's
        # scores (2026-10-05). get_rule_hybrids passes this through: True or
        # False from the searches it read (get_fold_pool_recipes already counts
        # a search pickled before the flag existed as pruned), and None only
        # when neither of its two reads returned anything. Until this date the
        # comment here said it was not passed through and a None was read as
        # pruned, so every block excluded fold 0. Now only a real True marks
        # fold 0 pruned; a None is recorded and decides nothing.
        _p = pool.get('fold0_pruned')
        pool_read.setdefault('fold0_pruned_by_fold', {})[str(fold_i)] = (
            None if _p is None else bool(_p))
        if _p is not None:
            pool_fold0_pruned = (bool(_p) if pool_fold0_pruned is None
                                 else (pool_fold0_pruned or bool(_p)))
        pool_read['selection_scope'] = 'other_folds_of_pool_search'
        pool_read['selection_note'] = (
            "each recipe ranked by its mean over the pool search's other "
            "folds; the searches ran over every row, so this holds out the "
            "fold's SCORES, not its rows")
        # One score per pool fold on every entry.
        _nf = {len(h['fold_scores']) for h in pool.get('hybrids') or []
               if h.get('fold_scores')}
        pool_read['n_folds_in_pool'] = (max(_nf) if _nf else None)
        pool_read['objectives'] = list(pool.get('objectives') or [])
        if pool.get('missing'):
            pool_read['rule_missing'][f"fold{fold_i}"] = list(pool['missing'])
        for w in (pool.get('warnings') or []):
            if w not in _pool_warnings_seen:
                _pool_warnings_seen.add(w)
                pool_read['warnings'].append(w)
                print(f"  [pool fold {fold_i}] {w}")
        _pool_cache[fold_i] = pool
        return pool

    # Fold 0's pool, reported only for what it ranked there: its base
    # representations were ranked on folds 1..4, not on the whole dataset.
    pool_first_fold = {}
    pool0 = _fold_pool(0)
    if pool0 is None:
        hybrid_error = pool_read['errors'].get('fold0', 'pool read failed')
    else:
        top3 = []
        for _obj, _fam in (pool0.get('by_objective') or {}).items():
            _ranked = (_fam.get('base_reps') or [])[:3]
            if _ranked:
                print(f"  Top-3 base reps for kNN, fold 0, {_obj} family "
                      f"(ranked on the other folds, "
                      f"{_fam.get('metric_name')}): "
                      + ', '.join(f"{b['rep']}={b['score_other_folds']:.4f}"
                                  for b in _ranked))
                top3.extend([(b['rep'], b['score_other_folds'])
                             for b in _ranked])
            for _h in (_fam.get('hybrids') or []):
                print(f"  Fold 0 kNN hybrid, {_obj} family {_h['slot']}: "
                      f"alloc={_h['allocation_method']}, "
                      f"imp={_h['importance_method']}, "
                      f"ev={_h['internal_evaluator']}, "
                      f"budget={_h['budget']} "
                      f"({_h['metric_name']}={_h['score_other_folds']:.4f} "
                      f"on {_h['n_other_folds']} other folds)")
            pool_first_fold[_obj] = {
                'metric_name': _fam.get('metric_name'),
                'lower_is_better': _fam.get('lower_is_better'),
                'missing': _fam.get('missing'),
                'base_reps_top2': [(b['rep'], b['score_other_folds'])
                                   for b in _ranked],
                'hybrids': [{'rank': h['rank'], 'slot': h['slot'],
                             'tier': h['tier'], 'name': h['name'],
                             'recipe': h['recipe'],
                             'metric_name': h['metric_name'],
                             'score_other_folds': h['score_other_folds']}
                            for h in (_fam.get('hybrids') or [])],
            }
        if not (pool0.get('hybrids') or []):
            hybrid_error = ('the pool held no kNN hybrid configuration for '
                            f'fold 0 under subset {POOL_SUBSET}')

    sorted_reps = sorted(reps.keys())
    sorted_augs = sorted(augmentations.keys()) if augmentations else []

    # Per-fold held-out score accumulators, keyed BY FOLD INDEX.
    #   clustering_fold[space][n_clusters][fold_i] = held-out coherence
    #   retrieval_fold[space]['unweighted'/'weighted'][fold_i] = held-out cost
    # 2026-09-12: these were lists appended to positionally. Under the decision a
    # space is one (family, rank) slot of the pool, and a slot can be missing on
    # one fold and present on the next, which would have put every later score at
    # the wrong fold index. Keyed by fold index and padded to NaN below, a
    # missing fold stays a missing fold.
    clustering_fold = {}
    retrieval_fold = {}
    # The same coherence under the coefficient-matched clusterer, when that arm
    # is on. Same shape, so the same aggregation and the same block shape apply.
    clustering_matched_fold = {}
    # The pooled form of the coherence, each cluster weighted by how many
    # held-out rows it holds instead of every cluster counting once.
    clustering_pooled_fold = {}

    def _acc_clu(space, nc, fold_i, val):
        clustering_fold.setdefault(space, {}).setdefault(nc, {})[fold_i] = val

    def _acc_clu_m(space, nc, fold_i, val):
        clustering_matched_fold.setdefault(space, {}).setdefault(
            nc, {})[fold_i] = val

    def _acc_clu_p(space, nc, fold_i, val):
        clustering_pooled_fold.setdefault(space, {}).setdefault(
            nc, {})[fold_i] = val

    def _acc_ret(space, cond, fold_i, val):
        retrieval_fold.setdefault(
            space, {'unweighted': {}, 'weighted': {}})[cond][fold_i] = val

    def _fold_series(by_fold, n):
        """Fold-index-aligned list, NaN where this fold produced no score."""
        return [(by_fold[i] if (i in by_fold and by_fold[i] is not None)
                 else float('nan')) for i in range(n)]

    # Which recipe each fold's slot held. A slot is (family, rank), and fold i's
    # rank-1 predictive hybrid can be a different recipe from fold j's.
    fold_recipes = {}
    # Which single representation the pool put first for each fold, and in which
    # metric. s13 scores every representation itself; this records the pool's
    # own per-fold pick alongside them.
    pool_base_rep = {}
    # pca_control_dims[space][fold_i] = how many components that control was
    # fitted with. One control per hybrid slot, plus the fixed-budget one.
    pca_control_dims = {}
    # pca_control_of[control] = the hybrid slot it controls for, and
    # pca_control_slot_width[control][fold_i] = that slot's feature count on
    # that fold. A reader checks the match by comparing the two.
    pca_control_of = {}
    pca_control_slot_width = {}
    # clusters_dropped[space][n_clusters][fold_i] = clusters that received fewer
    # than 2 held-out rows and so contributed no coherence term.
    clusters_dropped = {}

    # Each fold's scores are written to
    # s13_clustering_retrieval/<dataset>.partial.jsonl when the fold finishes
    # and read back on the next attempt (2026-09-25; _checkpoint.py). What is
    # stored is that fold's entry in every per-fold dict below (each is keyed
    # by fold index), merged back in fold order so the dicts come out with the
    # same keys in the same order as an uninterrupted run. KMeans, PCA and the
    # importance fits are seeded and nothing draws from a state shared between
    # folds. A fold in which a build, a scale, a PCA or a retrieval raised is
    # not stored, so it is retried.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's13_clustering_retrieval', dataset_name),
        run_identity('s13_clustering_retrieval', dataset_name, data, config,
                     kwargs=kwargs, budget=budget, k_retrieval=k_retrieval,
                     n_clusters_list=list(n_clusters_list),
                     matched_on=matched_on, n_folds=len(folds),
                     # 2026-10-02: spaces are the owner's set now.
                     hybrid_set='top2_general_top1_distance'))

    def _slice2(d, f):
        return {a: v[f] for a, v in d.items() if f in v}

    def _slice3(d, f):
        out = {}
        for a, bm in d.items():
            row = {b: v[f] for b, v in bm.items() if f in v}
            if row:
                out[a] = row
        return out

    def _merge3(d, sl, f):
        for a, row in sl.items():
            for b, val in row.items():
                d.setdefault(a, {}).setdefault(b, {})[f] = val

    print(f"\n  Held-out CV ({len(folds)} folds, group-aware)")
    for fold_i, (tr, va) in enumerate(folds):
        y_tr, y_va = labels[tr], labels[va]
        if _ck.done(fold=fold_i):
            _sv = _ck.get(fold=fold_i)
            _merge3(clustering_fold, _sv['clustering'], fold_i)
            _merge3(clustering_pooled_fold, _sv['clustering_pooled'], fold_i)
            _merge3(clusters_dropped, _sv['clusters_dropped'], fold_i)
            _merge3(clustering_matched_fold, _sv['clustering_matched'], fold_i)
            for _s, _row in _sv['retrieval'].items():
                for _c, _v in _row.items():
                    _acc_ret(_s, _c, fold_i, _v)
            for _s, _v in _sv['pca_control_dims'].items():
                pca_control_dims.setdefault(_s, {})[fold_i] = _v
            for _s, _v in _sv['pca_control_slot_width'].items():
                pca_control_slot_width.setdefault(_s, {})[fold_i] = _v
            pca_control_of.update(_sv['pca_control_of'])
            if _sv['fold_recipes'] is not None:
                fold_recipes[fold_i] = _sv['fold_recipes']
            if _sv['pool_base_rep'] is not None:
                pool_base_rep[fold_i] = _sv['pool_base_rep']
            continue
        _fold_threw = []

        # Assemble this fold's (train, val) matrices for every space.
        fold_spaces = {}
        for rname in sorted_reps:
            fold_spaces[f'rep_{rname}'] = (reps[rname][tr], reps[rname][va])

        # PCA-matched — fit on TRAIN only.
        Xc_tr = Xc_va = None
        try:
            Xc_tr = np.hstack([reps[r][tr] for r in sorted_reps]
                              + [augmentations[a][tr] for a in sorted_augs])
            Xc_va = np.hstack([reps[r][va] for r in sorted_reps]
                              + [augmentations[a][va] for a in sorted_augs])
        except Exception as e:
            _fold_threw.append(1)
            print(f"    fold {fold_i}: PCA input failed: "
                  f"{type(e).__name__}: {e}")

        # PCA spaces already fitted on this fold, keyed by how many components
        # they hold, so two hybrid slots that came out the same width share one
        # fit instead of paying for two.
        _pca_by_dim = {}

        def _pca_space(n_features):
            """A TRAIN-fitted PCA of `n_features` components; (pair, dim)."""
            if Xc_tr is None:
                return None, None
            pca_dim = min(int(n_features), Xc_tr.shape[1], len(tr) - 1)
            if pca_dim < 2:
                return None, None
            if pca_dim in _pca_by_dim:
                return _pca_by_dim[pca_dim], pca_dim
            try:
                sc = StandardScaler()
                pca = PCA(n_components=pca_dim, random_state=42)
                Xp_tr = pca.fit_transform(sc.fit_transform(Xc_tr))
                Xp_va = pca.transform(sc.transform(Xc_va))
            except Exception as e:
                _fold_threw.append(1)
                print(f"    fold {fold_i}: PCA failed ({pca_dim} components): "
                      f"{type(e).__name__}: {e}")
                return None, None
            _pca_by_dim[pca_dim] = (Xp_tr, Xp_va)
            return _pca_by_dim[pca_dim], pca_dim

        def _add_pca_control(space_name, n_features):
            """One PCA control of `n_features` features, fitted on TRAIN only."""
            if space_name in fold_spaces:
                return
            pair, pca_dim = _pca_space(n_features)
            if pair is None:
                return
            fold_spaces[space_name] = pair
            pca_control_dims.setdefault(space_name, {})[fold_i] = pca_dim

        # The fixed-width control, kept as it was.
        _add_pca_control('pca_matched', budget)

        # Hybrids — 2026-09-12: read from the pool, not searched here. One space
        # per (family, rank) slot: 'hybrid_predictive_r1' .. up to three per
        # family. The slot name is stable across folds; the recipe inside it is
        # whatever that fold's ranking on the OTHER folds put there, so it can
        # differ from fold to fold. Every returned recipe is built and scored —
        # the two families' stored scores are in different units, so nothing is
        # merged or dropped between them; s13's own statistic separates them.
        # Features are still selected on TRAIN only and applied to val.
        pool = _fold_pool(fold_i)
        _slot_blueprints = {}
        if pool is not None:
            # The model's top 2 single representations (owner's set); rank 1
            # under the keys it had, rank 2 beside it. s13 scores every
            # representation itself either way.
            _brs = pool.get('base_reps') or []
            if _brs:
                _br = _brs[0]
                pool_base_rep[fold_i] = {
                    'rep': _br['rep'],
                    'objective': _br['objective'],
                    'metric_name': _br['metric_name'],
                    'score_other_folds': _br['score_other_folds'],
                    'n_other_folds': _br.get('n_other_folds'),
                    'rep2': (_brs[1]['rep'] if len(_brs) > 1 else None),
                    'rep2_score_other_folds': (
                        _brs[1]['score_other_folds'] if len(_brs) > 1
                        else None),
                }
            for h in (pool.get('hybrids') or []):
                # One space per (family, slot) of the owner's set.
                sname = f"hybrid_{h['objective']}_{h['slot']}"
                blueprint = h.get('blueprint')
                if blueprint is None:
                    print(f"    fold {fold_i}: pool entry {h['name']} carries "
                          f"no blueprint")
                    continue
                aug_cond = h.get('aug_condition')
                try:
                    _use_augs = sorted_augs if aug_cond != 'none' else []
                    aug_tr = ({a: augmentations[a][tr] for a in _use_augs}
                              if _use_augs else None)
                    aug_va = ({a: augmentations[a][va] for a in _use_augs}
                              if _use_augs else None)
                    Xh_tr, Xh_va, _feat_info = build_fold_hybrid(
                        {r: reps[r][tr] for r in sorted_reps},
                        {r: reps[r][va] for r in sorted_reps},
                        y_tr, blueprint,
                        augmentations_train=aug_tr, augmentations_val=aug_va,
                        groups=(groups[tr] if groups is not None else None))
                    fold_spaces[sname] = (Xh_tr, Xh_va)
                    _slot_blueprints[sname] = blueprint
                    fold_recipes.setdefault(fold_i, {})[sname] = {
                        'allocation': blueprint.allocation_method,
                        'importance': blueprint.importance_method,
                        'evaluator': blueprint.internal_evaluator,
                        'budget': blueprint.budget,
                        # The number of features actually built, beside the
                        # budget (2026-10-05, owner's decision).
                        'width_built': int(np.asarray(Xh_tr).shape[1]),
                        'augmentation_strategy': blueprint.augmentation_strategy,
                        'objective': h['objective'],
                        'rank': h['rank'],
                        'slot': h['slot'],
                        'tier': h.get('tier'),
                        'budget_relaxed': h.get('budget_relaxed'),
                        'recipe': h.get('recipe'),
                        'pool_name': h.get('name'),
                        # The pool's own score of this recipe on the other
                        # folds, in ITS family's metric. The predictive family
                        # stores the dataset's metric and the similarity family
                        # stores a correlation named 'readacross'; neither is
                        # compared with the coherence and retrieval numbers
                        # below, which s13 computes itself.
                        'pool_metric': h.get('metric_name'),
                        'pool_score_other_folds': h.get('score_other_folds'),
                        'pool_n_other_folds': h.get('n_other_folds'),
                    }
                except Exception as e:
                    _fold_threw.append(1)
                    print(f"    fold {fold_i}: hybrid build failed "
                          f"({sname}): {type(e).__name__}: {str(e)[:80]}")

        # ── A PCA control PER HYBRID SLOT, at that slot's own width ────────
        # The fixed 100-feature control above matches a hybrid slot only by
        # coincidence: each slot's width is whatever its recipe chose, and the
        # pool's recipes are searched across [10, 25, 50, 100, 250, 500].
        #
        # 2026-09-20: the controls used to be named by WIDTH — 'pca_matched_25'.
        # A width that only occurred on one fold produced a control scored on
        # one fold while the hybrid it controlled for was scored on five, and
        # the two means were then read side by side. The control is now named
        # by the SLOT it controls for and refitted per fold at that fold's slot
        # width, so control and slot always rest on the same folds. Slots that
        # share a width on a fold share one PCA fit.
        for _sname in list(fold_spaces):
            if _sname not in _slot_blueprints:
                continue
            _w = int(np.asarray(fold_spaces[_sname][0]).shape[1])
            _cname = f'pca_for_{_sname}'
            _pair, _dim = _pca_space(_w)
            if _pair is None:
                continue
            fold_spaces[_cname] = _pair
            pca_control_dims.setdefault(_cname, {})[fold_i] = _dim
            pca_control_slot_width.setdefault(_cname, {})[fold_i] = _w
            pca_control_of[_cname] = _sname

        # Score every space on this held-out fold.
        for sname, (Xtr, Xva) in fold_spaces.items():
            try:
                Xtr_s, Xva_s = _scale_train_val(np.asarray(Xtr, dtype=np.float64),
                                                np.asarray(Xva, dtype=np.float64))
            except Exception as e:
                _fold_threw.append(1)
                print(f"    fold {fold_i}: scale failed ({sname}): "
                      f"{type(e).__name__}: {e}")
                continue

            for nc in n_clusters_list:
                _coh, _dropped, _pooled = _heldout_coherence(
                    Xtr_s, y_tr, Xva_s, y_va, nc, is_clf, method='kmeans')
                _acc_clu(sname, nc, fold_i, _coh)
                _acc_clu_p(sname, nc, fold_i, _pooled)
                clusters_dropped.setdefault(sname, {}).setdefault(
                    nc, {})[fold_i] = _dropped
                if matched_on and nc in matched_clusters:
                    try:
                        _cm, _dm, _ = _heldout_coherence(
                            Xtr_s, y_tr, Xva_s, y_va, nc, is_clf,
                            method=matched_method)
                    except Exception as e:
                        _fold_threw.append(1)
                        print(f"    fold {fold_i}: {matched_method} failed "
                              f"({sname}, nc={nc}): {type(e).__name__}: {e}")
                        _cm = float('nan')
                    _acc_clu_m(sname, nc, fold_i, _cm)

            _acc_ret(sname, 'unweighted', fold_i,
                     _heldout_retrieval(Xtr_s, y_tr, Xva_s, y_va,
                                        k_retrieval, is_clf))

            # Importance-weighted retrieval — hybrid slots only, weights from
            # TRAIN. Each slot uses the importance method its own pool recipe
            # was built with.
            if sname in _slot_blueprints:
                _bp = _slot_blueprints[sname]
                try:
                    imp = compute_feature_importance(
                        {'hybrid': Xtr_s}, y_tr,
                        method=getattr(_bp, 'importance_method',
                                       'lightgbm_gain')).get(
                        'hybrid', np.ones(Xtr_s.shape[1]))
                    # The power the importance is raised to is swept through
                    # KIRBY_IMPORTANCE_EXPONENT; unset gives the 0.5 this line
                    # used to spell out.
                    w = importance_feature_weights(imp)
                    w = w / (w.sum() + 1e-10)
                    _acc_ret(sname, 'weighted', fold_i,
                             _heldout_retrieval(Xtr_s * w, y_tr, Xva_s * w, y_va,
                                                k_retrieval, is_clf))
                except Exception as e:
                    _fold_threw.append(1)
                    print(f"    fold {fold_i}: weighted retrieval failed "
                          f"({sname}): {type(e).__name__}: {e}")
                    _acc_ret(sname, 'weighted', fold_i, float('nan'))

        if not _fold_threw:
            _pcw = _slice2(pca_control_slot_width, fold_i)
            _ck.put({
                'clustering': _slice3(clustering_fold, fold_i),
                'clustering_pooled': _slice3(clustering_pooled_fold, fold_i),
                'clusters_dropped': _slice3(clusters_dropped, fold_i),
                'clustering_matched': _slice3(clustering_matched_fold, fold_i),
                'retrieval': _slice3(retrieval_fold, fold_i),
                'pca_control_dims': _slice2(pca_control_dims, fold_i),
                'pca_control_slot_width': _pcw,
                'pca_control_of': {c: pca_control_of[c] for c in _pcw},
                'fold_recipes': fold_recipes.get(fold_i),
                'pool_base_rep': pool_base_rep.get(fold_i),
            }, fold=fold_i)

    # ── Fold-index-aligned score lists ──
    # 2026-09-12: every list below has one entry per fold, NaN where that fold
    # produced no score for that space. A space that fails on one fold no longer
    # shifts its later folds' scores onto the wrong fold index.
    clustering_fold = {s: {nc: _fold_series(byf, len(folds))
                           for nc, byf in ncmap.items()}
                       for s, ncmap in clustering_fold.items()}
    clustering_matched_fold = {s: {nc: _fold_series(byf, len(folds))
                                   for nc, byf in ncmap.items()}
                               for s, ncmap in clustering_matched_fold.items()}
    clustering_pooled_fold = {s: {nc: _fold_series(byf, len(folds))
                                  for nc, byf in ncmap.items()}
                              for s, ncmap in clustering_pooled_fold.items()}
    retrieval_fold = {s: {c: _fold_series(byf, len(folds))
                          for c, byf in conds.items()}
                      for s, conds in retrieval_fold.items()}

    # ── Aggregate held-out scores (nanmean across folds) ──
    clustering_results = {}
    for sname, ncmap in clustering_fold.items():
        entry = {}
        for nc, fs in ncmap.items():
            _drop = clusters_dropped.get(sname, {}).get(nc, {})
            entry[nc] = {
                'label_coherence': _nanmean(fs),
                'n_folds_used': int(sum(1 for s in fs
                                        if s is not None and np.isfinite(s))),
                # Each cluster weighted by its held-out row count instead of
                # every cluster counting once.
                'label_coherence_pooled': _nanmean(
                    clustering_pooled_fold.get(sname, {}).get(nc, [])),
                # Clusters that carried >=2 held-out rows and so contributed a
                # term, averaged over the folds that scored. Two spaces at one
                # cluster count can be scored on different numbers of terms.
                'mean_clusters_scored': (
                    float(nc - np.mean(list(_drop.values()))) if _drop
                    else float(nc)),
            }
            if matched_on:
                entry[nc]['label_coherence_matched'] = _nanmean(
                    clustering_matched_fold.get(sname, {}).get(nc, []))
        clustering_results[sname] = entry
        best_nc = max(entry, key=lambda k: (entry[k]['label_coherence']
                                            if np.isfinite(entry[k]['label_coherence'])
                                            else -np.inf), default=None)
        if best_nc is not None:
            print(f"    {sname}: best held-out coherence="
                  f"{entry[best_nc]['label_coherence']:.4f} (k={best_nc})")

    retrieval_results = {}
    for sname, conds in retrieval_fold.items():
        uw = _nanmean(conds['unweighted'])
        w = _nanmean(conds['weighted']) if _has_finite(conds['weighted']) else None
        retrieval_results[sname] = {
            'unweighted': {'mean_delta_y': uw},
            'weighted': ({'mean_delta_y': w} if w is not None else None),
        }
        w_str = f", weighted={w:.4f}" if w is not None else ""
        print(f"    {sname}: held-out mean_delta_y={uw:.4f}{w_str}")

    results = {
        'clustering': convert_numpy(clustering_results),
        'retrieval': convert_numpy(retrieval_results),
        'config': {
            'budget': budget,
            'k_retrieval': k_retrieval,
            'n_clusters_list': n_clusters_list,
            'n_folds': len(folds),
            'spaces': sorted(set(list(clustering_results.keys())
                                 + list(retrieval_results.keys()))),
            # The coefficient the retrieval half read, what the clustering half
            # minimised, and the clusterer that would honour the coefficient.
            'distance_coefficient': coefficient,
            'clustering_distance': 'squared_euclidean',
            'matched_clusterer': matched_method,
            'matched_clustering_scored': matched_on,
            'n_init': _N_INIT,
        },
        # How many components each PCA control was fitted with, per fold. One
        # control per distinct hybrid width plus the fixed-width one, so every
        # hybrid slot has a control of its own width to be read against.
        'pca_control_dims': {s: {int(f): int(d) for f, d in sorted(byf.items())}
                             for s, byf in sorted(pca_control_dims.items())},
        # Which hybrid slot each control controls for, and that slot's feature
        # count on each fold. The control is matched when its component count
        # equals the slot's feature count on the same fold.
        'pca_control_of': dict(sorted(pca_control_of.items())),
        'pca_control_slot_width': {
            s: {int(f): int(w) for f, w in sorted(byf.items())}
            for s, byf in sorted(pca_control_slot_width.items())},
        # Clusters that received fewer than 2 held-out rows, per space, per
        # cluster count, per fold. Those contributed no coherence term.
        'clusters_dropped': {
            s: {int(nc): {int(f): int(v) for f, v in sorted(byf.items())}
                for nc, byf in sorted(ncmap.items())}
            for s, ncmap in sorted(clusters_dropped.items())},
    }
    if top3:
        # 2026-09-12: the pool's own ranking for fold 0, one group of three per
        # family, each in that family's metric. It is no longer an RF ranking
        # from a search this file ran.
        results['top3_base_reps_first_fold'] = list(top3)
    if fold_recipes:
        # fold -> {slot: recipe}. One entry per (family, rank) slot the pool
        # filled on that fold; before 2026-09-12 there was one recipe per fold.
        results['fold_recipes'] = {int(fi): dict(r)
                                   for fi, r in sorted(fold_recipes.items())}
    if pool_base_rep:
        results['pool_base_rep_per_fold'] = {
            int(fi): dict(v) for fi, v in sorted(pool_base_rep.items())}
    results['pool_read'] = pool_read
    if pool_first_fold:
        results['pool_first_fold'] = pool_first_fold
    if hybrid_error is not None:
        results['hybrid_error'] = hybrid_error

    # ── Legacy summary (kept for downstream readers) — now on HELD-OUT means ──
    # Top-3 baseline reps per axis, and the hybrid-vs-best-rep win booleans.
    # These still use same-fold selection of the best rep, so they carry mild
    # optimism; the fully-honest number is the fold_reselection block below.
    rep_retrieval = {k: v['unweighted']['mean_delta_y']
                     for k, v in retrieval_results.items()
                     if k.startswith('rep_')
                     and np.isfinite(v['unweighted']['mean_delta_y'])}
    top3_retrieval = sorted(rep_retrieval.items(), key=lambda x: x[1])[:3]
    rep_clustering = {}
    for k, v in clustering_results.items():
        if k.startswith('rep_') and v:
            vals = [val['label_coherence'] for val in v.values()
                    if np.isfinite(val['label_coherence'])]
            if vals:
                rep_clustering[k] = max(vals)
    top3_clustering = sorted(rep_clustering.items(),
                             key=lambda x: x[1], reverse=True)[:3]
    results['top3_retrieval'] = top3_retrieval
    results['top3_clustering'] = top3_clustering

    # 2026-09-12: 'hybrid' was one space. It is now up to six — three pool ranks
    # per family — so the summary below takes the best hybrid slot on each axis
    # and names it. That maximum is read off the same folds it is compared on,
    # the same same-fold optimism the rest of this summary already carries; the
    # honest block re-picks per fold across every slot.
    hybrid_spaces = sorted(s for s in set(clustering_results)
                           | set(retrieval_results) if s.startswith('hybrid_'))
    _coh_by_slot = {}
    for s in hybrid_spaces:
        vals = [v['label_coherence']
                for v in clustering_results.get(s, {}).values()
                if np.isfinite(v['label_coherence'])]
        if vals:
            _coh_by_slot[s] = max(vals)
    _ret_by_slot = {}
    for s in hybrid_spaces:
        v = retrieval_results.get(s, {}).get('unweighted', {}).get('mean_delta_y')
        if v is not None and np.isfinite(v):
            _ret_by_slot[s] = v
    if _coh_by_slot or _ret_by_slot:
        h_coh_space = (max(_coh_by_slot, key=_coh_by_slot.get)
                       if _coh_by_slot else None)
        h_coh = _coh_by_slot.get(h_coh_space, float('nan'))
        h_ret_space = (min(_ret_by_slot, key=_ret_by_slot.get)
                       if _ret_by_slot else None)
        h_ret = _ret_by_slot.get(h_ret_space, float('nan'))
        best_rep_ret_name, best_rep_ret_val = (top3_retrieval[0]
                                               if top3_retrieval else ('?', 1e10))
        best_rep_coh_name, best_rep_coh_val = (top3_clustering[0]
                                               if top3_clustering else ('?', 0))
        results['summary'] = {
            'hybrid_coherence': h_coh,
            'hybrid_retrieval': h_ret,
            'hybrid_coherence_space': h_coh_space,
            'hybrid_retrieval_space': h_ret_space,
            'hybrid_spaces': hybrid_spaces,
            'best_rep_coherence': (best_rep_coh_name, best_rep_coh_val),
            'best_rep_retrieval': (best_rep_ret_name, best_rep_ret_val),
            'hybrid_clustering_wins': bool(h_coh > best_rep_coh_val),
            'hybrid_retrieval_wins': bool(h_ret < best_rep_ret_val),
            'metrics_agree': ((h_coh > best_rep_coh_val)
                              == (h_ret < best_rep_ret_val)),
        }
        print(f"\n  Best hybrid slot (held-out): coherence={h_coh:.4f} "
              f"({h_coh_space}) vs best_rep={best_rep_coh_val:.4f} "
              f"({'win' if h_coh > best_rep_coh_val else 'loss'}), "
              f"retrieval={h_ret:.4f} ({h_ret_space}) vs "
              f"best_rep={best_rep_ret_val:.4f} "
              f"({'win' if h_ret < best_rep_ret_val else 'loss'})")

    # ── Honest leave-one-fold-out re-score blocks (CONTRACT shape) ──
    # Two axes with OPPOSITE directions cannot share one fold_reselection block
    # (the helper takes a single lower_is_better), so each axis gets its own
    # identically-structured block. Each hybrid n_clusters / retrieval condition
    # is a candidate CELL and each single-rep (space x n_clusters / space) is a
    # baseline CELL, so fold_reselection re-picks the winning cell per held-out
    # fold instead of trusting the same-fold argmax. Single grouping model per
    # axis by design ('kmeans' / 'knn'). A reader feeds each block straight to
    # experiments._shared.fold_reselection.
    #
    # 2026-09-12: the candidate cells are no longer one searched recipe. Each
    # (family, rank) slot the pool filled is a cell, at each cluster count, so a
    # subset holding both families puts three predictive and three similarity
    # hybrids in as rival cells. They are comparable HERE because every cell's
    # fold scores are the coherence (or the retrieval cost) s13 measured itself;
    # the families' own stored scores, which are in two different units, are
    # kept out of this block and sit in fold_recipes.
    clu_candidates = []
    for sname in hybrid_spaces:
        for nc in n_clusters_list:
            fs = clustering_fold.get(sname, {}).get(nc)
            if fs and _has_finite(fs):
                clu_candidates.append({'name': f'{sname}|nc{nc}',
                                       'fold_scores': [float(s) for s in fs]})
    clu_baselines = {}
    for sname, ncmap in clustering_fold.items():
        if not sname.startswith('rep_'):
            continue
        for nc, fs in ncmap.items():
            if fs and _has_finite(fs):
                clu_baselines[f'{sname}|nc{nc}'] = [float(s) for s in fs]
    # The PCA controls, as a THIRD arm rather than a baseline. They are not
    # single representations, so putting them in clu_baselines would change what
    # 'best single representation' means; kept separately, a reader can re-pick
    # the hybrid against its own-width control fold by fold. Before 2026-09-20
    # these per-fold scores were not written out at all and the control could
    # only be read as a whole-run mean.
    clu_controls = {}
    for sname, ncmap in clustering_fold.items():
        if not sname.startswith('pca_'):
            continue
        for nc, fs in ncmap.items():
            if fs and _has_finite(fs):
                clu_controls[f'{sname}|nc{nc}'] = [float(s) for s in fs]

    ret_candidates = []
    for sname in hybrid_spaces:
        h_ret_fold = retrieval_fold.get(sname)
        if not h_ret_fold:
            continue
        if _has_finite(h_ret_fold['unweighted']):
            ret_candidates.append({
                'name': f'{sname}|unweighted_k{k_retrieval}',
                'fold_scores': [float(s) for s in h_ret_fold['unweighted']]})
        if _has_finite(h_ret_fold['weighted']):
            ret_candidates.append({
                'name': f'{sname}|weighted_k{k_retrieval}',
                'fold_scores': [float(s) for s in h_ret_fold['weighted']]})
    ret_baselines = {}
    ret_controls = {}
    for sname, conds in retrieval_fold.items():
        if _has_finite(conds['unweighted']):
            if sname.startswith('rep_'):
                ret_baselines[f'{sname}|k{k_retrieval}'] = [
                    float(s) for s in conds['unweighted']]
            elif sname.startswith('pca_'):
                ret_controls[f'{sname}|k{k_retrieval}'] = [
                    float(s) for s in conds['unweighted']]

    # Coherence is higher-better; retrieval COST is lower-better (both regression
    # |dy| and classification 1 - precision@k).
    #
    # 2026-09-12, both blocks: 'recipe_selection' now says which rows chose the
    # recipes — the pool search's other folds. That search ran over every row, so
    # what fold i's ranking held out is fold i's SCORES, not fold i's rows.
    # 'exclude_folds' is [0] when any pool member filtered its own candidate list
    # on fold 0's scores: fold 0's hybrid cells would otherwise be read off
    # recipes fold 0 helped keep, while the single representations passed through
    # no such filter, and the two arms would not pay the same held-out tax. A
    # search pickled before the flag existed reads as pruned inside the pool
    # read, which is the reading hr_exclude_folds takes. pool_fold0_pruned is
    # None only when no fold's read returned a flag at all (2026-10-05): then
    # no pool hybrid was read, nothing was filtered on fold 0, and no fold is
    # excluded.
    _exclude = [0] if pool_fold0_pruned else []
    results['fold_reselection_clustering'] = {
        'lower_is_better': False,
        'recipe_selection': 'other_folds_of_pool_search',
        'exclude_folds': _exclude,
        'fold0_pruned': pool_fold0_pruned,
        'pool_subset': POOL_SUBSET,
        'candidates_by_model': ({'kmeans': clu_candidates}
                                if clu_candidates else {}),
        'baselines_by_model': ({'kmeans': clu_baselines}
                               if clu_baselines else {}),
        'controls_by_model': ({'kmeans': clu_controls}
                              if clu_controls else {}),
        # What the k-means in this block minimised, and what the retrieval block
        # below measured. When the two names differ the file's two halves are
        # not answering the same question about distance.
        'clusterer': 'kmeans',
        'clusterer_distance': 'squared_euclidean',
        'distance_coefficient': coefficient,
    }
    results['fold_reselection_retrieval'] = {
        'lower_is_better': True,
        'recipe_selection': 'other_folds_of_pool_search',
        'exclude_folds': _exclude,
        'fold0_pruned': pool_fold0_pruned,
        'pool_subset': POOL_SUBSET,
        'candidates_by_model': ({'knn': ret_candidates}
                                if ret_candidates else {}),
        'baselines_by_model': ({'knn': ret_baselines}
                               if ret_baselines else {}),
        'controls_by_model': ({'knn': ret_controls}
                              if ret_controls else {}),
        'distance_coefficient': coefficient,
    }

    # ── The same clustering block under the coefficient-matched clusterer ──
    # Identical shape, identical cells, one difference: the clusterer honours
    # the process-wide distance coefficient instead of minimising squared
    # Euclidean distance. Reading the two deltas side by side says what the
    # mismatch is worth. Absent when the arm is off, and absent with a stated
    # reason when no clusterer of that shape exists for the coefficient.
    if not matched_on:
        results['clustering_matched_note'] = (
            f"coefficient={coefficient}; "
            + ('the clustering half already matches it (k-means is euclidean)'
               if matched_method == 'kmeans' else
               'no clusterer minimising summed distance under this coefficient '
               'is available, so the clustering half cannot honour it'
               if matched_method is None else
               f'matched arm off; set KIRBY_S13_MATCHED_CLUSTERING=1 to score '
               f'every space with {matched_method} as well'))
    else:
        m_cands, m_bases, m_ctrls = [], {}, {}
        for sname, ncmap in clustering_matched_fold.items():
            for nc, fs in ncmap.items():
                if not (fs and _has_finite(fs)):
                    continue
                cell = f'{sname}|nc{nc}'
                vals = [float(s) for s in fs]
                if sname.startswith('hybrid_'):
                    m_cands.append({'name': cell, 'fold_scores': vals})
                elif sname.startswith('rep_'):
                    m_bases[cell] = vals
                elif sname.startswith('pca_'):
                    m_ctrls[cell] = vals
        results['fold_reselection_clustering_matched'] = {
            'lower_is_better': False,
            'recipe_selection': 'other_folds_of_pool_search',
            'exclude_folds': _exclude,
            'fold0_pruned': pool_fold0_pruned,
            'pool_subset': POOL_SUBSET,
            'candidates_by_model': ({matched_method: m_cands}
                                    if m_cands else {}),
            'baselines_by_model': ({matched_method: m_bases}
                                   if m_bases else {}),
            'controls_by_model': ({matched_method: m_ctrls}
                                  if m_ctrls else {}),
            'clusterer': matched_method,
            'clusterer_distance': coefficient,
            'distance_coefficient': coefficient,
            'n_init': _N_INIT,
        }

    # Reuse the shared pure-arithmetic helper (no ML) to report the honest
    # hybrid-vs-best-single-rep delta inline for each axis.
    _axes = [('clustering', results['fold_reselection_clustering'], 'higher'),
             ('retrieval', results['fold_reselection_retrieval'], 'lower')]
    if 'fold_reselection_clustering_matched' in results:
        _axes.append(('clustering_matched',
                      results['fold_reselection_clustering_matched'], 'higher'))
    for axis, block, better in _axes:
        try:
            hr = fold_reselection(block['candidates_by_model'],
                                block['baselines_by_model'],
                                block['lower_is_better'],
                                exclude_folds=hr_exclude_folds(block),
                                # 2026-09-12: carry which rows chose the recipe
                                # into the inline result too, so it is not read
                                # as a whole-dataset pick.
                                recipe_selection=block['recipe_selection'])
            results[f'fold_reselection_{axis}_result'] = hr
            model_key = next(iter(block['candidates_by_model']), None)
            m = hr.get(model_key) if model_key else None
            if m is not None:
                flag = ' [single-fold, NOT honest]' if m.get('single_fold') else ''
                print(f"  Honest LOFO re-score ({axis}, {better}-better){flag}: "
                      f"hybrid={m['hybrid_reselected']:.4f}, "
                      f"best_single={m['best_single_reselected']:.4f}, "
                      f"delta={m['delta']:+.4f} (positive=hybrid better) "
                      f"on {m.get('n_paired_folds')} paired folds")
        except Exception as e:
            print(f"  [warn] fold_reselection ({axis}) failed: "
                  f"{type(e).__name__}: {e}")

        # The same re-pick against the PCA control of the hybrid's own width,
        # in place of the single representations. Positive still means the
        # hybrid won.
        ctrls = block.get('controls_by_model') or {}
        if ctrls and block['candidates_by_model']:
            try:
                hrc = fold_reselection(block['candidates_by_model'], ctrls,
                                       block['lower_is_better'],
                                       exclude_folds=hr_exclude_folds(block),
                                       recipe_selection=block['recipe_selection'])
                results[f'fold_reselection_{axis}_vs_pca_result'] = hrc
                mk = next(iter(ctrls))
                mc = hrc.get(mk)
                if mc is not None:
                    print(f"  Honest LOFO re-score ({axis}) against the PCA "
                          f"control: hybrid={mc['hybrid_reselected']:.4f}, "
                          f"best_pca={mc['best_single_reselected']:.4f}, "
                          f"delta={mc['delta']:+.4f} on "
                          f"{mc.get('n_paired_folds')} paired folds")
            except Exception as e:
                print(f"  [warn] fold_reselection vs PCA ({axis}) failed: "
                      f"{type(e).__name__}: {e}")

    # ── Per-cluster-count re-picks ─────────────────────────────────────────
    # The block above lets each arm choose its own cluster count per fold, so
    # the hybrid's winning cell and the single representation's can sit at two
    # different counts and the delta subtracts coherence at one granularity from
    # coherence at another. These are the same re-pick with the count held
    # fixed, one entry per count, so the two arms are read at the same
    # granularity.
    per_count = {}
    for axis, block in (('clustering', results['fold_reselection_clustering']),
                        ('clustering_matched',
                         results.get('fold_reselection_clustering_matched'))):
        if not block or not block['candidates_by_model']:
            continue
        mk = next(iter(block['candidates_by_model']))
        out = {}
        for nc in n_clusters_list:
            suffix = f'|nc{nc}'
            c = [x for x in block['candidates_by_model'][mk]
                 if x['name'].endswith(suffix)]
            b = {k: v for k, v in
                 (block['baselines_by_model'].get(mk) or {}).items()
                 if k.endswith(suffix)}
            if not c or not b:
                continue
            try:
                hr = fold_reselection({mk: c}, {mk: b}, False,
                                      exclude_folds=hr_exclude_folds(block),
                                      recipe_selection=block['recipe_selection'])
                r = hr.get(mk)
                if r is not None:
                    out[int(nc)] = {
                        'delta': r['delta'],
                        'hybrid_reselected': r['hybrid_reselected'],
                        'best_single_reselected': r['best_single_reselected'],
                        'n_paired_folds': r.get('n_paired_folds'),
                        'hybrid_winners_per_fold': r.get('hybrid_winners_per_fold'),
                        'baseline_winners_per_fold': r.get('baseline_winners_per_fold'),
                    }
            except Exception as e:
                print(f"  [warn] per-count re-pick ({axis}, nc={nc}) failed: "
                      f"{type(e).__name__}: {e}")
        if out:
            per_count[axis] = out
            print(f"  Held-out delta by cluster count ({axis}, coherence is "
                  f"unitless and higher-better; positive=hybrid won):")
            for nc in sorted(out):
                e = out[nc]
                print(f"    {nc:>3} clusters: delta={e['delta']:+.4f} "
                      f"(hybrid {e['hybrid_reselected']:.4f}, best single "
                      f"representation {e['best_single_reselected']:.4f}, "
                      f"{e['n_paired_folds']} paired folds)")
    if per_count:
        results['clustering_delta_by_cluster_count'] = per_count

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S13: Clustering & Retrieval')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's13_clustering_retrieval', args.dataset)
