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
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    compute_feature_importance, get_domain_metric,
    build_fold_hybrid, get_fold_pool_recipes,
    fold_reselection,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
    hr_exclude_folds,
    importance_feature_weights,
)
from kirby.find_best_hybrid import nested_cv_outer_folds

# The seed the folds are built with. The stored pool searches cross-validated
# over the same partition (5 folds, seed 42, group-aware), so this file's fold i
# and the pool's fold i are the same rows.
_FOLD_SEARCH_SEED = 42

# 2026-09-12 decision: s13 stops calling get_fold_best_search and reads the pool
# — best single representation plus the top three hybrids for its downstream
# model, chosen per fold from the other folds' scores. Similarity hybrids are in
# its subset.
POOL_SUBSET = 'predictive_and_similarity'
N_HYBRIDS_PER_FAMILY = 3


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


def _heldout_coherence(X_tr_s, y_tr, X_va_s, y_va, n_clusters, is_clf):
    """Fit KMeans on TRAIN, assign HELD-OUT points, score their label coherence.

    Regression: 1 - mean(within-cluster std(y_val)) / std(y_val_global).
    Classification: mean majority-class fraction among held-out points per
    cluster. Higher = the train-learned cluster geometry organises unseen
    labels well. Returns NaN when the fold cannot support the statistic (too few
    train points for n_clusters, degenerate labels, no cluster with >=2 held-out
    points, etc.) — the fold_reselection helper skips NaN folds.

    Both branches need at least 2 held-out rows in a cluster. A cluster holding
    one row has purity 1.0 by construction, so admitting singletons rewarded a
    space for scattering held-out points into clusters of one while the
    regression branch, which always required 2, did not.

    Returns (score, n_clusters_dropped): how many of the n_clusters received
    fewer than 2 held-out rows and so contributed no term.
    """
    if n_clusters >= len(y_tr):
        return float('nan'), 0
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_tr_s)
    va_lab = km.predict(X_va_s)

    if is_clf:
        if (y_va < 0).any():
            return float('nan'), 0
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
            return float('nan'), n_dropped
        return float(np.mean(purities)), n_dropped

    y_std = np.std(y_va)
    if y_std < 1e-10:
        return float('nan'), 0
    stds = []
    n_dropped = 0
    for c in range(n_clusters):
        mask = va_lab == c
        if mask.sum() >= 2:
            stds.append(np.std(y_va[mask]))
        else:
            n_dropped += 1
    if not stds:
        return float('nan'), n_dropped
    return float(1.0 - np.mean(stds) / y_std), n_dropped


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
    k_retrieval = kwargs.get('k_retrieval', 5)
    n_clusters_list = kwargs.get('n_clusters_list', [3, 5, 10])
    augmentations = data.get('augmentations')

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
          f"(subset={POOL_SUBSET}, {N_HYBRIDS_PER_FAMILY} hybrids per family, "
          f"model=knn, evaluators={DISTANCE_EVALUATORS}, "
          f"budgets={DISTANCE_BUDGETS})...")
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
            pool = get_fold_pool_recipes(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(folds),
                downstream_model='knn', budget=None,
                dataset_name=dataset_name, augmentations=augmentations,
                distance=True, data=data,
                metric=metric_name, task_type=task_type, domain=domain,
                pool_subset=POOL_SUBSET,
                n_hybrids=N_HYBRIDS_PER_FAMILY)
        except Exception as e:
            print(f"  [warn] pool read failed (fold {fold_i}): "
                  f"{type(e).__name__}: {e}")
            pool_read['errors'][f"fold{fold_i}"] = f"{type(e).__name__}: {e}"
            _pool_cache[fold_i] = None
            return None
        _p = pool.get('fold0_pruned')
        _p = True if _p is None else bool(_p)
        pool_fold0_pruned = (_p if pool_fold0_pruned is None
                             else (pool_fold0_pruned or _p))
        pool_read['selection_scope'] = pool.get('selection_scope')
        pool_read['selection_note'] = pool.get('selection_note')
        pool_read['n_folds_in_pool'] = pool.get('n_folds_in_pool')
        pool_read['objectives'] = list(pool.get('objectives') or [])
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
            _ranked = (_fam.get('base_reps_ranked') or [])[:3]
            if _ranked:
                print(f"  Top-3 base reps for kNN, fold 0, {_obj} family "
                      f"(ranked on the other folds, "
                      f"{_fam.get('metric_name')}): "
                      + ', '.join(f"{b['rep']}={b['score_other_folds']:.4f}"
                                  for b in _ranked))
                top3.extend([(b['rep'], b['score_other_folds'])
                             for b in _ranked])
            for _h in (_fam.get('hybrids') or []):
                print(f"  Fold 0 kNN hybrid, {_obj} family rank {_h['rank']}: "
                      f"alloc={_h['allocation_method']}, "
                      f"imp={_h['importance_method']}, "
                      f"ev={_h['internal_evaluator']}, "
                      f"budget={_h['budget']} "
                      f"({_h['metric_name']}={_h['score_other_folds']:.4f} "
                      f"on {_h['n_other_folds']} other folds)")
            pool_first_fold[_obj] = {
                'metric_name': _fam.get('metric_name'),
                'lower_is_better': _fam.get('lower_is_better'),
                'n_hybrid_candidates': _fam.get('n_hybrid_candidates'),
                'n_base_reps': _fam.get('n_base_reps'),
                'base_reps_top3': [(b['rep'], b['score_other_folds'])
                                   for b in _ranked],
                'hybrids': [{'rank': h['rank'], 'name': h['name'],
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

    def _acc_clu(space, nc, fold_i, val):
        clustering_fold.setdefault(space, {}).setdefault(nc, {})[fold_i] = val

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
    # fitted with. One control per distinct hybrid width, plus the fixed one.
    pca_control_dims = {}
    # clusters_dropped[space][n_clusters][fold_i] = clusters that received fewer
    # than 2 held-out rows and so contributed no coherence term.
    clusters_dropped = {}

    print(f"\n  Held-out CV ({len(folds)} folds, group-aware)")
    for fold_i, (tr, va) in enumerate(folds):
        y_tr, y_va = labels[tr], labels[va]

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
            print(f"    fold {fold_i}: PCA input failed: "
                  f"{type(e).__name__}: {e}")

        def _add_pca_control(space_name, n_features):
            """One PCA control of `n_features` features, fitted on TRAIN only."""
            if Xc_tr is None or space_name in fold_spaces:
                return
            pca_dim = min(int(n_features), Xc_tr.shape[1], len(tr) - 1)
            if pca_dim < 2:
                return
            try:
                sc = StandardScaler()
                pca = PCA(n_components=pca_dim, random_state=42)
                Xp_tr = pca.fit_transform(sc.fit_transform(Xc_tr))
                Xp_va = pca.transform(sc.transform(Xc_va))
                fold_spaces[space_name] = (Xp_tr, Xp_va)
                pca_control_dims.setdefault(space_name, {})[fold_i] = pca_dim
            except Exception as e:
                print(f"    fold {fold_i}: PCA failed ({space_name}, "
                      f"{pca_dim} features): {type(e).__name__}: {e}")

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
            if pool.get('base_rep') is not None:
                _br = pool['base_rep']
                pool_base_rep[fold_i] = {
                    'rep': _br['rep'],
                    'objective': _br['objective'],
                    'metric_name': _br['metric_name'],
                    'score_other_folds': _br['score_other_folds'],
                    'n_other_folds': _br['n_other_folds'],
                }
            for h in (pool.get('hybrids') or []):
                sname = f"hybrid_{h['objective']}_r{h['rank']}"
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
                        'augmentation_strategy': blueprint.augmentation_strategy,
                        'objective': h['objective'],
                        'rank': h['rank'],
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
                    print(f"    fold {fold_i}: hybrid build failed "
                          f"({sname}): {type(e).__name__}: {str(e)[:80]}")

        # ── A PCA control of every hybrid slot's own width ────────────────
        # The fixed 100-feature control above matches a hybrid slot only by
        # coincidence: each slot's width is whatever its recipe chose, and the
        # pool's recipes are searched across [10, 25, 50, 100, 250, 500]. So a
        # control is fitted at each distinct width this fold's hybrids came out
        # at, named by that width. The 100-feature control stays.
        for _sname in list(fold_spaces):
            if _sname not in _slot_blueprints:
                continue
            _w = int(np.asarray(fold_spaces[_sname][0]).shape[1])
            _add_pca_control(f'pca_matched_{_w}', _w)

        # Score every space on this held-out fold.
        for sname, (Xtr, Xva) in fold_spaces.items():
            try:
                Xtr_s, Xva_s = _scale_train_val(np.asarray(Xtr, dtype=np.float64),
                                                np.asarray(Xva, dtype=np.float64))
            except Exception as e:
                print(f"    fold {fold_i}: scale failed ({sname}): "
                      f"{type(e).__name__}: {e}")
                continue

            for nc in n_clusters_list:
                _coh, _dropped = _heldout_coherence(
                    Xtr_s, y_tr, Xva_s, y_va, nc, is_clf)
                _acc_clu(sname, nc, fold_i, _coh)
                clusters_dropped.setdefault(sname, {}).setdefault(
                    nc, {})[fold_i] = _dropped

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
                    print(f"    fold {fold_i}: weighted retrieval failed "
                          f"({sname}): {type(e).__name__}: {e}")
                    _acc_ret(sname, 'weighted', fold_i, float('nan'))

    # ── Fold-index-aligned score lists ──
    # 2026-09-12: every list below has one entry per fold, NaN where that fold
    # produced no score for that space. A space that fails on one fold no longer
    # shifts its later folds' scores onto the wrong fold index.
    clustering_fold = {s: {nc: _fold_series(byf, len(folds))
                           for nc, byf in ncmap.items()}
                       for s, ncmap in clustering_fold.items()}
    retrieval_fold = {s: {c: _fold_series(byf, len(folds))
                          for c, byf in conds.items()}
                      for s, conds in retrieval_fold.items()}

    # ── Aggregate held-out scores (nanmean across folds) ──
    clustering_results = {}
    for sname, ncmap in clustering_fold.items():
        entry = {}
        for nc, fs in ncmap.items():
            entry[nc] = {
                'label_coherence': _nanmean(fs),
                'n_folds_used': int(sum(1 for s in fs
                                        if s is not None and np.isfinite(s))),
            }
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
        },
        # How many components each PCA control was fitted with, per fold. One
        # control per distinct hybrid width plus the fixed-width one, so every
        # hybrid slot has a control of its own width to be read against.
        'pca_control_dims': {s: {int(f): int(d) for f, d in sorted(byf.items())}
                             for s, byf in sorted(pca_control_dims.items())},
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
    for sname, conds in retrieval_fold.items():
        if not sname.startswith('rep_'):
            continue
        if _has_finite(conds['unweighted']):
            ret_baselines[f'{sname}|k{k_retrieval}'] = [
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
    # search pickled before the flag existed reads as pruned, which is the
    # reading hr_exclude_folds takes.
    _exclude = [0] if (pool_fold0_pruned is None or pool_fold0_pruned) else []
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
    }

    # Reuse the shared pure-arithmetic helper (no ML) to report the honest
    # hybrid-vs-best-single-rep delta inline for each axis.
    for axis, block, better in (
            ('clustering', results['fold_reselection_clustering'], 'higher'),
            ('retrieval', results['fold_reselection_retrieval'], 'lower')):
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
            model_key = 'kmeans' if axis == 'clustering' else 'knn'
            m = hr.get(model_key)
            if m is not None:
                flag = ' [single-fold, NOT honest]' if m.get('single_fold') else ''
                print(f"  Honest LOFO re-score ({axis}, {better}-better){flag}: "
                      f"hybrid={m['hybrid_reselected']:.4f}, "
                      f"best_single={m['best_single_reselected']:.4f}, "
                      f"delta={m['delta']:+.4f} (positive=hybrid better)")
        except Exception as e:
            print(f"  [warn] fold_reselection ({axis}) failed: "
                  f"{type(e).__name__}: {e}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S13: Clustering & Retrieval')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's13_clustering_retrieval', args.dataset)
