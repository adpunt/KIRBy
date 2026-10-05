#!/usr/bin/env python3
"""Fed04: Cluster-Aware Importance.

Does the optimal KIRBy allocation vary across chemical space regions?
Uses FLuID's K-Means clustering to partition molecules, then runs KIRBy
importance independently per cluster. Compares whether structural FPs
dominate near activity cliffs while foundation model embeddings dominate
in smooth SAR regions.

Extends C07 (data complexity partitions) with FLuID's clustering.
Adds smooth_factor sweep to test cluster boundary regularization.

Aligned 2026-10-05 (handoff 2.1 R1-R6, 2.4): per outer fold, ONE search on
that fold's training rows (two-class labels, the dataset's decided grid,
pruning off) gives this model's 2 best hybrids at every decided budget; each
of the two recipes is run through every phase below on that fold. No
whole-dataset search, no hand-written fallback recipe.

Usage:
    python fed04_cluster_aware_importance.py --dataset herg
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dataclasses
import json
import time
import traceback
import numpy as np
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
)
from experiments.federated.federated_helpers import (
    smiles_to_fluid_df, cluster_training_only, compute_clf_metrics,
    extract_importance_profile, compare_importance_profiles,
    extract_selected_features, compare_selected_features,
    _resolve_match, fit_fed_classifier, fed_to_kirby_model,
    binarise_labels,
    smoke_grid, smoke_n_folds, smoke_k_silos,
    federated_budgets, fed_outer_folds, fed_fold_selection, recipe_reps,
    FED_FOLD_SEED,
)
from kirby.hybrid import apply_feature_selection
from kirby.find_best_hybrid import BUILTIN_METRICS


# ── Constants ────────────────────────────────────────────────────────────────
# smoke_* helpers are no-ops unless KIRBY_SMOKE_N is set.

EXPERIMENT_DIR = 'fed04_cluster_aware_importance'
# Part of the resume identity of both partial files (the per-call one and the
# per-fold one). Changed 2026-10-05 when the recipes moved to per-fold
# searches, so a partial file written by the whole-dataset version is moved
# aside instead of read back. Changed again the same day (suffix 'b') when the
# rebuilds took the search's metric and the 2 base representations became
# fitted arms, so a file from the first aligned version is not read back.
ALIGNMENT_ID = 'per_fold_selection_2026_10_05b'

K = smoke_k_silos(5)
SMOOTH_FACTORS = smoke_grid([0, 0.01, 0.05, 0.1, 0.2], [0, 0.05])
# The two hybrids per model per fold (handoff 2.1 R4). Until 2026-10-05 the
# slots were general1, general2 and distance1, read once from a whole-dataset
# search; fed04 measures no neighbours or distances, so answer 1 gives it no
# distance hybrid.
HYBRID_SLOTS = ('hybrid1', 'hybrid2')
# What the result file reports: MCC on the two-class labels, higher is better.
# run_all stamps the result file with these two (dirty-run bug 13: fed04 was
# stamped 'rmse, lower is better' on caco2 and logd).
REPORTED_METRIC = 'mcc'
REPORTED_LOWER_IS_BETTER = False


def _drop_zero_importance(fi):
    """Cut every selected feature whose importance is not above zero.

    create_hybrid stops at the last feature scored above zero, EXCEPT when no
    feature of a representation scored above zero: then it takes the
    representation's whole quota by column order (hybrid.py, the
    `_positive > 0` branch). On a small cluster lightgbm_gain often makes no
    split at all, and the dirty run (DIRTY_RUN_RESULTS.md, fed04) found two
    clusters whose "selection" was 250 features, every one of importance 0,
    taken in column order. Those are not a selection, so they are cut here
    (2026-10-05) and the width actually built is recorded beside the budget.
    A representation entry without per-feature importances is kept as built.

    Returns (feature_info cut, {rep: n features cut}).
    """
    out, dropped = {}, {}
    for rn, info in fi.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            out[rn] = info
            continue
        idx = np.asarray(info['selected_indices']).ravel()
        imp = info.get('importance_scores')
        imp = None if imp is None else np.asarray(imp, dtype=float).ravel()
        if imp is None or len(imp) != len(idx):
            out[rn] = info
            continue
        keep = imp > 0
        if keep.all():
            out[rn] = info
            continue
        dropped[rn] = int((~keep).sum())
        if keep.any():
            out[rn] = dict(info, selected_indices=idx[keep],
                           importance_scores=imp[keep],
                           n_features=int(keep.sum()))
        # A representation with nothing above zero leaves the selection.
    return out, dropped


def _width(fi):
    """How many features a feature_info actually selects."""
    return int(sum(len(np.asarray(v['selected_indices']).ravel())
                   for v in fi.values()
                   if isinstance(v, dict) and 'selected_indices' in v))


def _build(reps, labels, groups, recipe, augs=None):
    """One build of the recipe the fold's search chose, on the rows given,
    zero-importance features cut. Returns (feature_info, width built,
    {rep: n cut}); raises when nothing above zero is left.

    Every build (per cluster, per random part and on all training rows) goes
    through build_fold_hybrid with the search's own metric and direction
    (2026-10-05). Before, create_hybrid was called with neither, and each
    representation's share of the budget is its score read off with that
    metric (hybrid._score_rep_for_budget): without it the share came from the
    internal evaluator's accuracy, while the search ranked a split made by its
    own two-class metric. build_fold_hybrid also carries the recipe's other
    fields (candidate ranking, augmentation settings) that the old call left
    at create_hybrid's defaults. The evaluator is the recipe's, with 'match'
    already resolved to the run's own model; `reps` is already cut to the
    recipe's representation count (recipe_reps), so build_fold_hybrid does
    not cut again. The second matrix it returns (the same rows) is unused.
    """
    bp = dataclasses.replace(
        recipe['_blueprint'],
        internal_evaluator=recipe['internal_evaluator'],
        budget=int(recipe['budget']))
    use_augs = augs if recipe['aug_condition'] != 'none' else None
    _, _, fi = build_fold_hybrid(
        reps, reps, labels, bp,
        augmentations_train=use_augs, augmentations_val=use_augs,
        groups=groups,
        **fold_hybrid_scoring_kwargs(
            bp, model=None, metric_fn=recipe['_metric_fn'],
            lower_is_better=recipe['_lower_is_better']))
    fi, dropped = _drop_zero_importance(fi)
    width = _width(fi)
    if width == 0:
        raise ValueError('no selected feature scored above zero importance '
                         f'({sum(dropped.values())} features at importance 0 '
                         'cut)')
    return fi, width, dropped


def _compute_cluster_importance(train_reps, train_labels, silo_indices, k,
                                recipe, augs=None, train_groups=None):
    """Run create_hybrid per cluster and extract importance profiles.

    `recipe` is one hybrid from this fold's search (allocation, importance,
    budget, internal evaluator already resolved from 'match', augmentation
    condition); train_reps are already cut to the recipe's representations.
    There is no default recipe (2026-10-05): a cluster is built on the recipe
    the fold's search chose, or recorded as failed.

    Returns:
        (profiles, feature_infos, cluster_stats)
        - profiles: {cluster_id: {rep_name: importance_score}}, built clusters only
        - feature_infos: {cluster_id: feature_info_dict}, built clusters only
        - cluster_stats: {cluster_id: {n_samples, built, error | n_active,
          top_rep, budget, n_features_built, n_cut_zero_importance}} for
          EVERY cluster, so a failed cluster is in the file with its reason.
    """
    if recipe['internal_evaluator'] == 'match':
        raise ValueError("internal evaluator 'match' must be resolved to the "
                         "run's own model before a rebuild")
    profiles = {}
    feature_infos = {}
    cluster_stats = {}

    for s in range(1, k + 1):
        idx = silo_indices[s]
        if len(idx) < 20:
            cluster_stats[s] = {
                'n_samples': len(idx),
                'built': False,
                'error': 'too few samples (fewer than 20)',
            }
            continue

        cluster_reps = {r: arr[idx] for r, arr in train_reps.items()}
        cluster_labels = train_labels[idx]
        cluster_augs = ({a: v[idx] for a, v in augs.items()}
                        if augs else None)

        try:
            fi, width, dropped = _build(
                cluster_reps, cluster_labels,
                (train_groups[idx] if train_groups is not None else None),
                recipe, augs=cluster_augs)
            feature_infos[s] = fi
            profile = extract_importance_profile(fi)
            profiles[s] = profile

            top_rep = max(profile, key=profile.get) if profile else 'none'
            n_active = int(cluster_labels.sum())
            cluster_stats[s] = {
                'n_samples': len(idx),
                'built': True,
                'n_active': n_active,
                'n_inactive': len(idx) - n_active,
                'top_rep': top_rep,
                'n_reps_selected': len(profile),
                'budget': int(recipe['budget']),
                'n_features_built': width,
                'n_cut_zero_importance': int(sum(dropped.values())),
                'cut_zero_importance_by_rep': dropped,
            }
        except Exception as e:
            # Surface the traceback so a real bug (shape/import/NameError) in
            # the per-cluster create_hybrid path is debuggable from the
            # expensive run's logs rather than reduced to a string (FED-21).
            n_active = int(cluster_labels.sum())
            cluster_stats[s] = {
                'n_samples': len(idx),
                'built': False,
                'n_active': n_active,
                'n_inactive': len(idx) - n_active,
                'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc(),
            }
            traceback.print_exc()

    return profiles, feature_infos, cluster_stats


def _recipe_record(entry, kmodel, fold_i, budget, slot):
    """The recipe one slot runs, as plain fields for the result file, with
    'match' resolved to this run's own model (never rf)."""
    bp = entry.get('blueprint') if isinstance(entry, dict) else None
    if bp is None:
        # top_hybrids carries row.get('blueprint'), so an entry can arrive
        # without one. Raised here and recorded by the caller as a failed
        # slot, instead of an AttributeError that ends the whole run.
        raise ValueError(f'the search entry for {slot} at {budget} features '
                         f'carries no recipe (blueprint is None)')
    return {
        'slot': slot,
        'fold': int(fold_i),
        'label': entry.get('label'),
        'allocation_method': bp.allocation_method,
        'importance_method': bp.importance_method,
        'budget': int(entry.get('budget') or budget),
        'internal_evaluator_searched': bp.internal_evaluator,
        'internal_evaluator': _resolve_match(bp.internal_evaluator, kmodel),
        'rep_count': entry.get('rep_count', getattr(bp, 'rep_count', None)),
        'reps_searched': list(getattr(bp, 'reps', None) or []),
        'aug_condition': entry.get('aug_condition') or 'none',
        # The metric the search ranked this recipe by, and its direction.
        # Every rebuild sizes each representation's share of the budget with
        # this metric (_build), as the search did.
        'search_metric': getattr(bp, 'metric', None),
        'search_lower_is_better': bool(getattr(bp, 'lower_is_better', False)),
        # The search's own score: the mean over its inner folds, all on this
        # outer fold's training rows, by the KIRBy model it was ranked for.
        'train_rows_score': entry.get('score_mean'),
        'n_features_searched': entry.get('n_features'),
    }


def _fit_base_arms(*, base_names, fed_model, fed_seed, reps, labels_binary,
                   train_idx, test_idx):
    """The fold's 2 best base representations as two reference arms (handoff
    2.1 R4, 2026-10-05): each one whole, fitted on ALL training rows exactly
    as the global hybrid is (fit_fed_classifier, the run's model, the same
    under-sampler), scored on the held-out rows and on the training rows in
    the reported metric. One per fold, not per slot or budget: a base
    representation has no budget. A failed fit is recorded, not dropped."""
    y_train = labels_binary[train_idx]
    y_test = labels_binary[test_idx]
    out = []
    for rank, rn in enumerate(base_names, 1):
        rec = {'rank': rank, 'representation': rn}
        try:
            X_tr = reps[rn][train_idx]
            X_te = reps[rn][test_idx]
            clf = fit_fed_classifier(fed_model, X_tr, y_train, seed=fed_seed)
            rec['n_features'] = int(X_tr.shape[1])
            rec['held_out'] = convert_numpy(compute_clf_metrics(
                y_test, clf.predict(X_te), clf.predict_proba(X_te)[:, 1]))
            rec['train_rows'] = convert_numpy(compute_clf_metrics(
                y_train, clf.predict(X_tr), clf.predict_proba(X_tr)[:, 1]))
            print(f"  base {rank} {rn}: held-out MCC "
                  f"{rec['held_out'].get('mcc', float('nan')):.3f} "
                  f"({rec['n_features']} features, all training rows)")
        except Exception as e:
            traceback.print_exc()
            rec['error'] = f'{type(e).__name__}: {e}'
        out.append(rec)
    return out


def _spread(vals):
    """Mean and spread over folds. One fold has no spread, and says so
    (handoff 2.4 step 14) instead of reporting a standard deviation of 0."""
    vals = [float(v) for v in vals
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and not np.isnan(v)]
    if not vals:
        return {'mean': None, 'std': None, 'n_folds': 0}
    return {'mean': float(np.mean(vals)),
            'std': float(np.std(vals)) if len(vals) > 1 else None,
            'n_folds': len(vals),
            **({'note': 'one fold, no spread'} if len(vals) == 1 else {})}


def _fmt(agg):
    if agg.get('mean') is None:
        return 'n/a'
    if agg.get('std') is None:
        return f"{agg['mean']:.3f} (one fold, no spread)"
    return f"{agg['mean']:.3f} ± {agg['std']:.3f}"


def _measure(*, recipe, fed_model, fed_seed, fold_id, train_idx, test_idx,
             reps, labels_binary, groups, fluid_df_all, augs):
    """Every phase for one fold, one recipe (one slot at one budget).
    Returns the fold record."""
    budget = int(recipe['budget'])
    y_train = labels_binary[train_idx]
    y_test = labels_binary[test_idx]

    # The recipe's own representations: its list, cut to its representation
    # count by this fold's search scores (recipe_reps). Every build below,
    # per cluster, per random part and on all training rows, uses the same
    # set, so the arms differ in rows and nothing else. Until 2026-10-05 every
    # representation was used whatever count the recipe won at.
    train_reps_all = {r: arr[train_idx] for r, arr in reps.items()}
    test_reps_all = {r: arr[test_idx] for r, arr in reps.items()}
    train_reps, rc_note = recipe_reps(
        train_reps_all, recipe['_blueprint'],
        context=f"fed04 fold {fold_id} {recipe['slot']} budget {budget}")
    test_reps = {r: test_reps_all[r] for r in train_reps}
    train_augs = ({a: v[train_idx] for a, v in augs.items()}
                  if augs else None)
    train_groups = groups[train_idx] if groups is not None else None

    # CLUSTER TRAINING DATA ONLY — matching FLuID's pattern
    t0 = time.time()
    train_fluid_df = fluid_df_all.iloc[train_idx]
    train_silo_indices, cluster_data, kmeans_model = cluster_training_only(
        train_fluid_df, k=K, smooth_factor=0.05, seed=fed_seed)
    print(f"  Clustering (train only): K={K}, {time.time() - t0:.1f}s")

    for s in range(1, K + 1):
        idx = train_silo_indices[s]
        n_act = int(y_train[idx].sum())
        print(f"    Silo {s}: {len(idx)} samples "
              f"({n_act} active, {len(idx) - n_act} inactive)")

    # Assign test molecules to nearest training cluster centroid
    test_fps = np.asarray(list(fluid_df_all.iloc[test_idx].FP))
    test_silo = kmeans_model.predict(test_fps) + 1  # 1-indexed to match silos

    # ── Phase 2: Per-cluster importance ──────────────────────────────
    print("  --- Phase 2: Per-cluster importance (smooth=0.05) ---")
    profiles, feature_infos, cluster_stats = _compute_cluster_importance(
        train_reps, y_train, train_silo_indices, K, recipe,
        augs=train_augs, train_groups=train_groups)

    # Global baseline (full training set)
    fi_global, global_width, global_dropped = _build(
        train_reps, y_train, train_groups, recipe, augs=train_augs)
    global_profile = extract_importance_profile(fi_global)
    global_top = max(global_profile, key=global_profile.get) if global_profile else 'none'
    print(f"  Global top rep: {global_top} ({global_width} of {budget} "
          f"features built)")

    # ── Phase 2b: Null control — a RANDOM partition of the same sizes ─
    # Phase 3 asks whether two regions of chemical space split the budget
    # across the representations differently. Two subsamples of a few
    # hundred rows already split it differently from subsampling alone,
    # so the number Phase 3 reports has no scale without something to
    # read it against. This partitions the SAME training rows into the
    # SAME cluster sizes at random — no chemistry in the assignment — and
    # runs the identical per-cluster selection over it. The chemical
    # clusters are doing something only if their profiles agree LESS with
    # each other than the random parts of the same sizes do.
    _sizes = [len(train_silo_indices[s]) for s in range(1, K + 1)]
    _perm = np.random.RandomState(1000 + fed_seed + fold_id).permutation(
        len(train_idx))
    _null_indices, _cut = {}, 0
    for _s, _n in enumerate(_sizes, 1):
        _null_indices[_s] = _perm[_cut:_cut + _n]
        _cut += _n
    null_profiles, _, null_cluster_stats = _compute_cluster_importance(
        train_reps, y_train, _null_indices, K, recipe,
        augs=train_augs, train_groups=train_groups)

    # ── Phase 3: Compare importance profiles ─────────────────────────
    # Only over the clusters that built on BOTH sides (2026-10-05). In the
    # dirty run 1 to 2 chemical clusters per fold failed and were dropped
    # while the random parts never failed, so the chemical number was a mean
    # over 3 or 4 clusters and the random one over 5. Random part s has the
    # same size as chemical cluster s, so the pairing is by cluster number.
    _both = sorted(set(profiles) & set(null_profiles))
    mean_spearman, pairwise = compare_importance_profiles(
        {s: profiles[s] for s in _both})
    null_spearman, null_pairwise = compare_importance_profiles(
        {s: null_profiles[s] for s in _both})
    # The unmatched numbers, every cluster that built on its own side.
    mean_spearman_all, _ = compare_importance_profiles(profiles)
    null_spearman_all, _ = compare_importance_profiles(null_profiles)
    _ms_str = 'n/a' if mean_spearman is None else f"{mean_spearman:.3f}"
    _ns_str = 'n/a' if null_spearman is None else f"{null_spearman:.3f}"
    print(f"  Phase 3 mean pairwise Spearman over the {len(_both)} of {K} "
          f"clusters that built on both sides: {_ms_str}  "
          f"(random partition of the same sizes: {_ns_str})")
    cluster_failures = {
        'chemical': {s: st.get('error') for s, st in cluster_stats.items()
                     if not st.get('built')},
        'random': {s: st.get('error') for s, st in null_cluster_stats.items()
                   if not st.get('built')},
    }
    if cluster_failures['chemical'] or cluster_failures['random']:
        print(f"  clusters that did not build: chemical "
              f"{sorted(cluster_failures['chemical'])}, random "
              f"{sorted(cluster_failures['random'])}")
    rep_rankings = {}
    for s, profile in profiles.items():
        ranked = sorted(profile.items(), key=lambda x: -x[1])
        rep_rankings[s] = [r for r, _ in ranked]

    # ── Phase 3b: Compare the selected features themselves ───────────
    # The Spearman above ranks representations by how much of the budget
    # each one won. Two clusters can split 100 features across the same
    # representations in the same proportions and still take 100 different
    # descriptors and fingerprint bits, and that comparison says so.
    cluster_selections = {
        s: extract_selected_features(fi)
        for s, fi in feature_infos.items() if fi is not None}
    feature_overlap = compare_selected_features(cluster_selections)
    global_selection = extract_selected_features(fi_global)
    _global_set = set(global_selection['labels'])
    # How much of each cluster's own selection the whole-training-fold
    # selection also took. A cluster at 1.0 is asking for nothing the
    # global selection does not already give it.
    cluster_vs_global = {}
    for s, sel in cluster_selections.items():
        own = set(sel['labels'])
        cluster_vs_global[s] = {
            'n_selected': len(own),
            'n_also_in_global': len(own & _global_set),
            'fraction_also_in_global': (
                float(len(own & _global_set) / len(own)) if own else None),
            'not_in_global': sorted(own - _global_set),
        }
    _mj = feature_overlap.get('mean_jaccard')
    _nc = feature_overlap.get('n_clusters', 0)
    if _mj is None:
        print(f"  Phase 3b feature overlap: undefined — {_nc} cluster(s) "
              f"produced a selection, so there is no pair to compare")
    else:
        print(f"  Phase 3b mean pairwise feature overlap: {_mj:.3f} "
              f"across {_nc} clusters  "
              f"({feature_overlap['n_shared_by_all']} features taken by "
              f"every one of them; budget {budget} features)")
    for s in sorted(cluster_selections):
        _top = cluster_selections[s]['labels'][:5]
        print(f"    Cluster {s} top 5: {', '.join(_top)}")

    # ── Phase 4: Cluster-conditional hybrid ──────────────────────────
    # THREE arms, not two. The two that were here answer a question with
    # two things moving at once:
    #   global            — all training rows, ONE feature set for all.
    #   cluster_model     — this cluster's rows only, its OWN features.
    # A cluster's rows are a fifth of the fold and FLuID's classifier puts
    # every fit through RandomUnderSampler first, so on caco2 the
    # per-cluster forests trained on 58 to 596 rows against roughly a
    # thousand for the global one (measured 2026-09-18, all five folds).
    # Whatever the gap between those two arms is, most of it is the row
    # count, and the experiment reads it as the features.
    #   cluster_features  — ALL training rows, this cluster's features.
    # The third arm holds the rows fixed and varies only the feature set,
    # which is the question fed04 exists to ask. cluster_features minus
    # global is the value of choosing features per region; cluster_model
    # minus cluster_features is what fitting on one region's rows costs.
    per_cluster_models = {}
    per_cluster_models_allrows = {}
    per_cluster_fis = {}
    per_cluster_fit_rows = {}
    # A cluster whose features built can still fail to FIT: the under-sampler
    # on a cluster with one class, or svm's internal calibration folds on a
    # cluster left with a handful of rows after under-sampling. Until
    # 2026-10-05 that exception ended the whole record (fold, slot, budget),
    # losing the global arm and the random-partition comparison with it. Now
    # the cluster is recorded here and its held-out molecules are scored by
    # the whole-fold model in BOTH per-cluster arms, so the two arms still
    # differ in their training rows only.
    cluster_fit_failures = {}
    for s in range(1, K + 1):
        fi = feature_infos.get(s)
        if fi is None:
            continue
        idx = train_silo_indices[s]
        try:
            X_s = apply_feature_selection(train_reps, fi)
            clf = fit_fed_classifier(
                fed_model, X_s[idx], y_train[idx], seed=fed_seed)
            # Same features, every training row.
            clf_all = fit_fed_classifier(
                fed_model, X_s, y_train, seed=fed_seed)
        except Exception as e:
            traceback.print_exc()
            cluster_fit_failures[s] = {
                'n_samples': int(len(idx)),
                'n_active': int(y_train[idx].sum()),
                'error': f'{type(e).__name__}: {e}',
            }
            print(f"    cluster {s}: features built but the fit failed "
                  f"({type(e).__name__}); scored by the whole-fold model")
            continue
        per_cluster_models[s] = clf
        per_cluster_models_allrows[s] = clf_all
        per_cluster_fis[s] = fi
        # How many rows each arm actually fitted on, AFTER the
        # under-sampler, so the row-count confound is in the file rather
        # than only in this comment.
        _na = int(y_train[idx].sum())
        per_cluster_fit_rows[s] = {
            'n_cluster_rows': int(len(idx)),
            'n_cluster_rows_after_undersample': 2 * min(
                _na, int(len(idx)) - _na),
            'n_all_train_rows': int(len(y_train)),
            'n_all_train_rows_after_undersample': 2 * min(
                int(y_train.sum()), int(len(y_train) - y_train.sum())),
        }

    # Predict: use ORIGINAL FLuID cluster assignments for test molecules
    # (test_silo was computed earlier in this fold from the original
    # clustering — do NOT re-fit KMeans, which would produce misaligned
    # cluster IDs)
    per_cluster_test_X = {}
    for s, fi in per_cluster_fis.items():
        per_cluster_test_X[s] = apply_feature_selection(test_reps, fi)

    # Global hybrid baseline (compute first so fallback is available)
    X_global_train = apply_feature_selection(train_reps, fi_global)
    X_global_test = apply_feature_selection(test_reps, fi_global)
    clf_global = fit_fed_classifier(
        fed_model, X_global_train, y_train, seed=fed_seed)
    y_pred_global = clf_global.predict(X_global_test)
    y_prob_global = clf_global.predict_proba(X_global_test)[:, 1]
    # The training-row score of the global arm, in the reported metric, on
    # the rows it was fitted on (before the under-sampler).
    global_train_metrics = compute_clf_metrics(
        y_train, clf_global.predict(X_global_train),
        clf_global.predict_proba(X_global_train)[:, 1])

    # Route each held-out molecule to its own cluster, twice: once to the
    # model fitted on that cluster's rows, once to the model fitted on
    # every training row. Same routing, same features, different training
    # set — so the two differ by the row count and nothing else.
    def _route(models):
        y_p = y_pred_global.copy()
        y_q = y_prob_global.copy()
        miss = np.ones(len(test_idx), dtype=bool)
        for i in range(len(test_idx)):
            cluster = test_silo[i]
            if cluster in models and cluster in per_cluster_test_X:
                mdl = models[cluster]
                X_i = per_cluster_test_X[cluster][[i]]
                y_p[i] = mdl.predict(X_i)[0]
                proba = mdl.predict_proba(X_i)
                y_q[i] = proba[0, 1] if proba.shape[1] == 2 else 0.5
                miss[i] = False
        return y_p, y_q, miss

    y_pred_conditional, y_prob_conditional, fallback_mask = _route(
        per_cluster_models)
    y_pred_featonly, y_prob_featonly, _ = _route(
        per_cluster_models_allrows)
    n_fallback = int(fallback_mask.sum())
    if n_fallback > 0:
        print(f"  {n_fallback} test samples used global fallback")

    conditional_metrics = compute_clf_metrics(
        y_test, y_pred_conditional, y_prob_conditional)
    featonly_metrics = compute_clf_metrics(
        y_test, y_pred_featonly, y_prob_featonly)
    global_metrics = compute_clf_metrics(
        y_test, y_pred_global, y_prob_global)
    print(f"  MCC — per-cluster features + per-cluster rows "
          f"{conditional_metrics['mcc']:.3f} | per-cluster features + ALL "
          f"rows {featonly_metrics['mcc']:.3f} | global features + all "
          f"rows {global_metrics['mcc']:.3f}")

    # ── Phase 5: Smooth factor sweep ─────────────────────────────────
    smooth_sweep = {}
    for sf in SMOOTH_FACTORS:
        try:
            train_silo_idx_sf, _, _ = cluster_training_only(
                train_fluid_df, k=K, smooth_factor=sf, seed=fed_seed)
            prof_sf, _, stats_sf = _compute_cluster_importance(
                train_reps, y_train, train_silo_idx_sf, K, recipe,
                augs=train_augs, train_groups=train_groups)
            mean_sp, pw = compare_importance_profiles(prof_sf)
            smooth_sweep[str(sf)] = convert_numpy({
                'stability': mean_sp,
                'pairwise': pw,
                'n_clusters_built': len(prof_sf),
                'cluster_stats': stats_sf,
                'profiles': prof_sf,
            })
        except Exception as e:
            smooth_sweep[str(sf)] = {'error': f'{type(e).__name__}: {e}'}

    _rec = {k: v for k, v in recipe.items() if not k.startswith('_')}
    return {
        'fold': fold_id,
        'slot': recipe['slot'],
        'budget': budget,
        'recipe': convert_numpy(_rec),
        'rep_count_note': convert_numpy(rc_note),
        'reps_used': sorted(train_reps),
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'global_importance': convert_numpy({
            'profile': global_profile,
            'top_rep': global_top,
            'n_features_built': global_width,
            'n_cut_zero_importance': int(sum(global_dropped.values())),
        }),
        'per_cluster_importance': convert_numpy({
            'profiles': profiles,
            'cluster_stats': cluster_stats,
            'rep_rankings': rep_rankings,
        }),
        'cluster_failures': convert_numpy(cluster_failures),
        # Clusters whose features built and whose per-cluster fit failed
        # (Phase 4); their held-out molecules are in n_fallback.
        'cluster_fit_failures': convert_numpy(cluster_fit_failures),
        # Matched: the clusters that built on both sides (see Phase 3).
        'profile_stability': convert_numpy({
            'mean_spearman': mean_spearman,
            'pairwise': pairwise,
            'clusters_compared': _both,
            'n_clusters_compared': len(_both),
            'n_clusters_built': len(profiles),
            'mean_spearman_every_built_cluster': mean_spearman_all,
        }),
        'profile_stability_null': convert_numpy({
            'mean_spearman': null_spearman,
            'pairwise': null_pairwise,
            'clusters_compared': _both,
            'n_clusters_compared': len(_both),
            'n_clusters_built': len(null_profiles),
            'mean_spearman_every_built_cluster': null_spearman_all,
            'cluster_stats': null_cluster_stats,
            'partition': 'random, same cluster sizes as K-Means',
        }),
        # Which features, not just which representations. 'per_cluster'
        # lists every feature each cluster took, most important first;
        # 'overlap' says how far any two clusters agree; 'vs_global' says
        # how much each cluster wanted that the whole-training-fold
        # selection did not give it.
        'selected_features': convert_numpy({
            'per_cluster': cluster_selections,
            'overlap': feature_overlap,
            'global': global_selection,
            'vs_global': cluster_vs_global,
        }),
        'cluster_conditional_hybrid': convert_numpy(conditional_metrics),
        'cluster_features_all_rows': convert_numpy(featonly_metrics),
        'global_hybrid': convert_numpy(global_metrics),
        'global_hybrid_train_rows': convert_numpy(global_train_metrics),
        'per_cluster_fit_rows': convert_numpy(per_cluster_fit_rows),
        'smooth_factor_sweep': smooth_sweep,
        'n_fallback': n_fallback,
    }


def _summarise_slot(fold_records, budgets, headline_budget, n_folds):
    """The per-slot aggregate over folds. Every table reads the headline
    width; by_budget has the three arms at every width."""
    headline_records = [r for r in fold_records
                        if r.get('budget') == headline_budget]

    def _arm(records, key, metric):
        return _spread([(r.get(key) or {}).get(metric) for r in records])

    metric_keys = ['mcc', 'bac', 'f1', 'auroc', 'accuracy', 'f1_weighted']
    cond_agg = {m: _arm(headline_records, 'cluster_conditional_hybrid', m)
                for m in metric_keys}
    glob_agg = {m: _arm(headline_records, 'global_hybrid', m)
                for m in metric_keys}
    feat_agg = {m: _arm(headline_records, 'cluster_features_all_rows', m)
                for m in metric_keys}

    stab = _spread([r['profile_stability'].get('mean_spearman')
                    for r in headline_records])
    stab_null = _spread([r['profile_stability_null'].get('mean_spearman')
                         for r in headline_records])
    n_compared = [int(r['profile_stability'].get('n_clusters_compared') or 0)
                  for r in headline_records]
    n_failed_chem = [len((r.get('cluster_failures') or {}).get('chemical') or {})
                     for r in headline_records]
    n_failed_rand = [len((r.get('cluster_failures') or {}).get('random') or {})
                     for r in headline_records]

    # ── Which features this dataset prefers, across every fold ───────────
    # Each fold x cluster produced one selection of at most the budget. Counting
    # how many of those selections took each feature says what the dataset
    # wants everywhere (a high count) against what one region of chemical
    # space wants and the rest does not (a count of 1). The same counts from
    # another dataset's file are directly comparable, because a feature label
    # is the representation and the descriptor name or index, nothing else.
    _cluster_sel_counts = {}
    _n_cluster_selections = 0
    _global_sel_counts = {}
    _n_global_selections = 0
    _jaccards = []
    _widths = []
    for _r in headline_records:
        _sf = _r.get('selected_features') or {}
        for _s, _sel in (_sf.get('per_cluster') or {}).items():
            _n_cluster_selections += 1
            _widths.append(len(_sel.get('labels', [])))
            for _lab in _sel.get('labels', []):
                _cluster_sel_counts[_lab] = _cluster_sel_counts.get(_lab, 0) + 1
        _g = _sf.get('global') or {}
        if _g.get('labels'):
            _n_global_selections += 1
            for _lab in _g['labels']:
                _global_sel_counts[_lab] = _global_sel_counts.get(_lab, 0) + 1
        _j = (_sf.get('overlap') or {}).get('mean_jaccard')
        if isinstance(_j, (int, float)) and not np.isnan(_j):
            _jaccards.append(float(_j))

    _always = sorted(l for l, n in _cluster_sel_counts.items()
                     if n == _n_cluster_selections) if _n_cluster_selections else []
    _once = sorted(l for l, n in _cluster_sel_counts.items() if n == 1)
    _ranked_feats = sorted(_cluster_sel_counts.items(), key=lambda x: (-x[1], x[0]))

    dataset_feature_preference = {
        'n_cluster_selections': _n_cluster_selections,
        'n_global_selections': _n_global_selections,
        'budget_per_selection': headline_budget,
        # The width actually built, per selection: zero-importance features
        # are cut (2026-10-05), so a selection can be narrower than the budget.
        'widths_built': _widths,
        # Every feature any cluster took, with how many of the
        # n_cluster_selections selections took it. Full list, not a top slice.
        'cluster_selection_counts': dict(_ranked_feats),
        'global_selection_counts': _global_sel_counts,
        'taken_by_every_cluster_in_every_fold': _always,
        'n_taken_by_every_cluster_in_every_fold': len(_always),
        'taken_exactly_once': _once,
        'n_taken_exactly_once': len(_once),
        'n_distinct_features_ever_taken': len(_cluster_sel_counts),
        'mean_pairwise_feature_overlap': (
            float(np.mean(_jaccards)) if _jaccards else None),
    }

    # The smooth-factor sweep over folds.
    smooth_factor_stability = {}
    for _sf in SMOOTH_FACTORS:
        _key = str(_sf)
        smooth_factor_stability[_key] = _spread([
            ((_r.get('smooth_factor_sweep') or {}).get(_key) or {}).get('stability')
            for _r in headline_records])

    # How many held-out molecules landed in a cluster with no per-cluster
    # model and were scored by the whole-fold model instead. A large count
    # makes the two arms of the comparison partly the same arm, so it belongs
    # beside the two scores rather than only inside each fold.
    _fb = [int(_r.get('n_fallback', 0)) for _r in headline_records]
    _nte = [int(_r.get('n_test', 0)) for _r in headline_records]
    n_fallback_agg = dict(
        _spread(_fb), per_fold=_fb,
        mean_fraction_of_test=(
            float(np.mean([a / b for a, b in zip(_fb, _nte) if b]))
            if any(_nte) else None))

    # The three arms at every width. One row is one width; the numbers are
    # the mean over the folds of that width, in MCC.
    by_budget = {}
    for _b in budgets:
        _recs = [r for r in fold_records if r.get('budget') == _b]
        if not _recs:
            continue
        by_budget[str(_b)] = {
            'n_folds': len(_recs),
            'cluster_conditional_hybrid_mcc':
                _arm(_recs, 'cluster_conditional_hybrid', 'mcc'),
            'cluster_features_all_rows_mcc':
                _arm(_recs, 'cluster_features_all_rows', 'mcc'),
            'global_hybrid_mcc': _arm(_recs, 'global_hybrid', 'mcc'),
            'global_hybrid_train_rows_mcc':
                _arm(_recs, 'global_hybrid_train_rows', 'mcc'),
            'global_width_built': [
                (r.get('global_importance') or {}).get('n_features_built')
                for r in _recs],
        }

    aggregate = {
        'cluster_conditional_hybrid': cond_agg,
        'cluster_features_all_rows': feat_agg,
        'global_hybrid': glob_agg,
        # Both over the clusters that built on both sides, per fold.
        'mean_profile_stability': stab['mean'],
        'profile_stability': stab,
        'mean_profile_stability_null': stab_null['mean'],
        'profile_stability_null': stab_null,
        'n_clusters_compared_per_fold': n_compared,
        'n_chemical_clusters_failed_per_fold': n_failed_chem,
        'n_random_clusters_failed_per_fold': n_failed_rand,
        'smooth_factor_stability': convert_numpy(smooth_factor_stability),
        'n_fallback': convert_numpy(n_fallback_agg),
        'feature_preference': convert_numpy(dataset_feature_preference),
    }
    return convert_numpy(by_budget), aggregate


def _print_slot(slot, by_budget, aggregate, headline_budget, n_folds):
    print(f"\n{'─' * 70}")
    print(f"  {slot}: aggregated over {n_folds} fold(s), "
          f"headline budget {headline_budget} features:")
    print(f"\n  {'budget':>8} {'conditional':>12} {'feat_allrows':>13} "
          f"{'global':>10}   (mean MCC over folds)")
    for _b, _row in by_budget.items():
        def _m(a):
            return 'n/a' if a.get('mean') is None else f"{a['mean']:.4f}"
        print(f"  {_b:>8} {_m(_row['cluster_conditional_hybrid_mcc']):>12} "
              f"{_m(_row['cluster_features_all_rows_mcc']):>13} "
              f"{_m(_row['global_hybrid_mcc']):>10}")
    print(f"  Profile stability, chemical clusters: "
          f"{_fmt(aggregate['profile_stability'])}")
    print(f"  Same number over a RANDOM partition of the same cluster sizes: "
          f"{_fmt(aggregate['profile_stability_null'])}")
    print(f"  Both over the clusters that built on both sides; clusters "
          f"compared per fold: {aggregate['n_clusters_compared_per_fold']} "
          f"of {K}. Chemical clusters that failed per fold: "
          f"{aggregate['n_chemical_clusters_failed_per_fold']}; random: "
          f"{aggregate['n_random_clusters_failed_per_fold']}")
    cond = aggregate['cluster_conditional_hybrid']['mcc']
    feat = aggregate['cluster_features_all_rows']['mcc']
    glob = aggregate['global_hybrid']['mcc']
    print(f"  MCC, per-cluster features + per-cluster rows: {_fmt(cond)}")
    print(f"  MCC, per-cluster features + ALL training rows: {_fmt(feat)}")
    print(f"  MCC, global features + all training rows:      {_fmt(glob)}")
    if None not in (cond['mean'], feat['mean'], glob['mean']):
        print(f"  choosing features per region is worth "
              f"{feat['mean'] - glob['mean']:+.4f} MCC; "
              f"fitting on one region's rows costs "
              f"{cond['mean'] - feat['mean']:+.4f}")
    print(f"  Smooth factor -> mean pairwise Spearman over folds:")
    for _k, _v in aggregate['smooth_factor_stability'].items():
        print(f"    smooth_factor={_k}: {_fmt(_v)} "
              f"({_v['n_folds']} of {n_folds} folds defined)")
    _fbm = aggregate['n_fallback'].get('mean')
    print(f"  Held-out molecules scored by the whole-fold model instead: "
          f"{'n/a' if _fbm is None else f'{_fbm:.1f}'} per fold")
    _fp = aggregate['feature_preference']
    _ov = _fp['mean_pairwise_feature_overlap']
    _ov_str = 'n/a' if _ov is None else f"{_ov:.3f}"
    print(f"\n  Features: {_fp['n_distinct_features_ever_taken']} distinct "
          f"features taken across {_fp['n_cluster_selections']} selections "
          f"(budget {headline_budget} features; widths built "
          f"{sorted(set(_fp['widths_built']))})")
    print(f"  Mean pairwise overlap between two clusters' selections: {_ov_str}")
    print(f"  Taken by every cluster in every fold: "
          f"{_fp['n_taken_by_every_cluster_in_every_fold']} features")
    print(f"  Taken by exactly one cluster in one fold: "
          f"{_fp['n_taken_exactly_once']} features")
    _top = list(_fp['cluster_selection_counts'].items())[:10]
    if _top:
        print(f"  Most-taken 10 (count out of {_fp['n_cluster_selections']}):")
        for _lab, _n in _top:
            print(f"    {_n:>3}  {_lab}")
    print(f"  Full counts under aggregate.feature_preference")


def _run_one(data, config, output_dir, dataset_name, **kwargs):
    """Fed04: Cluster-aware importance."""
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity)
    print("\n" + "=" * 70)
    print("EXPERIMENT FED04: CLUSTER-AWARE IMPORTANCE")
    print("=" * 70)

    # The run's model is one of the dataset's two decided models (run_all,
    # fed_models_for); there is no 'rf' default any more (R1, 2026-10-05).
    fed_model = config['fed_model']
    fed_seed = config.get('fed_seed', 0)
    kmodel = fed_to_kirby_model(fed_model)

    # The decided general budgets of THIS dataset (lhasa_herg 100-2000,
    # caco2 and logd 500-2000), read here rather than at import.
    budgets = [int(b) for b in federated_budgets(dataset_name)]
    # Every table reads one width so it reads as it always did: 100 where
    # the dataset's budgets include it, else its smallest decided budget
    # (caco2 and logd start at 500). Every width is in by_budget.
    headline_budget = 100 if 100 in budgets else budgets[0]
    # The outer fold count is ONE setting, config['n_splits'] (handoff 2.4
    # step 14); 1 gives a single grouped held-out split.
    n_folds = smoke_n_folds(int(config['n_splits']))

    reps = data['reps']
    labels = data['labels']
    smiles = data['smiles']

    # ── Phase 1: Setup ───────────────────────────────────────────────────
    print("\n--- Phase 1: Setup ---")

    # Binarize labels (skip if already binary). The Lhasa loader sets
    # data['is_binary'], but the regular load_data('herg', ...) path returns
    # 0/1 labels WITHOUT that flag — so also detect already-binary labels by
    # value (unique values ⊆ {0,1}). Without this, a threshold on already-0/1
    # labels collapses every sample to class 0 (degenerate).
    labels_binary, cutoff_note = binarise_labels(
        dataset_name, labels, is_binary=bool(data.get('is_binary')))
    n_active = int(labels_binary.sum())
    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{n_active} active, {len(labels) - n_active} inactive "
          f"({cutoff_note})")

    # Compute Morgan FPs (for valid_mask)
    fluid_df_all, valid_mask = smiles_to_fluid_df(
        smiles, labels_binary, pki_threshold=0.5)

    # Filter to valid molecules
    groups = data.get('groups')
    if not valid_mask.all():
        valid_idx = np.where(valid_mask)[0]
        reps = {r: arr[valid_idx] for r, arr in reps.items()}
        labels_binary = labels_binary[valid_idx]
        smiles = [smiles[i] for i in valid_idx]
        labels = labels[valid_idx]
        if groups is not None:
            groups = groups[valid_idx]
        # P0 fix: reindex augmentations too — otherwise subsequent indexing
        # with post-filter indices corrupts non-Lhasa runs.
        if data.get('augmentations'):
            data = {**data, 'augmentations': {
                k: v[valid_idx] for k, v in data['augmentations'].items()}}

    n_samples = len(labels_binary)
    # Step 8 cut augmentation on lhasa_herg, caco2 and logd, so run_all loads
    # none and this is None there; a recipe only uses it when its search
    # condition is not 'none'.
    augs = data.get('augmentations')

    # SPLIT — scaffold-group-aware outer folds.
    # P1 fix: hard-fail when groups unavailable. federated_helpers.py:107-110
    # silently sets groups=None, which would trigger a random split
    # (forbidden by feedback_no_random_splits.md).
    if groups is None or len(np.unique(groups)) < max(2, n_folds):
        raise RuntimeError(
            f"fed04 requires scaffold groups for {n_folds}-fold CV "
            f"(got groups={'None' if groups is None else f'{len(np.unique(groups))} unique'}). "
            "Random splits are forbidden — fix the loader to provide groups.")

    # The outer folds (2026-10-05): fed_outer_folds, seed 42, the same rows
    # as the per-fold search store. They no longer depend on fed_seed, which
    # used to mix which molecules were held out into the spread over seeds.
    folds = list(fed_outer_folds(reps, labels_binary, groups, n_folds,
                                 'classification'))

    # Written after every fold (dirty-run bug 12): one unit per (model, seed,
    # fold) in <dataset>__folds.partial.jsonl, read back on the next attempt.
    _cfg_ident = {k: v for k, v in config.items()
                  if k not in ('fed_model', 'fed_seed')}
    fold_ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name,
                           suffix='folds'),
        run_identity(EXPERIMENT_DIR, dataset_name, data, _cfg_ident,
                     alignment=ALIGNMENT_ID))

    by_slot_records = {slot: [] for slot in HYBRID_SLOTS}
    selection_by_fold = []
    search_failures = []
    measure_failures = []
    base_arms_by_fold = []

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_id + 1}/{n_folds}: "
              f"{len(train_idx)} train / {len(test_idx)} test ===")
        _key = dict(fed_model=str(fed_model), fed_seed=int(fed_seed),
                    fold=int(fold_id))
        if fold_ck.done(**_key):
            unit = fold_ck.get(**_key)
            print(f"  [partial] fold {fold_id}: read back from {fold_ck.path}")
        else:
            unit = {'selection': None, 'records': {s: [] for s in HYBRID_SLOTS},
                    'search_failures': [], 'measure_failures': [],
                    'base_arms': None}
            # ONE search on this fold's training rows, on the two-class labels
            # every arm below trains on, so the recipe is ranked on the labels
            # it is used with. The decided grid was measured on the continuous
            # labels; running it on the two-class labels is accepted
            # (Adelaide, 2026-10-05). metric 'auto' resolves the two-class
            # metric. Shared by every budget and by both models of the run.
            try:
                sel = fed_fold_selection(
                    reps, labels_binary, groups, train_idx=train_idx,
                    fold_i=fold_id, n_folds=n_folds,
                    dataset_name=dataset_name, budgets=budgets,
                    models=[kmodel], task_type='classification',
                    metric='auto', augmentations=augs)
            except Exception as e:
                traceback.print_exc()
                sel = {kmodel: None,
                       '_errors': [{'error': f'{type(e).__name__}: {e}',
                                    'traceback': traceback.format_exc()}]}
            msel = sel.get(kmodel)
            if msel is None:
                # No fallback recipe (R2/R6): the fold is recorded as failed
                # and skipped for this model.
                # 'retry_on_resume': this fold is NOT written to the fold
                # file, and a call with such a fold is not written to the
                # per-call file (run() below), so the next attempt runs the
                # search again (R6, 2026-10-05). A drive dropping for seconds
                # used to be stored as a finished fold and never retried
                # without --force-rerun.
                unit['search_failures'].append({
                    'dataset': dataset_name, 'fold': fold_id, 'model': kmodel,
                    'error': 'the per-fold search returned nothing for this '
                             'model',
                    'retry_on_resume': True,
                    'search_errors': convert_numpy(sel.get('_errors') or [])})
                print(f"  fold {fold_id}: no per-fold search result for "
                      f"{kmodel}; this fold is skipped and recorded")
            else:
                sel_rec = {
                    'fold': fold_id,
                    'model': kmodel,
                    'fed_model': fed_model,
                    'selection_scope': msel.get('selection_scope',
                                                'fold_train_rows'),
                    'search_metric': msel.get('search_metric'),
                    'search_lower_is_better': msel.get('search_lower_is_better'),
                    # The 2 best base representations of this fold's search.
                    # Fitted as two reference arms below (unit['base_arms'],
                    # 2026-10-05, R4).
                    'base': list(msel.get('base') or []),
                    'base_detail': convert_numpy(
                        (msel.get('base_detail') or {}).get('selected')),
                    'search_errors': convert_numpy(sel.get('_errors') or []),
                    'by_budget': {},
                }
                # The 2 base representations, whole, on all training rows.
                unit['base_arms'] = {
                    'fold': fold_id, 'model': kmodel,
                    'arms': _fit_base_arms(
                        base_names=sel_rec['base'], fed_model=fed_model,
                        fed_seed=fed_seed, reps=reps,
                        labels_binary=labels_binary, train_idx=train_idx,
                        test_idx=test_idx)}
                if len(sel_rec['base']) < 2:
                    unit['search_failures'].append({
                        'dataset': dataset_name, 'fold': fold_id,
                        'model': kmodel,
                        'error': f"the search gave {len(sel_rec['base'])} "
                                 f"base representation(s), 2 wanted"})
                for b in budgets:
                    bb = (msel.get('by_budget') or {}).get(b) or {}
                    hyb = list(bb.get('hybrids') or [])
                    # A recipe per slot, or the reason there is none. An
                    # entry without a recipe is a failed slot, recorded.
                    recipes = []
                    for slot, entry in zip(HYBRID_SLOTS, hyb):
                        try:
                            recipe = _recipe_record(entry, kmodel, fold_id,
                                                    b, slot)
                            recipe['_blueprint'] = entry['blueprint']
                            _mname = (recipe['search_metric']
                                      or msel.get('search_metric'))
                            if _mname not in BUILTIN_METRICS:
                                raise ValueError(
                                    f'search metric {_mname!r} has no '
                                    f'scorer in BUILTIN_METRICS')
                            recipe['_metric_fn'] = BUILTIN_METRICS[_mname][0]
                            recipe['_lower_is_better'] = bool(
                                recipe['search_lower_is_better'])
                            recipes.append((slot, recipe))
                        except Exception as e:
                            traceback.print_exc()
                            unit['measure_failures'].append({
                                'dataset': dataset_name, 'fold': fold_id,
                                'model': kmodel, 'slot': slot, 'budget': b,
                                'error': f'{type(e).__name__}: {e}',
                                'traceback': traceback.format_exc()})
                    sel_rec['by_budget'][str(b)] = {
                        'hybrids': [
                            convert_numpy({k: v for k, v in r.items()
                                           if not k.startswith('_')})
                            for _, r in recipes],
                        'n_hybrids': len(hyb),
                        'hybrids_skipped_identical_features': convert_numpy([
                            {k: v for k, v in e.items()
                             if k in ('label', 'score_mean', 'aug_condition')}
                            if isinstance(e, dict) else e
                            for e in bb.get('hybrids_skipped_identical_features')
                            or []]),
                    }
                    if len(hyb) < len(HYBRID_SLOTS):
                        unit['search_failures'].append({
                            'dataset': dataset_name, 'fold': fold_id,
                            'model': kmodel, 'budget': b,
                            'error': f'the search gave {len(hyb)} distinct '
                                     f'hybrid(s) at {b} features, '
                                     f'{len(HYBRID_SLOTS)} wanted'})
                    for slot, recipe in recipes:
                        print(f"\n########## fold {fold_id} {slot} at {b} "
                              f"features: {recipe['allocation_method']} / "
                              f"{recipe['importance_method']} / "
                              f"{recipe['internal_evaluator']} / "
                              f"{recipe['rep_count'] or 'all'} representations "
                              f"##########")
                        try:
                            rec = _measure(
                                recipe=recipe, fed_model=fed_model,
                                fed_seed=fed_seed, fold_id=fold_id,
                                train_idx=train_idx, test_idx=test_idx,
                                reps=reps, labels_binary=labels_binary,
                                groups=groups, fluid_df_all=fluid_df_all,
                                augs=augs)
                            unit['records'][slot].append(rec)
                        except Exception as e:
                            traceback.print_exc()
                            unit['measure_failures'].append({
                                'dataset': dataset_name, 'fold': fold_id,
                                'model': kmodel, 'slot': slot, 'budget': b,
                                'error': f'{type(e).__name__}: {e}',
                                'traceback': traceback.format_exc()})
                unit['selection'] = sel_rec
            # JSON-plain before it is stored, and the same plain copy is used
            # below, so a fold read back from the file and a fold computed now
            # are the same objects (cluster numbers become string keys).
            unit = json.loads(json.dumps(convert_numpy(unit), default=str))
            if not any(f.get('retry_on_resume')
                       for f in unit['search_failures']):
                fold_ck.put(unit, **_key)

        if unit.get('selection') is not None:
            selection_by_fold.append(unit['selection'])
        if unit.get('base_arms') is not None:
            base_arms_by_fold.append(unit['base_arms'])
        search_failures.extend(unit.get('search_failures') or [])
        measure_failures.extend(unit.get('measure_failures') or [])
        for slot in HYBRID_SLOTS:
            by_slot_records[slot].extend(
                (unit.get('records') or {}).get(slot) or [])

    # ── Aggregate across folds, per slot ────────────────────────────────
    by_slot = {}
    for slot in HYBRID_SLOTS:
        recs = by_slot_records[slot]
        if not recs:
            continue
        bb, agg = _summarise_slot(recs, budgets, headline_budget, n_folds)
        _print_slot(slot, bb, agg, headline_budget, n_folds)
        by_slot[slot] = {'by_budget': bb, 'folds': recs, 'aggregate': agg}

    # The 2 base-representation arms over folds, by rank in the fold's
    # search. The representation at a rank can differ between folds, so the
    # names per fold are listed beside the held-out MCC.
    base_aggregate = {}
    for rank in (1, 2):
        _arms = [a for u in base_arms_by_fold for a in (u.get('arms') or [])
                 if a.get('rank') == rank]
        base_aggregate[f'base{rank}'] = {
            'representation_per_fold': [a.get('representation')
                                        for a in _arms],
            'held_out_mcc': _spread([(a.get('held_out') or {}).get('mcc')
                                     for a in _arms]),
            'train_rows_mcc': _spread([(a.get('train_rows') or {}).get('mcc')
                                       for a in _arms]),
            'n_failed': sum(1 for a in _arms if 'error' in a),
        }
        print(f"  base{rank} (whole representation, all training rows), "
              f"held-out MCC: {_fmt(base_aggregate[f'base{rank}']['held_out_mcc'])}"
              f"  {base_aggregate[f'base{rank}']['representation_per_fold']}")

    if search_failures or measure_failures:
        print(f"\n  failures recorded: {len(search_failures)} from the "
              f"per-fold search, {len(measure_failures)} while measuring")

    results = {
        'experiment': EXPERIMENT_DIR,
        'dataset': dataset_name,
        # What the arms are scored in (run_all stamps the file with these).
        'metric_name': REPORTED_METRIC,
        'lower_is_better': REPORTED_LOWER_IS_BETTER,
        'selection_scope': 'fold_train_rows',
        'config': {
            'k': K,
            'classification_cutoff': cutoff_note,
            'fed_model': fed_model,
            'model': kmodel,
            'fed_seed': fed_seed,
            'budgets_swept': budgets,
            'headline_budget': headline_budget,
            'smooth_factors': SMOOTH_FACTORS,
            'n_samples': n_samples,
            'n_folds': n_folds,
            'fold_seed': FED_FOLD_SEED,
            'hybrid_slots': list(HYBRID_SLOTS),
            'alignment': ALIGNMENT_ID,
            'recipe_source': ('per-fold search on the training rows, two-class '
                              'labels, decided general grid, pruning off'),
        },
        # Per fold: base representations, and per budget the 2 hybrids with
        # allocation, importance, budget, evaluator, representation count,
        # augmentation condition and the training-row score.
        'selection': selection_by_fold,
        'search_failures': search_failures,
        'measure_failures': measure_failures,
        'by_slot': by_slot,
        # The fold's 2 best base representations as reference arms, per fold
        # and over folds (MCC on the held-out rows and on the training rows).
        'base_arms': base_arms_by_fold,
        'base_aggregate': convert_numpy(base_aggregate),
        'rule': (f'per outer fold, the 2 best hybrids of {kmodel} at each '
                 f'decided budget from a search on that fold\'s training '
                 f'rows, each run through every phase'),
    }
    # The first slot keeps the top-level keys every reader already uses.
    _first = next(iter(by_slot), None)
    results['headline_slot'] = _first
    if _first is not None:
        results['by_budget'] = by_slot[_first]['by_budget']
        results['folds'] = by_slot[_first]['folds']
        results['aggregate'] = by_slot[_first]['aggregate']
    return results


# ── One (model, seed) call saved as it finishes (2026-09-25) ────────────────
# federated_helpers.run_all calls run() once per (fed_model, fed_seed) and
# writes fed04_cluster_aware_importance/<dataset>.json only after the last.
# Each call's return value is written to <dataset>.partial.jsonl the moment it
# returns and read back on the next attempt (the steps of
# _checkpoint.resumable_run, written out below), keyed
# on the model, the seed and any keyword argument, under an identity that
# includes the input fingerprint, the plain config, the KIRBY_* settings and
# ALIGNMENT_ID. Inside a call, each fold is saved as it finishes too
# (<dataset>__folds.partial.jsonl). --force-rerun / KIRBY_FORCE_RERUN=1
# discards both files.
def run(data, config, output_dir, dataset_name, **kwargs):
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity)
    _key = {'fed_model': str(config['fed_model']),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    # resumable_run's own steps (same sidecar, same identity), except that a
    # call with a fold whose per-fold search failed is NOT stored
    # (2026-10-05, R6): the next attempt runs that call again, reads the
    # finished folds back from the fold file and repeats only the failed
    # search. Through resumable_run the failure was stored as a finished
    # call and never retried without --force-rerun.
    _cfg = {k: v for k, v in config.items() if k not in _key}
    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, _cfg,
                     alignment=ALIGNMENT_ID))
    if ck.done(**_key):
        print(f'  [partial] {EXPERIMENT_DIR} {_key}: finished in an earlier '
              f'attempt, read back from {ck.path}')
        return ck.get(**_key)
    val = _run_one(data, config, output_dir, dataset_name, **kwargs)
    _retry = [f for f in (val.get('search_failures') or [])
              if f.get('retry_on_resume')]
    if _retry:
        print(f'  {len(_retry)} fold(s) whose per-fold search failed: this '
              f'call is not stored, so the next attempt retries them')
    else:
        ck.put(val, **_key)
    return val



if __name__ == '__main__':
    from experiments.federated.federated_helpers import fed_models_for
    parser = create_experiment_parser('Fed04: Cluster-Aware Importance')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    # One model of the dataset's two decided ones, as run_all would; run_all
    # is the path that runs both.
    config.setdefault('fed_model', fed_models_for(args.dataset)[0])
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, EXPERIMENT_DIR, args.dataset,
        metric_name=REPORTED_METRIC, lower_is_better=REPORTED_LOWER_IS_BETTER)
