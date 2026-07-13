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
  * folds are group-aware (make_cv_splitter, no random splits);
  * the hybrid's feature selection is done on the TRAIN fold only
    (build_fold_hybrid from a distance-aware blueprint) and PCA is fit on train
    only;
  * KMeans / the k-NN retrieval index are FIT ON TRAIN and the coherence /
    retrieval statistic is measured on the HELD-OUT fold's points
    (train cluster assignment for held-out points; queries answered by train
    neighbours, so there is no self-neighbour);
  * per-fold held-out scores are averaged across folds.
Both axes ALSO persist the per-fold held-out scores for BOTH arms (hybrid
candidate cells and single-rep baseline cells) in the CONTRACT-shaped blocks
`honest_rescore_clustering` and `honest_rescore_retrieval`. The two axes have
OPPOSITE metric directions (coherence higher-better, retrieval-cost
lower-better), so they cannot share one `honest_rescore` block (that helper
takes a single `lower_is_better`); hence two separate, identically-structured
blocks. A reader feeds each block straight to experiments._shared.honest_rescore.

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
from sklearn.neighbors import NearestNeighbors
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_fitted_pool_hybrid, extract_top3_base_reps,
    compute_feature_importance, get_domain_metric,
    make_cv_splitter, build_fold_hybrid, get_best_for_downstream,
    honest_rescore,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)


# =========================================================================
# LOCAL HELD-OUT STATISTICS (unsupervised — the shared honest_rescore helper
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
    points, etc.) — the honest_rescore helper skips NaN folds.
    """
    if n_clusters >= len(y_tr):
        return float('nan')
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_tr_s)
    va_lab = km.predict(X_va_s)

    if is_clf:
        if (y_va < 0).any():
            return float('nan')
        purities = []
        for c in range(n_clusters):
            mask = va_lab == c
            if mask.sum() >= 1:
                counts = np.bincount(y_va[mask].astype(int))
                purities.append(counts.max() / mask.sum())
        return float(np.mean(purities)) if purities else float('nan')

    y_std = np.std(y_va)
    if y_std < 1e-10:
        return float('nan')
    stds = []
    for c in range(n_clusters):
        mask = va_lab == c
        if mask.sum() >= 2:
            stds.append(np.std(y_va[mask]))
    if not stds:
        return float('nan')
    return float(1.0 - np.mean(stds) / y_std)


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
    nn = NearestNeighbors(n_neighbors=k_actual)
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
    search_speed = config.get('search_speed', 'full')
    n_folds = config['n_splits']
    budget = kwargs.get('budget', 100)
    k_retrieval = kwargs.get('k_retrieval', 5)
    n_clusters_list = kwargs.get('n_clusters_list', [3, 5, 10])
    augmentations = data.get('augmentations')

    # ── Distance-aware hybrid search (on-demand, disk-cached) ──
    # We only need the winning blueprint; the hybrid MATRIX is rebuilt per fold
    # (train-only) via build_fold_hybrid so no full-data selection leaks in.
    print(f"  Running distance-aware hybrid search "
          f"(evaluators={DISTANCE_EVALUATORS}, budgets={DISTANCE_BUDGETS})...")
    blueprint = None
    hybrid_error = None
    top3 = None
    try:
        _X_unused, _fi_unused, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=None,
            augmentations=augmentations,
            speed=search_speed,
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True)
        best_knn = get_best_for_downstream(search, 'knn')
        if best_knn and 'blueprint' in best_knn:
            blueprint = best_knn['blueprint']
            # 2026-04-21 cache-collision fix: trust the pool helper's
            # aug_condition over the (possibly stale) blueprint field.
            if _meta is not None and 'aug_condition' in _meta:
                blueprint.augmentation_strategy = _meta['aug_condition']
            print(f"  Best kNN config: alloc={blueprint.allocation_method}, "
                  f"imp={blueprint.importance_method}, "
                  f"ev={blueprint.internal_evaluator}, budget={blueprint.budget}")
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        # Record the failure explicitly so a missing hybrid arm is
        # distinguishable from a dataset that was never run.
        hybrid_error = f"{type(e).__name__}: {e}"
        print(f"  Hybrid search failed: {hybrid_error}")

    # ── Group-aware CV folds (shared by every space + both axes) ──
    sample_X = next(iter(reps.values()))
    try:
        splitter, use_groups = make_cv_splitter(n_folds, groups,
                                                task_type=task_type)
        folds = list(splitter.split(sample_X, labels, groups=groups))
    except Exception as e:
        # No random-split fallback (feedback_no_random_splits.md). Bail out
        # cleanly rather than crash the batch.
        print(f"  CV split failed: {type(e).__name__}: {e}")
        out = {'config': {'budget': budget, 'k_retrieval': k_retrieval,
                          'n_clusters_list': n_clusters_list, 'n_folds': n_folds},
               'cv_error': f"{type(e).__name__}: {e}"}
        if hybrid_error is not None:
            out['hybrid_error'] = hybrid_error
        return out

    sorted_reps = sorted(reps.keys())
    sorted_augs = sorted(augmentations.keys()) if augmentations else []

    # Per-fold held-out score accumulators.
    #   clustering_fold[space][n_clusters] = [held-out coherence per fold]
    #   retrieval_fold[space]['unweighted'/'weighted'] = [held-out cost per fold]
    clustering_fold = {}
    retrieval_fold = {}

    def _acc_clu(space, nc, val):
        clustering_fold.setdefault(space, {}).setdefault(nc, []).append(val)

    def _acc_ret(space, cond, val):
        retrieval_fold.setdefault(
            space, {'unweighted': [], 'weighted': []})[cond].append(val)

    print(f"\n  Held-out CV ({len(folds)} folds, group-aware)")
    for fold_i, (tr, va) in enumerate(folds):
        y_tr, y_va = labels[tr], labels[va]

        # Assemble this fold's (train, val) matrices for every space.
        fold_spaces = {}
        for rname in sorted_reps:
            fold_spaces[f'rep_{rname}'] = (reps[rname][tr], reps[rname][va])

        # PCA-matched — fit on TRAIN only.
        try:
            Xc_tr = np.hstack([reps[r][tr] for r in sorted_reps]
                              + [augmentations[a][tr] for a in sorted_augs])
            Xc_va = np.hstack([reps[r][va] for r in sorted_reps]
                              + [augmentations[a][va] for a in sorted_augs])
            pca_dim = min(budget, Xc_tr.shape[1], len(tr) - 1)
            if pca_dim >= 2:
                sc = StandardScaler()
                pca = PCA(n_components=pca_dim, random_state=42)
                Xp_tr = pca.fit_transform(sc.fit_transform(Xc_tr))
                Xp_va = pca.transform(sc.transform(Xc_va))
                fold_spaces['pca_matched'] = (Xp_tr, Xp_va)
        except Exception as e:
            print(f"    fold {fold_i}: PCA failed: {type(e).__name__}: {e}")

        # Hybrid — feature selection on TRAIN only, applied to val.
        if blueprint is not None:
            try:
                aug_tr = ({a: augmentations[a][tr] for a in sorted_augs}
                          if sorted_augs else None)
                aug_va = ({a: augmentations[a][va] for a in sorted_augs}
                          if sorted_augs else None)
                Xh_tr, Xh_va, _feat_info = build_fold_hybrid(
                    {r: reps[r][tr] for r in sorted_reps},
                    {r: reps[r][va] for r in sorted_reps},
                    y_tr, blueprint,
                    augmentations_train=aug_tr, augmentations_val=aug_va)
                fold_spaces['hybrid'] = (Xh_tr, Xh_va)
            except Exception as e:
                print(f"    fold {fold_i}: hybrid build failed: "
                      f"{type(e).__name__}: {str(e)[:80]}")

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
                _acc_clu(sname, nc,
                         _heldout_coherence(Xtr_s, y_tr, Xva_s, y_va, nc, is_clf))

            _acc_ret(sname, 'unweighted',
                     _heldout_retrieval(Xtr_s, y_tr, Xva_s, y_va,
                                        k_retrieval, is_clf))

            # Importance-weighted retrieval — hybrid only, weights from TRAIN.
            if sname == 'hybrid':
                try:
                    imp = compute_feature_importance(
                        {'hybrid': Xtr_s}, y_tr,
                        method='lightgbm_gain').get('hybrid',
                                                    np.ones(Xtr_s.shape[1]))
                    w = np.sqrt(np.maximum(imp, 0))
                    w = w / (w.sum() + 1e-10)
                    _acc_ret(sname, 'weighted',
                             _heldout_retrieval(Xtr_s * w, y_tr, Xva_s * w, y_va,
                                                k_retrieval, is_clf))
                except Exception as e:
                    print(f"    fold {fold_i}: weighted retrieval failed: "
                          f"{type(e).__name__}: {e}")
                    _acc_ret(sname, 'weighted', float('nan'))

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
    }
    if top3:
        results['top3_base_reps'] = list(top3)
    if hybrid_error is not None:
        results['hybrid_error'] = hybrid_error

    # ── Legacy summary (kept for downstream readers) — now on HELD-OUT means ──
    # Top-3 baseline reps per axis, and the hybrid-vs-best-rep win booleans.
    # These still use same-fold selection of the best rep, so they carry mild
    # optimism; the fully-honest number is the honest_rescore block below.
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

    if 'hybrid' in clustering_results and 'hybrid' in retrieval_results:
        h_coh = max((v['label_coherence']
                     for v in clustering_results['hybrid'].values()
                     if np.isfinite(v['label_coherence'])), default=float('nan'))
        h_ret = retrieval_results['hybrid']['unweighted']['mean_delta_y']
        best_rep_ret_name, best_rep_ret_val = (top3_retrieval[0]
                                               if top3_retrieval else ('?', 1e10))
        best_rep_coh_name, best_rep_coh_val = (top3_clustering[0]
                                               if top3_clustering else ('?', 0))
        results['summary'] = {
            'hybrid_coherence': h_coh,
            'hybrid_retrieval': h_ret,
            'best_rep_coherence': (best_rep_coh_name, best_rep_coh_val),
            'best_rep_retrieval': (best_rep_ret_name, best_rep_ret_val),
            'hybrid_clustering_wins': bool(h_coh > best_rep_coh_val),
            'hybrid_retrieval_wins': bool(h_ret < best_rep_ret_val),
            'metrics_agree': ((h_coh > best_rep_coh_val)
                              == (h_ret < best_rep_ret_val)),
        }
        print(f"\n  Hybrid (held-out): coherence={h_coh:.4f} vs "
              f"best_rep={best_rep_coh_val:.4f} "
              f"({'win' if h_coh > best_rep_coh_val else 'loss'}), "
              f"retrieval={h_ret:.4f} vs best_rep={best_rep_ret_val:.4f} "
              f"({'win' if h_ret < best_rep_ret_val else 'loss'})")

    # ── Honest leave-one-fold-out re-score blocks (CONTRACT shape) ──
    # Two axes with OPPOSITE directions cannot share one honest_rescore block
    # (the helper takes a single lower_is_better), so each axis gets its own
    # identically-structured block. Each hybrid n_clusters / retrieval condition
    # is a candidate CELL and each single-rep (space x n_clusters / space) is a
    # baseline CELL, so honest_rescore re-picks the winning cell per held-out
    # fold instead of trusting the same-fold argmax. Single grouping model per
    # axis by design ('kmeans' / 'knn'). A reader feeds each block straight to
    # experiments._shared.honest_rescore.
    clu_candidates = []
    for nc in n_clusters_list:
        fs = clustering_fold.get('hybrid', {}).get(nc)
        if fs and _has_finite(fs):
            clu_candidates.append({'name': f'hybrid_nc{nc}',
                                   'fold_scores': [float(s) for s in fs]})
    clu_baselines = {}
    for sname, ncmap in clustering_fold.items():
        if not sname.startswith('rep_'):
            continue
        for nc, fs in ncmap.items():
            if fs and _has_finite(fs):
                clu_baselines[f'{sname}|nc{nc}'] = [float(s) for s in fs]

    ret_candidates = []
    h_ret_fold = retrieval_fold.get('hybrid')
    if h_ret_fold:
        if _has_finite(h_ret_fold['unweighted']):
            ret_candidates.append({
                'name': f'hybrid_unweighted_k{k_retrieval}',
                'fold_scores': [float(s) for s in h_ret_fold['unweighted']]})
        if _has_finite(h_ret_fold['weighted']):
            ret_candidates.append({
                'name': f'hybrid_weighted_k{k_retrieval}',
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
    results['honest_rescore_clustering'] = {
        'lower_is_better': False,
        'candidates_by_model': ({'kmeans': clu_candidates}
                                if clu_candidates else {}),
        'baselines_by_model': ({'kmeans': clu_baselines}
                               if clu_baselines else {}),
    }
    results['honest_rescore_retrieval'] = {
        'lower_is_better': True,
        'candidates_by_model': ({'knn': ret_candidates}
                                if ret_candidates else {}),
        'baselines_by_model': ({'knn': ret_baselines}
                               if ret_baselines else {}),
    }

    # Reuse the shared pure-arithmetic helper (no ML) to report the honest
    # hybrid-vs-best-single-rep delta inline for each axis.
    for axis, block, better in (
            ('clustering', results['honest_rescore_clustering'], 'higher'),
            ('retrieval', results['honest_rescore_retrieval'], 'lower')):
        try:
            hr = honest_rescore(block['candidates_by_model'],
                                block['baselines_by_model'],
                                block['lower_is_better'])
            results[f'honest_rescore_{axis}_result'] = hr
            model_key = 'kmeans' if axis == 'clustering' else 'knn'
            m = hr.get(model_key)
            if m is not None:
                flag = ' [single-fold, NOT honest]' if m.get('single_fold') else ''
                print(f"  Honest LOFO re-score ({axis}, {better}-better){flag}: "
                      f"hybrid={m['hybrid_honest']:.4f}, "
                      f"best_single={m['best_single_honest']:.4f}, "
                      f"delta={m['delta']:+.4f} (positive=hybrid better)")
        except Exception as e:
            print(f"  [warn] honest_rescore ({axis}) failed: "
                  f"{type(e).__name__}: {e}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S13: Clustering & Retrieval')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's13_clustering_retrieval', args.dataset)
