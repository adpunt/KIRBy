"""Shared configuration and runner for per-dataset similarity experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from the pipeline readiness tiers.
Cross-dataset experiment (S09) stays in its own file and consumes
the per-dataset JSON outputs.

Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search.
Rep lists reused from complexity_helpers.DATASET_REPS.
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist as scipy_cdist

from kirby.similarity.distance_config import scipy_metric as scipy_distance_metric


def _pair_distances(Xs, pi, pj, block=8192):
    """Distance between row pi[t] and row pj[t], under --distance-metric.

    One distance per requested pair. It used to take the diagonal of a cdist
    call over a block of rows against itself, so a block of 4,096 pairs cost
    4,096 x 4,096 = 16.8 million distances to keep 4,096 of them: 4,095 of
    every 4,096 were computed and thrown away. That is why the callers sample
    5,000 pairs — at that cost 5,000 is already 17.6 million distances — and
    5,000 pairs is not enough to separate two representations whose Spearman
    differs by 0.01.

    Each of the four metrics the suite sweeps has a row-wise form, so there is
    nothing to compute off the diagonal:

      manhattan        sum |a - b|
      euclidean        sqrt(sum (a - b)^2)
      minkowski p=0.5  (sum |a - b|^0.5)^2
      cosine           1 - (a.b) / (|a| |b|)
    """
    import numpy as _np
    m, kw = scipy_distance_metric()
    out = _np.empty(len(pi), dtype=float)
    for start in range(0, len(pi), block):
        end = min(start + block, len(pi))
        a = _np.asarray(Xs[pi[start:end]], dtype=float)
        b = _np.asarray(Xs[pj[start:end]], dtype=float)
        if m == 'cityblock':
            out[start:end] = _np.abs(a - b).sum(axis=1)
        elif m == 'euclidean':
            out[start:end] = _np.sqrt(((a - b) ** 2).sum(axis=1))
        elif m == 'minkowski':
            p = float(kw.get('p', 2))
            out[start:end] = (_np.abs(a - b) ** p).sum(axis=1) ** (1.0 / p)
        elif m == 'cosine':
            num = (a * b).sum(axis=1)
            den = _np.linalg.norm(a, axis=1) * _np.linalg.norm(b, axis=1)
            # scipy returns 0 for a pair of all-zero rows; match it rather
            # than emitting nan into a Spearman.
            out[start:end] = _np.where(den > 0, 1.0 - num / _np.where(den > 0, den, 1.0), 0.0)
        else:
            # A metric added to distance_config without a row-wise form here.
            # Fall back to the diagonal rather than returning a wrong number.
            D = scipy_cdist(a, b, metric=m, **kw)
            out[start:end] = _np.diagonal(D)
    return out

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy,
    load_experiment_results, should_skip_experiment,
    experiment_result_is_empty,
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    input_fingerprint, fingerprint_mismatch,
)

# Reuse dataset rep configs — single source of truth
from experiments.complexity.complexity_helpers import DATASET_REPS


# =============================================================================
# PER-FOLD SELECTION AND kNN SETTINGS, SHARED BY EVERY SIMILARITY EXPERIMENT
# =============================================================================
# HANDOFF_EXPERIMENT_ALIGNMENT.md Part 2, chat 4 (2026-10-05). Every similarity
# experiment reads its folds, its representations, its hybrids and its kNN
# settings through the functions below, so the twelve cannot drift apart again.
#
# What each rule is, and where it was decided:
#   folds        sim_outer_folds: kirby.find_best_hybrid.nested_cv_outer_folds,
#                count from config['n_splits'] (handoff 2.4 step 14). Never a
#                literal. Seed SIM_FOLD_SEED = 42, the seed every per-fold
#                search is filed under (ocv{n}f{i}s42), so the noise suite and
#                this suite share one search per fold through the result store.
#   models       the dataset's two decided models (DECISIONS.md Step 1); knn
#                only through the distance search, where Step 9 runs it (R1).
#   hybrids      per outer fold, from searches on that fold's TRAINING rows
#                (R2): 2 base + 2 general hybrids per decided model
#                (fold_standard_selection, R4), a second hybrid with the same
#                selected features as the first skipped (2.3 item 8).
#   neighbour    an experiment that measures something with neighbours or
#   spaces       distances also gets (2.3 item 1): knn's top hybrid of the
#                distance search ('hybrid__distance1') and the top hybrid of the
#                similarity-objective search ('hybrid_similarity'), both per
#                fold, both on the decided grid (R3).
#   single-rep   knn's 2 best single representations, ranked by the per-fold
#   neighbour    distance search on the fold's training rows (Part 1 step 5,
#   arms         axis 1). The hybrid pool is never cut: pass the full
#                data['reps'] to anything that builds a hybrid.
#   kNN axes     knn_axes(): KNN_SEARCH_SPACE_RECORD.md axes 2, 4, 5, 6 --
#                distances per dataset group, 5/10/20 neighbours, rank and
#                inverse_square counting, scaffold group NOT excluded. Read from
#                tests/experiments/noise/similarity_space.py, the noise suite's
#                one copy, so both suites sweep the same values.
#   scaling      kirby.scaling.FeatureScaler fitted on the training rows: 0/1
#                features untouched, the rest standardised (2.3 item 17).
SIM_FOLD_SEED = 42

# Kept for any caller still importing the names; the per-fold function below
# uses them.
SIMILARITY_BASE_REP_COUNT = 2
SIMILARITY_BASE_REP_MODEL = 'knn'


def sim_outer_folds(data, config, seed=SIM_FOLD_SEED):
    """The outer (train_idx, test_idx) splits every similarity experiment uses.

    The count is config['n_splits'] (set by get_config / run_all from the one
    fold setting). A count of 1 is one held-out split, not 1-fold CV.
    """
    from kirby.find_best_hybrid import nested_cv_outer_folds
    n = int(config['n_splits'])
    groups = data.get('groups')
    return nested_cv_outer_folds(
        data['reps'], np.asarray(data['labels']),
        None if groups is None else np.asarray(groups), n, seed,
        data.get('task_type', 'regression'))


def fold_spread_note(values):
    """'one fold, no spread' when there is a single fold; else None. For any
    summary that would take a spread over folds."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return 'one fold, no spread' if len(vals) < 2 else None


def knn_axes(dataset_name, n_molecules):
    """The decided kNN settings for this dataset (KNN_SEARCH_SPACE_RECORD.md).

    Returns {'group', 'metrics', 'ks', 'schemes', 'scaffold_group_excluded',
    'k_max'}. metrics is axis 2 for the dataset's group (full_knn: manhattan,
    cosine, minkowski_p0.5; fast_knn: manhattan, cosine); ks is axis 4 (5, 10,
    20); schemes is axis 5 (rank, inverse_square); the scaffold group is never
    excluded (axis 6). The process-wide KIRBY_DISTANCE_METRIC is NOT read: it
    names one distance, and the record sweeps the group's list.
    """
    from experiments.noise import similarity_space as ss
    group = ss.knn_group(dataset_name, n_molecules)
    return {
        'group': group,
        'metrics': list(ss.GROUP_METRICS[group]),
        'ks': list(ss.SWEPT_KS),
        'schemes': list(ss.SWEPT_NEIGHBOUR_SCHEMES),
        'k_max': int(max(ss.SWEPT_KS)),
        'scaffold_group_excluded': False,
        'source': 'KNN_SEARCH_SPACE_RECORD.md axes 2, 4, 5, 6 (2026-10-05)',
    }


def scale_fold(X_train, X_other=None):
    """FeatureScaler fitted on the training rows; returns (train, other).

    Held-out values beyond the training range are cut to it (owner,
    2026-10-05, eda240c), so one extreme held-out molecule cannot dominate a
    distance.
    """
    from kirby.scaling import FeatureScaler
    sc = FeatureScaler().fit(np.asarray(X_train, dtype=np.float64))
    A = sc.transform(np.asarray(X_train, dtype=np.float64))
    if X_other is None:
        return A, None
    B = sc.transform(np.asarray(X_other, dtype=np.float64))
    B = np.clip(B, A.min(axis=0), A.max(axis=0))
    return A, B


def knn_lists(X_train, X_query=None, *, metric, k_max=20):
    """Each query row's k_max nearest training rows under one decided distance.

    Returns (idx, dist), each (n_query, k). With X_query None the query rows
    ARE the training rows and each row's own column is excluded by index (not
    by dropping the first column, which under ties can be a twin: dirty-run
    bug 7). Ties are broken by column index. No scaffold group is excluded
    (axis 6). Computed in row chunks, so memory stays bounded.
    """
    from experiments.noise import similarity_space as ss
    A = np.asarray(X_train, dtype=np.float64)
    self_rows = X_query is None
    Q = A if self_rows else np.asarray(X_query, dtype=np.float64)
    k = int(min(k_max, len(A) - (1 if self_rows else 0)))
    if k < 1:
        raise ValueError(f'knn_lists: {len(A)} training rows leave no neighbour')
    return ss._lists_from_distance(
        lambda s, e: ss.feature_distance(Q[s:e], A, metric),
        len(Q), len(A), k, self_rows=self_rows)


def knn_from_lists(y_train, idx, dist, *, k, scheme, is_clf):
    """The neighbours' label for each query row, from a list cut to k.

    Regression: the weighted mean. Classification: the class with the largest
    summed weight, plus each class's share of the weight (for AUC, which must
    be read off a score, not a 0/1 call). Returns (pred, proba, classes);
    proba and classes are None on regression.
    """
    from experiments.noise import similarity_space as ss
    y = np.asarray(y_train)
    i, d = np.asarray(idx)[:, :k], np.asarray(dist)[:, :k]
    w = np.asarray(ss.neighbour_weights(d, scheme), dtype=np.float64)
    # A query row whose distance is undefined to every training row (an
    # all-zero row under cosine is +inf from all of them) gets weight 0 on
    # every neighbour. Its k listed neighbours (lowest row numbers, by the tie
    # rule) then count equally, rather than the prediction being 0/0.
    dead = ~(w.sum(1) > 0)
    if dead.any():
        w[dead] = 1.0
    nb = y[i]
    if not is_clf:
        return (w * nb).sum(1) / w.sum(1), None, None
    classes = np.unique(y)
    votes = np.stack([(w * (nb == c)).sum(1) for c in classes], axis=1)
    tot = votes.sum(1, keepdims=True)
    proba = np.divide(votes, tot, out=np.full_like(votes, 1.0 / len(classes)),
                      where=tot > 0)
    return classes[votes.argmax(1)], proba, classes


def score_prediction(y_true, pred, proba, classes, metric_name, metric_fn):
    """The dataset's own metric. AUC is taken from the class shares, never
    from 0/1 calls (results audit, 473855a)."""
    if metric_name == 'auc' and proba is not None:
        from sklearn.metrics import roc_auc_score
        y_true = np.asarray(y_true)
        try:
            if len(classes) == 2:
                return float(roc_auc_score(y_true == classes[1], proba[:, 1]))
            return float(roc_auc_score(y_true, proba, multi_class='ovr',
                                       labels=list(classes)))
        except ValueError:
            return float('nan')
    return float(metric_fn(y_true, pred))


def knn_sweep(X_train, y_train, X_val, y_val, *, axes, metric_name, metric_fn,
              is_clf, scale=True):
    """Predict each held-out row from its nearest training rows, at every
    decided setting: one row per (distance, neighbour count, counting scheme).

    One neighbour list per distance serves every count and scheme. Features
    are scaled with scale_fold first unless scale=False (pass False for a
    matrix that is already scaled, or for raw 0/1 bits on purpose).
    """
    if scale:
        X_train, X_val = scale_fold(X_train, X_val)
    rows = []
    for metric in axes['metrics']:
        idx, dist = knn_lists(X_train, X_val, metric=metric,
                              k_max=axes['k_max'])
        for k in axes['ks']:
            if k > idx.shape[1]:
                rows.extend({'distance': metric, 'k': int(k), 'scheme': s,
                             'score': None,
                             'note': f'only {idx.shape[1]} training rows'}
                            for s in axes['schemes'])
                continue
            for scheme in axes['schemes']:
                pred, proba, classes = knn_from_lists(
                    y_train, idx, dist, k=k, scheme=scheme, is_clf=is_clf)
                rows.append({'distance': metric, 'k': int(k), 'scheme': scheme,
                             'score': score_prediction(y_val, pred, proba,
                                                       classes, metric_name,
                                                       metric_fn)})
    return rows


def best_knn_setting(rows, lower_is_better):
    """The best row of a knn_sweep, by its score in the metric's direction."""
    ok = [r for r in rows if r.get('score') is not None
          and np.isfinite(r['score'])]
    if not ok:
        return None
    return (min if lower_is_better else max)(ok, key=lambda r: r['score'])


# A fault in the code (a name, a signature, a key) is raised, never recorded as
# a search that "failed": a run that writes every arm as unavailable and exits 0
# hides it. Seen 2026-10-05 when fold_searches lost its memo helper.
_CODE_FAULTS = (NameError, AttributeError, TypeError, ImportError, SyntaxError)


def fold_similarity_selection(data, dataset_name, *, train_idx, fold_i,
                              n_outer_folds, neighbour_spaces,
                              seed=SIM_FOLD_SEED, verbose=False):
    """Everything a similarity experiment selects for one outer fold, chosen
    on that fold's training rows only.

    neighbour_spaces: True for an experiment that measures something with
        neighbours or distances (2.3 item 1: s02, s04, s05, s06, s08, s10, s12,
        s13). It adds knn's 2 best single representations, knn's top hybrid of
        the distance search and the top hybrid of the similarity search.

    Returns {
      'fold', 'n_outer_folds', 'selection_scope': 'fold_train_rows',
      'models': {m: {'base': [2 names], 'base_detail', 'hybrids': [2 entries,
                 each with 'blueprint', 'aug_condition', 'label',
                 'score_mean', ...], 'hybrids_skipped_identical_features'}
                 or None when that model's search failed},
      'knn_base': [2 names] or [],           (neighbour_spaces only)
      'knn_base_detail',
      'distance_hybrid': entry or None,      (knn's top, distance search)
      'similarity_hybrid': entry or None,    (top over every model scored)
      'unavailable': {what: why},
    }
    Rebuild any entry with build_fold_entry().
    """
    from experiments._shared import (
        fold_standard_selection, fold_searches, top_hybrids_across_models)
    from experiments.decided_grid import is_decided
    from experiments.dataset_policy import similarity_objective_undefined

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    task_type = data.get('task_type', 'auto')
    common = dict(train_idx=train_idx, fold_i=fold_i,
                  n_outer_folds=n_outer_folds, dataset_name=dataset_name,
                  augmentations=data.get('augmentations'),
                  task_type=task_type, seed=seed, verbose=verbose)
    out = {'fold': int(fold_i), 'n_outer_folds': int(n_outer_folds),
           'selection_scope': 'fold_train_rows', 'models': {},
           'knn_base': [], 'knn_base_detail': None,
           'distance_hybrid': None, 'similarity_hybrid': None,
           'unavailable': {}}

    try:
        sel = fold_standard_selection(reps, labels, groups, **common)
    except _CODE_FAULTS:
        raise
    except Exception as exc:
        # Recorded once for the fold; the caller decides whether the fold
        # can go on without the general hybrids.
        sel = {}
        out['unavailable']['general_search'] = (
            f'the per-fold general search raised {type(exc).__name__}: {exc}')
    for m, s in sel.items():
        if s is None:
            out['models'][m] = None
            out['unavailable'][f'general/{m}'] = (
                "the per-fold general search failed for this model")
            continue
        out['models'][m] = {k: s.get(k) for k in (
            'base', 'base_detail', 'hybrids',
            'hybrids_skipped_identical_features')}

    if not neighbour_spaces:
        return out

    if is_decided(dataset_name, 'general_distance'):
        try:
            sd = fold_standard_selection(reps, labels, groups, distance=True,
                                         models=['knn'], n_hybrids=1,
                                         distinct_features=False, **common)
            k_sel = sd.get('knn')
        except _CODE_FAULTS:
            raise
        except Exception as exc:
            k_sel = None
            out['unavailable']['distance_search'] = (
                f'the per-fold distance search raised '
                f'{type(exc).__name__}: {exc}')
        if k_sel is not None:
            out['knn_base'] = list(k_sel['base'])
            out['knn_base_detail'] = k_sel['base_detail']
            if k_sel['hybrids']:
                out['distance_hybrid'] = k_sel['hybrids'][0]
            else:
                out['unavailable']['distance_hybrid'] = (
                    'the per-fold distance search scored no knn hybrid')
            if not out['knn_base']:
                out['unavailable']['knn_base'] = (
                    k_sel['base_detail'] or {}).get('shortfall_reason') or (
                    'the per-fold distance search scored no knn baseline')
        else:
            out['unavailable'].setdefault(
                'distance_search', 'the per-fold distance search failed')
    else:
        out['unavailable']['distance_search'] = (
            'DECISIONS.md decides no distance search for this dataset')

    why = similarity_objective_undefined(dataset_name, labels, task_type)
    if why:
        out['unavailable']['similarity_hybrid'] = why
    else:
        try:
            ss_ = fold_searches(reps, labels, groups, tier='general',
                                objective='similarity', **common)
            srch = ss_.get('none')
            if srch is None:
                out['unavailable']['similarity_hybrid'] = (
                    'the per-fold similarity search failed')
            else:
                th = top_hybrids_across_models(
                    srch, n=1, require_full=False,
                    context=f'{dataset_name} similarity fold {fold_i}')
                if th['selected']:
                    out['similarity_hybrid'] = dict(th['selected'][0],
                                                    aug_condition='none')
                else:
                    out['unavailable']['similarity_hybrid'] = (
                        'the per-fold similarity search scored no hybrid')
        except _CODE_FAULTS:
            raise
        except Exception as exc:
            out['unavailable']['similarity_hybrid'] = (
                f'the per-fold similarity search raised '
                f'{type(exc).__name__}: {exc}')
    if verbose and out['unavailable']:
        print(f"  fold {fold_i}: unavailable {out['unavailable']}")
    return out


def build_fold_entry(data, entry, *, train_idx, val_idx, model_factory=None,
                     metric_fn=None, lower_is_better=None, budget_override=None):
    """Rebuild one selected hybrid on the fold's training rows and apply it to
    the held-out rows. Returns (X_train, X_val, feat_info).

    The representation scores passed are the recipe's own (blueprint.rep_scores),
    which for a per-fold recipe were measured on this fold's training rows --
    never a whole-dataset search's (dirty-run bug 1). A geometry-objective
    recipe gets no model (fold_hybrid_scoring_kwargs).
    """
    from experiments._shared import build_fold_hybrid, fold_hybrid_scoring_kwargs
    tr, va = np.asarray(train_idx), np.asarray(val_idx)
    reps = data['reps']
    y = np.asarray(data['labels'])
    groups = data.get('groups')
    bp = entry['blueprint']
    augs = (data.get('augmentations')
            if (entry.get('aug_condition') or 'none') != 'none' else None)
    return build_fold_hybrid(
        {k: np.asarray(v)[tr] for k, v in reps.items()},
        {k: np.asarray(v)[va] for k, v in reps.items()},
        y[tr], bp,
        augmentations_train=(None if not augs else
                             {a: np.asarray(X)[tr] for a, X in augs.items()}),
        augmentations_val=(None if not augs else
                           {a: np.asarray(X)[va] for a, X in augs.items()}),
        budget_override=budget_override,
        groups=None if groups is None else np.asarray(groups)[tr],
        precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
        **fold_hybrid_scoring_kwargs(
            bp, model=(model_factory() if model_factory else None),
            metric_fn=metric_fn, lower_is_better=lower_is_better))


def entry_record(entry):
    """What a result file stores about one selected hybrid (handoff 2.4 step
    11): allocation, importance, budget in features, evaluator, representation
    count, augmentation condition, its training-row score."""
    if entry is None:
        return None
    bp = entry.get('blueprint')
    g = (lambda a: getattr(bp, a, None)) if bp is not None else (lambda a: None)
    return {
        'label': entry.get('label'),
        'allocation': entry.get('allocation_method') or g('allocation_method'),
        'importance': entry.get('importance_method') or g('importance_method'),
        'budget_features': entry.get('budget') or g('budget'),
        'evaluator': (entry.get('internal_evaluator')
                      or g('internal_evaluator')),
        'rep_count': entry.get('rep_count') or g('rep_count'),
        'aug_condition': entry.get('aug_condition'),
        'objective': g('objective'),
        'train_rows_score': entry.get('score_mean'),
        'feature_signature_found': entry.get('feature_signature') is not None,
    }


def neighbour_arm_reps(reps, search=None, *, n=None, model=None,
                       held_out_fold=None, dataset_name=None, verbose=True):
    """Superseded by fold_similarity_selection(...)['knn_base'], which ranks
    on a per-fold distance search. Kept so an old import does not break; it
    refuses a whole-dataset search (R2)."""
    raise RuntimeError(
        'neighbour_arm_reps read a whole-dataset search; use '
        "fold_similarity_selection(..., neighbour_spaces=True)['knn_base'].")
# =============================================================================
# SHARED SIMILARITY METRIC FUNCTIONS
# =============================================================================

def _is_nominal(y):
    """Detect if labels are nominal multi-class (>2 discrete classes).

    DEPRECATED for direct use — prefer passing is_clf=True explicitly.
    Only reliable on the FULL label array, not individual elements.
    """
    unique = np.unique(y)
    return len(unique) > 2 and np.allclose(y, y.astype(int))


def _pairwise_delta(y_i, y_j, is_clf=False):
    """Compute pairwise activity difference.

    Regression / binary: |y_i - y_j| (continuous difference).
    Classification (is_clf=True): 1{y_i != y_j} (binary indicator).
    """
    if is_clf:
        return (y_i != y_j).astype(float)
    return np.abs(y_i - y_j)


def _pairwise_delta_matrix(y, is_clf=False):
    """Compute full n×n pairwise activity difference matrix."""
    if is_clf:
        return (y[:, None] != y[None, :]).astype(float)
    return np.abs(y[:, None] - y[None, :])


def similarity_activity_correlation(D, y, n_sample=5000, is_clf=False):
    """Spearman correlation between pairwise distance and activity difference.

    Regression / binary: |delta_y| (continuous).
    Classification (is_clf): 1{y_i != y_j} — do nearby samples share a class?

    Positive = good (large distance -> large activity difference means the
    representation preserves the structure-activity relationship).
    Returns (rho, p_value).
    """
    n = len(y)
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)

    if n_pairs > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_pairs, n_sample, replace=False)
        triu_i, triu_j = triu_i[idx], triu_j[idx]

    dists = D[triu_i, triu_j]
    delta_y = _pairwise_delta(y[triu_i], y[triu_j], is_clf=is_clf)

    rho, p = spearmanr(dists, delta_y)
    return float(rho), float(p)


def sac_pair_sample(n, n_sample, scalable=False, seed=42):
    """The molecule pairs every distance-activity Spearman here is taken over.

    One place, so the dense path and the nearest-neighbour path draw the same
    way and the permutation null below can be built for the whole dataset
    instead of once per representation.
    """
    if scalable:
        return _sample_offdiag_pairs(n, n_sample, seed)
    pi, pj = np.triu_indices(n, k=1)
    if len(pi) > n_sample:
        idx = np.random.RandomState(seed).choice(len(pi), n_sample,
                                                 replace=False)
        pi, pj = pi[idx], pj[idx]
    return pi, pj


def permuted_delta_ranks(labels, pair_i, pair_j, is_clf=False, n_perm=200,
                         seed=42, dtype=np.float32):
    """Ranked activity difference of those pairs under n_perm label shuffles.

    Nothing in it depends on the representation: the pairs are the same for
    every representation and the labels are shuffled the same way, so this is
    the whole permutation null for a dataset, built once. The per-representation
    loop used to rebuild all of it inside every representation, which is what
    made 200 shuffles expensive enough to keep the pair count at 5,000.

    Same RandomState(seed) draw order as that loop, so the shuffles are the
    ones it drew.
    """
    from scipy.stats import rankdata
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    out = np.empty((n_perm, len(pair_i)), dtype=dtype)
    for t in range(n_perm):
        y_perm = rng.permutation(labels)
        out[t] = rankdata(_pairwise_delta(y_perm[pair_i], y_perm[pair_j],
                                          is_clf=is_clf))
    return out


def sac_permutation_p(pair_dists, perm_ranks, sac_rho):
    """How often a shuffled activity column reached this representation's rho.

    The top counts shuffles whose absolute Spearman reached the real absolute
    Spearman; the bottom is the number of shuffles. A Spearman IS a Pearson on
    ranks, and the distance ranks are the same for every shuffle, so the whole
    null is one matrix product rather than 200 spearmanr calls.
    """
    from scipy.stats import rankdata
    rd = rankdata(pair_dists).astype(np.float64)
    rd -= rd.mean()
    dn = float(np.sqrt((rd ** 2).sum()))
    P = perm_ranks.astype(np.float64)
    P = P - P.mean(axis=1, keepdims=True)
    pn = np.sqrt((P ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        rhos = (P @ rd) / (pn * dn)
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return float('nan'), rhos
    return float(np.mean(np.abs(rhos) >= abs(sac_rho))), rhos


def compute_sali(D, y, eps=1e-6, is_clf=False):
    """SALI: activity_diff / (1 - sim(i,j)), sim = 1/(1+d).

    Classification (is_clf): uses binary indicator (same/different class).
    Returns per-sample max SALI and summary stats.
    """
    n = len(y)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0)

    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)
    denom = 1.0 - sim + eps
    sali_matrix = delta_y / denom
    # Pairs with identical features (distance 0) are left out (owner,
    # 2026-10-05): SALI divides by 1 - similarity, which is 0 for them, so the
    # published definition does not cover them. Before this they scored the
    # activity difference x 1,000,000 and set the mean (abagym_Ang2: largest
    # SALI 661,000 in all 10 representations). Counted instead.
    _same = (np.asarray(D) == 0)
    np.fill_diagonal(_same, False)
    sali_matrix = np.where(_same, np.nan, sali_matrix)
    np.fill_diagonal(sali_matrix, 0)
    n_identical = int(_same.sum() // 2)
    n_identical_diff = int((_same & (delta_y > 0)).sum() // 2)

    per_sample_max = np.nanmax(sali_matrix, axis=1)
    per_sample_mean = np.nanmean(sali_matrix, axis=1)

    return {
        'mean': float(np.mean(per_sample_max)),
        'max': float(np.max(per_sample_max)),
        'std': float(np.std(per_sample_max)),
        'median': float(np.median(per_sample_max)),
        'per_sample_mean_avg': float(np.mean(per_sample_mean)),
        'n_identical_pairs': n_identical,
        'n_identical_pairs_with_activity_difference': n_identical_diff,
    }


def violation_rate(D, y, k=5, is_clf=False):
    """Fraction of k-NN pairs with large activity difference.

    Regression / binary: |delta_y| > median(|delta_y|).
    Classification (is_clf): fraction of k-NN that are a different class.
    """
    from kirby.utils.noise_mitigation_methods import get_k_neighbors_from_distances

    n = len(y)
    k = min(k, n - 1)

    indices, _ = get_k_neighbors_from_distances(D, k)

    if is_clf:
        # Multi-class: fraction of k-NN that belong to a different class
        n_violations = 0
        n_total = 0
        for i in range(n):
            for j_idx in range(k):
                j = indices[i, j_idx]
                if y[i] != y[j]:
                    n_violations += 1
                n_total += 1
        return float(n_violations / n_total) if n_total > 0 else 0.0

    # Regression / binary: threshold-based
    triu_i, triu_j = np.triu_indices(n, k=1)
    if len(triu_i) > 50000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(triu_i), 50000, replace=False)
        all_delta = np.abs(y[triu_i[sample_idx]] - y[triu_j[sample_idx]])
    else:
        all_delta = np.abs(y[triu_i] - y[triu_j])
    median_delta = np.median(all_delta)

    n_violations = 0
    n_total = 0
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            if np.abs(y[i] - y[j]) > median_delta:
                n_violations += 1
            n_total += 1

    return float(n_violations / n_total) if n_total > 0 else 0.0


# =============================================================================
# SCALABLE SIMILARITY METRICS (no full n x n matrix — for large n)
# =============================================================================
# The three metrics above build a full n x n distance matrix, which OOMs above
# ~15k samples. These siblings compute the SAME three scores without it:
#   - SAC samples a fixed set of pairs (it already only used 5000),
#   - violation_rate / SALI use each point's nearest neighbours (a k-NN index),
# both bounded-memory (sklearn NearestNeighbors chunks its queries). Metric and
# standardization match compute_distance_matrix, which since 2026-09-04 reads
# --distance-metric rather than being Euclidean by omission. These three had the
# distance written out by hand -- np.linalg.norm here, metric='euclidean' there --
# so they went on being Euclidean after the helper stopped being.

def _standardize(X):
    from kirby.scaling import FeatureScaler
    return FeatureScaler().fit_transform(X)


def _sample_offdiag_pairs(n, n_sample=5000, seed=42):
    """Random distinct off-diagonal (i != j) index pairs WITHOUT enumerating
    all O(n^2) of them — np.triu_indices itself OOMs above ~15k."""
    rng = np.random.RandomState(seed)
    pi = rng.randint(0, n, size=n_sample * 2)
    pj = rng.randint(0, n, size=n_sample * 2)
    mask = pi != pj
    return pi[mask][:n_sample], pj[mask][:n_sample]


def similarity_activity_correlation_scalable(X, y, n_sample=5000, is_clf=False,
                                             seed=42):
    """SAC over a random sample of pairs only (no full matrix). Returns
    (rho, p, pair_i, pair_j, pair_dists); the pairs+distances are returned so a
    label-permutation test can reuse them without recomputing distances."""
    n = len(y)
    pi, pj = _sample_offdiag_pairs(n, n_sample, seed)
    Xs = _standardize(X)
    dists = _pair_distances(Xs, pi, pj)
    delta_y = _pairwise_delta(y[pi], y[pj], is_clf=is_clf)
    rho, p = spearmanr(dists, delta_y)
    return float(rho), float(p), pi, pj, dists


def violation_rate_scalable(X, y, k=5, is_clf=False, n_sample=50000, seed=42):
    """violation_rate via an exact k-NN index (self excluded, matching
    get_k_neighbors_from_distances). Regression threshold from sampled pairs."""
    from experiments._shared import make_nearest_neighbors  # honours --distance-metric
    n = len(y)
    k = min(k, n - 1)
    Xs = _standardize(X)
    _, idx = make_nearest_neighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs)
    idx = idx[:, 1:]  # drop self (nearest is always self at distance 0)
    if is_clf:
        return float(np.mean(y[idx] != y[:, None]))
    pi, pj = _sample_offdiag_pairs(n, n_sample, seed)
    median_delta = float(np.median(np.abs(y[pi] - y[pj])))
    return float(np.mean(np.abs(y[idx] - y[:, None]) > median_delta))


def compute_sali_scalable(X, y, eps=1e-6, is_clf=False, block=2048):
    """SALI without holding the full n x n matrix: distances are computed in
    row-blocks and reduced to each point's per-sample max/mean on the fly.
    Numerically IDENTICAL to compute_sali (same formula, metric, and
    diagonal/mean convention) — O(block*n) memory instead of O(n^2). (Time is
    still O(n^2); the win here is purely memory, which is what OOMs.)"""
    n = len(y)
    Xs = _standardize(X)
    y = np.asarray(y)
    per_sample_max = np.empty(n)
    per_sample_mean = np.empty(n)
    # Identical-feature pairs left out and counted, as in compute_sali
    # (owner, 2026-10-05). Each unordered pair is seen twice, from each side.
    _n_same = 0
    _n_same_diff = 0
    for start in range(0, n, block):
        end = min(start + block, n)
        _m, _kw = scipy_distance_metric()
        Db = scipy_cdist(Xs[start:end], Xs, metric=_m, **_kw)  # block x n
        sim = 1.0 / (1.0 + Db)
        if is_clf:
            dy = (y[start:end, None] != y[None, :]).astype(float)
        else:
            dy = np.abs(y[start:end, None] - y[None, :])
        sali_b = dy / (1.0 - sim + eps)
        _same = (Db == 0)
        _same[np.arange(end - start), np.arange(start, end)] = False
        _n_same += int(_same.sum())
        _n_same_diff += int((_same & (dy > 0)).sum())
        sali_b = np.where(_same, np.nan, sali_b)
        # zero the self entries (row i, column i) — matches np.fill_diagonal(0)
        sali_b[np.arange(end - start), np.arange(start, end)] = 0.0
        per_sample_max[start:end] = np.nanmax(sali_b, axis=1)
        per_sample_mean[start:end] = np.nanmean(sali_b, axis=1)
    return {
        'mean': float(np.mean(per_sample_max)),
        'max': float(np.max(per_sample_max)),
        'std': float(np.std(per_sample_max)),
        'median': float(np.median(per_sample_max)),
        'per_sample_mean_avg': float(np.mean(per_sample_mean)),
        'n_identical_pairs': _n_same // 2,
        'n_identical_pairs_with_activity_difference': _n_same_diff // 2,
    }


def identify_cliff_compounds_editdist(sequences, y, max_edits=5, activity_k=2.0,
                                      is_clf=False, similar_percentile=None):
    """Identify activity cliff compounds using pairwise edit distance (Hamming).

    Representation-independent cliff definition for protein/antibody data.
    Uses sequence Hamming distance (on aligned sequences of equal length)
    or Levenshtein distance (if lengths differ).

    Definition: cliff = structurally similar but large activity difference
    (|delta_y| > activity_k * MAD(y)).

    Args:
        sequences: list of str (aligned sequences, e.g., heavy chains).
        y: array of activity values.
        max_edits: absolute edit-distance cutoff for "structurally similar"
            (used only when ``similar_percentile`` is None).
        activity_k: Multiplier for MAD-based activity threshold.
        is_clf: If True, cliff = structurally similar + different class.
        similar_percentile: if set (e.g. 5.0), define "structurally similar" as
            the closest ``similar_percentile``% of nonzero pairwise edit
            distances instead of the absolute ``max_edits``. This is
            distribution-relative: on deep-mutational-scan antibody data an
            absolute ``max_edits=5`` flags a huge fraction of pairs as similar
            (the deleted s07 saw contamination saturate near 1.0); a percentile keeps the
            similar set a bounded fraction regardless of domain. See
            COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
            If a list/tuple of candidate percentiles is passed, the expensive
            edit-distance matrix is built ONCE and every candidate is swept;
            the one whose cliff-compound prevalence is closest to
            ``TARGET_CLIFF_PCT`` (10%) is auto-selected (preferring candidates
            with prevalence > 0). The returned ``info`` then also carries
            ``'percentile_sweep'`` and ``'selected_percentile'``.

    Returns:
        cliff_mask: boolean array, True for cliff compounds.
        info: dict with thresholds, counts, per-compound cliff pair counts.
    """
    TARGET_CLIFF_PCT = 10.0
    sweep_candidates = None
    if isinstance(similar_percentile, (list, tuple)):
        sweep_candidates = list(similar_percentile)
    n = len(y)
    y = np.asarray(y)

    # Pairwise edit distance (Hamming if equal length, else Levenshtein)
    same_len = len(set(len(s) for s in sequences)) == 1
    if same_len:
        # Fast Hamming: vectorised character comparison
        arr = np.array([list(s) for s in sequences])
        edit_dist = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            edit_dist[i] = (arr != arr[i]).sum(axis=1)
    else:
        # True Levenshtein edit distance (slower)
        def _levenshtein(s1, s2):
            if len(s1) < len(s2):
                return _levenshtein(s2, s1)
            prev = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                curr = [i + 1]
                for j, c2 in enumerate(s2):
                    curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                    prev[j] + (0 if c1 == c2 else 1)))
                prev = curr
            return prev[-1]

        edit_dist = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                d = _levenshtein(sequences[i], sequences[j])
                edit_dist[i, j] = d
                edit_dist[j, i] = d

    # Activity difference
    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)

    if is_clf:
        activity_threshold = 0.5
        mad_y = 0.0
    else:
        mad_y = float(np.median(np.abs(y - np.median(y))))
        activity_threshold = activity_k * mad_y

    # "Structurally similar" pairs: absolute cutoff, or distribution-relative
    # (closest similar_percentile% of nonzero edit distances). The matrices
    # above (edit_dist, delta_y) are the expensive part; given any edit_cutoff
    # the cliff stats are cheap boolean ops, so we factor them into a closure
    # that can be swept over several candidate percentiles essentially for free.
    _triu = edit_dist[np.triu_indices(n, k=1)]
    _nz = _triu[_triu > 0]

    def _percentile_cutoff(pct):
        return float(np.percentile(_nz, pct)) if _nz.size else 0.0

    def _cliff_stats(edit_cutoff):
        structural_mask = (edit_dist <= edit_cutoff) & (edit_dist > 0)
        cliff_pair_mask = structural_mask & (delta_y > activity_threshold)
        cliff_mask = cliff_pair_mask.any(axis=1)
        cliff_pair_counts = cliff_pair_mask.sum(axis=1)
        n_cliff_pairs = int(cliff_pair_mask.sum()) // 2
        n_cliff_compounds = int(cliff_mask.sum())
        pct_cliff = float(100.0 * n_cliff_compounds / n)
        return cliff_mask, cliff_pair_counts, n_cliff_pairs, n_cliff_compounds, pct_cliff

    percentile_sweep = None
    selected_percentile = None
    if sweep_candidates is not None:
        # Build the expensive matrices once (already done), then sweep cheaply.
        percentile_sweep = {}
        for cand in sweep_candidates:
            _, _, _, _, pct_cliff = _cliff_stats(_percentile_cutoff(cand))
            percentile_sweep[cand] = pct_cliff
        # Auto-select: prevalence closest to the 10% target, preferring
        # candidates with prevalence > 0 (only fall back to including 0%
        # candidates if every candidate is 0%).
        positive = [c for c in sweep_candidates if percentile_sweep[c] > 0]
        pool = positive if positive else list(sweep_candidates)
        selected_percentile = float(
            min(pool, key=lambda c: abs(percentile_sweep[c] - TARGET_CLIFF_PCT)))
        edit_cutoff = _percentile_cutoff(selected_percentile)
    elif similar_percentile is not None:
        edit_cutoff = _percentile_cutoff(similar_percentile)
    else:
        edit_cutoff = float(max_edits)

    (cliff_mask, cliff_pair_counts, n_cliff_pairs,
     n_cliff_compounds, pct_cliff_compounds) = _cliff_stats(edit_cutoff)

    info = {
        'method': 'edit_distance',
        'max_edits': max_edits,
        'similar_percentile': similar_percentile,
        'edit_cutoff': edit_cutoff,
        'activity_k': activity_k,
        'activity_threshold': activity_threshold,
        'mad_y': mad_y,
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': pct_cliff_compounds,
        'cliff_pair_counts': cliff_pair_counts,
        'mean_edit_dist': float(edit_dist[np.triu_indices(n, k=1)].mean()),
        'median_edit_dist': float(np.median(edit_dist[np.triu_indices(n, k=1)])),
    }
    if percentile_sweep is not None:
        info['percentile_sweep'] = percentile_sweep
        info['selected_percentile'] = selected_percentile

    return cliff_mask, info


def identify_cliff_compounds(ecfp4, y, tanimoto_threshold=0.55, activity_k=2.0,
                             is_clf=False):
    """Identify activity cliff compounds using ECFP4 Tanimoto.

    Definition based on Bajorath standard for ECFP4 (Stumpfe & Bajorath 2019):
      - Structural similarity: Tanimoto(ECFP4) >= tanimoto_threshold (default 0.55)
      - Activity difference: |delta_y| > activity_k * MAD(y)

    Classification (is_clf): cliff = similar structure but different class.

    Returns:
        cliff_mask: boolean array, True for cliff compounds
        info: dict with thresholds, counts, per-compound cliff pair counts
    """
    n = len(y)

    # Tanimoto similarity from binary ECFP4 (Tanimoto = 1 - Jaccard distance)
    jaccard_dist = scipy_cdist(ecfp4, ecfp4, metric='jaccard')
    tanimoto_sim = 1.0 - jaccard_dist
    np.fill_diagonal(tanimoto_sim, 0)

    # Activity difference matrix
    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)

    if is_clf:
        # Multi-class: cliff = high similarity AND different class
        activity_threshold = 0.5  # indicator is 0 or 1
        mad_y = 0.0
    else:
        # Regression: adaptive threshold k * MAD(y)
        mad_y = float(np.median(np.abs(y - np.median(y))))
        activity_threshold = activity_k * mad_y

    # Cliff pair mask: high similarity AND large activity difference
    cliff_pair_mask = (tanimoto_sim >= tanimoto_threshold) & (delta_y > activity_threshold)

    # Cliff compounds: involved in at least one cliff pair
    cliff_mask = cliff_pair_mask.any(axis=1)

    # Per-compound cliff pair count
    cliff_pair_counts = cliff_pair_mask.sum(axis=1)

    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2  # symmetric
    n_cliff_compounds = int(cliff_mask.sum())

    info = {
        'tanimoto_threshold': tanimoto_threshold,
        'activity_k': activity_k,
        'activity_threshold': activity_threshold,
        'mad_y': mad_y,
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n),
        'cliff_pair_counts': cliff_pair_counts,
    }

    return cliff_mask, info


# =============================================================================
# CALIBRATED CLIFF RULE — the one rule s04 and s12 both call
# =============================================================================
#
# WHY THIS EXISTS. Two files decided which molecules are activity cliffs and
# they did not agree. s04 called a pair a cliff when ECFP4 Tanimoto was at or
# above 0.55 and the activity gap exceeded 2 x the median absolute deviation of
# the labels. s12 called a pair a cliff when it was in the top half per cent of
# pairwise similarity on the standardised concatenation of every representation,
# with the same 2 x median absolute deviation gap. On herg_ki s04 called 451 of
# 1412 molecules cliffs; s12 called 295 of 1412. Same dataset, same molecules,
# two answers, and neither file recorded which rule had produced its number.
#
# WHAT WAS MISCALIBRATED. The activity gap scaled with each dataset's own
# spread, so the share of molecules it flagged was whatever that dataset's label
# distribution happened to give. Measured 2026-09-19 at Tanimoto 0.55 and a gap
# of 2 x the median absolute deviation: 1625 of 2158 caco2 molecules, 451 of
# 1412 herg_ki molecules, 125 of 2641 adme_mdr1 molecules. As shares of each
# dataset that is 75.3%, 31.9% and 4.7%. At 75.3% the cliff stratum is most of
# the dataset and its score cannot differ much from the overall score. At 4.7%
# the 71 cliff pairs spread over 5 folds leave 20 to 30 cliff molecules in a
# held-out fold.
#
# WHAT THIS DOES INSTEAD. The structural side is unchanged: ECFP4 Tanimoto at or
# above 0.55, the convention identify_cliff_compounds cites Stumpfe & Bajorath
# 2019 for. The activity side stops being a fixed multiple of the median
# absolute deviation. It is solved, per dataset, so the share of molecules the
# rule flags is the same on every dataset. The multiplier it lands on is
# reported back in units of the median absolute deviation, so how far each
# dataset moved from the old 2.0 is readable.
#
# WHERE THIS DEPARTS FROM THE CITED CONVENTION, STATED. Stumpfe & Bajorath fix
# the activity side as well as the structural one: a 100-fold potency
# difference, two log units. That is defined for potency. caco2's label is log
# permeability with a median absolute deviation of 0.2085, and a two-log-unit
# gap flags no molecule there at all. The structural cutoff is the half of the
# convention that transfers across these endpoints, and it is the half kept.

# The share of the dataset's molecules the cliff stratum is calibrated to.
# Chosen from the sweep in the calibration run of 2026-09-20: see the block
# comment above for what the uncalibrated rule gave.
#
# 15.0 sits between two constraints measured on these three datasets:
#   below it   at 10% the thinnest held-out fold carries 19 cliff molecules on
#              herg_ki and 23 on caco2, under the 30 a fold needs to give a
#              stable cliff score.
#   above it   at 20% adme_mdr1's activity gap falls to 0.21x its median
#              absolute deviation, which is too small a difference to call a
#              cliff.
# Even at 15% adme_mdr1 needs 0.50x MAD, which is weak — it has 2,318 scaffolds
# among 2,641 molecules, so genuinely similar pairs are scarce.
#
# KIRBY_CLIFF_TARGET_PCT sweeps it. Unset gives 15.0, so every stored number
# reproduces. s04 and s12 record the achieved share and the gap in MAD units
# per run, so the two constraints above are checkable from any result file
# rather than only from this comment.
CLIFF_TARGET_PCT = float(
    os.environ.get('KIRBY_CLIFF_TARGET_PCT', '').strip() or 15.0)

# The structural cutoff of Stumpfe & Bajorath 2019 for ECFP4, unchanged.
CLIFF_TANIMOTO_CUTOFF = 0.55

# For a dataset with no ECFP4 — no SMILES, or a non-molecular domain — the
# structural side is the top this-many per cent of pairwise similarity on the
# standardised concatenation of every representation. Held fixed so that the
# solved activity gap is the only free quantity; with both free the target share
# does not determine a rule.
CLIFF_SIMILARITY_PERCENTILE = 1.0


def solve_activity_gap(structural_mask, y, target_pct, is_clf=False,
                       n_steps=80):
    """The activity gap that makes `target_pct` per cent of molecules cliffs.

    `structural_mask` is the boolean n x n matrix of structurally similar pairs,
    diagonal already false. The share of molecules flagged falls as the gap
    rises, so a bisection finds it. The share moves in steps, so the gap
    returned is the one whose share is nearest the target, not one that hits it.

    Returns (gap, achieved_pct). On a classification task the activity side is
    the class-mismatch indicator and there is nothing to solve, so the gap comes
    back as 0.5 and the achieved share is whatever the structural side gives.
    """
    n = len(y)
    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)

    def _pct(gap):
        m = structural_mask & (delta_y > gap)
        np.fill_diagonal(m, False)
        return 100.0 * float(m.any(axis=1).sum()) / n

    if is_clf:
        return 0.5, _pct(0.5)

    lo, hi = 0.0, float(np.max(delta_y)) if n else 1.0
    if hi <= 0:
        return 0.0, _pct(0.0)
    best = (lo, _pct(lo))
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        p = _pct(mid)
        if abs(p - target_pct) < abs(best[1] - target_pct):
            best = (mid, p)
        if p > target_pct:
            lo = mid
        else:
            hi = mid
    return float(best[0]), float(best[1])


def calibrated_cliff_rule(structural_sim, y, structural_cutoff,
                          structural_source, target_pct=CLIFF_TARGET_PCT,
                          is_clf=False, activity_k=None):
    """Cliff molecules under the calibrated rule, and the rule that produced them.

    Args:
        structural_sim: n x n similarity, diagonal ignored. ECFP4 Tanimoto for a
            molecular dataset; 1/(1+d) on the standardised concatenation of
            every representation otherwise.
        y: the labels, in the dataset's own units.
        structural_cutoff: the similarity at or above which a pair counts as
            structurally similar.
        structural_source: a sentence saying where structural_sim came from,
            stored so a result file says which rule wrote it.
        target_pct: the share of the dataset's molecules to flag. The activity
            gap is solved to reach it.
        activity_k: set this to skip the calibration and use a gap of
            activity_k x the median absolute deviation of y, which is what s04
            and s12 both did before. The uncalibrated share is recorded either
            way, so the two are always comparable in a stored result.

    Returns (cliff_mask, info). info['rule'] names which of the two ran.
    """
    y = np.asarray(y)
    n = len(y)
    mad_y = 0.0 if is_clf else float(np.median(np.abs(y - np.median(y))))
    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)

    structural_mask = np.asarray(structural_sim) >= structural_cutoff
    np.fill_diagonal(structural_mask, False)
    n_structural_pairs = int(structural_mask.sum()) // 2

    def _stats(gap):
        m = structural_mask & (delta_y > gap)
        np.fill_diagonal(m, False)
        mask = m.any(axis=1)
        return mask, m.sum(axis=1), int(m.sum()) // 2, int(mask.sum())

    # What the uncalibrated rule gives, recorded whichever rule runs, so the
    # calibrated count is never the only count in the file.
    gap_uncal = 0.5 if is_clf else 2.0 * mad_y
    _, _, pairs_uncal, n_uncal = _stats(gap_uncal)

    if activity_k is not None or is_clf:
        k = 2.0 if activity_k is None else float(activity_k)
        gap = 0.5 if is_clf else k * mad_y
        rule = 'fixed_mad_multiple'
        achieved = None
    else:
        gap, achieved = solve_activity_gap(structural_mask, y, target_pct)
        k = (gap / mad_y) if mad_y > 0 else float('nan')
        rule = 'calibrated_to_target_share'

    cliff_mask, pair_counts, n_pairs, n_cliff = _stats(gap)

    info = {
        'rule': rule,
        'structural_source': structural_source,
        'structural_cutoff': float(structural_cutoff),
        'n_structural_pairs': n_structural_pairs,
        'target_pct_molecules': (None if rule == 'fixed_mad_multiple'
                                 else float(target_pct)),
        'activity_threshold': float(gap),
        'activity_k': float(k),
        'activity_k_note': (
            'the activity gap divided by the median absolute deviation of the '
            'labels, so it is comparable with the 2.0 both files used before'),
        'mad_y': mad_y,
        'n_molecules': int(n),
        'n_cliff_compounds': n_cliff,
        'n_cliff_pairs': n_pairs,
        'pct_cliff_compounds': float(100.0 * n_cliff / n) if n else 0.0,
        'cliff_pair_counts': pair_counts,
        # The rule both files ran before this one existed, on these same rows.
        'uncalibrated_activity_k': 2.0,
        'uncalibrated_activity_threshold': float(gap_uncal),
        'uncalibrated_n_cliff_compounds': n_uncal,
        'uncalibrated_n_cliff_pairs': pairs_uncal,
        'uncalibrated_pct_cliff_compounds': (float(100.0 * n_uncal / n)
                                             if n else 0.0),
        # Kept for the readers of the older key names.
        'tanimoto_threshold': float(structural_cutoff),
    }
    if achieved is not None:
        info['achieved_pct_molecules'] = float(achieved)
    return cliff_mask, info


def cliff_structural_similarity(data, reps, is_molecular=True,
                                metric='manhattan'):
    """The n x n similarity the cliff rule reads, and where it came from.

    ECFP4 Tanimoto built from the dataset's own SMILES where there are SMILES
    and rdkit can parse them — the same matrix s04 built, so both files now
    partition a molecular dataset identically. Otherwise 1/(1+d) on the
    scaled concatenation of every representation, d taken under `metric`, one
    of the decided distances (similarity_space.FEATURE_METRICS). manhattan,
    the default, is in both kNN groups' decided lists.

    Returns (similarity, cutoff, source_sentence, kind) where kind is
    'smiles_ecfp4_tanimoto' or 'concatenation_similarity_percentile'.
    """
    n_rows = len(data['labels'])
    smiles = data.get('smiles')
    if is_molecular and smiles is not None and len(smiles) == n_rows:
        try:
            fp, n_unparsed = _cliff_ecfp4_bits(smiles)
        except Exception as exc:
            print(f"  cliff ECFP4 from SMILES failed "
                  f"({type(exc).__name__}: {exc}); falling back to the "
                  f"standardised concatenation of every representation")
        else:
            # Tanimoto on binary rows as one matrix product: the intersection is
            # fp @ fp.T and the union is |a| + |b| - intersection. Identical to
            # 1 - scipy's jaccard distance on these rows and far cheaper.
            f = np.asarray(fp, dtype=np.float64)
            inter = f @ f.T
            cnt = f.sum(axis=1)
            union = cnt[:, None] + cnt[None, :] - inter
            sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
            np.fill_diagonal(sim, 0.0)
            return (sim, CLIFF_TANIMOTO_CUTOFF,
                    (f'ECFP4 radius 2 over 2048 bits, built from this '
                     f"dataset's {n_rows} SMILES; {n_unparsed} of them rdkit "
                     f'could not parse and each got a private feature, so it is '
                     f'a cliff partner of nothing. Tanimoto at or above '
                     f'{CLIFF_TANIMOTO_CUTOFF}, the ECFP4 convention of '
                     f'Stumpfe & Bajorath 2019'),
                    'smiles_ecfp4_tanimoto')

    # DECIDED 2026-10-06 (Adelaide delegated): the fallback takes d under a
    # decided distance through similarity_space.feature_distance, on features
    # scaled with FeatureScaler (0/1 features unscaled, the rest standardised).
    # Before this date it took the one process-wide distance
    # (KIRBY_DISTANCE_METRIC) through compute_distance_matrix on StandardScaler
    # features. The scaling is fitted over every row: features only, no labels.
    from kirby.scaling import FeatureScaler
    from experiments.noise.similarity_space import (
        feature_distance, FEATURE_METRICS)
    if metric not in FEATURE_METRICS:
        raise ValueError(f'cliff distance {metric!r} is not one of the '
                         f'decided distances {FEATURE_METRICS}')
    X = np.hstack([np.asarray(reps[k], dtype=np.float64)
                   for k in sorted(reps.keys())])
    X = FeatureScaler().fit(X).transform(X)
    D = feature_distance(X, X, metric)
    sim = 1.0 / (1.0 + D)          # an inf distance (cosine, no spread) -> 0
    del D
    np.fill_diagonal(sim, 0.0)
    off = sim[~np.eye(len(sim), dtype=bool)]
    cutoff = float(np.percentile(off, 100.0 - CLIFF_SIMILARITY_PERCENTILE))
    return (sim, cutoff,
            (f'1/(1+d) on the concatenation of every representation, '
             f'{X.shape[1]} features, 0/1 features unscaled and the rest '
             f'standardised over every row, d taken with {metric}; '
             f'structurally similar means the top '
             f'{CLIFF_SIMILARITY_PERCENTILE} per cent of pairs by that '
             f'similarity. No ECFP4 was available for this dataset'),
            'concatenation_similarity_percentile')


def _cliff_ecfp4_bits(smiles_list, n_bits=2048, radius=2):
    """ECFP4 bit matrix for the cliff rule. Returns (matrix, n_unparsed).

    A molecule rdkit cannot parse gets one private feature of its own rather
    than an all-zero row: two all-zero rows are Tanimoto 1 to each other, which
    would make every unparsable molecule a cliff partner of every other one.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mols = [Chem.MolFromSmiles(smi) if smi else None for smi in smiles_list]
    unparsed = [i for i, m in enumerate(mols) if m is None]
    fp_array = np.zeros((len(mols), n_bits + len(unparsed)), dtype=np.float32)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for bit in fp.GetOnBits():
            fp_array[i, bit] = 1.0
    for slot, i in enumerate(unparsed):
        fp_array[i, n_bits + slot] = 1.0
    return fp_array, len(unparsed)


# =============================================================================
# ANTIBODY SEQUENCE CLIFF ANALYSIS (classification-aware edit-distance sweep)
# =============================================================================
#
# References for threshold choices and precedent:
#   Mason et al. 2021, Nat Biomed Eng — trastuzumab CDR-H3 DMS, Hamming ≤ 1/2/3
#     neighborhoods with Hamming ≤ 2 as local-cliff default. Closest analog
#     for binary binder/non-binder flip analysis.
#   Adams et al. 2019, eLife — Ab-14 DMS, single/double mutant epistasis at
#     Hamming 1 and 2.
#   Phillips et al. 2023, eLife — CR9114/CR6261 DMS, pairs at Hamming 1 for
#     epistasis, ≤ 2 for local landscape.
#
# Caveat: classification cliffs are under-formalized for antibodies. Hamming
# on full V-region is length-biased; for variable-length CDR-H3 see Gupta 2017
# normalized Levenshtein. Cite Mason 2021 as the primary methodological anchor.

def _cdr_cache_path(dataset_name, heavy_seqs=None):
    """Path for cached IMGT CDR strings, keyed on the sequences they came from.

    This used to be the dataset name alone. The file holds one CDR string per
    antibody, in order, so it is only valid for the exact sequence list it was
    built from — and a run on a filtered or extended set of antibodies read the
    older, differently-long list and paired the first N of them against its own
    N labels. Every distance, cliff and neighbour number downstream is then
    computed on another antibody's CDRs. The sequences decide the file now; the
    dataset name stays in it only so the directory is readable.
    """
    from pathlib import Path
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    stem = str(dataset_name)
    if heavy_seqs is not None:
        import hashlib
        h = hashlib.sha1(
            '\n'.join(map(str, heavy_seqs)).encode()).hexdigest()[:12]
        stem = f'{stem}__{len(heavy_seqs)}n_{h}'
    return cache_root / 'cdr_indices' / f'{stem}.pkl'


def _extract_cdr_sequences(heavy_seqs, dataset_name=None):
    """Extract CDR-only substrings from heavy chains using ANARCI (IMGT).

    Results are cached per dataset under ~/.cache/kirby/cdr_indices/ since
    ANARCI is slow (~seconds per sequence). Returns list of CDR-concatenated
    strings (CDR1+CDR2+CDR3). Returns None if ANARCI unavailable or fails.
    """
    import pickle
    cache_path = (_cdr_cache_path(dataset_name, heavy_seqs)
                  if dataset_name else None)
    if cache_path and cache_path.exists():
        try:
            return pickle.loads(cache_path.read_bytes())
        except Exception:
            pass

    try:
        from kirby.representations.antibody import get_cdr_residue_indices
    except Exception:
        return None

    cdr_seqs = []
    failures = 0
    for seq in heavy_seqs:
        try:
            idxs = get_cdr_residue_indices(seq)
            if not idxs:
                failures += 1
                cdr_seqs.append(seq)  # fallback to full sequence for this one
                continue
            cdr_seqs.append(''.join(seq[i] for i in idxs))
        except Exception:
            failures += 1
            cdr_seqs.append(seq)

    if failures > 0.5 * len(heavy_seqs):
        # More than half failed — likely ANARCI not installed or broken.
        # Signal failure to caller so they can skip the CDR scope entirely.
        return None

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(pickle.dumps(cdr_seqs))
        except Exception:
            pass  # cache write failure is non-fatal

    return cdr_seqs


def antibody_sequence_cliff_sweep(sequences, labels, is_clf,
                                  max_edits_sweep=(1, 2, 3, 5),
                                  max_compounds=3000, scope_label='full_V',
                                  activity_k=2.0):
    """Run edit-distance cliff detection across multiple Hamming thresholds.

    For classification: cliff = pair within Hamming <= max_edits but with
    different class labels (binder/non-binder flip, polyreactive/specific
    flip, etc.).

    Args:
        sequences: list of str (heavy chains, or CDR-concatenated strings).
        labels: array of activity or class labels.
        is_clf: True for classification datasets.
        max_edits_sweep: Hamming thresholds to report (sensitivity sweep).
        max_compounds: subsample cap — n² Hamming matrix grows fast.
        scope_label: 'full_V' or 'cdr_only' — for reporting/logging only.
        activity_k: MAD multiplier for regression cliff threshold (unused
            for classification; see identify_cliff_compounds_editdist).

    Returns:
        dict: {
            'scope': scope_label,
            'n_analyzed': int,
            'thresholds': {max_edits: {
                'n_cliff_pairs': int,
                'n_cliff_compounds': int,
                'pct_cliff_compounds': float,
                'mean_edit_dist': float,
                'median_edit_dist': float,
            }},
            'notes': str,  # interpretation caveats
        }
    """
    n = len(sequences)
    y = np.asarray(labels)

    if n > max_compounds:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_compounds, replace=False)
        sequences = [sequences[i] for i in idx]
        y = y[idx]
        n = max_compounds

    result = {
        'scope': scope_label,
        'n_analyzed': n,
        'thresholds': {},
    }

    for max_edits in max_edits_sweep:
        try:
            _, info = identify_cliff_compounds_editdist(
                sequences, y, max_edits=max_edits, activity_k=activity_k,
                is_clf=is_clf)
            # Drop the per-compound count array (too large for JSON).
            info_clean = {k: v for k, v in info.items()
                          if k not in ('cliff_pair_counts',)}
            result['thresholds'][int(max_edits)] = info_clean
        except Exception as e:
            # SIMA-15: surface the failure on the console (exception class +
            # threshold) so a blown threshold is visible at run time, not only
            # by reading the saved JSON afterwards. Error is still captured in
            # the result dict for downstream aggregation.
            print(f"    WARNING: cliff sweep failed at max_edits={max_edits} "
                  f"({scope_label}): {type(e).__name__}: {e}")
            result['thresholds'][int(max_edits)] = {'error': str(e)}

    if is_clf:
        result['notes'] = (
            "Classification cliff = pair within Hamming <= max_edits but "
            "with different class labels. Binary indicator is coarser than "
            "continuous |delta_y|, so interpretation should be qualitative "
            "(cliff density, flip frequency) rather than quantitative. "
            "Primary anchor: Mason et al. 2021 (trastuzumab CDR-H3 DMS). "
            "See also Phillips 2023, Adams 2019."
        )
    else:
        result['notes'] = (
            "Regression cliff = pair within Hamming <= max_edits with "
            "|delta_y| > activity_k * MAD(y). Anchor: Mason 2021 DMS."
        )

    return result


def run_antibody_cliff_analysis(data, is_clf, dataset_name=None,
                                max_edits_sweep=(1, 2, 3, 5)):
    """Driver: run sequence cliff sweep at full V-region and CDR-only scopes.

    Args:
        data: dict from load_data(). Must contain 'labels' and either
            'heavy_seqs' (antibody) or 'sequences' (nucleotide).
        is_clf: True for classification datasets.
        dataset_name: Optional dataset name for CDR cache keying. If provided,
            extracted CDR indices are cached under ~/.cache/kirby/cdr_indices/
            so subsequent runs skip ANARCI. Falls back to `data.get('dataset_name')`
            if not passed.
        max_edits_sweep: Hamming thresholds to sweep.

    Returns a dict or None if the dataset has no sequence data (non-antibody,
    non-nucleotide, or sequences not exposed by the loader).
    """
    heavy_seqs = data.get('heavy_seqs')
    if heavy_seqs is None or len(heavy_seqs) == 0:
        # Antibody datasets expose heavy_seqs; nucleotide uses sequences key.
        seqs = data.get('sequences')
        if seqs is None or len(seqs) == 0:
            return None
        heavy_seqs = list(seqs)

    labels = data['labels']
    if dataset_name is None:
        dataset_name = data.get('dataset_name')
    results = {}

    # Scope 1: full V-region (fast, always available).
    print(f"  Sequence cliff sweep (full V-region, n={len(heavy_seqs)})...")
    results['full_V'] = antibody_sequence_cliff_sweep(
        list(heavy_seqs), labels, is_clf,
        max_edits_sweep=max_edits_sweep, scope_label='full_V')
    for thresh, info in results['full_V']['thresholds'].items():
        if 'error' not in info:
            print(f"    Hamming<={thresh}: {info['n_cliff_pairs']} pairs, "
                  f"{info['n_cliff_compounds']} cliff compounds "
                  f"({info.get('pct_cliff_compounds', 0):.1f}%)")

    # Scope 2: CDR-only (opt-in, slower due to ANARCI).
    # Caches CDR extraction per dataset so repeated runs are fast.
    cdr_seqs = _extract_cdr_sequences(list(heavy_seqs), dataset_name=dataset_name)
    if cdr_seqs is not None:
        print(f"  Sequence cliff sweep (CDR-only, from ANARCI/IMGT)...")
        results['cdr_only'] = antibody_sequence_cliff_sweep(
            cdr_seqs, labels, is_clf,
            max_edits_sweep=max_edits_sweep, scope_label='cdr_only')
        for thresh, info in results['cdr_only']['thresholds'].items():
            if 'error' not in info:
                print(f"    Hamming<={thresh}: {info['n_cliff_pairs']} pairs, "
                      f"{info['n_cliff_compounds']} cliff compounds "
                      f"({info.get('pct_cliff_compounds', 0):.1f}%)")
    else:
        results['cdr_only'] = {'skipped': True,
                               'reason': 'ANARCI unavailable or failed on >50% of sequences'}
        print(f"  Sequence cliff sweep (CDR-only): SKIPPED (ANARCI unavailable)")

    return results


# Similarity-specific extra params (layered on top of the DEFAULT_* config)
SIMILARITY_EXTRA = {
    # Which importance method weights the features INSIDE the distance s02
    # scores. This is not KIRBY_LOCAL_IMPORTANCE, which picks the method the
    # hybrid search uses to rank features while it spends the budget -- that
    # one changes how the hybrid is built, this one changes how the distance
    # is weighted. Like k_values below, this key sits in the config every
    # similarity run is built from, so it always won and s02 only ever
    # measured lightgbm_gain. The env var now wins when it is set.
    'weight_methods': ([v.strip() for v in
                        os.environ.get('KIRBY_S02_WEIGHT_METHODS', '').split(',')
                        if v.strip()] or ['lightgbm_gain']),
    # 'k_values' (s05's neighbour counts, KIRBY_S05_K_VALUES) removed 2026-10-05:
    # every similarity experiment reads 5, 10, 20 from knn_axes (axis 4).
    # How many molecule pairs every distance-activity Spearman is taken over,
    # in s01, s02 and s08. It was 5,000 because _pair_distances cost 4,096
    # distances for every pair it kept; it now costs one, so 5,000 was paying
    # for precision it did not need to give up. Measured on herg_ki, ecfp4:
    # the rho reads 0.0580 at 5,000 pairs, 0.0491 at 50,000 and 0.0461 at
    # 200,000, and the gap between neighbouring representations in that table
    # is about 0.005 — so at 5,000 the ranking moved with the pair draw.
    #
    # It does NOT make a single rho precise. Every pair is drawn from the same
    # molecules, so the permutation null's spread is set by the molecule count
    # and barely moves: 0.0178 at 5,000 pairs against 0.0132 at 50,000 on those
    # 1,412 molecules. That spread is stored per representation as
    # sac_permutation_null_sd, and it is what a difference between two
    # representations has to clear.
    'n_sample': 50000,
}


# ── Single-dataset experiment list ─────────────────────────────────────────
# S09 is cross-dataset — run separately.

SINGLE_DATASET_EXPERIMENTS = [
    's01',   # similarity baseline (standalone)
    's02',   # hybrid similarity
    's03',   # feature concentration
    's04',   # activity cliffs
    's05',   # kNN prediction
    's06',   # applicability domain
    's08',   # optimal dimension (standalone)
    's10',   # importance downstream (standalone)
    's11',   # bayesian optimization (standalone)
    's12',   # cross-domain cliffs (standalone, generic)
    's13',   # clustering + retrieval (standalone)
]


def _import_experiment(exp_id: str):
    """Dynamically import a similarity experiment module by ID (e.g. 's01')."""
    module_map = {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.similarity.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    }
    return name_map.get(exp_id, exp_id)


def _screen_rep_pool(reps: dict, pool_label: str):
    """Zero-fill partial NaN, drop degenerate reps, and report both.

    A rep whose matrix is entirely NaN, or entirely zero once NaN is filled,
    would still enter the pool and receive budget in the hybrid search, so it
    is removed here instead. A partly-NaN rep keeps the existing zero fill;
    the fraction of entries replaced is returned so it reaches the saved
    result rather than only stdout.

    Args:
        reps: rep name -> feature matrix.
        pool_label: 'base' or 'augmentation' — used in the audit and prints.

    Returns:
        (kept, audit). `kept` holds the surviving matrices, NaN already
        filled. `audit` has 'kept' (rep -> fraction of entries replaced by
        zero) and 'dropped' (rep -> {'reason', 'nan_fraction'}), with
        reasons 'all_nan', 'all_zero_after_nan_fill', 'all_zero',
        'empty_matrix'.
    """
    kept = {}
    audit = {'pool': pool_label, 'kept': {}, 'dropped': {}}
    for rname, X in reps.items():
        X = np.asarray(X)
        if X.size == 0:
            audit['dropped'][rname] = {'reason': 'empty_matrix',
                                       'nan_fraction': None}
            print(f"  DROP {pool_label} rep {rname}: empty_matrix")
            continue
        nan_count = int(np.isnan(X).sum())
        nan_fraction = nan_count / X.size
        if nan_count == X.size:
            audit['dropped'][rname] = {'reason': 'all_nan',
                                       'nan_fraction': 1.0}
            print(f"  DROP {pool_label} rep {rname}: all_nan "
                  f"({nan_count} of {X.size} entries)")
            continue
        if nan_count > 0:
            print(f"  WARNING: {rname} has {nan_count} NaN values "
                  f"({100 * nan_fraction:.1f}%) — imputing 0")
            X = np.nan_to_num(X, nan=0.0)
        if not np.any(X):
            reason = 'all_zero_after_nan_fill' if nan_count > 0 else 'all_zero'
            audit['dropped'][rname] = {'reason': reason,
                                       'nan_fraction': nan_fraction}
            print(f"  DROP {pool_label} rep {rname}: {reason} "
                  f"({100 * nan_fraction:.1f}% NaN before the fill)")
            continue
        kept[rname] = X
        audit['kept'][rname] = nan_fraction
    return kept, audit


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None,
            pruning_mode: str = 'default_pruning',
            force_rerun: bool = False,
            precompute_objectives=None):
    """Load data once and run all single-dataset similarity experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: fast tier + similarity extras).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
        pruning_mode: One of PRUNING_MODES. 'default_pruning' = legacy
            single-stage prune. 'aggressive_pruning' = F16-validated multi-
            stage pruning (checkpoints=[25,50,100], td=0.5) applied globally
            to all models, no n_samples gate. F16: ~58% fold-0 eval reduction.
        precompute_objectives: extra search objectives to warm before the
            experiments run, e.g. ('similarity',). Default None = the
            predictive searches alone, which is what this function always did.
            'similarity' chooses features so that each molecule's activity is
            predicted from its five nearest neighbours in the distance the
            features define; find_best_hybrid stores it under its own cache
            tier, so it is an extra pass and the predictive searches are
            untouched. An experiment that asks the pool for a similarity-built
            hybrid reads it off DISK (the pool refuses to serve a non-predictive
            objective from the in-memory dicts, which record no objective), so
            this pass exists to put it there once instead of rebuilding it
            inside the first experiment that asks.
    """
    if config is None:
        from kirby.find_best_hybrid import (
            DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS, DEFAULT_N_FOLDS,
        )
        from experiments._shared import FULL_BUDGETS
        config = {
            'n_splits': DEFAULT_N_FOLDS,
            'allocations': DEFAULT_ALLOCATION_METHODS,
            'importance_methods': DEFAULT_IMPORTANCE_METHODS,
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': FULL_BUDGETS,
            'models': DOWNSTREAM_MODELS,
        }
        config.update(SIMILARITY_EXTRA)
    # Search axes from the decided grid (R1, R3), whoever built the config.
    from experiments._shared import decided_config
    config = decided_config(dict(config), dataset_name)
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)

    # Resolve rep list
    if rep_names is None:
        ds_reps = DATASET_REPS.get(dataset_name)
        if ds_reps is None:
            raise ValueError(
                f"No rep list for '{dataset_name}'. Pass rep_names explicitly "
                f"or add it to DATASET_REPS in complexity_helpers.py."
            )
        rep_names = ds_reps['general']

    print(f"\n{'=' * 70}")
    print(f"SIMILARITY EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    # ── Load data ONCE ──
    t0 = time.time()
    print(f"\nLoading {dataset_name}...")
    data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # NaN guard: some reps produce NaN for certain sequences. Partial NaN is
    # still replaced with 0 and warned about; a rep that is entirely NaN, or
    # entirely zero once filled, leaves the pool instead of reaching
    # precompute_hybrid_searches and drawing budget.
    reps_data, _base_audit = _screen_rep_pool(data.get('reps', {}), 'base')
    if _base_audit['dropped'] and len(reps_data) < 2:
        _reasons = {k: v['reason'] for k, v in _base_audit['dropped'].items()}
        raise ValueError(
            f"{dataset_name}: {len(reps_data)} usable representation(s) left "
            f"after the NaN/zero screen dropped {_reasons}. A hybrid needs at "
            f"least 2."
        )
    data['reps'] = reps_data

    # ── Load augmentation reps ──
    aug_rep_names = DATASET_REPS.get(dataset_name, {}).get('augmentation', [])
    _aug_audit = {'pool': 'augmentation', 'kept': {}, 'dropped': {}}
    if aug_rep_names:
        # A name the base screen dropped is no longer in reps_data, so it lands
        # in `missing`, is reloaded, and is dropped again by the same screen.
        missing = [r for r in aug_rep_names if r not in reps_data]
        if missing:
            print(f"  Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing)
            aug_reps, _aug_audit = _screen_rep_pool(aug_data['reps'], 'augmentation')
            data['augmentations'] = aug_reps or None
        else:
            data['augmentations'] = {r: reps_data[r] for r in aug_rep_names}
            _aug_audit['kept'] = {r: _base_audit['kept'][r] for r in aug_rep_names}
        if data['augmentations']:
            print(f"  Augmentations ({len(data['augmentations'])}): "
                  f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
        else:
            print(f"  Augmentations (0): all {len(aug_rep_names)} dropped by "
                  f"the NaN/zero screen")
    else:
        data['augmentations'] = None

    # ── Record the screen ──
    # 'kept' fractions are the share of matrix entries replaced by zero;
    # 'dropped' names why each rep left the pool. Saved before any experiment
    # runs so the record survives a run that dies part-way.
    rep_pool_audit = {
        'dataset': dataset_name,
        'n_base_requested': len(rep_names),
        'n_base_kept': len(reps_data),
        'base': _base_audit,
        'augmentation': _aug_audit,
    }
    data['rep_pool_audit'] = rep_pool_audit
    save_experiment_results(rep_pool_audit, output_dir,
                            '_rep_pool_audit', dataset_name)

    # ── No whole-dataset searches (R2, R6; _shared.no_startup_searches) ──
    # This also drops the opt-in extra-objective pass (precompute_objectives):
    # a similarity-objective search is built per fold by the experiment that
    # needs it, inside the dataset's decided cuts (R3).
    from experiments._shared import no_startup_searches
    precompute_errors = no_startup_searches(data, pruning_mode, 'similarity')
    if precompute_objectives:
        print(f"  precompute_objectives={list(precompute_objectives)} ignored "
              f"for the same reason.")

    # ── Record precompute failures ──
    # Written before any experiment runs, so the record survives a run that
    # dies part-way. Each record carries tier, dataset, aug_condition,
    # error_type, error and traceback.
    precompute_error_audit = {
        'dataset': dataset_name,
        'n_failures': sum(len(v) for v in precompute_errors.values()),
        'by_tier': {k: list(v) for k, v in precompute_errors.items()},
    }
    if precompute_error_audit['n_failures']:
        print(f"  Precompute failures recorded: "
              f"{precompute_error_audit['n_failures']}")
    save_experiment_results(precompute_error_audit, output_dir,
                            '_precompute_errors', dataset_name)

    # What every result file written below is stamped with, and what a
    # resumed run compares its existing files against.
    _fingerprint = input_fingerprint(data, config)

    # ── Run experiments in order ──
    results = {}
    for exp_id in experiments:
        # Resume: skip an experiment whose output JSON already exists AND holds
        # a measurement. Lets a walled/restarted job advance instead of
        # re-running finished experiments. force_rerun=True recomputes always.
        #
        # A file existing is not proof the run produced anything. 55 files under
        # tests/results hold no finite number outside the metadata and status
        # keys — an all-failed run, a refusal, a section left as {} (listed in
        # EMPTY_RESULT_FILES_2026_08_27.md). Under an exists() test every one of
        # them was skipped on every resubmission. save_experiment_results now
        # stamps 'result_status'; experiment_result_is_empty reads it back.
        # A file carrying the {'skipped': True} marker classifies as 'skipped',
        # not 'empty', so a deliberate skip still resumes as before.
        #
        # classify_unmarked=True — the chosen meaning for a file written before
        # the stamp existed. Such a file carries neither 'result_status' nor
        # 'has_computed_results', and on 2026-08-27 every one of the 1541 files
        # under tests/results was such a file. Re-deriving the status from the
        # file's own sections is what makes this check act on them at all; left
        # at the default, an unmarked empty file still reads as finished.
        # A file that will not parse is recomputed for the same reason: no
        # measurement can be read out of it.
        _out_json = os.path.join(
            output_dir, _experiment_full_name(exp_id), f"{dataset_name}.json")
        if not force_rerun and os.path.exists(_out_json):
            try:
                with open(_out_json) as _f:
                    _prev = json.load(_f)
                _prev_empty = experiment_result_is_empty(
                    _prev, classify_unmarked=True)
                _why = 'records no measurement'
            except (ValueError, OSError) as _e:
                _prev_empty = True
                _why = f'could not be read ({_e})'
            if not _prev_empty:
                # A file with numbers in it can still be the wrong numbers.
                # Nothing in it said which fold grouping, labels or
                # representations produced them, so a resume after any of those
                # changed kept it. input_fingerprint records them; a mismatch
                # recomputes.
                _drift = fingerprint_mismatch(
                    _prev.get('input_fingerprint'), _fingerprint)
                if _drift:
                    _prev_empty = True
                    _why = _drift
            if _prev_empty:
                print(f"  RERUN {exp_id}: existing output {_why} — "
                      f"recomputing ({_out_json})")
            else:
                print(f"  RESUME {exp_id}: already done — skipping ({_out_json})")
                results[exp_id] = {'resumed': True}
                continue
        print(f"\n{'─' * 70}")
        t_exp = time.time()

        # Large-n / domain scope policy: write a {'skipped':...} marker so the
        # wave-audit counts it as intentionally-skipped (see _shared.should_skip_
        # experiment + REPRODUCIBILITY_LEDGER.md §E).
        _skip = should_skip_experiment(exp_id, data)
        if _skip is not None:
            print(f"  SKIP {exp_id}: {_skip['reason']}")
            results[exp_id] = _skip
            save_experiment_results(_skip, output_dir,
                                    _experiment_full_name(exp_id), dataset_name)
            continue

        # An experiment that cannot be imported is a FAILED experiment, not a
        # skip: until 2026-09-25 this printed a line and continued, leaving no
        # file and no error record. Recorded now, with the traceback, so
        # run_dataset.py writes it to <output>/_errors/ and exits 1.
        try:
            mod = _import_experiment(exp_id)
        except Exception as e:
            import traceback
            print(f"  {exp_id} FAILED: import failed "
                  f"({type(e).__name__}: {e})")
            traceback.print_exc()
            results[exp_id] = {'error': str(e) or type(e).__name__,
                               'error_kind': f'import:{type(e).__name__}',
                               'traceback': traceback.format_exc()}
            continue

        # Build kwargs — pass similarity-specific params
        kwargs = {}
        if exp_id in ('s01', 's02', 's08'):
            kwargs['n_sample'] = config.get('n_sample', 5000)
        if exp_id == 's02':
            kwargs['weight_methods'] = config.get('weight_methods', ['lightgbm_gain'])
        # Budgets, neighbour counts and models are no longer passed from here
        # (handoff Part 2, chat 4, 2026-10-05). Each experiment reads the
        # dataset's decided budgets (decided_grid.decided_search) and the kNN
        # record's settings (knn_axes) itself, so a run cannot carry the old
        # FULL_BUDGETS / DISTANCE_BUDGETS / k = 5 and 10 into a search or a
        # neighbour count.
        # SMOKE ONLY: s11 runs conditions x n_seeds(=3) x max_iterations(=100)
        # GP-refit BO loops — minutes-to-hours locally. When KIRBY_SMOKE_N is
        # set, shrink the BO loop so the whole experiment COMPLETES and writes
        # output (it has no early checkpoint save; a killed run = no JSON). This
        # is gated on the smoke env var so server full runs use the defaults.
        if exp_id == 's11' and os.environ.get('KIRBY_SMOKE_N'):
            kwargs['max_iterations'] = int(os.environ.get('KIRBY_SMOKE_S11_ITERS', '').strip() or '8')
            kwargs['n_seeds'] = int(os.environ.get('KIRBY_SMOKE_S11_SEEDS', '').strip() or '1')
        # KIRBY_S11_CONDITIONS names the arms to run, comma separated. s11's run()
        # has always taken a `conditions` list; nothing exposed it.
        #
        # Why it is needed here: `hybrid_bnn` fits an MC-dropout network and is
        # what kills every local s11 run. Measured 2026-09-23 on caco2 at 250
        # molecules, 3 picks and 1 seed, with nothing else on the machine: the
        # nine conditions before it finished and were checkpointed, and the
        # process was killed inside hybrid_bnn with no traceback. It matches the
        # earlier record of the same condition killing three processes at 14 GB
        # of 15 GB swap and recording nothing. That is a broken arm on this
        # machine, not a slow one, and it is the stated reason for leaving it
        # out of a local run. It is not cut from the experiment: name it in
        # KIRBY_S11_CONDITIONS, or run s11 on the cluster, and it runs.
        if exp_id == 's11':
            _c = os.environ.get('KIRBY_S11_CONDITIONS')
            if _c:
                kwargs['conditions'] = [x.strip() for x in _c.split(',') if x.strip()]
        # s03 used to be handed rf, lightgbm, dnn here; it now uses the
        # dataset's two decided models (decided_grid) itself.

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name, **kwargs)
            results[exp_id] = exp_result

            # Save per-experiment JSON
            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(exp_result, output_dir, exp_full_name,
                                    dataset_name, fingerprint=_fingerprint)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e) or type(e).__name__,
                               'error_kind': type(e).__name__,
                               'traceback': traceback.format_exc()}

        except SystemExit as e:
            # SystemExit is not an Exception: a library that calls sys.exit()
            # instead of raising walks past the handler above and ends the
            # whole run with nothing said. complexity_helpers.run_all catches
            # it for this reason (complexity_helpers.py:1242-1263). Same
            # handling here, added 2026-09-25.
            print(f"  {exp_id} FAILED: a library called sys.exit({e.code}) "
                  f"instead of raising. Caught so the remaining experiments "
                  f"still run.")
            import traceback
            traceback.print_exc()
            results[exp_id] = {
                'error': f'SystemExit({e.code}) — a library exited the process',
                'error_kind': 'SystemExit',
                'traceback': traceback.format_exc()}

        except KeyboardInterrupt:
            # Deliberate: Ctrl-C and SIGINT must still stop the run.
            raise

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} — {len(results)} experiments in {total:.0f}s")
    if _base_audit['dropped'] or _aug_audit['dropped']:
        print(f"Reps dropped by the NaN/zero screen: "
              f"base={ {k: v['reason'] for k, v in _base_audit['dropped'].items()} } "
              f"augmentation={ {k: v['reason'] for k, v in _aug_audit['dropped'].items()} }")
    print(f"{'=' * 70}")

    # Underscore key so it does not collide with an experiment ID.
    results['_rep_pool_audit'] = rep_pool_audit
    results['_precompute_errors'] = precompute_error_audit
    return results
