#!/usr/bin/env python3
"""N09: LNL Method Survey with Hybrids — Learning with Noisy Labels methods.

Paper: Noise

Scope / claim (honest framing): this experiment asks a RELATIVE question —
does adding the hybrid representation change the RANKING of learning-with-
noisy-labels (LNL) methods — under SYNTHETIC noise injected on the TRAINING
labels only. Validation labels are left clean. It does NOT establish an
absolute noise-robustness result for any method or representation: the signal
of interest is whether the base->hybrid (and hybrid->weighted-hybrid) swap
re-orders which LNL methods come out on top.

Methods tested:
  - 12 total in LNL_REGISTRY
  - 10 support both regression and classification (MSE<->CE swap inside the
    method, see lnl_methods.py).
  - 2 are regression-only (heteroscedastic_gp, evidential_kernel); they are
    auto-filtered when task_type='classification' via REGRESSION_ONLY_LNL.

ALIGNED 2026-10-05 (chat 5, HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4; Adelaide's
2.3 "Still open" group B, n09 decided):
  - One model, the neural network (dnn). Every condition here trains it: the
    plain baseline and all twelve label-noise methods are neural-network
    training schemes, so dnn is fixed by the question.
  - Feature spaces per outer fold and label set: dnn's 2 best single
    representations and 2 best hybrids (base_1, base_2, hybrid,
    hybrid__general2), ranked by one dnn search per (kind, level, draw, fold)
    on that pass's DAMAGED training labels (level 0: the labels as delivered).
    Plus the importance-weighted copy of hybrid 1 (weighted_hybrid), a fifth
    feature space. RESTORED 2026-10-07 (Adelaide): both, reverting the
    2026-10-06 clean-label search and the drop of the weighted copy.
  - The per-fold dnn search: on a dataset whose two decided models include dnn
    it runs the decided grid with models=dnn; on the other datasets it runs
    decided_grid.fast_model_search_kwargs(ds, 'dnn') -- the fast lists
    intersected with the dataset's cuts (Adelaide 2026-10-05). Nothing is read
    from a whole-dataset search any more.
  - The plain-model comparison is the plain dnn only; the dataset's second
    decided model is not fitted (DECIDED 2026-10-06, Adelaide delegated).
  - Damage: the decided kinds and levels (noise_helpers.decided_conditions /
    decided_levels / inject_condition); 3 draws per non-zero level
    (noise_helpers.draws_for: 1 under censoring), level 0 (the labels as
    delivered) once.
  - Folds: noise_helpers.noise_outer_folds (config['n_splits']), the folds
    the per-fold searches are filed under. Any count, including 1.
  - distance_selection's neighbours, in the method survey: on each pass and
    feature space, the top (setting, k, scheme) of the neighbour-setting
    screen (n07.screen_settings, scored on a held-back fifth of the training
    molecules). Its old default -- its own 10 neighbours in the trained
    features, unweighted -- is no longer run.
  - distance_selection's neighbour sweep follows the kNN record: settings by
    dataset group, k in (5, 10, 20), both counting schemes, scaffold-mates
    kept, only the screen's top 3 (setting, k, scheme) refitted. Its D' rows
    are fitted with dnn only.
  - Neighbour spaces: dnn's own 2 + 2, the fold's distance hybrid and the
    fold's similarity hybrid. On a dataset whose two decided models lack dnn,
    the distance hybrid is the first decided model's that has one.

Conditions per level, once per feature space c in base_1, base_2, hybrid,
hybrid__general2 and weighted_hybrid:
  <c>_no_lnl  — feature space c + standard DNN (baseline)
  <c>_lnl     — feature space c + each LNL method

Outputs per level:
  - delta_from_clean: signed (damaged score - level-0 score), in the dataset's
    own metric; always defined. This is the robustness number on fast_knn
    datasets, where auc_norm is absent by decision (KNN record, noise grid).
  - score_ratio_pct: ratio*100 when both sides are positive, else NaN.
  - vs_plain_dnn_same_labels: each method against the plain DNN trained on the
    same labels (the headline, owner's decision 2026-09-27).

Usage:
    python n09_lnl_survey.py --dataset logd --mode intermediary
    python n09_lnl_survey.py --dataset qm9 --mode full
"""
import sys
import os
import hashlib
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments.dataset_policy import noise_groups_for
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (
    detection_scores, sweep_conditions, decided_conditions, decided_levels,
    inject_condition, level_unit, dataset_knn_group, noise_outer_folds,
    fold_spread, fold_configurations, fold_model_inputs, configuration_record,
    draws_for)
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, get_domain_metric,
)
from kirby.utils.lnl_methods import (
    get_lnl_method, LNL_REGISTRY, REGRESSION_ONLY_LNL, DistanceBasedSelection,
    _standard_dnn_train_predict, _standard_dnn_train_predict_clf,
)

ALL_METHODS = list(LNL_REGISTRY.keys())

# The model every condition here trains, and the model the per-fold search
# ranks representations and recipes for. Fixed by the question (Adelaide,
# 2026-10-05, handoff 2.3 group B).
BASE_RANK_MODEL = 'dnn'

# Three damage draws per non-zero level; level 0 (the labels as delivered) is
# run once (KNN record, noise grid; noise_helpers.inject_condition). Under
# censoring only the first is run (noise_helpers.draws_for).
NOISE_DRAWS = (0, 1, 2)

# The fitted feature spaces, in the order they are reported. The first four are
# the 2 + 2 of handoff 2.1 R4/R5 (noise_helpers.CONFIGURATION_SPACES); the
# fifth is the importance-weighted copy of hybrid 1.
# RESTORED 2026-10-07 (Adelaide): the weighted copy of hybrid 1 is back as a
# fifth feature space (dropped 2026-10-06 in f2d2c98).
FIT_SPACES = ('base_1', 'base_2', 'hybrid', 'hybrid__general2')
WEIGHTED_SPACE = 'weighted_hybrid'
ALL_SPACES = FIT_SPACES + (WEIGHTED_SPACE,)
HYBRID_FIT_SPACES = ('hybrid', 'hybrid__general2')
BASE_FIT_SPACES = ('base_1', 'base_2')

# RESTORED 2026-10-07 (Adelaide): the per-fold dnn search sees each pass's
# DAMAGED training labels. One search per (kind, level, draw, fold), so the
# representations and recipes are chosen with no clean label in view (the
# 4 June audit fix, AUDIT P1#5; NOISE_AUDIT_STATUS_2026_06_04.md). Per fold
# that is 1 search for level 0 plus one per non-zero level and draw of each
# kind (noise_helpers.draws_for: 1 draw under censoring).
# How the damaged labels reach the search: they are written into the label
# column itself (labels=, in _dnn_fold_entry), so the search trains AND
# scores its inner folds on them. find_best_hybrid's train_labels= is not
# used: it trains on that column but scores the inner folds on labels=.
# Each search is filed under condition_suffix 'n09dnn_y<hash of its training
# labels>', which rides on find_best_hybrid's tier, so no two label sets
# share a stored search or result row. The check that hybrid 2's features
# differ from hybrid 1's runs on the same damaged labels.
# This reverts 2026-10-06 (f2d2c98), which ran one search per fold on the
# clean training labels; 'clean' still selects that path.
SEARCH_LABELS = 'damaged'

# Identity of this version's checkpoint lines. Lines written before carry
# another fingerprint and are ignored (handoff 2.4 item 11). 2026-10-07:
# damaged-label search and the weighted copy of hybrid 1 restored.
ALIGNMENT_VERSION = 'restored_2026-10-07'

# Stamped into each condition file's input fingerprint (run() below), so a
# condition file written by an earlier version of n09 is recomputed by
# sweep_conditions instead of skipped. Same mechanism as n07's RESULT_LAYOUT.
RESULT_LAYOUT = 'restored_2026-10-07'


class _BaselineDNN:
    """The plain DNN of `_run_baseline`, with a fit/predict interface.

    create_hybrid's accept-or-reject step deep-copies whatever model it is
    given, fits it on the candidate feature set and scores it with the
    dataset's own metric. Without a model it falls back to the internal
    evaluator scoring r2 (regression) or accuracy (classification), so the
    features are kept or dropped by a different learner and a different
    quantity than the one the condition is then measured with.

    `_standard_dnn_train_predict` trains and predicts in one call, so fit
    stores the training rows and predict runs the network. Same
    hyperparameters, same random_state and — on a classification set — the
    same P(y=positive) output as `_run_baseline`, which is what AUROC needs.
    Holds arrays only, so deepcopy works.
    """

    def __init__(self, random_state=42, is_clf=False, classes_=None):
        self.random_state = random_state
        self.is_clf = is_clf
        self.classes_in = classes_

    def get_params(self, deep=True):
        return {'random_state': self.random_state, 'is_clf': self.is_clf,
                'classes_': self.classes_in}

    def fit(self, X, y, groups=None):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        return self

    def predict(self, X):
        return _run_baseline(self.X_train_, self.y_train_, np.asarray(X),
                             None, random_state=self.random_state,
                             is_clf=self.is_clf, classes_=self.classes_in)


def _run_baseline(X_train, y_train, X_val, y_val, random_state=42,
                  is_clf=False, classes_=None):
    """Standard DNN baseline (no LNL). Routes to clf trainer when is_clf=True."""
    if is_clf:
        return _standard_dnn_train_predict_clf(
            X_train, y_train, X_val,
            epochs=200, hidden_sizes=(256, 128, 64),
            dropout=0.2, lr=0.001, batch_size=256,
            patience=20, random_state=random_state,
            classes_=classes_,
        )
    return _standard_dnn_train_predict(
        X_train, y_train, X_val,
        epochs=200, hidden_sizes=(256, 128, 64),
        dropout=0.2, lr=0.001, batch_size=256,
        patience=20, random_state=random_state,
    )


def _record_detection(acc, cond_key, method_name, result, corrupted_mask,
                      noise_condition):
    """Grade one method's per-molecule suspicion score, if it exposes one.

    Three of the twelve learning methods rank every training molecule by how wrong
    its label looks -- `small_loss` by its own squared error, `confident_learning`
    by its error against an ensemble, `distance_selection` by how far it sits from
    its neighbours. Each now hands that ranking out in its metadata. Until
    2026-09-12 all three were graded only by the score after they threw molecules
    away, never on whether they threw the right ones away.

    The other nine expose no per-molecule ranking, so nothing is recorded for them.
    Nothing is recorded either under the six conditions where every training
    molecule moves -- see `noise_helpers.DETECTION_IS_DEFINED`.

    The suspicion array itself is read and dropped. Only the three numbers are kept.
    """
    if corrupted_mask is None:
        return
    sus = (result or {}).get('metadata', {}).get('suspicion_scores')
    if sus is None:
        return
    det = detection_scores(corrupted_mask, sus, noise_condition)
    if det is not None:
        acc.setdefault(cond_key, {}).setdefault(method_name, []).append(det)


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes."""
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted', 'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None
                and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


class _SchemeDistanceSelection(DistanceBasedSelection):
    """distance_selection with each neighbour counted by the swept scheme.

    2026-10-05 (chat 5). KNN record axis 5 sweeps how much each neighbour
    counts ('rank', 'inverse_square'). similarity_sweep hands the scheme over
    on the neighbour lists ('scheme' key), but DistanceBasedSelection
    (lnl_methods.py) takes a plain mean and spread of the neighbours' labels
    and never reads it, so the two schemes would give the same rows. Here the
    neighbours' mean, spread and class disagreement are weighted by
    similarity_space.neighbour_weights. Everything after the suspicion score
    (85th-percentile threshold, keep at least 10, refit) is the parent's.

    With no scheme on the lists (or no lists) it is the parent unchanged.
    Since 2026-10-06 the method survey's distance_selection also runs through
    this class, on the top screened (setting, k, scheme) of each pass.
    """

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        L = self.precomputed_neighbors
        scheme = L.get('scheme') if isinstance(L, dict) else None
        if scheme is None:
            return super().train_and_predict(X_train, y_train, X_val, y_val)
        from experiments.noise import similarity_space as ss
        X_t = np.asarray(X_train, dtype=np.float32)
        y_arr = np.asarray(y_train).ravel()
        k_actual = min(int(self.k), len(y_arr) - 1)
        idx, dist = ss.neighbours_at(L, k_actual, which='train')
        w = ss.neighbour_weights(dist, scheme)
        w_sum = w.sum(axis=1)
        nb = y_arr[np.asarray(idx)]
        if self.is_clf:
            disagreement = (w * (nb != y_arr[:, None])).sum(axis=1) / w_sum
        else:
            nb = nb.astype(float)
            mean = ss.weighted_neighbour_label(y_arr.astype(float), idx, dist,
                                               scheme)
            var = (w * (nb - mean[:, None]) ** 2).sum(axis=1) / w_sum
            disagreement = np.abs(y_arr - mean) / (np.sqrt(var) + 1e-10)
        threshold = np.percentile(disagreement, self.threshold_pct)
        flagged = disagreement > threshold
        keep_mask = ~flagged
        if keep_mask.sum() < 10:
            keep_mask = np.ones(len(y_arr), dtype=bool)
        y_pred = self._default_predict(X_t[keep_mask],
                                       np.asarray(y_train)[keep_mask], X_val)
        return {'y_pred': y_pred,
                'metadata': {
                    'suspicion_scores': np.asarray(disagreement, dtype=float),
                    'n_flagged': int(flagged.sum()),
                    'n_kept': int(keep_mask.sum()),
                    'threshold': float(threshold),
                    'mean_disagreement': float(disagreement.mean()),
                    'scheme': scheme}}


# In-process memo of the per-fold dnn selections, keyed by (dataset, fold,
# fold count, training-label hash). The search sees each pass's damaged
# training labels, so each (kind, level, draw) of a fold has its own entry;
# level 0's labels are the same under every kind, so the conditions of one
# sweep share that one. The search itself is stored on disk by
# find_best_hybrid and read back across runs.
_DNN_ENTRY_MEMO = {}
_NEIGHBOUR_MEMO = {}


def _label_token(y):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(y)).tobytes()
                        ).hexdigest()[:12]


def _distinct_dnn_hybrids(pooled, n_hybrids, *, data, dataset_name, fold_i,
                          train_idx, val_idx, y_train, factory, metric):
    """The first n_hybrids of the ranked candidates whose selected features
    differ (Adelaide 2026-10-05, handoff 2.1), each built exactly as the
    hybrid and hybrid__general2 spaces are built: on this fold's training
    rows, with _BaselineDNN (`factory`) in the accept-or-reject step and the
    metric the conditions are scored with, through the noise cache
    (cached_hybrid_for_fold, want_feature_info=True; the same cache entry
    _weighted_copy reads for hybrid 1).

    _shared._distinct_feature_hybrids is not used here: it builds every
    candidate with the registry dnn from get_model_factories, a different
    network from the one these spaces are built and trained with, so the
    feature sets it compares need not be the ones trained on (review
    2026-10-05). A candidate whose build fails is kept with
    feature_signature None and the error recorded, as there.
    """
    from experiments._shared import hybrid_feature_signature
    from experiments.noise import noise_cache as nc
    tr = np.asarray(train_idx)
    va = np.asarray(val_idx)
    reps = data['reps']
    g = data.get('groups')
    metric_name, metric_fn, lower = metric
    chosen, skipped, seen = [], [], set()
    for e in pooled:
        if len(chosen) >= n_hybrids:
            break
        bp = e.get('blueprint')
        augs = (data.get('augmentations')
                if (e.get('aug_condition') or 'none') != 'none' else None)
        sig, err = None, None
        try:
            built = nc.cached_hybrid_for_fold(
                dataset_name, {k: v[tr] for k, v in reps.items()},
                {k: v[va] for k, v in reps.items()}, np.asarray(y_train), bp,
                None if g is None else np.asarray(g)[tr], factory(),
                metric_fn, lower, model_name=BASE_RANK_MODEL,
                metric_name=metric_name, train_idx=tr,
                augmentations_train=(None if augs is None else
                                     {a: X[tr] for a, X in augs.items()}),
                augmentations_val=(None if augs is None else
                                   {a: X[va] for a, X in augs.items()}),
                want_feature_info=True)
            if built is None or built[0] is None:
                err = 'the build failed'
            else:
                sig = hybrid_feature_signature(built[2])
        except Exception as exc:
            from kirby.result_cache import StaleResultDB as _Stale
            if isinstance(exc, _Stale):
                raise
            err = f'{type(exc).__name__}: {exc}'
        if sig is not None and sig in seen:
            skipped.append({'label': e.get('label'),
                            'aug_condition': e.get('aug_condition'),
                            'score_mean': e.get('score_mean'),
                            'same_features_as': next(
                                c['label'] for c in chosen
                                if c.get('feature_signature') == sig)})
            print(f"  {dataset_name} fold {fold_i} dnn: hybrid "
                  f"{e.get('label')} selects the same features as one "
                  f"already chosen; taking the next")
            continue
        if sig is not None:
            seen.add(sig)
        chosen.append(dict(e, feature_signature=sig, signature_error=err))
    return chosen, skipped


def _dnn_fold_entry(data, dataset_name, *, fold_i, train_idx, val_idx,
                    n_folds, y_search_train, is_clf, classes_, task_type,
                    metric):
    """dnn's 2 best single representations and 2 best hybrids for one outer
    fold, from one search on that fold's training rows with the labels
    `y_search_train` (the pass's damaged training labels; SEARCH_LABELS).
    They are written into the label column passed as labels=, so the search
    trains and scores its inner folds on them.

    The entry has the shape noise_helpers.fold_configurations gives one model
    ('base', 'hybrids', 'recipes', 'unavailable', 'fit_configurations',
    'hybrids_skipped_identical_features', 'selection_scope'), so
    fold_model_inputs and configuration_record read it unchanged. It is built
    here and not by fold_configurations because that runs the dataset's two
    decided models on the decided grid only; it has no way to run dnn alone on
    the fast grid (the 16 datasets whose two models lack dnn) or to name the
    estimator.

    The search fits _BaselineDNN, the network every condition here trains
    (2026-09-28: the recipe used to be chosen by a random forest). Its
    condition_suffix carries the training-label hash, so a search on one label
    set is never read back for another, nor for the dataset's standard
    two-model per-fold search.
    """
    from experiments._shared import (search_on_fold, top_base_reps,
                                     top_hybrids,
                                     STANDARD_N_BASE, STANDARD_N_HYBRIDS)
    from experiments.decided_grid import (decided_models,
                                          fast_model_search_kwargs)
    train_idx = np.asarray(train_idx)
    labels_full = np.array(data['labels'], copy=True)
    labels_full[train_idx] = np.asarray(y_search_train)
    token = _label_token(labels_full[train_idx])
    key = (dataset_name, int(fold_i), int(n_folds), token)
    if key in _DNN_ENTRY_MEMO:
        return _DNN_ENTRY_MEMO[key]
    suffix = f'n09dnn_y{token}'
    model_obj = {BASE_RANK_MODEL: _BaselineDNN(
        random_state=42 + int(fold_i), is_clf=is_clf, classes_=classes_)}
    decided = list(decided_models(dataset_name, 'general'))
    common = dict(train_idx=train_idx, fold_i=int(fold_i),
                  n_outer_folds=int(n_folds), dataset_name=dataset_name,
                  augmentations=data.get('augmentations'),
                  task_type=task_type, condition_suffix=suffix)
    if BASE_RANK_MODEL in decided:
        grid_used = 'decided grid, models=dnn (dnn is one of the two decided)'
        fell_back = []
        searches = search_on_fold(data['reps'], labels_full,
                                  data.get('groups'), models=model_obj,
                                  **common)
    else:
        kw, fell_back = fast_model_search_kwargs(dataset_name,
                                                 BASE_RANK_MODEL)
        kw['models'] = model_obj
        grid_used = ('fast lists inside the decided cuts, models=dnn (dnn is '
                     f'not one of the decided {decided})')
        searches = search_on_fold(data['reps'], labels_full,
                                  data.get('groups'), fixed_grid=kw, **common)
    base_search = searches.get('none')
    entry_meta = {'search_grid': grid_used, 'fast_fell_back': list(fell_back),
                  'decided_models': decided, 'condition_suffix': suffix,
                  'search_labels': SEARCH_LABELS}
    if base_search is None:
        fails = getattr(searches, 'failures', None) or []
        why = ('the per-fold dnn search failed: '
               f"{[f.get('error') or f.get('error_type') for f in fails]}")
        print(f"    fold {fold_i}: {why}")
        entry = {'failed': why, **entry_meta}
        _DNN_ENTRY_MEMO[key] = entry
        return entry
    ctx = f'{dataset_name} n09 dnn fold {fold_i}'
    base = top_base_reps(base_search, BASE_RANK_MODEL, n=STANDARD_N_BASE,
                         context=ctx, require_full=False)
    pooled = []
    for cond, srch in searches.items():
        if srch is None:
            continue
        th = top_hybrids(srch, BASE_RANK_MODEL, n=STANDARD_N_HYBRIDS + 4,
                         context=ctx, require_full=False)
        pooled += [dict(e, aug_condition=cond) for e in th['selected']]
    lower = base_search.lower_is_better
    pooled.sort(key=lambda e: ((e['score_mean'] if lower else -e['score_mean']),
                               e['label']))
    skipped = []
    if len(pooled) > 1:
        # A second hybrid whose selected features equal the first's is
        # skipped (Adelaide 2026-10-05, handoff 2.1).
        # Built with _BaselineDNN and the conditions' metric, as the
        # fitted hybrid spaces are (review 2026-10-05).
        chosen, skipped = _distinct_dnn_hybrids(
            pooled, STANDARD_N_HYBRIDS, data=data, dataset_name=dataset_name,
            fold_i=fold_i, train_idx=train_idx, val_idx=val_idx,
            y_train=labels_full[train_idx],
            factory=lambda: _BaselineDNN(random_state=42 + int(fold_i),
                                         is_clf=is_clf, classes_=classes_),
            metric=metric)
    else:
        chosen = pooled[:STANDARD_N_HYBRIDS]
    # configuration_record reads 'evaluator' and 'n_reps'; top_hybrids names
    # them 'internal_evaluator' and 'rep_count'.
    chosen = [dict(h, evaluator=h.get('internal_evaluator'),
                   n_reps=h.get('rep_count')) for h in chosen]
    recipes, unavailable = {}, {}
    for i, sp in enumerate(HYBRID_FIT_SPACES):
        if len(chosen) > i:
            recipes[sp] = (chosen[i]['blueprint'],
                           chosen[i].get('aug_condition', 'none'))
        else:
            unavailable[sp] = (f'the per-fold dnn search gave fewer than '
                               f'{i + 1} hybrids with distinct features')
    bases = [e['rep'] for e in base['selected']]
    for i, sp in enumerate(BASE_FIT_SPACES):
        if len(bases) <= i:
            unavailable[sp] = base.get('shortfall_reason') or (
                f'the per-fold dnn search scored fewer than {i + 1} '
                f'representations')
    entry = {
        'base': bases,
        'base_training_row_scores': {e['rep']: e.get('score')
                                     for e in base['selected']},
        'hybrids': chosen,
        'hybrids_skipped_identical_features': skipped,
        'fit_configurations': bases + [h['label'] for h in chosen],
        'recipes': recipes,
        'unavailable': unavailable,
        'selection_scope': 'fold_train_rows',
        'lower_is_better': bool(lower),
        **entry_meta,
    }
    _DNN_ENTRY_MEMO[key] = entry
    return entry


def _neighbour_recipes(data, dataset_name, *, fold_i, train_idx, n_folds):
    """The distance hybrid and the similarity hybrid of one outer fold, as
    neighbour spaces only for distance_selection's sweep (handoff 2.3 item 1).

    From noise_helpers.fold_configurations: the dataset's standard per-fold
    searches on the clean training labels (shared with every other noise
    experiment's store). 'hybrid__distance1' is dnn's top hybrid of the
    distance search; 'hybrid_similarity' is the top hybrid of the similarity
    search, which fits no model. On a dataset whose two decided models lack
    dnn, 'hybrid__distance1' is the top distance hybrid of the first decided
    model (decided_models order) that has one. Returns ({space: (Blueprint,
    aug)}, {space: why unavailable}, {space: model it came from}).
    """
    key = (dataset_name, int(fold_i), int(n_folds))
    if key in _NEIGHBOUR_MEMO:
        return _NEIGHBOUR_MEMO[key]
    from experiments.decided_grid import decided_models
    recipes, unavailable, source = {}, {}, {}
    try:
        conf = fold_configurations(data, dataset_name, fold_i=int(fold_i),
                                   train_idx=train_idx, n_folds=int(n_folds))
    except Exception as exc:
        why = (f'the per-fold standard searches raised '
               f'{type(exc).__name__}: {exc}')
        _NEIGHBOUR_MEMO[key] = ({}, {'hybrid__distance1': why,
                                     'hybrid_similarity': why}, {})
        return _NEIGHBOUR_MEMO[key]
    if BASE_RANK_MODEL in conf:
        dnn_e = conf.get(BASE_RANK_MODEL)
        if dnn_e is not None and 'hybrid__distance1' in dnn_e['recipes']:
            recipes['hybrid__distance1'] = dnn_e['recipes']['hybrid__distance1']
            source['hybrid__distance1'] = BASE_RANK_MODEL
        else:
            unavailable['hybrid__distance1'] = (
                'the per-fold dnn search failed' if dnn_e is None else
                dnn_e['unavailable'].get(
                    'hybrid__distance1',
                    'no distance hybrid for dnn on this fold'))
    else:
        # DECIDED 2026-10-06 (Adelaide delegated): on a dataset whose two
        # decided models lack dnn, the neighbour space is the other decided
        # model's distance hybrid: the first decided model, in decided_models
        # order, whose per-fold distance search gave one.
        order = [m for m in decided_models(dataset_name, 'general')
                 if m in conf] + [m for m in conf
                                  if m not in decided_models(dataset_name,
                                                             'general')]
        whys = {}
        for m in order:
            e = conf.get(m)
            if e is not None and 'hybrid__distance1' in e['recipes']:
                recipes['hybrid__distance1'] = e['recipes']['hybrid__distance1']
                source['hybrid__distance1'] = m
                break
            whys[m] = ('the per-fold search failed' if e is None else
                       e['unavailable'].get('hybrid__distance1',
                                            'no distance hybrid'))
        else:
            unavailable['hybrid__distance1'] = (
                f'dnn is not decided here and no decided model has a distance '
                f'hybrid on this fold: {whys}')
    for m, e in conf.items():
        if e is not None and 'hybrid_similarity' in e['recipes']:
            recipes['hybrid_similarity'] = e['recipes']['hybrid_similarity']
            source['hybrid_similarity'] = m
            break
    else:
        whys = [e['unavailable'].get('hybrid_similarity') for e in conf.values()
                if e is not None]
        unavailable['hybrid_similarity'] = (whys[0] if whys and whys[0] else
                                            'the per-fold similarity search '
                                            'gave no hybrid')
    _NEIGHBOUR_MEMO[key] = (recipes, unavailable, source)
    return recipes, unavailable, source


def _weighted_copy(data, dataset_name, entry, *, train_idx, val_idx, y_train,
                   factory, metric, X_hyb_tr, X_hyb_va):
    """The importance-weighted copy of hybrid 1 (conditions E/F of the old
    file), or (None, None, why).

    RESTORED 2026-10-07 (Adelaide), as it was before f2d2c98.

    The weights are the per-fold importance scores of the build's own
    selected features (flatten_fold_weights), square-rooted and normalised to
    mean 1, multiplied into the standardised hybrid matrix the hybrid arm is
    trained on (AUDIT P1#4: no re-scaling afterwards, which would divide the
    weights straight back out).

    fold_model_inputs does not return which features a build selected, so
    the build is asked once more for its feature_info through the same cache
    entry kind (want_feature_info=True is its own entry). The two builds have
    the same inputs; the copy is made only when the second build's
    standardised matrix equals the hybrid arm's, so the weights are known to
    belong to those features.
    """
    from experiments._shared import flatten_fold_weights
    from experiments.noise import noise_cache as nc
    from experiments.noise.n07_remediation import _scale_pair
    rec = (entry.get('recipes') or {}).get('hybrid')
    if rec is None:
        return None, None, 'no hybrid 1 on this fold'
    bp, aug_cond = rec
    tr = np.asarray(train_idx)
    va = np.asarray(val_idx)
    reps = data['reps']
    augs = data.get('augmentations') if (aug_cond or 'none') != 'none' else None
    metric_name, metric_fn, lower = metric
    g = data.get('groups')
    built = nc.cached_hybrid_for_fold(
        dataset_name, {k: v[tr] for k, v in reps.items()},
        {k: v[va] for k, v in reps.items()}, y_train, bp,
        None if g is None else np.asarray(g)[tr], factory(), metric_fn, lower,
        model_name=BASE_RANK_MODEL, metric_name=metric_name, train_idx=tr,
        augmentations_train=(None if augs is None else
                             {a: X[tr] for a, X in augs.items()}),
        augmentations_val=(None if augs is None else
                           {a: X[va] for a, X in augs.items()}),
        want_feature_info=True)
    if built is None or built[0] is None:
        return None, None, 'the feature_info build failed'
    X_tr_s, _X_va_s = _scale_pair(built[0], built[1])
    if (X_tr_s.shape != X_hyb_tr.shape
            or not np.allclose(X_tr_s, X_hyb_tr, equal_nan=True)):
        return None, None, ('the feature_info build differs from the hybrid '
                            'arm\'s matrix, so its weights cannot be matched '
                            'to those features')
    fold_weights = flatten_fold_weights(built[2])
    if len(fold_weights) != X_hyb_tr.shape[1]:
        return None, None, (f'weight count {len(fold_weights)} differs from '
                            f'the hybrid width {X_hyb_tr.shape[1]}')
    w = np.sqrt(np.maximum(fold_weights, 0))
    w = (w / (w.mean() + 1e-10)).astype(X_hyb_tr.dtype, copy=False)
    return X_hyb_tr * w, X_hyb_va * w, None


def _top_screened_lists(data, dataset_name, settings, *, X_tr, X_va,
                        train_idx, val_idx, y_train, y_val, groups_tr,
                        recipes, base_reps, factory, metric, task_type,
                        is_clf):
    """The top (setting, k, scheme) of the neighbour-setting screen for one
    pass and feature space, and its neighbour lists with the scheme on them.

    DECIDED 2026-10-06 (Adelaide delegated): distance_selection in the method
    survey takes its neighbours from here instead of its own 10 nearest
    neighbours in the trained features. The spaces, lists and screen are built
    exactly as n07.similarity_sweep builds them for the same (feature space,
    fold, draw), through the same caches, and the screen gets inner= so it
    predicts a held-back fifth of the training molecules.

    Returns (lists, k, record) or (None, None, record) with record['reason'].
    """
    from experiments.noise.n07_remediation import (
        fold_similarity_spaces, fold_neighbours, screen_settings)
    if not settings:
        return None, None, {'reason': 'no similarity settings for this dataset'}
    spaces_tr, spaces_va, _unav = fold_similarity_spaces(
        dataset_name, settings, model_input_tr=X_tr, model_input_va=X_va,
        reps=data['reps'], train_idx=train_idx, val_idx=val_idx,
        y_train=y_train, groups_train=groups_tr, base_reps=base_reps,
        recipes=recipes, augmentations=data.get('augmentations'),
        model_name=BASE_RANK_MODEL, model_factory=factory, metric=metric,
        task_type=task_type)
    smi = data.get('smiles')
    smi_tr = None if smi is None else [smi[i] for i in np.asarray(train_idx)]
    smi_va = None if smi is None else [smi[i] for i in np.asarray(val_idx)]
    nbrs = fold_neighbours(dataset_name, settings, spaces_tr, spaces_va,
                           y_train, smiles_train=smi_tr, smiles_val=smi_va,
                           is_clf=is_clf)
    kept, table = screen_settings(
        settings, nbrs, y_train, metric=metric, is_clf=is_clf, y_val=y_val,
        inner={'dataset_name': dataset_name, 'spaces_train': spaces_tr,
               'y_train': y_train, 'groups_train': groups_tr,
               'smiles_train': smi_tr, 'is_clf': is_clf})
    if not kept:
        return None, None, {'reason': 'the screen scored no (setting, k, '
                                      'scheme) on this pass'}
    s, k, scheme = kept[0]
    rec = {'setting': s.name, 'k': int(k), 'scheme': scheme,
           'scored_on': next((r.get('scored_on') for r in table), None),
           'score': next((r.get('score') for r in table
                          if (r['setting'], r['k'], r['scheme'])
                          == (s.name, k, scheme)), None)}
    return dict(nbrs[s.name], scheme=scheme), int(k), rec


def _per_fold_means(folds, scores):
    """Average the draws inside each fold: {fold: mean}, fold order kept."""
    by = {}
    for f, s in zip(folds, scores):
        by.setdefault(int(f), []).append(float(s))
    return {f: float(np.mean(v)) for f, v in by.items()}


def _agg(scores, folds, draws):
    """One arm at one level: the mean over (fold, draw) passes, the per-pass
    scores, and the spread over folds (draws averaged inside each fold first;
    noise_helpers.fold_spread, 'one fold, no spread' on one fold)."""
    if not scores:
        return {'mean': float('nan'), 'n_folds': 0, 'folds': [],
                'fold_draws': [], 'fold_scores': [],
                'spread_over_folds': fold_spread([])}
    return {
        'mean': float(np.mean(scores)),
        # n_folds counts (fold, draw) passes, as it always has here.
        'n_folds': len(scores),
        # The folds behind that mean. Compare two conditions only over the
        # (fold, draw) passes both of them scored.
        'folds': list(folds),
        'fold_draws': list(draws),
        'fold_scores': [float(x) for x in scores],
        'spread_over_folds': fold_spread(
            list(_per_fold_means(folds, scores).values())),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """n09_lnl_survey across every decided damage kind for this dataset.

    The kind is an axis this experiment sweeps, not something a caller runs it
    once per. The sweep, the per-kind checkpointing, the resume and the stride
    slice are noise_helpers.sweep_conditions. Each kind's file is written as it
    finishes, and inside a kind each (level, fold, draw) pass is checkpointed
    as it finishes (n07.FoldCheckpoint).

    Its headline is a downstream score, which is defined under every kind, so
    it sweeps the whole decided list rather than only the kinds where a
    corrupted SUBSET exists. It also records detection, and detection_scores
    already returns None where that is undefined.
    """
    # RESULT_LAYOUT (and the env fold filter) enter each condition file's
    # input fingerprint through 'noise_strategies', as n07 does, so a file
    # written by an earlier n09 version is recomputed, not skipped.
    fp_config = dict(config or {})
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, 'n09_lnl_survey',
        data, fp_config, output_dir, dataset_name,
        default_conditions=decided_conditions(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N09: LNL METHOD SURVEY WITH HYBRIDS")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    noise_condition = kwargs.get('noise_condition') or decided_conditions(data)[0]
    # The decided levels for this kind on this dataset's group, 0 first
    # (noise_helpers.decided_levels; KNN record, noise grid).
    levels = [float(x) for x in decided_levels(data, dataset_name,
                                               noise_condition)]
    noise_groups, _n_ng, _lg = noise_groups_for(data)
    noise_provenance = {}
    methods_to_test = kwargs.get('methods', ALL_METHODS)

    # heteroscedastic_gp and evidential_kernel have no classification analog
    # (continuous-noise-variance / NIG distribution math doesn't transform via
    # a loss-function swap). All 10 other methods support clf via MSE->CE in
    # their inner training. See REGRESSION_ONLY_LNL in lnl_methods.py.
    if is_clf:
        methods_to_test = [m for m in methods_to_test
                           if m not in REGRESSION_ONLY_LNL]

    # Resolve the full class set up front so per-fold y_train subsets all
    # train classifiers with the same n_classes outputs.
    classes_ = np.unique(labels) if is_clf else None

    # Outer folds: the ones the per-fold searches are filed under, count from
    # config['n_splits'] (handoff 2.4 item 14). Any count, including 1.
    outer_folds, n_folds = noise_outer_folds(data, config)

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name} "
          f"({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Damage: {noise_condition}, levels {levels} "
          f"({level_unit(noise_condition, is_clf)})")
    print(f"  Methods: {len(methods_to_test)}")
    print(f"  Folds: {[f for f, _, _ in outer_folds]} of {n_folds}")
    print(f"  Per-fold dnn search on the {SEARCH_LABELS} training labels")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'models': [BASE_RANK_MODEL],
        'models_note': ('dnn only: every condition trains the neural network '
                        '(handoff 2.3 group B, n09). The feature spaces are '
                        "dnn's 2 best single representations and 2 best "
                        'hybrids per fold and label set, from one search per '
                        '(kind, level, draw, fold) on the damaged training '
                        'labels, plus the weighted copy of hybrid 1. No '
                        'second model.'),
        'configuration_spaces': list(ALL_SPACES),
        'selection_scope': 'fold_train_rows',
        'search_labels': SEARCH_LABELS,
        'result_layout': RESULT_LAYOUT,
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in outer_folds],
        'knn_group': dataset_knn_group(data, dataset_name),
        # 'sigmas' kept as the key analysers read; the values are the levels.
        'sigmas': levels,
        'levels': levels,
        'level_unit': level_unit(noise_condition, is_clf),
        'noise_draws_per_nonzero_level': len(draws_for(noise_condition,
                                                       NOISE_DRAWS)),
        'noise_condition': noise_condition,
        'noise_provenance': noise_provenance,
        'methods': methods_to_test,
        'task_type': task_type,
        'is_clf': is_clf,
        'n_classes': int(len(classes_)) if is_clf else None,
        # One entry per (level, fold, draw) pass: which representations and
        # hybrids dnn was given on that pass (configuration_record), their
        # scores on the fold's training rows (the search's inner folds), which
        # spaces could not be built and why.
        'configurations': [],
        'per_sigma': {},
    }

    # ── distance_selection's neighbour sweep (kNN record) ─────────────────
    # distance_selection finds neighbours and drops training molecules, so it
    # is swept over this dataset group's similarity settings, k in (5, 10, 20)
    # and both counting schemes; the per-dataset screen keeps the top 3
    # (setting, k, scheme) and only those are refitted. Its kept rows are
    # written as D-versus-D' rows.
    #
    # TODO(similarity-sweep 2026-09-25): the other eleven methods change how
    # the DNN trains, not the training set, so there is no D' to fit other
    # models on and they write no rows. small_loss and confident_learning do
    # drop molecules but do not return which; mentornet reads a neighbour
    # list but carves its checkpoint-selection rows out of X_train first
    # (lnl_methods.py, MentorNet.train_and_predict), so a list over the
    # training fold no longer lines up and it is not swept. Each needs a
    # decision on what its D' is.
    from experiments.noise.n07_remediation import (
        FoldCheckpoint, similarity_record, sweep_settings, RowSink,
        sweep_context, similarity_sweep)
    from experiments._shared import input_fingerprint
    smiles = data.get('smiles')
    n_molecules = (len(set(smiles)) if smiles is not None and len(smiles)
                   else len(labels))
    sim_settings = sweep_settings(dataset_name, n_molecules,
                                  has_smiles=smiles is not None)
    sim_sink = RowSink(output_dir, 'n09_lnl_survey', dataset_name,
                       kwargs.get('noise_condition'))
    # factories: only dnn's, read for the accept-or-reject step of the
    # neighbour-space hybrid builds; set per fold below (its seed is the
    # fold's). The D' rows are fitted by n07.row_models with dnn only.
    sim_ctx = sweep_context(
        experiment='n09_lnl_survey', dataset_name=dataset_name, sink=sim_sink,
        settings=sim_settings, metric=metric, task_type=task_type,
        condition=noise_condition, level=None, data=data, factories={})
    _ds_by_sim = {}
    neighbour_space_record = {}
    ckpt = FoldCheckpoint(output_dir, 'n09_lnl_survey', dataset_name,
                          condition=kwargs.get('noise_condition'),
                          tag='passes', fingerprint={
                              'input': input_fingerprint(data, config),
                              'noise_condition': noise_condition,
                              'methods': list(methods_to_test),
                              'levels': levels,
                              'n_folds': int(n_folds),
                              'search_labels': SEARCH_LABELS,
                              'alignment': ALIGNMENT_VERSION,
                              'result_layout': RESULT_LAYOUT})

    _plain_keys = [f'{c}_no_lnl' for c in ALL_SPACES]
    _lnl_keys = [f'{c}_lnl' for c in ALL_SPACES]

    for sigma in levels:
        print(f"\n  === Level {sigma} ===")
        t_sigma = time.time()

        sigma_results = {k: [] for k in _plain_keys}
        # Which fold each score came from. A fold where a hybrid build fails
        # drops out of that arm but not the others, so without this two means
        # are averages over different folds and their difference is not a
        # paired comparison.
        sigma_folds = {k: [] for k in _plain_keys}
        # Which noise draw each score came from, aligned with sigma_folds.
        sigma_draws = {k: [] for k in _plain_keys}
        # The methods run at level 0 too (owner's decision 2026-09-27): the
        # plain DNN on the labels as delivered (D) against each method on the
        # same labels (D'). Detection is not graded at 0.
        for _ck in _lnl_keys:
            sigma_results[_ck] = {m: [] for m in methods_to_test}
            sigma_folds[_ck] = {m: [] for m in methods_to_test}
            sigma_draws[_ck] = {m: [] for m in methods_to_test}
        detection_acc = {}

        # THREE NOISE DRAWS PER FOLD at a non-zero level, averaged (user
        # decision 2026-09-12); one under censoring, whose draws are
        # identical (noise_helpers.draws_for, 2026-10-06). Level 0 is the labels as delivered on every
        # task, so it runs once (dirty-run bug 6 closed by noise_spec).
        _draws = (draws_for(noise_condition, NOISE_DRAWS) if sigma > 0
                  else NOISE_DRAWS[:1])
        _passes = [(_fi, _tr, _va, _nd)
                   for _fi, _tr, _va in outer_folds for _nd in _draws]
        _outer_results = sigma_results
        for fold_idx, train_idx, val_idx, noise_seed in _passes:
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Damage on the training rows only (noise_helpers.inject_condition
            # dispatches every decided kind). Level 0 returns the labels as
            # delivered and an all-False mask.
            y_train, _cmask, _prov = inject_condition(
                data, dataset_name, noise_condition, sigma, noise_seed,
                train_idx=train_idx, groups=noise_groups)
            noise_provenance.setdefault(
                f'{noise_condition}|{sigma}|{noise_seed}', _prov)

            # ── One (level, fold, draw) pass, checkpointed ──
            # n09 changes no training data for eleven of its methods, so each
            # pass is written to a checkpoint as it finishes and read back on
            # resume (n07.FoldCheckpoint). The pass writes into pass-local
            # accumulators, appended to the level's in the same order, so the
            # means are the same numbers whether a pass ran or was read back.
            def _one_pass(fold_idx=fold_idx, train_idx=train_idx,
                          val_idx=val_idx, noise_seed=noise_seed,
                          y_train=y_train, y_val=y_val, _cmask=_cmask,
                          y_train_clean=y_train_clean):
                sigma_results = {k: ([] if not isinstance(v, dict) else
                                     {m: [] for m in v})
                                 for k, v in _outer_results.items()}
                sigma_folds = {k: ([] if not isinstance(v, dict) else
                                   {m: [] for m in v})
                               for k, v in _outer_results.items()}
                sigma_draws = {k: ([] if not isinstance(v, dict) else
                                   {m: [] for m in v})
                               for k, v in _outer_results.items()}
                detection_acc = {}
                train_groups_subset = (np.asarray(groups)[train_idx]
                                       if groups is not None else None)

                # dnn's 2 + 2 for this fold, searched on its training rows
                # with the labels SEARCH_LABELS names (RESTORED 2026-10-07:
                # this pass's damaged labels).
                y_search = (y_train if SEARCH_LABELS == 'damaged'
                            else y_train_clean)
                try:
                    entry = _dnn_fold_entry(
                        data, dataset_name, fold_i=fold_idx,
                        train_idx=train_idx, val_idx=val_idx,
                        n_folds=n_folds, y_search_train=y_search,
                        is_clf=is_clf, classes_=classes_,
                        task_type=task_type, metric=metric)
                except Exception as exc:
                    from kirby.result_cache import StaleResultDB as _Stale
                    if isinstance(exc, _Stale):
                        raise
                    entry = {'failed': (f'the per-fold dnn search raised '
                                        f'{type(exc).__name__}: {exc}')}
                    print(f"    fold {fold_idx}: {entry['failed']}")

                def _factory(fold_idx=fold_idx):
                    return _BaselineDNN(random_state=42 + fold_idx,
                                        is_clf=is_clf, classes_=classes_)

                X_by_space, unavailable, names = {}, {}, {}
                if entry.get('failed'):
                    unavailable = {sp: entry['failed'] for sp in ALL_SPACES}
                else:
                    # Built on this fold's training rows with the labels the
                    # pass holds; hybrids from the per-fold recipe, whose
                    # representation scores are the fold search's own
                    # (dirty-run bug 1). The accept-or-reject step fits the
                    # same plain DNN every condition is scored with.
                    X_by_space, unavailable, names = fold_model_inputs(
                        data, dataset_name, BASE_RANK_MODEL, entry,
                        train_idx=train_idx, val_idx=val_idx,
                        y_train=y_train, factory=_factory, metric=metric,
                        spaces=FIT_SPACES)
                    # RESTORED 2026-10-07 (Adelaide): the importance-weighted
                    # copy of hybrid 1 is a fifth feature space beyond the
                    # 2 + 2, because weighted-versus-plain hybrid is part of
                    # this experiment's question.
                    if 'hybrid' in X_by_space:
                        try:
                            Xw_tr, Xw_va, why = _weighted_copy(
                                data, dataset_name, entry,
                                train_idx=train_idx, val_idx=val_idx,
                                y_train=y_train, factory=_factory,
                                metric=metric,
                                X_hyb_tr=X_by_space['hybrid'][0],
                                X_hyb_va=X_by_space['hybrid'][1])
                        except Exception as exc:
                            from kirby.result_cache import \
                                StaleResultDB as _Stale
                            if isinstance(exc, _Stale):
                                raise
                            Xw_tr, Xw_va = None, None
                            why = (f'weighted build raised '
                                   f'{type(exc).__name__}: {exc}')
                        if Xw_tr is not None:
                            X_by_space[WEIGHTED_SPACE] = (Xw_tr, Xw_va)
                        else:
                            unavailable[WEIGHTED_SPACE] = why
                    else:
                        unavailable.setdefault(
                            WEIGHTED_SPACE, 'no hybrid 1 on this pass')
                for sp in ALL_SPACES:
                    if sp not in X_by_space:
                        print(f"    fold {fold_idx} draw {noise_seed}: "
                              f"{sp} unavailable: {unavailable.get(sp)}")

                # The neighbour spaces distance_selection reads: dnn's own
                # base and hybrid recipes of this fold plus the distance and
                # similarity hybrids of the fold's standard searches.
                _uses_neighbours = ('distance_selection' in methods_to_test
                                    and not entry.get('failed'))
                sim_recipes = dict(entry.get('recipes') or {})
                if _uses_neighbours:
                    nb_rec, nb_unavail, nb_source = _neighbour_recipes(
                        data, dataset_name, fold_i=fold_idx,
                        train_idx=train_idx, n_folds=n_folds)
                    neighbour_space_record.setdefault(str(fold_idx), {
                        'spaces': sorted(nb_rec), 'unavailable': nb_unavail,
                        'model_each_came_from': nb_source})
                    sim_recipes.update(nb_rec)

                # DECIDED 2026-10-06 (Adelaide delegated): distance_selection
                # in the method survey uses the top (setting, k, scheme) of
                # this pass's screen on each feature space, not its own 10
                # neighbours in the trained features. Where the screen gives
                # nothing, distance_selection is recorded as not run.
                screened, screened_rec = {}, {}
                if _uses_neighbours:
                    for sp in ALL_SPACES:
                        if sp not in X_by_space:
                            continue
                        try:
                            _L, _k, _rec = _top_screened_lists(
                                data, dataset_name, sim_settings,
                                X_tr=X_by_space[sp][0], X_va=X_by_space[sp][1],
                                train_idx=train_idx, val_idx=val_idx,
                                y_train=y_train, y_val=y_val,
                                groups_tr=train_groups_subset,
                                recipes=sim_recipes,
                                base_reps=list(entry.get('base') or []),
                                factory=_factory, metric=metric,
                                task_type=task_type, is_clf=is_clf)
                        except Exception as exc:
                            from kirby.result_cache import \
                                StaleResultDB as _Stale
                            if isinstance(exc, _Stale):
                                raise
                            _L, _k, _rec = None, None, {
                                'reason': (f'the screen raised '
                                           f'{type(exc).__name__}: {exc}')}
                        screened_rec[sp] = _rec
                        if _L is not None:
                            screened[sp] = (_L, _k)
                        else:
                            print(f"    fold {fold_idx} draw {noise_seed}: "
                                  f"{sp} distance_selection not run: "
                                  f"{_rec.get('reason')}")

                config_rec = {
                    'level': float(sigma), 'fold': int(fold_idx),
                    'draw': int(noise_seed),
                    'configuration': (configuration_record(entry, names)
                                      if not entry.get('failed') else
                                      {'available': False}),
                    'base_training_row_scores':
                        entry.get('base_training_row_scores'),
                    'search_grid': entry.get('search_grid'),
                    'fast_fell_back': entry.get('fast_fell_back'),
                    'condition_suffix': entry.get('condition_suffix'),
                    'unavailable': {sp: str(unavailable.get(sp))
                                    for sp in ALL_SPACES
                                    if sp not in X_by_space},
                    # distance_selection's neighbours per feature space: the
                    # screen's top (setting, k, scheme), or why it was not run.
                    'distance_selection_neighbours': screened_rec,
                }

                lnl_kwargs = dict(random_state=42 + fold_idx,
                                  is_clf=is_clf, classes_=classes_)
                for sp in ALL_SPACES:
                    if sp not in X_by_space:
                        continue
                    X_tr_s, X_va_s = X_by_space[sp]
                    # Plain DNN on this feature space.
                    try:
                        y_pred = _run_baseline(X_tr_s, y_train, X_va_s, y_val,
                                               random_state=42 + fold_idx,
                                               is_clf=is_clf, classes_=classes_)
                        sc = float(metric_fn(y_val, y_pred))
                        sigma_results[f'{sp}_no_lnl'].append(sc)
                        sigma_folds[f'{sp}_no_lnl'].append(fold_idx)
                        sigma_draws[f'{sp}_no_lnl'].append(noise_seed)
                    except Exception as e:
                        print(f"    {sp} plain DNN failed: {e}")
                    # Each label-noise method on this feature space.
                    _ck = f'{sp}_lnl'
                    for method_name in methods_to_test:
                        try:
                            if method_name == 'distance_selection':
                                if sp not in screened:
                                    continue
                                _L, _k = screened[sp]
                                method = _SchemeDistanceSelection(
                                    k=_k, precomputed_neighbors=_L,
                                    **lnl_kwargs)
                            else:
                                method = get_lnl_method(method_name,
                                                        **lnl_kwargs)
                            # Keeps each method's checkpoint-selection carve
                            # scaffold-disjoint inside the training fold.
                            method.selection_groups = train_groups_subset
                            res = method.train_and_predict(X_tr_s, y_train,
                                                           X_va_s, y_val)
                            sc = float(metric_fn(y_val, res['y_pred']))
                            _record_detection(detection_acc, _ck, method_name,
                                              res, _cmask, noise_condition)
                            sigma_results[_ck][method_name].append(sc)
                            sigma_folds[_ck][method_name].append(fold_idx)
                            sigma_draws[_ck][method_name].append(noise_seed)
                        except Exception as e:
                            print(f"    {sp}/{method_name} failed: {e}")

                # ── distance_selection under the screened similarity settings
                # On each of the four fitted spaces (not the weighted copy, as
                # before 2026-10-06). The screen keeps the top 3 (setting, k,
                # scheme), and only those are refitted.
                _sim_scores = []
                if _uses_neighbours and sim_sink.enabled:
                    sim_ctx['factories'] = {BASE_RANK_MODEL: _factory}
                    for _arm in FIT_SPACES:
                        if _arm not in X_by_space:
                            continue
                        _Xtr, _Xva = X_by_space[_arm]

                        def _ds_dprimes(setting, lists, k, _Xtr=_Xtr,
                                        _Xva=_Xva, _arm=_arm):
                            try:
                                _mth = _SchemeDistanceSelection(
                                    k=k, precomputed_neighbors=lists,
                                    **lnl_kwargs)
                                _mth.selection_groups = train_groups_subset
                                _res = _mth.train_and_predict(_Xtr, y_train,
                                                              _Xva, y_val)
                            except Exception as e:
                                print(f"    distance_selection "
                                      f"{setting.name} k={k} failed: {e}")
                                yield 'distance_selection', 'threshold_pct=85', None
                                return
                            _sim_scores.append({
                                'arm': _arm, 'setting': setting.name,
                                'k': int(k), 'scheme': lists.get('scheme'),
                                'score': float(metric_fn(y_val,
                                                         _res['y_pred']))})
                            # The rows the method kept, rebuilt from what it
                            # returns: flagged = suspicion above its
                            # threshold, every row kept when fewer than 10
                            # would remain.
                            _md = _res['metadata']
                            _keep = ~(np.asarray(_md['suspicion_scores'])
                                      > float(_md['threshold']))
                            if _keep.sum() < 10:
                                _keep = np.ones(len(y_train), dtype=bool)
                            yield ('distance_selection', 'threshold_pct=85',
                                   {'X': _Xtr[_keep],
                                    'y': np.asarray(y_train)[_keep],
                                    'keep': _keep, 'sample_weight': None,
                                    'label_column': None})
                        # DECIDED 2026-10-06 (Adelaide delegated): the D'
                        # rows are fitted with dnn only. n07.row_models fits
                        # the model_name passed here and nothing else.
                        similarity_sweep(
                            sim_ctx, model_name=BASE_RANK_MODEL,
                            model_input=f'{BASE_RANK_MODEL}:{_arm}',
                            X_tr=_Xtr, X_va=_Xva,
                            y_tr_noisy=y_train, y_tr_clean=y_train_clean,
                            y_val=y_val, train_idx=train_idx,
                            val_idx=val_idx,
                            groups_tr=train_groups_subset, fold=fold_idx,
                            draw=noise_seed, recipes=sim_recipes,
                            base_reps=list(entry.get('base') or []),
                            make_dprimes=_ds_dprimes, level=sigma)
                return {'results': sigma_results, 'folds': sigma_folds,
                        'draws': sigma_draws, 'detection': detection_acc,
                        'configuration': config_rec,
                        'sim': _sim_scores}

            _pass = ckpt.cached([float(sigma), int(fold_idx), int(noise_seed)],
                                _one_pass)
            for _ck, _v in _pass['results'].items():
                if isinstance(_v, dict):
                    for _m, _xs in _v.items():
                        sigma_results[_ck][_m].extend(_xs)
                        sigma_folds[_ck][_m].extend(_pass['folds'][_ck][_m])
                        sigma_draws[_ck][_m].extend(_pass['draws'][_ck][_m])
                else:
                    sigma_results[_ck].extend(_v)
                    sigma_folds[_ck].extend(_pass['folds'][_ck])
                    sigma_draws[_ck].extend(_pass['draws'][_ck])
            for _ck, _by_m in _pass['detection'].items():
                for _m, _rows in _by_m.items():
                    detection_acc.setdefault(_ck, {}).setdefault(
                        _m, []).extend(_rows)
            results['configurations'].append(_pass['configuration'])
            for _r in _pass.get('sim') or []:
                _kk = f"k={_r['k']}|scheme={_r.get('scheme')}"
                (_ds_by_sim.setdefault(str(sigma), {}).setdefault(_r['arm'], {})
                 .setdefault(_r['setting'], {}).setdefault(_kk, [])
                 .append(_r['score']))

        # Aggregate the level's passes.
        agg = {}
        for cond_key in _plain_keys:
            agg[cond_key] = _agg(sigma_results[cond_key],
                                 sigma_folds[cond_key], sigma_draws[cond_key])
        for cond_key in _lnl_keys:
            agg[cond_key] = {
                m: _agg(sigma_results[cond_key][m], sigma_folds[cond_key][m],
                        sigma_draws[cond_key][m])
                for m in methods_to_test}

        if detection_acc:
            # How well each method FOUND the corrupted molecules, before any
            # repair. Present only under kinds where a corrupted subset exists.
            agg['detection'] = {
                _ck: {_m: _mean_detection(_rows) for _m, _rows in _by_method.items()}
                for _ck, _by_method in detection_acc.items()}

        results['per_sigma'][str(sigma)] = convert_numpy(agg)

        print("    " + ", ".join(
            f"{sp} DNN: {agg[f'{sp}_no_lnl']['mean']:.4f}" for sp in ALL_SPACES))
        for cond in _lnl_keys:
            best_name, best_score = _best_method(agg[cond], methods_to_test,
                                                 lower_is_better)
            if best_name:
                print(f"    Best {cond}: {best_name} ({best_score:.4f})")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # =========================================================================
    # DIFFERENCE FROM LEVEL 0
    # =========================================================================
    # For each (level, condition), against the same feature space's level-0
    # score:
    #   delta_from_clean: signed (damaged - level 0), in the dataset's metric.
    #     The robustness number on fast_knn datasets (auc_norm absent there by
    #     decision; KNN record, noise grid).
    #   score_ratio_pct: ratio as a percentage; only defined when both scores
    #     are positive (a ratio of negative R^2 is meaningless).
    clean_sigma = results['per_sigma'].get(str(0.0), {})
    clean_ref = {sp: clean_sigma.get(f'{sp}_no_lnl', {}).get('mean', float('nan'))
                 for sp in ALL_SPACES}

    def _score_ratio_pct(noisy_val, clean_val):
        """Score ratio in percent. A plain ratio, not bounded to 100. NaN when
        either side is missing or not positive."""
        if np.isnan(noisy_val) or np.isnan(clean_val):
            return float('nan')
        if clean_val <= 1e-10 or noisy_val <= 1e-10:
            return float('nan')
        if lower_is_better:
            return float(clean_val / noisy_val * 100)
        return float(noisy_val / clean_val * 100)

    def _pair(noisy_val, clean_val):
        delta = (float('nan') if np.isnan(noisy_val) or np.isnan(clean_val)
                 else float(noisy_val - clean_val))
        return {'delta_from_clean': delta,
                'score_ratio_pct': _score_ratio_pct(noisy_val, clean_val)}

    retention = {}
    for sigma_key, sigma_data in results['per_sigma'].items():
        if float(sigma_key) == 0.0:
            continue
        sigma_ret = {}
        for sp in ALL_SPACES:
            noisy = sigma_data.get(f'{sp}_no_lnl', {}).get('mean', float('nan'))
            sigma_ret[f'{sp}_no_lnl'] = _pair(noisy, clean_ref[sp])
            sigma_ret[f'{sp}_lnl'] = {
                m: _pair(sigma_data.get(f'{sp}_lnl', {}).get(m, {}).get(
                    'mean', float('nan')), clean_ref[sp])
                for m in methods_to_test}
        retention[sigma_key] = sigma_ret
    results['retention'] = convert_numpy(retention)
    results['retention_schema'] = (
        'Each entry has {delta_from_clean: damaged minus level 0 for the same '
        'feature space, in the dataset metric; score_ratio_pct: ratio*100 when '
        'both sides are positive else NaN}. lower_is_better gives the '
        'direction.')

    # =========================================================================
    # THE HEADLINE, 2026-09-27 (owner's decision): each method against the
    # plain DNN trained on THE SAME LABELS -- same level, fold, noise draw and
    # feature space. gain is sign-normalised: positive means the method beat
    # the plain DNN, in the dataset's own metric. Passes are paired on
    # (fold, draw); a pass only one side scored is left out and counted.
    # =========================================================================
    vs_plain = {}
    for sigma_key, sigma_data in results['per_sigma'].items():
        per_cond = {}
        for cond in _lnl_keys:
            plain = sigma_data.get(cond[:-len('_lnl')] + '_no_lnl', {})
            per_method = {}
            for method_name in methods_to_test:
                meth = sigma_data.get(cond, {}).get(method_name, {})
                pairs, n_unpaired = _paired(plain, meth)
                gains = [((p - m) if lower_is_better else (m - p))
                         for _, p, m in pairs]
                per_fold_gain = _per_fold_means([k[0] for k, _, _ in pairs],
                                                gains)
                per_method[method_name] = {
                    'plain_dnn_mean': (float(np.mean([p for _, p, _ in pairs]))
                                       if pairs else float('nan')),
                    'method_mean': (float(np.mean([m for _, _, m in pairs]))
                                    if pairs else float('nan')),
                    'gain_mean': (float(np.mean(gains)) if gains
                                  else float('nan')),
                    'gain_per_pass': [float(g) for g in gains],
                    'gain_spread_over_folds': fold_spread(
                        list(per_fold_gain.values())),
                    'passes': [list(k) for k, _, _ in pairs],
                    'passes_won': int(sum(g > 0 for g in gains)),
                    'n_passes': len(gains),
                    'n_passes_scored_by_one_side_only': int(n_unpaired),
                }
            per_cond[cond] = per_method
        vs_plain[sigma_key] = per_cond
    results['vs_plain_dnn_same_labels'] = convert_numpy(vs_plain)
    results['vs_plain_dnn_same_labels_schema'] = (
        'Per level, per feature space, per method: the plain DNN and the '
        'method, both trained on the same labels (as delivered at level 0, '
        'damaged above it), scored on the same held-out fold, paired on '
        '(fold, draw). gain > 0 means the method beat the plain DNN, in the '
        'dataset metric. This is the headline; `retention` compares against '
        'level 0 instead.')
    print("\n  Each method against the plain DNN on the same labels "
          "(passes won of passes, mean gain):")
    for sigma_key, per_cond in vs_plain.items():
        for cond, per_method in per_cond.items():
            cells = [f"{m} {r['passes_won']} of {r['n_passes']} "
                     f"{r['gain_mean']:+.4f}"
                     for m, r in per_method.items() if r['n_passes']]
            if cells:
                print(f"    level {sigma_key} {cond}: " + ", ".join(cells))

    # =========================================================================
    # SUMMARY, at the highest level
    # =========================================================================
    summary = {f'clean_{sp}': clean_ref[sp] for sp in ALL_SPACES}
    highest_sigma = str(max(s for s in levels if s > 0))
    hs_data = results['per_sigma'].get(highest_sigma, {})
    summary['summary_level'] = float(highest_sigma)

    for cond in _lnl_keys:
        best_name, best_score = _best_method(hs_data.get(cond, {}),
                                             methods_to_test, lower_is_better)
        summary[f'best_{cond}'] = best_name
        summary[f'best_{cond}_score'] = (float(best_score) if best_name
                                         else None)

    # Which representation each base space was, counted over passes: the
    # ranking is redone on every fold and label set.
    _names_by_space = {sp: Counter() for sp in ALL_SPACES}
    for rec in results['configurations']:
        for sp, nm in ((rec.get('configuration') or {})
                       .get('configuration_names') or {}).items():
            _names_by_space.setdefault(sp, Counter())[str(nm)] += 1
    summary['configuration_names_by_pass_count'] = {
        sp: dict(c) for sp, c in _names_by_space.items() if c}

    # Methods where a hybrid beats a single representation, each hybrid
    # against each single representation, over the passes both arms scored.
    for hyb in HYBRID_FIT_SPACES:
        out = {}
        for b in BASE_FIT_SPACES:
            improves, harms = [], []
            for m in methods_to_test:
                _b_arm = hs_data.get(f'{b}_lnl', {}).get(m, {})
                _h_arm = hs_data.get(f'{hyb}_lnl', {}).get(m, {})
                base_s, hyb_s = _shared_fold_means(_b_arm, _h_arm)
                if np.isnan(base_s) or np.isnan(hyb_s):
                    if _b_arm.get('n_folds') and _h_arm.get('n_folds'):
                        print(f"    {m}: {b} and {hyb} share no (fold, draw) "
                              f"pass at level {highest_sigma}; left out")
                    continue
                if _better(hyb_s, base_s, lower_is_better):
                    improves.append(m)
                elif _better(base_s, hyb_s, lower_is_better):
                    harms.append(m)
            out[b] = {'hybrid_improves': improves, 'hybrid_harms': harms,
                      'n_hybrid_improves': len(improves),
                      'n_hybrid_harms': len(harms)}
        summary['hybrid_vs_base' if hyb == 'hybrid'
                else f'{hyb}_vs_base'] = out

    # Methods where the weighted copy beats hybrid 1 (does weighting help?)
    # RESTORED 2026-10-07 (Adelaide), with the weighted copy.
    whybrid_improves, whybrid_harms = [], []
    for m in methods_to_test:
        _hyb_arm = hs_data.get('hybrid_lnl', {}).get(m, {})
        _whyb_arm = hs_data.get(f'{WEIGHTED_SPACE}_lnl', {}).get(m, {})
        hyb_s, whyb_s = _shared_fold_means(_hyb_arm, _whyb_arm)
        if np.isnan(hyb_s) or np.isnan(whyb_s):
            if _hyb_arm.get('n_folds') and _whyb_arm.get('n_folds'):
                print(f"    {m}: hybrid and weighted hybrid share no "
                      f"(fold, draw) pass at level {highest_sigma}; left out")
            continue
        if _better(whyb_s, hyb_s, lower_is_better):
            whybrid_improves.append(m)
        elif _better(hyb_s, whyb_s, lower_is_better):
            whybrid_harms.append(m)
    summary['weighted_improves_over_hybrid'] = whybrid_improves
    summary['weighted_harms_vs_hybrid'] = whybrid_harms
    summary['n_weighted_improves'] = len(whybrid_improves)
    summary['n_weighted_harms'] = len(whybrid_harms)

    # Did the top-3 ranking change? None when an arm never populated, so an
    # empty list is never read as a changed ranking.
    top3 = {sp: _top3(hs_data.get(f'{sp}_lnl', {}), methods_to_test,
                      lower_is_better) for sp in ALL_SPACES}
    for sp in ALL_SPACES:
        summary[f'top3_{sp}'] = top3[sp]

    def _changed(a, b):
        return None if (not top3[a] or not top3[b]) else top3[a] != top3[b]
    summary['ranking_changed'] = {b: _changed(b, 'hybrid')
                                  for b in BASE_FIT_SPACES}
    summary['ranking_changed_hybrid__general2'] = {
        b: _changed(b, 'hybrid__general2') for b in BASE_FIT_SPACES}
    summary['ranking_changed_weighted'] = _changed('hybrid', WEIGHTED_SPACE)

    results['summary'] = convert_numpy(summary)
    results['similarity'] = similarity_record(
        sim_settings, sim_sink, swept_methods=['distance_selection'],
        spaces_unavailable=sim_ctx['unavailable'],
        neighbour_spaces_by_fold=neighbour_space_record,
        checkpoint_file=ckpt.path, passes_read_back=ckpt.n_resumed)
    # The per-dataset screen of every (model input, fold, draw, level): each
    # (setting, k, scheme) scored with no model fit, and which 3 were kept.
    results['similarity_screen'] = convert_numpy(sim_ctx.get('screen') or [])
    # {level: {arm: {setting: {'k=..|scheme=..': mean DNN score over
    # (fold, draw)}}}}, in the dataset's metric.
    results['distance_selection_by_similarity'] = {
        _lv: {_a: {_sn: {_k: {'mean': float(np.mean(_v)), 'n': len(_v)}
                         for _k, _v in _byk.items()}
                   for _sn, _byk in _bys.items()}
              for _a, _bys in _bya.items()}
        for _lv, _bya in _ds_by_sim.items()}

    print(f"\n  === SUMMARY (level {highest_sigma}) ===")
    print("  Level 0: " + ", ".join(f"{sp}={clean_ref[sp]:.4f}"
                                    for sp in ALL_SPACES))
    for sp in ALL_SPACES:
        print(f"  Best {sp}_lnl: {summary.get(f'best_{sp}_lnl')}")
    for b, v in summary['hybrid_vs_base'].items():
        print(f"  Hybrid 1 improves over {b}: {v['n_hybrid_improves']} of "
              f"{len(methods_to_test)} methods")
    print(f"  Weighted improves over hybrid 1: {len(whybrid_improves)} of "
          f"{len(methods_to_test)} methods")
    print(f"  Ranking changed (hybrid 1 vs base): {summary['ranking_changed']}")
    print(f"  Ranking changed (weighted): {summary['ranking_changed_weighted']}")
    for sp in ALL_SPACES:
        print(f"  Top-3 {sp}: {top3[sp]}")

    return results


def _better(a, b, lower_is_better):
    return (a < b) if lower_is_better else (a > b)


def _best_method(cond_data, methods, lower_is_better):
    best_name = None
    best_score = float('inf') if lower_is_better else float('-inf')
    for m in methods:
        s = (cond_data or {}).get(m, {}).get('mean', float('nan'))
        if s is None or np.isnan(s):
            continue
        if _better(s, best_score, lower_is_better):
            best_score, best_name = s, m
    return best_name, best_score


def _top3(cond_data, methods, lower_is_better):
    scores = {}
    for m in methods:
        s = (cond_data or {}).get(m, {}).get('mean', float('nan'))
        if s is not None and not np.isnan(s):
            scores[m] = s
    return sorted(scores, key=scores.get, reverse=not lower_is_better)[:3]


def _paired(a, b):
    """Pairs (plain score, method score) over the passes both scored."""
    ka = {(f, d): s for f, d, s in zip(a.get('folds', []),
                                        a.get('fold_draws', []),
                                        a.get('fold_scores', []))}
    kb = {(f, d): s for f, d, s in zip(b.get('folds', []),
                                        b.get('fold_draws', []),
                                        b.get('fold_scores', []))}
    shared = sorted(set(ka) & set(kb))
    return ([(k, ka[k], kb[k]) for k in shared], len(set(ka) ^ set(kb)))


def _shared_fold_means(a, b):
    """Each arm's mean over the (fold, draw) passes both arms scored, or
    (nan, nan) when they share none or either lacks its per-pass lists."""
    def _passes(d):
        f, g, sc = d.get('folds'), d.get('fold_draws'), d.get('fold_scores')
        if f is None or g is None or sc is None:
            return None
        if not (len(f) == len(g) == len(sc)):
            return None
        return [((int(fi), int(gi)), float(x)) for fi, gi, x in zip(f, g, sc)]

    pa, pb = _passes(a), _passes(b)
    if pa is None or pb is None:
        return float('nan'), float('nan')
    shared = {k for k, _ in pa} & {k for k, _ in pb}
    ka = [x for k, x in pa if k in shared]
    kb = [x for k, x in pb if k in shared]
    if not ka or not kb:
        return float('nan'), float('nan')
    return float(np.mean(ka)), float(np.mean(kb))


if __name__ == '__main__':
    parser = create_experiment_parser('N09: LNL Method Survey')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n09_lnl_survey', args.dataset)
