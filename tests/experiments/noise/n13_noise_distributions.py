#!/usr/bin/env python3
"""N13: Synthetic-Noise SENSITIVITY Study.

Paper: Noise

This is a SENSITIVITY study, not an absolute-robustness benchmark. It measures
how method rankings and hybrid advantage SHIFT when the injected synthetic-noise
DISTRIBUTION is changed, holding everything else fixed. All noise here is
artificially injected onto clean labels; the numbers describe sensitivity to the
choice of synthetic noise model, NOT a claim about robustness to real-world
label noise. Read every cell as "how much does the result move under this
injected distribution", never as "method X is robust to noise".

ALIGNED 2026-10-05 (chat 5, HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.4):
  - Kinds of damage: the decided six on a regression dataset (gaussian,
    outlier_p05, grouped_shifted, censoring, shift_only_p10,
    sparse_dprime_up_p10; KNN_SEARCH_SPACE_RECORD "The noise grid", Adelaide
    2026-10-05), at the decided levels for the dataset's group
    (noise_helpers.decided_conditions / decided_levels / inject_condition).
    Until 2026-10-05 n13 swept gaussian, student_t at nu 10, 5 and 3, laplace,
    outlier at 1%, 5% and 10%, grouped_wider, grouped_shifted and censoring;
    the decision cut eight of those. A classification dataset keeps its six
    flip kinds and the flip-rate levels: no decision cut them (f37 measured
    regression only).
  - Level 0 is the labels as delivered. It is run once per fold, not once per
    kind and draw, and reported under every kind. Every non-zero level is
    drawn 3 times.
  - Folds: noise_helpers.noise_outer_folds, the folds the per-fold searches
    are filed under; any count, including 1. Every summary over folds goes
    through fold_spread. No significance test anywhere.
  - Models: the dataset's two decided models (DECISIONS.md Step 1). Per fold,
    each model's 2 best single representations and 2 best hybrids, all ranked
    by searches on that fold's training rows (noise_helpers.
    fold_configurations), so 2 models x 4 = 8 fitted configurations. Nothing
    is read from a whole-dataset search.
  - The distance hybrid ('hybrid__distance1', the model's top hybrid of the
    distance search) and the similarity hybrid ('hybrid_similarity', the top
    hybrid of the similarity-objective search) are spaces noise is DETECTED
    in (Track B) and neighbour spaces of the neighbour sweep. No model is
    fitted on them.

Two evaluation tracks:
  Track A: ALL 12 LNL methods from the registry, each training its own neural
           network, in each of dnn's 4 configurations. Read against the plain
           network trained on the same labels in the same configuration.
           ⛔ Only on datasets whose two decided models include dnn; see the
           OPEN note at TRACK_A_MODEL.
  Track B: noise detectors (local_variance_ratio, gaussian_process, ngboost
           when installed) score each training molecule in every detection
           space of a model; the scores become sample weights for that model
           fitted on each of its 2 single representations. SAMPLE-WEIGHT
           remediation only: a model whose fit() ignores sample_weight is
           left out, never fitted unweighted and mislabelled as remediated.
           local_variance_ratio reads neighbours, so it runs only under the
           neighbour sweep, never plainly in a detection space (2026-10-05).

The neighbour sweep (KNN_SEARCH_SPACE_RECORD): Track B's neighbour detector
and Track A's distance_selection read their neighbours from every setting of
this dataset's group (n07.sweep_settings), k in 5, 10, 20, both counting
schemes, scaffold-mates kept in the lists. Grading a detector fits no model,
so it covers every (setting, k). Refits (the D-versus-D' rows and
distance_selection's network) go only to the screen's top 3 (setting, k,
scheme) per pass (n07.screen_settings).

Usage:
    python n13_noise_distributions.py --dataset logd
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, get_domain_metric,
    get_model_factories, input_fingerprint,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    detection_scores, decided_conditions, decided_levels, inject_condition,
    level_unit, dataset_knn_group, noise_outer_folds, fold_spread,
    fold_configurations, fold_model_inputs, configuration_record,
    CONFIGURATION_SPACES, DECIDED_REGRESSION_CONDITIONS,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.lnl_methods import (
    get_lnl_method, DistanceBasedSelection, _standard_dnn_train_predict,
    _standard_dnn_train_predict_clf, LNL_REGISTRY, REGRESSION_ONLY_LNL)
from kirby.utils.noise_estimation import get_noise_estimator, weights_from_scores
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, sweep_schemes, screen_settings, screen_top,
    usable_ks, fold_similarity_spaces, fold_neighbours, RowSink, sweep_context,
    dprime_rows, row_identity, detector_scores_with, cleaned_from_weights,
    similarity_record, FoldCheckpoint,
)
from experiments.noise import noise_cache as _nc
from experiments.noise.noise_cache import NEIGHBOUR_DETECTORS


# What this file writes since chat 5 (2026-10-05). It is in the checkpoint's
# fingerprint and in the rows file's name, so a checkpoint line or a row
# written by the whole-dataset design is never read back.
RESULT_LAYOUT = 'n13_fold_aligned_2026-10-05b'  # b: neighbour fixes after review

# Draws per non-zero level. Level 0 is drawn once.
NOISE_SEEDS = [0, 1, 2]

# Test the FULL LNL method registry (all 12), NOT a pre-selected shortlist.
# n09's question is whether hybrids RERANK the methods, so pre-filtering n13
# to a pre-hybrid shortlist would bias the cross-distribution result.
# Regression-only methods are gated out on classification datasets below.
LNL_METHODS = list(LNL_REGISTRY.keys())
ESTIMATORS = ['local_variance_ratio', 'gaussian_process']
# ngboost is optional
try:
    get_noise_estimator('ngboost')
    ESTIMATORS.append('ngboost')
except Exception:
    pass

LNL_EPOCHS = 50  # reduced for speed

# Every Track A method trains a neural network, so Track A runs in dnn's own
# 4 configurations (its 2 best single representations and 2 best hybrids from
# the per-fold search).
#
# OPEN (chat 5): 16 of the 43 decided datasets do not have dnn among their two
# models, so the per-fold search ranks nothing for dnn there and Track A does
# not run on them (the result file says so under 'track_a_skipped'). Until
# 2026-10-05 n13 read dnn's recipe from the whole-dataset search on every
# dataset. Two ways to restore it, both Adelaide's call: (a) the per-fold dnn
# search she approved for n09 (decided_grid.fast_model_search_kwargs(ds,
# 'dnn') + _shared.search_on_fold(..., fixed_grid=...), already written as
# n09_lnl_survey._dnn_fold_entry); (b) train the LNL networks in the two
# decided models' configurations, whose features were chosen for another
# model (the owner rejected that mismatch on 2026-09-28). 2026-10-05.
TRACK_A_MODEL = 'dnn'

# The spaces noise is detected in for Track B, per model: its 4 fitted
# configurations, then the two neighbour-only hybrids (handoff 2.3 item 1).
NEIGHBOUR_ONLY_SPACES = ('hybrid__distance1', 'hybrid_similarity')
DETECTION_SPACES = tuple(CONFIGURATION_SPACES) + NEIGHBOUR_ONLY_SPACES
HYBRID_SPACES_REBUILT = ('hybrid', 'hybrid__general2') + NEIGHBOUR_ONLY_SPACES
BASE_SPACES = ('base_1', 'base_2')

# Models whose fit() does NOT accept a `sample_weight` argument are found on
# the first weighted fit (TypeError) and left out of Track B. The seed is
# empty: dnn's fit gained sample_weight on 2026-09-17.
UNWEIGHTED_TRACK_B_SEED = frozenset()
UNWEIGHTED_TRACK_B_MODELS = set(UNWEIGHTED_TRACK_B_SEED)

# The kinds this file swept before the 2026-10-05 noise-grid decision that are
# no longer swept, recorded on the result file.
CUT_BY_NOISE_GRID_DECISION = ['student_t_nu10', 'student_t_nu5',
                              'student_t_nu3', 'laplace', 'outlier_p01',
                              'outlier_p10', 'grouped_wider']


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


def _plain_dnn(X_tr, y_tr, X_va, *, seed, is_clf, classes_):
    """The plain network every Track A method is read against.

    2026-10-05: on a classification dataset this was the regression network
    (_standard_dnn_train_predict z-scores the class labels and regresses on
    them). The classifier twin returns what the metric expects.
    """
    if is_clf:
        return _standard_dnn_train_predict_clf(
            X_tr, y_tr, X_va, epochs=LNL_EPOCHS, patience=10,
            random_state=seed, classes_=classes_)
    return _standard_dnn_train_predict(
        X_tr, y_tr, X_va, epochs=LNL_EPOCHS, patience=10, random_state=seed)


def _neighbour_detector_scores(ds, est_name, X_tr, y_tr, lists, k, scheme, *,
                               is_clf):
    """One neighbour detector's suspicion scores from a swept neighbour list,
    with each neighbour counted by the swept scheme (kNN record axis 5).

    2026-10-05 (chat 5). noise_cache.cached_detector_scores, which
    n07.detector_scores_with calls, takes the neighbours' plain mean and spread
    and never reads the scheme, so 'rank' and 'inverse_square' gave the same
    scores. local_variance_ratio (the one neighbour detector n13 runs) is
    computed here with the arithmetic of LocalVarianceRatioEstimator
    (noise_estimation.py), the neighbours' mean, spread, dominant class and
    purity weighted by similarity_space.neighbour_weights. Same floors as the
    estimator. Classification uses the estimator's classification branch;
    until 2026-10-05 n13 always ran the regression branch on class labels.
    Any other neighbour detector goes through detector_scores_with with is_clf,
    unweighted. None when the scores cannot be computed.
    """
    if est_name != 'local_variance_ratio':
        return detector_scores_with(ds, est_name, X_tr, y_tr, lists, k,
                                    is_clf=is_clf)
    from experiments.noise import similarity_space as ss
    try:
        y = np.asarray(y_tr).ravel()
        k_use = int(min(int(k), len(y) - 1))
        idx, dist = ss.neighbours_at(lists, k_use, which='train')
        idx = np.asarray(idx)
        w = ss.neighbour_weights(dist, scheme)
        w_sum = w.sum(axis=1)
        nb = y[idx]
        if is_clf:
            disagreement = (w * (nb != y[:, None])).sum(axis=1) / w_sum
            dominant = ss.weighted_neighbour_label(y, idx, dist, scheme,
                                                   is_clf=True)
            purity = (w * (nb == dominant[:, None])).sum(axis=1) / w_sum
            floor = max(1.0 / idx.shape[1], 1e-3)
            return disagreement / np.maximum(1.0 - purity, floor)
        yf = y.astype(float)
        mean = ss.weighted_neighbour_label(yf, idx, dist, scheme)
        sd = np.sqrt((w * (nb.astype(float) - mean[:, None]) ** 2).sum(axis=1)
                     / w_sum)
        floor = max(0.01 * float(np.std(yf)), 1e-3)
        return np.abs(yf - mean) / np.maximum(sd, floor)
    except Exception as exc:
        print(f"      {est_name} k={k} {scheme} failed: "
              f"{type(exc).__name__}: {exc}")
        return None


class _SchemeDistanceSelection(DistanceBasedSelection):
    """distance_selection with each neighbour counted by the swept scheme.

    2026-10-05 (chat 5). The same subclass as n09_lnl_survey's: the swept lists
    carry the scheme ('scheme' key), but DistanceBasedSelection
    (lnl_methods.py) takes the neighbours' plain mean and spread and never
    reads it. Here the mean, spread and class disagreement are weighted by
    similarity_space.neighbour_weights. Everything after the suspicion score
    (85th-percentile threshold, keep at least 10, refit) is the parent's.
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


def _sweep_pass(ctx, *, model_name, model_input, X_tr, X_va, y_tr_noisy,
                y_tr_clean, y_val, train_idx, val_idx, groups_tr, fold, draw,
                level, condition, recipes, base_reps, make_dprimes,
                grade=None):
    """One (model input, fold, draw) pass of the neighbour sweep.

    The same steps as n07.similarity_sweep, with one difference: the D-versus-D'
    rows are fitted with this pass's own model only. similarity_sweep adds rf
    and svm to every row (n07.row_models), which are not the dataset's decided
    models (R1). 2026-10-05.

    grade(setting, lists, k): scoring only, no model fit, called for every
        (setting, k) the lists hold.
    make_dprimes(setting, lists, k, scheme): called only for the screen's top
        (setting, k, scheme); yields (method, method_setting, cleaned).
    No group is excluded from the lists (axis 6: scaffold-mates stay in).
    """
    sink = ctx['sink']
    ds = ctx['dataset_name']
    settings = ctx['settings']
    spaces_tr, spaces_va, unavail = fold_similarity_spaces(
        ds, settings, model_input_tr=X_tr, model_input_va=X_va,
        reps=ctx['reps'], train_idx=train_idx, val_idx=val_idx,
        y_train=y_tr_noisy, groups_train=groups_tr, base_reps=base_reps,
        recipes=recipes, augmentations=ctx['augmentations'],
        model_name=model_name, model_factory=ctx['factories'].get(model_name),
        metric=ctx['metric'], task_type=ctx['task_type'])
    smi = ctx.get('smiles')
    smi_tr = None if smi is None else [smi[i] for i in np.asarray(train_idx)]
    smi_va = None if smi is None else [smi[i] for i in np.asarray(val_idx)]
    nbrs = fold_neighbours(ds, settings, spaces_tr, spaces_va, y_tr_noisy,
                           smiles_train=smi_tr, smiles_val=smi_va,
                           is_clf=ctx['is_clf'])
    for s in settings:
        L = nbrs.get(s.name)
        if L is None or not L.get('available'):
            ctx['unavailable'].setdefault(
                f'{model_input}|{s.name}',
                unavail.get(s.space) or (L or {}).get('reason'))
    if grade is not None:
        for s in settings:
            L = nbrs.get(s.name)
            for k in usable_ks(L):
                grade(s, L, k)
    kept, table = screen_settings(settings, nbrs, y_tr_noisy,
                                  metric=ctx['metric'], is_clf=ctx['is_clf'],
                                  y_val=y_val)
    ctx['screen'].append({
        'model_input': model_input, 'fold': int(fold), 'draw': int(draw),
        'level': float(level), 'condition': condition,
        'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
        'table': table})
    base_id = dict(dataset_name=ds, experiment=ctx['experiment'],
                   model_input=model_input, noise_condition=condition,
                   level=level, draw=draw, fold=fold)
    for s, k, scheme in kept:
        L = dict(nbrs[s.name], scheme=scheme)
        for method, msetting, cleaned in make_dprimes(s, L, k, scheme):
            if sink is None or not sink.enabled:
                continue
            ms = f'k={k}|scheme={scheme}'
            ms = ms if not msetting else f'{ms}|{msetting}'
            dprime_rows(sink, dataset_name=ds, cleaned=cleaned, X_tr=X_tr,
                        y_D=y_tr_noisy, X_va=X_va, y_va=y_val,
                        metric=ctx['metric'], task_type=ctx['task_type'],
                        y_clean=y_tr_clean, models=(model_name,),
                        is_clf=ctx['is_clf'],
                        identity=row_identity(
                            method=method, method_setting=ms,
                            similarity_setting=s.name, **base_id))


def run(data, config, output_dir, dataset_name, **kwargs):
    """N13: Alternative noise distributions, per outer fold."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N13: ALTERNATIVE NOISE DISTRIBUTIONS")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    classes_ = np.unique(labels) if is_clf else None
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    predict_mode = _nc.predict_mode_for(metric_name)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    knn_grp = dataset_knn_group(data, dataset_name)

    # R1: the dataset's two decided models, nothing else. knn runs only
    # inside the distance search. Replaces config['models'] /
    # DOWNSTREAM_MODELS_NOISE (five models) and the dnn this file always added.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    model_factories = get_model_factories(models, task_type=task_type,
                                          n_samples=len(labels))
    track_a_on = TRACK_A_MODEL in models
    track_a_skipped = (None if track_a_on else
                       f'{TRACK_A_MODEL} is not one of this dataset\'s decided '
                       f'models {models}, so no per-fold search ranks '
                       f'configurations for it (OPEN, see TRACK_A_MODEL)')
    print(f"  Models (decided): {models}")
    print(f"  Dataset: {dataset_name} ({len(labels)} rows, {n_molecules} "
          f"molecules, group {knn_grp})")
    print(f"  Metric: {metric_name} (lower is better: {lower_is_better})")
    if track_a_skipped:
        print(f"  Track A OFF: {track_a_skipped}")

    lnl_methods = [m for m in LNL_METHODS
                   if not (is_clf and m in REGRESSION_ONLY_LNL)]
    estimators = [e for e in ESTIMATORS if not (is_clf and e == 'gaussian_process')]
    sim_estimators = [e for e in estimators if e in NEIGHBOUR_DETECTORS]

    # The decided kinds of damage. A caller (run_all under a named condition)
    # may narrow them; a kind outside the decided set is refused, not run.
    noise_types = list(decided_conditions(data))
    noise_types_refused = {}
    _asked = kwargs.get('noise_types')
    if _asked:
        for c in _asked:
            if c not in noise_types:
                noise_types_refused[c] = (
                    'not one of the decided kinds for this task '
                    '(KNN_SEARCH_SPACE_RECORD, noise grid, 2026-10-05)')
        noise_types = [c for c in noise_types if c in _asked]
        for c, why in noise_types_refused.items():
            print(f"  {c}: not run, {why}")
    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    noise_types_dropped = {}
    if noise_groups is None:
        for c in [c for c in noise_types if c in GROUPED_STRATEGIES]:
            noise_types_dropped[c] = (
                'this dataset has no SMILES to build scaffold groups from; '
                'grouped noise over invented groups is uniform noise wearing '
                'a grouped name')
        noise_types = [c for c in noise_types if c not in noise_types_dropped]
        for c, why in noise_types_dropped.items():
            print(f"  {c}: DROPPED, {why}")
    else:
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_group_share:.1%} of molecules")
    levels_by_condition = {c: [float(x) for x in
                               decided_levels(data, dataset_name, c)]
                           for c in noise_types}
    for c in noise_types:
        print(f"  {c}: levels {levels_by_condition[c]} "
              f"({level_unit(c, is_clf)})")
    if not noise_types:
        return {'skipped': True, 'reason': 'no kind of damage left to run',
                'noise_types_refused': noise_types_refused,
                'noise_types_dropped': noise_types_dropped,
                'metric': metric_name, 'lower_is_better': lower_is_better}

    folds, n_folds = noise_outer_folds(data, config)
    print(f"  Folds: {[f for f, _, _ in folds]} of {n_folds}")

    # The neighbour settings this dataset's group sweeps.
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)
    sink = RowSink(output_dir, 'n13_noise_distributions', dataset_name,
                   RESULT_LAYOUT)
    sweep_ctx = sweep_context(
        experiment='n13_noise_distributions', dataset_name=dataset_name,
        sink=sink, settings=sim_settings, metric=metric, task_type=task_type,
        condition=None, level=None, data=data, factories=model_factories)
    print(f"  Neighbour sweep: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x schemes {list(sweep_schemes())}, top "
          f"{screen_top()} refitted; rows -> {sink.path}")

    ckpt = FoldCheckpoint(
        output_dir, 'n13_noise_distributions', dataset_name, tag='units',
        fingerprint={
            'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
            'n_folds': int(n_folds), 'models': models,
            'noise_seeds': list(NOISE_SEEDS),
            'levels': levels_by_condition, 'lnl_methods': list(lnl_methods),
            'estimators': list(estimators), 'lnl_epochs': LNL_EPOCHS,
            'settings': [s.name for s in sim_settings],
            'screen_top': screen_top()})

    global UNWEIGHTED_TRACK_B_MODELS
    UNWEIGHTED_TRACK_B_MODELS = set(UNWEIGHTED_TRACK_B_SEED)

    fold_records = {}
    units = []   # [{'fold', 'condition', 'level', 'draw', 'value'}]

    for fold_i, train_idx, val_idx in folds:
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue
        y_tr_clean = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]

        # The single-representation configurations choose no features from
        # the labels, so one pair per model serves every level and draw.
        base_inputs = {}
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            if m not in model_factories:
                frec['models'][m] = {'available': False,
                                     'why': 'no model factory for this name'}
                continue
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr_clean,
                factory=model_factories[m], metric=metric, spaces=BASE_SPACES)
            base_inputs[m] = (entry, ins, unav)
            rec = configuration_record(entry, names)
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        fold_records[str(fold_i)] = frec

        def _unit(condition, level, draw, _tr=train_idx, _va=val_idx,
                  _ytr=y_tr_clean, _yva=y_val, _g=g_tr, _bi=base_inputs,
                  _fold=fold_i):
            # Level 0 is the labels as delivered whatever the kind; condition
            # None marks the one shared level-0 pass.
            y_noisy, cmask, prov = inject_condition(
                data, dataset_name, condition or noise_types[0], level, draw,
                train_idx=_tr, groups=noise_groups)
            cmask = np.asarray(cmask, dtype=bool)
            graded = condition is not None and float(level) > 0.0
            seed_combined = 42 + 1000 * int(_fold) + int(draw)
            val = {'provenance': prov, 'n_corrupted': int(cmask.sum()),
                   'config_scores': {}, 'n_features': {},
                   'track_a': {}, 'track_a_plain': {}, 'track_b': {},
                   'detection': {}, 'sim_detection': {}, 'track_a_sim': {},
                   'unavailable': {}}
            sweep_ctx['screen'] = []
            noise_score_memo = {}

            for m, (entry, ins_b, unav_b) in _bi.items():
                factory = model_factories[m]
                # The hybrids choose their features from the labels the run
                # holds, so they are rebuilt from the damaged labels.
                ins_h, unav_h, _n = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=_tr, val_idx=_va,
                    y_train=y_noisy, factory=factory, metric=metric,
                    spaces=HYBRID_SPACES_REBUILT)
                spaces = dict(ins_b)
                spaces.update(ins_h)
                for sp in DETECTION_SPACES:
                    if sp not in spaces:
                        val['unavailable'].setdefault(m, {})[sp] = (
                            (unav_b if sp in BASE_SPACES else unav_h).get(sp)
                            or 'not built')

                # The 8 configurations (R5): each model plainly fitted on its 4.
                for sp in CONFIGURATION_SPACES:
                    if sp not in spaces:
                        continue
                    X_tr, X_va = spaces[sp]
                    val['n_features'].setdefault(m, {})[sp] = int(
                        np.shape(X_tr)[1])
                    try:
                        val['config_scores'].setdefault(m, {})[sp] = float(
                            metric_fn(_yva, _nc.cached_fit_predict(
                                dataset_name, m, factory, X_tr, y_noisy, X_va,
                                predict_mode=predict_mode)))
                    except Exception as exc:
                        val['unavailable'].setdefault(m, {})[sp] = (
                            f'fit raised {type(exc).__name__}: {exc}')

                # ── Track B: detect in every detection space, weight the
                # model's fit on each of its 2 single representations. ──
                for sp in DETECTION_SPACES:
                    if sp not in spaces:
                        continue
                    # A space two models share (the same single
                    # representation, the similarity hybrid) is scored once.
                    if sp in BASE_SPACES and entry and len(entry.get('base') or []) > BASE_SPACES.index(sp):
                        memo_id = f"rep:{entry['base'][BASE_SPACES.index(sp)]}"
                    elif sp == 'hybrid_similarity':
                        memo_id = 'hybrid_similarity'
                    else:
                        memo_id = f'{m}:{sp}'
                    dkey = f'{m}:{sp}'
                    X_space = spaces[sp][0]
                    # A neighbour detector (local_variance_ratio) is not run
                    # here. Called plainly it finds its own 10 neighbours in
                    # this space with the process-wide distance and no
                    # counting scheme: on the 4 configurations that is the
                    # model_input setting the kNN record cut (degenerate), and
                    # on the 2 neighbour-only hybrids it is one fixed k and a
                    # distance outside the group's list. Its weighted refits
                    # would also go out for every space, not only the screen's
                    # top 3. It runs only under the neighbour sweep below,
                    # where those hybrids are among the settings. 2026-10-05.
                    for est_name in estimators:
                        if est_name in NEIGHBOUR_DETECTORS:
                            continue
                        mk = (memo_id, est_name)
                        if mk not in noise_score_memo:
                            try:
                                noise_score_memo[mk] = get_noise_estimator(
                                    est_name).estimate(X_space, y_noisy)
                            except Exception as exc:
                                noise_score_memo[mk] = None
                                val['unavailable'].setdefault(m, {})[
                                    f'{sp}|{est_name}'] = (
                                    f'estimator raised {type(exc).__name__}: '
                                    f'{exc}')
                        noise_scores = noise_score_memo[mk]
                        if noise_scores is None:
                            continue
                        if graded:
                            _det = detection_scores(cmask, noise_scores,
                                                    condition)
                            if _det is not None:
                                (val['detection'].setdefault(dkey, {})
                                 [est_name]) = _det
                        weights = weights_from_scores(
                            noise_scores, strength=0.1, method='exponential')
                        if m in UNWEIGHTED_TRACK_B_MODELS:
                            continue
                        for bsp in BASE_SPACES:
                            if bsp not in spaces:
                                continue
                            Xb_tr, Xb_va = spaces[bsp]
                            try:
                                pred = _nc.cached_fit_predict(
                                    dataset_name, m, factory, Xb_tr, y_noisy,
                                    Xb_va, sample_weight=weights,
                                    predict_mode=predict_mode)
                            except TypeError:
                                # fit() takes no sample_weight: not
                                # remediable this way. Recorded, never fitted
                                # unweighted under a remediated name.
                                UNWEIGHTED_TRACK_B_MODELS.add(m)
                                break
                            except Exception:
                                continue
                            (val['track_b'].setdefault(dkey, {})
                             .setdefault(est_name, {}))[f'{m}@{bsp}'] = float(
                                metric_fn(_yva, pred))

                # ── Track B's neighbour detector under the neighbour sweep,
                # one pass per single-representation configuration. ──
                if sim_estimators and m not in UNWEIGHTED_TRACK_B_MODELS:
                    for bsp in BASE_SPACES:
                        if bsp not in spaces:
                            continue
                        Xb_tr, Xb_va = spaces[bsp]

                        # Grading fits no model, so it covers every
                        # (setting, k, scheme). Keyed 'k=<k>|<scheme>'.
                        # is_clf is the dataset's (until 2026-10-05 it was
                        # always False, the regression score on class labels).
                        def _grade(setting, lists, k, _x=Xb_tr):
                            if not graded:
                                return
                            for est_name in sim_estimators:
                                for sch in sweep_schemes():
                                    sc = _neighbour_detector_scores(
                                        dataset_name, est_name, _x, y_noisy,
                                        lists, k, sch, is_clf=is_clf)
                                    if sc is None:
                                        continue
                                    _det = detection_scores(cmask, sc,
                                                            condition)
                                    if _det is not None:
                                        (val['sim_detection']
                                         .setdefault(est_name, {})
                                         .setdefault(setting.name, {})
                                         .setdefault(f'k={k}|{sch}', [])
                                         .append(_det))

                        def _dprimes(setting, lists, k, scheme, _x=Xb_tr):
                            for est_name in sim_estimators:
                                sc = _neighbour_detector_scores(
                                    dataset_name, est_name, _x, y_noisy,
                                    lists, k, scheme, is_clf=is_clf)
                                if sc is None:
                                    yield est_name, 'exponential_0.1', None
                                    continue
                                w = weights_from_scores(sc, strength=0.1,
                                                        method='exponential')
                                yield (est_name, 'exponential_0.1',
                                       cleaned_from_weights(_x, y_noisy, w))

                        _sweep_pass(
                            sweep_ctx, model_name=m,
                            model_input=f'{m}@{bsp}', X_tr=Xb_tr, X_va=Xb_va,
                            y_tr_noisy=y_noisy, y_tr_clean=_ytr, y_val=_yva,
                            train_idx=_tr, val_idx=_va, groups_tr=_g,
                            fold=_fold, draw=draw, level=float(level),
                            condition=condition,
                            recipes=(entry or {}).get('recipes') or {},
                            base_reps=(entry or {}).get('base') or [],
                            make_dprimes=_dprimes, grade=_grade)

                # ── Track A: the LNL methods, in dnn's 4 configurations. ──
                if not (track_a_on and m == TRACK_A_MODEL):
                    continue
                for sp in CONFIGURATION_SPACES:
                    if sp not in spaces:
                        continue
                    X_tr, X_va = spaces[sp]
                    try:
                        val['track_a_plain'][sp] = float(metric_fn(
                            _yva, _plain_dnn(X_tr, y_noisy, X_va,
                                             seed=seed_combined, is_clf=is_clf,
                                             classes_=classes_)))
                    except Exception:
                        val['track_a_plain'][sp] = None
                    for method_name in lnl_methods:
                        # distance_selection called plainly finds its own 10
                        # neighbours in the features being trained on (the
                        # model_input setting the kNN record cut, degenerate)
                        # and trains a network on that one setting. It runs
                        # only under the sweep below, on the screen's top 3
                        # (setting, k, scheme). 2026-10-05.
                        if method_name == 'distance_selection':
                            continue
                        # OPEN (chat 5): mentornet also takes its 10 nearest
                        # neighbours in the features being trained on (the
                        # model_input setting), but only their mean distance,
                        # as one input to its weighting network, never their
                        # labels. It cannot take a swept list: it carves its
                        # checkpoint-selection rows out of the training rows
                        # first, so a list over the training fold no longer
                        # lines up (n09 says the same). Options: keep it as
                        # is (current), drop it from Track A, or change
                        # MentorNet to carve after reading the list. Adelaide's.
                        # 2026-10-05.
                        try:
                            mth = get_lnl_method(
                                method_name, epochs=LNL_EPOCHS, patience=10,
                                random_state=seed_combined, is_clf=is_clf,
                                classes_=classes_)
                            res = mth.train_and_predict(X_tr, y_noisy, X_va,
                                                        _yva)
                            val['track_a'].setdefault(sp, {})[method_name] = \
                                float(metric_fn(_yva, res['y_pred']))
                        except Exception:
                            pass

                    # distance_selection finds neighbours and drops training
                    # molecules, so it is swept. Training its network is a
                    # refit, so only the screen's top 3 are run.
                    # Its score is recorded whether or not a rows file is
                    # written (_sweep_pass writes rows only when the sink is
                    # on), so distance_selection is never left out.
                    if 'distance_selection' not in lnl_methods:
                        continue

                    def _ds_dprimes(setting, lists, k, scheme, _Xtr=X_tr,
                                    _Xva=X_va, _sp=sp):
                        try:
                            # Counts each neighbour by the swept scheme.
                            _mth = _SchemeDistanceSelection(
                                k=k, precomputed_neighbors=lists,
                                epochs=LNL_EPOCHS, patience=10,
                                random_state=seed_combined, is_clf=is_clf,
                                classes_=classes_)
                            _res = _mth.train_and_predict(_Xtr, y_noisy,
                                                          _Xva, _yva)
                        except Exception as _e:
                            print(f"      distance_selection {setting.name} "
                                  f"k={k} {scheme} failed: {_e}")
                            yield ('distance_selection', 'threshold_pct=85',
                                   None)
                            return
                        (val['track_a_sim'].setdefault(_sp, {})[
                            f'{setting.name}|k={k}|{scheme}']) = float(
                            metric_fn(_yva, _res['y_pred']))
                        # The kept rows (lnl_methods: flagged = suspicion
                        # above its threshold; all kept when fewer than 10
                        # would remain).
                        _md = _res['metadata']
                        _keep = ~(np.asarray(_md['suspicion_scores'])
                                  > float(_md['threshold']))
                        if _keep.sum() < 10:
                            _keep = np.ones(len(y_noisy), dtype=bool)
                        yield ('distance_selection', 'threshold_pct=85',
                               {'X': _Xtr[_keep],
                                'y': np.asarray(y_noisy)[_keep],
                                'keep': _keep, 'sample_weight': None,
                                'label_column': None})

                    _sweep_pass(
                        sweep_ctx, model_name=TRACK_A_MODEL,
                        model_input=f'{TRACK_A_MODEL}:{sp}', X_tr=X_tr,
                        X_va=X_va, y_tr_noisy=y_noisy, y_tr_clean=_ytr,
                        y_val=_yva, train_idx=_tr, val_idx=_va, groups_tr=_g,
                        fold=_fold, draw=draw, level=float(level),
                        condition=condition,
                        recipes=(entry or {}).get('recipes') or {},
                        base_reps=(entry or {}).get('base') or [],
                        make_dprimes=_ds_dprimes)

            val['screen'] = sweep_ctx.pop('screen', [])
            val['unweighted'] = sorted(UNWEIGHTED_TRACK_B_MODELS)
            return val

        def _run_unit(key, condition, level, draw):
            v = ckpt.get(key)
            if v is None:
                print(f"    {condition or 'level 0'}, level {level:g}, "
                      f"draw {draw}")
                t0 = time.time()
                v = _unit(condition, float(level), int(draw))
                ckpt.put(key, v)
                print(f"      [{time.time() - t0:.0f}s]")
            else:
                print(f"    {condition or 'level 0'}, level {level:g}, draw "
                      f"{draw}: read back from {ckpt.path}")
            UNWEIGHTED_TRACK_B_MODELS.update(v.get('unweighted') or [])
            units.append({'fold': int(fold_i), 'condition': condition,
                          'level': float(level), 'draw': int(draw),
                          'value': v})

        # Level 0 once per fold (the labels as delivered), then every
        # non-zero level of every kind, 3 draws each. Written as each
        # finishes (FoldCheckpoint).
        _run_unit(['unit', int(fold_i), 'level0'], None, 0.0, 0)
        for condition in noise_types:
            for level in levels_by_condition[condition]:
                if float(level) == 0.0:
                    continue
                for draw in NOISE_SEEDS:
                    _run_unit(['unit', int(fold_i), condition, float(level),
                               int(draw)], condition, level, draw)

    # ── Aggregation. A fold's number is the mean over its draws; every
    # spread is over folds (fold_spread). No significance test anywhere. ──
    def _by_fold(cond, level, getter):
        """{fold: mean over draws} of getter(unit value), skipping None."""
        sel = [u for u in units
               if (u['condition'] is None if float(level) == 0.0
                   else (u['condition'] == cond and u['level'] == float(level)))]
        out = {}
        for u in sel:
            x = getter(u['value'])
            if x is None or not np.isfinite(x):
                continue
            out.setdefault(u['fold'], []).append(float(x))
        return {f: float(np.mean(v)) for f, v in out.items()}

    def _get(*path):
        def g(v):
            for p in path:
                if not isinstance(v, dict) or p not in v:
                    return None
                v = v[p]
            return v
        return g

    unit_keys = sorted({f'{m}@{sp}' for u in units
                        for m, d in (u['value'].get('config_scores') or {}).items()
                        for sp in d})
    track_a_spaces = sorted({sp for u in units
                             for sp in (u['value'].get('track_a_plain') or {})})
    track_b_keys = sorted({(dk, e, uk) for u in units
                           for dk, de in (u['value'].get('track_b') or {}).items()
                           for e, du in de.items() for uk in du})
    det_keys = sorted({(dk, e) for u in units
                       for dk, de in (u['value'].get('detection') or {}).items()
                       for e in de})

    def _unit_path(uk):
        m, sp = uk.split('@', 1)
        return ('config_scores', m, sp)

    clean_by_fold = {uk: _by_fold(None, 0.0, _get(*_unit_path(uk)))
                     for uk in unit_keys}
    clean_plain_by_fold = {sp: _by_fold(None, 0.0, _get('track_a_plain', sp))
                           for sp in track_a_spaces}

    def _mean(d):
        return float(np.mean(list(d.values()))) if d else float('nan')

    folds_run = [int(f) for f, _, _ in folds]

    def _spread_by_fold(d):
        """fold_spread over {fold: score}, plus each score under its fold
        number and the folds that have none (a failed fit or an unbuilt
        space), so a per-fold number in the file can be tied to its fold.
        2026-10-05."""
        out = fold_spread([d[f] for f in sorted(d)])
        out['by_fold'] = {str(f): float(d[f]) for f in sorted(d)}
        out['folds_missing'] = [f for f in folds_run if f not in d]
        return out

    def _vs_anchor(mean_s, anchor):
        # retention_pct only when both sides are positive; same guard as n09.
        if (not np.isnan(anchor) and anchor > 1e-10 and mean_s > 1e-10):
            retention = (float(anchor / mean_s * 100) if lower_is_better
                         else float(mean_s / anchor * 100))
        else:
            retention = float('nan')
        delta = float(mean_s - anchor) if not np.isnan(anchor) else float('nan')
        oriented = (float((anchor - mean_s) if lower_is_better
                          else (mean_s - anchor))
                    if not np.isnan(anchor) else float('nan'))
        return retention, delta, oriented

    per_noise_type = {}
    for cond in noise_types:
        nt_result = {'per_strength': {}}
        for level in levels_by_condition[cond]:
            lk = str(float(level))
            noisy_by_fold = {uk: _by_fold(cond, level, _get(*_unit_path(uk)))
                             for uk in unit_keys}
            sr = {
                'level_unit': level_unit(cond, is_clf),
                # Mean over folds of the per-fold draw means, keyed
                # '<model>@<configuration>'. Kept under the old names for the
                # readers (noise_ceiling_analyse, n06).
                'noisy_score_by_model': {uk: _mean(v)
                                         for uk, v in noisy_by_fold.items()},
                'clean_score_by_model': {uk: _mean(v)
                                         for uk, v in clean_by_fold.items()},
                'configuration_scores': {uk: _spread_by_fold(v)
                                         for uk, v in noisy_by_fold.items()},
                # The robustness number on every dataset, and the only one
                # on fast_knn (auc_norm is absent there by decision): damaged
                # minus clean, per fold, in the dataset's own metric and
                # direction.
                'delta_from_clean': {uk: _spread_by_fold(
                    {f: noisy_by_fold[uk][f] - clean_by_fold[uk][f]
                     for f in noisy_by_fold[uk] if f in clean_by_fold[uk]})
                    for uk in unit_keys},
                'track_a': {}, 'track_a_vs_plain_dnn_same_labels': {},
                'track_b': {}, 'detection': {},
            }
            # Track A: each method against the clean plain network on base_1
            # and base_2, and against the plain network on the same labels in
            # the same configuration (the headline, 2026-09-27).
            anchors = {b: _mean(clean_plain_by_fold.get(b) or {})
                       for b in BASE_SPACES}
            for sp in track_a_spaces:
                methods = sorted({mn for u in units
                                  for mn in ((u['value'].get('track_a') or {})
                                             .get(sp) or {})})
                space_a, vs_plain = {}, {}
                for mn in methods:
                    by_f = _by_fold(cond, level, _get('track_a', sp, mn))
                    if not by_f:
                        continue
                    mean_s = _mean(by_f)
                    row = {'score': mean_s,
                           'per_fold': _spread_by_fold(by_f),
                           'retention': {}, 'delta_from_clean_dnn': {},
                           'delta_from_clean_dnn_oriented': {}}
                    for a, anc in anchors.items():
                        r_, d_, o_ = _vs_anchor(mean_s, anc)
                        row['retention'][a] = r_
                        row['delta_from_clean_dnn'][a] = d_
                        row['delta_from_clean_dnn_oriented'][a] = o_
                    space_a[mn] = row
                    pairs = [(u['value']['track_a_plain'].get(sp),
                              u['value']['track_a'][sp][mn]) for u in units
                             if (u['condition'] is None if float(level) == 0.0
                                 else (u['condition'] == cond
                                       and u['level'] == float(level)))
                             and mn in ((u['value'].get('track_a') or {})
                                        .get(sp) or {})
                             and (u['value'].get('track_a_plain') or {})
                             .get(sp) is not None]
                    if pairs:
                        gains = [((p - q) if lower_is_better else (q - p))
                                 for p, q in pairs]
                        vs_plain[mn] = {
                            'plain_dnn_mean': float(np.mean([p for p, _ in pairs])),
                            'method_mean': float(np.mean([q for _, q in pairs])),
                            'gain_mean': float(np.mean(gains)),
                            'gain_per_pass': [float(g) for g in gains],
                            'passes_won': int(sum(g > 0 for g in gains)),
                            'n_passes': len(gains)}
                sr['track_a'][sp] = space_a
                sr['track_a_vs_plain_dnn_same_labels'][sp] = vs_plain
            sr['clean_score_dnn'] = anchors
            # Track B: recovery = (weighted - damaged) / (clean - damaged)
            # x 100, from the fold means of the same unit.
            for dk, est, uk in track_b_keys:
                by_f = _by_fold(cond, level, _get('track_b', dk, est, uk))
                if not by_f:
                    continue
                denoised = _mean(by_f)
                mn_noisy = sr['noisy_score_by_model'].get(uk, float('nan'))
                mn_clean = sr['clean_score_by_model'].get(uk, float('nan'))
                gap = mn_clean - mn_noisy
                recovery = (float((denoised - mn_noisy) / gap * 100)
                            if np.isfinite(gap) and abs(gap) > 1e-10
                            else float('nan'))
                m_, bsp = uk.split('@', 1)
                (sr['track_b'].setdefault(dk, {}).setdefault(est, {}))[uk] = {
                    'denoised': denoised, 'recovery': recovery,
                    'per_fold': _spread_by_fold(by_f),
                    'downstream_model': m_, 'fitted_on': bsp}
            for dk, est in det_keys:
                rows = [u['value']['detection'][dk][est] for u in units
                        if u['condition'] == cond
                        and u['level'] == float(level)
                        and est in ((u['value'].get('detection') or {})
                                    .get(dk) or {})]
                if rows:
                    sr['detection'].setdefault(dk, {})[est] = \
                        _mean_detection(rows)
            nt_result['per_strength'][lk] = convert_numpy(sr)
        per_noise_type[cond] = nt_result

    # The neighbour sweep's own numbers, per '<kind>:<level>'.
    track_a_by_similarity, detection_by_similarity = {}, {}
    screen_by_unit = []
    for u in units:
        cl = ('level0:0.0' if u['condition'] is None
              else f"{u['condition']}:{u['level']}")
        for sp, d in (u['value'].get('track_a_sim') or {}).items():
            for sk, sc in d.items():
                (track_a_by_similarity.setdefault(cl, {}).setdefault(sp, {})
                 .setdefault(sk, {}))[str(u['fold'])] = sc
        for e, by_s in (u['value'].get('sim_detection') or {}).items():
            for sn, by_k in by_s.items():
                for k, rows in by_k.items():
                    (detection_by_similarity.setdefault(cl, {})
                     .setdefault(e, {}).setdefault(sn, {})
                     .setdefault(k, [])).extend(rows)
        for s in u['value'].get('screen') or []:
            screen_by_unit.append(dict(s, unit_fold=u['fold'],
                                       unit_condition=u['condition'],
                                       unit_draw=u['draw']))
    detection_by_similarity = {
        cl: {e: {sn: {k: _mean_detection(rows) for k, rows in by_k.items()}
                 for sn, by_k in by_s.items()}
             for e, by_s in by_e.items()}
        for cl, by_e in detection_by_similarity.items()}

    results = {
        # A SYNTHETIC-noise SENSITIVITY study: how rankings and hybrid
        # advantage SHIFT across injected distributions, NOT an absolute
        # robustness claim about real-world label noise.
        'study_type': 'synthetic_noise_sensitivity',
        'is_absolute_robustness_claim': False,
        'noise_model': 'synthetic (labels injected onto clean targets)',
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'knn_group': knn_grp,
        'models': models,
        'n_folds': int(n_folds),
        'folds_run': folds_run,
        'selection_scope': 'fold_train_rows',
        'configurations': list(CONFIGURATION_SPACES),
        'detection_spaces': list(DETECTION_SPACES),
        'folds': fold_records,
        'noise_types': noise_types,
        'noise_types_refused': noise_types_refused,
        'noise_types_dropped': noise_types_dropped,
        'noise_types_cut_by_decision': (
            [] if is_clf else [c for c in CUT_BY_NOISE_GRID_DECISION
                               if c not in DECIDED_REGRESSION_CONDITIONS]),
        'levels_by_condition': levels_by_condition,
        'level_units': {c: level_unit(c, is_clf) for c in noise_types},
        'noise_seeds': list(NOISE_SEEDS),
        'level_zero': 'the labels as delivered, run once per fold and '
                      'reported under every kind',
        'auc_norm': ('not computed by n13; delta_from_clean (damaged minus '
                     'clean, per fold) is the robustness number'),
        'n_noise_groups': n_noise_groups,
        'largest_group_share': largest_group_share,
        'lnl_methods': lnl_methods,
        'estimators': estimators,
        'track_a_model': TRACK_A_MODEL,
        'track_a_skipped': track_a_skipped,
        'track_b_unweighted_models': sorted(UNWEIGHTED_TRACK_B_MODELS),
        'noise_provenance': {
            f"{u['condition'] or 'level0'}:{u['level']}:draw{u['draw']}:"
            f"fold{u['fold']}": u['value'].get('provenance')
            for u in units},
        'per_noise_type': per_noise_type,
        'similarity': similarity_record(
            sim_settings, sink, estimators_swept=sim_estimators,
            schemes_swept=list(sweep_schemes()), screen_top=screen_top(),
            spaces_unavailable=sweep_ctx['unavailable'],
            checkpoint_file=ckpt.path,
            units_read_back_from_checkpoint=ckpt.n_resumed),
        'screen': screen_by_unit,
        'track_a_by_similarity': track_a_by_similarity,
        'detection_by_similarity': detection_by_similarity,
        'unavailable_by_unit': {
            f"{u['condition'] or 'level0'}:{u['level']}:draw{u['draw']}:"
            f"fold{u['fold']}": u['value'].get('unavailable')
            for u in units if u['value'].get('unavailable')},
    }

    results['summary'] = _compute_summary(results)
    s = results['summary']
    print(f"\n  === SUMMARY ===")
    print(f"  Gaussian rank transfers: {s.get('gaussian_rank_transfers')}")
    print(f"  Hybrid wins per kind: {s.get('hybrid_win_rate_by_noise_type')}")
    if s.get('best_method_per_distribution'):
        for nt, m in s['best_method_per_distribution'].items():
            print(f"    {nt}: best={m}")
    return convert_numpy(results)


def _compute_summary(results):
    """Which Track A method is best under each kind, whether the gaussian
    winner is the winner under every kind, and how often each hybrid
    configuration beats each single-representation configuration, per kind.

    Counts only; no significance test.
    """
    if 'lower_is_better' not in results:
        raise KeyError(
            "n13._compute_summary needs results['lower_is_better']: which way "
            "the dataset's metric runs. run() writes it, so a results dict "
            "without it was built by hand. It is not defaulted, because a "
            "guessed direction reverses every win count below and still "
            "returns a full set of numbers.")
    lower = bool(results['lower_is_better'])
    hybs = ('hybrid', 'hybrid__general2')
    best_per_dist = {}
    wins = {h: {b: {} for b in BASE_SPACES} for h in hybs}
    total = {h: {b: {} for b in BASE_SPACES} for h in hybs}

    for nt, nt_data in results.get('per_noise_type', {}).items():
        best_method, best_score = None, None
        for h in hybs:
            for b in BASE_SPACES:
                wins[h][b][nt] = 0
                total[h][b][nt] = 0
        for sk, sd in nt_data.get('per_strength', {}).items():
            ta = sd.get('track_a', {}) or {}
            for space_data in ta.values():
                for mn, mv in space_data.items():
                    s = mv.get('score', float('nan'))
                    if s is None or np.isnan(s):
                        continue
                    if best_score is None or (s < best_score if lower
                                              else s > best_score):
                        best_score, best_method = s, mn

            def _avg(sp):
                v = [mv.get('score') for mv in (ta.get(sp) or {}).values()
                     if mv.get('score') is not None
                     and not np.isnan(mv.get('score'))]
                return float(np.mean(v)) if v else None
            for h in hybs:
                h_avg = _avg(h)
                for b in BASE_SPACES:
                    b_avg = _avg(b)
                    if h_avg is None or b_avg is None:
                        continue
                    total[h][b][nt] += 1
                    if (h_avg < b_avg) if lower else (h_avg > b_avg):
                        wins[h][b][nt] += 1
        if best_method:
            best_per_dist[nt] = best_method

    gaussian_best = best_per_dist.get('gaussian')
    transfers = (all(best_per_dist.get(nt) == gaussian_best
                     for nt in best_per_dist) if gaussian_best else None)
    rates = {h: {b: {nt: wins[h][b][nt] / total[h][b][nt]
                     for nt in total[h][b] if total[h][b][nt] > 0}
                 for b in BASE_SPACES} for h in hybs}
    # Restored 2026-10-05 for tests/analysis/firstpass_notready.read_n13,
    # which reads it: the arithmetic of the file before chat 5 (the mean over
    # kinds of the per-kind win rate is above one half), now per hybrid
    # configuration and per single representation.
    persists = {h: {b: (bool(np.mean(list(r.values())) > 0.5) if r else None)
                    for b, r in rates[h].items()} for h in hybs}
    return convert_numpy({
        'best_method_per_distribution': best_per_dist,
        'gaussian_rank_transfers': transfers,
        'hybrid_advantage_persists': persists,
        # Per kind: of the levels where both were scored, how many the hybrid
        # configuration's mean Track A score beat the single representation's.
        'hybrid_wins_by_noise_type': wins,
        'hybrid_total_by_noise_type': total,
        'hybrid_win_rate_by_noise_type': rates,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('N13: Noise Distributions')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n13_noise_distributions',
                            args.dataset)
