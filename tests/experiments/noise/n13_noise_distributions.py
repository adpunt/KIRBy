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

Conditions swept (regression): gaussian, student_t at nu = 10, 5 and 3, laplace,
outlier at p = 1%, 5% and 10%, grouped_wider and grouped_shifted. Every one of them
delivers the SAME amount of noise at a given level, so a difference between two cells
is a difference of shape. Censoring is not here: its level is the fraction of labels
clipped rather than a dose, so it is not dose-matched and belongs on its own axis.
  - 'multiplicative' was REMOVED: for mean-centred / arbitrarily-shifted targets
    y*(1+eps) is not a meaningful label-noise model (it scales by absolute value,
    so its magnitude depends on an arbitrary origin), so it was dropped rather
    than reported as a comparable distribution.
  - 'uniform' was REMOVED with the redesign. It assumed the error depends on the
    measured value, and that was tested directly on 16,844 repeat measurements and
    refuted.
  - 'partial' is gone as a separate axis. A corrupted subset is what outlier_p01,
    outlier_p05 and outlier_p10 are, with the fraction carried by the condition,
    so every condition sweeps one shared level grid and every cell is comparable.
  - Classification sweeps the six noiseInject flip models instead: uniform,
    class_imbalance, binary_asymmetric, instance_noise, class_dependent and
    confusion_directed. A flipped class label has no magnitude, so dose matching
    does not apply and the level is a flip rate.
  - grouped_wider and grouped_shifted need one scaffold group per molecule and are
    dropped on datasets with no SMILES rather than run over invented groups.

Three kinds of feature space are compared in both tracks: single
representations, a hybrid built from the predictive-objective search, and a
hybrid built from the similarity-objective search (2026-09-12 decision: n13's
pool subset includes similarity hybrids, and the pool replaces the hand-built
four-block stack this file used to carry).

THE SINGLE REPRESENTATIONS, owner's decision 2026-09-25: each downstream
model's top 2, ranked by that model's own score in the stored search
(_shared.noise_base_reps_by_model). Spaces are named 'base_<k>_<rep>'.
  Track A trains the DNN, so its single-representation spaces are the DNN's
    top 2, and each Track A row is read against the clean DNN score on each of
    them ('base_1' and 'base_2' anchors).
  Track B detects noise in each model's own top 2 (a representation shared by
    several models at one rank is built once, and its rows are fitted only for
    those models) and in the hybrids. Every weighted fit of a model is done
    twice, once on each of its two single representations -- including the rows
    whose noise was detected in a hybrid. Each (model, rank) pair is an
    evaluation unit keyed '<model>@base_<k>' with its own clean score, noisy
    score and gap. Every '*_by_model' score field is keyed by that unit.
  The per-fold single-recipe block lists both of a model's representations as
    its baselines.
Which representations those are is written to 'base_reps_by_model'.

Two evaluation tracks:
  Track A: ALL 12 LNL methods from the registry (not a pre-hybrid shortlist)
           (data-level, training-loop, GP) — retention metric
  Track B: Selected estimators covering each family tested in N11
           (kNN-based, GP-based, gradient-boosted) — recovery metric.
           SAMPLE-WEIGHT remediation only: models whose fit() ignores
           sample_weight (e.g. MLP) are excluded, never fitted unweighted and
           mislabelled as remediated.

Usage:
    python n13_noise_distributions.py --dataset logd --mode intermediary
    python n13_noise_distributions.py --dataset qm9 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    DOWNSTREAM_MODELS_NOISE,
    get_precomputed_search, get_best_search_for_downstream, extract_top3_base_reps,
    build_fold_hybrid, fold_reselection, build_fold_reselection_input,
    hr_exclude_folds,
    # The top-2 selectors and the two hybrid deltas, added 2026-09-18. All four
    # read rows the search already stored and fit nothing.
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    searched_models, resolve_comparison_models,
    # Ranks every model's stored hybrid configurations together and returns the
    # best. Added 2026-09-28 to replace the places that handed a model with no
    # stored search of its own the random forest's recipe.
    top_hybrids_across_models,
)
from kirby.find_best_hybrid import OBJECTIVE_SIMILARITY
# 2026-09-12 decision: n13's pool subset includes similarity hybrids, and the
# pool replaces the hand-built stack. `_similarity_hybrid_space` (the stacked
# count + physchem-minus-saboteurs + 3D-shape + fine-2D matrix, with two of its
# four blocks chosen against MCS structural overlap) is no longer imported here.
# The file itself is untouched — n11 still imports it.
from experiments.dataset_policy import (
    noise_dose_units, noise_groups_for, reference_noise_level,
    noise_levels as dataset_noise_levels,
)
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for, detection_scores,
)
from kirby.noise_spec import (
    CENSORING_STRATEGIES,
    CLASSIFICATION_STRATEGIES, DOSE_MATCHED_STRATEGIES, GROUPED_STRATEGIES,
)
from kirby.utils.lnl_methods import (
    get_lnl_method, _standard_dnn_train_predict,
    _standard_dnn_train_predict_clf, LNL_REGISTRY)
from kirby.utils.noise_estimation import get_noise_estimator, weights_from_scores
# The similarity-sweep kit and the disk caches, added 2026-09-25 (plan
# sections 1-3). See n07_remediation's SIMILARITY-SWEEP KIT block.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, RowSink, sweep_context, similarity_sweep,
    detector_scores_with, cleaned_from_weights, similarity_record,
    FoldCheckpoint,
)
from experiments.noise import noise_cache as _nc
from experiments.noise.noise_cache import NEIGHBOUR_DETECTORS


# The conditions. This experiment's whole subject is whether the SHAPE of the noise
# matters, so it runs the full set rather than a sample of it.
#
# It used to run four of its own: gaussian, uniform, heavy_tailed and partial, each at
# `strength * (max(y) - min(y))`. Two problems, both fatal to the comparison it was
# making. They were not dose-matched, so a difference between them was partly a
# difference of AMOUNT -- at one nominal setting the old six delivered between 0.49x
# and 2.00x the Gaussian amount, and their apparent severity ordering was entirely
# explained by that. And 'partial' needed a second axis of its own (a corrupted-subset
# fraction at a fixed strength) that was not comparable cell-for-cell with the others.
#
# Under the dose-matched set both problems go away. `uniform` is gone: the Uniform
# SHAPE was deleted, because its premise -- that error depends on the measured value --
# was tested on 16,844 repeat measurements and refuted. `partial` is what
# outlier_p01 / outlier_p05 / outlier_p10 now are, with the corrupted fraction folded
# into the condition, so every condition sweeps the same single level grid.
NOISE_TYPES = None       # resolved per task_type in run()

# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy.
STRENGTHS = None

# Multi-seed noise injection: aggregate scores across NOISE_SEEDS to reduce
# dependence on a single noise realization. Per-fold injection still uses
# distinct seeds (combined with fold index below) so each (fold, seed) draw
# is unique.
NOISE_SEEDS = [0, 1, 2]

# 2026-09-12 decision: the part of the precomputed search pool n13 may take a
# hybrid from includes the searches run under objective='similarity' — the ones
# whose features were chosen so that the distance geometry predicts each
# molecule's activity from its five nearest neighbours — not only the
# prediction-built ones. This is NOT the distance=True flag (a different
# evaluator and budget lineage, still prediction-built); n13 does not pass it.
# The two families' stored scores are in different units, so the pool does not
# rank one against the other: it returns the first family named here as the
# winner and every family's own best under metadata['by_objective'], which is
# where the hybrid_sim arm below gets its recipe.
POOL_SUBSET = 'predictive_and_similarity'

# Test the FULL LNL method registry (all 12), NOT a pre-selected shortlist.
# The previous 4-method list ('dividemix', 'small_loss', 'uncertainty_curriculum',
# 'heteroscedastic_gp') was the winner set from the OLD pre-hybrid confirmation
# report. But n09's whole question is whether hybrids RERANK the methods — so
# pre-filtering n13 to a pre-hybrid shortlist would bias the cross-distribution
# robustness result. Testing all 12 keeps n13 self-contained (no dependency on
# n09's output) at ~3x the track-A training cost. Regression-only methods are
# still gated out for classification datasets below (REGRESSION_ONLY_LNL).
LNL_METHODS = list(LNL_REGISTRY.keys())
ESTIMATORS = ['local_variance_ratio', 'gaussian_process']
# ngboost is optional
try:
    get_noise_estimator('ngboost')
    ESTIMATORS.append('ngboost')
except Exception:
    pass

LNL_EPOCHS = 50  # reduced for speed

# The model whose ranking picks Track A's two single representations. Every
# Track A method trains a DNN (_standard_dnn_train_predict and the LNL methods'
# own networks), so the DNN's ranking is the one that applies.
TRACK_A_MODEL = 'dnn'

# Downstream models whose fit() does NOT accept a `sample_weight` argument, so
# they cannot participate in the Track-B sample-weight remediation track. The
# loop below adds any model that raises TypeError on a weighted fit. The seed is
# empty: it used to hold 'dnn', because kirby/models/dnn.py was
# fit(self, X, y, groups=None), and on 2026-09-17 both DNNRegressor.fit and
# DNNClassifier.fit gained sample_weight. A seeded name is never tried, so
# leaving 'dnn' here would have kept it out of track_b after the fix.
# Rows for these models are intentionally absent from track_b (not recorded as a
# zero-recovery result) and the names are surfaced in
# results['track_b_unweighted_models'].
UNWEIGHTED_TRACK_B_SEED = frozenset()
UNWEIGHTED_TRACK_B_MODELS = set(UNWEIGHTED_TRACK_B_SEED)


def _bp_sig(bp):
    """Stable recipe id for a blueprint.

    Doubles as (a) the fold_reselection candidate 'name' (per-fold winner label)
    and (b) the de-dup key so identical per-model blueprints only trigger one
    (expensive) create_hybrid build per fold.
    """
    return "{}|{}|b{}|{}|aug={}".format(
        getattr(bp, 'allocation_method', None),
        getattr(bp, 'importance_method', None),
        getattr(bp, 'budget', None),
        getattr(bp, 'internal_evaluator', None),
        getattr(bp, 'augmentation_strategy', None),
    )


# The private `_inject_noise` this module used to carry has been deleted, along
# with its four hand-rolled distributions and the separate 'partial' axis. Every
# condition now comes from noiseInject through `noise_helpers.inject_train_labels`,
# which solves in closed form for the scale that delivers the requested dose and
# records what it actually delivered.
def _build_spaces(train_reps, val_reps, y_train, augmentations_train,
                  augmentations_val, bp, base_spaces, groups=None,
                  model=None, metric_fn=None, lower_is_better=None,
                  objective=None, is_clf=None):
    """Build the single-representation spaces + hybrid, with the hybrid's val
    features.

    The hybrid goes through build_fold_hybrid, not a bare create_hybrid call.
    The bare call passed only allocation_method / importance_method /
    internal_evaluator / budget / augmentations / augmentation_strategy, so
    accept-or-reject inside the build was scored by the internal evaluator's
    r2 rather than by fitting the real downstream model and calling the
    dataset's metric, and candidate_ranking, augmentation_margin,
    augmentation_fraction and aug_per_source_budget were dropped — the rebuild
    was not the recipe the search stored. build_fold_hybrid forwards all of
    them, and it returns the validation features too, so the separate replay
    of apply_feature_selection on the val reps is gone.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        val_reps: {rep_name: X_val} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations_train: {aug_name: X_train} already sliced, or None.
        augmentations_val: {aug_name: X_val} already sliced, or None.
        bp: Blueprint from get_precomputed_search, for the model that scores
            the hybrid space. None when the pool holds no recipe for that
            model; the returned `spaces` then carries the single
            representations only and no 'hybrid' key.
        base_spaces: {space name: representation} -- the single
            representations noise is detected in, each scaled on this fold's
            training rows.
        model: The downstream estimator INSTANCE the accept-or-reject step
            scores with, or None to fall back on the internal evaluator.
        metric_fn / lower_is_better: The dataset's metric and its direction,
            used with `model`.
        objective: The search family this recipe came from. Under
            objective='similarity' create_hybrid fits no model, so the model
            and the dataset metric are refused there and the objective goes in
            instead.

    Returns (spaces, hybrid_val_scaled_or_None).
    """
    spaces = {}
    hybrid_val_s = None

    for _sp, _rn in base_spaces.items():
        sc = StandardScaler()
        spaces[_sp] = sc.fit_transform(train_reps[_rn])

    # No recipe for the model that scores the hybrid space, so there is no
    # hybrid space. Another model's recipe is not substituted.
    if bp is None:
        return spaces, None

    # The pool subset offers both search families, so read which one this
    # recipe came from before deciding what may be passed. build_fold_hybrid
    # raises when a similarity-objective recipe is handed a model or a metric,
    # and it falls back on the recipe's own objective field the same way.
    _obj = objective if objective is not None else getattr(bp, 'objective', None)
    if _obj == OBJECTIVE_SIMILARITY:
        _obj_kw = {'objective': OBJECTIVE_SIMILARITY, 'is_clf': is_clf}
    else:
        _obj_kw = {'model': model, 'metric_fn': metric_fn,
                   'lower_is_better': lower_is_better}

    try:
        X_h, X_h_val, _ = build_fold_hybrid(
            train_reps, val_reps, y_train, bp,
            augmentations_train=augmentations_train,
            augmentations_val=augmentations_val,
            groups=groups, **_obj_kw)
        scaler_h = StandardScaler()
        spaces['hybrid'] = scaler_h.fit_transform(X_h)
        hybrid_val_s = scaler_h.transform(X_h_val)
    except Exception as e:
        print(f"      hybrid failed: {e}")

    return spaces, hybrid_val_s


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


def _top2_and_deltas(search, models, where, search_by_model=None):
    """The top 2 base representations, the top 2 hybrids and the two hybrid
    deltas, per downstream model. 0 new model fits.

    `search_by_model` names the SearchResult the pool returned for each model.
    Each model's rows are read out of its own. A model absent from it is
    recorded and skipped, never read out of another model's search. `search` is
    then only the one handed to select_model_rep_pairs, which takes a single
    SearchResult and walks every model in it; which model's it is goes on the
    result file as 'top2_pairs_search_of_model'.

    `models` is what the caller asks for -- n13's own roster, from
    config['models'] or DOWNSTREAM_MODELS_NOISE. Two lists can disagree with it:
    the models a stored search scored, which resolve_comparison_models reads off
    the search, and the models the pool returned a search for at all. A name
    dropped by either is written to 'top2_extra_failures' with the reason.

    Added 2026-09-18 on the owner's instruction: the top 2 go in beside what
    this file already reports, and nothing already reported moves. The six keys
    returned here are new and nothing reads them -- not the summary, not the
    printed output, not any other experiment.

    Every selector reads rows the search already stored: baselines[rep][model]
    (find_best_hybrid.py:5524-5525) and all_results (:1536). Nothing is fitted,
    so the whole block costs 0 model fits and only dict passes in wall-clock
    time. select_model_rep_pairs walks those same two structures a second time,
    which is why it is called rather than reassembled from the per-model
    results above it.

    require_full=False on every call. A downstream model the search never
    scored -- svm past its 10,000-molecule cap (find_best_hybrid.py:2969-2978),
    or one a registered skip policy dropped (:2841-2848) -- then comes back as
    a short list carrying its own shortfall_reason. Under the default True it
    raises ValueError instead, which would end a run that has already spent
    hours of fits, for the sake of an extra.

    `hybrid=` is handed to hybrid_deltas rather than left to its default. The
    default path reads `_picked['selected'][0]['row']` (_shared.py:7432) and
    the rows top_hybrids builds carry no 'row' key (_shared.py:7043-7066), so
    the default raises KeyError. The row passed in is that same top-ranked one,
    and it carries the 'score_mean' and 'blueprint' hybrid_deltas reads.

    'blueprint' is dropped from every stored row. A Blueprint is the whole
    recipe (find_best_hybrid.py:1344) and save_experiment_results json.dumps
    this dict (_shared.py:8021), so leaving the object in would write a full
    recipe into each of the 4 rows per downstream model.
    """
    def _no_blueprint(rows):
        return [{k: v for k, v in r.items() if k != 'blueprint'} for r in rows]

    block = {
        'top2_extra_new_model_fits': 0,
        'top2_base_reps': {},
        'top2_hybrids': {},
        'top2_pairs_by_model': {},
        'hybrid_deltas_by_model': {},
        'top2_extra_failures': {},
    }

    _requested = list(models)
    _compare = []
    for m in _requested:
        # NO FALLBACK TO `search`. Until now one search was fetched for rf and
        # all five models were read out of it, so four models' top-2
        # representations, top-2 hybrid configurations and hybrid deltas were
        # ranked on rows the pool chose FOR rf and written under their own
        # names. The pool returns the best stored configuration for the model it
        # is asked about, so the answer for lightgbm can be a different member
        # of the pool. A model the caller passed no search for is recorded below
        # and skipped.
        _s_m = (search_by_model or {}).get(m)
        if _s_m is None:
            block['top2_extra_failures']['no_search/%s' % m] = (
                'the pool returned no stored search for this downstream '
                'model, so there are no rows of its own to rank')
            continue
        _keep, _dropped = resolve_comparison_models(_s_m, [m])
        if not _keep:
            block['top2_extra_failures']['not_scored/%s' % m] = _dropped[0][1]
            continue
        _compare.append(m)
    block['top2_models_requested'] = _requested
    block['top2_models_compared'] = list(_compare)
    block['top2_models_scored_by_each_search'] = {
        _m: searched_models(_s) for _m, _s in (search_by_model or {}).items()}
    for _k, _why in block['top2_extra_failures'].items():
        print(f"    top-2 rows absent, {_k}: {_why}")

    for m in _compare:
        _s_m = search_by_model[m]
        ctx = '{}/{}'.format(where, m)
        try:
            _b = top_base_reps(_s_m, m, n=2, require_full=False, context=ctx)
            block['top2_base_reps'][m] = {
                'selected': _no_blueprint(_b['selected']),
                'n_returned': _b['n_returned'],
                'n_reps_in_baselines': _b['n_reps_in_baselines'],
                'n_reps_scored_for_model': _b['n_reps_scored_for_model'],
                'score_basis': _b['score_basis'],
                'tie_at_cut': _b['tie_at_cut'],
                'shortfall_reason': _b['shortfall_reason'],
            }
        except Exception as exc:
            block['top2_extra_failures']['top_base_reps/%s' % m] = \
                '%s: %s' % (type(exc).__name__, exc)
            print(f"    top-2 base representations skipped for {m}: "
                  f"{type(exc).__name__}: {exc}")

        _h = None
        try:
            _h = top_hybrids(_s_m, m, n=2, require_full=False, context=ctx)
            block['top2_hybrids'][m] = {
                'selected': _no_blueprint(_h['selected']),
                'n_returned': _h['n_returned'],
                'score_basis': _h['score_basis'],
                'ranked_over': _h['ranked_over'],
                'tie_at_cut': _h['tie_at_cut'],
                'notes': _h['notes'],
                'shortfall_reason': _h['shortfall_reason'],
            }
        except Exception as exc:
            _h = None
            block['top2_extra_failures']['top_hybrids/%s' % m] = \
                '%s: %s' % (type(exc).__name__, exc)
            print(f"    top-2 hybrids skipped for {m}: "
                  f"{type(exc).__name__}: {exc}")

        # The two reference numbers for this model's top-ranked hybrid: against
        # the mean of the representations the recipe's per-fold budget split
        # actually gave features to, and against that model's single best base
        # representation. Positive means the hybrid is better, on either metric
        # direction. A representation listed in the recipe but given 0 features
        # on every fold is left out of the first one.
        try:
            if _h is not None and _h['selected']:
                block['hybrid_deltas_by_model'][m] = hybrid_deltas(
                    _s_m, m, hybrid=_h['selected'][0], context=ctx)
        except Exception as exc:
            block['top2_extra_failures']['hybrid_deltas/%s' % m] = \
                '%s: %s' % (type(exc).__name__, exc)
            print(f"    hybrid deltas skipped for {m}: "
                  f"{type(exc).__name__}: {exc}")

    # The same two selectors again, across every downstream model at once, for
    # the counts that exist only across models: how many of the 4 rows per
    # model came back, and which representations or recipes two models both
    # picked. 'per_model' is deliberately not kept -- it is those same rows a
    # second time, with the Blueprint objects still in them.
    try:
        _p = select_model_rep_pairs(search, models=list(models), n_base=2,
                                    n_hybrids=2, require_full=False,
                                    context=where)
        block['top2_pairs_by_model'] = {
            'pairs': _p['pairs'],
            'n_pairs': _p['n_pairs'],
            'n_pairs_possible': _p['n_pairs_possible'],
            'n_distinct_units': _p['n_distinct_units'],
            'collapsed': _p['collapsed'],
            'shortfalls': _p['shortfalls'],
        }
    except Exception as exc:
        block['top2_extra_failures']['select_model_rep_pairs'] = \
            '%s: %s' % (type(exc).__name__, exc)
        print(f"    top-2 grid skipped: {type(exc).__name__}: {exc}")

    return block


def run(data, config, output_dir, dataset_name, **kwargs):
    """N13: Alternative noise distributions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N13: ALTERNATIVE NOISE DISTRIBUTIONS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # 2026-09-20: `noise_helpers.run_all` puts the caller's model list in
    # config['models'] and passes no `models` kwarg. Reading only the module
    # default meant the configured list was silently dropped and every run
    # scored the full five-model roster. Config first, kwarg as override.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS_NOISE)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type,
        n_samples=len(labels))
    print(f"  Models: {list(model_factories.keys())}")
    # SVM internally caps training data at _max_samples=10000. On big
    # datasets (e.g. qm9 at 10K, kiba at 30K+) this triggers SVM-only and
    # silently makes the SVM column an apples-to-oranges comparison against
    # the other models that see the full data. Surface the situation
    # explicitly when SVM is in the set and the dataset is at risk.
    if 'svm' in model_factories and len(labels) > 10000:
        print(f"  WARNING: dataset has {len(labels)} samples — SVM will "
              f"internally cap training data to 10000, so SVM rows below "
              f"are NOT comparable head-to-head with the other models. "
              f"Consider dropping 'svm' from DOWNSTREAM_MODELS_NOISE for "
              f"this dataset.")

    sample_X = next(iter(reps.values()))

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    # Filter out regression-only methods for classification
    # Canonical regression-only set lives in lnl_methods.py; mirroring it
    # locally previously missed evidential_kernel and would silently crash on
    # clf datasets (NIG distribution is regression-only).
    from kirby.utils.lnl_methods import REGRESSION_ONLY_LNL
    lnl_methods = [m for m in LNL_METHODS if not (is_clf and m in REGRESSION_ONLY_LNL)]
    estimators = [e for e in ESTIMATORS if not (is_clf and e == 'gaussian_process')]

    # Regression and classification take different condition sets, because a flipped
    # class label has no magnitude and so cannot be dose-matched.
    noise_types = kwargs.get('noise_types') or NOISE_TYPES
    if noise_types is None:
        # CENSORING JOINED 2026-09-12, owner's decision. It is the one condition
        # that moves labels in ONE direction: every label above a cut-point is
        # replaced by that cut-point, so the corrupted molecules are chosen by
        # their own value. The sister study measured it at roughly twelve times
        # any difference between the zero-mean shapes. It was excluded here only
        # because this file swept DOSE_MATCHED_STRATEGIES, and censoring's level
        # is a fraction of labels clipped rather than an amount of error, so it
        # cannot share the level axis. It gets its own level grid below.
        noise_types = (list(CLASSIFICATION_STRATEGIES) if is_clf
                       else list(DOSE_MATCHED_STRATEGIES) + list(CENSORING_STRATEGIES))
    # A regression condition on a classification dataset is not a different
    # shape: noise_spec_for replaces every one of them with the uniform flip,
    # and seeds the draw from the replaced name, so they come out as identical
    # copies of one flip. A caller-given list (the local runner sets four
    # regression shapes) therefore keeps only its flip models, and when it
    # names none this dataset sweeps the six flip models instead. 2026-09-28.
    noise_types_requested = list(noise_types)
    if is_clf:
        _clf_kept = [c for c in noise_types if c in CLASSIFICATION_STRATEGIES]
        if len(_clf_kept) < len(noise_types):
            noise_types = _clf_kept or list(CLASSIFICATION_STRATEGIES)
            print(f"  classification: the requested conditions "
                  f"{noise_types_requested} would all run as one uniform flip, "
                  f"so the flip models {noise_types} run instead")
    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    if noise_groups is None:
        _dropped = [c for c in noise_types if c in GROUPED_STRATEGIES]
        if _dropped:
            print(f"  {', '.join(_dropped)}: DROPPED, this dataset has no SMILES to "
                  f"build scaffold groups from. Grouped noise over invented groups is "
                  f"uniform noise wearing a grouped name.")
            noise_types = [c for c in noise_types if c not in _dropped]
    else:
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_group_share:.1%} of molecules")

    strengths = kwargs.get('strengths') or kwargs.get('levels') or STRENGTHS
    if strengths is None:
        strengths = list(dataset_noise_levels(dataset_name, is_clf=is_clf))
    noise_provenance = {}

    print(f"  Noise conditions: {noise_types}")
    censoring_levels = list(kwargs.get('censoring_levels')
                            or dataset_noise_levels(dataset_name,
                                                    strategy='censoring'))

    def _levels_for_condition(condition):
        """The level grid this condition sweeps.

        Ten conditions sweep an AMOUNT of error, as a fraction of the clean label
        spread. Censoring sweeps a FRACTION OF LABELS CLIPPED. They are different
        quantities and putting them on one axis was the defect the sister study
        called 'a knob that means different things in different conditions'.
        """
        return censoring_levels if condition in CENSORING_STRATEGIES else strengths

    print(f"  Levels ({noise_dose_units(dataset_name)}): {strengths}")
    if any(c in CENSORING_STRATEGIES for c in noise_types):
        print(f"  Censored fractions (censoring only): {censoring_levels}")
    print(f"  LNL methods: {lnl_methods}")
    print(f"  Estimators: {estimators}")
    print(f"  Folds: {n_folds}")

    # Get blueprint ONCE from pool helper (uses clean labels)
    # 2026-09-12 decision: ask the pool for the widened subset — searches run
    # under objective='similarity' are eligible here alongside the
    # prediction-built ones. The returned winner is still the predictive one
    # whenever that family is on disk, because the two families' scores are in
    # different units and the pool takes the first family the subset names.
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
        pool_subset=POOL_SUBSET,
    )
    # Always define `top3` up front so the results dict below never trips a
    # NameError when search is None (the old code only set it inside the
    # success branch).
    top3 = None
    if search is not None:
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (ranked by RF baseline): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    bp = meta['best_config']['blueprint'] if meta else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}
    print(f"  Pool subset: {meta.get('pool_subset')} — 'hybrid' recipe from "
          f"the {meta.get('objective')} search "
          f"(chosen by {meta.get('objective_choice')}, stored "
          f"{meta.get('metric_name')})")

    # Extra arm: the similarity-objective hybrid, 'hybrid_sim'.
    #
    # 2026-09-12 decision: n13's pool subset includes similarity hybrids, and
    # the pool replaces the hand-built stack. Until today this arm was the
    # four-block matrix from _similarity_hybrid_space — count +
    # physchem-minus-saboteurs + 3D shape + fine-2D, with the physchem and
    # fine-2D blocks chosen against MCS structural overlap over sampled
    # molecule pairs, and no activity label anywhere in the choice. It now
    # comes from the stored search run under objective='similarity' for this
    # dataset, the same way the predictive hybrid comes from the search run
    # under objective='predictive'. Two things change with it: the recipe is a
    # searched allocation method / budget / importance method / internal
    # evaluator instead of a fixed four-block stack, and the features are
    # refit per fold on that fold's noisy training labels rather than chosen
    # label-blind.
    #
    # The arm is on only when that search family is on disk for this dataset;
    # tests/precompute_hybrids.py <dataset> --with-similarity writes it.
    _by_objective = (meta.get('by_objective') or {}) if meta else {}
    sim_entry = _by_objective.get(OBJECTIVE_SIMILARITY)
    bp_sim = sim_entry['best_config']['blueprint'] if sim_entry else None
    sim_aug_condition = sim_entry.get('aug_condition') if sim_entry else None
    sim_arm_on = bp_sim is not None
    sim_build_failures = {}
    if sim_arm_on:
        if getattr(bp_sim, 'augmentation_strategy', None) != sim_aug_condition:
            try:
                bp_sim.augmentation_strategy = sim_aug_condition
            except Exception:
                pass
        print(f"  hybrid_sim arm ON: recipe from the similarity-objective "
              f"search ({sim_entry['tier']}/{sim_entry['aug_condition']}, "
              f"stored {sim_entry['metric_name']}="
              f"{sim_entry['score_mean']:.4f}), rebuilt per fold on that "
              f"fold's noisy training labels")
    else:
        print("  hybrid_sim arm OFF: this dataset has no stored search under "
              "objective='similarity' (write one with "
              "tests/precompute_hybrids.py <dataset> --with-similarity)")

    # Align the blueprint's stored aug_strategy with the pool slot it actually
    # came from (cache-collision guard, see project_cache_collision_fix).
    aug_condition = meta.get('aug_condition', 'uncapped') if meta else 'uncapped'
    if getattr(bp, 'augmentation_strategy', None) != aug_condition:
        try:
            bp.augmentation_strategy = aug_condition
        except Exception:
            pass
    # The augmentations the 'hybrid' space may draw on are gated by the
    # aug condition of the recipe that space is built from, which is
    # TRACK_A_MODEL's and is resolved after the per-model lookups below.

    # ── Per-model MATCHED blueprints, one recipe per downstream model ──
    # Each model's hybrid space is built from a recipe the pool stored for
    # THAT model. Distinct recipes are de-duplicated per fold (via _bp_sig)
    # before the create_hybrid call to bound cost.
    #
    # TRACK_A_MODEL is looked up here too, whether or not it is in the roster,
    # because the 'hybrid' space every Track A method trains on is built from
    # its recipe.
    #
    # 2026-09-12 decision: this resolution takes the widened pool subset too —
    # every place n13 picks a hybrid out of the pool offers the
    # similarity-built searches, not only the prediction-built ones. Which
    # family each model's recipe came from is recorded in
    # results['reselected_hybrid_objective_by_model'].
    bp_by_model = {}
    aug_by_model = {}
    objective_by_model = {}
    # The stored search each model's recipe came from, which also ranks that
    # model's single representations below.
    search_by_model = {}
    for _mname in sorted(set(model_factories) | {TRACK_A_MODEL}):
        try:
            _s_m, _m_m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
                pool_subset=POOL_SUBSET,
            )
        except Exception as _exc:
            print(f"  WARNING: matched blueprint fetch failed for {_mname} ({_exc})")
            continue
        if _s_m is not None:
            search_by_model[_mname] = _s_m
        if _m_m is None:
            continue
        _bp_m = _m_m['best_config']['blueprint']
        objective_by_model[_mname] = _m_m.get('objective')
        _augc_m = _m_m.get('aug_condition', 'uncapped')
        if getattr(_bp_m, 'augmentation_strategy', None) != _augc_m:
            try:
                _bp_m.augmentation_strategy = _augc_m
            except Exception:
                pass
        bp_by_model[_mname] = _bp_m
        aug_by_model[_mname] = _augc_m
    # Every model with its own recipe is marked so before the fill below.
    hybrid_recipe_source_by_model = {m: 'own_search' for m in bp_by_model}
    hybrid_recipe_chosen_by_model = {}

    # ── A model the pool holds no recipe of its own for ──
    # Until 2026-09-28 the loop that stood here copied `bp` -- the recipe the
    # pool returned when it was asked for a random forest -- into every such
    # model, and the per-fold hybrid space below then trained that model on features a
    # random forest's search had chosen while the row was written under the
    # model's own name. Those two disagree about which model the features were
    # chosen for, so the number could not be read as that model's hybrid score.
    #
    # Owner's decision 2026-09-28: rank every model's stored configurations
    # together and take the best one, rather than one named model's. This is
    # still not the model's own recipe, so the name of the model whose score put
    # that configuration on top is recorded per model in
    # results['hybrid_recipe_chosen_by_model'], and
    # results['hybrid_recipe_source_by_model'] says of each model whether its
    # recipe came from its own stored search or from this ranking.
    #
    # The rows being ranked all come from `search`, the one search the pool
    # returned above, so they are one search family. POOL_SUBSETS names the
    # prediction family first and the pool returns the first family it finds, so
    # that family is 'predictive' on every dataset that has a prediction search
    # on disk.
    cross_bp = None
    cross_aug = aug_condition
    cross_objective = meta.get('objective') if meta else None
    cross_chosen_by = None
    cross_block = None
    cross_absent_reason = None
    try:
        _xm = top_hybrids_across_models(search, n=2)
    except Exception as _exc:
        _xm = None
        cross_absent_reason = (f'ranking the stored configurations over every '
                               f'model raised {type(_exc).__name__}: {_exc}')
        print(f"  ranking hybrid configurations over every model failed "
              f"({type(_exc).__name__}: {_exc})")
    if _xm is not None:
        _cand = next((e for e in _xm.get('selected') or []
                      if e.get('blueprint') is not None), None)
        if _cand is None:
            cross_absent_reason = (
                _xm.get('shortfall_reason')
                or 'none of the stored configurations carries a blueprint, so '
                   'no feature set can be rebuilt from them')
        else:
            cross_bp = _cand['blueprint']
            cross_chosen_by = _cand.get('recipe_chosen_by')
            if getattr(cross_bp, 'augmentation_strategy', None) != cross_aug:
                try:
                    cross_bp.augmentation_strategy = cross_aug
                except Exception:
                    pass
            cross_block = {
                'recipe_chosen_by': cross_chosen_by,
                'models_ranked_over': list(_xm.get('models_ranked_over') or []),
                'n_models_ranked_over': (_xm.get('ranked_over') or {}).get(
                    'n_models'),
                'n_configurations_pooled': (_xm.get('ranked_over') or {}).get(
                    'n_configurations_pooled'),
                'score_mean': _cand.get('score_mean'),
                'score_is_in': ('the metric the search stored, '
                                f"{getattr(search, 'metric_name', None)}, "
                                f"produced by {cross_chosen_by} and not by the "
                                f"model it is handed to"),
                'allocation_method': _cand.get('allocation_method'),
                'importance_method': _cand.get('importance_method'),
                'budget': _cand.get('budget'),
                'internal_evaluator': _cand.get('internal_evaluator'),
                'objective': cross_objective,
                'aug_condition': cross_aug,
            }
            print(f"  best hybrid configuration over every model the search "
                  f"scored: allocation {_cand.get('allocation_method')}, budget "
                  f"{_cand.get('budget')} features, importance "
                  f"{_cand.get('importance_method')}, internal evaluator "
                  f"{_cand.get('internal_evaluator')} — chosen by "
                  f"{cross_chosen_by}'s score "
                  f"({getattr(search, 'metric_name', None)}="
                  f"{_cand.get('score_mean')})")

    # TRACK_A_MODEL is filled too, whether or not this dataset's gating put it
    # in the roster: the 'hybrid' space every Track A method trains on is built
    # from its recipe.
    for _mname in sorted(set(model_factories) | {TRACK_A_MODEL}):
        if _mname in bp_by_model:
            continue
        if cross_bp is None:
            continue
        bp_by_model[_mname] = cross_bp
        aug_by_model[_mname] = cross_aug
        objective_by_model[_mname] = cross_objective
        hybrid_recipe_source_by_model[_mname] = 'across_models'
        hybrid_recipe_chosen_by_model[_mname] = cross_chosen_by
        print(f"  {_mname}: the pool holds no recipe searched for {_mname}, so "
              f"its hybrid space is built from the best configuration over "
              f"every model the search scored, which {cross_chosen_by}'s score "
              f"chose")

    # Still nothing after the fill: no configuration over any model could be
    # rebuilt. These models have no hybrid rows at all rather than rows holding
    # a number produced on features chosen for a different model.
    hybrid_unscored_models = sorted(m for m in model_factories
                                    if m not in bp_by_model)
    for _mname in hybrid_unscored_models:
        print(f"  {_mname}: no hybrid space at all -- {cross_absent_reason}")

    # ── The 'hybrid' space Track A trains on, and Track B estimates noise in ──
    # Every Track A method trains a neural network
    # (_standard_dnn_train_predict and the label-noise methods' own networks),
    # so this space is built from TRACK_A_MODEL's own recipe. Until 2026-09-28
    # it was built from `bp`, the random forest's recipe, while the network that
    # scored it was a different model: the features were chosen for one model
    # and every Track A number was produced by another.
    #
    # When the pool holds no recipe searched for TRACK_A_MODEL, the fill above
    # has put the best configuration over every model the search scored here
    # instead, and results['hybrid_recipe_chosen_by_model'] names the model whose
    # score chose it. None only when no configuration over any model could be
    # rebuilt; the 'hybrid' space is then absent from Track A and from Track B's
    # detection spaces.
    bp_track_a = bp_by_model.get(TRACK_A_MODEL)
    aug_condition_track_a = aug_by_model.get(TRACK_A_MODEL, 'uncapped')
    bp_objective = objective_by_model.get(TRACK_A_MODEL)
    # The estimator the 'hybrid' space's accept-or-reject step is scored with:
    # the same model the search that produced this recipe fitted. None when
    # this dataset's gating dropped TRACK_A_MODEL from the roster, and
    # build_fold_hybrid then falls back on the internal evaluator.
    hybrid_model_factory = model_factories.get(TRACK_A_MODEL)
    if bp_track_a is None:
        print(f"  'hybrid' space OFF: the pool holds no recipe searched for "
              f"{TRACK_A_MODEL}, which is the model every Track A method "
              f"trains, and no configuration over the other models could be "
              f"rebuilt either -- {cross_absent_reason}")
    elif hybrid_recipe_source_by_model.get(TRACK_A_MODEL) == 'across_models':
        print(f"  'hybrid' space built from the best configuration over every "
              f"model the search scored, which {cross_chosen_by}'s score chose; "
              f"the pool holds none searched for {TRACK_A_MODEL}")
    # Gate the augmentations so a 'none' pool slot really propagates no
    # augmentation features into the 'hybrid' space.
    use_augmentations = (augmentations if aug_condition_track_a != 'none'
                         else None)

    # ── Each model's top-2 single representations ──
    # Ranked by that model's own score in its own stored search. Track A trains
    # the DNN, so the DNN is ranked too whether or not it is in the roster. A
    # `base_reps` kwarg ({model: [reps]}) overrides the ranking, for a smoke or a
    # direct caller.
    #
    # ⛔ A MODEL WITH NO STORED SEARCH OF ITS OWN IS LEFT UNRANKED HERE, and the
    # reason is written per model to results['base_reps_missing']. Until
    # 2026-09-28 this line read `search_by_model.get(m) or search`, so such a
    # model was ranked in the search the pool returned when it was asked for a
    # random forest, and results['base_reps_search_model'] recorded 'rf' for it.
    # These are single representations, not hybrid features: the cross-model
    # ranking that fills a missing hybrid recipe above ranks stored hybrid
    # configurations, and there is no equivalent of it for a base-representation
    # ranking, so nothing is substituted. A model left out here has no single
    # representation to be fitted on and so no evaluation unit at all.
    _rank_models = list(model_factories) + (
        [TRACK_A_MODEL] if TRACK_A_MODEL not in model_factories else [])
    _base_override = kwargs.get('base_reps')
    if _base_override is not None:
        base_by_model = {m: list(_base_override[m])[:NOISE_BASE_REP_COUNT]
                         for m in _rank_models if _base_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in _rank_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: search_by_model.get(m) for m in _rank_models},
            _rank_models, reps=reps, dataset_name=dataset_name)
    eval_models = [m for m in model_factories if m in base_by_model]
    # Evaluation units for Track B and the reference scores: one per (model,
    # rank), '<model>@base_<k>'.
    eval_units = {f'{m}@base_{k}': (m, k, rn)
                  for m in eval_models
                  for k, rn in enumerate(base_by_model[m], start=1)}
    # Track B detection spaces: {space: representation} and the models each
    # is fitted for.
    base_spaces = {}
    base_space_models = {}
    for _k in range(1, NOISE_BASE_REP_COUNT + 1):
        for _rn, _ms in base_rank_groups(base_by_model, _k,
                                         eval_models).items():
            base_spaces[f'base_{_k}_{_rn}'] = _rn
            base_space_models[f'base_{_k}_{_rn}'] = list(_ms)
    # Track A's single-representation spaces: the DNN's top 2.
    track_a_base = {f'base_{_k}_{_rn}': (_k, _rn) for _k, _rn in enumerate(
        base_by_model.get(TRACK_A_MODEL) or [], start=1)}
    for _u, (_m, _k, _rn) in eval_units.items():
        print(f"  evaluation unit {_u}: {_m} fitted on {_rn}")
    for _sp, (_k, _rn) in track_a_base.items():
        print(f"  Track A single representation rank {_k} "
              f"({TRACK_A_MODEL}): {_rn}")
    for _m, _why in base_missing.items():
        print(f"  {_m}: no single representation -- {_why}")
    if not eval_units and not track_a_base:
        print("  no model has a single representation to be fitted on, so "
              "nothing can be scored")
        return {'error': 'no_single_representation', 'metric': metric_name,
                'base_reps_by_model': base_by_model,
                'base_reps_missing': base_missing}
    # The headline 'clean_score'/'noisy_score' fields were random forest on
    # the one base representation. They are now rf on its rank-1 one.
    _ref_unit = 'rf@base_1'

    # Reset the unweighted-model registry to its seed for this run (module-level
    # mutable set persists across run() calls in a single process).
    global UNWEIGHTED_TRACK_B_MODELS
    # From the one seed above. This line held its own literal {'dnn'} until
    # 2026-09-17, so the module constant and the per-run reset could disagree.
    UNWEIGHTED_TRACK_B_MODELS = set(UNWEIGHTED_TRACK_B_SEED)

    results = {
        # This experiment is a SYNTHETIC-noise SENSITIVITY study: it reports how
        # rankings/hybrid-advantage SHIFT across injected noise distributions,
        # NOT an absolute robustness claim about real-world label noise.
        'study_type': 'synthetic_noise_sensitivity',
        'is_absolute_robustness_claim': False,
        'noise_model': 'synthetic (labels injected onto clean targets)',
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        # Every condition now sweeps ONE axis -- the level -- and delivers the same
        # amount of noise at a given level, so cells are comparable across the whole
        # grid. The old 'partial' track had an axis of its own that was not.
        'noise_types': noise_types,
        'noise_types_requested': noise_types_requested,
        'noise_types_dropped': ['multiplicative', 'uniform'],
        'strengths': strengths,
        'dose_units': noise_dose_units(dataset_name),
        'noise_provenance': noise_provenance,
        'n_noise_groups': n_noise_groups,
        'largest_group_share': largest_group_share,
        'lnl_methods': lnl_methods,
        'estimators': estimators,
        # Downstream models participating in the honest per-fold re-score arm
        # (gated by n_samples via get_model_factories — never a fixed set).
        'models': list(model_factories.keys()),
        # Each model's top 2 single representations, from its own score in the
        # stored search. See the module docstring.
        'base_reps_by_model': base_by_model,
        'base_reps_missing': base_missing,
        # Every model here was ranked in its OWN stored search. A model with no
        # stored search of its own is absent from 'base_reps_by_model' and its
        # reason is in 'base_reps_missing'; nothing is ranked in another model's
        # search, which is what this key used to record as 'rf'.
        'base_reps_search_model': {m: m for m in base_by_model},
        'base_reps_unranked_models': sorted(base_missing),
        'base_reps_no_substitution': (
            'a model with no stored search of its own is left unranked; its '
            'single representations are not taken from another model\'s search'),
        'track_a_base_reps_ranked_by': TRACK_A_MODEL,
        'evaluation_units': {u: {'model': m, 'rank': k, 'representation': rn}
                             for u, (m, k, rn) in eval_units.items()},
        'evaluation_unit_key': (
            "'<model>@base_<k>': that model fitted on its rank-k single "
            "representation. Every *_by_model score field is keyed by it."),
        'track_b_base_detection_spaces': {
            sp: {'representation': rn, 'models': base_space_models[sp]}
            for sp, rn in base_spaces.items()},
        'reference_unit': _ref_unit,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # Track-B sample-weight remediation excludes these models because their
        # fit() ignores sample_weight; their rows are absent from track_b
        # rather than masquerading as zero-recovery noisy-baseline ties.
        'track_b_unweighted_models': sorted(UNWEIGHTED_TRACK_B_MODELS),
        # 2026-09-12 decision: the pool subset this file draws its hybrids from
        # includes the similarity-built searches. Which family each hybrid
        # actually came from is recorded here, so a reader can tell the two
        # apart. 'objective' is the family of the recipe driving the 'hybrid'
        # space (Track A and Track B) and 'objective_choice' says how the pool
        # picked it: 'only_family' when just one family was on disk,
        # 'subset_order' when both were and their scores are in different
        # units, 'score' when they share a metric.
        'pool_subset': (meta.get('pool_subset') if meta else POOL_SUBSET),
        'hybrid_objective': (meta.get('objective') if meta else None),
        'hybrid_objective_choice': (meta.get('objective_choice') if meta
                                    else None),
        'hybrid_metric_name': (meta.get('metric_name') if meta else None),
        'pool_by_objective': convert_numpy({
            _o: {'tier': _d.get('tier'),
                 'aug_condition': _d.get('aug_condition'),
                 'score_mean': _d.get('score_mean'),
                 'metric_name': _d.get('metric_name'),
                 'lower_is_better': _d.get('lower_is_better')}
            for _o, _d in _by_objective.items()}),
        # Family of the per-model matched recipe used by the honest per-fold
        # hybrid space below, one entry per downstream model the pool held a
        # recipe for.
        'reselected_hybrid_objective_by_model': dict(objective_by_model),
        # Where each model's hybrid recipe came from. 'own_search' means the
        # stored search the pool returned when it was asked for that model.
        # 'across_models' means the pool held none for it, so the recipe is the
        # best configuration over every model the search scored, and
        # 'hybrid_recipe_chosen_by_model' names the model whose score put that
        # configuration on top. Nothing is taken from one named model and written
        # under another's name without one of these two entries saying so.
        'hybrid_recipe_source_by_model': dict(hybrid_recipe_source_by_model),
        'hybrid_recipe_chosen_by_model': dict(hybrid_recipe_chosen_by_model),
        # The one configuration that fill uses, and how many models' stored rows
        # were ranked together to find it. None when nothing could be ranked, and
        # 'hybrid_recipe_across_models_absent_reason' then says why.
        'hybrid_recipe_across_models': convert_numpy(cross_block),
        'hybrid_recipe_across_models_absent_reason': cross_absent_reason,
        # Downstream models with no hybrid space at all: the pool holds no recipe
        # searched for them AND no configuration over the other models could be
        # rebuilt. Their hybrid rows are absent rather than holding a number
        # produced on features chosen for a different model.
        'hybrid_unscored_models': hybrid_unscored_models,
        # The model whose recipe builds the 'hybrid' space that Track A trains on
        # and Track B estimates noise in, whether a recipe was found, and where
        # it came from. With no recipe there is no 'hybrid' space in either track.
        'track_a_hybrid_recipe_model': TRACK_A_MODEL,
        'track_a_hybrid_recipe_present': bp_track_a is not None,
        'track_a_hybrid_recipe_source': hybrid_recipe_source_by_model.get(
            TRACK_A_MODEL),
        'track_a_hybrid_recipe_chosen_by': hybrid_recipe_chosen_by_model.get(
            TRACK_A_MODEL),
        'per_noise_type': {},
    }

    # ── Top-2 rows and the two hybrid deltas, beside what is reported ──
    # Owner's instruction 2026-09-18: add the top 2 as an extra and change
    # nothing already reported. 'top3_base_reps' above is untouched and still
    # holds the three names extract_top3_base_reps ranked on the rf baseline;
    # these keys sit next to it.
    #
    # ONE SEARCH PER DOWNSTREAM MODEL. `search` was fetched with
    # downstream_model='rf' and all five models' rows used to be read out of it,
    # so four models' top-2 representations, top-2 hybrid configurations and
    # hybrid deltas came from rows the pool chose for a random forest and were
    # written under their own names. `search_by_model` above already holds the
    # search the pool returned for each model, from the per-model lookups that
    # fetched their matched recipes, and each model is read out of its own here.
    # No model falls back to rf's. 0 new model fits: every helper reads rows the
    # search already stored.
    #
    # select_model_rep_pairs takes ONE SearchResult and walks every model in it,
    # so it is given the search of the model named in
    # 'top2_pairs_search_of_model'.
    if search_by_model:
        _grid_model = next(
            (m for m in model_factories if m in search_by_model),
            next(iter(search_by_model)))
        results['top2_pairs_search_of_model'] = _grid_model
        results.update(_top2_and_deltas(
            search_by_model[_grid_model], list(model_factories.keys()),
            f'n13/{dataset_name}', search_by_model=search_by_model))
    else:
        results['top2_extra_absent_reason'] = (
            'the pool returned a recipe but no SearchResult for any downstream '
            'model, so there are no stored rows to rank and no baseline scores '
            'to difference against')

    # The plain DNN every Track A method is read against. On a classification
    # dataset it is the cross-entropy network the label-noise methods train
    # (LNLMethod._default_predict routes there on is_clf), returning the
    # positive-class probability. Until 2026-09-28 both calls below used the
    # regression trainer on every dataset, so on a classification set the
    # methods were compared with a squared-error network on the class codes.
    if is_clf:
        _all_classes = np.unique(labels)

        def _plain_dnn(X_tr, y_tr, X_va, **kw):
            return _standard_dnn_train_predict_clf(X_tr, y_tr, X_va,
                                                   classes_=_all_classes, **kw)
    else:
        _plain_dnn = _standard_dnn_train_predict

    # ── Cache fold splits and label-independent eval features ──
    clean_scores_by_model = {}
    # Track A is DNN-based (all LNL methods use _standard_dnn_train_predict
    # internally). Use a dedicated DNN clean baseline for Track A retention
    # so we're comparing DNN-with-LNL vs DNN-without-LNL, not against the
    # RF baseline used by Track B and noisy-baseline reporting.
    # One list per Track A anchor: the DNN on its rank-k single representation.
    clean_scores_dnn = {f'base_{_k}': [] for _k, _rn in track_a_base.values()}
    fold_data = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                        labels, mode=mode):
        y_train_clean = labels[train_idx]
        y_val = labels[val_idx]

        train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
        val_reps_fold = {k: v[val_idx] for k, v in reps.items()}

        # Single-representation features (independent of labels), one scaled
        # pair per representation any unit or Track A space uses.
        # Cached 2026-09-25 (noise_cache.cached_scaled): the same
        # StandardScaler arithmetic, stored once per (representation content,
        # fold).
        _by_rep = {}
        for _rn in ({r for _m, _k, r in eval_units.values()}
                    | {r for _k, r in track_a_base.values()}):
            _by_rep[_rn] = _nc.cached_scaled(dataset_name, reps[_rn],
                                             train_idx, val_idx)
        eval_by_unit = {u: _by_rep[rn] for u, (_m, _k, rn) in eval_units.items()}

        # Clean baseline per unit (Track B reference)
        for _u, (mname, _k, _rn) in eval_units.items():
            if _u not in clean_scores_by_model:
                clean_scores_by_model[_u] = []
            # Through noise_cache.cached_fit_predict since 2026-09-25: the
            # same fit and model.predict, stored once per (features, labels,
            # model) and read back by the D-versus-D' rows.
            try:
                clean_scores_by_model[_u].append(float(metric_fn(
                    y_val, _nc.cached_fit_predict(
                        dataset_name, mname, model_factories[mname],
                        eval_by_unit[_u][0], y_train_clean,
                        eval_by_unit[_u][1],
                        predict_mode=_nc.predict_mode_for(metric_name)))))
            except Exception:
                pass

        # DNN clean baseline on each of its two single representations
        # (matched-evaluator anchors for Track A)
        for _sp, (_k, _rn) in track_a_base.items():
            try:
                dnn_pred = _plain_dnn(
                    _by_rep[_rn][0], y_train_clean, _by_rep[_rn][1],
                    epochs=LNL_EPOCHS, patience=10, random_state=42)
                clean_scores_dnn[f'base_{_k}'].append(
                    float(metric_fn(y_val, dnn_pred)))
            except Exception:
                pass

        # Slice augmentations through the aug-condition-aware shim so 'none'
        # pool slots really propagate no aug columns.
        train_augs = {k: v[train_idx] for k, v in use_augmentations.items()} if use_augmentations else None
        # val_augs was missing — the Track A hybrid path below tries to read
        # fd.get('val_augs'), couldn't find it, dropped the augmentation
        # columns from the val feature matrix, and scaler_h.transform then
        # failed silently (column-count mismatch). Hybrid Track A was thus
        # skipped on every augmentation dataset.
        val_augs = {k: v[val_idx] for k, v in use_augmentations.items()} if use_augmentations else None

        # 2026-09-12: nothing is precomputed per fold for hybrid_sim any more.
        # The hand-built four-block stack was label-blind, so one build per
        # fold served every noise condition. The pool recipe that replaced it
        # selects features from the labels, so it is rebuilt inside the seed
        # loop below from that fold's noisy training labels, exactly like the
        # predictive hybrid.
        fold_data.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'y_train_clean': y_train_clean,
            'y_val': y_val,
            'train_reps_fold': train_reps_fold,
            'val_reps_fold': val_reps_fold,
            'train_augs': train_augs,
            'val_augs': val_augs,
            'eval_by_unit': eval_by_unit,
            'track_a_base': {_sp: _by_rep[_rn]
                             for _sp, (_k, _rn) in track_a_base.items()},
        })

    clean_mean_by_model = {mn: float(np.mean(s)) if s else float('nan')
                           for mn, s in clean_scores_by_model.items()}
    # Primary clean score for top-level reporting still uses RF; Track A
    # uses the DNN-specific clean baseline below for matched-evaluator
    # retention math.
    clean_mean = clean_mean_by_model.get(_ref_unit, float('nan'))
    clean_mean_dnn = {_a: (float(np.mean(_v)) if _v else float('nan'))
                      for _a, _v in clean_scores_dnn.items()}

    # A sensitivity study has no single "the" condition, so the top-level
    # contract block exposes a CANONICAL moderate-noise slice (gaussian
    # sigma=0.3 when present, else the first computed cell). The full per-
    # (noise_type, strength) set is also kept nested. See the assignment after
    # the main loop.
    canonical_reselected = None
    canonical_reselected_key = None
    # A sensitivity study has no single "the" condition, so the canonical slice is
    # Gaussian at one unit of real error for this dataset -- 0.54 log units on
    # herg_ki, 0.35 on caco2, half the label SD where nothing is published.
    _canonical_level = min(strengths, key=lambda v: abs(v - reference_noise_level(dataset_name))) \
        if strengths else 0.0

    # ── The similarity axis and the checkpoint, added 2026-09-25 ──────────
    # Plan sections 1-2. Track B's neighbour estimator (local_variance_ratio;
    # noise_cache.NEIGHBOUR_DETECTORS) reads its neighbours from every
    # similarity setting at every k in SWEPT_KS, and its sample-weight repair
    # (exponential at strength 0.1, as Track B) is written as D-versus-D' rows
    # for RF, SVM and the unit's model, on each unit's single representation.
    # Everything Track A and Track B already report is unchanged.
    #
    # Track A's distance_selection finds neighbours and drops training
    # molecules, so it is swept too (see Track A in the loop).
    #
    # TODO(similarity-sweep 2026-09-25): Track A's other methods change how
    # the network trains, not the training set, so there is no D' to fit RF
    # and SVM on and they write no rows. small_loss and confident_learning do
    # drop molecules but do not return which; mentornet reads a neighbour list
    # but carves its checkpoint-selection rows out of X_train first
    # (lnl_methods.py, MentorNet.train_and_predict), so a list over the
    # training fold no longer lines up. Each needs a decision on its D'.
    sim_settings = sweep_settings()
    sink = RowSink(output_dir, 'n13_noise_distributions', dataset_name)
    sim_estimators = [e for e in estimators if e in NEIGHBOUR_DETECTORS]
    sweep_ctx = sweep_context(
        experiment='n13_noise_distributions', dataset_name=dataset_name,
        sink=sink, settings=sim_settings,
        metric=(metric_name, metric_fn, lower_is_better), task_type=task_type,
        condition=None, level=None, data=data, factories=model_factories)
    sim_detection = {}
    track_a_sim = {}
    # The DNN's rows need a DNN factory beside the roster's.
    sweep_ctx['factories'] = dict(get_model_factories(
        ['rf', 'svm', 'dnn'], task_type=task_type, n_samples=len(labels)),
        **model_factories)

    def _sim_recipes_for(mname):
        """The two hybrid spaces this model's similarity settings read.

        'hybrid' is the prediction-objective recipe. It is the model's own where
        the pool held one, and otherwise the best configuration over every model
        the search scored, which the fill above already put in bp_by_model.
        results['hybrid_recipe_source_by_model'] says which of the two it was per
        model and results['hybrid_recipe_chosen_by_model'] names the model whose
        score chose a cross-model one.

        Until 2026-09-28 this opened with the recipe the pool returned when it
        was asked for a random forest, and kept it for every model whose own
        recipe came from the nearest-neighbour family or was missing. A random
        forest's recipe is no longer put in another model's place.

        'hybrid_similarity' is the nearest-neighbour-objective recipe, one per
        dataset, present when that search family is on disk for this dataset.

        A recipe from the nearest-neighbour family never fills the 'hybrid' slot:
        the two are different feature spaces and comparing them is the point of
        having both. POOL_SUBSET names the prediction family first and the pool
        returns the first family it finds, so a model whose recipe came from the
        nearest-neighbour family means this dataset has no prediction search on
        disk at all, and its 'hybrid' slot is reported unavailable by
        fold_similarity_spaces rather than filled.
        """
        rec = {}
        _bp_m = bp_by_model.get(mname)
        if (_bp_m is not None
                and objective_by_model.get(mname) != OBJECTIVE_SIMILARITY):
            rec['hybrid'] = (_bp_m, aug_by_model.get(mname, aug_condition))
        if bp_sim is not None:
            rec['hybrid_similarity'] = (bp_sim, sim_aug_condition)
        return rec

    # n13 wrote its result once, at the end, after every (condition, level)
    # had run -- days on a large dataset. Each (condition, level) is now
    # written as it finishes and read back on the next run
    # (n07_remediation.FoldCheckpoint), keyed on the inputs; a changed input
    # recomputes. Floats round-trip through JSON exactly, so a resumed file is
    # the same numbers.
    from experiments._shared import input_fingerprint
    ckpt = FoldCheckpoint(output_dir, 'n13_noise_distributions', dataset_name,
                          tag='levels', fingerprint={
                              'input': input_fingerprint(data, config),
                              'noise_seeds': list(NOISE_SEEDS),
                              'lnl_methods': list(lnl_methods),
                              'estimators': list(estimators),
                              'models': list(model_factories),
                              'lnl_epochs': LNL_EPOCHS})
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x Track B estimators {sim_estimators}; rows -> "
          f"{sink.path}; checkpoint -> {ckpt.path}")

    # ── Main loop: noise type × strength ──
    for noise_type in noise_types:
        print(f"\n  === {noise_type} ===")
        t_nt = time.time()
        nt_result = {'per_strength': {}}

        # One axis for the ten dose-matched conditions; censoring has its own,
        # because its level counts labels clipped and theirs measures an amount of
        # error. The corrupted fraction that used to need a third axis is carried
        # by the condition itself (outlier_p01/p05/p10).
        for strength in _levels_for_condition(noise_type):
            strength_key = str(strength)

            # Resume: a (condition, level) already on disk is read back, with
            # the provenance, build failures and unweighted models it added.
            _ck = ckpt.get([noise_type, strength_key])
            if _ck is not None:
                strength_result = _ck['strength_result']
                noise_provenance.update(_ck.get('noise_provenance') or {})
                for _kf, _nf in (_ck.get('sim_build_failures') or {}).items():
                    sim_build_failures[_kf] = (sim_build_failures.get(_kf, 0)
                                               + int(_nf))
                UNWEIGHTED_TRACK_B_MODELS.update(_ck.get('unweighted') or [])
                if _ck.get('track_a_sim'):
                    track_a_sim[f'{noise_type}:{strength_key}'] = \
                        _ck['track_a_sim']
                if _ck.get('detection_sim'):
                    sim_detection[f'{noise_type}:{strength_key}'] = \
                        _ck['detection_sim']
                reselection_block = strength_result[
                    'per_fold_scores_single_recipe']
                _is_canonical = (noise_type == 'gaussian'
                                 and abs(strength - _canonical_level) < 1e-9)
                if _is_canonical or canonical_reselected is None:
                    canonical_reselected = reselection_block
                    canonical_reselected_key = f'{noise_type}:{strength_key}'
                nt_result['per_strength'][strength_key] = strength_result
                print(f"    σ={strength_key}: read back from {ckpt.path}")
                continue
            _prov_before = set(noise_provenance)
            _sbf_before = dict(sim_build_failures)

            noisy_baseline_by_model = {u: [] for u in eval_units}
            # {space: {method: [scores]}} for track A
            track_a_acc = {}
            # {space: {method: [(plain DNN score, method score)]}}, one pair
            # per (fold, draw) where both fits succeeded. 2026-09-27.
            track_a_pairs = {}
            # {space: {estimator: {model: [scores]}}} for track B
            track_b_acc = {}
            # {space: {estimator: [detection reading per (fold, draw)]}}. Stays
            # empty unless the condition is one with a corrupted subset.
            detection_acc = {}

            # Honest per-fold arms for the shared fold_reselection helper:
            #   reselected_base_folds[model][fold] = [per-seed base-rep scores]
            #   reselected_hyb_folds[model][fold]  = [per-seed matched-hybrid scores]
            # Each fold entry is averaged over NOISE_SEEDS below, then laid out
            # fold-aligned so fold_reselection can leave-one-fold-out re-select
            # each arm's winner WITHOUT the held-out fold (the arithmetic
            # embodiment of the n09 nested-selection template). We do NOT run a
            # per-fold find_best_hybrid inner search here: that would multiply
            # the noise_type x strength x fold x seed matrix by a full search
            # and bust the 7-day wall; the LOFO re-selection over the cached
            # matched-blueprint candidates gives the held-out-fold-free
            # selection instead.
            # reselected_base_folds[model][rep][fold], one entry per single
            # representation of that model.
            reselected_base_folds = {mn: {} for mn in model_factories}
            reselected_hyb_folds = {mn: {} for mn in model_factories}
            reselected_hyb_name = {}  # model -> its blueprint signature (candidate name)

            for fi, fd in enumerate(fold_data):
                for noise_seed in NOISE_SEEDS:
                    # One seed per (fold, noise draw) for the label-noise
                    # learners below. It was referenced there and never
                    # defined, so every Track A method raised NameError into
                    # a bare except and Track A produced nothing at all.
                    _seed_combined = 42 + 1000 * fi + noise_seed
                    # Per-(condition, level, fold, seed) noise realization
                    _spec = noise_spec_for(dataset_name, noise_type, strength,
                                           noise_seed=noise_seed,
                                           is_clf=is_clf)
                    _tr_idx = fd['train_idx']
                    y_noisy, _cmask, _prov = inject_train_labels(
                        labels, _spec, train_idx=_tr_idx, is_clf=is_clf,
                        groups=noise_groups, dataset_name=dataset_name)
                    noise_provenance.setdefault(_spec.spec_id, _prov)

                    # Noisy baseline per unit
                    for _u, (mname, _k, _rn) in eval_units.items():
                        try:
                            _xtr, _xva = fd['eval_by_unit'][_u]
                            _sc_base = float(metric_fn(
                                fd['y_val'], _nc.cached_fit_predict(
                                    dataset_name, mname, model_factories[mname],
                                    _xtr, y_noisy, _xva,
                                    predict_mode=_nc.predict_mode_for(
                                        metric_name))))
                            noisy_baseline_by_model[_u].append(_sc_base)
                            # Per-fold single-representation arm for
                            # fold_reselection (reuses this same fit — no extra
                            # training cost).
                            (reselected_base_folds[mname].setdefault(_rn, {})
                             .setdefault(fi, []).append(_sc_base))
                        except Exception:
                            pass

                    # Build spaces from NOISY labels (realistic — no oracle
                    # leak). The 'hybrid' space is built from TRACK_A_MODEL's
                    # own recipe, because every method scored on it trains that
                    # model. bp_track_a is None when the pool holds no recipe
                    # for it, and _build_spaces then returns the single
                    # representations only.
                    spaces, hybrid_val_s = _build_spaces(
                        fd['train_reps_fold'], fd['val_reps_fold'], y_noisy,
                        fd['train_augs'], fd['val_augs'],
                        bp_track_a, base_spaces,
                        groups=(groups[fd['train_idx']]
                                if groups is not None else None),
                        model=(hybrid_model_factory()
                               if hybrid_model_factory is not None else None),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better,
                        objective=bp_objective, is_clf=is_clf)

                    # similarity-objective hybrid arm. 2026-09-12 decision: the
                    # pool replaces the hand-built four-block stack, so this is
                    # the recipe the similarity-objective search picked, refit
                    # on this fold's training rows and the same noisy labels
                    # the predictive hybrid above sees.
                    #
                    # No downstream model and no dataset metric go into the
                    # build: under objective='similarity' create_hybrid scores
                    # its own neighbour geometry and fits no model, and
                    # build_fold_hybrid refuses a model/metric_fn there.
                    #
                    # The old stack was taken as built, not rescaled, because
                    # each of its four blocks carried a weight in the distance.
                    # A pool-built hybrid has no such blocks, so it is scaled
                    # the same way the predictive hybrid is.
                    hybrid_sim_val_s = None
                    if bp_sim is not None:
                        try:
                            if sim_aug_condition != 'none' and augmentations:
                                _tr_aug_s = {k: v[fd['train_idx']]
                                             for k, v in augmentations.items()}
                                _va_aug_s = {k: v[fd['val_idx']]
                                             for k, v in augmentations.items()}
                            else:
                                _tr_aug_s = _va_aug_s = None
                            _Xs_tr, _Xs_va, _ = build_fold_hybrid(
                                fd['train_reps_fold'], fd['val_reps_fold'],
                                y_noisy, bp_sim,
                                augmentations_train=_tr_aug_s,
                                augmentations_val=_va_aug_s,
                                groups=(groups[fd['train_idx']]
                                        if groups is not None else None),
                                objective=OBJECTIVE_SIMILARITY,
                                is_clf=is_clf)
                            _sc_sim = StandardScaler()
                            spaces['hybrid_sim'] = _sc_sim.fit_transform(_Xs_tr)
                            hybrid_sim_val_s = _sc_sim.transform(_Xs_va)
                        except Exception as _exc:
                            _key_f = f"{type(_exc).__name__}: {_exc}"[:200]
                            if _key_f not in sim_build_failures:
                                print(f"      hybrid_sim build failed: {_key_f}")
                            sim_build_failures[_key_f] = \
                                sim_build_failures.get(_key_f, 0) + 1

                    # ── Honest hybrid arm: train each model PLAINLY (no LNL,
                    # no sample weights) on its MATCHED hybrid space, built from
                    # that model's blueprint on the NOISY labels. This is the
                    # hybrid candidate the fold_reselection helper compares against
                    # the base-rep baseline arm. Identical blueprints across
                    # models share one create_hybrid build via _hyb_space_cache.
                    _hyb_space_cache = {}  # bp_sig -> (X_tr_s, X_va_s) or None
                    for mname, mfactory in model_factories.items():
                        _bp_m = bp_by_model.get(mname)
                        if _bp_m is None:
                            continue
                        # 2026-09-12: the recipe name carries the search family
                        # it came from, so two families that happen to share an
                        # allocation method / budget / importance method /
                        # internal evaluator are still told apart in the
                        # per-fold score block.
                        _obj_m = objective_by_model.get(mname)
                        _sig = f"{_obj_m}|{_bp_sig(_bp_m)}"
                        reselected_hyb_name[mname] = _sig
                        if _sig not in _hyb_space_cache:
                            try:
                                _augc_m = aug_by_model.get(mname, 'uncapped')
                                if _augc_m != 'none' and augmentations:
                                    _tr_aug = {k: v[fd['train_idx']]
                                               for k, v in augmentations.items()}
                                    _va_aug = {k: v[fd['val_idx']]
                                               for k, v in augmentations.items()}
                                else:
                                    _tr_aug = _va_aug = None
                                # A recipe from the similarity family is refit
                                # under its own objective, which fits no
                                # downstream model — passing the model and the
                                # dataset metric there is refused, so they go
                                # in only for a prediction-built recipe.
                                if _obj_m == OBJECTIVE_SIMILARITY:
                                    _obj_kw = {'objective': OBJECTIVE_SIMILARITY,
                                               'is_clf': is_clf}
                                else:
                                    _obj_kw = {'model': mfactory(),
                                               'metric_fn': metric_fn,
                                               'lower_is_better': lower_is_better}
                                _Xh_tr, _Xh_va, _ = build_fold_hybrid(
                                    fd['train_reps_fold'], fd['val_reps_fold'],
                                    y_noisy, _bp_m,
                                    augmentations_train=_tr_aug,
                                    augmentations_val=_va_aug,
                                    groups=(groups[fd['train_idx']] if groups is not None else None),
                                    **_obj_kw)
                                _sc_h = StandardScaler()
                                _hyb_space_cache[_sig] = (
                                    _sc_h.fit_transform(_Xh_tr),
                                    _sc_h.transform(_Xh_va))
                            except Exception:
                                _hyb_space_cache[_sig] = None
                        _built = _hyb_space_cache[_sig]
                        if _built is None:
                            continue
                        _Xh_tr_s, _Xh_va_s = _built
                        try:
                            m = mfactory()
                            m.fit(_Xh_tr_s, y_noisy)
                            reselected_hyb_folds[mname].setdefault(fi, []).append(
                                float(metric_fn(fd['y_val'], _nc._predict(
                                    m, _Xh_va_s,
                                    _nc.predict_mode_for(metric_name)))))
                        except Exception:
                            pass

                    # The hybrid's val features come back from _build_spaces
                    # with the train side, so there is no second build and no
                    # replay of the feature selection on the val reps here.

                    # ── Track A: LNL methods ──
                    # The DNN's two single representations, then the hybrids.
                    _track_a_spaces = [(_sp, _x[0], _x[1]) for _sp, _x
                                       in fd['track_a_base'].items()]
                    if 'hybrid' in spaces and hybrid_val_s is not None:
                        _track_a_spaces.append(
                            ('hybrid', spaces['hybrid'], hybrid_val_s))
                    if 'hybrid_sim' in spaces and hybrid_sim_val_s is not None:
                        _track_a_spaces.append(
                            ('hybrid_sim', spaces['hybrid_sim'], hybrid_sim_val_s))
                    for space_name, X_train_s, X_val_s in _track_a_spaces:

                        if space_name not in track_a_acc:
                            track_a_acc[space_name] = {}

                        # D: the plain DNN on the SAME labels the methods get
                        # (recorded at level 0.0, injected above), same space
                        # and seed. Owner's decision 2026-09-27: every method
                        # is read against this, not against the clean-label
                        # DNN, which mixes the noise damage into the number.
                        try:
                            _plain_pred = _plain_dnn(
                                X_train_s, y_noisy, X_val_s,
                                epochs=LNL_EPOCHS, patience=10,
                                random_state=_seed_combined)
                            _plain_score = float(metric_fn(fd['y_val'],
                                                           _plain_pred))
                        except Exception:
                            _plain_score = None

                        for method_name in lnl_methods:
                            if method_name not in track_a_acc[space_name]:
                                track_a_acc[space_name][method_name] = []
                            try:
                                m = get_lnl_method(
                                    method_name, epochs=LNL_EPOCHS,
                                    patience=10, random_state=_seed_combined,
                                    is_clf=is_clf,
                                    classes_=(np.unique(labels) if is_clf else None))
                                res = m.train_and_predict(X_train_s, y_noisy,
                                                          X_val_s, fd['y_val'])
                                score = float(metric_fn(fd['y_val'], res['y_pred']))
                                track_a_acc[space_name][method_name].append(score)
                                if _plain_score is not None:
                                    (track_a_pairs.setdefault(space_name, {})
                                     .setdefault(method_name, [])
                                     .append((_plain_score, score)))
                            except Exception:
                                pass

                    # ── Track A's distance_selection under every similarity
                    # setting, 2026-09-25. The one LNL method that finds
                    # neighbours and drops training molecules
                    # (lnl_methods.DistanceBasedSelection takes a neighbour
                    # list since 2026-09-25). Its DNN score per (space,
                    # setting, k) goes to results['track_a_by_similarity'];
                    # the rows it kept are D', written as rows for RF, SVM
                    # and the DNN. The neighbour spaces are the DNN's.
                    if ('distance_selection' in lnl_methods and sink.enabled):
                        _g_tr = (groups[fd['train_idx']]
                                 if groups is not None else None)
                        for space_name, X_train_s, X_val_s in _track_a_spaces:
                            def _ds_dprimes(setting, lists, k,
                                            _Xtr=X_train_s, _Xva=X_val_s,
                                            _sp=space_name):
                                try:
                                    _mth = get_lnl_method(
                                        'distance_selection', k=k,
                                        precomputed_neighbors=lists,
                                        epochs=LNL_EPOCHS, patience=10,
                                        random_state=_seed_combined,
                                        is_clf=is_clf,
                                        classes_=(np.unique(labels) if is_clf
                                                  else None))
                                    _res = _mth.train_and_predict(
                                        _Xtr, y_noisy, _Xva, fd['y_val'])
                                except Exception as _e:
                                    print(f"      distance_selection "
                                          f"{setting.name} k={k} failed: {_e}")
                                    yield ('distance_selection',
                                           'threshold_pct=85', None)
                                    return
                                (track_a_sim.setdefault(
                                    f'{noise_type}:{strength_key}', {})
                                 .setdefault(_sp, {})
                                 .setdefault(setting.name, {})
                                 .setdefault(str(k), []).append(float(
                                     metric_fn(fd['y_val'], _res['y_pred']))))
                                # The kept rows, rebuilt from what the method
                                # returns (lnl_methods.py: flagged = suspicion
                                # above its threshold; all kept when fewer
                                # than 10 would remain).
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
                            similarity_sweep(
                                sweep_ctx, model_name='dnn',
                                model_input=f'dnn:{space_name}', X_tr=X_train_s,
                                X_va=X_val_s, y_tr_noisy=y_noisy,
                                y_tr_clean=fd['y_train_clean'],
                                y_val=fd['y_val'], train_idx=fd['train_idx'],
                                val_idx=fd['val_idx'], groups_tr=_g_tr,
                                fold=fi, draw=noise_seed,
                                recipes=_sim_recipes_for(TRACK_A_MODEL),
                                base_reps=base_by_model.get(TRACK_A_MODEL, []),
                                make_dprimes=_ds_dprimes, level=strength,
                                condition=noise_type)

                    # ── Track B: Estimation + remediation ──
                    # Asymmetry by design: the noise-detection step iterates
                    # over the single-representation and hybrid spaces (so the
                    # estimator sees different geometry), but the model that
                    # consumes the resulting sample weights ALWAYS trains on a
                    # single representation -- each of that model's top 2 in
                    # turn, one evaluation unit each. That makes Track B
                    # specifically ask "does hybrid-derived noise detection
                    # improve single-representation training?", distinct from
                    # Track A's "does hybrid feature space help LNL methods
                    # that train on hybrid?". Recovery denominator (clean/noisy
                    # baselines) is the same unit's, so the comparison is
                    # apples-to-apples.
                    for space_name in list(base_spaces) + ['hybrid', 'hybrid_sim']:
                        if space_name not in spaces:
                            continue
                        X_space = spaces[space_name]
                        # A single-representation space is fitted only for the
                        # models that rank that representation there.
                        _open = base_space_models.get(space_name)

                        if space_name not in track_b_acc:
                            track_b_acc[space_name] = {}

                        for est_name in estimators:
                            if est_name not in track_b_acc[space_name]:
                                track_b_acc[space_name][est_name] = {}
                            try:
                                est = get_noise_estimator(est_name)
                                noise_scores = est.estimate(X_space, y_noisy)
                                # How well this estimator FOUND the corrupted
                                # molecules, before any repair. Owner's decision
                                # 2026-09-12. Absent under the six conditions where
                                # every training molecule moves. Costs no model fit.
                                _det = detection_scores(_cmask, noise_scores,
                                                        noise_type)
                                if _det is not None:
                                    (detection_acc
                                     .setdefault(space_name, {})
                                     .setdefault(est_name, [])
                                     .append(_det))
                                weights = weights_from_scores(noise_scores, strength=0.1,
                                                              method='exponential')
                                for _u, (mname, _k, _rn) in eval_units.items():
                                    if _open is not None and mname not in _open:
                                        continue
                                    mfactory = model_factories[mname]
                                    if _u not in track_b_acc[space_name][est_name]:
                                        track_b_acc[space_name][est_name][_u] = []
                                    # Track B is a SAMPLE-WEIGHT remediation track:
                                    # the only thing distinguishing it from the
                                    # noisy baseline is the per-sample reliability
                                    # weight. Models whose fit() does NOT accept
                                    # `sample_weight` (sklearn MLPRegressor /
                                    # MLPClassifier) cannot be remediated this
                                    # way — a fallback unweighted fit would
                                    # reproduce the noisy baseline EXACTLY
                                    # (recovery == 0 by construction) and silently
                                    # inflate "no recovery" counts. Skip such
                                    # models entirely and record them so the
                                    # aggregation can mark them as N/A rather than
                                    # as a genuine zero-recovery result.
                                    if mname in UNWEIGHTED_TRACK_B_MODELS:
                                        continue
                                    try:
                                        m = mfactory()
                                        _xtr, _xva = fd['eval_by_unit'][_u]
                                        try:
                                            m.fit(_xtr, y_noisy,
                                                  sample_weight=weights)
                                        except TypeError:
                                            # Model does not support sample_weight:
                                            # it is not remediable in Track B.
                                            # Remember it and skip — do NOT record
                                            # an unweighted score masquerading as
                                            # a remediated one.
                                            UNWEIGHTED_TRACK_B_MODELS.add(mname)
                                            continue
                                        # DIRTY-LOCAL 2026-09-28: AUC needs
                                        # the positive-class probability, not
                                        # the hard class (predict_mode_for).
                                        score = float(metric_fn(fd['y_val'],
                                                                 _nc._predict(
                                            m, _xva,
                                            _nc.predict_mode_for(metric_name))))
                                        track_b_acc[space_name][est_name][_u].append(score)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                    # ── The similarity sweep for this (fold, draw). ──
                    # One pass per evaluation unit the Track B repair is
                    # fitted for. The estimator is built as Track B builds it
                    # (get_noise_estimator(name): k=10, the process distance,
                    # is_clf left False, no group exclusion), so the lists
                    # are built without group exclusion and MODEL_INPUT at
                    # k=10 is Track B's own neighbour lookup in the unit's
                    # representation.
                    for _u, (mname, _k_rank, _rn) in eval_units.items():
                        if (not sim_estimators or not sink.enabled
                                or mname in UNWEIGHTED_TRACK_B_MODELS):
                            continue
                        _xtr, _xva = fd['eval_by_unit'][_u]

                        def _dprimes(setting, lists, k, _xtr=_xtr):
                            for est_name in sim_estimators:
                                sc = detector_scores_with(
                                    dataset_name, est_name, _xtr, y_noisy,
                                    lists, k, is_clf=False)
                                if sc is None:
                                    yield est_name, 'exponential_0.1', None
                                    continue
                                _det = detection_scores(_cmask, sc, noise_type)
                                if _det is not None:
                                    (sim_detection
                                     .setdefault(f'{noise_type}:{strength_key}',
                                                 {})
                                     .setdefault(est_name, {})
                                     .setdefault(setting.name, {})
                                     .setdefault(str(k), []).append(_det))
                                w = weights_from_scores(sc, strength=0.1,
                                                        method='exponential')
                                yield (est_name, 'exponential_0.1',
                                       cleaned_from_weights(_xtr, y_noisy, w))
                        similarity_sweep(
                            sweep_ctx, model_name=mname,
                            model_input=f'{_u}:{_rn}',
                            X_tr=_xtr, X_va=_xva, y_tr_noisy=y_noisy,
                            y_tr_clean=fd['y_train_clean'], y_val=fd['y_val'],
                            train_idx=fd['train_idx'], val_idx=fd['val_idx'],
                            groups_tr=(groups[fd['train_idx']]
                                       if groups is not None else None),
                            fold=fi, draw=noise_seed,
                            recipes=_sim_recipes_for(mname),
                            base_reps=base_by_model.get(mname, []),
                            make_dprimes=_dprimes, level=strength,
                            condition=noise_type)

            # Aggregate per model
            noisy_mean_by_model = {mn: float(np.mean(s)) if s else float('nan')
                                   for mn, s in noisy_baseline_by_model.items()}
            noisy_mean = noisy_mean_by_model.get(_ref_unit, float('nan'))
            # Top-level gap is anchored to rf on its rank-1 single
            # representation (Track B uses per-unit gaps below; Track A uses
            # the DNN-specific gaps). Kept here for the JSON output's headline
            # 'noise_gap' field; 'reference_unit' names it.
            noise_gap_rf = clean_mean - noisy_mean

            strength_result = {
                'noisy_score_by_model': noisy_mean_by_model,
                'clean_score_by_model': clean_mean_by_model,
                'noisy_score': noisy_mean,
                'clean_score': clean_mean,
                'noise_gap_rf': noise_gap_rf,
                'track_a': {},
                'track_b': {},
            }

            # Track A aggregation (DNN-specific — retention is anchored to the
            # DNN clean baseline so it's a matched-evaluator number, not an
            # RF-anchored mismatch as it was previously). There are two
            # anchors, the clean DNN on its rank-1 and on its rank-2 single
            # representation, and every row is read against both: each of
            # 'retention', 'delta_from_clean_dnn' and its oriented twin is a
            # dict keyed 'base_1' / 'base_2'.
            def _vs_anchor(mean_s, anchor):
                # retention_pct only when both sides have the same sign
                # and the denominator is bounded away from zero; same
                # robustness guard as n09 (R² can be negative).
                if (not np.isnan(anchor)
                        and anchor > 1e-10 and mean_s > 1e-10):
                    if lower_is_better:
                        retention = float(anchor / mean_s * 100)
                    else:
                        retention = float(mean_s / anchor * 100)
                else:
                    retention = float('nan')
                delta = (float(mean_s - anchor) if not np.isnan(anchor)
                         else float('nan'))
                # Same difference, sign-normalised so a positive number always
                # means this method beat the clean-label DNN. The raw one runs
                # the dataset's own way and reads backwards on rmse. Added
                # 2026-09-10 (metric-direction audit).
                oriented = (float((anchor - mean_s) if lower_is_better
                                  else (mean_s - anchor))
                            if not np.isnan(anchor) else float('nan'))
                return retention, delta, oriented

            for space_name, methods in track_a_acc.items():
                space_a = {}
                for method_name, score_list in methods.items():
                    if score_list:
                        mean_s = float(np.mean(score_list))
                        _row = {'score': mean_s, 'retention': {},
                                'delta_from_clean_dnn': {},
                                'delta_from_clean_dnn_oriented': {}}
                        for _a, _anchor in clean_mean_dnn.items():
                            _r, _d, _o = _vs_anchor(mean_s, _anchor)
                            _row['retention'][_a] = _r
                            _row['delta_from_clean_dnn'][_a] = _d
                            _row['delta_from_clean_dnn_oriented'][_a] = _o
                        space_a[method_name] = _row
                strength_result['track_a'][space_name] = space_a
            strength_result['clean_score_dnn'] = clean_mean_dnn

            # THE TRACK A HEADLINE, 2026-09-27: each method against the plain
            # DNN on the same labels, same space, same (fold, draw). gain > 0
            # means the method won, in the dataset metric.
            _vs_plain = {}
            for space_name, by_m in track_a_pairs.items():
                _vs_plain[space_name] = {}
                for method_name, prs in by_m.items():
                    gains = [((p - m) if lower_is_better else (m - p))
                             for p, m in prs]
                    _vs_plain[space_name][method_name] = {
                        'plain_dnn_mean': float(np.mean([p for p, _ in prs])),
                        'method_mean': float(np.mean([m for _, m in prs])),
                        'gain_mean': float(np.mean(gains)),
                        'gain_per_pass': [float(g) for g in gains],
                        'passes_won': int(sum(g > 0 for g in gains)),
                        'n_passes': len(gains),
                    }
            strength_result['track_a_vs_plain_dnn_same_labels'] = _vs_plain

            # Track B aggregation (per estimator × per model)
            for space_name, est_dict in track_b_acc.items():
                space_b = {}
                for est_name, model_dict in est_dict.items():
                    est_results = {}
                    for mname, score_list in model_dict.items():
                        if score_list:
                            denoised = float(np.mean(score_list))
                            mn_noisy = noisy_mean_by_model.get(mname, noisy_mean)
                            mn_clean = clean_mean_by_model.get(mname, clean_mean)
                            mn_gap = mn_clean - mn_noisy
                            recovery = float((denoised - mn_noisy) / mn_gap * 100) if abs(mn_gap) > 1e-10 else float('nan')
                            est_results[mname] = {
                                'denoised': denoised,
                                'recovery': recovery,
                                # 'mname' is the evaluation unit; these split it.
                                'downstream_model': eval_units[mname][0],
                                'fitted_on': f'base_{eval_units[mname][1]}',
                                'fitted_on_representation': eval_units[mname][2],
                            }
                    space_b[est_name] = est_results
                strength_result['track_b'][space_name] = space_b

            # ── Per-fold scores for this (noise_type, strength) ──
            # Fold-aligned per-arm scores (mean over NOISE_SEEDS within each
            # fold; np.nan for folds a model failed on), in the shape the shared
            # fold_reselection helper consumes.
            #
            # The hybrid side is NOT a re-selection. fold_reselection's job is
            # to re-pick each side's winner from the folds other than the
            # held-out one, and it needs a list of recipes to pick from. n13
            # gives it ONE hybrid recipe per model, so there is nothing to
            # re-pick on that side. The base side holds the model's top 2 single
            # representations (2026-09-25), so fold_reselection picks between
            # those two without the held-out fold. The key used to be called
            # 'fold_reselection', which promised a hybrid re-selection; it is
            # named for what it holds.
            #
            # That one recipe per model came from the clean-label search pool
            # (get_best_search_for_downstream above), like every recipe in this
            # suite. Making the re-selection real means feeding every searched
            # recipe's per-fold scores, which costs one refit per
            # noise_type x strength x fold x seed per recipe.
            hr_candidates = {}
            hr_baselines = {}
            for mn in model_factories:
                base_map = reselected_base_folds.get(mn, {})
                hyb_map = reselected_hyb_folds.get(mn, {})
                _hb = {}
                for _rn, _fmap in base_map.items():
                    if any(_fmap.get(fi) for fi in range(n_folds)):
                        _hb[_rn] = [float(np.mean(_fmap[fi]))
                                    if _fmap.get(fi) else float('nan')
                                    for fi in range(n_folds)]
                if _hb:
                    hr_baselines[mn] = _hb
                if any(hyb_map.get(fi) for fi in range(n_folds)):
                    hr_candidates[mn] = [{
                        'name': reselected_hyb_name.get(mn, 'hybrid'),
                        'fold_scores': [
                            float(np.mean(hyb_map[fi]))
                            if hyb_map.get(fi) else float('nan')
                            for fi in range(n_folds)],
                    }]
            reselection_block = {
                'lower_is_better': lower_is_better,
                # Every fold counts. The candidates here are the arms this file built itself,
                # not a search's recipe list, so no fold filtered them.
                'exclude_folds': [],
                'candidates_by_model': hr_candidates,
                'baselines_by_model': hr_baselines,
                # One hybrid recipe per model, so no leave-one-fold-out
                # re-selection happens on that side; the base side is the
                # model's top 2 single representations. Stated here so a
                # reader of the file does not have to infer it from the list
                # lengths.
                'n_candidate_recipes_per_model': {
                    mn: len(v) for mn, v in hr_candidates.items()},
                'n_single_representations_per_model': {
                    mn: len(v) for mn, v in hr_baselines.items()},
                'reselection': ('hybrid: none, single recipe per model; '
                                'base: leave-one-fold-out between the '
                                "model's top 2 single representations"),
                'recipe_source': 'clean_label_search_pool',
            }
            strength_result['per_fold_scores_single_recipe'] = reselection_block

            # Stash the canonical top-level slice (moderate gaussian noise) so a
            # reader can json.load(...)['per_fold_scores_single_recipe'] directly.
            _is_canonical = (noise_type == 'gaussian'
                             and abs(strength - _canonical_level) < 1e-9)
            if _is_canonical or canonical_reselected is None:
                canonical_reselected = reselection_block
                canonical_reselected_key = f'{noise_type}:{strength_key}'

            if detection_acc:
                strength_result['detection'] = {
                    _sp: {_e: _mean_detection(_rows) for _e, _rows in _by_est.items()}
                    for _sp, _by_est in detection_acc.items()}

            nt_result['per_strength'][strength_key] = convert_numpy(strength_result)
            ckpt.put([noise_type, strength_key], {
                'strength_result': nt_result['per_strength'][strength_key],
                'noise_provenance': {k: v for k, v in noise_provenance.items()
                                     if k not in _prov_before},
                'sim_build_failures': {
                    k: v - _sbf_before.get(k, 0)
                    for k, v in sim_build_failures.items()
                    if v != _sbf_before.get(k, 0)},
                'unweighted': sorted(UNWEIGHTED_TRACK_B_MODELS),
                'track_a_sim': track_a_sim.get(f'{noise_type}:{strength_key}'),
                'detection_sim': sim_detection.get(
                    f'{noise_type}:{strength_key}')})

            # Print compact
            print(f"    σ={strength_key}: noisy={noisy_mean:.4f}", end='')
            # Best track A
            best_a = None
            best_a_score = float('-inf') if not lower_is_better else float('inf')
            for sp_d in strength_result['track_a'].values():
                for mn, mv in sp_d.items():
                    s = mv.get('score', float('nan'))
                    if not np.isnan(s):
                        if (not lower_is_better and s > best_a_score) or \
                           (lower_is_better and s < best_a_score):
                            best_a_score = s
                            best_a = mn
            if best_a:
                print(f", best_LNL={best_a}({best_a_score:.4f})", end='')
            print()

        results['per_noise_type'][noise_type] = convert_numpy(nt_result)
        print(f"    [{time.time() - t_nt:.0f}s]")

    # Refresh after the loop in case extra models were auto-detected as
    # sample_weight-incompatible (TypeError on weighted fit).
    results['track_b_unweighted_models'] = sorted(UNWEIGHTED_TRACK_B_MODELS)

    # ── Top-level per-fold score block ──
    # A sensitivity study has no single condition; expose a CANONICAL moderate-
    # noise slice here (gaussian sigma=0.3 when available, else the first
    # computed cell). The full per-(noise_type, strength) set is preserved under
    # per_noise_type[*]['per_strength'][*]['per_fold_scores_single_recipe'].
    #
    # Deliberately NOT written under the key 'fold_reselection': that name says
    # the winning recipe was re-picked without the fold it is scored on, and
    # here there is only one recipe per model to pick. The delta printed below
    # is one hybrid recipe against one base representation, fold for fold.
    if canonical_reselected is not None:
        results['per_fold_scores_single_recipe'] = convert_numpy(canonical_reselected)
        results['per_fold_scores_condition'] = canonical_reselected_key
        # Sanity log via the shared helper (pure arithmetic, no model fitting).
        try:
            _hr = fold_reselection(canonical_reselected['candidates_by_model'],
                                 canonical_reselected['baselines_by_model'],
                                 lower_is_better,
                                 exclude_folds=hr_exclude_folds(canonical_reselected))
            print(f"\n  === PER-FOLD SCORES, ONE RECIPE PER MODEL "
                  f"({canonical_reselected_key}) ===")
            print("  One hybrid recipe per model, so the hybrid side is not "
                  "re-picked; the base side is re-picked between the model's "
                  "top 2 single representations.")
            for _mn, _r in _hr.items():
                print(f"    {_mn}: hybrid={_r['hybrid_reselected']:.4f} "
                      f"base={_r['best_single_reselected']:.4f} "
                      f"delta={_r['delta']:+.4f} single_fold={_r['single_fold']}")
        except Exception as _exc:
            print(f"  per-fold score summary skipped ({_exc})")

    # Where the 'hybrid_sim' space came from. 2026-09-12 decision: the pool
    # replaces the hand-built four-block stack, so this records the stored
    # similarity-objective search the recipe was read from, not a molecule
    # cache. 'recipe_refit_rows' = the rows the features were refit on, one
    # rebuild per (noise condition, level, fold, noise seed).
    results['hybrid_sim'] = convert_numpy({
        'arm_on': bool(sim_arm_on),
        'source': 'pool_search_objective_similarity',
        'objective': OBJECTIVE_SIMILARITY if sim_arm_on else None,
        'tier': sim_entry.get('tier') if sim_entry else None,
        'aug_condition': sim_aug_condition,
        'search_score_mean': sim_entry.get('score_mean') if sim_entry else None,
        'search_metric_name': (sim_entry.get('metric_name') if sim_entry
                               else None),
        'recipe_refit_rows': 'fold_train_rows_noisy_labels' if sim_arm_on else None,
        'skip': (None if sim_arm_on else
                 "no stored search under objective='similarity' for this "
                 "dataset"),
        'build_failures': dict(sim_build_failures),
    })

    # The similarity axis, 2026-09-25. Every key above is n13's own tracks,
    # unchanged. The swept numbers are the D-versus-D' rows in
    # results['similarity']['rows_file'] and, for detection,
    # {'<condition>:<level>': {estimator: {setting: {k: mean reading}}}}.
    results['similarity'] = similarity_record(
        sim_settings, sink, estimators_swept=sim_estimators,
        spaces_unavailable=sweep_ctx['unavailable'],
        checkpoint_file=ckpt.path,
        levels_read_back_from_checkpoint=ckpt.n_resumed)
    # {'<condition>:<level>': {space: {setting: {k: mean DNN score}}}}.
    results['track_a_by_similarity'] = {
        _cl: {_sp: {_sn: {_k: {'mean': float(np.mean(_v)), 'n': len(_v)}
                          for _k, _v in _byk.items()}
                    for _sn, _byk in _bys.items()}
              for _sp, _bys in _bysp.items()}
        for _cl, _bysp in track_a_sim.items()}
    results['detection_by_similarity'] = {
        _cl: {_e: {_sn: {_k: _mean_detection(_rows)
                         for _k, _rows in _byk.items()}
                   for _sn, _byk in _bys.items()}
              for _e, _bys in _bye.items()}
        for _cl, _bye in sim_detection.items()}

    # Summary
    results['summary'] = _compute_summary(results)
    s = results['summary']
    print(f"\n  === SUMMARY ===")
    print(f"  Gaussian rank transfers: {s.get('gaussian_rank_transfers')}")
    print(f"  Hybrid advantage persists: {s.get('hybrid_advantage_persists')}")
    print(f"  Similarity-objective hybrid vs base rep: "
          f"{s.get('hybrid_sim_advantage_persists')}")
    if s.get('best_method_per_distribution'):
        for nt, m in s['best_method_per_distribution'].items():
            print(f"    {nt}: best={m}")

    return results


def _compute_summary(results):
    """Compute summary: does Gaussian ranking transfer? Does hybrid help across distributions?"""
    # Collect best Track A method per (noise_type, space)
    best_per_dist = {}  # noise_type -> best method name
    # Hybrid advantage tally: count by NOISE TYPE, not by (noise_type x level).
    # Every condition now sweeps the SAME single level grid, so no condition
    # contributes more cells than any other -- the inflation the old 'partial'
    # track caused, with its own separate axis, is gone. The per-type
    # normalisation is kept anyway, so an unequal grid could not silently
    # reintroduce it.
    #
    # 2026-09-25: Track A holds the DNN's rank-1 and rank-2 single
    # representations ('base_1_<rep>', 'base_2_<rep>'). Every tally below is
    # kept once per rank -- the hybrid against base_1, and against base_2 --
    # never with the two ranks averaged together.
    _ranks = [f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1)]
    hybrid_wins_by_type = {_b: {} for _b in _ranks}
    hybrid_total_by_type = {_b: {} for _b in _ranks}
    # 2026-09-12: 'hybrid_sim' — the space built from the similarity-objective
    # search — gets its own tally against the single representation alone, so
    # the two hybrid families can be read apart. The two buckets above do not
    # pool it into the baseline.
    hybrid_sim_wins_by_type = {_b: {} for _b in _ranks}
    hybrid_sim_total_by_type = {_b: {} for _b in _ranks}

    for nt, nt_data in results.get('per_noise_type', {}).items():
        best_method = None
        best_score = float('-inf')
        # results['lower_is_better'] is written unconditionally at the top of
        # run() (the dict literal that builds `results`), so a plain read is
        # right and a defaulted False would only hide a real absence.
        # Fixed 2026-09-10 (metric-direction audit).
        #
        # It STILL is not defaulted, and the reason is worth keeping. Which way a
        # metric runs decides which of two scores wins every comparison below. A
        # default would pick a direction for a caller who never said one, and on
        # half the datasets it would pick the wrong one and still return numbers.
        # What changed on 2026-09-16 is only the message: a bare KeyError here
        # says nothing about what to pass.
        if 'lower_is_better' not in results:
            raise KeyError(
                "n13._compute_summary needs results['lower_is_better']: which way "
                "the dataset's metric runs. run() writes it, so a results dict "
                "without it was built by hand. It is not defaulted, because a "
                "guessed direction reverses every win count below and still "
                "returns a full set of numbers.")
        lower = bool(results['lower_is_better'])
        for _b in _ranks:
            hybrid_wins_by_type[_b][nt] = 0
            hybrid_total_by_type[_b][nt] = 0
            hybrid_sim_wins_by_type[_b][nt] = 0
            hybrid_sim_total_by_type[_b][nt] = 0

        for sk, sd in nt_data.get('per_strength', {}).items():
            for space_name, space_data in sd.get('track_a', {}).items():
                for mn, mv in space_data.items():
                    s = mv.get('score', float('nan'))
                    if np.isnan(s):
                        continue
                    if lower:
                        if best_method is None or s < best_score:
                            best_score = s
                            best_method = mn
                    else:
                        if s > best_score:
                            best_score = s
                            best_method = mn

            # Check hybrid advantage (Track A) at this (noise_type, strength),
            # once against each rank of the single representation.
            def _scores(pred):
                out = []
                for space_name, space_data in sd.get('track_a', {}).items():
                    if not pred(space_name):
                        continue
                    for mv in space_data.values():
                        s = mv.get('score', float('nan'))
                        if not np.isnan(s):
                            out.append(s)
                return out
            hybrid_scores = _scores(lambda sp: sp == 'hybrid')
            # 2026-09-12: the similarity-objective space, counted separately.
            hybrid_sim_scores = _scores(lambda sp: sp == 'hybrid_sim')
            for _b in _ranks:
                base_scores = _scores(lambda sp: sp.startswith(f'{_b}_'))
                if base_scores and hybrid_scores:
                    hybrid_total_by_type[_b][nt] += 1
                    b_avg = np.mean(base_scores)
                    h_avg = np.mean(hybrid_scores)
                    if (not lower and h_avg > b_avg) or (lower and h_avg < b_avg):
                        hybrid_wins_by_type[_b][nt] += 1
                if base_scores and hybrid_sim_scores:
                    hybrid_sim_total_by_type[_b][nt] += 1
                    b_avg = np.mean(base_scores)
                    hs_avg = np.mean(hybrid_sim_scores)
                    if (not lower and hs_avg > b_avg) or (lower and hs_avg < b_avg):
                        hybrid_sim_wins_by_type[_b][nt] += 1

        if best_method:
            best_per_dist[nt] = best_method

    # Aggregate two ways:
    #   - "raw" totals (sum of every (noise_type, strength) cell) — kept for
    #     backwards compat with downstream readers.
    #   - "per_type" win rate (mean of within-type win rates) — the
    #     inflation-resistant headline. Each noise type contributes equally
    #     regardless of how many strength variants it has.
    #   Each is kept once per rank of the single representation.
    def _rates(wins, total):
        return {nt: (wins[nt] / total[nt]) for nt in total if total[nt] > 0}

    hybrid_wins = {_b: sum(hybrid_wins_by_type[_b].values()) for _b in _ranks}
    hybrid_total = {_b: sum(hybrid_total_by_type[_b].values()) for _b in _ranks}
    per_type_rates = {_b: _rates(hybrid_wins_by_type[_b],
                                 hybrid_total_by_type[_b]) for _b in _ranks}
    persists_by_type = {
        _b: (np.mean(list(r.values())) > 0.5 if r else None)
        for _b, r in per_type_rates.items()}

    # Same arithmetic for the similarity-objective space, against each single
    # representation alone. Added 2026-09-12 with the pool subset change.
    sim_per_type_rates = {_b: _rates(hybrid_sim_wins_by_type[_b],
                                     hybrid_sim_total_by_type[_b])
                          for _b in _ranks}
    sim_persists_by_type = {
        _b: (np.mean(list(r.values())) > 0.5 if r else None)
        for _b, r in sim_per_type_rates.items()}

    # Does Gaussian winner transfer?
    gaussian_best = best_per_dist.get('gaussian')
    if gaussian_best:
        # The noise types this run actually produced. The module-level
        # NOISE_TYPES is None until run() resolves it per task_type, so
        # iterating it here raised TypeError and killed the whole summary.
        transfers = all(best_per_dist.get(nt) == gaussian_best
                        for nt in best_per_dist)
    else:
        transfers = None

    # Every condition sweeps the SAME axis now -- the noise level -- and delivers
    # the same amount at a given level, so every cell is comparable with every other
    # and they all share one x-axis. Censoring is the one condition that does not
    # belong on it, because its level is the fraction of labels clipped rather than
    # a dose; it is swept on its own axis and n13 does not include it.
    sweep_axis_by_noise_type = {
        nt: 'noise_level' for nt in results.get('per_noise_type', {})
    }

    return convert_numpy({
        'best_method_per_distribution': best_per_dist,
        'gaussian_rank_transfers': transfers,
        'sweep_axis_by_noise_type': sweep_axis_by_noise_type,
        # Headline flag: hybrid wins more than half of noise types on average.
        # One vote per condition, whatever its level grid. Every hybrid field
        # below is a dict keyed 'base_1' / 'base_2': the hybrid against the
        # DNN's rank-1 and rank-2 single representation.
        'hybrid_advantage_persists': persists_by_type,
        # Legacy raw totals (still per-(type, strength), for backwards
        # compatibility with readers that consumed the old fields).
        'hybrid_wins': hybrid_wins,
        'hybrid_total': hybrid_total,
        # Per-type breakdown — the supplementary view.
        'hybrid_wins_by_noise_type': hybrid_wins_by_type,
        'hybrid_total_by_noise_type': hybrid_total_by_type,
        'hybrid_win_rate_by_noise_type': per_type_rates,
        # The similarity-objective space on its own, against the base
        # representation only. Empty when that search family was not on disk
        # for this dataset, so the arm never ran.
        'hybrid_sim_advantage_persists': sim_persists_by_type,
        'hybrid_sim_wins_by_noise_type': hybrid_sim_wins_by_type,
        'hybrid_sim_total_by_noise_type': hybrid_sim_total_by_type,
        'hybrid_sim_win_rate_by_noise_type': sim_per_type_rates,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('N13: Noise Distributions')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n13_noise_distributions',
                            args.dataset)
