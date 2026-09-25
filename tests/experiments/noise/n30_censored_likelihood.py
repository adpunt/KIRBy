#!/usr/bin/env python3
"""N30: when labels are clipped at an assay limit, lift them, drop them, or fit a censored likelihood?

Paper: Noise.

THE QUESTION. Under `censoring` every training label above a cut-point is
recorded as the cut-point. The molecules sitting at the limit are known exactly,
so no detector is involved. What is left is what to DO with them:

  lift them          raise every label at the limit by an estimated amount
  drop them          remove them from the training set
  winsorise          pull every label toward an out-of-fold prediction, but only
                     by the part of its distance beyond a threshold
  change the loss    keep the clipped labels, and tell the model they are lower
                     bounds (a censored likelihood), or fit a robust loss

The lifts were measured once already, outside the repository, in
`/Volumes/seagate/local_noise_run/loop_headroom/censoring_probe.py` and its
FINDINGS.md sections 6 and 11-14 (2026-09-23, herg_ki and caco2, base
representation, 3 folds). This file ports them into the suite, adds the hybrid
spaces, and adds the censored likelihood the research report of 2026-09-24
proposed as its B4, and the residual winsorisation it proposed as its B5.

THE SINGLE-REPRESENTATION COMPARISON is each downstream model's top 2 base
representations by that model's own score in the stored search -- spaces
base_1 and base_2 (_shared.noise_base_reps_by_model, owner's decision
2026-09-25). Which representations those are is written to
`base_reps_by_model`. A model the pool holds no search for has neither.

WHAT THE INJECTOR DOES, READ IN THIS SESSION. `noiseInject/core.py:736-779`,
reached through `noise_helpers.inject_train_labels`:

  - side 'upper' (`core.py:73`): values ABOVE the limit are replaced by it.
  - the limit is the (1 - level) quantile of the FULL label column, so the
    level is the fraction of labels clipped, not a dose in label units
    (`kirby/noise_spec.py:219`).
  - the limit comes back on the provenance row as `censoring_limit`.
  - the draw does not enter: the clip is `np.minimum(y, limit)`. MEASURED on
    500 labels at level 0.25: noise seeds 0, 1 and 2 give bit-identical
    columns. So this file runs ONE draw per level by default, and says so on
    the result file. Asking for more draws with --draws repeats the same fit.

WHICH ROWS COUNT AS CLIPPED. The rows whose label equals the limit, which is what
a practitioner holding only the recorded labels can see. That is not always the
injector's mask: on a coarsely recorded assay some labels sat ON the limit before
the clip and were never moved (`core.py:782-815`). Both counts are written for
every pass, `molecules_at_the_limit` and `molecules_the_clip_moved`.

THE ARMS. Every arm is scored on the same held-out molecules against their
untouched labels. `clipped` is the arm every other one is paired against.

  clean                      fitted on the training labels before the clip. The
                             most a label treatment can recover.
  clipped                    fitted on the clipped labels as they are.
  constant_lift              every label at the limit gets the SAME lift: the
                             mean over those rows of sd * phi(a) / (1 - Phi(a)),
                             a = (limit - p) / sd.
  per_molecule_lift          each label at the limit gets its own lift,
                             limit + sd * phi(a) / (1 - Phi(a)).
  per_molecule_lift_shuffled/seed<k>
                             the same lifts dealt to different molecules at the
                             limit. It separates "the labels were raised" from
                             "each molecule got its right amount".
  truncated_normal_mean      each label at the limit set to p + sd * phi(a) /
                             (1 - Phi(a)), floored at the limit. See the note
                             below on why this is not the same as
                             per_molecule_lift.
  drop_clipped               the rows at the limit removed.
  winsor_c<c>                every training label moved to
                             p + clip(y - p, -c * s, c * s), where
                             s = 1.4826 * median |y - p|. A label within c * s of
                             the prediction is not moved at all.

In every arm above, p is a group-aware out-of-fold LightGBM prediction
(`kirby.utils.noisy_labels.oof_predict`), computed on this fold's training
molecules only, in the feature space the arm is fitted in. `sd` is the standard
deviation of y - p over the rows NOT at the limit. FINDINGS.md section 6: taking it
over every row mixes in the distance the clip moved a label.

⛔ TWO LIFT FORMULAS, AND THE PROBE USED THE FIRST. For a normal with mean p and
spread sd, the mean of what lies above the limit is p + sd * phi(a) / (1 - Phi(a)).
`censoring_probe.py` computes limit + sd * phi(a) / (1 - Phi(a)), which is larger
by (limit - p). Every number in FINDINGS.md sections 6 and 11-14 is from the
probe's version, so `constant_lift` and `per_molecule_lift` keep it, so that they
reproduce those numbers. `truncated_normal_mean` is the other version.

THE MODEL SWAPS. Three arms change the model, not the labels. Each one is paired
against the model it replaces, fitted on the same clipped labels in the same
feature matrix.

  xgboost_aft      XGBoost with objective survival:aft and a normal error
                   distribution. A row at the limit is given the interval
                   [limit, +inf); every other row an interval of zero width at
                   its label. AFT models log(time), so each label enters as
                   time = exp(y - centre), and a prediction comes back as
                   log(prediction) + centre. With that map the normal
                   applies to the label itself, not to a shifted copy of it.
                   MEASURED in this session on toy data: xgboost 2.1.4's
                   predict() returns exp(margin), so log() is the inverse. The
                   error scale the likelihood uses is `sd` above, read off
                   the rows not at the limit. Paired against xgboost on
                   `clipped` and against xgboost on `constant_lift`.
  rf_l1            a random forest with absolute-error splits, paired against
                   rf on `clipped`.
  lightgbm_huber   LightGBM with the Huber objective, paired against lightgbm on
                   `clipped`.

rf_l1 and lightgbm_huber are tier 'extra' in `find_best_hybrid.DEFAULT_MODELS`
and no stored search holds a recipe for them. So in a hybrid space each one is
fitted in the feature matrix built for the model it replaces: rf's for rf_l1,
lightgbm's for lightgbm_huber. When that model has no matrix in the space, the
swap is not run there, and the reason is recorded. KIRBy changes the input, not
the model, so the three swaps are comparators for the label arms, not proposals.

THE RECOVERY SHARE. Per (level, space, model), over the passes all three scored:

    (clipped - arm) / (clipped - clean)   where lower wins
    (arm - clipped) / (clean - clipped)   where higher wins

It is refused, with the reason written, where the clean-to-clipped gap is under 2%
of the clean score. That is the floor n07 and n11 use. For a model swap the
denominator is the replaced model's own clean-to-clipped gap.

WHAT IS NOT BUILT HERE. adme_hlm has 958 of its 3,087 labels at exactly its
minimum, and adme_rlm has 346 of 3,054. Whether that is a real assay floor is
the owner's decision and is not settled. So no real-floor path exists in this
file; it runs the injected condition only.

WHERE THE FEATURES COME FROM. The recipe for each downstream model is read from
the precomputed pool, which was searched on the clean labels. Each fold then
rebuilds the hybrid from that fold's CLIPPED training labels, once, and every arm
shares it. So `clean` is the clean labels in clipped-label features, and no label
treatment reaches the feature choice.

ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 2-3). Every label arm
writes one D-versus-D' row per pass against `clipped`, with `clean` as
score_clean; the model swaps change the model, not the data, and write none.
The accumulators are checkpointed per pass; the caches of plan section 3 are
used. controls.py's twins run with KIRBY_NOISE_CONTROLS=1.

Usage:
    python n30_censored_likelihood.py --dataset herg_ki --mode full
    python n30_censored_likelihood.py --dataset caco2 --mode full --levels 0.1,0.25
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time
import zlib

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, hybrid_for_fold,
    pool_recipe_per_downstream_model, group_models_by_recipe,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.utils.noisy_labels import oof_predict
# The D-versus-D' rows, the per-pass checkpoint, the caches and the controls
# (2026-09-25, plan sections 2-3). They live in n23 for the reason given at
# the top of its shared block.
from experiments.noise import noise_cache as nc
from experiments.noise.controls import controls_enabled
from experiments.noise.n23_uncertainty_cleaning import (
    RowSink, PassCheckpoint, ensure_dprime_models, fit_score_cached,
    control_twins, as_cleanup_dict, fit_cleaned, cached_base_cell,
    cached_hybrid_cell, NO_NEIGHBOURS,
)

# The only condition this file is defined for. Every arm reads the limit.
CENSORING_CONDITION = 'censoring'
DEFAULT_NOISE_CONDITION = CENSORING_CONDITION

# ONE draw. The clip is np.minimum(y, limit) and no seed enters it, so draws 0, 1
# and 2 give bit-identical labels (measured, see the docstring). Three draws
# would be three copies of one fit.
NOISE_DRAWS = (0,)

# Winsorisation thresholds, in units of 1.4826 * median |y - p|. The report's
# B5 range, 1.0 to 1.5.
WINSOR_C = (1.0, 1.25, 1.5)

# Replicates of the shuffled-lift control. Each is reported on its own; no mean
# across them is written.
SHUFFLE_SEEDS = (0, 1, 2)

# Inner folds for the out-of-fold prediction p. Same as n19.
OOF_SPLITS = 5

# The recovery share is refused below this share of the clean score. n07:294,
# n11:272.
GAP_MIN_FRAC = 0.02

# The model swaps, each with the model it replaces.
COMPARATORS = {'rf_l1': 'rf', 'lightgbm_huber': 'lightgbm'}
AFT_MODEL = 'xgboost_aft'
AFT_PARENT = 'xgboost'

# XGBoost settings for the censored fit, matched to
# find_best_hybrid._get_xgb_regressor (n_estimators=100, colsample_bytree=0.5,
# random_state=42) and to XGBRegressor's own defaults for the rest.
AFT_PARAMS = {'eta': 0.3, 'max_depth': 6, 'colsample_bytree': 0.5,
              'seed': 42, 'verbosity': 0, 'nthread': 1}
AFT_ROUNDS = 100

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

# Arms that need rows at the limit. At level 0.0 no row is at a limit and these
# are undefined, not zero.
NEEDS_A_LIMIT = ('constant_lift', 'per_molecule_lift', 'truncated_normal_mean',
                 'drop_clipped')


def build_arm_names(winsor_c, shuffle_seeds):
    """Every label arm, in the order they are fitted."""
    out = ['clean', 'clipped', 'constant_lift', 'per_molecule_lift']
    out += [f'per_molecule_lift_shuffled/seed{s}' for s in shuffle_seeds]
    out += ['truncated_normal_mean', 'drop_clipped']
    out += [f'winsor_c{c:g}' for c in winsor_c]
    return out


def rows_at_limit(y_clipped, limit, tol=1e-12):
    """The training rows whose recorded label equals the limit.

    None for the limit means nothing was clipped, and no row is returned.
    """
    y = np.asarray(y_clipped, dtype=float)
    if limit is None or not np.isfinite(limit):
        return np.zeros(len(y), dtype=bool)
    return y >= float(limit) - tol


def residual_sd(y, p, at):
    """Standard deviation of y - p over the rows NOT at the limit.

    FINDINGS.md section 6: over every row it mixes in how far the clip moved a label.
    """
    r = (np.asarray(y, dtype=float) - np.asarray(p, dtype=float))[~at]
    if r.size < 2:
        return float('nan')
    return float(np.std(r))


def mills_ratio(limit, p, sd):
    """phi(a) / (1 - Phi(a)) with a = (limit - p) / sd, per row."""
    a = (float(limit) - np.asarray(p, dtype=float)) / max(float(sd), 1e-9)
    return norm.pdf(a) / np.clip(1.0 - norm.cdf(a), 1e-9, None)


def lift_columns(y_clipped, p, at, limit, sd, shuffle_seeds, seed_key):
    """The five lift arms, each a full training label column.

    Ported from censoring_probe.py:268-316 (`impute_tobit`,
    `impute_constant_mean_lift`, `impute_tobit_shuffled<k>`), where the lift is
    `limit + sd * mills`. `truncated_normal_mean` is `p + sd * mills` floored
    at the limit, which is the mean of a normal truncated at the limit.

    `seed_key` makes each pass's shuffle its own draw; the same seed on the same
    pass gives the same permutation.

    Returns ({arm: column}, {'mean_lift': ..., 'spread_of_lift': ...}).
    """
    y = np.asarray(y_clipped, dtype=float)
    m = mills_ratio(limit, p, sd)
    per_mol = y.copy()
    per_mol[at] = np.maximum(limit, limit + sd * m[at])
    lift = per_mol[at] - limit

    const = y.copy()
    const[at] = limit + float(np.mean(lift)) if lift.size else limit

    trunc = y.copy()
    trunc[at] = np.maximum(limit, np.asarray(p, dtype=float)[at] + sd * m[at])

    cols = {'constant_lift': const, 'per_molecule_lift': per_mol,
            'truncated_normal_mean': trunc}
    for s in shuffle_seeds:
        perm = np.random.default_rng([int(s)] + list(seed_key)).permutation(
            lift.size)
        sh = y.copy()
        sh[at] = limit + lift[perm]
        cols[f'per_molecule_lift_shuffled/seed{s}'] = sh
    info = {'mean_lift_label_units': (float(np.mean(lift)) if lift.size
                                      else float('nan')),
            'spread_of_lift_label_units': (float(np.std(lift)) if lift.size
                                           else float('nan'))}
    return cols, info


def winsorise(y, p, c):
    """y moved to p + clip(y - p, -c*s, c*s), s = 1.4826 * median |y - p|.

    Returns (column, s, how many labels moved).
    """
    y = np.asarray(y, dtype=float)
    r = y - np.asarray(p, dtype=float)
    s = 1.4826 * float(np.median(np.abs(r)))
    if not np.isfinite(s) or s <= 0:
        return y.copy(), s, 0
    r_c = np.clip(r, -c * s, c * s)
    out = np.asarray(p, dtype=float) + r_c
    moved = int(np.sum(np.abs(r_c - r) > 1e-12))
    # A label that did not move keeps its exact value, not p + (y - p).
    out[np.abs(r_c - r) <= 1e-12] = y[np.abs(r_c - r) <= 1e-12]
    return out, s, moved


def fit_aft_predict(X_tr, y_clipped, at, X_va, sd, params=None, rounds=None):
    """XGBoost survival:aft on interval labels, predictions on the label scale.

    Each label enters as time = exp(y - centre). A row at the limit is the
    interval [exp(limit - centre), +inf); every other row a zero-width
    interval at its own time. AFT's normal is on log(time), which is then
    y - centre, so the error model applies to the label itself.

    Returns predictions for X_va in label units.
    """
    import xgboost as xgb
    y = np.asarray(y_clipped, dtype=float)
    centre = float(np.median(y))
    t = np.exp(y - centre)
    upper = t.copy()
    upper[at] = np.inf
    dtr = xgb.DMatrix(np.asarray(X_tr, dtype=float))
    dtr.set_float_info('label_lower_bound', t)
    dtr.set_float_info('label_upper_bound', upper)
    p = dict(AFT_PARAMS if params is None else params)
    p.update({'objective': 'survival:aft',
              'aft_loss_distribution': 'normal',
              'aft_loss_distribution_scale': float(max(sd, 1e-3))})
    booster = xgb.train(p, dtr, int(AFT_ROUNDS if rounds is None else rounds))
    pred_time = booster.predict(xgb.DMatrix(np.asarray(X_va, dtype=float)))
    return np.log(np.clip(pred_time, 1e-300, None)) + centre


def recovery_share(clean, clipped, arm, lower_is_better,
                   gap_min_frac=GAP_MIN_FRAC):
    """Share of the clean-to-clipped gap an arm recovers, or (None, why)."""
    if not all(np.isfinite(v) for v in (clean, clipped, arm)):
        return None, 'a score is missing'
    gap = (clipped - clean) if lower_is_better else (clean - clipped)
    floor = gap_min_frac * abs(clean)
    if gap < floor:
        return None, (f'the clean-to-clipped gap is {gap:+.6f}, under '
                      f'{gap_min_frac:g} of the clean score ({floor:.6f})')
    gain = (clipped - arm) if lower_is_better else (arm - clipped)
    return float(gain / gap), None


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, metric_name,
               keep=None):
    """One fit, one held-out score, or (None, why)."""
    try:
        m = factory()
        if keep is None:
            m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr[keep], np.asarray(y_tr)[keep])
        return float(metric_fn(
            y_va, _predict_for_metric(m, X_va, metric_name))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _levels_and_draws(dataset_name, levels=None, draws=None):
    """(level, draw) pairs. One draw, see NOISE_DRAWS."""
    levels = (noise_levels(dataset_name, strategy=CENSORING_CONDITION,
                           is_clf=False) if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    return [(float(l), int(d)) for l in levels for d in draws]


def run(data, config, output_dir, dataset_name, **kwargs):
    """N30: lift, drop, winsorise, or fit a censored likelihood on clipped labels."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N30: CLIPPED LABELS -- LIFT, DROP, OR CENSORED LIKELIHOOD")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    if condition != CENSORING_CONDITION:
        raise ValueError(
            f"n30: condition {condition!r} clips nothing. Every arm here reads "
            f"the limit a clipped label sits at, so n30 runs under "
            f"{CENSORING_CONDITION!r} only.")

    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        # A skip with its reason, not a raise: run_all reaches n30 on every
        # dataset, and n19, n26, n27 and n29 skip a class label the same way.
        print(f"  SKIPPED: {dataset_name} is {task_type}. A class label has no "
              f"value above a limit to lift, so censoring is regression only.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    winsor_c = [float(c) for c in (kwargs.get('winsor_c') or WINSOR_C)]
    shuffle_seeds = [int(s) for s in
                     (kwargs.get('shuffle_seeds') or SHUFFLE_SEEDS)]
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    arm_names = build_arm_names(winsor_c, shuffle_seeds)

    # ⛔ config FIRST, then the kwarg. run_all passes the model list in
    # config['models']; see n24:418-422.
    # rf and svm are added when the caller left them out: every D-versus-D'
    # row needs them (ensure_dprime_models, owner decision 2026-09-25).
    downstream_models = ensure_dprime_models(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n30: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")
    # The swaps run only where the model they replace is in the run.
    comparators = {c: p for c, p in COMPARATORS.items()
                   if p in downstream_models}
    comp_factories = get_model_factories(
        list(comparators), task_type=task_type, n_samples=len(labels))
    comparators = {c: p for c, p in comparators.items() if c in comp_factories}
    run_aft = (AFT_PARENT in downstream_models
               and not kwargs.get('skip_aft', False))
    swaps = dict(comparators)
    if run_aft:
        swaps[AFT_MODEL] = AFT_PARENT

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')
    noise_groups = noise_groups_for(data)[0]

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': noise_dose_units(dataset_name),
        'level_means': 'the fraction of the full label column clipped to the limit',
        'noise_draws': list(NOISE_DRAWS if kwargs.get('draws') is None
                            else kwargs['draws']),
        'noise_draws_note': (
            'The clip is np.minimum(y, limit) and no seed enters it, so noise '
            'seeds 0, 1 and 2 give bit-identical labels. One draw per level by '
            'default. More draws repeat the same fit.'),
        'models': downstream_models,
        'model_swaps': swaps,
        'model_swaps_note': (
            'Each swap is fitted on the clipped labels in the feature matrix '
            'built for the model it replaces, and paired against that model on '
            'the clipped labels. rf_l1 and lightgbm_huber have no recipe of '
            'their own in the pool. xgboost_aft enters each label as '
            'exp(label - centre) and returns log(prediction) + centre.'),
        'arms': arm_names,
        'arm_to_beat': 'clipped',
        'winsor_c': winsor_c,
        'shuffle_seeds': shuffle_seeds,
        'oof_splits': oof_splits,
        'gap_min_frac': GAP_MIN_FRAC,
        'aft_params': dict(AFT_PARAMS, rounds=AFT_ROUNDS,
                           distribution='normal',
                           scale='sd of y - p over the rows not at the limit'),
        'lift_formula_note': (
            'constant_lift and per_molecule_lift use limit + sd * mills, as '
            'censoring_probe.py did, so they reproduce FINDINGS.md sections 6 '
            'and 11-14. truncated_normal_mean uses p + sd * mills floored at '
            'the limit, the mean of a normal truncated at the limit. The two '
            'differ by limit - p on each row.'),
        'clipped_rows_rule': (
            'the training rows whose recorded label equals the limit. '
            'molecules_the_clip_moved is the injector\'s own count, which '
            'leaves out labels that sat on the limit before the clip.'),
        'features_are_fixed_note': (
            'Each feature space is built once per (level, fold) from the '
            'CLIPPED training labels and shared by every arm, clean included. '
            'The recipe was searched on clean labels. No label treatment '
            'reaches the feature choice.'),
        'claim_scope': (
            'The censoring is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
        'per_level': {},
        'clip_per_pass': {},
    }

    splits = list(cv_split(groups, n_folds, next(iter(reps.values())), labels,
                           mode=mode))

    # ── The recipe per downstream model, n23:733-904's pattern ──
    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    _raised = [d['lookup_error'] for d in no_recipe.values()
               if d['lookup_error']]
    results['hybrid_lookup_error'] = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}

    recipes_by_model, recipe_for_space_by_model = {}, {}
    hybrid_unavailable = {}
    for _m in downstream_models:
        _m_recipes, _m_unavail = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipes_by_model[_m] = _m_recipes
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint']
                                         for r in _m_recipes}
        for _sp, _why in _m_unavail.items():
            hybrid_unavailable.setdefault(_sp, {})[_m] = _why
    results['hybrid_spaces_unavailable'] = hybrid_unavailable

    _ordered = []
    for _m in downstream_models:
        for r in recipes_by_model.get(_m, ()):
            if r['space'] not in _ordered:
                _ordered.append(r['space'])

    # ── The single-representation comparison, per downstream model ──
    if base_reps_override is not None:
        base_by_model = {m: list(base_reps_override[m])
                         for m in downstream_models if base_reps_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in downstream_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: d['search'] for m, d in by_model.items()}, downstream_models,
            reps=reps, dataset_name=dataset_name)
    results['base_reps_by_model'] = base_by_model
    results['base_reps_missing'] = base_missing
    base_spaces = tuple(
        f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1)
        if base_rank_groups(base_by_model, _k, downstream_models))

    spaces = base_spaces + tuple(_ordered)
    results['hybrid_available'] = bool(_ordered)
    if _ordered:
        results['hybrid_recipe_by_model'] = convert_numpy({
            m: {r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in rs}
            for m, rs in recipes_by_model.items() if rs})
        for m, rs in recipes_by_model.items():
            for r in rs:
                print(f"  {m} / {r['space']}: family {r['objective']!r}, "
                      f"budget {r['budget']} features")
    else:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']}; the "
             f"pool was never read. Every arm ran in the single "
             f"representations only.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Every arm ran in the single '
             f'representations only.'))
        print(f"  no hybrid recipe; single representations only")
    results['spaces_run'] = list(spaces)

    passes = _levels_and_draws(dataset_name, levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} molecules, {len(splits)} folds, models "
          f"{downstream_models}, swaps {swaps}")
    print(f"  levels {results['noise_levels']} (fraction clipped), "
          f"{len(passes)} (level, draw) passes, {len(arm_names)} label arms")
    print(f"  spaces {list(spaces)}")

    # acc[level][space][arm][model][pass] = score
    acc = {}
    errors = {}
    clip_acc = {}
    t0 = time.time()

    # ── Rows and the per-pass checkpoint (2026-09-25) ──
    # D is `clipped`, the labels as recorded; every label arm but `clean` is
    # a D'; `clean` is score_clean. The model swaps change the model, not the
    # training data, so they write no row. The per_molecule_lift_shuffled
    # arms are this file's own control and are written as D' rows like the
    # others.
    _metric = (metric_name, metric_fn, lower_is_better)
    rows = RowSink(output_dir, 'n30_censored_likelihood', dataset_name,
                   experiment='n30', metric=_metric,
                   condition=kwargs.get('noise_condition'), data=data,
                   config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n30_censored_likelihood', dataset_name,
        condition=kwargs.get('noise_condition'), data=data, config=config,
        source_file=__file__,
        key={'condition': condition, 'winsor_c': winsor_c,
             'shuffle_seeds': shuffle_seeds, 'oof_splits': oof_splits,
             'passes': passes, 'models': list(downstream_models),
             'swaps': swaps, 'base_reps': base_by_model,
             'n_splits': n_folds})
    done_passes = set()
    _state = ckpt.load()
    if _state:
        acc, errors, clip_acc = (_state['acc'], _state['errors'],
                                 _state['clip_acc'])
        done_passes = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_passes)} (level, draw, "
              f"fold) passes already done")

    for level, draw in passes:
        lvl_key = f'{level:g}'
        spec = noise_spec_for(dataset_name, condition, level,
                              noise_seed=draw, is_clf=False)
        cell = acc.setdefault(lvl_key, {})
        print(f"\n  --- censoring / fraction {lvl_key} / draw {draw} ---")

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if pass_key in done_passes:
                print(f"    {pass_key}: on disk in the checkpoint")
                continue
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = groups[train_idx] if groups is not None else None
            try:
                y0, cmask, prov = inject_train_labels(
                    labels, spec, train_idx=train_idx, is_clf=False,
                    groups=noise_groups, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            limit = prov.get('censoring_limit') if level > 0 else None
            limit = (float(limit) if limit is not None
                     and np.isfinite(float(limit)) else None)
            at = rows_at_limit(y0, limit)
            n_tr = int(len(y0))
            clip_rec = {
                'pass': pass_key, 'limit': limit,
                'training_molecules': n_tr,
                'molecules_at_the_limit': int(at.sum()),
                'molecules_the_clip_moved': int(np.sum(cmask)),
                'how_far_above_the_limit_the_true_labels_sit_label_units': (
                    float(np.mean(y_clean_tr[at]) - limit)
                    if at.any() else float('nan')),
                'per_space': {},
            }

            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}

            # {space: [(X_tr, X_va, models on it)]}, n23:960-1010's shape.
            # base_k holds one matrix per distinct representation at rank k,
            # shared by the models that rank it there.
            fold_spaces = {}
            for _k in range(1, NOISE_BASE_REP_COUNT + 1):
                _bsp = f'base_{_k}'
                if _bsp not in spaces:
                    continue
                _bcells = []
                for _rn, _ms in base_rank_groups(base_by_model, _k,
                                                 downstream_models).items():
                    # noise_cache.cached_scaled since 2026-09-25.
                    _Xt, _Xv = cached_base_cell(dataset_name, reps[_rn],
                                                train_idx, val_idx)
                    _bcells.append((_Xt, _Xv, list(_ms)))
                fold_spaces[_bsp] = _bcells
            for _sp in spaces:
                if _sp.startswith('base_'):
                    continue
                _cells = []
                for _bp_m, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    # noise_cache.cached_hybrid_for_fold since 2026-09-25:
                    # the same build for the same draw is shared with n23,
                    # n24, n25, n31-n35, n38, n40 and n41 (plan section 3).
                    X_tr_h, X_va_h = cached_hybrid_cell(
                        dataset_name, train_reps, val_reps, y0, _bp_m, g_tr,
                        None if _shareable else _mnames[0],
                        None if _shareable
                        else model_factories[_mnames[0]](),
                        _metric, train_idx)
                    if X_tr_h is None:
                        errors[f'{lvl_key}|{_sp}/{_mnames[0]}|f{fold_i}|'
                               f'build'] = 'the hybrid could not be built'
                        continue
                    sh = StandardScaler()
                    _cells.append((sh.fit_transform(X_tr_h),
                                   sh.transform(X_va_h), list(_mnames)))
                if _cells:
                    fold_spaces[_sp] = _cells

            for space in spaces:
                for X_tr_s, X_va_s, cell_models in fold_spaces.get(space, ()):
                    _ekey = (space if cell_models == list(downstream_models)
                             else f'{space}/{cell_models[0]}')
                    space_cell = cell.setdefault(space, {})

                    # p, the out-of-fold prediction every lift and winsor arm
                    # reads, on this matrix and these clipped labels.
                    try:
                        # noise_cache.cached_oof_predict since 2026-09-25:
                        # the same call (seed 0), stored.
                        p = nc.cached_oof_predict(dataset_name, X_tr_s, y0,
                                                  g_tr, n_splits=oof_splits)
                    except Exception as exc:
                        errors[f'{lvl_key}|{_ekey}|f{fold_i}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        p = None

                    cols = {'clean': (y_clean_tr, None),
                            'clipped': (y0, None)}
                    sp_rec = {}
                    sd = float('nan')
                    if p is not None:
                        sd = residual_sd(y0, p, at)
                        sp_rec['residual_sd_rows_not_at_limit'] = sd
                        if at.any() and np.isfinite(sd):
                            lc, linfo = lift_columns(
                                y0, p, at, limit, sd, shuffle_seeds,
                                seed_key=(fold_i, int(round(level * 1000)),
                                          draw))
                            sp_rec.update(linfo)
                            for a, col in lc.items():
                                cols[a] = (col, None)
                            cols['drop_clipped'] = (y0, ~at)
                        for c in winsor_c:
                            col, s, moved = winsorise(y0, p, c)
                            cols[f'winsor_c{c:g}'] = (col, None)
                            sp_rec[f'winsor_c{c:g}'] = {
                                'threshold_label_units': float(c * s),
                                'labels_moved': moved}
                    clip_rec['per_space'][_ekey] = sp_rec

                    for arm in arm_names:
                        if arm not in cols:
                            continue
                        y_use, keep = cols[arm]
                        for mname in cell_models:
                            if arm in ('clean', 'clipped'):
                                # The clean-label fit and the no-cleaning fit
                                # on D, through noise_cache for a seeded model
                                # (plan section 3). keep is None for both.
                                s, err = fit_score_cached(
                                    dataset_name, mname,
                                    model_factories[mname], X_tr_s, y_use,
                                    X_va_s, y_val, metric_fn, metric_name)
                            else:
                                s, err = _fit_score(
                                    model_factories[mname], X_tr_s, y_use,
                                    X_va_s, y_val, metric_fn, metric_name,
                                    keep=keep)
                            if s is None:
                                errors[f'{lvl_key}|{_ekey}|{arm}|{mname}|'
                                       f'f{fold_i}'] = err
                                continue
                            (space_cell.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = s

                    # ── The D-versus-D' rows and the controls (2026-09-25) ──
                    for mname in cell_models:
                        sD = (space_cell.get('clipped') or {}).get(
                            mname, {}).get(pass_key)
                        sC = (space_cell.get('clean') or {}).get(
                            mname, {}).get(pass_key)
                        for arm in arm_names:
                            if arm in ('clean', 'clipped') or arm not in cols:
                                continue
                            sP = (space_cell.get(arm) or {}).get(
                                mname, {}).get(pass_key)
                            if sP is None:
                                continue
                            rows.add(method=arm.split('/')[0],
                                     method_setting=arm,
                                     similarity_setting=NO_NEIGHBOURS,
                                     model_input=space, model=mname,
                                     noise_condition=condition,
                                     level=float(level), draw=int(draw),
                                     fold=fold_i, score_D=sD, score_Dprime=sP,
                                     score_clean=sC)
                    # controls.py's twins of every label arm but the shuffled
                    # lifts, which are this file's own control, when
                    # KIRBY_NOISE_CONTROLS=1.
                    if controls_enabled():
                        for arm in arm_names:
                            if (arm in ('clean', 'clipped') or arm not in cols
                                    or arm.startswith(
                                        'per_molecule_lift_shuffled')):
                                continue
                            y_use, keep = cols[arm]
                            _cd = (as_cleanup_dict(X_tr_s, y0, keep=keep)
                                   if keep is not None else
                                   as_cleanup_dict(X_tr_s, y0, labels=y_use))
                            _seed = (int(draw), int(fold_i), int(zlib.crc32(
                                f'{lvl_key}|{_ekey}|{arm}'.encode())
                                & 0x7FFFFFFF))
                            for cname, tw, why in control_twins(
                                    _cd, X_tr_s, y0, _seed):
                                if tw is None:
                                    errors[f'{lvl_key}|{_ekey}|{arm}|{cname}|'
                                           f'f{fold_i}'] = why
                                    continue
                                for mname in cell_models:
                                    sP, err = fit_cleaned(
                                        model_factories[mname], tw, X_va_s,
                                        y_val, metric_fn, metric_name)
                                    rows.add(
                                        method=arm.split('/')[0],
                                        method_setting=f'{arm}|control={cname}',
                                        similarity_setting=NO_NEIGHBOURS,
                                        model_input=space, model=mname,
                                        noise_condition=condition,
                                        level=float(level), draw=int(draw),
                                        fold=fold_i,
                                        score_D=(space_cell.get('clipped')
                                                 or {}).get(mname, {}).get(
                                                     pass_key),
                                        score_Dprime=sP,
                                        score_clean=(space_cell.get('clean')
                                                     or {}).get(mname, {}).get(
                                                         pass_key))

                    # ── The model swaps, on the clipped labels ──
                    for swap, parent in swaps.items():
                        if parent not in cell_models:
                            continue
                        if swap == AFT_MODEL:
                            if not np.isfinite(sd):
                                errors[f'{lvl_key}|{_ekey}|{swap}|f{fold_i}'] = (
                                    'no residual sd, so no AFT error scale')
                                continue
                            try:
                                pred = fit_aft_predict(X_tr_s, y0, at,
                                                       X_va_s, sd)
                                s = float(metric_fn(y_val, pred))
                            except Exception as exc:
                                errors[f'{lvl_key}|{_ekey}|{swap}|f{fold_i}'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                        else:
                            s, err = _fit_score(
                                comp_factories[swap], X_tr_s, y0, X_va_s,
                                y_val, metric_fn, metric_name)
                            if s is None:
                                errors[f'{lvl_key}|{_ekey}|{swap}|'
                                       f'f{fold_i}'] = err
                                continue
                        (space_cell.setdefault('clipped', {})
                         .setdefault(swap, {})[pass_key]) = s

            clip_acc.setdefault(lvl_key, []).append(clip_rec)
            # The pass is finished: its rows to disk, then the accumulators.
            rows.flush()
            done_passes.add(pass_key)
            ckpt.save({'acc': acc, 'errors': errors, 'clip_acc': clip_acc,
                       'done': sorted(done_passes)})
            print(f"    fraction {lvl_key} fold {fold_i}: "
                  f"{clip_rec['molecules_at_the_limit']} of {n_tr} training "
                  f"molecules at the limit {limit} "
                  f"[{time.time() - t0:.0f}s]")

    results['per_level'] = convert_numpy(acc)
    results['clip_per_pass'] = convert_numpy(clip_acc)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, arm_names, downstream_models, swaps, metric_name,
        lower_is_better))
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    ckpt.clear()
    return results


def _paired(base, got, lower_is_better):
    """One arm against another, on the passes both scored. Positive means better.

    n24:981-1006. The direction is read off the dataset.
    """
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_passes_paired': int(gains.size),
        'passes_won': int((gains > tol).sum()),
        'passes_lost': int((gains < -tol).sum()),
        'passes_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_pass_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'pass_gains': [float(x) for x in gains],
    }


def _mean_on(d, keys):
    vals = [d[k] for k in keys if k in d and np.isfinite(d[k])]
    return float(np.mean(vals)) if vals else float('nan')


def _share(clean_d, clipped_d, arm_d, lower_is_better):
    """Recovery share over the passes all three scored."""
    keys = sorted(set(clean_d) & set(clipped_d) & set(arm_d))
    if not keys:
        return {'recovery_share': None,
                'recovery_share_refused_because': 'no pass scored all three',
                'n_passes': 0}
    c, k, a = (_mean_on(clean_d, keys), _mean_on(clipped_d, keys),
               _mean_on(arm_d, keys))
    share, why = recovery_share(c, k, a, lower_is_better)
    return {'recovery_share': share,
            'recovery_share_refused_because': why,
            'clean_mean': c, 'clipped_mean': k, 'arm_mean': a,
            'n_passes': len(keys)}


def _summarise(acc, arm_names, models, swaps, metric_name, lower_is_better):
    """Every label arm against clipped, and every swap against its parent.

    The headline is a count of passes won, paired. A pass is one (draw, fold).
    """
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against the clipped labels, paired per '
                        'pass; a pass is one (draw, fold)'),
           'per_level': {}}
    for lvl, by_space in acc.items():
        lrow = {}
        for space, by_arm in by_space.items():
            clipped = by_arm.get('clipped') or {}
            clean = by_arm.get('clean') or {}
            srow = {'per_model': {}, 'model_swaps': {}}
            for m in models:
                b = clipped.get(m) or {}
                if not b:
                    continue
                row = {'clipped_mean': _mean_on(b, list(b)),
                       'per_arm': {}}
                for arm in arm_names:
                    if arm == 'clipped':
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    entry = {'vs_clipped': _paired(b, g, lower_is_better)}
                    if arm != 'clean':
                        entry.update(_share(clean.get(m) or {}, b, g,
                                            lower_is_better))
                    row['per_arm'][arm] = entry
                srow['per_model'][m] = row
            for swap, parent in swaps.items():
                g = clipped.get(swap) or {}
                b = clipped.get(parent) or {}
                if not g or not b:
                    continue
                entry = {'replaces': parent,
                         'vs_parent_on_clipped': _paired(b, g,
                                                         lower_is_better)}
                entry.update(_share(clean.get(parent) or {}, b, g,
                                    lower_is_better))
                if swap == AFT_MODEL:
                    lifted = (by_arm.get('constant_lift') or {}).get(parent) or {}
                    if lifted:
                        entry['vs_parent_on_constant_lift'] = _paired(
                            lifted, g, lower_is_better)
                srow['model_swaps'][swap] = entry
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass against the clipped labels)")
    for lvl, lrow in s['per_level'].items():
        print(f"\n  fraction clipped {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow['per_model'].items():
                print(f"    {space}/{m}: clipped {row['clipped_mean']:.4f}")
                for arm, e in row['per_arm'].items():
                    v = e['vs_clipped']
                    sh = e.get('recovery_share')
                    tail = (f", recovers {sh:.2f} of the clean-to-clipped gap"
                            if sh is not None else '')
                    print(f"      {arm:34s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} (spread "
                          f"{v['spread_over_pass_gains']:.4f}){tail}")
            for swap, e in srow['model_swaps'].items():
                v = e['vs_parent_on_clipped']
                print(f"    {space}/{swap} vs {e['replaces']} on clipped: won "
                      f"{v['passes_won']} of {v['n_passes_paired']} passes, "
                      f"mean {v['mean_gain']:+.4f}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N30: clipped labels -- lift, drop, winsorise, or a censored likelihood')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Only {CENSORING_CONDITION!r} is accepted '
                             f'(default {DEFAULT_NOISE_CONDITION})')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated fractions clipped. Default: the '
                             'shared censoring grid')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default 0; '
                             'the clip does not depend on the draw)')
    parser.add_argument('--winsor-c', type=str, default=None,
                        help='Comma-separated winsorisation thresholds, in '
                             'units of 1.4826 * median |y - p| (default '
                             f'{",".join(str(c) for c in WINSOR_C)})')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides the 1 or 5 --mode gives')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {}
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.winsor_c:
        kw['winsor_c'] = [float(x) for x in args.winsor_c.split(',')]

    res = run(data, config, args.output, args.dataset, **kw)
    suffix = kw.get('noise_condition') or DEFAULT_NOISE_CONDITION
    if args.tag:
        suffix = f'{suffix}_{args.tag}'
    save_experiment_results(res, args.output, 'n30_censored_likelihood',
                            args.dataset, filename_suffix=suffix)


if __name__ == '__main__':
    main()
