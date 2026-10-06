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

THE CONFIGURATIONS (chat 5 alignment, 2026-10-05; HANDOFF_EXPERIMENT_ALIGNMENT
2.4). The dataset's two decided models (DECISIONS.md Step 1) only. Per outer
fold, per model: its 2 best base representations (base_1, base_2) and its 2
best general hybrids (hybrid, hybrid__general2), all ranked by searches run on
that fold's training rows (noise_helpers.fold_configurations). 2 models x 4 =
8 fitted configurations per fold. What each fold chose is written to
`configurations_per_fold`. Until 2026-10-05 the base representations and up to
six hybrid spaces per model came from whole-dataset stored searches, for five
models plus rf and svm. n30 measures nothing with neighbours, so the distance
and similarity hybrids are not used.

WHAT THE INJECTOR DOES, READ IN THIS SESSION. `noiseInject/core.py:736-779`,
reached through `noise_helpers.inject_train_labels`:

  - side 'upper' (`core.py:73`): values ABOVE the limit are replaced by it.
  - the limit is the (1 - level) quantile of the FULL label column, so the
    level is the fraction of labels clipped, not a dose in label units
    (`kirby/noise_spec.py:219`).
  - the limit comes back on the provenance row as `censoring_limit`.
  - the draw does not enter: the clip is `np.minimum(y, limit)`. MEASURED on
    500 labels at level 0.25: noise seeds 0, 1 and 2 give bit-identical
    columns. So this file runs ONE draw per level (noise_helpers.draws_for,
    decided 2026-10-06), and says so on the result file. A --draws list is
    cut to its first draw.

WHICH ROWS COUNT AS CLIPPED. The rows whose label equals the limit, which is what
a practitioner holding only the recorded labels can see. Since 2026-09-28 the
limit itself is read off the training labels too (`limit_from_labels`: the
largest label, when at least 1% of training molecules and at least 2 sit on
it), never from the injector; the injector's limit is written beside it for
evaluation only. So level 0.0 runs the lift and drop arms too wherever the
recorded labels already pile on their top value. That is not always the
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
and no search runs them. So each one is fitted in the feature matrices of the
model it replaces: rf's configurations for rf_l1, lightgbm's for
lightgbm_huber, xgboost's for xgboost_aft. Since 2026-10-05 a swap runs only
where the model it replaces is one of the dataset's two decided models; the
others are listed in `model_swaps_not_run` with the reason. KIRBy changes the input, not
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

WHERE THE FEATURES COME FROM. Each fold's recipes come from that fold's
searches, run on its training rows with the clean labels. Each fold then
rebuilds each hybrid from that fold's CLIPPED training labels, once per level,
and every arm shares it. So `clean` is the clean labels in clipped-label features, and no label
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

import time
import zlib

import numpy as np
from scipy.stats import norm

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.dataset_policy import noise_groups_for, noise_dose_units
from experiments.decided_grid import decided_models
# CHAT 5 ALIGNMENT (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT 2.4): folds,
# per-fold configurations, the decided levels and one injection call. These
# replace cv_split, get_best_search_for_downstream,
# pool_recipe_per_downstream_model, pool_recipes_by_family,
# noise_base_reps_by_model, group_models_by_recipe and DOWNSTREAM_MODELS.
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, inject_condition, decided_levels,
    dataset_knn_group, level_unit, CONFIGURATION_SPACES, draws_for,
)
from experiments.noise.n07_remediation import _predict_for_metric
# The D-versus-D' rows, the per-pass checkpoint, the caches and the controls
# (2026-09-25, plan sections 2-3). They live in n23 for the reason given at
# the top of its shared block.
from experiments.noise import noise_cache as nc
from experiments.noise.controls import controls_enabled
from experiments.noise.n23_uncertainty_cleaning import (
    RowSink, PassCheckpoint, fit_score_cached,
    control_twins, as_cleanup_dict, fit_cleaned, NO_NEIGHBOURS,
)

# The only condition this file is defined for. Every arm reads the limit.
CENSORING_CONDITION = 'censoring'
DEFAULT_NOISE_CONDITION = CENSORING_CONDITION

# ONE draw. The clip is np.minimum(y, limit) and no seed enters it, so draws 0, 1
# and 2 give bit-identical labels (measured, see the docstring). Three draws
# would be three copies of one fit.
# DECIDED 2026-10-06 (Adelaide delegated): one draw under censoring, through
# noise_helpers.draws_for. Re-measured 2026-10-05 through
# noise_helpers.inject_condition('censoring', level 0.25), 500 labels: draws
# 0, 1 and 2 return bit-identical training labels.
NOISE_DRAWS = (0,)

# Bumped 2026-10-05 (chat 5): the configurations are now chosen per outer fold
# by searches on that fold's training rows, for the dataset's two decided
# models. The rows file and the checkpoint carry this tag, so rows and passes
# written by the whole-dataset design (same model_input names, 'base_1',
# 'hybrid') are never read back as done.
RESULT_LAYOUT = 'fold_aligned_2026-10-05'

# The hybrid configurations, rebuilt per (level, fold) from the clipped
# training labels. base_1 and base_2 choose no features from the labels and
# are built once per fold.
BASE_CONFIGURATIONS = ('base_1', 'base_2')
HYBRID_CONFIGURATIONS = ('hybrid', 'hybrid__general2')

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


# 2026-09-28, owner's decision: the limit is read off the recorded training
# labels, never off the injector. A user holding clipped labels sees many of
# them piled on one top value. The largest training label counts as a limit
# when at least this share of training molecules (and at least 2) sit exactly
# on it; otherwise there is no limit and the lift and drop arms do not run.
MIN_SHARE_AT_LIMIT = 0.01


def limit_from_labels(y, min_share=MIN_SHARE_AT_LIMIT, tol=1e-12):
    """The upper limit a user would read off the labels, or None."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return None
    top = float(np.max(y))
    n_at = int(np.sum(y >= top - tol))
    return top if n_at >= max(2, int(np.ceil(min_share * y.size))) else None


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
    # float32 rule (handoff 2.3 item 18, 2026-10-05): the feature matrix is
    # handed over in its own dtype, never converted to float64 here. Labels
    # and the interval bounds stay float64.
    dtr = xgb.DMatrix(np.asarray(X_tr))
    dtr.set_float_info('label_lower_bound', t)
    dtr.set_float_info('label_upper_bound', upper)
    p = dict(AFT_PARAMS if params is None else params)
    p.update({'objective': 'survival:aft',
              'aft_loss_distribution': 'normal',
              'aft_loss_distribution_scale': float(max(sd, 1e-3))})
    booster = xgb.train(p, dtr, int(AFT_ROUNDS if rounds is None else rounds))
    pred_time = booster.predict(xgb.DMatrix(np.asarray(X_va)))
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


def _levels_and_draws(data, dataset_name, levels=None, draws=None):
    """(level, draw) pairs. One draw (draws_for), see NOISE_DRAWS. Level 0
    is run once.

    The levels are the decided censoring grid for this dataset's group
    (noise_helpers.decided_levels, Adelaide 2026-10-05): 0, 0.1, 0.25, 0.4 on
    full_knn and 0, 0.25 on fast_knn, as the fraction of labels clipped. A
    `levels` kwarg (a smoke) overrides.
    """
    levels = (decided_levels(data, dataset_name, CENSORING_CONDITION)
              if levels is None else levels)
    draws = tuple(draws_for(CENSORING_CONDITION,
                            NOISE_DRAWS if draws is None else draws))
    out = []
    for l in levels:
        for d in ((draws[0],) if float(l) == 0.0 else draws):
            out.append((float(l), int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N30: lift, drop, winsorise, or fit a censored likelihood on clipped labels.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    What each outer fold does, in order:
      1. Folds are noise_helpers.noise_outer_folds: the splits the per-fold
         searches are filed under, the count from config['n_splits'] or
         KIRBY_NOISE_N_FOLDS. Until 2026-10-05 they were _shared.cv_split.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels, decided grid, pruning off) and gives each
         of the dataset's two decided models its 2 best base representations
         and its 2 best general hybrids. Nothing is read from a whole-dataset
         search. Until 2026-10-05 the recipes and the base representations
         came from whole-dataset stored searches, ranked on every row, for five
         models plus rf and svm added by ensure_dprime_models, with up to six
         hybrid spaces per model (general, distance and similarity families).
      3. Each model is fitted on its 4 configurations (2 models x 4 = 8, R5):
         base_1 and base_2 built once per fold, the two hybrids rebuilt on the
         fold's training rows from the CLIPPED labels of each level, through
         noise_helpers.fold_model_inputs, which passes the fold search's own
         representation scores (dirty-run bug 1).
      4. Every label arm runs in each configuration with that configuration's
         model only. A model swap runs only in its parent model's
         configurations.
    n30 measures nothing with neighbours or distances, so the distance hybrid
    and the similarity hybrid are not used here (handoff 2.3 item 1).
    Each (level, draw, fold) pass is checkpointed as it finishes, and its rows
    go to disk first.
    """
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
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    if task_type != 'regression':
        # A skip with its reason, not a raise: run_all reaches n30 on every
        # dataset, and n19, n26, n27 and n29 skip a class label the same way.
        print(f"  SKIPPED: {dataset_name} is {task_type}. A class label has no "
              f"value above a limit to lift, so censoring is regression only.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    _metric = (metric_name, metric_fn, lower_is_better)

    winsor_c = [float(c) for c in (kwargs.get('winsor_c') or WINSOR_C)]
    shuffle_seeds = [int(s) for s in
                     (kwargs.get('shuffle_seeds') or SHUFFLE_SEEDS)]
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    arm_names = build_arm_names(winsor_c, shuffle_seeds)

    # R1 (chat 5, 2026-10-05): the dataset's two decided models
    # (DECISIONS.md Step 1), nothing else; knn runs only inside the distance
    # search. Replaces config['models'] (run_all hands every experiment five)
    # and ensure_dprime_models, which added rf and svm to every list. A
    # `models` kwarg (a smoke, a direct caller) can only narrow the decided
    # pair; a name outside it is dropped and written down.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        models = [m for m in asked if m in decided]
        models_not_decided = [m for m in asked if m not in decided]
    else:
        models = list(decided)
        models_not_decided = []
    if config.get('models'):
        print(f"  config['models'] {list(config['models'])} is not read: "
              f"n30 fits the decided models {decided}")
    if kwargs.get('base_reps') is not None:
        print("  base_reps given by the caller are not read: each fold's base "
              "representations come from that fold's search")
    model_factories = get_model_factories(
        models, task_type=task_type, n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n30: no decided model could be built for {dataset_name}. "
            f"Decided {decided}, asked {list(asked or [])}, factories "
            f"{sorted(model_factories)}.")

    # The swaps run only where the model they replace is one of the decided
    # pair. Until 2026-10-05 the run held five models, so rf, lightgbm and
    # xgboost were always present.
    # DECIDED 2026-10-06 (Adelaide delegated): a swap runs only where the
    # model it replaces is one of the dataset's two decided models. On a
    # dataset whose pair holds none of rf, lightgbm and xgboost (herg_ki,
    # caco2, adme_*, the svm/dnn and gp pairs) no swap runs, and each is
    # listed in model_swaps_not_run with the reason.
    comparators = {c: p for c, p in COMPARATORS.items() if p in models}
    comp_factories = get_model_factories(
        list(comparators), task_type=task_type, n_samples=len(labels))
    comparators = {c: p for c, p in comparators.items() if c in comp_factories}
    run_aft = (AFT_PARENT in models and not kwargs.get('skip_aft', False))
    swaps = dict(comparators)
    if run_aft:
        swaps[AFT_MODEL] = AFT_PARENT
    swaps_not_run = {
        s: (f'{p} is not one of the decided models {decided}' if p not in models
            else 'no model factory for this swap on this dataset'
            if s != AFT_MODEL else 'skip_aft was passed')
        for s, p in list(COMPARATORS.items()) + [(AFT_MODEL, AFT_PARENT)]
        if s not in swaps}

    noise_groups = noise_groups_for(data)[0]
    knn_grp = dataset_knn_group(data, dataset_name)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'result_layout': RESULT_LAYOUT,
        'selection_scope': 'fold_train_rows',
        'noise_condition': condition,
        'level_unit': level_unit(condition, False),
        'knn_group': knn_grp,
        'dose_units': noise_dose_units(dataset_name),
        'level_means': 'the fraction of the full label column clipped to the limit',
        'noise_draws': list(draws_for(
            CENSORING_CONDITION,
            NOISE_DRAWS if kwargs.get('draws') is None else kwargs['draws'])),
        'noise_draws_note': (
            'The clip is np.minimum(y, limit) and no seed enters it, so noise '
            'seeds 0, 1 and 2 give bit-identical labels. One draw per level '
            '(noise_helpers.draws_for, decided 2026-10-06); a longer draws '
            'list is cut to its first draw. Level 0 runs once.'),
        'models': models,
        'models_decided': decided,
        'models_asked_but_not_decided': models_not_decided,
        'models_without_factory': models_without_factory,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'model_swaps': swaps,
        'model_swaps_not_run': swaps_not_run,
        'model_swaps_note': (
            'Each swap is fitted on the clipped labels in the feature matrix '
            'built for the model it replaces, and paired against that model on '
            'the clipped labels. A swap runs only where the model it replaces '
            'is one of the dataset\'s two decided models. rf_l1 and '
            'lightgbm_huber have no search of their own. xgboost_aft enters '
            'each label as exp(label - centre) and returns '
            'log(prediction) + centre.'),
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
            "Each decided model has four configurations per outer fold: its "
            "2 best single representations and its 2 best general hybrids, "
            "all ranked by searches on that fold's training rows with the "
            "clean labels (fold_configurations). The hybrids are rebuilt on "
            "the fold's training molecules from the CLIPPED labels of each "
            "level, once, and shared by every arm, clean included. No label "
            "treatment reaches the feature choice."),
        'robustness_note': (
            'clipped_minus_clean in the summary is the clipped-label score '
            'minus the clean-label score, per model and configuration, in the '
            "dataset's own metric (read with lower_is_better). No auc_norm is "
            'computed here; on fast_knn datasets it is absent by decision.'),
        'claim_scope': (
            'The censoring is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
        'per_level': {},
        'clip_per_pass': {},
    }

    folds, n_folds = noise_outer_folds(data, config)
    results['n_folds'] = int(n_folds)
    results['folds_run'] = [int(f) for f, _, _ in folds]

    passes = _levels_and_draws(data, dataset_name, levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    n_cfg = len(models) * len(CONFIGURATION_SPACES)
    results['fitted_configurations_per_fold'] = n_cfg
    print(f"  {len(labels)} molecules, {n_folds} outer folds (running "
          f"{results['folds_run']}), decided models {models}, swaps {swaps}, "
          f"group {knn_grp}")
    print(f"  levels {results['noise_levels']} (fraction clipped), "
          f"{len(passes)} (level, draw) passes, {len(arm_names)} label arms")
    print(f"  {n_cfg} fitted configurations per fold ({len(models)} models x "
          f"{len(CONFIGURATION_SPACES)}: {list(CONFIGURATION_SPACES)})")

    # acc[level][space][arm][model][pass] = score; space is the configuration
    # name (base_1, base_2, hybrid, hybrid__general2) of the model it is
    # keyed under, so base_1 of one model and base_1 of the other are
    # different representations.
    acc = {}
    errors = {}
    clip_acc = {}
    # Per outer fold: what each model was fitted on (item 11 of handoff 2.4).
    fold_records = {}
    # Configurations that could not be built, per (level, model, space).
    config_unavailable = {}
    t0 = time.time()

    # ── Rows and the per-pass checkpoint (2026-09-25; layout tag 2026-10-05) ──
    # D is `clipped`, the labels as recorded; every label arm but `clean` is
    # a D'; `clean` is score_clean. The model swaps change the model, not the
    # training data, so they write no row. The per_molecule_lift_shuffled
    # arms are this file's own control and are written as D' rows like the
    # others.
    cond_tag = f'{condition}__{RESULT_LAYOUT}'
    rows = RowSink(output_dir, 'n30_censored_likelihood', dataset_name,
                   experiment='n30', metric=_metric,
                   condition=cond_tag, data=data, config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n30_censored_likelihood', dataset_name,
        condition=cond_tag, data=data, config=config,
        source_file=__file__,
        key={'layout': RESULT_LAYOUT, 'condition': condition,
             'winsor_c': winsor_c, 'shuffle_seeds': shuffle_seeds,
             'oof_splits': oof_splits, 'passes': passes,
             'models': list(models), 'swaps': swaps,
             'n_splits': int(n_folds), 'folds_run': results['folds_run']})
    done_passes = set()
    _state = ckpt.load()
    if _state:
        acc, errors, clip_acc = (_state['acc'], _state['errors'],
                                 _state['clip_acc'])
        fold_records = _state.get('fold_records', {})
        config_unavailable = _state.get('config_unavailable', {})
        done_passes = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_passes)} (level, draw, "
              f"fold) passes already done")

    def _save_state():
        ckpt.save({'acc': acc, 'errors': errors, 'clip_acc': clip_acc,
                   'fold_records': fold_records,
                   'config_unavailable': config_unavailable,
                   'done': sorted(done_passes)})

    for fold_i, train_idx, val_idx in folds:
        _fold_ids = [f'l{l:g}d{d}f{fold_i}' for l, d in passes]
        if (str(fold_i) in fold_records
                and all(p in done_passes for p in _fold_ids)):
            print(f"\n  Fold {fold_i}: every pass is on disk in the checkpoint")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            why = (f'the per-fold searches failed: {type(exc).__name__}: '
                   f'{exc}')
            print(f"    {why}")
            errors[f'f{fold_i}|fold_configurations'] = why
            fold_records[str(fold_i)] = {'error': why}
            _save_state()
            continue
        y_val = labels[val_idx]
        y_clean_tr = labels[train_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]

        # base_1 and base_2 of each model: no features are chosen from the
        # labels, so one matrix per fold serves every level.
        base_inputs = {}
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_clean_tr,
                factory=model_factories[m], metric=_metric,
                spaces=BASE_CONFIGURATIONS)
            base_inputs[m] = (ins, unav)
            rec = configuration_record(entry, names)
            rec['base_unavailable'] = {sp: why for sp, why in unav.items()
                                       if sp in BASE_CONFIGURATIONS}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            rec['training_row_score_note'] = (
                "hybrids[*].score_mean is the fold search's score of that "
                "hybrid on this fold's training rows (its own inner folds), "
                'in the search metric. The base representations\' search '
                'scores are not handed through fold_configurations. The '
                'held-out scores are in held_out_score below and in '
                'per_level, keyed by pass.')
            # held_out_score[level][space] = {'clean': .., 'clipped': ..}
            rec['held_out_score'] = {}
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        # A resumed fold keeps the held-out scores its finished passes wrote.
        _prev = (fold_records.get(str(fold_i)) or {}).get('models') or {}
        for m, rec in frec['models'].items():
            rec['held_out_score'].update(
                (_prev.get(m) or {}).get('held_out_score') or {})
        fold_records[str(fold_i)] = frec

        for level, draw in passes:
            lvl_key = f'{level:g}'
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if pass_key in done_passes:
                print(f"    {pass_key}: on disk in the checkpoint")
                continue
            cell = acc.setdefault(lvl_key, {})
            print(f"\n    --- fold {fold_i}, censoring / fraction {lvl_key} / "
                  f"draw {draw} ---")
            try:
                y0, cmask, prov = inject_condition(
                    data, dataset_name, condition, float(level), int(draw),
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            limit = limit_from_labels(y0)
            # Evaluation only: the injector's own limit, never handed to an arm.
            _inj = (prov or {}).get('censoring_limit') if level > 0 else None
            _inj = (float(_inj) if _inj is not None
                    and np.isfinite(float(_inj)) else None)
            at = rows_at_limit(y0, limit)
            n_tr = int(len(y0))
            clip_rec = {
                'pass': pass_key, 'limit': limit,
                'limit_read_from': 'the training labels (limit_from_labels)',
                'limit_from_injector_for_evaluation': _inj,
                'training_molecules': n_tr,
                'molecules_at_the_limit': int(at.sum()),
                'molecules_the_clip_moved': int(np.sum(cmask)),
                'how_far_above_the_limit_the_true_labels_sit_label_units': (
                    float(np.mean(y_clean_tr[at]) - limit)
                    if at.any() else float('nan')),
                'per_space': {},
            }

            # Each model's 4 configurations: base_1 and base_2 from the fold,
            # the two hybrids rebuilt from this level's clipped labels.
            configs = []
            for m in models:
                entry = entries.get(m)
                hyb, unav_h, _names = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=train_idx,
                    val_idx=val_idx, y_train=y0,
                    factory=model_factories[m], metric=_metric,
                    spaces=HYBRID_CONFIGURATIONS)
                b_ins, b_unav = base_inputs[m]
                for sp in CONFIGURATION_SPACES:
                    src_ = hyb if sp in HYBRID_CONFIGURATIONS else b_ins
                    unav_ = unav_h if sp in HYBRID_CONFIGURATIONS else b_unav
                    pair = src_.get(sp)
                    if pair is None:
                        (config_unavailable.setdefault(lvl_key, {})
                         .setdefault(m, {}).setdefault(
                             sp, unav_.get(sp) or 'not built'))
                        continue
                    configs.append((m, sp, pair[0], pair[1]))

            for mname, space, X_tr_s, X_va_s in configs:
                _ekey = f'{space}/{mname}'
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
                    if arm in ('clean', 'clipped'):
                        # The clean-label fit and the no-cleaning fit on D,
                        # through noise_cache for a seeded model (plan
                        # section 3). keep is None for both.
                        s, err = fit_score_cached(
                            dataset_name, mname, model_factories[mname],
                            X_tr_s, y_use, X_va_s, y_val, metric_fn,
                            metric_name)
                    else:
                        s, err = _fit_score(
                            model_factories[mname], X_tr_s, y_use, X_va_s,
                            y_val, metric_fn, metric_name, keep=keep)
                    if s is None:
                        errors[f'{lvl_key}|{_ekey}|{arm}|f{fold_i}'] = err
                        continue
                    (space_cell.setdefault(arm, {})
                     .setdefault(mname, {})[pass_key]) = s

                sD = (space_cell.get('clipped') or {}).get(
                    mname, {}).get(pass_key)
                sC = (space_cell.get('clean') or {}).get(
                    mname, {}).get(pass_key)
                (frec['models'][mname]['held_out_score']
                 .setdefault(lvl_key, {}))[space] = {'clean': sC,
                                                     'clipped': sD}

                # ── The D-versus-D' rows and the controls (2026-09-25) ──
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
                                fold=fold_i, score_D=sD,
                                score_Dprime=sP, score_clean=sC)

                # ── The model swaps, on the clipped labels, in the parent
                #    model's own configurations only ──
                for swap, parent in swaps.items():
                    if parent != mname:
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
            _save_state()
            print(f"    fraction {lvl_key} fold {fold_i}: "
                  f"{clip_rec['molecules_at_the_limit']} of {n_tr} training "
                  f"molecules at the limit {limit}, {len(configs)} "
                  f"configurations [{time.time() - t0:.0f}s]")

    results['per_level'] = convert_numpy(acc)
    results['clip_per_pass'] = convert_numpy(clip_acc)
    results['configurations_per_fold'] = convert_numpy(fold_records)
    results['configurations_unavailable'] = convert_numpy(config_unavailable)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, arm_names, models, swaps, metric_name, lower_is_better))
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    ckpt.clear()
    return results


def _outer_fold_of(pass_key):
    """The outer fold number of a pass key 'l<level>d<draw>f<fold>'."""
    return int(str(pass_key).rsplit('f', 1)[1])


def _by_fold(pairs):
    """{fold: mean over that fold's draws} from [(pass_key, value)]."""
    by = {}
    for k, v in pairs:
        by.setdefault(_outer_fold_of(k), []).append(float(v))
    return {f: float(np.mean(v)) for f, v in sorted(by.items())}


def _paired(base, got, lower_is_better):
    """One arm against another, on the passes both scored. Positive means better.

    n24:981-1006. The direction is read off the dataset.

    2026-10-05 (chat 5): the spread is over OUTER FOLDS
    (noise_helpers.fold_spread): each fold's gain is the mean over its draws,
    and the spread is the standard deviation of those, or None with 'one
    fold, no spread' when one fold ran. Until then it was the standard
    deviation over every (draw, fold) pass, which is 0 on one pass.
    passes_won, passes_lost and passes_tied still count (draw, fold) passes.
    """
    shared = sorted(set(base) & set(got))
    pairs = [(k, (base[k] - got[k]) if lower_is_better else (got[k] - base[k]))
             for k in shared]
    pairs = [(k, float(g)) for k, g in pairs if np.isfinite(g)]
    gains = np.array([g for _k, g in pairs], dtype=float)
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    fs = fold_spread(list(_by_fold(pairs).values()))
    return {
        'n_passes_paired': int(gains.size),
        'passes_won': int((gains > tol).sum()),
        'passes_lost': int((gains < -tol).sum()),
        'passes_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'gain_over_folds': fs,
        'pass_gains': {k: g for k, g in pairs},
    }


def _clipped_minus_clean(clean_d, clipped_d):
    """The plain difference damaged minus clean, per outer fold, in the
    dataset's own metric (read it with lower_is_better). The robustness
    number the noise-grid decision asks for on fast_knn datasets, written on
    every dataset (Adelaide 2026-10-05)."""
    keys = sorted(set(clean_d) & set(clipped_d))
    pairs = [(k, clipped_d[k] - clean_d[k]) for k in keys
             if np.isfinite(clipped_d[k]) and np.isfinite(clean_d[k])]
    return fold_spread(list(_by_fold(pairs).values()))


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
                       'clipped_minus_clean': _clipped_minus_clean(
                           clean.get(m) or {}, b),
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


def _spread_text(fs):
    if fs.get('sd') is None:
        return fs.get('note') or 'no spread'
    return f"fold sd {fs['sd']:.4f} over {fs['n_folds']} folds"


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
                          f"{v['mean_gain']:+.4f} "
                          f"({_spread_text(v['gain_over_folds'])}){tail}")
            for swap, e in srow['model_swaps'].items():
                v = e['vs_parent_on_clipped']
                print(f"    {space}/{swap} vs {e['replaces']} on clipped: won "
                      f"{v['passes_won']} of {v['n_passes_paired']} passes, "
                      f"mean {v['mean_gain']:+.4f}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the fold spread of the gains behind it is not a gain.")


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
                        help='Comma-separated injection replicates. Cut to '
                             'the first (draws_for): the clip does not depend '
                             'on the draw')
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
