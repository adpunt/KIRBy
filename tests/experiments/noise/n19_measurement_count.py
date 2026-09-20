#!/usr/bin/env python3
"""N19: what a measurement count and a measurement spread are worth.

Paper: Noise.

THE QUESTION. A molecule measured six times carries a better label than one
measured once. Nothing in this suite has ever used that, because until
2026-09-16 no loader kept it: every hERG loader averaged a molecule's repeat
measurements together and wrote one number (NOISE_AUDIT_2026_09_10.md 11.1).
Two datasets keep them now -- `herg_ic50`, one row per ChEMBL record, and
`bindingdb_kd`, one row per measured (drug, protein) pair.

NOTHING IS INJECTED. The disagreement is what different laboratories got when
they measured the same thing. This is n17's regime, and n17's split is reused
rather than rebuilt.

HOW A MOLECULE IS GRADED. One row per molecule. A training molecule contributes
a label built from its own measurements. A held-out molecule is graded against
the mean of the measurements not used for training. A molecule is never on both
sides, so nothing a molecule trained on is also what it is graded against.
`n17_replicate_removal.build_entities` builds all of that and is imported here.

TWO WAYS OF BUILDING A TRAINING LABEL, AND THE DIFFERENCE IS THE POINT.

  single    ONE measurement, chosen at random. What a label looks like when it
            came from one laboratory. Its error variance is that molecule's own
            within-molecule variance, and it does NOT fall with the measurement
            count -- one measurement of a molecule measured six times is no more
            precise than one measurement of a molecule measured twice.
  averaged  the mean of ALL that molecule's measurements. What a loader normally
            hands you. Its error variance is the within-molecule variance
            DIVIDED BY the measurement count, so the count is now a precision.

A training molecule and a held-out molecule are different molecules, so a
training molecule using every one of its own measurements leaks nothing.

That split is what makes the weights below testable rather than assumed. Under
`single` the count is not a precision and weighting by it should do nothing.
Under `averaged` it is, and weighting by it is the inverse-variance weight that
meta-analysis has used since Hedges & Olkin 1985. Both run under both, so which
weight wins is measured rather than argued from the algebra.

SIX WAYS OF CHOOSING WHAT TO TRAIN ON. All six are graded on the same held-out
molecules against the same consensus labels, in the same feature space, with the
same folds. Three set a per-molecule fit weight; two change the training label;
one does nothing.

  untouched            every training molecule at its label, unweighted. The
                       thing to beat.
  count_weighted       weight proportional to how many times the molecule was
                       measured.
  spread_weighted      weight proportional to one over the variance of that
                       molecule's own measurements.
  precision_weighted   weight proportional to the count divided by that
                       variance -- the inverse-variance weight for an averaged
                       label.
  shrunk               the training label moves toward a group-aware
                       out-of-fold prediction, and moves further the less
                       precise it is. Each molecule's own measurement variance
                       sets how far.
  calibrated           the same shrinkage, except the variance is not each
                       molecule's own. It is fitted against the measurement
                       count on the molecules measured three or more times, and
                       then applied to every molecule including the ones
                       measured once, which have no variance of their own.

`shrunk` and `calibrated` differ in one thing: whether a molecule's shrinkage
uses its own spread, which is a noisy number at two or three measurements, or a
spread read off a curve fitted on the molecules that have enough measurements to
estimate one. That second one is the trusted-subset idea -- learn the noise
function where it can be learned, apply it where it cannot.

WHAT IS AND IS NOT AN ORACLE. A molecule's own measurement spread is information
a practitioner genuinely holds: they have the measurements. It is not the
held-out answer, and no held-out molecule's label enters any weight or any
shrinkage anywhere in this file. It IS the same quantity n17's `ceiling` ranks
on, and n17 calls that an oracle -- but n17 uses it to REMOVE molecules and
calls it an oracle because no detector can beat it, not because it leaks. Here
it is used as a weight on the molecule's own row.

WHERE THE HYBRID COMES IN, AND THE LIMIT ON IT. Every condition runs in one base
representation and in one hybrid per family of stored search, each recipe coming
from the precomputed pool and each rebuilt per fold through `build_fold_hybrid`,
the same way n02, n07, n11, n12 and n13 reach a hybrid. There are two families:
a search that chose its features to predict the label, and one that chose them
so that each molecule's activity is predicted by its five nearest neighbours in
the distance those features define. Their spaces are 'hybrid' and
'hybrid_similarity'. Before 2026-09-19 only the first was built and the second
was written into the result file as a score nothing had measured in.

The hybrid is built ONCE per (fold, draw, label construction), on the UNTOUCHED
training labels, and all six conditions share it. So the only thing differing
between two conditions is the weight or the label handed to the downstream fit.
Whether a recipe searched on the shrunk labels would do better is a different
question and it is n20's.

`sample_weight` appears zero times in `src/kirby/hybrid.py` and zero times in
`src/kirby/find_best_hybrid.py`. So the three weighted conditions weight the
downstream fit and NOT the feature choice, in the hybrid space and in the base
space alike. `weights_reach_feature_selection` is written False on every result
row, because a reader who sees a hybrid beaten by a weight would otherwise
assume the weight had been in the budget split.

NO ORACLE CAVEAT TO CARRY. n11, n12 and n13 record that their recipe was chosen
on clean labels and their experiment then corrupts them. Here nothing is
corrupted and there is no clean column, so the recipe saw the same labels this
experiment trains on. `recipe_saw_clean_labels` is written False for that reason.

Usage:
    python n19_measurement_count.py --dataset herg_ic50 --mode full
    python n19_measurement_count.py --dataset bindingdb_kd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    # One feature space per search family the pool holds, added 2026-09-19.
    # n11's loop, in one place. See SPACE_FOR_FAMILY below.
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    pick_noise_base_rep,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n17_replicate_removal import (
    build_entities, MIN_MEASUREMENTS,
)
# One per-fold hybrid build, shared with n17 and n18. It lived here until
# 2026-09-17; n17 and n18 then needed it and a third copy was the alternative.
from experiments.noise.noise_helpers import hybrid_for_fold
from kirby.utils.noisy_labels import oof_predict

# How a training molecule's label is built from its own measurements.
LABEL_BUILDS = ('single', 'averaged')

# What is done to the training molecules before the downstream fit.
CONDITIONS = ('untouched', 'count_weighted', 'spread_weighted',
              'precision_weighted', 'shrunk', 'calibrated')

# The three that set a per-molecule fit weight. A model whose fit() refuses a
# weight is absent from these three rather than fitted unweighted and recorded
# as if it had been weighted -- the mistake n13 names at :155-162.
WEIGHTED_CONDITIONS = ('count_weighted', 'spread_weighted',
                       'precision_weighted')

# The two that change the training label instead.
RELABEL_CONDITIONS = ('shrunk', 'calibrated')

# Which feature space each condition is measured in. All of them, always.
#
# 2026-09-19: 'hybrid_similarity' was added. The pool is asked for two families
# of stored search (see POOL_SUBSET) and until today it built one of them.
# POOL_SUBSETS names the families in PREFERENCE order and the pool returns the
# first that has an entry, because their stored scores are in different units;
# a predictive recipe always exists, so the geometry family was written into
# the result file and never measured in. 'hybrid' is still the predictive
# recipe, so every number reported before today still means what it meant.
SPACES = ('base', 'hybrid', 'hybrid_similarity')

# Which result-file space each family of stored search is built into.
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

# How many times WHICH measurement becomes the `single` training label, and
# which measurements form each held-out molecule's consensus, are re-drawn.
# Three, matching n07, n09, n11, n12, n13, n15 and n17.
N_DRAWS = 3

# Folds for the out-of-fold prediction the two relabelling conditions shrink
# toward. It runs inside the training molecules of an outer fold only.
OOF_SPLITS = 5

# Which searches the recipe may come from. Same as n02, n06, n07, n11, n12 and
# n13 since 2026-09-12: a search whose features were chosen to predict the
# label, and one whose features were chosen so that the distance they define
# predicts each molecule's activity from its five nearest neighbours.
POOL_SUBSET = 'predictive_and_similarity'

# A weight is a ratio, so it needs a floor: a molecule whose measurements all
# agreed has a variance of zero and an infinite weight. The floor is a share of
# the median defined variance, so it scales with the dataset rather than being
# a number in the label's units.
VARIANCE_FLOOR_SHARE = 0.01


def measurement_stats(labels, replicate_groups):
    """Per-molecule measurement count, mean and variance, in entity order.

    Entity order is `build_entities`' order -- molecules sorted by the row their
    first measurement sits on -- and the caller checks that the two agree rather
    than trusting that they do.

    Returns a dict of arrays:
      row_index       the first row of this molecule, to read features from
      n_measurements  how many times it was measured
      mean_all        the mean of ALL its measurements
      var_all         the variance of its measurements, ddof=1, NaN below two

    ddof=1, for the reason n17 records at :176-184: with ddof=0 the measured
    spread grows with how many times a molecule was measured whatever its real
    disagreement, so a quiet molecule measured six times outranks a noisy one
    measured twice. Everything downstream of this ranks or divides by it.

    Reviewed 2026-09-18 alongside n17's identical floor-of-2 question: no
    change requested here. `var_all` at k=2 feeds `weights_for`'s sample weights and
    `shrink_toward`'s label shrinkage directly, so it moves what gets fit, not
    just what gets reported -- that was the point of checking it, and the
    decision was to leave it as-is.
    """
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(replicate_groups)

    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    gids = sorted(order, key=lambda g: order[g][0])

    n = len(gids)
    out = {
        'row_index': np.zeros(n, dtype=np.int64),
        'n_measurements': np.zeros(n, dtype=np.int64),
        'mean_all': np.zeros(n, dtype=np.float64),
        'var_all': np.full(n, np.nan, dtype=np.float64),
    }
    for j, g in enumerate(gids):
        rows = np.asarray(order[g], dtype=np.int64)
        vals = labels[rows]
        out['row_index'][j] = rows[0]
        out['n_measurements'][j] = len(rows)
        out['mean_all'][j] = float(np.mean(vals))
        if len(rows) >= 2:
            out['var_all'][j] = float(np.var(vals, ddof=1))
    return out


def training_labels(ent, stats, build):
    """The training label for every molecule, and how many measurements it rests on.

    `single`   one measurement, chosen at random -- what `build_entities` put in
               `train_label`. It rests on ONE measurement whatever the molecule's
               count, which is exactly why the count is not a precision here.
    `averaged` the mean of all of this molecule's measurements. It rests on all
               of them.

    A molecule measured once gets the same number either way.

    Returns (y_train, n_behind_label).
    """
    if build == 'single':
        # `build_entities` puts ONE measurement in `train_label` only for the
        # molecules it can grade. A molecule measured once or twice cannot be
        # graded, so it gets the mean of its measurements instead -- n17's
        # choice, at :185-192, and the right one: there is nothing to hold out
        # for it and the mean is the best single number its rows support.
        #
        # So the count behind a `single` label is 1 for a gradeable molecule and
        # its whole measurement count for the rest. Writing 1 everywhere would
        # say a molecule measured twice carries twice the error variance it
        # does, and underweight it by that much.
        n_behind = np.where(ent['gradeable'], 1.0,
                            stats['n_measurements'].astype(np.float64))
        return ent['train_label'].copy(), n_behind
    if build == 'averaged':
        return (stats['mean_all'].copy(),
                stats['n_measurements'].astype(np.float64))
    raise ValueError(f"unknown label build {build!r}")


def label_variance(stats, n_behind, floor_share=VARIANCE_FLOOR_SHARE):
    """The error variance of each molecule's TRAINING label, and where it came from.

    A molecule's measurements disagree by `var_all`. A label resting on
    `n_behind` of them has an error variance of `var_all / n_behind` -- one
    measurement carries the whole of it, a mean of six carries a sixth.

    A molecule measured once has no variance of its own. It is given the median
    of the ones that do, rather than zero (which is an infinite weight) or
    dropped (which changes how many molecules train, for a reason that has
    nothing to do with the question). How many were filled that way is returned
    and written to the result file.

    Returns (variance, n_filled, floor).
    """
    var = np.asarray(stats['var_all'], dtype=np.float64).copy()
    known = np.isfinite(var)
    if not known.any():
        raise RuntimeError(
            "no molecule in this dataset was measured more than once, so no "
            "measurement variance exists and n19 has nothing to weight by.")
    median_var = float(np.median(var[known]))
    floor = max(floor_share * median_var, 1e-12)
    var[~known] = median_var
    var = np.maximum(var, floor)
    return var / np.maximum(n_behind, 1.0), int((~known).sum()), float(floor)


def calibrated_variance(stats, n_behind, trusted_min=MIN_MEASUREMENTS,
                        floor_share=VARIANCE_FLOOR_SHARE):
    """The error variance read off a curve fitted on the well-measured molecules.

    The trusted subset is the molecules measured `trusted_min` or more times --
    the only ones whose own variance is worth much. Within it, the mean variance
    is taken at each distinct measurement count. A molecule is then given the
    fitted variance for its own count, and a count the trusted subset never saw
    takes the nearest count it did see.

    This is the difference between this and `shrunk`: `shrunk` uses a molecule's
    own variance, which at two measurements is one number from two points.
    This uses a variance estimated across every molecule measured that many
    times.

    Returns (variance, fit, n_trusted). `fit` is {count: variance} and is
    written to the result file, because it IS the noise function and a reader
    should be able to see whether it falls with the count or is flat.
    """
    var = np.asarray(stats['var_all'], dtype=np.float64)
    k = np.asarray(stats['n_measurements'], dtype=np.int64)
    trusted = np.isfinite(var) & (k >= int(trusted_min))
    if not trusted.any():
        raise RuntimeError(
            f"no molecule was measured {trusted_min} or more times, so there "
            f"is no trusted subset to fit a noise function on.")

    fit = {int(c): float(np.mean(var[trusted & (k == c)]))
           for c in np.unique(k[trusted])}
    counts = np.array(sorted(fit), dtype=np.int64)
    values = np.array([fit[int(c)] for c in counts], dtype=np.float64)

    median_var = float(np.median(var[np.isfinite(var)]))
    floor = max(floor_share * median_var, 1e-12)
    # Nearest count the trusted subset actually saw. A molecule measured once
    # takes the smallest fitted count, which is the most pessimistic of them.
    nearest = np.abs(k[:, None] - counts[None, :]).argmin(axis=1)
    fitted = np.maximum(values[nearest], floor)
    return (fitted / np.maximum(n_behind, 1.0),
            {int(c): float(v) for c, v in zip(counts, values)},
            int(trusted.sum()))


def weights_for(condition, n_measurements, variance, n_behind):
    """The per-molecule fit weight, normalised to average one.

    Average one rather than sum one, so a weighted fit and an unweighted fit see
    the same total weight and the difference between them is the SHAPE of the
    weighting and not how much of it there is.

    count_weighted      how many times the molecule was measured
    spread_weighted     one over the variance of its own measurements
    precision_weighted  the count divided by that variance -- the
                        inverse-variance weight for a label that is the mean of
                        that many measurements
    """
    k = np.asarray(n_measurements, dtype=np.float64)
    # `variance` is already var_all / n_behind. The per-measurement variance is
    # what the two spread weights divide by, so multiply it back out.
    per_measurement = np.asarray(variance, dtype=np.float64) * np.maximum(
        n_behind, 1.0)
    if condition == 'count_weighted':
        w = k
    elif condition == 'spread_weighted':
        w = 1.0 / per_measurement
    elif condition == 'precision_weighted':
        w = k / per_measurement
    else:
        raise ValueError(f"{condition!r} does not set a weight")
    mean_w = float(np.mean(w))
    if not np.isfinite(mean_w) or mean_w <= 0:
        raise RuntimeError(f"{condition}: the weights average {mean_w}, which "
                           f"cannot be normalised.")
    return w / mean_w


def shrink_toward(y, oof, variance):
    """Move each training label toward the out-of-fold prediction.

    The amount is `variance / (variance + tau2)`, where tau2 is how far the
    labels sit from the out-of-fold prediction overall. A molecule whose label
    is precise barely moves; one whose label rests on a single noisy
    measurement moves most of the way. That is the James-Stein form, and it is
    one of the families `kirby/utils/noisy_labels/__init__.py:16-19` records as
    deferred.

    Returns (y_shrunk, shrinkage, tau2).
    """
    y = np.asarray(y, dtype=np.float64)
    oof = np.asarray(oof, dtype=np.float64)
    tau2 = float(np.var(y - oof))
    if not np.isfinite(tau2) or tau2 <= 0:
        # Nothing to shrink toward: the prediction already reproduces the label
        # exactly, so every label would move the whole way for no reason.
        return y.copy(), np.zeros_like(y), tau2
    s = np.asarray(variance, dtype=np.float64) / (
        np.asarray(variance, dtype=np.float64) + tau2)
    return (1.0 - s) * y + s * oof, s, tau2


def _fit_and_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, weights):
    """One fit, one held-out score, or None with the reason recorded.

    A model whose fit() refuses a per-molecule weight comes back with
    'refuses_sample_weight' and the caller drops it from the weighted
    conditions. It is never fitted unweighted and recorded as if it had been
    weighted -- that is the mistake n13 names at :155-162 and guards the same
    way.
    """
    try:
        m = factory()
        if weights is None:
            m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr, y_tr, sample_weight=weights)
        return float(metric_fn(y_va, m.predict(X_va))), None
    except TypeError as exc:
        if weights is not None and 'sample_weight' in str(exc):
            return None, 'refuses_sample_weight'
        return None, f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _drop_blueprint(obj):
    """Strip the pickled Blueprint out of the top-2 blocks before they are saved.

    It does serialise -- convert_numpy routes it through Blueprint.to_dict
    (find_best_hybrid.py:1438) -- but it carries kept_augmentations, which is
    one list of feature indices per fold per donor representation, and every
    recipe field on it (allocation method, importance method, budget, internal
    evaluator, step size, representation count) is already a key of the entry
    it hangs off. Which representations actually fed a configuration is
    recorded by hybrid_deltas under 'source_reps'.
    """
    if isinstance(obj, dict):
        return {k: _drop_blueprint(v) for k, v in obj.items()
                if k != 'blueprint'}
    if isinstance(obj, list):
        return [_drop_blueprint(v) for v in obj]
    return obj


def _print_deltas(deltas_by_model, models, metric_name):
    """The top configuration against its two reference numbers, per model.

    Two numbers and not one, because they answer different questions. The first
    is the configuration against the MEAN of the representations the budget
    actually gave features to, read off Blueprint.allocation_per_fold, so a
    representation given 0 features is not in that mean. The second is it
    against the single best base representation for the same downstream model.
    A configuration can win the first and lose the second.
    """
    if not deltas_by_model:
        return
    print(f"\n  TOP HYBRID vs TWO REFERENCE NUMBERS ({metric_name}, "
          f"+ means the hybrid won):")
    for m in models:
        hd = deltas_by_model.get(m)
        if not hd:
            continue
        dsm, dbr = hd.get('delta_vs_source_mean'), hd.get('delta_vs_best_rep')
        src_mean, n_src = hd.get('source_mean'), hd.get('n_source_reps')
        best_rep, best_sc = hd.get('best_rep'), hd.get('best_rep_score')
        left = (f"vs the mean of the {n_src} representations that fed it "
                f"({src_mean:.4f}): {dsm:+.4f}"
                if dsm is not None and src_mean is not None
                else "vs the mean of its source representations: --")
        right = (f"vs the best single representation {best_rep} "
                 f"({best_sc:.4f}): {dbr:+.4f}"
                 if dbr is not None and best_sc is not None
                 else "vs the best single representation: --")
        print(f"    {m}: {left}; {right}")


def run(data, config, output_dir, dataset_name, **kwargs):
    """N19: the measurement count and the measurement spread as weight and as shrinkage."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N19: WHAT A MEASUREMENT COUNT AND A SPREAD ARE WORTH")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    reps = data['reps']
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A repeat measurement "
              f"of a class label either agrees or does not, so there is no "
              f"spread in the label's own units and no consensus mean.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        print(f"  SKIPPED: {dataset_name}'s loader exposes no repeat structure, "
              f"so no molecule has a measurement count or a spread. "
              f"NOISE_AUDIT_2026_09_10.md 11.1.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    n_draws = int(kwargs.get('n_draws', N_DRAWS))
    label_builds = list(kwargs.get('label_builds') or LABEL_BUILDS)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))

    stats = measurement_stats(labels, rep_groups)
    probe = build_entities(labels, rep_groups, seed=0)
    if not np.array_equal(probe['n_measurements'], stats['n_measurements']):
        raise RuntimeError(
            "n19: measurement_stats and build_entities disagree about how many "
            "times each molecule was measured, so they are not in the same "
            "order and every weight would be attached to the wrong molecule.")
    if not np.array_equal(probe['row_index'], stats['row_index']):
        raise RuntimeError(
            "n19: measurement_stats and build_entities disagree about which row "
            "represents each molecule.")

    n_ent = len(stats['n_measurements'])
    n_grade = int(probe['gradeable'].sum())
    counts = {int(k): int((stats['n_measurements'] >= k).sum())
              for k in (2, 3, 4, 5)}
    print(f"  {len(labels)} measurements over {n_ent} molecules")
    for k, v in counts.items():
        print(f"    molecules measured {k}+ times: {v} of {n_ent} "
              f"({100.0 * v / max(n_ent, 1):.2f}%)")
    print(f"  gradeable molecules (>= {MIN_MEASUREMENTS} measurements): {n_grade}")

    # ⛔ config FIRST, then the kwarg. run_all passes the caller's model list in
    # `config['models']` and never as a `models` kwarg -- the only kwarg it ever
    # forwards is noise_condition (noise_helpers.py:1053). This file read
    # kwargs alone, so the list the runner was asked for was silently dropped
    # and every run fell back to the module default. Eight files in this
    # directory had it; the same files read config['n_splits'] correctly two
    # lines away. Fixed 2026-09-20. An explicit kwarg still wins, which is what
    # a smoke or a direct caller uses.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_ent)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n19: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'models': downstream_models,
        'label_builds': label_builds,
        'conditions': list(CONDITIONS),
        'spaces': list(SPACES),
        'n_draws': n_draws,
        'min_measurements_to_grade': MIN_MEASUREMENTS,
        'oof_splits': oof_splits,
        'feasibility': {
            'n_rows': int(len(labels)),
            'n_molecules': n_ent,
            'n_molecules_gradeable': n_grade,
            'molecules_by_min_measurements': counts,
        },
        # Two facts a reader needs before reading any number below, both stated
        # rather than left to be inferred from the absence of a caveat.
        'weights_reach_feature_selection': False,
        'weights_reach_feature_selection_note': (
            'sample_weight appears nowhere in src/kirby/hybrid.py or '
            'src/kirby/find_best_hybrid.py. The three weighted conditions '
            'weight the downstream fit only. The features were chosen without '
            'any weight, in the base space and the hybrid space alike.'),
        'recipe_saw_clean_labels': False,
        'recipe_saw_clean_labels_note': (
            'Nothing is injected here and there is no clean label column, so '
            'the recipe saw the same labels this experiment trains on. The '
            'oracle caveat n11, n12 and n13 carry does not apply.'),
        'per_draw': {},
    }

    if n_grade < 30:
        print(f"  SKIPPED: {n_grade} gradeable molecules is too few to split "
              f"into {n_folds} folds and read a difference between six "
              f"conditions.")
        results['skipped'] = 'too_few_gradeable_molecules'
        return results

    # ── The hybrid recipe, fetched once ──
    # The pool is keyed on the dataset as the precompute built it, so the lookup
    # is handed the ROW-level representations, labels and fold grouping, exactly
    # as n02, n11, n12 and n13 hand them over. What comes back is a recipe -- a
    # representation set, an allocation method, an importance method, a budget
    # and an internal evaluator -- and the features are rebuilt per fold on the
    # entity rows below.
    augmentations = data.get('augmentations')
    bp = None
    # Bound before the try so the top-2 block below can test it. Until
    # 2026-09-18 nothing read `search` and an exception here left the name
    # unbound; reading it would have raised UnboundLocalError on the one path
    # this file is built to survive -- a dataset the precompute never covered.
    search = None
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, data.get('groups'),
            downstream_model=downstream_models[0], budget=None,
            augmentations=augmentations,
            dataset_name=dataset_name, task_type=task_type,
            data=data,
            pool_subset=POOL_SUBSET,
        )
        bp = meta['best_config']['blueprint'] if meta else None
    except Exception as exc:
        meta = None
        print(f"  hybrid recipe lookup failed: {type(exc).__name__}: {exc}")

    # One recipe per family the pool holds, and a written reason for each family
    # that produced none. `bp` stays the recipe the pool RETURNED, which is what
    # every 'hybrid_*' key below already described.
    hybrid_recipes, hybrid_unavailable = pool_recipes_by_family(
        meta, SPACE_FOR_FAMILY,
        eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
    recipe_for_space = {r['space']: r['blueprint'] for r in hybrid_recipes}
    results['hybrid_spaces_unavailable'] = hybrid_unavailable

    if not hybrid_recipes:
        # Every condition still runs, in the base space alone. Dropping the run
        # would lose six real answers because one of several spaces is missing.
        spaces = ('base',)
        results['hybrid_available'] = False
        results['hybrid_unavailable_reason'] = (
            'no recipe came back from the precomputed pool for subset '
            f'{POOL_SUBSET!r}, so the hybrid space could not be built. Every '
            'condition ran in the base representation only.')
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"base representation only")
    else:
        spaces = ('base',) + tuple(r['space'] for r in hybrid_recipes)
        results['hybrid_available'] = True
        results['hybrid_family'] = meta['objective']
        results['hybrid_family_choice'] = meta['objective_choice']
        results['hybrid_pool_subset'] = meta['pool_subset']
        results['hybrid_tier'] = meta['tier']
        results['hybrid_aug_condition'] = meta['aug_condition']
        results['hybrid_budget'] = getattr(bp, 'budget', None)
        results['hybrid_reps'] = sorted(getattr(bp, 'reps', []) or [])
        results['hybrid_pool_by_objective'] = convert_numpy({
            o: {'tier': d['tier'], 'aug_condition': d['aug_condition'],
                'score_mean': d['score_mean'], 'metric_name': d['metric_name'],
                'lower_is_better': d['lower_is_better']}
            for o, d in (meta.get('by_objective') or {}).items()
        })
        print(f"  hybrid recipe: {len(results['hybrid_reps'])} representations, "
              f"budget {results['hybrid_budget']}, searched under objective "
              f"{meta['objective']} (pool subset {meta['pool_subset']})")
        results['hybrid_space_of_pool_winner'] = SPACE_FOR_FAMILY.get(
            meta['objective'], f"hybrid_{meta['objective']}")
        results['hybrid_spaces'] = convert_numpy({
            r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'is_the_recipe_the_pool_returned': r['is_pool_winner'],
                'tier': r['tier'], 'aug_condition': r['aug_condition'],
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in hybrid_recipes})
        results['hybrid_spaces_note'] = (
            'One hybrid space per family of stored search. A predictive search '
            'chose its features to predict the label and its search_score_mean '
            "is in the dataset's own metric. A similarity search chose its "
            "features so that each molecule's activity is predicted by its "
            'five nearest neighbours in the distance those features define, '
            'fits no downstream model, and its search_score_mean is that '
            'correlation. The two search scores are in different units and '
            'nothing here ranks one against the other. Every space below is '
            "scored in this experiment's own conditions, in the dataset's own "
            'metric, on the same folds.')
        for r in hybrid_recipes:
            print(f"  hybrid space {r['space']}: family {r['objective']!r}, "
                  f"budget {r['budget']} features, allocation "
                  f"{r['allocation_method']}, tier {r['tier']}/"
                  f"{r['aug_condition']}, search score "
                  f"{r['search_score_mean']} in {r['search_metric_name']!r}")
        for _sp, _why in hybrid_unavailable.items():
            print(f"  no {_sp} space: {_why}")
    results['spaces_run'] = list(spaces)

    # Which representation the base space is. The pool's own preference order is
    # for injected-noise experiments picking one fixed matrix; here the base
    # space exists to be the thing the hybrid is read against, so it is the
    # representation pick_noise_base_rep names, and is recorded on the result.
    # pick_noise_base_rep, not sorted(reps)[0]. The alphabetical pick is the
    # fallback that function was written to replace: it makes the control arm a
    # matter of spelling, and it can hand the control slot to a structure-tuned
    # sim_* recipe, which is a treatment this study built. n11, n17 and n21
    # already call it; these four did not. Changed 2026-09-18.
    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    results['base_rep'] = base_rep_name
    print(f"  base representation: {base_rep_name}")

    # ── What each downstream model would have picked for itself ──
    # Added 2026-09-18, ADDITIVE. Nothing above or below moves: `base_rep` is
    # still the one pick_noise_base_rep names, `spaces_run` is still the same 2
    # spaces, the hybrid is still the 1 recipe fetched for
    # downstream_models[0], and every paired fold count, mean gain and spread
    # in 'per_draw' and 'summary' is still measured in those 2 spaces. These 4
    # keys sit beside them.
    #
    # What was absent without them: this file scores up to 5 downstream models
    # on 1 base representation and 1 hybrid recipe, and the result file
    # recorded neither which representations nor which configurations the other
    # 4 rank highest. 'gain_by_space' compares a weighting's gain in the base
    # representation against its gain in the hybrid, and for 4 of the 5 models
    # that hybrid was chosen for a different model.
    #
    # 0 new model fits. All four read rows the search already stored:
    # search.baselines for the representations, search.all_results for the
    # configurations.
    #
    # They go at the TOP level of `results`, never inside 'per_draw', which is
    # keyed by str(draw) and read back by _fold_scores (:773) as
    # results['per_draw'].values() -- a non-draw entry there would be walked as
    # a draw block.
    results['top2_base_reps'] = {}
    results['top2_hybrids'] = {}
    results['top2_pairs_by_model'] = {}
    results['hybrid_deltas_by_model'] = {}
    if search is None:
        results['top2_unavailable_reason'] = (
            'the pool returned no SearchResult, so there are no stored rows to '
            'rank. Every condition number in this file stands.')
    else:
        for _m in downstream_models:
            # require_full=False: a downstream model this search never scored
            # comes back a short list carrying its own shortfall_reason,
            # instead of raising before the first draw has run.
            try:
                results['top2_base_reps'][_m] = top_base_reps(
                    search, _m, 2, require_full=False)
            except Exception as _e:
                print(f"  top-2 base representations for {_m} unavailable: "
                      f"{type(_e).__name__}: {_e}")
            _th = None
            try:
                _th = top_hybrids(search, _m, 2, require_full=False)
                results['top2_hybrids'][_m] = _drop_blueprint(_th)
            except Exception as _e:
                print(f"  top-2 hybrid configurations for {_m} unavailable: "
                      f"{type(_e).__name__}: {_e}")
            # The rank-1 configuration is handed over explicitly. Called with
            # hybrid=None, hybrid_deltas picks its own and reads it as
            # _picked['selected'][0]['row'] (_shared.py:7432); a selected entry
            # from top_hybrids carries no 'row' key, so that path raises
            # KeyError('row'). Read in _shared.py this session. The entry does
            # carry 'score_mean' and 'blueprint', which is all hybrid_deltas
            # reads off the configuration it is given.
            if _th and _th['selected']:
                try:
                    results['hybrid_deltas_by_model'][_m] = hybrid_deltas(
                        search, _m, hybrid=_th['selected'][0])
                except Exception as _e:
                    print(f"  hybrid deltas for {_m} unavailable: "
                          f"{type(_e).__name__}: {_e}")
        # The same grid in one block: per downstream model, 2 base
        # representations and 2 hybrid configurations, plus 'n_distinct_units'
        # -- how many separate things would have to be BUILT once one
        # representation picked by three downstream models is counted once.
        try:
            results['top2_pairs_by_model'] = _drop_blueprint(
                select_model_rep_pairs(search, models=list(downstream_models),
                                       n_base=2, n_hybrids=2,
                                       require_full=False))
        except Exception as _e:
            print(f"  top-2 model-and-representation grid unavailable: "
                  f"{type(_e).__name__}: {_e}")
        _print_deltas(results['hybrid_deltas_by_model'], downstream_models,
                      metric_name)

    # Models that refused a per-molecule weight, discovered by being refused
    # rather than assumed. Written to the result file so a missing cell in a
    # weighted condition is traceable to the model rather than to the data.
    refuses_weight = set()

    # ── The experiment ──
    for draw in range(n_draws):
        t_draw = time.time()
        print(f"\n  === draw {draw} ===")
        ent = build_entities(labels, rep_groups, seed=draw)
        y_grade_all = ent['grade_label']
        gradeable = ent['gradeable']

        row_groups = data.get('groups')
        ent_groups = (np.asarray(row_groups)[stats['row_index']]
                      if row_groups is not None else None)

        # One row per molecule, and only the rows this experiment reads are
        # cast. On a 2,048-bit fingerprint over 30,000 rows the whole-matrix
        # cast is about 490 MB copied and then mostly discarded -- n17 records
        # the same at :335-340.
        ent_reps = {name: np.asarray(mat[stats['row_index']], dtype=np.float64)
                    for name, mat in reps.items()}
        X_base_all = ent_reps[base_rep_name]

        draw_block = {'per_label_build': {}}

        for build in label_builds:
            y_train_all, n_behind = training_labels(ent, stats, build)
            var_own, n_filled, var_floor = label_variance(stats, n_behind)
            var_cal, noise_fit, n_trusted = calibrated_variance(stats, n_behind)

            build_block = {
                'n_molecules_with_no_variance_of_their_own': n_filled,
                'variance_floor_label_units_squared': var_floor,
                'n_molecules_in_trusted_subset': n_trusted,
                # The noise function itself: measurement count -> the mean
                # variance of the molecules measured that many times. A reader
                # should be able to see whether it falls with the count.
                'fitted_variance_by_measurement_count': noise_fit,
                'per_space': {},
            }
            print(f"    label build {build!r}: "
                  f"{n_filled} molecules have no variance of their own, "
                  f"trusted subset {n_trusted}")

            # {space: {condition: {model: {fold key: score}}}}
            #
            # Keyed by the fold, NOT appended to a list. Two conditions have to
            # be compared on the SAME folds: a condition whose fit raised on
            # fold 2 appends nothing, and two lists of different lengths then
            # line fold 3 up against fold 2. The gain that comes out of that is
            # a difference between folds, which on this data is several times
            # larger than any difference between conditions.
            acc = {s: {c: {m: {} for m in downstream_models}
                       for c in CONDITIONS} for s in spaces}
            errors = {}
            tau2_seen = {s: [] for s in spaces}
            shrink_seen = {s: [] for s in spaces}
            folds_scored = 0

            splits = list(cv_split(ent_groups, n_folds, X_base_all,
                                   y_train_all))
            for fold_i, (tr, va) in enumerate(splits):
                fold_key = f'd{draw}f{fold_i}'
                va_graded = va[gradeable[va]]
                if len(va_graded) < 5:
                    print(f"      fold {fold_i}: {len(va_graded)} gradeable "
                          f"molecules held out, skipped")
                    continue
                folds_scored += 1
                y_tr = y_train_all[tr]
                y_va = y_grade_all[va_graded]
                g_tr = ent_groups[tr] if ent_groups is not None else None

                # Every feature space, all built from this fold's training
                # molecules only. One hybrid per family of stored search:
                # hybrid_for_fold reads the objective off the recipe and drops
                # the model, the metric and its direction under a geometry
                # recipe, where build_fold_hybrid refuses the pair, so one call
                # serves both families.
                fold_spaces = {}
                sc = StandardScaler()
                fold_spaces['base'] = (sc.fit_transform(X_base_all[tr]),
                                       sc.transform(X_base_all[va_graded]))
                for _sp in spaces:
                    _bp = recipe_for_space.get(_sp)
                    if _bp is None:
                        continue
                    X_tr_h, X_va_h = hybrid_for_fold(
                        {k: v[tr] for k, v in ent_reps.items()},
                        {k: v[va_graded] for k, v in ent_reps.items()},
                        # The hybrid is built on the UNTOUCHED training labels,
                        # once per fold, and all six conditions share it. So the
                        # only thing differing between two conditions is what
                        # the downstream fit was handed. Whether a recipe
                        # searched on the shrunk labels would do better is n20's
                        # question, not this one's.
                        y_tr, _bp, g_tr,
                        model_factories[downstream_models[0]](),
                        metric_fn, lower_is_better)
                    if X_tr_h is not None:
                        sc_h = StandardScaler()
                        fold_spaces[_sp] = (
                            sc_h.fit_transform(X_tr_h), sc_h.transform(X_va_h))

                for space in spaces:
                    if space not in fold_spaces:
                        continue
                    X_tr_s, X_va_s = fold_spaces[space]

                    # The out-of-fold prediction the two relabelling conditions
                    # shrink toward, computed inside this fold's TRAINING
                    # molecules only and in the space the model will be fitted
                    # in. Group-aware, because a random inner fold would put a
                    # molecule's scaffold family on both sides and deflate the
                    # residual the shrinkage keys on
                    # (noisy_labels/core.py:3-8).
                    y_shrunk = y_calib = None
                    try:
                        oof = oof_predict(X_tr_s, y_tr, g_tr,
                                          n_splits=oof_splits)
                        y_shrunk, s_own, tau2 = shrink_toward(
                            y_tr, oof, var_own[tr])
                        y_calib, _s_cal, _ = shrink_toward(
                            y_tr, oof, var_cal[tr])
                        tau2_seen[space].append(float(tau2))
                        shrink_seen[space].append(float(np.mean(s_own)))
                    except Exception as exc:
                        errors[f'{space}|oof'] = f'{type(exc).__name__}: {exc}'
                        print(f"      fold {fold_i} {space}: out-of-fold "
                              f"prediction failed: {type(exc).__name__}: {exc}")

                    for cond in CONDITIONS:
                        if cond in RELABEL_CONDITIONS:
                            y_use = y_shrunk if cond == 'shrunk' else y_calib
                            if y_use is None:
                                continue
                            w = None
                        elif cond in WEIGHTED_CONDITIONS:
                            y_use = y_tr
                            w = weights_for(cond,
                                            stats['n_measurements'][tr],
                                            var_own[tr], n_behind[tr])
                        else:
                            y_use, w = y_tr, None

                        for mname in downstream_models:
                            if w is not None and mname in refuses_weight:
                                continue
                            score, err = _fit_and_score(
                                model_factories[mname], X_tr_s, y_use,
                                X_va_s, y_va, metric_fn, w)
                            if err == 'refuses_sample_weight':
                                refuses_weight.add(mname)
                                continue
                            if score is None:
                                errors[f'{space}|{cond}|{mname}|{fold_key}'] = err
                                continue
                            acc[space][cond][mname][fold_key] = score

            for space in spaces:
                build_block['per_space'][space] = {
                    'mean_tau2_over_folds': (
                        float(np.mean(tau2_seen[space]))
                        if tau2_seen[space] else float('nan')),
                    'mean_shrinkage_over_folds': (
                        float(np.mean(shrink_seen[space]))
                        if shrink_seen[space] else float('nan')),
                    'per_condition': {
                        c: {'per_model': {
                            m: {'mean': (float(np.mean(list(v.values()))) if v
                                         else float('nan')),
                                # Keyed by the fold that produced it, so the
                                # summary can pair two conditions on the folds
                                # they both scored.
                                'fold_scores': {k: float(x)
                                                for k, x in v.items()},
                                'n_folds': len(v)}
                            for m, v in acc[space][c].items()}}
                        for c in CONDITIONS},
                }
            build_block['n_folds_scored'] = folds_scored
            if errors:
                build_block['_errors'] = errors
            draw_block['per_label_build'][build] = convert_numpy(build_block)

        results['per_draw'][str(draw)] = draw_block
        print(f"    [{time.time() - t_draw:.0f}s]")

    results['models_refusing_sample_weight'] = sorted(refuses_weight)
    results['summary'] = _summarise(results, downstream_models, label_builds,
                                    spaces, metric_name, lower_is_better)
    _print_summary(results, metric_name)

    save_experiment_results(results, output_dir, 'n19_measurement_count',
                            dataset_name)
    return results


def _fold_scores(results, build, space, cond, model):
    """Every fold score for one cell, keyed by the fold that produced it.

    Pooled across draws, and the keys carry the draw, so two conditions can be
    lined up on exactly the folds they both scored.
    """
    out = {}
    for draw in results['per_draw'].values():
        blk = draw['per_label_build'].get(build, {})
        sp = blk.get('per_space', {}).get(space)
        if not sp:
            continue
        cell = sp['per_condition'][cond]['per_model'].get(model)
        if cell:
            out.update(cell['fold_scores'])
    return out


def _summarise(results, models, label_builds, spaces, metric_name,
               lower_is_better):
    """One row per (label build, space, model): every condition against untouched.

    THE HEADLINE IS A COUNT OF FOLDS, not a mean, and the reason is n18's at
    :83-88. Averaging hides a condition that wins big on one fold and loses on
    four. It also hides something worse here: the held-out molecules are graded
    against a consensus of real measurements, so a fold holding out the
    badly-measured molecules scores worse whatever condition is running. MEASURED
    on the smoke's own data: the spread of the raw scores across folds was 0.26
    to 0.32 while every difference between conditions was 0.01 to 0.18. Read
    unpaired, none of them is a difference.

    So every comparison here is PAIRED: the same fold, the same molecules, the
    same grading labels, two conditions. What varies between folds is common to
    both and cancels.

    Positive gain always means better, whichever way the metric runs.
    """
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': 'folds won against untouched, not mean gain',
           'comparison': 'paired per fold',
           'per_label_build': {}}

    for build in label_builds:
        build_row = {'per_space': {}}
        for space in spaces:
            space_row = {'per_model': {}}
            for m in models:
                base = _fold_scores(results, build, space, 'untouched', m)
                base_vals = list(base.values())
                row = {'untouched_mean': (float(np.mean(base_vals))
                                          if base_vals else float('nan')),
                       'untouched_n_folds': len(base),
                       'per_condition': {}}
                for cond in CONDITIONS:
                    got = _fold_scores(results, build, space, cond, m)
                    shared = sorted(set(base) & set(got))
                    gains = np.array(
                        [(base[k] - got[k]) if lower_is_better
                         else (got[k] - base[k]) for k in shared],
                        dtype=float)
                    gains = gains[np.isfinite(gains)]
                    # A fold is only WON when the two fits actually scored
                    # differently. Two conditions can be the identical fit --
                    # weighting by a measurement count every molecule shares
                    # gives every molecule the same weight -- and the gain is
                    # then floating-point dust. MEASURED on the smoke: a mean
                    # gain of -1.4e-17 over eight folds, one of which came out
                    # positive. Counting that as a win says a condition beat
                    # untouched when it WAS untouched.
                    scale = max(abs(row['untouched_mean']), 1.0)
                    tol = 1e-9 * scale
                    row['per_condition'][cond] = {
                        'mean': (float(np.mean(list(got.values())))
                                 if got else float('nan')),
                        'n_folds': len(got),
                        # The folds BOTH this condition and untouched scored.
                        # Everything below is over those and only those.
                        'n_folds_paired': int(gains.size),
                        'folds_won': int((gains > tol).sum()),
                        'folds_lost': int((gains < -tol).sum()),
                        # Folds where the two scored the same number. A large
                        # count here means the two conditions are one fit.
                        'folds_tied': int((np.abs(gains) <= tol).sum()),
                        'tie_tolerance': float(tol),
                        'mean_gain': (float(gains.mean()) if gains.size
                                      else float('nan')),
                        'spread_over_fold_gains': (float(gains.std())
                                                   if gains.size
                                                   else float('nan')),
                        'fold_gains': [float(x) for x in gains],
                        # A gain that is not larger than the spread of the
                        # per-fold gains behind it is not a gain. Stated as a
                        # comparison of two measured numbers, not a test: at
                        # three draws over five folds there is no replicate
                        # count that would support a confidence interval, and
                        # the suite's standing rule is raw spread only.
                        'gain_exceeds_its_own_spread': bool(
                            gains.size > 0
                            and gains.mean() > gains.std()),
                    }
                space_row['per_model'][m] = row
            build_row['per_space'][space] = space_row
        out['per_label_build'][build] = build_row

    # Does a weighting help MORE in a searched space than in a fixed one? That
    # is the question n02 asks of importance weights and nothing has asked of a
    # measurement count. Only defined where both spaces ran.
    def _cross_for(hybrid_space):
        """One hybrid space against the base representation, per condition."""
        cross = {}
        for build in label_builds:
            per_cond = {}
            for cond in CONDITIONS:
                per_model = {}
                for m in models:
                    b = out['per_label_build'][build]['per_space']['base'][
                        'per_model'][m]['per_condition'][cond]['mean_gain']
                    h = out['per_label_build'][build]['per_space'][
                        hybrid_space]['per_model'][m][
                            'per_condition'][cond]['mean_gain']
                    per_model[m] = {
                        'gain_in_base': b,
                        'gain_in_hybrid': h,
                        'hybrid_minus_base': (float(h - b)
                                              if np.isfinite(b)
                                              and np.isfinite(h)
                                              else float('nan')),
                    }
                per_cond[cond] = per_model
            cross[build] = per_cond
        return cross

    # Every hybrid space this run built, each against the base representation.
    # 2026-09-19: there can be more than one, so this is keyed by space name.
    # 'gain_by_space' below keeps the shape it had, holding the predictive
    # space alone, so nothing already reported moves.
    _hybrid_spaces = [sp for sp in spaces if sp != 'base']
    if _hybrid_spaces and 'base' in spaces:
        out['gain_by_hybrid_space'] = {sp: _cross_for(sp)
                                       for sp in _hybrid_spaces}
    if 'hybrid' in spaces and 'base' in spaces:
        cross = _cross_for('hybrid')
        out['gain_by_space'] = cross
        out['gain_by_space_note'] = (
            "Each entry is one condition's mean paired gain over untouched in "
            'the base representation and in the hybrid, and the difference '
            'between the two. It says whether the weighting or the shrinkage is '
            'worth more where the features were chosen. The features were '
            'chosen without any weight in both spaces. '
            "'gain_by_space' holds the space built from the predictive search "
            "alone; 'gain_by_hybrid_space' holds the same table for every "
            'hybrid space this run built, keyed by space name.')
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per fold against untouched)")
    if not results.get('hybrid_available', True):
        print("  no hybrid recipe was available; base representation only")
    if results.get('models_refusing_sample_weight'):
        print(f"  refused a per-molecule weight, so absent from the three "
              f"weighted conditions: "
              f"{', '.join(results['models_refusing_sample_weight'])}")
    for build, brow in s['per_label_build'].items():
        what = ('one measurement per molecule' if build == 'single'
                else 'the mean of every measurement')
        print(f"\n  training label = {what}")
        for space, srow in brow['per_space'].items():
            print(f"    {space}:")
            for m, row in srow['per_model'].items():
                if not np.isfinite(row['untouched_mean']):
                    print(f"      {m}: no fold scored")
                    continue
                print(f"      {m}  untouched {row['untouched_mean']:.4f} "
                      f"over {row['untouched_n_folds']} folds")
                for cond in CONDITIONS:
                    if cond == 'untouched':
                        continue
                    c = row['per_condition'][cond]
                    if not c['n_folds_paired']:
                        print(f"        {cond:19s} no fold scored both")
                        continue
                    mark = '  >spread' if c['gain_exceeds_its_own_spread'] else ''
                    print(f"        {cond:19s} won {c['folds_won']} of "
                          f"{c['n_folds_paired']} folds, "
                          f"mean {c['mean_gain']:+.4f} "
                          f"(spread {c['spread_over_fold_gains']:.4f}){mark}")
    print("\n  The count of folds won is the headline. A mean gain no larger "
          "than the")
    print("  spread of the per-fold gains behind it is not a gain.")
    if 'gain_by_space' in s:
        print("  'gain_by_space' in the result file holds the same gains in "
              "both spaces and the difference.")


def main():
    parser = create_experiment_parser(
        'N19: the measurement count and spread as weight and as shrinkage')
    parser.add_argument('--n-draws', type=int, default=N_DRAWS,
                        help='How many times the training measurement is re-drawn')
    parser.add_argument('--label-builds', type=str, default=None,
                        help="Comma-separated: single, averaged")
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {'n_draws': args.n_draws}
    if args.label_builds:
        kw['label_builds'] = [x.strip() for x in args.label_builds.split(',')]
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
