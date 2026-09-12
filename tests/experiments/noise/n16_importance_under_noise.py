#!/usr/bin/env python3
"""N16: Does feature importance move when the labels get noisier.

Paper: Noise (trains NO downstream models).

THE QUESTION. Do certain features become more or less relevant as the labels get
noisier. Every other noise experiment asks what noise does to a SCORE. This one asks
what noise does to the ORDERING OF FEATURES that the score is built on, which nothing
in the suite records. n12 computes the vector this needs — once per level, per fold,
per scoring method — uses it to scale features before a distance, and discards it
(n12_importance_weighted_distance.py:214, :239, :164). n15 records the version one
level up: which REPRESENTATIONS the clean-label and noisy-label searches picked. That
is representations, not features.

WHAT IT OUTPUTS, per (representation, scoring method, condition, level, fold, noise
replicate):
  1. n_scored — how many features carry any importance at all. This is a first-order
     answer on its own: measured on caco2 fold 0 with clean labels, lightgbm_gain
     scores 209 of ecfp4's 2048 features while deeplift scores 1777 of the same 2048.
  2. Rank agreement against the SAME cell at level 0.0 — Spearman and Kendall tau_b,
     each over all features and over the features either level scored non-zero, plus
     top-k set overlap. Reused from f30_analyse._pair_stats.
  3. Turnover in the top set — which features left the clean top set and which
     entered, plus Nogueira phi-hat set stability over the whole level sweep. The top
     set is capped at how many features the scorer actually ranked, never at the
     representation's width: lightgbm_gain scores 39 of diff_maccs's 167 features on
     buchwald_hartwig, so an uncapped top-50 overlap is 1.00 at every level (tied
     zeros, ordered by index, identical on both sides) while the scored ranking moves
     from Spearman 0.980 to 0.920.

TWO ARMS, AND ALL THREE ROWS ARE REPORTED FOR BOTH.
  base   — every representation in the pool, scored whole. The feature set is a fixed
           matrix, so a vector at one level indexes the same features as at another
           and rows 2 and 3 are defined by construction.
  hybrid — the hybrid the recipe builds at that level, via build_fold_hybrid. Row 1 is
           how many of its features carry any importance. Row 2 needs care, because
           two levels select DIFFERENT features and the two vectors cannot be lined up
           elementwise: every hybrid column is keyed back to its
           (representation, source index) through feat_info[rep]['selected_indices']
           and the ranking is correlated over the features BOTH levels selected, with
           the size of that overlap reported beside every correlation. Row 3 is the
           turnover in the selected set itself, per representation.

THE RECIPE IS HELD FIXED at the clean-label precomputed one, at every level. Three
reasons. Re-searching per level is not affordable — measured off
/Volumes/seagate/kirby_cache/result_db/logd.db, a partial logd search (the pruned 10,
five budgets from 25 to 300, 1,275 stored configurations) sums to 52,002 seconds of
fitting, and the default grid is 2,400 configurations over budgets up to 2000.
Re-searching also moves the budget, the allocation method, the importance method and
the internal evaluator, so feature drift and recipe drift would arrive mixed; n15
already owns recipe drift. And this experiment makes NO performance claim, so the
oracle caveat n11 and n12 carry — the recipe was chosen having seen clean labels —
does not transfer to it. Every result row says so.

THE TWO ARMS SCORE ON DIFFERENT ROWS, by design. The base arm scores the fold's whole
training block. create_hybrid, inside build_fold_hybrid, scores its internal training
part only, for the reason at hybrid.py:5686-5691 — importance computed over the
held-out part too let "keep the base as it is" win more often than it should. So each
arm is compared only against its own level-0.0 row, never against the other, and the
base arm's vector is NEVER passed in as precomputed_importance.

NOISE. This does NOT use noise_helpers.inject_train_labels, and as of 2026-09-12 only
ONE of the three reasons it did not still holds.

  GONE: the fold no longer reaches either seed, and the amount is no longer solved
  against the fold's own training labels. Both were fixed on 2026-09-12 —
  inject_train_labels now draws once over the whole label column and indexes the
  training rows out, and noise_spec_for no longer takes a fold. That is what
  RERUN_PLAN.md 3.3b ruled on 2026-09-04. Measured after the change on 600 molecules
  over five scaffold folds: 0 of 600 molecules carry more than one corrupted value,
  in gaussian, outlier_p10, grouped_wider, grouped_shifted and censoring alike.

  STANDS: the level still enters noise_seed (noise_helpers.py, `seed = selection_seed
  + level*1000*7919`), so each level is a fresh shape draw. Which MOLECULES are hit
  nests across levels — selection_seed is level-free — but the directions do not.
  This file needs the directions to nest, because every comparison it makes is
  against its own level-0.0 vector, so it keeps its own injection.
  Measured on 400 molecules through that route: the error at level 0.15 and at level
  0.30 have Spearman +0.015 (gaussian) and +0.037 (outlier_p10), signs agreeing on 200
  and 196 of 400 — chance.
Measured on 400 molecules through that route: the error at level 0.15 and at level
0.30 have Spearman +0.015 (gaussian) and +0.037 (outlier_p10), signs agreeing on 200
and 196 of 400 — chance.

Every comparison here is against the level-0.0 vector, so the levels have to NEST:
same molecules, same directions, more magnitude. Both seeds are therefore held
constant across the level grid and varied only across noise replicates. Measured on
400 molecules with both held: the error at level 0.30 is the error at level 0.15 times
2.000000 on every molecule, minimum equal to maximum, in gaussian, student_t_nu3,
laplace and outlier_p10. The assertion is re-run at run time rather than trusted.

SCOPE: regression only, for two reasons in code. calibrate_flip_probability binary
searches the flip rate with a tolerance of 0.01 and no zero case
(noiseInject/calibration.py:78-93), so a classification level of 0.0 returns about
0.0078 and flips roughly 0.8% of training labels — the reference row is not clean. And
the classification injector takes one seed only (core.py:965, noise_spec.py:420-421),
so selection_seed is dropped and the levels cannot nest at all.

Usage:
    python n16_importance_under_noise.py --dataset logd --mode intermediary
    python n16_importance_under_noise.py --dataset caco2 --mode full --replicates 3
    python n16_importance_under_noise.py --dataset logd --noise-condition outlier_p10
    python n16_importance_under_noise.py --dataset logd --arm base
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser, parse_rep_args,
    save_experiment_results, convert_numpy, cv_split, get_domain_metric,
    get_best_search_for_downstream, build_fold_hybrid, get_model_factories,
)
from experiments.dataset_policy import (
    noise_levels as dataset_noise_levels, noise_groups_for, noise_dose_units,
)
# The four rank-agreement statistics already in this repository, with their
# provenance recorded at f30_analyse.py:9-19. _pair_stats reports every feature-rank
# statistic TWICE — over all features and over the features either side scored
# non-zero — and the second is the honest one here for the same reason it is there:
# lightgbm_gain leaves most features at exactly 0.0 and np.argsort orders tied zeros
# by index, so an all-features Spearman measures how two tied-zero blocks line up.
# f30_analyse imports stdlib, numpy and scipy only, so this pulls in no torch.
from experiments.framework.f30_analyse import _pair_stats, nogueira_stability
from kirby.hybrid import compute_feature_importance
from kirby.noise_spec import (
    NoiseSpec, calibrate, make_noisy_labels_verbose, noise_provenance_row,
)


# The three the engine can actually pick: find_best_hybrid.py:3240
# DEFAULT_IMPORTANCE_METHODS. f30's other 28 scorers are candidates under
# evaluation, not things a recipe can select, so a drift measured in one of them
# would not describe any hybrid this repo builds.
IMPORTANCE_METHODS = ['deeplift', 'lightgbm_gain', 'treeshap']

# Gaussian is the reference shape and what n09, n11, n12 and n15 each inject. The
# other ten regression conditions are reachable one run each through
# --noise-condition, and the condition is in BOTH the output filename stem and every
# npz key, so a second condition does not overwrite the first (the smaller of the two
# fixes NOISE_AUDIT_2026_09_10.md section 10 decision 4 lists).
NOISE_CONDITION = 'gaussian'

# How many independent noise draws per level. One draw gives no run-to-run spread, so
# a level-to-level movement has nothing to clear. Each replicate re-seeds BOTH seeds,
# so it is a different corruption of the same molecules at every level of that
# replicate's own nested ladder.
N_REPLICATES = 3

# The size of the "top" whose membership is tracked. 100 features is the smallest
# canonical budget above the engine's own default of 100 (create_hybrid budget=100,
# hybrid.py:5288), so the tracked set is the size of a real allocation rather than an
# arbitrary slice.
TOP_K = 100

# Below this many features selected at BOTH levels, a rank correlation between the
# hybrid's two orderings is not reported. _pair_stats itself returns nan under 3
# (f30_analyse.py:70-73); 10 is the point below which the correlation says more about
# which handful survived than about how they were ranked. Folds under it are counted
# and named, never dropped silently.
MIN_SHARED_FOR_RANK = 10

# Base seeds for the two independent streams. Held apart because they answer different
# questions: SELECTION_BASE decides WHICH molecules are corrupted and must not vary
# with the level, SHAPE_BASE decides HOW HARD and must not either, or the levels stop
# nesting. 42 and 1337 mirror the training/validation stream split at
# tests/noise_column.py:115.
SELECTION_BASE = 42
SHAPE_BASE = 1337

# Tolerance on the nesting assertion. The ratio is a float division of two products of
# the same unit draw, so it is exact to rounding; 1e-9 catches a changed seed formula
# while tolerating the last bit.
NEST_RTOL = 1e-9


def _specs_for_ladder(condition, levels, replicate, dataset_name):
    """One NoiseSpec per level, seeded so the levels NEST within this replicate.

    Both seeds are constant across `levels` and both move with `replicate`. That is
    the whole point: NoiseSpec's own docstring (noise_spec.py:159-165) says a level
    sweep that varies `noise_seed` per level changes the affected molecules at every
    level, so the curve mixes more corruption with a different corruption. Holding
    `noise_seed` as well as `selection_seed` goes one step further and fixes the unit
    shape draw too, so a level only SCALES it.

    zlib.crc32, not hash(): Python randomises string hashing per process, so hash()
    here would draw different noise on every run while this docstring promised the
    opposite. Same reasoning as noise_column.py:117-119.
    """
    import zlib
    digest = zlib.crc32(f"{dataset_name}|{condition}".encode('utf-8'))
    sel = (SELECTION_BASE * 1000003 + digest
           + int(replicate) * 104729) & 0x7FFFFFFF
    shp = (SHAPE_BASE * 1000003 + digest
           + int(replicate) * 15485863) & 0x7FFFFFFF
    units = noise_dose_units(dataset_name)
    return [NoiseSpec(strategy=condition, target_effective_noise=float(lv),
                      noise_seed=shp, selection_seed=sel, dose_units=units)
            for lv in levels]


def _inject_ladder(labels, specs, levels, noise_groups):
    """Corrupt the FULL label column once per level, and check the ladder nests.

    The amount is solved on the full clean label column (kirby.noise_spec.calibrate,
    resolved once at noise_spec.py:325-330) and the realisation covers every row, so
    each fold indexes its own training positions out of one draw. A molecule's
    corruption is then a property of the molecule, not of which fold it landed in —
    the property NOISE_AUDIT_2026_09_10.md section 1.1 measured the suite as lacking
    (600 of 600 molecules carrying more than one corrupted value).

    Returns (per_level, provenance) where per_level[level] = {'y': full noisy column,
    'epsilon': per-row error, 'realised': delivered amount in label units}.

    RAISES if the ladder does not nest. The whole experiment compares a vector at one
    level against the vector at level 0.0, and if the higher level is a fresh draw
    rather than the same draw scaled, that comparison has two things moving in it and
    cannot be read. Better to stop than to write rows nobody can interpret.
    """
    y = np.asarray(labels, dtype=float).ravel()
    per_level, provenance = {}, {}
    for spec, level in zip(specs, levels):
        magnitude = calibrate(y, spec, is_clf=False)
        res = make_noisy_labels_verbose(y, spec, magnitude=magnitude,
                                        groups=noise_groups)
        row = noise_provenance_row(res, spec)
        per_level[level] = {
            'y': np.asarray(res.y_noisy, dtype=float),
            'epsilon': np.asarray(res.epsilon, dtype=float),
            'realised': row.get('realised_dose_label_units'),
            'target': row.get('target_dose_label_units'),
        }
        provenance[spec.spec_id] = row

    # --- the nesting check ------------------------------------------------------
    # noiseInject warns on a draw that misses its target amount (DoseWarning,
    # core.py:653-661) and find_best_hybrid.py:74 discards every warning in the
    # process, re-admitting KirbyWarning only (_warn.py:39) — _shared.py:52 does the
    # same. So the delivered amount is PRINTED here; a warning cannot reach the log.
    nonzero = [lv for lv in levels if float(lv) > 0.0]
    if nonzero:
        ref_lv = min(nonzero)
        ref_eps = per_level[ref_lv]['epsilon']
        moved = np.abs(ref_eps) > 0
        for lv in nonzero:
            want = float(lv) / float(ref_lv)
            got = per_level[lv]['epsilon'][moved] / ref_eps[moved]
            if not np.allclose(got, want, rtol=NEST_RTOL):
                raise RuntimeError(
                    f"the noise ladder does not nest: at level {lv} the error is "
                    f"{got.min():.9f} to {got.max():.9f} times the error at level "
                    f"{ref_lv}, and it must be {want:.9f} on every molecule. Every "
                    f"comparison this experiment makes is against level 0.0, so a "
                    f"level that is a fresh draw rather than the same draw scaled "
                    f"puts two moving things in one number. Check that "
                    f"_specs_for_ladder still holds noise_seed AND selection_seed "
                    f"constant across the level grid.")
    return per_level, provenance


def _score_one(X, y, method, seed):
    """One representation's per-feature importance vector, on the rows given.

    compute_feature_importance fits an estimator on (X, y) — the labels are its
    second positional argument (hybrid.py:4253) — so under label noise it must be
    handed the NOISY training labels, which is what the engine itself does and says
    at find_best_hybrid.py:5744-5747. Returns the vector and the seconds it took.

    Only `random_state` is passed through. The regression/classification split is
    read off the labels inside (`_is_classification_task`, hybrid.py:4308) and there
    is no `task_type` or `n_jobs` keyword — unknown keys are read with kwargs.get and
    silently ignored, which would make a typo look like it had taken effect.

    A failure RAISES. The silent drop at find_best_hybrid.py:4473 would turn a scorer
    that cannot run on a representation into a missing row that reads as "no drift".
    """
    t0 = time.time()
    out = compute_feature_importance({'x': X}, y, method=method,
                                     random_state=seed)
    v = np.asarray(out['x'], dtype=float).ravel()
    if v.shape[0] != X.shape[1]:
        raise ValueError(
            f"{method} returned {v.shape[0]} scores for {X.shape[1]} features")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{method} returned non-finite scores")
    return v, time.time() - t0


def _prescreen_width(X, cap=2000):
    """How many of this representation's features can be scored at all.

    compute_feature_importance pre-screens anything wider than 2000 features to its
    top-variance 2000 and writes exactly 0.0 for the rest (hybrid.py:4359, :4379-4383
    for treeshap; the same cap on lightgbm_gain and xgboost_gain). The variance is
    read off the feature matrix and never off the labels (hybrid.py:4380), so the
    zeroed block is byte-identical at every noise level and cannot masquerade as
    drift. Recorded per representation so a reader can see which rows carry one.
    """
    return int(min(X.shape[1], cap))


def _usable_methods(methods, X, y):
    """Drop the scoring methods that cannot run here, and say which, once.

    Each method is tried once on a small slice before the sweep starts. A method that
    raises on the slice raises on every cell, so without this check a broken one
    produces one failure row per (representation, level, fold, replicate) and the
    result reads as if the method had run and found nothing.

    Measured 2026-09-12 on this machine: `treeshap` cannot run at all — `shap` 0.47.2
    imports `transformers.modeling_tf_utils`, which calls `MessageFactory.GetPrototype`,
    deleted in protobuf 6.33.6. It filled 96 of one run's cells with the same import
    error. It works wherever the f30 result files were produced, so this is an
    environment state, not a defect in the scorer.

    Returns (usable, unavailable) where unavailable is {method: error string}.
    """
    n = min(60, X.shape[0])
    w = min(20, X.shape[1])
    Xp = np.asarray(X)[:n, :w]
    yp = np.asarray(y)[:n]
    usable, unavailable = [], {}
    for m in methods:
        try:
            _score_one(Xp, yp, m, seed=0)
            usable.append(m)
        except Exception as exc:
            unavailable[m] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return usable, unavailable


def _hybrid_column_sources(feat_info, n_cols):
    """Where each column of the built hybrid came from, in column order.

    Returns a list of (source, index) of length n_cols, or None when it cannot be
    reconciled with the matrix that was actually built.

    The order is `apply_feature_selection`'s (hybrid.py:5140-5149): iterate
    `feature_info` in insertion order, skip the metadata keys, and take each
    representation's `selected_indices` in stored order. Extra admitted features are
    appended after the base block, in `augmentation_info` insertion order
    (hybrid.py:5216-5221), and are labelled `aug:<donor>` so they can never be
    matched against a base feature of the same index.

    Reconciled rather than trusted, and the cell is dropped rather than guessed when
    the count disagrees. A mapping that is off by one would silently compare one
    feature's importance at level 0.0 against a different feature's at level 0.5, and
    the resulting rank correlation would look like a real number.
    """
    sources = []
    for name, info in feat_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        for idx in np.asarray(info['selected_indices'], dtype=int).ravel():
            sources.append((str(name), int(idx)))
    aug = feat_info.get('augmentation_info') or {}
    if isinstance(aug, dict):
        for donor, info in aug.items():
            if not isinstance(info, dict):
                continue
            idxs = info.get('selected_indices', info.get('indices'))
            if idxs is None:
                continue
            for idx in np.asarray(idxs, dtype=int).ravel():
                sources.append((f'aug:{donor}', int(idx)))
    if len(sources) != int(n_cols):
        return None
    return sources


def _turnover(a, b, k):
    """Which of the top-k features left, and which arrived, going from a to b.

    TWO top-k sizes, and the second is the one to read.

    `top_k` is the size asked for, capped at the representation's width. That is the
    figure f30's `_pair_stats` uses and it is kept so the two are comparable — but it
    is unreadable whenever the scorer left more than (width - k) features at exactly
    zero. Measured on buchwald_hartwig: lightgbm_gain scores 39 of diff_maccs's 167
    features, so a top-50 set is those 39 plus 11 tied zeros that np.argsort orders by
    index, identically on both sides. The overlap comes back 1.00 at every noise level
    and reads as "nothing moved" when the scored ranking had in fact moved (Spearman
    0.980 down to 0.920 over the same levels).

    `top_k_effective` caps k at the number of features the REFERENCE side actually
    scored, so the set contains only features the scorer ranked. Where a scorer
    touches everything the two figures are identical.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_scored_ref = int((a != 0).sum())
    out = {}
    for tag, kk in (('', int(min(k, len(a)))),
                    ('_effective', int(min(k, n_scored_ref)))):
        if kk <= 0:
            out.update({f'top_k{tag}': 0, f'n_left{tag}': 0, f'n_entered{tag}': 0,
                        f'retained_fraction{tag}': float('nan')})
            continue
        sa = set(np.argsort(a)[-kk:].tolist())
        sb = set(np.argsort(b)[-kk:].tolist())
        out.update({
            f'top_k{tag}': kk,
            f'n_left{tag}': int(len(sa - sb)),
            f'n_entered{tag}': int(len(sb - sa)),
            # The denominator is the reference side's top set, and it is named here
            # so the number is never read as a fraction of the representation.
            f'retained_fraction{tag}': float(len(sa & sb) / kk),
        })
    out['n_scored_reference'] = n_scored_ref
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N16: feature importance across noise levels. Trains no downstream models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N16: FEATURE IMPORTANCE ACROSS NOISE LEVELS")
    print("=" * 70)

    domain = data.get('domain', 'molecular')
    task_type = data['task_type']
    labels = np.asarray(data['labels'], dtype=float).ravel()
    groups = data.get('groups')
    reps = data.get('reps') or {}
    augmentations = data.get('augmentations')
    n = len(labels)

    # --- Scope guards (skip markers, not errors) --------------------------------
    if task_type != 'regression':
        print("  SKIP: classification — a requested level of 0.0 still flips about "
              "0.8% of training labels (noiseInject/calibration.py:78-93), so there "
              "is no clean reference row, and the classification injector drops "
              "selection_seed so the levels cannot nest.")
        return {'skipped': True,
                'reason': 'classification has no clean reference level and cannot nest',
                'policy': 'regression_only', 'domain': domain, 'n_samples': n}
    if len(reps) < 2:
        print(f"  SKIP: {len(reps)} representations loaded — nothing to compare.")
        return {'skipped': True, 'reason': f'only {len(reps)} representations',
                'policy': 'needs_two_reps', 'domain': domain, 'n_samples': n}

    n_folds = config['n_splits']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    condition = kwargs.get('noise_condition', NOISE_CONDITION)
    levels = kwargs.get('levels') or kwargs.get('sigmas')
    if levels is None:
        levels = list(dataset_noise_levels(dataset_name, strategy=condition))
    levels = sorted(float(lv) for lv in levels)
    n_replicates = int(kwargs.get('replicates', N_REPLICATES))
    arm = kwargs.get('arm', 'both')
    top_k = int(kwargs.get('top_k', TOP_K))
    imp_methods = list(kwargs.get('importance_methods') or IMPORTANCE_METHODS)

    # Every comparison below is against the level-0.0 vector, so a grid without it —
    # or with only one level — has no answer in it. The smoke harness passes
    # levels=[0.3] under KIRBY_SMOKE_N (_smoke_harness.py:233), which is exactly this
    # case, and a skip naming the grid is more use than rows that cannot be read.
    if 0.0 not in levels or len(levels) < 2:
        print(f"  SKIP: levels={levels} — every comparison is against level 0.0, so "
              f"the grid needs 0.0 and at least one level above it.")
        return {'skipped': True, 'reason': f'grid has no usable reference: {levels}',
                'policy': 'needs_zero_level_and_one_more',
                'levels': levels, 'domain': domain, 'n_samples': n}

    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    spec_probe = NoiseSpec(strategy=condition, target_effective_noise=0.0)
    if spec_probe.needs_groups and noise_groups is None:
        print(f"  SKIP: {condition} needs one scaffold group per molecule and this "
              f"dataset exposes no SMILES.")
        return {'skipped': True, 'reason': f'{condition} needs scaffold groups',
                'policy': 'condition_needs_groups', 'domain': domain, 'n_samples': n}

    rep_order = sorted(reps)

    # Which pool actually loaded, by name. The antibody and reaction loaders drop a
    # failed representation with a warning rather than an error (_shared.py:2925,
    # :3096), so a run can quietly measure nine representations and report as if it
    # measured ten. f30 checks this for the same reason (f30_importance_methods.py:
    # 113-128). Recorded rather than raised, because a missing representation still
    # leaves every other one's drift readable.
    pool_expected, pool_missing, pool_extra = None, [], []
    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
        pool_expected = list((DATASET_REPS.get(dataset_name) or {}).get('general') or [])
    except Exception as exc:
        print(f"  NOTE: could not read the signed-off pool: {type(exc).__name__}: {exc}")
    if pool_expected:
        pool_missing = sorted(set(pool_expected) - set(rep_order))
        pool_extra = sorted(set(rep_order) - set(pool_expected))
        if pool_missing or pool_extra:
            print(f"  NOTE: loaded pool differs from the signed-off "
                  f"{len(pool_expected)}: missing={pool_missing} extra={pool_extra}")

    print(f"  Dataset: {dataset_name} ({domain}, n={n}, metric={metric_name})")
    print(f"  Representations: {len(rep_order)} — "
          f"{ {r: int(reps[r].shape[1]) for r in rep_order} }")
    # Drop any scoring method that cannot run in this environment, before the sweep
    # rather than once per cell. See _usable_methods.
    _probe_X = reps[rep_order[0]]
    imp_methods, imp_unavailable = _usable_methods(imp_methods, _probe_X, labels)
    if imp_unavailable:
        for m, err in imp_unavailable.items():
            print(f"  SCORING METHOD UNAVAILABLE HERE — {m}: {err}")
        print(f"  Continuing with {len(imp_methods)}: {imp_methods}")
    if not imp_methods:
        print("  SKIP: no scoring method can run in this environment.")
        return {'skipped': True,
                'reason': 'no importance method could run',
                'policy': 'no_usable_importance_method',
                'importance_methods_unavailable': imp_unavailable,
                'domain': domain, 'n_samples': n}
    print(f"  Scoring methods: {imp_methods}")
    print(f"  Condition: {condition}   levels: {levels}")
    print(f"  Noise replicates: {n_replicates}   folds: {n_folds}   top_k: {top_k}")
    print(f"  Arm: {arm}")

    # --- the recipe, held fixed at every level ----------------------------------
    # From the CLEAN-LABEL precompute that noise_helpers.run_all builds once per
    # dataset before any experiment runs (noise_helpers.py:351-367), so nine other
    # experiments in the same job share the cost of it. Held FIXED across the level
    # grid on purpose: re-searching per level would move the budget, the allocation
    # method, the importance method and the internal evaluator too, and feature drift
    # and recipe drift would arrive mixed. n15 measures recipe drift directly.
    bp, recipe_note = None, None
    if arm in ('hybrid', 'both'):
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            dataset_name=dataset_name, task_type=task_type,
            data=data,
        )
        bp = meta['best_config']['blueprint'] if meta else None
        if bp is None:
            print("  NOTE: no recipe in the pool — running the base arm only.")
            arm = 'base'
        else:
            recipe_note = meta.get('selection_note')
            print(f"  Recipe: reps={sorted(getattr(bp, 'reps', []) or [])} "
                  f"budget={bp.budget} allocation={bp.allocation_method} "
                  f"importance={bp.importance_method} "
                  f"evaluator={bp.internal_evaluator}")

    # The downstream estimator instance the hybrid will be fed. create_hybrid hands it
    # to the admission step, and WITHOUT metric_fn alongside it every allocation score
    # comes back -inf silently (_shared.py:4471-4480), so the two travel together.
    model_inst = None
    if arm in ('hybrid', 'both'):
        model_inst = get_model_factories(['rf'], task_type=task_type,
                                         n_samples=n)['rf']()

    results = {
        'experiment': 'n16_importance_under_noise',
        'dataset': dataset_name,
        'domain': domain,
        'n_samples': n,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'trains_downstream_models': False,
        'arm': arm,
        'noise_condition': condition,
        'levels': levels,
        'dose_units': noise_dose_units(dataset_name),
        'n_replicates': n_replicates,
        'n_folds': n_folds,
        'n_folds_scored': None,   # set once the folds are built, below
        'top_k': top_k,
        'importance_methods': imp_methods,
        'importance_methods_unavailable': imp_unavailable,
        'rep_order': rep_order,
        'rep_widths': {r: int(reps[r].shape[1]) for r in rep_order},
        'rep_prescreen_width': {r: _prescreen_width(reps[r]) for r in rep_order},
        'pool_expected': pool_expected,
        'pool_missing': pool_missing,
        'pool_extra': pool_extra,
        'n_noise_groups': int(n_noise_groups),
        'largest_noise_group_share': float(largest_group_share),
        # Stated on the row so a reader does not import n11's and n12's caveat: the
        # recipe was chosen on clean labels, and that is the CONTROL here, not a leak,
        # because this experiment reports no performance number for it to inflate.
        'recipe_lineage': 'clean_label_precompute_held_fixed_across_levels',
        'recipe_selection_note': recipe_note,
        'oracle_caveat_applies': False,
        'oracle_caveat_note': (
            'The recipe was selected on clean labels and held fixed at every level, '
            'which is what makes the level-to-level comparison controlled. No '
            'performance number is reported for the hybrid, so the best-case-hybrid '
            'warning n11 and n12 carry does not transfer to this result.'),
        'blueprint': (None if bp is None else {
            'reps': sorted(getattr(bp, 'reps', []) or []),
            'allocation_method': bp.allocation_method,
            'importance_method': bp.importance_method,
            'budget': bp.budget,
            'internal_evaluator': bp.internal_evaluator,
            'augmentation_strategy': getattr(bp, 'augmentation_strategy', None),
        }),
        'noise_provenance': {},
        'dose_check': [],
        'per_replicate': {},
        'scoring_seconds': [],
        'errors': [],
    }

    # Folds come from the CLEAN labels and are reused at every level. cv_split uses
    # GroupKFold on regression and never reads the labels (_shared.py:1850-1861), so
    # this is identity rather than choice — but it is asserted below rather than
    # trusted, because a changed splitter would silently make every level a different
    # partition and the drift unreadable.
    sample_X = reps[rep_order[0]]
    folds = list(cv_split(groups, n_folds, sample_X, labels))
    # --max-folds keeps the same splitter and stops early (get_config puts it in the
    # config at _shared.py:6377-6378). Honoured here so a shortened run reports the
    # folds it actually scored instead of silently scoring all of them.
    max_folds = config.get('max_folds')
    if max_folds:
        folds = folds[:int(max_folds)]
    results['n_folds_scored'] = len(folds)
    fold_signature = [(tr.tolist(), va.tolist()) for tr, va in folds]

    t_start = time.time()
    for replicate in range(n_replicates):
        specs = _specs_for_ladder(condition, levels, replicate, dataset_name)
        per_level, prov = _inject_ladder(labels, specs, levels, noise_groups)
        results['noise_provenance'].update(prov)

        for lv in levels:
            pl = per_level[lv]
            asked = pl['target']
            got = pl['realised']
            # Printed, not warned: see _inject_ladder.
            print(f"  [r{replicate} level {lv:g}] delivered "
                  f"{'nan' if got is None else f'{got:.6f}'} label units against a "
                  f"target of {'nan' if asked is None else f'{asked:.6f}'}")
            results['dose_check'].append({
                'replicate': replicate, 'level': lv,
                'target_dose_label_units': asked,
                'realised_dose_label_units': got,
            })

        rep_block = {}
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if (train_idx.tolist(), val_idx.tolist()) != fold_signature[fold_idx]:
                raise RuntimeError(
                    f"fold {fold_idx} moved between noise levels. The folds are made "
                    f"from the clean labels once and every level must reuse them, or "
                    f"a vector at one level is scored on different rows from the "
                    f"vector it is compared against.")

            train_reps = {r: reps[r][train_idx] for r in rep_order}
            train_augs = ({k: v[train_idx] for k, v in augmentations.items()}
                          if augmentations else None)
            g_train = None if groups is None else np.asarray(groups)[train_idx]

            # The hybrid arm is handed ONLY the representations the recipe scored.
            # build_fold_hybrid does not subset to blueprint.reps itself, and
            # handing it the whole pool lets the allocation rank in representations
            # the search never considered, so the rebuilt hybrid holds a different
            # feature mix from the published one. n12 subsets for the same reason
            # (n12_importance_weighted_distance.py:131-143, its audit P1 finding).
            # The BASE arm keeps the whole pool: whether a representation the recipe
            # rejected changes its own ranking under noise is part of the question.
            bp_reps = getattr(bp, 'reps', None) if bp is not None else None
            if bp_reps:
                reps_for_bp = {r: train_reps[r] for r in bp_reps if r in train_reps}
                if not reps_for_bp:
                    # Named, not silent: an empty subset would otherwise build a
                    # hybrid out of nothing.
                    print(f"    NOTE: none of the recipe's representations "
                          f"{sorted(bp_reps)} are in this fold's pool; falling back "
                          f"to the whole pool for the hybrid arm.")
                    reps_for_bp = train_reps
            else:
                reps_for_bp = train_reps

            # ---- BASE ARM: every representation, scored whole ------------------
            if arm in ('base', 'both'):
                t_fold = time.time()
                n_calls = 0
                for method in imp_methods:
                    for rname in rep_order:
                        for lv in levels:
                            y_tr = per_level[lv]['y'][train_idx]
                            key = ('base', method, rname, lv, fold_idx)
                            try:
                                v, secs = _score_one(
                                    train_reps[rname], y_tr, method,
                                    seed=42 + fold_idx)
                            except Exception as exc:
                                msg = f"{type(exc).__name__}: {exc}"
                                print(f"    base {method}/{rname} level {lv:g}: {msg}")
                                results['errors'].append(
                                    {'arm': 'base', 'method': method, 'rep': rname,
                                     'level': lv, 'fold': fold_idx, 'error': msg})
                                continue
                            rep_block[key] = v.astype(np.float32)
                            n_calls += 1
                            results['scoring_seconds'].append(
                                {'arm': 'base', 'method': method, 'rep': rname,
                                 'level': lv, 'fold': fold_idx,
                                 'n_features': int(v.size), 'seconds': round(secs, 3)})
                # One line per fold, so a long run says where it is. Without it the
                # base arm printed nothing between the injection lines and the end,
                # and a six-hour sweep had no progress to read at all.
                _done = (replicate * len(folds)) + fold_idx + 1
                _total = n_replicates * len(folds)
                _el = time.time() - t_fold
                print(f"    [r{replicate} f{fold_idx}] {n_calls} scores in "
                      f"{_el:.0f}s — fold {_done} of {_total}", flush=True)

            # ---- HYBRID ARM: which features the recipe takes -------------------
            if arm in ('hybrid', 'both') and bp is not None:
                for lv in levels:
                    y_tr = per_level[lv]['y'][train_idx]
                    t0 = time.time()
                    try:
                        # build_fold_hybrid, NOT a bare create_hybrid: it forwards the
                        # WHOLE recipe. n12 passes six fields and drops
                        # candidate_ranking, augmentation_margin,
                        # augmentation_fraction, augmentation_threshold,
                        # augmentation_budget, aug_per_source_budget,
                        # aug_max_candidates and step_size, so its hybrid can hold
                        # different features from the one that won its search.
                        # groups is sliced to this fold's training rows: without it the
                        # internal admission holdout falls back to a RandomState(42)
                        # 20% random carve sharing scaffolds with its own training
                        # rows (_shared.py:4379-4386).
                        X_tr_h, _, feat_info = build_fold_hybrid(
                            reps_for_bp, reps_for_bp, y_tr, bp,
                            augmentations_train=train_augs,
                            augmentations_val=train_augs,
                            groups=g_train,
                            model=model_inst, metric_fn=metric_fn,
                            lower_is_better=lower_is_better,
                        )
                    except Exception as exc:
                        msg = f"{type(exc).__name__}: {exc}"
                        print(f"    hybrid level {lv:g} fold {fold_idx}: {msg}")
                        results['errors'].append(
                            {'arm': 'hybrid', 'level': lv, 'fold': fold_idx,
                             'error': msg})
                        continue
                    secs = time.time() - t0
                    # WHICH features the recipe took, per representation.
                    for rname in rep_order:
                        info = feat_info.get(rname)
                        if not isinstance(info, dict):
                            continue
                        sel = np.asarray(info.get('selected_indices', []), dtype=int)
                        rep_block[('hybrid', 'recipe', rname, lv, fold_idx)] = sel
                    results['scoring_seconds'].append(
                        {'arm': 'hybrid', 'method': 'recipe', 'rep': '__all__',
                         'level': lv, 'fold': fold_idx,
                         'n_features': None, 'seconds': round(secs, 3)})
                    print(f"    [r{replicate} f{fold_idx} level {lv:g}] "
                          f"build_fold_hybrid {secs:.1f}s")

                    # HOW THE HYBRID'S OWN FEATURES RANK, so the hybrid gets the
                    # same two rows the base representations get and not only the
                    # set-turnover one. The hybrid is `budget` features wide, so
                    # scoring it costs a fraction of scoring a 2048-feature
                    # representation and all three methods are run.
                    X_h = np.asarray(X_tr_h)
                    srcs = _hybrid_column_sources(feat_info, X_h.shape[1])
                    if srcs is None:
                        msg = (f"could not map {X_h.shape[1]} hybrid columns onto "
                               f"the recipe's selected indices; the rank rows for "
                               f"this cell are dropped rather than guessed")
                        print(f"    hybrid level {lv:g} fold {fold_idx}: {msg}")
                        results['errors'].append(
                            {'arm': 'hybrid', 'method': 'columns', 'level': lv,
                             'fold': fold_idx, 'error': msg})
                        continue
                    rep_block[('hybrid_src', 'map', '__all__', lv, fold_idx)] = srcs
                    for method in imp_methods:
                        try:
                            v, hsecs = _score_one(X_h, y_tr, method,
                                                  seed=42 + fold_idx)
                        except Exception as exc:
                            msg = f"{type(exc).__name__}: {exc}"
                            print(f"    hybrid {method} level {lv:g}: {msg}")
                            results['errors'].append(
                                {'arm': 'hybrid', 'method': method, 'rep': '__all__',
                                 'level': lv, 'fold': fold_idx, 'error': msg})
                            continue
                        rep_block[('hybrid_imp', method, '__all__', lv,
                                   fold_idx)] = v.astype(np.float32)
                        results['scoring_seconds'].append(
                            {'arm': 'hybrid', 'method': method, 'rep': '__hybrid__',
                             'level': lv, 'fold': fold_idx,
                             'n_features': int(v.size),
                             'seconds': round(hsecs, 3)})

        results['per_replicate'][str(replicate)] = _summarise_replicate(
            rep_block, rep_order, imp_methods, levels, len(folds), reps, top_k, arm)

        # Per-feature vectors go beside the JSON, not into it: written inline they
        # would be tens of megabytes of numbers no reader opens by eye. Convention
        # copied from n06_bindingdb_ground_truth.py:890-915.
        _save_vectors(rep_block, output_dir, dataset_name, condition, replicate,
                      results)

    results['seconds_total'] = round(time.time() - t_start, 1)
    results['summary'] = _summarise(results, levels, imp_methods, rep_order, arm)
    print(f"\n  Done in {results['seconds_total']:.1f}s. "
          f"{len(results['errors'])} cells failed.")
    return convert_numpy(results)


def _save_vectors(rep_block, output_dir, dataset_name, condition, replicate, results):
    """Write one replicate's per-feature arrays next to the dataset's JSON."""
    out_dir = os.path.join(output_dir, 'n16_importance_under_noise')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f'{dataset_name}__{condition}__r{replicate}__vectors.npz')
    payload = {}
    for (a, method, rname, lv, fold_idx), v in rep_block.items():
        stem = (f'{a}__{rname}__{method}__{condition}'
                f'__e{int(round(lv * 1000)):04d}__f{fold_idx}__r{replicate}')
        if a == 'hybrid_src':
            # A list of (source, index) pairs. Split into two arrays rather than let
            # savez coerce the pairs into a string matrix, so the index side stays an
            # integer and a reader does not have to parse it back out of text.
            payload[f'{stem}__name'] = np.array([str(s[0]) for s in v])
            payload[f'{stem}__index'] = np.array([int(s[1]) for s in v],
                                                 dtype=np.int32)
        else:
            payload[stem] = v
    np.savez_compressed(path, **payload)
    results.setdefault('vector_files', []).append(path)
    results.setdefault('vector_keys', {})[str(replicate)] = sorted(payload)
    print(f"  [saved {path}] {len(payload)} arrays")


def _summarise_replicate(rep_block, rep_order, imp_methods, levels, n_folds,
                         reps, top_k, arm):
    """Per (method, representation, level): the three rows, against level 0.0."""
    out = {'base': {}, 'hybrid': {}, 'hybrid_importance': {}}

    if arm in ('base', 'both'):
        for method in imp_methods:
            for rname in rep_order:
                width = int(reps[rname].shape[1])
                cells = {}
                for lv in levels:
                    per_fold, sets = [], []
                    for f in range(n_folds):
                        v = rep_block.get(('base', method, rname, lv, f))
                        ref = rep_block.get(('base', method, rname, 0.0, f))
                        if v is None or ref is None:
                            continue
                        v = np.asarray(v, dtype=float)
                        ref = np.asarray(ref, dtype=float)
                        row = {
                            'n_scored': int((v != 0).sum()),
                            'n_features': int(v.size),
                        }
                        row.update(_pair_stats(ref, v, top_k))
                        row.update(_turnover(ref, v, top_k))
                        per_fold.append(row)
                        k = int(min(top_k, v.size))
                        sets.append(set(np.argsort(v)[-k:].tolist()))
                    if not per_fold:
                        continue
                    cells[f'{lv:g}'] = {
                        'n_folds': len(per_fold),
                        'per_fold': per_fold,
                        # Means across folds, each named by what it counts.
                        'mean_n_scored_features': float(np.mean(
                            [r['n_scored'] for r in per_fold])),
                        'mean_spearman_both_scored': float(np.mean(
                            [r['spearman_both_scored'] for r in per_fold])),
                        'mean_spearman_all_features': float(np.mean(
                            [r['spearman_all'] for r in per_fold])),
                        'mean_kendall_both_scored': float(np.mean(
                            [r['kendall_both_scored'] for r in per_fold])),
                        'mean_topk_retained_fraction': float(np.mean(
                            [r['retained_fraction'] for r in per_fold])),
                        # The readable one: the top set capped at how many features
                        # the scorer actually ranked, so it holds no tied zeros.
                        'mean_topk_effective_retained_fraction': float(np.mean(
                            [r['retained_fraction_effective'] for r in per_fold])),
                        'mean_topk_effective_size': float(np.mean(
                            [r['top_k_effective'] for r in per_fold])),
                        'nogueira_topk_stability_across_folds': (
                            nogueira_stability(sets, width) if len(sets) > 1
                            else float('nan')),
                    }
                if cells:
                    out['base'].setdefault(method, {})[rname] = cells

    if arm in ('hybrid', 'both'):
        # ROW 1 and ROW 2 for the hybrid's own features, the same two the base
        # representations get. Two levels select different features, so the vectors
        # cannot be lined up elementwise; they are keyed by (representation, source
        # index) and correlated over the features BOTH levels selected. The size of
        # that overlap is reported next to every correlation, because a correlation
        # over a handful of shared features is weak evidence however high it is.
        for method in imp_methods:
            cells = {}
            for lv in levels:
                per_fold = []
                for f in range(n_folds):
                    v = rep_block.get(('hybrid_imp', method, '__all__', lv, f))
                    ref = rep_block.get(('hybrid_imp', method, '__all__', 0.0, f))
                    smap = rep_block.get(('hybrid_src', 'map', '__all__', lv, f))
                    rmap = rep_block.get(('hybrid_src', 'map', '__all__', 0.0, f))
                    if v is None or ref is None or smap is None or rmap is None:
                        continue
                    cur = dict(zip([tuple(s) for s in smap],
                                   np.asarray(v, dtype=float)))
                    base = dict(zip([tuple(s) for s in rmap],
                                    np.asarray(ref, dtype=float)))
                    shared = sorted(set(cur) & set(base))
                    row = {
                        'n_hybrid_features': int(np.asarray(v).size),
                        'n_hybrid_features_at_level_zero': int(np.asarray(ref).size),
                        'n_features_selected_at_both_levels': len(shared),
                        'n_scored': int((np.asarray(v, dtype=float) != 0).sum()),
                    }
                    if len(shared) >= MIN_SHARED_FOR_RANK:
                        a = np.array([base[k] for k in shared], dtype=float)
                        b = np.array([cur[k] for k in shared], dtype=float)
                        row.update(_pair_stats(a, b, top_k))
                        row.update(_turnover(a, b, top_k))
                    per_fold.append(row)
                if not per_fold:
                    continue
                ranked = [r for r in per_fold if 'spearman_both_scored' in r]
                cell = {
                    'n_folds': len(per_fold),
                    'n_folds_with_enough_shared_features': len(ranked),
                    'per_fold': per_fold,
                    'mean_n_hybrid_features': float(np.mean(
                        [r['n_hybrid_features'] for r in per_fold])),
                    'mean_n_scored_features': float(np.mean(
                        [r['n_scored'] for r in per_fold])),
                    'mean_n_features_selected_at_both_levels': float(np.mean(
                        [r['n_features_selected_at_both_levels'] for r in per_fold])),
                }
                if ranked:
                    cell.update({
                        'mean_spearman_both_scored': float(np.mean(
                            [r['spearman_both_scored'] for r in ranked])),
                        'mean_spearman_all_features': float(np.mean(
                            [r['spearman_all'] for r in ranked])),
                        'mean_kendall_both_scored': float(np.mean(
                            [r['kendall_both_scored'] for r in ranked])),
                        'mean_topk_effective_retained_fraction': float(np.mean(
                            [r['retained_fraction_effective'] for r in ranked])),
                        'mean_topk_effective_size': float(np.mean(
                            [r['top_k_effective'] for r in ranked])),
                    })
                cells[f'{lv:g}'] = cell
            if cells:
                out['hybrid_importance'].setdefault(method, {})['__hybrid__'] = cells

        # ROW 3: which SOURCE features the recipe took, per representation.
        for rname in rep_order:
            width = int(reps[rname].shape[1])
            cells = {}
            for lv in levels:
                per_fold, sets = [], []
                for f in range(n_folds):
                    sel = rep_block.get(('hybrid', 'recipe', rname, lv, f))
                    ref = rep_block.get(('hybrid', 'recipe', rname, 0.0, f))
                    if sel is None or ref is None:
                        continue
                    s, r = set(np.asarray(sel).tolist()), set(np.asarray(ref).tolist())
                    per_fold.append({
                        'n_selected_features': len(s),
                        'n_selected_features_at_level_zero': len(r),
                        'n_left': len(r - s),
                        'n_entered': len(s - r),
                        'retained_fraction_of_level_zero': (
                            float(len(r & s) / len(r)) if r else float('nan')),
                    })
                    sets.append(s)
                if not per_fold:
                    continue
                cells[f'{lv:g}'] = {
                    'n_folds': len(per_fold),
                    'per_fold': per_fold,
                    'mean_n_selected_features': float(np.mean(
                        [r['n_selected_features'] for r in per_fold])),
                    'mean_retained_fraction_of_level_zero': float(np.mean(
                        [r['retained_fraction_of_level_zero'] for r in per_fold])),
                    'nogueira_selection_stability_across_folds': (
                        nogueira_stability(sets, width) if len(sets) > 1
                        else float('nan')),
                }
            if cells:
                out['hybrid'][rname] = cells

    return out


def _summarise(results, levels, imp_methods, rep_order, arm):
    """The one screen a reader needs: what moved, per scoring method, per level.

    Never pooled across datasets and never averaged with another dataset's numbers —
    a Spearman on rmse-scored logd and on spearman-scored flab are not the same
    quantity. Every figure here is within one dataset.
    """
    summary = {'base': {}, 'hybrid': {}, 'hybrid_importance': {}}
    reps_per_rep = results['per_replicate']

    if arm in ('base', 'both'):
        for method in imp_methods:
            per_level = {}
            for lv in levels:
                rhos, scored, retained = [], [], []
                for block in reps_per_rep.values():
                    for rname, cells in block.get('base', {}).get(method, {}).items():
                        c = cells.get(f'{lv:g}')
                        if c is None:
                            continue
                        rhos.append(c['mean_spearman_both_scored'])
                        scored.append(c['mean_n_scored_features'])
                        # The effective one: capped at how many features the scorer
                        # actually ranked, so the set holds no tied zeros.
                        retained.append(
                            c['mean_topk_effective_retained_fraction'])
                if not rhos:
                    continue
                per_level[f'{lv:g}'] = {
                    'n_cells': len(rhos),
                    'median_spearman_vs_level_zero': float(np.median(rhos)),
                    'min_spearman_vs_level_zero': float(np.min(rhos)),
                    'max_spearman_vs_level_zero': float(np.max(rhos)),
                    'median_n_scored_features': float(np.median(scored)),
                    'median_topk_effective_retained_fraction': float(
                        np.median(retained)),
                }
            if per_level:
                summary['base'][method] = per_level

    if arm in ('hybrid', 'both'):
        # The hybrid's own feature ranking, the counterpart of the base table above.
        for method in imp_methods:
            per_level = {}
            for lv in levels:
                rhos, shared, scored = [], [], []
                for block in reps_per_rep.values():
                    cells = (block.get('hybrid_importance', {})
                             .get(method, {}).get('__hybrid__', {}))
                    c = cells.get(f'{lv:g}')
                    if c is None:
                        continue
                    shared.append(c['mean_n_features_selected_at_both_levels'])
                    scored.append(c['mean_n_scored_features'])
                    if 'mean_spearman_both_scored' in c:
                        rhos.append(c['mean_spearman_both_scored'])
                if not shared:
                    continue
                row = {
                    'n_cells': len(shared),
                    'median_n_features_selected_at_both_levels': float(
                        np.median(shared)),
                    'median_n_scored_features': float(np.median(scored)),
                }
                if rhos:
                    row.update({
                        'n_cells_with_a_correlation': len(rhos),
                        'median_spearman_vs_level_zero': float(np.median(rhos)),
                        'min_spearman_vs_level_zero': float(np.min(rhos)),
                        'max_spearman_vs_level_zero': float(np.max(rhos)),
                    })
                per_level[f'{lv:g}'] = row
            if per_level:
                summary['hybrid_importance'][method] = per_level

        per_level = {}
        for lv in levels:
            kept, nsel = [], []
            for block in reps_per_rep.values():
                for rname, cells in block.get('hybrid', {}).items():
                    c = cells.get(f'{lv:g}')
                    if c is None:
                        continue
                    kept.append(c['mean_retained_fraction_of_level_zero'])
                    nsel.append(c['mean_n_selected_features'])
            if not kept:
                continue
            per_level[f'{lv:g}'] = {
                'n_cells': len(kept),
                'median_retained_fraction_of_level_zero': float(np.median(kept)),
                'median_n_selected_features': float(np.median(nsel)),
            }
        if per_level:
            summary['hybrid'] = per_level

    return summary


if __name__ == '__main__':
    parser = create_experiment_parser(
        'N16: feature importance across noise levels')
    parser.add_argument('--noise-condition', default=NOISE_CONDITION,
                        help='noiseInject regression condition name '
                             '(default: gaussian)')
    parser.add_argument('--replicates', type=int, default=N_REPLICATES,
                        help='independent noise draws per level (default: 3)')
    parser.add_argument('--arm', default='both',
                        choices=['base', 'hybrid', 'both'],
                        help='base representations, the hybrid selection, or both')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help='size of the tracked top-of-the-ranking set')
    parser.add_argument('--levels', type=str, default=None,
                        help='comma-separated noise levels; must include 0.0 '
                             '(default: the shared grid for this dataset)')
    parser.add_argument('--importance-methods', type=str, default=None,
                        help='comma-separated scoring methods '
                             '(default: deeplift,lightgbm_gain,treeshap)')
    args = parser.parse_args()
    config = get_config(args)
    # The signed-off pool comes for free: load_data auto-filters to
    # DATASET_REPS[ds]['general'] whenever rep_names is None (_shared.py:3510-3520),
    # which is the same list noise_helpers.run_all passes explicitly
    # (noise_helpers.py:281-288). So a standalone run measures the same
    # representations a suite run does without asking for them. A comma-separated
    # --reps goes through parse_rep_args and overrides that.
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names)
    results = run(data, config, args.output, args.dataset,
                  noise_condition=args.noise_condition,
                  replicates=args.replicates, arm=args.arm, top_k=args.top_k,
                  levels=([float(x) for x in args.levels.split(',')]
                          if args.levels else None),
                  importance_methods=(args.importance_methods.split(',')
                                      if args.importance_methods else None))
    # The condition goes in the filename on a standalone run, so two conditions do
    # not overwrite each other. Under noise_helpers.run_all the filename is the plain
    # dataset name (noise_helpers.py:494-495) and the condition lives inside the file
    # as `noise_condition`; a second condition through the suite overwrites the first,
    # which is NOISE_AUDIT_2026_09_10.md section 10 decision 4 and not settled here.
    # The npz files carry the condition either way and never collide.
    #
    # metric_name and lower_is_better are passed explicitly: registry_lookup would be
    # handed 'logd__gaussian', find nothing, and leave the file with no metric name
    # for the direction gate to read (_shared.py:6089-6109).
    save_experiment_results(
        results, args.output,
        'n16_importance_under_noise',
        f'{args.dataset}__{args.noise_condition}',
        metric_name=results.get('metric'),
        lower_is_better=results.get('lower_is_better'))
