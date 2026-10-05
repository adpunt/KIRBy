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
     each over all features, over the features EITHER level scored non-zero, and
     over the features BOTH levels scored non-zero, plus top-k set overlap. The
     first two come from f30_analyse._pair_stats, which calls the either-level set
     'both_scored'; this file renames it 'either_scored' (dirty-run bug, n16
     "column name", fixed 2026-10-05) and computes the both-levels set itself.
  2b. THE FLOOR (added 2026-10-05, dirty-run bug "no level-0 floor"). The same
     cell at level 0.0 scored a second time with another scorer seed, compared
     with the first exactly as a noisy level is. Movement a noisy level shows
     that is no larger than this floor is the scorer's own instability, not the
     labels. The across-fold Nogueira stability at level 0.0 is the second
     floor and is reported beside it.
  3. Turnover in the top set — which features left the clean top set and which
     entered, plus Nogueira phi-hat set stability over the whole level sweep. The top
     set is capped at how many features the scorer actually ranked, never at the
     representation's width: lightgbm_gain scores 39 of diff_maccs's 167 features on
     buchwald_hartwig, so an uncapped top-50 overlap is 1.00 at every level (tied
     zeros, ordered by index, identical on both sides) while the scored ranking moves
     from Spearman 0.980 to 0.920.

TWO ARMS, AND ALL THREE ROWS ARE REPORTED FOR BOTH.

ALIGNED 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4, chat 5). Per outer fold,
per decided model (the dataset's two from DECISIONS.md Step 1): its 2 best base
representations and its 2 best hybrids, both ranked by ONE search on that fold's
training rows (_shared.fold_standard_selection, clean labels, decided grid,
pruning off). 2 models x (2 base + 2 hybrids) = the 8 configurations of R5. Until
2026-10-05 the base arm scored every representation in the pool and the hybrid arm
held one recipe, rf's best in a search over every row (R1, R2, R4, R5 of the
audit). A representation that is one of both models' top 2 is scored once; the
result file names which models picked it.
  base   — each fold's selected base representations, scored whole. The feature
           set is a fixed matrix, so a vector at one level indexes the same
           features as at another and rows 2 and 3 are defined by construction.
  hybrid — each model's two per-fold recipes, rebuilt at every level on that
           level's training labels with that model as the admission model
           (noise_cache.cached_hybrid_for_fold). Row 1 is
           how many of its features carry any importance. Row 2 needs care, because
           two levels select DIFFERENT features and the two vectors cannot be lined up
           elementwise: every hybrid column is keyed back to its
           (representation, source index) through feat_info[rep]['selected_indices']
           and the ranking is correlated over the features BOTH levels selected, with
           the size of that overlap reported beside every correlation. Row 3 is the
           turnover in the selected set itself, per representation.

THE RECIPE IS HELD FIXED at the clean-label one, at every level. Since 2026-10-05
that is each fold's clean-label search on its own training rows, not the
whole-dataset precompute (which saw the held-out rows). Three reasons for holding
it fixed. Re-searching per level is not affordable — measured off
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

NOISE (changed 2026-10-05). Every level comes from noise_helpers.inject_condition,
the one route every noise experiment uses, with the decided kinds and levels
(noise_helpers.decided_conditions / decided_levels: gaussian, outlier_p05,
grouped_shifted, censoring, shift_only_p10, sparse_dprime_up_p10; full_knn
datasets 0, 0.5, 1.0, 1.5 and censoring 0, 0.10, 0.25, 0.40; fast_knn datasets 0,
1.0 and censoring 0, 0.25). 3 draws per non-zero level; level 0 is the labels as
delivered, the same in every draw, and every score of it is read from noise_cache
after the first. Until 2026-10-05 this file built its own NoiseSpec through
TrainingNoise, which cannot inject shift_only_p10 or sparse_dprime_up_p10.

Every comparison here is against the level-0.0 vector, so the levels have to NEST:
same molecules, same directions, more magnitude. The draw seed holds the level out
for all six kinds. Checked 2026-10-05 on 400 synthetic labels through
inject_condition: for the five kinds other than censoring the error at level 1.0
and 1.5 is exactly 2 and 3 times the error at 0.5 on every molecule; under
censoring the labels clipped at 0.10 are a subset of those clipped at 0.25 and
0.40, moved the same way and at least as far. The check is re-run at run time
(_check_nesting) rather than trusted.

SCOPE: regression only. The first of the two reasons this rested on is gone: since
2026-10-05 level 0 on class labels is the labels as delivered (noise_spec,
handoff 2.3 item 10). The second, that the classification injector takes one seed
only so the flip levels may not nest, has not been re-checked.

Usage:
    python n16_importance_under_noise.py --dataset logd --mode intermediary
    python n16_importance_under_noise.py --dataset caco2 --mode full --replicates 3
    python n16_importance_under_noise.py --dataset logd --noise-condition outlier_p05
    python n16_importance_under_noise.py --dataset logd --arm base
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (
    sweep_conditions, conditions_for_task,
    # CHAT 5 ALIGNMENT block (2026-10-05): folds, the decided noise grid, the
    # injection route and the result record every noise experiment shares.
    noise_outer_folds, fold_spread, decided_levels, inject_condition,
    level_unit, configuration_record, dataset_knn_group,
)

from experiments._shared import (
    load_data, get_config, create_experiment_parser, parse_rep_args,
    save_experiment_results, convert_numpy, get_domain_metric,
    get_model_factories, fold_standard_selection,
)
from experiments.dataset_policy import noise_groups_for, noise_dose_units
# The four rank-agreement statistics already in this repository, with their
# provenance recorded at f30_analyse.py:9-19. _pair_stats reports every feature-rank
# statistic over all features and over the features EITHER side scored non-zero
# (it names that set 'both_scored'; _rank_stats below renames it). The second is
# the honest one here for the same reason it is there: lightgbm_gain leaves most
# features at exactly 0.0 and np.argsort orders tied zeros by index, so an
# all-features Spearman measures how two tied-zero blocks line up.
# f30_analyse imports stdlib, numpy and scipy only, so this pulls in no torch.
from experiments.framework.f30_analyse import _pair_stats, nogueira_stability
from kirby.hybrid import compute_feature_importance
# The disk cache and the per-fold checkpoint, added 2026-09-25 (plan section
# 3). n16 finds no neighbours and fits no downstream model, so it takes no
# similarity setting and writes no D-versus-D' rows (plan section 6). For the
# same reason it takes no distance hybrid and no similarity hybrid as a
# neighbour space (handoff 2.3 item 1: those are for experiments that measure
# something with neighbours or distances; checked in this file 2026-10-05).
from experiments.noise import noise_cache as nc
from experiments.noise import _sim_axis_n15_n22 as sx

# The name the checkpoint and the result directory are filed under.
EXP_FULL_NAME = 'n16_importance_under_noise'

# Bumped 2026-10-05 (chat 5 alignment): per-fold 2 + 2 selection per decided
# model, decided noise grid, renamed rank columns, the floor. In the checkpoint
# signature, so a partial checkpoint from the old design is never resumed, and
# checked against every finished condition file by run() (_old_schema_conditions),
# so a finished file from the old design is re-run rather than kept.
RESULT_SCHEMA = 'n16_chat5_2026_10_05'


# The three the engine had as DEFAULT_IMPORTANCE_METHODS (find_best_hybrid.py).
# f30's other 28 scorers are candidates under evaluation, not things a recipe can
# select, so a drift measured in one of them would not describe any hybrid this
# repo builds.
# OPEN (chat 5): DECISIONS.md Step 3 now cuts importance methods per dataset for
# the SEARCH (deeplift cut on herg_ki; qm9 keeps lightgbm_gain only). Here the
# methods are the measured quantity, not a search axis, so all three are still
# run on every dataset. Whether to narrow them to each dataset's decided ones is
# Adelaide's to decide; the hybrid's own recipe already uses only decided ones.
IMPORTANCE_METHODS = ['deeplift', 'lightgbm_gain', 'treeshap']

# Used only when the caller names one condition. A suite run sweeps the six
# decided regression kinds (noise_helpers.conditions_for_task), and the condition
# is in BOTH the output filename stem and every npz key, so one condition does not
# overwrite another.
NOISE_CONDITION = 'gaussian'

# How many independent noise draws per non-zero level: 3, the decided count
# (KNN_SEARCH_SPACE_RECORD.md, "The noise grid"). Each draw is a different
# corruption of the same molecules at every level of that draw's own nested
# ladder. Level 0 is the same labels in every draw.
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

# The floor (added 2026-10-05): level 0.0 is scored a second time with the scorer
# seed moved by this much, so a reader can see how far a ranking moves with clean
# labels and nothing changed but the scorer's own randomness.
FLOOR_SEED_OFFSET = 1000

# Tolerance on the nesting assertion. The ratio is a float division of two products of
# the same unit draw, so it is exact to rounding; 1e-9 catches a changed seed formula
# while tolerating the last bit.
NEST_RTOL = 1e-9


def _inject_ladder(data, dataset_name, condition, levels, replicate, folds,
                   noise_groups):
    """Every fold's training labels at every level for one draw.

    Rewritten 2026-10-05 onto noise_helpers.inject_condition, the route every
    noise experiment uses (handoff 2.4, chat 5). The draw is made over the whole
    label column and each fold's training rows taken out, so a molecule keeps
    the same draw in every fold it trains in.

    Returns (per_level, provenance) where per_level[level][fold_i] is that
    fold's noisy training labels and provenance is one record per level (the
    first fold's; the draw is the same column in every fold).

    RAISES if the ladder does not nest (_check_nesting). The whole experiment
    compares a vector at one level against the vector at level 0.0, and if the
    higher level is a fresh draw rather than the same draw made larger, that
    comparison has two things moving in it and cannot be read.
    """
    y_all = np.asarray(data['labels'], dtype=float).ravel()
    per_level, provenance = {}, {}
    for lv in levels:
        per_level[lv] = {}
        for fold_i, tr, _va in folds:
            y_tr, _mask, prov = inject_condition(
                data, dataset_name, condition, float(lv), int(replicate),
                train_idx=tr, groups=noise_groups)
            per_level[lv][fold_i] = np.asarray(y_tr, dtype=float)
            if lv not in provenance:
                provenance[lv] = dict(prov or {}, fold=int(fold_i))
    first_fold, first_tr, _ = folds[0]
    eps = {lv: per_level[lv][first_fold] - y_all[first_tr] for lv in levels}
    _check_nesting(condition, levels, eps)
    return per_level, provenance


def _check_nesting(condition, levels, eps):
    """Raise unless the non-zero levels of one draw nest.

    Five of the six decided kinds scale one draw: the error at level L2 is
    L2/L1 times the error at level L1 on every moved molecule. censoring clips
    a share of labels, so its error does not scale; there the rows clipped at a
    lower level must be among those clipped at a higher one, moved the same way
    and at least as far.
    """
    nonzero = [lv for lv in levels if float(lv) > 0.0]
    if len(nonzero) < 2:
        return
    ref_lv = min(nonzero)
    ref = eps[ref_lv]
    moved = np.abs(ref) > 0
    for lv in nonzero:
        if lv == ref_lv:
            continue
        cur = eps[lv]
        if condition == 'censoring':
            ok = (np.all(np.abs(cur[moved]) > 0)
                  and np.all(np.sign(cur[moved]) == np.sign(ref[moved]))
                  and np.all(np.abs(cur[moved]) >= np.abs(ref[moved]) - 1e-12))
            detail = ('the labels clipped at the lower level are not all '
                      'clipped at least as far, the same way, at this level')
        else:
            want = float(lv) / float(ref_lv)
            got = cur[moved] / ref[moved]
            ok = bool(np.allclose(got, want, rtol=NEST_RTOL))
            detail = (f'the error is {got.min():.9f} to {got.max():.9f} times '
                      f'the error at level {ref_lv}, and it must be '
                      f'{want:.9f} on every molecule')
        if not ok:
            raise RuntimeError(
                f"the noise ladder does not nest under {condition} at level "
                f"{lv}: {detail}. Every comparison this experiment makes is "
                f"against level 0.0, so a level that is a fresh draw rather "
                f"than the same draw made larger puts two moving things in "
                f"one number. Check that noise_helpers.noise_draw_seed still "
                f"holds the level out of the seed.")


def _score_one(X, y, method, seed, dataset=None):
    """One representation's per-feature importance vector, on the rows given.

    `dataset`, added 2026-09-25 (plan section 3): given, the vector is read
    from noise_cache when (X, y, method, seed) were scored before -- keyed on
    the CONTENT of X and of the label vector, so level 0.0, which hands every
    replicate and every condition the same clean labels, is scored once per
    (representation, fold, method) instead of once per replicate per
    condition. compute_feature_importance is seeded (random_state), and the
    thread counts torch and LightGBM read are in every key. A failure raises
    and is never stored. The seconds returned are this call's, so a cached
    read reports its own few milliseconds.

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

    def _compute():
        out = compute_feature_importance({'x': X}, y, method=method,
                                         random_state=seed)
        return np.asarray(out['x'], dtype=float).ravel()
    if dataset is None:
        v = _compute()
    else:
        v = np.asarray(nc.cached(
            dataset, 'importance',
            {'X': nc.array_hash(X), 'y': nc.array_hash(y),
             'method': str(method), 'seed': int(seed)}, _compute))
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


def _rank_stats(a, b, top_k):
    """Feature-rank agreement between two importance vectors over the same features.

    f30_analyse._pair_stats, with its 'both_scored' figures renamed
    'either_scored': its mask is (a != 0) | (b != 0), the features EITHER side
    scored (dirty-run bug, n16 "column name", 2026-10-05). The features BOTH
    sides scored, (a != 0) & (b != 0), are computed here as
    'spearman_both_scored' / 'kendall_both_scored', nan under 3 features. The
    two set sizes are reported beside them.
    """
    from scipy.stats import spearmanr, kendalltau
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = _pair_stats(a, b, top_k)
    out = {'spearman_all': s['spearman_all'], 'kendall_all': s['kendall_all'],
           'spearman_either_scored': s['spearman_both_scored'],
           'kendall_either_scored': s['kendall_both_scored'],
           'topk_overlap': s['topk_overlap'], 'topk': s['topk']}
    either = (a != 0) | (b != 0)
    both = (a != 0) & (b != 0)
    out['n_either_scored'] = int(either.sum())
    out['n_both_scored'] = int(both.sum())
    if both.sum() >= 3:
        out['spearman_both_scored'] = float(spearmanr(a[both], b[both])[0])
        out['kendall_both_scored'] = float(
            kendalltau(a[both], b[both], variant='b')[0])
    else:
        out['spearman_both_scored'] = float('nan')
        out['kendall_both_scored'] = float('nan')
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """n16_importance_under_noise across every condition this dataset can be given.

    The condition is an axis this experiment sweeps, not something a caller
    runs it once per. The sweep, the per-condition checkpointing, the resume
    and the stride slice are noise_helpers.sweep_conditions. The default list
    is the six decided regression kinds (conditions_for_task, 2026-10-05).

    Files from the old design are re-run (2026-10-05). sweep_conditions keeps
    a condition file whose input_fingerprint matches, and that fingerprint
    holds the rows, labels, groups, representations, fold count, metric and
    noise grid, never RESULT_SCHEMA. So a finished file from the old design
    (every pool representation, one rf recipe) would be kept as done and never
    re-run, and n16_analyse would then refuse it as old-schema. Before the
    normal sweep, every condition this call covers whose file on disk carries
    another result_schema is run again with force_rerun, on its own.
    """
    default = conditions_for_task(data)
    stale = ([] if kwargs.get('force_rerun') or output_dir is None
             else _old_schema_conditions(kwargs, default, output_dir,
                                         dataset_name))
    if stale:
        print(f"  Old-schema n16 files, re-run first: {stale}")
        pre = {k: v for k, v in kwargs.items()
               if k not in ('noise_conditions', 'noise_condition',
                            'sweep_slice', 'force_rerun')}
        pre['noise_conditions'] = list(stale)
        pre['force_rerun'] = True
        sweep_conditions(
            _run_one_condition, EXP_FULL_NAME,
            data, config, output_dir, dataset_name,
            default_conditions=default, kwargs=pre,
            groups_needed=GROUPED_STRATEGIES)
    # The normal sweep writes the index file over all conditions; the ones just
    # re-run now carry this schema and a matching fingerprint, so it keeps them.
    return sweep_conditions(
        _run_one_condition, EXP_FULL_NAME,
        data, config, output_dir, dataset_name,
        default_conditions=default, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _old_schema_conditions(kwargs, default, output_dir, dataset_name):
    """Conditions this call covers whose file on disk has another result_schema.

    Resolves the condition list and the stride slice the same way
    noise_helpers.sweep_conditions does (noise_conditions, else
    noise_condition, else the default; then conditions[k::n]), so a sliced
    array task only re-runs conditions in its own slice. A file that cannot be
    read is left to sweep_conditions, which already re-runs it.
    """
    import json
    from pathlib import Path
    conds = kwargs.get('noise_conditions')
    if conds is None:
        one = kwargs.get('noise_condition')
        conds = [one] if one else list(default)
    conds = [str(c) for c in conds]
    spec = kwargs.get('sweep_slice')
    if spec:
        k, n = (int(x) for x in str(spec).split('/'))
        if 0 <= k < n:
            conds = conds[k::n]
    stale = []
    for c in conds:
        f = Path(output_dir) / EXP_FULL_NAME / f'{dataset_name}__{c}.json'
        if not (f.exists() and f.stat().st_size > 0):
            continue
        try:
            with open(f) as fh:
                prev = json.load(fh)
        except (ValueError, OSError):
            continue
        if prev.get('result_schema') != RESULT_SCHEMA:
            stale.append(c)
    return stale


def _select_fold(data, dataset_name, fold_i, train_idx, n_folds):
    """{model: entry} for one outer fold: each decided model's 2 base
    representations and 2 hybrid recipes, from one search on this fold's
    training rows with clean labels (_shared.fold_standard_selection).

    Called with the same arguments noise_helpers.fold_configurations hands it
    (through fold_neighbour_space_recipes), so the stored per-fold general
    search is the one every other noise experiment reads. fold_configurations
    itself is not used because it also builds the distance search and the
    similarity search, whose hybrids are neighbour spaces only and n16 measures
    nothing with neighbours (2026-10-05).

    entry = {'base': [2 names], 'base_scores': {name: training-row score},
             'hybrids': [top_hybrids entries], 'recipes': {'hybrid': (Blueprint,
             aug condition), 'hybrid__general2': (...)}, 'unavailable': {space:
             why}, 'hybrids_skipped_identical_features': [...]}; None for a model
             whose search failed.
    """
    sel = fold_standard_selection(
        data['reps'], np.asarray(data['labels']), data.get('groups'),
        train_idx=np.asarray(train_idx), fold_i=int(fold_i),
        n_outer_folds=int(n_folds), dataset_name=dataset_name,
        augmentations=data.get('augmentations'),
        task_type=data.get('task_type') or 'auto', seed=42,
        label_noise=None, verbose=False, condition_suffix=None)
    out = {}
    for m, s_m in sel.items():
        if s_m is None:
            out[m] = None
            continue
        recipes, unavailable = {}, {}
        for space, i in (('hybrid', 0), ('hybrid__general2', 1)):
            if len(s_m['hybrids']) > i:
                h = s_m['hybrids'][i]
                recipes[space] = (h['blueprint'], h.get('aug_condition', 'none'))
            else:
                unavailable[space] = (f'the per-fold general search gave this '
                                      f'model fewer than {i + 1} hybrids with '
                                      f'distinct features')
        for i, sp in enumerate(('base_1', 'base_2')):
            if len(s_m['base']) <= i:
                unavailable[sp] = (
                    (s_m.get('base_detail') or {}).get('shortfall_reason')
                    or f'the per-fold search scored fewer than {i + 1} '
                       f'representations for this model')
        out[m] = {
            'base': list(s_m['base']),
            'base_scores': {e['rep']: e.get('score') for e in
                            (s_m.get('base_detail') or {}).get('selected', [])},
            'hybrids': s_m['hybrids'],
            'hybrids_skipped_identical_features':
                s_m.get('hybrids_skipped_identical_features', []),
            'recipes': recipes,
            'unavailable': unavailable,
            'fit_configurations': (list(s_m['base'])
                                   + [h['label'] for h in s_m['hybrids']]),
        }
    return out


# The two hybrid configurations of one model, in rank order: the space names
# fold_configurations uses, and the short name this file files them under.
HYBRID_SLOTS = (('hybrid', 'hybrid1'), ('hybrid__general2', 'hybrid2'))


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
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
    # OPEN (chat 5): classification is still skipped. Of the two reasons below,
    # the first no longer holds since 2026-10-05 (level 0 on class labels is
    # the labels as delivered, handoff 2.3 item 10). The second has not been
    # re-checked. Running n16 on the 15 classification datasets is Adelaide's
    # to decide; nothing here changed what runs.
    if task_type != 'regression':
        print("  SKIP: classification — the classification injector takes one "
              "seed only, so the flip levels may not nest (not re-checked "
              "2026-10-05; the level-0 flip reason is gone).")
        return {'skipped': True,
                'reason': 'classification: the flip levels may not nest',
                'policy': 'regression_only', 'domain': domain, 'n_samples': n}
    if len(reps) < 2:
        print(f"  SKIP: {len(reps)} representations loaded — nothing to compare.")
        return {'skipped': True, 'reason': f'only {len(reps)} representations',
                'policy': 'needs_two_reps', 'domain': domain, 'n_samples': n}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    condition = kwargs.get('noise_condition', NOISE_CONDITION)
    levels = kwargs.get('levels') or kwargs.get('sigmas')
    levels_source = 'caller'
    if levels is None:
        # The decided grid (noise_helpers.decided_levels, 2026-10-05): by the
        # dataset's kNN group and, for censoring, its own share grid.
        levels = list(decided_levels(data, dataset_name, condition))
        levels_source = 'decided'
    levels = sorted(float(lv) for lv in levels)
    n_replicates = int(kwargs.get('replicates', N_REPLICATES))
    arm = kwargs.get('arm', 'both')
    top_k = int(kwargs.get('top_k', TOP_K))
    imp_methods = list(kwargs.get('importance_methods') or IMPORTANCE_METHODS)

    # Every comparison below is against the level-0.0 vector, so a grid without it —
    # or with only one level — has no answer in it.
    if 0.0 not in levels or len(levels) < 2:
        print(f"  SKIP: levels={levels} — every comparison is against level 0.0, so "
              f"the grid needs 0.0 and at least one level above it.")
        return {'skipped': True, 'reason': f'grid has no usable reference: {levels}',
                'policy': 'needs_zero_level_and_one_more',
                'levels': levels, 'domain': domain, 'n_samples': n}

    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    if condition in GROUPED_STRATEGIES and noise_groups is None:
        print(f"  SKIP: {condition} needs one scaffold group per molecule and this "
              f"dataset exposes no SMILES.")
        return {'skipped': True, 'reason': f'{condition} needs scaffold groups',
                'policy': 'condition_needs_groups', 'domain': domain, 'n_samples': n}

    rep_order = sorted(reps)

    # Which pool actually loaded, by name. The antibody and reaction loaders drop a
    # failed representation with a warning rather than an error, so a run can
    # quietly hold nine representations and report as if it held ten. Recorded
    # rather than raised.
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
    print(f"  Representations loaded: {len(rep_order)} — "
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

    # --- outer folds: one setting, any count (handoff 2.4 item 14) --------------
    # noise_outer_folds: kirby.find_best_hybrid.nested_cv_outer_folds, the folds
    # the per-fold searches are filed under, counted from config['n_splits']
    # (KIRBY_NOISE_N_FOLDS overrides; KIRBY_NOISE_ONLY_FOLDS runs a subset).
    # Replaces cv_split, 2026-10-05.
    folds, n_folds = noise_outer_folds(data, config)
    # --max-folds keeps the same split and stops early (get_config puts it in
    # the config). Honoured so a shortened run reports the folds it scored.
    max_folds = config.get('max_folds')
    if max_folds:
        folds = folds[:int(max_folds)]
    fold_ids = [int(f) for f, _, _ in folds]

    from experiments.decided_grid import decided_models
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type, n_samples=n)

    print(f"  Scoring methods: {imp_methods}")
    print(f"  Condition: {condition}   levels: {levels} ({levels_source}; "
          f"{level_unit(condition)})")
    print(f"  Noise draws: {n_replicates}   folds: {fold_ids} of {n_folds}   "
          f"top_k: {top_k}")
    print(f"  Decided models: {models}   arm: {arm}")

    results = {
        'experiment': 'n16_importance_under_noise',
        'result_schema': RESULT_SCHEMA,
        'dataset': dataset_name,
        'domain': domain,
        'n_samples': n,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'trains_downstream_models': False,
        'held_out_score': ('none: n16 fits no downstream model; each '
                           'configuration\'s training-row score from the '
                           'per-fold search is in fold_configurations'),
        'arm': arm,
        'models': models,
        'knn_group': dataset_knn_group(data, dataset_name),
        'noise_condition': condition,
        'levels': levels,
        'levels_source': levels_source,
        'level_unit': level_unit(condition),
        'dose_units': noise_dose_units(dataset_name),
        'n_replicates': n_replicates,
        'n_folds': n_folds,
        'folds_run': fold_ids,
        'n_folds_scored': len(fold_ids),
        'top_k': top_k,
        'floor_seed_offset': FLOOR_SEED_OFFSET,
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
        'selection_scope': 'fold_train_rows',
        # Stated on the row so a reader does not import n11's and n12's caveat: the
        # recipe was chosen on clean labels, and that is the CONTROL here, not a leak,
        # because this experiment reports no performance number for it to inflate.
        'recipe_lineage': 'clean_label_per_fold_search_held_fixed_across_levels',
        'oracle_caveat_applies': False,
        'oracle_caveat_note': (
            'Each fold\'s recipes were selected on that fold\'s clean training '
            'labels and held fixed at every level, which is what makes the '
            'level-to-level comparison controlled. No performance number is '
            'reported for the hybrid.'),
        # {fold: {model: configuration_record}}, the 2 + 2 per model (R5).
        'fold_configurations': {},
        # {fold: {representation: [models that picked it]}}: the base arm.
        'fold_base_reps': {},
        'noise_provenance': {},
        'dose_check': [],
        'per_replicate': {},
        'scoring_seconds': [],
        'errors': [],
        'complete': False,
    }

    # ── The per-fold checkpoint ──
    # One condition is written as a whole by sweep_conditions; inside it this
    # pickles the results and the draw's vectors after every (draw, fold), so a
    # killed condition resumes at the next fold. RESULT_SCHEMA and the fold
    # list are in the signature (2026-10-05), so a checkpoint from the old
    # whole-pool, one-recipe design is never resumed.
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'schema': RESULT_SCHEMA, 'levels': levels,
            'replicates': n_replicates, 'arm': arm, 'top_k': top_k,
            'methods': imp_methods, 'n_folds': n_folds, 'folds': fold_ids,
            'models': models}),
        condition=condition)
    _st0 = ck.state or {}
    if _st0:
        results = _st0['results']

    selections = {}

    def _selection(fold_i, train_idx):
        if fold_i not in selections:
            print(f"    [f{fold_i}] per-fold selection: 2 base + 2 hybrids per "
                  f"model {models}", flush=True)
            selections[fold_i] = _select_fold(data, dataset_name, fold_i,
                                              train_idx, n_folds)
        return selections[fold_i]

    t_start = time.time()
    for replicate in range(n_replicates):
        if str(replicate) in results['per_replicate']:
            print(f"  draw {replicate}: already in the checkpoint, skipped")
            continue
        _resumed = _st0.get('replicate') == replicate
        per_level, prov = _inject_ladder(data, dataset_name, condition, levels,
                                         replicate, folds, noise_groups)
        results['noise_provenance'][str(replicate)] = {
            f'{lv:g}': p for lv, p in prov.items()}

        if not _resumed:
            for lv in levels:
                p = prov.get(lv) or {}
                asked = p.get('target_dose_label_units')
                got = p.get('realised_dose_label_units')
                # Printed, not warned: this process silences warnings at import.
                print(f"  [d{replicate} level {lv:g}] delivered "
                      f"{'n/a' if got is None else f'{got:.6f}'} label units "
                      f"against a target of "
                      f"{'n/a' if asked is None else f'{asked:.6f}'}")
                results['dose_check'].append({
                    'replicate': replicate, 'level': lv,
                    'target_dose_label_units': asked,
                    'realised_dose_label_units': got,
                })

        rep_block = _st0['rep_block'] if _resumed else {}
        for fold_i, train_idx, val_idx in folds:
            if ck.is_done(f'r{replicate}f{fold_i}'):
                print(f"    [d{replicate} f{fold_i}] already in the "
                      f"checkpoint, skipped")
                continue
            sel = _selection(fold_i, train_idx)

            # The result record of this fold's 8 configurations (2.4 item 11).
            if str(fold_i) not in results['fold_configurations']:
                rec = {}
                for m in models:
                    e = sel.get(m)
                    r = configuration_record(e)
                    if e is not None:
                        r['base_training_row_scores'] = e.get('base_scores')
                    if m not in factories:
                        r['admission_model_unavailable'] = (
                            f'{m} has no model factory at n={n} '
                            f'(get_model_factories); its hybrids are not built')
                    rec[m] = r
                results['fold_configurations'][str(fold_i)] = rec
                base_by = {}
                for m in models:
                    for rn in ((sel.get(m) or {}).get('base') or []):
                        base_by.setdefault(rn, []).append(m)
                results['fold_base_reps'][str(fold_i)] = base_by
            base_names = sorted(results['fold_base_reps'][str(fold_i)])

            train_reps = {r: reps[r][train_idx] for r in rep_order}
            val_reps = {r: reps[r][val_idx] for r in rep_order}
            g_train = None if groups is None else np.asarray(groups)[train_idx]

            # ---- BASE ARM: each selected representation, scored whole ----------
            if arm in ('base', 'both'):
                t_fold = time.time()
                n_calls = 0
                for method in imp_methods:
                    for rname in base_names:
                        for lv in levels:
                            y_tr = per_level[lv][fold_i]
                            # Level 0 scored twice: the reference, and the floor
                            # with another scorer seed.
                            runs = [('base', 42 + fold_i)]
                            if lv == 0.0:
                                runs.append(('base_floor',
                                             42 + fold_i + FLOOR_SEED_OFFSET))
                            for tag, seed in runs:
                                try:
                                    v, secs = _score_one(
                                        train_reps[rname], y_tr, method,
                                        seed=seed, dataset=dataset_name)
                                except Exception as exc:
                                    msg = f"{type(exc).__name__}: {exc}"
                                    print(f"    {tag} {method}/{rname} level "
                                          f"{lv:g}: {msg}")
                                    results['errors'].append(
                                        {'arm': tag, 'method': method,
                                         'rep': rname, 'level': lv,
                                         'fold': fold_i, 'error': msg})
                                    continue
                                rep_block[(tag, method, rname, lv, fold_i)] = \
                                    v.astype(np.float32)
                                n_calls += 1
                                results['scoring_seconds'].append(
                                    {'arm': tag, 'method': method, 'rep': rname,
                                     'level': lv, 'fold': fold_i,
                                     'n_features': int(v.size),
                                     'seconds': round(secs, 3)})
                print(f"    [d{replicate} f{fold_i}] base arm {base_names}: "
                      f"{n_calls} scores in {time.time() - t_fold:.0f}s",
                      flush=True)

            # ---- HYBRID ARM: which features each recipe takes ------------------
            if arm in ('hybrid', 'both'):
                for m in models:
                    entry = sel.get(m)
                    if entry is None or m not in factories:
                        continue
                    for space, slot in HYBRID_SLOTS:
                        rec = (entry.get('recipes') or {}).get(space)
                        if rec is None or rec[0] is None:
                            continue
                        bp, aug_cond = rec
                        cfg = f'{m}.{slot}'
                        augs = (augmentations
                                if (aug_cond or 'none') != 'none' else None)
                        for lv in levels:
                            y_tr = per_level[lv][fold_i]
                            t0 = time.time()
                            built = _build_hybrid(
                                dataset_name, train_reps, val_reps, y_tr, bp,
                                g_train, m, factories[m], metric_fn,
                                metric_name, lower_is_better, train_idx,
                                val_idx, augs)
                            if built[0] is None:
                                msg = (built[2] if isinstance(built[2], str)
                                       else 'the hybrid build failed')
                                print(f"    hybrid {cfg} level {lv:g} fold "
                                      f"{fold_i}: {msg}")
                                results['errors'].append(
                                    {'arm': 'hybrid', 'config': cfg,
                                     'level': lv, 'fold': fold_i,
                                     'error': msg})
                                continue
                            X_tr_h, feat_info = built[0], built[2]
                            secs = time.time() - t0
                            # WHICH features the recipe took, per representation.
                            for rname in rep_order:
                                info = feat_info.get(rname)
                                if not isinstance(info, dict):
                                    continue
                                sel_idx = np.asarray(
                                    info.get('selected_indices', []), dtype=int)
                                rep_block[('hybrid', cfg, rname, lv, fold_i)] = \
                                    sel_idx
                            results['scoring_seconds'].append(
                                {'arm': 'hybrid', 'method': 'recipe',
                                 'config': cfg, 'level': lv, 'fold': fold_i,
                                 'n_features': None, 'seconds': round(secs, 3)})
                            print(f"    [d{replicate} f{fold_i} level {lv:g}] "
                                  f"{cfg} built in {secs:.1f}s")

                            # HOW THE HYBRID'S OWN FEATURES RANK, so the hybrid
                            # gets the same rows the base representations get.
                            X_h = np.asarray(X_tr_h)
                            srcs = _hybrid_column_sources(feat_info, X_h.shape[1])
                            if srcs is None:
                                msg = (f"could not map {X_h.shape[1]} hybrid "
                                       f"columns onto the recipe's selected "
                                       f"indices; the rank rows for this cell "
                                       f"are dropped rather than guessed")
                                print(f"    hybrid {cfg} level {lv:g} fold "
                                      f"{fold_i}: {msg}")
                                results['errors'].append(
                                    {'arm': 'hybrid', 'config': cfg,
                                     'method': 'columns', 'level': lv,
                                     'fold': fold_i, 'error': msg})
                                continue
                            rep_block[('hybrid_src', cfg, '__all__', lv,
                                       fold_i)] = srcs
                            for method in imp_methods:
                                runs = [('hybrid_imp', 42 + fold_i)]
                                if lv == 0.0:
                                    runs.append(('hybrid_imp_floor',
                                                 42 + fold_i + FLOOR_SEED_OFFSET))
                                for tag, seed in runs:
                                    try:
                                        v, hsecs = _score_one(
                                            X_h, y_tr, method, seed=seed,
                                            dataset=dataset_name)
                                    except Exception as exc:
                                        msg = f"{type(exc).__name__}: {exc}"
                                        print(f"    {tag} {cfg} {method} level "
                                              f"{lv:g}: {msg}")
                                        results['errors'].append(
                                            {'arm': tag, 'config': cfg,
                                             'method': method, 'level': lv,
                                             'fold': fold_i, 'error': msg})
                                        continue
                                    rep_block[(tag, method, cfg, lv, fold_i)] = \
                                        v.astype(np.float32)
                                    results['scoring_seconds'].append(
                                        {'arm': tag, 'method': method,
                                         'config': cfg, 'level': lv,
                                         'fold': fold_i,
                                         'n_features': int(v.size),
                                         'seconds': round(hsecs, 3)})

            ck.save(f'r{replicate}f{fold_i}',
                    {'results': results, 'replicate': replicate,
                     'rep_block': rep_block})

        results['per_replicate'][str(replicate)] = _summarise_replicate(
            rep_block, rep_order, imp_methods, levels, fold_ids, reps, top_k,
            arm, _hybrid_configs(models))

        # Per-feature vectors go beside the JSON, not into it.
        _save_vectors(rep_block, output_dir, dataset_name, condition, replicate,
                      results)

        ck.save(f'r{replicate}', {'results': results, 'replicate': None})

    results['seconds_total'] = round(time.time() - t_start, 1)
    results['summary'] = _summarise(results, levels, imp_methods, arm)
    # Complete = every draw and every fold of the split ran. A run under
    # --max-folds or KIRBY_NOISE_ONLY_FOLDS is written with complete False and
    # n16_analyse says so.
    results['complete'] = bool(
        len(results['per_replicate']) == n_replicates
        and len(fold_ids) == n_folds)
    print(f"\n  Done in {results['seconds_total']:.1f}s. "
          f"{len(results['errors'])} cells failed.")
    # sweep_conditions writes this condition's file from the return value.
    ck.clear()
    return convert_numpy(results)


def _hybrid_configs(models):
    return [f'{m}.{slot}' for m in models for _sp, slot in HYBRID_SLOTS]


def _build_hybrid(dataset, train_reps, val_reps, y_tr, bp, g_train, model_name,
                  factory, metric_fn, metric_name, lower_is_better, train_idx,
                  val_idx, augs):
    """One per-fold recipe rebuilt on these training labels, with the decided
    model as the admission model, through noise_cache.cached_hybrid_for_fold
    (the same route noise_helpers.fold_model_inputs takes). The recipe's
    representation scores are the fold search's own (dirty-run bug 1).
    Returns (X_train, X_val, feature_info), or (None, None, why)."""
    from experiments.noise.noise_helpers import recipe_is_geometry
    geometry = recipe_is_geometry(bp)
    try:
        built = nc.cached_hybrid_for_fold(
            dataset, train_reps, val_reps, y_tr, bp, g_train,
            None if geometry else factory(), metric_fn, lower_is_better,
            model_name=None if geometry else model_name,
            metric_name=metric_name, train_idx=np.asarray(train_idx),
            augmentations_train=(None if augs is None else
                                 {a: X[train_idx] for a, X in augs.items()}),
            augmentations_val=(None if augs is None else
                               {a: X[val_idx] for a, X in augs.items()}),
            want_feature_info=True)
    except Exception as exc:
        return None, None, f'build raised {type(exc).__name__}: {exc}'
    if built is None or built[0] is None:
        return None, None, 'the hybrid build failed'
    return built


def _save_vectors(rep_block, output_dir, dataset_name, condition, replicate, results):
    """Write one draw's per-feature arrays next to the dataset's JSON."""
    if output_dir is None:
        return
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


def _spread(rows, key):
    """fold_spread of one figure over the per-fold rows that carry it."""
    return fold_spread([r.get(key) for r in rows])


def _base_cells(rep_block, method, rname, levels, fold_ids, width, top_k):
    """The per-level cells of one (method, representation) in the base arm."""
    cells = {}
    for lv in levels:
        per_fold, sets = [], []
        for f in fold_ids:
            v = rep_block.get(('base', method, rname, lv, f))
            ref = rep_block.get(('base', method, rname, 0.0, f))
            if v is None or ref is None:
                continue
            v = np.asarray(v, dtype=float)
            ref = np.asarray(ref, dtype=float)
            row = {'fold': f, 'n_scored': int((v != 0).sum()),
                   'n_features': int(v.size)}
            row.update(_rank_stats(ref, v, top_k))
            row.update(_turnover(ref, v, top_k))
            per_fold.append(row)
            k = int(min(top_k, v.size))
            sets.append(set(np.argsort(v)[-k:].tolist()))
        if not per_fold:
            continue
        cells[f'{lv:g}'] = _cell_from_rows(per_fold)
        cells[f'{lv:g}']['nogueira_topk_stability_across_folds'] = (
            nogueira_stability(sets, width) if len(sets) > 1 else
            'one fold, no spread')
    # THE FLOOR: level 0.0 against itself under another scorer seed.
    floor_rows = []
    for f in fold_ids:
        ref = rep_block.get(('base', method, rname, 0.0, f))
        flo = rep_block.get(('base_floor', method, rname, 0.0, f))
        if ref is None or flo is None:
            continue
        row = {'fold': f, 'n_scored': int((np.asarray(flo) != 0).sum()),
               'n_features': int(np.asarray(flo).size)}
        row.update(_rank_stats(ref, flo, top_k))
        row.update(_turnover(ref, flo, top_k))
        floor_rows.append(row)
    if floor_rows:
        cells['floor_seed'] = _cell_from_rows(floor_rows)
    return cells


def _cell_from_rows(per_fold):
    """One cell: the per-fold rows and each figure's spread over folds."""
    return {
        'n_folds': len(per_fold),
        'per_fold': per_fold,
        'n_scored_features': _spread(per_fold, 'n_scored'),
        'spearman_either_scored': _spread(per_fold, 'spearman_either_scored'),
        'spearman_both_scored': _spread(per_fold, 'spearman_both_scored'),
        'spearman_all_features': _spread(per_fold, 'spearman_all'),
        'kendall_either_scored': _spread(per_fold, 'kendall_either_scored'),
        'topk_retained_fraction': _spread(per_fold, 'retained_fraction'),
        # The readable one: the top set capped at how many features the scorer
        # actually ranked, so it holds no tied zeros.
        'topk_effective_retained_fraction': _spread(
            per_fold, 'retained_fraction_effective'),
        'topk_effective_size': _spread(per_fold, 'top_k_effective'),
    }


def _summarise_replicate(rep_block, rep_order, imp_methods, levels, fold_ids,
                         reps, top_k, arm, hybrid_configs):
    """Per (method, representation or hybrid configuration, level): the three
    rows against level 0.0, plus the floor. Every figure over folds is a
    noise_helpers.fold_spread (mean, sd, per fold; 'one fold, no spread' on
    one fold)."""
    out = {'base': {}, 'hybrid': {}, 'hybrid_importance': {}}

    if arm in ('base', 'both'):
        for method in imp_methods:
            for rname in rep_order:
                cells = _base_cells(rep_block, method, rname, levels, fold_ids,
                                    int(reps[rname].shape[1]), top_k)
                if cells:
                    out['base'].setdefault(method, {})[rname] = cells

    if arm in ('hybrid', 'both'):
        # ROW 1 and ROW 2 for each hybrid configuration's own features. Two levels
        # select different features, so the vectors cannot be lined up
        # elementwise; they are keyed by (representation, source index) and
        # correlated over the features BOTH levels selected. The size of that
        # overlap is reported next to every correlation.
        for method in imp_methods:
            for cfg in hybrid_configs:
                cells = {}
                for lv in levels:
                    per_fold = []
                    for f in fold_ids:
                        v = rep_block.get(('hybrid_imp', method, cfg, lv, f))
                        ref = rep_block.get(('hybrid_imp', method, cfg, 0.0, f))
                        smap = rep_block.get(('hybrid_src', cfg, '__all__', lv, f))
                        rmap = rep_block.get(('hybrid_src', cfg, '__all__', 0.0, f))
                        if v is None or ref is None or smap is None or rmap is None:
                            continue
                        cur = dict(zip([tuple(s) for s in smap],
                                       np.asarray(v, dtype=float)))
                        base = dict(zip([tuple(s) for s in rmap],
                                        np.asarray(ref, dtype=float)))
                        shared = sorted(set(cur) & set(base))
                        row = {
                            'fold': f,
                            'n_hybrid_features': int(np.asarray(v).size),
                            'n_hybrid_features_at_level_zero':
                                int(np.asarray(ref).size),
                            'n_features_selected_at_both_levels': len(shared),
                            'n_scored': int((np.asarray(v, dtype=float) != 0).sum()),
                        }
                        if len(shared) >= MIN_SHARED_FOR_RANK:
                            a = np.array([base[k] for k in shared], dtype=float)
                            b = np.array([cur[k] for k in shared], dtype=float)
                            row.update(_rank_stats(a, b, top_k))
                            row.update(_turnover(a, b, top_k))
                        per_fold.append(row)
                    if not per_fold:
                        continue
                    ranked = [r for r in per_fold if 'spearman_either_scored' in r]
                    cell = {
                        'n_folds': len(per_fold),
                        'n_folds_with_enough_shared_features': len(ranked),
                        'per_fold': per_fold,
                        'n_hybrid_features': _spread(per_fold, 'n_hybrid_features'),
                        'n_scored_features': _spread(per_fold, 'n_scored'),
                        'n_features_selected_at_both_levels': _spread(
                            per_fold, 'n_features_selected_at_both_levels'),
                    }
                    if ranked:
                        cell.update({
                            'spearman_either_scored': _spread(
                                ranked, 'spearman_either_scored'),
                            'spearman_both_scored': _spread(
                                ranked, 'spearman_both_scored'),
                            'spearman_all_features': _spread(ranked, 'spearman_all'),
                            'kendall_either_scored': _spread(
                                ranked, 'kendall_either_scored'),
                            'topk_effective_retained_fraction': _spread(
                                ranked, 'retained_fraction_effective'),
                            'topk_effective_size': _spread(ranked, 'top_k_effective'),
                        })
                    cells[f'{lv:g}'] = cell
                # THE FLOOR: the level-0.0 hybrid scored again with another seed.
                floor_rows = []
                for f in fold_ids:
                    ref = rep_block.get(('hybrid_imp', method, cfg, 0.0, f))
                    flo = rep_block.get(('hybrid_imp_floor', method, cfg, 0.0, f))
                    if ref is None or flo is None:
                        continue
                    row = {'fold': f,
                           'n_scored': int((np.asarray(flo) != 0).sum())}
                    row.update(_rank_stats(ref, flo, top_k))
                    row.update(_turnover(ref, flo, top_k))
                    floor_rows.append(row)
                if floor_rows:
                    cells['floor_seed'] = _cell_from_rows(floor_rows)
                if cells:
                    out['hybrid_importance'].setdefault(method, {})[cfg] = cells

        # ROW 3: which SOURCE features each recipe took, per representation.
        # The recipe differs between folds (each fold has its own search), so
        # the across-fold Nogueira figure here mixes recipes; the per-fold
        # retained fraction against level 0.0 is the figure to read.
        for cfg in hybrid_configs:
            for rname in rep_order:
                width = int(reps[rname].shape[1])
                cells = {}
                for lv in levels:
                    per_fold, sets = [], []
                    for f in fold_ids:
                        sel = rep_block.get(('hybrid', cfg, rname, lv, f))
                        ref = rep_block.get(('hybrid', cfg, rname, 0.0, f))
                        if sel is None or ref is None:
                            continue
                        s = set(np.asarray(sel).tolist())
                        r = set(np.asarray(ref).tolist())
                        per_fold.append({
                            'fold': f,
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
                        'n_selected_features': _spread(
                            per_fold, 'n_selected_features'),
                        'retained_fraction_of_level_zero': _spread(
                            per_fold, 'retained_fraction_of_level_zero'),
                        'nogueira_selection_stability_across_folds': (
                            nogueira_stability(sets, width) if len(sets) > 1
                            else 'one fold, no spread'),
                    }
                if cells:
                    out['hybrid'].setdefault(cfg, {})[rname] = cells

    return out


def _mean_of(spread):
    """The mean out of a fold_spread dict, or None."""
    return spread.get('mean') if isinstance(spread, dict) else None


def _summarise(results, levels, imp_methods, arm):
    """The one screen a reader needs: what moved, per scoring method, per level,
    beside the floor.

    Never pooled across datasets and never averaged with another dataset's numbers —
    a Spearman on rmse-scored logd and on spearman-scored flab are not the same
    quantity. Every figure here is within one dataset. A median here is over the
    (draw, representation) cells of one level, each cell being a mean over folds.
    """
    summary = {'base': {}, 'hybrid': {}, 'hybrid_importance': {}}
    reps_per_rep = results['per_replicate']
    keys = [f'{lv:g}' for lv in levels] + ['floor_seed']

    def _row(vals_rho, vals_scored, vals_kept):
        vals_rho = [x for x in vals_rho if x is not None and np.isfinite(x)]
        if not vals_rho:
            return None
        out = {'n_cells': len(vals_rho),
               'median_spearman_either_scored_vs_level_zero':
                   float(np.median(vals_rho)),
               'min_spearman_either_scored_vs_level_zero': float(np.min(vals_rho)),
               'max_spearman_either_scored_vs_level_zero': float(np.max(vals_rho))}
        sc = [x for x in vals_scored if x is not None]
        if sc:
            out['median_n_scored_features'] = float(np.median(sc))
        kp = [x for x in vals_kept if x is not None and np.isfinite(x)]
        if kp:
            out['median_topk_effective_retained_fraction'] = float(np.median(kp))
        return out

    if arm in ('base', 'both'):
        for method in imp_methods:
            per_level = {}
            for key in keys:
                rhos, scored, kept = [], [], []
                for block in reps_per_rep.values():
                    for rname, cells in block.get('base', {}).get(method, {}).items():
                        c = cells.get(key)
                        if c is None:
                            continue
                        rhos.append(_mean_of(c['spearman_either_scored']))
                        scored.append(_mean_of(c['n_scored_features']))
                        kept.append(_mean_of(c['topk_effective_retained_fraction']))
                r = _row(rhos, scored, kept)
                if r:
                    per_level[key] = r
            if per_level:
                summary['base'][method] = per_level

    if arm in ('hybrid', 'both'):
        for method in imp_methods:
            per_level = {}
            for key in keys:
                rhos, scored, kept, shared = [], [], [], []
                for block in reps_per_rep.values():
                    for cfg, cells in (block.get('hybrid_importance', {})
                                       .get(method, {}).items()):
                        c = cells.get(key)
                        if c is None:
                            continue
                        scored.append(_mean_of(c.get('n_scored_features')))
                        if 'n_features_selected_at_both_levels' in c:
                            shared.append(_mean_of(
                                c['n_features_selected_at_both_levels']))
                        if 'spearman_either_scored' in c:
                            rhos.append(_mean_of(c['spearman_either_scored']))
                            kept.append(_mean_of(
                                c['topk_effective_retained_fraction']))
                r = _row(rhos, scored, kept) or {}
                sh = [x for x in shared if x is not None]
                if sh:
                    r['median_n_features_selected_at_both_levels'] = float(
                        np.median(sh))
                    r.setdefault('n_cells', len(sh))
                if r:
                    per_level[key] = r
            if per_level:
                summary['hybrid_importance'][method] = per_level

        per_level = {}
        for lv in levels:
            kept, nsel = [], []
            for block in reps_per_rep.values():
                for cfg, by_rep in block.get('hybrid', {}).items():
                    for rname, cells in by_rep.items():
                        c = cells.get(f'{lv:g}')
                        if c is None:
                            continue
                        kept.append(_mean_of(c['retained_fraction_of_level_zero']))
                        nsel.append(_mean_of(c['n_selected_features']))
            kept = [x for x in kept if x is not None and np.isfinite(x)]
            nsel = [x for x in nsel if x is not None]
            if not kept:
                continue
            per_level[f'{lv:g}'] = {
                'n_cells': len(kept),
                'median_retained_fraction_of_level_zero': float(np.median(kept)),
                'median_n_selected_features': (float(np.median(nsel))
                                               if nsel else None),
            }
        if per_level:
            summary['hybrid'] = per_level

    return summary


if __name__ == '__main__':
    parser = create_experiment_parser(
        'N16: feature importance across noise levels')
    parser.add_argument('--noise-condition', default=None,
                        help='one damage kind (default: the six decided '
                             'regression kinds, swept)')
    parser.add_argument('--replicates', type=int, default=N_REPLICATES,
                        help='independent noise draws per level (default: 3)')
    parser.add_argument('--arm', default='both',
                        choices=['base', 'hybrid', 'both'],
                        help='base representations, the hybrid selection, or both')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help='size of the tracked top-of-the-ranking set')
    parser.add_argument('--levels', type=str, default=None,
                        help='comma-separated noise levels; must include 0.0 '
                             '(default: the decided grid for this dataset)')
    parser.add_argument('--importance-methods', type=str, default=None,
                        help='comma-separated scoring methods '
                             '(default: deeplift,lightgbm_gain,treeshap)')
    args = parser.parse_args()
    config = get_config(args)
    # The signed-off pool comes for free: load_data auto-filters to
    # DATASET_REPS[ds]['general'] whenever rep_names is None. A comma-separated
    # --reps goes through parse_rep_args and overrides that.
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names)
    # Through run(), so a standalone run sweeps the decided conditions exactly
    # as the suite does, and writes <dataset>__<condition>.json per condition
    # as each finishes (sweep_conditions). Before 2026-10-05 a standalone run
    # called one condition and saved once at the end.
    kw = dict(replicates=args.replicates, arm=args.arm, top_k=args.top_k,
              levels=([float(x) for x in args.levels.split(',')]
                      if args.levels else None),
              importance_methods=(args.importance_methods.split(',')
                                  if args.importance_methods else None))
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    index = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(index, args.output, 'n16_importance_under_noise',
                            args.dataset)
