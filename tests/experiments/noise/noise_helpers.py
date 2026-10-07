"""Shared configuration and runner for per-dataset noise experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from the pipeline readiness tiers.
Every experiment here runs on one dataset. All but one are independent of each
other and may run in any order; n06 is not, and runs last. Its headline is the
rank correlation between its own ordering of the noise estimators and the
ordering n11 and n13 measured under injected noise, and it reads those two out
of the files they write in the same run.

Experiments:
  N01-N08: Original noise suite (cliffs, weighting, SAS, GP-LOO, detection,
           remediation, estimation survey)
  N06: real measurement disagreement — repeat measurements and two curators,
       never injected noise
  N09: LNL method survey with hybrids (12 training-loop methods)
  N10: Hybrid neighborhood for similarity-dependent LNL
  N11: Noise estimation × remediation grid (full confirmation report grid)
  N12: Feature importance-weighted distance for noise estimation
  N13: Alternative noise distributions (uniform, multiplicative, heavy-tailed, partial)
  N14: Activity-cliff landscape survey (gate; trains no downstream models)
  N15: Clean-searched vs noise-searched hybrid recipe (oracle advantage)
  N16: Feature importance across noise levels (gate; trains no downstream models)

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

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy, should_skip_experiment,
    experiment_result_is_empty,
    DOWNSTREAM_MODELS_NOISE, INTERNAL_EVALUATORS,
    input_fingerprint, fingerprint_mismatch,
)

# Reuse dataset rep configs — single source of truth
from experiments.complexity.complexity_helpers import DATASET_REPS


# ── Experiment list ────────────────────────────────────────────────────────
# All standalone, no ordering constraints.

SINGLE_DATASET_EXPERIMENTS = [
    # c10 kept its name when it moved here from the complexity suite on
    # 2026-09-12 (your decision). Renaming it to an n-number would break every
    # reference to c10 in EXPERIMENTS_FINALIZED.md and the result directory
    # every stored file already sits in.
    'c10',   # noise robustness of the selected features (moved from complexity)
    'n01',   # cliff-stratified prediction error (sub-4 only; sub-1/2/3 cut)
    'n02',   # importance-weighted sample weighting
    'n07',   # noise remediation strategies
    'n09',   # LNL method survey with hybrids (12 methods)
    'n11',   # noise estimation x remediation grid
    'n12',   # feature importance-weighted distance
    'n13',   # alternative noise distributions
    'n14',   # activity-cliff landscape survey (gate; no downstream models)
    'n15',   # clean-searched vs noise-searched hybrid blueprint (oracle advantage)
    'n16',   # feature importance across noise levels (gate; no downstream models)
    # n17 injects nothing. It trains on ONE real measurement per entity and
    # grades on the mean of that entity's others, so the noise is the
    # disagreement between laboratories. A dataset whose loader collapses its
    # repeats before anything sees them writes a skip carrying the reason,
    # which as of 2026-09-16 is most of them.
    'n17',   # removal on real repeat-measurement disagreement
    # n18 injects nothing either, and unlike n17 it needs no molecule measured
    # twice. It trains on the labels as they come, cleans them, trains again,
    # and counts the folds the cleaned model won. So it runs on every dataset.
    'n18',   # does cleaning help on the labels as they come
    # n19 injects nothing either. It needs a loader that kept one row per
    # MEASUREMENT, which as of 2026-09-16 is herg_ic50 and bindingdb_kd; every
    # other dataset writes a skip carrying the reason.
    'n19',   # the measurement count and spread as weight and as shrinkage
    # n20 searches its recipe inside each fold rather than reading the pool,
    # because which labels a search was allowed to see is its whole subject.
    'n20',   # does the cleaning have to come before the recipe search
    # n22 injects nothing. Its suspect set is ChEMBL's own curator field, which
    # reaches an experiment only through the herg_ic50 loader; every other
    # dataset writes a skip carrying the reason, the same way n17 and n19 do.
    # It is OUTLIER noise and nothing else -- a value out of range or a number
    # typed in wrong (your ruling, 2026-09-17, NOISE_AUDIT_2026_09_10.md 16.5).
    'n22',   # detector precision and recall against ChEMBL's curator flag
    # n21 and n23 were written on 2026-09-17/18 and registered in the import
    # maps below but never in THIS list, so neither could be reached through
    # run_dataset.py or a SLURM job -- only by calling the module by hand.
    # n21 takes a condition (HONOURS_NOISE_CONDITION, below) and runs only
    # under the five in DETECTION_IS_DEFINED; its default is outlier_p10.
    'n21',   # which distance the nearest neighbours are found in
    'n23',   # cleaning driven by a per-molecule uncertainty estimate
    # n24 and n25 were written and committed (4c85148, a4f5579) but never
    # listed here or in the two import maps below, so run_dataset.py could not
    # reach them -- the same gap n21 and n23 had. Registered 2026-09-25. Both
    # take a condition (n24:435, n25:786); n25 only the five in
    # DETECTION_IS_DEFINED (n25:787-793), see CONDITION_LIMITED_TO.
    'n24',   # delete the flagged molecules, rewrite them, or weigh them down
    'n25',   # exempting activity-cliff molecules from cleaning
    # n26-n30 added 2026-09-24 from an outside research report on why cleaning
    # fails at this accuracy (n28's arms went into n19 instead of a new file).
    # n26 and n29 inject nothing and need repeat measurements; every dataset
    # without them writes a skip carrying the reason, as n17 and n19 do. n29
    # also needs the ChEMBL document of each measurement, herg_ic50 only.
    # n30 names its own condition, censoring, so it runs in the job with no
    # condition and is left out of every job that names one.
    'n26',   # label-noise variance from repeat measurements, beside model error
    'n27',   # what fewer training molecules costs; how label error reaches test
    'n29',   # one offset per ChEMBL document, fitted and removed
    'n30',   # clipped labels: lift, drop, winsorise, or censored likelihood
    # n31-n36 added 2026-09-24 from a second outside report, on whether cleaning
    # pays at higher R². n32, n33 and n34 take a condition. n31 names its own
    # five conditions, two of them the shift_only_* ones outside the NoiseInject
    # registry, so it runs only in the job with no condition. n35 injects
    # gaussian by construction, like n27. n36 needs repeat measurements, like n19.
    'n31',   # predicted vs measured detector quality; where flagging breaks even
    'n32',   # trimmed refits, two learners, and four more detectors
    'n33',   # robust losses, target transform, variance weights, mixup
    'n34',   # a random offset per scaffold group instead of subtracting its mean
    'n35',   # label disagreement between near-identical molecules
    'n36',   # which molecules to measure a second time
    # n37-n41 added 2026-09-25 from a third outside report, on noise spread over
    # every label against rare large errors. n40 and n41 take a condition. n38
    # sweeps the conditions sized by the model's own error (RELATIVE_CONDITIONS),
    # which are not in the registry, so like n31 it runs only in the job with no
    # condition. n37 is diagnostics on clean labels; n39 injects its own
    # simulated sources.
    'n37',   # how large the label noise is against the model error
    'n38',   # noise sized by the model's error, not by the label spread
    'n39',   # repeat measurements from sources whose errors are correlated
    'n40',   # retuned models, seed averaging, stacking, a GP with a noise term
    'n41',   # flags with a controlled false-flag rate, and three more detectors
    # n42 added 2026-09-25 (plan section 6): every data-changing method the
    # files above own, imported from them, crossed with every similarity
    # setting and every model input, one D-versus-D' row per fit. It takes a
    # condition (HONOURS_NOISE_CONDITION) and sweeps conditions_for_task when
    # given none. Before n06 because n06 must stay last.
    'n42',   # every mitigation method x similarity setting x model input
    # n06 runs LAST, and that is load-bearing: its headline is the rank
    # correlation between its own ordering of the noise estimators and the
    # ordering n11 and n13 measured under injected noise, and it reads those
    # two orderings out of the result files they write in this same run. Put it
    # earlier and it finds no stored ranking for the dataset in hand and
    # records that instead of the number.
    'n06',   # real measurement disagreement, from a ground-truth provider
]
# CUT 2026-07-07 (irredeemable by design — see AUDIT_NOISE.md):
# n03 sas_landscape, n04 gp_loo_detection, n05 noise_injection_benchmark,
# n08 estimation_survey, n10 hybrid_neighborhood_lnl.

# n06 was excluded from this list until 2026-09-11, on the grounds that it was
# dataset-specific and had to be run by hand. It no longer is: its ground truth
# comes from a provider, and a dataset where no provider finds a repeat
# structure writes a skip carrying the count it found rather than being left
# out of the run.


# ── The levels n15 sweeps ──────────────────────────────────────────────────
# This used to be the grid a noise-aware blueprint was PRECOMPUTED at. That
# precompute was deleted on 2026-09-11 (see the block further down), so this
# constant now controls exactly one thing: the levels n15 runs at. Nothing has
# to be kept in step with it any more.
#
# Levels are TARGET EFFECTIVE NOISE: the noise actually DELIVERED, measured as a
# root-mean-square, as a fraction of the label SD. A level therefore means the same
# thing on every dataset and in every condition — unlike the five ad-hoc, mutually
# incompatible `_inject_noise` definitions the suite grew, four of which scaled by the
# label RANGE and so ran at roughly 4-6x the level they claimed.
#
# CHANGED 2026-09-12, user decision. It was (0.05, 0.10, 0.20), and no value in
# it appeared on the shared grid n09/n11/n12/n13 sweep — so n15's answer could
# not be read against the experiments it exists to settle. It is now the bottom
# three entries of that shared grid (experiments.dataset_policy line 376).
#
# 0.0 is a control, not a wasted run. At level 0.0 the labels are untouched, so
# both recipes see the same values and the gap between them should be zero. It
# is not guaranteed to be: the clean recipe comes from the searches held in
# memory, built with losing allocation methods dropped after fold 0
# (`default_pruning`, threshold 0.15), while the noisy one goes through
# get_precomputed_search, which sets adaptive_pruning=False (_shared.py:5125).
# So the two searches choose from lists of different lengths even when the
# labels are identical, and the 0.0 row measures that on its own. Every other
# level can have it subtracted.
#
# NOT covered: 0.5, 0.75 and 1.0, the top three of the shared six. 0.5 is the
# level dataset_policy.reference_noise_level() names for any experiment needing
# one number instead of a sweep, so a reading of n15 cannot be carried to it.
# Whether the clean-label advantage grows or shrinks as the noise rises is not
# measured anywhere, so nothing here says which way that matters.
#
# ONE noise_seed. The blueprint should not swing with the random draw; if it does,
# that is itself the finding, and it is what the clean-vs-noisy experiment measures.
# ⛔ THE WHOLE SHARED GRID SINCE 2026-09-18, OWNER'S DECISION.
#
# This was (0.0, 0.15, 0.3) — the bottom three of the six every other noise
# experiment sweeps. n15 measures how much a hybrid gains from having been
# searched on the clean labels, and n11 and n12 carry ORACLE-BIASED warnings
# that only come off if that gain is near zero. Measuring it at 0.0, 0.15 and
# 0.3 and not at 0.5, 0.75 and 1.0 answered the question exactly where the
# corruption is mildest and left it open where it is worst — which is where an
# oracle head start would be largest.
#
# It is not a constant any more. The grid differs by task type: regression
# sweeps (0.0, 0.15, 0.3, 0.5, 0.75, 1.0) and classification (0.0, 0.05, 0.1,
# 0.2, 0.3, 0.4), because on a classification dataset the level is a flip rate
# and 1.0 would inverst every label. noisy_search_specs reads
# dataset_policy.noise_levels() so n15's numbers can be read straight against
# every other noise experiment's.
#
# ⛔ WHAT IT COSTS. Every level is an extra RECIPE SEARCH per arm, not an extra
# round of model fits, because n15 searches inside the experiment. Draws go
# from 7 (one at level 0.0 plus three at each of two levels) to 16 (one plus
# three at each of five) on a regression dataset, so about 2.3 times the
# searches. The fast tier and the aggressive pruning schedule in n15 are what
# absorb it.
#
# Kept as the name every reader knows, now a fallback for a caller that names
# no task type.
# ⛔ A FALLBACK ONLY. The live path above reads dataset_policy.noise_levels, so
# this fires only when the policy cannot answer. It is kept in step with
# _DEFAULT_NOISE_LEVELS by hand, and the two disagreeing silently is the whole
# hazard: a fallback that differs from the shared ladder would put n15's curve on
# a different axis from every other experiment's while still looking right.
# Corrected 2026-09-21 with the policy, from (0.0, 0.15, 0.3, 0.5, 0.75, 1.0).
NOISY_SEARCH_LEVELS = (0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
NOISY_SEARCH_SEED = 0  # kept for readers; the sweep uses NOISY_SEARCH_SEEDS below


NOISY_SEARCH_SEEDS = (0, 1, 2)


def noisy_search_specs(task_type: str, noise_seeds=None, dataset_name=None):
    """The NoiseSpec list n15 sweeps, for this task type.

    THREE DRAWS PER LEVEL, not one. Owner's decision 2026-09-12. n07, n09, n11,
    n12 and n13 already average three corruptions; n15 averaged one, so a single
    unlucky draw set its whole answer. Since the level came out of the seed the
    same day, one draw now serves the whole level sweep, which makes a single
    draw worse rather than better.

    n15 pays more for this than the others do. Everywhere else an extra draw is
    an extra round of model fits; here it is an extra RECIPE SEARCH per level per
    arm, because n15 searches its recipe inside the experiment. That cost is what
    the fast tier and the adaptive pruning schedule in n15 are there to absorb.

    At level 0.0 on a regression dataset the draw delivers exactly zero, so three
    draws are three identical runs and one is kept. NOT on a classification set:
    calibrate_flip_probability returns a floor rather than zero at a target of
    0.0 — measured at 0.001953 on 200 labels and 0.0078125 on 400 — so a few
    labels flip there, and a different number of them on every draw.
    """
    from kirby.noise_spec import DEFAULT_STRATEGY_CLF, DEFAULT_STRATEGY_REG
    is_clf = (task_type == 'classification')
    cond = DEFAULT_STRATEGY_CLF if is_clf else DEFAULT_STRATEGY_REG
    seeds = tuple(NOISY_SEARCH_SEEDS if noise_seeds is None else noise_seeds)
    # The shared grid for THIS task type, so n15's levels line up with every
    # other noise experiment's and the numbers can be read against each other.
    # Falls back to the module constant if the policy cannot answer.
    try:
        from experiments.dataset_policy import noise_levels as _policy_levels
        levels = tuple(_policy_levels(None, is_clf=is_clf))
    except Exception:
        levels = NOISY_SEARCH_LEVELS
    if not levels:
        levels = NOISY_SEARCH_LEVELS
    out = []
    for lvl in levels:
        lvl_seeds = seeds if (lvl > 0 or is_clf) else seeds[:1]
        for sd in lvl_seeds:
            # noise_spec_for, the suite's one rule, since 2026-09-30 (owner:
            # every experiment injects noise the same way). default_spec seeded
            # the draw with the bare draw index (0, 1, 2), so n15's gaussian
            # noise was a different draw from every other experiment's at the
            # same condition, level and draw.
            out.append(noise_spec_for(dataset_name, cond, lvl,
                                      noise_seed=int(sd), is_clf=is_clf))
    return out


# ── ONE injector for the whole noise suite ─────────────────────────────────
# n07, n09, n11, n12 and n13 each carried a private `_inject_noise`, and no two of
# them meant the same thing: n07 corrupted a fraction of rows at 0.3 of the label
# range, n09/n11/n12 added Gaussian noise to every row at a fraction of the range,
# and n13 had four distributions of its own. None went through noiseInject. They are
# replaced by the two functions below.

def noise_draw_seed(condition, draw, base_seed=42):
    """THE seed of one noise draw: condition, draw index and base seed only.

    The level is not in it and the fold is not in it (noise_spec_for's notes
    below say why). Every injector in the suite uses this number -- the
    registry conditions through noise_spec_for, the shift and relative
    conditions through inject_shift_train_labels / inject_relative_train_labels
    -- so the same (condition, level, draw) is the same noise in every
    experiment. Made the one rule on 2026-09-30 (owner: "make it uniform");
    before that n31, n32/n41, n38 and n40 each seeded shift and relative
    conditions their own way, with the fold inside the seed.
    """
    import zlib
    digest = zlib.crc32(str(condition).encode('utf-8'))
    return (int(base_seed) * 1000003 + digest
            + int(draw) * 15485863) & 0x7FFFFFFF


def noise_spec_for(dataset_name, condition, level, base_seed=42,
                   noise_seed=0, is_clf=False):
    """A NoiseSpec whose draw is unique per (condition, level, draw, base seed).

    THE FOLD IS DELIBERATELY NOT IN IT. It used to be: the signature carried a
    `fold_idx` that went into both seeds, so every fold drew its own realisation
    and a molecule's corruption depended on which fold it landed in. Drawing over
    the whole label column is not enough on its own to fix that — the SPEC has to
    be the same in every fold too, or the whole-column draw is simply repeated
    with a different seed. Both halves are `RERUN_PLAN.md` 3.3b in
    ~/repos/qsar_qm_models, settled by the author 2026-09-04, whose own note says
    the injector's seed "already excludes the fold". KIRBy's did not. Removed
    2026-09-12; the argument is gone rather than ignored, so a call written for
    the old signature fails with a TypeError.

    What is kept is `noise_seed`, the replicate index n11 and n13 sweep as
    [0, 1, 2]. That is a deliberate axis: several realisations of the same
    condition at the same level, which is what protects a result from a freak
    draw. The fold never was that axis — it confounded the realisation with the
    split.

    The injector advances its generator on every draw, so re-using one across a
    sweep makes a level's realisation depend on how many levels ran before it —
    and a run of part of the grid then would not reproduce the matching rows of a
    full run. The seed depends on the CONDITION as well, or two conditions at one
    level would share a realisation and a comparison between them would measure
    nothing.

    zlib.crc32, NOT hash(): Python randomises string hashing per process, so hash()
    would make the noise different on every run.

    WHICH molecules are hit is seeded separately, from everything EXCEPT the level.
    Without that the outlier and grouped conditions pick a different set at every
    level, so a level sweep varies the amount and the molecules at once and the
    levels do not nest. Measured on noiseInject's 120-molecule fixture: the
    level-free pattern against the realised scale gave Spearman -0.101 for
    outlier_p10 before this, and 1.000 after (noiseInject d39d611, 2026-08-27).
    """
    import zlib
    from kirby.noise_spec import (CLASSIFICATION_STRATEGIES, DEFAULT_STRATEGY_CLF,
                                  NoiseSpec)
    from experiments.dataset_policy import noise_dose_units

    if is_clf and condition not in CLASSIFICATION_STRATEGIES:
        # A flipped class label has no magnitude, so none of the dose-matched
        # conditions has a classification counterpart and the level stops being a
        # dose and becomes a flip rate. Fall back to the symmetric flip model rather
        # than hand a regression condition to the classification injector, which
        # raises. An experiment that wants a specific flip model names one of
        # CLASSIFICATION_STRATEGIES instead.
        condition = DEFAULT_STRATEGY_CLF

    selection_seed = noise_draw_seed(condition, noise_seed, base_seed)
    # THE LEVEL IS NOT IN THE SEED. It used to be, as
    # `selection_seed + round(level*1000)*7919`, so going from level 0.15 to
    # level 0.30 threw the first set of label errors away and rolled a fresh
    # one. A molecule pushed up at 0.15 was pushed down at 0.30 about half the
    # time, so two points on a level curve differed by the draw as much as by
    # the amount of noise. Measured on 1000 labels, gaussian: the error at 0.15
    # and the error at 0.30 had Spearman +0.037 and pointed the same way on 500
    # of 1000 molecules -- a coin flip. Measured on ridge with 400 training
    # molecules and 10 draws per level, the spread across draws at one level was
    # 1.0x to 2.0x the step to the next level.
    #
    # With one seed for the whole sweep the levels are the same errors scaled:
    # measured, the error at 0.30 is exactly 2.0000 times the error at 0.15 on
    # every molecule, Spearman +1.000, same direction on 1000 of 1000. A level
    # curve then moves for one reason.
    #
    # `noise_seed` (the replicate index n07, n11 and n13 sweep as [0, 1, 2])
    # still varies the draw, so several independent curves are still available
    # -- that axis is what protects a result from an unlucky draw, and it is now
    # the ONLY thing that does. n09 and n12 run one replicate and so ride on one
    # draw; the SHAPE of their curve is sound, its absolute position is not.
    #
    # noiseInject XORs the selection stream with 0x5CA1E before use
    # (core.py:_selection_rng), so who gets hit and how hard stay separate
    # streams even though both are derived from this one number. The level is
    # still in `spec_id` as e####, so cache entries for two levels never collide.
    #
    # User decision, 2026-09-12.
    seed = selection_seed
    return NoiseSpec(strategy=condition, target_effective_noise=float(level),
                     noise_seed=seed, selection_seed=selection_seed,
                     dose_units=noise_dose_units(dataset_name))


def inject_train_labels(y_all, spec, *, train_idx, is_clf=False, groups=None,
                        dataset_name=None):
    """Corrupt one fold's TRAINING labels, and say what was corrupted.

    Args:
        y_all:     the FULL clean label column, every molecule in the dataset.
        spec:      a NoiseSpec, normally from noise_spec_for().
        train_idx: positions of this fold's training molecules in `y_all`.
                   Keyword-only and REQUIRED, so a call written for the old
                   per-fold signature fails with a TypeError instead of quietly
                   meaning something different.
        groups:    the FULL scaffold-group array from dataset_policy.
                   noise_groups_for(), or None. NOT the fold's slice.
        dataset_name: used only to look up the measurement error the labels
                   already carry, which goes onto the provenance row. Omitting
                   it leaves those three fields absent rather than wrong.

    Returns (y_noisy_train, corrupt_mask_train, provenance). The first two are
    already sliced to `train_idx`. `corrupt_mask_train` is the ground truth for
    "did the estimator find the corrupted molecules". `provenance` describes the
    whole-column draw, so it is the same row whichever fold asked for it.

    WHAT IS TAKEN FROM THE TRAINING LABELS
    --------------------------------------
    As the NoiseInject paper's Methods say: the level is a fraction of the
    standard deviation of this fold's clean TRAINING labels, the censoring limit
    is set from the clean training labels, and grouped-wider widens families
    covering 20% of the training molecules. Until 2026-09-27 all three came from
    the whole label column, so the held-out labels set how the training labels
    were corrupted. See `kirby.noise_spec.inject_training_rows`.

    The random draws are still made over the whole column and depend on no label,
    so a molecule keeps the same draw in every fold it trains in (RERUN_PLAN.md
    3.3b, 2026-09-04). The gate is
    `tests/test_noise_is_a_property_of_the_molecule.py`.

    Held-out molecules are never read here: only `train_idx` is returned. A
    molecule that is held out in one fold and training in another is corrupted in
    the second only, which is what the ruling asks for.

    A MISSED DRAW RAISES
    --------------------
    noiseInject checks what it delivered against the band its own tolerance
    allows, and warns. That warning cannot survive this process: find_best_hybrid
    and _shared each call warnings.filterwarnings('ignore') at import, and every
    noise experiment imports them. Measured: an injection asked for 0.5 label
    units and made to deliver 1.5016 printed nothing at all.

    So the warning is caught here and re-raised as noiseInject's own DoseError,
    which is what that class is exported for. The band is noiseInject's, computed
    by its own `dose_tolerance` — it is not re-implemented here, because a
    specification with two implementations is how the two halves of this study
    drifted apart before.
    """
    import warnings
    import numpy as np
    from noiseInject import DoseError, DoseWarning
    from kirby.noise_spec import make_noisy_train_labels, noise_provenance_row

    y_all = np.asarray(y_all)
    train_idx = np.asarray(train_idx)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_all):
            raise ValueError(
                f"inject_train_labels: groups has {len(groups)} entries and the "
                f"label column has {len(y_all)}. Pass the FULL group array from "
                f"dataset_policy.noise_groups_for(), not this fold's slice — the "
                f"draw runs over the whole column.")

    if is_clf:
        y_noisy_train, _ = make_noisy_train_labels(y_all, spec, train_idx,
                                                   is_clf=True)
        return (y_noisy_train, y_noisy_train != y_all[train_idx],
                {'noise_spec_id': spec.spec_id})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DoseWarning)
        _, res = make_noisy_train_labels(y_all, spec, train_idx, is_clf=False,
                                         groups=groups)
    for w in caught:
        if issubclass(w.category, DoseWarning):
            raise DoseError(
                f"{spec.spec_id}: {w.message}. Raised rather than warned: this "
                f"process silences warnings at import, so a scale error would "
                f"otherwise be invisible on every draw.")

    scale = np.asarray(res.noise_scale, dtype=float)
    if spec.scale_is_degenerate or scale.max() - scale.min() <= 1e-12:
        # Every molecule gets the same scale, so "which ones are corrupted" is every
        # one of them that actually moved. For the five shape-only conditions that is
        # all of them — the question of WHERE the noise is is undefined there, not
        # answered with an empty mask.
        mask_train = np.asarray(res.epsilon) != 0.0
    else:
        mask_train = scale > (scale.min() + 1e-12)
    prov = noise_provenance_row(res, spec)
    prov.update(_already_there(dataset_name, y_all[train_idx], spec))
    return np.asarray(res.y_noisy), mask_train, prov


def _already_there(dataset_name, y_clean, spec):
    """What the labels were already carrying, beside what we added.

    NOISE_DESIGN.md section 4d in ~/repos/qsar_qm_models, raised by the author
    2026-08-27: the experimental labels are not clean to begin with, and QM9's
    are. Error adds in quadrature, so injecting level L on labels already
    carrying b of their own spread leaves sqrt(b^2 + L^2), not L.

    Measured on KIRBy's own label files, 2026-09-12. logd is the LogD column of
    openadmet_train.csv, n = 5039, spread 1.1905 log units against a published
    0.48 (Niu et al. 2024), so b = 0.403. herg_ki is chembl_herg_ki.csv,
    n = 1415, spread 0.9143 pKi units against a published 0.54 (Kramer et al.
    2012), so b = 0.591. On herg_ki that makes the level-0.0 row a run at 0.591
    of the label spread, and level 0.15 a run at 0.609 — a change of 3%.

    Three fields, or none. `reference_error_as_fraction_of_spread` returns None
    wherever no assay error has been published — QM9, the reaction sets, the
    antibody sets — and there the level IS the whole of the noise, so writing a
    zero would assert something nobody measured.

    This does not change what is injected. It records what the number means.
    """
    import numpy as np
    from experiments.dataset_policy import (REFERENCE_ERROR_LABEL_UNITS,
                                            reference_error_as_fraction_of_spread)
    if dataset_name is None:
        return {}
    b = reference_error_as_fraction_of_spread(dataset_name, y_clean)
    if b is None:
        return {}
    level = float(spec.target_effective_noise)
    out = {
        'baseline_error_fraction_of_spread': float(b),
        'baseline_error_label_units': float(
            REFERENCE_ERROR_LABEL_UNITS[dataset_name]),
        'baseline_error_source': dataset_name,
    }
    if not spec.is_censoring:
        # Censoring's level is a fraction of labels clipped, not an amount, so
        # it does not add in quadrature with anything and gets no total.
        out['total_noise_fraction_of_spread'] = float(np.sqrt(b * b + level * level))
        # A model cannot explain the part of the variance that is measurement
        # error, so this is the ceiling on R-squared before anything is injected.
        out['r2_ceiling_from_baseline_error'] = float(1.0 - b * b)
    return out


# The conditions under which "which molecules are corrupted" is a real question.
# Under the other six every training molecule's label moves, so a list of the ones
# that moved is a list of all of them, precision is 1.0 by construction and the number
# says nothing. Naming them here rather than testing the mask means a detection score
# is suppressed on purpose, not by accident on a draw where every molecule happened
# to move.
#
# `censoring` is the newest member and the only one whose corrupted set is chosen by
# the label's own value: every label above a cut-point is replaced by that cut-point.
# Owner's decision 2026-09-12 added it to n09, n11 and n12.
# ADDED 2026-09-24 for n31 and n32 only, from the second research report.
# A random share of the TRAINING molecules has its label moved UP by delta label
# SDs; every other label is untouched. None of the registry's conditions is a
# one-direction shift on a subset: outlier_only_* widens the error on a subset
# (zero mean), grouped_shifted moves every molecule. Kept out of the NoiseInject
# registry on purpose, so the nine files that loop over the registry's conditions
# do not gain a job per dataset. Level = delta in label SDs.
SHIFT_ONLY_CONDITIONS = {'shift_only_p05': 0.05, 'shift_only_p10': 0.10}
SHIFT_ONLY_DELTAS_SD = (0.5, 1.0, 2.0)

# ADDED 2026-09-25 for n38-n41, from the third research report; MOVED into
# noiseInject 2026-09-25 (noiseInject/relative.py, RelativeNoiseInjector). The
# conditions and their draws live there; what stays here is KIRBy's: the unit
# `resid_sd` (oof_residual_sd below, the SD of lightgbm's out-of-fold residual on
# the clean training labels, base space, written beside every result) and the
# level grids the experiments sweep.
#
#   dense_oofmse         every training label gets N(0, level * resid_sd^2).
#   sparse_dprime_pXX    XX% of training molecules moved by level * resid_sd,
#                        sign drawn at random per molecule.
#   sparse_dprime_up_pXX the same, every one moved up.
#   hetero_seen          noise SD follows the molecule's heavy-atom count.
#   hetero_hidden        noise SD is a per-molecule lognormal draw.
#                        Both hetero conditions: mean injected variance is
#                        level * resid_sd^2, the same as dense_oofmse.
from noiseInject import (  # noqa: E402
    RELATIVE_CONDITIONS, RELATIVE_SPARSE_SHARES, RelativeNoiseInjector,
)
RELATIVE_DENSE_LEVELS = (0.1, 0.3, 1.0)     # share of OOF MSE
RELATIVE_DPRIME_LEVELS = (1.0, 2.0, 4.0)    # residual SDs


def relative_levels(condition):
    """The level grid of a RELATIVE_CONDITIONS member, 0 included."""
    if condition not in RELATIVE_CONDITIONS:
        raise ValueError(f'relative_levels: unknown condition {condition!r}')
    grid = (RELATIVE_DPRIME_LEVELS if condition.startswith('sparse_dprime')
            else RELATIVE_DENSE_LEVELS)
    return (0.0,) + tuple(grid)


DETECTION_IS_DEFINED = ('outlier_p01', 'outlier_p05', 'outlier_p10',
                        'outlier_only_p01', 'outlier_only_p05',
                        'outlier_only_p10',
                        'grouped_wider', 'censoring',
                        'shift_only_p05', 'shift_only_p10') \
    + tuple(c for c in RELATIVE_CONDITIONS if c.startswith('sparse_dprime'))

# ADDED 2026-09-23. The three `outlier_only_*` conditions leave the unselected
# molecules' labels bit-identical, where the three `outlier_*` conditions widen
# the selected fraction and still move every other label. Detection is defined
# under both -- some molecules are in the suspect set and some are not -- but the
# ORACLE is not the same thing under the two. Under outlier_p10 the widened tenth
# carries half the injected squared error, so removing exactly those molecules
# leaves the other half behind; under outlier_only_p10 it carries all of it.
# Measured 2026-09-23 on 2,000 Gaussian labels: 47.8% against 100.0%.

# Suspect sets that nothing here injected, so they cannot be reached by
# --noise-condition and are deliberately NOT in the tuple above: n21 offers
# that tuple as its condition choices and refuses anything outside it, and a
# name no injector can produce does not belong in a list of injectable names.
# What they share with the five is the only property detection grading needs —
# SOME molecules are in the suspect set and some are not, so precision is not
# 1.0 by construction.
#
#   curator_flag   ChEMBL's own `data_validity_comment`, written by its
#                  curators. n22 grades against it. 301 of the 12,008 hERG
#                  IC50 records carry one, counted 2026-09-17 under
#                  ChEMBL_37.
DETECTION_DEFINED_WITHOUT_INJECTION = ('curator_flag',)


def inject_shift_train_labels(y_all, *, train_idx, condition, delta_sd, draw,
                              base_seed=42):
    """Move a share of the labels up by delta_sd label SDs; return one fold's rows.

    y_all      the FULL clean label column
    train_idx  this fold's training rows
    condition  a key of SHIFT_ONLY_CONDITIONS; it sets the share moved
    delta_sd   the shift, in SDs of this fold's TRAINING labels (the unit
               inject_train_labels scales every registry condition by)
    draw       the draw index; the seed is noise_draw_seed(condition, draw)

    WHICH MOLECULES MOVE IS DRAWN OVER THE WHOLE COLUMN and the fold's training
    rows are then taken, as inject_training_rows does for the registry
    conditions. So a molecule is moved or not whatever fold it trains in, and
    the share moved within a fold is the condition's share up to sampling.
    Until 2026-09-30 the draw was over the training slice with a seed each
    caller built itself (n31 crc32 with the fold, n32/n41 draw*1000+fold), and
    the shift was in SDs of the whole column.

    Returns (y_noisy_train, moved_mask_train, provenance), the same shape as
    inject_train_labels, so detection_scores grades it the same way.
    """
    import numpy as np
    if condition not in SHIFT_ONLY_CONDITIONS:
        raise ValueError(f'inject_shift_train_labels: unknown condition {condition!r}')
    y_all = np.asarray(y_all, dtype=float).ravel()
    train_idx = np.asarray(train_idx)
    share = SHIFT_ONLY_CONDITIONS[condition]
    n_all = len(y_all)
    seed = noise_draw_seed(condition, draw, base_seed)
    moved_all = np.zeros(n_all, dtype=bool)
    n_move = int(round(share * n_all))
    if n_move > 0 and float(delta_sd) != 0.0:
        rng = np.random.RandomState(seed ^ 0x5CA1E)
        moved_all[rng.choice(n_all, size=n_move, replace=False)] = True
    y_train = y_all[train_idx].copy()
    label_sd = float(np.std(y_train))
    shift = float(delta_sd) * label_sd
    moved = moved_all[train_idx]
    y_train[moved] += shift
    return y_train, moved, {
        'condition': condition, 'share': share, 'delta_sd': float(delta_sd),
        'shift_label_units': shift, 'label_sd_training': label_sd,
        'n_moved': int(moved.sum()), 'n_training': int(len(train_idx)),
        'seed': int(seed), 'draw': int(draw),
        'drawn_over': 'whole column, then training rows',
    }


def heavy_atom_counts(smiles):
    """Heavy-atom count per SMILES; NaN where RDKit cannot parse it."""
    import numpy as np
    from rdkit import Chem
    out = np.full(len(smiles), np.nan)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s)) if s is not None else None
        if m is not None:
            out[i] = m.GetNumHeavyAtoms()
    return out


def oof_residual_sd(X, y, groups=None, n_splits=5, seed=0):
    """SD of lightgbm's out-of-fold residual on (X, y): the unit of RELATIVE_CONDITIONS.

    Scaffold groups are respected when given, as everywhere else in the suite.
    Returns (resid_sd, oof_prediction).
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold, KFold
    from kirby.find_best_hybrid import get_model
    y = np.asarray(y, dtype=float).ravel()
    n_splits = max(2, min(n_splits, len(y)))
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splits = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splits = KFold(n_splits=n_splits, shuffle=True,
                       random_state=seed).split(X, y)
    p = np.zeros_like(y)
    for tr, va in splits:
        m = get_model('lightgbm', 'regression')
        m.fit(X[tr], y[tr])
        p[va] = m.predict(X[va])
    return float(np.std(y - p)), p


def inject_relative_train_labels(y_all, *, train_idx, condition, level, draw,
                                 resid_sd, heavy_atoms=None, base_seed=42,
                                 y_base_train=None):
    """One fold's training labels through noiseInject's RelativeNoiseInjector.

    Shared by n32, n38, n40, n41 and n42. `level` is in units of `resid_sd`,
    this fold's out-of-fold residual SD. The seed is
    noise_draw_seed(condition, draw).

    THE NOISE IS DRAWN OVER THE WHOLE COLUMN and the fold's training rows are
    then taken, as inject_training_rows does for the registry conditions, so a
    molecule's noise does not depend on its fold. noiseInject's noise does not
    read the label values (relative.py: epsilon from rng, unit, heavy_atoms),
    so drawing it over the column and adding the training rows' share to the
    fold's labels is the same as injecting into those labels.
    Until 2026-09-30 the draw was over the training slice and each caller built
    its own seed with the fold inside it (n32/n41 draw*1000+fold, n38
    crc32(...fold), n40 crc32(dataset|...fold)).

    heavy_atoms   one per molecule of the WHOLE column (hetero_seen only)
    y_base_train  optional replacement for the fold's clean training labels
                  (n38's semi-synthetic labels); the noise is added to these.

    Returns (y_noisy_train, moved_mask_train, provenance).
    """
    import numpy as np
    y_all = np.asarray(y_all, dtype=float).ravel()
    train_idx = np.asarray(train_idx)
    seed = noise_draw_seed(condition, draw, base_seed)
    if heavy_atoms is not None and len(np.asarray(heavy_atoms).ravel()) != len(y_all):
        raise ValueError('inject_relative_train_labels: heavy_atoms must have one '
                         'entry per molecule of the whole column, got '
                         f'{len(np.asarray(heavy_atoms).ravel())} for {len(y_all)}')
    res = RelativeNoiseInjector.from_condition(
        condition, random_state=int(seed)).inject_verbose(
            y_all, level=level, unit=resid_sd, heavy_atoms=heavy_atoms)
    eps_tr = np.asarray(res.epsilon)[train_idx]
    base = (np.asarray(y_base_train, dtype=float).ravel()
            if y_base_train is not None else y_all[train_idx])
    y_noisy = base + eps_tr
    moved = np.asarray(res.moved_mask)[train_idx]
    prov = res.as_row()
    prov.update({'resid_sd': float(resid_sd), 'seed': int(seed), 'draw': int(draw),
                 'drawn_over': 'whole column, then training rows',
                 'n_moved_training': int(moved.sum())})
    return y_noisy, moved, prov


def cleanlab_suspicion(X, y, model_name='lightgbm', seed=0, cv_n_folds=5):
    """cleanlab's regression label-quality score, turned into a suspicion score.

    cleanlab.regression.learn.CleanLearning (Zhou, Mueller et al., arXiv:2305.16583)
    scores each label from an out-of-fold residual scaled by estimated aleatoric
    and epistemic uncertainty; its own bootstrap refits give the epistemic part.
    Its internal folds are random, not scaffold-grouped: it is a detector run on
    one training fold, and the held-out fold never reaches it.

    Returns 1 - label_quality per molecule, so higher is more suspicious.
    """
    import numpy as np
    from cleanlab.regression.learn import CleanLearning
    from kirby.find_best_hybrid import get_model
    cl = CleanLearning(get_model(model_name, 'regression'),
                       cv_n_folds=cv_n_folds, seed=int(seed), verbose=False)
    issues = cl.find_label_issues(np.asarray(X), np.asarray(y, dtype=float).ravel())
    return 1.0 - issues['label_quality'].to_numpy(dtype=float)


def gamma_test(D, y, k_max=10):
    """Gamma test (Stefansson, Koncar & Jones 1997): noise variance from neighbours.

    D      square distance matrix between the molecules (diagonal ignored)
    y      their labels
    For each k = 1..k_max, delta_k = mean over molecules of the mean SQUARED
    distance to their k nearest neighbours, gamma_k = mean of half the squared
    label difference to the same neighbours. A straight line gamma = A + B*delta
    through the k_max points; the intercept A estimates the label-noise variance.
    It also holds variation the features cannot resolve, so it is an upper bound.

    Differs from n35's close_pairs estimate: per-molecule neighbours rather than
    the closest share of all pairs, squared distance, and k_max points.
    """
    import numpy as np
    D = np.asarray(D, dtype=float).copy()
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    k_max = int(min(k_max, n - 1))
    if k_max < 2:
        return None
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k_max]
    d_sorted = np.take_along_axis(D, idx, axis=1)
    dy2 = 0.5 * (y[idx] - y[:, None]) ** 2
    delta = np.array([np.mean(d_sorted[:, k] ** 2) for k in range(k_max)])
    gamma = np.array([np.mean(dy2[:, k]) for k in range(k_max)])
    # the k-th neighbour's statistics, per Jones' formulation (not the running mean)
    B, A = np.polyfit(delta, gamma, 1)
    var_y = float(np.var(y))
    return {'intercept': float(A), 'slope': float(B), 'k_max': k_max,
            'delta': delta.tolist(), 'gamma': gamma.tolist(),
            'v_ratio': float(A / var_y) if var_y > 0 else None,
            'var_y': var_y, 'n': int(n)}


def detection_is_defined(condition) -> bool:
    """Whether a detection score means anything under this condition."""
    # A flip kind on a classification set flips a share of the labels and
    # leaves the rest, so a corrupted subset exists (added 2026-10-05, chat 5;
    # n21, n23 and n24 had each worked around its absence).
    from kirby.noise_spec import CLASSIFICATION_STRATEGIES
    return (str(condition) in DETECTION_IS_DEFINED
            or str(condition) in DETECTION_DEFINED_WITHOUT_INJECTION
            or str(condition) in CLASSIFICATION_STRATEGIES)


def detection_scores(corrupted_mask, suspicion, condition, *, higher_is_suspicious=True):
    """Grade a per-molecule suspicion score against the molecules that really moved.

    Until 2026-09-12 every detector in the noise directory was graded only by the
    score AFTER repair. Nothing ever checked whether it found the right molecules.
    `inject_train_labels` already returns the list of molecules whose label moved --
    its own docstring calls that list the ground truth for exactly this -- and all
    seven lines that received it threw it away.

    corrupted_mask   one boolean per training molecule: did this label move.
                     Under `curator_flag` nothing moved and this is instead
                     whether ChEMBL's curators marked the record suspect; the
                     three numbers below are computed identically either way,
                     which is the point of grading both through here.
    suspicion        one float per training molecule: how wrong this label looks
    condition        the noise condition name, checked against
                     DETECTION_IS_DEFINED and
                     DETECTION_DEFINED_WITHOUT_INJECTION

    Returns None when the condition is one where every molecule moves, or when the
    two arrays disagree on length, or when the draw produced no corrupted molecule
    and no clean one. Returning None rather than a number is deliberate: a suppressed
    figure is readable, a meaningless 1.0 is not.

    Three numbers, and what each counts:
      detection_average_precision  area under the precision-recall curve, over all
                                   cut-offs. The headline: it needs no threshold, so
                                   it does not inherit the cut-off defect of section
                                   6.9, where a filter's ruler grows with the noise.
      detection_precision_at_k     of the k molecules the detector ranks most
                                   suspicious, the share that really moved. k is the
                                   number that really moved, so a perfect detector
                                   scores 1.0 and a coin flip scores the corrupted
                                   share of the training set.
      detection_recall_at_k        of the molecules that really moved, the share in
                                   that same top k. At this k it equals the precision;
                                   both are written because a reader who sees one
                                   number alone cannot tell which one it is.
      detection_auroc              the chance a randomly chosen moved molecule
                                   scores above a randomly chosen untouched one.
                                   Added 2026-09-24 so the suite's numbers can sit
                                   beside the predicted-ceiling formulas in n31,
                                   which are written as AUROC. 0.5 is a coin flip
                                   whatever the corrupted share.
    Plus the two counts they are fractions of, so no bare ratio is ever written:
      detection_n_corrupted        molecules whose label moved
      detection_n_training         molecules in the training fold
    """
    import numpy as np
    if not detection_is_defined(condition):
        return None
    mask = np.asarray(corrupted_mask).ravel().astype(bool)
    sus = np.asarray(suspicion, dtype=float).ravel()
    if mask.size == 0 or sus.size != mask.size:
        return None
    n_corrupt = int(mask.sum())
    if n_corrupt == 0 or n_corrupt == mask.size:
        return None
    if not np.all(np.isfinite(sus)):
        sus = np.where(np.isfinite(sus), sus, np.nanmin(sus[np.isfinite(sus)]) - 1.0
                       if np.any(np.isfinite(sus)) else 0.0)
    if not higher_is_suspicious:
        sus = -sus
    try:
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(mask, sus))
    except Exception:
        ap = float('nan')
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(mask, sus))
    except Exception:
        auroc = float('nan')
    order = np.argsort(-sus, kind='stable')
    top = order[:n_corrupt]
    hits = int(mask[top].sum())
    return {
        'detection_average_precision': ap,
        'detection_precision_at_k': hits / float(n_corrupt),
        'detection_recall_at_k': hits / float(n_corrupt),
        'detection_auroc': auroc,
        'detection_n_corrupted': n_corrupt,
        'detection_n_training': int(mask.size),
        'detection_condition': str(condition),
    }


def tail_scores(y_clean_test, pred, share=0.1):
    """Error on the extremes of the clean held-out labels, and calibration slope.

    Added 2026-09-24 for n31-n36. Cleaning that rewrites labels toward a model's
    prediction pulls extreme labels inward, and the overall rmse can hide that.

    y_clean_test  the CLEAN held-out labels
    pred          the model's predictions for the same molecules
    share         the fraction of held-out molecules counted as each tail

    Returns:
      tail_rmse_top           rmse over the held-out molecules whose clean label
                              is in the top `share`
      tail_rmse_bottom        the same for the bottom `share`
      tail_n_each             molecules in each tail
      calibration_slope       slope of clean label regressed on prediction. 1 is
                              calibrated; below 1 the predictions are spread too
                              wide, above 1 too narrow
    None when fewer than 20 held-out molecules, since a tenth is then 1 molecule.
    """
    import numpy as np
    y = np.asarray(y_clean_test, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if y.size != p.size or y.size < 20:
        return None
    k = max(1, int(round(share * y.size)))
    order = np.argsort(y, kind='stable')
    lo, hi = order[:k], order[-k:]
    var_p = float(np.var(p))
    slope = (float(np.cov(p, y, bias=True)[0, 1]) / var_p) if var_p > 0 else float('nan')
    return {
        'tail_rmse_top': float(np.sqrt(np.mean((y[hi] - p[hi]) ** 2))),
        'tail_rmse_bottom': float(np.sqrt(np.mean((y[lo] - p[lo]) ** 2))),
        'tail_n_each': int(k),
        'calibration_slope': slope,
    }


def hybrid_for_fold(train_reps, val_reps, y_train, bp, groups_train,
                    model, metric_fn, lower_is_better,
                    augmentations_train=None, augmentations_val=None,
                    want_feature_info=False):
    """The hybrid feature matrices for one fold, or (None, None) if it could not build.

    Written for n19 on 2026-09-16 and moved here on 2026-09-17, when n17 and n18
    needed the same thing and it would otherwise have been a third copy. n19
    imports it from here; nothing else changed about what it does.

    The two augmentation arguments were added on 2026-09-17 for n21, which is
    the first caller whose recipe may carry an augmentation strategy. They
    default to None, which is what every earlier caller passed implicitly, so
    n17, n18, n19 and n20 build exactly the features they built before.

    The same shape as n12's `_build_fold_shared`: the representation pool is cut
    down to the ones the recipe actually scored, so the per-fold allocation
    cannot rank in representations the search never considered. Under a recipe
    from the similarity family no downstream model is fitted inside the build,
    so the model, the metric and its direction are not passed -- build_fold_hybrid
    refuses the pair.

    Args:
        train_reps: {name: matrix} for this fold's TRAINING rows only.
        val_reps: {name: matrix} for the rows this fold is graded on.
        y_train: one label per training row. Whatever the caller passes here is
            what ranks the features and splits the budget, so a caller that
            wants the features chosen without a held-out label must pass the
            training labels alone.
        groups_train: the fold grouping, already sliced to the training rows.
        model: the downstream estimator INSTANCE the accept/reject test is
            scored with, or None under a geometry recipe.
        metric_fn, lower_is_better: that test's metric and its direction.
        augmentations_train, augmentations_val: {name: matrix} donor blocks,
            already sliced to this fold's rows, or None. A recipe that names an
            augmentation strategy needs them or it rebuilds a narrower feature
            space than the one the search chose.
        want_feature_info: return a third value, WHICH FEATURES the build
            selected. `build_fold_hybrid` has always returned it third and
            every caller here dropped it on the floor, so no experiment could
            say which features a recipe chose -- only how wide it was. A
            Blueprint does not carry them: it carries the representations, the
            allocation method, the importance method, the budget and the
            evaluator, and the features exist only once the hybrid is built.
            The value is {rep_name: {'selected_indices': array of column ids
            LOCAL to that representation's own matrix, 'importance_scores':
            array, 'n_features': int}} plus an 'allocation' entry. Because the
            indices are local (hybrid.py:2236), a feature's identity is the
            PAIR (representation, index) -- two representations both selecting
            their column 7 are two different features, not one.
            False (the default) returns the two matrices alone, which is what
            n17, n18, n19 and n20's own fold loop pass.
    """
    from experiments._shared import build_fold_hybrid
    from kirby.find_best_hybrid import _is_geometry_objective

    bp_reps = getattr(bp, 'reps', None)
    if bp_reps:
        tr = {r: train_reps[r] for r in bp_reps if r in train_reps}
        if not tr:
            tr = train_reps
    else:
        tr = train_reps
    va = {r: val_reps[r] for r in tr if r in val_reps} or tr

    objective = getattr(bp, 'objective', None)
    geometry = objective is not None and _is_geometry_objective(objective)
    try:
        X_tr_h, X_va_h, feat_info = build_fold_hybrid(
            tr, va, y_train, bp,
            augmentations_train=augmentations_train,
            augmentations_val=augmentations_val,
            groups=groups_train,
            **({} if geometry else {'model': model, 'metric_fn': metric_fn,
                                    'lower_is_better': lower_is_better}),
        )
        if want_feature_info:
            return X_tr_h, X_va_h, feat_info
        return X_tr_h, X_va_h
    except Exception as exc:
        print(f"      hybrid build failed: {type(exc).__name__}: {exc}")
        if want_feature_info:
            return None, None, None
        return None, None


# ⛔ THE ORDER THE CONDITIONS RUN IN, AND IT IS NOT THE REGISTRY'S.
#
# A sweep of fourteen conditions is most of a day per dataset and the jobs
# carrying it have a wall clock. Each condition is written as it finishes, so a
# job that runs out of time keeps what it got -- which makes the ORDER decide
# what you end up holding. In registry order the first five are gaussian and
# the three student-t shapes and laplace: all zero-mean, all constant-width,
# and the case the literature review says cleaning cannot help in. A job that
# reached only those would have measured nothing that could have come out
# differently.
#
# So the ones that can show a difference go first:
#
#   grouped_shifted   a constant offset per scaffold family. The one condition
#                     here that is NOT zero-mean, and the review's "systematic
#                     per-source bias" -- the case it says relabeling helps
#                     most. Never run.
#   outlier_only_p10  the widened tenth is the ONLY thing that moved, so
#                     removing exactly those molecules can recover all of the
#                     injected error.
#   outlier_p10       the same p and lam with every other label ALSO moved, so
#                     the same removal recovers half. The pair is the
#                     comparison: it separates "the detector cannot find them"
#                     from "finding them does not help".
#   gaussian          the control. Zero-mean and constant-width, where the
#                     review predicts no gain.
#   student_t_nu3     the heaviest tail, where a robust loss should matter most.
#   censoring         every label above a cut-point replaced by it. The only
#                     condition whose corrupted molecules are chosen by the
#                     label's own value.
#
# The rest follow in registry order. Nothing is cut -- this is an ORDER, not a
# selection, and a job with time for all fourteen runs all fourteen.
SWEEP_ORDER = ('grouped_shifted', 'outlier_only_p10', 'outlier_p10',
               'gaussian', 'student_t_nu3', 'censoring',
               'outlier_only_p05', 'outlier_p05', 'grouped_wider',
               'laplace', 'outlier_only_p01', 'outlier_p01',
               'student_t_nu5', 'student_t_nu10')


def conditions_for_task(data, detection_is_the_headline=False, promoted=None):
    """Every condition this dataset can be given, for an experiment to sweep.

    A classification set takes the flip models and a regression set the shapes:
    a flipped class label has no magnitude, so the two families are not
    interchangeable and noise_spec_for substitutes one for the other silently
    if handed the wrong kind.

    `detection_is_the_headline` narrows it to the conditions where a corrupted
    SUBSET exists. Use it only where the number the experiment REPORTS is a
    detection score -- under a constant-width condition every training label
    moves, the suspect set is every molecule and precision is 1.0 by
    construction. An experiment whose headline is a downstream score and which
    merely also records detection does NOT need this: detection_scores already
    returns None where it is undefined.
    """
    from kirby.noise_spec import CLASSIFICATION_STRATEGIES
    if data.get('task_type') == 'classification':
        return list(CLASSIFICATION_STRATEGIES)
    # The six decided kinds (DECIDED_REGRESSION_CONDITIONS, f37, Adelaide
    # 2026-10-05). Three of them are outside the NoiseInject registry; inject
    # every one through inject_condition. SWEEP_ORDER above is the old
    # fourteen, kept for the record.
    ordered = list(DECIDED_REGRESSION_CONDITIONS)
    # The screen leaves out SCREEN_DROPPED_CONDITIONS; a promoted run (the
    # argument when the caller knows, else KIRBY_NOISE_PROMOTE) sweeps all six.
    if not (promoted if promoted is not None else promoted_run()):
        ordered = [c for c in ordered if c not in SCREEN_DROPPED_CONDITIONS]
    if detection_is_the_headline:
        return [c for c in ordered if c in DETECTION_IS_DEFINED]
    return ordered


def sweep_conditions(run_one, exp_id, data, config, output_dir,
                     dataset_name, default_conditions, kwargs,
                     groups_needed=None):
    """Run one experiment across several injected conditions, checkpointing each.

    THE CONDITION IS AN AXIS AN EXPERIMENT SWEEPS, not something a caller runs
    it once per. This is that sweep, in one place, because seven experiments
    need it and three near-identical copies of a loop is how n17, n18 and n23
    ended up each carrying the same per-model build defect.

    WHY EACH CONDITION IS WRITTEN AS IT LANDS. One condition on one dataset
    measured between 1,991 and 7,095 seconds locally at 3 folds on rf alone, so
    a sweep is most of a day and it WILL be interrupted -- by a wall clock, by a
    kill, by a laptop closing. Writing once at the end loses all of it. Each
    condition goes to its own file through save_experiment_results'
    `filename_suffix`, which exists for exactly this, and a condition whose file
    is already there is skipped. An interrupted run loses the condition in
    flight and nothing else.

    Args:
        run_one: the experiment's own single-condition entry point, called as
            run_one(data, config, output_dir, dataset_name,
                    noise_condition=<name>, **kwargs).
        exp_id: the result directory name, e.g. 'n23_uncertainty_cleaning'.
        default_conditions: what to sweep when the caller names none.
        kwargs: the experiment's kwargs. `noise_conditions` (a list),
            `noise_condition` (one name, becoming a one-element list),
            `sweep_slice` ('K/N') and `force_rerun` are consumed here.
        groups_needed: conditions that need scaffold groups. They are SKIPPED
            with their reason when the dataset has none, never attempted: one
            impossible condition must not end the other thirteen. The drop
            happens BEFORE the slice so array tasks stay balanced.

    Returns an index of what was written, not the scores -- those are in the
    per-condition files.
    """
    from experiments._shared import save_experiment_results

    conditions = kwargs.pop('noise_conditions', None)
    if conditions is None:
        one = kwargs.get('noise_condition')
        conditions = [one] if one else list(default_conditions)
    conditions = [str(c) for c in conditions]
    kwargs.pop('noise_condition', None)

    skipped = {}
    if groups_needed:
        # Imported here, not at module scope: dataset_policy imports from this
        # module, so a top-level import would be circular.
        from experiments.dataset_policy import noise_groups_for
        noise_groups, _, _ = noise_groups_for(data)
        if noise_groups is None:
            for c in [c for c in conditions if c in groups_needed]:
                skipped[c] = (
                    f'{dataset_name} loads no SMILES, so there are no scaffold '
                    f'groups for this condition to act on. Every fold would '
                    f'raise inside the injector and the file would carry no '
                    f'score.')
            conditions = [c for c in conditions if c not in skipped]

    slice_spec = kwargs.pop('sweep_slice', None)
    if slice_spec:
        k, n = (int(x) for x in str(slice_spec).split('/'))
        if not (0 <= k < n):
            raise ValueError(f'sweep_slice {slice_spec!r}: need 0 <= K < N')
        # By STRIDE, not by block, so a part-finished array covers a spread of
        # conditions rather than the first few. Same rule as the OPIG sweep.
        conditions = conditions[k::n]

    if output_dir is None:
        raise ValueError(
            f"{exp_id}: sweeping conditions needs an output directory, because "
            f"each condition is WRITTEN as it finishes and a condition already "
            f"on disk is skipped -- that is what makes an interrupted sweep "
            f"resumable. A caller with nowhere to write wants one condition, "
            f"not a sweep: call the experiment's _run_one_condition directly.")

    # force_rerun reaches here from run_dataset.py --force-rerun through
    # run_all, which hands it to every run() that forwards its kwargs to this
    # function (noise_helpers._takes_force_rerun). Before 2026-09-25 run_all
    # never passed it, so a condition file on disk was skipped even under
    # --force-rerun.
    force = bool(kwargs.pop('force_rerun', False))
    # What each condition file is stamped with and compared against on
    # resume: the same input_fingerprint run_all stamps its files with and
    # compares (run_all, 'Resume' block). Before 2026-09-25 a condition file
    # was kept whenever it existed and was non-empty, so a change of fold
    # grouping, labels or representations reused numbers computed from the
    # old data. A file written before this stamp has no fingerprint and is
    # recomputed once, as run_all does with its own files.
    from experiments._shared import (
        input_fingerprint, fingerprint_mismatch, experiment_result_is_empty)
    fingerprint = input_fingerprint(data, config)
    done, ran = {}, []
    for cond in conditions:
        dest = Path(output_dir) / exp_id / f'{dataset_name}__{cond}.json'
        if dest.exists() and dest.stat().st_size > 0 and not force:
            try:
                with open(dest) as fh:
                    prev = json.load(fh)
                # A caller whose data carries no labels gets fingerprint {}
                # (input_fingerprint): nothing to compare, so the file stands.
                why = ('records no measurement'
                       if experiment_result_is_empty(prev,
                                                     classify_unmarked=True)
                       else None if not fingerprint
                       else fingerprint_mismatch(prev.get('input_fingerprint'),
                                                 fingerprint))
            except (ValueError, OSError) as exc:
                why = f'could not be read ({exc})'
            if not why:
                print(f"  {cond}: already on disk, skipping ({dest})")
                done[cond] = str(dest)
                continue
            print(f"  {cond}: RERUN, existing file {why} ({dest})")
        print(f"\n{'=' * 70}\n{exp_id.upper()} CONDITION {cond}\n{'=' * 70}")
        t0 = time.time()
        res = run_one(data, config, output_dir, dataset_name,
                      noise_condition=cond, **kwargs)
        save_experiment_results(res, output_dir, exp_id, dataset_name,
                                fingerprint=fingerprint,
                                filename_suffix=cond)
        print(f"  {cond} done in {time.time() - t0:.0f}s -> {dest}")
        done[cond] = str(dest)
        ran.append(cond)

    for cond, why in skipped.items():
        print(f"  {cond}: not run -- {why}")

    # ⛔ THE COUNTS ARE NOT DECORATION. This index is what run_all saves at
    # <dir>/<dataset>.json -- the exact path verify_run_status.py:179 and
    # audit_done_dataset_gaps.py:144 check to decide whether an experiment ran.
    # save_experiment_results stamps result_status from whether the payload
    # holds a computed quantity, and a bare list of filenames holds none, so a
    # sweep of fourteen finished conditions was being filed as
    # result_status='empty', has_computed_results=False. A completed run read
    # as a failed one. These counts are the computed quantity.
    return {'swept_conditions': conditions,
            'conditions_written': done,
            'conditions_run_this_time': ran,
            'conditions_skipped_and_why': skipped,
            'sweep_slice': slice_spec,
            'n_conditions_swept': len(conditions),
            'n_conditions_on_disk': len(done),
            'n_conditions_run_this_time': len(ran),
            'n_conditions_skipped': len(skipped),
            'what_a_condition_file_holds': (
                'one condition of this experiment, at '
                '<dir>/<dataset>__<condition>.json. This file is the INDEX of '
                'that sweep, not a result: the scores are in the per-condition '
                'files it names.')}


def group_models_by_recipe(recipe_by_model, models):
    """Which downstream models can share one built feature matrix, and which cannot.

    WHY IT EXISTS. The recipe is looked up per downstream model, so five models
    means five builds per fold per space. Under a recipe from the similarity
    family that is waste: no downstream model is fitted inside the build and the
    model is not passed to it, so two models holding the SAME such recipe get
    bit-identical matrices.

    IT DOES NOT ASSUME THEY HOLD THE SAME ONE. get_best_search_for_downstream
    skips a stored search the model is absent from and then takes
    get_best_for_downstream for it (_shared.py:6133-6138), so two models can come
    back with different similarity recipes. Recipes are compared, never assumed
    equal, and _same_recipe returns True only on a plain True -- a comparison
    that raises, or returns an array, counts as different. So the worst case is
    one build per model, which is what the callers did before this existed.

    Under a predictive recipe the model IS fitted inside the build, so those are
    never grouped however equal the recipes look.

    Args:
        recipe_by_model: {model: recipe} -- a model absent from it, or mapped to
            None, gets no build here rather than another model's.
        models: the downstream models to consider, in the order to build them.

    Returns:
        [(recipe, [model, ...], shareable)]. `shareable` is True when the build
        fits no downstream model, which is when the caller passes None in place
        of a model instance and when the models in the list share the build.
    """
    groups = []
    for name in models:
        bp = (recipe_by_model or {}).get(name)
        if bp is None:
            continue
        shareable = recipe_is_geometry(bp)
        placed = False
        if shareable:
            for g in groups:
                if not g[2]:
                    continue
                if g[0] is bp or _same_recipe(g[0], bp):
                    g[1].append(name)
                    placed = True
                    break
        if not placed:
            groups.append((bp, [name], shareable))
    return groups


def recipe_is_geometry(bp):
    """Whether this recipe chose its features by the geometry they define.

    Under one of these no downstream model is fitted inside the build, so the
    feature matrix it produces depends on the recipe alone -- which is what lets
    two downstream models holding the same one share a single build. Anything
    it cannot read off the blueprint is False, so an unreadable recipe is built
    per model exactly as before.
    """
    try:
        from kirby.find_best_hybrid import _is_geometry_objective
    except ImportError:
        return False
    obj = getattr(bp, 'objective', None)
    return obj is not None and bool(_is_geometry_objective(obj))


def _same_recipe(a, b):
    """Whether two stored recipes are the same recipe, conservatively.

    Used to decide whether one built feature matrix can serve two downstream
    models. Getting this WRONG in the permissive direction would hand a model
    another model's features, which is the defect the per-model lookup exists
    to remove -- so anything but a plain True means "treat them as different"
    and build twice. Identity is checked by the caller before this is reached.

    A blueprint's __eq__ may be identity-based, may be a numpy-style elementwise
    comparison that has no truth value, or may raise. All three land on False.
    """
    if a is b:
        return True
    try:
        eq = (a == b)
    except Exception:
        return False
    return eq is True


def pool_recipe_per_downstream_model(lookup, models):
    """The stored search and its recipe for EACH downstream model, not for one.

    ⛔ WHOLE-DATASET, SUPERSEDED 2026-10-05. This reads the stored search over
    every row (rule R2 of HANDOFF_EXPERIMENT_ALIGNMENT.md) and get_rule_hybrids'
    set of 2 general + 1 distance hybrids beside 2 base representations, which
    is 10 fitted configurations for two models where R5 wants 8. Its per-fold
    replacement is _shared.fold_neighbour_space_recipes. Kept unchanged until
    each noise experiment is moved to it (chat 5), so no caller breaks first.

    WHY IT EXISTS. n17, n18 and n23 each fetched ONE recipe, asked for
    downstream_models[0] -- 'rf' in every run this suite has made -- and then
    scored every downstream model on the features that one recipe chose. A run
    over the five models in _shared.DOWNSTREAM_MODELS -- rf, xgboost, svm, dnn,
    lightgbm -- wrote five numbers measured in the random forest's feature
    space, so a difference between two of them was a difference between two
    estimators fitted on one estimator's features. Changed 2026-09-23 in the
    three files at once.

    Args:
        lookup: the pool reader with everything but the model already bound,
            called once per model as lookup(downstream_model=<name>). In the
            three callers it is functools.partial over
            _shared.get_best_search_for_downstream, so the representations, the
            labels, the fold grouping, the dataset name and the pool subset are
            fixed and the model is the only thing that moves.
        models: the downstream model names, in the order the caller listed them.

    WHAT HAPPENS TO A MODEL THE POOL HOLDS NO SEARCH FOR. Read out of
    _shared.get_best_search_for_downstream in this session: it returns the pair
    (None, None) -- after printing a line that starts '[pool] WARNING: no
    usable hybrid for model=' -- when no member of the pool scored that model
    at that budget, and it REFUSES the cross-model fallback that
    get_best_for_downstream would otherwise take (the best configuration from
    any model, carrying that other model's score). It RAISES instead when a
    stored search cannot be opened at all (result_cache.StaleResultDB, which
    _iter_pool_searches re-raises on purpose) or when the call is malformed.
    Both land in the second return value here, saying which of the two
    happened, and such a model gets no hybrid space from the caller: it is
    never handed another model's recipe, which is the defect this function
    exists to remove.

    WHAT IT COSTS. One lookup per model where there was one lookup per run. The
    searches are not re-run: the model asked for is in no cache key -- the
    search pickle is named by dataset, tier and augmentation strategy
    (find_best_hybrid._search_cache_path), get_precomputed_search passes no
    `models` so the model list in the stored search's identity is the same in
    every one of these calls, and the in-memory
    data['precomputed_searches'] dict is keyed by augmentation condition alone
    and is written back into by _iter_pool_searches -- so every call after the
    first reads the same stored SearchResult and picks a different row out of
    it with get_best_for_downstream. What the per-fold loop
    pays is separate and belongs to the caller: one hybrid BUILD per (fold,
    space, model) where there was one per (fold, space).

    Returns:
        (by_model, no_recipe).

        by_model: {model: {'search': SearchResult or None, 'meta': the pool
        metadata dict, 'blueprint': the recipe}} for every model the pool
        answered for, in the order `models` names them.

        no_recipe: {model: {'why': one sentence, 'lookup_error': 'Type:
        message' or None}} for every model it did not.
    """
    by_model, no_recipe = {}, {}
    # THE OWNER'S SET, 2026-10-02: top 2 general + top 1 distance hybrids per
    # model and family (_shared.get_rule_hybrids). The lookup below still finds
    # the stored search; the rule set is read beside it from the same bound
    # arguments and handed to pool_recipes_by_family through meta, so every
    # caller of this function builds all of the rule's spaces.
    import functools as _ft
    from experiments._shared import get_rule_hybrids as _rule
    _rule_args, _rule_kw = None, None
    # Only for the real pool reader. A smoke test that swaps in a fake lookup
    # must not trigger real searches through the rule read.
    from experiments._shared import (
        get_best_search_for_downstream as _real_lookup)
    if isinstance(lookup, _ft.partial) and lookup.func is _real_lookup:
        _rule_args = lookup.args
        _rule_kw = {k: v for k, v in (lookup.keywords or {}).items()
                    if k not in ('distance', 'return_fitted', 'downstream_model')}
    for name in models:
        _rule_set = None
        if _rule_args is not None:
            try:
                _rule_set = _rule(*_rule_args, downstream_model=name, **_rule_kw)
            except Exception as exc:
                from kirby.result_cache import StaleResultDB as _Stale
                if isinstance(exc, _Stale):
                    raise
                print(f"  rule hybrid read failed for {name}: "
                      f"{type(exc).__name__}: {exc}")
        try:
            search, meta = lookup(downstream_model=name)
        except Exception as exc:
            err = f'{type(exc).__name__}: {exc}'
            no_recipe[name] = {
                'why': (f'the pool lookup for this model RAISED {err}, so the '
                        f'pool was never read for it and this is not a '
                        f'statement about what it holds'),
                'lookup_error': err}
            print(f"  hybrid recipe lookup failed for {name}: {err}")
            continue
        if _rule_set is not None and _rule_set.get('hybrids'):
            meta = dict(meta or {})
            meta['rule_hybrids'] = _rule_set
        bp = ((meta or {}).get('best_config') or {}).get('blueprint')
        if bp is None and meta and meta.get('rule_hybrids'):
            bp = meta['rule_hybrids']['hybrids'][0]['blueprint']
        if bp is None:
            no_recipe[name] = {
                'why': ('the precomputed pool holds no stored search that '
                        'scored this downstream model at this budget, so no '
                        'recipe was searched for it; it is NOT given another '
                        "model's recipe"
                        if not meta else
                        'the pool returned a winner for this model but that '
                        'configuration carried no recipe, so no feature space '
                        'could be rebuilt from it'),
                'lookup_error': None}
            continue
        by_model[name] = {'search': search, 'meta': meta, 'blueprint': bp}
    return by_model, no_recipe


def _import_experiment(exp_id: str):
    """Dynamically import a noise experiment module by ID (e.g. 'n01')."""
    module_map = {
        'c10': 'c10_noise_robustness',
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n06': 'n06_bindingdb_ground_truth',
        'n07': 'n07_remediation',
        'n09': 'n09_lnl_survey',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
        'n14': 'n14_activity_cliff_landscape',
        'n15': 'n15_clean_vs_noisy_hybrid',
        'n16': 'n16_importance_under_noise',
        'n17': 'n17_replicate_removal',
        'n18': 'n18_clean_the_real_labels',
        'n19': 'n19_measurement_count',
        'n20': 'n20_clean_before_search',
        'n21': 'n21_neighbour_distance',
        'n22': 'n22_curator_flag',
        'n23': 'n23_uncertainty_cleaning',
        'n24': 'n24_delete_or_keep',
        'n25': 'n25_cliff_exemption',
        'n26': 'n26_noise_variance_from_repeats',
        'n27': 'n27_removal_cost_and_propagation',
        'n29': 'n29_document_bias',
        'n30': 'n30_censored_likelihood',
        'n31': 'n31_regime_and_breakeven',
        'n32': 'n32_iterative_and_dual_selection',
        'n33': 'n33_robust_training',
        'n34': 'n34_group_random_effects',
        'n35': 'n35_near_duplicate_disagreement',
        'n36': 'n36_remeasure_priority',
        'n37': 'n37_noise_budget',
        'n38': 'n38_noise_relative_to_model_error',
        'n39': 'n39_correlated_sources',
        'n40': 'n40_model_side_regularisation',
        'n41': 'n41_conformal_fdr_detection',
        'n42': 'n42_similarity_noise_sweep',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.noise.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        'c10': 'c10_noise_robustness',
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        # n06 was absent here while it was hand-run. Without it the resume
        # check and the saved file both went to a directory named 'n06', which
        # is not where anything looks for it.
        'n06': 'n06_bindingdb_ground_truth',
        'n07': 'n07_remediation',
        'n09': 'n09_lnl_survey',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
        'n14': 'n14_activity_cliff_landscape',
        'n15': 'n15_clean_vs_noisy_hybrid',
        'n16': 'n16_importance_under_noise',
        'n17': 'n17_replicate_removal',
        'n18': 'n18_clean_the_real_labels',
        'n19': 'n19_measurement_count',
        'n20': 'n20_clean_before_search',
        'n21': 'n21_neighbour_distance',
        'n22': 'n22_curator_flag',
        'n23': 'n23_uncertainty_cleaning',
        'n24': 'n24_delete_or_keep',
        'n25': 'n25_cliff_exemption',
        'n26': 'n26_noise_variance_from_repeats',
        'n27': 'n27_removal_cost_and_propagation',
        'n29': 'n29_document_bias',
        'n30': 'n30_censored_likelihood',
        'n31': 'n31_regime_and_breakeven',
        'n32': 'n32_iterative_and_dual_selection',
        'n33': 'n33_robust_training',
        'n34': 'n34_group_random_effects',
        'n35': 'n35_near_duplicate_disagreement',
        'n36': 'n36_remeasure_priority',
        'n37': 'n37_noise_budget',
        'n38': 'n38_noise_relative_to_model_error',
        'n39': 'n39_correlated_sources',
        'n40': 'n40_model_side_regularisation',
        'n41': 'n41_conformal_fdr_detection',
        'n42': 'n42_similarity_noise_sweep',
    }
    return name_map.get(exp_id, exp_id)


# Which experiments can be pointed at a condition, and how they take it.
# n13 sweeps every condition by default and takes a LIST; the others run one and
# take a name. n07 hardcodes the outlier condition, n15 is gaussian by
# construction, and n01/n02/n06/n14 inject nothing — none of them can honour a
# condition, so asking raises rather than being ignored.
# n17 is not here either, and for the same reason as n01/n02/n06/n14: it injects
# nothing, so there is no condition to point it at. Nor is n19, for the same
# reason -- its noise is the disagreement between laboratories and nobody chose
# its shape. Nor is n22: its suspect set is ChEMBL's own curator field and
# nobody here chose that either.
# n21 takes a condition but only the five in DETECTION_IS_DEFINED, because
# average precision is its headline and under the other six every training
# molecule's label moves. It raises on the other six naming all five.
# n07 joined 2026-09-23: it sweeps through sweep_conditions now (c46b71e), which
# reads `noise_condition`, so it can run one condition like the others. Left
# out, a job given a condition dropped it and n07 ran only in the no-condition
# job, all fourteen conditions end to end.
# n26, n27, n29 and n30 are not here either. n26 and n29 inject nothing. n27
# injects gaussian by construction, like n15. n30 runs censoring only, named in
# its own file, so a job asking for any other condition must not reach it.
# n32, n33 and n34 joined 2026-09-24. n32 writes a skip under a condition where
# every label moves; n34 writes a skip under any condition but the two grouped
# ones. n31, n35 and n36 are left out for the reasons given at their entries in
# SINGLE_DATASET_EXPERIMENTS.
# n40 and n41 joined 2026-09-25. n41 writes a skip under a condition where every
# label moves, as n32 does. n37, n38 and n39 are left out for the reasons at
# their entries in SINGLE_DATASET_EXPERIMENTS.
# n24 and n25 joined 2026-09-25: both read `noise_condition`
# (n24:435, n25:786).
# n42 joined 2026-09-25: it sweeps through sweep_conditions, which reads
# `noise_condition` (n42.run).
HONOURS_NOISE_CONDITION = {'n07', 'n09', 'n11', 'n12', 'n16', 'n20', 'n21', 'n23',
                           'n24', 'n25',
                           'n32', 'n33', 'n34', 'n40', 'n41', 'n42'}
SWEEPS_ALL_CONDITIONS = {'n13'}

# Experiments that take a condition but RAISE under any condition outside a
# set: n21 (n21:36-40) and n25 (n25:787-793) need some molecules left
# uncorrupted, so only DETECTION_IS_DEFINED. Since 2026-09-25 a failed
# experiment makes run_dataset.py exit 1, so a job named with, say, gaussian
# would fail every time on a condition these two cannot run by design.
# run_all writes a skip marker carrying the reason for them instead, the way
# should_skip_experiment does; an explicit sweep in their own run() is
# unaffected.
# n24 joined 2026-09-27: it raises outside DETECTION_IS_DEFINED
# (n24_delete_or_keep.py:740-751), and a server check on herg gave it 'uniform'.
# It now gets the same skip marker as n21 and n25 instead of a crash.
# 2026-10-05 (chat 5): the decided kinds where a corrupted subset exists, plus
# the classification flip kinds, which also corrupt a subset.
from kirby.noise_spec import CLASSIFICATION_STRATEGIES as _CLF_KINDS  # noqa: E402
# Kept in step with DECIDED_REGRESSION_CONDITIONS (defined further down) by
# the assert next to it.
_DETECTION_KINDS = (('outlier_p05', 'censoring', 'shift_only_p10',
                     'sparse_dprime_up_p10') + tuple(_CLF_KINDS))
CONDITION_LIMITED_TO = {'n21': _DETECTION_KINDS,
                        'n24': _DETECTION_KINDS,
                        'n25': _DETECTION_KINDS}


def _takes_force_rerun(mod):
    """Whether mod.run can be handed force_rerun without breaking it.

    True when run() names force_rerun, or when it hands its kwargs to
    sweep_conditions, which pops force_rerun (noise_helpers.sweep_conditions)
    before calling the single-condition entry point. Any other run() is not
    given the keyword: a **kwargs that is forwarded on to a function without
    it would raise. Added 2026-09-25 so --force-rerun reaches the condition
    files a sweep writes; before, run_all never passed it and a condition
    file on disk was always skipped.
    """
    import inspect
    fn = getattr(mod, 'run', None)
    if fn is None:
        return False
    try:
        if 'force_rerun' in inspect.signature(fn).parameters:
            return True
        return 'sweep_conditions(' in inspect.getsource(fn)
    except (TypeError, OSError, ValueError):
        return False


def _error_record(exp_id, exc, kind=None):
    """What run_all stores for a failed experiment. The traceback text is
    kept so run_dataset.py can write it to <output>/_errors/ (2026-09-25)."""
    import traceback
    return {'error': str(exc) or type(exc).__name__,
            'error_kind': kind or type(exc).__name__,
            'traceback': traceback.format_exc()}


# Datasets that reuse another dataset's DATASET_REPS entry (see run_all).
REP_LIST_FROM = {'herg_ic50': 'herg_ki'}


def load_noise_data(dataset_name, rep_names=None):
    """(data, rep_names): the dataset as run_all loads it, before any search.

    The approved representation list (DATASET_REPS, or REP_LIST_FROM), the
    NaN guard, and the augmentation representations in data['augmentations'].
    Split out of run_all on 2026-10-02 so a caller that must not start a
    search (framework f36) loads the same data without run_all's precompute.
    """
    # Resolve rep list
    if rep_names is None:
        # herg_ic50 has no DATASET_REPS entry of its own: it takes herg_ki's
        # representations (same target, same molecule type; user decision).
        # Mapped here rather than added to DATASET_REPS, so the precompute and
        # status scripts that loop over every DATASET_REPS key do not pick it up.
        ds_reps = DATASET_REPS.get(REP_LIST_FROM.get(dataset_name, dataset_name))
        if ds_reps is None:
            raise ValueError(
                f"No rep list for '{dataset_name}'. Pass rep_names explicitly "
                f"or add it to DATASET_REPS in complexity_helpers.py."
            )
        rep_names = ds_reps['general']

    t0 = time.time()
    print(f"\nLoading {dataset_name}...")
    data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # NaN guard: some reps produce NaN for certain sequences. Replace with 0 and warn.
    import numpy as np
    reps_data = data.get('reps', {})
    for rname, X in list(reps_data.items()):
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            pct = 100 * nan_count / X.size
            print(f"  WARNING: {rname} has {nan_count} NaN values ({pct:.1f}%) — imputing 0")
            reps_data[rname] = np.nan_to_num(X, nan=0.0)

    # ── Load augmentation reps ──
    aug_rep_names = DATASET_REPS.get(dataset_name, {}).get('augmentation', [])
    if aug_rep_names:
        missing = [r for r in aug_rep_names if r not in reps_data]
        if missing:
            print(f"  Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing)
            aug_reps = {}
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_reps[rn] = arr
            data['augmentations'] = aug_reps
        else:
            data['augmentations'] = {r: reps_data[r] for r in aug_rep_names}
        print(f"  Augmentations ({len(data['augmentations'])}): "
              f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
    else:
        data['augmentations'] = None

    return data, rep_names


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None,
            pruning_mode: str = 'default_pruning',
            force_rerun: bool = False,
            noise_condition: str = None):
    """Load data once and run all single-dataset noise experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: built from the DEFAULT_* constants).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
        noise_condition: one of kirby.noise_spec.REGRESSION_STRATEGIES (or the
            classification flip models). None leaves every experiment on its own
            default, which is gaussian for n09/n11/n12 and the outlier condition
            for n07. Anything else is forwarded to the experiment AND written
            into the output filename, so two conditions on one dataset no longer
            land on one file. Before 2026-09-12 the condition reached neither the
            path nor the resume fingerprint, so the second run of a dataset
            overwrote the first or was skipped as already done — which is why
            nothing ever passed one.
        pruning_mode: One of PRUNING_MODES. 'default_pruning' = legacy
            single-stage prune. 'aggressive_pruning' = F16-validated multi-
            stage pruning (checkpoints=[25,50,100], td=0.5) applied globally
            to all models, no n_samples gate. F16: ~58% fold-0 eval reduction.
    """
    if config is None:
        from kirby.find_best_hybrid import (
            DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS, DEFAULT_N_FOLDS,
        )
        # Production precompute uses FULL_BUDGETS =
        # [10,25,50,100,250,500,1000,2000], which CONTAINS the hybrid budgets
        # n10 requests ([25,100,250]). An earlier four-entry list
        # ([50,500,1000,2000]) did not, so n10's strict budget guard skipped
        # all three and the hybrid-vs-base comparison (n10's whole point)
        # came back silently empty. Both the in-memory precompute below and the SLURM
        # disk cache (tests/precompute_hybrids.py) already build
        # FULL_BUDGETS, so this only makes config['budgets'] truthful and keeps
        # n10's requested hybrid budgets reachable.
        from experiments._shared import FULL_BUDGETS
        config = {
            'n_splits': DEFAULT_N_FOLDS,
            'allocations': DEFAULT_ALLOCATION_METHODS,
            'importance_methods': DEFAULT_IMPORTANCE_METHODS,
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': FULL_BUDGETS,
            'models': DOWNSTREAM_MODELS_NOISE,
        }
    # Search axes from the decided grid (R1, R3), whoever built the config.
    from experiments._shared import decided_config
    config = decided_config(dict(config), dataset_name)
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)
        if noise_condition:
            # A condition names noise that only some experiments inject. The
            # rest pin their own condition in the file and read no keyword, and
            # the loop below RAISES on the first one rather than running
            # gaussian under a filename that says otherwise. `c10` is first in
            # the default list and is not one of the eight, so before this every
            # job submitted with a condition and no explicit experiment list
            # died on its first experiment having done nothing.
            #
            # Only the default list is narrowed. An explicit list that names an
            # experiment which cannot honour the condition still raises: asking
            # for n07 under grouped_shifted is a thing to be told about, not to
            # have silently dropped.
            _takes = HONOURS_NOISE_CONDITION | SWEEPS_ALL_CONDITIONS
            _dropped = [e for e in experiments if e not in _takes]
            experiments = [e for e in experiments if e in _takes]
            print(f"  noise_condition={noise_condition!r}: running the "
                  f"{len(experiments)} experiments that inject it "
                  f"({', '.join(experiments)}).")
            if _dropped:
                print(f"  not run under this condition, they name their own: "
                      f"{', '.join(_dropped)}")
            if not experiments:
                raise ValueError(
                    f"noise_condition={noise_condition!r} was asked for and none "
                    f"of the default experiments injects a condition. This is a "
                    f"bug in HONOURS_NOISE_CONDITION or SINGLE_DATASET_EXPERIMENTS.")

    # Resolve rep list (load_noise_data resolves it the same way; done here
    # first so the banner can name the representations).
    if rep_names is None:
        ds_reps = DATASET_REPS.get(REP_LIST_FROM.get(dataset_name, dataset_name))
        if ds_reps is None:
            raise ValueError(
                f"No rep list for '{dataset_name}'. Pass rep_names explicitly "
                f"or add it to DATASET_REPS in complexity_helpers.py."
            )
        rep_names = ds_reps['general']
    print(f"\n{'=' * 70}")
    print(f"NOISE EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    t0 = time.time()
    data, rep_names = load_noise_data(dataset_name, rep_names)
    import numpy as np
    reps_data = data.get('reps', {})

    # ── No whole-dataset searches (R2, R6; _shared.no_startup_searches) ──
    # This includes the similarity-objective pass that ran here until
    # 2026-10-05; a similarity-objective search is built per fold by the
    # experiment that needs it.
    from experiments._shared import no_startup_searches
    precompute_errors = no_startup_searches(data, pruning_mode, 'noise')

    # ── NOISY-LABEL hybrid searches: NOT precomputed, deliberately ──
    # Removed 2026-09-11. This block built one full search per noise level, on
    # every dataset a noise run touched, and nothing read any of them.
    #
    # Two things disagreed. The block built searches that TRAIN on corrupted
    # activity values and pick their winner against the CLEAN ones. n15 — the
    # only experiment in the suite that asks for a noisy search — passes
    # score_on_noisy_labels=True (n15:127), which asks for the version that
    # picks its winner on the corrupted values too. find_best_hybrid files the
    # two under different tiers, so they could never meet: n15 missed on every
    # layer and paid for its own searches on top of these.
    #
    # The dict stays, empty, because get_best_search_for_downstream still reads
    # it for a caller that asks for a noisy search WITHOUT
    # score_on_noisy_labels (_shared.py:5220). No such caller exists in the
    # suite today; if one appears it will build what it needs on its first call
    # and land it in the result store, the same way n15 already does.
    data['precomputed_searches_noisy'] = {}

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

    # ── Run experiments ──
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
        # The condition is part of the identity of the run, so it is part of the
        # filename and of the file the resume check looks for.
        _sfx = f"__{noise_condition}" if noise_condition else ""
        _out_json = os.path.join(
            output_dir, _experiment_full_name(exp_id),
            f"{dataset_name}{_sfx}.json")
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
                                    _experiment_full_name(exp_id), dataset_name,
                                    filename_suffix=noise_condition)
            continue

        # An experiment that cannot be imported is a FAILED experiment, not a
        # skip. Until 2026-09-25 this printed a line and continued, so a
        # syntax error or a missing module left no file, no error record and a
        # run that looked finished. Recorded now; run_dataset.py exits 1.
        try:
            mod = _import_experiment(exp_id)
        except Exception as e:
            print(f"  {exp_id} FAILED: import failed "
                  f"({type(e).__name__}: {e})")
            import traceback
            traceback.print_exc()
            results[exp_id] = _error_record(exp_id, e, kind=(
                f'import:{type(e).__name__}'))
            continue

        _limited = CONDITION_LIMITED_TO.get(exp_id)
        if noise_condition and _limited and noise_condition not in _limited:
            _skip = {'skipped': True,
                     'reason': (f'{exp_id} runs only under {list(_limited)}: '
                                f'under {noise_condition!r} every training '
                                f'label moves and no molecule has a right '
                                f'one to find'),
                     'noise_condition': noise_condition}
            print(f"  SKIP {exp_id}: {_skip['reason']}")
            results[exp_id] = _skip
            save_experiment_results(_skip, output_dir,
                                    _experiment_full_name(exp_id), dataset_name,
                                    filename_suffix=noise_condition)
            continue

        try:
            # Only some experiments take a condition, and the ones that do not
            # would swallow the keyword through **kwargs and run gaussian while
            # the filename said otherwise. Refuse instead.
            _exp_kwargs = {}
            if noise_condition:
                if exp_id in HONOURS_NOISE_CONDITION:
                    _exp_kwargs = {'noise_condition': noise_condition}
                elif exp_id in SWEEPS_ALL_CONDITIONS:
                    _exp_kwargs = {'noise_types': [noise_condition]}
                else:
                    raise ValueError(
                        f"run_all: noise_condition={noise_condition!r} was asked "
                        f"for, and {exp_id} cannot honour it — it names its own "
                        f"condition in the file and reads no keyword. Run it "
                        f"without noise_condition, or list only the experiments "
                        f"in {sorted(HONOURS_NOISE_CONDITION | SWEEPS_ALL_CONDITIONS)}.")
            if _takes_force_rerun(mod):
                _exp_kwargs['force_rerun'] = bool(force_rerun)
            _exp_config = one_fold_screen_config(config, exp_id, data=data,
                                                 dataset_name=dataset_name)
            exp_result = mod.run(data, _exp_config, output_dir, dataset_name,
                                 **_exp_kwargs)
            results[exp_id] = exp_result

            exp_full_name = _experiment_full_name(exp_id)
            # ⛔ A SWEEP UNDER A NAMED CONDITION HAS ALREADY WRITTEN ITS FILE.
            # sweep_conditions saves each condition to
            # <exp>/<dataset>__<condition>.json as it lands and returns an
            # INDEX of those files. Saving that index here under the same
            # suffix wrote it to the same path and replaced the scores with the
            # list of filenames. Without a condition the index goes to
            # <dataset>.json, a different file, which is what the status
            # scripts read -- so that save stays.
            if (noise_condition and isinstance(exp_result, dict)
                    and 'conditions_written' in exp_result):
                print(f"  {exp_id}: condition file written by the sweep itself "
                      f"-- not overwriting it with the sweep index")
            else:
                save_experiment_results(exp_result, output_dir, exp_full_name,
                                        dataset_name, fingerprint=_fingerprint,
                                        filename_suffix=noise_condition)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = _error_record(exp_id, e)

        except SystemExit as e:
            # SystemExit is not an Exception: a library that calls sys.exit()
            # instead of raising walks past the handler above and ends the
            # whole run with nothing said. complexity_helpers.run_all catches
            # it for this reason (complexity_helpers.py:1242-1263, c21 on
            # flab_thermo_tresanco, 2026-09-25). Same handling here, 2026-09-25.
            print(f"  {exp_id} FAILED: a library called sys.exit({e.code}) "
                  f"instead of raising. Caught so the remaining experiments "
                  f"still run.")
            import traceback
            traceback.print_exc()
            results[exp_id] = _error_record(
                exp_id, f'SystemExit({e.code}) — a library exited the process',
                kind='SystemExit')

        except KeyboardInterrupt:
            # Deliberate: Ctrl-C and SIGINT must still stop the run.
            raise

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} — {len(results)} experiments in {total:.0f}s")
    print(f"{'=' * 70}")

    # ── Extrapolate every sweep back to zero error ──────────────────────────
    # A reading of the files just written, not a run: it trains nothing and loads no
    # dataset, so it costs seconds at the end of a job that ran for hours. Each
    # injecting experiment scored six levels of added error; the labels were already
    # carrying b of their own before anything was added, error adds in quadrature, so
    # a level L sits at a total of sqrt(b^2 + L^2). Running the curve back to a total
    # of zero estimates the score with the training labels exact.
    #
    # It goes here rather than in a script somebody runs afterwards because the
    # numbers it reads are only complete once the experiments above have finished, and
    # a step that has to be remembered is a step that gets missed.
    #
    # `write_readout` catches its own exceptions and returns None. This try/except is
    # the second one on purpose: an ImportError on the module itself would otherwise
    # take down a job that had already produced every experiment's results.
    print(f"\n  Extrapolating each noise sweep back to zero error...")
    try:
        from experiments.noise.noise_ceiling_analyse import write_readout
        write_readout(output_dir, dataset_name)
    except Exception as _e:
        print(f"  [noise ceiling] not written and the run continues — "
              f"{type(_e).__name__}: {_e}")

    # Underscore key so it does not collide with an experiment ID.
    results['_precompute_errors'] = precompute_error_audit
    return results


# ══════════════════════════════════════════════════════════════════════════
# CHAT 5 ALIGNMENT (2026-10-05): the shared pieces every noise experiment
# calls. HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.4, KNN_SEARCH_SPACE_RECORD.md.
# ══════════════════════════════════════════════════════════════════════════

# ── The decided noise grid (KNN_SEARCH_SPACE_RECORD, "The noise grid",
#    Adelaide 2026-10-05; f37) ─────────────────────────────────────────────
# Six kinds of damage on a regression dataset. f37 measured regression only,
# so a classification dataset keeps its flip models and flip-rate grid
# (CLASSIFICATION_STRATEGIES, dataset_policy.CLASSIFICATION_NOISE_LEVELS).
DECIDED_REGRESSION_CONDITIONS = ('gaussian', 'outlier_p05', 'grouped_shifted',
                                 'censoring', 'shift_only_p10',
                                 'sparse_dprime_up_p10')
assert set(_DETECTION_KINDS) - set(_CLF_KINDS) == {
    c for c in DECIDED_REGRESSION_CONDITIONS if c in DETECTION_IS_DEFINED}

# Levels per kind, by dataset group (similarity_space.knn_group). Each kind
# reads the level in its own unit, as f37 did: a fraction of the clean
# training-label SD (registry kinds), a fraction of labels clipped
# (censoring), label SDs (shift_only_p10), SDs of lightgbm's out-of-fold
# residual (sparse_dprime_up_p10). 0 is the clean labels and is run once.
DECIDED_LEVELS = {'full_knn': (0.0, 0.5, 1.0, 1.5),
                  'fast_knn': (0.0, 1.0)}
DECIDED_CENSORING_LEVELS = {'full_knn': (0.0, 0.10, 0.25, 0.40),
                            'fast_knn': (0.0, 0.25)}
# ── The one-pass SCREEN (Adelaide 2026-10-07) ──────────────────────────────
# The first pass over the noise experiments runs two levels on every dataset
# (0 and the reporting level 1.0; censoring 0 and 0.25, the fast_knn grid) and
# five of the six kinds. A PROMOTED run (KIRBY_NOISE_PROMOTE names the
# experiment; the ledger's noise_full stage) runs the full grid above: all
# four levels on a full_knn dataset and all six kinds.
# Dropped from the screen: sparse_dprime_up_p10. f37 (scratchpad/f37_results,
# pred_corr_L1.csv) found its held-out predictions correlate with
# shift_only_p10's at 0.931 (caco2), 0.938 (herg_ki) and 0.970 (logd) against
# 0.951, 0.946 and 0.972 for two draws of shift_only_p10 itself, so the two
# kinds do the same thing to a model; shift_only_p10 is the cheaper of the two
# (sparse_dprime needs lightgbm's out-of-fold residual first).
SCREEN_LEVELS = (0.0, 1.0)
SCREEN_CENSORING_LEVELS = (0.0, 0.25)
SCREEN_DROPPED_CONDITIONS = ('sparse_dprime_up_p10',)
# Classification screen (Adelaide 2026-10-07): flip rates 0 and 0.10, all six
# flip models; a promoted run sweeps dataset_policy.CLASSIFICATION_NOISE_LEVELS.
# 0.10: on a balanced binary set the flipped labels correlate 0.80 with the
# true ones, against 0.71 for the regression level 1.0 (13 of the 15
# classification datasets are binary); humanness_therapeutic's n20 search
# already lost 0.093 auc between 0 and 0.10 (0.891 to 0.798).
SCREEN_CLASSIFICATION_LEVELS = (0.0, 0.10)


def promoted_run():
    """True inside a promoted (full-grid) run: KIRBY_NOISE_PROMOTE is set.
    The ledger sets it only for noise_full units; by hand, export it."""
    return bool(str(os.environ.get('KIRBY_NOISE_PROMOTE') or '').strip())


# DECIDED, Adelaide 2026-10-06: fast_knn gets level 0 and 1.0 (censoring 0 and
# 0.25). 1.0 is the reporting level (dataset_policy: "The REPORTING level ...
# is 1.0"); 0.25 is censoring's middle value. logd is in fast_knn, with the
# other fast_knn datasets. auc_norm is absent there by decision; the plain
# difference against level 0 is reported instead.

# The representation sparse_dprime's unit is measured on: the first of the
# dataset's pruned representations as loaded (DATASET_REPS order), one choice
# for every model and configuration so the damage is identical across them
# (f37 used ecfp4 for the same reason).


def dataset_knn_group(data, dataset_name):
    """'full_knn' or 'fast_knn' for this dataset, from its molecule count
    (more than 5000 is fast_knn). On a dataset with one row per measurement
    (herg_ic50) the count is distinct molecules when SMILES are loaded."""
    from experiments.noise import similarity_space as ss
    smi = data.get('smiles')
    n = (len(set(smi)) if smi is not None and len(smi)
         else len(data['labels']))
    return ss.knn_group(dataset_name, n)


def decided_conditions(data):
    """The damage kinds this dataset is swept over."""
    from kirby.noise_spec import CLASSIFICATION_STRATEGIES
    if data.get('task_type') == 'classification':
        return list(CLASSIFICATION_STRATEGIES)
    return conditions_for_task(data)


def decided_levels(data, dataset_name, condition):
    """The levels one kind is swept at on this dataset, 0 first."""
    from experiments.dataset_policy import CLASSIFICATION_NOISE_LEVELS
    if data.get('task_type') == 'classification':
        if not promoted_run():                  # the screen
            return SCREEN_CLASSIFICATION_LEVELS
        return tuple(CLASSIFICATION_NOISE_LEVELS)
    g = dataset_knn_group(data, dataset_name)
    if not promoted_run():                      # the screen: two levels everywhere
        return SCREEN_CENSORING_LEVELS if condition == 'censoring' else SCREEN_LEVELS
    if condition == 'censoring':
        return DECIDED_CENSORING_LEVELS[g]
    return DECIDED_LEVELS[g]


def level_unit(condition, is_clf=False):
    if is_clf:
        return 'share of training labels flipped'
    if condition == 'censoring':
        return 'share of training labels clipped'
    if condition in SHIFT_ONLY_CONDITIONS:
        return 'clean training-label SDs (the shift)'
    if condition in RELATIVE_CONDITIONS:
        return "SDs of lightgbm's out-of-fold residual on the clean training labels"
    return 'fraction of the clean training-label SD'


_RESID_SD_CACHE = {}


def inject_condition(data, dataset_name, condition, level, draw, *,
                     train_idx, groups=None, base_seed=42, heavy_atoms=None):
    """ONE call for every kept kind of damage: (y_noisy_train, corrupted mask
    over the training rows, provenance). Routes to inject_train_labels (the
    registry kinds and the classification flips), inject_shift_train_labels
    (shift_only_*) or inject_relative_train_labels (sparse_dprime*), so an
    experiment does not carry the dispatch itself. Level 0 returns the clean
    training labels and an all-False mask.

    groups: the FULL scaffold-group array (dataset_policy.noise_groups_for);
    looked up when None.
    """
    import numpy as np
    y_all = np.asarray(data['labels'])
    tr = np.asarray(train_idx)
    is_clf = data.get('task_type') == 'classification'
    prov_base = {'condition': condition, 'level': float(level),
                 'draw': int(draw),
                 'level_unit': level_unit(condition, is_clf)}
    if float(level) == 0.0:
        return (y_all[tr].copy(), np.zeros(len(tr), dtype=bool),
                dict(prov_base, note='level 0: the labels as delivered'))
    if condition in SHIFT_ONLY_CONDITIONS:
        y, m, p = inject_shift_train_labels(
            y_all, train_idx=tr, condition=condition, delta_sd=float(level),
            draw=int(draw), base_seed=base_seed)
        return y, m, dict(prov_base, **p)
    if condition in RELATIVE_CONDITIONS:
        from experiments._shared import as_rep_dtype
        from kirby.scaling import FeatureScaler
        rep = next(iter(data['reps']))
        key = (dataset_name, rep, len(y_all), hash(tr.tobytes()))
        if key not in _RESID_SD_CACHE:
            X = FeatureScaler().fit_transform(as_rep_dtype(
                np.asarray(data['reps'][rep])[tr]))
            g_tr = (None if data.get('groups') is None
                    else np.asarray(data['groups'])[tr])
            _RESID_SD_CACHE[key] = oof_residual_sd(
                X, y_all[tr].astype(float), groups=g_tr)[0]
        y, m, p = inject_relative_train_labels(
            y_all, train_idx=tr, condition=condition, level=float(level),
            draw=int(draw), resid_sd=_RESID_SD_CACHE[key],
            heavy_atoms=heavy_atoms, base_seed=base_seed)
        return y, m, dict(prov_base, resid_sd_representation=rep, **p)
    if groups is None:
        from experiments.dataset_policy import noise_groups_for
        groups = noise_groups_for(data)[0]
    spec = noise_spec_for(dataset_name, condition, float(level),
                          base_seed=base_seed, noise_seed=int(draw),
                          is_clf=is_clf)
    y, m, p = inject_train_labels(y_all, spec, train_idx=tr, is_clf=is_clf,
                                  groups=groups, dataset_name=dataset_name)
    return y, m, dict(prov_base, **(p or {}))


# ── Outer folds: one setting, any count (handoff 2.4 item 14) ──────────────
# The fold rule is Adelaide's to decide (2.12). The count comes from the
# experiment config (config['n_splits']), overridable with KIRBY_NOISE_N_FOLDS.
# KIRBY_NOISE_ONLY_FOLDS="0" (or "0,3") runs only those folds of that split,
# so a one-fold screen is fold 0 of the same split the full run uses and the
# per-fold searches it stores are reused when the other folds run.

def noise_n_folds(config):
    raw = os.environ.get('KIRBY_NOISE_N_FOLDS')
    n = int(raw) if raw not in (None, '') else int(config['n_splits'])
    if n < 1:
        raise ValueError(f'outer fold count must be at least 1, got {n}')
    return n


def noise_outer_folds(data, config, seed=42):
    """[(fold number, train_idx, val_idx)] and the fold count.

    The splits are kirby.find_best_hybrid.nested_cv_outer_folds, the ones the
    per-fold searches are filed under (ocv{n}f{i}s{seed}), so a hybrid searched
    on a fold's training rows is scored on that fold's held-out rows. A count
    of 1 is one group-aware 80/20 split, not 1-fold CV.
    """
    import numpy as np
    from kirby.find_best_hybrid import nested_cv_outer_folds
    n = noise_n_folds(config)
    folds = nested_cv_outer_folds(
        data['reps'], np.asarray(data['labels']), data.get('groups'), n, seed,
        data.get('task_type') or 'regression')
    raw = os.environ.get('KIRBY_NOISE_ONLY_FOLDS')
    only = (None if raw in (None, '') else
            {int(x) for x in str(raw).split(',') if x.strip() != ''})
    _src = 'KIRBY_NOISE_ONLY_FOLDS'
    if only is None and config.get('only_folds') is not None:
        only = {int(f) for f in config['only_folds']}
        raw = ','.join(str(f) for f in sorted(only))
        _src = 'one-fold screen'
    out = [(i, np.asarray(tr), np.asarray(va)) for i, (tr, va) in enumerate(folds)
           if only is None or i in only]
    if only is not None:
        print(f"  {_src} ({raw}): running folds "
              f"{[i for i, _, _ in out]} of {n}")
    return out, n


def fold_spread(values):
    """Mean and spread over folds, or the note that there is none. Never a
    division by zero on one fold, never a significance test."""
    import numpy as np
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v:
        return {'mean': None, 'sd': None, 'n_folds': 0,
                'per_fold': [], 'note': 'no fold scored'}
    if len(v) == 1:
        return {'mean': v[0], 'sd': None, 'n_folds': 1, 'per_fold': v,
                'note': 'one fold, no spread'}
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)),
            'min': min(v), 'max': max(v), 'n_folds': len(v), 'per_fold': v}


# ── The 8 configurations (R5) per fold ─────────────────────────────────────
# Per decided model: its 2 best base representations and 2 best hybrids, all
# ranked by a search on that fold's training rows (fold_standard_selection),
# so 2 models x 4 = 8 fitted configurations. The distance hybrid and the
# similarity hybrid are neighbour spaces only (handoff 2.3 item 1).
CONFIGURATION_SPACES = ('base_1', 'base_2', 'hybrid', 'hybrid__general2')

# ⛔ OPEN (handoff 2.12): which labels the per-fold searches see. They see the
# clean training labels (label_noise=None), as the whole-dataset precompute
# did. n15 and n20 search on damaged labels on purpose and pass their own.


def fold_configurations(data, dataset_name, *, fold_i, train_idx, n_folds,
                        seed=42, label_noise=None, verbose=False,
                        condition_suffix=None, train_labels=None,
                        noise_groups=None):
    """{model: entry} for one outer fold: _shared.fold_neighbour_space_recipes
    on this dataset's pruned representations and decided grid. Each entry has
    'base' (2 representation names), 'hybrids', 'recipes' (every hybrid
    space, neighbour spaces included), 'unavailable' and
    'fit_configurations'. A model whose search failed maps to None.

    condition_suffix: REQUIRED whenever `data` is not the dataset as loaded --
    rows dropped or collapsed (herg_ic50 one row per molecule, handoff 2.3
    item 14), or labels other than the clean ones -- so the stored per-fold
    search of one row set is never read for another."""
    import numpy as np
    from experiments._shared import fold_neighbour_space_recipes
    return fold_neighbour_space_recipes(
        data['reps'], np.asarray(data['labels']), data.get('groups'),
        train_idx=np.asarray(train_idx), fold_i=int(fold_i),
        n_outer_folds=int(n_folds), dataset_name=dataset_name,
        augmentations=data.get('augmentations'),
        task_type=data.get('task_type') or 'auto', seed=seed,
        label_noise=label_noise, verbose=verbose,
        condition_suffix=condition_suffix, train_labels=train_labels,
        noise_groups=noise_groups)


def fold_model_inputs(data, dataset_name, model_name, entry, *, train_idx,
                      val_idx, y_train, factory=None, metric=None,
                      spaces=CONFIGURATION_SPACES):
    """The standardised (X_train, X_val) of each fitted configuration of one
    model on one fold: {space: (X_tr, X_va)}, {space: why unavailable}, and
    {space: what it is} (the representation name or the recipe label).

    Hybrids are rebuilt on this fold's training rows from the per-fold
    recipe, with the labels the experiment holds (y_train), through the
    noise cache; the recipe's representation scores are the fold search's own
    (dirty-run bug 1). 0/1 features are left unscaled (kirby.scaling).
    """
    import numpy as np
    from experiments.noise.n07_remediation import fold_similarity_spaces
    if entry is None:
        return {}, {sp: 'the per-fold search failed for this model'
                    for sp in spaces}, {}
    tr, va, unavail = fold_similarity_spaces(
        dataset_name, [], model_input_tr=None, model_input_va=None,
        reps=data['reps'], train_idx=np.asarray(train_idx),
        val_idx=np.asarray(val_idx), y_train=y_train,
        groups_train=(None if data.get('groups') is None
                      else np.asarray(data['groups'])[np.asarray(train_idx)]),
        base_reps=entry.get('base') or [], recipes=entry.get('recipes') or {},
        augmentations=data.get('augmentations'), model_name=model_name,
        model_factory=factory, metric=metric,
        task_type=data.get('task_type'), extra_spaces=tuple(spaces))
    unavail = dict(unavail)
    # The per-fold search's own reason wins over fold_similarity_spaces'
    # generic one.
    for sp, why in (entry.get('unavailable') or {}).items():
        if sp in spaces and sp not in tr:
            unavail[sp] = why
    out = {sp: (tr[sp], va[sp]) for sp in spaces if sp in tr}
    names = {}
    for i, sp in enumerate(('base_1', 'base_2')):
        if sp in out and len(entry.get('base') or []) > i:
            names[sp] = entry['base'][i]
    for i, sp in enumerate(('hybrid', 'hybrid__general2')):
        if sp in out and len(entry.get('hybrids') or []) > i:
            names[sp] = entry['hybrids'][i].get('label')
    return out, unavail, names


def configuration_record(entry, names=None):
    """What a result file says about one model's configurations on one fold."""
    if entry is None:
        return {'available': False}
    bd = entry.get('base_detail') or {}
    rec = {'base': list(entry.get('base') or []),
           # each base representation's score in the per-fold search (mean
           # over its inner folds on this fold's training rows)
           'base_training_row_scores': [
               {k: e.get(k) for k in ('rep', 'score')}
               for e in (bd.get('selected') or [])],
           'hybrids': [{k: h.get(k) for k in (
               'label', 'score_mean', 'budget', 'allocation_method',
               'importance_method', 'internal_evaluator', 'rep_count',
               'aug_condition')}
               for h in (entry.get('hybrids') or [])],
           'hybrids_skipped_identical_features': [
               h.get('label') for h in
               (entry.get('hybrids_skipped_identical_features') or [])],
           'neighbour_spaces': sorted((entry.get('recipes') or {})),
           'unavailable': dict(entry.get('unavailable') or {}),
           'selection_scope': 'fold_train_rows'}
    if names:
        rec['configuration_names'] = dict(names)
    return rec


def draws_for(condition, draws):
    """The noise draws to run for one damage kind.

    Censoring clips at a fixed fraction of the training labels and reads no
    seed, so its draws give bit-identical labels (checked 2026-10-05 at 0.25
    clipped, 400 and 500 labels). DECIDED 2026-10-06 (Adelaide delegated): one
    draw under censoring, every other kind its full list.
    """
    draws = tuple(draws)
    if str(condition) == 'censoring':
        return draws[:1]
    return draws


# ── One-fold screen first (Adelaide, 2026-10-06) ───────────────────────────
# The noise experiments that compare many methods run fold 0 only, the way the
# preliminary noise runs did. Adelaide promotes the ones that look promising or
# interesting to all folds for the real results. The descriptive experiments,
# which compare no list of methods, run every fold. fold 0 is a fold of the
# same split the full run uses, so its per-fold searches are reused on
# promotion.
#   Promote:   KIRBY_NOISE_PROMOTE=n20,n25   (or 'all')
#   Override:  KIRBY_NOISE_ONLY_FOLDS still wins over both.
# The folds run go into every result file's input fingerprint, so a promoted
# run never reads the one-fold file back as finished.
DESCRIPTIVE_NOISE_EXPERIMENTS = frozenset({
    'c10',   # how the 8 configurations degrade; no methods compared
    'n01',   # prediction error on activity cliffs
    'n14',   # activity-cliff landscape; fits no model
    'n15',   # clean-searched against noise-searched recipe: 2 arms
    'n16',   # how feature importance moves under damage
    'n26',   # label-noise variance from repeat measurements
    'n27',   # what fewer training molecules cost
    'n37',   # how large the label noise is against model error
})
ONE_FOLD_SCREEN_FOLDS = (0,)


def one_fold_screen_config(config, exp_id, data=None, dataset_name=None):
    """config for one experiment: fold 0 only for a many-method noise
    experiment that has not been promoted; the first K folds for a descriptive
    experiment listed in _shared.OUTER_FOLDS_RUN (n15, n26: their own hybrid
    searches per fold; Adelaide 2026-10-07, same rule as c06/c07/c13/c23) on a
    dataset outside _shared.ALL_FOLDS_DATASETS; unchanged otherwise."""
    promoted = {x.strip() for x in str(os.environ.get('KIRBY_NOISE_PROMOTE')
                                       or '').split(',') if x.strip()}
    if 'all' in promoted or exp_id in promoted:
        return config
    if exp_id in DESCRIPTIVE_NOISE_EXPERIMENTS:
        from experiments._shared import OUTER_FOLDS_RUN, ALL_FOLDS_DATASETS
        k = OUTER_FOLDS_RUN.get(exp_id)
        ds = dataset_name or (data or {}).get('dataset_name')
        if (k and config.get('only_folds') is None and k < int(config.get('n_splits') or 0)
                and ds not in ALL_FOLDS_DATASETS):
            out = dict(config)
            out['only_folds'] = tuple(range(k))
            print(f"  {exp_id}: running outer folds 0..{k - 1} of the "
                  f"{config.get('n_splits')}-fold layout (OUTER_FOLDS_RUN)")
            return out
        return config
    out = dict(config)
    out['only_folds'] = ONE_FOLD_SCREEN_FOLDS
    print(f"  {exp_id}: one-fold screen, fold 0 of {config.get('n_splits')} "
          f"(promote with KIRBY_NOISE_PROMOTE={exp_id})")
    return out
