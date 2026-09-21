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


def noisy_search_specs(task_type: str, noise_seeds=None):
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
    from kirby.noise_spec import default_spec
    is_clf = (task_type == 'classification')
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
            out.append(default_spec(is_clf=is_clf, target_effective_noise=lvl,
                                    noise_seed=int(sd)))
    return out


# ── ONE injector for the whole noise suite ─────────────────────────────────
# n07, n09, n11, n12 and n13 each carried a private `_inject_noise`, and no two of
# them meant the same thing: n07 corrupted a fraction of rows at 0.3 of the label
# range, n09/n11/n12 added Gaussian noise to every row at a fraction of the range,
# and n13 had four distributions of its own. None went through noiseInject. They are
# replaced by the two functions below.

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

    digest = zlib.crc32(str(condition).encode('utf-8'))
    selection_seed = (int(base_seed) * 1000003 + digest
                      + int(noise_seed) * 15485863) & 0x7FFFFFFF
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

    A MOLECULE'S CORRUPTION IS A PROPERTY OF THE MOLECULE
    -----------------------------------------------------
    The draw runs ONCE over the whole label column and the training rows are
    indexed out of the result. It used to run on each fold's training labels, so
    a molecule got different noise depending on which fold it landed in.
    Measured on 600 molecules over five scaffold folds at level 0.5: 600 of 600
    molecules carried more than one corrupted value, under gaussian, outlier_p10,
    grouped_wider and grouped_shifted alike, and the delivered amount ranged from
    4.9% apart across folds (gaussian) to 23.9% (grouped_shifted). Drawing once
    gives 0 of 600 and one delivered amount.

    Two things follow from the whole-column draw and are not separately
    negotiable, because there is no fold-free alternative to either:
      - the amount is solved against the FULL column's clean spread, not the
        fold's training spread;
      - the cut-point for a label-keyed condition (censoring) is the full
        column's quantile, not the fold's.

    This is `RERUN_PLAN.md` 3.3b in ~/repos/qsar_qm_models, settled by the author
    2026-09-04, and the gate is `tests/test_noise_is_a_property_of_the_molecule.py`.
    `tests/alternative_data_noise_robustness.py` moved to it at commit dd58f4f;
    the noise suite moved 2026-09-12.

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
    from kirby.noise_spec import make_noisy_labels, make_noisy_labels_verbose, \
        noise_provenance_row

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
        y_noisy_all = make_noisy_labels(y_all, spec, is_clf=True)
        mask_all = (y_noisy_all != y_all)
        return (y_noisy_all[train_idx], mask_all[train_idx],
                {'noise_spec_id': spec.spec_id})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DoseWarning)
        res = make_noisy_labels_verbose(y_all, spec, groups=groups)
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
        mask_all = np.asarray(res.epsilon) != 0.0
    else:
        mask_all = scale > (scale.min() + 1e-12)
    y_noisy_all = np.asarray(res.y_noisy)
    prov = noise_provenance_row(res, spec)
    prov.update(_already_there(dataset_name, y_all, spec))
    return y_noisy_all[train_idx], mask_all[train_idx], prov


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
DETECTION_IS_DEFINED = ('outlier_p01', 'outlier_p05', 'outlier_p10',
                        'grouped_wider', 'censoring')

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


def detection_is_defined(condition) -> bool:
    """Whether a detection score means anything under this condition."""
    return (str(condition) in DETECTION_IS_DEFINED
            or str(condition) in DETECTION_DEFINED_WITHOUT_INJECTION)


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
    order = np.argsort(-sus, kind='stable')
    top = order[:n_corrupt]
    hits = int(mask[top].sum())
    return {
        'detection_average_precision': ap,
        'detection_precision_at_k': hits / float(n_corrupt),
        'detection_recall_at_k': hits / float(n_corrupt),
        'detection_n_corrupted': n_corrupt,
        'detection_n_training': int(mask.size),
        'detection_condition': str(condition),
    }


def hybrid_for_fold(train_reps, val_reps, y_train, bp, groups_train,
                    model, metric_fn, lower_is_better,
                    augmentations_train=None, augmentations_val=None):
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
        X_tr_h, X_va_h, _ = build_fold_hybrid(
            tr, va, y_train, bp,
            augmentations_train=augmentations_train,
            augmentations_val=augmentations_val,
            groups=groups_train,
            **({} if geometry else {'model': model, 'metric_fn': metric_fn,
                                    'lower_is_better': lower_is_better}),
        )
        return X_tr_h, X_va_h
    except Exception as exc:
        print(f"      hybrid build failed: {type(exc).__name__}: {exc}")
        return None, None


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
HONOURS_NOISE_CONDITION = {'n09', 'n11', 'n12', 'n16', 'n20', 'n21', 'n23'}
SWEEPS_ALL_CONDITIONS = {'n13'}


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
    print(f"NOISE EXPERIMENTS: {dataset_name.upper()}")
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

    # ── Resolve pruning_mode + apply n_samples gate ──
    from experiments._shared import resolve_pruning_mode, apply_pruning_gate
    _sched, _ovr_template, _min_n = resolve_pruning_mode(pruning_mode)
    _model_overrides = apply_pruning_gate(
        _ovr_template, len(data['labels']), _min_n)
    print(f"\n  Pruning mode: {pruning_mode}")
    if _model_overrides is not None:
        print(f"    overrides active for: {sorted(_model_overrides.keys())}")
    elif _ovr_template is not None:
        print(f"    overrides gated off (n={len(data['labels'])} < {_min_n})")

    # ── Precompute hybrid searches (general reps) ──
    # precompute_hybrid_searches catches a per-aug_condition exception, stores
    # None under that key, and appends one record to `.failures`. Those records
    # are copied into data['precompute_errors'][tier] here and written to
    # _precompute_errors/<dataset>.json below, so a failed aug_condition is
    # named in the output instead of existing only as a printed line.
    _task_type = data['task_type']
    precompute_errors = data.setdefault('precompute_errors', {})
    from experiments._shared import precompute_failures
    if len(reps_data) >= 2:
        from experiments._shared import precompute_hybrid_searches
        print(f"\n  Precomputing hybrid searches (general, {len(reps_data)} reps)...")
        precompute_results = precompute_hybrid_searches(
            reps=reps_data,
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name, tier='general',
            adaptive_pruning_schedule=_sched,
            model_pruning_aggressiveness=_model_overrides,
        )
        data['precomputed_searches'] = precompute_results
        precompute_errors['general'] = precompute_failures(precompute_results)

        # The SIMILARITY-objective search, built here so it is paid for once
        # per dataset instead of by whichever experiment asks first. n06 scores
        # a hybrid whose features were chosen by distance beside the one chosen
        # by prediction, and n02, n07 and n11 are due the same arm. Nothing
        # reads the return value: find_best_hybrid namespaces its own store by
        # objective, so an experiment that later asks for objective='similarity'
        # finds this search instead of running one inside the experiment.
        #
        # It is a full search, and it doubles this step. That is the cost of
        # having the distance-built arm at all, and it is paid on the cluster
        # once rather than inside a timed experiment.
        print(f"\n  Precomputing similarity-objective searches "
              f"({len(reps_data)} reps)...")
        try:
            sim_results = precompute_hybrid_searches(
                reps=reps_data,
                labels=data['labels'],
                groups=data.get('groups'),
                config=config,
                augmentations=data.get('augmentations'),
                task_type=_task_type,
                domain=data.get('domain', 'molecular'),
                dataset_name=dataset_name, tier='general',
                adaptive_pruning_schedule=_sched,
                model_pruning_aggressiveness=_model_overrides,
                objective='similarity',
            )
            precompute_errors['general_similarity'] = precompute_failures(
                sim_results)
        except Exception as exc:
            # A dataset where the geometry objective cannot be scored must not
            # take the whole noise run down with it; the arm is then absent and
            # every experiment that wanted it records that.
            print(f"  similarity-objective precompute failed: "
                  f"{type(exc).__name__}: {exc}")
            precompute_errors['general_similarity'] = [{
                'tier': 'general_similarity', 'dataset': dataset_name,
                'error_type': type(exc).__name__, 'error': str(exc)}]
    else:
        data['precomputed_searches'] = {}
        precompute_errors['general'] = []

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

        try:
            mod = _import_experiment(exp_id)
        except (ImportError, KeyError) as e:
            print(f"  SKIP {exp_id}: import failed ({e})")
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
            exp_result = mod.run(data, config, output_dir, dataset_name,
                                 **_exp_kwargs)
            results[exp_id] = exp_result

            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(exp_result, output_dir, exp_full_name,
                                    dataset_name, fingerprint=_fingerprint,
                                    filename_suffix=noise_condition)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e)}

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
