#!/usr/bin/env python3
"""C21: Single-Rep Augmentation Test.

Paper: Complexity
Question: a top-tier standalone rep already does well and hybridisation adds
    little. Can a handful of INDIVIDUAL extra features nudge it further?

Design: for each base rep in DATASET_REPS[dataset]['general'] (the 10-rep pool):
    A_baseline      base rep alone
    B_aug           base + the dataset's augmentation-category reps, uncapped.
                    These are the small engineered feature sets the surveys
                    assigned to 'augmentation' (PIPELINE_READINESS_TODO.md).
    C_topreps       base + the top-N features of every OTHER pool rep,
                    each capped to N by per-fold RF importance.
    D_both          B and C together (aug reps uncapped, pool reps capped to N).

    H0_hybrid         the multi-rep pool hybrid, no extras (reference)
    H1_hybrid_aug     that hybrid + the augmentation features

This file answers the PLACEMENT question: do extras do more on one strong
representation at its full width, or on the multi-representation hybrid at a
budget. Every B/C/D condition and every H1 arm is run under each admission
method named by --strategies and kept by the decided grid (2026-10-05: only
'uncapped', on ginkgo_colloidal and ginkgo_hic; H1 is off until the owner
decides, so only the single-representation half of this question runs).
The methods themselves are compared against each other in the framework
augmentation experiment, which sweeps the ranker, the per-donor cap, the size
rule and the margin.

NO SEARCH. Every non-baseline condition constructs the blueprint directly — one
base rep gets 100% of the budget at its full native dimension. Allocation and
internal-evaluator cannot change a single-rep selection. Importance can
(corrected 2026-10-05, tracker C21a): base features with importance 0 are not
kept, so it is taken from the decided grid. The H_* arms build a fixed
multi-rep recipe rather than picking one by argmax over the same folds they
are then scored on.

Two decoy donors are planted into the same pool as the real ones, unless
--no-decoys: a block of pure Gaussian noise as wide as the real donors
combined, and a block of features copied out of the base representation and
roughed up by 5% of each feature's own standard deviation. A working admission
rule takes ZERO features from either. The counts print on the delta line and
are stored per fold, so a win bought by admitting noise cannot be read without
seeing that it was.

WHERE THE AUGMENTATION ARMS BELONG (user decision, 2026-09-10). Building an
augmented feature set is precompute work, not experiment work: every base
representation, capped and uncapped, from the curated donor pool and from the
full one — 10 representations x 2 cap settings x 2 donor pools x 5 folds = 200
builds per dataset — made alongside the hybrid precompute, and read from the
store by whatever wants them. tests/precompute_hybrids.py does not emit those
arms today: it builds conditions on the 10-representation pool only. Adding
them is a change to that file, not to this one, so it is not made here. Until
it lands, this file builds its own augmented features per fold, which is the
duplicate work that decision removes.

Conditions are skipped, not faked, when their pool is empty: on datasets whose
survey found no augmentation rep helps, `augmentation` is legitimately [] and
B/D do not run. Since 2026-10-05, C (and every other extra-feature arm) is
also skipped wherever DECISIONS.md Step 8 cut augmentation, which is 41 of 43
datasets; see the ALIGNED block below.

ALIGNED 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4; selection rule (B) of
2.3 "Still open": every representation plus the extra-feature conditions,
fixed-recipe hybrids, the dataset's decided models, per fold).
  - Models: only decided_models(ds, 'general'). The old fallback to rf,
    xgboost and lightgbm is gone; a caller naming models is narrowed to the
    decided two, and naming none of them raises.
  - Extra-feature arms (B, C, D, Rc, Rb and H1) run only where DECISIONS.md
    Step 8 keeps augmentation: the distance search of ginkgo_colloidal and
    ginkgo_hic, condition 'uncapped' (2.3 item 4). Everywhere else they are
    skipped with a printed reason, and A_baseline and H0 still run. 'capped'
    is cut on every dataset, so it is dropped with a printed line.
  - H1 is off on every dataset for now: its recipe comes from the decided
    general grid, which keeps no augmentation anywhere, and Step 8 keeps
    augmentation on the distance search only. The skip reason is printed and
    stored as h1_skipped_reason until the owner decides.
  - H0/H1 recipe: each part is taken from the dataset's decided general grid.
    The file's own value is kept where the grid has it; where it does not, the
    replacement is printed and stored on every row (_hybrid_ref_recipe). The
    budgets are the shipped 500/1000/2000 features that the grid keeps, or
    the grid's whole list where it keeps none of them.
  - Outer folds: kirby.find_best_hybrid.nested_cv_outer_folds with
    config['n_splits'], drawn once and shared by every arm. One fold works:
    it is a single held-out split, and every spread then reads None.
  - Feature scaling: kirby.scaling.FeatureScaler (0/1 features unscaled,
    2.3 item 17). Decoy features are float32 like every representation
    (2.3 item 18).
  - Every row carries a per-fold list (fold, rows, held-out score, recipe)
    and selection_scope 'fold_train_rows'. RESULT_VERSION is new, so a file
    written before this cannot be mistaken for one written after.

Usage:
    python c21_single_rep_augmentation.py --dataset herg_ki --mode intermediary
    python c21_single_rep_augmentation.py --dataset herg_ki --base-reps mhg_gnn \
        --strategies none,all,capped,uncapped
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, get_domain_metric,
    build_fold_hybrid, plant_decoys, decoy_counts, as_rep_dtype,
    make_noise_donor, make_echo_donor, NOISE_DONOR, ECHO_DONOR)
from experiments.decided_grid import (
    decided_search, decided_models, decided_aug_conditions, is_decided)
from kirby.find_best_hybrid import (
    Blueprint, _predict_for_metric, nested_cv_outer_folds)
from kirby.scaling import FeatureScaler
from experiments.complexity.complexity_helpers import DATASET_REPS

# Stamped on the result file. New 2026-10-05 with the per-fold alignment, so a
# file written by the earlier code (five models, fixed weighted_global recipe,
# capped arms on every dataset) is told apart from one written after it.
RESULT_VERSION = 'c21_per_fold_2026-10-05'

# Outer-fold seed, the same one nested_cv_outer_folds is given by the per-fold
# searches of every other experiment, so c21's folds are their folds.
OUTER_FOLD_SEED = 42

# Floor on the denominator of relative_improvement (= delta / |base_score|).
# On antibody tasks the base Spearman rho routinely lands near zero, and an
# unfloored |base_score| manufactures absurd relative improvements (e.g. a
# base rho of 0.004 turned a modest +0.12 delta into a bogus +27.8). Below
# this magnitude the base score is not a meaningful signal, so we clamp the
# denominator to keep the ratio finite and interpretable. Base scores at or
# above the floor are unaffected.
REL_IMPROVEMENT_DENOM_FLOOR = 0.1

# Features allowed from EACH donor in conditions C and D: every other top_tier
# rep is capped to its top DEFAULT_TOP_N features per fold before pooling
# (_build_conditions: {k: top_n for k in other_top}, the per_source budget).
# Corrected 2026-10-05: this comment used to say it was the number of donor
# REPRESENTATIONS offered. It is not: every other top_tier rep is offered.
# It was 5 until 2026-09-12.
DEFAULT_TOP_N = 9

# Downstream-model seeds averaged within each fold before any delta is taken.
# Adding columns changes which features a stochastic tree model sees, which
# moves the score by itself: on tresanco, permuting the column order of an
# otherwise IDENTICAL 48-column block moved a fold Spearman across a 0.032
# range — larger than any effect this experiment is trying to detect. One seed
# is therefore not a measurement.
DEFAULT_N_SEEDS = 3

# Fixed recipe for the multi-rep reference arms (H0/H1). NOT searched: the
# previous E_* arms picked a blueprint by argmax over the same folds they were
# then scored on, which gave the hybrid reference a winner's curse the
# fixed-config single-rep arms do not have, and made the two incomparable.
# 'weighted_global' pools features across reps and ranks once globally, so a
# large near-tied pool does not shatter the budget into per-rep dust.
HYBRID_REF_ALLOCATION = 'weighted_global'
HYBRID_REF_IMPORTANCE = 'lightgbm_gain'
# Pre-registered headline budget, fixed BEFORE seeing any result so that
# quoting it is not a selection.
HYBRID_REF_HEADLINE_BUDGET = 1000
HYBRID_REF_BUDGETS_SHIPPED = (500, 1000, 2000)

# ── LOCAL CONDITION CUTS ────────────────────────────────────────────────────
# What follows cuts THIS FILE'S OWN condition lists, and nothing else. No row
# is ever dropped: all 672 flab_thermo_tresanco antibodies, all 2,158 caco2
# molecules and so on are in every fold of every arm below, before and after.
#
# WHY. c21 has never once run to completion on the laptop grid. Its last
# attempt on flab_thermo_tresanco, the smallest dataset at 672 antibodies,
# died at 05:10 on 2026-09-25 after 4022 s having finished four arms of the
# hundred and ten it had queued: ten base representations x eleven arms each.
#
# WHAT ONE ARM COSTS, profiled here 2026-09-25 on flab_thermo_tresanco,
# saprot as the base, random forest, one of the three folds. One
# build_fold_hybrid call took 112 s and split four ways:
#     33 s  scoring the representation so performance_weighted can weight it
#     22 s  lightgbm_gain importance over the base representation's features
#     28 s  the out-of-fold residuals the candidate features are ranked against
#     25 s  the five block sizes the admission rule scores, 0/5/10/20/40
# plus the three downstream seeds fitted on the finished matrix. The first of
# those four is removed outright below -- one representation cannot be
# out-weighted by itself -- and the same call then took 74.5 s against 161.3 s,
# with an identical matrix out of both. The other three are per arm and there
# is no way to reach them from this file: build_fold_hybrid forwards no
# aug_precomputed_ranking, so the residuals are recomputed for every arm of the
# same base representation and fold even though hybrid.py says in its own
# comment that they cannot differ.
#
# The machine was running eight other experiments throughout, so every second
# above is a busy-laptop second, not a clean one.
#
# HOW THE KNOBS BEHAVE. Each is read from the environment. Unset, and with
# KIRBY_LOCAL_GRID unset too, every one of them is the value the cluster has
# always run, so a server run is byte-for-byte unchanged. Under
# KIRBY_LOCAL_GRID the local default applies, and an explicit KIRBY_C21_* var
# beats both. Whatever ends up in force is printed at the top of the run under
# "local condition cuts", so no result exists without its cuts beside it.


def _local_grid():
    """True when the laptop grid in env.sh is active (KIRBY_LOCAL_GRID)."""
    return bool(os.environ.get('KIRBY_LOCAL_GRID'))


def _env_list(name, shipped, local=None, cast=str):
    """One condition list: explicit environment variable, else the local
    default when the laptop grid is on, else the shipped value."""
    raw = os.environ.get(name)
    if raw:
        picked = tuple(cast(x.strip()) for x in raw.split(',') if x.strip())
        if picked:
            return picked, f'{name}={raw}'
    if local is not None and _local_grid():
        return tuple(local), 'KIRBY_LOCAL_GRID'
    return tuple(shipped), 'shipped'


# Budgets for the H0/H1 multi-representation reference arms. Each budget is a
# separate fixed arm costing one build per model per fold, and an H build
# ranks lightgbm_gain over the WHOLE pool -- five representations at 1000
# features each here, not one. Measured on flab_thermo_tresanco 2026-09-25,
# budget 1000, random forest, one fold: 250.8 s for H0 and 276.4 s for H1,
# against 97 s for a whole single-representation arm on the same fold. Three
# budgets is three times that.
#
# Local runs keep the single pre-registered headline budget of 1000. What is
# lost is the shape of the H curve across budgets, which nothing in this
# file's summary claims.
HYBRID_REF_BUDGETS, HYBRID_REF_BUDGETS_SOURCE = _env_list(
    'KIRBY_C21_HYBRID_BUDGETS', HYBRID_REF_BUDGETS_SHIPPED,
    local=(HYBRID_REF_HEADLINE_BUDGET,), cast=int)

# The five ways of deciding which extra features to keep. This used to be the
# single hardcoded string 'uncapped', so c21 answered "where do extras go"
# for one admission rule only. The names are exactly what create_hybrid
# dispatches on; an unrecognised one raises there and lists the valid five.
#
#   none            no extras at all -- the control
#   all             every donor's top augmentation_budget features, no test
#   capped          extras must DISPLACE base features; total stays in budget
#   uncapped        extras are appended OUTSIDE the budget
#
# 'greedy_ablation' was CUT on 2026-08-26 -- "No greedy at all" -- and is not in
# this list. The engine still dispatches the name so old recipes do not crash,
# and naming it explicitly here still works so the cut can be checked.
ALL_STRATEGIES = ('none', 'all', 'capped', 'uncapped')
CUT_STRATEGIES = ('greedy_ablation',)   # dispatchable, never selected
# Both admission rules by default (owner's decision, 2026-09-10): the showcase
# is every base representation, capped and uncapped, with donors from the
# curated augmentation pool and from the other base representations. The
# curated-versus-full axis is already the B / C / D conditions; this line is the
# capped-versus-uncapped half, and it ran uncapped only until 2026-09-12.
DEFAULT_STRATEGIES_SHIPPED = ('capped', 'uncapped')
# Every admission method doubles the arm count, and an arm is a build per model
# per fold. Local runs keep 'capped' alone: it is the harder of the two, since
# extras have to displace base features rather than being appended outside the
# budget, so a win under it is the one worth seeing first. What is lost is the
# capped-versus-uncapped comparison itself, which is the 2026-09-12 half of the
# showcase and has to come back from a server run.
DEFAULT_STRATEGIES, DEFAULT_STRATEGIES_SOURCE = _env_list(
    'KIRBY_C21_STRATEGIES', DEFAULT_STRATEGIES_SHIPPED, local=('capped',))

# Which of the non-baseline conditions are built. A_baseline is not optional --
# every delta in the file is measured against it.
#   B   base + the curated augmentation representations, uncapped
#   C   base + the top-N features of each other base representation
#   D   B and C together
#   Rc  the matched-width random-column control for C
#   Rb  the matched-width random-column control for B
# MEASURED, flab_thermo_tresanco 2026-09-25, one base representation, one
# admission method, three folds, the two models rf and svm, the whole run timed
# end to end with eight other experiments sharing the laptop:
#     106 s  python start-up, loading 672 antibodies and the search precompute
#     201 s  A_baseline, both models
#     315 s  B under rf          213 s  B under svm
#     704 s  C under rf          ~500 s C under svm (killed before it landed)
#     750 s  H0, the hybrid reference, rf only, at budget 1000
# C costs 2.2x what B does because its donors are the other four base
# representations at a thousand features each, against the curated pool's
# forty-eight, and the candidate ranking scores every one of those columns.
# A + B + C + H0 comes to about 2790 s, which is 46 minutes; the cap is 40.
#
# So the local default is B alone, with H0. That runs in about 1585 s, 26
# minutes, measured from the pieces above.
#
# ⛔ WHAT IS LOST, and these are the cuts to argue with first.
#   C   the other base representations as donors -- "can other top reps donate
#       a few good features". Under rf it was the LARGER effect of the two:
#       delta +0.0223 against B's +0.0080, and it took nothing from the
#       pure-noise donor. It is also the only condition that runs at all on a
#       dataset whose survey found no augmentation representation helps.
#   Rc, Rb  the matched-width random-column controls. Without them a positive
#       delta in B or C cannot be told apart from the model having been handed
#       more features, so nothing the local grid produces is readable as a
#       selection effect.
#   D   the union of B and C.
# Both alternatives fit the cap if the other half is given up, and both are one
# environment variable:
#     KIRBY_C21_CONDITIONS=B,C       with KIRBY_C21_HYBRID_ARMS=  -> ~34 min
#     KIRBY_C21_CONDITIONS=B,C,Rc,Rb                              -> needs a
#       longer cap than forty minutes on this machine.
ALL_CONDITION_KEYS = ('B', 'C', 'D', 'Rc', 'Rb')
CONDITION_KEYS, CONDITION_KEYS_SOURCE = _env_list(
    'KIRBY_C21_CONDITIONS', ALL_CONDITION_KEYS, local=('B',))

# How many base representations the survey loop covers, as a count taken off
# the front of the pool in DATASET_REPS order, or a comma-separated list of
# names. The whole pool is the shipped behaviour and the docstring already
# calls it "a 10x more expensive survey". Local runs take one: the base-
# representation loop is a pure multiplier on everything above, so one base is
# one tenth of the run. What is lost is the survey -- whether the answer holds
# for every base representation or only the first. --base-reps still overrides
# this from the command line.
C21_BASE_REP_LIMIT = os.environ.get('KIRBY_C21_BASE_REPS') or (
    '1' if _local_grid() else '')

# The multi-representation reference arms, and how many downstream models they
# are scored under.
#
# ⛔ THESE TWO ARE THE EXPENSIVE ONES AND THE CUT NEEDS A DECISION. Measured on
# flab_thermo_tresanco 2026-09-25, five representations at 1000 features each,
# budget 1000, random forest, one fold: ONE H build took 250.8 s, against 97 s
# for a whole single-representation arm on the same fold. H0 over two models
# and three folds is 1506 s on its own, H1 more, and the two together are
# roughly fifty minutes before a single single-representation arm has run. The
# difference is the pool: an H build ranks lightgbm_gain over five thousand
# features and scores five representations against each other, a
# single-representation build ranks one thousand and scores nothing.
#
# The local default keeps H0, the hybrid with no extras, under the first model
# only, and drops H1. What that costs is the half of the placement question
# that asks whether extras do more ON the hybrid -- H0 still gives the
# single-representation arms a hybrid to be compared against, but nothing local
# measures the hybrid WITH extras. There is no arrangement of this file that
# fits both H arms and a readable single-representation set into forty minutes
# on this laptop; which half to keep is the user's call, and both are one
# environment variable away:
#     KIRBY_C21_HYBRID_ARMS=h0,h1   KIRBY_C21_HYBRID_MODELS=2
HYBRID_ARMS, HYBRID_ARMS_SOURCE = _env_list(
    'KIRBY_C21_HYBRID_ARMS', ('h0', 'h1'), local=('h0',))
HYBRID_MODEL_LIMIT = os.environ.get('KIRBY_C21_HYBRID_MODELS') or (
    '1' if _local_grid() else '')

# Features copied out of the base representation to build the near-copy decoy,
# and how hard they are roughed up (as a multiple of each feature's own
# standard deviation). Matches scratchpad/aug_decoy_check.py, which produced
# the logd and herg_ki decoy counts recorded in AUGMENTATION.md.
DECOY_ECHO_COLS = 64
DECOY_ECHO_JITTER = 0.05


# ── THE DECIDED GRID, applied to this file's fixed recipes (2026-10-05) ──────
# c21 runs no search, but rule R3 says the dataset's decided budgets,
# importance methods, allocation methods, internal evaluators and
# representation counts apply to every hybrid build. The file's own value is
# kept wherever the dataset's general grid has it. Where it does not, the
# preference list below is walked, then the grid's first member is taken. The
# choice is printed and stored on every row, so no number is read without it.
#
# What this changes, read from decided_grid.json 2026-10-05 (43 datasets):
#   allocation  weighted_global is in no dataset's grid. wglobal_power2 (the
#               same global ranking across representations, weights squared)
#               is in 25; the other 18 get their first, performance_weighted.
#   importance  lightgbm_gain is cut on buchwald_hartwig, cdr_flexibility,
#               ginkgo_pr_cho and ginkgo_titer; they get deeplift.
#   evaluator   lightgbm is cut on abagym_VEGF and ginkgo_titer; they get match.
#   rep count   the whole pool everywhere except buchwald_hartwig (2 only).
#   budgets     see _hybrid_ref_budgets.
# ⛔ These replacements are Claude's, not the owner's: listed as an open
# question in the handoff for her to confirm or change.
HYBRID_REF_PREFERENCE = {
    'allocation_methods': (HYBRID_REF_ALLOCATION, 'wglobal_power2'),
    'importance_methods': (HYBRID_REF_IMPORTANCE,),
    'internal_evaluators': ('lightgbm',),
}
# The single-representation arms' recipe (no allocation to make: one
# representation takes the whole budget). importance is NOT inert here: at
# budget = the base width, create_hybrid keeps only base features whose
# importance is above zero (tracker C21a), so it follows the grid too.
SINGLE_REP_PREFERENCE = {
    'allocation_methods': ('performance_weighted',),
    'importance_methods': ('lightgbm_gain',),
    'internal_evaluators': ('match',),
}


def _pick_decided(grid, axis, preferred):
    """(value, how) for one recipe axis: the first preferred value the grid
    keeps, else the grid's first member. `how` is 'file' when the file's own
    value survived and 'decided_grid' when it was replaced."""
    decided = list(grid[axis])
    for i, p in enumerate(preferred):
        if p in decided:
            return p, ('file' if i == 0 else 'decided_grid')
    return decided[0], 'decided_grid'


def _hybrid_ref_budgets(grid, requested):
    """The H0/H1 budgets in features: the requested ones the grid keeps, or
    the grid's whole list when it keeps none (abagym_Ang2, cdr_flexibility and
    suzuki keep none of 500/1000/2000). Returns (budgets, note or None)."""
    decided = [int(b) for b in grid['budgets']]
    kept = [int(b) for b in requested if int(b) in decided]
    dropped = [int(b) for b in requested if int(b) not in decided]
    if kept:
        note = (f'budgets {dropped} features not in the decided grid '
                f'{decided}; dropped' if dropped else None)
        return kept, note
    return decided, (f'none of the budgets {list(requested)} features is in '
                     f'the decided grid; using its whole list {decided}')


def _hybrid_ref_recipe(dataset_name, requested_budgets):
    """The fixed H0/H1 recipe for this dataset, every part from its decided
    general grid. Returns a dict that is printed and stored on every H row."""
    grid = decided_search(dataset_name, 'general')
    out, replaced = {}, {}
    for axis, pref in HYBRID_REF_PREFERENCE.items():
        val, how = _pick_decided(grid, axis, pref)
        out[axis] = val
        if how != 'file':
            replaced[axis] = {'file_value': pref[0], 'used': val}
    # rep count: the whole pool (None) when the grid keeps it, else the
    # largest count it keeps.
    rcs = list(grid['rep_counts'])
    if None in rcs:
        out['rep_count'] = None
    else:
        out['rep_count'] = int(max(r for r in rcs if r is not None))
        replaced['rep_count'] = {'file_value': 'whole pool',
                                 'used': out['rep_count']}
    budgets, bnote = _hybrid_ref_budgets(grid, requested_budgets)
    out['budgets'] = budgets
    if bnote:
        replaced['budgets'] = {'file_value': list(requested_budgets),
                               'used': budgets, 'note': bnote}
    out['replaced_from_file'] = replaced
    out['grid'] = {k: grid[k] for k in ('allocation_methods',
                                        'importance_methods',
                                        'internal_evaluators', 'rep_counts',
                                        'budgets')}
    return out


def _single_rep_recipe(dataset_name):
    """The single-representation arms' importance and evaluator, from the
    decided general grid. Returns {axis: value, 'replaced_from_file': {...}}."""
    grid = decided_search(dataset_name, 'general')
    out, replaced = {}, {}
    for axis, pref in SINGLE_REP_PREFERENCE.items():
        val, how = _pick_decided(grid, axis, pref)
        out[axis] = val
        if how != 'file':
            replaced[axis] = {'file_value': pref[0], 'used': val}
    out['replaced_from_file'] = replaced
    return out


def _decided_strategies(dataset_name, requested):
    """The admission methods this dataset may run, and why the rest are not.

    Handoff 2.3 item 4 (owner, 2026-10-05): an experiment comparing with and
    without added features runs that comparison only where DECISIONS.md Step 8
    keeps augmentation -- the distance search of ginkgo_colloidal and
    ginkgo_hic, condition 'uncapped'. Returns (kept, reason or None)."""
    allowed = []
    if is_decided(dataset_name, 'general_distance'):
        allowed = [c for c in decided_aug_conditions(dataset_name,
                                                     'general_distance')
                   if c != 'none']
    kept = [s for s in requested if s in allowed]
    dropped = [s for s in requested if s not in allowed]
    if not allowed:
        return [], (f'DECISIONS.md Step 8 cut augmentation on {dataset_name} '
                    f'(general and distance search), so the extra-feature '
                    f'arms B, C, D, Rc, Rb and H1 are skipped (handoff 2.3 '
                    f'item 4). A_baseline and H0 still run.')
    if dropped:
        return kept, (f'admission methods {dropped} not kept by DECISIONS.md '
                      f'Step 8 on {dataset_name} (kept: {allowed}); dropped.'
                      + ('' if kept else ' Nothing left, so the extra-feature '
                         'arms are skipped.'))
    return kept, None


def _spread(scores):
    """Standard deviation over folds, or None when fewer than two folds
    scored: one held-out split has no spread (handoff 2.4 step 14)."""
    vals = [s for s in scores if s is not None and np.isfinite(s)]
    if len(vals) < 2:
        return None
    return float(np.std(vals))


def _fold_rank_scores(train_reps, y_train, groups_train, evaluator,
                      model_factory, metric_name, metric_fn, lower_is_better):
    """One score per representation on this fold's TRAINING rows, higher is
    better, for a recipe whose representation count is below the pool.

    Same carve the engine uses for its allocation scores
    (kirby.hybrid.internal_holdout_split, 20% of the training rows by group)
    and the same evaluator call (_evaluate_allocation). With
    evaluator 'match' the downstream model itself is fitted. The held-out
    rows of the outer fold are never seen. Used only where the decided grid
    has no whole-pool count (buchwald_hartwig, count 2).
    """
    from kirby.hybrid import (internal_holdout_split, _evaluate_allocation,
                              RANKING_METRICS)
    tr_i, va_i = internal_holdout_split(len(y_train), y_train, groups_train,
                                        0.2)
    scores = {}
    for name, X in train_reps.items():
        try:
            if evaluator == 'match':
                s = _evaluate_allocation(
                    X[tr_i], X[va_i], y_train[tr_i], y_train[va_i],
                    model=model_factory(), metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
            else:
                graded = {}
                _, y_pred = _evaluate_allocation(
                    X[tr_i], X[va_i], y_train[tr_i], y_train[va_i],
                    internal_evaluator=evaluator, return_predictions=True,
                    graded_out=graded)
                y_use = (graded.get('y') if (metric_name in RANKING_METRICS
                                             and graded.get('y') is not None)
                         else y_pred)
                raw = float(metric_fn(y_train[va_i], y_use))
                s = -raw if lower_is_better else raw
            # A score that could not be computed ranks last, not first: NaN
            # would sort unpredictably in build_fold_hybrid's ranking.
            scores[name] = float(s) if np.isfinite(s) else float('-inf')
        except Exception as e:
            print(f"      ranking score for {name} failed - {e}")
            scores[name] = float('-inf')
    return scores


def _seeded(factory, seed):
    """Return a fresh model with its random_state pinned to `seed` when it has one."""
    m = factory()
    try:
        if 'random_state' in m.get_params():
            m.set_params(random_state=seed)
    except Exception:
        pass
    return m


def _fit_score_seeds(Xtr, ytr, Xva, yva, factory, metric_fn, n_seeds):
    """Mean score over n_seeds model seeds on ONE fold.

    The averaging happens before any delta is taken, so model stochasticity
    does not masquerade as an effect of the added columns.
    """
    vals = []
    for s in range(n_seeds):
        m = _seeded(factory, 42 + s)
        m.fit(Xtr, ytr)
        # _predict_for_metric, not .predict: on an AUROC dataset the hard class
        # labels give a degraded AUROC, and every delta in this file is a
        # difference of two of these scores.
        vals.append(float(metric_fn(yva, _predict_for_metric(m, Xva, metric_fn))))
    return float(np.mean(vals)), vals


def _baseline_fold_scores(X_base, labels, folds,
                          model_factory, metric_fn, n_seeds):
    """Fold scores for the base representation ALONE, no donors anywhere.

    Split out of the arm loop so it is computed once per (base
    representation, model) instead of once per arm. It reads no condition, no
    admission method and no donor pool, and the model seeds are fixed at
    42..42+n_seeds-1, so every arm that shared a base representation and a
    model was refitting the identical three folds and getting the identical
    numbers. NOTHING in any result changes: the same list is now stored
    against each arm instead of recomputed for it.
    """
    out = []
    # folds: the outer splits drawn once in run() (2026-10-05), shared by
    # every arm. FeatureScaler leaves 0/1 features unscaled (2.3 item 17).
    for train_idx, val_idx in folds:
        scaler = FeatureScaler()
        Xtr = scaler.fit_transform(X_base[train_idx])
        Xva = scaler.transform(X_base[val_idx])
        mean_b, _ = _fit_score_seeds(Xtr, labels[train_idx], Xva,
                                     labels[val_idx], model_factory,
                                     metric_fn, n_seeds)
        out.append(mean_b)
    return out


def _effective_donor_width(donors, per_source_budget):
    """How many features the real donors actually offer, in total.

    Sum over donors of min(that donor's per-source cap, its full width). A
    donor absent from the cap dict contributes every one of its features.
    """
    caps = per_source_budget or {}
    return int(sum(min(int(caps.get(name, v.shape[1])), int(v.shape[1]))
                   for name, v in donors.items()))


def _plant_capped_decoys(donors, base_X, rng, per_source_budget,
                         echo_cols=DECOY_ECHO_COLS,
                         echo_jitter=DECOY_ECHO_JITTER):
    """Plant both decoy donors, the noise block sized to what the real donors
    actually offer under this condition's per-donor cap.

    `_shared.plant_decoys` takes no width argument: it sets the noise donor's
    width to the sum of the donors' FULL widths. In C_topreps and D_both each
    other pool representation is capped to top_n features by
    aug_per_source_budget, so the real donors offer top_n features each
    (DEFAULT_TOP_N, 9 since 2026-09-12; this said 5 until 2026-10-05) while the
    noise block arrived thousands of features wide and the decoy count it
    produced was a count against a donor no real one was competing with.

    Effective width here is the sum over donors of min(that donor's cap, its
    full width). Donors with no cap contribute all of their features, which is
    what B_aug and the R_* controls do.

    Both decoys are cast to float32 (2026-10-05, handoff 2.3 item 18): the
    shared builders return float64, and one float64 block turned the whole
    hybrid matrix float64.
    """
    out = dict(donors)
    n_real = _effective_donor_width(donors, per_source_budget)
    if n_real > 0:
        out[NOISE_DONOR] = as_rep_dtype(
            make_noise_donor(base_X.shape[0], n_real, rng))
    echo = make_echo_donor(base_X, echo_cols, echo_jitter, rng)
    if echo is not None:
        out[ECHO_DONOR] = as_rep_dtype(echo)
    return out


def _build_conditions(base_name, top_tier_reps, aug_reps, top_n, rng=None,
                      with_controls=True, condition_keys=None):
    """Return {condition_name: (aug_pool, per_source_budget)}.

    per_source_budget is a dict so the wide top-tier reps get capped to top_n
    while the small engineered augmentation reps stay uncapped — a source
    absent from the dict contributes all of its columns.

    Two matched-width CONTROLS are added when with_controls is set. Adding any
    columns changes model capacity, so without them a win in B/C/D cannot be
    told apart from "the model simply got more features":

      R_rand_n{top_n}  offers top_n columns picked UNIFORMLY AT RANDOM from each
                       other base rep, instead of its top_n by importance. Same
                       width as C, same donors, same admission path — so C minus
                       R isolates the SELECTION, not the capacity.
      R_randaug        offers as many randomly-chosen columns (drawn from the
                       other base reps) as the curated augmentation pool has.
                       Same width as B, so B minus R_randaug isolates whether
                       the curated blocks carry something the base pool does not.

    Random columns carry no label information, so drawing them once outside the
    fold loop leaks nothing.
    """
    other_top = {k: v for k, v in top_tier_reps.items() if k != base_name}
    aug_only = {k: v for k, v in aug_reps.items() if k != base_name}
    # Which non-baseline conditions to build. None means every one of them,
    # which is what a caller that knows nothing about the local knobs gets.
    wanted = set(condition_keys if condition_keys is not None
                 else ALL_CONDITION_KEYS)

    conditions = {'A_baseline': (None, None)}
    if aug_only and 'B' in wanted:
        conditions['B_aug'] = (aug_only, None)
    if other_top and 'C' in wanted:
        conditions[f'C_topreps_n{top_n}'] = (
            other_top, {k: top_n for k in other_top})
    if aug_only and other_top and 'D' in wanted:
        both = {**aug_only, **other_top}
        conditions[f'D_both_n{top_n}'] = (
            both, {k: top_n for k in other_top})

    if not with_controls or not other_top:
        return conditions
    rng = rng if rng is not None else np.random.RandomState(0)

    # Control for C: same donors, same top_n-per-donor width, random columns.
    rand_pool = {}
    for k, v in other_top.items():
        n_take = min(top_n, v.shape[1])
        cols = rng.choice(v.shape[1], size=n_take, replace=False)
        rand_pool[f'{k}__rand'] = v[:, np.sort(cols)]
    if 'Rc' in wanted:
        conditions[f'R_rand_n{top_n}'] = (rand_pool, None)

    # Control for B: as many random columns as the curated pool is wide,
    # spread evenly over the other base reps.
    if aug_only and 'Rb' in wanted:
        aug_width = sum(v.shape[1] for v in aug_only.values())
        per_donor = max(1, int(np.ceil(aug_width / len(other_top))))
        randaug, taken = {}, 0
        for k, v in other_top.items():
            if taken >= aug_width:
                break
            n_take = min(per_donor, v.shape[1], aug_width - taken)
            cols = rng.choice(v.shape[1], size=n_take, replace=False)
            randaug[f'{k}__randaug'] = v[:, np.sort(cols)]
            taken += n_take
        if randaug:
            conditions['R_randaug'] = (randaug, None)
    return conditions


def _score_hybrid_condition(cond_name, top_tier_reps, aug_dict, aug_condition,
                            labels, groups, models, metric_fn, lower_is_better,
                            folds, recipe, task_type, metric_name,
                            n_seeds=DEFAULT_N_SEEDS,
                            candidate_ranking='residual_correlation',
                            augmentation_budget=20, max_candidates=50,
                            with_decoys=True):
    """Score a MULTI-rep hybrid on the same folds as the single-rep arms.

    FIXED CONFIGURATION, NO SEARCH. The blueprint is constructed, not fetched.

    The previous implementation pulled a blueprint from get_precomputed_search
    and took `search.best_by_model` — an argmax over the same folds the arm was
    then scored on. That gave the hybrid reference a winner's curse the
    fixed-config single-rep arms (A/B/C/D) do not carry, so the two could not be
    compared, which is the whole point of having the reference. It also required
    a precomputed search that does not exist locally; without one it silently
    fell back to a full find_best_hybrid over every budget.

    Here every budget in recipe['budgets'] is a separate fixed arm and all of
    them are reported. Nothing is selected, so each fold mean is unbiased.

    2026-10-05: `recipe` is _hybrid_ref_recipe(dataset): allocation,
    importance, evaluator, representation count and budgets from the dataset's
    decided grid. `folds` are the outer splits drawn once in run(). Every
    build is fitted on the fold's training rows only; where the recipe's
    representation count is below the pool, the representations are ranked
    by scores computed on those training rows (_fold_rank_scores), never by
    whole-dataset scores (dirty-run bug 1).
    """
    rows = []
    rep_count = recipe.get('rep_count')
    evaluator = recipe['internal_evaluators']
    needs_rank = rep_count is not None and int(rep_count) < len(top_tier_reps)
    # {(fold, model or None): scores}. The scores do not depend on the budget,
    # so they are computed once per fold (per model when the evaluator is the
    # downstream model itself) and reused by every budget.
    rank_cache = {}

    for budget in recipe['budgets']:
        bp = Blueprint(
            reps=sorted(top_tier_reps),
            allocation_method=recipe['allocation_methods'],
            importance_method=recipe['importance_methods'],
            internal_evaluator=evaluator,
            budget=budget,
            step_size=5,
            augmentation_strategy=aug_condition,
            aug_per_source_budget=None,
            # On the recipe as of 2026-08-26. Without them the per-fold rebuild
            # fell back to create_hybrid's own defaults regardless of what this
            # arm was configured with.
            candidate_ranking=candidate_ranking,
            augmentation_budget=augmentation_budget,
            aug_max_candidates=max_candidates,
            metric=metric_name,
            task_type=task_type,
            lower_is_better=lower_is_better,
            rep_count=rep_count)
        arm = f'{cond_name}_b{budget}'

        for model_name, model_factory in models.items():
            scores, dims, kept = [], [], set()
            width_per_fold = []
            noise_taken, echo_taken = [], []
            per_fold = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                y_train, y_val = labels[train_idx], labels[val_idx]
                width_per_fold.append(None)
                tr = {k: v[train_idx] for k, v in top_tier_reps.items()}
                va = {k: v[val_idx] for k, v in top_tier_reps.items()}
                g_tr = groups[train_idx] if groups is not None else None
                fold_rec = {'fold': int(fold_idx),
                            'n_train_rows': int(len(train_idx)),
                            'n_heldout_rows': int(len(val_idx)),
                            'heldout_score': None,
                            'training_rows_score': None,
                            'width_built': None}

                rank_scores = None
                if needs_rank:
                    _rk = (fold_idx, model_name if evaluator == 'match' else None)
                    if _rk not in rank_cache:
                        rank_cache[_rk] = _fold_rank_scores(
                            tr, y_train, g_tr, evaluator, model_factory,
                            metric_name, metric_fn, lower_is_better)
                    rank_scores = rank_cache[_rk]
                    _ranked = sorted(rank_scores,
                                     key=lambda r: (-rank_scores[r], r))
                    fold_rec['reps_used'] = sorted(_ranked[:int(rep_count)])
                    fold_rec['rep_rank_scores_train_rows'] = dict(rank_scores)
                else:
                    fold_rec['reps_used'] = sorted(top_tier_reps)

                pool = aug_dict
                if aug_dict and with_decoys:
                    # There is no single base representation here to copy
                    # features out of -- the base is a hybrid drawn from ten of
                    # them, and which features it keeps is not known until
                    # after this call. So the near-copy is built from the
                    # CANDIDATE pool: a few features from each pool
                    # representation. A copied feature may therefore not be in
                    # the finished hybrid at all, which makes this a weaker
                    # decoy here than on the single-representation arms. Stated
                    # rather than left as a silent difference.
                    _r = np.random.RandomState(2000 + fold_idx)
                    _slices = []
                    _per = max(1, int(np.ceil(DECOY_ECHO_COLS
                                              / max(1, len(top_tier_reps)))))
                    for _v in top_tier_reps.values():
                        _take = _r.choice(_v.shape[1],
                                          size=min(_per, _v.shape[1]),
                                          replace=False)
                        _slices.append(_v[:, np.sort(_take)])
                    pool = plant_decoys(aug_dict, np.hstack(_slices), _r,
                                        echo_cols=DECOY_ECHO_COLS,
                                        echo_jitter=DECOY_ECHO_JITTER)
                    # float32 like every representation (2026-10-05, 2.3
                    # item 18); plant_decoys returns float64 decoys.
                    pool = {k: as_rep_dtype(v) for k, v in pool.items()}

                tr_aug = ({k: v[train_idx] for k, v in pool.items()}
                          if pool else None)
                va_aug = ({k: v[val_idx] for k, v in pool.items()}
                          if pool else None)
                try:
                    Xtr, Xva, finfo = build_fold_hybrid(
                        tr, va, y_train, bp,
                        augmentations_train=tr_aug, augmentations_val=va_aug,
                        groups=g_tr,
                        model=model_factory(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better,
                        # Only when the recipe uses fewer representations
                        # than the pool: this fold's training-row scores,
                        # which also set the budget split, as the search's
                        # own allocation scores do (dirty-run bug 1).
                        precomputed_alloc_scores=rank_scores)
                    # 0/1 features stay unscaled (2026-10-05, 2.3 item 17).
                    sc = FeatureScaler()
                    Xtr = sc.fit_transform(Xtr)
                    Xva = sc.transform(Xva)
                    mean_s, _ = _fit_score_seeds(Xtr, y_train, Xva, y_val,
                                                 model_factory, metric_fn, n_seeds)
                    scores.append(mean_s)
                    dims.append(int(Xtr.shape[1]))
                    width_per_fold[-1] = int(Xtr.shape[1])
                    _ai = finfo.get('augmentation_info') or {}
                    kept.update(_ai.keys())
                    _d = decoy_counts(_ai)
                    noise_taken.append(_d['noise'])
                    echo_taken.append(_d['echo'])
                    fold_rec.update(heldout_score=mean_s,
                                    width_built=int(Xtr.shape[1]),
                                    kept_augmentations=sorted(_ai.keys()),
                                    decoys=_d)
                except Exception as e:
                    print(f"    {arm}/{model_name}: fold failed - {e}")
                    scores.append(float('nan'))
                    fold_rec['error'] = f'{type(e).__name__}: {e}'
                per_fold.append(fold_rec)

            ok = not np.all(np.isnan(scores))
            headline = recipe.get('headline_budget')
            rows.append({
                'condition': arm,
                'condition_family': cond_name,
                'budget': budget,
                'headline': headline is not None and budget == headline,
                'model': model_name,
                'base_rep': f'HYBRID({len(top_tier_reps)} reps)',
                'base_tier': 'multi_rep_hybrid_fixed',
                'hybrid_mean': float(np.nanmean(scores)) if ok else float('nan'),
                # None with one scored fold: no spread (2026-10-05).
                'hybrid_std': _spread(scores) if ok else None,
                'hybrid_scores': scores,
                'mean_dim': float(np.mean(dims)) if dims else float('nan'),
                # 2026-10-05: features actually built per fold, same order as
                # hybrid_scores (None where the fold failed). H0's budgets can
                # build nearly the same width, which is why its rows barely
                # move; read the budget against this.
                'width_built_per_fold': width_per_fold,
                'kept_augmentations': sorted(kept),
                'aug_status': 'ok' if ok else 'all_folds_failed',
                'hybrid_reps': sorted(top_tier_reps),
                'aug_pool_reps': sorted(aug_dict) if aug_dict else [],
                'n_seeds': n_seeds,
                'augmentation_strategy': aug_condition,
                'candidate_ranking': candidate_ranking,
                'aug_max_candidates': max_candidates,
                'decoy_noise_per_fold': noise_taken,
                'decoy_echo_per_fold': echo_taken,
                'decoys_planted': bool(with_decoys and aug_dict),
                # The recipe in plain fields (2026-10-05, handoff 2.4 step
                # 11), and which parts the decided grid replaced.
                'hybrid_recipe': {
                    'allocation_method': recipe['allocation_methods'],
                    'importance_method': recipe['importance_methods'],
                    'internal_evaluator': evaluator,
                    'budget_features': int(budget),
                    'rep_count': rep_count,
                    'n_reps_in_pool': len(top_tier_reps),
                    'aug_condition': aug_condition,
                    'replaced_from_file': recipe.get('replaced_from_file', {}),
                },
                'selection': 'none: fixed recipe, nothing ranked',
                'selection_scope': 'fold_train_rows',
                'per_fold': per_fold,
                'blueprint': bp.to_dict(),
            })
            if ok:
                star = (' *headline' if headline is not None
                        and budget == headline else '')
                decoy = (f" decoys=noise {sum(noise_taken)}/echo {sum(echo_taken)}"
                         if with_decoys and aug_dict else "")
                print(f"    {arm}/{model_name}: {np.nanmean(scores):.4f} "
                      f"(dim~{np.mean(dims):.0f}, kept={sorted(kept)}){decoy}{star}")
    return rows


def run(data, config, output_dir, dataset_name, top_n=DEFAULT_TOP_N,
        base_reps=None, model_names=None, n_seeds=DEFAULT_N_SEEDS,
        hybrid_budgets=HYBRID_REF_BUDGETS, with_controls=True,
        skip_single_rep=False, strategies=DEFAULT_STRATEGIES,
        candidate_ranking='residual_correlation',
        augmentation_budget=20, max_candidates=50,
        with_decoys=True, **kwargs):
    """C21: Single-Rep Augmentation Test.

    base_reps: restrict the base-rep loop to these names. Default (None) is
        every rep in the pool, which is a survey: 10 bases x 2 models x 5
        conditions x folds. Pin it to the dataset's best rep to get the
        best-vs-best comparison at ~1/10th the cost.
    model_names: restrict the downstream models. Default (2026-10-05) is the
        dataset's two decided models, decided_models(ds, 'general'). Names
        outside them are dropped with a printed line; none left raises.
    n_seeds: downstream-model seeds averaged within each fold (see
        DEFAULT_N_SEEDS for why one is not enough).
    strategies: which admission methods to run, from ALL_STRATEGIES. Each one
        becomes its own arm on every non-baseline condition, named
        '<condition>__<method>'. Default is DEFAULT_STRATEGIES, ('capped',
        'uncapped') on the server. 2026-10-05: narrowed to what DECISIONS.md
        Step 8 keeps on this dataset (_decided_strategies), which is
        'uncapped' on ginkgo_colloidal and ginkgo_hic and nothing elsewhere.
    hybrid_budgets: the H0/H1 budgets asked for, in features. Narrowed to the
        decided grid by _hybrid_ref_recipe.
    candidate_ranking: 'residual_correlation' ranks a candidate feature by what
        the base representation could NOT explain; 'rf_importance' ranks it by
        how well it predicts on its own. Read only by 'capped' and 'uncapped'.
    augmentation_budget: features taken from EACH donor by 'all' and
        'greedy_ablation'. Not the same setting as the per-donor cap, which
        applies to 'capped' and 'uncapped'.
    max_candidates: how long the pooled ranked candidate list is allowed to be
        before a block size is chosen from it. 50 is the engine default and is
        what every run before 2026-08-26 used.
    with_decoys: plant a pure-noise donor and a jittered near-copy of the base
        representation into the same pool as the real donors, and count how many
        features each fold takes from them. A working rule takes zero of both.
        They add no conditions. Turn them off only to reproduce an older run.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C21: SINGLE-REP AUGMENTATION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)
    # Outer fold count from the one setting (handoff 2.4 step 14). 1 is a
    # single held-out split, not 1-fold cross-validation.
    n_folds = int(config['n_splits'])

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    # Fail on a bad method name here rather than on the first fold. create_hybrid
    # raises too, but only after the data is loaded and the base arm is scored.
    strategies = tuple(strategies)
    bad = [s for s in strategies
           if s not in ALL_STRATEGIES and s not in CUT_STRATEGIES]
    if bad:
        raise ValueError(
            f"Unknown augmentation strategy {bad}. "
            f"Choose from: {', '.join(ALL_STRATEGIES)}.")
    revived = [s for s in strategies if s in CUT_STRATEGIES]
    if revived:
        # Named explicitly, so it runs -- but it is not part of the study and
        # the log has to say so, or a later reader finds greedy results sitting
        # beside the others with nothing marking them as cut.
        print(f"  WARNING: {', '.join(revived)} is CUT (2026-08-26, "
              f"'No greedy at all'). It runs because you named it. It is not "
              f"part of the study and must not be reported as if it were.")
    if candidate_ranking not in ('residual_correlation', 'rf_importance'):
        raise ValueError(
            f"Unknown candidate_ranking {candidate_ranking!r}. Choose from: "
            f"'residual_correlation', 'rf_importance'.")
    # 2026-10-05: only the augmentation conditions DECISIONS.md Step 8 keeps
    # on this dataset (handoff 2.3 item 4). With none kept, every
    # extra-feature arm is skipped and the reason is printed and stored.
    strategies_requested = strategies
    _kept, aug_skip_reason = _decided_strategies(dataset_name,
                                                 strategies_requested)
    strategies = tuple(_kept)
    if aug_skip_reason:
        print(f"  Augmentation: {aug_skip_reason}")
    print(f"  Admission methods: {', '.join(strategies) or 'none'} "
          f"(asked {', '.join(strategies_requested)} "
          f"[{DEFAULT_STRATEGIES_SOURCE}])")
    # Every condition list this file cut, and what set it. A result produced
    # under a cut says so in its own log.
    print(f"  local condition cuts: conditions="
          f"{','.join(CONDITION_KEYS)} [{CONDITION_KEYS_SOURCE}]; "
          f"hybrid reference budgets="
          f"{','.join(str(b) for b in HYBRID_REF_BUDGETS)} "
          f"[{HYBRID_REF_BUDGETS_SOURCE}]; base representations="
          f"{C21_BASE_REP_LIMIT or 'the whole pool'}; reference arms="
          f"{','.join(HYBRID_ARMS)} [{HYBRID_ARMS_SOURCE}] under "
          f"{HYBRID_MODEL_LIMIT or 'every'} model")
    print(f"  Candidate ranking: {candidate_ranking}  "
          f"(list cut to top {max_candidates})")
    print(f"  Planted decoys: {'noise + near-copy' if with_decoys else 'none'}")

    # Models (2026-10-05, rule R1): only the dataset's two decided models,
    # decided_models(ds, 'general'). The old default of rf, xgboost and
    # lightgbm, and config['models'] (five models under the old run_all), are
    # gone. knn is never one of them; c21 runs no distance search.
    _decided_m = list(decided_models(dataset_name, 'general'))
    if model_names:
        _asked_m = list(model_names)
        model_names = [m for m in _asked_m if m in _decided_m]
        _dropped_m = [m for m in _asked_m if m not in _decided_m]
        if not model_names:
            raise ValueError(
                f"{dataset_name}: models {_asked_m} has nothing DECISIONS.md "
                f"keeps (decided: {_decided_m}).")
        if _dropped_m:
            print(f"  Models {_dropped_m} not decided for {dataset_name}; "
                  f"dropped.")
    else:
        model_names = _decided_m
    models = get_model_factories(model_names, task_type=task_type, n_samples=n_samples)
    print(f"  Models: {list(models.keys())}  (x{n_seeds} seeds averaged per fold)")

    ds_config = DATASET_REPS.get(dataset_name, {})
    # The 10-rep pool (the retired 'top_tier' == 'general'); var name kept.
    #
    # cut_pool is the same representation cut complexity run_all applies to
    # the pool it loads, and it is a no-op when KIRBY_LOCAL_REPS is unset, so
    # the cluster runs all ten as before. This file read DATASET_REPS
    # directly, so a local run that had cut the pool to five watched c21 load
    # the other five straight back in and run the whole ten-representation
    # survey -- the log line "Loading 5 reps not already in memory" on
    # flab_thermo_tresanco 2026-09-25 is that happening.
    #
    # WHAT THE CUT COSTS. The base-representation loop is the survey, so five
    # is half the bases; and the donors in C_topreps and D_both are the OTHER
    # bases, so it also takes each donor pool from nine representations to
    # four. It is a compute cut for a laptop and says nothing about which
    # representations belong in the pool. ROWS ARE NOT TOUCHED: all 672
    # antibodies are in every fold either way.
    from experiments._local_grid import cut_pool
    top_tier_names = cut_pool(dataset_name, ds_config.get('general', []))
    # Augmentation representations only where an admission method survived
    # Step 8 (2026-10-05). Elsewhere none is loaded or offered, so no
    # extra-feature arm can be built by accident.
    aug_rep_names = ds_config.get('augmentation', []) if strategies else []

    # Augmentation reps: prefer the ones run_all already loaded. complexity
    # run_all pops aug-only reps out of data['reps'] (so find_best_hybrid never
    # sees a rep in both the base and augmentation sets) and parks them in
    # data['augmentations']. Re-deriving them from data['reps'] would silently
    # find nothing there.
    aug_reps = dict(data.get('augmentations') or {}) if strategies else {}

    # Base reps are TOP TIER. c21 asks "can the BEST rep be nudged further" —
    # the general pool is the hybrid search pool, not the answer to that.
    missing = [r for r in top_tier_names if r not in reps]
    missing += [r for r in aug_rep_names if r not in reps and r not in aug_reps]
    if missing:
        print(f"  Loading {len(missing)} reps not already in memory: {missing}")
        extra = load_data(dataset_name, rep_preset='all', rep_names=missing)
        for rname, arr in extra['reps'].items():
            if rname in aug_rep_names:
                aug_reps[rname] = arr
            else:
                reps[rname] = arr

    top_tier_reps = {r: reps[r] for r in top_tier_names if r in reps}
    aug_reps = {r: v for r, v in aug_reps.items() if r in aug_rep_names}

    if not top_tier_reps:
        print(f"  No top_tier reps available for {dataset_name} — skipping C21")
        return {'error': 'no_top_tier_reps', 'dataset': dataset_name}

    print(f"  Base reps (top_tier, {len(top_tier_reps)}): "
          f"{', '.join(f'{k}={v.shape[1]}D' for k, v in top_tier_reps.items())}")
    if aug_reps:
        print(f"  Augmentation reps ({len(aug_reps)}): "
              f"{', '.join(f'{k}={v.shape[1]}D' for k, v in aug_reps.items())}")
    elif not strategies:
        print(f"  Augmentation reps: not used on {dataset_name} "
              f"(augmentation cut by DECISIONS.md Step 8)")
    else:
        # Not a defect: the survey tested augmentation on this dataset and none
        # helped, so PIPELINE_READINESS_TODO records "Augmentation | none".
        print(f"  Augmentation reps: NONE configured for {dataset_name} "
              f"(survey found none helped) — B/D conditions will be skipped")
    print(f"  Top-N features taken from each other top_tier rep: {top_n}")

    # The outer folds, drawn ONCE (2026-10-05, handoff 2.4 steps 5 and 14):
    # the same splitter and seed the per-fold searches of every other
    # experiment use, and config['n_splits'] folds. Every arm below is fitted
    # on a fold's training rows and scored on its held-out rows.
    folds = nested_cv_outer_folds(top_tier_reps, labels, groups, n_folds,
                                  OUTER_FOLD_SEED, task_type)
    n_folds_run = len(folds)
    print(f"  Outer folds: {n_folds_run} (asked {n_folds})"
          + ('; one held-out split, so no spread over folds'
             if n_folds_run == 1 else ''))

    # The fixed recipes, every part from the decided general grid.
    href = _hybrid_ref_recipe(dataset_name, hybrid_budgets)
    href['headline_budget'] = (HYBRID_REF_HEADLINE_BUDGET
                               if HYBRID_REF_HEADLINE_BUDGET in href['budgets']
                               else None)
    srec = _single_rep_recipe(dataset_name)
    print(f"  Hybrid reference recipe: allocation {href['allocation_methods']}, "
          f"importance {href['importance_methods']}, evaluator "
          f"{href['internal_evaluators']}, representations "
          f"{href['rep_count'] or 'whole pool'}, budgets {href['budgets']} "
          f"features (headline {href['headline_budget']})")
    for _ax, _r in href['replaced_from_file'].items():
        print(f"    [decided grid] {_ax}: file value {_r['file_value']} -> "
              f"{_r['used']}")
    print(f"  Single-representation recipe: importance "
          f"{srec['importance_methods']}, evaluator "
          f"{srec['internal_evaluators']}"
          + (f" (replaced from file: {srec['replaced_from_file']})"
             if srec['replaced_from_file'] else ''))

    # Pin the base-rep loop when asked. Unpinned this is a 10-rep survey; the
    # goal is best-rep x best-model, so normally exactly one name comes in here.
    if base_reps:
        wanted = [r for r in base_reps if r in top_tier_reps]
        missing_bases = [r for r in base_reps if r not in top_tier_reps]
        if missing_bases:
            raise ValueError(
                f"--base-reps named {missing_bases} which are not in "
                f"{dataset_name}'s pool ({sorted(top_tier_reps)}).")
        base_loop = {r: top_tier_reps[r] for r in wanted}
        print(f"  Base reps PINNED to {wanted} "
              f"(pool has {len(top_tier_reps)}; donors still come from all of it)")
    else:
        base_loop = top_tier_reps
        # KIRBY_C21_BASE_REPS: a count off the front of the pool, or names.
        # Empty means the whole pool, which is the shipped survey.
        _lim = (C21_BASE_REP_LIMIT or '').strip()
        if _lim:
            if _lim.isdigit():
                _keep = list(top_tier_reps)[:int(_lim)]
            else:
                _asked = [x.strip() for x in _lim.split(',') if x.strip()]
                _keep = [r for r in top_tier_reps if r in _asked]
            if _keep and len(_keep) < len(top_tier_reps):
                _dropped = [r for r in top_tier_reps if r not in _keep]
                print(f"  [local] base-representation loop cut to {_keep}; "
                      f"not run as a base: {_dropped}. Every one of them is "
                      f"still a donor in C and D, and every molecule is still "
                      f"in every fold.")
                base_loop = {r: top_tier_reps[r] for r in _keep}

    rng = np.random.RandomState(0)
    rep_results = []
    # {(base representation, model): [one score per fold]} for the base
    # representation on its own. See _baseline_fold_scores.
    base_score_cache = {}

    if skip_single_rep:
        print("  Single-rep arms SKIPPED (skip_single_rep) — "
              "reference arms only")
        base_loop = {}

    for base_name, X_base in base_loop.items():
        base_dim = X_base.shape[1]
        print(f"\n  {base_name} ({base_dim}D)")

        conditions = _build_conditions(base_name, top_tier_reps, aug_reps, top_n,
                                       rng=rng, with_controls=with_controls,
                                       condition_keys=CONDITION_KEYS)
        if len(conditions) == 1:
            print(f"    only baseline available (no aug reps, no other top reps)")

        # (condition, admission method) pairs. The baseline condition offers no
        # donors at all, so it runs once and the method axis does not apply to
        # it. Every other condition runs under each requested method.
        arms = []
        for cond_name, (aug_dict, per_source) in conditions.items():
            if aug_dict is None:
                arms.append((cond_name, None, None, 'none'))
            else:
                for _strat in strategies:
                    arms.append((cond_name, aug_dict, per_source, _strat))

        for cond_name, aug_dict, per_source, strategy in arms:
            is_baseline = aug_dict is None
            arm_name = cond_name if is_baseline else f'{cond_name}__{strategy}'

            # No search. With one base rep the recipe is fully determined:
            # 100% of the budget goes to that rep, and budget = its full native
            # dimension. Allocation and internal_evaluator cannot change a
            # single-rep selection. Importance CAN (corrected 2026-10-05,
            # tracker C21a): base features with importance 0 are not kept, so
            # it is taken from the decided grid (_single_rep_recipe). The
            # admission method chooses which extra features to keep.
            bp = None
            if not is_baseline:
                bp = Blueprint(
                    reps=[base_name],
                    allocation_method=srec['allocation_methods'],  # 1 rep -> 100%
                    importance_method=srec['importance_methods'],
                    internal_evaluator=srec['internal_evaluators'],  # inert (no search)
                    budget=base_dim,                           # keep all base features
                    step_size=5,
                    augmentation_strategy=strategy,
                    aug_per_source_budget=per_source,
                    # Recorded on the recipe as of 2026-08-26. Before that a
                    # rebuild silently used create_hybrid's own defaults for
                    # all four, whatever the run was configured with.
                    candidate_ranking=candidate_ranking,
                    augmentation_budget=augmentation_budget,
                    aug_max_candidates=max_candidates,
                    metric=metric_name,
                    task_type=task_type,
                    lower_is_better=lower_is_better)

            for model_name, model_factory in models.items():
                # Every downstream model shares one blueprint, but the
                # extra-feature SELECTION is not model-independent: the model
                # passed below scores each candidate block size (corrected
                # 2026-10-05; the comment here said the opposite). So the
                # admitted extras are chosen by the decided models only.
                bp_model = bp

                # The base-alone score depends on the base representation, the
                # model and the fold and on nothing else, so it is computed
                # once per (base representation, model) and reused by every
                # arm built on it. It used to be refitted inside this loop for
                # all eleven arms: identical inputs, identical seeds,
                # identical numbers. Measured on flab_thermo_tresanco
                # 2026-09-25, 3 folds x 3 seeds: 45 s for rf and 34 s for svm
                # per pass, and eleven arms x two models paid it ten more
                # times than once. No stored number moves.
                _bkey = (base_name, model_name)
                if _bkey not in base_score_cache:
                    base_score_cache[_bkey] = _baseline_fold_scores(
                        X_base, labels, folds,
                        model_factory, metric_fn, n_seeds)
                scores_base = list(base_score_cache[_bkey])
                scores_aug = []
                fold_aug_info = []

                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    if is_baseline or bp is None:
                        scores_aug.append(float('nan'))
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                              'final_dim': base_dim,
                                              'n_base_kept': base_dim,
                                              'n_base_removed': 0})
                        continue

                    train_base = {base_name: X_base[train_idx]}
                    val_base = {base_name: X_base[val_idx]}

                    # Two donors a working admission rule must take NOTHING
                    # from: a block of pure Gaussian noise as wide as the real
                    # donors actually offer under this condition's per-donor
                    # cap, and a block of features copied out of the base
                    # representation and roughed up by 5% of each feature's own
                    # standard deviation. The near-copy is the only one that
                    # catches a rule rewarding what the base already carries.
                    #
                    # Built full length once per fold with a fresh seed, so the
                    # same features are sliced for training and validation.
                    # They join the same pool as the real donors, so they cost
                    # no extra conditions -- what they buy is a count.
                    pool = aug_dict
                    if with_decoys:
                        pool = _plant_capped_decoys(
                            aug_dict, X_base,
                            np.random.RandomState(1000 + fold_idx),
                            per_source)

                    train_augs = {rn: pool[rn][train_idx] for rn in pool}
                    val_augs = {rn: pool[rn][val_idx] for rn in pool}

                    try:
                        # build_fold_hybrid -> create_hybrid -> augment_uncapped.
                        # per_source rides on the blueprint; passed explicitly
                        # too for clarity. No search anywhere in this path.
                        Xtr_h, Xva_h, feature_info = build_fold_hybrid(
                            train_base, val_base, y_train, bp_model,
                            augmentations_train=train_augs,
                            augmentations_val=val_augs,
                            aug_per_source_budget=per_source,
                            groups=(groups[train_idx] if groups is not None else None),
                            model=model_factory(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better,
                            # THERE IS ONE REPRESENTATION HERE. An allocation
                            # proportional to how the representations score
                            # against each other has nothing to compare, so the
                            # one representation takes the whole budget
                            # whatever its score is. Left to itself
                            # allocate_performance_weighted fits the model on
                            # it anyway to produce that number.
                            #
                            # Measured on flab_thermo_tresanco 2026-09-25,
                            # saprot as the base, random forest, fold 0, the
                            # same call run both ways back to back: 161.3 s
                            # without this line and 74.5 s with it. The two
                            # builds came back with identical selected feature
                            # indices, an identical training matrix, an
                            # identical validation matrix, the same budget
                            # split {saprot: 1000} and the same ten admitted
                            # mutation_features columns.
                            # ⛔ Single-representation arms ONLY. The H arms
                            # below have five or ten representations and the
                            # score decides how the budget splits between them,
                            # so they must measure it.
                            precomputed_alloc_scores={base_name: 1.0})

                        # 0/1 features stay unscaled (2026-10-05, 2.3 item 17).
                        scaler_h = FeatureScaler()
                        Xtr_h = scaler_h.fit_transform(Xtr_h)
                        Xva_h = scaler_h.transform(Xva_h)
                        mean_h, _ = _fit_score_seeds(Xtr_h, y_train, Xva_h, y_val,
                                                     model_factory, metric_fn, n_seeds)
                        scores_aug.append(mean_h)

                        aug_info = feature_info.get('augmentation_info', {})
                        # Per-source detail: which rep donated how many columns.
                        kept_detail = {
                            k: int(v.get('n_features', 0))
                            for k, v in (aug_info or {}).items()
                        }
                        # 2026-10-05: counted from the recipe, not from the
                        # width. n_added was max(0, final_dim - base_dim),
                        # which read 0 whenever base features were removed:
                        # the hybrid at budget=base_dim keeps only base
                        # features with importance above zero (about 1300
                        # base features removed on one dataset), before any
                        # extra is admitted. n_base_removed counts every base
                        # feature not in the final matrix, for either reason;
                        # n_base_displaced is the part capped evicted to make
                        # room for extras (phase_b_replaced, 0 for uncapped).
                        n_added = int(sum(kept_detail.values()))
                        _bi = feature_info.get(base_name)
                        n_base_kept = (int(_bi.get('n_features', 0))
                                       if isinstance(_bi, dict)
                                       else int(Xtr_h.shape[1]) - n_added)
                        fold_aug_info.append({
                            'kept': sorted(kept_detail),
                            'kept_detail': kept_detail,
                            'n_added': n_added,
                            'n_base_kept': n_base_kept,
                            'n_base_removed': int(base_dim - n_base_kept),
                            'n_base_displaced': int(
                                feature_info.get('phase_b_replaced') or 0),
                            'final_dim': Xtr_h.shape[1],
                            # How many features this fold took from each decoy.
                            # Both should be 0. Anything above 0 says the rule
                            # admits pure noise, or admits near-copies of what
                            # the base representation already carries.
                            'decoys': decoy_counts(aug_info),
                        })
                    except Exception as e:
                        print(f"    {model_name} fold {fold_idx}: augmentation failed - {e}")
                        scores_aug.append(float('nan'))
                        # Build failed: nothing was built, so the counts are
                        # None rather than a base_dim that was never fitted.
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                              'final_dim': base_dim,
                                              'n_base_kept': None,
                                              'n_base_removed': None,
                                              'build_failed': True,
                                              'decoys': {'noise': 0, 'echo': 0}})

                base_mean = float(np.nanmean(scores_base))
                # A search that found no winning config makes every augmented
                # fold NaN. Flag it explicitly rather than emitting a bare NaN
                # aug_mean, which is what nan-saturated the old c21 results.
                if is_baseline:
                    aug_status = 'baseline'
                elif bp is None:
                    aug_status = 'search_no_config'
                elif np.all(np.isnan(scores_aug)):
                    aug_status = 'all_folds_failed'
                else:
                    aug_status = 'ok'

                if aug_status in ('baseline', 'search_no_config', 'all_folds_failed'):
                    aug_mean = float('nan')
                    delta = 0.0 if is_baseline else float('nan')
                else:
                    aug_mean = float(np.nanmean(scores_aug))
                    delta = aug_mean - base_mean

                all_kept = set()
                total_added = []
                noise_taken, echo_taken = [], []
                for fi in fold_aug_info:
                    all_kept.update(fi['kept'])
                    total_added.append(fi['n_added'])
                    _d = fi.get('decoys') or {}
                    noise_taken.append(int(_d.get('noise', 0)))
                    echo_taken.append(int(_d.get('echo', 0)))

                result = {
                    'base_rep': base_name,
                    'base_tier': 'top_tier',
                    'condition': cond_name,
                    'arm': arm_name,
                    'augmentation_strategy': 'none' if is_baseline else strategy,
                    'candidate_ranking': candidate_ranking,
                    'aug_max_candidates': max_candidates,
                    # Per fold, how many features came from each planted decoy.
                    # A working rule takes zero of both.
                    'decoy_noise_per_fold': noise_taken,
                    'decoy_echo_per_fold': echo_taken,
                    'decoys_planted': bool(with_decoys) and not is_baseline,
                    # How many features the noise donor was built with, and how
                    # many the real donors offered it against. The two are the
                    # same number by construction; both are stored so the decoy
                    # counts above can be read against a width.
                    'decoy_noise_width': (
                        _effective_donor_width(aug_dict, per_source)
                        if (with_decoys and aug_dict) else 0),
                    'real_donor_width': (
                        _effective_donor_width(aug_dict, per_source)
                        if aug_dict else 0),
                    'model': model_name,
                    'base_dim': base_dim,
                    'base_mean': base_mean,
                    # None with one scored fold: no spread (2026-10-05).
                    'base_std': _spread(scores_base),
                    'base_scores': scores_base,
                    'aug_mean': aug_mean,
                    'aug_std': _spread(scores_aug) if not is_baseline else None,
                    'aug_scores': scores_aug,
                    'delta': delta,
                    'aug_status': aug_status,
                    # Sign-normalise: positive == improvement regardless of
                    # metric direction. Without this, RMSE/MAE wins are stored
                    # as negative relative_improvement and any downstream filter
                    # on positive sign rejects them.
                    'relative_improvement': (
                        (-1 if lower_is_better else 1) * delta
                        / max(abs(base_mean), REL_IMPROVEMENT_DENOM_FLOOR)
                        if not np.isnan(delta) else float('nan')),
                    'kept_augmentations': sorted(all_kept),
                    # Mean over folds of extra features admitted (2026-10-05:
                    # now the true count, see n_added in fold_aug_info).
                    'mean_features_added': float(np.mean(total_added)),
                    # Per fold: base features kept, base features removed,
                    # extra features added (2026-10-05).
                    'n_base_kept_per_fold': [fi.get('n_base_kept')
                                             for fi in fold_aug_info],
                    'n_base_removed_per_fold': [fi.get('n_base_removed')
                                                for fi in fold_aug_info],
                    'n_added_per_fold': [fi['n_added'] for fi in fold_aug_info],
                    'fold_aug_info': fold_aug_info,
                    'aug_pool_reps': sorted(aug_dict) if aug_dict else [],
                    'per_source_budget': per_source,
                    'top_n': top_n,
                    # What top_n counts (2026-10-05, dirty-run bug "top_n
                    # meaning"): features taken from EACH other base
                    # representation in C and D, not a number of donors.
                    'top_n_counts': 'features per other base representation',
                    # The recipe in plain fields (handoff 2.4 step 11). One
                    # representation, so allocation has nothing to split and
                    # the budget is the base width in features.
                    'recipe': (None if is_baseline else {
                        'allocation_method': srec['allocation_methods'],
                        'importance_method': srec['importance_methods'],
                        'internal_evaluator': srec['internal_evaluators'],
                        'budget_features': int(base_dim),
                        'rep_count': 1,
                        'aug_condition': strategy,
                        'replaced_from_file': srec['replaced_from_file'],
                    }),
                    'selection': 'none: every representation and condition '
                                 'is run, nothing ranked',
                    'selection_scope': 'fold_train_rows',
                    # One entry per outer fold (2026-10-05, handoff 2.4 step
                    # 11). Every feature set was built and every model fitted
                    # on the fold's training rows; the scores are on its
                    # held-out rows. Nothing here is ranked, so there is no
                    # training-row score to report.
                    'per_fold': [
                        dict({'fold': int(_fi),
                              'n_train_rows': int(len(folds[_fi][0])),
                              'n_heldout_rows': int(len(folds[_fi][1])),
                              'base_heldout_score': scores_base[_fi],
                              'aug_heldout_score': (None if is_baseline
                                                    else scores_aug[_fi]),
                              'training_rows_score': None},
                             **fold_aug_info[_fi])
                        for _fi in range(len(folds))],
                    'blueprint': bp.to_dict() if bp else None,
                }
                rep_results.append(result)

                sign = -1 if lower_is_better else 1
                if aug_status not in ('ok', 'baseline'):
                    print(f"    {arm_name}/{model_name}: base={base_mean:.4f}  "
                          f"SKIPPED ({aug_status})")
                elif is_baseline:
                    print(f"    {arm_name}/{model_name}: base={base_mean:.4f}")
                else:
                    status = "+" if sign * delta > 0 else "-" if sign * delta < 0 else "="
                    # Decoy counts print on the same line as the delta, so a
                    # win that was bought by admitting noise cannot be read
                    # without seeing that it was.
                    decoy = (f"  decoys=noise {sum(noise_taken)}/echo {sum(echo_taken)}"
                             if with_decoys else "")
                    print(f"    {arm_name}/{model_name}: base={base_mean:.4f}  "
                          f"aug={aug_mean:.4f}  delta={sign * delta:+.4f} [{status}]  "
                          f"kept={sorted(all_kept)}  added={np.mean(total_added):.1f}"
                          f"{decoy}")

    # ── Multi-rep reference arms: the c02 top-tier hybrid, and that hybrid
    #    plus augmentation features. Scored on the SAME folds and the SAME
    #    models as the single-rep arms, so the numbers compare directly.
    hybrid_results = []
    # Which downstream models the reference arms are scored under. One H build
    # over the whole pool cost 250.8 s against 97 s for a whole
    # single-representation arm (flab_thermo_tresanco, 2026-09-25), so the
    # model axis is the difference between the H arms fitting in a local run
    # and not. Unset, this is every model the run uses, which is the shipped
    # behaviour.
    h_models = models
    _hm = (HYBRID_MODEL_LIMIT or '').strip()
    if _hm:
        if _hm.isdigit():
            _hkeep = list(models)[:int(_hm)]
        else:
            _hasked = [x.strip() for x in _hm.split(',') if x.strip()]
            _hkeep = [m for m in models if m in _hasked]
        if _hkeep and len(_hkeep) < len(models):
            print(f"  [local] reference arms scored under {_hkeep} only; "
                  f"{[m for m in models if m not in _hkeep]} has "
                  f"single-representation arms but no hybrid to read them "
                  f"against.")
            h_models = {m: models[m] for m in _hkeep}
    if len(top_tier_reps) >= 2 and 'h0' in HYBRID_ARMS:
        print(f"\n  HYBRID reference arms ({len(top_tier_reps)} top-tier reps)")
        hybrid_results += _score_hybrid_condition(
            'H0_hybrid', top_tier_reps, None, 'none',
            labels, groups, h_models, metric_fn, lower_is_better,
            folds, href, task_type, metric_name, n_seeds=n_seeds)
    # H1 adds extra features to H0, whose recipe comes from the decided
    # GENERAL grid. Step 8 keeps augmentation on the distance search only
    # (ginkgo_colloidal, ginkgo_hic); the general grid's aug_conditions are
    # ['none'] on every dataset. So "general recipe + extras" is decided
    # nowhere, and H1 stays off until the owner chooses (2026-10-05).
    h1_skip_reason = None
    _gen_aug = [c for c in decided_aug_conditions(dataset_name, 'general')
                if c != 'none']
    if strategies and not _gen_aug:
        h1_skip_reason = (
            f'H1 would add extra features to the general-grid hybrid H0, but '
            f'the decided general grid of {dataset_name} keeps no '
            f'augmentation (Step 8 keeps it on the distance search only). '
            f'Off until the owner decides.')
    if len(top_tier_reps) >= 2 and 'h1' in HYBRID_ARMS:
        if h1_skip_reason:
            print(f"    H1_hybrid_aug: skipped ({h1_skip_reason})")
        elif aug_reps:
            # USE CASE A: small augmented blocks added to the MULTI-REP hybrid.
            # One arm per admission method, so the placement question (extras
            # onto one representation, or onto the hybrid) is asked of every
            # method rather than only of 'uncapped'.
            for _strat in strategies:
                if _strat == 'none':
                    continue  # H0_hybrid already is that arm
                hybrid_results += _score_hybrid_condition(
                    f'H1_hybrid_aug__{_strat}', top_tier_reps, aug_reps, _strat,
                    labels, groups, h_models, metric_fn, lower_is_better,
                    folds, href, task_type, metric_name, n_seeds=n_seeds,
                    candidate_ranking=candidate_ranking,
                    augmentation_budget=augmentation_budget,
                    max_candidates=max_candidates,
                    with_decoys=with_decoys)
        elif not strategies:
            print(f"    H1_hybrid_aug: skipped ({aug_skip_reason})")
        else:
            print("    H1_hybrid_aug: skipped (no augmentation reps configured)")
    if len(top_tier_reps) < 2:
        print(f"\n  HYBRID reference arms: skipped "
              f"({len(top_tier_reps)} top-tier rep — a hybrid needs >= 2)")
    elif set(HYBRID_ARMS) != {'h0', 'h1'}:
        print(f"  [local] reference arms run: {','.join(HYBRID_ARMS)} "
              f"[{HYBRID_ARMS_SOURCE}]. "
              f"{'H1, the hybrid WITH extras, did not run.' if 'h1' not in HYBRID_ARMS else ''}"
              f"{'H0, the hybrid with no extras, did not run, so H1 has no reference.' if 'h0' not in HYBRID_ARMS else ''}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Head-to-head: best single base rep vs the c02 hybrid vs every augmented
    # arm, per model. This is the comparison the experiment exists to make.
    sign = -1 if lower_is_better else 1
    # The rows are picked by sign * score, so the right rows are chosen; the
    # scores PRINTED beside them are raw, in the dataset's own metric. Say which
    # way that metric runs. Fixed 2026-09-10 (metric-direction audit).
    print(f"\n  HEAD-TO-HEAD ({metric_name}, "
          f"{'lower' if lower_is_better else 'higher'} = better; "
          f"delta is sign-normalised so + is an improvement)")
    for model_name in models:
        print(f"\n  [{model_name}]")
        base_rows = [r for r in rep_results
                     if r['condition'] == 'A_baseline' and r['model'] == model_name
                     and not np.isnan(r['base_mean'])]
        if base_rows:
            # Picked on the same held-out scores it prints (2026-10-05:
            # labelled as such). A print only; nothing stored is chosen
            # this way.
            best_base = max(base_rows, key=lambda r: sign * r['base_mean'])
            print(f"    highest held-out single base rep (picked on these "
                  f"scores): {best_base['base_rep']:<22s} "
                  f"{best_base['base_mean']:+.4f}")
        aug_rows = [r for r in rep_results
                    if r['condition'] != 'A_baseline' and r['model'] == model_name
                    and r.get('aug_status') == 'ok']
        for cond in sorted(set(r['condition'] for r in aug_rows)):
            cr = [r for r in aug_rows if r['condition'] == cond]
            best = max(cr, key=lambda r: sign * r['aug_mean'])
            print(f"    {cond:<21s}: {best['base_rep']:<22s} "
                  f"{best['aug_mean']:+.4f}  (delta {sign * best['delta']:+.4f} "
                  f"vs its own base)")
        for r in hybrid_results:
            if r['model'] == model_name and r.get('aug_status') == 'ok':
                print(f"    {r['condition']:<21s}: {'(hybrid)':<22s} "
                      f"{r['hybrid_mean']:+.4f}")

    # Only aggregate over results that were actually evaluable. Skipped ones
    # carry a NaN delta and would poison the mean / argmax; count them
    # separately so a NaN-saturated run reads as "N skipped" rather than a
    # meaningless NaN mean_delta.
    aug_results_all = [r for r in rep_results if r['condition'] != 'A_baseline']
    aug_results = [r for r in aug_results_all if r.get('aug_status') == 'ok']
    n_skipped = len(aug_results_all) - len(aug_results)
    n_tests = len(aug_results)
    improved = hurt = 0
    mean_delta = 0.0
    if n_skipped:
        skip_reasons = {}
        for r in aug_results_all:
            if r.get('aug_status') != 'ok':
                skip_reasons[r['aug_status']] = skip_reasons.get(r['aug_status'], 0) + 1
        print(f"  Skipped (not evaluable): {n_skipped} ({skip_reasons})")
    if n_tests > 0:
        sign = -1 if lower_is_better else 1
        improved = sum(1 for r in aug_results if sign * r['delta'] > 0)
        hurt = sum(1 for r in aug_results if sign * r['delta'] < 0)
        mean_delta = float(np.mean([sign * r['delta'] for r in aug_results]))
        all_conditions = sorted(set(r['condition'] for r in aug_results))
        # The BASE REPRESENTATIONS ACTUALLY LOOPED, not the size of the pool.
        # This line read len(top_tier_reps) and so announced "5 top-tier reps"
        # on a run whose base-representation loop had been cut to one -- the
        # three numbers multiplied to something other than the count beside
        # them. The pool is printed separately above, as donors.
        print(f"  Total evaluable tests: {n_tests} "
              f"({len(base_loop)} base representations x {len(models)} models "
              f"x {len(all_conditions)} conditions; donors come from the "
              f"{len(top_tier_reps)}-representation pool)")
        print(f"  Augmentation improved: {improved}/{n_tests}")
        print(f"  Augmentation hurt:     {hurt}/{n_tests}")
        print(f"  Mean delta:            {mean_delta:+.4f}")

        for cond in all_conditions:
            cond_results = [r for r in aug_results if r['condition'] == cond]
            cond_improved = sum(1 for r in cond_results if sign * r['delta'] > 0)
            cond_mean = float(np.mean([sign * r['delta'] for r in cond_results]))
            best_c = max(cond_results, key=lambda r: sign * r['delta'])
            print(f"  {cond}: {cond_improved}/{len(cond_results)} improved, "
                  f"mean={cond_mean:+.4f}, "
                  f"best={best_c['base_rep']}/{best_c['model']} "
                  f"delta={sign * best_c['delta']:+.4f}")
    elif not strategies:
        # 2026-10-05: no extra-feature arm was built, by decision, not failure.
        print(f"  No augmentation results: {aug_skip_reason}")
    else:
        print(f"  No evaluable augmentation results "
              f"(all {len(aug_results_all)} non-baseline tests were skipped).")

    return convert_numpy({
        'result_version': RESULT_VERSION,
        'metric': metric_name,
        'metric_name': metric_name,
        # From get_domain_metric, never assumed.
        'lower_is_better': lower_is_better,
        'n_folds': n_folds_run,
        'n_folds_asked': n_folds,
        'outer_fold_seed': OUTER_FOLD_SEED,
        'n_samples': n_samples,
        # 2026-10-05 (handoff 2.4 step 11): the dataset's decided models, and
        # the scope every feature set and fit was confined to.
        'models': list(models),
        'selection': ('none: every representation and every kept '
                      'condition is run; hybrids are fixed recipes'),
        'selection_scope': 'fold_train_rows',
        'admission_methods_asked': list(strategies_requested),
        'admission_methods_run': list(strategies),
        'augmentation_skipped_reason': aug_skip_reason,
        'h1_skipped_reason': h1_skip_reason,
        'hybrid_reference_recipe': {
            k: href[k] for k in ('allocation_methods', 'importance_methods',
                                 'internal_evaluators', 'rep_count',
                                 'budgets', 'headline_budget',
                                 'replaced_from_file', 'grid')},
        'single_rep_recipe': srec,
        'top_n': top_n,
        'base_reps': sorted(top_tier_reps),
        'base_dims': {k: int(v.shape[1]) for k, v in top_tier_reps.items()},
        # The c02 top-tier hybrid and that hybrid + augmentation features,
        # scored on the same folds/models as the single-rep arms.
        'hybrid_results': hybrid_results,
        'aug_reps': sorted(aug_reps),
        'aug_dims': {k: int(v.shape[1]) for k, v in aug_reps.items()},
        'rep_results': rep_results,
        'summary': {
            'n_tests': n_tests,          # evaluable (aug_status == 'ok') only
            'n_skipped': n_skipped,      # search_no_config / all_folds_failed
            'improved': improved if n_tests > 0 else 0,
            'hurt': hurt if n_tests > 0 else 0,
            'mean_delta': mean_delta if n_tests > 0 else 0.0,
            # When every non-baseline test was skipped the experiment produced
            # no augmentation signal — flag it so a NaN-saturated run is not
            # mistaken for a "no improvement" result.
            'all_skipped': (n_tests == 0 and n_skipped > 0),
        },
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C21: Single-Rep Augmentation',
                                      has_delta=False, has_augment=False)
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                        help='Features taken from each other top_tier rep '
                             f'(default {DEFAULT_TOP_N})')
    parser.add_argument('--base-reps', type=str, default=None,
                        help='Comma-separated base reps to use (default: the '
                             'whole pool, which is a 10x more expensive survey). '
                             'Pin this to the dataset best rep.')
    # --models is NOT registered here. create_experiment_parser already adds it
    # (with the same meaning, comma-separated, default None). Registering it
    # twice made argparse raise on import of this block, so every command-line
    # start of this script died before parse_args -- including the two examples
    # in the docstring above. It reached the cluster only via
    # run_dataset.py complexity <ds>, which calls run() in process, so the crash
    # was invisible. Found and removed 2026-08-26.
    parser.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS,
                        help=f'Model seeds averaged per fold (default {DEFAULT_N_SEEDS})')
    parser.add_argument('--hybrid-budgets', type=str,
                        default=','.join(str(b) for b in HYBRID_REF_BUDGETS),
                        help='Comma-separated budgets for the fixed multi-rep '
                             'reference arms H0/H1')
    parser.add_argument('--no-controls', action='store_true',
                        help='Skip the matched-width random-column control arms')
    parser.add_argument('--strategies', type=str,
                        default=','.join(DEFAULT_STRATEGIES),
                        help='Comma-separated admission methods, from '
                             f'{",".join(ALL_STRATEGIES)} '
                             f'(default {",".join(DEFAULT_STRATEGIES)})')
    parser.add_argument('--candidate-ranking', type=str,
                        default='residual_correlation',
                        choices=['residual_correlation', 'rf_importance'],
                        help='How pooled candidate features are ranked. Read '
                             'only by capped and uncapped.')
    parser.add_argument('--augmentation-budget', type=int, default=20,
                        help='Features taken from EACH donor by the all and '
                             'greedy_ablation methods (default 20)')
    parser.add_argument('--max-candidates', type=int, default=50,
                        help='Length of the pooled ranked candidate list a '
                             'block size is chosen from (default 50)')
    parser.add_argument('--no-decoys', action='store_true',
                        help='Do not plant the pure-noise and near-copy donors')
    args = parser.parse_args()
    config = get_config(args)

    _split = lambda s: [x.strip() for x in s.split(',') if x.strip()]
    base_reps = _split(args.base_reps) if args.base_reps else None
    model_names = _split(args.models) if args.models else None
    hybrid_budgets = tuple(int(b) for b in _split(args.hybrid_budgets))
    strategies = tuple(_split(args.strategies))

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset, top_n=args.top_n,
                  base_reps=base_reps, model_names=model_names,
                  n_seeds=args.n_seeds, hybrid_budgets=hybrid_budgets,
                  with_controls=not args.no_controls,
                  strategies=strategies,
                  candidate_ranking=args.candidate_ranking,
                  augmentation_budget=args.augmentation_budget,
                  max_candidates=args.max_candidates,
                  with_decoys=not args.no_decoys)
    save_experiment_results(results, args.output, 'c21_single_rep_augmentation',
                            args.dataset)
