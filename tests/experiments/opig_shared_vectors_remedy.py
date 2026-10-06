#!/usr/bin/env python3
"""
Can a hybrid or an augmentation stop a representation merging two molecules?

THE QUESTION. opig_shared_vectors_census.py counts how many molecules a
representation gives one identical row, and turns that count into the best score
any model could reach. This script tries to fix it and measures whether the fix
worked, on two numbers at once:

  1. the score on held-out molecules, in the dataset's own metric;
  2. how many molecules still share a row IN THE MATRIX THAT WAS BUILT for
     that fold.

Both have to move, on the same outer folds, or the claim is not made.

THE FOUR THINGS COMPARED, all on one outer-fold partition.

  A   the target representation on its own.
  B   the target plus a donor representation's features, swept over the
      augmentation settings.
  C1  hybrids of the target with the first pair of partner representations.
  C2  the same with the second pair, and so on for every pair named.
  D1  C1 plus the donor, swept the same way as B. D2 likewise.

REPRESENTATIONS (2026-10-05). Only the dataset's pruned list
(DATASET_REPS[ds]['general']). The target, both partners of every pair and the
--reps pool must be on it; a name off it raises. Until 2026-10-05 the target was
chemberta (384 features) and the second pair used usrcat. Neither is on
herg_ki's list: herg_ki carries chembert (256 features), a different model, and
usrcat is an augmentation representation. So the defaults are now:

  target   'auto': the representation on the list that merges the most
           genuinely different molecules, measured on the loaded matrices
           (choose_representations). In the census of 29 Sep, on herg_ki's
           cached matrices of that day, that was graph_kernel, 105 of 1,412
           molecules, 56 of them stereoisomers. NOT re-measured 2026-10-05:
           the run prints and records the counts it measures ('choice' in
           the result file). If ecfp4 or pdv now merges more than
           graph_kernel, 'auto' picks a named partner as the target and the
           run raises; name the target with --target then.
  pairs    per dataset (DEFAULT_PARTNER_SETS). herg_ki: ecfp4 + pdv, both
           blind to stereochemistry (in the 29 Sep census each tied the same
           56 stereoisomers), so that pair can move the score and cannot
           split those 56; and ecfp4 + mhg_gnn, where mhg_gnn tied no
           molecule, so that pair can. Running both is what separates "a
           hybrid helps" from "a hybrid that can see the difference helps".
           Every other dataset (ginkgo_colloidal and ginkgo_hic included,
           whose lists hold protein embeddings only): one pair picked by
           measurement, the two representations that leave fewest of the
           target's molecules sharing a row.

Which target and which partners is Adelaide's choice (DEFAULT_TARGET,
DEFAULT_PARTNER_SETS).

HOW C IS SCORED (2026-10-05). Per outer fold, one search on that fold's
training rows over the target and the pair, on the dataset's decided grid
(_shared.fold_standard_selection; pruning off). Per decided model it gives the
2 best hybrids (a second hybrid whose selected features equal the first's is
skipped for the next) and the 2 best base representations of that pool. Each
is fitted on the fold's training rows and scored on its held-out rows, as is
the target alone (A). Each hybrid is rebuilt with build_fold_hybrid from its
recipe and the per-representation scores of that same per-fold search. No
distance hybrid (answer 1: this experiment measures no neighbours).

B AND D (answer 4, 2026-10-05). The comparisons with added donor features run
only where Step 8 keeps augmentation: the distance search of ginkgo_colloidal
and ginkgo_hic. On every other dataset, herg_ki included, they are skipped and
the reason is printed and written to the result file. Where they run, only the
augmentation conditions Step 8 keeps are swept ('uncapped' on those two).

B and D run on ginkgo_colloidal and ginkgo_hic, adding paratope and structural
(their added-features representations; handoff 2.3 items 12-13), and are
skipped elsewhere with the reason in the result file.
Per sweep point, per decided model, per outer fold: the 2 best hybrids of
that fold's inner search (a second whose selected features equal the first's
is skipped for the next), each rebuilt on the fold's training rows and scored
on its held-out rows, and the 2 best base representations. The models are the
dataset's two general decided models; knn, which the distance search also
scores, is left out (B_D_INCLUDE_KNN; answer 1 gives knn and distance
hybrids to the census only).

CACHE CHECK. Before anything is fitted, a sample of rows of every loaded
representation that has a builder here is rebuilt and compared with the
cached matrix (graph_kernel is rebuilt whole: its features depend on every
molecule). A mismatch raises. A representation with no builder here is
recorded as not checked; if it is the target, a partner or the donor, the
run refuses (REFUSE_UNCHECKED_COMPARED_REPS) unless --skip-cache-check.

Comparison A is not a search of its own: find_best_hybrid refuses a pool of one
representation with no donor. The target is fitted alone on each fold's
training rows and scored on the held-out rows by the same scorer as every
other slot.

WHY THE TIE COUNT IS REPORTED THREE WAYS. Concatenating features can only ever
split a tied group, never merge one, so "hybrid ties <= target ties" would be
true by construction if the hybrid used all of the target's features. It does
not: it selects a budget's worth. So each fold reports

  ties of the target representation at its full width      (what A fits)
  ties of the target's features the hybrid kept            (the width control)
  ties of the whole built hybrid                           (the answer)

The middle number is what separates "the partner broke the tie" from "the budget
changed the width".

SETTINGS THAT REACH THE ENGINE BUT DO NOTHING. create_hybrid dispatches the four
augmentation strategies at hybrid.py:6180, and each one reads a different subset
of the settings:

  candidate_ranking, aug_per_source_budget, augmentation_margin,
  augmentation_fraction   are read by 'uncapped' and 'capped' only
  augmentation_budget                        is read by 'all' and nothing else
  augmentation_threshold                     is read by the cut greedy path only

So this script never crosses a setting with a strategy that ignores it. Doing so
returns an identical score under a different label, which reads as a measured
null and is not one. aug_max_candidates is not swept at all: it is not a
parameter of find_best_hybrid and passing it raises TypeError; the search pins it
at 50 (find_best_hybrid.py:5007).

Every sweep point is also checked for being a no-op after the fact: a point that
admitted zero donor features is printed as no-op rather than as a result.

Usage:
    # how big is the grid, before submitting anything
    python tests/experiments/opig_shared_vectors_remedy.py \\
        --datasets herg_ki --count-only

    # the run (outer fold count from the experiment config, get_config)
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache \\
        python tests/experiments/opig_shared_vectors_remedy.py \\
        --datasets herg_ki
"""
import os
import sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# LightGBM links libomp and torch links libiomp5; whichever loads its pool second
# exits 139 with no traceback on this Mac (src/kirby/hybrid.py:3512). One thread
# is not a cure on its own, but it removes the multi-threaded fit that triggers
# it. Server runs are untouched because the variable is only set when absent.
if sys.platform == 'darwin':
    os.environ.setdefault("KIRBY_LGBM_THREADS", "1")

import json
import time
import hashlib
import argparse
import warnings
from collections import Counter
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')

from experiments._shared import (add_distance_metric_arg,          # noqa: E402
                                 apply_distance_metric_arg,
                                 convert_numpy)
import opig_shared_vectors_census as census                        # noqa: E402

# ===== CONFIGURATION ======================================================

SEED = 42

# The search arms B and D run under. Answer 4 (2026-10-05): the comparisons
# with added donor features run only where Step 8 keeps augmentation, which is
# the distance search of ginkgo_colloidal and ginkgo_hic. The name must contain
# 'distance': decided_grid reads it to choose the distance row of DECISIONS.md
# (models with knn, budgets 100-500 features). It is not a precompute tier, so
# nothing stored by the precompute is read under it.
TIER_DISTANCE = 'general_distance_shared_vec'
AUG_ARMS_TIER = TIER_DISTANCE

# The checkpoint identity of the result layout. Bumped 2026-10-05 when C moved
# to the per-fold standard selection (2 base + 2 hybrids per model per fold,
# no distance hybrid), so rows a run of the old layout left in the
# .partial.jsonl are never read back into the new file.
#
# Bumped again 2026-10-05 (b): B and D now report 2 hybrids per model per
# fold, knn left out, and the cache check covers graph_kernel and mhg_gnn.
#
# Bumped 2026-10-06 with this pass's other changes, so rows an earlier
# attempt stored are not read back.
RESULT_LAYOUT = 'per_fold_standard_selection_2026_10_06b'

DEFAULT_DATASETS = ['herg_ki']

# The target and the three-representation sets. Changed 2026-10-05 from
# chemberta / (ecfp4, pdv) / (ecfp4, usrcat), agreed 2026-09-16, because
# chemberta and usrcat are not on herg_ki's pruned list (only
# DATASET_REPS[ds]['general'] is used now). 'auto' picks the representation
# on the list that merges the most genuinely different molecules. The second
# pair swaps usrcat for mhg_gnn, which tied no herg_ki molecule in the census
# of 29 Sep, as usrcat did. ⛔ Adelaide's choice; see the result file's
# 'choice' block for what was picked.
#
# The pairs are per dataset since 2026-10-05: a named pair is only valid on
# a dataset whose pruned list holds both names, and ginkgo_colloidal and
# ginkgo_hic (the only datasets where B and D run, answer 4) hold protein
# embeddings only. A dataset not named here gets one pair picked by
# measurement (choose_representations).
DEFAULT_TARGET = 'auto'
DEFAULT_PARTNER_SETS = {
    'herg_ki': (('ecfp4', 'pdv'), ('ecfp4', 'mhg_gnn')),
}

# B and D run under the distance search, whose decided models include knn
# (Step 9). Answer 1 gives knn and distance hybrids to
# opig_shared_vectors_census only, so B and D score the dataset's two general
# decided models and leave knn out. ⛔ Adelaide's choice (answer 4 puts B and
# D on the distance search; answer 1 keeps knn out of this experiment). True
# adds knn back to the rows of B and D.
B_D_INCLUDE_KNN = False

# Representations that are never added features of a hybrid here.
# 2026-10-06: empty. paratope and structural ARE the added features of the
# distance search on ginkgo_colloidal and ginkgo_hic (Adelaide 2026-10-05,
# handoff 2.3 items 12-13), so arms B and D run on those two datasets. A
# short-lived exclusion of them here was reverted.
EXCLUDED_DONORS = ()

# A representation the claim is about (the target, a partner, the donor)
# whose cached matrix could not be compared with its builder. True: the run
# refuses, as it does on a mismatch (the dirty run's chemberta matrix tied no
# molecule and nothing noticed). False: printed and recorded only.
# --skip-cache-check overrides either.
REFUSE_UNCHECKED_COMPARED_REPS = True
DEFAULT_OUTPUT = os.path.join(HERE, '..', 'results', 'comparisons')

ONE_FOLD_NOTE = 'one fold, no spread'

# The order slots are reported in, per model, for every C arm.
C_SLOTS = ('hybrid1', 'hybrid2', 'base1', 'base2')

# The four ranking rules f24 still runs. The two deeplift rules were dropped
# 2026-09-07 for taking 10 of every 14 kept features from a planted pure-noise
# donor (AUGMENTATION.md section 10).
RANKINGS = ('residual_correlation', 'rf_importance', 'lightgbm_gain', 'mutual_info')
PER_SOURCE_BUDGETS = (None, 5, 10, 20)
# Multiples of the target representation's own spread across outer folds, so the
# number means the same thing on a dataset scored in RMSE and one scored in AUC.
#
# This IS the sensitivity sweep for that margin -- reviewed 2026-09-18 after a
# question about whether the margin should be tested at different sizes. It
# already is, at 0/0.25x/0.5x/1.0x, crossed with every ranking rule, budget
# cap and fraction. Decision: leave the design as-is; run the full sweep
# when there's a reason to read its result, not before.
MARGIN_MULTIPLES = (0.0, 0.25, 0.5, 1.0)
FRACTIONS = (None, 0.05, 0.10, 0.25)
AUGMENTATION_BUDGETS = (5, 20, 50)          # read by strategy 'all' only
STRATEGIES = ('none', 'uncapped', 'capped', 'all')


def sweep_points(fold_spread, strategies=STRATEGIES):
    """Every augmentation setting worth running, and nothing that is ignored.

    fold_spread is the target representation's standard deviation across the
    outer folds, in the dataset's own metric. A margin is stated as a multiple of
    it so a block of donor features has to beat "keep nothing" by more than the
    run-to-run wobble before it is admitted.

    strategies: the augmentation conditions to sweep. Since 2026-10-05 the
    caller passes 'none' plus the conditions decided_aug_conditions keeps for
    the dataset and search (R3), so a condition Step 8 cut is never run.
    """
    points = []
    for strategy in STRATEGIES:
        if strategy not in strategies:
            continue
        if strategy == 'none':
            points.append({'augmentation_strategy': 'none'})
        elif strategy == 'all':
            for budget in AUGMENTATION_BUDGETS:
                points.append({'augmentation_strategy': 'all',
                               'augmentation_budget': budget})
        else:
            for ranking, per_source, mult, fraction in product(
                    RANKINGS, PER_SOURCE_BUDGETS, MARGIN_MULTIPLES, FRACTIONS):
                # ⛔ THE MARGIN IS DEAD WHENEVER A FRACTION IS SET. Skipped
                # since 2026-09-18, owner's decision.
                #
                # The margin is how much better a block of donor features has
                # to score than keeping none before it is admitted. A fraction
                # says: do not run that test at all, just take the top
                # round(fraction x width) by rank. hybrid.py:4111 states it --
                # "margin is not consulted; the two settings are alternatives,
                # not a stack" -- and hybrid.py:4366 repeats it for the capped
                # path.
                #
                # So every (strategy, ranking, per-source budget, fraction)
                # combination with a fraction set ran the 4 margin multiples as
                # 4 IDENTICAL searches. The dropped points are DEGENERATE: they
                # re-run a search whose every input the engine treats
                # identically. That is one of the three stated reasons a cut is
                # allowed. The margin axis is still swept in full on the
                # fraction=None combinations, which is where it acts.
                if fraction is not None and mult != MARGIN_MULTIPLES[0]:
                    continue
                points.append({'augmentation_strategy': strategy,
                               'candidate_ranking': ranking,
                               'aug_per_source_budget': per_source,
                               'augmentation_margin': mult * (fold_spread or 0.0),
                               'augmentation_margin_multiple': mult,
                               'augmentation_fraction': fraction})
    return points


def point_slug(point):
    """A short stable name for one sweep point.

    It goes into condition_suffix. Without it every sweep point writes the same
    search cache filename -- the path is {dataset}__{tier}__{strategy}.pkl
    (find_best_hybrid.py:153) and carries none of the other settings -- so each
    point evicts the one before it and nothing is ever reused.
    """
    parts = [point['augmentation_strategy']]
    for key, short in (('candidate_ranking', 'rank'),
                       ('aug_per_source_budget', 'cap'),
                       ('augmentation_margin_multiple', 'marg'),
                       ('augmentation_fraction', 'frac'),
                       ('augmentation_budget', 'augbud')):
        if key in point:
            parts.append(f'{short}{point[key]}')
    return '_'.join(str(p).replace('.', 'p') for p in parts)


def engine_kwargs(point):
    """The sweep point with this script's own bookkeeping keys removed."""
    return {k: v for k, v in point.items()
            if k not in ('augmentation_margin_multiple',)}


# ===== PICKING THE TARGET, THE PARTNERS AND THE DONOR ====================

def choose_representations(data, dataset_name, pool, donors,
                           target=None, partner_sets=None, donor=None,
                           need_donor=True):
    """Pick by measurement: which representation merges most, and what splits it.

    target       the representation that merges the most genuinely different
                 molecules -- the one the claim is about.
    partner_sets one or more pairs to hybridise the target with. Named pairs are
                 kept as given; otherwise the two representations that leave
                 fewest of the target's molecules tied are picked.
    donor        the augmentation representation that leaves fewest of the
                 target's molecules tied when laid beside it. Only picked when
                 need_donor (arms B or D run on this dataset).

    All three are chosen on every row of the dataset, before any fold exists.
    That is a choice of WHICH comparisons to run, not of any score, and it is recorded
    in the output as a whole-data choice so it is never read as an honest pick.
    """
    reps = data['reps']
    partitions = {r: census.row_partition(reps[r])[0]
                  for r in reps if reps[r].ndim == 2}

    def still_tied_with(a, b):
        _, sizes = census.joint_partition(partitions[a], partitions[b])
        return int((sizes > 1).sum())

    merged = {}
    if target is None:
        keys = census.input_keys(data, data.get('domain'))
        stereo = census.stereo_stripped_keys(data, data.get('domain'))
        for r in pool:
            if r not in partitions:
                continue
            X = reps[r]
            inv, sizes = census.row_partition(X)
            zero = ~np.any(X != 0, axis=1)
            merged[r] = census.attribute_groups(
                inv, sizes, zero, keys, stereo)['n_representation_merged']
        if not merged or max(merged.values()) == 0:
            raise RuntimeError(
                f'{dataset_name}: no representation in the pool merges any two '
                f'molecules, so there is nothing for this experiment to fix. '
                f'Name one with --target to run it anyway.')
        # Most merged first, then the name, so a tie resolves the same way on
        # every run.
        target = sorted(merged, key=lambda r: (-merged[r], r))[0]
    if target not in partitions:
        raise RuntimeError(f'{dataset_name}: target {target} has no matrix here.')

    partners_picked_by = 'named'
    if not partner_sets:
        others = [r for r in pool if r != target and r in partitions]
        others.sort(key=lambda r: (still_tied_with(target, r), r))
        partner_sets = [tuple(others[:2])]
        partners_picked_by = ('the two representations that leave fewest of '
                              'the target\'s molecules sharing a row, '
                              'measured on the loaded matrices')
    partner_sets = [tuple(ps) for ps in partner_sets]
    for ps in partner_sets:
        if len(ps) != 2:
            raise RuntimeError(f'{dataset_name}: each set needs two partners, '
                               f'got {ps}')
        if target in ps:
            raise RuntimeError(
                f'{dataset_name}: the target {target} is also a partner in '
                f'{ps}. Name the target or the pairs with --target / '
                f'--partner-sets so the two differ.')
        for r in ps:
            if r not in partitions:
                raise RuntimeError(
                    f'{dataset_name}: {r} has no matrix here. Build it first.')

    used = {target} | {r for ps in partner_sets for r in ps}
    if need_donor and donor is None:
        cands = [r for r in donors if r in partitions and r not in used]
        if not cands:
            raise RuntimeError(
                f'{dataset_name}: no augmentation representation left to be '
                f'the donor (DATASET_REPS[{dataset_name!r}][\'augmentation\'] '
                f'minus the target and partners).')
        donor = min(cands, key=lambda r: (still_tied_with(target, r), r))

    _, sizes_t = census.row_partition(reps[target])

    def still_tied_with_set(names):
        inv = partitions[target]
        for r in names:
            inv, sizes = census.joint_partition(inv, partitions[r])
        return int((sizes > 1).sum())

    choice = {
        'target': target, 'partner_sets': [list(ps) for ps in partner_sets],
        'donor': donor,
        'chosen_on': 'every row of the dataset, before folds -- picks which '
                     'comparisons run, not any score',
        'target_picked_by': ('named' if not merged else
                             'most molecules merged by the representation, '
                             'measured on the loaded matrices'),
        'partner_sets_picked_by': partners_picked_by,
        'molecules_merged_per_representation': merged or None,
        'target_molecules_sharing_a_row': int((sizes_t > 1).sum()),
        'still_tied_target_plus_each_partner': {
            p: still_tied_with(target, p)
            for ps in partner_sets for p in ps},
        'still_tied_with_each_whole_set': {
            '+'.join(ps): still_tied_with_set(ps) for ps in partner_sets},
        'still_tied_target_plus_donor': (still_tied_with(target, donor)
                                         if donor is not None else None),
    }
    return choice


# ===== TIE COUNTING ON A BUILT MATRIX ====================================

def tie_stats(X, decimals=None):
    """How many rows of this matrix repeat another row, and in how many groups.

    Rows are compared in the matrix's own precision. Before 2026-10-05 every
    matrix was converted to float64 first; float32 is now the precision of
    every representation (answer 18), and two float32 rows are identical
    exactly when their float64 copies are, so nothing is converted.
    """
    A = np.asarray(X)
    if decimals is not None:
        A = np.round(A, decimals)
    A = np.ascontiguousarray(A)
    keys = [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in A]
    counts = Counter(keys)
    inv = np.empty(len(keys), dtype=np.int64)
    ids = {}
    for i, k in enumerate(keys):
        inv[i] = ids.setdefault(k, len(ids))
    return {'n_rows': int(A.shape[0]), 'n_features': int(A.shape[1]),
            'n_tied_rows': int(sum(c for c in counts.values() if c > 1)),
            'n_tied_groups': int(sum(1 for c in counts.values() if c > 1)),
            '_inv': inv}


def stack_fold_matrix(n_rows, tr, te, X_tr, X_te):
    """The fold's training and held-out rows of one built matrix, in row order."""
    X_fold = np.empty((n_rows, X_tr.shape[1]),
                      dtype=np.result_type(X_tr, X_te))
    X_fold[tr] = X_tr
    X_fold[te] = X_te
    return X_fold


def rebuild_fold_matrix(reps, augs, labels, groups, tr, te, blueprint,
                        model_name, task_type, metric_fn, lower_is_better):
    """Rebuild the hybrid the engine scored on this outer fold, all rows.

    Arms B and D only (scored by nested_cv_evaluate). This mirrors
    find_best_hybrid.py:10955-11000 call for call, so the matrix counted here
    is the matrix that produced the score. apply_blueprint is used rather than
    build_fold_hybrid because apply_blueprint subsets to the blueprint's own
    representation list and build_fold_hybrid does not.
    """
    from kirby.find_best_hybrid import Blueprint, apply_blueprint, get_model
    from kirby.hybrid import apply_feature_selection, apply_augmentation_selection

    train_reps = {r: X[tr] for r, X in reps.items()}
    test_reps = {r: X[te] for r, X in reps.items()}
    train_augs = {r: X[tr] for r, X in augs.items()} if augs else None
    test_augs = {r: X[te] for r, X in augs.items()} if augs else None
    y_train = labels[tr]
    train_groups = groups[tr] if groups is not None else None

    rb_model = get_model(model_name, task_type, n_samples=len(tr))
    if metric_fn is None:
        rb_model = None
    X_tr_h, finfo = apply_blueprint(
        train_reps, blueprint, labels=y_train, augmentations=train_augs,
        groups=train_groups, model=rb_model,
        metric_fn=(metric_fn if rb_model is not None else None),
        lower_is_better=lower_is_better, row_index=np.asarray(tr),
        structural_pairs=None, is_clf=(task_type == 'classification'))

    bp = (blueprint if isinstance(blueprint, Blueprint)
          else Blueprint.from_dict(blueprint))
    te_bp = {k: test_reps[k] for k in bp.reps if k in test_reps}
    X_te_h = apply_feature_selection(te_bp, finfo, base_only=True)

    aug_info = finfo.get('augmentation_info')
    if aug_info:
        _te = dict(test_augs) if test_augs else {}
        for name, idx in (finfo.get('demoted_reps') or {}).items():
            if name in test_reps:
                _te[name] = test_reps[name][:, idx]
        if _te:
            block = apply_augmentation_selection(_te, aug_info)
            if block is not None:
                X_te_h = np.hstack([X_te_h, block])

    # The matrix's own precision, not float64 (answer 18, 2026-10-05).
    return stack_fold_matrix(labels.size, tr, te, X_tr_h, X_te_h), finfo


def fold_tie_report(reps, X_fold, finfo, target, labels, task_type,
                    metric_name, decimals=None):
    """The three tie counts for one fold, each with the ceiling it implies."""
    out = {}
    full = tie_stats(reps[target], decimals)
    out['target_full_width'] = _with_ceiling(full, labels, task_type, metric_name)

    entry = finfo.get(target)
    if isinstance(entry, dict) and 'selected_indices' in entry:
        idx = np.asarray(entry['selected_indices'], dtype=int)
        kept = tie_stats(reps[target][:, idx], decimals) if idx.size else None
    else:
        kept = None
    out['target_columns_the_hybrid_kept'] = (
        _with_ceiling(kept, labels, task_type, metric_name) if kept else
        {'note': 'the hybrid gave this representation no features'})

    built = tie_stats(X_fold, decimals)
    out['built_hybrid'] = _with_ceiling(built, labels, task_type, metric_name)
    return out


def _with_ceiling(stats, labels, task_type, metric_name):
    inv = stats.pop('_inv')
    ceiling = census.metric_ceiling(inv, labels, task_type, metric_name)
    ceiling.pop('n_molecules', None)
    stats['percent_tied'] = round(100.0 * stats['n_tied_rows'] /
                                  max(stats['n_rows'], 1), 3)
    stats['ceiling'] = ceiling
    return stats


# ===== ARMS C (AND A): THE PER-FOLD STANDARD SELECTION ===================

def arm_namespace(fold_i, n_outer, arm, pool_reps):
    """The per-fold search store name for one arm's pool.

    Representations are not part of the stored-result key, so each pool's
    searches are filed under the fold's nested-CV namespace plus the arm and
    its representation names. Without it C2's search would read C1's rows,
    and a run with different partners would read an older run's.
    """
    from kirby.find_best_hybrid import nested_cv_cache_namespace
    return (f'{nested_cv_cache_namespace(fold_i, n_outer, SEED)}'
            f'__shvec_{arm}_{"+".join(sorted(pool_reps))}')


def _heldout_score(X_tr, X_te, y_tr, y_te, model_name, task_type, metric_fn):
    """Fit on the fold's training rows, score the held-out rows.

    The same scorer nested_cv_evaluate uses (find_best_hybrid._outer_fit_score):
    FeatureScaler (0/1 features unscaled, the rest standardised; answer 17)
    and missing-value filling fitted on the training rows only, the engine's
    default model for that name (svm with standardised labels). Returns
    (score, None) or (None, why).
    """
    from kirby.find_best_hybrid import _outer_fit_score
    try:
        s, _ = _outer_fit_score(X_tr, y_tr, X_te, y_te, model_name, task_type,
                                metric_fn, seed=SEED)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    if s is None or not np.isfinite(s):
        return None, 'the fit returned no finite score (model unavailable or failed)'
    return float(s), None


def _fold_search_failures(pool_reps, labels, groups, tr, k, n_outer,
                          dataset_name, task_type, metric_name, domain, ns):
    """The recorded failures of the search fold_standard_selection just ran.

    Reads the in-process memo with the arguments fold_standard_selection
    passed (general tier, no models), so nothing is searched again.
    """
    from experiments._shared import fold_searches
    try:
        srch = fold_searches(
            pool_reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            task_type=task_type, metric=metric_name, domain=domain,
            seed=SEED, cache_namespace=ns)
        return [{kk: (vv if isinstance(vv, (int, float, str, type(None)))
                      else str(vv)) for kk, vv in f.items()}
                for f in (getattr(srch, 'failures', None) or [])]
    except Exception as e:  # reading the memo must not stop the fold
        return [{'error': f'could not read the failure record: '
                          f'{type(e).__name__}: {e}'}]


def run_arm_fold(arm, pool_reps, data, dataset_name, folds, k, models, metric,
                 target, args):
    """One outer fold of one C arm: A, 2 base representations, 2 hybrids.

    The search runs on this fold's training rows over the arm's pool
    (fold_standard_selection, general search, decided grid, pruning off,
    augmentation 'none' because no donor is passed). Every slot is fitted on
    the training rows and scored on the held-out rows. A failed search, or a
    model short of 2 base representations or 2 hybrids, is recorded with
    dataset, arm, fold, model and the reason; the missing slots are left
    empty, never filled.

    Returns the fold record that goes into the result file.
    """
    from experiments._shared import (fold_standard_selection,
                                     build_fold_hybrid,
                                     fold_hybrid_scoring_kwargs)
    from kirby.find_best_hybrid import get_model

    metric_name, metric_fn, lower = metric
    labels = np.asarray(data['labels'])
    groups = (np.asarray(data['groups']) if data.get('groups') is not None
              else None)
    task_type = data['task_type']
    n_outer = len(folds)
    tr, te = folds[k]
    ns = arm_namespace(k, n_outer, arm, pool_reps)

    rec = {'dataset': dataset_name, 'arm': arm, 'fold': int(k),
           'n_outer_folds': n_outer,
           'n_train': int(len(tr)), 'n_heldout': int(len(te)),
           'cache_namespace': ns, 'search_tier': 'general',
           'representations': sorted(pool_reps),
           'target_representation': target,
           'selection_scope': 'fold_train_rows',
           'metric': metric_name, 'lower_is_better': bool(lower),
           'search_failures': [], 'shortfall': [], 'models': {}}

    # A search that RAISES (an undecided grid, a failed build inside the
    # identical-features check) is recorded like one that returned nothing:
    # one failure per model with dataset, arm, fold and the error, and the
    # fold's slots stay empty (R2/R6, 2026-10-05). It used to stop the whole
    # script, so later arms and datasets never ran and the failing fold was
    # never written.
    try:
        sel = fold_standard_selection(
            pool_reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name, models=models,
            task_type=task_type, metric=metric_name,
            domain=data.get('domain'), seed=SEED, cache_namespace=ns)
    except Exception as exc:
        err = f'{type(exc).__name__}: {exc}'
        for m in models:
            rec['search_failures'].append({
                'dataset': dataset_name, 'arm': arm, 'fold': int(k),
                'model': m, 'errors': [{'error': err}]})
        print(f'    {arm} fold {k} SEARCH RAISED, slots skipped for '
              f'{", ".join(models)}: {err}', flush=True)
        return rec

    failed = sorted(m for m, s in sel.items() if s is None)
    if failed:
        errs = _fold_search_failures(pool_reps, labels, groups, tr, k, n_outer,
                                     dataset_name, task_type, metric_name,
                                     data.get('domain'), ns)
        for m in failed:
            rec['search_failures'].append({
                'dataset': dataset_name, 'arm': arm, 'fold': int(k),
                'model': m, 'errors': errs})
            print(f'    {arm} fold {k} {m:8s} SEARCH FAILED, slots skipped: '
                  f'{[e.get("error") or e.get("error_type") for e in errs]}',
                  flush=True)

    tr_reps = {r: np.asarray(X)[tr] for r, X in pool_reps.items()}
    te_reps = {r: np.asarray(X)[te] for r, X in pool_reps.items()}
    y_tr, y_te = labels[tr], labels[te]

    for m, s in sel.items():
        if s is None:
            continue
        mrec = {'slots': {}}
        # A: the target alone, same fold, same scorer.
        sc, why = _heldout_score(tr_reps[target], te_reps[target], y_tr, y_te,
                                 m, task_type, metric_fn)
        mrec['target_alone'] = {'rep': target, 'heldout_score': sc,
                                'error': why}

        bd = s.get('base_detail') or {}
        base_sel = bd.get('selected') or []
        if len(base_sel) < 2 or len(s['hybrids']) < 2:
            rec['shortfall'].append({
                'dataset': dataset_name, 'arm': arm, 'fold': int(k),
                'model': m, 'n_base': len(base_sel),
                'n_hybrids': len(s['hybrids']),
                'reason': bd.get('shortfall_reason')
                          or 'fewer than 2 hybrids with distinct selected '
                             'features in this fold\'s search'})

        for i, e in enumerate(base_sel, 1):
            rep = e['rep']
            sc, why = _heldout_score(tr_reps[rep], te_reps[rep], y_tr, y_te,
                                     m, task_type, metric_fn)
            mrec['slots'][f'base{i}'] = {
                'rep': rep,
                # Mean over the search's inner folds, inside this outer fold's
                # training rows.
                'training_rows_score': e.get('score'),
                'heldout_score': sc, 'error': why}

        for i, e in enumerate(s['hybrids'], 1):
            bp = e['blueprint']
            row = {'label': e.get('label'),
                   'allocation_method': e.get('allocation_method'),
                   'importance_method': e.get('importance_method'),
                   'budget_features': e.get('budget'),
                   'internal_evaluator': e.get('internal_evaluator'),
                   'representation_count': e.get('rep_count'),
                   'augmentation_condition': e.get('aug_condition'),
                   'training_rows_score': e.get('score_mean'),
                   'heldout_score': None, 'error': None, 'ties': None}
            try:
                X_tr_h, X_te_h, fi = build_fold_hybrid(
                    tr_reps, te_reps, y_tr, bp,
                    groups=(groups[tr] if groups is not None else None),
                    # This fold's search measured these on this fold's
                    # training rows; reused so the budget split is the one
                    # that search settled on (dirty-run bug 1).
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=get_model(m, task_type, n_samples=len(tr)),
                        metric_fn=metric_fn, lower_is_better=lower))
            except Exception as ex:
                row['error'] = f'build failed: {type(ex).__name__}: {ex}'
                print(f'    {arm} fold {k} {m:8s} hybrid{i} {row["error"]}',
                      flush=True)
                mrec['slots'][f'hybrid{i}'] = row
                continue
            row['features_by_rep'] = {
                r: int(len(info['selected_indices']))
                for r, info in fi.items()
                if isinstance(info, dict) and 'selected_indices' in info}
            row['n_features'] = int(np.asarray(X_tr_h).shape[1])
            sc, why = _heldout_score(X_tr_h, X_te_h, y_tr, y_te, m, task_type,
                                     metric_fn)
            row['heldout_score'], row['error'] = sc, why
            if not args.no_ties:
                X_fold = stack_fold_matrix(labels.size, tr, te,
                                           np.asarray(X_tr_h),
                                           np.asarray(X_te_h))
                ties = fold_tie_report(pool_reps, X_fold, fi, target, labels,
                                       task_type, metric_name,
                                       args.tie_decimals)
                ties['fold'] = int(k)
                ties['hybrid_width'] = int(X_fold.shape[1])
                row['ties'] = ties
            mrec['slots'][f'hybrid{i}'] = row
        mrec['hybrids_skipped_identical_features'] = [
            x.get('label') for x in
            (s.get('hybrids_skipped_identical_features') or [])]
        rec['models'][m] = mrec

        def _fmt(v):
            return '—' if v is None else f'{v:.4f}'
        print(f'    {arm} fold {k} {m:8s} {metric_name}: {target} alone '
              f'{_fmt(mrec["target_alone"]["heldout_score"])}; '
              + '; '.join(f'{slot} {_fmt(mrec["slots"][slot]["heldout_score"])}'
                          for slot in C_SLOTS if slot in mrec['slots']),
              flush=True)
    return rec


def _spread(values):
    """Standard deviation over folds, or the one-fold note."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return ONE_FOLD_NOTE if len(vals) == 1 else None
    return float(np.std(vals, ddof=1))


def arm_rows(arm, fold_recs, target, metric_name, lower):
    """One row per (model, slot) of one C arm, over the folds finished so far.

    Each row carries the held-out score per fold, the target alone on the same
    folds, the paired difference signed so positive is better, and for a
    hybrid the recipe and the three tie counts per fold.
    """
    fold_recs = sorted(fold_recs, key=lambda r: r['fold'])
    models = sorted({m for r in fold_recs for m in r['models']})
    rows = []
    for m in models:
        for slot in C_SLOTS:
            folds, scores, base_scores, reps, recipes, ties = [], [], [], [], [], []
            for r in fold_recs:
                mr = r['models'].get(m)
                if not mr or slot not in mr['slots']:
                    continue
                e = mr['slots'][slot]
                folds.append(r['fold'])
                scores.append(e.get('heldout_score'))
                base_scores.append(mr['target_alone'].get('heldout_score'))
                reps.append(e.get('rep'))
                if slot.startswith('hybrid'):
                    recipes.append({kk: e.get(kk) for kk in (
                        'label', 'allocation_method', 'importance_method',
                        'budget_features', 'internal_evaluator',
                        'representation_count', 'augmentation_condition',
                        'n_features', 'features_by_rep',
                        'training_rows_score')})
                    ties.append(e.get('ties'))
            if not folds:
                continue
            paired = [i for i, (h, b) in enumerate(zip(scores, base_scores))
                      if h is not None and b is not None]
            deltas = [((base_scores[i] - scores[i]) if lower
                       else (scores[i] - base_scores[i])) for i in paired]
            rows.append({
                'comparison': f'{arm}__{slot}', 'arm': arm, 'slot': slot,
                'search': 'general', 'sweep_point': None, 'model': m,
                'condition_suffix': None,
                'metric_name': metric_name, 'lower_is_better': bool(lower),
                'target_representation': target,
                'folds': folds,
                'slot_fold_scores': scores,
                'target_alone_fold_scores': base_scores,
                'base_representation_per_fold': (reps if slot.startswith('base')
                                                  else None),
                'recipes_per_fold': (recipes if slot.startswith('hybrid')
                                     else None),
                'paired_folds': [folds[i] for i in paired],
                'delta_vs_target_alone_per_fold': deltas,
                'delta_vs_target_alone_mean': (float(np.mean(deltas))
                                               if deltas else None),
                'delta_vs_target_alone_spread': _spread(deltas),
                'n_paired_folds': len(deltas),
                'ties_per_fold': ties,
                'donor_features_admitted': None,
                'no_op': None,
                'pool_representations': sorted(fold_recs[0]['representations']),
                'selection_scope': 'fold_train_rows',
            })
    return rows


# ===== ARMS B AND D: THE AUGMENTATION SWEEP ==============================

def run_comparison(name, data, dataset_name, reps, donor_reps, n_outer, args,
                   models, point=None, verbose=True):
    """One nested_cv_evaluate call for one sweep point of arm B or D.

    Only reached where Step 8 keeps augmentation (answer 4): the distance
    search (AUG_ARMS_TIER) of ginkgo_colloidal and ginkgo_hic. The search is on
    the dataset's decided grid (through _shared._apply_dataset_search_speed
    with the real dataset name): its models (plus knn, Step 9), budgets,
    importance methods, allocation methods and internal evaluators.

    models: the downstream models named to the search, which narrows the
    decided list to them. main passes the dataset's two general decided
    models, without knn unless B_D_INCLUDE_KNN (2026-10-05).

    WHY NOT THE STANDARD PATH. fold_standard_selection / search_on_fold pass
    no augmentation settings beyond the condition's name, and this sweep is
    over candidate_ranking, aug_per_source_budget, augmentation_margin,
    augmentation_fraction and augmentation_budget. So B and D call
    nested_cv_evaluate directly, as before.
    """
    from experiments._shared import (_apply_dataset_search_speed,
                                     _resolve_search_metric)
    from kirby.find_best_hybrid import (nested_cv_evaluate,
                                        nested_cv_outer_folds)

    labels = np.asarray(data['labels'])
    groups = data['groups']
    kwargs = dict(engine_kwargs(point or {'augmentation_strategy': 'none'}))
    suffix = point_slug(point) if point else None

    metric, task_type = _resolve_search_metric(
        labels, task_type=data['task_type'], metric='auto',
        domain=data.get('domain'), dataset_name=dataset_name)
    search = dict(
        tier=AUG_ARMS_TIER, models=list(models),
        task_type=task_type, metric=metric, cache=True,
        # rep_counts is pinned to the whole pool. A blueprint records the pool it
        # was chosen from, not the subset a rep_count picked, and
        # nested_cv_evaluate's outer rebuild re-applies the pool -- so a searched
        # rep_count would make the honest score describe a wider feature set than
        # the inner search chose.
        rep_counts=[None],
        # tune=None reads KIRBY_TUNE from the environment, which would silently
        # tune this run if it happens to be set in the shell.
        tune=False,
        condition_suffix=suffix, **kwargs)
    _apply_dataset_search_speed(search, dataset_name)

    result = nested_cv_evaluate(
        base_reps=reps, labels=labels, groups=groups,
        n_outer_folds=n_outer, seed=SEED, verbose=verbose,
        dataset_name=dataset_name,
        augmentations=(donor_reps or None), **search)

    outer_folds = nested_cv_outer_folds(
        reps, labels, groups, n_outer, SEED, task_type)
    if len(outer_folds) != result.n_outer_folds:
        raise RuntimeError(
            f'{dataset_name} {name}: regenerated {len(outer_folds)} outer folds '
            f'but the run scored {result.n_outer_folds}. The splitter reduces '
            f'n_splits to the unique-group count, so the two must be read from '
            f'the same call.')
    return result, outer_folds


def summarise_comparison(name, point, result, outer_folds, data, reps, donor_reps,
                         target, models, args):
    """Scores, paired deltas against A, and the tie counts per fold (B and D).

    Per decided model (models; knn only if B_D_INCLUDE_KNN) and outer fold:
    the 2 best hybrids of that fold's inner search (top_hybrids on
    result.inner_searches[i], ranked by its mean over inner folds of the
    outer fold's training rows). Each is rebuilt on the fold's training rows
    (rebuild_fold_matrix, the engine's own path) and scored on its held-out
    rows by the same scorer as C. A lower-ranked hybrid whose selected
    features equal one already taken is skipped for the next (answer 8).
    Until 2026-10-05 B and D reported one hybrid per model, the winner
    nested_cv_evaluate picked, and every model it scored, knn included.

    The rows carry the arm's name with a 'd' appended: their recipes come
    from the distance search, the only search Step 8 keeps augmentation on.
    """
    from experiments._shared import (top_hybrids, hybrid_feature_signature,
                                     STANDARD_N_HYBRIDS)
    labels = np.asarray(data['labels'])
    groups = data['groups']
    task_type = data['task_type']
    lower = bool(result.lower_is_better)
    metric_name = result.metric_name
    from kirby.find_best_hybrid import _resolve_metric
    metric_fn, _lb = _resolve_metric(metric_name)
    inner = list(result.inner_searches or [])
    n_folds = result.n_outer_folds
    stem = f'{name}d'

    rows = []
    for model in models:
        entry = (result.by_model or {}).get(model) or {}
        base = ((result.baselines_by_model or {}).get(model, {})
                .get(target, {}).get('fold_scores') or [None] * n_folds)
        engine_winner = entry.get('hybrid_fold_scores') or [None] * n_folds
        # per slot, per fold
        slots = {i: {'scores': [None] * n_folds, 'recipes': [None] * n_folds,
                     'ties': [None] * n_folds, 'errors': [None] * n_folds}
                 for i in range(1, STANDARD_N_HYBRIDS + 1)}
        missing, skipped_identical = [], []
        for fi_ in range(n_folds):
            s = inner[fi_] if fi_ < len(inner) else None
            if s is None:
                missing.append(f'outer fold {fi_}: the inner search failed')
                continue
            th = top_hybrids(s, model, n=STANDARD_N_HYBRIDS + 4,
                             require_full=False,
                             context=f'opig_shared_vectors_remedy {stem} '
                                     f'fold {fi_} model {model}')
            tr, te = outer_folds[fi_]
            seen, slot = {}, 1
            for e in th['selected']:
                if slot > STANDARD_N_HYBRIDS:
                    break
                bp = e.get('blueprint')
                rec = _recipe_record(bp) if bp is not None else {}
                rec.update({'label': e.get('label'),
                            'training_rows_score': e.get('score_mean')})
                if bp is None:
                    slots[slot]['recipes'][fi_] = rec
                    slots[slot]['errors'][fi_] = 'the search row has no recipe'
                    slot += 1
                    continue
                try:
                    X_fold, finfo = rebuild_fold_matrix(
                        reps, donor_reps, labels, groups, tr, te, bp, model,
                        task_type, metric_fn, lower)
                except Exception as exc:
                    slots[slot]['recipes'][fi_] = rec
                    slots[slot]['errors'][fi_] = (
                        f'build failed: {type(exc).__name__}: {exc}')
                    slot += 1
                    continue
                sig = hybrid_feature_signature(finfo)
                if sig in seen:
                    skipped_identical.append({
                        'fold': fi_, 'label': e.get('label'),
                        'same_features_as': seen[sig]})
                    continue
                seen[sig] = e.get('label')
                sc, why = _heldout_score(X_fold[tr], X_fold[te], labels[tr],
                                         labels[te], model, task_type,
                                         metric_fn)
                rec['n_features'] = int(X_fold.shape[1])
                rec['donor_features_admitted'] = list(
                    (finfo.get('augmentation_info') or {}).keys())
                slots[slot]['scores'][fi_] = sc
                slots[slot]['errors'][fi_] = why
                slots[slot]['recipes'][fi_] = rec
                if not args.no_ties:
                    t = fold_tie_report(reps, X_fold, finfo, target, labels,
                                        task_type, metric_name,
                                        args.tie_decimals)
                    t['fold'] = fi_
                    t['kept_augmentations'] = rec['donor_features_admitted']
                    t['hybrid_width'] = int(X_fold.shape[1])
                    slots[slot]['ties'][fi_] = t
                slot += 1
            if slot <= STANDARD_N_HYBRIDS:
                missing.append(f'outer fold {fi_}: {slot - 1} hybrid(s) with '
                               f'distinct selected features in the inner search')

        for i, sl in slots.items():
            scores = sl['scores']
            paired = [j for j, (h, b) in enumerate(zip(scores, base))
                      if h is not None and b is not None
                      and np.isfinite(h) and np.isfinite(b)]
            deltas = [((base[j] - scores[j]) if lower
                       else (scores[j] - base[j])) for j in paired]
            admitted = sorted({d for r in sl['recipes'] if r
                               for d in (r.get('donor_features_admitted') or [])})
            built_any = any(r and 'n_features' in r for r in sl['recipes'])
            rows.append({
                'comparison': f'{stem}__hybrid{i}', 'arm': name,
                'slot': f'hybrid{i}', 'search': AUG_ARMS_TIER,
                'sweep_point': point, 'model': model,
                'condition_suffix': point_slug(point) if point else None,
                'metric_name': metric_name, 'lower_is_better': lower,
                'n_outer_folds': n_folds,
                'target_representation': target,
                'slot_fold_scores': scores,
                'slot_fold_errors': sl['errors'],
                'target_alone_fold_scores': base,
                # The held-out score nested_cv_evaluate took for its own
                # winner, beside hybrid1's: the two should agree when the
                # winner is the top row of the inner search.
                'engine_winner_fold_scores': (engine_winner if i == 1
                                              else None),
                'paired_folds': paired,
                'delta_vs_target_alone_per_fold': deltas,
                'delta_vs_target_alone_mean': (float(np.mean(deltas))
                                               if deltas else None),
                'delta_vs_target_alone_spread': _spread(deltas),
                'n_paired_folds': len(deltas),
                'ties_per_fold': sl['ties'],
                'donor_features_admitted': admitted,
                'no_op': (point is not None
                          and point.get('augmentation_strategy') != 'none'
                          and built_any and not admitted),
                'pool_representations': sorted(reps),
                'recipes_per_fold': sl['recipes'],
                'hybrids_skipped_identical_features': (skipped_identical
                                                       if i == 1 else []),
                'selection_scope': 'fold_train_rows',
                'rule_missing': ([f'{stem} hybrids for {model}: {w}'
                                  for w in missing] if i == 1 else []),
            })
    rows.extend(_base_rows(name, point, result, reps, target, models))
    return rows


def _recipe_record(bp):
    """What one fold's hybrid was: allocation, importance, budget, evaluator,
    the representations it was built from (the whole pool: rep_counts is
    pinned to it) and the augmentation setting."""
    if bp is None:
        return None
    from kirby.find_best_hybrid import Blueprint
    b = bp if isinstance(bp, Blueprint) else Blueprint.from_dict(bp)
    return {'allocation_method': b.allocation_method,
            'importance_method': b.importance_method,
            'budget_features': b.budget,
            'internal_evaluator': b.internal_evaluator,
            'representations': list(b.reps),
            'representation_count': 'whole pool',
            'augmentation_condition': b.augmentation_strategy}


def _base_rows(name, point, result, reps, target, models):
    """The 2 best base representations per model per outer fold (B and D).

    Each fold's two are ranked by that fold's inner search -- its own mean over
    inner folds of the outer fold's training rows (top_base_reps on
    result.inner_searches[i]) -- and then read off the score nested_cv_evaluate
    already took for every representation on the held-out fold. No new fit.
    The paired delta is against the target representation alone, as in the
    hybrid rows. A pool smaller than two (arm B: the target alone) yields one
    base row and says so in rule_missing.
    """
    from experiments._shared import top_base_reps, STANDARD_N_BASE
    lower = bool(result.lower_is_better)
    inner = list(result.inner_searches or [])
    n_folds = result.n_outer_folds
    stem = f'{name}d'
    rows = []
    # Only the models main asked for (knn left out unless B_D_INCLUDE_KNN,
    # 2026-10-05); it used to be every model the search scored.
    for model in models:
        per_rep = (result.baselines_by_model or {}).get(model, {})
        target_scores = (per_rep.get(target) or {}).get('fold_scores') or []
        picks = [[None] * n_folds for _ in range(STANDARD_N_BASE)]
        missing = []
        for i in range(n_folds):
            s = inner[i] if i < len(inner) else None
            if s is None:
                missing.append(f'outer fold {i}: the inner search failed')
                continue
            sel = top_base_reps(s, model, n=STANDARD_N_BASE, require_full=False,
                                context=f'opig_shared_vectors_remedy {stem} '
                                        f'fold {i} model {model}')
            if not sel['complete']:
                missing.append(f'outer fold {i}: {sel["shortfall_reason"]}')
            for k, e in enumerate(sel['selected']):
                picks[k][i] = e['rep']
        for k in range(STANDARD_N_BASE):
            reps_k = picks[k]
            if not any(reps_k):
                continue
            scores = [((per_rep.get(r) or {}).get('fold_scores') or [None] * n_folds)[i]
                      if r else None for i, r in enumerate(reps_k)]
            paired = [i for i, (h, b) in enumerate(zip(scores, target_scores))
                      if h is not None and b is not None
                      and np.isfinite(h) and np.isfinite(b)]
            deltas = [((target_scores[i] - scores[i]) if lower
                       else (scores[i] - target_scores[i])) for i in paired]
            rows.append({
                'comparison': f'{stem}__base{k + 1}',
                'arm': name, 'slot': f'base{k + 1}', 'search': AUG_ARMS_TIER,
                'sweep_point': point, 'model': model,
                'condition_suffix': point_slug(point) if point else None,
                'metric_name': result.metric_name, 'lower_is_better': lower,
                'n_outer_folds': n_folds,
                'target_representation': target,
                'base_representation_per_fold': reps_k,
                'slot_fold_scores': scores,
                'target_alone_fold_scores': target_scores,
                'paired_folds': paired,
                'delta_vs_target_alone_per_fold': deltas,
                'delta_vs_target_alone_mean': (float(np.mean(deltas))
                                               if deltas else None),
                'delta_vs_target_alone_spread': _spread(deltas),
                'n_paired_folds': len(deltas),
                'ties_per_fold': [],
                'donor_features_admitted': None,
                'no_op': None,
                'pool_representations': sorted(reps),
                'selection_scope': 'fold_train_rows',
                # Written once, on base1, so the per-dataset list does not
                # repeat each fold's gap.
                'rule_missing': ([f'{stem} base ranking for {model}: {w}'
                                  for w in missing] if k == 0 else []),
            })
    return rows


# ===== ANALYSIS ===========================================================

def mean_of(rows_ties, where, field):
    vals = [t[where][field] for t in rows_ties
            if t and isinstance(t.get(where), dict) and field in t[where]]
    return float(np.mean(vals)) if vals else None


def print_headline(dataset_name, choice, rows, models_common):
    print('\n' + '=' * 110)
    print(f'{dataset_name}: does the fix move BOTH numbers, on the same folds?')
    sets = ' | '.join('+'.join(ps) for ps in choice['partner_sets'])
    print(f'target = {choice["target"]}   partner sets = {sets}'
          f'   donor = {choice["donor"] or "(none: B and D not run)"}')
    print('score delta is the row minus the target representation alone, signed so '
          'positive is better; mean over folds.')
    print('tied rows counts molecules sharing a row in the matrix that fold was '
          'scored on, mean over folds.')
    print('=' * 110)
    for model in sorted(models_common):
        print(f'\n  model {model}')
        print(f'    {"which":14s} {"setting":30s} {"score delta":>12s} '
              f'{"folds":>6s} {"tied: full":>11s} {"kept feats":>10s} '
              f'{"hybrid":>8s}')
        for r in rows:
            if r['model'] != model:
                continue
            ties = [t for t in (r['ties_per_fold'] or []) if t]
            full = mean_of(ties, 'target_full_width', 'n_tied_rows')
            kept = mean_of(ties, 'target_columns_the_hybrid_kept', 'n_tied_rows')
            built = mean_of(ties, 'built_hybrid', 'n_tied_rows')
            d = r['delta_vs_target_alone_mean']
            label = r['condition_suffix'] or '-'
            if r['no_op']:
                label += '  [no-op: no donor features admitted]'
            print(f'    {r["comparison"]:14s} {label[:30]:30s} '
                  f'{"—" if d is None else f"{d:+12.4f}"} '
                  f'{r["n_paired_folds"]:6d} '
                  f'{"—" if full is None else f"{full:11.1f}"} '
                  f'{"—" if kept is None else f"{kept:10.1f}"} '
                  f'{"—" if built is None else f"{built:8.1f}"}')


def write_output(path, rows, per_dataset, per_fold):
    """Write the whole result file now (after every fold and every sweep
    point, dirty-run bug 12). Written to a temporary name and moved, so a job
    killed mid-write leaves the previous complete file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(convert_numpy({'result_layout': RESULT_LAYOUT,
                                 'rows': rows, 'per_dataset': per_dataset,
                                 'per_fold': per_fold}), f, indent=1)
    os.replace(tmp, path)


# ===== IS THE CACHED MATRIX WHAT THE BUILDER RETURNS TODAY? ==============

# Added 2026-10-05. census.freshness_check rebuilds only the representations
# in its own map (ecfp4, maccs, avalon, atom_pair, top_torsion, rdkfp, pdv,
# chemberta, chembert) and passes over every other one without a word. On
# herg_ki the target 'auto' resolves to graph_kernel (29 Sep census) and the
# second pair uses mhg_gnn, so the two representations the tie counts depend
# on most were never compared with their builders -- the same blind spot
# that let the dirty run count ties on a chemberta matrix that had none.
#
#   graph_kernel  rebuilt on EVERY row: its features are the WL labels found
#                 across all molecules, minus those absent from over 95% of
#                 them, so a sample of rows yields different features.
#   mhg_gnn       a frozen encoder applied one molecule at a time, so a
#                 sample of rows is enough, as for chembert.
_WHOLE_MATRIX_BUILDERS = {'graph_kernel': 'create_graph_kernel'}
_SAMPLE_BUILDERS = {'mhg_gnn': 'create_mhg_gnn'}


def _compare(fresh, cached, rows_checked, tol):
    if fresh.shape != cached.shape:
        return {'checked': True, 'rows_checked': int(rows_checked),
                'cached_shape': list(cached.shape),
                'rebuilt_shape': list(fresh.shape), 'matches': False,
                'note': 'the rebuilt matrix has a different shape'}
    diff = np.abs(fresh - cached)
    n_bad = int((diff > tol).any(axis=1).sum()) if diff.size else 0
    return {'checked': True, 'rows_checked': int(rows_checked),
            'rows_that_differ': n_bad,
            'largest_difference': float(diff.max()) if diff.size else 0.0,
            'matches': bool(n_bad == 0)}


def cache_check(data, rep_names, n_rows, tol=1e-4):
    """Per representation: does the cached matrix match its builder today?

    Returns {rep: record}. Every name in rep_names gets a record, so a
    representation that was not checked says so in the result file:
      {'checked': True, 'matches': bool, ...}   compared
      {'checked': False, 'error': ...}          the builder raised
      {'checked': False, 'note': ...}           no builder here
    """
    rep_names = list(rep_names)
    smiles = data.get('smiles')
    if smiles is None:
        return {r: {'checked': False,
                    'note': 'no SMILES on this dataset; this script has no '
                            'builder to compare the cache with'}
                for r in rep_names}
    out = {}
    for r, v in (census.freshness_check(data, rep_names, n_rows=n_rows,
                                        tol=tol) or {}).items():
        if r in rep_names and isinstance(v, dict):
            out[r] = dict(v, checked=('matches' in v))
    import kirby.representations.molecular as M
    n = len(smiles)
    idx = np.unique(np.linspace(0, n - 1, min(n_rows, n)).astype(int))
    for r in rep_names:
        if r in out:
            continue
        cached = data['reps'].get(r)
        if cached is None:
            out[r] = {'checked': False, 'note': 'not loaded'}
            continue
        whole = r in _WHOLE_MATRIX_BUILDERS
        fn_name = _WHOLE_MATRIX_BUILDERS.get(r) or _SAMPLE_BUILDERS.get(r)
        if fn_name is None or not hasattr(M, fn_name):
            out[r] = {'checked': False,
                      'note': 'no builder in this check; the cached matrix '
                              'was not compared with the code'}
            continue
        rows = np.arange(n) if whole else idx
        try:
            fresh = np.asarray(getattr(M, fn_name)([smiles[i] for i in rows]),
                               dtype=np.float32)
        except Exception as exc:
            out[r] = {'checked': False,
                      'error': f'{type(exc).__name__}: {exc}'}
            continue
        out[r] = _compare(fresh, np.asarray(cached)[rows], len(rows), tol)
        out[r]['rebuilt'] = 'every row' if whole else 'a sample of rows'
    return out


# ===== MAIN ===============================================================

def _check_on_list(names, allowed, ds, what):
    off = sorted(set(names) - set(allowed))
    if off:
        raise RuntimeError(
            f'{ds}: {what} {", ".join(off)} not on the dataset\'s pruned list '
            f'(DATASET_REPS[{ds!r}]). Only those representations are used '
            f'(2026-10-05). The list: {", ".join(allowed)}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--target', default=DEFAULT_TARGET,
                    help='representation to fix, from the dataset\'s pruned '
                         'list. "auto" (the default) uses the one that merges '
                         'the most genuinely different molecules.')
    ap.add_argument('--partner-sets', nargs='+', dest='partner_sets',
                    default=None,
                    help='one or more comma-separated pairs from the pruned '
                         'list, each hybridised with the target, used on every '
                         'dataset named. Pass "auto" to pick one pair by '
                         'measurement. Default: DEFAULT_PARTNER_SETS for the '
                         'dataset, "auto" where it names none.')
    ap.add_argument('--donor', default=None,
                    help='augmentation representation for B and D (from the '
                         "dataset's augmentation list); only used where B and "
                         'D run.')
    ap.add_argument('--reps', default=None,
                    help='comma-separated narrowing of the pruned list the '
                         'target is picked from; a name off the list raises')
    # There is no --speed and no --budgets: the decided grid sets each
    # dataset's budgets and speed class (decided_grid.apply_decided_grid).
    ap.add_argument('--models', default=None,
                    help='comma-separated downstream models. Default None: the '
                         "dataset's two decided models (DECISIONS.md Step 1). "
                         'A list here is narrowed to those; naming none raises.')
    ap.add_argument('--n-outer-folds', type=int, default=None,
                    dest='n_outer_folds',
                    help='outer held-out folds. Default: the experiment '
                         "config's n_splits (get_config). 1 is a single "
                         'grouped held-out split.')
    ap.add_argument('--compare', nargs='+', default=['A', 'B', 'C', 'D'],
                    choices=['A', 'B', 'C', 'D'],
                    help='which of the four to run (A is scored inside C)')
    ap.add_argument('--tie-decimals', type=int, default=None, dest='tie_decimals',
                    help='round features to this many decimals before comparing '
                         'rows; default compares them exactly')
    ap.add_argument('--no-ties', action='store_true',
                    help='scores only, skip the tie counts')
    ap.add_argument('--skip-cache-check', action='store_true',
                    dest='skip_cache_check',
                    help='run even when a cached matrix no longer matches its '
                         'builder. The result then describes the cache, not the '
                         'pipeline.')
    ap.add_argument('--cache-check-rows', type=int, default=150,
                    dest='cache_check_rows',
                    help='how many rows to rebuild for that check')
    ap.add_argument('--sweep-slice', default=None, dest='sweep_slice',
                    help='K/N — run only every Nth augmentation setting, '
                         'starting at K (B and D only). The whole grid still '
                         'runs; it is spread over N tasks. Nothing is cut. '
                         'Every task writes its own result file, named by the '
                         'slice.')
    ap.add_argument('--fold-spread', type=float, default=None,
                    dest='fold_spread',
                    help='The target representation\'s spread across outer '
                         'folds, in the dataset metric. A margin is a multiple '
                         'of it. Pass the number a C-only run printed and the '
                         'sliced tasks skip re-running C to get it.')
    ap.add_argument('--count-only', action='store_true',
                    help='print the size of the grid and stop')
    ap.add_argument('--smoke', action='store_true',
                    help='B and D: one sweep point per strategy. Folds, '
                         'budgets and models are the normal ones.')
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    ap.add_argument('--force-rerun', action='store_true',
                    help='Recompute what an earlier attempt finished '
                         '(<output>.partial.jsonl).')
    ap = add_distance_metric_arg(ap)
    args = ap.parse_args()
    apply_distance_metric_arg(args)

    args.models = ([m.strip() for m in args.models.split(',') if m.strip()]
                   if args.models else None)

    from experiments._shared import load_data, get_config, get_domain_metric
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.decided_grid import decided_models, decided_aug_conditions
    from kirby.find_best_hybrid import nested_cv_outer_folds

    class _Cfg:
        mode = 'full'
    _cfg = _Cfg()
    # get_config sets the process-wide distance again from its argument's
    # distance_metric, and _Cfg had none, so it reset whatever
    # --distance-metric had set to the default (2026-10-05). It reaches knn
    # and the distance search of B and D.
    _cfg.distance_metric = getattr(args, 'distance_metric', None)
    config = get_config(_cfg)
    # The outer fold count comes from ONE setting: the experiment config's
    # n_splits, unless --n-outer-folds names another (2026-10-05). It used to
    # be a literal default of 5 here.
    n_outer_asked = int(args.n_outer_folds or config['n_splits'])

    # Each finished unit is written to <output>.partial.jsonl and read back on
    # the next attempt (experiments/_checkpoint.py). Since 2026-10-05 the units
    # are (dataset, C arm, fold) and (dataset, B/D arm, sweep point), and the
    # JSON itself is rewritten after every unit. Before, the C arms were never
    # stored and the JSON was written only at the end, so the dirty run's
    # job, killed at its 3-hour cap, saved neither C1 nor C2 (dirty-run bug
    # 12). A unit that raised is recorded as an error row, is not stored, is
    # retried next time, and makes the script exit 1.
    _tag = ''
    if args.sweep_slice:
        _k, _n = (int(x) for x in args.sweep_slice.split('/'))
        if not 0 <= _k < _n:
            raise ValueError(f'--sweep-slice {args.sweep_slice}: need 0 <= K < N')
        _tag = f'_slice{_k}of{_n}'
    out_path = os.path.join(args.output_dir,
                            f'opig_shared_vectors_remedy{_tag}.json')
    from experiments._checkpoint import script_checkpoint, exit_if_failed
    # Not built under --count-only (2026-10-05): count_only was part of the
    # identity, so the count the docstring recommends before submitting
    # opened the sidecar under a different identity and moved every finished
    # fold of the real run aside to .stale. compare and smoke are left out of
    # the identity too: each unit is keyed by its own arm and sweep point, and
    # its result does not depend on which other arms or points the run does.
    _ck = None
    if not args.count_only:
        _ck = script_checkpoint(
            'opig_shared_vectors_remedy', out_path, args,
            exclude=('datasets', 'output_dir', 'count_only', 'compare',
                     'smoke'),
            result_layout=RESULT_LAYOUT)
    _failed = []

    bd_rows, all_meta, c_records = [], [], {}

    def _all_rows():
        rows = []
        for (ds_, arm_), recs in c_records.items():
            if not recs:
                continue
            for r in arm_rows(arm_, recs, recs[0]['target_representation'],
                              recs[0]['metric'], recs[0]['lower_is_better']):
                r['dataset'] = ds_
                rows.append(r)
        return rows + bd_rows

    def _flush():
        if args.count_only:
            return
        write_output(out_path, _all_rows(), all_meta,
                     [r for recs in c_records.values() for r in recs])

    def _partner_sets_for(ds):
        """The pairs for one dataset: --partner-sets, else the dataset's
        DEFAULT_PARTNER_SETS, else None (one pair picked by measurement)."""
        if args.partner_sets is not None:
            if args.partner_sets == ['auto']:
                return None
            return [tuple(x.strip() for x in ps.split(','))
                    for ps in args.partner_sets]
        named_default = DEFAULT_PARTNER_SETS.get(ds)
        return [tuple(ps) for ps in named_default] if named_default else None

    def _named_for(ds):
        named = set()
        if args.target and args.target != 'auto':
            named.add(args.target)
        for ps in (_partner_sets_for(ds) or []):
            named.update(ps)
        return named

    # Every dataset's names are checked against its pruned list BEFORE the
    # first one runs (2026-10-05), so a name off one dataset's list stops
    # the run at start-up, not hours in when that dataset is reached.
    for ds in args.datasets:
        general = list(DATASET_REPS[ds]['general'])
        if args.reps:
            _check_on_list([r.strip() for r in args.reps.split(',')
                            if r.strip()], general, ds, '--reps')
        _check_on_list(_named_for(ds), general, ds,
                       'the named target or partners')

    t0 = time.time()
    for ds in args.datasets:
        entry = DATASET_REPS[ds]
        general = list(entry['general'])
        pool = general
        if args.reps:
            pool = [r.strip() for r in args.reps.split(',') if r.strip()]
        named = _named_for(ds)
        pool = sorted(set(pool) | named)

        models = list(decided_models(ds, 'general'))
        if args.models:
            dropped = [m for m in args.models if m not in models]
            if dropped:
                print(f'{ds}: --models {", ".join(dropped)} not decided for '
                      f'this dataset, left out', flush=True)
            models = [m for m in args.models if m in models]
            if not models:
                raise RuntimeError(f'{ds}: --models names none of the decided '
                                   f'models {decided_models(ds, "general")}')

        # B and D (answer 4, 2026-10-05): only where Step 8 keeps an
        # augmentation condition on the search they run under, and only that
        # condition is swept.
        want_aug = 'B' in args.compare or 'D' in args.compare
        aug_conds = [c for c in decided_aug_conditions(ds, AUG_ARMS_TIER)
                     if c != 'none']
        # The added features left after paratope and structural are taken out
        # (EXCLUDED_DONORS, decided 2026-10-06). With none left there is
        # nothing for B or D to add, so they are skipped with the reason.
        allowed_donors = [r for r in (entry.get('augmentation') or [])
                          if r not in EXCLUDED_DONORS]
        run_aug = want_aug and bool(aug_conds) and bool(allowed_donors)
        aug_skip_reason = None
        if want_aug and not run_aug:
            if aug_conds and not allowed_donors:
                aug_skip_reason = (
                    f'B and D skipped on {ds}: its added-features '
                    f'representations are '
                    f'{", ".join(entry.get("augmentation") or []) or "(none)"}, '
                    f'and none is left to add to the target.')
            else:
                aug_skip_reason = (
                    f'B and D skipped on {ds}: DECISIONS.md Step 8 keeps no '
                    f'augmentation on its distance search (decided conditions: '
                    f'{decided_aug_conditions(ds, AUG_ARMS_TIER)}), so a '
                    f'comparison with added donor features is not run here '
                    f'(answer 4, 2026-10-05).')
            print(f'{ds}: {aug_skip_reason}', flush=True)
        donors = list(allowed_donors) if run_aug else []
        if args.donor and run_aug:
            _check_on_list([args.donor], donors, ds, '--donor')

        want = sorted(set(pool) | set(donors))
        have = census.cached_reps(ds)
        missing = [r for r in want if r not in have]
        if missing:
            print(f'{ds}: not cached, left out ({len(missing)}): '
                  f'{", ".join(missing)}', flush=True)
        pool = [r for r in pool if r in have]
        donors = [r for r in donors if r in have]

        print(f'\n=== {ds} ({time.time() - t0:.0f}s) ===', flush=True)
        data = load_data(ds, rep_preset='full', rep_names=pool + donors)
        reps_all = data['reps']

        # The whole experiment is about how many molecules a representation
        # merges, so a cached matrix that no longer matches its builder makes
        # every number here describe a representation the pipeline does not use.
        # The dirty run passed --skip-cache-check and loaded a chemberta
        # matrix that tied no molecule, so its tie counts measured nothing.
        # Since 2026-10-05 every loaded representation gets a record, checked
        # or not, and graph_kernel and mhg_gnn are checked (cache_check).
        fresh = ('skipped: --skip-cache-check' if args.skip_cache_check
                 else None)
        if not args.skip_cache_check and not args.count_only:
            fresh = cache_check(data, sorted(reps_all),
                                n_rows=args.cache_check_rows)
            unchecked = sorted(r for r, v in fresh.items()
                               if not v.get('checked'))
            if unchecked:
                print(f'  cache NOT compared with the builder for '
                      f'{len(unchecked)} representation(s): '
                      f'{", ".join(unchecked)}', flush=True)
            stale = [r for r, v in fresh.items()
                     if isinstance(v, dict) and v.get('matches') is False]
            if stale:
                detail = '; '.join(
                    f'{r}: {fresh[r].get("rows_that_differ")} of '
                    f'{fresh[r].get("rows_checked")} sampled rows differ'
                    for r in stale)
                raise RuntimeError(
                    f'{ds}: the cached matrix no longer matches what the '
                    f'builder returns for {", ".join(stale)} ({detail}). '
                    f'Rebuild those representations before running this, or '
                    f'pass --skip-cache-check and say so on the result.')

        choice = choose_representations(
            data, ds, pool, donors,
            target=(None if args.target == 'auto' else args.target),
            partner_sets=_partner_sets_for(ds),
            donor=(args.donor if run_aug else None), need_donor=run_aug)
        target = choice['target']
        partner_sets = [tuple(ps) for ps in choice['partner_sets']]
        donor = choice['donor']

        # The representations the claim is about must have been compared
        # with their builders (REFUSE_UNCHECKED_COMPARED_REPS, 2026-10-05).
        # A dataset with no SMILES has no builder here at all; that is
        # printed and recorded, not refused.
        compared = [target] + [r for ps in partner_sets for r in ps] + (
            [donor] if donor is not None else [])
        if isinstance(fresh, dict) and data.get('smiles') is not None:
            not_ok = sorted({r for r in compared
                             if not (fresh.get(r) or {}).get('checked')})
            if not_ok:
                why = '; '.join(
                    f'{r}: {(fresh.get(r) or {}).get("error") or (fresh.get(r) or {}).get("note") or "no record"}'
                    for r in not_ok)
                msg = (f'{ds}: the cached matrix of {", ".join(not_ok)} '
                       f'(target, partner or donor) was not compared with '
                       f'its builder ({why}). The tie counts depend on it.')
                if REFUSE_UNCHECKED_COMPARED_REPS:
                    raise RuntimeError(
                        msg + ' Make the builder available, or pass '
                              '--skip-cache-check and say so on the result.')
                print(f'  ⛔ {msg}', flush=True)
        elif isinstance(fresh, dict):
            print(f'  ⛔ {ds}: no SMILES, so no cached matrix was compared '
                  f'with a builder; the tie counts describe the cache.',
                  flush=True)
        print(f'  target={target}  donor={donor or "(none)"}', flush=True)
        for i, ps in enumerate(partner_sets, 1):
            print(f'  set {i}: {target} + {" + ".join(ps)}', flush=True)
        print(f'  {target} alone: {choice["target_molecules_sharing_a_row"]} of '
              f'{len(data["labels"])} molecules share a row', flush=True)
        if choice['target_molecules_sharing_a_row'] == 0:
            print(f'  ⛔ {target} gives every molecule its own row on these '
                  f'matrices, so every tie count below is 0 by construction.',
                  flush=True)
        for p, n in choice['still_tied_target_plus_each_partner'].items():
            print(f'    with {p} beside it: {n} still share a row', flush=True)
        for name, n in choice['still_tied_with_each_whole_set'].items():
            print(f'    with {name} beside it: {n} still share a row', flush=True)
        if donor is not None:
            print(f'    with the donor {donor} beside it: '
                  f'{choice["still_tied_target_plus_donor"]} still share a row',
                  flush=True)

        hybrid_pools = {f'C{i}': {r: reps_all[r] for r in (target,) + ps}
                        for i, ps in enumerate(partner_sets, 1)}
        target_only = {target: reps_all[target]}
        donor_reps = {donor: reps_all[donor]} if donor is not None else {}

        labels = np.asarray(data['labels'])
        task_type = data['task_type']
        folds = nested_cv_outer_folds(next(iter(hybrid_pools.values())), labels,
                                      data['groups'], n_outer_asked, SEED,
                                      task_type)
        n_outer = len(folds)
        if n_outer != n_outer_asked:
            print(f'  {n_outer_asked} outer folds asked, {n_outer} made (the '
                  f'splitter caps the count at the number of groups)',
                  flush=True)

        strategies = ('none',) + tuple(aug_conds)
        if args.count_only:
            run_c = 'C' in args.compare or 'A' in args.compare
            print(f'\n  outer folds: {n_outer}')
            print(f'  C arms: {len(hybrid_pools) if run_c else 0}, one '
                  f'general search per arm per fold = '
                  f'{(len(hybrid_pools) if run_c else 0) * n_outer} '
                  f'per-fold searches')
            if run_aug:
                points = sweep_points(1.0, strategies)
                print(f'  B and D sweep points per arm: {len(points)} '
                      f'(strategies {", ".join(strategies)})')
                n_augmented = ((1 if 'B' in args.compare else 0)
                               + (len(hybrid_pools) if 'D' in args.compare
                                  else 0))
                print(f'  B and D nested-CV runs: {len(points) * n_augmented}, '
                      f'each with {n_outer} per-fold searches')
            else:
                print('  B and D: not run on this dataset')
            continue

        metric = get_domain_metric(data)
        metric_name, _, lower = metric
        meta = {'dataset': ds, 'choice': choice,
                'rule': ('per C arm, per decided model, per outer fold: the '
                         'target alone (A), the 2 best base representations '
                         'and the 2 best hybrids of a search on that fold\'s '
                         'training rows over the arm\'s pool (rows '
                         '<arm>__base1/2, <arm>__hybrid1/2). No distance '
                         'hybrid (answer 1, 2026-10-05). B and D, only where '
                         'Step 8 keeps augmentation: per sweep point, per '
                         'decided model, per outer fold, the 2 best hybrids '
                         'of that fold\'s inner distance search with '
                         'distinct selected features (rows <arm>d__hybrid1/2) '
                         'and 2 base representations (<arm>d__base1/2); knn '
                         'only if B_D_INCLUDE_KNN.'),
                'selection_scope': 'fold_train_rows',
                'search_grid': 'DECISIONS.md decided grid per dataset and '
                               'search (decided_grid.py), pruning off',
                'metric': metric_name, 'lower_is_better': bool(lower),
                'models': models,
                'n_outer_folds': n_outer,
                # Per representation: whether a rebuilt sample of rows
                # matches the cached matrix; a representation whose builder
                # raised carries 'error' and was NOT checked.
                'cache_check': fresh,
                'augmentation_arms': {
                    'run': run_aug, 'skip_reason': aug_skip_reason,
                    'search': AUG_ARMS_TIER,
                    # Representations never used as added features
                    # (2026-10-06).
                    'excluded_donors': list(EXCLUDED_DONORS),
                    'strategies': list(strategies) if run_aug else []},
                'search_failures': [], 'shortfall': [], 'rule_missing': []}
        all_meta.append(meta)

        # ---- C (and A): per fold, written after every fold -------------
        if 'C' in args.compare or 'A' in args.compare:
            for arm, rep_pool in hybrid_pools.items():
                recs = c_records.setdefault((ds, arm), [])
                for k in range(n_outer):
                    _key = dict(dataset=ds, arm=arm, fold=int(k),
                                n_outer=int(n_outer), reps=sorted(rep_pool))
                    if _ck.done(**_key):
                        recs.append(_ck.get(**_key))
                        print(f'  {arm} fold {k}: finished in an earlier '
                              f'attempt, read back', flush=True)
                    else:
                        print(f'  {arm} fold {k} of {n_outer} '
                              f'({time.time() - t0:.0f}s)', flush=True)
                        rec = run_arm_fold(arm, rep_pool, data, ds, folds, k,
                                           models, metric, target, args)
                        recs.append(rec)
                        if rec['search_failures']:
                            _failed.append(f'{ds}/{arm}/fold{k}')
                        else:
                            # A fold whose search failed is in the result
                            # file but not stored, so the next attempt
                            # searches it again.
                            _ck.put(convert_numpy(rec), **_key)
                    meta['search_failures'] = [
                        f for (d_, _a), rs in c_records.items() if d_ == ds
                        for r in rs for f in r['search_failures']]
                    meta['shortfall'] = [
                        f for (d_, _a), rs in c_records.items() if d_ == ds
                        for r in rs for f in r['shortfall']]
                    _flush()

        # ---- B and D: only where Step 8 keeps augmentation -------------
        spread = args.fold_spread
        if run_aug and spread is None:
            # The margin unit: the target's spread across outer folds, the
            # largest over the decided models. It is read off the OUTER
            # held-out folds and then sits inside every fold's search of arms
            # B and D -- an open question for Adelaide, not fixed here.
            for (d_, _a), recs in c_records.items():
                if d_ != ds:
                    continue
                for m in models:
                    sd = _spread([(r['models'].get(m) or {}).get(
                        'target_alone', {}).get('heldout_score') for r in recs])
                    if isinstance(sd, float):
                        spread = sd if spread is None else max(spread, sd)
            if spread is None:
                raise RuntimeError(
                    f'{ds}: no spread for the target across outer folds ('
                    f'{n_outer} fold(s); one fold has no spread), so a margin '
                    f'multiple has no unit. Keep C in --compare with 2 or more '
                    f'folds, or pass --fold-spread.')
        meta['fold_spread_used_for_margins'] = spread

        if run_aug:
            # The models of B and D: the dataset's general decided models
            # (as C), plus knn only if B_D_INCLUDE_KNN (2026-10-05).
            bd_models = list(models)
            if (B_D_INCLUDE_KNN and 'knn' in decided_models(ds, AUG_ARMS_TIER)
                    and 'knn' not in bd_models):
                bd_models.append('knn')
            meta['augmentation_arms']['models'] = bd_models
            points = sweep_points(spread or 0.0, strategies)
            if args.sweep_slice:
                points = points[_k::_n]
                print(f'  slice {_k} of {_n}: {len(points)} of the '
                      f'{len(sweep_points(spread or 0.0, strategies))} settings',
                      flush=True)
            if args.smoke:
                seen, trimmed = set(), []
                for p in points:
                    if p['augmentation_strategy'] in seen:
                        continue
                    seen.add(p['augmentation_strategy'])
                    trimmed.append(p)
                points = trimmed
            meta['n_sweep_points'] = len(points)

            to_run = [('B', target_only)]
            for name, rep_pool in hybrid_pools.items():
                to_run.append(('D' + name[1:], rep_pool))
            for which, which_reps in to_run:
                if which[0] not in args.compare:
                    continue
                for i, point in enumerate(points):
                    print(f'  {which} point {i + 1}/{len(points)}: '
                          f'{point_slug(point)} ({time.time() - t0:.0f}s)',
                          flush=True)
                    _pkey = dict(dataset=ds, comparison=which,
                                 search=AUG_ARMS_TIER, n_outer=int(n_outer),
                                 point=point_slug(point),
                                 reps=sorted(which_reps))
                    if _ck.done(**_pkey):
                        bd_rows.extend(_ck.get(**_pkey))
                        print('    finished in an earlier attempt, read back',
                              flush=True)
                        continue
                    try:
                        res, bfolds = run_comparison(
                            which, data, ds, which_reps, donor_reps, n_outer,
                            args, bd_models, point=point, verbose=False)
                        _new_rows = summarise_comparison(
                            which, point, res, bfolds, data, which_reps,
                            donor_reps, target, bd_models, args)
                    except Exception as exc:
                        print(f'    FAILED: {type(exc).__name__}: {exc}',
                              flush=True)
                        bd_rows.append({
                            'comparison': which, 'search': AUG_ARMS_TIER,
                            'dataset': ds, 'sweep_point': point,
                            'error': f'{type(exc).__name__}: {exc}'})
                        _failed.append(f'{ds}/{which}/{point_slug(point)}')
                        _flush()
                        continue
                    for r in _new_rows:
                        r['dataset'] = ds
                    _ck.put(_new_rows, **_pkey)
                    bd_rows.extend(_new_rows)
                    _flush()

        ds_rows = [r for r in _all_rows() if r.get('dataset') == ds]
        meta['rule_missing'] = [w for r in ds_rows
                                for w in (r.get('rule_missing') or [])]
        by_which = {}
        for r in ds_rows:
            if 'model' in r:
                by_which.setdefault(r['comparison'], set()).add(r['model'])
        common = set.intersection(*by_which.values()) if by_which else set()
        dropped = set().union(*by_which.values()) - common if by_which else set()
        if dropped:
            print(f'\n  models present in some rows and not others, left out '
                  f'of the table: {", ".join(sorted(dropped))}')
            print('  (a failed search leaves a model out of that arm; knn '
                  'appears only in B and D, and only if B_D_INCLUDE_KNN)')
        meta['models_in_every_row'] = sorted(common)
        meta['models_left_out_of_table'] = sorted(dropped)
        print_headline(ds, choice,
                       [r for r in ds_rows if r.get('model') in common], common)
        _flush()

    if args.count_only:
        return
    _flush()
    print(f'\nwrote {out_path}')
    exit_if_failed(_failed, 'unit')


if __name__ == '__main__':
    main()
