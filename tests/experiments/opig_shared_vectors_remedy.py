#!/usr/bin/env python3
"""
Can a hybrid or an augmentation stop a representation merging two molecules?

THE QUESTION. opig_shared_vectors_census.py counts how many molecules a
representation gives one identical row, and turns that count into the best score
any model could reach. This script tries to fix it and measures whether the fix
worked, on two numbers at once:

  1. the score on held-out molecules, in the dataset's own metric;
  2. how many molecules still share a row IN THE MATRIX THE ENGINE ACTUALLY
     BUILT for that fold.

Both have to move, on the same outer folds, or the claim is not made.

THE FOUR THINGS COMPARED, all on one outer-fold partition.

  A   the target representation on its own.
  B   the target plus a donor representation's features, swept over the
      augmentation settings.
  C1  a hybrid of the target with the first pair of partner representations.
  C2  the same with the second pair, and so on for every pair named.
  D1  C1 plus the donor, swept the same way as B. D2 likewise.

Two pairs are run by default, ChemBERTa with ECFP4 and PDV, and ChemBERTa with
ECFP4 and usrcat. ECFP4 and PDV are both blind to stereochemistry, and on herg_ki
they repeat the same 56 rows ChemBERTa's own reader merges, so that pair can move
the score and cannot move the count. usrcat comes off a 3D conformer and repeats
no rows there. Running both is what separates "a hybrid helps" from "a hybrid
that can see the difference helps".

Comparison A is not a run of its own: find_best_hybrid refuses a pool of one
representation with no donor (find_best_hybrid.py:7866). It is read off comparison C's
own baseline table, where every representation in the pool is scored on the
untouched outer folds by the same code that scored the hybrid.

WHY THE TIE COUNT IS REPORTED THREE WAYS. Concatenating features can only ever
split a tied group, never merge one, so "hybrid ties <= target ties" would be
true by construction if the hybrid used all of the target's features. It does
not: it selects a budget's worth. So each fold reports

  ties of the target representation at its full width      (what A fits)
  ties of the target's columns the hybrid kept             (the width control)
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

Every sweep point is also checked after the fact. A point that admitted zero
donor features is printed as no-op. A point whose hybrid took exactly the same
columns as an earlier point's, on every fold, is printed as "same matrices as"
that point. The second check compares rows inside one output file; across
--sweep-slice files, compare 'recipe_signature_per_fold'.

Usage:
    # how big is the grid, before submitting anything
    python tests/experiments/opig_shared_vectors_remedy.py \\
        --datasets herg_ki --count-only

    # local plumbing check, minutes
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache \\
        python tests/experiments/opig_shared_vectors_remedy.py \\
        --datasets herg_ki --smoke

    # the run
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache \\
        python tests/experiments/opig_shared_vectors_remedy.py \\
        --datasets herg_ki qm9 --speed full --n-outer-folds 5
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

TIER = 'shared_vec'      # a private cache identity; does not touch the shared DBs
DEFAULT_DATASETS = ['herg_ki']

# The target and the two three-representation sets, agreed 2026-09-16. ECFP4 and
# PDV are both blind to stereochemistry: create_ecfp4 leaves includeChirality at
# its default and PDV is RDKit's 200 descriptors. Measured on herg_ki, ecfp4 and
# pdv repeat the same 56 rows ChemBERTa's own token sequences merge, so that set
# can move the score and cannot move the count. usrcat is built from a 3D
# conformer and repeats no rows on herg_ki, so the second set is the one that
# can. Running both is what tells the two apart.
DEFAULT_TARGET = 'chemberta'
DEFAULT_PARTNER_SETS = (('ecfp4', 'pdv'), ('ecfp4', 'usrcat'))
DEFAULT_OUTPUT = os.path.join(HERE, '..', 'results', 'comparisons')

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
# cap and fraction (172 configurations). Only a 14-row smoke slice has
# actually run so far (tests/results/comparisons/opig_shared_vectors_remedy.json),
# not the full sweep. Decision: leave the design as-is; run the full sweep
# when there's a reason to read its result, not before.
MARGIN_MULTIPLES = (0.0, 0.25, 0.5, 1.0)
FRACTIONS = (None, 0.05, 0.10, 0.25)
AUGMENTATION_BUDGETS = (5, 20, 50)          # read by strategy 'all' only
STRATEGIES = ('none', 'uncapped', 'capped', 'all')


def sweep_points(fold_spread):
    """Every augmentation setting worth running, and nothing that is ignored.

    fold_spread is the target representation's standard deviation across the
    outer folds, in the dataset's own metric. A margin is stated as a multiple of
    it so a block of donor features has to beat "keep nothing" by more than the
    run-to-run wobble before it is admitted.
    """
    points = []
    for strategy in STRATEGIES:
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
                # 4 IDENTICAL searches. They could not even share a cache: the
                # condition suffix carries the margin, so each of the 4 paid a
                # full run.
                #
                # 2 strategies x 4 rankings x 4 per-source budgets x 3 non-None
                # fractions = 96 combinations, each wasting 3 of its 4 margin
                # multiples: 288 duplicate sweep points of 516. The sweep is 228
                # points now and nothing it measured is lost -- the margin axis
                # is still swept in full on the 4 fraction=None combinations per
                # (strategy, ranking, per-source budget), which is where it acts.
                #
                # This is not a cut for compute. The dropped points are
                # DEGENERATE: they re-run a search whose every input the engine
                # treats identically. That is one of the three stated reasons a
                # cut is allowed.
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


def _content_fingerprint(mats):
    """One short hash over the names and bytes of every matrix, in name order."""
    h = hashlib.blake2b(digest_size=12)
    for name in sorted(mats):
        A = np.ascontiguousarray(mats[name])
        h.update(name.encode())
        h.update(str(A.dtype).encode())
        h.update(str(A.shape).encode())
        h.update(A.tobytes())
    return h.hexdigest()


def engine_kwargs(point):
    """The sweep point with this script's own bookkeeping keys removed."""
    return {k: v for k, v in point.items()
            if k not in ('augmentation_margin_multiple',)}


# ===== PICKING THE TARGET, THE PARTNERS AND THE DONOR ====================

def choose_representations(data, dataset_name, pool, donors,
                           target=None, partner_sets=None, donor=None):
    """Pick by measurement: which representation merges most, and what splits it.

    target       the representation that merges the most genuinely different
                 molecules -- the one the claim is about.
    partner_sets one or more pairs to hybridise the target with. Named pairs are
                 kept as given; otherwise the two representations that leave
                 fewest of the target's molecules tied are picked.
    donor        the augmentation representation that leaves fewest of the
                 target's molecules tied when laid beside it.

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

    if target is None:
        merged = {}
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
        target = max(merged, key=merged.get)

    if not partner_sets:
        others = [r for r in pool if r != target and r in partitions]
        others.sort(key=lambda r: (still_tied_with(target, r), r))
        partner_sets = [tuple(others[:2])]
    partner_sets = [tuple(ps) for ps in partner_sets]
    for ps in partner_sets:
        if len(ps) != 2:
            raise RuntimeError(f'{dataset_name}: each set needs two partners, '
                               f'got {ps}')
        for r in ps:
            if r not in partitions:
                raise RuntimeError(
                    f'{dataset_name}: {r} has no matrix here. Name it in --reps '
                    f'or build it first.')

    used = {target} | {r for ps in partner_sets for r in ps}
    if donor is None:
        cands = [r for r in donors if r in partitions and r not in used]
        if not cands:
            cands = [r for r in pool if r in partitions and r not in used]
        if not cands:
            raise RuntimeError(f'{dataset_name}: no donor representation left')
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
        'target_molecules_sharing_a_row': int((sizes_t > 1).sum()),
        'still_tied_target_plus_each_partner': {
            p: still_tied_with(target, p)
            for ps in partner_sets for p in ps},
        'still_tied_with_each_whole_set': {
            '+'.join(ps): still_tied_with_set(ps) for ps in partner_sets},
        'still_tied_target_plus_donor': still_tied_with(target, donor),
    }
    return choice


# ===== TIE COUNTING ON A BUILT MATRIX ====================================

def tie_stats(X, decimals=None):
    """How many rows of this matrix repeat another row, and in how many groups."""
    A = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if decimals is not None:
        A = np.round(A, decimals)
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


def rebuild_fold_matrix(reps, augs, labels, groups, tr, te, blueprint,
                        model_name, task_type, metric_fn, lower_is_better):
    """Rebuild the hybrid the engine scored on this outer fold, all rows.

    This mirrors find_best_hybrid.py:10128-10160 call for call, so the matrix
    counted here is the matrix that produced the score. apply_blueprint is used
    rather than build_fold_hybrid because apply_blueprint subsets to the
    blueprint's own representation list and build_fold_hybrid does not.
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

    X_fold = np.empty((labels.size, X_tr_h.shape[1]), dtype=np.float64)
    X_fold[tr] = X_tr_h
    X_fold[te] = X_te_h
    return X_fold, finfo


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


# ===== ONE ARM ============================================================

def run_comparison(name, data, dataset_name, reps, donor_reps, config, args,
            point=None, verbose=True):
    """One nested_cv_evaluate call, plus the per-fold tie counts it makes possible."""
    from experiments._shared import run_nested_cv
    from kirby.find_best_hybrid import nested_cv_outer_folds

    labels = np.asarray(data['labels'])
    groups = data['groups']
    kwargs = dict(engine_kwargs(point or {}))
    suffix = point_slug(point) if point else None

    result = run_nested_cv(
        reps=reps, labels=labels, groups=groups, config=config,
        dataset_name=dataset_name, task_type=data['task_type'],
        domain=data.get('domain'), tier=TIER,
        augmentations=(donor_reps or None),
        budgets=args.budgets, n_outer_folds=args.n_outer_folds,
        speed=args.speed,
        **({'models': args.models} if args.models else {}),
        # rep_counts is pinned to the whole pool. A blueprint records the pool it
        # was chosen from, not the subset a rep_count picked, and the outer
        # rebuild re-applies the pool -- so a searched rep_count would make the
        # honest score describe a wider feature set than the inner search chose.
        rep_counts=[None],
        # tune=None reads KIRBY_TUNE from the environment, which would silently
        # tune this run if it happens to be set in the shell.
        tune=False,
        condition_suffix=suffix,
        verbose=verbose, **kwargs)

    outer_folds = nested_cv_outer_folds(
        reps, labels, groups, args.n_outer_folds or config.get('n_splits', 5),
        42, data['task_type'])
    if len(outer_folds) != result.n_outer_folds:
        raise RuntimeError(
            f'{dataset_name} {name}: regenerated {len(outer_folds)} outer folds '
            f'but the run scored {result.n_outer_folds}. The splitter reduces '
            f'n_splits to the unique-group count, so the two must be read from '
            f'the same call.')
    return result, outer_folds


def summarise_comparison(name, point, result, outer_folds, data, reps, donor_reps,
                  target, args):
    """Scores, paired deltas against A, and the tie counts per fold."""
    labels = np.asarray(data['labels'])
    groups = data['groups']
    task_type = data['task_type']
    lower = bool(result.lower_is_better)
    metric_name = result.metric_name
    from kirby.find_best_hybrid import _resolve_metric
    metric_fn, _lb = _resolve_metric(metric_name)

    rows = []
    for model, entry in (result.by_model or {}).items():
        hybrid = entry.get('hybrid_fold_scores') or []
        base = ((result.baselines_by_model or {}).get(model, {})
                .get(target, {}).get('fold_scores') or [])
        paired = [(i, h, b) for i, (h, b) in enumerate(zip(hybrid, base))
                  if h is not None and b is not None
                  and np.isfinite(h) and np.isfinite(b)]
        deltas = [((b - h) if lower else (h - b)) for _, h, b in paired]

        ties_per_fold = []
        if not args.no_ties:
            for fold_i, _, _ in paired:
                bp = (entry.get('blueprints_per_fold') or [None] * 99)[fold_i]
                if bp is None:
                    ties_per_fold.append(None)
                    continue
                tr, te = outer_folds[fold_i]
                X_fold, finfo = rebuild_fold_matrix(
                    reps, donor_reps, labels, groups, tr, te, bp, model,
                    task_type, metric_fn, lower)
                rep_ties = fold_tie_report(
                    reps, X_fold, finfo, target, labels, task_type,
                    metric_name, args.tie_decimals)
                rep_ties['fold'] = fold_i
                rep_ties['kept_augmentations'] = list(
                    (finfo.get('augmentation_info') or {}).keys()) or []
                rep_ties['hybrid_width'] = int(X_fold.shape[1])
                # The rebuild is only "the matrix that was scored" if it has
                # the engine's width on this fold. Recorded, and loud if not.
                _eng_w = (entry.get('n_features_per_fold') or [None] * 99)[fold_i]
                rep_ties['engine_width'] = _eng_w
                rep_ties['width_matches_engine'] = (
                    None if _eng_w is None else int(_eng_w) == int(X_fold.shape[1]))
                if rep_ties['width_matches_engine'] is False:
                    print(f'    !! {name} {model} fold {fold_i}: rebuilt width '
                          f'{X_fold.shape[1]} features, engine scored '
                          f'{_eng_w}. The tie counts for this fold do not '
                          f'describe the scored matrix.', flush=True)
                rep_ties['recipe_signature'] = _recipe_signature(finfo)
                ties_per_fold.append(rep_ties)

        # Only the DONOR's names count as admitted donor features. A base
        # representation auto-demoted into augmentation_info (top_k allocations)
        # would otherwise make a point that admitted nothing from the donor
        # read as a real augmentation result.
        donor_names = set(donor_reps or {})
        admitted = sorted({name for t in ties_per_fold if t
                           for name in t['kept_augmentations']
                           if name in donor_names})
        rows.append({
            'comparison': name, 'sweep_point': point, 'model': model,
            'condition_suffix': point_slug(point) if point else None,
            'metric_name': metric_name, 'lower_is_better': lower,
            'n_outer_folds': result.n_outer_folds,
            'target_representation': target,
            'hybrid_fold_scores': hybrid,
            'target_alone_fold_scores': base,
            'paired_folds': [i for i, _, _ in paired],
            'delta_vs_target_alone_per_fold': deltas,
            'delta_vs_target_alone_mean': (float(np.mean(deltas))
                                           if deltas else None),
            'n_paired_folds': len(deltas),
            'ties_per_fold': ties_per_fold,
            'donor_features_admitted': admitted,
            # Without the rebuild there is nothing to read admission from, so
            # say "not checked" rather than calling every point a no-op.
            'no_op': (None if args.no_ties else
                      (point is not None
                       and point.get('augmentation_strategy') != 'none'
                       and not admitted)),
            'recipe_signature_per_fold': [
                (t or {}).get('recipe_signature') for t in ties_per_fold],
        })
    return rows


def _recipe_signature(finfo):
    """A hash of exactly which columns this fold's hybrid took, base and donor.

    Two sweep points with the same signature on every fold built the same
    matrix, so their scores are one measurement filed under two settings.
    """
    h = hashlib.blake2b(digest_size=10)
    for key in sorted(k for k, v in finfo.items()
                      if isinstance(v, dict) and 'selected_indices' in v):
        h.update(f'base:{key}:'.encode())
        h.update(np.asarray(finfo[key]['selected_indices'], dtype=np.int64)
                 .tobytes())
    for key, v in sorted((finfo.get('augmentation_info') or {}).items()):
        h.update(f'aug:{key}:'.encode())
        h.update(np.asarray(v.get('selected_indices', []), dtype=np.int64)
                 .tobytes())
    return h.hexdigest()


def mark_duplicate_points(rows):
    """Mark every row whose built matrices repeat an earlier row's, fold by fold.

    Same dataset, comparison and model, and the same recipe signature on every
    fold. Only rows inside this file can be compared; a sliced run compares
    across slice files by reading 'recipe_signature_per_fold'.
    """
    first = {}
    for r in rows:
        sigs = r.get('recipe_signature_per_fold')
        if not sigs or any(s is None for s in sigs) or 'model' not in r:
            continue
        # D<i> is C<i>'s pool plus the donor, so D<i> at strategy 'none' is
        # compared with C<i>; B has its own pool.
        pool_id = ('C' + r['comparison'][1:] if r['comparison'].startswith('D')
                   else r['comparison'])
        key = (r.get('dataset'), pool_id, r['model'], tuple(sigs))
        if key in first:
            r['same_matrices_as'] = first[key]
        else:
            first[key] = f"{r['comparison']} {r.get('condition_suffix') or '-'}"


# ===== ANALYSIS ===========================================================

def mean_of(rows_ties, where, field):
    vals = [t[where][field] for t in rows_ties
            if t and isinstance(t.get(where), dict) and field in t[where]]
    return float(np.mean(vals)) if vals else None


def print_headline(dataset_name, choice, rows, models_common):
    print('\n' + '=' * 104)
    print(f'{dataset_name}: does the fix move BOTH numbers, on the same folds?')
    sets = ' | '.join('+'.join(ps) for ps in choice['partner_sets'])
    print(f'target = {choice["target"]}   partner sets = {sets}'
          f'   donor = {choice["donor"]}')
    print('score delta is the row minus the target representation alone, signed so '
          'positive is better.')
    print('tied rows counts molecules sharing a row in the matrix that fold was '
          'scored on.')
    print('=' * 104)
    for model in sorted(models_common):
        print(f'\n  model {model}')
        print(f'    {"which":6s} {"score delta":>12s} '
              f'{"folds":>6s} {"tied: full":>11s} {"kept cols":>10s} '
              f'{"hybrid":>8s}  setting')
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
            if r.get('same_matrices_as'):
                label += f"  [same matrices as {r['same_matrices_as']}]"
            # The setting goes LAST and whole. It used to be cut to 34
            # characters, which is shorter than most sweep-point names
            # ('uncapped_rankresidual_correlation_cap...'), so different points
            # printed under one label and the no-op tag was never visible.
            print(f'    {r["comparison"]:6s} '
                  f'{"—" if d is None else f"{d:+12.4f}":>12s} '
                  f'{r["n_paired_folds"]:6d} '
                  f'{"—" if full is None else f"{full:11.1f}":>11s} '
                  f'{"—" if kept is None else f"{kept:10.1f}":>10s} '
                  f'{"—" if built is None else f"{built:8.1f}":>8s}  {label}')


# ===== MAIN ===============================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--target', default=DEFAULT_TARGET,
                    help='representation to fix. Pass "auto" to use the one that '
                         'merges the most genuinely different molecules.')
    ap.add_argument('--partner-sets', nargs='+', dest='partner_sets',
                    default=['%s' % ','.join(ps) for ps in DEFAULT_PARTNER_SETS],
                    help='one or more comma-separated pairs, each hybridised '
                         'with the target. Pass "auto" to pick one pair by '
                         'measurement.')
    ap.add_argument('--donor', default=None)
    ap.add_argument('--reps', default=None,
                    help='comma-separated pool override')
    ap.add_argument('--speed', default='full', choices=['fast', 'full'])
    ap.add_argument('--models', default=None,
                    help='comma-separated downstream models. Default None, which '
                         'is what the precompute wrote its cache under, so a list '
                         'here files under its own key and hits none of it. Use '
                         'it for a plumbing check, not for a result.')
    ap.add_argument('--budgets', default=None,
                    help='comma-separated feature budgets; default FULL_BUDGETS')
    ap.add_argument('--n-outer-folds', type=int, default=5, dest='n_outer_folds',
                    help='outer held-out folds. Passed explicitly because '
                         'get_config gives any mode but "full" a single fold, '
                         'and a delta over one fold prints like a delta over five.')
    ap.add_argument('--compare', nargs='+', default=['A', 'B', 'C', 'D'],
                    choices=['A', 'B', 'C', 'D'],
                    help='which of the four to run')
    ap.add_argument('--tie-decimals', type=int, default=None, dest='tie_decimals',
                    help='round features to this many decimals before comparing '
                         'rows; default compares them exactly')
    ap.add_argument('--no-ties', action='store_true',
                    help='scores only, skip the per-fold rebuild')
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
                         'starting at K. The whole grid still runs; it is spread '
                         'over N array tasks. Nothing is cut. Every task writes '
                         'its own result file, named by the slice.')
    ap.add_argument('--fold-spread', type=float, default=None,
                    dest='fold_spread',
                    help='The target representation\'s spread across outer '
                         'folds, in the dataset metric. A margin is a multiple '
                         'of it. Pass the number a C-only run printed and the '
                         'array tasks skip re-running C to get it.')
    ap.add_argument('--count-only', action='store_true',
                    help='print the size of the grid and stop')
    ap.add_argument('--smoke', action='store_true',
                    help='one sweep point per strategy, 2 outer folds, budget 25')
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    ap.add_argument('--force-rerun', action='store_true',
                    help='Recompute what an earlier attempt finished '
                         '(<output>.partial.jsonl).')
    ap = add_distance_metric_arg(ap)
    args = ap.parse_args()
    apply_distance_metric_arg(args)

    args.budgets = ([int(b) for b in args.budgets.split(',') if b.strip()]
                    if args.budgets else None)
    args.models = ([m.strip() for m in args.models.split(',') if m.strip()]
                   if args.models else None)
    if args.smoke:
        from experiments._shared import smoke_budgets
        args.budgets = args.budgets or smoke_budgets()
        args.n_outer_folds = min(args.n_outer_folds, 2)

    from experiments._shared import load_data, get_config
    from experiments.complexity.complexity_helpers import DATASET_REPS

    class _Cfg:
        mode = 'full'
    config = get_config(_Cfg())

    # Each (dataset, comparison, sweep point) is written to
    # <output>.partial.jsonl when it finishes and read back on the next
    # attempt (2026-09-25; experiments/_checkpoint.py). What is stored is the
    # rows summarise_comparison produced for it, which is what goes into the
    # JSON. The sidecar is named after the output file, so each --sweep-slice
    # keeps its own. A sweep point that raised is still recorded as an error
    # row, is not stored, is retried next time, and makes the script exit 1
    # after the JSON is written. The C comparisons are not stored per unit.
    _tag = ''
    if args.sweep_slice:
        _k, _n = (int(x) for x in args.sweep_slice.split('/'))
        _tag = f'_slice{_k}of{_n}'
    from experiments._checkpoint import script_checkpoint, exit_if_failed
    _ck = script_checkpoint(
        'opig_shared_vectors_remedy',
        os.path.join(args.output_dir,
                     f'opig_shared_vectors_remedy{_tag}.json'),
        args, exclude=('datasets', 'output_dir'))
    _failed = []

    all_rows, all_meta = [], []
    t0 = time.time()
    for ds in args.datasets:
        entry = DATASET_REPS[ds]
        pool = ([r.strip() for r in args.reps.split(',')] if args.reps
                else list(entry['general']))
        donors = list(entry.get('augmentation') or [])
        named = set()
        if args.target and args.target != 'auto':
            named.add(args.target)
        if args.partner_sets != ['auto']:
            for ps in args.partner_sets:
                named.update(x.strip() for x in ps.split(','))
        if args.donor:
            named.add(args.donor)
        pool = sorted(set(pool) | named)
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
        # herg_ki's cached chemberta says 0 molecules share a row and the
        # builder says 130, so this is not hypothetical. Checked, not assumed.
        # Written into the result either way. The flag's own help says "say so
        # on the result", and until 2026-09-28 nothing did.
        cache_check = {'skipped': bool(args.skip_cache_check)}
        if not args.skip_cache_check and not args.count_only:
            fresh = census.freshness_check(data, sorted(reps_all),
                                           n_rows=args.cache_check_rows)
            cache_check['result'] = fresh
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
            partner_sets=(None if args.partner_sets == ['auto'] else
                          [tuple(x.strip() for x in ps.split(','))
                           for ps in args.partner_sets]),
            donor=args.donor)
        target = choice['target']
        partner_sets = [tuple(ps) for ps in choice['partner_sets']]
        donor = choice['donor']
        print(f'  target={target}  donor={donor}', flush=True)
        for i, ps in enumerate(partner_sets, 1):
            print(f'  set {i}: {target} + {" + ".join(ps)}', flush=True)
        print(f'  {target} alone: {choice["target_molecules_sharing_a_row"]} of '
              f'{len(data["labels"])} molecules share a row', flush=True)
        if choice['target_molecules_sharing_a_row'] == 0:
            # A named target skips the "nothing merges" refusal the automatic
            # pick has. The run still answers whether the score moves, but the
            # tie count starts at zero, so the count half of the question
            # cannot be asked. herg_ki's cached chemberta is this case.
            print(f'  !! {target} gives every molecule its own row in the '
                  f'matrix this run loaded, so no fix can lower the tie '
                  f'count. Only the score half of the question is answered.'
                  + ('  The cache check was skipped, so this may be a stale '
                     'cached matrix.' if args.skip_cache_check else ''),
                  flush=True)
        for p, n in choice['still_tied_target_plus_each_partner'].items():
            print(f'    with {p} beside it: {n} still share a row', flush=True)
        for name, n in choice['still_tied_with_each_whole_set'].items():
            print(f'    with {name} beside it: {n} still share a row', flush=True)
        print(f'    with the donor {donor} beside it: '
              f'{choice["still_tied_target_plus_donor"]} still share a row',
              flush=True)

        hybrid_pools = {f'C{i}': {r: reps_all[r] for r in (target,) + ps}
                        for i, ps in enumerate(partner_sets, 1)}
        target_only = {target: reps_all[target]}
        donor_reps = {donor: reps_all[donor]}

        if args.count_only:
            # The margin axis needs a spread that only a run can give, so count
            # it at its full length and say the spread is not known yet.
            points = sweep_points(1.0)
            print(f'\n  sweep points per augmented comparison: {len(points)}')
            for s_name in STRATEGIES:
                n = sum(1 for p in points
                        if p['augmentation_strategy'] == s_name)
                print(f'    strategy {s_name:9s} {n:5d}')
            n_sets = (1 if args.partner_sets == ['auto']
                      else len(args.partner_sets))
            n_augmented = ((1 if 'B' in args.compare else 0)
                           + (n_sets if 'D' in args.compare else 0))
            n_c = n_sets if 'C' in args.compare else 0
            print(f'  partner sets: {n_sets}')
            print(f'  nested-CV runs for this dataset: '
                  f'{n_c + len(points) * n_augmented} '
                  f'({n_c} without a donor, {len(points)} for each of '
                  f'{n_augmented} swept ones)')
            continue

        # Every C first: each carries A in its baseline table, and the
        # target's spread across outer folds is what a margin is a multiple of.
        # An array task that was handed the spread skips this, because C is the
        # same run in every task and repeating it N times is the one thing
        # slicing the sweep is meant to avoid.
        spread, c_results = args.fold_spread, {}
        run_c = 'C' in args.compare and args.fold_spread is None
        # Was `if run_c or 'C' in args.compare`, which is just "C was asked
        # for": a task handed --fold-spread re-ran C anyway, and then took the
        # max of the handed spread and the new one, so the margin unit was not
        # the number passed in. Fixed 2026-09-28 (dirty local audit).
        if run_c:
            for name, rep_pool in hybrid_pools.items():
                res, folds = run_comparison(name, data, ds, rep_pool, None,
                                            config, args)
                c_results[name] = (res, folds, rep_pool)
                for model, base in (res.baselines_by_model or {}).items():
                    sd = (base.get(target) or {}).get('std')
                    if sd is not None and np.isfinite(sd):
                        spread = (float(sd) if spread is None
                                  else max(spread, float(sd)))
        if spread is None and ('B' in args.compare or 'D' in args.compare):
            raise RuntimeError(
                f'{ds}: no spread for the target across outer folds, so a '
                f'margin multiple has no unit. Either keep C in --compare or '
                f'pass --fold-spread from a run that printed it.')
        metric_name = (next(iter(c_results.values()))[0].metric_name
                       if c_results else '(from --fold-spread)')
        print(f'  {target} spread across outer folds, worst model: '
              f'{"unknown" if spread is None else f"{spread:.4f}"} '
              f'{metric_name}', flush=True)

        points = sweep_points(spread or 0.0)
        slice_tag = ''
        if args.sweep_slice:
            k, n = (int(x) for x in args.sweep_slice.split('/'))
            if not 0 <= k < n:
                raise ValueError(f'--sweep-slice {args.sweep_slice}: need 0 <= K < N')
            points = points[k::n]
            slice_tag = f'_slice{k}of{n}'
            print(f'  slice {k} of {n}: {len(points)} of the '
                  f'{len(sweep_points(spread or 0.0))} settings', flush=True)
        if args.smoke:
            seen, trimmed = set(), []
            for p in points:
                if p['augmentation_strategy'] in seen:
                    continue
                seen.add(p['augmentation_strategy'])
                trimmed.append(p)
            points = trimmed

        for name, (res, folds, rep_pool) in c_results.items():
            all_rows.extend(summarise_comparison(
                name, None, res, folds, data, rep_pool, None, target, args))

        to_run = [('B', target_only)]
        for name, rep_pool in hybrid_pools.items():
            to_run.append(('D' + name[1:], rep_pool))
        for which, which_reps in to_run:
            if which[0] not in args.compare:
                continue
            for i, point in enumerate(points):
                print(f'  {which} point {i + 1}/{len(points)}: '
                      f'{point_slug(point)} ({time.time() - t0:.0f}s)', flush=True)
                # The unit also depends on the donor (chosen by measurement, not
                # an argument), the margin unit (spread) and the matrices'
                # CONTENT. None of those was in the key or the header, so after
                # a representation was rebuilt at the same width -- the fix for
                # the stale chemberta this file warns about -- the old rows were
                # read back as if they described the new matrix.
                _pkey = dict(dataset=ds, comparison=which,
                             point=point_slug(point),
                             reps=sorted(which_reps),
                             target=target, donor=donor,
                             spread=spread,
                             content=_content_fingerprint(
                                 {**which_reps, **donor_reps}))
                if _ck.done(**_pkey):
                    all_rows.extend(_ck.get(**_pkey))
                    print('    finished in an earlier attempt, read back',
                          flush=True)
                    continue
                try:
                    res, folds = run_comparison(which, data, ds, which_reps,
                                                donor_reps, config, args,
                                                point=point, verbose=False)
                except Exception as exc:
                    print(f'    FAILED: {type(exc).__name__}: {exc}', flush=True)
                    all_rows.append({'comparison': which, 'dataset': ds,
                                     'sweep_point': point,
                                     'error': f'{type(exc).__name__}: {exc}'})
                    _failed.append(f'{ds}/{which}/{point_slug(point)}')
                    continue
                _new_rows = summarise_comparison(
                    which, point, res, folds, data, which_reps, donor_reps,
                    target, args)
                _ck.put(_new_rows, **_pkey)
                all_rows.extend(_new_rows)

        for r in all_rows:
            r.setdefault('dataset', ds)
        mark_duplicate_points([r for r in all_rows if r.get('dataset') == ds])
        by_which = {}
        for r in all_rows:
            if r.get('dataset') == ds and 'model' in r:
                by_which.setdefault(r['comparison'], set()).add(r['model'])
        common = set.intersection(*by_which.values()) if by_which else set()
        dropped = set().union(*by_which.values()) - common if by_which else set()
        if dropped:
            print(f'\n  models present in some of the four and not others, left '
                  f'out of the table: {", ".join(sorted(dropped))}')
            print('  (dataset_policy._DATASET_STRATEGY_SKIPS drops models per '
                  'augmentation strategy, so one of the four can lose a '
                  'model silently)')
        print_headline(ds, choice,
                       [r for r in all_rows
                        if r.get('dataset') == ds and r.get('model') in common],
                       common)
        all_meta.append({'dataset': ds, 'choice': choice,
                         'cache_check': cache_check,
                         'tie_count_can_move': (
                             choice['target_molecules_sharing_a_row'] > 0),
                         'fold_spread_used_for_margins': spread,
                         'n_sweep_points': len(points),
                         'models_in_all_four': sorted(common),
                         'models_left_out': sorted(dropped)})

    if args.count_only:
        exit_if_failed(_failed, 'sweep point')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    tag = ''
    if args.sweep_slice:
        k, n = (int(x) for x in args.sweep_slice.split('/'))
        tag = f'_slice{k}of{n}'
    path = os.path.join(args.output_dir,
                        f'opig_shared_vectors_remedy{tag}.json')
    with open(path, 'w') as f:
        json.dump(convert_numpy({'rows': all_rows, 'per_dataset': all_meta}),
                  f, indent=1)
    print(f'\nwrote {path}')
    exit_if_failed(_failed, 'sweep point')


if __name__ == '__main__':
    main()
