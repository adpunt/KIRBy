#!/usr/bin/env python3
"""f33 replay — apply every pruning rule to the recorded fold scores.

Nothing is fitted here. f33_pruning_recalibration.py ran each dataset's grid with
adaptive_pruning=False and stored every configuration's per-fold score; this file
replays the two prunes off that record at any setting, which is what makes the
two-fold question free to answer.

THE TWO RULES, mirrored from the engine.

  End-of-fold-0 allocation prune (find_best_hybrid.py:6083). Group by (importance
  method, internal evaluator, model, budget). In each group rank the allocation
  methods by their fold-0 score and give a point to the best `strategy_top_n`.
  Drop a method that scored in fewer than `final_strategy_threshold` of the
  groups. A non-finite score sinks to worst, exactly as the engine's sentinel at
  line 6072 does, so a method whose configurations failed loses those placements.

  Mid-flight tail drop (_compute_tail_prunes, find_best_hybrid.py:3265). At a
  checkpoint budget, inside each (model, evaluator) group restricted to
  configurations at budget <= the checkpoint, rank by fold-0 score and keep
  max(min_configs, round(n * (1 - tail_drop))). The dropped (allocation,
  importance) pairs are never evaluated at any higher budget in that group.
  Checkpoints fire only on an exact budget match (line 6004), so a checkpoint
  that is not one of the budgets run does nothing at all.

TWO FOLDS INSTEAD OF ONE. The engine has no such setting — `checkpoints` are
budgets, not folds, and the vote reads fold 0 and nothing else. Two forms are
replayed here because they can disagree:
  mean   — rank each group on the mean of the folds voting, threshold applied once
  points — rank each voting fold separately, add the points, threshold multiplied
           by the number of voting folds

WHAT COUNTS AS SAFE. The truth is the allocation method with the best fold-mean
per (model, budget), which is what the unpruned search returns. A rule is safe on
a dataset when every one of those winners survives it. When one does not, the
report prints the winner's fold-mean and the best surviving method's fold-mean in
the dataset's own metric, then the difference between them.

TIME. `time_seconds` is the whole configuration across all its folds, not one
fold, so a per-fold saving is that number divided by the number of folds. Stated
as the approximation it is.

Usage:
    python tests/experiments/framework/f33_analyse.py
    python tests/experiments/framework/f33_analyse.py --budgets 50,150,300
    python tests/experiments/framework/f33_analyse.py --datasets caco2
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = _ROOT / 'tests' / 'results' / 'framework' / 'f33_pruning'

# The shipped values, src/kirby/find_best_hybrid.py DEFAULT_PRUNING_SCHEDULE.
SHIPPED_TOP_N = 3
SHIPPED_THRESHOLD = 0.15
SHIPPED_MIN_CONFIGS = 1

TOP_N_VALUES = [1, 2, 3, 4]
THRESHOLD_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TAIL_DROP_VALUES = [0.25, 0.50, 0.75]
# The checkpoints tests/experiments/_shared.py PRUNING_MODES['aggressive_pruning']
# ships, plus each single budget on its own so a schedule can be read one
# checkpoint at a time.
AGGRESSIVE_CHECKPOINTS = [25, 50, 100]


# =============================================================================
# LOADING
# =============================================================================

def load_records(results_dir, datasets=None, budgets=None):
    out = []
    for path in sorted(Path(results_dir).glob('*.json')):
        if path.stem.endswith('__smoke'):
            continue
        with open(path) as f:
            rec = json.load(f)
        if datasets and rec['dataset'] not in datasets:
            continue
        if budgets is not None:
            keep = set(budgets)
            rec['results'] = [r for r in rec['results'] if r['budget'] in keep]
            rec['budgets_run'] = sorted(b for b in rec['budgets_run'] if b in keep)
        if not rec['results']:
            print(f"  skip {path.name}: no configurations at these budgets")
            continue
        out.append(rec)
    return out


def _worst(lower_is_better):
    return float('inf') if lower_is_better else float('-inf')


def _fold_score(row, fold_idx, lower_is_better):
    """One fold's score, with a missing or non-finite value sunk to worst.

    The engine does exactly this (find_best_hybrid.py:6072): a configuration that
    failed must not take a top place away from one that scored.
    """
    fs = row['fold_scores']
    if fold_idx >= len(fs):
        return _worst(lower_is_better)
    s = fs[fold_idx]
    if s is None or not np.isfinite(s):
        return _worst(lower_is_better)
    return float(s)


def _fold_mean(row, fold_idxs, lower_is_better):
    vals = [_fold_score(row, k, lower_is_better) for k in fold_idxs]
    if any(not np.isfinite(v) for v in vals):
        return _worst(lower_is_better)
    return float(np.mean(vals))


def _full_mean(row, lower_is_better):
    fs = [s for s in row['fold_scores'] if s is not None and np.isfinite(s)]
    if not fs:
        return _worst(lower_is_better)
    return float(np.mean(fs))


def _better(a, b, lower_is_better):
    return a < b if lower_is_better else a > b


# =============================================================================
# THE TRUTH: what an unpruned search returns
# =============================================================================

def model_budget_winners(rec):
    """{(model, budget): (winning allocation method, its fold-mean)}.

    Plus {(model, budget): {allocation method: fold-mean}} so a lost winner can be
    priced against the best method that survived.
    """
    lib = rec['lower_is_better']
    by_cell = defaultdict(dict)
    for r in rec['results']:
        cell = (r['model'], r['budget'])
        m = _full_mean(r, lib)
        # One allocation method appears once per (importance, evaluator) too, so
        # the method's number for this cell is its best over those.
        prev = by_cell[cell].get(r['allocation_method'])
        if prev is None or _better(m, prev, lib):
            by_cell[cell][r['allocation_method']] = m
    winners = {}
    for cell, scores in by_cell.items():
        best = min(scores, key=scores.get) if lib else max(scores, key=scores.get)
        winners[cell] = (best, scores[best])
    return winners, by_cell


# =============================================================================
# RULE 1 — the allocation vote
# =============================================================================

def allocation_vote(rec, vote_folds, top_n, threshold_frac, combine='mean',
                    excluded_pairs=None):
    """Replay the end-of-fold-0 vote, optionally over more than one fold.

    vote_folds  which folds feed the vote, e.g. [0] for the shipped rule
    combine     'mean'   rank on the mean of vote_folds, threshold applied once
                'points' rank each fold separately, add points, threshold scaled
    excluded_pairs  {(model, evaluator): {(allocation, importance)}} already
                dropped mid-flight; those configurations do not vote, which is
                what the engine does at line 6062.

    Returns (kept, pruned, place_counts, n_groups, threshold_places).
    """
    lib = rec['lower_is_better']
    methods = rec['grid']['allocation_methods']
    excluded_pairs = excluded_pairs or {}

    groups = defaultdict(list)
    for r in rec['results']:
        grp = (r['model'], r['internal_evaluator'])
        if (r['allocation_method'], r['importance_method']) in excluded_pairs.get(grp, ()):
            continue
        key = (r['importance_method'], r['internal_evaluator'],
               r['model'], r['budget'])
        groups[key].append(r)
    n_groups = len(groups)

    place_counts = defaultdict(int)
    if combine == 'mean':
        rank_folds = [list(vote_folds)]
        threshold_places = n_groups * threshold_frac
    else:
        rank_folds = [[k] for k in vote_folds]
        threshold_places = n_groups * threshold_frac * len(vote_folds)

    for entries in groups.values():
        for folds in rank_folds:
            ranked = sorted(entries,
                            key=lambda r: _fold_mean(r, folds, lib),
                            reverse=not lib)
            for r in ranked[:top_n]:
                place_counts[r['allocation_method']] += 1

    pruned = [m for m in methods if place_counts.get(m, 0) < threshold_places]
    kept = [m for m in methods if m not in pruned]
    return kept, pruned, dict(place_counts), n_groups, threshold_places


# =============================================================================
# RULE 2 — the mid-flight tail drop
# =============================================================================

def tail_drop(rec, checkpoints, frac, min_configs=SHIPPED_MIN_CONFIGS):
    """Replay _compute_tail_prunes over a list of checkpoint budgets.

    Only checkpoints that are exactly one of the budgets run can fire — the
    engine's `cur_b in _all_checkpoints` test. Returns
    ({(model, evaluator): {(allocation, importance)}}, checkpoints that fired).
    """
    lib = rec['lower_is_better']
    budgets_run = set(rec['budgets_run'])
    fired = [b for b in sorted(checkpoints) if b in budgets_run]
    dropped = defaultdict(set)
    if frac <= 0:
        return dict(dropped), fired

    for ckpt in fired:
        by_group = defaultdict(list)
        for r in rec['results']:
            if r['budget'] > ckpt:
                continue
            grp = (r['model'], r['internal_evaluator'])
            pair = (r['allocation_method'], r['importance_method'])
            if pair in dropped.get(grp, ()):
                continue
            by_group[grp].append(r)
        for grp, entries in by_group.items():
            ranked = sorted(entries, key=lambda r: _fold_score(r, 0, lib),
                            reverse=not lib)
            n_keep = max(min_configs, int(round(len(ranked) * (1 - frac))))
            kept_pairs = {(r['allocation_method'], r['importance_method'])
                          for r in ranked[:n_keep]}
            all_pairs = {(r['allocation_method'], r['importance_method'])
                         for r in ranked}
            dropped[grp] |= (all_pairs - kept_pairs)
    return {k: v for k, v in dropped.items() if v}, fired


# =============================================================================
# COST AND SAFETY
# =============================================================================

def winners_lost(rec, kept_methods, dropped_pairs=None):
    """Which (model, budget) winners the surviving methods can no longer reach.

    Returns a list of (model, budget, winner, winner_score, best_kept,
    best_kept_method, difference). Empty means safe.
    """
    lib = rec['lower_is_better']
    winners, by_cell = model_budget_winners(rec)
    kept = set(kept_methods)
    dropped_pairs = dropped_pairs or {}

    # A (model, budget) cell can also lose its winner because every (allocation,
    # importance) pair carrying it was tail-dropped, so recompute reachability
    # from the rows rather than from the method list alone.
    reachable = defaultdict(dict)
    for r in rec['results']:
        if r['allocation_method'] not in kept:
            continue
        grp = (r['model'], r['internal_evaluator'])
        if (r['allocation_method'], r['importance_method']) in dropped_pairs.get(grp, ()):
            continue
        cell = (r['model'], r['budget'])
        m = _full_mean(r, lib)
        prev = reachable[cell].get(r['allocation_method'])
        if prev is None or _better(m, prev, lib):
            reachable[cell][r['allocation_method']] = m

    lost = []
    for cell, (winner, winner_score) in sorted(winners.items()):
        surviving = reachable.get(cell, {})
        if winner in surviving:
            continue
        if not surviving:
            lost.append((*cell, winner, winner_score, None, None, None))
            continue
        best_method = (min(surviving, key=surviving.get) if lib
                       else max(surviving, key=surviving.get))
        best_score = surviving[best_method]
        lost.append((*cell, winner, winner_score, best_score, best_method,
                     abs(winner_score - best_score)))
    return lost, len(winners)


def seconds_saved(rec, pruned_methods, vote_folds, dropped_pairs=None):
    """Wall time the rule does not spend, from the recorded time_seconds.

    A tail-dropped configuration above its checkpoint is never evaluated, so all
    of its folds are saved. A configuration whose allocation method loses the vote
    has already paid the folds that voted, so the saving is the rest of them.
    `time_seconds` covers the whole configuration across its folds, so this
    divides by the fold count.
    """
    n_folds = rec['n_folds_run']
    dropped_pairs = dropped_pairs or {}
    pruned = set(pruned_methods)
    saved = 0.0
    total = 0.0
    for r in rec['results']:
        t = r['time_seconds']
        if t is None:
            continue
        total += t
        grp = (r['model'], r['internal_evaluator'])
        pair = (r['allocation_method'], r['importance_method'])
        if pair in dropped_pairs.get(grp, ()):
            saved += t
        elif r['allocation_method'] in pruned:
            saved += t * (n_folds - len(vote_folds)) / n_folds
    return saved, total


# =============================================================================
# REPORTING
# =============================================================================

def _fmt_lost(rec, lost, limit=5):
    lines = []
    for (model, budget, winner, ws, bk, bkm, diff) in lost[:limit]:
        if bk is None:
            lines.append(f"      {model}/budget {budget}: winner {winner} "
                         f"({ws:.4f} {rec['metric']}) — nothing survived")
        else:
            lines.append(f"      {model}/budget {budget}: winner {winner} "
                         f"{ws:.4f} {rec['metric']}, best surviving {bkm} "
                         f"{bk:.4f}, difference {diff:.4f}")
    if len(lost) > limit:
        lines.append(f"      ... and {len(lost) - limit} more")
    return lines


def report_dataset(rec):
    lib = rec['lower_is_better']
    direction = 'lower is better' if lib else 'higher is better'
    winners, by_cell = model_budget_winners(rec)
    n_folds = rec['n_folds_run']

    print(f"\n{'=' * 78}")
    print(f"  {rec['dataset']} — {rec['n_samples']} rows, "
          f"{rec['n_representations']} representations, "
          f"metric {rec['metric']} ({direction})")
    print(f"  {len(rec['results'])} configurations, budgets "
          f"{rec['budgets_run']}, {n_folds} folds")
    print(f"{'=' * 78}")

    print(f"\n  Winning allocation method per (model, budget) — "
          f"{len(winners)} cells, this is what a rule has to keep reachable:")
    for (model, budget), (winner, score) in sorted(winners.items()):
        print(f"    {model:9s} budget {budget:4d}: {winner:22s} "
              f"{score:.4f} {rec['metric']}")

    print(f"\n  Win counts across those {len(winners)} cells:")
    counts = defaultdict(int)
    for (winner, _) in winners.values():
        counts[winner] += 1
    for m in rec['grid']['allocation_methods']:
        print(f"    {m:22s} {counts.get(m, 0):3d} cells")

    # --- the shipped rule, and the sweep around it ---
    print(f"\n  The allocation vote on fold 0 alone "
          f"(the shipped rule is top-{SHIPPED_TOP_N} at {SHIPPED_THRESHOLD:.0%}):")
    for top_n in TOP_N_VALUES:
        for thr in THRESHOLD_VALUES:
            kept, pruned, places, n_groups, thr_places = allocation_vote(
                rec, [0], top_n, thr)
            lost, n_cells = winners_lost(rec, kept)
            saved, total = seconds_saved(rec, pruned, [0])
            mark = '  <-- shipped' if (top_n == SHIPPED_TOP_N
                                       and abs(thr - SHIPPED_THRESHOLD) < 1e-9) else ''
            state = 'SAFE' if not lost else f'LOST {len(lost)}/{n_cells}'
            print(f"    top-{top_n} at {thr:>5.0%}: dropped "
                  f"{len(pruned)}/{len(rec['grid']['allocation_methods'])} "
                  f"{sorted(pruned) if pruned else ''} — {state}, "
                  f"saves {saved / 3600:.2f} h of {total / 3600:.2f} h{mark}")
            for line in _fmt_lost(rec, lost):
                print(line)

    # --- is the fold-0 vote noisy? ---
    print(f"\n  The same shipped setting run on each fold on its own — "
          f"whether the vote depends on which fold cast it:")
    per_fold = {}
    for k in range(n_folds):
        kept, pruned, places, n_groups, thr_places = allocation_vote(
            rec, [k], SHIPPED_TOP_N, SHIPPED_THRESHOLD)
        per_fold[k] = tuple(sorted(pruned))
        print(f"    fold {k}: dropped {sorted(pruned) if pruned else 'nothing'} "
              f"(places needed {thr_places:.1f} of {n_groups} groups; "
              f"counts " + ', '.join(f"{m} {places.get(m, 0)}"
                                     for m in rec['grid']['allocation_methods']) + ")")
    agree = len(set(per_fold.values())) == 1
    print(f"    the {n_folds} folds " +
          ("all drop the same set" if agree else "do NOT drop the same set"))

    # --- two folds instead of one ---
    print(f"\n  Two folds instead of one, at the shipped setting:")
    for combine in ('mean', 'points'):
        kept, pruned, places, n_groups, thr_places = allocation_vote(
            rec, [0, 1], SHIPPED_TOP_N, SHIPPED_THRESHOLD, combine=combine)
        lost, n_cells = winners_lost(rec, kept)
        saved, total = seconds_saved(rec, pruned, [0, 1])
        saved1, _ = seconds_saved(
            rec, allocation_vote(rec, [0], SHIPPED_TOP_N, SHIPPED_THRESHOLD)[1], [0])
        state = 'SAFE' if not lost else f'LOST {len(lost)}/{n_cells}'
        label = ('ranked on the mean of folds 0 and 1' if combine == 'mean'
                 else 'points added over folds 0 and 1, threshold doubled')
        print(f"    {label}: dropped "
              f"{sorted(pruned) if pruned else 'nothing'} — {state}, "
              f"saves {saved / 3600:.2f} h against {saved1 / 3600:.2f} h "
              f"for the fold-0 vote")
        for line in _fmt_lost(rec, lost):
            print(line)

    # --- the mid-flight tail drop ---
    print(f"\n  The mid-flight tail drop, then the shipped vote on what survives:")
    checkpoint_sets = [[b] for b in rec['budgets_run']] + [AGGRESSIVE_CHECKPOINTS]
    seen = set()
    for ckpts in checkpoint_sets:
        key = tuple(sorted(ckpts))
        if key in seen:
            continue
        seen.add(key)
        for frac in TAIL_DROP_VALUES:
            pairs, fired = tail_drop(rec, ckpts, frac)
            if not fired:
                print(f"    checkpoints {sorted(ckpts)} at {frac:.0%}: "
                      f"no checkpoint is one of the budgets run — nothing fires")
                continue
            kept, pruned, _, _, _ = allocation_vote(
                rec, [0], SHIPPED_TOP_N, SHIPPED_THRESHOLD, excluded_pairs=pairs)
            lost, n_cells = winners_lost(rec, kept, dropped_pairs=pairs)
            saved, total = seconds_saved(rec, pruned, [0], dropped_pairs=pairs)
            n_pairs = sum(len(v) for v in pairs.values())
            state = 'SAFE' if not lost else f'LOST {len(lost)}/{n_cells}'
            mark = ('  <-- aggressive_pruning' if key == tuple(AGGRESSIVE_CHECKPOINTS)
                    and abs(frac - 0.50) < 1e-9 else '')
            print(f"    checkpoints {sorted(ckpts)} (fired {fired}) at {frac:.0%}: "
                  f"{n_pairs} (allocation, importance) pairs dropped, vote then "
                  f"dropped {sorted(pruned) if pruned else 'nothing'} — {state}, "
                  f"saves {saved / 3600:.2f} h of {total / 3600:.2f} h{mark}")
            for line in _fmt_lost(rec, lost):
                print(line)

    return {
        'dataset': rec['dataset'],
        'metric': rec['metric'],
        'n_cells': len(winners),
        'win_counts': dict(counts),
        'per_fold_pruned': {k: list(v) for k, v in per_fold.items()},
        'folds_agree': agree,
    }


def report_across(summaries):
    if len(summaries) < 2:
        return
    print(f"\n{'=' * 78}")
    print(f"  ACROSS {len(summaries)} DATASETS — counts, never a mean")
    print(f"{'=' * 78}")
    print(f"\n  How many (model, budget) cells each allocation method wins, "
          f"per dataset:")
    methods = sorted({m for s in summaries for m in s['win_counts']})
    head = f"  {'dataset':24s}" + ''.join(f" {m[:14]:>15s}" for m in methods)
    print(head)
    print('  ' + '-' * (len(head) - 2))
    for s in summaries:
        row = f"  {s['dataset']:24s}"
        for m in methods:
            row += f" {s['win_counts'].get(m, 0):>15d}"
        print(row + f"   (of {s['n_cells']} cells)")
    n_agree = sum(1 for s in summaries if s['folds_agree'])
    print(f"\n  Datasets where all folds vote the same way at the shipped "
          f"setting: {n_agree} of {len(summaries)}")
    for s in summaries:
        if not s['folds_agree']:
            print(f"    {s['dataset']}: " + '; '.join(
                f"fold {k} dropped {v or 'nothing'}"
                for k, v in sorted(s['per_fold_pruned'].items())))


def main():
    ap = argparse.ArgumentParser(description='f33 pruning replay')
    ap.add_argument('--results-dir', type=str, default=str(RESULTS_DIR))
    ap.add_argument('--datasets', type=str, default=None)
    ap.add_argument('--budgets', type=str, default=None,
                    help='restrict the replay to these budgets, e.g. 50,150,300')
    args = ap.parse_args()

    datasets = ([d.strip() for d in args.datasets.split(',')]
                if args.datasets else None)
    budgets = ([int(b) for b in args.budgets.split(',')]
               if args.budgets else None)

    records = load_records(args.results_dir, datasets, budgets)
    if not records:
        print(f"No f33 records in {args.results_dir}")
        return 1

    print(f"f33 pruning replay — {len(records)} dataset(s), nothing refitted")
    if budgets:
        print(f"restricted to budgets {budgets}")
    summaries = [report_dataset(rec) for rec in records]
    report_across(summaries)
    return 0


if __name__ == '__main__':
    sys.exit(main())
