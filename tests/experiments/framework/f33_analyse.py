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


def _n(r):
    """This configuration's representation count. '' when the record predates
    the axis, so an old record and a new one never share a group by accident."""
    v = r.get('rep_count', 'absent')
    return 'all' if v is None else v


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
        grp = (r['model'], r['internal_evaluator'], _n(r))
        if (r['allocation_method'], r['importance_method']) in excluded_pairs.get(grp, ()):
            continue
        key = (r['importance_method'], r['internal_evaluator'],
               r['model'], r['budget'], _n(r))
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
            grp = (r['model'], r['internal_evaluator'], _n(r))
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
        grp = (r['model'], r['internal_evaluator'], _n(r))
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
        grp = (r['model'], r['internal_evaluator'], _n(r))
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


# =============================================================================
# STEP 3 — how much the voting fold can actually separate
# =============================================================================

VOTE_SCHEMES = [
    ('fold 0', [0], 'mean'),
    ('fold 1', [1], 'mean'),
    ('fold 2', [2], 'mean'),
    ('fold 3', [3], 'mean'),
    ('fold 4', [4], 'mean'),
    ('folds 0+1 mean', [0, 1], 'mean'),
    ('folds 0+1 points', [0, 1], 'points'),
]


def separability(rec, vote_folds, top_n=SHIPPED_TOP_N):
    """How many of the groups the voting fold can tell apart.

    In a group of 5 allocation methods, a tie is broken by the order the methods
    sit in the list, not by any score. Three counts are returned:
      all_tied     every one of the 5 scores in the group is the same number
      edge_tied    the score at place top_n equals the score at place top_n+1, so
                   which method takes the last point is decided by list order
      mean_distinct  average count of distinct score values in a group
    """
    lib = rec['lower_is_better']
    groups = defaultdict(list)
    for r in rec['results']:
        key = (r['importance_method'], r['internal_evaluator'],
               r['model'], r['budget'], _n(r))
        groups[key].append(r)

    all_tied = 0
    edge_tied = 0
    distinct = []
    for entries in groups.values():
        vals = [_fold_mean(r, vote_folds, lib) for r in entries]
        finite = [v for v in vals if np.isfinite(v)]
        uniq = len(set(round(v, 12) for v in finite)) + (1 if len(finite) < len(vals) else 0)
        distinct.append(uniq)
        if uniq <= 1:
            all_tied += 1
        srt = sorted(vals, reverse=not lib)
        if len(srt) > top_n and srt[top_n - 1] == srt[top_n]:
            edge_tied += 1
    return {'n_groups': len(groups), 'all_tied': all_tied,
            'edge_tied': edge_tied,
            'mean_distinct': float(np.mean(distinct)) if distinct else 0.0}


def print_rep_count_axis(rec):
    """What the representation-count axis buys, and whether it is prunable.

    Two questions, both per dataset and never averaged across them:
      1. which count wins each of the (model, budget) cells, and what the whole
         search loses if that count is not reachable;
      2. whether one or two folds can tell which count is going to lose.
    """
    lib = rec['lower_is_better']
    counts = sorted({_n(r) for r in rec['results']}, key=lambda v: (v == 'all', v))
    if len(counts) < 2:
        print(f"\n  Representation count: this record has only {counts} — "
              f"the axis was not swept, nothing to report.")
        return

    print(f"\n  THE REPRESENTATION-COUNT AXIS. One configuration is restricted to "
          f"the N representations")
    print(f"  that fold's allocation scores rank highest; 'all' is the whole pool "
          f"of {rec['n_representations']}.")

    # --- which count wins each cell ---
    best = {}
    for r in rec['results']:
        cell = (r['model'], r['budget'])
        m = _full_mean(r, lib)
        if cell not in best or _better(m, best[cell][0], lib):
            best[cell] = (m, r)
    wins = defaultdict(int)
    for _, r in best.values():
        wins[_n(r)] += 1
    hrs = defaultdict(float)
    for r in rec['results']:
        if r['time_seconds']:
            hrs[_n(r)] += r['time_seconds'] / 3600
    tot = sum(hrs.values())
    print(f"\n    {'count':>8s}{'cells won':>12s}{'hours':>10s}{'share':>8s}")
    for c in counts:
        print(f"    {str(c):>8s}{wins.get(c, 0):>6d} of {len(best):<4d}"
              f"{hrs.get(c, 0):>9.2f}h{hrs.get(c, 0) / tot * 100 if tot else 0:>7.1f}%")

    # --- what each count alone would reach ---
    print(f"\n    If only ONE count were searched, the best that search could "
          f"return, in {rec['metric']}:")
    overall = (min if lib else max)(_full_mean(r, lib) for r in rec['results'])
    for c in counts:
        rows = [r for r in rec['results'] if _n(r) == c]
        b = (min if lib else max)(_full_mean(r, lib) for r in rows)
        h = sum(r['time_seconds'] for r in rows if r['time_seconds']) / 3600
        print(f"      count {str(c):>4s}: {b:.4f}  "
              f"({abs(b - overall):.4f} short of the whole axis' {overall:.4f})  "
              f"{h:.2f} h of {tot:.2f} h")

    # --- can one or two folds tell which count will lose? ---
    print(f"\n    Cutting a count on the evidence of the first folds. A count is cut")
    print(f"    when its best score so far is more than the threshold behind the best")
    print(f"    score so far. lost = of the {len(best)} (model, budget) winners.")
    for folds, label in (([0], 'after fold 0'), ([0, 1], 'after folds 0 and 1')):
        cells = []
        for thr in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
            sofar = {}
            for r in rec['results']:
                v = _fold_mean(r, folds, lib)
                c = _n(r)
                if c not in sofar or _better(v, sofar[c], lib):
                    sofar[c] = v
            bb = (min if lib else max)(sofar.values())
            cut = {c for c, v in sofar.items()
                   if (abs(v - bb) / abs(bb) if lib else (bb - v) / abs(bb)) > thr}
            if len(cut) >= len(sofar):
                cut = set(list(cut)[:len(sofar) - 1])
            surv = [r for r in rec['results'] if _n(r) not in cut]
            reach = {}
            for r in surv:
                cell = (r['model'], r['budget'])
                m = _full_mean(r, lib)
                if cell not in reach or _better(m, reach[cell], lib):
                    reach[cell] = m
            lost = [(c, m) for c, (m, _) in best.items()
                    if c not in reach or abs(reach[c] - m) > 1e-12]
            gaps = [abs(m - reach[c]) for c, m in lost if c in reach]
            saved = sum(r['time_seconds'] for r in rec['results']
                        if _n(r) in cut and r['time_seconds']) \
                * (rec['n_folds_run'] - len(folds)) / rec['n_folds_run'] / 3600
            cells.append((thr, sorted(cut, key=str), len(lost),
                          float(np.mean(gaps)) if gaps else 0.0, saved))
        print(f"      {label}:")
        for thr, cut, nl, gap, sv in cells:
            print(f"        {thr:>5.0%}  cuts {str(cut) if cut else 'nothing':<22s} "
                  f"lost {nl:>2d} of {len(best):<3d} mean gap {gap:.4f} "
                  f"{rec['metric']}  saves {sv:.2f} h")


def print_separability(rec):
    print(f"\n  How much each voting fold can separate. One group is one")
    print(f"  (importance method, internal evaluator, model, budget), holding the 5")
    print(f"  allocation methods. A tie is broken by list order, not by score, so a")
    print(f"  fold with many ties is voting on the order the methods are written in.")
    print(f"    {'voting fold':>18s} {'groups':>8s} {'all 5 tied':>12s} "
          f"{'3rd=4th tied':>14s} {'distinct scores':>16s}")
    for label, folds, combine in VOTE_SCHEMES:
        if combine == 'points':
            continue
        if max(folds) >= rec['n_folds_run']:
            continue
        s = separability(rec, folds)
        print(f"    {label:>18s} {s['n_groups']:>8d} "
              f"{s['all_tied']:>6d} ({s['all_tied'] / s['n_groups']:>4.0%}) "
              f"{s['edge_tied']:>7d} ({s['edge_tied'] / s['n_groups']:>4.0%}) "
              f"{s['mean_distinct']:>15.2f} of 5")


def print_vote_sweep(rec):
    """Step 3 in full: every voting scheme x every N x every threshold."""
    methods = rec['grid']['allocation_methods']
    print(f"\n  STEP 3 — the allocation vote swept over N, threshold and which fold votes.")
    print(f"  Cell reads deleted/lost: how many of the {len(methods)} allocation methods")
    print(f"  the setting deletes, then how many of the {len(model_budget_winners(rec)[0])} "
          f"(model, budget) winners")
    print(f"  can no longer be reached. '.' means it deletes nothing. "
          f"'{{n}}/SAFE' means it deletes n and still reaches every winner.")

    seen_outcomes = {}
    for label, folds, combine in VOTE_SCHEMES:
        if max(folds) >= rec['n_folds_run']:
            continue
        print(f"\n    voting on {label}:")
        print(f"      {'N':>3s} " + ''.join(f"{t:>12.0%}" for t in THRESHOLD_VALUES))
        for top_n in TOP_N_VALUES:
            cells = []
            for thr in THRESHOLD_VALUES:
                kept, pruned, places, n_groups, thr_places = allocation_vote(
                    rec, folds, top_n, thr, combine=combine)
                if not pruned:
                    cells.append(f"{'.':>12s}")
                    continue
                lost, n_cells = winners_lost(rec, kept)
                key = (label, tuple(sorted(pruned)))
                if key not in seen_outcomes:
                    saved, total = seconds_saved(rec, pruned, folds)
                    diffs = [d for (*_, d) in lost if d is not None and np.isfinite(d)]
                    seen_outcomes[key] = {
                        'pruned': sorted(pruned), 'lost': lost, 'n_cells': n_cells,
                        'saved': saved, 'total': total,
                        'mean_diff': float(np.mean(diffs)) if diffs else 0.0,
                        'settings': [],
                    }
                seen_outcomes[key]['settings'].append((top_n, thr))
                txt = f"{len(pruned)}/SAFE" if not lost else f"{len(pruned)}/{len(lost)}"
                star = '*' if (top_n == SHIPPED_TOP_N
                               and abs(thr - SHIPPED_THRESHOLD) < 1e-9
                               and label == 'fold 0') else ' '
                cells.append(f"{txt:>11s}{star}")
            print(f"      {top_n:>3d} " + ''.join(cells))
    print(f"      * = the shipped setting, top-{SHIPPED_TOP_N} at {SHIPPED_THRESHOLD:.0%} on fold 0")

    print(f"\n    Every distinct deletion the sweep produced, priced:")
    if not seen_outcomes:
        print(f"      none — no setting anywhere in the sweep deletes a method")
        return
    for (label, _), o in sorted(seen_outcomes.items(),
                                key=lambda kv: (kv[0][0], len(kv[1]['pruned']))):
        st = o['settings']
        rng = (f"N {min(n for n, _ in st)}-{max(n for n, _ in st)}, "
               f"threshold {min(t for _, t in st):.0%}-{max(t for _, t in st):.0%}")
        print(f"      voting on {label}, deletes {o['pruned']} ({rng}, "
              f"{len(st)} of {len(TOP_N_VALUES) * len(THRESHOLD_VALUES)} settings)")
        if not o['lost']:
            print(f"        every one of the {o['n_cells']} winners still reachable; "
                  f"saves {o['saved'] / 3600:.2f} h of {o['total'] / 3600:.2f} h")
        else:
            print(f"        {len(o['lost'])} of {o['n_cells']} winners lost, mean gap "
                  f"{o['mean_diff']:.4f} {rec['metric']}; "
                  f"saves {o['saved'] / 3600:.2f} h of {o['total'] / 3600:.2f} h")
            for line in _fmt_lost(rec, o['lost'], limit=25):
                print(line)


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

    print_rep_count_axis(rec)
    print_separability(rec)
    print_vote_sweep(rec)

    # --- is the fold-0 vote noisy? ---
    print(f"\n  The shipped setting run on each fold on its own — "
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

    fast_summary = report_fast(rec)

    return {
        'dataset': rec['dataset'],
        'metric': rec['metric'],
        'fast': fast_summary,
        'n_cells': len(winners),
        'win_counts': dict(counts),
        'per_fold_pruned': {k: list(v) for k, v in per_fold.items()},
        'folds_agree': agree,
    }


# =============================================================================
# SIDE QUESTION — is the fast search's selection adequate?
# =============================================================================

def fast_lists():
    """The four lists `speed='fast'` substitutes, read from the engine.

    Read at runtime rather than copied, so a change to any of them shows up here
    instead of being silently compared against a stale copy.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[3]
    if str(_root / 'src') not in _sys.path:
        _sys.path.insert(0, str(_root / 'src'))
    from kirby.find_best_hybrid import (
        FAST_ALLOCATION_METHODS, FAST_IMPORTANCE_METHODS,
        FAST_INTERNAL_EVALUATORS, FAST_MODEL_TIER, DEFAULT_MODELS,
    )
    models = sorted(n for n, spec in DEFAULT_MODELS.items()
                    if spec.get('tier') == FAST_MODEL_TIER)
    return {
        'allocation': list(FAST_ALLOCATION_METHODS),
        'importance': list(FAST_IMPORTANCE_METHODS),
        'evaluators': list(FAST_INTERNAL_EVALUATORS),
        'models': models,
    }


def _fast_evaluators_for(model, fast_evaluators):
    """`match` resolves to the model's own type, the same expansion the search does."""
    evs = {e for e in fast_evaluators if e != 'match'}
    if 'match' in fast_evaluators:
        evs.add(model)
    return evs


def _fast_reachable(row, fast):
    return (row['model'] in fast['models']
            and row['allocation_method'] in fast['allocation']
            and row['importance_method'] in fast['importance']
            and row['internal_evaluator'] in _fast_evaluators_for(row['model'],
                                                                  fast['evaluators']))


def report_fast(rec):
    """What the fast lists can and cannot reach in this record.

    The fast search is a SUBSET of the grid f33 recorded, so this needs no extra
    fitting. It asks two things. On the (model, budget) cells fast still runs, is
    the configuration the full grid found reachable at all? And what does the
    whole fast search return against what the whole full search returns?

    One consequence worth keeping in view: every one of these four lists is in the
    search cache key (find_best_hybrid.py _make_search_meta), so a fast run files
    separately and cannot be topped up into a full one later.
    """
    lib = rec['lower_is_better']
    fast = fast_lists()
    methods = rec['grid']['allocation_methods']
    importance = rec['grid']['importance_methods']
    all_models = rec['grid']['models']

    print(f"\n  FAST SEARCH — what speed='fast' can reach in this record")
    print(f"    allocation: {fast['allocation']}")
    print(f"      out of reach: {sorted(set(methods) - set(fast['allocation'])) or 'none'}")
    print(f"    importance: {fast['importance']}")
    print(f"      out of reach: {sorted(set(importance) - set(fast['importance'])) or 'none'}")
    print(f"    internal evaluators: {fast['evaluators']} "
          f"('match' becomes the model's own type)")
    print(f"    models: {fast['models']}")
    print(f"      out of reach: {sorted(set(all_models) - set(fast['models'])) or 'none'}")

    rows = rec['results']
    reachable = [r for r in rows if _fast_reachable(r, fast)]
    if not reachable:
        print(f"    nothing in this record is reachable under the fast lists")
        return None

    # Whole-search comparison: best configuration either search can return.
    def _best_of(rs):
        scored = [(r, _full_mean(r, lib)) for r in rs]
        return (min(scored, key=lambda t: t[1]) if lib
                else max(scored, key=lambda t: t[1]))

    full_best, full_score = _best_of(rows)
    fast_best, fast_score = _best_of(reachable)
    print(f"\n    best the FULL grid returns: {full_best['model']}/"
          f"{full_best['allocation_method']}/{full_best['importance_method']}/"
          f"budget {full_best['budget']}/evaluator {full_best['internal_evaluator']}"
          f" = {full_score:.4f} {rec['metric']}")
    print(f"    best the FAST search returns: {fast_best['model']}/"
          f"{fast_best['allocation_method']}/{fast_best['importance_method']}/"
          f"budget {fast_best['budget']}/evaluator {fast_best['internal_evaluator']}"
          f" = {fast_score:.4f} {rec['metric']}")
    print(f"    difference: {abs(full_score - fast_score):.4f} {rec['metric']}")

    # Per-cell, restricted to the models fast still runs — so a miss is the fault
    # of the allocation, importance or evaluator list, not of dropping a model.
    print(f"\n    On the (model, budget) cells fast still runs, is the full "
          f"grid's winner reachable?")
    cells = sorted({(r['model'], r['budget']) for r in rows
                    if r['model'] in fast['models']})
    n_miss = 0
    axis_blame = defaultdict(int)
    for cell in cells:
        in_cell = [r for r in rows if (r['model'], r['budget']) == cell]
        win, win_score = _best_of(in_cell)
        in_cell_fast = [r for r in in_cell if _fast_reachable(r, fast)]
        if not in_cell_fast:
            print(f"      {cell[0]:9s} budget {cell[1]:4d}: nothing reachable")
            n_miss += 1
            continue
        fbest, fscore = _best_of(in_cell_fast)
        same = (win['allocation_method'] == fbest['allocation_method']
                and win['importance_method'] == fbest['importance_method']
                and win['internal_evaluator'] == fbest['internal_evaluator'])
        if same:
            print(f"      {cell[0]:9s} budget {cell[1]:4d}: reached "
                  f"({win['allocation_method']}/{win['importance_method']}/"
                  f"{win['internal_evaluator']}) {win_score:.4f} {rec['metric']}")
            continue
        n_miss += 1
        blame = []
        if win['allocation_method'] not in fast['allocation']:
            blame.append('allocation')
        if win['importance_method'] not in fast['importance']:
            blame.append('importance')
        if win['internal_evaluator'] not in _fast_evaluators_for(cell[0],
                                                                 fast['evaluators']):
            blame.append('evaluator')
        for b in blame:
            axis_blame[b] += 1
        print(f"      {cell[0]:9s} budget {cell[1]:4d}: MISSED — full winner "
              f"{win['allocation_method']}/{win['importance_method']}/"
              f"{win['internal_evaluator']} {win_score:.4f}, best fast "
              f"{fbest['allocation_method']}/{fbest['importance_method']}/"
              f"{fbest['internal_evaluator']} {fscore:.4f}, "
              f"difference {abs(win_score - fscore):.4f} "
              f"({', '.join(blame) if blame else 'a lower-scoring combination'} "
              f"out of reach)")
    print(f"    {n_miss} of {len(cells)} cells miss the full grid's winner")
    if axis_blame:
        print(f"    which list excluded it: " + ', '.join(
            f"{k} on {v} cells" for k, v in sorted(axis_blame.items())))

    t_fast = sum(r['time_seconds'] for r in reachable if r['time_seconds'])
    t_full = sum(r['time_seconds'] for r in rows if r['time_seconds'])
    print(f"    cost: {len(reachable)} of {len(rows)} configurations, "
          f"{t_fast / 3600:.2f} h of {t_full / 3600:.2f} h measured")

    return {
        'n_cells': len(cells), 'n_miss': n_miss,
        'axis_blame': dict(axis_blame),
        'full_score': full_score, 'fast_score': fast_score,
        'seconds_fast': t_fast, 'seconds_full': t_full,
        'n_configs_fast': len(reachable), 'n_configs_full': len(rows),
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
    fasts = [s for s in summaries if s.get('fast')]
    if fasts:
        print(f"\n  Fast search: cells whose winner the fast lists cannot reach, "
              f"and what the whole search loses:")
        print(f"  {'dataset':24s} {'cells missed':>14s} {'full best':>11s} "
              f"{'fast best':>11s} {'difference':>11s} {'hours':>14s}")
        for s in fasts:
            f = s['fast']
            print(f"  {s['dataset']:24s} {f['n_miss']:>6d} of {f['n_cells']:<5d} "
                  f"{f['full_score']:>11.4f} {f['fast_score']:>11.4f} "
                  f"{abs(f['full_score'] - f['fast_score']):>11.4f} "
                  f"{f['seconds_fast'] / 3600:>6.2f} of {f['seconds_full'] / 3600:<5.2f}")
        blame = defaultdict(int)
        for s in fasts:
            for k, v in s['fast']['axis_blame'].items():
                blame[k] += v
        if blame:
            print(f"  which fast list excluded the winner, summed over datasets: "
                  + ', '.join(f"{k} on {v} cells" for k, v in sorted(blame.items())))

    n_agree = sum(1 for s in summaries if s['folds_agree'])
    print(f"\n  Datasets where all folds vote the same way at the shipped "
          f"setting: {n_agree} of {len(summaries)}")
    for s in summaries:
        if not s['folds_agree']:
            print(f"    {s['dataset']}: " + '; '.join(
                f"fold {k} dropped {v or 'nothing'}"
                for k, v in sorted(s['per_fold_pruned'].items())))


# =============================================================================
# THE RULE COMPARISON TABLE — one row per rule, one column per dataset
# =============================================================================

def _rule_cell(rec, kind, **kw):
    """Run one rule on one dataset. Returns (n_pruned, n_cells_lost, mean_difference)."""
    lib = rec['lower_is_better']
    if kind == 'vote':
        kept, pruned, _, _, _ = allocation_vote(
            rec, kw['folds'], kw['top_n'], kw['threshold'], combine=kw.get('combine', 'mean'))
        lost, n_cells = winners_lost(rec, kept)
        pairs = None
    else:
        pairs, fired = tail_drop(rec, kw['checkpoints'], kw['frac'])
        if not fired:
            return None
        kept, pruned, _, _, _ = allocation_vote(
            rec, [0], SHIPPED_TOP_N, SHIPPED_THRESHOLD, excluded_pairs=pairs)
        lost, n_cells = winners_lost(rec, kept, dropped_pairs=pairs)
    diffs = [d for (*_, d) in lost if d is not None and np.isfinite(d)]
    return (len(pruned), len(lost), n_cells,
            float(np.mean(diffs)) if diffs else 0.0)


def rule_rows():
    """Every rule the table reports, in order, as (label, kind, kwargs)."""
    rows = []
    for top_n in TOP_N_VALUES:
        for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            label = f'fold0_top{top_n}_lt{int(thr*100)}pct'
            if top_n == SHIPPED_TOP_N and abs(thr - SHIPPED_THRESHOLD) < 1e-9:
                label += '  <-SHIPPED'
            rows.append((label, 'vote',
                         dict(folds=[0], top_n=top_n, threshold=thr)))
    for k in range(1, 5):
        rows.append((f'fold{k}only_top3_lt15pct', 'vote',
                     dict(folds=[k], top_n=SHIPPED_TOP_N, threshold=SHIPPED_THRESHOLD)))
    rows.append(('folds01_mean_top3_lt15pct', 'vote',
                 dict(folds=[0, 1], top_n=SHIPPED_TOP_N,
                      threshold=SHIPPED_THRESHOLD, combine='mean')))
    rows.append(('folds01_points_top3_lt15pct', 'vote',
                 dict(folds=[0, 1], top_n=SHIPPED_TOP_N,
                      threshold=SHIPPED_THRESHOLD, combine='points')))
    for frac in TAIL_DROP_VALUES:
        label = f'taildrop_25_50_100_at{int(frac*100)}pct'
        if abs(frac - 0.50) < 1e-9:
            label += '  <-AGGRESSIVE'
        rows.append((label, 'tail',
                     dict(checkpoints=AGGRESSIVE_CHECKPOINTS, frac=frac)))
    return rows


def print_rule_table(records):
    """The table: rule down the side, dataset across the top.

    Each cell reads "Np  NL  d=X.XXXX":
      Np       how many of the 5 allocation methods the rule deletes
      NL       how many of the (model, budget) cells lose the method that won them
      d=       mean gap, in the dataset's own metric, between the method that won a
               lost cell and the best method the rule left behind
    A cell reading "Np  SAFE" loses no winner.
    """
    rows = rule_rows()
    names = [r['dataset'] for r in records]
    w = 22
    print(f"\n{'=' * (30 + (w + 3) * len(names))}")
    print(f"  PRUNING RULE COMPARISON — {len(records)} dataset(s)")
    print(f"{'=' * (30 + (w + 3) * len(names))}")
    print(f"\n  Each cell: allocation methods deleted / (model, budget) cells whose")
    print(f"  winner is lost / mean gap to the best method left, in that dataset's metric.")
    print(f"  Every dataset has 5 allocation methods and 25 cells "
          f"(nt_promoter_tata has 20, it runs 4 models).\n")
    header = f"  {'Rule':30s}" + ''.join(f" | {n[:w]:>{w}s}" for n in names)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for label, kind, kw in rows:
        cells = []
        for rec in records:
            out = _rule_cell(rec, kind, **kw)
            if out is None:
                cells.append('no checkpoint fires')
                continue
            n_pruned, n_lost, n_cells, mean_diff = out
            if n_lost == 0:
                cells.append(f'{n_pruned}p  SAFE')
            else:
                cells.append(f'{n_pruned}p {n_lost:2d}/{n_cells}L d={mean_diff:.4f}')
        print(f"  {label:30s}" + ''.join(f" | {c:>{w}s}" for c in cells))

    print(f"\n  Best configuration each dataset's full grid found:")
    for rec in records:
        lib = rec['lower_is_better']
        best = (min(rec['results'], key=lambda r: _full_mean(r, lib)) if lib
                else max(rec['results'], key=lambda r: _full_mean(r, lib)))
        arrow = 'lower is better' if lib else 'higher is better'
        print(f"    {rec['dataset']:24s} {best['model']}/"
              f"{best['allocation_method']}/{best['importance_method']}/"
              f"budget {best['budget']}/evaluator {best['internal_evaluator']}"
              f" = {_full_mean(best, lib):.4f} {rec['metric']} ({arrow})")

    print(f"\n  How many of the 25 (model, budget) cells each allocation method wins:")
    methods = records[0]['grid']['allocation_methods']
    head = f"  {'dataset':24s}" + ''.join(f" {m[:20]:>21s}" for m in methods)
    print(head)
    print('  ' + '-' * (len(head) - 2))
    for rec in records:
        winners, _ = model_budget_winners(rec)
        c = defaultdict(int)
        for (wname, _) in winners.values():
            c[wname] += 1
        print(f"  {rec['dataset']:24s}" + ''.join(f" {c.get(m, 0):>21d}" for m in methods))


# =============================================================================
# WHERE TO PRUNE — what each axis costs and what it buys
# =============================================================================

def axis_cost_and_wins(rec):
    """For every member of every axis: hours spent on it, and cells it wins.

    The prune deletes allocation methods today. This asks the prior question:
    across all five axes, which members are expensive and win nothing? Those are
    the places worth pruning, whatever mechanism does the pruning.

    "Cells won" counts the 25 (model, budget) cells whose best configuration
    contains that member. Budget and model ARE the cell key, so for those two the
    count is over models instead: how many models have their best at that budget,
    and how many budgets each model wins outright.
    """
    lib = rec['lower_is_better']
    rows = rec['results']
    winners = {}
    for r in rows:
        cell = (r['model'], r['budget'])
        m = _full_mean(r, lib)
        cur = winners.get(cell)
        if cur is None or _better(m, cur[1], lib):
            winners[cell] = (r, m)

    out = {}
    for axis, field in (('allocation method', 'allocation_method'),
                        ('importance method', 'importance_method'),
                        ('internal evaluator', 'internal_evaluator')):
        members = sorted({r[field] for r in rows})
        stats = {}
        for mem in members:
            secs = sum(r['time_seconds'] for r in rows
                       if r[field] == mem and r['time_seconds'])
            wins = sum(1 for (w, _) in winners.values() if w[field] == mem)
            stats[mem] = (secs, wins, len(winners))
        out[axis] = stats

    # Budget: how many models put their best at this budget.
    best_per_model = {}
    for (model, budget), (r, m) in winners.items():
        cur = best_per_model.get(model)
        if cur is None or _better(m, cur[1], lib):
            best_per_model[model] = (budget, m)
    bstats = {}
    for b in sorted({r['budget'] for r in rows}):
        secs = sum(r['time_seconds'] for r in rows
                   if r['budget'] == b and r['time_seconds'])
        wins = sum(1 for (bb, _) in best_per_model.values() if bb == b)
        bstats[str(b)] = (secs, wins, len(best_per_model))
    out['budget'] = bstats

    # Model: the model is the answer the search returns, so "wins" is whether it
    # holds the best score on the whole dataset.
    overall = (min(rows, key=lambda r: _full_mean(r, lib)) if lib
               else max(rows, key=lambda r: _full_mean(r, lib)))
    mstats = {}
    for mo in sorted({r['model'] for r in rows}):
        secs = sum(r['time_seconds'] for r in rows
                   if r['model'] == mo and r['time_seconds'])
        mstats[mo] = (secs, 1 if overall['model'] == mo else 0, 1)
    out['model'] = mstats
    return out


def print_where_to_prune(records):
    print(f"\n{'=' * 96}")
    print(f"  WHERE TO PRUNE — what each axis member costs, and what it wins")
    print(f"{'=' * 96}")
    print(f"\n  hours = measured wall time on every configuration containing that member.")
    print(f"  wins  = (model, budget) cells whose best configuration contains it, out of 25.")
    print(f"          For budget: how many of the 5 models put their best there.")
    print(f"          For model: whether it holds the dataset's single best score.")
    print(f"  A member with many hours and no wins is a place to prune.")

    for rec in records:
        stats = axis_cost_and_wins(rec)
        total = sum(r['time_seconds'] for r in rec['results'] if r['time_seconds'])
        print(f"\n  {rec['dataset']} — {rec['n_samples']} rows, metric {rec['metric']}, "
              f"{total / 3600:.2f} h measured over {len(rec['results'])} configurations")
        for axis, members in stats.items():
            print(f"    {axis}")
            ordered = sorted(members.items(), key=lambda kv: -kv[1][0])
            for mem, (secs, wins, n_cells) in ordered:
                bar = '#' * int(round(30 * secs / total)) if total else ''
                print(f"      {mem:22s} {secs / 3600:6.2f} h "
                      f"({secs / total * 100:4.1f}%)  wins {wins:2d}/{n_cells:<2d} {bar}")


def print_tradeoff(records):
    """HOW to prune: every rule ordered by what it saves, against what it costs."""
    print(f"\n{'=' * 96}")
    print(f"  HOW TO PRUNE — every setting, by hours saved against winners lost")
    print(f"{'=' * 96}")
    for rec in records:
        total = sum(r['time_seconds'] for r in rec['results'] if r['time_seconds'])
        n_folds = rec['n_folds_run']
        print(f"\n  {rec['dataset']} — {total / 3600:.2f} h measured, metric {rec['metric']}")
        print(f"    {'setting':32s} {'saves':>16s} {'cells lost':>12s} "
              f"{'mean gap':>12s}  methods deleted")
        entries = []
        for label, kind, kw in rule_rows():
            if kind == 'vote':
                kept, pruned, _, _, _ = allocation_vote(
                    rec, kw['folds'], kw['top_n'], kw['threshold'],
                    combine=kw.get('combine', 'mean'))
                lost, n_cells = winners_lost(rec, kept)
                saved, _ = seconds_saved(rec, pruned, kw['folds'])
                pairs = None
            else:
                pairs, fired = tail_drop(rec, kw['checkpoints'], kw['frac'])
                if not fired:
                    continue
                kept, pruned, _, _, _ = allocation_vote(
                    rec, [0], SHIPPED_TOP_N, SHIPPED_THRESHOLD, excluded_pairs=pairs)
                lost, n_cells = winners_lost(rec, kept, dropped_pairs=pairs)
                saved, _ = seconds_saved(rec, pruned, [0], dropped_pairs=pairs)
            diffs = [d for (*_, d) in lost if d is not None and np.isfinite(d)]
            entries.append((saved, len(lost), n_cells,
                            float(np.mean(diffs)) if diffs else 0.0,
                            pruned, label))
        for saved, n_lost, n_cells, gap, pruned, label in sorted(entries,
                                                                 key=lambda e: -e[0]):
            gap_s = f'{gap:.4f}' if n_lost else '-'
            print(f"    {label:32s} {saved / 3600:6.2f} h "
                  f"({saved / total * 100:4.1f}%) {n_lost:5d}/{n_cells:<6d} "
                  f"{gap_s:>12s}  {','.join(sorted(pruned)) if pruned else '-'}")


def main():
    ap = argparse.ArgumentParser(description='f33 pruning replay')
    ap.add_argument('--results-dir', type=str, default=str(RESULTS_DIR))
    ap.add_argument('--datasets', type=str, default=None)
    ap.add_argument('--table-only', action='store_true',
                    help='print only the rule comparison table')
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
    if args.table_only:
        print_where_to_prune(records)
        print_tradeoff(records)
        print_rule_table(records)
        return 0
    summaries = [report_dataset(rec) for rec in records]
    print_rule_table(records)
    report_across(summaries)
    return 0


if __name__ == '__main__':
    sys.exit(main())
