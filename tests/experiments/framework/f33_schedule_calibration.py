#!/usr/bin/env python3
"""f33 schedule calibration — the f16 analysis, re-run on today's search space.

This is f16_multistage_pruning.py's analysis, definition for definition, applied
to the record f33_pruning_recalibration.py produced. f16 chose the shipped
`aggressive_pruning` schedule (checkpoints [25, 50, 100], tail_drop 0.5) on ONE
dataset, caco2, in April, over 13 allocation methods of which only two survive.
Same tables, same pass rule, current search space.

DEFINITIONS, copied from f16 so the two are comparable.

  baseline_best_for_model
      the best score_mean over every configuration of that model, unpruned.

  pruned_best_for_model
      the same, over the configurations a schedule leaves alive. The schedule is
      the mid-flight tail drop at its checkpoints followed by the end-of-fold-0
      allocation vote at the shipped top-3 / 15%, exactly as `_run_search` runs
      them in that order.

  rel_loss = (pruned - baseline) / |baseline|      when lower is better
           = (baseline - pruned) / |baseline|      when higher is better
      POSITIVE means the pruned search returned a WORSE answer.

  pct_saved
      fold evaluations the schedule does not run, over the model's total.
      A configuration whose (allocation, importance) pair is tail-dropped skips
      all five folds at budgets above the checkpoint, and the four remaining
      folds at budgets at or below it, where fold 0 has already been paid.
      A configuration whose allocation method loses the end-of-fold-0 vote skips
      four of its five folds.

  trivial preservation (T)
      the model's unpruned best sits at a budget at or below the highest
      checkpoint, so the schedule was never given the chance to delete it.

  PASS
      rel_loss <= 2% AND not trivial.

Usage:
    python tests/experiments/framework/f33_schedule_calibration.py
    python tests/experiments/framework/f33_schedule_calibration.py --datasets logd
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'tests' / 'experiments' / 'framework'))

from f33_analyse import (  # noqa: E402
    load_records, allocation_vote, tail_drop, _full_mean, _better,
    SHIPPED_TOP_N, SHIPPED_THRESHOLD,
)

RESULTS_DIR = _ROOT / 'tests' / 'results' / 'framework' / 'f33_pruning'

# f16's own grid, unchanged.
PASS_THRESHOLD = 0.02
CHECKPOINT_CONFIGS = {
    '[25]':         [25],
    '[50]':         [50],
    '[100]':        [100],
    '[25,50]':      [25, 50],
    '[25,50,100]':  [25, 50, 100],
    '[10,25,50]':   [10, 25, 50],
}
# 0.0 is NOT in f16's grid and is added here on purpose: at tail_drop 0 the only
# thing running is the end-of-fold-0 allocation vote, which f16 held fixed at
# top-3 / 15% inside every cell. Without this row a loss the vote caused is read
# off the table as the tail drop's.
TAIL_DROPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def baseline_best_by_model(rec):
    """{model: {'score_mean': float, 'budget': int}} over the unpruned grid."""
    lib = rec['lower_is_better']
    out = {}
    for r in rec['results']:
        m = r['model']
        s = _full_mean(r, lib)
        if m not in out or _better(s, out[m]['score_mean'], lib):
            out[m] = {'score_mean': s, 'budget': r['budget']}
    return out


def evaluate_cell(rec, model, checkpoints, td, baseline, vote_threshold=None):
    """One (model, checkpoint config, tail_drop) cell of the table."""
    lib = rec['lower_is_better']
    n_folds = rec['n_folds_run']

    thr = SHIPPED_THRESHOLD if vote_threshold is None else vote_threshold
    pairs, fired = tail_drop(rec, checkpoints, td)
    if thr > 0:
        kept_alloc, pruned_alloc, _, _, _ = allocation_vote(
            rec, [0], SHIPPED_TOP_N, thr, excluded_pairs=pairs)
    else:
        kept_alloc = list(rec['grid']['allocation_methods'])
    kept_set = set(kept_alloc)

    rows = [r for r in rec['results'] if r['model'] == model]
    total_evals = len(rows) * n_folds
    avoided = 0
    alive = []
    max_ckpt = max(fired) if fired else 0
    for r in rows:
        grp = (r['model'], r['internal_evaluator'],
               'all' if r.get('rep_count', 'absent') is None
               else r.get('rep_count', 'absent'))
        pair = (r['allocation_method'], r['importance_method'])
        if pair in pairs.get(grp, ()):
            avoided += n_folds if r['budget'] > max_ckpt else (n_folds - 1)
            continue
        if r['allocation_method'] not in kept_set:
            avoided += n_folds - 1
            continue
        alive.append(r)

    pct_saved = (avoided / total_evals * 100) if total_evals else 0.0
    if not alive:
        return {'rel_loss': float('inf'), 'pct_saved': pct_saved,
                'trivial_preservation': False, 'pruned_best': None,
                'n_alive': 0, 'n_total': len(rows)}

    scored = [(r, _full_mean(r, lib)) for r in alive]
    pruned_best = (min(scored, key=lambda t: t[1])[1] if lib
                   else max(scored, key=lambda t: t[1])[1])
    base = baseline['score_mean']
    delta = (pruned_best - base) if lib else (base - pruned_best)
    rel_loss = delta / abs(base) if base else float('inf')
    trivial = bool(max_ckpt > 0 and baseline['budget'] <= max_ckpt)
    return {'rel_loss': rel_loss, 'pct_saved': pct_saved,
            'trivial_preservation': trivial, 'pruned_best': pruned_best,
            'n_alive': len(alive), 'n_total': len(rows)}


def survivors(rec, checkpoints, td, vote_threshold=None, models_pruned=None):
    """Configurations a schedule leaves alive.

    models_pruned: apply the schedule only to these models. Every other model
    keeps its whole grid, which is the shape f16 called an override.
    """
    thr = SHIPPED_THRESHOLD if vote_threshold is None else vote_threshold
    pairs, fired = tail_drop(rec, checkpoints, td)
    if thr > 0:
        kept_alloc, _, _, _, _ = allocation_vote(
            rec, [0], SHIPPED_TOP_N, thr, excluded_pairs=pairs)
    else:
        kept_alloc = list(rec['grid']['allocation_methods'])
    kept_set = set(kept_alloc)
    out = []
    for r in rec['results']:
        if models_pruned is not None and r['model'] not in models_pruned:
            out.append(r)
            continue
        grp = (r['model'], r['internal_evaluator'],
               'all' if r.get('rep_count', 'absent') is None
               else r.get('rep_count', 'absent'))
        if (r['allocation_method'], r['importance_method']) in pairs.get(grp, ()):
            continue
        if r['allocation_method'] not in kept_set:
            continue
        out.append(r)
    return out, pairs, fired


def _config_id(r):
    # The representation count is part of a configuration's identity: the same
    # recipe over the best 3 representations and over all 10 are two different
    # configurations with two different scores.
    return (r['model'], r['allocation_method'], r['importance_method'],
            r['budget'], r['internal_evaluator'],
            'all' if r.get('rep_count', 'absent') is None
            else r.get('rep_count', 'absent'))


def top_n_overlap(rec, alive, n):
    """How many of the unpruned grid's best n configurations survive.

    Returns (n_surviving, n). One configuration is one
    (model, allocation method, importance method, budget, internal evaluator).
    """
    lib = rec['lower_is_better']
    ranked = sorted(rec['results'], key=lambda r: _full_mean(r, lib),
                    reverse=not lib)
    want = {_config_id(r) for r in ranked[:n]}
    have = {_config_id(r) for r in alive}
    return len(want & have), n


def override_table(records, vote_threshold=None):
    """f16's override arms: the schedule on one model only, against all models.

    Each row is one (schedule, which models it is applied to). The columns say
    what it saves in fold evaluations, how many of the unpruned best three and
    best ten configurations survive, and the worst any model's best score moves.
    """
    print(f"\n{'=' * 100}")
    print(f"  Applying a schedule to SOME models only — f16's override arms")
    print(f"  One row is one schedule applied to the models named. "
          f"saves = fold evaluations not run, over the whole grid's.")
    print(f"  top3 / top10 = how many of the unpruned grid's best three and best ten")
    print(f"  configurations are still alive. worst move = the largest amount any")
    print(f"  model's best score got worse, in that dataset's own metric.")
    print(f"{'=' * 100}")
    arms = []
    for ckpts in ([25], [25, 50, 100]):
        for td in (0.3, 0.5, 0.7):
            for who, label in ((None, 'every model'),
                               (('svm',), 'svm only'),
                               (('dnn',), 'dnn only'),
                               (('svm', 'dnn'), 'svm and dnn')):
                arms.append((ckpts, td, who, label))
    for rec in records:
        lib = rec['lower_is_better']
        n_folds = rec['n_folds_run']
        total_evals = len(rec['results']) * n_folds
        base_by_model = baseline_best_by_model(rec)
        print(f"\n  {rec['dataset']} — metric {rec['metric']}, "
              f"{len(rec['results'])} configurations, {total_evals} fold evaluations")
        print(f"    {'checkpoints':>14s} {'tail_drop':>10s} {'applied to':>14s} "
              f"{'saves':>8s} {'top3':>7s} {'top10':>7s} {'worst move':>12s}")
        for ckpts, td, who, label in arms:
            alive, pairs, fired = survivors(rec, ckpts, td, vote_threshold, who)
            if not fired:
                continue
            kept_ids = {_config_id(r) for r in alive}
            avoided = 0
            max_ckpt = max(fired)
            for r in rec['results']:
                if _config_id(r) in kept_ids:
                    continue
                avoided += n_folds if r['budget'] > max_ckpt else (n_folds - 1)
            t3, _ = top_n_overlap(rec, alive, 3)
            t10, _ = top_n_overlap(rec, alive, 10)
            worst = 0.0
            for m, base in base_by_model.items():
                mine = [r for r in alive if r['model'] == m]
                if not mine:
                    worst = float('inf')
                    continue
                b = (min(mine, key=lambda r: _full_mean(r, lib)) if lib
                     else max(mine, key=lambda r: _full_mean(r, lib)))
                sc = _full_mean(b, lib)
                move = (sc - base['score_mean']) if lib else (base['score_mean'] - sc)
                worst = max(worst, move)
            wtxt = 'lost a model' if not np.isfinite(worst) else f'{worst:.4f}'
            print(f"    {str(ckpts):>14s} {td:>10.1f} {label:>14s} "
                  f"{avoided / total_evals * 100:>7.1f}% {t3:>4d}/3  {t10:>5d}/10 "
                  f"{wtxt:>12s}")


def per_dataset_tables(rec, vote_threshold=None):
    """f16's table 1: rows = tail_drop, columns = models, cells = loss% / saved%."""
    baseline = baseline_best_by_model(rec)
    models = sorted(baseline)
    print(f"\n{'=' * 100}")
    print(f"  {rec['dataset']} — {rec['n_samples']} rows, metric {rec['metric']}, "
          f"{len(rec['results'])} configurations, {rec['n_folds_run']} folds")
    print(f"{'=' * 100}")
    print(f"  unpruned best per model (score, and the budget it sits at):")
    for m in models:
        print(f"    {m:10s} {baseline[m]['score_mean']:.4f} {rec['metric']}"
              f"  at budget {baseline[m]['budget']}")

    cells = {}
    for ckpt_name, ckpts in CHECKPOINT_CONFIGS.items():
        cells[ckpt_name] = {}
        fires = [b for b in ckpts if b in set(rec['budgets_run'])]
        note = '' if fires else '   (no checkpoint is one of the budgets run)'
        print(f"\n  checkpoint = {ckpts}  fires at {fires}{note}"
              f"   (T = trivial, the model's best sits at or below the last checkpoint)")
        print(f"  {'tail_drop':>10s}  " + "  ".join(f"{m:>15s}" for m in models))
        for td in TAIL_DROPS:
            row = {}
            chunks = []
            for m in models:
                c = evaluate_cell(rec, m, ckpts, td, baseline[m], vote_threshold)
                row[m] = c
                T = 'T' if c['trivial_preservation'] else ' '
                if not np.isfinite(c['rel_loss']):
                    chunks.append(f"  {T} INF /{c['pct_saved']:5.1f}%")
                else:
                    chunks.append(f"  {T} {c['rel_loss'] * 100:+6.2f}%/{c['pct_saved']:5.1f}%")
            cells[ckpt_name][f"{td:.1f}"] = row
            print(f"  {td:>10.1f}  " + "".join(chunks))
    return {'dataset': rec['dataset'], 'metric': rec['metric'],
            'lower_is_better': rec['lower_is_better'],
            'n_samples': rec['n_samples'],
            'baseline_best_by_model': baseline, 'cells': cells}


def max_safe_tail_drop(per_dataset, models):
    """f16's table 2: the largest tail_drop that passes on every non-trivial dataset."""
    out = {}
    for ckpt_name in CHECKPOINT_CONFIGS:
        out[ckpt_name] = {}
        for m in models:
            best_td, used, avg_saved = None, [], 0.0
            had_evidence = False
            for td in sorted(TAIL_DROPS, reverse=True):
                key = f"{td:.1f}"
                nontrivial, trivial = [], []
                for ds in per_dataset:
                    c = ds['cells'].get(ckpt_name, {}).get(key, {}).get(m)
                    if c is None:
                        continue
                    (trivial if c['trivial_preservation'] else nontrivial).append(
                        (ds['dataset'], c))
                if not nontrivial:
                    continue
                had_evidence = True
                if all(c['rel_loss'] <= PASS_THRESHOLD for _, c in nontrivial):
                    best_td = td
                    used = [n for n, _ in nontrivial]
                    vals = [c['pct_saved'] for _, c in nontrivial + trivial]
                    avg_saved = sum(vals) / len(vals) if vals else 0.0
                    break
            out[ckpt_name][m] = {'max_safe_tail_drop': best_td,
                                 'had_evidence': had_evidence,
                                 'nontrivial_datasets_used': used,
                                 'avg_pct_saved_at_max': avg_saved}
    return out


def print_max_safe(max_safe, models):
    print(f"\n{'=' * 100}")
    print(f"  Largest tail_drop that holds, per model and checkpoint list")
    print(f"  holds = the pruned search's best is within {PASS_THRESHOLD * 100:.0f}% "
          f"of the unpruned best, on every dataset where the schedule had a chance to fail")
    print(f"  FAILS AT EVERY VALUE = even the gentlest tail drop costs this model "
          f"more than {PASS_THRESHOLD * 100:.0f}% on some dataset")
    print(f"  'no chance to fail' = the model's best always sat at or below the "
          f"last checkpoint, so nothing could delete it")
    print(f"{'=' * 100}")
    print(f"  {'model':10s}  " + "  ".join(f"{c:>22s}" for c in CHECKPOINT_CONFIGS))
    for m in models:
        chunks = []
        for ckpt_name in CHECKPOINT_CONFIGS:
            e = max_safe[ckpt_name].get(m, {})
            t = e.get('max_safe_tail_drop')
            if t is None:
                chunks.append(f"  {'FAILS AT EVERY VALUE' if e.get('had_evidence') else 'no chance to fail':>22s}")
            else:
                chunks.append(f"  tail_drop={t:.1f} ({len(e['nontrivial_datasets_used'])}ds,"
                              f" saves {e['avg_pct_saved_at_max']:.0f}%)")
        print(f"  {m:10s}  " + "".join(chunks))


def universal_combos(per_dataset, models):
    """f16's table 3: every (checkpoint list, tail_drop) where no model fails anywhere."""
    rows = []
    for ckpt_name in CHECKPOINT_CONFIGS:
        for td in TAIL_DROPS:
            key = f"{td:.1f}"
            nontrivial, saved = [], []
            for ds in per_dataset:
                for m in models:
                    c = ds['cells'].get(ckpt_name, {}).get(key, {}).get(m)
                    if c is None:
                        continue
                    saved.append(c['pct_saved'])
                    if not c['trivial_preservation']:
                        nontrivial.append((ds['dataset'], m, c))
            if not nontrivial:
                continue
            if all(c['rel_loss'] <= PASS_THRESHOLD for _, _, c in nontrivial):
                worst = max(c['rel_loss'] for _, _, c in nontrivial)
                rows.append((ckpt_name, td,
                             sum(saved) / len(saved) if saved else 0.0,
                             len(nontrivial), worst))
    return sorted(rows, key=lambda r: -r[2])


def print_universal(rows):
    print(f"\n{'=' * 100}")
    print(f"  Schedules where no model loses more than {PASS_THRESHOLD * 100:.0f}% "
          f"on any dataset, ordered by what they save")
    print(f"{'=' * 100}")
    if not rows:
        print(f"  none — every schedule costs some model more than "
              f"{PASS_THRESHOLD * 100:.0f}% on some dataset")
        return
    print(f"  {'checkpoints':>14s}  {'tail_drop':>10s}  {'saves':>8s}  "
          f"{'model-dataset pairs it could have failed':>42s}  {'worst loss':>11s}")
    for ckpt_name, td, saved, n_nt, worst in rows:
        print(f"  {ckpt_name:>14s}  {td:>10.1f}  {saved:>7.1f}%  {n_nt:>42d}  "
              f"{worst * 100:>10.2f}%")


def main():
    ap = argparse.ArgumentParser(description='f33 schedule calibration, f16 style')
    ap.add_argument('--results-dir', type=str, default=str(RESULTS_DIR))
    ap.add_argument('--datasets', type=str, default=None)
    ap.add_argument('--vote-off', action='store_true',
                    help='disable the end-of-fold-0 allocation vote, so the tables '
                         'show what the mid-flight tail drop does on its own')
    ap.add_argument('-o', '--output', type=str, default=None,
                    help='write the calibration JSON here')
    args = ap.parse_args()

    datasets = ([d.strip() for d in args.datasets.split(',')]
                if args.datasets else None)
    records = load_records(args.results_dir, datasets, None)
    if not records:
        print(f"No f33 records in {args.results_dir}")
        return 1

    print(f"f33 schedule calibration — the f16 analysis on today's search space")
    print(f"{len(records)} dataset(s): {[r['dataset'] for r in records]}")
    print(f"pass = the pruned search's best is within {PASS_THRESHOLD * 100:.0f}% "
          f"of the unpruned best")

    vt = 0.0 if args.vote_off else None
    if args.vote_off:
        print(f"END-OF-FOLD-0 ALLOCATION VOTE IS OFF — these tables are the "
              f"mid-flight tail drop alone")
    per_dataset = [per_dataset_tables(rec, vt) for rec in records]
    models = sorted({m for d in per_dataset for m in d['baseline_best_by_model']})
    ms = max_safe_tail_drop(per_dataset, models)
    print_max_safe(ms, models)
    print_universal(universal_combos(per_dataset, models))
    override_table(records, vt)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'experiment': 'f33_schedule_calibration',
                       'pass_threshold': PASS_THRESHOLD,
                       'checkpoint_configs': CHECKPOINT_CONFIGS,
                       'tail_drops': TAIL_DROPS,
                       'datasets_evaluated': [d['dataset'] for d in per_dataset],
                       'per_dataset': per_dataset,
                       'max_safe_tail_drop': ms}, f, indent=2, default=str)
        print(f"\n  written to {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
