#!/usr/bin/env python3
"""f32 read-out — does the best normalisation follow the metric or the dataset?

Reads every f32_<dataset>__fold<n>.json written by
f32_budget_split_normalisation.py and prints three things.

1. Per dataset, per weighting scheme, per budget: each arm's mean held-out score
   across the folds, in that dataset's own metric, alongside the raw pair
   (the arm's score and today's score) so nothing has to be read as a delta.
2. The comparison the experiment exists for: for each metric with two datasets,
   whether both datasets put the same arm first.
3. Whether the two controls held on every fold.

Usage:
    python tests/experiments/framework/f32_analyse.py
    python tests/experiments/framework/f32_analyse.py --input <dir>
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np


def load(input_dir):
    files = sorted(Path(input_dir).glob('f32_*__fold*.json'))
    payloads = []
    for f in files:
        try:
            payloads.append(json.loads(f.read_text()))
        except Exception as e:
            print(f'  unreadable: {f.name} ({type(e).__name__}: {e})')
    return payloads


# A fold this small cannot rank five arms against each other: on
# tcga_brca_pam50 fold 0 the held-out block is 3 molecules, and 28 of that
# fold's 30 cells scored f1_weighted exactly 1.0. Folds below this are dropped
# from the means and NAMED in the output — a stated reason, not a silent cut.
MIN_HELDOUT_ROWS = 20


def mean_across_folds(payloads):
    """(dataset, arm, weighting, budget) -> list of one held-out score per fold."""
    cells = defaultdict(list)
    meta = {}
    dropped = []
    for p in payloads:
        meta[p['dataset']] = {'metric': p['metric'],
                              'lower_is_better': p['lower_is_better'],
                              'n_samples': p.get('n_samples')}
        n_heldout = p.get('n_heldout_rows')
        if n_heldout is not None and n_heldout < MIN_HELDOUT_ROWS:
            dropped.append((p['dataset'], p['fold'], n_heldout))
            continue
        for r in p['rows']:
            if r.get('error') or r.get('primary') is None:
                continue
            cells[(r['dataset'], r['arm'], r['weighting'], r['budget'])].append(
                float(r['primary']))
    return cells, meta, dropped


def rank_arms(scores_by_arm, lower_is_better):
    """Arms best first. `scores_by_arm` is {arm: mean score in the metric}."""
    return sorted(scores_by_arm.items(), key=lambda kv: kv[1],
                  reverse=not lower_is_better)


def best_arms(scores_by_arm, lower_is_better):
    """Every arm that is first, not just the one that happens to sort first.

    Arms that pick the same features score identically — under rank_exp_d0.3,
    which reads only the order, that is ALL of them. Calling one of those the
    winner would turn a tie into a preference and make the metric-versus-dataset
    comparison read as disagreement when nothing disagreed.
    """
    ranked = rank_arms(scores_by_arm, lower_is_better)
    top = ranked[0][1]
    return frozenset(arm for arm, value in ranked if value == top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.environ.get(
        'F32_OUT', str(_ROOT / 'tests' / 'results' / 'framework')))
    args = ap.parse_args()

    payloads = load(args.input)
    if not payloads:
        print(f'no f32 result files in {args.input}')
        return 1
    print(f'{len(payloads)} result files from {args.input}\n')

    cells, meta, dropped = mean_across_folds(payloads)
    datasets = sorted(meta)
    for ds, fold, n in dropped:
        print(f'  dropped {ds} fold {fold}: the held-out block is {n} '
              f'molecules, under the {MIN_HELDOUT_ROWS} needed to tell five '
              f'arms apart. Its cells are in the result file, not in the means '
              f'below.')
    if dropped:
        print()

    # -- 1. per dataset ------------------------------------------------------
    winners = {}          # (dataset, weighting, budget) -> best arms
    spread_counts = []    # one per combination: did first place clear the spread
    for ds in datasets:
        m = meta[ds]
        metric, lower = m['metric'], m['lower_is_better']
        direction = 'lower is better' if lower else 'higher is better'
        print(f'=== {ds} — {metric}, {direction} ===')
        keys = {(w, b) for (d, _a, w, b) in cells if d == ds}
        for weighting, budget in sorted(keys):
            per_arm, n_folds, per_fold = {}, {}, {}
            for (d, arm, w, b), vals in cells.items():
                if d == ds and w == weighting and b == budget:
                    per_arm[arm] = float(np.mean(vals))
                    n_folds[arm] = len(vals)
                    per_fold[arm] = vals
            if not per_arm:
                continue
            today = per_arm.get('raw')
            ranked = rank_arms(per_arm, lower)
            winners[(ds, weighting, budget)] = best_arms(per_arm, lower)
            folds_seen = max(n_folds.values())
            print(f'  {weighting}, budget {budget} features, '
                  f'{folds_seen} folds:')
            for arm, value in ranked:
                vals = per_fold[arm]
                lo, hi = min(vals), max(vals)
                against = ('' if today is None or arm == 'raw'
                           else f'   today {today:.4f} -> {value:.4f}')
                mark = '  <- today' if arm == 'raw' else ''
                print(f'    {arm:14s} {metric} {value:.4f}'
                      f'  (folds ran {lo:.4f} to {hi:.4f}, '
                      f'{n_folds[arm]} of them){against}{mark}')
            # A gap between two arms that is smaller than the spread one arm
            # shows across its own folds is not something these five folds can
            # separate. Printed, not tested.
            if today is not None and 'raw' in per_fold:
                raw_spread = max(per_fold['raw']) - min(per_fold['raw'])
                gap = abs(ranked[0][1] - today)
                beat_spread = ranked[0][0] != 'raw' and gap > raw_spread
                spread_counts.append(beat_spread)
                if ranked[0][0] != 'raw':
                    verdict = 'larger than' if beat_spread else 'smaller than'
                    print(f'    first place beats today by {gap:.4f} {metric}, '
                          f'{verdict} the {raw_spread:.4f} today itself moves '
                          f'across its folds')
        print()

    # -- 2. metric or dataset ------------------------------------------------
    by_metric = defaultdict(list)
    for ds in datasets:
        by_metric[meta[ds]['metric']].append(ds)

    print('=== is any arm ahead by more than the folds themselves move? ===')
    n_beat = sum(1 for x in spread_counts if x)
    print(f'  In {n_beat} of {len(spread_counts)} '
          f'dataset-by-weighting-by-budget combinations the first-placed arm '
          f'beats today by more than today moves across its own folds. '
          f'Everywhere else the ordering of the arms sits inside the fold '
          f'spread, and these folds cannot separate them.\n')

    print('=== does the winning arm follow the metric or the dataset? ===')
    for metric, names in sorted(by_metric.items()):
        if len(names) < 2:
            print(f'  {metric}: one dataset ({names[0]}) — this cannot separate '
                  f'the metric from the dataset, and no claim is made either way')
            continue
        conditions = {(w, b) for (d, w, b) in winners if d in names}
        agree, disagree = [], []
        for weighting, budget in sorted(conditions):
            picks = {d: winners.get((d, weighting, budget)) for d in names}
            if None in picks.values():
                continue
            shared = frozenset.intersection(*picks.values())
            if shared:
                agree.append((weighting, budget, shared))
            else:
                disagree.append((weighting, budget, picks))
        total = len(agree) + len(disagree)
        print(f'  {metric} ({len(names)} datasets: {", ".join(names)}): all '
              f'of them share a first-placed arm in {len(agree)} of {total} '
              f'weighting-and-budget combinations')
        for weighting, budget, picks in disagree:
            shown = ', '.join(f'{d} picks {"/".join(sorted(a))}'
                              for d, a in picks.items())
            print(f'      differ at {weighting}, budget {budget}: {shown}')
        tied = [(w, b, s) for w, b, s in agree if len(s) > 1]
        for weighting, budget, shared in tied:
            print(f'      tied at {weighting}, budget {budget}: '
                  f'{"/".join(sorted(shared))} score identically on both')
    print()

    # -- 3. the controls -----------------------------------------------------
    print('=== controls ===')
    # Re-derived from each file's rows rather than read off its stored `checks`,
    # so files written before the strictness condition was corrected on
    # 2026-09-10 are judged by the same rule as the ones written after it.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import f32_budget_split_normalisation as F
    rechecked = {}
    for p in payloads:
        try:
            raw = {k: (v if v is not None else float('-inf'))
                   for k, v in p['raw_scores'].items()}
            rechecked[(p['dataset'], p['fold'])] = F._check_invariants(
                p['dataset'], p['fold'], p['rows'], raw, p['metric'])
        except Exception:
            rechecked[(p['dataset'], p['fold'])] = p.get('checks') or {}

    for name in ('rank_exp_split_unchanged', 'skill_matches_raw'):
        held, failed, skipped, conditional = 0, [], 0, 0
        for p in payloads:
            result = rechecked[(p['dataset'], p['fold'])].get(name)
            if result is None:
                skipped += 1
                continue
            if result.get('holds'):
                held += 1
            else:
                failed.append((p['dataset'], p['fold'], result.get('note', '')))
            if not result.get('strict'):
                conditional += 1
        print(f'  {name}: held on {held} folds, failed on {len(failed)}, '
              f'not applicable on {skipped}, and was conditional rather than '
              f'exact on {conditional}')
        for ds, fold, note in failed:
            print(f'      FAILED {ds} fold {fold}: {note}')

    # -- the AUROC note ------------------------------------------------------
    hard = sorted({p['dataset'] for p in payloads if p.get('auc_on_hard_labels')})
    if hard:
        print(f'\n  On {", ".join(hard)} the budget split is sized by an AUROC '
              f'computed on hard class labels, which is '
              f'(sensitivity + specificity) / 2. Chance is still 0.5, so the '
              f'2*(auc - 0.5) rule is unaffected.')

    suspect = [(p['dataset'], p['fold'], p['raw_scores_outside_metric_range'])
               for p in payloads if p.get('raw_scores_outside_metric_range')]
    if suspect:
        print('\n  Scores outside what the metric can return (the evaluator\'s '
              'own r2 or accuracy showing through):')
        for ds, fold, names in suspect:
            print(f'      {ds} fold {fold}: {names}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
