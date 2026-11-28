#!/usr/bin/env python3
"""The best trained arm per target, read off a finished moleculeace_cliff_model run.

WHY THIS EXISTS. Her instruction of 2026-11-28, her words: "I've never cared about
averages. I would only in practice use the BEST one ... This is a TOOL. I am studying
noise mitigation this is a TOOL and i want the best version of that tool."

`moleculeace_cliff_model.py` pools every target and every fold into one list per arm and
reports that list's median. This reads the same output file and reports, per target, the
highest-scoring trained arm and which model and representation it was. Nothing is re-run
and nothing is re-fitted: the per-target, per-fold, per-arm rows are already in the file.

TWO THINGS IT PRINTS, because "the best" has two readings and the choice is hers.

  per target        for each target, the median across that target's folds is taken for
                    each model-and-representation pair, and the pair with the highest
                    median is printed beside its number. This is the tool that picks its
                    own settings per target.
  one setting       the model-and-representation pair whose per-target medians are highest
                    the most often is NOT printed, because a count of targets is not a
                    score. Instead every pair is printed with its lowest and highest
                    per-target median, so one setting can be chosen by the floor it
                    guarantees rather than by a tally.

Matthews correlation is the number throughout: -1 to 1, 0 is chance. Accuracy is not
printed, because the share of flagged molecules runs 8.8 to 54.0 per cent across the 30
targets and accuracy tracks that share.

    python tests/experiments/cliffs/moleculeace_cliff_model_best.py \
        --input /data/stat-ecr/scat9264/cliff_tbd7/moleculeace_cliff_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _per_target_best(results):
    """One record per target: its best trained model-and-representation pair."""
    out = []
    for res in results:
        rows = res.get('rows') or []
        if not rows:
            out.append({'target': res.get('target'),
                        'refused': res.get('refused'), 'best': None})
            continue
        per_pair = {}
        for row in rows:
            if row.get('arm') != 'trained_model':
                continue
            if row.get('matthews_correlation') is None:
                continue
            key = (row['model'], row['representation'])
            per_pair.setdefault(key, []).append(row['matthews_correlation'])
        if not per_pair:
            out.append({'target': res.get('target'),
                        'refused': 'no trained arm scored on this target',
                        'best': None})
            continue
        medians = {k: float(np.median(v)) for k, v in per_pair.items()}
        rules = {}
        for row in rows:
            if row.get('arm', '').startswith('rule:') \
                    and row.get('matthews_correlation') is not None:
                rules.setdefault(row['arm'], []).append(
                    row['matthews_correlation'])
        best_key = max(medians, key=lambda k: medians[k])
        out.append({
            'target': res.get('target'),
            'n_folds_scored': len(per_pair[best_key]),
            'best': {'model': best_key[0], 'representation': best_key[1],
                     'matthews_correlation_median': medians[best_key]},
            'all_pairs': {f'{m}|{r}': v for (m, r), v in medians.items()},
            'rules': {k: float(np.median(v)) for k, v in rules.items()},
        })
    return out


def _pair_spread(per_target):
    """One record per model-and-representation pair: its floor and ceiling.

    The number in each is a per-target median, so the lowest is the worst target that
    pair faced and the highest is the best. No count of targets appears.
    """
    acc = {}
    for rec in per_target:
        for key, val in (rec.get('all_pairs') or {}).items():
            acc.setdefault(key, []).append(val)
    out = []
    for key, vals in acc.items():
        vals = sorted(vals)
        model, rep = key.split('|', 1)
        out.append({'model': model, 'representation': rep,
                    'lowest_target': vals[0], 'highest_target': vals[-1],
                    'median_target': float(np.median(vals))})
    out.sort(key=lambda r: -r['lowest_target'])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True,
                    help='a finished moleculeace_cliff_model.json')
    ap.add_argument('--output', default=None,
                    help='where to write the two tables as JSON; optional')
    a = ap.parse_args(argv)

    payload = json.loads(Path(a.input).read_text())
    results = payload.get('per_target') or payload.get('results') or []
    if not results:
        raise SystemExit(
            f'{a.input} holds no per-target rows: its keys are '
            f'{sorted(payload)}')

    per_target = _per_target_best(results)
    spread = _pair_spread(per_target)

    print('\nBest trained arm per target. The number is the median Matthews '
          'correlation\nacross that target\'s folds. 0 is chance.\n')
    print(f'{"target":24s}{"model":10s}{"representation":18s}'
          f'{"matthews":>10s}{"their rule":>12s}')
    for rec in per_target:
        if rec['best'] is None:
            print(f'{str(rec["target"]):24s}{rec.get("refused", "refused")}')
            continue
        b = rec['best']
        rule = rec['rules'].get('rule:n01')
        print(f'{rec["target"]:24s}{b["model"]:10s}'
              f'{b["representation"]:18s}'
              f'{b["matthews_correlation_median"]:10.4f}'
              f'{(f"{rule:12.4f}" if rule is not None else "           -")}')

    print('\nEvery model-and-representation pair, by the worst target it faced.\n'
          'Each number is a per-target median Matthews correlation.\n')
    print(f'{"model":10s}{"representation":18s}{"lowest":>10s}'
          f'{"median":>10s}{"highest":>10s}')
    for r in spread:
        print(f'{r["model"]:10s}{r["representation"]:18s}'
              f'{r["lowest_target"]:10.4f}{r["median_target"]:10.4f}'
              f'{r["highest_target"]:10.4f}')
    print()

    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {'what_it_is': 'the best trained arm per target, and every '
                           'model-and-representation pair with the lowest and '
                           'highest per-target median it reached',
             'metric': 'Matthews correlation, -1 to 1, 0 is chance',
             'read_from': str(a.input),
             'per_target': per_target, 'pairs': spread}, indent=2))
        print(f'written to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
