#!/usr/bin/env python3
"""Reads the n02 weight-shuffle result and prints the two arms side by side.

WHAT THE RUN MEASURED. On the same 3 outer folds and the same three models,
three arms: no weighting at all, the weight vector n02 ships at its 4-to-1 row
weight ratio between the most and the least suspect molecule, and that SAME
vector handed to the training rows in a permuted order under each of five
named seeds. Every arm holds the identical multiset of weights, so what the
shuffled arms break is only the pairing between a row and its weight.

WHAT THIS PRINTS. Per model, in the dataset's own metric: the unweighted
score, the scored arm's distance from it, and the five shuffled arms' distance
from it as a SPAN -- lowest seed to highest -- because at five seeds the
reading is the span and not any one seed's mean. A positive number always
means the weights helped, on both metrics.

The header also reprints the top-to-bottom row weight ratio of the scored arm
and of each shuffled arm. Those must be equal to four decimal places: if they
are not, the shuffled arm is not holding the weights the scored arm holds and
the comparison is not the one this run intended.
"""
import argparse
import json
import os

import numpy as np

DEFAULT_DIR = '/Volumes/seagate/local_noise_run/loop_n02'


def read_one(path):
    with open(path) as fh:
        d = json.load(fh)
    metric = d['metric']
    lower = d['lower_is_better']
    seeds = d.get('shuffle_seeds') or []
    print(f"\n{'=' * 72}")
    print(f"{d['dataset']}: {d['n_molecules']} molecules, metric {metric} "
          f"({'lower' if lower else 'higher'} wins), representation "
          f"{d['control_representation']}, space {d['space']}, "
          f"{d['folds_scored']} of {d['folds_requested']} outer folds scored")
    seed_text = (', '.join(str(s) for s in seeds) if seeds
                 else 'NONE -- this file has no shuffled arm')
    print(f"  seeds the weights were reordered under: {seed_text}")
    print(f"  how the reorder was drawn: {d.get('shuffle_draw')}")

    ratios = d.get('weight_ratio_top_to_bottom', {})
    scored_arm = None
    for arm in ratios:
        if '|shuffled_seed' not in arm and arm != 'unweighted':
            scored_arm = arm
            break
    if scored_arm is None:
        print('  no scored arm in this file')
        return
    r_scored = ratios[scored_arm]
    print(f"  heaviest row weight divided by the lightest, scored arm "
          f"{scored_arm}: {r_scored:.4f}")
    bad = []
    for s in seeds:
        r = ratios.get(f'{scored_arm}|shuffled_seed{s}')
        if r is None or abs(r - r_scored) > 1e-4:
            bad.append((s, r))
    if bad:
        print(f"  ⛔ {len(bad)} of the {len(seeds)} shuffled arms do NOT hold "
              f"the scored arm's weights: {bad}")
    else:
        print(f"  all {len(seeds)} shuffled arms hold the same heaviest-to-"
              f"lightest ratio as the scored arm")

    if d.get('notes'):
        print(f"  notes: {d['notes']}")

    print(f"\n  one row per model. 'distance from no weighting' is in "
          f"{metric}, positive means the weights helped.")
    print(f"  {'model':<10} {'no weighting':>13} {'scored':>9} "
          f"{'shuffled span over the 5 seeds':>34}")
    for mname, rows in d['per_model'].items():
        uw = rows['unweighted']['mean']
        g_scored = rows[scored_arm]['gain']
        gs = [rows[f'{scored_arm}|shuffled_seed{s}']['gain']
              for s in seeds if f'{scored_arm}|shuffled_seed{s}' in rows]
        if not gs:
            print(f"  {mname:<10} {uw:>13.4f} {g_scored:>+9.4f} "
                  f"{'no shuffled arm':>34}")
            continue
        span = f"{min(gs):+.4f} to {max(gs):+.4f}  (median {np.median(gs):+.4f})"
        print(f"  {mname:<10} {uw:>13.4f} {g_scored:>+9.4f} {span:>34}")
        print(f"  {'':<10} {'':>13} {'':>9} "
              f"seed by seed: "
              f"{', '.join(f'{s}: {g:+.4f}' for s, g in zip(seeds, gs))}")

    # The reading the loop was opened for, stated for the whole dataset.
    g_scored_all, g_shuf_all = [], []
    for mname, rows in d['per_model'].items():
        g_scored_all.append(rows[scored_arm]['gain'])
        g_shuf_all += [rows[f'{scored_arm}|shuffled_seed{s}']['gain']
                       for s in seeds if f'{scored_arm}|shuffled_seed{s}' in rows]
    if g_shuf_all:
        print(f"\n  averaged over the {len(g_scored_all)} models: the scored "
              f"pairing is {np.mean(g_scored_all):+.4f} {metric} from no "
              f"weighting.")
        print(f"  the {len(g_shuf_all)} shuffled readings "
              f"({len(seeds)} seeds x {len(g_scored_all)} models) run "
              f"{min(g_shuf_all):+.4f} to {max(g_shuf_all):+.4f} {metric}, "
              f"median {np.median(g_shuf_all):+.4f}.")
        n_worse = sum(1 for g in g_shuf_all if g < np.mean(g_scored_all))
        print(f"  {n_worse} of the {len(g_shuf_all)} shuffled readings sit "
              f"below the scored arm's average.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default=DEFAULT_DIR)
    p.add_argument('--datasets', default='flab_thermo_tresanco,herg_ki')
    p.add_argument('--tag', default='shuffle')
    p.add_argument('--space', default='base')
    a = p.parse_args()
    found = 0
    for ds in [x.strip() for x in a.datasets.split(',') if x.strip()]:
        path = os.path.join(a.dir,
                            f'score_sweep_{ds}_{a.space}_{a.tag}.json')
        if not os.path.exists(path):
            print(f'not written yet: {path}')
            continue
        found += 1
        read_one(path)
    if not found:
        print('\nnothing to read. The run is registered with the supervisor as '
              'n02_weight_shuffle_control.')


if __name__ == '__main__':
    main()
