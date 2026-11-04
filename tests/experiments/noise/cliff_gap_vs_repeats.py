#!/usr/bin/env python3
"""Is a rule's label gap bigger than the same compound's own repeats disagree by?

Paper 3, activity cliffs. Fits no model, uses no split, trains no cleanup.

THE QUESTION. A cliff rule calls a pair of molecules a cliff when their labels
sit more than some gap apart. Where one compound was measured more than once,
the spread of its own repeats is how far two measurements of one thing sit
apart in that assay. A gap smaller than that cannot be told apart from the
assay disagreeing with itself. So for each rule: how does its gap compare with
what this dataset's own repeats disagree by, in the label's own units?

This is the first of the nine cliff-versus-noise checks whose unit is label
units rather than a count or a flag. The others compare one set of molecules
with another.

WHY THIS DATASET AND NO OTHER, TODAY. The three datasets the cliff census has
run on -- hERG Ki, Caco-2 and MDR1-MDCK -- expose no repeat structure to the
loader, so each of them falls back to Kramer's 0.54 pKi, which is a figure for
public Ki data and the wrong assay for a log permeability or a log efflux
ratio. The per-record hERG IC50 pull is different: it keeps one row per
measurement, so a compound measured four times has four rows and its own
spread can be computed. 1,218 of its 9,802 molecules were measured more than
once.

WHAT ONE MOLECULE'S LABEL IS, and it matters here more than anywhere. A
molecule's label is the MEAN of its measurements in pIC50, which is the same
choice `cliff_curator_flag.py` makes and n29 calls its `mean_label` arm, so
the two measurements are comparable. The repeat spread is then the spread of
the measurements that mean was taken over.

TWO SPREADS, AND THEY ARE NOT THE SAME NUMBER.

  * A molecule measured more than once (2 or more records).
  * A molecule whose repeats DISAGREE (spread above 0). A compound measured
    three times at the same value has repeats and no disagreement, and it says
    nothing about how far the assay wanders. `compute_replicate_stds` draws
    exactly this distinction and both counts are reported.

The spread is the population standard deviation, ddof=0, which is
`_ground_truth_providers.GROUP_SPREAD_DDOF`. Not a choice made here.

THREE COMPARISONS PER RULE, all in pIC50.

  1. The rule's gap against the median repeat spread of this dataset, as a
     multiple. A gap of 1.0 against a median spread of 0.2 is 5 times what the
     assay repeats to. **The reference is the median over the molecules whose
     repeats disagree, not over every repeated molecule.** Measured
     2026-11-04: 566 of hERG IC50's 1,218 repeated molecules have repeats
     that agree to the digit, which drags the median over all of them to
     0.0014 pIC50 and makes a multiple of it meaningless -- the first run of
     this file printed 718x for a gap of 0.98. Repeats that agree exactly are
     as likely to be one measurement entered twice as an assay that repeats
     perfectly, and either way they carry nothing about how far the assay
     wanders. Both medians are in the output file.
  2. The share of repeated molecules whose OWN repeats disagree by more than
     the rule's gap. These are compounds where one measurement and another of
     the same compound would clear the rule's gap on their own.
  3. Of the rule's cliff pairs where both molecules were measured more than
     once, the share whose label gap is below the larger of the two molecules'
     own spreads, and the share below the sum of the two. These are the cliff
     pairs that the two compounds' own measurement spread could account for.

NONE OF THE THREE IS A TEST. No p-value, no null, no resampling. They are the
raw spreads and the raw gaps, compared.

WHAT A RESULT DOES NOT SAY. A rule whose gap sits below the repeat spread is
not thereby wrong; it is a rule whose cliff set contains pairs that two
measurements of one compound could have produced. Whether that matters is the
rule choice, which is Adelaide's.

RUN
    python tests/experiments/noise/cliff_gap_vs_repeats.py
    python tests/experiments/noise/cliff_gap_vs_repeats.py --rules n01,moleculeace

The per-record pull has to exist. If it does not, fetch it with
    python tests/experiments/noise/_probe_chembl_herg.py --standard-type IC50
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (os.path.join(_ROOT, 'src'), os.path.join(_ROOT, 'tests'),
           os.path.join(_ROOT, 'tests', 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.noise import cliff_rules as cr  # noqa: E402
from experiments.noise._ground_truth_providers import (  # noqa: E402
    GROUP_SPREAD_DDOF, MIN_REPLICATES)

DATASET = 'herg_ic50'
DEFAULT_OUTPUT = '/Volumes/seagate/cliff_gap_vs_repeats'

#: Kramer et al.'s spread for public Ki data, already in the repo at
#: `cliff_rules.KRAMER_NOISE_FLOOR_PKI`. Printed beside this dataset's own
#: number so the two can be read against each other, not as a substitute.
KRAMER = cr.KRAMER_NOISE_FLOOR_PKI


def load_molecules():
    """One row per molecule: mean pIC50, and the spread of its own repeats."""
    from kirby.datasets.herg import load_herg_ic50_measurements

    d = load_herg_ic50_measurements()
    smiles = np.asarray(d['smiles'], dtype=object)
    record_labels = np.asarray(d['labels'], dtype=float)
    groups = np.asarray(d['replicate_groups'])

    order = np.unique(groups)
    mol_smiles, mol_labels, mol_spread, mol_n = [], [], [], []
    for gid in order:
        rows = np.flatnonzero(groups == gid)
        vals = record_labels[rows]
        mol_smiles.append(str(smiles[rows[0]]))
        mol_labels.append(float(vals.mean()))
        mol_n.append(int(len(rows)))
        if len(rows) >= MIN_REPLICATES:
            mol_spread.append(float(np.std(vals, ddof=GROUP_SPREAD_DDOF)))
        else:
            mol_spread.append(np.nan)

    return {
        'smiles': mol_smiles,
        'labels': np.asarray(mol_labels, dtype=float),
        'spread': np.asarray(mol_spread, dtype=float),
        'n_records': np.asarray(mol_n, dtype=int),
        'n_records_total': int(len(record_labels)),
    }


def spread_summary(spread, n_records):
    """What this dataset's repeats disagree by, in the label's own units."""
    repeated = n_records >= MIN_REPLICATES
    s = spread[repeated]
    disagree = s > 0
    q = ([float(np.percentile(s, p)) for p in (25, 50, 75, 90, 95, 99)]
         if len(s) else [None] * 6)
    qd = ([float(np.percentile(s[disagree], p))
           for p in (25, 50, 75, 90, 95, 99)]
          if int(disagree.sum()) else [None] * 6)
    return {
        'n_molecules': int(len(spread)),
        'n_molecules_measured_more_than_once': int(repeated.sum()),
        'n_molecules_whose_repeats_disagree': int(disagree.sum()),
        'n_molecules_whose_repeats_agree_exactly':
            int((repeated.sum() - disagree.sum())),
        'max_records_on_one_molecule': int(n_records.max()),
        'spread_label_units_over_repeated_molecules': {
            'p25': q[0], 'median': q[1], 'p75': q[2],
            'p90': q[3], 'p95': q[4], 'p99': q[5],
            'mean': float(s.mean()) if len(s) else None,
            'max': float(s.max()) if len(s) else None,
        },
        'spread_label_units_over_disagreeing_molecules_only': {
            'p25': qd[0], 'median': qd[1], 'p75': qd[2],
            'p90': qd[3], 'p95': qd[4], 'p99': qd[5],
        },
        'kramer_noise_floor_pki': KRAMER,
        'ddof': GROUP_SPREAD_DDOF,
        'min_replicates': MIN_REPLICATES,
    }


def compare(sets, ctx, spread, repeated, gap_units, median_spread,
            median_all):
    """The three comparisons, for one rule. Every number names its unit."""
    out = {
        'gap_label_units': gap_units,
        'median_repeat_spread_of_disagreeing_molecules_label_units':
            median_spread,
        'median_repeat_spread_of_all_repeated_molecules_label_units':
            median_all,
    }

    # 1. the gap as a multiple of what this dataset's repeats disagree by
    if gap_units is not None and median_spread and median_spread > 0:
        out['gap_as_multiple_of_median_repeat_spread'] = float(
            gap_units / median_spread)
    else:
        out['gap_as_multiple_of_median_repeat_spread'] = None
    out['gap_as_multiple_of_kramer_floor'] = (
        float(gap_units / KRAMER) if gap_units is not None else None)

    # 2. repeated molecules whose own repeats disagree by more than the gap
    s = spread[repeated]
    if gap_units is not None and len(s):
        over = int((s > gap_units).sum())
        out['n_repeated_molecules_whose_own_spread_exceeds_the_gap'] = over
        out['n_repeated_molecules'] = int(len(s))
        out['pct_repeated_molecules_whose_own_spread_exceeds_the_gap'] = (
            100.0 * over / len(s))
    else:
        out['n_repeated_molecules_whose_own_spread_exceeds_the_gap'] = None
        out['n_repeated_molecules'] = int(len(s))
        out['pct_repeated_molecules_whose_own_spread_exceeds_the_gap'] = None

    # 3. cliff pairs whose gap the two molecules' own spreads could account for
    pair_mask = np.asarray(sets['pair_mask'], dtype=bool)
    i, j = np.nonzero(np.triu(pair_mask, 1))
    out['n_cliff_pairs'] = int(len(i))
    both = repeated[i] & repeated[j]
    out['n_cliff_pairs_both_molecules_repeated'] = int(both.sum())
    if int(both.sum()):
        ii, jj = i[both], j[both]
        pair_gap = np.abs(ctx.labels[ii] - ctx.labels[jj])
        s_i, s_j = spread[ii], spread[jj]
        larger = np.maximum(s_i, s_j)
        total = s_i + s_j
        n_b = int(both.sum())
        out['pct_of_those_pairs_gap_below_larger_molecule_spread'] = (
            100.0 * float((pair_gap < larger).sum()) / n_b)
        out['pct_of_those_pairs_gap_below_sum_of_the_two_spreads'] = (
            100.0 * float((pair_gap < total).sum()) / n_b)
        out['median_gap_of_those_pairs_label_units'] = float(
            np.median(pair_gap))
        out['median_larger_spread_of_those_pairs_label_units'] = float(
            np.median(larger))
    else:
        out['pct_of_those_pairs_gap_below_larger_molecule_spread'] = None
        out['pct_of_those_pairs_gap_below_sum_of_the_two_spreads'] = None
        out['median_gap_of_those_pairs_label_units'] = None
        out['median_larger_spread_of_those_pairs_label_units'] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rules', default='',
                    help='comma-separated names from cliff_rules.RULES; '
                         'default every rule')
    ap.add_argument('--output', default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    t0 = time.time()
    mol = load_molecules()
    labels = mol['labels']
    spread = mol['spread']
    n_records = mol['n_records']
    repeated = n_records >= MIN_REPLICATES
    n = len(labels)

    summ = spread_summary(spread, n_records)
    med_all = summ['spread_label_units_over_repeated_molecules']['median']
    med = summ[
        'spread_label_units_over_disagreeing_molecules_only']['median']

    print(f"{DATASET}: {n} molecules, from {mol['n_records_total']} ChEMBL "
          f"records, label = mean pIC50 of a molecule's own measurements")
    print(f"  {summ['n_molecules_measured_more_than_once']} molecules were "
          f"measured more than once, of which "
          f"{summ['n_molecules_whose_repeats_disagree']} have repeats that "
          f"disagree at all")
    print(f"  the spread of one molecule's own repeats, in pIC50, over those "
          f"{summ['n_molecules_measured_more_than_once']} molecules:")
    d = summ['spread_label_units_over_repeated_molecules']
    print(f"    median {d['median']:.4f}, quartiles {d['p25']:.4f} and "
          f"{d['p75']:.4f}, 90th {d['p90']:.4f}, 95th {d['p95']:.4f}, "
          f"99th {d['p99']:.4f}, largest {d['max']:.4f}")
    print(f"  {summ['n_molecules_whose_repeats_agree_exactly']} of those "
          f"{summ['n_molecules_measured_more_than_once']} molecules have "
          f"repeats that agree to the digit, which is what drags that median "
          f"to {med_all:.4f}. Every comparison below uses the median over the "
          f"{summ['n_molecules_whose_repeats_disagree']} molecules whose "
          f"repeats disagree instead:")
    e = summ['spread_label_units_over_disagreeing_molecules_only']
    print(f"    median {e['median']:.4f}, quartiles {e['p25']:.4f} and "
          f"{e['p75']:.4f}, 90th {e['p90']:.4f}, 95th {e['p95']:.4f}, "
          f"99th {e['p99']:.4f}")
    print(f"  against Kramer's {KRAMER} pKi, the figure the census falls back "
          f"to where a dataset has no repeats: this dataset's median "
          f"disagreement is {med / KRAMER:.3f} of it")
    print()

    ctx = cr.CliffContext(labels, smiles=mol['smiles'], is_clf=False,
                          dataset_name=DATASET)

    names = ([s.strip() for s in args.rules.split(',') if s.strip()]
             or list(cr.RULES))
    rows, skipped = {}, {}
    for name in names:
        similarity, gap = cr.RULES[name]
        t1 = time.time()
        try:
            sets = cr.cliff_sets(ctx, similarity, gap)
        except Exception as exc:
            skipped[name] = f'{type(exc).__name__}: {exc}'
            print(f"  {name:24s} NOT RUN  {type(exc).__name__}: "
                  f"{str(exc)[:90]}")
            continue
        gap_units = sets['gap_info'].get('cutoff_label_units')
        row = compare(sets, ctx, spread, repeated, gap_units, med, med_all)
        row.update({'similarity': similarity, 'gap': gap,
                    'n_cliff_molecules': sets['n_cliff_molecules'],
                    'gap_info': sets['gap_info'],
                    'seconds': round(time.time() - t1, 1)})
        rows[name] = row
        g = ('n/a' if gap_units is None else f'{gap_units:.4f}')
        mult = row['gap_as_multiple_of_median_repeat_spread']
        mult_s = 'n/a' if mult is None else f'{mult:5.2f}x'
        pct = row['pct_of_those_pairs_gap_below_sum_of_the_two_spreads']
        pct_s = 'n/a' if pct is None else f'{pct:5.2f}%'
        print(f"  {name:24s} gap {g:>7s} pIC50 = {mult_s} the median repeat "
              f"spread; of its {row['n_cliff_pairs_both_molecules_repeated']} "
              f"cliff pairs with both molecules repeated, {pct_s} have a gap "
              f"below the two molecules' spreads added together "
              f"({row['seconds']}s)")

    print()
    print("One line above is one rule. The gap is in pIC50. The multiple is "
          "that gap divided by the median spread of one molecule's own "
          "repeats on this dataset, over the molecules whose repeats "
          "disagree. The last share counts cliff pairs where "
          "both molecules were measured more than once, and asks how many of "
          "them have a label difference smaller than the two molecules' own "
          "measurement spreads added together.")
    print("A rule is not wrong for having such pairs. The number says how "
          "much of its cliff set two measurements of the same compounds could "
          "have produced.")

    out = {
        'dataset': DATASET,
        'unit': 'one molecule',
        'label_per_molecule': 'mean of its measurements, pIC50',
        'n_records_total': mol['n_records_total'],
        'repeat_spread': summ,
        'rules': rows,
        'rules_not_run': skipped,
        'what_this_is_not': (
            'no test, no p-value, no null and no resampling. The raw repeat '
            'spreads and the raw label gaps, compared in pIC50.'),
        'seconds': round(time.time() - t0, 1),
    }
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f'{DATASET}.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nWritten to {path}  ({out['seconds']}s)")


if __name__ == '__main__':
    main()
