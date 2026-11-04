#!/usr/bin/env python3
"""Smoke: AC1's parts. The dataset list, both binary hybrids, and the row the
census writes, with the gap in label units on it.

It builds the hybrid through the debug path (one named budget), not through the
search, so it runs in a minute. Which features a search would pick does not
change whether a row comes out or what is on it.

    python3 experiments/cliffs/_smoke_ac1.py --dataset herg_ki
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments.cliffs import ac1_definition_counts as ac1
from experiments.cliffs.binary_hybrid import EXPECTED_KIND
from experiments.noise import cliff_rules as cr

FAILS = []

#: The columns every row must carry, and what one of them counts.
REQUIRED = (
    'similarity', 'gap', 'feature_space', 'n_molecules', 'n_similar_pairs',
    'n_cliff_pairs', 'n_cliff_molecules', 'pct_cliff_molecules',
    'gap_cutoff_label_units', 'gap_over_kramer_floor',
    'gap_over_own_repeat_spread', 'own_repeat_spread_label_units',
    'similarity_cutoff', 'similarity_used_labels',
)


def check(name, condition, detail=''):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}"
          f"{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def main(argv=None):
    import argparse

    from experiments._shared import load_data
    from experiments.noise.cliff_census import own_repeat_spread

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--budget', type=int, default=250)
    ap.add_argument('--reps', default='ecfp4,avalon,atom_pair,graph_kernel',
                    help='kept small so the smoke does not load ten '
                         'representations to discard six')
    args = ap.parse_args(argv)

    print('\nthe dataset list')
    ds = ac1.datasets()
    check('every dataset in it is molecular and in the registry',
          len(ds) == 14, f'{len(ds)} datasets: {ds}')
    check('the three QM9 targets, both hERG sets and Dablander\'s three are '
          'in it',
          all(d in ds for d in ('qm9', 'qm9_mu', 'qm9_alpha', 'herg_ki',
                                'herg_ic50', 'chembl_qsar_d2',
                                'chembl_qsar_fxa', 'chembl_qsar_mpro')))
    check('no antibody dataset is in it',
          not any(d.startswith(('flab_', 'abagym_', 'ginkgo_', 'tz_',
                                'cdr_')) for d in ds))

    data = load_data(args.dataset,
                     rep_names=[r.strip() for r in args.reps.split(',')])
    labels = np.asarray(data['labels'], dtype=float).ravel()
    print(f"\n{args.dataset}: {len(labels)} molecules, "
          f"{len(data['reps'])} representations")

    spread, spread_info = own_repeat_spread(args.dataset, labels, data=data)
    check('the repeat spread is either a number in label units or a stated '
          'reason for having none',
          (spread is not None) or bool(spread_info.get('reason')),
          (f'{spread:.4f} label units' if spread is not None
           else str(spread_info.get('reason'))[:90]))

    print('\nboth binary hybrids')
    hybrids, refused = ac1.binary_hybrids(
        data, args.dataset, source=f'budget:{args.budget}')
    for k, why in refused.items():
        print(f'    no hybrid for {k}: {why}')
    check('both restrictions produced a hybrid', len(hybrids) == 2,
          f'{sorted(hybrids)}')
    for label, (rep_name, X, prov) in hybrids.items():
        expected = EXPECTED_KIND[prov['restriction']]
        check(f'{label}: the kind read from the assembled matrix is '
              f'{expected}',
              prov['kind_read_from_the_assembled_matrix'] == expected,
              f"read {prov['kind_read_from_the_assembled_matrix']}, "
              f"{prov['n_features_assembled']} features from "
              f"{prov['representations']}")
        check(f'{label}: it is loaded under the name the criteria look for',
              rep_name in ('hybrid_bits', 'hybrid_counts'), rep_name)
        check(f'{label}: the coefficient list came from the kind',
              ('tversky' in prov['coefficients_defined_on_it'])
              == (expected == 'binary'),
              f"{prov['coefficients_defined_on_it']}")

    ctx = cr.CliffContext(
        labels, smiles=data.get('smiles'),
        reps={k: np.asarray(v, dtype=np.float64)
              for k, v in data['reps'].items()},
        is_clf=(data.get('task_type') == 'classification'),
        dataset_name=args.dataset)

    print('\nthe rows, on three criteria and two gaps')
    keep_sims = ('ecfp4_tanimoto_055', 'all_features_top_1pct',
                 'hybrid_bits_tanimoto_090', 'hybrid_bits_tversky_090',
                 'hybrid_counts_soergel_090')
    keep_gaps = ('mad_2x', 'absolute_1')
    rows, masks, errors, excluded = ac1.census(
        ctx, hybrids, spread, verbose=True, sims=keep_sims, gaps=keep_gaps)
    check('no row failed to compute', not errors,
          '; '.join(f'{k}: {v}' for k, v in list(errors.items())[:3]))
    check('rows came out', len(rows) > 0, f'{len(rows)} rows')
    missing = {c for r in rows for c in REQUIRED if c not in r}
    check('every row carries every column', not missing, f'missing {missing}')
    check("every row has its gap in the label's own units",
          all(r['gap_cutoff_label_units'] is not None for r in rows),
          f"{sum(r['gap_cutoff_label_units'] is None for r in rows)} rows "
          f"without one")
    check('every row has a mask of the molecules it flagged',
          len(masks) == len(rows), f'{len(masks)} masks, {len(rows)} rows')
    check('a molecule count never exceeds the dataset',
          all(r['n_cliff_molecules'] <= len(labels) for r in rows))
    check('cliff pairs never exceed similar pairs',
          all(r['n_cliff_pairs'] <= r['n_similar_pairs'] for r in rows))

    hyb = [r for r in rows if r['feature_space'] != 'the dataset itself']
    plain = [r for r in rows if r['feature_space'] == 'the dataset itself']
    check('the hybrid rows say their similarity side used the labels',
          bool(hyb) and all(r['similarity_used_labels'] for r in hyb),
          f'{len(hyb)} hybrid rows')
    check('the rows that did not use the labels say so',
          bool(plain) and not any(r['similarity_used_labels'] for r in plain),
          f'{len(plain)} rows on the dataset itself')
    check('every hybrid row carries the recipe that built its hybrid',
          all(r.get('hybrid', {}).get('how') for r in hyb))
    check('tversky ran on the 0/1 hybrid and soergel on the count one',
          {r['similarity'] for r in hyb} >= {'hybrid_bits_tversky_090',
                                             'hybrid_counts_soergel_090'},
          f"{sorted({r['similarity'] for r in hyb})}")

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main())
