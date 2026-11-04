#!/usr/bin/env python3
"""Smoke: a cliff definition computed on one fold's training rows uses nothing
outside them.

The thing being checked. A definition reads the labels twice: once for the
threshold (the median absolute deviation of the label column) and once for the
pairing (which molecules a molecule is paired with). Computed over the whole
dataset, both reads reach the test fold. subset_context restricts both.

The checks that matter are the last three. Slicing the parent's cached
similarity matrix must equal building it from the subset's molecules alone,
or the reuse is wrong. The glued features must NOT be the parent's sliced,
because they are z-scored and the parent's means are over every molecule.

    python3 experiments/noise/_smoke_subset_context.py --dataset herg_ki
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments.noise import cliff_rules as cr

FAILS = []


def check(name, condition, detail=''):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}"
          f"{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def main(argv=None):
    import argparse

    from experiments._shared import load_data

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--reps', default='ecfp4,avalon,atom_pair,graph_kernel')
    ap.add_argument('--gap', default='mad_2x')
    ap.add_argument('--similarity', default='ecfp4_tanimoto_055')
    args = ap.parse_args(argv)

    data = load_data(args.dataset,
                     rep_names=[r.strip() for r in args.reps.split(',')])
    labels = np.asarray(data['labels'], dtype=float).ravel()
    reps = {k: np.asarray(v, dtype=np.float64) for k, v in data['reps'].items()}

    ctx = cr.CliffContext(labels, smiles=data.get('smiles'), reps=reps,
                          is_clf=(data.get('task_type') == 'classification'),
                          dataset_name=args.dataset)
    n = ctx.n
    print(f"\n{args.dataset}: {n} molecules, {len(reps)} representations")

    # One fold boundary: the last fifth is held out.
    rng = np.random.RandomState(0)
    order = rng.permutation(n)
    train, test = np.sort(order[:int(0.8 * n)]), np.sort(order[int(0.8 * n):])
    print(f"one fold: {len(train)} training rows, {len(test)} held out")

    whole = cr.cliff_sets(ctx, args.similarity, args.gap)
    sub = cr.subset_context(ctx, train)
    part = cr.cliff_sets(sub, args.similarity, args.gap)

    print("\nwhat each context computed")
    print(f"{'':>26} {'molecules':>10} {'threshold':>12} "
          f"{'similar pairs':>14} {'cliff molecules':>16}")
    for tag, r in (('the whole dataset', whole), ("one fold's training", part)):
        print(f"{tag:>26} {r['n_molecules']:>10,} "
              f"{r['gap_info']['cutoff_label_units']:>12.4f} "
              f"{r['n_similar_pairs']:>14,} {r['n_cliff_molecules']:>16,}")

    print("\nthe restriction")
    check('the subset context holds only the training rows',
          sub.n == len(train), f'{sub.n} of {n}')
    check('its labels are the training rows\' labels',
          bool(np.array_equal(sub.labels, labels[train])))
    check('it records what it is a subset of',
          (sub.subset_of or {}).get('n_molecules_in_parent') == n
          and part['subset_of'] is not None,
          str(sub.subset_of))
    check('the whole-dataset row says it is not a subset',
          whole['subset_of'] is None)

    print("\nthe threshold was recomputed, not inherited")
    mad_all = float(np.median(np.abs(labels - np.median(labels))))
    mad_tr = float(np.median(np.abs(labels[train] - np.median(labels[train]))))
    check('the threshold follows the training rows\' own spread',
          abs(part['gap_info']['cutoff_label_units'] - 2.0 * mad_tr) < 1e-9,
          f"{part['gap_info']['cutoff_label_units']:.4f} label units against "
          f"{2.0 * mad_tr:.4f} from the training rows and "
          f"{2.0 * mad_all:.4f} from all {n}")

    print("\nno pair crosses the fold boundary")
    pm = part['pair_mask']
    check('the pair matrix is the size of the training set',
          pm.shape == (len(train), len(train)), str(pm.shape))
    check('every cliff molecule it flags is a training molecule',
          len(part['molecule_mask']) == len(train))

    print("\nthe structural reuse is correct")
    rebuilt = cr.CliffContext(labels[train],
                              smiles=[data['smiles'][i] for i in train],
                              reps={k: X[train] for k, X in reps.items()},
                              dataset_name=args.dataset)
    fresh = cr.cliff_sets(rebuilt, args.similarity, args.gap)
    check('slicing the parent\'s similarity matrix equals building it from '
          'the training molecules alone',
          fresh['n_similar_pairs'] == part['n_similar_pairs']
          and bool(np.array_equal(fresh['pair_mask'], pm)),
          f"{part['n_similar_pairs']:,} similar pairs sliced against "
          f"{fresh['n_similar_pairs']:,} rebuilt")

    print("\nthe two things that must not be inherited")
    _ = ctx.features
    check('the glued features are rebuilt, so the z-scoring does not use '
          'held-out molecules',
          not np.allclose(cr.subset_context(ctx, train).features,
                          ctx.features[train]),
          f"largest difference "
          f"{float(np.abs(cr.subset_context(ctx, train).features - ctx.features[train]).max()):.4g}")
    check('the glued features match a rebuild from the training rows alone',
          bool(np.allclose(cr.subset_context(ctx, train).features,
                           rebuilt.features)))
    check('the pairwise activity gap is rebuilt from the subset\'s labels',
          bool(np.allclose(cr.subset_context(ctx, train).gap_matrix,
                           rebuilt.gap_matrix)))

    print("\nlabels as the method sees them, not the clean ones")
    noisy = labels.copy()
    noisy[train[:50]] += 5.0
    sub_noisy = cr.subset_context(ctx, train, labels=noisy)
    check('passing a corrupted label column changes the threshold',
          abs(sub_noisy.labels[0] - sub.labels[0]) >= 0
          and not np.array_equal(sub_noisy.labels, sub.labels))
    r_noisy = cr.cliff_sets(sub_noisy, args.similarity, args.gap)
    check('the threshold came from the corrupted labels',
          r_noisy['gap_info']['cutoff_label_units']
          != part['gap_info']['cutoff_label_units'],
          f"{r_noisy['gap_info']['cutoff_label_units']:.4f} against "
          f"{part['gap_info']['cutoff_label_units']:.4f} label units")
    check('a label column one per subset row is accepted too',
          cr.subset_context(ctx, train, labels=noisy[train]).n == len(train))

    print("\nevery similarity criterion that runs on the dataset still runs "
          "on a fold")
    ok_whole, _ = cr.available_similarity_criteria(ctx)
    ok_sub, _ = cr.available_similarity_criteria(sub)
    lost = [c for c in ok_whole if c not in ok_sub]
    check('none is lost by restricting the context', not lost, f'lost {lost}')
    failed = []
    for name in ok_sub:
        try:
            cr.cliff_sets(sub, name, args.gap)
        except Exception as e:
            failed.append(f'{name}: {type(e).__name__}')
    check('each one computes on the fold', not failed,
          f"{len(ok_sub)} criteria; failures {failed[:4]}")

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main())
