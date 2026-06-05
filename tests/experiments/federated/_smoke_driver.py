#!/usr/bin/env python3
"""Local smoke driver for federated experiments (test-only, NOT for server).

run_dataset.py cannot pass a local rep list, so on this laptop run_all resolves
DATASET_REPS['herg']['general'] which includes foundation-model reps (molformer,
grover, mhg_gnn, graph_kernel, chemberta...) that cannot be computed locally.

This driver calls federated_helpers.run_all directly with:
  - a small set of locally-computable reps (rep_names override)
  - the smoke config (already shrinks folds/silos/grids under KIRBY_SMOKE_N)
  - FULL_BUDGETS shrunk to [25] (precompute_hybrid_searches ignores
    config['budgets'] and always reads _shared.FULL_BUDGETS).

Usage (always set KIRBY_SMOKE_N + single-thread env):
  KIRBY_SMOKE_N=200 python tests/experiments/federated/_smoke_driver.py \
      herg --experiments fed01 fed02 fed04 --reps maccs pdv atom_pair ecfp4 avalon
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--experiments', nargs='+', default=None)
    p.add_argument('--reps', nargs='+', default=['maccs', 'pdv', 'atom_pair',
                                                 'ecfp4', 'avalon'])
    p.add_argument('--output', default='tests/results/_smoke_2026_06_05/federated')
    args = p.parse_args()

    if not os.environ.get('KIRBY_SMOKE_N', '').strip():
        sys.exit("Refusing to run without KIRBY_SMOKE_N set (smoke-only).")

    # Shrink the precompute budget grid (precompute ignores config['budgets']).
    import experiments._shared as _shared
    _shared.FULL_BUDGETS = [25]

    from experiments.federated import federated_helpers as fh
    fh.run_all(
        dataset_name=args.dataset,
        output_dir=args.output,
        experiments=args.experiments,
        rep_names=args.reps,
    )


if __name__ == '__main__':
    main()
