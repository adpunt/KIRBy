#!/usr/bin/env python3
"""Local smoke driver for federated experiments (test-only, NOT for server).

run_dataset.py cannot pass a local rep list, so on this laptop run_all resolves
DATASET_REPS['herg']['general'] which includes foundation-model reps (molformer,
grover, mhg_gnn, graph_kernel, chemberta...) that cannot be computed locally.

This driver calls federated_helpers.run_all directly with:
  - a small set of locally-computable reps (rep_names override)
  - the smoke config (already shrinks folds/silos/grids under KIRBY_SMOKE_N)
  - FULL_BUDGETS shrunk (precompute_hybrid_searches ignores config['budgets']
    and always reads _shared.FULL_BUDGETS)
  - optional overrides for the searched importance methods, the downstream
    federated model families and the seed list.

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


def _patch_shap_transformers_probe():
    """Local-env only: stop shap.TreeExplainer importing TensorFlow/transformers.

    shap's Explainer constructor calls is_transformers_lm(model), which does a
    getattr on the lazily-loaded `transformers` module. On this laptop
    transformers 4.45.1 sits next to protobuf 6.33.6 and that getattr raises
    `'FieldDescriptor' object has no attribute 'is_repeated'`, so EVERY
    treeshap recipe dies before computing anything. The probe only asks whether
    the model is a language model; a RandomForest never is. Returning False
    skips the import and leaves the SHAP values untouched.

    Test-only and local-only: nothing under src/ is changed, and the server
    environment never runs this file.
    """
    try:
        import shap.explainers._explainer as _e
        _e.is_transformers_lm = lambda model: False
    except Exception as exc:                      # pragma: no cover
        print(f"  [SMOKE] shap probe patch skipped: {exc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--experiments', nargs='+', default=None)
    p.add_argument('--reps', nargs='+', default=['maccs', 'pdv', 'atom_pair',
                                                 'ecfp4', 'avalon'])
    p.add_argument('--budgets', nargs='+', type=int, default=[25],
                   help='FULL_BUDGETS override for the precompute search.')
    p.add_argument('--importance', nargs='+', default=None,
                   help="Searched importance methods, e.g. lightgbm_gain. "
                        "Default: the engine's own three.")
    p.add_argument('--fed-models', nargs='+', default=None,
                   help='Downstream federated families (rf xgb dnn).')
    p.add_argument('--fed-seeds', nargs='+', type=int, default=None)
    p.add_argument('--output', default='tests/results/_smoke_2026_06_05/federated')
    args = p.parse_args()

    if not (os.environ.get('KIRBY_SMOKE_N', '').strip()
            or os.environ.get('KIRBY_SMOKE_GRID', '').strip() not in ('', '0')):
        sys.exit("Refusing to run without KIRBY_SMOKE_N or KIRBY_SMOKE_GRID "
                 "set (test driver only). KIRBY_SMOKE_GRID=1 keeps every row "
                 "and narrows the search; KIRBY_SMOKE_N=<n> also caps rows.")

    _patch_shap_transformers_probe()

    # Shrink the precompute budget grid (precompute ignores config['budgets']).
    import experiments._shared as _shared
    _shared.FULL_BUDGETS = list(args.budgets)

    if args.importance:
        import kirby.find_best_hybrid as _fbh
        _fbh.DEFAULT_IMPORTANCE_METHODS = list(args.importance)
        print(f"  [SMOKE] importance methods -> {args.importance}")

    from experiments.federated import federated_helpers as fh

    config = dict(fh._get_federated_config())
    if args.importance:
        config['importance_methods'] = list(args.importance)
    if args.fed_models:
        config['fed_models'] = list(args.fed_models)
    if args.fed_seeds:
        config['fed_seeds'] = list(args.fed_seeds)

    fh.run_all(
        dataset_name=args.dataset,
        output_dir=args.output,
        experiments=args.experiments,
        config=config,
        rep_names=args.reps,
    )


if __name__ == '__main__':
    main()
