"""Local smoke runner for similarity experiments (small real dataset).

Injects a tiny fast config (shrinks FULL_BUDGETS so precompute is tractable)
and calls similarity_helpers.run_all on a small dataset with locally-loadable
reps (ecfp4/maccs — no foundation models, no server). Smoke test only.

Usage:
  KIRBY_SMOKE_N=200 python tests/experiments/similarity/_run_similarity_locally.py \
      <dataset> <exp1> <exp2> ...

Env:
  KIRBY_LOCAL_DS   default dataset (default logd)
  KIRBY_LOCAL_REPS comma-separated rep override (default ecfp4,maccs)
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))   # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))          # tests/experiments/
warnings.filterwarnings('ignore')

import experiments._shared as _shared  # noqa: E402
import kirby.find_best_hybrid as _fbh  # noqa: E402
from kirby.find_best_hybrid import SPEED_TIERS  # noqa: E402
from experiments.similarity import similarity_helpers  # noqa: E402

# SMOKE ONLY: precompute_hybrid_searches hardcodes models=None and speed='full',
# so find_best_hybrid runs ALL of DEFAULT_MODELS (rf/xgboost/ridge/knn/lightgbm/
# mlp/svm/gp) for baselines, allocation evaluators, AND the config search —
# regardless of config['models']. On macOS the gp/lightgbm/mlp paths are slow
# and the TF+OpenMP double-load can segfault (exit 139). Trim DEFAULT_MODELS to
# rf+knn (knn is needed by the distance-tier experiments s04/s05/s06/s08) for
# the LOCAL smoke run only. This is an in-memory override in this throwaway
# driver; src and on-disk config are untouched, so server full runs are
# unaffected.
_SMOKE_MODELS = ('rf', 'knn', 'ridge')
_fbh.DEFAULT_MODELS = {k: v for k, v in _fbh.DEFAULT_MODELS.items()
                       if k in _SMOKE_MODELS}

# SMOKE ONLY: precompute / find_best_hybrid take importance + allocation +
# evaluator lists from SPEED_TIERS['full'] (NOT config). The default importance
# set ['deeplift', 'lightgbm_gain', 'treeshap'] imports TensorFlow for deeplift,
# which on macOS triggers an OpenMP double-load segfault (exit 139) the moment
# any subsequent threaded fit runs alongside it. Drop deeplift/treeshap and the
# TF-free evaluators so the smoke search is both TF-free and tiny. In-memory
# override only; the on-disk SPEED_TIERS (server) is untouched.
_fbh.SPEED_TIERS['full']['importance_methods'] = ['lightgbm_gain']
_fbh.SPEED_TIERS['full']['allocation_methods'] = ['performance_weighted', 'top_2']
_fbh.SPEED_TIERS['full']['internal_evaluators'] = ['match', 'rf']


def fast_config():
    # precompute reads _shared.FULL_BUDGETS (NOT config['budgets']) — shrink it.
    _shared.FULL_BUDGETS = [25]
    tier = SPEED_TIERS['full']
    cfg = {
        'n_splits': 2,
        'search_speed': 'full',
        'allocations': tier['allocation_methods'][:2],
        'importance_methods': tier['importance_methods'][:1],
        'internal_evaluators': ['rf'],
        'budgets': [25],
        'models': ['rf'],
    }
    cfg.update(similarity_helpers.SIMILARITY_EXTRA)
    return cfg


def main(argv):
    ds = argv[0] if argv else os.environ.get('KIRBY_LOCAL_DS', 'logd')
    exps = argv[1:] or list(similarity_helpers.SINGLE_DATASET_EXPERIMENTS)
    reps = os.environ.get('KIRBY_LOCAL_REPS', 'ecfp4,maccs').split(',')

    # Also shrink the distance-tier budgets used by s04/s05/s06/s07/s10/s13.
    _shared.DISTANCE_BUDGETS = [25]
    # run_all passes _shared.DISTANCE_EVALUATORS (knn/rf/mlp/lightgbm/svm) into
    # the distance-tier precompute explicitly; shrink to one so the smoke
    # precompute is tractable. Read at call time -> in-memory override suffices.
    _shared.DISTANCE_EVALUATORS = ['knn']

    # SMOKE ONLY: the dataset's augmentation reps (e.g. selformer foundation
    # model, 3D pharmacophore/usrcat/molclr, atom_pair/top_torsion) are
    # expensive or server-only and dominate local wall time. run_all pulls them
    # from DATASET_REPS[ds]['augmentation']; blank that entry in-memory so the
    # smoke run exercises the experiment code path without them. The on-disk
    # config is untouched (server full runs unaffected).
    from experiments.complexity.complexity_helpers import DATASET_REPS
    if ds in DATASET_REPS:
        DATASET_REPS[ds] = {**DATASET_REPS[ds], 'augmentation': []}

    cfg = fast_config()
    out = os.path.join('tests', 'results', '_smoke_2026_06_05', 'similarity_out')
    os.makedirs(out, exist_ok=True)

    similarity_helpers.run_all(
        dataset_name=ds,
        output_dir=out,
        experiments=exps,
        config=cfg,
        rep_names=reps,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
