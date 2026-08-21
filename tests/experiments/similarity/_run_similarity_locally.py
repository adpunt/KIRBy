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
# KIRBY_LOCAL_MODELS overrides the smoke model set (comma-separated). Used by the
# §V cache gate to prove the changed-MODEL-list reuse path: adding a model must
# reuse the old per-model rows from the SQLite ResultCache and compute ONLY the
# new model. Default keeps rf/knn/ridge (knn needed by the distance-tier exps).
_SMOKE_MODELS = tuple(os.environ.get('KIRBY_LOCAL_MODELS', 'rf,knn,ridge').split(','))
_fbh.DEFAULT_MODELS = {k: v for k, v in _fbh.DEFAULT_MODELS.items()
                       if k in _SMOKE_MODELS}

# SMOKE ONLY: precompute / find_best_hybrid take importance + allocation +
# evaluator lists from the module-level DEFAULT_* constants (NOT config). The
# default importance set ['deeplift', 'lightgbm_gain', 'treeshap'] imports
# TensorFlow for deeplift, which on macOS triggers an OpenMP double-load segfault
# (exit 139) the moment any subsequent threaded fit runs alongside it. Drop
# deeplift/treeshap and the TF-free evaluators so the smoke search is both
# TF-free and tiny. In-memory reassignment only; the src constants (server)
# are untouched.
_fbh.DEFAULT_IMPORTANCE_METHODS = ['lightgbm_gain']
_fbh.DEFAULT_ALLOCATION_METHODS = ['performance_weighted', 'top_2']
_fbh.DEFAULT_INTERNAL_EVALUATORS = ['match', 'rf']


def fast_config():
    # precompute reads _shared.FULL_BUDGETS (NOT config['budgets']) — shrink it.
    _shared.FULL_BUDGETS = [25]
    cfg = {
        'n_splits': 2,
        'search_speed': 'full',
        'allocations': _fbh.DEFAULT_ALLOCATION_METHODS[:2],
        'importance_methods': _fbh.DEFAULT_IMPORTANCE_METHODS[:1],
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

    # SMOKE ONLY (molecular path): _shared._load_data_impl's molecular branch
    # calls get_molecular_representations(smiles, rep_preset) with rep_preset
    # ='all' and only filters to rep_names AFTERWARD (_shared.py:2548-2550) — so
    # it computes ALL ~33 molecular reps (incl. foundation models grover/
    # molformer/chemberta/mhg_gnn that hang or fail locally) before discarding
    # all but ecfp4/maccs. The antibody/DTA/nucleotide branches correctly pass
    # the rep list as the preset; the molecular branch does not (SHARED-CODE
    # finding #M1 in the report). Work around it locally by wrapping the
    # molecular rep fn so a comma-list preset only computes the wanted reps.
    # Driver-only; src untouched, server runs unaffected.
    try:
        import find_best_hybrid_molecular as _fbhm  # noqa: E402
        _orig_get = _fbhm.get_representations
        _wanted = set(reps)

        def _smoke_get_representations(smiles, preset_or_list, *a, **kw):
            if preset_or_list == 'all':
                preset_or_list = ','.join(reps)
            return _orig_get(smiles, preset_or_list, *a, **kw)
        _fbhm.get_representations = _smoke_get_representations
    except Exception as _e:
        print(f"  [smoke] molecular rep-fn wrap skipped: {_e}")

    cfg = fast_config()
    out_root = os.environ.get('KIRBY_LOCAL_OUT',
                              os.path.join('tests', 'results', '_smoke_V_2026_06_10', 'similarity_out'))
    out = out_root
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
