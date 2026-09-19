"""Run the similarity wave end to end on a laptop, on whole small datasets.

Not a smoke test. Every molecule of the dataset is used; what is shrunk is the
searched grid, not the data. The grid itself is cut through the KIRBY_LOCAL_*
variables that tests/experiments/_local_grid.py reads, so this file sets those
and adds only what is specific to the similarity wave:

  representations   7 that RDKit builds in minutes, in place of the 10 named
                    in DATASET_REPS (grover, molformer, mhg_gnn, smited are
                    server-only). Six are substructure fingerprints and pdv is
                    200 physicochemical descriptors, which is the one that
                    should win alone on a permeability endpoint — the pool is
                    picked so the single-representation baseline is a hard one.
                    mordred was in it and came out: its build is 30-50 minutes
                    per dataset on a loaded laptop and it measures the same kind
                    of quantity pdv does.
  augmentation      emptied, so the precompute runs one condition instead of
                    three
  objectives        the similarity-objective precompute is warmed as well as
                    the predictive one. Six experiments ask the pool for
                    recipes built to predict each molecule from its five
                    nearest neighbours, and nothing in the repository writes
                    that family, so without this they cold-build it inside
                    whichever experiment asks first.

⛔ Every number out of this is a smoke number. The allocation, importance and
budget axes are cut for compute, so no cell here is evidence about which of
them wins.

Usage:
  python tests/experiments/similarity/_local_autoresearch.py <dataset> [exp ...]
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))   # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))          # tests/experiments/
warnings.filterwarnings('ignore')

POOL = os.environ.get(
    'KIRBY_LOCAL_POOL',
    'ecfp4,maccs,avalon,atom_pair,top_torsion,pdv,ecfp4_count'
).split(',')

# find_best_hybrid runs DEFAULT_N_FOLDS=5 and precompute_hybrid_searches does
# not forward a fold count, so the stored per-fold scores are always 5 folds at
# seed 42. Every experiment splits with nested_cv_outer_folds(config['n_splits'])
# and get_fold_pool_recipes ranks the pool's fold i against the caller's fold i,
# so the two counts have to agree.
N_SPLITS = 5

# Read by _local_grid at _shared import time, so they are set before it.
_GRID = {
    'KIRBY_LOCAL_GRID': '1',
    'KIRBY_LOCAL_BUDGETS': '25,100,250',
    'KIRBY_LOCAL_MODELS': 'rf',
    'KIRBY_LOCAL_EVALUATORS': 'rf',
    'KIRBY_LOCAL_IMPORTANCE': 'lightgbm_gain',
    'KIRBY_LOCAL_ALLOCATIONS': 'performance_weighted,power_2',
    'KIRBY_LOCAL_REP_COUNTS': '3,none',
}
for _k, _v in _GRID.items():
    os.environ.setdefault(_k, _v)
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

import experiments._shared as _shared  # noqa: E402
from experiments.similarity import similarity_helpers  # noqa: E402


def config():
    cfg = {
        'n_splits': N_SPLITS,
        'allocations': os.environ['KIRBY_LOCAL_ALLOCATIONS'].split(','),
        'importance_methods': ['lightgbm_gain'],
        'internal_evaluators': list(_shared.INTERNAL_EVALUATORS),
        'budgets': list(_shared.FULL_BUDGETS),
        'models': list(_shared.DOWNSTREAM_MODELS),
    }
    cfg.update(similarity_helpers.SIMILARITY_EXTRA)
    # s02 has two axes beyond budget: which distance lineage the recipe comes
    # from, and which search family. Measuring both fully is 12 cells and about
    # three hours a dataset here, so this run measures them one at a time and
    # says which. KIRBY_S02_LINEAGES / KIRBY_S02_FAMILIES override.
    cfg['s02_lineages'] = os.environ.get(
        'KIRBY_S02_LINEAGES', 'general,general_distance').split(',')
    cfg['s02_families'] = [f for f in os.environ.get(
        'KIRBY_S02_FAMILIES', '').split(',') if f] or None
    return cfg


def main(argv):
    ds = argv[0]
    exps = argv[1:] or list(similarity_helpers.SINGLE_DATASET_EXPERIMENTS)

    # The augmentation representations named in DATASET_REPS are 3D conformer
    # sets and foundation models. Blanking the entry leaves the 'none'
    # condition alone instead of none/uncapped/capped.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    if ds in DATASET_REPS:
        DATASET_REPS[ds] = {**DATASET_REPS[ds], 'augmentation': []}

    # _load_data_impl's molecular branch asks get_representations for the preset
    # 'all' and filters afterwards, so all ~33 molecular representations would
    # be built and thrown away. Rewrite the preset to the list actually wanted.
    try:
        import find_best_hybrid_molecular as _fbhm
        _orig = _fbhm.get_representations

        def _only_wanted(smiles, preset_or_list, *a, **kw):
            if preset_or_list == 'all':
                preset_or_list = ','.join(POOL)
            return _orig(smiles, preset_or_list, *a, **kw)
        _fbhm.get_representations = _only_wanted
    except Exception as e:
        print(f"  [local] molecular rep-fn wrap skipped: {e}")

    out = os.environ.get('KIRBY_LOCAL_OUT',
                         os.path.join('tests', 'results_local_sim'))
    os.makedirs(out, exist_ok=True)

    similarity_helpers.run_all(
        dataset_name=ds,
        output_dir=out,
        experiments=exps,
        config=config(),
        rep_names=list(POOL),
        precompute_objectives=('similarity',),
    )


if __name__ == '__main__':
    main(sys.argv[1:])
