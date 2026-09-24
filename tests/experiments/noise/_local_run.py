#!/usr/bin/env python3
"""Run the noise suite on one dataset locally, with every SEARCH axis cut to
the smallest thing that still runs.

⛔ NOT a results run for the paper. What is cut and what is not:

  CUT (search and sweep axes)      1 allocation method, 1 importance scorer,
                                   1 internal evaluator, 2 budgets, 2
                                   downstream models, 3 inner search folds,
                                   4 injected noise levels instead of 6, and
                                   the per-experiment cuts in SWEEP_CUTS below.
  NOT CUT                          every molecule of the dataset, every
                                   representation in its approved pool, every
                                   cleanup, every detector, every arm, and the
                                   5 scaffold-grouped outer folds each
                                   experiment holds out.

Usage:
    python tests/experiments/noise/_local_run.py caco2 n14 n18
    python tests/experiments/noise/_local_run.py herg_ki n07 --condition outlier_p10
"""
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = '/Volumes/seagate/local_noise_run/results'

os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
# ⛔ torch and lightgbm load two different OpenMP runtimes on macOS and a
# multi-threaded fit of one after the other exits 139 with no traceback. The
# pool size is what triggers it and only OMP_NUM_THREADS sets that. See
# project_fold_hook_and_lgbm_segfault_2026_09_02.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))

MODELS = ['rf', 'lightgbm']
BUDGETS = [50, 250]
# Two different fold counts, and they are not the same knob.
#   SEARCH_FOLDS  the inner folds a configuration is scored on, and the folds
#                 the per-representation baselines are measured over.
#   EXP_FOLDS     the scaffold-grouped folds an experiment holds out. 5, the
#                 suite's own design: n18's headline is a count of folds won
#                 and 3 folds cannot carry it.
SEARCH_FOLDS = 3
EXP_FOLDS = 5


def shrink(models=None, budgets=None, search_folds=None, augment=False):
    models = list(models or MODELS)
    budgets = list(budgets or BUDGETS)
    search_folds = search_folds or SEARCH_FOLDS

    import kirby.find_best_hybrid as F
    F.DEFAULT_ALLOCATION_METHODS = ['performance_weighted']
    F.FAST_ALLOCATION_METHODS = ['performance_weighted']
    F.DEFAULT_IMPORTANCE_METHODS = ['lightgbm_gain']
    F.FAST_IMPORTANCE_METHODS = ['lightgbm_gain']
    F.DEFAULT_INTERNAL_EVALUATORS = ['match']
    F.FAST_INTERNAL_EVALUATORS = ['match']
    # [3, None] and not [3]: under objective='similarity' a rep_count other
    # than the whole pool needs a per-fold ranking, and a build older than
    # 2026-09-18 16:33 does not compute one, so every such configuration fails
    # and the geometry arm comes back empty.
    F.DEFAULT_REP_COUNTS = [3, None]
    F.FAST_REP_COUNTS = [3, None]
    F.DEFAULT_N_FOLDS = search_folds

    # This laptop runs several experiment suites at once. n_jobs changes timing
    # only, never a score.
    _nj = int(os.environ.get('KIRBY_RF_JOBS', '2'))
    import sklearn.ensemble as _se
    import sklearn.neighbors as _sn
    F.DEFAULT_MODELS['rf']['regressor'] = lambda: _se.RandomForestRegressor(
        n_estimators=100, max_features='sqrt', random_state=42, n_jobs=_nj)
    F.DEFAULT_MODELS['rf']['classifier'] = lambda: _se.RandomForestClassifier(
        n_estimators=100, max_features='sqrt', random_state=42, n_jobs=_nj)
    F.DEFAULT_MODELS['knn']['regressor'] = lambda: _sn.KNeighborsRegressor(
        n_neighbors=5, n_jobs=_nj)
    F.DEFAULT_MODELS['knn']['classifier'] = lambda: _sn.KNeighborsClassifier(
        n_neighbors=5, n_jobs=_nj)
    # Retier rather than delete, so a models=None site resolves to `models`
    # while an experiment that names dnn or svm explicitly still gets it —
    # the list branch of _resolve_models never looks at the tier.
    for _name, _spec in F.DEFAULT_MODELS.items():
        if _spec.get('tier') in ('fast', 'full'):
            _spec['tier'] = 'fast' if _name in models else 'extra'

    if not augment:
        # The donor-feature axis is orthogonal to every noise question and it
        # triples the precompute (none / uncapped / capped).
        from experiments.complexity import complexity_helpers as CH
        for _v in CH.DATASET_REPS.values():
            _v['augmentation'] = []

    import experiments._shared as S
    S.FULL_BUDGETS = budgets
    S.DISTANCE_BUDGETS = budgets
    S.DOWNSTREAM_MODELS = models
    S.DOWNSTREAM_MODELS_NOISE = models
    S.DISTANCE_MODELS = models + ['knn']
    S.DISTANCE_EVALUATORS = ['knn']
    S.INTERNAL_EVALUATORS = ['match']
    return {'models': models, 'budgets': budgets, 'search_folds': search_folds}


def config(models=None, budgets=None, exp_folds=None):
    return {
        'n_splits': exp_folds or EXP_FOLDS,
        'allocations': ['performance_weighted'],
        'importance_methods': ['lightgbm_gain'],
        'internal_evaluators': ['match'],
        'budgets': list(budgets or BUDGETS),
        'models': list(models or MODELS),
    }


def sweep_cuts(exp_ids):
    """Shrink each experiment's own sweep. Every entry says what it cut."""
    import importlib
    notes = {}

    import experiments.dataset_policy as DP
    # 4 is the fewest _curve_robustness accepts, so the ceiling extrapolation
    # at the end of a run still has a curve to fit.
    DP._DEFAULT_NOISE_LEVELS = (0.0, 0.15, 0.3, 0.75)
    DP.CENSORING_LEVELS_DETECT = (0.0, 0.10, 0.25)
    notes['levels'] = '6 injected levels -> 4'

    def mod(name):
        return importlib.import_module(f'experiments.noise.{name}')

    def force(m, **kw):
        """Wrap run() so it always gets these kwargs.

        n09, n11 and n12 read their draw count as kwargs.get('noise_seeds',
        [0, 1, 2]) — a literal default with no module constant behind it — and
        run_all forwards no kwargs of its own.
        """
        _orig = m.run

        def _wrapped(data, cfg, output_dir, dataset_name, **kwargs):
            kwargs.update(kw)
            return _orig(data, cfg, output_dir, dataset_name, **kwargs)
        m.run = _wrapped

    if 'c10' in exp_ids:
        m = mod('c10_noise_robustness')
        m.REGRESSION_STRATEGIES = ['gaussian', 'outlier_p10', 'grouped_wider']
        notes['c10'] = '11 shapes -> 3, so 56 recipe searches -> 13'

    if 'n02' in exp_ids:
        m = mod('n02_sample_weighting')
        m.WEIGHTED_MODELS = list(MODELS)
        m.DEFAULT_IMPORTANCE_METHODS = ['lightgbm_gain']
        notes['n02'] = '5 weighted models -> 2, deeplift dropped'

    if 'n07' in exp_ids:
        m = mod('n07_remediation')
        m.NOISE_SEEDS = [0]
        notes['n07'] = '3 noise draws -> 1'

    if 'n09' in exp_ids:
        m = mod('n09_lnl_survey')
        m.NOISY_SEARCH_BUDGETS = [100]
        force(m, noise_seeds=[0])
        notes['n09'] = '3 search budgets -> 1, 3 draws -> 1'

    if 'n11' in exp_ids:
        m = mod('n11_estimation_remediation_grid')
        # ⛔ TWO LISTS, AND `ALPHAS` IS NEITHER OF THEM. n11 split its one
        # swept list in two on 2026-09-18 -- a smoothing alpha is a fraction of
        # the way from the label to a target, a weighting strength is an
        # unbounded row-weight ratio -- and kept `ALPHAS` as an alias bound once
        # at import, which the result file reports and no loop reads. This
        # entry rebound the alias alone, so from 2026-09-18 to 2026-09-20 it
        # reported cutting the swept values 4 -> 2 and cut nothing: both loops
        # still ran all four. Fixed 2026-09-20.
        m.SMOOTH_ALPHAS = [0.3, 0.7]
        # 2-to-1 is what n07 and n18 used before 2026-09-18; 4-to-1 is what n18
        # then measured as the winner on two domains. Keeping both means the
        # iteration run can still tell them apart. A results run sweeps all four.
        m.WEIGHT_STRENGTHS = [1.0, 3.0]
        m.ALPHAS = m.SMOOTH_ALPHAS
        m.WEIGHT_METHODS = ['inverse']
        m.SMOOTH_METHODS = ['proportional', 'threshold']
        m.NEIGHBOR_METRICS = ['manhattan']
        m.EXTRA_TARGETS = ('oof',)
        m.TORCH_ESTIMATORS = []
        m.OPTIONAL_ESTIMATORS = []
        force(m, noise_seeds=[0])
        notes['n11'] = ('4 smoothing alphas -> 2, 4 weight strengths -> 2 '
                        '(2-to-1 and 4-to-1), 3 weightings -> 1, 4 smoothers '
                        '-> 2, 4 distances -> 1, 2 extra targets -> 1 (the '
                        'Gaussian-process target is the dropped one), torch '
                        'and ngboost detectors dropped, 3 draws -> 1')

    if 'n12' in exp_ids:
        m = mod('n12_importance_weighted_distance')
        # Same reason as n21: this file exists to ask whether scaling features
        # by their importance improves neighbour-based detection, so the
        # importance scorer and the neighbour count stay.
        force(m, noise_seeds=[0])
        notes['n12'] = ('3 draws -> 1; k 5/10/20 and both importance scorers '
                        'kept -- they are the question')

    if 'n13' in exp_ids:
        m = mod('n13_noise_distributions')
        m.NOISE_SEEDS = [0]
        m.LNL_EPOCHS = 20
        # One shape per family: light-tailed, heavy-tailed, a corrupted subset,
        # and one keyed to the scaffold groups. The question is whether a
        # ranking transfers BETWEEN shapes, so more than one has to run.
        m.NOISE_TYPES = ['gaussian', 'student_t_nu3', 'outlier_p10',
                         'grouped_wider']
        notes['n13'] = '3 draws -> 1, 50 epochs -> 20, 11 shapes -> 4'

    if 'n16' in exp_ids:
        m = mod('n16_importance_under_noise')
        m.IMPORTANCE_METHODS = ['lightgbm_gain']
        m.N_REPLICATES = 1
        notes['n16'] = '3 importance scorers -> 1, 3 replicates -> 1'

    if 'n17' in exp_ids:
        mod('n17_replicate_removal').N_DRAWS = 1
        notes['n17'] = '3 draws -> 1'

    if 'n19' in exp_ids:
        mod('n19_measurement_count').N_DRAWS = 1
        notes['n19'] = '3 draws -> 1'

    if 'n20' in exp_ids:
        m = mod('n20_clean_before_search')
        m.NOISE_DRAWS = (0,)
        m.FLAG_RATES = (0.10,)
        # tune=True turns one downstream fit into 61 for rf and lightgbm. The
        # compute audit puts n20 at ~121 million fits per dataset with it on
        # and ~2.9 million with it off. Both arms keep identical settings,
        # which is what the comparison needs; only the cost changes.
        _orig = m.search_settings

        def _untuned(n_samples, pruning_mode='aggressive_pruning'):
            out = _orig(n_samples, pruning_mode)
            out['tune'] = False
            out['tune_knn_k'] = False
            out.pop('tune_n_iter', None)
            return out
        m.search_settings = _untuned
        notes['n20'] = ('3 draws -> 1, 3 flag rates -> 1, tuning off '
                        '(~40x, and it applies to both arms alike)')

    if 'n21' in exp_ids:
        m = mod('n21_neighbour_distance')
        # ⛔ NOT CUT: the neighbour count, the importance scorer, the distance
        # metric and the importance exponent. Those four ARE the question --
        # which way of computing the similarity finds the corrupted molecules
        # -- and cutting them is cutting the experiment. Only replication is.
        m.NOISE_SEEDS = (0,)
        notes['n21'] = ('3 draws -> 1; every similarity axis kept '
                        '(k 5/10/20, lightgbm_gain + treeshap, '
                        'euclidean + manhattan, exponent 0.5 + 1.0)')

    if 'n23' in exp_ids:
        m = mod('n23_uncertainty_cleaning')
        m.NOISE_DRAWS = (0,)
        m.DROP_SHARES = (0.10,)
        m.MOVE_STRENGTHS = (0.5,)
        m.MOVE_METHODS = ('proportional',)
        notes['n23'] = ('3 draws -> 1, 3 drop shares -> 1, 3 strengths -> 1, '
                        '2 movers -> 1')

    if 'n26' in exp_ids:
        m = mod('n26_noise_variance_from_repeats')
        m.N_SPLIT_HALF_REPEATS = 5
        m.INJECTED_VARIANCES = (0.25,)
        notes['n26'] = ('20 split-half draws -> 5, 3 injected variances in the '
                        'estimator check -> 1')

    if 'n27' in exp_ids:
        m = mod('n27_removal_cost_and_propagation')
        m.REPEATS = 3
        # 0.9 kept beside 1.0: removing a tenth is the operation the
        # 2026-09-23 headroom probe priced.
        m.SHARES = (0.9, 1.0)
        m.NOISE_DRAWS = (0,)
        m.ALPHAS = (0.5, 1.0)
        m.OOF_SPLITS = 3
        notes['n27'] = ('20 repeats -> 3, 4 shares -> 2 (0.9 and 1.0), 3 draws '
                        '-> 1, 3 relabelling fractions -> 2, 5 inner folds -> 3')

    if 'n29' in exp_ids:
        m = mod('n29_document_bias')
        m.N_BOOTSTRAP = 50
        m.OOF_SPLITS = 3
        notes['n29'] = '200 bootstrap resamples -> 50, 5 inner folds -> 3'

    if 'n30' in exp_ids:
        m = mod('n30_censored_likelihood')
        m.WINSOR_C = (1.0,)
        m.SHUFFLE_SEEDS = (0,)
        m.OOF_SPLITS = 3
        notes['n30'] = ('3 winsorising thresholds -> 1, 3 shuffle seeds -> 1, '
                        '5 inner folds -> 3')

    return notes


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('experiments', nargs='+')
    p.add_argument('--out', default=OUT)
    p.add_argument('--models', nargs='+', default=None)
    p.add_argument('--budgets', nargs='+', type=int, default=None)
    p.add_argument('--folds', type=int, default=None,
                   help='outer folds each experiment holds out (default 5)')
    p.add_argument('--search-folds', dest='search_folds', type=int, default=None,
                   help='inner folds the recipe search scores on (default 3)')
    p.add_argument('--force', action='store_true')
    p.add_argument('--condition', default=None)
    p.add_argument('--augment', action='store_true')
    a = p.parse_args()

    print('SHRINK:', shrink(a.models, a.budgets, a.search_folds,
                            augment=a.augment))
    print('SWEEP CUTS:', sweep_cuts(set(a.experiments)))

    from experiments.noise import noise_helpers as NH
    kw = {'noise_condition': a.condition} if a.condition else {}
    t0 = time.time()
    NH.run_all(dataset_name=a.dataset, output_dir=a.out,
               experiments=a.experiments,
               config=config(a.models, a.budgets, a.folds),
               force_rerun=a.force, **kw)
    print(f"\nTOTAL {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
