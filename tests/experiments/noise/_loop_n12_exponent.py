#!/usr/bin/env python3
"""n12's importance exponent, both arms, on n12's own numbers.

THE QUESTION. n12 multiplies each z-scored feature by its importance raised to
a power, then finds each molecule's nearest neighbours in that space. The power
has been 0.5 since the file was written. The neighbours are found under
manhattan, which kirby.similarity.distance_config has made the default since it
existed. Under manhattan the per-feature term of the distance is the weight
times the absolute difference, so the power that makes a feature's weight equal
its importance is 1.0, not 0.5.

n21 measured the pair on flab_thermo_tresanco and the matched power won 3 of
the 4 scorer-and-distance pairs it ran. This script asks the same question of
n12, on n12's folds, n12's corruption and n12's estimators.

WHAT IT RUNS. One process, one data load, two calls to n12.run with nothing
different between them but n12.IMPORTANCE_EXPONENT. The folds are the same
because the split is seeded, the corruption is the same because the noise seed
is passed explicitly, and the reading is detection average precision against
the molecules whose labels actually moved -- which no downstream model touches.

⛔ ROWS ARE NEVER CUT. All 672 antibodies. What is cut is the search space:
3 outer folds, 1 corruption draw, one injected level, one downstream model,
budget 50 features. See _fast_loop.py for why.

⛔ CACHE. flab_thermo_tresanco's ten representations and its 11,692 stored
search rows are both under /Volumes/seagate/kirby_cache, which is where the
n12 run of 2026-09-20 11:00 read them from, so that is what KIRBY_CACHE_DIR is
set to. Nothing here deletes, moves or renames a cache file. If that store is
still missing the distance_metric column that commit 9c6f7d7 added, the pool
open raises StaleResultDB and this script says so in one line and exits 0,
because the rebuild is the owner's to approve. Under a supervisor that retries
on a timer, the run then starts on its own once the store is rebuilt.
"""
import os

for _k, _v in {
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'KMP_DUPLICATE_LIB_OK': 'TRUE',
        'KIRBY_LGBM_THREADS': '1', 'KIRBY_RF_JOBS': '1',
        # The representation cache holds every rep this dataset needs, so no
        # model is downloaded. Without these two, loading saprot asks
        # huggingface.co for a config file it already has on disk and spends
        # its time on connection retries.
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'KIRBY_CACHE_DIR': '/Volumes/seagate/kirby_cache',
}.items():
    os.environ.setdefault(_k, _v)

if os.environ.get('KIRBY_SMOKE_N', '').strip():
    raise SystemExit(
        "KIRBY_SMOKE_N is set. It is a ROW CAP inside load_data, not a speed "
        "flag, and a detector graded against a corrupted subset has nothing to "
        "find once the rows are cut. Unset it.")

import json
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np

DATASET = 'flab_thermo_tresanco'
CONDITION = 'outlier_p10'
LEVELS = [0.5]
EXP_FOLDS = 3
NOISE_SEEDS = [0]
BUDGETS = [50]
EXP_MODELS = ['rf']
ARMS = [('exponent_0.5', 0.5), ('exponent_1.0', 1.0)]
OUT = '/Volumes/seagate/local_noise_run/loop_n12'


def _arm_path(arm_name):
    return os.path.join(OUT, f'{DATASET}__{CONDITION}__{arm_name}.json')


def main():
    # The supervisor re-runs a job on a timer whatever its exit code, so a run
    # that already produced both arms says so and costs nothing. Delete the two
    # files to re-measure.
    if all(os.path.exists(_arm_path(a)) for a, _ in ARMS):
        print('both arms already on disk in ' + OUT + '; nothing to do')
        out_by_arm = {}
        for a, _ in ARMS:
            with open(_arm_path(a)) as f:
                out_by_arm[a] = json.load(f)
        compare(out_by_arm)
        return 0

    import _local_run as LR
    sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments', 'noise'))
    print('ITERATION SETTINGS:', LR.shrink(models=None, budgets=BUDGETS,
                                           search_folds=2), flush=True)
    import experiments.dataset_policy as DP
    DP._DEFAULT_NOISE_LEVELS = tuple(LEVELS)
    DP.CENSORING_LEVELS_DETECT = tuple(LEVELS)

    from experiments._shared import load_data
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from kirby.result_cache import StaleResultDB
    from kirby.similarity.distance_config import get_distance_metric
    from experiments.noise import n12_importance_weighted_distance as n12

    print('DISTANCE IN FORCE:', get_distance_metric(), flush=True)
    print('EXPONENT THAT DISTANCE MATCHES:',
          n12.EXPONENT_MATCHED_TO_METRIC.get(get_distance_metric()), flush=True)

    rep_names = DATASET_REPS[DATASET]['general']
    t0 = time.time()
    data = load_data(DATASET, rep_preset='all', rep_names=rep_names)
    for r, X in list(data['reps'].items()):
        if int(np.isnan(X).sum()):
            data['reps'][r] = np.nan_to_num(X, nan=0.0)
    data['augmentations'] = None
    print(f'loaded {len(data["labels"])} rows in {time.time() - t0:.1f}s',
          flush=True)

    config = LR.config(models=None, budgets=BUDGETS, exp_folds=EXP_FOLDS)
    config['n_splits'] = EXP_FOLDS
    os.makedirs(OUT, exist_ok=True)

    out_by_arm = {}
    for arm_name, exponent in ARMS:
        n12.IMPORTANCE_EXPONENT = exponent
        print(f'\n{"=" * 70}\nARM {arm_name}: n12.IMPORTANCE_EXPONENT = '
              f'{n12.IMPORTANCE_EXPONENT}\n{"=" * 70}', flush=True)
        t = time.time()
        try:
            res = n12.run(data, config, OUT, DATASET,
                          models=EXP_MODELS,
                          noise_condition=CONDITION,
                          noise_seeds=NOISE_SEEDS,
                          sigmas=LEVELS)
        except StaleResultDB as e:
            print('BLOCKED: the search store this run reads is missing a '
                  'column that commit 9c6f7d7 added, so the pool cannot be '
                  'opened. Nothing was deleted or moved. Rebuilding it is the '
                  f'owner\'s call.\n  {e}', flush=True)
            return 0
        print(f'ARM {arm_name} took {time.time() - t:.0f}s', flush=True)
        path = os.path.join(OUT, f'{DATASET}__{CONDITION}__{arm_name}.json')
        from experiments._shared import convert_numpy
        with open(path, 'w') as f:
            json.dump(convert_numpy(res), f, indent=1)
        print('wrote', path, flush=True)
        out_by_arm[arm_name] = res

    compare(out_by_arm)
    return 0


def compare(out_by_arm):
    """Print the paired reading, one line per scorer and condition."""
    arms = [a for a, _ in ARMS if a in out_by_arm]
    if len(arms) < 2:
        print('\nonly one arm produced a result; nothing to compare')
        return
    a0, a1 = arms[0], arms[1]
    print('\n' + '=' * 78)
    print('DETECTION AVERAGE PRECISION against the molecules whose labels '
          'moved.')
    print('One row is one importance scorer, one feature space and one '
          'neighbour count.')
    print('=' * 78)
    rows = []
    for sigma in sorted(out_by_arm[a0].get('per_sigma', {})):
        s0 = out_by_arm[a0]['per_sigma'][sigma].get('per_importance_method', {})
        s1 = out_by_arm[a1]['per_sigma'][sigma].get('per_importance_method', {})
        for imp in sorted(s0):
            for cond in sorted(s0.get(imp, {})):
                d0 = (s0[imp][cond] or {}).get('detection') or {}
                d1 = ((s1.get(imp) or {}).get(cond) or {}).get('detection') or {}
                for est in sorted(d0):
                    v0 = (d0.get(est) or {}).get('detection_average_precision')
                    v1 = (d1.get(est) or {}).get('detection_average_precision')
                    if v0 is None or v1 is None:
                        continue
                    if not (np.isfinite(v0) and np.isfinite(v1)):
                        continue
                    rows.append((imp, cond, est, float(v0), float(v1)))
    if not rows:
        print('no paired readings')
        return
    print(f'{"scorer":<15}{"space":<32}{"estimator":<26}'
          f'{a0:>14}{a1:>14}')
    for imp, cond, est, v0, v1 in rows:
        print(f'{imp:<15}{cond:<32}{est:<26}{v0:>14.4f}{v1:>14.4f}')
    wins = sum(1 for _, _, _, v0, v1 in rows if v1 > v0)
    ties = sum(1 for _, _, _, v0, v1 in rows if v1 == v0)
    print(f'\nreadings where {a1} scored higher than {a0}: {wins} of '
          f'{len(rows)} paired readings ({ties} of them tied)')
    m0 = float(np.mean([r[3] for r in rows]))
    m1 = float(np.mean([r[4] for r in rows]))
    print(f'mean average precision over those {len(rows)} readings: '
          f'{a0} {m0:.4f}, {a1} {m1:.4f}')


if __name__ == '__main__':
    sys.exit(main())
