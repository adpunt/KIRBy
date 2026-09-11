#!/usr/bin/env python3
"""n06 on REAL rows, scoped to minutes.

The synthetic smoke (_smoke_n06.py) checks arithmetic whose answer is known on
paper. This one checks that the thing runs on data nobody made up: real repeat
measurements, a real hybrid build per fold, real estimators. Reading the code
is not evidence it works — the last audit of this kind found four defects by
reading and four more on the first real run.

It is NOT a results run. The rows are capped, the budget is one small number,
only the cheap local representations are loaded, and only the neighbour-based
estimators run unless --all-estimators is passed. What it prints is how long a
fold costs, which is what sizes the cluster job.

THE SUBSET KEEPS WHOLE PAIRS. A drug-protein pair measured three times only
carries a disagreement if all three rows are in hand; drawing rows at random
turns repeat measurements into singletons and quietly shrinks the ground truth.

Usage:
    python tests/experiments/noise/_smoke_n06_e2e.py --subsample 1200
    python tests/experiments/noise/_smoke_n06_e2e.py --dataset herg_ki
    python tests/experiments/noise/_smoke_n06_e2e.py --all-estimators
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

# n06 resolves its estimator list the way n11 does, which on a machine with
# torch is 14 estimators; the torch and boosting ones are most of the cost.
# Measured on 1,200 BindingDB rows: one fold took 689 seconds with all 14.
# KIRBY_NOISE_LIGHT_ESTIMATORS is the switch n11 already has for this, and it
# has to be set BEFORE n06 is imported. The cluster runs the full list.
if '--all-estimators' not in sys.argv:
    os.environ['KIRBY_NOISE_LIGHT_ESTIMATORS'] = '1'

import argparse  # noqa: E402
import warnings  # noqa: E402
import time  # noqa: E402
import json  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import experiments._shared as _shared  # noqa: E402
from experiments._shared import (  # noqa: E402
    load_data, precompute_hybrid_searches, save_experiment_results,
)

_shared.FULL_BUDGETS = [25]

# One importance scorer for the local run. The default list includes treeshap,
# and on this machine shap.TreeExplainer imports transformers, which imports
# tensorflow, which fails on the installed protobuf ('FieldDescriptor' object
# has no attribute 'is_repeated'). Every treeshap configuration then raises and
# the search spends its time on the traceback. This only scopes the smoke, and
# it changes the cache key, so nothing here can be mistaken for a real search.
import kirby.find_best_hybrid as _fbh  # noqa: E402
_fbh.DEFAULT_IMPORTANCE_METHODS = ['lightgbm_gain']

# Two cheap representations per dataset, both computable on a laptop. The point
# is to run the real path on real rows, not to produce a number anyone quotes.
SMOKE_REPS = {
    'bindingdb_kd': ['drug_maccs', 'protein_aac'],   # 167 and 20 features
    'herg_ki': ['ecfp4', 'pdv'],
    'herg': ['ecfp4', 'pdv'],
}

# Results go under tests/results, never a scratch directory: three complexity
# files were dead for weeks because they read from /private/tmp.
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results',
    '_smoke')


def tiny_config():
    return {'n_splits': 2, 'allocations': ['greedy'],
            'importance_methods': ['lightgbm_gain'],
            'internal_evaluators': ['rf'], 'budgets': [25], 'models': ['rf']}


def subset_keeping_whole_pairs(data, n_rows, seed=0):
    """Take about n_rows rows, never splitting a drug-protein pair.

    Pairs measured more than once are taken first — they are the only rows that
    carry a ground truth at all — and rows measured once fill the rest, so the
    estimators still have neighbours that are not repeats.

    A dataset with no drug and protein ids has no pairs to keep whole, and its
    ground truth comes from matching molecules to another curator instead. Such
    a dataset is left whole: cutting rows there only throws away matches.
    """
    md = data.get('metadata') or {}
    drug_ids, protein_ids = md.get('drug_ids'), md.get('protein_ids')
    n = len(data['labels'])
    if drug_ids is None or protein_ids is None:
        print(f"  no drug and protein ids on {n} rows — nothing to subset "
              f"without breaking the molecule match, keeping every row")
        return data

    by_pair = {}
    for i, key in enumerate(f'{d}|{p}' for d, p in zip(drug_ids, protein_ids)):
        by_pair.setdefault(key, []).append(i)
    repeated = [rows for rows in by_pair.values() if len(rows) > 1]
    singles = [rows[0] for rows in by_pair.values() if len(rows) == 1]

    rng = np.random.default_rng(seed)
    rng.shuffle(repeated)
    keep, n_pairs = [], 0
    for rows in repeated:
        if len(keep) + len(rows) > n_rows * 0.6:
            break
        keep.extend(rows)
        n_pairs += 1
    n_repeat_rows = len(keep)
    fill = rng.choice(singles, min(len(singles), max(0, n_rows - len(keep))),
                      replace=False)
    keep = np.sort(np.concatenate([np.array(keep, dtype=int), fill]))

    data['labels'] = np.asarray(data['labels'])[keep]
    data['reps'] = {k: X[keep] for k, X in data['reps'].items()}
    if data.get('groups') is not None:
        data['groups'] = np.asarray(data['groups'])[keep]
    for mk in ('drug_ids', 'protein_ids', 'drug_smiles', 'protein_seqs'):
        v = md.get(mk)
        if v is not None and len(v) == n:
            md[mk] = ([v[i] for i in keep] if not isinstance(v, np.ndarray)
                      else np.asarray(v)[keep])
    data['metadata'] = md
    # The molecule list has to be cut with the rows, or the cross-curator
    # provider matches row 900 to the wrong molecule.
    for key in ('drug_smiles', 'smiles'):
        v = data.get(key)
        if v is not None and len(v) == n:
            data[key] = [v[i] for i in keep]
    print(f"  kept {len(keep)} rows: {n_repeat_rows} of them sit in "
          f"{n_pairs} pairs measured more than once, the rest measured once")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subsample', type=int, default=1200,
                    help='0 keeps every row')
    ap.add_argument('--dataset', default='bindingdb_kd')
    ap.add_argument('--reps', default=None,
                    help="comma-separated; defaults to this dataset's entry "
                         "in SMOKE_REPS")
    ap.add_argument('--output', default=DEFAULT_OUTPUT)
    ap.add_argument('--all-estimators', action='store_true',
                    help='run the full list n06 resolves instead of the '
                         'neighbour-based ones; this is the cluster cost')
    ap.add_argument('--similarity-arm', action='store_true',
                    help='also build the hybrid whose features were chosen by '
                         'distance. It needs a similarity-objective search, '
                         'and builds one if none is cached, which is not fast')
    args = ap.parse_args()

    print(f"=== n06 end to end, real {args.dataset} rows ===")
    print(f"  estimators: "
          f"{'the full list' if args.all_estimators else 'neighbour-based only'}")
    t0 = time.time()
    reps = (args.reps.split(',') if args.reps
            else SMOKE_REPS.get(args.dataset))
    if not reps:
        raise SystemExit(f"no cheap representation pair is registered for "
                         f"{args.dataset}; pass --reps a,b")
    data = load_data(args.dataset, rep_preset='all', rep_names=reps)
    for k, X in list(data['reps'].items()):
        if np.isnan(X).any():
            data['reps'][k] = np.nan_to_num(X, nan=0.0)
    print(f"  loaded {len(data['labels'])} rows in {time.time() - t0:.1f}s")

    if args.subsample and len(data['labels']) > args.subsample:
        data = subset_keeping_whole_pairs(data, args.subsample)
    data['augmentations'] = None

    config = tiny_config()
    t0 = time.time()
    data['precomputed_searches'] = precompute_hybrid_searches(
        reps=data['reps'], labels=data['labels'], groups=data.get('groups'),
        config=config, augmentations=None, task_type=data['task_type'],
        domain=data.get('domain', 'molecular'),
        dataset_name=args.dataset, tier='general', internal_evaluators=['rf'])
    print(f"  precompute done in {time.time() - t0:.1f}s")

    from n06_bindingdb_ground_truth import run
    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()
    results = run(data, config, args.output, args.dataset, budget=25,
                  similarity_arm=args.similarity_arm)
    wall = time.time() - t0

    # What a smoke has to check, beyond not crashing.
    assert results.get('metric'), 'the result file must carry metric'
    assert results.get('task_type'), 'the result file must carry task_type'
    if results.get('skipped'):
        print(f"\n  n06 SKIPPED: {results['reason']}")
        print(f"  provider counts: "
              f"{json.dumps(results['providers'], indent=2)[:1200]}")
        sys.exit(1)

    for name, block in results['ground_truth'].items():
        print(f"\n  {name}: {block['n_groups']} groups over "
              f"{block['n_rows_covered']} rows, scored by {block['statistic']}")
        for space, per_est in block['per_space'].items():
            print(f"    {space}:")
            rows = [(e, v.get('value'), v.get('n_replicate_groups'),
                     v.get('refused')) for e, v in per_est.items()]
            for e, v, n, why in sorted(rows,
                                       key=lambda r: (r[1] is None,
                                                      -(r[1] or 0))):
                shown = '  n/a' if v is None else f'{v:+.3f}'
                tail = f'  ({why})' if why else ''
                print(f"      {e:<24s} {shown}  over {n} groups{tail}")
        minus = block['minus_comparator']
        for space, row in minus['per_space'].items():
            named = {e: d for e, d in row.items() if d is not None}
            if named:
                print(f"    {space} minus {minus['comparator_space']}: "
                      + ', '.join(f'{e} {d:+.3f}' for e, d in named.items()))

    assert os.path.exists(results['per_row_scores_file']), \
        'the per-molecule scores were not written'
    save_experiment_results(results, args.output,
                            'n06_bindingdb_ground_truth', args.dataset)

    n_scored = sum(
        1 for block in results['ground_truth'].values()
        for per_est in block['per_space'].values()
        for v in per_est.values() if v.get('value') is not None)
    print(f"\n=== n06 ran in {wall:.0f}s on {len(data['labels'])} rows, "
          f"{n_scored} (space, estimator) pairs produced a number ===")
    print(f"  seconds per space: {results['seconds_per_space']}")
    for space, per_est in results.get('seconds_per_estimator', {}).items():
        worst = list(per_est.items())[:4]
        print(f"  {space}: " + ', '.join(f'{e} {v:.0f}s' for e, v in worst))
    if n_scored == 0:
        print("  NOTHING was scored — that is a failure, not a pass")
        sys.exit(2)


if __name__ == '__main__':
    main()
