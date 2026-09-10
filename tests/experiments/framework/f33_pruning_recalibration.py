#!/usr/bin/env python3
"""f33 — record every configuration on every fold, unpruned, so both prunes can
be replayed off the record instead of re-fitted.

WHY THIS EXISTS. Two answers are open in FRAMEWORK_FINALIZATION.md 3.4: whether
the end-of-fold-0 allocation prune stays on by default, and whether the two-fold
version gets built. Every pruning number in circulation was fitted in April on 13
allocation methods (f15). DEFAULT_ALLOCATION_METHODS is five now and only two of
those five — performance_weighted and power_2 — were in the April run. The brief
is PRUNING_RECALIBRATION_HANDOFF.md, 2026-09-02.

WHAT THE TWO PRUNES ARE, read from the code on 2026-09-10.

  End-of-fold-0 allocation prune. find_best_hybrid.py:6083. After fold 0 the
  results are grouped by (importance method, internal evaluator, model, budget);
  in each group the best `strategy_top_n` allocation methods score a point; a
  method is dropped from the remaining folds when it scored in fewer than
  `final_strategy_threshold` of the groups. Defaults 3 and 0.15. It is ON by
  default: find_best_hybrid.py:6818 takes adaptive_pruning=True and line 3229
  turns that into 0.15. A configuration that CRASHES on fold 0 contributes no
  score, so its allocation method loses that placement (line 6049) — a method can
  be pruned for crashing.

  Mid-flight tail drop. find_best_hybrid.py:6004. At a checkpoint the worst
  `tail_drop_per_checkpoint` fraction of (allocation, importance) pairs in each
  (model, evaluator) group is dropped. A checkpoint fires only on an EXACT budget
  match, so the shipped aggressive_pruning checkpoints [25, 50, 100] fire once at
  the default budgets [50, 150, 300]. Off by default.

THE SHAPE IS f15'S AND THE LOOP IS NOT. f15 scored every configuration on every
fold and stored the per-fold list; f15_pruning_analysis.py and f16 then replayed
rules off that record with no extra fits. That is what makes the two-fold question
free — the record already holds all five folds. But two things in f15's own
scaffolding are wrong against today's engine:

  1. f15 precomputed each representation's allocation score on the OUTER fold's
     validation rows and handed them in as precomputed_alloc_scores. create_hybrid
     scores them on its own carve inside the fold's TRAINING rows. Doing it f15's
     way changed which features were selected — measured 2026-08-27 on three
     configurations, reverted in 5ea8475.
  2. f15 precomputed importance over the whole fold's training rows. hybrid.py:5285
     computes it on the internal training part only.

The engine does both correctly and caches them per fold to disk
(find_best_hybrid.py:5377-5470), through the same internal_holdout_split that
create_hybrid calls. So this file runs the engine unpruned and keeps what it
returns; it does not re-implement the loop.

WHAT IS RECORDED. SearchResult.all_results carries one dict per configuration with
fold_scores (one entry per fold), time_seconds (the WHOLE configuration across all
its folds, not per fold), allocation_method, importance_method, internal_evaluator,
budget and model. Those rows also land in <cache>/result_db/<dataset>.db, so the
run doubles as precompute for (general, none, predictive) and resumes if stopped.

BUDGETS. The default search runs [50, 150, 300]; aggressive_pruning's checkpoints
are [25, 50, 100] and fire only on an exact match. Running the union means one
record answers both — f33_analyse.py restricts to any subset of budgets.

Usage:
    python tests/experiments/framework/f33_pruning_recalibration.py --check
    python tests/experiments/framework/f33_pruning_recalibration.py --smoke
    python tests/experiments/framework/f33_pruning_recalibration.py --datasets tcga_brca_pam50
    python tests/experiments/framework/f33_pruning_recalibration.py --datasets logd,caco2
"""

import os
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
# MEASURED 2026-09-03 (f29): a torch DNN fit placed after LightGBM in the same
# process sat at 0% CPU for 22 minutes on this machine; the same fit alone took
# 44 s. MEASURED 2026-09-04 (f31): the same ordering segfaults outright at
# dnn.py:113 with no traceback. This grid has lightgbm and dnn as downstream
# models AND as internal evaluators, so it is exposed to both. These two
# variables are what f31 and f32 ran under, and f32 completed nine datasets on
# 2026-09-10. Must be set before any import that pulls in libomp.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import faulthandler
faulthandler.enable()   # the segfault above prints nothing without this

import sys
# Line-buffered so a long run streams into a log file instead of appearing at the
# end. kirby's own prints do not pass flush=True.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import time
import argparse
import warnings
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# The nine datasets whose signed-off 10-representation pool is complete in
# /Volumes/seagate/kirby_cache/rep_cache, checked file by file against
# DATASET_REPS[name]['general'] on 2026-09-10. The same nine f32 ran.
#
# Two more folders exist on that disk and are NOT here because their pools are
# incomplete, not because anything was cut: abcov_kd holds 2 of its 10
# representations, qm9 holds 8 of 10 (graphmvp and molformer missing). Building
# either pool locally puts that dataset back in this list.
DATASETS = [
    'buchwald_hartwig',        # 3955 rows, rmse
    'caco2',                   # 2158 rows, rmse
    'flab_expression_koenig',  # 4275 rows, spearman
    'flab_thermo_tresanco',    #  672 rows, spearman
    'herg_ki',                 # 1412 rows, rmse
    'humanness_therapeutic',   #  564 rows, auc
    'logd',                    # 5028 rows, rmse
    'nt_promoter_tata',        # 5274 rows, mcc
    'tcga_brca_pam50',         #  821 rows, f1_weighted
]

# The three the default search runs, plus the two aggressive_pruning checkpoints
# that a default search never reaches. f33_analyse.py reports the shipped rule
# over [50, 150, 300] and the aggressive schedule over all five, separately.
BUDGETS = [25, 50, 100, 150, 300]
DEFAULT_SEARCH_BUDGETS = [50, 150, 300]

N_FOLDS = 5
SEED = 42

# None on all four means the engine's own defaults, which is the point: this
# measures the prune on the search space as it actually stands. Written out here
# so --check can print what they resolved to.
ALLOCATION_METHODS = None    # -> DEFAULT_ALLOCATION_METHODS, five methods
IMPORTANCE_METHODS = None    # -> deeplift, lightgbm_gain, treeshap
INTERNAL_EVALUATORS = None   # -> match, rf, lightgbm, dnn
MODELS = None                # -> rf, lightgbm, xgboost, svm, dnn, less any the
                             #    dataset policy skips (nt_* loses xgboost)

TIER = 'general'
AUG_STRATEGY = 'none'

OUT_DIR = _ROOT / 'tests' / 'results' / 'framework' / 'f33_pruning'


# =============================================================================
# GRID RESOLUTION — shared by --check and the run, so they cannot disagree
# =============================================================================

def resolve_grid(dataset_name, task_type, n_samples, models=None, budgets=None,
                 n_folds=None):
    """What the engine will search for this dataset.

    Mirrors find_best_hybrid.py:5255-5305: the evaluator named 'match' becomes
    the model's own type and duplicates collapse, and the model set goes through
    the dataset policy. Returns a dict, and every number f33_analyse.py checks
    against is in it.
    """
    from kirby.find_best_hybrid import (
        DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS,
        DEFAULT_INTERNAL_EVALUATORS, _resolve_models,
    )

    allocation = list(ALLOCATION_METHODS or DEFAULT_ALLOCATION_METHODS)
    importance = list(IMPORTANCE_METHODS or DEFAULT_IMPORTANCE_METHODS)
    evaluators = list(INTERNAL_EVALUATORS or DEFAULT_INTERNAL_EVALUATORS)

    budgets = list(budgets if budgets is not None else BUDGETS)
    n_folds = int(n_folds if n_folds is not None else N_FOLDS)
    resolved = _resolve_models(models if models is not None else MODELS,
                               task_type, n_samples=n_samples,
                               dataset_name=dataset_name, strategy=AUG_STRATEGY)
    model_names = sorted(resolved.keys())

    use_matched = 'match' in evaluators
    explicit = [e for e in evaluators if e != 'match']
    match_map = {'rf': 'rf', 'lightgbm': 'lightgbm', 'xgboost': 'xgboost',
                 'mlp': 'mlp', 'svm': 'svm', 'gp': 'gp', 'ridge': 'ridge',
                 'knn': 'knn', 'catboost': 'catboost', 'dnn': 'dnn',
                 'cnn': 'cnn', 'bnn': 'bnn', 'ngboost': 'ngboost'}

    pairs = []
    for m in model_names:
        evs = list(explicit)
        if use_matched:
            matched = match_map.get(m)
            if matched and matched not in evs:
                evs.append(matched)
        for ev in (evs or ['rf']):
            pairs.append((m, ev))

    n_groups_all = len(importance) * len(pairs) * len(budgets)
    n_groups_default = len(importance) * len(pairs) * len(DEFAULT_SEARCH_BUDGETS)
    return {
        'allocation_methods': allocation,
        'importance_methods': importance,
        'internal_evaluators': evaluators,
        'models': model_names,
        'model_evaluator_pairs': [list(p) for p in pairs],
        'budgets': budgets,
        'default_search_budgets': list(DEFAULT_SEARCH_BUDGETS),
        'n_folds': n_folds,
        'n_model_evaluator_pairs': len(pairs),
        'n_configurations': len(allocation) * n_groups_all,
        'n_groups_all_budgets': n_groups_all,
        'n_groups_default_budgets': n_groups_default,
        'n_configuration_folds': len(allocation) * n_groups_all * n_folds,
    }


def load_one(dataset_name):
    """Representations, labels, groups and the registry's metric for one dataset."""
    from experiments._shared import load_data
    from kirby.registry import require

    info = require(dataset_name)
    data = load_data(dataset_name)
    if data['task_type'] != info.task_type:
        raise ValueError(
            f"{dataset_name}: loader says task_type={data['task_type']!r}, "
            f"registry says {info.task_type!r}")
    reps = data['reps']
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    groups = None if groups is None else np.asarray(groups)
    return reps, labels, groups, info


# =============================================================================
# THE RUN
# =============================================================================

def run_dataset(dataset_name, budgets=None, models=None, n_folds=None,
                out_dir=OUT_DIR, verbose=True, tag=''):
    """One unpruned search. Writes the per-configuration per-fold record."""
    from kirby.find_best_hybrid import find_best_hybrid

    budgets = list(budgets or BUDGETS)
    n_folds = int(n_folds or N_FOLDS)

    t0 = time.time()
    reps, labels, groups, info = load_one(dataset_name)
    print(f"[{datetime.now():%H:%M:%S}] {dataset_name}: {len(labels)} rows, "
          f"{len(reps)} representations, metric={info.metric} "
          f"(lower_is_better={info.lower_is_better}), loaded in "
          f"{time.time() - t0:.0f}s", flush=True)

    grid = resolve_grid(dataset_name, info.task_type, len(labels),
                        models=models if models is not None else MODELS,
                        budgets=budgets, n_folds=n_folds)
    print(f"  {len(grid['allocation_methods'])} allocation methods x "
          f"{len(grid['importance_methods'])} importance methods x "
          f"{len(budgets)} budgets x {grid['n_model_evaluator_pairs']} "
          f"(model, internal evaluator) pairs", flush=True)
    print(f"  models: {grid['models']}", flush=True)

    t_search = time.time()
    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        dataset_name=dataset_name,
        tier=TIER,
        metric=info.metric,
        task_type=info.task_type,
        budgets=budgets,
        models=models if models is not None else MODELS,
        allocation_methods=ALLOCATION_METHODS,
        importance_methods=IMPORTANCE_METHODS,
        internal_evaluators=INTERNAL_EVALUATORS,
        n_folds=n_folds,
        seed=SEED,
        augmentations=None,
        augmentation_strategy=AUG_STRATEGY,
        # The whole point. True is the shipped default and would drop allocation
        # methods after fold 0, which is the thing being measured.
        adaptive_pruning=False,
        two_phase=False,
        cache=True,
        cache_dir=os.environ['KIRBY_CACHE_DIR'],
        verbose=verbose,
    )
    search_seconds = time.time() - t_search

    rows = []
    for r in (result.all_results or []):
        rows.append({
            'allocation_method': r.get('allocation_method'),
            'importance_method': r.get('importance_method'),
            'internal_evaluator': r.get('internal_evaluator'),
            'model': r.get('model'),
            'budget': r.get('budget'),
            # One entry per fold, in fold order. This is the whole record every
            # pruning rule is replayed against.
            'fold_scores': [None if s is None or not np.isfinite(s) else float(s)
                            for s in (r.get('fold_scores') or [])],
            'score_mean': (float(r['score_mean'])
                           if r.get('score_mean') is not None
                           and np.isfinite(r['score_mean']) else None),
            # The WHOLE configuration across all its folds, not one fold.
            'time_seconds': (float(r['time_seconds'])
                             if r.get('time_seconds') is not None else None),
            'n_features': r.get('n_features'),
        })

    payload = {
        'experiment': 'f33_pruning_recalibration',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dataset': dataset_name,
        'n_samples': int(len(labels)),
        'n_representations': len(reps),
        'representations': sorted(reps.keys()),
        'metric': info.metric,
        'lower_is_better': bool(info.lower_is_better),
        'task_type': info.task_type,
        'seed': SEED,
        'adaptive_pruning': False,
        'augmentation_strategy': AUG_STRATEGY,
        'tier': TIER,
        'grid': grid,
        'budgets_run': budgets,
        'n_folds_run': n_folds,
        'search_seconds': search_seconds,
        'results': rows,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{dataset_name}{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"[{datetime.now():%H:%M:%S}] {dataset_name}: {len(rows)} "
          f"configurations in {search_seconds / 3600:.2f} h "
          f"-> {out_path}", flush=True)
    _report_unpruned(payload)
    return payload


def _report_unpruned(payload):
    """Every allocation method present at every budget is the proof pruning was off.

    A method that was pruned after fold 0 would be missing from the budgets it
    never reached, so this is the check that the record is what it claims.
    """
    from collections import defaultdict
    want = set(payload['grid']['allocation_methods'])
    by_budget = defaultdict(set)
    n_partial = 0
    n_folds = payload['n_folds_run']
    for r in payload['results']:
        by_budget[r['budget']].add(r['allocation_method'])
        fs = r['fold_scores']
        if len(fs) != n_folds or any(s is None for s in fs):
            n_partial += 1
    missing = {b: sorted(want - got) for b, got in by_budget.items()
               if want - got}
    if missing:
        print(f"  !! allocation methods absent at some budgets: {missing}")
    else:
        print(f"  all {len(want)} allocation methods present at all "
              f"{len(by_budget)} budgets — pruning was off")
    if n_partial:
        print(f"  {n_partial} of {len(payload['results'])} configurations have "
              f"a missing or non-finite fold score")


# =============================================================================
# --check and --smoke
# =============================================================================

def do_check(datasets):
    """Resolve the grid and the pools without fitting anything."""
    import experiments.dataset_policy  # noqa: F401  registers the skip policy
    from kirby.registry import require

    print("f33 grid check — nothing is fitted here\n")
    ok = True
    for ds in datasets:
        try:
            reps, labels, groups, info = load_one(ds)
        except Exception as e:
            ok = False
            print(f"{ds:24s} LOAD FAILED: {type(e).__name__}: {e}")
            continue
        grid = resolve_grid(ds, info.task_type, len(labels))
        n_groups = grid['n_groups_all_budgets']
        print(f"{ds}")
        print(f"  {len(labels)} rows, {len(reps)} representations, "
              f"metric={info.metric}, task_type={info.task_type}")
        print(f"  models: {grid['models']}")
        print(f"  internal evaluators: {grid['internal_evaluators']} -> "
              f"{grid['n_model_evaluator_pairs']} (model, evaluator) pairs")
        print(f"  allocation: {grid['allocation_methods']}")
        print(f"  importance: {grid['importance_methods']}")
        print(f"  budgets: {grid['budgets']}")
        print(f"  {grid['n_configurations']} configurations x "
              f"{grid['n_folds']} folds = "
              f"{grid['n_configuration_folds']} configuration-folds")
        # What the shipped threshold means at these numbers: each group hands a
        # point to the best strategy_top_n=3 of the allocation methods.
        n_alloc = len(grid['allocation_methods'])
        print(f"  vote over all {len(grid['budgets'])} budgets: {n_groups} groups, "
              f"{n_groups * 3} points over {n_alloc} methods, "
              f"a method is dropped below {n_groups * 0.15:.1f} groups at 0.15")
        print(f"  vote over the {len(grid['default_search_budgets'])} default "
              f"budgets: {grid['n_groups_default_budgets']} groups, "
              f"dropped below {grid['n_groups_default_budgets'] * 0.15:.1f}")
        print()
    return ok


def do_smoke(dataset):
    """One dataset, one budget, one model, two folds — on real data."""
    print(f"f33 smoke on {dataset}: one budget, one model, two folds\n")
    payload = run_dataset(dataset, budgets=[50], models=['lightgbm'],
                          n_folds=2, tag='__smoke')
    rows = payload['results']
    problems = []
    if not rows:
        problems.append('all_results came back empty')
    for r in rows[:5]:
        if len(r['fold_scores']) != 2:
            problems.append(
                f"fold_scores has {len(r['fold_scores'])} entries, expected 2")
            break
    if rows and all(r['time_seconds'] is None for r in rows):
        problems.append('no configuration carries time_seconds')
    want = set(payload['grid']['allocation_methods'])
    got = {r['allocation_method'] for r in rows}
    if want - got:
        problems.append(f"allocation methods missing: {sorted(want - got)}")
    print()
    if problems:
        for p in problems:
            print(f"  SMOKE FAILED: {p}")
        return False
    print(f"  smoke passed: {len(rows)} configurations, "
          f"{len(got)} allocation methods, two fold scores each")
    return True


def main():
    ap = argparse.ArgumentParser(description='f33 pruning recalibration recorder')
    ap.add_argument('--datasets', type=str, default=None,
                    help='comma-separated; default is all nine')
    ap.add_argument('--budgets', type=str, default=None,
                    help='comma-separated override of the budget list')
    ap.add_argument('--folds', type=int, default=None)
    ap.add_argument('--check', action='store_true',
                    help='resolve the grid and the pools, fit nothing')
    ap.add_argument('--smoke', action='store_true',
                    help='one dataset, one budget, one model, two folds')
    ap.add_argument('--smoke-dataset', type=str, default='tcga_brca_pam50')
    ap.add_argument('-o', '--output', type=str, default=str(OUT_DIR))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    import experiments.dataset_policy  # noqa: F401  registers skip + budget policy

    datasets = ([d.strip() for d in args.datasets.split(',') if d.strip()]
                if args.datasets else list(DATASETS))
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        print(f"note: not in the f33 roster, running anyway: {unknown}")

    if args.check:
        sys.exit(0 if do_check(datasets) else 1)

    if args.smoke:
        sys.exit(0 if do_smoke(args.smoke_dataset) else 1)

    budgets = ([int(b) for b in args.budgets.split(',')]
               if args.budgets else None)

    failed = []
    for ds in datasets:
        try:
            run_dataset(ds, budgets=budgets, n_folds=args.folds,
                        out_dir=args.output, verbose=not args.quiet)
        except Exception as e:
            failed.append((ds, f'{type(e).__name__}: {e}'))
            print(f"!! {ds} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    if failed:
        print("\nfailed datasets:")
        for ds, err in failed:
            print(f"  {ds}: {err}")
        sys.exit(1)


if __name__ == '__main__':
    main()
