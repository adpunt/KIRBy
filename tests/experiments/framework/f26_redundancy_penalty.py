#!/usr/bin/env python3
"""F26: does penalising redundancy in the allocation step help, and which
measurement of redundancy?

Paper: Framework. This decides how KIRBy allocates a budget, so it is a
framework experiment, not a noise or similarity one.

THE QUESTION
    `allocate_performance_weighted` splits a feature budget across the base
    representations in proportion to a weight per representation. One weighting
    scheme, `entropy`, subtracts a redundancy penalty: a representation that
    duplicates a higher-scoring one gets less budget. Two things are unknown.

      1. Is the penalty worth having at all, against no penalty?
      2. Redundancy has two candidate measurements, and they disagree. Which
         one, if either, wins?

WHY IT MATTERS MORE FOR A SIMILARITY OBJECTIVE THAN A PREDICTIVE ONE
    A tree can ignore a duplicated feature at no cost. A distance sums over
    every feature it is given, so two near-duplicate representations each
    handed 200 features make the shared direction count twice and tilt the
    metric. The penalty has never been run under a similarity objective. That
    gap is the reason this experiment exists.

THE TWO MEASUREMENTS
    entropy_*   redundancy = |Pearson r| between two representations'
                per-sample row norms ||x_i||. On molecular features the row
                norm is close to a molecule-size proxy. The penalty multiplies
                the representation's SCORE, which `_sanitize_baseline_scores`
                has already shifted to be non-negative.
    pairdist_*  redundancy = |Spearman rho| between two representations'
                z-scored pairwise Euclidean distance vectors over a shared set
                of sampled row pairs -- the quantity a concatenated distance
                actually sums over. The penalty multiplies the representation's
                RANK, which the shift cannot flatten.

    Measured on 8 cached herg_ki representations (2026-08-26,
    tests/test_pairdist_redundancy.py): mean |difference| between the two
    redundancy matrices 0.319, rank agreement rho -0.153. They are not two
    views of the same number.

THE ARMS (11)
    linear              no penalty, score base       -- the reference for entropy_*
    rank_linear         no penalty, rank base        -- the reference for pairdist_*
    entropy_0.5/1.0/2.0        row-norm redundancy, score base
    pairdist_0.5/1.0/2.0       pairdist redundancy, rank base
    pairdist_score_0.5/1.0/2.0 pairdist redundancy, score base

    Both no-penalty arms are in because pairdist_* changes two things against
    entropy_* at once (the redundancy quantity AND the base quantity). Without
    rank_linear a rank-base effect would be read as a redundancy-measurement
    effect. pairdist_score_* isolates it from the other side.

CROSSED WITH
    objective: predictive, similarity (read-across)
    budget:    the full list, [10, 25, 50, 100, 250, 500, 1000, 2000]
    dataset:   one per run. Reported per dataset, never averaged across them.

WHAT IS REPORTED
    delta_mean_of_folds from nested_cv_evaluate -- the hybrid minus the best
    single representation, paired per outer fold, both arms having paid the
    identical held-out tax. Never search_mean, never .all_results.
    Plus the same paired delta per budget, and the two redundancy matrices over
    the dataset's own representations.

SKIPS ARE LOGGED. Any (objective, arm) that fails is written into the output
under 'skipped' with its error. A missing cell never reads as a cell that ran
and produced nothing.

WALL TIME. Each (objective, arm) is one nested_cv_evaluate: 5 outer folds x
8 budgets x 3 importance methods x 5 internal evaluators x the model roster.
Eleven arms is about 1.2x the config count of ONE default search (the default
allocation list has 9 entries), and there are 3 objectives, so a full dataset
is roughly 3.7 default nested-CV runs. That is a SERVER job --
tests/slurm_scripts/run_f26_redundancy_penalty.sh, one array task per dataset.
Estimated 12-40 hours per dataset depending on n and representation widths;
that is a structural estimate from the config count, NOT a measured time.
Do not run the full grid locally.

Usage:
    # server, one dataset (this is what the SLURM array task calls)
    python tests/experiments/framework/f26_redundancy_penalty.py --dataset herg_ki

    # local plumbing check -- cheapest possible, proves the arms dispatch
    python tests/experiments/framework/f26_redundancy_penalty.py \\
        --dataset logd --smoke

    # redundancy matrices only, no search at all (seconds)
    python tests/experiments/framework/f26_redundancy_penalty.py \\
        --dataset herg_ki --redundancy-only
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))     # tests/
sys.path.insert(0, os.path.join(HERE, '..'))           # tests/experiments/

TIER = 'f26_redund'          # own cache identity; touches no existing result DB

# (arm name, allocation method registered in NAMED_ALLOCATIONS, what it is)
ARMS = [
    ('linear',             'performance_weighted', 'no penalty, score base'),
    ('rank_linear',        'rank_linear',          'no penalty, rank base'),
    ('entropy_0.5',        'entropy_0.5',          'row-norm redundancy, score base'),
    ('entropy_1.0',        'entropy_1.0',          'row-norm redundancy, score base'),
    ('entropy_2.0',        'entropy_2.0',          'row-norm redundancy, score base'),
    ('pairdist_0.5',       'pairdist_0.5',         'pairdist redundancy, rank base'),
    ('pairdist_1.0',       'pairdist_1.0',         'pairdist redundancy, rank base'),
    ('pairdist_2.0',       'pairdist_2.0',         'pairdist redundancy, rank base'),
    ('pairdist_score_0.5', 'pairdist_score_0.5',   'pairdist redundancy, score base'),
    ('pairdist_score_1.0', 'pairdist_score_1.0',   'pairdist redundancy, score base'),
    ('pairdist_score_2.0', 'pairdist_score_2.0',   'pairdist redundancy, score base'),
]

# similarity_sac was CUT by the owner on 2026-08-26 and now raises. The two
# live objectives are a fitted model's held-out score, and activity read-across
# on the held-out rows. Under read-across the per-representation score that this
# penalty multiplies is the read-across rank -- until 2026-08-26 it was sac,
# which was measured to be the worst of the four available scores.
OBJECTIVES = ['predictive', 'similarity']

# The no-penalty reference each arm is read against.
REFERENCE_ARM = {
    'entropy_0.5': 'linear', 'entropy_1.0': 'linear', 'entropy_2.0': 'linear',
    'pairdist_0.5': 'rank_linear', 'pairdist_1.0': 'rank_linear',
    'pairdist_2.0': 'rank_linear',
    'pairdist_score_0.5': 'linear', 'pairdist_score_1.0': 'linear',
    'pairdist_score_2.0': 'linear',
}


# ------------------------------------------------------------------ redundancy
def redundancy_report(reps, n_pairs=5000, seed=42):
    """Both redundancy matrices over the dataset's own representations.

    Read-only, no search, no model fit. This is the direct evidence for
    'the two measurements disagree', separate from any allocation outcome.
    """
    from kirby.hybrid import redundancy_between_reps
    out = {}
    for measure in ('rownorm', 'pairdist'):
        try:
            r = redundancy_between_reps(reps, measure=measure, n_pairs=n_pairs,
                                        seed=seed)
        except Exception as exc:
            out[measure] = {'error': repr(exc)}
            continue
        if r is None:
            out[measure] = {'error': 'too few rows to sample pairs'}
            continue
        out[measure] = {'names': r['names'],
                        'matrix': [[float(v) for v in row] for row in r['matrix']]}

    both = out.get('rownorm'), out.get('pairdist')
    if all(isinstance(b, dict) and 'matrix' in b for b in both):
        from scipy.stats import spearmanr
        A = np.asarray(both[0]['matrix'])
        B = np.asarray(both[1]['matrix'])
        iu = np.triu_indices(A.shape[0], k=1)
        a, b = A[iu], B[iu]
        rho, _ = spearmanr(a, b)
        out['comparison'] = {
            'n_pairs_of_reps': int(len(a)),
            'rownorm_mean': float(a.mean()), 'rownorm_median': float(np.median(a)),
            'rownorm_min': float(a.min()), 'rownorm_max': float(a.max()),
            'pairdist_mean': float(b.mean()), 'pairdist_median': float(np.median(b)),
            'pairdist_min': float(b.min()), 'pairdist_max': float(b.max()),
            'mean_abs_difference': float(np.abs(a - b).mean()),
            'rank_agreement_spearman': (float(rho) if np.isfinite(rho) else None),
        }
    return out


# ------------------------------------------------------------------- one arm
def _paired_delta(hybrid_folds, baseline_folds, lower_is_better):
    """Mean of the per-fold (hybrid - best single rep) differences.

    The same arithmetic nested_cv_evaluate uses for delta_mean_of_folds, reused
    here to get the paired delta at ONE budget, which it does not report.
    """
    per_fold, paired = [], []
    n = max(len(hybrid_folds or []), len(baseline_folds or []))
    for f in range(n):
        hv = hybrid_folds[f] if hybrid_folds and f < len(hybrid_folds) else None
        bv = baseline_folds[f] if baseline_folds and f < len(baseline_folds) else None
        if hv is None or bv is None or not np.isfinite(hv) or not np.isfinite(bv):
            per_fold.append(None)
            continue
        d = (bv - hv) if lower_is_better else (hv - bv)
        per_fold.append(float(d))
        paired.append(float(d))
    return (float(np.mean(paired)) if paired else None), per_fold, len(paired)


def summarise_arm(result):
    """Pull delta_mean_of_folds per model, and the paired delta per budget."""
    lower = bool(result.lower_is_better)
    out = {'metric_name': result.metric_name,
           'lower_is_better': lower,
           'n_outer_folds': int(result.n_outer_folds),
           'by_model': {}, 'by_model_budget': {}}

    for model, e in (result.by_model or {}).items():
        out['by_model'][model] = {
            # THE number. Paired per outer fold, both arms held out.
            'delta_mean_of_folds': (None if not np.isfinite(e['delta_mean_of_folds'])
                                    else float(e['delta_mean_of_folds'])),
            'n_paired_folds': int(e['n_paired_folds']),
            'hybrid_mean': (None if not np.isfinite(e['hybrid_mean'])
                            else float(e['hybrid_mean'])),
            'best_single_mean': (None if not np.isfinite(e['best_single_mean'])
                                 else float(e['best_single_mean'])),
            'best_single_reps_per_fold': e.get('best_single_reps_per_fold'),
            'per_fold_delta': e.get('per_fold_delta'),
        }

    for model, per_budget in (result.by_model_budget or {}).items():
        base_folds = ((result.by_model or {}).get(model, {})
                      .get('best_single_fold_scores'))
        out['by_model_budget'][model] = {}
        for budget, b in per_budget.items():
            d, per_fold, n_paired = _paired_delta(b.get('fold_scores'),
                                                  base_folds, lower)
            out['by_model_budget'][model][str(budget)] = {
                'delta_mean_of_folds': d,
                'n_paired_folds': n_paired,
                'per_fold_delta': per_fold,
                'hybrid_honest_mean': (None if not np.isfinite(b['honest_mean'])
                                       else float(b['honest_mean'])),
            }
    return out


# ---------------------------------------------------------------------- main
def _result_path(args, dataset):
    """The file this dataset's result is written to. One place, so the
    end-of-run write and the after-every-cell write can never disagree."""
    suffix = ('__redundancy_only' if args.redundancy_only
              else ('__smoke' if args.smoke else ''))
    return os.path.join(args.output_dir,
                        f'f26_redundancy_penalty_{dataset}{suffix}.json')


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_result(out, path):
    """Write to a temporary file in the same directory, then rename over the
    target. A rename is atomic on one filesystem, so a job killed mid-write
    leaves the previous complete file rather than a truncated one."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.partial'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _make_flusher(args):
    """Rewrite the whole result after every (objective, arm) cell.

    out['progress'] records how many of the planned cells have been attempted,
    which of them are stored, and when the last write happened, so a partial
    file is never mistaken for a finished run.
    """
    path = _result_path(args, args.dataset)

    def flush(out, objective, arm, state, t0):
        prog = out.setdefault('progress', {
            'cells_planned': len(out['objectives']) * len(out['arms']),
            'cells_attempted': 0,
            'complete': False,
        })
        if objective is not None:
            prog['cells_attempted'] += 1
            prog['last_cell'] = {'objective': objective, 'arm': arm,
                                 'state': state}
        prog['elapsed_seconds'] = round(time.time() - t0, 1)
        out['elapsed_seconds'] = prog['elapsed_seconds']
        try:
            _write_result(out, path)
        except Exception as exc:
            # A failed write must not lose the compute that produced the cell.
            print(f'  [flush] could not write {path}: {exc!r}', flush=True)
            return
        if objective is not None:
            print(f"  [flush] {prog['cells_attempted']}/"
                  f"{prog['cells_planned']} cells -> {os.path.basename(path)}",
                  flush=True)

    return flush


def run_dataset(ds, args):
    from experiments._shared import (load_data, run_nested_cv, get_config,
                                     FULL_BUDGETS, parse_rep_args)

    config = get_config(args)
    if args.smoke and args.reps == 'full' and not args.framework:
        # Plumbing only: three cheap representations, one budget, one objective.
        args.reps = 'ecfp4,maccs,avalon'
    rep_preset, rep_names = parse_rep_args(args)

    data = load_data(ds, rep_preset, rep_names=rep_names)
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')

    budgets = ([int(b) for b in args.budgets.split(',')] if args.budgets
               else ([25] if args.smoke else list(FULL_BUDGETS)))
    objectives = ([o.strip() for o in args.objectives.split(',')] if args.objectives
                  else (['predictive'] if args.smoke else list(OBJECTIVES)))
    arms = ([a for a in ARMS if a[0] in
             {x.strip() for x in args.arms.split(',')}] if args.arms
            else (ARMS[:4] if args.smoke else ARMS))

    models = ([m.strip() for m in args.models.split(',')] if args.models
              else None)
    importance = ([i.strip() for i in args.importance.split(',')]
                  if args.importance else None)

    out = {
        'experiment': 'f26_redundancy_penalty',
        'dataset': ds,
        'tier': TIER,
        'n_samples': int(len(labels)),
        'reps': {k: list(v.shape) for k, v in sorted(reps.items())},
        'budgets': budgets,
        'objectives': objectives,
        'arms': [{'name': a, 'allocation_method': m, 'is': d} for a, m, d in arms],
        'reference_arm': REFERENCE_ARM,
        'smoke': bool(args.smoke),
        'models': models,
        'importance_methods': importance,
        'n_outer_folds_requested': args.outer_folds or config['n_splits'],
        'redundancy': redundancy_report(reps),
        'results': {},
        'skipped': [],
    }

    if args.redundancy_only:
        return out

    # Each (objective, arm) is one nested_cv_evaluate and takes hours. The file
    # used to be written once, after all 22 of them, so a job that died at hour
    # 40 of its 47-hour ceiling left nothing on disk. It is now rewritten after
    # every cell, complete or skipped, and 'progress' says how far it got.
    _flush = _make_flusher(args)

    t0 = time.time()
    for objective in objectives:
        out['results'][objective] = {}
        for arm_name, alloc, _desc in arms:
            print(f'\n=== {ds} | objective={objective} | arm={arm_name} '
                  f'({time.time() - t0:.0f}s elapsed) ===', flush=True)
            try:
                result = run_nested_cv(
                    reps=reps, labels=labels, groups=groups, config=config,
                    dataset_name=ds,
                    task_type=data.get('task_type', 'auto'),
                    domain=data.get('domain'),
                    tier=TIER,
                    budgets=budgets,
                    allocation_methods=[alloc],
                    objective=objective,
                    n_outer_folds=args.outer_folds,
                    **({'models': models} if models else {}),
                    **({'importance_methods': importance} if importance else {}),
                    # One search cache file per arm. allocation_methods is in
                    # the search META but not in the search FILENAME, so
                    # without this every arm writes the same file and evicts
                    # the previous one -- correct results, all recomputed cold.
                    #
                    # condition_suffix, NOT cache_namespace. cache_namespace
                    # renames the DATASET part of every cache path, which also
                    # renames the per-(representation, model) baseline files;
                    # each arm would then refit every baseline from scratch,
                    # 11 times over. The baseline score of a single
                    # representation does not depend on the allocation method,
                    # so all 11 arms should share it. condition_suffix lands in
                    # the search filename only.
                    condition_suffix=f'alloc-{arm_name}',
                    verbose=True,
                )
                out['results'][objective][arm_name] = summarise_arm(result)
                _flush(out, objective, arm_name, 'done', t0)
            except Exception as exc:
                # A missing cell must not read as a cell that ran and found
                # nothing. Write the failure down.
                out['skipped'].append({
                    'objective': objective, 'arm': arm_name,
                    'allocation_method': alloc,
                    'error': repr(exc),
                    'traceback': traceback.format_exc(limit=8),
                })
                print(f'  SKIPPED {objective}/{arm_name}: {exc!r}', flush=True)
                _flush(out, objective, arm_name, 'skipped', t0)

    out['elapsed_seconds'] = round(time.time() - t0, 1)
    out['progress']['complete'] = True
    _flush(out, None, None, 'complete', t0)
    return out


def print_report(out):
    ds = out['dataset']
    print('\n' + '=' * 88)
    print(f'F26 REDUNDANCY PENALTY -- {ds}   (n={out["n_samples"]}, '
          f'{len(out["reps"])} representations)')
    print('=' * 88)

    c = (out.get('redundancy') or {}).get('comparison')
    if c:
        print('\nRedundancy between representations, both measurements, '
              f'{c["n_pairs_of_reps"]} pairs:')
        print(f'  row-norm (entropy_*):  mean {c["rownorm_mean"]:.3f}  '
              f'median {c["rownorm_median"]:.3f}  '
              f'range {c["rownorm_min"]:.3f}-{c["rownorm_max"]:.3f}')
        print(f'  pairdist (pairdist_*): mean {c["pairdist_mean"]:.3f}  '
              f'median {c["pairdist_median"]:.3f}  '
              f'range {c["pairdist_min"]:.3f}-{c["pairdist_max"]:.3f}')
        print(f'  mean |difference| {c["mean_abs_difference"]:.3f}; '
              f'rank agreement rho {c["rank_agreement_spearman"]}')

    for objective, arms in (out.get('results') or {}).items():
        print(f'\n--- objective = {objective} ---')
        models = sorted({m for a in arms.values() for m in a.get('by_model', {})})
        for model in models:
            metric = next((a['metric_name'] for a in arms.values()
                           if a.get('by_model', {}).get(model)), '?')
            print(f'\n  model = {model}   (metric {metric}; '
                  f'delta = hybrid minus best single rep, + is better)')
            print(f'    {"arm":22s} {"delta":>9s} {"folds":>6s}  reference')
            for arm_name, _alloc, _d in ARMS:
                a = arms.get(arm_name)
                if not a or model not in a.get('by_model', {}):
                    continue
                e = a['by_model'][model]
                d = e['delta_mean_of_folds']
                ref = REFERENCE_ARM.get(arm_name, '--')
                ref_e = (arms.get(ref, {}).get('by_model', {}).get(model)
                         if ref != '--' else None)
                rd = ref_e['delta_mean_of_folds'] if ref_e else None
                vs = ('' if rd is None or d is None
                      else f'  vs {ref}: {d - rd:+.4f}')
                ds_str = f'{d:+9.4f}' if d is not None else f'{"n/a":>9s}'
                print(f'    {arm_name:22s} {ds_str} {e["n_paired_folds"]:>6d}  '
                      f'{ref}{vs}')

    if out.get('skipped'):
        print(f'\nSKIPPED CELLS ({len(out["skipped"])}):')
        for s in out['skipped']:
            print(f'  {s["objective"]}/{s["arm"]}: {s["error"]}')
    else:
        print('\nNo cells skipped.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--reps', default='full',
                    help="rep preset name, or a comma-separated pool override")
    ap.add_argument('--framework', action='store_true',
                    help='use the curated FRAMEWORK_TASKS pool for this dataset')
    ap.add_argument('--mode', default='full', choices=['fast', 'full',
                                                       'intermediary'])
    ap.add_argument('--budgets', default=None,
                    help='comma-separated; default the full list')
    ap.add_argument('--objectives', default=None,
                    help=f'comma-separated; default {",".join(OBJECTIVES)}')
    ap.add_argument('--arms', default=None,
                    help='comma-separated arm names; default all 11')
    ap.add_argument('--redundancy-only', action='store_true',
                    dest='redundancy_only',
                    help='report both redundancy matrices and stop. No search.')
    ap.add_argument('--models', default=None,
                    help='comma-separated downstream models. Default None, '
                         'which is the roster the SLURM precompute caches under '
                         '-- change it only for a plumbing check.')
    ap.add_argument('--outer-folds', type=int, default=None, dest='outer_folds',
                    help='outer CV folds. Default the config n_splits (5).')
    ap.add_argument('--importance', default=None,
                    help='comma-separated importance methods. Default all three.')
    ap.add_argument('--smoke', action='store_true',
                    help='plumbing check: 3 cheap reps, 1 budget, 1 objective, '
                         '4 arms. Proves dispatch, measures nothing.')
    ap.add_argument('--output-dir', dest='output_dir',
                    default=os.path.join(HERE, '..', 'results'))
    args = ap.parse_args()

    out = run_dataset(args.dataset, args)
    print_report(out)

    path = _result_path(args, args.dataset)
    _write_result(out, path)
    print(f'\nwrote {path}')
    return out


if __name__ == '__main__':
    main()
