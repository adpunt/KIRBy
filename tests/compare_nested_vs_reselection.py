#!/usr/bin/env python
"""Nested CV at the fast setting against per-fold reselection, on one dataset.

THE QUESTION. Nineteen experiments report a hybrid-minus-baseline number built
by per-fold reselection: the search runs once over every molecule, and the
winner is then re-picked for each held-out fold from the per-fold scores that
search already produced. Nothing is refitted. It is cheap, and the scores it
re-picks over came out of a search that had already seen every fold, for
pruning and for candidate selection.

Five experiments instead run nested cross-validation: one search per outer
fold, on that fold's training molecules only, with both the hybrid and every
single representation then scored on the molecules that fold never saw. That is
the number the reselection one is standing in for.

Nobody has measured how far apart the two are. This does that, on one dataset,
for every downstream model, and prints the pair.

WHAT IT COSTS. The nested arm runs at `speed='fast'` -- 3 allocation methods x 1
importance method x 2 representation counts x 8 budgets x 9 (model, evaluator)
pairs = 432 configurations, against 8,160 at the full setting. Five outer folds
x 432 configurations x 5 inner folds is about 10,800 model fits per dataset.
The reselection arm reads a stored search and fits nothing.

⛔ THE TWO ARMS DO NOT SEARCH THE SAME GRID. The nested arm is fast by design,
because a full-grid nested run is about 204,000 model fits per dataset and does
not fit a 4-hour wall. So a difference between the two numbers is the two
things TOGETHER: holding data out, and searching a shorter list. This run
cannot separate them. It answers "is the shortcut close enough to the real
thing, at a cost I can afford", which is the decision on the table. It does not
answer "how much does holding data out cost on its own".

USAGE
    python tests/compare_nested_vs_reselection.py herg_ki
    python tests/compare_nested_vs_reselection.py caco2 --n-outer-folds 5

On the cluster, inside a 4-hour allocation:
    srun --time=04:00:00 --cpus-per-task=8 --mem=32G \
        python tests/compare_nested_vs_reselection.py herg_ki
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from experiments._shared import (
    load_data, get_config, save_experiment_results, convert_numpy,
    run_nested_cv, get_best_search_for_downstream,
    build_fold_reselection_input, fold_reselection, hr_exclude_folds,
    DOWNSTREAM_MODELS,
)


def _fmt(v, width=9):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'-':>{width}}"
    return f"{v:>{width}.4f}"


def run(dataset_name, args):
    # ⛔ args.reps is a comma-separated NAME LIST, not a preset. Until
    # 2026-09-19 it was passed as load_data's second positional argument, which
    # is rep_preset, and was never split -- so --reps silently loaded the full
    # 10-representation pool and the flag did nothing.
    _rep_names = ([r.strip() for r in args.reps.split(',') if r.strip()]
                  if args.reps else None)
    data = load_data(dataset_name, rep_names=_rep_names)
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    task_type = data['task_type']

    print(f"\n{'=' * 74}")
    print(f"  {dataset_name}: {len(labels)} molecules, "
          f"{len(reps)} representations, {args.n_outer_folds} outer folds")
    print(f"{'=' * 74}")

    out = {
        'dataset': dataset_name,
        'n_molecules': int(len(labels)),
        'n_representations': len(reps),
        'n_outer_folds': args.n_outer_folds,
        'nested_speed': 'fast',
        # Stated in the file so a reader of the JSON does not have to find this
        # docstring: the nested arm searched a shorter list than the stored
        # search the reselection arm reads, so the gap below mixes two changes.
        'caveat': ('the nested arm ran at speed=fast (432 configurations) and '
                   'the reselection arm reads a stored search at the full '
                   'setting (8,160). A difference is holding data out AND the '
                   'shorter list together, not either alone.'),
    }

    # ---- arm 1: nested cross-validation, fast inner search -----------------
    print("\n  [1/2] nested CV, one search per outer fold, speed=fast")
    t0 = time.time()
    nested = run_nested_cv(
        reps, labels, groups,
        config={'models': DOWNSTREAM_MODELS},
        dataset_name=dataset_name,
        augmentations=augmentations,
        task_type=task_type,
        n_outer_folds=args.n_outer_folds,
        speed='fast',
        verbose=args.verbose,
    )
    nested_seconds = time.time() - t0
    print(f"        {nested_seconds / 60:.1f} minutes")
    out['nested_seconds'] = round(nested_seconds, 1)
    out['nested'] = {
        m: {'hybrid_mean': e.get('hybrid_mean'),
            'best_single_mean': e.get('best_single_mean'),
            'delta_mean_of_folds': e.get('delta_mean_of_folds'),
            'n_paired_folds': e.get('n_paired_folds'),
            'best_single_reps_per_fold': e.get('best_single_reps_per_fold')}
        for m, e in (nested.by_model or {}).items()
    }

    # ⛔ WRITE ARM 1 BEFORE ARM 2 RUNS. Added 2026-09-19 after a run on ARC
    # spent 138.6 minutes on the nested arm, hit the wall in arm 2, and wrote
    # nothing at all: the file was dumped once at the end, so a failure in the
    # cheap arm threw away the expensive one.
    _partial = os.path.join(args.output_dir, f'{dataset_name}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    out['status'] = 'nested arm only — arm 2 has not run yet'
    with open(_partial, 'w') as _f:
        json.dump(convert_numpy(out), _f, indent=1, default=str)
    print(f"        nested arm saved to {_partial}")

    # ---- arm 2: per-fold reselection over one whole-dataset search ---------
    print("\n  [2/2] per-fold reselection over one stored whole-dataset search")
    t0 = time.time()
    # ⛔ ARM 2 READS A STORED SEARCH. IT MUST NOT BUILD ONE.
    # When nothing is stored, get_best_search_for_downstream falls through to a
    # live find_best_hybrid at FULL speed — 8,160 configurations — which is not
    # the quantity this arm is supposed to measure and does not finish inside a
    # 4-hour wall. That is exactly what happened on ARC on 2026-09-19, on a
    # dataset whose result database had just been deleted for the pdv rebuild.
    # KIRBY_NO_COLD_SEARCH makes the engine refuse instead.
    _prev_cold = os.environ.get('KIRBY_NO_COLD_SEARCH')
    os.environ['KIRBY_NO_COLD_SEARCH'] = '0.5'
    search = meta = None
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            data=data,
            dataset_name=dataset_name,
            task_type=task_type,
        )
    except Exception as _exc:
        print(f"        no stored search to read — {type(_exc).__name__}: "
              f"{str(_exc)[:160]}")
        out['reselection_skipped'] = (
            f'{type(_exc).__name__}: {str(_exc)[:400]}. Arm 2 reads a stored '
            f'search and refuses to build one. Fill this dataset\'s result '
            f'database first, then re-run.')
    finally:
        if _prev_cold is None:
            os.environ.pop('KIRBY_NO_COLD_SEARCH', None)
        else:
            os.environ['KIRBY_NO_COLD_SEARCH'] = _prev_cold

    resel = {}
    if search is None:
        print("        reselection arm skipped — nothing stored to read")
        # Do NOT overwrite a reason the except block already wrote: that one
        # names the engine's refusal and how far short the stored rows fell,
        # and 'no stored search' says less.
        out.setdefault('reselection_skipped',
                       'no stored search for this dataset')
    else:
        block = build_fold_reselection_input(search)
        resel = fold_reselection(
            block['candidates_by_model'],
            block['baselines_by_model'],
            block['lower_is_better'],
            metric_name=block.get('metric_name'),
            exclude_folds=hr_exclude_folds(block),
        )
        out['reselection'] = {
            m: {'hybrid_mean': e.get('hybrid_reselected'),
                'best_single_mean': e.get('best_single_reselected'),
                'delta_mean_of_folds': e.get('delta_mean_of_folds'),
                'n_paired_folds': e.get('n_paired_folds')}
            for m, e in resel.items()
        }
    resel_seconds = time.time() - t0
    print(f"        {resel_seconds:.1f} seconds")
    out['reselection_seconds'] = round(resel_seconds, 1)

    # ---- the comparison ----------------------------------------------------
    metric = nested.metric_name
    lower = nested.lower_is_better
    print(f"\n  hybrid minus best single representation, in {metric} "
          f"({'lower' if lower else 'higher'} is better)")
    print(f"  one row is one downstream model; both columns are the same "
          f"quantity, measured two ways\n")
    print(f"    {'model':<10} {'nested':>9} {'reselection':>12} "
          f"{'gap':>9}  {'same sign?':>11}")

    rows = {}
    for m in sorted(set(out['nested']) | set(out.get('reselection', {}))):
        n_d = (out['nested'].get(m) or {}).get('delta_mean_of_folds')
        r_d = (out.get('reselection', {}).get(m) or {}).get('delta_mean_of_folds')
        gap = (n_d - r_d) if (n_d is not None and r_d is not None) else None
        same = (None if gap is None
                else bool((n_d > 0) == (r_d > 0)))
        rows[m] = {'nested_delta': n_d, 'reselection_delta': r_d,
                   'gap': gap, 'same_sign': same}
        print(f"    {m:<10} {_fmt(n_d)} {_fmt(r_d, 12)} {_fmt(gap)}  "
              f"{('yes' if same else 'NO') if same is not None else '-':>11}")
    out['comparison'] = rows

    _gaps = [abs(r['gap']) for r in rows.values() if r['gap'] is not None]
    _flips = [m for m, r in rows.items() if r['same_sign'] is False]
    if _gaps:
        print(f"\n    largest gap over {len(_gaps)} downstream models: "
              f"{max(_gaps):.4f} {metric}")
        out['max_abs_gap'] = float(max(_gaps))
    if _flips:
        print(f"    ⛔ the two disagree on whether the hybrid won for: "
              f"{', '.join(sorted(_flips))}")
    elif _gaps:
        print(f"    the two agree on the sign for every downstream model")
    out['models_disagreeing_on_sign'] = sorted(_flips)
    print(f"\n    nested arm {nested_seconds / 60:.1f} minutes, "
          f"reselection arm {resel_seconds:.1f} seconds")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--reps', default=None)
    ap.add_argument('--n-outer-folds', type=int, default=5)
    ap.add_argument('--output-dir', default='tests/results/nested_vs_reselection')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    result = run(args.dataset, args)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f'{args.dataset}.json')
    with open(path, 'w') as f:
        json.dump(convert_numpy(result), f, indent=1, default=str)
    print(f"\n  wrote {path}")
