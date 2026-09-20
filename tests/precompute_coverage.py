#!/usr/bin/env python3
"""Configurations written against configurations the full grid asks for, per pass.

`check_precompute_status.py` answers "has this dataset got at least one row for
this (tier, augmentation strategy)". That was the right question when a
condition was the unit. It is not any more: one pass is thousands of
configurations, and a dataset with 1 of 8,160 reads as done.

This answers three questions instead, straight off `result_db/<dataset>.db`:

  1. how many configurations each pass has stored, against how many the full
     grid asks for
  2. how long a configuration actually took, from the `time_seconds` the engine
     records on every row
  3. therefore how many hours the rest of that pass will take

A PASS is one (tier, objective, augmentation strategy, tuned) group -- the
identity columns that separate one precompute invocation from another, minus
the ones fixed by the dataset (`labels_hash`, `metric`, `task_type`,
`rep_pool`, folds, seed).

HOW THE EXPECTED COUNT IS BUILT. From the engine's own lists, not from a
number written down anywhere:

    allocation methods x importance methods x budgets x representation counts
      x (model, evaluator) pairs

`match` is not an evaluator -- it adds a model's own evaluator when that is not
already in the list (find_best_hybrid.py:5739), so the last term is counted per
model rather than as a product. A representation count at or above the pool
size collapses onto the whole pool, so counts are filtered against the pool
size read off the rows themselves. A non-predictive objective computes no
per-fold allocation scores, so its only count is the whole pool.

⛔ READS ONLY. Opens every database read-only, fits nothing, writes nothing.

Usage:
    export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
    python tests/precompute_coverage.py --trial
    python tests/precompute_coverage.py caco2
    python tests/precompute_coverage.py --all --incomplete-only
"""
import argparse
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

TRIAL_DATASETS = ['buchwald_hartwig', 'caco2', 'flab_expression_koenig',
                  'tcga_brca_pam50', 'logd']


def cache_root():
    env = os.environ.get('KIRBY_CACHE_DIR')
    if env:
        return Path(env), 'KIRBY_CACHE_DIR'
    return Path.home() / '.cache' / 'kirby', 'DEFAULT (no KIRBY_CACHE_DIR)'


def expected_configs(n_pool, model_names, evaluator_names, n_alloc, n_importance,
                     n_budgets, rep_counts):
    """Configurations the full grid asks for. Mirrors find_best_hybrid exactly."""
    explicit = [e for e in evaluator_names if e != 'match']
    use_matched = 'match' in evaluator_names
    pairs = 0
    for m in model_names:
        evs = list(explicit)
        if use_matched and m not in evs:
            evs.append(m)
        pairs += len(evs)
    # A count at or above the pool size is the whole pool, and files once.
    counts = {None if (c is None or c >= n_pool) else c for c in rep_counts}
    return n_alloc * n_importance * n_budgets * len(counts) * pairs, pairs, len(counts)


def scan(db_path):
    """Rows grouped by pass. Read-only; a locked writer never blocks us."""
    uri = f'file:{db_path}?mode=ro'
    con = sqlite3.connect(uri, uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            'SELECT tier, objective, aug_strategy, tune, rep_pool, model, '
            '       internal_evaluator, budget, allocation_method, '
            '       importance_method, rep_count, time_seconds '
            'FROM results').fetchall()
    finally:
        con.close()

    passes = defaultdict(lambda: {
        'n': 0, 'times': [], 'models': set(), 'evaluators': set(),
        'budgets': set(), 'alloc': set(), 'importance': set(),
        'rep_counts': set(), 'rep_pool': None})
    for r in rows:
        key = (r['tier'], r['objective'], r['aug_strategy'], r['tune'] or '')
        p = passes[key]
        p['n'] += 1
        if r['time_seconds'] is not None:
            p['times'].append(float(r['time_seconds']))
        p['models'].add(r['model'])
        p['evaluators'].add(r['internal_evaluator'])
        p['budgets'].add(r['budget'])
        p['alloc'].add(r['allocation_method'])
        p['importance'].add(r['importance_method'])
        p['rep_counts'].add(r['rep_count'])
        p['rep_pool'] = r['rep_pool']
    return passes


def pool_size(rep_pool_text):
    """Representations in the pool, from the stored sorted name/width pairs."""
    if not rep_pool_text:
        return None
    # Stored canonicalised; count names rather than parsing the exact form.
    import re
    names = re.findall(r"'([A-Za-z0-9_]+)'\s*,\s*\d+", rep_pool_text)
    if names:
        return len(set(names))
    return None


def main():
    ap = argparse.ArgumentParser(
        description='Configurations stored against configurations asked for.')
    ap.add_argument('datasets', nargs='*')
    ap.add_argument('--trial', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--incomplete-only', action='store_true',
                    help='Hide passes that are complete')
    args = ap.parse_args()

    root, how = cache_root()
    db_dir = root / 'result_db'
    print("ENVIRONMENT")
    print(f"  cache root      {root}")
    print(f"  set by          {how}")
    print(f"  result_db       {db_dir}  "
          f"({'exists' if db_dir.exists() else 'ABSENT'})")
    if how.startswith('DEFAULT'):
        print("  ⛔ KIRBY_CACHE_DIR is not set. Source setup.sh, or this reads "
              "an empty directory.")
    print()
    if not db_dir.exists():
        print("Nothing has been precomputed under this cache root.")
        return 2

    from kirby.find_best_hybrid import (
        DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS,
        DEFAULT_INTERNAL_EVALUATORS, DEFAULT_REP_COUNTS,
        FAST_ALLOCATION_METHODS, FAST_IMPORTANCE_METHODS,
        FAST_INTERNAL_EVALUATORS, FAST_REP_COUNTS,
    )
    from experiments.dataset_policy import dataset_search_speed
    from experiments._shared import (
        FULL_BUDGETS, DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
        DOWNSTREAM_MODELS, DISTANCE_MODELS,
    )
    from experiments.complexity.complexity_helpers import DATASET_REPS

    if args.all:
        datasets = sorted(DATASET_REPS)
    elif args.trial:
        datasets = list(TRIAL_DATASETS)
    elif args.datasets:
        datasets = args.datasets
    else:
        ap.print_help()
        return 1

    grand_done = grand_want = 0
    grand_remaining_hours = 0.0

    for ds in datasets:
        db = db_dir / f'{ds}.db'
        if not db.exists():
            print(f"{ds}: no result database — nothing precomputed")
            print()
            continue
        try:
            passes = scan(db)
        except sqlite3.Error as e:
            print(f"{ds}: could not read {db.name} — {e}")
            print()
            continue
        if not passes:
            print(f"{ds}: database is empty")
            print()
            continue

        # ⛔ COUNT AGAINST THE GRID THIS DATASET ACTUALLY SEARCHES.
        # Until 2026-09-21 this always counted against the full grid, so the
        # twelve datasets searched at fast could never print above about 10%
        # and looked barely started when they were nearly done. Both the
        # precompute and, since the same day, every experiment search those
        # twelve at speed='fast'.
        _speed = dataset_search_speed(ds)
        _alloc = (FAST_ALLOCATION_METHODS if _speed == 'fast'
                  else DEFAULT_ALLOCATION_METHODS)
        _imp = (FAST_IMPORTANCE_METHODS if _speed == 'fast'
                else DEFAULT_IMPORTANCE_METHODS)
        _evals = (FAST_INTERNAL_EVALUATORS if _speed == 'fast'
                  else DEFAULT_INTERNAL_EVALUATORS)
        _counts_all = FAST_REP_COUNTS if _speed == 'fast' else DEFAULT_REP_COUNTS

        print(f"{ds}" + (f"   [speed=fast: {len(_alloc)} allocation methods, "
                         f"{len(_imp)} importance method"
                         f"{'s' if len(_imp) != 1 else ''}, "
                         f"{len(_counts_all)} representation counts]"
                         if _speed == 'fast' else ""))
        print(f"  {'pass (tier / objective / augmentation / tuned)':52s} "
              f"{'stored':>8s} {'asked':>8s} {'%':>5s} {'s/config':>9s} {'h left':>8s}")
        print(f"  {'-'*52} {'-'*8} {'-'*8} {'-'*5} {'-'*9} {'-'*8}")
        for key in sorted(passes):
            tier, objective, aug, tune = key
            p = passes[key]
            is_distance = tier.startswith('general_distance')
            is_predictive = (objective == 'predictive')
            n_pool = pool_size(p['rep_pool']) or len(
                DATASET_REPS.get(ds, {}).get('general', []) or [10])

            models = DISTANCE_MODELS if is_distance else DOWNSTREAM_MODELS
            evaluators = DISTANCE_EVALUATORS if is_distance else _evals
            budgets = DISTANCE_BUDGETS if is_distance else FULL_BUDGETS
            counts = [None] if not is_predictive else _counts_all

            want, pairs, n_counts = expected_configs(
                n_pool, models, evaluators,
                len(_alloc), len(_imp),
                len(budgets), counts)

            done = p['n']
            pct = 100.0 * done / want if want else 0.0
            med = statistics.median(p['times']) if p['times'] else None
            hours_left = ((want - done) * med / 3600.0
                          if med is not None and want > done else 0.0)

            if args.incomplete_only and done >= want:
                continue
            label = f"{tier} / {objective} / {aug}" + (f" / {tune}" if tune else "")
            print(f"  {label:52.52s} {done:8d} {want:8d} {pct:4.0f}% "
                  f"{(f'{med:.1f}' if med is not None else '—'):>9s} "
                  f"{hours_left:8.1f}")

            # ⛔ SAY WHEN THE MEDIAN IS MEASURED ON A CHEAPER GRID THAN THE ONE
            # STILL TO RUN. Seconds per configuration is not flat across the
            # axes: it rises with the budget, with the representation count and
            # above all with the model. Measured on caco2 2026-09-14, rf only:
            # 2.3s at budget 2000 on 2 representations against 15.2s at the
            # same budget on all ten. svm and dnn are dearer again. So a median
            # taken over rows that cover only some of the models understates
            # what is left, and 'h left' is a floor, not an estimate.
            missing_models = sorted(set(models) - p['models'])
            if missing_models and done < want:
                print(f"      ⛔ the seconds above are measured over "
                      f"{sorted(p['models'])} only. {missing_models} have not "
                      f"run here, so 'h left' is a FLOOR.")
            missing_evals = sorted(set(e for e in evaluators if e != 'match')
                                   - p['evaluators'])
            if missing_evals and done < want:
                print(f"      the same for internal evaluators: "
                      f"{missing_evals} have not run.")

            grand_done += done
            grand_want += want
            grand_remaining_hours += hours_left
        print()

    print("TOTAL")
    print(f"  {grand_done} configurations stored of {grand_want} the full grid asks for"
          f" ({100.0 * grand_done / grand_want if grand_want else 0:.1f}%)")
    print(f"  {grand_remaining_hours:.0f} core-hours left at the median "
          f"seconds-per-configuration measured above")
    print()
    print("  One configuration is one (allocation method, importance method, budget,")
    print("  representation count, model, internal evaluator) scored over all 5 folds.")
    print("  'h left' is single-process core-hours, not wall time on N cores.")
    print()
    print("  ⛔ TWO THINGS 'h left' DOES NOT COUNT, both paid once per pass and")
    print("     both large. Before the first configuration is scored a search")
    print("     fits every representation's own baseline, then the per-fold")
    print("     importance scores, then the per-fold allocation scores -- the")
    print("     last of those once per internal evaluator. Measured on caco2")
    print("     2026-09-14, one allocation method / one importance method / one")
    print("     evaluator / rf: the 16 configurations took 1.5 minutes of the")
    print("     run's 12.7 minutes. The other 11 minutes were that setup.")
    print("     It does not grow with the grid, so it matters most for a SMALL")
    print("     pass and least for a full one -- but a wall has to cover it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
