#!/usr/bin/env python
"""Check EXPERIMENT status (distinct from precompute status).

check_precompute_status.py measures the per-dataset SQLite result store — i.e.
whether the cached hybrid SEARCHES (general/top/distance tiers) a dataset's
experiments depend on are built. It does NOT measure whether the experiments
themselves have run.

Each per-dataset job (run_dataset.py CATEGORY DATASET) does two things in order:
  1. build precompute (the searches)  <- what check_precompute_status sees
  2. run the experiments and save each to
        tests/results/<category>/<exp_id>/<dataset>.json   <- what THIS sees

So "precompute DONE" can be true while zero experiments have run (the common
case when precompute ate the whole wall-clock and the job died before step 2).
This tool counts the experiment JSONs that exist, per dataset per category.

A present JSON means the runner reached that experiment (it ran, or recorded a
deliberate skip). A missing JSON means the job never got there. Some absences
are legitimate (clf-only datasets skip some experiments, n-gates, etc.), so
treat the ratio as "experiments reached", not a hard completeness contract.

Usage:
    python tests/check_experiment_status.py                 # all datasets
    python tests/check_experiment_status.py --dataset herg  # one dataset
    python tests/check_experiment_status.py --results-dir tests/results
    python tests/check_experiment_status.py --missing       # only not-fully-run
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

# Category -> results subdir (mirrors run_dataset.py's default_output tails).
CATEGORIES = ('complexity', 'similarity', 'noise')


def _experiment_ids(category):
    """The experiment IDs a category runs, from its helper (source of truth)."""
    import importlib
    mod = importlib.import_module(f'experiments.{category}.{category}_helpers')
    return list(mod.SINGLE_DATASET_EXPERIMENTS)


def _domain(name):
    if name.startswith(('tcga_', 'luad', 'brca')):
        return 'tcga'
    if name.startswith(('davis', 'kiba', 'bindingdb')):
        return 'dta'
    if name.startswith(('buchwald', 'doyle', 'suzuki', 'bh_')):
        return 'reaction'
    if any(name.startswith(p) for p in (
            'flab_', 'abagym_', 'abcov_', 'ddg_', 'tz_', 'humanness',
            'ginkgo_', 'graphinity_')):
        return 'antibody'
    if name.startswith(('nt_', 'gb_')):
        return 'nucleotide'
    return 'molecular'


def scan_dataset(dataset, results_dir, exp_ids_by_cat):
    """Return {category: (n_present, n_expected, [missing_ids])} for a dataset."""
    out = {}
    for cat in CATEGORIES:
        ids = exp_ids_by_cat[cat]
        present, missing = [], []
        for eid in ids:
            # save_experiment_results writes <results>/<cat>/<exp_dir>/<ds>.json.
            # The exp_dir is the experiment_id passed by run_all; it begins with
            # the short id (e.g. 'c02' -> 'c02_complexity_grid'), so match by
            # prefix to stay robust to the suffix.
            cat_root = Path(results_dir) / cat
            hit = False
            if cat_root.is_dir():
                for exp_dir in cat_root.iterdir():
                    if exp_dir.is_dir() and exp_dir.name.split('_')[0] == eid:
                        if (exp_dir / f'{dataset}.json').exists():
                            hit = True
                            break
            (present if hit else missing).append(eid)
        out[cat] = (len(present), len(ids), missing)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', help='check a single dataset')
    ap.add_argument('--results-dir', default='tests/results',
                    help='base results dir (default: tests/results)')
    ap.add_argument('--missing', action='store_true',
                    help='show only datasets with any experiment not yet run')
    args = ap.parse_args()

    from experiments.complexity.complexity_helpers import DATASET_REPS
    exp_ids_by_cat = {cat: _experiment_ids(cat) for cat in CATEGORIES}
    totals = {cat: len(ids) for cat, ids in exp_ids_by_cat.items()}

    datasets = ([args.dataset] if args.dataset
                else sorted(DATASET_REPS.keys()))
    if args.dataset and args.dataset not in DATASET_REPS:
        print(f"ERROR: '{args.dataset}' not in DATASET_REPS")
        sys.exit(1)

    by_domain = {}
    for ds in datasets:
        by_domain.setdefault(_domain(ds), []).append(ds)

    grand = {cat: [0, 0] for cat in CATEGORIES}  # [present, expected]
    n_full = n_none = n_partial = 0

    for domain in sorted(by_domain):
        rows = []
        for ds in by_domain[domain]:
            scan = scan_dataset(ds, args.results_dir, exp_ids_by_cat)
            tot_present = sum(scan[c][0] for c in CATEGORIES)
            tot_expected = sum(scan[c][1] for c in CATEGORIES)
            if tot_present == 0:
                state = 'none'
                n_none += 1
            elif tot_present == tot_expected:
                state = 'FULL'
                n_full += 1
            else:
                state = 'partial'
                n_partial += 1
            for c in CATEGORIES:
                grand[c][0] += scan[c][0]
                grand[c][1] += scan[c][1]
            if args.missing and state == 'FULL':
                continue
            rows.append((ds, state, scan))

        if not rows:
            continue
        print(f"\n=== {domain.upper()} ===")
        print(f"  {'dataset':<36} {'state':<8} "
              f"{'cx':<7} {'sim':<7} {'noise':<7}")
        print(f"  {'-'*36} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        for ds, state, scan in rows:
            cx = f"{scan['complexity'][0]}/{scan['complexity'][1]}"
            sim = f"{scan['similarity'][0]}/{scan['similarity'][1]}"
            noi = f"{scan['noise'][0]}/{scan['noise'][1]}"
            print(f"  {ds:<36} {state:<8} {cx:<7} {sim:<7} {noi:<7}")

    total_ds = n_full + n_partial + n_none
    print(f"\n{'='*64}")
    print(f"Datasets: {total_ds}  |  fully run: {n_full}  |  "
          f"partial: {n_partial}  |  not started: {n_none}")
    for cat in CATEGORIES:
        p, e = grand[cat]
        print(f"  {cat:<11}: {p}/{e} experiment-results present "
              f"({totals[cat]} per dataset)")


if __name__ == '__main__':
    main()
