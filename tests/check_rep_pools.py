#!/usr/bin/env python3
"""Are a dataset's representations on disk, before a precompute job starts?

Pure file existence — no representation is built, no dataset is loaded, no model
is fitted. Safe on a login node.

WHY THIS EXISTS. `precompute_hybrids.py` refuses a dataset whose pool did not
fully load (find_best_hybrid's rep_pool is part of the result database's
identity, so a search over 7 of 10 representations is a different search that no
experiment will ever read). That refusal happens AFTER the job has queued and
loaded. This says the same thing in two seconds, before submission.

⛔ THE ENVIRONMENT IS THE FIRST THING THIS PRINTS, AND IT IS NOT DECORATION.
The representation cache lives under $KIRBY_CACHE_DIR, which `setup.sh` sets to
/data/stat-ecr/scat9264/.cache/kirby on ARC. Run this without sourcing setup.sh
and the variable is unset, the path falls back to ~/.cache/kirby, every file is
absent, and the report reads "10 of 10 missing" for every dataset — a wrong
answer, not an error. So the cache root and whether it was inherited or
defaulted are printed first, and a cache root that does not exist is a hard
failure rather than a table of zeros.

Usage:
    export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
    python tests/check_rep_pools.py caco2 logd
    python tests/check_rep_pools.py --trial          # the five trial datasets
    python tests/check_rep_pools.py --all
    python tests/check_rep_pools.py --all --missing-only
    python tests/check_rep_pools.py --trial --quiet  # exit code only
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

# The five datasets the 2026-09-14 run trials on, before the other 37.
TRIAL_DATASETS = [
    'buchwald_hartwig',
    'caco2',
    'flab_expression_koenig',
    'tcga_brca_pam50',
    'logd',
]


def cache_root():
    """(path, how_it_was_set). Mirrors _shared._rep_cache_dir exactly."""
    env = os.environ.get('KIRBY_CACHE_DIR')
    if env:
        return Path(env) / 'rep_cache', 'KIRBY_CACHE_DIR'
    return Path.home() / '.cache' / 'kirby' / 'rep_cache', 'DEFAULT (no KIRBY_CACHE_DIR)'


def print_environment(root, how):
    """What this process is actually reading. Printed before any result."""
    venv = os.environ.get('VIRTUAL_ENV')
    print("ENVIRONMENT")
    print(f"  python          {sys.version.split()[0]}  ({sys.executable})")
    print(f"  virtualenv      {venv or 'NONE ACTIVE'}")
    print(f"  cache root      {root}")
    print(f"  set by          {how}")
    print(f"  root exists     {'yes' if root.exists() else 'NO'}")
    if how.startswith('DEFAULT'):
        print()
        print("  ⛔ KIRBY_CACHE_DIR is not set. On ARC that variable is set by")
        print("     setup.sh and every representation lives under it. Without it")
        print("     this reads an empty directory and reports everything missing.")
        print("     Run:  export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh")
    print()


def present_reps(root, dataset):
    d = root / dataset
    if not d.exists():
        return set()
    return {p.stem for p in d.glob('*.npy')}


def conditions_reachable(n_pool_missing, n_aug, n_aug_missing):
    """Which precompute conditions this dataset can actually run.

    Mirrors precompute_hybrids.py: an incomplete pool refuses the dataset
    outright; an incomplete augmentation set keeps 'none' and drops the other
    two; no augmentation representations at all means 'none' only.
    """
    if n_pool_missing:
        return []
    if n_aug == 0 or n_aug_missing:
        return ['none']
    return ['none', 'uncapped', 'capped']


def main():
    ap = argparse.ArgumentParser(
        description="Check that datasets' representations are on disk.")
    ap.add_argument('datasets', nargs='*', help='Dataset names')
    ap.add_argument('--trial', action='store_true',
                    help='The five trial datasets')
    ap.add_argument('--all', action='store_true',
                    help='Every dataset in DATASET_REPS')
    ap.add_argument('--missing-only', action='store_true',
                    help='Print only datasets that are not fully ready')
    ap.add_argument('--quiet', action='store_true',
                    help='Print the environment and the verdict line only')
    ap.add_argument('--emit-conditions', action='store_true',
                    help='Machine-readable: one "<dataset> <cond,cond,...>" '
                         'line per dataset on stdout and nothing else, naming '
                         'the precompute conditions that dataset can actually '
                         'run. A dataset that would be refused emits no line. '
                         'For a submit script to build its plan from.')
    args = ap.parse_args()

    root, how = cache_root()
    if not args.emit_conditions:
        print_environment(root, how)

    if not root.exists():
        if args.emit_conditions:
            # Say nothing on stdout so a caller cannot mistake an unset
            # KIRBY_CACHE_DIR for "no dataset is ready"; say why on stderr.
            print(f"check_rep_pools: cache root {root} does not exist "
                  f"({how}). Source setup.sh.", file=sys.stderr)
            return 2
        print(f"REFUSED: the representation cache root does not exist.")
        print(f"         {root}")
        print(f"         This is what a wrong environment looks like. Source")
        print(f"         setup.sh, or set KIRBY_CACHE_DIR, and run again.")
        return 2

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

    unknown = [d for d in datasets if d not in DATASET_REPS]
    if unknown:
        print(f"Not in DATASET_REPS, nothing to check: {unknown}")
        datasets = [d for d in datasets if d in DATASET_REPS]

    rows = []
    for ds in datasets:
        cfg = DATASET_REPS[ds]
        pool = list(cfg.get('general', []))
        aug = list(cfg.get('augmentation', []))
        have = present_reps(root, ds)
        pool_missing = [r for r in pool if r not in have]
        aug_missing = [r for r in aug if r not in have]
        conds = conditions_reachable(len(pool_missing), len(aug), len(aug_missing))
        rows.append({
            'dataset': ds, 'pool': pool, 'aug': aug,
            'pool_missing': pool_missing, 'aug_missing': aug_missing,
            'conditions': conds,
        })

    if args.emit_conditions:
        for r in rows:
            if r['conditions']:
                print(f"{r['dataset']} {','.join(r['conditions'])}")
        return 1 if any(not r['conditions'] for r in rows) else 0

    refused = [r for r in rows if not r['conditions']]
    partial = [r for r in rows if r['conditions'] == ['none'] and r['aug']]
    ready = [r for r in rows if len(r['conditions']) == 3
             or (r['conditions'] == ['none'] and not r['aug'])]

    if not args.quiet:
        print("PER DATASET — counts are representations, not features")
        print(f"  {'dataset':30s} {'pool':>9s} {'augmentation':>13s}  conditions it can run")
        print(f"  {'-'*30} {'-'*9} {'-'*13}  {'-'*28}")
        for r in rows:
            if args.missing_only and r in ready:
                continue
            n_pool, n_pm = len(r['pool']), len(r['pool_missing'])
            n_aug, n_am = len(r['aug']), len(r['aug_missing'])
            conds = ', '.join(r['conditions']) if r['conditions'] else 'NONE — REFUSED'
            print(f"  {r['dataset']:30s} {n_pool - n_pm:4d} of {n_pool:<2d} "
                  f"{n_aug - n_am:6d} of {n_aug:<2d}  {conds}")
            if r['pool_missing']:
                print(f"      pool not on disk ({len(r['pool_missing'])}): "
                      f"{r['pool_missing']}")
            if r['aug_missing']:
                print(f"      augmentation not on disk ({len(r['aug_missing'])}): "
                      f"{r['aug_missing']}")
        print()

    print("VERDICT")
    print(f"  {len(ready)} of {len(rows)} datasets can run every condition they have")
    if partial:
        print(f"  {len(partial)} of {len(rows)} can run 'none' only — augmentation "
              f"representations absent: {[r['dataset'] for r in partial]}")
    if refused:
        print(f"  {len(refused)} of {len(rows)} would be REFUSED by "
              f"precompute_hybrids.py — pool incomplete: "
              f"{[r['dataset'] for r in refused]}")

    # Non-zero when anything would be refused, so a submit script can gate on it.
    return 1 if refused else 0


if __name__ == '__main__':
    sys.exit(main())
