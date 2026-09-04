#!/usr/bin/env python
"""Build the pruned representation pool for a dataset and stop.

Fills `$KIRBY_CACHE_DIR/rep_cache/<dataset>/<rep>.npy` — one array per
representation, computed once over every row. Fits no model, runs no
cross-validation, chooses no recipe.

WHICH REPRESENTATIONS. `DATASET_REPS[ds]['general']`, the pool the user signed
off during rep pruning (REP_PRUNING_LOG.md). Nothing else: not `augmentation`,
not `similarity`, not `ft_reps`. `--include-augmentation` adds the augmentation
list; the other two have no whole-dataset matrix, because each is fitted inside
a fold.

WHY IT IS NOT ONE load_data CALL. `_cache_rep_dict` hands every missing name to
the builder in one call (_shared.py:284), and the molecular and TCGA builders
raise when everything in that call fails (find_best_hybrid_molecular.py:649,
find_best_hybrid_tcga.py:438). The exception escapes load_data, so the
representations already on disk are discarded and the dataset load dies. That is
what stopped herg in August with 17 of 18 representations cached. This script
tries the whole pool first, and on an exception retries one representation at a
time so a single failure costs one representation.

EXIT CODE. Non-zero when any requested representation is still absent at the
end, with the names printed. A run that builds nine of ten exits 1.

Usage:
    python tests/precompute_reps.py logd
    python tests/precompute_reps.py logd caco2 herg_ki
    python tests/precompute_reps.py --all
    python tests/precompute_reps.py --list          # roster, one name per line
    python tests/precompute_reps.py logd --dry-run  # what is missing, build nothing

One dataset per process. Two processes on the same dataset race: the temporary
file `_shared.py` writes before its atomic rename is `<rep>.tmp.npy`, the same
path in both (_shared.py:213).
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments._shared import load_data, _rep_cache_dir  # noqa: E402
from experiments.complexity.complexity_helpers import DATASET_REPS  # noqa: E402


def wanted_reps(dataset, include_augmentation=False):
    """The representation names this script is responsible for."""
    cfg = DATASET_REPS[dataset]
    names = list(cfg.get('general', []))
    if include_augmentation:
        for r in cfg.get('augmentation', []):
            if r not in names:
                names.append(r)
    return names


def on_disk(dataset, names):
    """Split names into (cached, missing) by what is in rep_cache already."""
    d = _rep_cache_dir() / dataset
    cached = [r for r in names if (d / f'{r}.npy').exists()]
    missing = [r for r in names if r not in cached]
    return cached, missing


def build(dataset, names):
    """Build `names` for `dataset`. Returns the names that raised."""
    if not names:
        return []
    try:
        load_data(dataset, rep_preset='all', rep_names=names)
        return []
    except Exception:
        print(f"  {dataset}: the pool raised as one call — retrying one at a "
              f"time so the survivors are kept", flush=True)
        traceback.print_exc()

    failed = []
    for r in names:
        _, still_missing = on_disk(dataset, [r])
        if not still_missing:
            continue
        try:
            load_data(dataset, rep_preset='all', rep_names=[r])
        except Exception:
            print(f"  {dataset}/{r}: FAILED", flush=True)
            traceback.print_exc()
            failed.append(r)
    return failed


def run(dataset, include_augmentation=False, dry_run=False):
    """Build one dataset's pool. Returns the names still absent at the end."""
    names = wanted_reps(dataset, include_augmentation)
    cached, missing = on_disk(dataset, names)
    print(f"\n=== {dataset}: {len(names)} representations, "
          f"{len(cached)} cached, {len(missing)} to build", flush=True)
    if cached:
        print(f"  cached:  {', '.join(cached)}", flush=True)
    if missing:
        print(f"  to build: {', '.join(missing)}", flush=True)

    if dry_run or not missing:
        return missing

    build(dataset, missing)

    # Ask the disk, not the return value: load_data drops a representation that
    # raised inside the builder without raising itself.
    _, still_missing = on_disk(dataset, names)
    if still_missing:
        print(f"  {dataset}: MISSING {len(still_missing)} of {len(names)}: "
              f"{', '.join(still_missing)}", flush=True)
    else:
        print(f"  {dataset}: all {len(names)} on disk", flush=True)
    return still_missing


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('datasets', nargs='*', help='dataset names')
    p.add_argument('--all', action='store_true',
                   help='every key in DATASET_REPS')
    p.add_argument('--list', action='store_true',
                   help='print the roster in the order --all uses, and stop')
    p.add_argument('--include-augmentation', action='store_true',
                   help="also build DATASET_REPS[ds]['augmentation']")
    p.add_argument('--dry-run', action='store_true',
                   help='report what is missing, build nothing')
    args = p.parse_args()

    roster = sorted(DATASET_REPS.keys())

    if args.list:
        for ds in roster:
            print(ds)
        return 0

    if args.all:
        datasets = roster
    elif args.datasets:
        datasets = args.datasets
    else:
        p.error('give dataset names, or --all, or --list')

    unknown = [d for d in datasets if d not in DATASET_REPS]
    if unknown:
        print(f"ERROR: not in DATASET_REPS: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(f"rep_cache root: {_rep_cache_dir()}")
    print(f"KIRBY_CACHE_DIR={os.environ.get('KIRBY_CACHE_DIR', '(unset)')}")

    incomplete = {}
    for ds in datasets:
        try:
            still_missing = run(ds, args.include_augmentation, args.dry_run)
        except Exception:
            print(f"  {ds}: the dataset itself did not load", flush=True)
            traceback.print_exc()
            still_missing = wanted_reps(ds, args.include_augmentation)
        if still_missing:
            incomplete[ds] = still_missing

    print("\n=== summary")
    for ds in datasets:
        names = wanted_reps(ds, args.include_augmentation)
        gap = incomplete.get(ds, [])
        state = 'COMPLETE' if not gap else f"MISSING {len(gap)}: {', '.join(gap)}"
        print(f"  {ds:28s} {len(names) - len(gap):2d}/{len(names):2d}  {state}")

    if args.dry_run:
        return 0
    return 1 if incomplete else 0


if __name__ == '__main__':
    sys.exit(main())
