#!/usr/bin/env python
"""Check precompute status against the hybrid registry.

Reads ~/.cache/kirby/hybrid_registry.json and compares every dataset in
DATASET_REPS against the standard conditions it's supposed to have:

    general/none, top/none, general/uncapped, top/uncapped,
    general/capped, top/capped

Datasets without augmentation only get general/none + top/none.
Distance conditions are not checked.

Usage:
    python tests/check_precompute_status.py                # full report
    python tests/check_precompute_status.py --dataset NAME # one dataset
    python tests/check_precompute_status.py --done         # only DONE
    python tests/check_precompute_status.py --incomplete   # only not-DONE
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

from experiments.complexity.complexity_helpers import DATASET_REPS


def _registry_path():
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return cache_root / 'hybrid_registry.json'


def _get_domain(name):
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


def expected_conditions(cfg):
    """Return set of expected standard condition keys for a dataset config."""
    general = cfg.get('general', [])
    top = cfg.get('top_tier', [])
    aug = cfg.get('augmentation', [])
    expected = set()

    if len(general) >= 2:
        expected.add('general/none')
        if aug:
            expected.add('general/uncapped')
            expected.add('general/capped')
    elif len(general) == 1 and aug:
        expected.add('general/uncapped')

    if len(top) >= 2:
        expected.add('top/none')
        if aug:
            expected.add('top/uncapped')
            expected.add('top/capped')
    elif len(top) == 1 and (general or aug):
        expected.add('top/uncapped')

    return expected


def classify(expected, found):
    """Return ('DONE'|'partial'|'missing', n_found, missing_set)."""
    found_std = {k for k in found
                 if '/' in k
                 and not k.startswith(('general_distance', 'top_distance'))
                 and not k.startswith('top+borderline')}
    have = found_std & expected
    missing = expected - found_std
    if not missing:
        return 'DONE', len(have), missing
    if not have:
        return 'missing', 0, missing
    return 'partial', len(have), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', help='check a single dataset')
    ap.add_argument('--done', action='store_true', help='show only DONE')
    ap.add_argument('--incomplete', action='store_true',
                    help='show only partial/missing')
    args = ap.parse_args()

    reg_path = _registry_path()
    if not reg_path.exists():
        print(f"No registry at {reg_path}")
        sys.exit(1)
    registry = json.loads(reg_path.read_text())

    # Select datasets to check
    if args.dataset:
        if args.dataset not in DATASET_REPS:
            print(f"ERROR: '{args.dataset}' not in DATASET_REPS")
            sys.exit(1)
        datasets = [args.dataset]
    else:
        datasets = sorted(DATASET_REPS.keys())

    # Group by domain
    by_domain = {}
    for ds in datasets:
        by_domain.setdefault(_get_domain(ds), []).append(ds)

    totals = {'DONE': 0, 'partial': 0, 'missing': 0}

    for domain in sorted(by_domain.keys()):
        rows = []
        for ds in by_domain[domain]:
            cfg = DATASET_REPS[ds]
            expected = expected_conditions(cfg)
            if not expected:
                continue
            found = set(registry.get(ds, {}).get('configs', {}).keys())
            status, n_have, missing = classify(expected, found)
            totals[status] += 1

            if args.done and status != 'DONE':
                continue
            if args.incomplete and status == 'DONE':
                continue

            rows.append((ds, status, n_have, len(expected), sorted(missing)))

        if not rows:
            continue

        print(f"\n=== {domain.upper()} ===")
        print(f"  {'dataset':<38} {'status':<8} {'found':<8} missing")
        print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*40}")
        for ds, status, n_have, n_exp, missing in rows:
            miss_str = ','.join(missing) if missing else ''
            print(f"  {ds:<38} {status:<8} {n_have}/{n_exp:<6} {miss_str}")

    total = sum(totals.values())
    print(f"\n{'='*70}")
    print(f"Total datasets: {total}  |  "
          f"DONE: {totals['DONE']}  |  "
          f"partial: {totals['partial']}  |  "
          f"missing: {totals['missing']}")
    if totals['DONE']:
        print(f"\nReady for experiments: {totals['DONE']}/{total} datasets")


if __name__ == '__main__':
    main()
