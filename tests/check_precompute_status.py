#!/usr/bin/env python
"""Check precompute status by validating cache files on disk.

Walks ~/.cache/kirby/searches/ (or $KIRBY_CACHE_DIR/searches/) and for each
dataset in DATASET_REPS verifies that the expected condition files exist,
load cleanly, carry a valid _cache_meta header, and contain non-empty
best_by_model / all_results.

Standard expected conditions:
    general__none,  general__uncapped,  general__capped
    top__none,      top__uncapped,      top__capped

Datasets without augmentation get the 'none' conditions only. The 'capped'
conditions also require augmentations. Distance conditions are not checked
by default — pass --distance to include them.

This replaces the old registry-based status check, which only counted
registry keys and could not detect empty / stale / corrupt cache files.

Usage:
    python tests/check_precompute_status.py                # full report
    python tests/check_precompute_status.py --dataset NAME # one dataset
    python tests/check_precompute_status.py --done         # only DONE
    python tests/check_precompute_status.py --incomplete   # only not-DONE
    python tests/check_precompute_status.py --distance     # include distance
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

from experiments.complexity.complexity_helpers import DATASET_REPS


def _searches_dir() -> Path:
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return cache_root / 'searches'


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


def expected_conditions(cfg, include_distance=False):
    """Return list of expected (tier, aug_strategy) tuples for a dataset."""
    general = cfg.get('general', [])
    top = cfg.get('top_tier', [])
    aug = cfg.get('augmentation', [])
    expected = []

    if len(general) >= 2:
        expected.append(('general', 'none'))
        if aug:
            expected.append(('general', 'uncapped'))
            expected.append(('general', 'capped'))
    elif len(general) == 1 and aug:
        expected.append(('general', 'uncapped'))

    if len(top) >= 2:
        expected.append(('top', 'none'))
        if aug:
            expected.append(('top', 'uncapped'))
            expected.append(('top', 'capped'))
    elif len(top) == 1 and (general or aug):
        expected.append(('top', 'uncapped'))

    if include_distance:
        if len(general) >= 2:
            expected.append(('general_distance', 'none'))
            if aug:
                expected.append(('general_distance', 'uncapped'))
        if len(top) >= 2:
            expected.append(('top_distance', 'none'))
            if aug:
                expected.append(('top_distance', 'uncapped'))

    return expected


def _check_file(path: Path) -> tuple:
    """Return ('ok'|'missing'|'corrupt'|'empty'|'stale', detail_str)."""
    if not path.exists():
        return ('missing', '')
    try:
        wrapper = pickle.loads(path.read_bytes())
    except Exception as e:
        return ('corrupt', f'{type(e).__name__}: {e}')
    if not isinstance(wrapper, dict):
        return ('stale', 'pre-redesign SearchResult (no wrapper)')
    if '_cache_meta' not in wrapper or 'payload' not in wrapper:
        return ('stale', 'missing _cache_meta header')
    meta = wrapper.get('_cache_meta') or {}
    payload = wrapper.get('payload')
    # Validate structural fields of payload
    best = getattr(payload, 'best_by_model', None)
    all_r = getattr(payload, 'all_results', None)
    if not best or not all_r:
        return ('empty',
                f"best_by_model={len(best) if best else 0} "
                f"all_results={len(all_r) if all_r else 0}")
    return ('ok',
            f"n_results={len(all_r)} "
            f"models={sorted(best.keys())} "
            f"metric={meta.get('metric_name', '?')}")


def classify_dataset(dataset, cfg, searches_dir, include_distance):
    expected = expected_conditions(cfg, include_distance=include_distance)
    results = []
    n_ok = 0
    for tier, strat in expected:
        fname = f"{dataset}__{tier}__{strat}__full.pkl"
        path = searches_dir / fname
        status, detail = _check_file(path)
        results.append((tier, strat, status, detail))
        if status == 'ok':
            n_ok += 1
    if n_ok == len(expected) and expected:
        overall = 'DONE'
    elif n_ok == 0:
        overall = 'missing'
    else:
        overall = 'partial'
    return overall, n_ok, len(expected), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', help='check a single dataset')
    ap.add_argument('--done', action='store_true', help='show only DONE')
    ap.add_argument('--incomplete', action='store_true',
                    help='show only partial/missing')
    ap.add_argument('--distance', action='store_true',
                    help='also check distance conditions')
    ap.add_argument('--verbose', action='store_true',
                    help='print per-condition status')
    args = ap.parse_args()

    searches_dir = _searches_dir()
    if not searches_dir.exists():
        print(f"No search cache at {searches_dir}")
        print("  (will be created on first precompute run)")
        sys.exit(0 if args.incomplete or not args.done else 1)

    # Select datasets to check
    if args.dataset:
        if args.dataset not in DATASET_REPS:
            print(f"ERROR: '{args.dataset}' not in DATASET_REPS")
            sys.exit(1)
        datasets = [args.dataset]
    else:
        datasets = sorted(DATASET_REPS.keys())

    by_domain = {}
    for ds in datasets:
        by_domain.setdefault(_get_domain(ds), []).append(ds)

    totals = {'DONE': 0, 'partial': 0, 'missing': 0}

    for domain in sorted(by_domain.keys()):
        rows = []
        for ds in by_domain[domain]:
            cfg = DATASET_REPS[ds]
            overall, n_ok, n_exp, results = classify_dataset(
                ds, cfg, searches_dir, include_distance=args.distance)
            if n_exp == 0:
                continue
            totals[overall] += 1
            if args.done and overall != 'DONE':
                continue
            if args.incomplete and overall == 'DONE':
                continue
            rows.append((ds, overall, n_ok, n_exp, results))

        if not rows:
            continue

        print(f"\n=== {domain.upper()} ===")
        print(f"  {'dataset':<38} {'status':<8} {'found':<8} missing/bad")
        print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*40}")
        for ds, overall, n_ok, n_exp, results in rows:
            bad = [
                f"{tier}/{strat}={status}"
                for tier, strat, status, _ in results
                if status != 'ok'
            ]
            bad_str = ', '.join(bad) if bad else ''
            print(f"  {ds:<38} {overall:<8} {n_ok}/{n_exp:<6} {bad_str}")
            if args.verbose:
                for tier, strat, status, detail in results:
                    mark = '✓' if status == 'ok' else 'X'
                    print(f"      {mark} {tier}/{strat}: {status} {detail}")

    total = sum(totals.values())
    print(f"\n{'='*70}")
    print(f"Total datasets checked: {total}  |  "
          f"DONE: {totals['DONE']}  |  "
          f"partial: {totals['partial']}  |  "
          f"missing: {totals['missing']}")
    if totals['DONE']:
        print(f"\nReady for experiments: {totals['DONE']}/{total} datasets")


if __name__ == '__main__':
    main()
