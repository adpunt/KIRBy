#!/usr/bin/env python
"""Check precompute status by validating cache files on disk.

Walks ~/.cache/kirby/searches/ (or $KIRBY_CACHE_DIR/searches/) and for each
dataset in DATASET_REPS verifies that the expected condition files exist,
load cleanly, carry a valid _cache_meta header, and contain non-empty
best_by_model / all_results.

Standard expected conditions (drives the strict found/expected ratio):
    general__none,  general__uncapped,  general__capped
    top__none,      top__uncapped,      top__capped

Datasets without augmentation get the 'none' conditions only. The 'capped'
conditions also require augmentations. Distance conditions are not checked
by default — pass --distance to include them.

Additionally reports c02 per-model top-rep-dim caches (additive, not part
of the strict ratio since the expected count is dynamic per dataset):
    {dataset}__{tier}__{aug}__full__top_rep_dim_{model}.pkl
These are written by c02_complexity_grid.py via condition_suffix and survive
across c02 re-runs. The 'top_rep_dim' column shows ok/total for whatever was
found.

Usage:
    python tests/check_precompute_status.py                # full report
    python tests/check_precompute_status.py --dataset NAME # one dataset
    python tests/check_precompute_status.py --done         # only DONE
    python tests/check_precompute_status.py --incomplete   # only not-DONE
    python tests/check_precompute_status.py --distance     # include distance
    python tests/check_precompute_status.py --verbose      # per-condition + per-model top_rep_dim
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


def find_top_rep_dim_files(dataset, searches_dir):
    """Find c02 per-model top-rep-dim cache files for a dataset.

    These are written by c02_complexity_grid.py (per-model search at the
    model's best baseline rep dim, condition_suffix='top_rep_dim_<model>').
    Cache name: {dataset}__{tier}__{aug}__full__top_rep_dim_<model>.pkl

    Returns list of (filename, model, status, detail) tuples.
    """
    pattern = f"{dataset}__*__full__top_rep_dim_*.pkl"
    out = []
    for path in sorted(searches_dir.glob(pattern)):
        # Extract the model name from the suffix
        stem = path.stem  # filename without .pkl
        # Suffix is everything after '__top_rep_dim_'
        marker = '__top_rep_dim_'
        if marker in stem:
            model = stem.split(marker, 1)[1]
        else:
            model = '?'
        status, detail = _check_file(path)
        out.append((path.name, model, status, detail))
    return out


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
    # c02 per-model top-rep-dim caches (additive — count separately, not
    # part of the strict precompute denominator since their expected count
    # is dynamic per dataset).
    top_rep_dim_files = find_top_rep_dim_files(dataset, searches_dir)
    if n_ok == len(expected) and expected:
        overall = 'DONE'
    elif n_ok == 0:
        overall = 'missing'
    else:
        overall = 'partial'
    return overall, n_ok, len(expected), results, top_rep_dim_files


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
            overall, n_ok, n_exp, results, top_rep_dim_files = classify_dataset(
                ds, cfg, searches_dir, include_distance=args.distance)
            if n_exp == 0:
                continue
            totals[overall] += 1
            if args.done and overall != 'DONE':
                continue
            if args.incomplete and overall == 'DONE':
                continue
            rows.append((ds, overall, n_ok, n_exp, results, top_rep_dim_files))

        if not rows:
            continue

        print(f"\n=== {domain.upper()} ===")
        print(f"  {'dataset':<38} {'status':<8} {'found':<8} {'top_rep_dim':<12} missing/bad")
        print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*12} {'-'*40}")
        for ds, overall, n_ok, n_exp, results, top_rep_dim_files in rows:
            bad = [
                f"{tier}/{strat}={status}"
                for tier, strat, status, _ in results
                if status != 'ok'
            ]
            n_trd_ok = sum(1 for _, _, st, _ in top_rep_dim_files if st == 'ok')
            n_trd = len(top_rep_dim_files)
            trd_str = f"{n_trd_ok}/{n_trd}" if n_trd else '-'
            bad_str = ', '.join(bad) if bad else ''
            print(f"  {ds:<38} {overall:<8} {n_ok}/{n_exp:<6} {trd_str:<12} {bad_str}")
            if args.verbose:
                for tier, strat, status, detail in results:
                    mark = '✓' if status == 'ok' else 'X'
                    print(f"      {mark} {tier}/{strat}: {status} {detail}")
                for fname, model, status, detail in top_rep_dim_files:
                    mark = '✓' if status == 'ok' else 'X'
                    print(f"      {mark} top_rep_dim/{model}: {status} {detail}")

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
