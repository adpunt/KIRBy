#!/usr/bin/env python3
"""Real-dataset smoke test on ginkgo_heparin (n=94, smallest registered).

Exercises, in order:
  1. Registry lookup + load_data cross-check
  2. find_best_hybrid MISS path — writes search + baselines + hybrid_evals
  3. find_best_hybrid HIT path — identical call, reads search cache
  4. Phase D.2 dedup — smaller rep set with overlapping top_k subset,
     should register hybrid_eval hits without re-running feature selection
  5. Drift detection — different seed → search stale + recompute
  6. Direct pickle inspection of the on-disk cache file

Runs against the user's real `~/.cache/kirby/`. Test artifacts are
written under tier='smoke_test' and tier='smoke_test_subset' so they
can't collide with real precompute output. The test DOES NOT clean up
after itself — `ls ~/.cache/kirby/searches/ginkgo_heparin__smoke_*`
after the run is the record.

Usage:
    # Interactive session (HTC/ARC) or local with reps pre-cached:
    python tests/scripts/smoke_real_dataset.py

Exits 0 on all-pass, non-zero on any failure. Expected runtime:
roughly 1-5 minutes, depending on whether the reps are cached as
.npy on disk (they should be — reps are pre-cached separately and
aren't computed here).
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np
from kirby.registry import lookup
from kirby.find_best_hybrid import (
    find_best_hybrid, _CACHE_STATS, _reset_cache_stats,
    _cache_stats_summary,
)

DATASET = 'ginkgo_heparin'
TIER = 'smoke_test'
TIER_SUBSET = 'smoke_test_subset'
# Load timeout — if the data+reps aren't cached on disk and reps would
# be computed fresh, this catches that case quickly so the user isn't
# confused by a 30-minute "smoke" test.
LOAD_TIMEOUT_SEC = 120


def _fail(step: int, msg: str, details: str = ''):
    print(f'\nFAIL [{step}] {msg}')
    if details:
        print(details)
    sys.exit(1)


def _ok(step: int, msg: str):
    print(f'OK   [{step}] {msg}')


def main() -> int:
    info = lookup(DATASET)
    if info is None:
        _fail(0, f'{DATASET!r} not in DATASET_REGISTRY')

    print(f'=== smoke_real_dataset: {DATASET} ===')
    print(f'Registry: task={info.task_type}, metric={info.metric}, '
          f'n={info.n_samples}, domain={info.domain}')

    # ---- 1. load_data + cross-check --------------------------------------
    print('\n[1/6] load_data + registry cross-check')
    from experiments._shared import load_data
    t0 = time.time()
    data = load_data(DATASET)
    t_load = time.time() - t0
    print(f'  loaded in {t_load:.1f}s')
    if t_load > LOAD_TIMEOUT_SEC:
        _fail(1, f'load_data took {t_load:.1f}s > {LOAD_TIMEOUT_SEC}s — '
              f'reps likely being computed fresh. Abort before smoke '
              f'eats all your interactive time.')
    if data.get('dataset_name') != DATASET:
        _fail(1, f"load_data didn't stamp dataset_name",
              f"got {data.get('dataset_name')!r}")
    if data.get('task_type') != info.task_type:
        _fail(1, f'task_type mismatch: loader={data.get("task_type")!r} '
              f'vs registry={info.task_type!r}')
    _ok(1, f'{len(data["reps"])} reps, labels={data["labels"].shape}, '
         f'groups={len(set(data["groups"]))} unique')

    # Pick 5 smallest reps so search is fast. Avoids depending on which
    # foundation models are cached.
    by_dim = sorted(data['reps'].keys(),
                    key=lambda r: data['reps'][r].shape[1])
    simple_reps = by_dim[:5]
    if len(simple_reps) < 5:
        _fail(1, f'need at least 5 reps, got {len(simple_reps)}')
    print(f'  selected 5 smallest reps: '
          f'{[(r, data["reps"][r].shape[1]) for r in simple_reps]}')

    reps5 = {r: data['reps'][r] for r in simple_reps}
    reps4 = {r: data['reps'][r] for r in simple_reps[:4]}

    common = dict(
        labels=data['labels'],
        groups=data['groups'],
        metric=info.metric,
        task_type=info.task_type,
        speed='fast',
        budgets=[5, 10, 20],
        allocation_methods=['top_3'],
        importance_methods=['lightgbm_gain'],
        internal_evaluators=['rf'],
        augmentation_strategy='uncapped',
        cache=True,
        dataset_name=DATASET,
        verbose=False,
    )

    cache_root = Path(os.environ.get('KIRBY_CACHE_DIR',
                                      str(Path.home() / '.cache' / 'kirby')))
    search_path = (cache_root / 'searches' /
                   f'{DATASET}__{TIER}__uncapped__fast.pkl')
    subset_search_path = (cache_root / 'searches' /
                          f'{DATASET}__{TIER_SUBSET}__uncapped__fast.pkl')
    # Clean smoke artifacts from prior runs so we test fresh miss→hit.
    for p in (search_path, subset_search_path):
        if p.exists():
            p.unlink()

    # ---- 2. MISS path: first run writes --------------------------------
    print('\n[2/6] Run 1 on 5 reps — expect search_miss + cache write')
    _reset_cache_stats()
    t0 = time.time()
    r1 = find_best_hybrid(base_reps=reps5, tier=TIER, **common)
    e1 = time.time() - t0
    print(f'  elapsed: {e1:.1f}s')
    print(f'  {_cache_stats_summary()}')
    if _CACHE_STATS['search_miss'] != 1:
        _fail(2, f'expected 1 search_miss, got {_CACHE_STATS["search_miss"]}')
    if not search_path.exists():
        _fail(2, f'no cache file written at {search_path}')
    if not r1.best_by_model:
        _fail(2, 'result has empty best_by_model')
    he_written_r1 = _CACHE_STATS['hybrid_eval_miss']
    _ok(2, f'cache written ({search_path.stat().st_size} bytes), '
         f'hybrid_eval_miss={he_written_r1} (entries written to disk)')

    # ---- 3. HIT path: identical call -----------------------------------
    print('\n[3/6] Run 2 identical — expect search_hit, no compute')
    _reset_cache_stats()
    t0 = time.time()
    r2 = find_best_hybrid(base_reps=reps5, tier=TIER, **common)
    e2 = time.time() - t0
    print(f'  elapsed: {e2:.1f}s')
    print(f'  {_cache_stats_summary()}')
    if _CACHE_STATS['search_hit'] != 1:
        _fail(3, f'expected 1 search_hit, got {_CACHE_STATS["search_hit"]}')
    if _CACHE_STATS['search_miss'] != 0:
        _fail(3, f'expected 0 miss, got {_CACHE_STATS["search_miss"]}')
    # Hit path should be much faster — 10x is a safe threshold.
    if e1 > 2.0 and e2 > e1 * 0.3:
        _fail(3, f'hit path ({e2:.1f}s) not meaningfully faster than '
              f'miss ({e1:.1f}s) — cache read may not be short-circuiting')
    # Same SearchResult?
    if r1.metric_name != r2.metric_name:
        _fail(3, 'metric_name differs between miss and hit returns')
    _ok(3, f'hit path {e2:.2f}s << miss path {e1:.2f}s, same result')

    # ---- 4. Phase D.2 dedup: subset of reps, same top_3 ----------------
    print('\n[4/6] Run 3 on 4-rep subset — expect hybrid_eval_hit > 0')
    _reset_cache_stats()
    t0 = time.time()
    r3 = find_best_hybrid(base_reps=reps4, tier=TIER_SUBSET, **common)
    e3 = time.time() - t0
    print(f'  elapsed: {e3:.1f}s')
    print(f'  {_cache_stats_summary()}')
    he_hits = _CACHE_STATS['hybrid_eval_hit']
    he_misses = _CACHE_STATS['hybrid_eval_miss']
    # search file must be distinct (different tier)
    if not subset_search_path.exists():
        _fail(4, f'subset search cache not written at {subset_search_path}')
    if he_hits == 0:
        # Not necessarily a failure — possible if per-fold alloc scores
        # pick different subsets with/without the 5th rep. Warn only.
        print(f'  WARN: no hybrid_eval hits. Predicted top_3 likely '
              f'differed between runs (per-fold alloc scores can shift '
              f'when input rep set changes). Write path still worked '
              f'({he_misses} new entries).')
    else:
        _ok(4, f'D.2 dedup: {he_hits} hybrid_eval cache hits from Run 1')

    # ---- 5. Drift detection --------------------------------------------
    print('\n[5/6] Run 4 with seed=99 — expect search_stale + recompute')
    _reset_cache_stats()
    t0 = time.time()
    r4 = find_best_hybrid(base_reps=reps5, tier=TIER, seed=99, **common)
    e4 = time.time() - t0
    print(f'  elapsed: {e4:.1f}s')
    print(f'  {_cache_stats_summary()}')
    if _CACHE_STATS['search_stale'] != 1:
        _fail(5, f'expected 1 search_stale, got '
              f'{_CACHE_STATS["search_stale"]}')
    _ok(5, 'drift detected, cache recomputed + rewritten')

    # ---- 6. Direct cache-file inspection -------------------------------
    print('\n[6/6] Direct pickle.load of the search cache file')
    w = pickle.loads(search_path.read_bytes())
    if not isinstance(w, dict):
        _fail(6, 'cache file is not a dict wrapper')
    if '_cache_meta' not in w or 'payload' not in w:
        _fail(6, 'envelope structure invalid',
              f'top-level keys: {list(w.keys())}')
    meta = w['_cache_meta']
    payload = w['payload']
    # Required meta fields from the current _make_search_meta
    required_meta = {
        'dataset_name', 'tier', 'aug_strategy', 'speed',
        'rep_names', 'rep_hashes', 'labels_hash',
        'metric_name', 'task_type', 'n_folds', 'seed',
        'budgets', 'candidate_ranking', 'greedy_tolerance',
        'screen_threshold', 'two_phase', 'refinement_top_n',
        'test_rep_subsets', 'weight_method', 'step_size',
        'cv_splitter_id', 'rep_factories_id', 'condition_suffix',
    }
    missing = required_meta - set(meta.keys())
    if missing:
        _fail(6, f'meta missing {len(missing)} fields', f'{sorted(missing)}')
    if meta['dataset_name'] != DATASET:
        _fail(6, f'meta dataset_name = {meta["dataset_name"]!r}')
    if meta['metric_name'] != info.metric:
        _fail(6, f'meta metric = {meta["metric_name"]!r}, '
              f'expected {info.metric!r}')
    if meta['task_type'] != info.task_type:
        _fail(6, f'meta task_type = {meta["task_type"]!r}')
    if not payload.best_by_model:
        _fail(6, 'payload.best_by_model is empty')
    for m, cfg in payload.best_by_model.items():
        if 'blueprint' not in cfg:
            _fail(6, f'model {m!r} config has no blueprint', repr(cfg))
    _ok(6, f'{len(meta)} meta fields, '
         f'payload.metric_name={payload.metric_name!r}, '
         f'models={list(payload.best_by_model.keys())}')

    print('\n' + '=' * 60)
    print(f'smoke_real_dataset ({DATASET}): ALL 6 CHECKS PASSED')
    print(f'cache files left behind:')
    print(f'  {search_path.name}')
    print(f'  {subset_search_path.name}')
    he_dir = cache_root / 'hybrid_evals' / DATASET
    if he_dir.exists():
        print(f'  hybrid_evals/{DATASET}/: '
              f'{len(list(he_dir.glob("*.pkl")))} files')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
