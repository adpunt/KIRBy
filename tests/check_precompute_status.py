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
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments._shared import (
    FULL_BUDGETS, DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
from kirby.find_best_hybrid import SPEED_TIERS

# Standard (non-distance) precompute always runs speed='full' and passes
# internal_evaluators=None into find_best_hybrid, which then normalises None
# to the speed-tier default (SPEED_TIERS['full']['internal_evaluators']) BEFORE
# writing _cache_meta (find_best_hybrid.py ~line 4115/4232). So the value the
# engine compares against is the tier default, NOT None — deriving None here
# would false-flag every healthy standard cache as drift.
_FULL_TIER_EVALUATORS = tuple(sorted(
    SPEED_TIERS['full'].get('internal_evaluators') or []))

# Fields the running engine (find_best_hybrid._load_cache_with_meta) compares
# between the cache's stored _cache_meta and the meta the CURRENT code builds
# (_make_search_meta). A mismatch on ANY of these makes the engine reject the
# cache ("meta drift on '<field>'") and rebuild cold — even though the file
# exists and is structurally valid. The engine's ignore_keys
# (code_timestamp / rep_hashes / aug_hashes / rep_hash) are deliberately
# excluded because neural-rep value-hashes are not bit-reproducible.
#
# Of the validated fields, these are the ones we can derive WITHOUT loading the
# dataset (cheap — from DATASET_REPS + _shared constants + the condition):
#   internal_evaluators, rep_names, budgets, aug_names
# The rest (n_samples, labels_hash, groups_hash, rep_shapes, metric_name,
# task_type, model_names, n_folds, seed, …) need the actual data array or a
# full pipeline run, so we cannot assert them here — they are reported as
# "unchecked" rather than silently treated as OK.
_CHEAP_DRIFT_FIELDS = ('internal_evaluators', 'rep_names', 'budgets', 'aug_names')


def _expected_meta_cheap(dataset, cfg, tier, strat):
    """Return the cheap-to-derive subset of _cache_meta the CURRENT code
    would build for a (dataset, tier, strat) precompute condition.

    Mirrors precompute_hybrids.precompute_dataset (which reps/budgets/
    evaluators it passes into find_best_hybrid) and find_best_hybrid.
    _make_search_meta (how those become meta fields). Only the four
    _CHEAP_DRIFT_FIELDS are returned; everything else needs the data array.
    """
    general = cfg.get('general', [])
    top = cfg.get('top_tier', [])
    aug = cfg.get('augmentation', [])

    is_distance = tier.endswith('_distance')
    base_tier = tier[:-len('_distance')] if is_distance else tier

    # rep_names: base_reps the precompute requests for this tier.
    # _make_search_meta stores tuple(sorted(base_reps.keys())). 'top+borderline'
    # uses the top set as base; the borderline reps go into the aug pool.
    if base_tier in ('top', 'top+borderline'):
        rep_set = top
    else:  # 'general'
        rep_set = general
    rep_names = tuple(sorted(rep_set))

    # budgets: distance tiers use DISTANCE_BUDGETS, standard use FULL_BUDGETS
    # (speed='full' is the only supported precompute speed).
    budgets = tuple(DISTANCE_BUDGETS if is_distance else FULL_BUDGETS)

    # internal_evaluators: distance tiers pass DISTANCE_EVALUATORS explicitly;
    # standard tiers pass None, which find_best_hybrid normalises to the
    # speed='full' tier default before storing it in _cache_meta. Both end up
    # as tuple(sorted(...)) in the stored meta.
    if is_distance:
        internal_evaluators = tuple(sorted(DISTANCE_EVALUATORS))
    else:
        internal_evaluators = _FULL_TIER_EVALUATORS

    # aug_names: 'none' strategy passes no augmentations → stored as ().
    # uncapped/capped pass the dataset's augmentation list. (For the
    # 'top+borderline' condition the pool also includes general-minus-top,
    # but the checker never validates that condition for the DONE count.)
    if strat == 'none':
        aug_names = ()
    else:
        aug_names = tuple(sorted(aug))

    return {
        'rep_names': rep_names,
        'budgets': budgets,
        'internal_evaluators': internal_evaluators,
        'aug_names': aug_names,
    }


def _detect_drift(stored_meta, expected_cheap):
    """Compare stored _cache_meta against the cheap expected values the
    current code would produce. Return a list of (field, stored, expected)
    for each field that DRIFTS (i.e. the engine would reject the cache).

    A field absent from stored_meta is also drift — the engine treats
    'k not in stored' as a miss just like a value mismatch.
    """
    drifts = []
    for field, expected_val in expected_cheap.items():
        if field not in stored_meta:
            drifts.append((field, '<absent>', expected_val))
            continue
        stored_val = stored_meta[field]
        # Normalise stored sequences to tuples for an apples-to-apples
        # compare (pickled meta already stores tuples, but be defensive
        # against lists from any legacy writer).
        if isinstance(stored_val, list):
            stored_val = tuple(stored_val)
        if stored_val != expected_val:
            drifts.append((field, stored_val, expected_val))
    return drifts


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


# Top/capped is only expected when n_samples <= TOP_CAPPED_MAX_N. Above
# that threshold both precompute_hybrids.py and experiments._shared skip
# the capped strategy (projects past the 7-day partition). When n is
# unknown — no cache yet on disk — we default to EXPECTING top/capped so
# missing-cache datasets still surface in the "partial" report.
TOP_CAPPED_MAX_N = 250


def _peek_n_samples(searches_dir, dataset):
    """Return n_samples for a dataset by reading meta from any cached
    search file. Returns None when no cache exists yet.

    Order of preference: general/none → top/none → general/uncapped →
    top/uncapped. Stops at the first hit; this is purely a metadata peek,
    not the full _check_file pipeline."""
    for tier, strat in [('general', 'none'), ('top', 'none'),
                        ('general', 'uncapped'), ('top', 'uncapped')]:
        fname = f"{dataset}__{_sanitize(tier)}__{_sanitize(strat)}__full.pkl"
        path = searches_dir / fname
        if not path.exists():
            continue
        try:
            wrapper = pickle.loads(path.read_bytes())
        except Exception:
            continue
        if not isinstance(wrapper, dict):
            continue
        meta = wrapper.get('_cache_meta') or {}
        n = meta.get('n_samples')
        if n is not None:
            return int(n)
    return None


def expected_conditions(cfg, include_distance=False, dataset_name=None,
                        n_samples=None):
    """Return list of expected (tier, aug_strategy) tuples for a dataset.

    n_samples (when known) controls whether top/capped is expected:
    only datasets with n <= TOP_CAPPED_MAX_N have it. Pass None to expect
    top/capped (the conservative default for not-yet-precomputed datasets).
    """
    general = cfg.get('general', [])
    top = cfg.get('top_tier', [])
    aug = cfg.get('augmentation', [])
    expected = []

    if len(general) >= 2:
        expected.append(('general', 'none'))
        if aug:
            expected.append(('general', 'uncapped'))
            # general/capped dropped 2026-05-22: 9x cost vs uncapped on the
            # large general-rep pool; all consumers now read top/capped.
    elif len(general) == 1 and aug:
        expected.append(('general', 'uncapped'))

    if len(top) >= 2:
        expected.append(('top', 'none'))
        if aug:
            expected.append(('top', 'uncapped'))
            # n_samples is None when no cache exists yet for this dataset
            # — we still expect top/capped so the "missing" status surfaces.
            if n_samples is None or n_samples <= TOP_CAPPED_MAX_N:
                expected.append(('top', 'capped'))
            # Note: precompute_hybrids.py also writes ('top+borderline',
            # uncapped) when general has reps outside top. We do NOT
            # require it in the DONE count — many caches predate when
            # that condition was added, and requiring it would mark
            # previously-DONE datasets as partial overnight. The file is
            # consumed by experiments via direct path lookup, not via
            # this checker.
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


def _check_file(path: Path, expected_cheap: dict = None) -> tuple:
    """Return ('ok'|'missing'|'corrupt'|'empty'|'stale'|'drift', detail, extras).

    'drift' is reported when the file exists and is structurally valid BUT its
    stored _cache_meta disagrees with the meta the current code would build for
    at least one cheap-to-derive field (rep_names / budgets / aug_names /
    internal_evaluators). The running engine rejects such a file on load
    ("meta drift on '<field>'") and rebuilds it cold, so it must NOT count as
    DONE. Pass expected_cheap=None (e.g. top_rep_dim caches, n-peeks) to skip
    the drift check entirely.
    """
    if not path.exists():
        return ('missing', '', {})
    try:
        wrapper = pickle.loads(path.read_bytes())
    except Exception as e:
        return ('corrupt', f'{type(e).__name__}: {e}', {})
    if not isinstance(wrapper, dict):
        return ('stale', 'pre-redesign SearchResult (no wrapper)', {})
    if '_cache_meta' not in wrapper or 'payload' not in wrapper:
        return ('stale', 'missing _cache_meta header', {})
    meta = wrapper.get('_cache_meta') or {}
    payload = wrapper.get('payload')
    best = getattr(payload, 'best_by_model', None)
    all_r = getattr(payload, 'all_results', None)
    if not best or not all_r:
        return ('empty',
                f"best_by_model={len(best) if best else 0} "
                f"all_results={len(all_r) if all_r else 0}",
                {})
    # Clamp expected budgets to THIS cache's n_samples, mirroring the engine's
    # at-build clamp (find_best_hybrid drops budgets > n and appends n as the
    # at-n boundary). Without this, every dataset with n < max(FULL_BUDGETS)
    # would false-flag a 'budgets' drift even though the engine reuses the cache
    # cleanly — the opposite false signal. n comes from the stored meta (the
    # same n the engine clamped against when it wrote this file).
    if expected_cheap and 'budgets' in expected_cheap:
        _n = meta.get('n_samples')
        if _n:
            _b = list(expected_cheap['budgets'])
            _clamped = [b for b in _b if b <= _n]
            if any(b > _n for b in _b) and _n not in _clamped:
                _clamped.append(_n)
            if not _clamped:
                _clamped = [_n]
            expected_cheap = {**expected_cheap, 'budgets': tuple(_clamped)}
    # Drift check: would the CURRENT engine accept this cache? Compare the
    # cheap-to-derive meta fields. Any mismatch = the engine rebuilds cold.
    if expected_cheap:
        drifts = _detect_drift(meta, expected_cheap)
        if drifts:
            detail = '; '.join(
                f"{f}(stored={s!r}, expected={e!r})" for f, s, e in drifts)
            return ('drift', detail,
                    {'drift_fields': [f for f, _, _ in drifts],
                     'mtime': path.stat().st_mtime})
    # Per-model breakdown: sum time_seconds and count configs grouped by model.
    # Each entry in all_results is a dict with 'model' and 'time_seconds'.
    per_model: dict = {}
    for r in all_r:
        if not isinstance(r, dict):
            continue
        mdl = r.get('model') or '?'
        t = r.get('time_seconds') or 0.0
        if mdl not in per_model:
            per_model[mdl] = {'wall_seconds': 0.0, 'n_configs': 0}
        per_model[mdl]['wall_seconds'] += float(t)
        per_model[mdl]['n_configs'] += 1
    # Try to grab dataset n and rep dim from meta if present (best-effort)
    n_samples = meta.get('n_samples')
    extras = {
        'n_results': len(all_r),
        'wall_seconds': float(sum(
            (r.get('time_seconds') or 0.0) for r in all_r if isinstance(r, dict))),
        'code_timestamp': meta.get('code_timestamp'),
        'mtime': path.stat().st_mtime,
        'per_model': per_model,
        'n_samples': n_samples,
    }
    return ('ok',
            f"n_results={len(all_r)} "
            f"models={sorted(best.keys())} "
            f"metric={meta.get('metric_name', '?')}",
            extras)


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
        # top_rep_dim caches have a dynamic per-model rep set / condition;
        # we don't reconstruct their expected meta, so skip the drift check.
        status, detail, extras = _check_file(path, expected_cheap=None)
        out.append((path.name, model, status, detail, extras))
    return out


def _sanitize(s):
    """Match find_best_hybrid._sanitize_filename_part — keep the cache
    filename lookup in sync with how precompute_hybrids writes them.
    Without this, tiers like 'top+borderline' would never match the file
    'top_borderline' that's actually on disk."""
    return str(s).replace('+', '_').replace('/', '_').replace(' ', '_')


def classify_dataset(dataset, cfg, searches_dir, include_distance):
    # Peek at any cached file to learn n_samples — drives whether
    # top/capped is expected. None ⇒ no cache yet ⇒ default to expecting.
    n_samples = _peek_n_samples(searches_dir, dataset)
    expected = expected_conditions(
        cfg, include_distance=include_distance, dataset_name=dataset,
        n_samples=n_samples)
    results = []
    n_ok = 0
    n_drift = 0
    for tier, strat in expected:
        fname = f"{dataset}__{_sanitize(tier)}__{_sanitize(strat)}__full.pkl"
        path = searches_dir / fname
        expected_cheap = _expected_meta_cheap(dataset, cfg, tier, strat)
        status, detail, extras = _check_file(path, expected_cheap=expected_cheap)
        results.append((tier, strat, status, detail, extras))
        if status == 'ok':
            n_ok += 1
        elif status == 'drift':
            n_drift += 1
    # c02 per-model top-rep-dim caches (additive — count separately, not
    # part of the strict precompute denominator since their expected count
    # is dynamic per dataset).
    top_rep_dim_files = find_top_rep_dim_files(dataset, searches_dir)
    # DONE means "present AND the current code will actually accept every
    # cache" — a drifted condition (engine rebuilds cold) is NOT done.
    if n_ok == len(expected) and expected:
        overall = 'DONE'
    elif n_drift and (n_ok + n_drift) == len(expected):
        # All conditions are present but at least one will be rebuilt cold.
        overall = 'drift'
    elif n_ok == 0 and n_drift == 0:
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
    ap.add_argument('--show-drift', action='store_true',
                    help='for drifted conditions, print which _cache_meta '
                         'field(s) differ from what the current code expects '
                         '(the fields the running engine would reject on load)')
    ap.add_argument('--timing', action='store_true',
                    help='print wall time + write date per ok condition '
                         '(uses time_seconds + mtime already in cache files)')
    ap.add_argument('--names-only', action='store_true',
                    help='print ONLY dataset names (one per line), no tables. '
                         'Combine with --done/--incomplete/--distance to get a '
                         'machine-readable list for scripting (submit wave).')
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

    # Machine-readable mode: print just the dataset names matching the
    # done/incomplete filter (used by submit_rerun_wave.sh to classify).
    if args.names_only:
        for ds in datasets:
            cfg = DATASET_REPS[ds]
            overall, n_ok, n_exp, _results, _trd = classify_dataset(
                ds, cfg, searches_dir, include_distance=args.distance)
            if n_exp == 0:
                continue
            if args.done and overall != 'DONE':
                continue
            if args.incomplete and overall == 'DONE':
                continue
            print(ds)
        return

    by_domain = {}
    for ds in datasets:
        by_domain.setdefault(_get_domain(ds), []).append(ds)

    totals = {'DONE': 0, 'drift': 0, 'partial': 0, 'missing': 0}

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
                for tier, strat, status, _, _ in results
                if status != 'ok'
            ]
            n_trd_ok = sum(1 for _, _, st, _, _ in top_rep_dim_files if st == 'ok')
            n_trd = len(top_rep_dim_files)
            trd_str = f"{n_trd_ok}/{n_trd}" if n_trd else '-'
            bad_str = ', '.join(bad) if bad else ''
            print(f"  {ds:<38} {overall:<8} {n_ok}/{n_exp:<6} {trd_str:<12} {bad_str}")
            if args.show_drift:
                for tier, strat, status, detail, _ in results:
                    if status == 'drift':
                        print(f"      ! {tier}/{strat} DRIFT — engine will "
                              f"rebuild cold: {detail}")
            if args.verbose:
                for tier, strat, status, detail, _ in results:
                    mark = '✓' if status == 'ok' else 'X'
                    print(f"      {mark} {tier}/{strat}: {status} {detail}")
                for fname, model, status, detail, _ in top_rep_dim_files:
                    mark = '✓' if status == 'ok' else 'X'
                    print(f"      {mark} top_rep_dim/{model}: {status} {detail}")
            if args.timing:
                for tier, strat, status, _, extras in results:
                    if status != 'ok':
                        continue
                    ws = extras.get('wall_seconds') or 0.0
                    nr = extras.get('n_results') or 0
                    mt = extras.get('mtime')
                    ns = extras.get('n_samples')
                    when = (datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M')
                            if mt else '?')
                    n_str = f" n={ns}" if ns else ""
                    print(f"      [time] {tier}/{strat}:{n_str} "
                          f"{ws/3600:.2f}h ({ws:.0f}s)  "
                          f"n_configs={nr}  written={when}")
                    pm = extras.get('per_model') or {}
                    # Sort by wall_seconds desc so slow models surface first
                    for mdl, d in sorted(pm.items(),
                                         key=lambda kv: -kv[1]['wall_seconds']):
                        mw = d['wall_seconds']
                        mn = d['n_configs']
                        per_cfg = (mw / mn) if mn else 0.0
                        share = (mw / ws * 100) if ws else 0.0
                        print(f"          {mdl:<10s} "
                              f"{mw/3600:6.2f}h ({mw:7.0f}s)  "
                              f"n_cfg={mn:<5d} "
                              f"per_cfg={per_cfg:6.2f}s  "
                              f"share={share:5.1f}%")
                for fname, model, status, _, extras in top_rep_dim_files:
                    if status != 'ok':
                        continue
                    ws = extras.get('wall_seconds') or 0.0
                    nr = extras.get('n_results') or 0
                    mt = extras.get('mtime')
                    when = (datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M')
                            if mt else '?')
                    print(f"      [time] top_rep_dim/{model}: "
                          f"{ws/3600:.2f}h ({ws:.0f}s)  "
                          f"n_configs={nr}  written={when}")

    total = sum(totals.values())
    print(f"\n{'='*70}")
    print(f"Total datasets checked: {total}  |  "
          f"DONE: {totals['DONE']}  |  "
          f"drift (will rebuild): {totals['drift']}  |  "
          f"partial: {totals['partial']}  |  "
          f"missing: {totals['missing']}")
    if totals['drift']:
        print(f"\nWARNING: {totals['drift']} dataset(s) have all conditions "
              f"present but at least one DRIFTS — the running engine will "
              f"reject and rebuild those caches cold. They are NOT DONE. "
              f"Run with --show-drift to see the offending _cache_meta "
              f"field(s).")
    if totals['DONE']:
        print(f"\nReady for experiments: {totals['DONE']}/{total} datasets")


if __name__ == '__main__':
    main()
