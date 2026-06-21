#!/usr/bin/env python
"""Check precompute status by reading the per-dataset SQLite result store.

Walks $KIRBY_CACHE_DIR/result_db/ (one <dataset>.db per dataset, written by
the cache redesign in result_cache.py) and, for each dataset in DATASET_REPS,
checks that the expected precompute CONDITIONS are present. A condition is a
(tier, aug_strategy) pair; in the SQLite store these are just rows in the
`results` table, so "present" means "≥1 row with that tier+aug_strategy".

Standard expected conditions (drives the strict found/expected ratio):
    general__none,  general__uncapped,  general__capped
    top__none,      top__uncapped,      top__capped

Datasets without augmentation get the 'none' conditions only. Distance
conditions (general_distance / top_distance) are not checked by default —
pass --distance to include them.

The redesign removed cache versioning/drift entirely (a layout change means
"wipe + rebuild", never a migration), so there is no longer a 'drift' state:
a condition is simply present or missing.

EXPERIMENT STAGE: rows whose tier is an experiment tier (e.g. c06_variant,
c21_aac, s03_*) live in the same DB. The 'exp' column counts how many such
experiment tiers a dataset has — i.e. whether it has moved past precompute
into the experiments themselves.

Usage:
    python tests/check_precompute_status.py                # full report
    python tests/check_precompute_status.py --dataset NAME # one dataset
    python tests/check_precompute_status.py --done         # only DONE
    python tests/check_precompute_status.py --incomplete   # only not-DONE
    python tests/check_precompute_status.py --distance     # include distance
    python tests/check_precompute_status.py --timing       # wall time per cond
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

from experiments.complexity.complexity_helpers import DATASET_REPS

# Core precompute tiers (everything else found in a DB is an experiment tier).
CORE_TIER_PREFIXES = ('general', 'top')

# Top/capped is only expected when n_samples <= TOP_CAPPED_MAX_N. The SQLite
# store does not carry n_samples, so we expect top/capped conservatively (a
# dataset that legitimately skipped capped for large n will show as 'partial'
# on that one condition — same conservative signal the old tool gave when it
# could not peek n).
TOP_CAPPED_MAX_N = 250


def _result_db_dir() -> Path:
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return cache_root / 'result_db'


# ── Experiment progress (the SECOND question) ─────────────────────────────
# Precompute = the cached SEARCHES (result_db, above). Experiments = the c/s/n
# runs that consume those searches and save to tests/results/<cat>/<exp>/<ds>.json.
# A job builds precompute first, then runs experiments, so a dataset can be
# precompute-DONE with zero experiments run. We surface BOTH here so this is the
# only status tool you need.
EXP_CATEGORIES = ('complexity', 'similarity', 'noise')


def _exp_ids_by_cat():
    """Experiment IDs per category, from each helper (source of truth).
    If a helper can't import, that category is reported as unavailable (None)."""
    import importlib
    out = {}
    for cat in EXP_CATEGORIES:
        try:
            mod = importlib.import_module(f'experiments.{cat}.{cat}_helpers')
            out[cat] = list(mod.SINGLE_DATASET_EXPERIMENTS)
        except Exception:
            out[cat] = None
    return out


def _scan_experiments(dataset, results_dir, ids_by_cat):
    """{category: (n_present, n_total)} of experiment JSONs for a dataset."""
    out = {}
    for cat in EXP_CATEGORIES:
        ids = ids_by_cat.get(cat)
        if not ids:
            out[cat] = (0, 0)
            continue
        cat_root = Path(results_dir) / cat
        dirs = list(cat_root.iterdir()) if cat_root.is_dir() else []
        present = 0
        for eid in ids:
            # save_experiment_results writes <results>/<cat>/<exp_dir>/<ds>.json;
            # exp_dir starts with the short id (e.g. 'c02' -> 'c02_complexity_grid').
            for d in dirs:
                if (d.is_dir() and d.name.split('_')[0] == eid
                        and (d / f'{dataset}.json').exists()):
                    present += 1
                    break
        out[cat] = (present, len(ids))
    return out


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


def expected_conditions(cfg, include_distance=False, n_samples=None):
    """Return list of expected (tier, aug_strategy) tuples for a dataset.

    n_samples (when known) controls whether top/capped is expected: only
    datasets with n <= TOP_CAPPED_MAX_N have it. Pass None to expect
    top/capped (the conservative default — the SQLite store has no n).
    """
    general = cfg.get('general', [])
    top = cfg.get('top_tier', [])
    aug = cfg.get('augmentation', [])
    expected = []

    if len(general) >= 2:
        expected.append(('general', 'none'))
        if aug:
            expected.append(('general', 'uncapped'))
    elif len(general) == 1 and aug:
        expected.append(('general', 'uncapped'))

    if len(top) >= 2:
        expected.append(('top', 'none'))
        if aug:
            expected.append(('top', 'uncapped'))
            if n_samples is None or n_samples <= TOP_CAPPED_MAX_N:
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


def _is_experiment_tier(tier: str) -> bool:
    """True for experiment tiers (c06_variant, c21_aac, s03_...) — anything
    that is NOT one of the core general*/top* precompute tiers."""
    return not tier.startswith(CORE_TIER_PREFIXES)


def _scan_db(db_path: Path):
    """Read one dataset's result DB.

    Returns (cond_map, exp_tiers) where:
      cond_map[(tier, aug_strategy)] = {
          'n_configs', 'wall_seconds', 'mtime', 'per_model': {model: {...}}}
      exp_tiers = {tier: n_configs} for experiment tiers present.
    """
    cond_map = {}
    exp_tiers = {}
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    except sqlite3.OperationalError:
        return cond_map, exp_tiers
    try:
        cur = conn.execute(
            'SELECT tier, aug_strategy, model, '
            'COUNT(*), SUM(time_seconds), MAX(created_at) '
            'FROM results GROUP BY tier, aug_strategy, model')
        rows = cur.fetchall()
    except sqlite3.Error:
        conn.close()
        return cond_map, exp_tiers
    conn.close()

    for tier, aug, model, n_cfg, wall, mtime in rows:
        aug = aug or 'none'
        wall = wall or 0.0
        if _is_experiment_tier(tier):
            exp_tiers[tier] = exp_tiers.get(tier, 0) + (n_cfg or 0)
            continue
        key = (tier, aug)
        d = cond_map.setdefault(
            key, {'n_configs': 0, 'wall_seconds': 0.0, 'mtime': None,
                  'per_model': {}})
        d['n_configs'] += (n_cfg or 0)
        d['wall_seconds'] += wall
        if mtime and (d['mtime'] is None or mtime > d['mtime']):
            d['mtime'] = mtime
        d['per_model'][model or '?'] = {
            'n_configs': n_cfg or 0, 'wall_seconds': wall}
    return cond_map, exp_tiers


def classify_dataset(dataset, cfg, db_dir, include_distance):
    """Return (overall, n_ok, n_expected, results, exp_tiers).

    results items: (tier, aug, status, detail, extras) — status is 'ok' or
    'missing'. extras mirrors the old tool's timing fields so --timing works.
    exp_tiers: dict {experiment_tier: n_configs}.
    """
    db_path = db_dir / f'{dataset}.db'
    cond_map, exp_tiers = ({}, {})
    if db_path.exists():
        cond_map, exp_tiers = _scan_db(db_path)

    expected = expected_conditions(
        cfg, include_distance=include_distance, n_samples=None)
    results = []
    n_ok = 0
    for tier, strat in expected:
        d = cond_map.get((tier, strat))
        if d:
            n_ok += 1
            extras = {
                'wall_seconds': d['wall_seconds'],
                'n_results': d['n_configs'],
                'mtime': d['mtime'],
                'n_samples': None,
                'per_model': d['per_model'],
            }
            results.append((tier, strat, 'ok',
                            f"n_configs={d['n_configs']}", extras))
        else:
            results.append((tier, strat, 'missing', '', {}))

    if expected and n_ok == len(expected):
        overall = 'DONE'
    elif n_ok == 0 and not exp_tiers:
        overall = 'missing'
    else:
        overall = 'partial'
    return overall, n_ok, len(expected), results, exp_tiers


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
    ap.add_argument('--timing', action='store_true',
                    help='print wall time + write date per ok condition '
                         '(from time_seconds + created_at in the result DB)')
    ap.add_argument('--results-dir', default='tests/results',
                    help='base experiment-results dir (default: tests/results); '
                         'drives the cx/sim/noise experiment columns')
    ap.add_argument('--no-experiments', action='store_true',
                    help='hide the experiment columns (precompute only)')
    ap.add_argument('--names-only', action='store_true',
                    help='print ONLY dataset names (one per line), no tables. '
                         'Combine with --done/--incomplete for scripting.')
    # Accepted for backward compatibility; the redesign removed drift.
    ap.add_argument('--show-drift', action='store_true',
                    help='(no-op: the SQLite store has no drift state)')
    args = ap.parse_args()

    db_dir = _result_db_dir()
    if not db_dir.exists():
        print(f"No result cache at {db_dir}")
        print("  (will be created on first precompute run)")
        sys.exit(0 if args.incomplete or not args.done else 1)

    if args.dataset:
        if args.dataset not in DATASET_REPS:
            print(f"ERROR: '{args.dataset}' not in DATASET_REPS")
            sys.exit(1)
        datasets = [args.dataset]
    else:
        datasets = sorted(DATASET_REPS.keys())

    if args.names_only:
        for ds in datasets:
            cfg = DATASET_REPS[ds]
            overall, n_ok, n_exp, _r, _e = classify_dataset(
                ds, cfg, db_dir, include_distance=args.distance)
            if n_exp == 0:
                continue
            if args.done and overall != 'DONE':
                continue
            if args.incomplete and overall == 'DONE':
                continue
            print(ds)
        return

    show_exp = not args.no_experiments
    ids_by_cat = _exp_ids_by_cat() if show_exp else {}

    by_domain = {}
    for ds in datasets:
        by_domain.setdefault(_get_domain(ds), []).append(ds)

    totals = {'DONE': 0, 'partial': 0, 'missing': 0}
    exp_grand = {c: [0, 0] for c in EXP_CATEGORIES}  # [present, total]

    for domain in sorted(by_domain.keys()):
        rows = []
        for ds in by_domain[domain]:
            cfg = DATASET_REPS[ds]
            overall, n_ok, n_exp, results, exp_tiers = classify_dataset(
                ds, cfg, db_dir, include_distance=args.distance)
            if n_exp == 0:
                continue
            totals[overall] += 1
            if args.done and overall != 'DONE':
                continue
            if args.incomplete and overall == 'DONE':
                continue
            rows.append((ds, overall, n_ok, n_exp, results, exp_tiers))

        if not rows:
            continue

        print(f"\n=== {domain.upper()} ===")
        if show_exp:
            print(f"  {'dataset':<34} {'precomp':<8} {'pre#':<6} "
                  f"{'cx':<7} {'sim':<7} {'noise':<7} missing-precompute")
            print(f"  {'-'*34} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*30}")
        else:
            print(f"  {'dataset':<34} {'precomp':<8} {'pre#':<6} missing-precompute")
            print(f"  {'-'*34} {'-'*8} {'-'*6} {'-'*40}")
        for ds, overall, n_ok, n_exp, results, exp_tiers in rows:
            missing = [f"{tier}/{strat}"
                       for tier, strat, status, _, _ in results
                       if status != 'ok']
            miss_str = ', '.join(missing) if missing else ''
            pre_n = f"{n_ok}/{n_exp}"
            if show_exp:
                e = _scan_experiments(ds, args.results_dir, ids_by_cat)
                for c in EXP_CATEGORIES:
                    exp_grand[c][0] += e[c][0]
                    exp_grand[c][1] += e[c][1]
                cx = f"{e['complexity'][0]}/{e['complexity'][1]}"
                sim = f"{e['similarity'][0]}/{e['similarity'][1]}"
                noi = f"{e['noise'][0]}/{e['noise'][1]}"
                print(f"  {ds:<34} {overall:<8} {pre_n:<6} "
                      f"{cx:<7} {sim:<7} {noi:<7} {miss_str}")
            else:
                print(f"  {ds:<34} {overall:<8} {pre_n:<6} {miss_str}")
            if args.verbose:
                for tier, strat, status, detail, _ in results:
                    mark = '✓' if status == 'ok' else 'X'
                    print(f"      {mark} {tier}/{strat}: {status} {detail}")
                if exp_tiers:
                    et = ', '.join(f"{t}({n})" for t, n in sorted(exp_tiers.items()))
                    print(f"      experiments: {et}")
            if args.timing:
                for tier, strat, status, _, extras in results:
                    if status != 'ok':
                        continue
                    ws = extras.get('wall_seconds') or 0.0
                    nr = extras.get('n_results') or 0
                    mt = extras.get('mtime')
                    when = (datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M')
                            if mt else '?')
                    print(f"      [time] {tier}/{strat}: "
                          f"{ws/3600:.2f}h ({ws:.0f}s)  "
                          f"n_configs={nr}  written={when}")
                    pm = extras.get('per_model') or {}
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

    total = sum(totals.values())
    print(f"\n{'='*70}")
    print(f"PRECOMPUTE  datasets: {total}  |  "
          f"DONE: {totals['DONE']}  |  "
          f"partial: {totals['partial']}  |  "
          f"missing: {totals['missing']}")
    if totals['DONE']:
        print(f"  ({totals['DONE']}/{total} datasets have precompute built)")
    if show_exp:
        print(f"EXPERIMENTS (results JSONs in {args.results_dir}):")
        for c in EXP_CATEGORIES:
            p, e = exp_grand[c]
            if ids_by_cat.get(c) is None:
                print(f"  {c:<11}: (helper import failed — run on the server)")
            else:
                print(f"  {c:<11}: {p}/{e} run")


if __name__ == '__main__':
    main()
