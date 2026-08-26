#!/usr/bin/env python
"""Check precompute status by reading the per-dataset SQLite result store.

Walks $KIRBY_CACHE_DIR/result_db/ (one <dataset>.db per dataset, written by
the cache redesign in result_cache.py) and, for each dataset in DATASET_REPS,
checks that the expected precompute CONDITIONS are present. A condition is a
(tier, aug_strategy) pair; in the SQLite store these are just rows in the
`results` table, so "present" means "≥1 row with that tier+aug_strategy".

Readiness is reported PER EXPERIMENT FAMILY, because the families do not
consume the same conditions. Two pools exist:

    MAIN  general/none, general/uncapped,
          general/capped   (only when n <= CAPPED_MAX_N)
    DIST  general_distance/{none,uncapped}

    complexity  needs MAIN   (only c03 also reads DIST)
    noise       needs MAIN   (no noise experiment touches the distance pool)
    similarity  needs MAIN + DIST

There is ONE rep pool per dataset ('general'); the 'top' tier was retired
2026-08-24. capped now rides on 'general' and is precomputed only for
n <= CAPPED_MAX_N. The per-family verdict below avoids a dataset whose ONLY gap
is the distance conditions looking "not done" when it is in fact fully ready for
complexity and noise.

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
    python tests/check_precompute_status.py --timing       # wall time per cond

    # --done/--incomplete filter on ONE family (default: complexity)
    python tests/check_precompute_status.py --family complexity --incomplete
    python tests/check_precompute_status.py --family similarity --done
    python tests/check_precompute_status.py --family all --incomplete

    # machine-readable, for the submit scripts
    python tests/check_precompute_status.py --family complexity \
        --incomplete --names-only   # -> needs a complexity job (it builds MAIN+DIST)
    python tests/check_precompute_status.py --family complexity \
        --done --names-only         # -> ready to run cx + noise now

    --distance is a back-compat alias for --family all.
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path


def _fresh(ts, now=None):
    """Human relative age of a unix timestamp ('30s', '4m', '3h', '5d').

    This is the 'is it alive?' signal: a 'DONE' dataset still being built
    shows a recent write (seconds/minutes), a finished idle one shows days.
    """
    if not ts:
        return '—'
    now = now if now is not None else time.time()
    dt = max(0.0, now - ts)
    if dt < 90:
        return f'{int(dt)}s'
    if dt < 5400:
        return f'{int(dt / 60)}m'
    if dt < 129600:
        return f'{int(dt / 3600)}h'
    return f'{int(dt / 86400)}d'

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))

from experiments.complexity.complexity_helpers import DATASET_REPS

# Registry carries per-dataset n_samples (populated when complexity_helpers ->
# _shared -> dataset_policy imported above). Used to decide whether general/capped
# is even expected for a dataset.
try:
    from kirby.registry import lookup as _registry_lookup
except Exception:
    _registry_lookup = None


def _n_samples_for(dataset):
    """Registry n for the dataset, or None if unknown."""
    if _registry_lookup is None:
        return None
    try:
        info = _registry_lookup(dataset)
        return getattr(info, 'n_samples', None) if info else None
    except Exception:
        return None


# Core precompute tiers (everything else found in a DB is an experiment tier).
CORE_TIER_PREFIXES = ('general',)

# general/capped is ONLY built for n <= CAPPED_MAX_N; precompute deliberately
# skips it above that (it blows the 7-day wall). n_samples now comes from the
# dataset registry (_n_samples_for). When n is unknown we assume LARGE and do
# NOT expect capped — otherwise every large dataset is perma-'partial' on a
# phantom condition it will never have (the old behaviour, which froze the DONE
# count: only small-n datasets could ever reach DONE).
#
# IMPORTED, not restated. This file carried its own copy of the number until
# 2026-08-26. Every other reader already takes it from experiments._shared, so a
# copy here meant the gate could be changed in one place and silently disagree
# with the checker that reports on it.
from experiments._shared import CAPPED_MAX_N  # noqa: E402

# Which precompute POOL each experiment family consumes. Established 2026-07-09
# by a static read of every c*/s*/n* experiment file:
#   - noise: every pool read passes distance=False, and noise_helpers never
#     precomputes a distance tier -> MAIN only.
#   - complexity: MAIN for c01..c24; c03_cross_dataset_transfer is the ONLY
#     complexity experiment that reads the distance pool (it is reported as a
#     footnote, not as a gate, so one experiment can't mark 23 others 'not done').
#   - similarity: s04/s05/s06/s07/s08/s10/s13 read the distance pool -> MAIN+DIST.
# NB a complexity JOB *builds* both pools (complexity_helpers.precompute_dataset
# precomputes general_distance), which is why complexity is the
# universal builder that similarity can depend on.
FAMILY_NEEDS = {
    'complexity': ('main',),
    'noise': ('main',),
    'similarity': ('main', 'dist'),
    'all': ('main', 'dist'),
}
FAMILIES = ('complexity', 'similarity', 'noise')


def _pool_of(tier: str) -> str:
    """'dist' for the distance lineage, else 'main'."""
    return 'dist' if tier.endswith('_distance') else 'main'


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


def main_conditions(cfg, n_samples=None):
    """Expected (tier, aug_strategy) tuples in the MAIN pool.

    n_samples (when known) controls whether general/capped is expected: only
    datasets with n <= CAPPED_MAX_N have it. Unknown n -> assume LARGE and
    do NOT expect capped, otherwise every large dataset is perma-'partial' on a
    phantom condition it will never have.
    """
    # One 10-rep pool ('general'); the 'top' tier is retired (2026-08-24).
    # 'general' now carries capped (n <= CAPPED_MAX_N only).
    general = cfg.get('general', [])
    aug = cfg.get('augmentation', [])
    expected = []

    if len(general) >= 2:
        expected.append(('general', 'none'))
        if aug:
            expected.append(('general', 'uncapped'))
            if n_samples is not None and n_samples <= CAPPED_MAX_N:
                expected.append(('general', 'capped'))
    elif len(general) == 1 and aug:
        expected.append(('general', 'uncapped'))

    return expected


def dist_conditions(cfg):
    """Expected (tier, aug_strategy) tuples in the DIST pool.

    Consumed only by the similarity distance experiments and complexity c03.
    Distance precompute never emits 'capped' (no consumer needs it).
    """
    general = cfg.get('general', [])
    aug = cfg.get('augmentation', [])
    expected = []

    if len(general) >= 2:
        expected.append(('general_distance', 'none'))
        if aug:
            expected.append(('general_distance', 'uncapped'))

    return expected


def expected_conditions(cfg, include_distance=False, n_samples=None):
    """Back-compat shim: MAIN pool, plus DIST when include_distance.

    Kept because tests/slurm_scripts/resubmit_missing*.sh import this name.
    New code should call main_conditions/dist_conditions directly.
    """
    expected = main_conditions(cfg, n_samples=n_samples)
    if include_distance:
        expected += dist_conditions(cfg)
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


def _scan_pool(conds, cond_map):
    """Check one pool's conditions against the DB. -> (rows, n_ok)."""
    rows = []
    n_ok = 0
    for tier, strat in conds:
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
            rows.append((tier, strat, 'ok',
                         f"n_configs={d['n_configs']}", extras))
        else:
            rows.append((tier, strat, 'missing', '', {}))
    return rows, n_ok


def classify_dataset(dataset, cfg, db_dir, include_distance=False):
    """Return (families, results, exp_tiers).

    families: {family: (status, n_ok, n_expected)} for complexity/similarity/
        noise/all — status is 'DONE' | 'partial' | 'missing'.
    results: [(tier, aug, status, detail, extras)] across BOTH pools, so
        --verbose/--timing show every condition regardless of family.
    exp_tiers: {experiment_tier: n_configs}.

    include_distance is accepted and ignored — the pools are always both
    scanned now, and the family map decides what counts as expected.
    """
    db_path = db_dir / f'{dataset}.db'
    cond_map, exp_tiers = ({}, {})
    if db_path.exists():
        cond_map, exp_tiers = _scan_db(db_path)

    n_samples = _n_samples_for(dataset)
    pools = {
        'main': main_conditions(cfg, n_samples=n_samples),
        'dist': dist_conditions(cfg),
    }
    scanned = {p: _scan_pool(c, cond_map) for p, c in pools.items()}

    # A dataset with literally nothing in its DB is 'missing', not 'partial'.
    has_any = bool(cond_map) or bool(exp_tiers)

    families = {}
    for family, needed in FAMILY_NEEDS.items():
        n_ok = sum(scanned[p][1] for p in needed)
        n_exp = sum(len(pools[p]) for p in needed)
        if n_exp and n_ok == n_exp:
            status = 'DONE'
        elif n_ok == 0 and not has_any:
            status = 'missing'
        else:
            status = 'partial'
        families[family] = (status, n_ok, n_exp)

    results = scanned['main'][0] + scanned['dist'][0]
    return families, results, exp_tiers


# Reduced near-term keep-set (2026-07-08) — MIRROR of the KEEP_DS list in
# tests/slurm_scripts/cancel_to_keepset.sh. Keep the two in sync.
PRIORITY_DATASETS = [
    # Molecular (5)
    'adme_mdr1', 'adme_hlm', 'caco2', 'herg_ki', 'qm9',
    # Antibody (13)
    'flab_expression_koenig', 'ddg_combined', 'abagym_EGFR', 'abcov_kd',
    'ginkgo_hic', 'ginkgo_tm2', 'ginkgo_titer', 'ginkgo_pr_cho',
    'ginkgo_acsins', 'ginkgo_colloidal', 'ginkgo_purity', 'ginkgo_aggregation',
    # Other domains (2)
    'nt_promoter_tata', 'gb_human_enhancers_cohn',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', help='check a single dataset')
    ap.add_argument('--datasets', help='comma-separated list of datasets to check')
    ap.add_argument('--priority', action='store_true',
                    help=f'check ONLY the {len(PRIORITY_DATASETS)} near-term '
                         'keep-set datasets (mirror of cancel_to_keepset.sh)')
    ap.add_argument('--family', choices=sorted(FAMILY_NEEDS.keys()),
                    default='complexity',
                    help='which experiment family --done/--incomplete/'
                         '--names-only filter on (default: complexity, the '
                         'family whose job builds both pools). '
                         "'all' = MAIN+DIST, the old --distance behaviour.")
    ap.add_argument('--done', action='store_true',
                    help='show only datasets DONE for --family')
    ap.add_argument('--incomplete', action='store_true',
                    help='show only datasets partial/missing for --family')
    ap.add_argument('--distance', action='store_true',
                    help='back-compat alias for --family all')
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

    # --distance predates the per-family verdict; it meant "require the distance
    # tiers too", which is exactly --family all.
    if args.distance:
        args.family = 'all'

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
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
        bad = [d for d in datasets if d not in DATASET_REPS]
        if bad:
            print(f"ERROR: not in DATASET_REPS: {', '.join(bad)}")
            sys.exit(1)
    elif args.priority:
        datasets = [d for d in PRIORITY_DATASETS if d in DATASET_REPS]
        missing = [d for d in PRIORITY_DATASETS if d not in DATASET_REPS]
        if missing:
            print(f"WARNING: priority datasets not in DATASET_REPS: {', '.join(missing)}")
    else:
        datasets = sorted(DATASET_REPS.keys())

    if args.names_only:
        for ds in datasets:
            cfg = DATASET_REPS[ds]
            families, _r, _e = classify_dataset(ds, cfg, db_dir)
            overall, _n_ok, n_exp = families[args.family]
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

    # Per-family precompute tallies (the old single tally hid that a dataset can
    # be ready for one family and not another).
    totals = {f: {'DONE': 0, 'partial': 0, 'missing': 0} for f in FAMILIES}
    exp_grand = {c: [0, 0] for c in EXP_CATEGORIES}  # [present, total]

    def _cell(fam_tuple):
        status, n_ok, n_exp = fam_tuple
        if n_exp == 0:
            return '—'
        return 'DONE' if status == 'DONE' else f'{n_ok}/{n_exp}'

    for domain in sorted(by_domain.keys()):
        rows = []
        for ds in by_domain[domain]:
            cfg = DATASET_REPS[ds]
            families, results, exp_tiers = classify_dataset(ds, cfg, db_dir)
            sel_status, _sel_ok, sel_exp = families[args.family]
            if sel_exp == 0:
                continue
            for f in FAMILIES:
                totals[f][families[f][0]] += 1
            if args.done and sel_status != 'DONE':
                continue
            if args.incomplete and sel_status == 'DONE':
                continue
            rows.append((ds, families, results, exp_tiers))

        if not rows:
            continue

        print(f"\n=== {domain.upper()} ===")
        if show_exp:
            print(f"  {'dataset':<34} {'cx-pre':<7} {'sim-pre':<8} {'nz-pre':<7} "
                  f"{'rows':<8} {'fresh':<6} "
                  f"{'cx':<7} {'sim':<7} {'noise':<7} missing-for-{args.family}")
            print(f"  {'-'*34} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*6} "
                  f"{'-'*7} {'-'*7} {'-'*7} {'-'*30}")
        else:
            print(f"  {'dataset':<34} {'cx-pre':<7} {'sim-pre':<8} {'nz-pre':<7} "
                  f"{'rows':<8} {'fresh':<6} missing-for-{args.family}")
            print(f"  {'-'*34} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*40}")
        for ds, families, results, exp_tiers in rows:
            # Only report gaps that matter for the family being asked about, so
            # a distance-only gap never reads as "complexity not done".
            needed = FAMILY_NEEDS[args.family]
            missing = [f"{tier}/{strat}"
                       for tier, strat, status, _, _ in results
                       if status != 'ok' and _pool_of(tier) in needed]
            miss_str = ', '.join(missing) if missing else ''
            # Row fill + write-freshness across present conditions. This is the
            # antidote to the misleading binary DONE: a 'DONE' dataset still
            # being built shows a large-but-growing row count and a fresh write
            # time, so 'DONE' no longer means 'finished' silently.
            total_rows = sum((extras.get('n_results') or 0)
                             for _t, _s, st, _d, extras in results if st == 'ok')
            mtimes = [extras.get('mtime')
                      for _t, _s, st, _d, extras in results
                      if st == 'ok' and extras.get('mtime')]
            fresh = _fresh(max(mtimes)) if mtimes else '—'
            rows_str = f"{total_rows}"
            cxp = _cell(families['complexity'])
            simp = _cell(families['similarity'])
            nzp = _cell(families['noise'])
            if show_exp:
                e = _scan_experiments(ds, args.results_dir, ids_by_cat)
                for c in EXP_CATEGORIES:
                    exp_grand[c][0] += e[c][0]
                    exp_grand[c][1] += e[c][1]
                cx = f"{e['complexity'][0]}/{e['complexity'][1]}"
                sim = f"{e['similarity'][0]}/{e['similarity'][1]}"
                noi = f"{e['noise'][0]}/{e['noise'][1]}"
                print(f"  {ds:<34} {cxp:<7} {simp:<8} {nzp:<7} "
                      f"{rows_str:<8} {fresh:<6} "
                      f"{cx:<7} {sim:<7} {noi:<7} {miss_str}")
            else:
                print(f"  {ds:<34} {cxp:<7} {simp:<8} {nzp:<7} "
                      f"{rows_str:<8} {fresh:<6} {miss_str}")
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

    print(f"\n{'='*70}")
    print("PRECOMPUTE readiness, per experiment family "
          "(complexity needs MAIN; similarity needs MAIN+DIST; noise needs MAIN)")
    for f in FAMILIES:
        t = totals[f]
        n = sum(t.values())
        note = '  [c03 also needs DIST]' if f == 'complexity' else ''
        print(f"  {f:<11}: DONE {t['DONE']:>3}  |  partial {t['partial']:>3}  "
              f"|  missing {t['missing']:>3}   (of {n}){note}")
    print(f"  filtering on --family {args.family}")
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
