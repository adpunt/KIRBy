#!/usr/bin/env python3
"""Check one prelim noise run that was made twice with identical settings.

Plan section 7, item 2. Reads what two runs of the same (dataset, experiment,
condition, fold, level) left on disk and prints three things:

  (a) FINISHED   each run's exit code (when given), its run_one manifest
                 status, the result JSON, and the per-fold output rows (JSONL,
                 d_vs_dprime.ROW_FIELDS) for every fold that ran. The rows of
                 run 2 must equal the rows of run 1.
  (b) CACHE      hits against misses per cache kind, from the two
                 KIRBY_NOISE_CACHE_REPORT files noise_cache.py writes at exit.
                 Run 2 must be all hits: any miss or bypass is a FAIL.
  (c) INPUTS     what the result file says was read -- recipes, representations,
                 budget, pruning mode, tier, metric and direction -- beside what
                 the precompute stored for that dataset. The stored side is read
                 with the calls the experiments make (precompute_hybrid_searches,
                 then get_best_search_for_downstream through
                 pool_recipe_per_downstream_model, then noise_base_reps_by_model),
                 inside a READ-ONLY GUARD: every function in find_best_hybrid and
                 kirby.hybrid that fits a model or scores features is replaced by
                 one that raises, and KIRBY_NO_COLD_SEARCH=1.0 refuses any search
                 with less than every configuration stored. Anything the engine
                 tried to compute is counted and printed as a FAIL: the stored
                 search does not cover what the experiment asks for. Nothing is
                 ever searched.

Exit code 1 on any FAIL, 0 otherwise.

Usage (what check_one.sbatch runs):
    python tests/experiments/noise/check_noise_run.py \
        --dataset caco2 --experiment n07 --condition outlier_p10 \
        --out1 $PRELIM/run1 --out2 $PRELIM/run2 \
        --report1 $PRELIM/cache_run1.json --report2 $PRELIM/cache_run2.json \
        --rc1 0 --rc2 0 --pruning-mode calibrated

    --no-stored skips the stored-search half of (c): no data is loaded.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))

# Kept in step with d_vs_dprime.py by importing it; it has no heavy imports.
from experiments.noise.d_vs_dprime import KEY_FIELDS, ROW_FIELDS  # noqa: E402

SCORE_FIELDS = ('score_D', 'score_Dprime', 'score_clean', 'gain')

# Keys in a result JSON that name what the run read. Printed with their path.
RECIPE_KEYS = ('tier', 'aug_condition', 'objective', 'objective_choice',
               'pool_subset', 'budget', 'hybrid_budget', 'blueprint',
               'allocation_method', 'importance_method', 'internal_evaluator',
               'augmentation_strategy', 'base_reps_by_model', 'base_reps',
               'hybrid_available', 'hybrid_unavailable_by_model',
               'hybrid_unavailable_reason', 'pruning_mode', 'speed',
               'search_speed', 'models', 'downstream_models')


class Verdict:
    def __init__(self):
        self.fails = []

    def fail(self, section, msg):
        self.fails.append(f"{section}: {msg}")
        print(f"  FAIL  {msg}")

    @staticmethod
    def ok(msg):
        print(f"  PASS  {msg}")

    @staticmethod
    def note(msg):
        print(f"  NOTE  {msg}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        return {'_unreadable': f'{type(exc).__name__}: {exc}'}


def _short(v, n=160):
    s = json.dumps(v, default=str, sort_keys=True) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n - 3] + '...'


def _walk(obj, path=''):
    """Yield (path, key, value) for every dict entry, depth first."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, k, v
            yield from _walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            yield from _walk(v, f"{path}[{i}]")


def result_json_path(out_dir, experiment, dataset, condition, exp_dir=None):
    sfx = ''
    if condition:
        clean = re.sub(r'[^A-Za-z0-9._-]+', '_', str(condition)).strip('_')
        sfx = f"__{clean}" if clean else ''
    dirs = ([os.path.join(out_dir, exp_dir)] if exp_dir else
            sorted(glob.glob(os.path.join(out_dir, f"{experiment}_*"))))
    for d in dirs:
        p = os.path.join(d, f"{dataset}{sfx}.json")
        if os.path.exists(p):
            return p
    return os.path.join(dirs[0], f"{dataset}{sfx}.json") if dirs else None


def read_rows(out_dir, dataset, experiment):
    """Every complete JSONL row under out_dir for this dataset and experiment."""
    rows, files = [], []
    for p in sorted(glob.glob(os.path.join(out_dir, '**', '*.jsonl'),
                              recursive=True)):
        n_before = len(rows)
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a half-written last line
                if (r.get('dataset') == dataset and
                        str(r.get('experiment', '')).split('_')[0] == experiment):
                    rows.append(r)
        if len(rows) > n_before:
            files.append((p, len(rows) - n_before))
    return rows, files


def _key(r):
    return tuple(json.dumps(r.get(f), sort_keys=True) for f in KEY_FIELDS)


def _same(a, b):
    if a is None or b is None:
        return a is b
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    return fa == fb


# ---------------------------------------------------------------------------
# (a) finished, rows on disk
# ---------------------------------------------------------------------------
# Experiments that write no D vs D' rows by design: they measure something and
# change no training data (grep for a row writer finds none in these files,
# 2026-09-27). For them "no rows" is not a failure; the result file is the check.
NO_ROWS_BY_DESIGN = {'c10', 'n01', 'n06', 'n14', 'n15', 'n16', 'n26', 'n37'}


def check_finished(args, V):
    print("\n(a) BOTH RUNS FINISHED, ROWS ON DISK FOR EVERY FOLD THAT RAN")
    per_run = {}
    for tag, out, rc in (('run 1', args.out1, args.rc1), ('run 2', args.out2, args.rc2)):
        man = _load_json(os.path.join(out, '_run_one_manifest.json')) or {}
        if rc is not None:
            (V.ok if rc == 0 else lambda m: V.fail('a', m))(
                f"{tag}: exit code {rc}")
        if man:
            st = man.get('status')
            msg = (f"{tag}: run_one status '{st}'"
                   + (f" -- {man.get('error')}" if man.get('error') else ''))
            if st in ('ok', 'skipped'):
                V.ok(msg)
            else:
                V.fail('a', msg)
        else:
            V.note(f"{tag}: no _run_one_manifest.json in {out} (run not made "
                   f"through run_one.py, or it died before writing one)")
        # run_one drops the condition for an experiment that names its own, and
        # the filename then carries none; the manifest says which was used.
        cond = man['condition'] if 'condition' in man else args.condition
        rj = result_json_path(out, args.experiment, args.dataset, cond,
                              man.get('experiment_dir'))
        res = _load_json(rj)
        if res is None:
            V.fail('a', f"{tag}: no result file ({rj})")
        elif '_unreadable' in res:
            V.fail('a', f"{tag}: result file unreadable ({rj}): {res['_unreadable']}")
        else:
            st = res.get('result_status')
            msg = f"{tag}: result file {rj} (result_status={st!r})"
            (V.ok if st in ('ok', 'skipped') else lambda m: V.fail('a', m))(msg)
            if st == 'skipped' or res.get('skipped'):
                # The experiment decided this dataset is out of its scope and
                # said why (e.g. regression-only, or needs repeat measurements).
                # That is a pass for a crash-and-cache check, not a failure.
                args._skipped = True
                V.note(f"{tag}: SKIPPED by the experiment -- "
                       f"{_short(res.get('reason') or res.get('skip_reason') or res.get('why') or '')}")
        rows, files = read_rows(out, args.dataset, args.experiment)
        per_run[tag] = {'manifest': man, 'result': res, 'result_path': rj,
                        'rows': rows}
        for p, n in files:
            print(f"        {n} rows in {p}")

        # Which folds were expected: the fold run_one kept, else --expect-folds,
        # else every fold of the result file's split.
        if args.expect_folds:
            want = {int(x) for x in args.expect_folds.split(',')}
        elif man.get('fold') is not None:
            want = {int(man['fold'])}
        else:
            n = ((res or {}).get('input_fingerprint') or {}).get('n_splits')
            want = set(range(int(n))) if n else None
        got = {r.get('fold') for r in rows}
        _no_rows_ok = (args.experiment in NO_ROWS_BY_DESIGN
                       or getattr(args, '_skipped', False))
        if not rows and _no_rows_ok:
            V.note(f"{tag}: no output rows, as expected "
                   f"({'skipped' if getattr(args, '_skipped', False) else 'this experiment writes none'})")
        elif not rows:
            V.fail('a', f"{tag}: no output rows (*.jsonl with dataset="
                        f"{args.dataset}, experiment={args.experiment}) under {out}")
        elif want is None:
            V.note(f"{tag}: {len(rows)} rows, folds {sorted(got, key=str)}; the "
                   f"expected fold set is unknown")
        else:
            missing = sorted(want - got)
            if missing:
                V.fail('a', f"{tag}: {len(rows)} rows, folds {sorted(got, key=str)}; "
                            f"no rows for fold(s) {missing}")
            else:
                V.ok(f"{tag}: {len(rows)} rows, every expected fold present "
                     f"({sorted(want)})")
        bad = [r for r in rows if set(r) - set(ROW_FIELDS)
               or set(ROW_FIELDS) - set(r)]
        if bad:
            V.fail('a', f"{tag}: {len(bad)} rows do not carry exactly "
                        f"d_vs_dprime.ROW_FIELDS")

    r1 = {_key(r): r for r in per_run['run 1']['rows']}
    r2 = {_key(r): r for r in per_run['run 2']['rows']}
    if r1 or r2:
        only1, only2 = set(r1) - set(r2), set(r2) - set(r1)
        differ = [k for k in set(r1) & set(r2)
                  if not all(_same(r1[k].get(f), r2[k].get(f)) for f in SCORE_FIELDS)]
        if only1 or only2 or differ:
            V.fail('a', f"run 2 rows do not match run 1: {len(only1)} only in "
                        f"run 1, {len(only2)} only in run 2, {len(differ)} with a "
                        f"different score_D, score_Dprime, score_clean or gain")
            for k in differ[:5]:
                print(f"        {dict(zip(KEY_FIELDS, k))}")
                print(f"          run 1 {[r1[k].get(f) for f in SCORE_FIELDS]}")
                print(f"          run 2 {[r2[k].get(f) for f in SCORE_FIELDS]}")
        else:
            V.ok(f"run 2's {len(r2)} rows equal run 1's, score for score")
    return per_run


# ---------------------------------------------------------------------------
# (b) cache
# ---------------------------------------------------------------------------
def check_cache(args, V):
    print("\n(b) CACHE: HITS AGAINST MISSES PER KIND (run 2 must be all hits)")
    rep1, rep2 = _load_json(args.report1), _load_json(args.report2)
    for tag, path, rep in (('run 1', args.report1, rep1), ('run 2', args.report2, rep2)):
        if rep is None:
            print(f"  {tag}: no report at {path}. noise_cache writes one only when "
                  f"the process made at least one cache call and exited normally.")
        elif '_unreadable' in rep:
            V.fail('b', f"{tag}: report unreadable ({path}): {rep['_unreadable']}")
    rep1 = rep1 if isinstance(rep1, dict) and '_unreadable' not in rep1 else {}
    rep2 = rep2 if isinstance(rep2, dict) and '_unreadable' not in rep2 else {}
    if not rep1 and not rep2:
        if getattr(args, '_skipped', False):
            V.note("no cache calls: the experiment skipped this dataset")
            return
        V.fail('b', "neither run recorded a cache call, so this experiment does "
                    "not go through noise_cache yet and run 2 cannot be shown to "
                    "be all hits")
        return
    kinds = sorted(set(rep1) | set(rep2))
    print(f"  {'kind':<22} {'run 1 hits':>10} {'misses':>7} {'bypass':>7}   "
          f"{'run 2 hits':>10} {'misses':>7} {'bypass':>7}")
    for k in kinds:
        a, b = rep1.get(k, {}), rep2.get(k, {})
        print(f"  {k:<22} {a.get('hits', 0):>10} {a.get('misses', 0):>7} "
              f"{a.get('bypass', 0):>7}   {b.get('hits', 0):>10} "
              f"{b.get('misses', 0):>7} {b.get('bypass', 0):>7}")
    for k in kinds:
        a, b = rep1.get(k, {}), rep2.get(k, {})
        if k not in rep2:
            V.fail('b', f"{k}: run 1 made {sum(a.values())} calls, run 2 made none")
        elif b.get('misses', 0) or b.get('bypass', 0):
            V.fail('b', f"{k}: run 2 had {b.get('misses', 0)} misses and "
                        f"{b.get('bypass', 0)} bypassed calls out of "
                        f"{sum(b.values())}")
        else:
            calls1 = sum(a.values())
            extra = (f"; run 1 made {calls1} calls" if calls1 != b.get('hits', 0)
                     else '')
            V.ok(f"{k}: run 2 all hits ({b.get('hits', 0)}){extra}")


# ---------------------------------------------------------------------------
# (c) what was read, next to what was stored
# ---------------------------------------------------------------------------
_BP_RE = {
    'reps': re.compile(r"reps=\[([^\]]*)\]"),
    'budget': re.compile(r"budget=(\d+)"),
    'allocation_method': re.compile(r"allocation_method='([^']*)'"),
    'importance_method': re.compile(r"importance_method='([^']*)'"),
    'augmentation_strategy': re.compile(r"augmentation_strategy='([^']*)'"),
}


def blueprint_summary(bp):
    """(sorted reps, budget, allocation, importance, aug) from a Blueprint, a
    dict, or the repr string a JSON may carry. None when it is none of them."""
    if bp is None:
        return None
    if isinstance(bp, str):
        if 'budget=' not in bp:
            return None
        got = {k: (m.group(1) if (m := rx.search(bp)) else None)
               for k, rx in _BP_RE.items()}
        reps = tuple(sorted(x.strip().strip("'\"") for x in
                            (got['reps'] or '').split(',') if x.strip()))
        return (reps, int(got['budget']) if got['budget'] else None,
                got['allocation_method'], got['importance_method'],
                got['augmentation_strategy'])
    get = (bp.get if isinstance(bp, dict) else lambda k, d=None: getattr(bp, k, d))
    reps = get('reps')
    if reps is None or get('budget') is None:
        return None
    return (tuple(sorted(str(r) for r in reps)), int(get('budget')),
            get('allocation_method'), get('importance_method'),
            get('augmentation_strategy'))


def recorded_inputs(res):
    """What the result file says it read."""
    rec = {'metric_name': res.get('metric_name'),
           'lower_is_better': res.get('lower_is_better'),
           'distance_metric': res.get('distance_metric'),
           'fingerprint': res.get('input_fingerprint') or {},
           'by_key': {}, 'blueprints': []}
    for path, k, v in _walk(res):
        if k in RECIPE_KEYS:
            rec['by_key'].setdefault(k, []).append((path, v))
        s = blueprint_summary(v) if k in ('blueprint', 'recipe') else None
        if s is not None:
            rec['blueprints'].append((path, s))
    return rec


def print_recorded(rec, args, V):
    print("  -- what the result file says it read --")
    fp = rec['fingerprint']
    print(f"  metric            {rec['metric_name']!r}, lower_is_better="
          f"{rec['lower_is_better']!r}")
    print(f"  distance          {rec['distance_metric']!r}")
    if fp:
        print(f"  molecules         {fp.get('n_samples')}   folds in config "
              f"{fp.get('n_splits')}   groups {fp.get('n_groups')}")
        print(f"  representations   " + ', '.join(
            f"{n} ({w} features)" for n, w in fp.get('reps', [])))
    else:
        V.note("the result file has no input_fingerprint")
    for k in RECIPE_KEYS:
        occ = rec['by_key'].get(k)
        if not occ:
            continue
        print(f"  {k} ({len(occ)} place{'s' if len(occ) > 1 else ''} in the file)")
        for path, v in occ[:4]:
            print(f"      {path} = {_short(v)}")
        if len(occ) > 4:
            print(f"      ... {len(occ) - 4} more")
    pm = {str(v) for _, v in rec['by_key'].get('pruning_mode', [])}
    if not pm:
        V.note(f"the result file does not record a pruning mode; the job passed "
               f"--pruning-mode {args.pruning_mode}")
    elif pm != {args.pruning_mode}:
        V.fail('c', f"result file records pruning mode {sorted(pm)}, the job "
                    f"passed {args.pruning_mode}")
    if not rec['blueprints']:
        V.note("the result file records no recipe it can be matched on "
               "(no 'blueprint' entry with reps and budget)")


class ReadOnlyRefusal(RuntimeError):
    pass


def install_read_only_guard():
    """Make every model fit or feature score in the search raise, and count it.

    The names replaced are the ones the search reaches at call time:
      find_best_hybrid._evaluate_single         one model per representation
                                                (the baselines, on a cache miss)
      find_best_hybrid._evaluate_hybrid_config  one configuration
      kirby.hybrid.compute_feature_importance   the importance pre-pass
      kirby.hybrid._evaluate_allocation         the allocation pre-pass
    The last two are imported inside the search function body, so replacing
    the module attribute reaches them.
    """
    os.environ['KIRBY_NO_COLD_SEARCH'] = '1.0'
    import kirby.find_best_hybrid as E
    import kirby.hybrid as H
    blocked = {}

    def _block(mod, name):
        def f(*a, **k):
            blocked[name] = blocked.get(name, 0) + 1
            raise ReadOnlyRefusal(
                f"read-only check: {name} would compute; the stored search does "
                f"not hold it")
        setattr(mod, name, f)

    for mod, name in ((E, '_evaluate_single'), (E, '_evaluate_hybrid_config'),
                      (H, 'compute_feature_importance'), (H, '_evaluate_allocation')):
        if hasattr(mod, name):
            _block(mod, name)
    return blocked


def read_stored(args, V):
    """The stored searches for this dataset, read the way run_all and the
    experiments read them, under the read-only guard."""
    import functools
    import numpy as np
    blocked = install_read_only_guard()

    from kirby.find_best_hybrid import (DEFAULT_ALLOCATION_METHODS,
                                        DEFAULT_IMPORTANCE_METHODS,
                                        DEFAULT_N_FOLDS)
    from kirby.similarity.distance_config import (DEFAULT_DISTANCE_METRIC,
                                                  set_distance_metric)
    set_distance_metric(os.environ.get('KIRBY_DISTANCE_METRIC')
                        or DEFAULT_DISTANCE_METRIC)
    from experiments._shared import (
        load_data, get_best_search_for_downstream, noise_base_reps_by_model,
        get_domain_metric, resolve_pruning_mode, apply_pruning_gate,
        precompute_hybrid_searches, precompute_failures, FULL_BUDGETS,
        DOWNSTREAM_MODELS_NOISE, INTERNAL_EVALUATORS)
    from experiments.noise.noise_helpers import pool_recipe_per_downstream_model
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.dataset_policy import dataset_search_speed

    ds = args.dataset
    rep_names = DATASET_REPS[ds]['general']
    # The same two loads run_all makes (noise_helpers.run_all).
    data = load_data(ds, rep_preset='all', rep_names=rep_names)
    reps = data['reps']
    for k, X in list(reps.items()):
        if np.isnan(X).any():
            reps[k] = np.nan_to_num(X, nan=0.0)
    aug_names = DATASET_REPS.get(ds, {}).get('augmentation', [])
    augs = None
    if aug_names:
        missing = [r for r in aug_names if r not in reps]
        if missing:
            ad = load_data(ds, rep_preset='all', rep_names=missing)
            augs = {k: np.nan_to_num(v, nan=0.0) for k, v in ad['reps'].items()}
        else:
            augs = {r: reps[r] for r in aug_names}
    data['augmentations'] = augs

    # The config run_all builds when it is given none.
    config = {'n_splits': DEFAULT_N_FOLDS,
              'allocations': DEFAULT_ALLOCATION_METHODS,
              'importance_methods': DEFAULT_IMPORTANCE_METHODS,
              'internal_evaluators': INTERNAL_EVALUATORS,
              'budgets': FULL_BUDGETS, 'models': DOWNSTREAM_MODELS_NOISE}
    sched, ovr, min_n = resolve_pruning_mode(args.pruning_mode)
    overrides = apply_pruning_gate(ovr, len(data['labels']), min_n)

    stored = {'speed': dataset_search_speed(ds), 'reps': sorted(
        (k, int(v.shape[1])) for k, v in reps.items()),
        'augmentations': sorted(augs) if augs else [], 'members': {},
        'by_model': {}, 'no_recipe': {}, 'base_reps': {}, 'blocked': blocked}
    metric_name, _fn, lib = get_domain_metric(data)
    stored['metric_name'], stored['lower_is_better'] = metric_name, bool(lib)

    try:
        pre = precompute_hybrid_searches(
            reps=reps, labels=data['labels'], groups=data.get('groups'),
            config=config, augmentations=augs, task_type=data['task_type'],
            domain=data.get('domain', 'molecular'), dataset_name=ds,
            tier='general', adaptive_pruning_schedule=sched,
            model_pruning_aggressiveness=overrides)
    except Exception as exc:
        V.fail('c', f"reading the stored searches raised {type(exc).__name__}: {exc}")
        return stored
    data['precomputed_searches'] = pre
    for cond, s in (pre or {}).items():
        stored['members'][cond] = (
            'absent' if s is None else
            f"{len(getattr(s, 'all_results', []) or [])} configurations, "
            f"metric {getattr(s, 'metric_name', '?')}, lower_is_better="
            f"{getattr(s, 'lower_is_better', '?')}")
    stored['failures'] = precompute_failures(pre)

    models = list(args.models.split(',')) if args.models else list(
        DOWNSTREAM_MODELS_NOISE)
    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(get_best_search_for_downstream, reps, data['labels'],
                          data.get('groups'), budget=args.budget,
                          augmentations=augs, dataset_name=ds,
                          task_type=data['task_type'], data=data,
                          pool_subset=args.pool_subset),
        models)
    for m, d in by_model.items():
        meta, bp, s = d['meta'] or {}, d['blueprint'], d['search']
        stored['by_model'][m] = {
            'tier': meta.get('tier'), 'aug_condition': meta.get('aug_condition'),
            'objective': meta.get('objective'),
            'score_mean': (meta.get('best_config') or {}).get('score_mean'),
            'blueprint': blueprint_summary(bp),
            'internal_evaluator': getattr(bp, 'internal_evaluator', None),
            'search_metric': getattr(s, 'metric_name', None),
            'search_lower_is_better': getattr(s, 'lower_is_better', None)}
    stored['no_recipe'] = {m: d['why'] for m, d in no_recipe.items()}
    base, base_missing = noise_base_reps_by_model(
        {m: d['search'] for m, d in by_model.items()}, models, reps=reps,
        dataset_name=ds, verbose=False)
    stored['base_reps'], stored['base_missing'] = base, base_missing
    return stored


def print_stored(stored, args, V):
    print("  -- what the precompute stored (read-only) --")
    print(f"  search speed      {stored['speed']} (dataset_policy.FAST_SPEED_DATASETS)")
    print(f"  metric            {stored['metric_name']!r}, lower_is_better="
          f"{stored['lower_is_better']!r} (get_domain_metric)")
    print(f"  representations   " + ', '.join(f"{n} ({w} features)"
                                            for n, w in stored['reps']))
    print(f"  augmentations     {', '.join(stored['augmentations']) or 'none'}")
    for cond, s in stored['members'].items():
        print(f"  general/{cond:<9} {s}")
    for f in stored.get('failures') or []:
        V.fail('c', f"stored search general/{f.get('aug_condition')}: "
                    f"{f.get('error_type')}: {_short(f.get('error'), 200)}")
    print(f"  pool subset {args.pool_subset!r}, budget "
          f"{args.budget if args.budget is not None else 'any'}:")
    for m, d in stored['by_model'].items():
        bp = d['blueprint']
        print(f"    {m:<9} {d['tier']}/{d['aug_condition']}/{d['objective']}  "
              f"score {d['score_mean']} ({d['search_metric']}, "
              f"lower_is_better={d['search_lower_is_better']})")
        if bp:
            print(f"              budget {bp[1]} features, {bp[2]}, {bp[3]}, "
                  f"evaluator {d['internal_evaluator']}, augmentation {bp[4]}")
            print(f"              representations {', '.join(bp[0])}")
    for m, why in stored['no_recipe'].items():
        print(f"    {m:<9} no recipe: {why}")
    for m, names in stored['base_reps'].items():
        print(f"    {m:<9} top single representations: {', '.join(names)}")
    if stored['blocked']:
        V.fail('c', "the search tried to compute while being read: " + ', '.join(
            f"{k} x{n}" for k, n in sorted(stored['blocked'].items()))
            + ". The stored search does not hold everything this run asks for, "
              "so the experiment would search or be refused.")
    else:
        V.ok("the stored searches were read without a single fit")


def compare(rec, stored, V):
    print("  -- the two side by side --")
    if rec['metric_name'] is not None:
        if rec['metric_name'] == stored['metric_name']:
            V.ok(f"metric: file {rec['metric_name']!r}, stored {stored['metric_name']!r}")
        else:
            V.fail('c', f"metric: file {rec['metric_name']!r}, stored "
                        f"{stored['metric_name']!r}")
    if rec['lower_is_better'] is not None:
        if bool(rec['lower_is_better']) == stored['lower_is_better']:
            V.ok(f"direction: lower_is_better {bool(rec['lower_is_better'])} in both")
        else:
            V.fail('c', f"direction: file lower_is_better={rec['lower_is_better']}, "
                        f"stored {stored['lower_is_better']}")
    for m, d in stored['by_model'].items():
        if (d['search_lower_is_better'] is not None and
                bool(d['search_lower_is_better']) != stored['lower_is_better']):
            V.fail('c', f"{m}: the stored search says lower_is_better="
                        f"{d['search_lower_is_better']}, the dataset says "
                        f"{stored['lower_is_better']}")
    fp_reps = sorted(tuple(x) for x in rec['fingerprint'].get('reps', []))
    if fp_reps:
        if fp_reps == [tuple(x) for x in stored['reps']]:
            V.ok(f"representations: the same {len(fp_reps)} names and widths")
        else:
            V.fail('c', f"representations differ: file {fp_reps}, loaded now "
                        f"{stored['reps']}")
    want = {m: d['blueprint'] for m, d in stored['by_model'].items() if d['blueprint']}
    for path, s in rec['blueprints']:
        hit = [m for m, bp in want.items() if bp == s]
        if hit:
            V.ok(f"recipe at {path} is the stored recipe for {', '.join(hit)}")
        else:
            V.fail('c', f"recipe at {path} ({s[1]} features, {s[2]}, {s[3]}, "
                        f"{', '.join(s[0])}) matches no stored recipe")
    rec_base = dict(rec['by_key'].get('base_reps_by_model', [(None, {})])[0][1] or {})
    for m, names in rec_base.items():
        got = stored['base_reps'].get(m)
        if got is None:
            continue
        if list(names) == list(got):
            V.ok(f"{m}: top single representations match ({', '.join(names)})")
        else:
            V.fail('c', f"{m}: file top single representations {names}, stored "
                        f"{got}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--experiment', required=True, help='e.g. n07')
    ap.add_argument('--condition', default=None,
                    help='the condition in the result filename, if any')
    ap.add_argument('--out1', required=True)
    ap.add_argument('--out2', required=True)
    ap.add_argument('--report1', required=True)
    ap.add_argument('--report2', required=True)
    ap.add_argument('--rc1', type=int, default=None)
    ap.add_argument('--rc2', type=int, default=None)
    ap.add_argument('--pruning-mode', default='calibrated')
    ap.add_argument('--pool-subset', default=None,
                    help='as the experiment passes it; default predictive_only')
    ap.add_argument('--budget', type=int, default=None,
                    help='as the experiment passes it; default any budget')
    ap.add_argument('--models', default=None,
                    help='comma list; default DOWNSTREAM_MODELS_NOISE')
    ap.add_argument('--expect-folds', default=None,
                    help='comma list of fold indices; default from the manifest')
    ap.add_argument('--no-stored', action='store_true',
                    help='skip reading the stored searches (no data loaded)')
    args = ap.parse_args(argv)

    V = Verdict()
    print(f"check_noise_run: {args.dataset} {args.experiment} "
          f"condition={args.condition or 'experiment default'}")
    per_run = check_finished(args, V)
    check_cache(args, V)

    print("\n(c) WHAT WAS READ, NEXT TO WHAT THE PRECOMPUTE STORED")
    res = per_run['run 1']['result']
    rec = None
    if isinstance(res, dict) and '_unreadable' not in res:
        rec = recorded_inputs(res)
        print_recorded(rec, args, V)
    else:
        V.fail('c', "run 1 left no readable result file, so nothing it read can "
                    "be shown")
    if args.no_stored:
        V.note("--no-stored: the stored searches were not read")
    else:
        try:
            stored = read_stored(args, V)
        except Exception as exc:
            V.fail('c', f"reading the stored searches raised "
                        f"{type(exc).__name__}: {exc}")
            stored = None
        if stored is not None:
            print_stored(stored, args, V)
            if rec is not None:
                compare(rec, stored, V)

    print()
    if V.fails:
        print(f"RESULT: FAIL ({len(V.fails)})")
        for f in V.fails:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
