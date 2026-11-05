#!/usr/bin/env python3
"""The Papers 2 and 3 ledger: which dataset is at which step, what ran where,
and how long each unit took. The plan is plan.py; the code is the
experiments' ledger (tests/experiments/orchestrate/ledger.py), subclassed the
way the screening ledger is, so the lock, claims, liveness check, retries,
heartbeats, reset and release-claim are the ones already in use.

Files under the run root:
    ledger.json  ledger.lock  events.jsonl  PAPERS.md  logs/
    RUN_SETTINGS.json   every setting the run depends on, and the gate.
                        Written once by init; after that only Adelaide edits
                        it. Read at every claim and by every unit.
    cache/              this run's own KIRBY_CACHE_DIR. Search results,
                        baselines, fold scores and checkpoints start empty
                        here; the representation files and downloaded data
                        are links into the thesis cache, which is never
                        written to except by a representation build adding a
                        file that was missing.
    complexity/ similarity/ noise/   the experiments' result files

`tests/results/CURRENT_PAPERS` (one line, untracked) names the live run. The
old `CURRENT_RUN` and the screen's `CURRENT_SCREEN` are never touched.

    python tests/experiments/papers/ledger.py init --run /data/stat-ecr/scat9264/papers/run_2026_11_03
    python tests/experiments/papers/ledger.py progress          one line per dataset
    python tests/experiments/papers/ledger.py status
    python tests/experiments/papers/ledger.py show herg_ki
    python tests/experiments/papers/ledger.py settings          the settings file as the ledger reads it
    python tests/experiments/papers/ledger.py reset herg_ki search_f2
    python tests/experiments/papers/ledger.py release-claim herg_ki
    python tests/experiments/papers/ledger.py extend --datasets <name>
    python tests/experiments/papers/ledger.py set-priority herg_ki 0
    python tests/experiments/papers/ledger.py sync
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.orchestrate import ledger as L_mod   # noqa: E402
from experiments.papers import plan as P              # noqa: E402

CURRENT_PAPERS_FILE = TESTS / 'results' / 'CURRENT_PAPERS'
SETTINGS_NAME = 'RUN_SETTINGS.json'
now = L_mod.now
CLOSED = ('done', 'done_with_failures', 'empty')

# Top-level entries of the thesis cache that hold search results or anything
# computed from a search. They start empty in the run's own cache; every
# other entry (representation files, downloaded data, structures, model
# outputs that do not depend on a search) is linked.
FRESH_CACHE_ENTRIES = (
    'result_db', 'searches', 'baselines', 'fold_scores', 'folds',
    'hybrid_evals', 'hybrid_fold_cache', 'hybrid_search',
    'create_hybrid_cache', 'hybrid_registry.json', 'noise_cache',
    'fed_autoresearch',
)
# Never linked at all: patient data (CLAUDE.md), and leftovers no search reads
# (old checkpoints at the top of the cache, stale or trashed copies, one-off
# experiment folders, the old precompute gate). Seen on the server 2026-11-04.
NEVER_LINK_PREFIXES = ('tcga', 'annitia', 'nhanes')
NEVER_LINK_SUFFIXES = ('.ckpt.pkl', '.TRASH', '_stale_20260918')
NEVER_LINK_NAMES = ('c31_weighted_global', 'c32_allocation_cross', 'validity',
                    'precompute_gate.txt')


def default_settings(cache_dir, shared_cache):
    return {
        '_what': ('Every setting the Papers 2 and 3 run depends on. Edit by hand; '
                  'the ledger reads it at every claim and every unit reads it at start.'),
        'searches': {
            'pruned': {'space': 'full', 'pruning': 'calibrated'},
            'fast': {'space': 'fast', 'pruning': None},
            'full': {'space': 'full', 'pruning': None},
        },
        'experiments_search': None,
        'aug_conditions': ['none'],
        'n_outer_folds': 5,
        'cache_dir': str(cache_dir),
        'shared_cache': str(shared_cache) if shared_cache else None,
        'open': {k: False for k in P.GATED_STEPS},
        'future_work': [],
        # Paper 2's meta-learner over models with their own hybrids. The step
        # holds until the three nulls and the dataset list are filled in; she
        # names datasets from results and runtime (2026-11-05). candidate_models
        # is her list. net_sizes is the nets she adds to it from dnn_small (64,
        # 32) and dnn_wide (512, 256, 128): [] for none.
        'meta_learner': {
            'datasets': [],
            'candidate_models': ['rf', 'dnn', 'lightgbm', 'svm', 'gp', 'xgboost', 'ngboost'],
            'net_sizes': None,
            'inner_folds': None,
            'min_rows': None,
        },
        'notes': [],
    }


def meta_learner_hold(settings, dataset):
    """Why the meta-learner step may not start on this dataset, or None."""
    m = settings.get('meta_learner') or {}
    for key in ('net_sizes', 'inner_folds', 'min_rows'):
        if m.get(key) is None:
            return f'meta_learner.{key} is not set in RUN_SETTINGS.json'
    if not m.get('candidate_models'):
        return 'meta_learner.candidate_models is empty in RUN_SETTINGS.json'
    if dataset not in (m.get('datasets') or []):
        return 'not named in meta_learner.datasets in RUN_SETTINGS.json'
    return None


def resolve_run(run=None):
    cand = run or os.environ.get('KIRBY_PAPERS_RUN')
    if not cand and CURRENT_PAPERS_FILE.exists():
        cand = CURRENT_PAPERS_FILE.read_text().strip() or None
    if not cand:
        return None
    p = Path(cand)
    return p if p.is_absolute() else REPO / p


def read_settings(root):
    return json.loads((Path(root) / SETTINGS_NAME).read_text())


def search_cache(settings, search):
    """Each search's own cache: <cache_dir>_<search>."""
    return f"{settings['cache_dir']}_{search}"


def settings_env(settings, search=None):
    """The environment a unit runs with. search names one of the three
    searches; None is the reps step, which only needs the shared files."""
    env = {'KIRBY_CACHE_DIR': settings['cache_dir'],
           'KIRBY_SEARCH_AUG': ','.join(settings.get('aug_conditions') or ['none'])}
    if search:
        cfg = settings['searches'][search]
        env.update(KIRBY_CACHE_DIR=search_cache(settings, search),
                   KIRBY_SEARCH_SPACE=cfg['space'],
                   KIRBY_SEARCH_PRUNING=cfg.get('pruning') or 'none')
    return env


def prepare_cache(cache_dir, shared_cache):
    """Make the run's cache: an empty folder per FRESH_CACHE_ENTRIES name
    (created by the code when first written), a link for every other entry
    of the shared cache. Never removes or overwrites anything. Returns
    (linked, left_fresh)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    linked, fresh = [], []
    if not shared_cache:
        return linked, fresh
    shared = Path(shared_cache)
    if not shared.is_dir():
        raise SystemExit(f'shared cache {shared} is not a folder')
    if shared.resolve() == cache_dir.resolve():
        raise SystemExit('the run cache must not be the shared cache')
    for entry in sorted(shared.iterdir()):
        dest = cache_dir / entry.name
        if entry.name in FRESH_CACHE_ENTRIES:
            fresh.append(entry.name)
            continue
        low = entry.name.lower()
        if (low.startswith(NEVER_LINK_PREFIXES) or entry.name.endswith(NEVER_LINK_SUFFIXES)
                or entry.name in NEVER_LINK_NAMES):
            continue
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(entry)
        linked.append(entry.name)
    return linked, fresh


class PapersLedger(L_mod.Ledger):
    def __init__(self, run_root):
        super().__init__(run_root)
        self.md = self.root / 'PAPERS.md'
        self._settings = None

    # ── settings and holds ────────────────────────────────────────────────

    def settings(self, reload=True):
        if reload or self._settings is None:
            self._settings = read_settings(self.root)
        return self._settings

    def hold_reason(self, row, stage):
        """Why this step of this dataset may not start now, or None."""
        st = self.settings(reload=False)
        name = stage['name']
        if row['dataset'] in (st.get('future_work') or []):
            return 'moved to future work (RUN_SETTINGS.json)'
        reps = self.stage_of(row, 'reps')
        if name != 'reps' and reps and reps['status'] == 'done_with_failures':
            return 'its representation files failed to build (see Failed units)'
        if name in P.GATED_STEPS and not (st.get('open') or {}).get(name):
            return f'step {name} is not open in RUN_SETTINGS.json'
        if name in ('paper2', 'paper3', P.META_STEP) and not st.get('experiments_search'):
            return 'experiments_search is not set in RUN_SETTINGS.json'
        if name == P.META_STEP:
            why = meta_learner_hold(st, row['dataset'])
            if why:
                return why
        if name != 'reps' and not P.models_recorded(row['dataset']):
            return 'its two models are not recorded (no row in decided_grid.json)'
        return None

    # ── what differs from the experiments' ledger ─────────────────────────

    def init(self, rows, settings, run_name=None, force=False):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not force:
            raise FileExistsError(f'{self.path} exists; use extend to add datasets')
        sp = self.root / SETTINGS_NAME
        if not sp.exists():
            sp.write_text(json.dumps(settings, indent=1) + '\n')
        self.state = {'run': run_name or self.root.name, 'root': str(self.root),
                      'created': now(), 'kind': 'papers', 'rows': rows}
        for sub in ('logs', P.META_DIR) + tuple(P.CATEGORY_DIR.values()):
            (self.root / sub).mkdir(exist_ok=True)
        self.save()
        self.event(what='init', rows=list(rows))
        CURRENT_PAPERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_PAPERS_FILE.write_text(str(self.root) + '\n')

    def save(self):
        tmp = self.path.with_suffix('.json.tmp.%d' % os.getpid())
        tmp.write_text(json.dumps(self.state, separators=(',', ':')))
        os.replace(tmp, self.path)
        try:
            self.md.write_text(render_md(self.state, self))
        except Exception as exc:          # the document must never block a run
            print(f'papers ledger: PAPERS.md not rendered: {exc}', file=sys.stderr)

    def searches_ready(self, row):
        """Every fold of the search the experiments read is ok on this row."""
        chosen = self.settings(reload=False).get('experiments_search')
        s = self.stage_of(row, f'search_{chosen}') if chosen else None
        if s is None:
            return False
        return all(u['status'] in ('ok', 'skipped') for u in s['units'].values())

    def open_stages(self, row):
        """Steps a job may work in: reps first; the three searches once reps
        is closed; paper2 and paper3 once the chosen search's folds are ok. A
        held step is left out (hold_reason)."""
        self.settings()
        out = []
        reps = self.stage_of(row, 'reps')
        for s in row['stages']:
            if s['status'] in CLOSED:
                continue
            if self.hold_reason(row, s):
                continue
            if s['name'] != 'reps' and reps and reps['status'] not in CLOSED:
                continue
            if s['name'] in ('paper2', 'paper3', P.META_STEP) and not self.searches_ready(row):
                continue
            out.append(s)
        return out

    def runnable_units(self, row, stage, mem_gb=None, only_unit=None):
        """The parent's rule, except that a held step has nothing runnable,
        even for a unit named by hand: the gate cannot be passed from the
        command line, only from RUN_SETTINGS.json. One unit can be held on its
        own too (unit_hold_reason)."""
        self.settings()
        if self.hold_reason(row, stage):
            return []
        units = super().runnable_units(row, stage, mem_gb, only_unit)
        return [k for k in units if not self.unit_hold_reason(row, stage, k)]

    def unit_hold_reason(self, row, stage, unit_key):
        """Why this ONE unit may not start now, or None.

        Added 2026-11-04 for the activity-cliff units. Two of their settings
        are not in RUN_SETTINGS.json but in
        tests/experiments/cliffs/CLIFF_SETTINGS.json, and one of them --
        Adelaide's M and her L -- comes out of the noise experiments' own
        results, so the unit that reads them cannot run until she writes them.
        The check is made at every claim, so writing the file releases the unit
        with no ledger command, the same way rebuilding decided_grid.json
        releases a dataset.
        """
        # By the unit's own name, not by its category: unit_category reads
        # the first letter for the older suites and raises on 'reps'.
        try:
            from experiments.cliffs.settings import unset as cliff_unset
            from experiments.cliffs.required import REQUIRED_SETTINGS
        except Exception:
            return None
        need = REQUIRED_SETTINGS.get(unit_key.split('__', 1)[0], ())
        if not need:
            return None
        missing = [n for n in cliff_unset() if n in need]
        if missing:
            return (f'{", ".join(missing)} has no value in '
                    f'cliffs/CLIFF_SETTINGS.json')
        return None

    def deps_state(self, row, stage, unit_key):
        state = 'ok'
        for d in P.unit_deps(stage['name'], unit_key):
            for k, u in stage['units'].items():
                if k != d and self.unit_exp(k) != d:
                    continue
                if u['status'] in ('ok', 'skipped'):
                    continue
                if u['status'] == 'failed' and not self.unit_runnable(u):
                    return 'blocked'
                state = 'waiting'
        return state

    def finish_unit(self, name, stage_name, unit_key, status, seconds=None,
                    rc=None, error=None, log=None):
        row = self.row(name)
        if status == 'ok' and stage_name in P.RESULT_STEPS:
            f = P.result_file(self.root, stage_name, name, unit_key)
            if f is None or not f.exists():
                status, error = 'failed', f'exited 0 but wrote no result file {f}'
            else:
                try:
                    d = json.loads(f.read_text())
                    rs = d.get('result_status')
                    if rs == 'skipped' or (rs is None and d.get('skipped')):
                        status, error = 'skipped', d.get('reason') or 'the experiment skipped itself'
                    elif rs not in (None, 'ok'):
                        status, error = 'failed', f'result_status {rs!r} in {f}'
                except Exception as exc:
                    status, error = 'failed', f'unreadable result file {f}: {exc}'
        # Skip the parent's file check (it reads orchestrate's layout).
        stage = self.stage_of(row, stage_name)
        u = stage['units'][unit_key]
        if status != 'ok' and self.experiment_skipped(unit_key):
            status, error = 'skipped', f'experiment skipped: {self.experiment_skipped(unit_key)}'
        u.update(status=status, ended=now(), rc=rc, error=error)
        if seconds is not None:
            u['seconds'] = round(float(seconds), 1)
        if log:
            u['log'] = log
        self.event(what='finish', row=name, stage=stage_name, unit=unit_key,
                   status=status, seconds=u['seconds'], rc=rc)
        self.close_stage_if_done(name, stage)

    def scan_disk(self):
        found = {}
        for name, row in self.state['rows'].items():
            for stage in row['stages']:
                if stage['name'] not in P.RESULT_STEPS:
                    continue
                for k, u in stage['units'].items():
                    if u['status'] in ('ok', 'skipped'):
                        continue
                    f = P.result_file(self.root, stage['name'], name, k)
                    if f is None or not f.exists():
                        continue
                    try:
                        d = json.loads(f.read_text())
                    except Exception:
                        continue
                    st = d.get('result_status')
                    if st is None and d.get('skipped'):
                        st = 'skipped'
                    if st in ('ok', 'skipped'):
                        found[(name, stage['name'], k)] = (
                            st, d.get('timestamp'),
                            None if st == 'ok' else d.get('reason'), d.get('seconds'))
        return found

    def replan(self):
        """Rebuild every row from plan.py, keeping the units that still exist
        (status, attempts, times) and the row's claim and priority. Steps the
        plan no longer has are dropped. Call inside locked(), with no worker
        running. Returns the count of rows."""
        for name, old in list(self.state['rows'].items()):
            new = P.build_row(name)
            new.update(priority=old['priority'], claim=old['claim'], mem=old['mem'])
            have = {s['name']: s for s in old['stages']}
            for i, st in enumerate(new['stages']):
                if st['name'] in have:
                    new['stages'][i] = have[st['name']]
            self.state['rows'][name] = new
        self.event(what='replan', rows=len(self.state['rows']))
        return len(self.state['rows'])

    def extend(self, names):
        added = {}
        for name in names:
            if name in self.state['rows']:
                fresh = P.build_row(name)
                row = self.state['rows'][name]
                n = 0
                for fs in fresh['stages']:
                    s = self.stage_of(row, fs['name'])
                    if s is None:
                        row['stages'].append(fs)
                        n += len(fs['units'])
                        continue
                    for k, u in fs['units'].items():
                        if k not in s['units']:
                            s['units'][k] = u
                            n += 1
                            if s['status'] in CLOSED:
                                s['status'], s['ended'] = 'running', None
                if n:
                    added[name] = f'{n} new unit(s)'
                continue
            self.state['rows'][name] = P.build_row(name)
            added[name] = 'new row'
        self.event(what='extend', added=added)
        return added


# ── The document ───────────────────────────────────────────────────────────

def _h(seconds):
    if not seconds:
        return ''
    return f'{seconds / 3600:.1f} h' if seconds >= 600 else f'{int(seconds)} s'


def _cell(stage, hold):
    units = list(stage['units'].values())
    if not units:
        return '–'
    c = Counter(u['status'] for u in units)
    secs = sum(u.get('seconds') or 0 for u in units)
    cell = f"{c['ok']}/{len(units)}"
    for k in ('running', 'failed'):
        if c[k]:
            cell += f', {c[k]} {k}'
    if secs:
        cell += f', {_h(secs)}'
    if hold and stage['status'] not in CLOSED:
        cell += ' (held)'
    return cell


def render_progress(state):
    """One plain-text table, one row per dataset: `ledger.py progress`. Each
    step is units finished / units planned; a skipped unit counts as finished
    and is also counted in its own column."""
    rows = state['rows']
    head = ['dataset'] + list(P.STEPS) + ['running', 'failed', 'skipped', 'hours']
    table = [head]
    for name in sorted(rows, key=lambda k: (rows[k]['priority'], k)):
        r = rows[name]
        cells, all_units = [], []
        for sn in P.STEPS:
            s = L_mod.Ledger.stage_of(r, sn)
            us = list(s['units'].values()) if s else []
            all_units += us
            cells.append(f"{sum(u['status'] in ('ok', 'skipped') for u in us)}/{len(us)}")
        count = Counter(u['status'] for u in all_units)
        secs = sum(u.get('seconds') or 0 for u in all_units)
        table.append([name] + cells + [str(count['running']), str(count['failed']),
                                       str(count['skipped']), f'{secs / 3600:.1f}'])
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    return '\n'.join('  '.join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in table)


def render_md(state, L=None):
    rows = state['rows']
    try:
        st = L.settings() if L else None
    except Exception:
        st = None
    out = [f"# Papers 2 and 3 ledger: {state['run']}", '',
           f"Root `{state['root']}`. Created {state['created']}. Rewritten after every change. "
           'Runbook: `tests/slurm_scripts/runbook/PAPERS.md`.', '']
    if st:
        opened = ', '.join(k for k, v in (st.get('open') or {}).items() if v) or 'none'
        out += [f"Settings (`{SETTINGS_NAME}`): augmentation {', '.join(st.get('aug_conditions') or [])}, "
                f"{st.get('n_outer_folds')} outer folds. Steps open: {opened}. "
                f"Search the experiments read: {st.get('experiments_search') or 'not chosen'}. "
                f"Future work: {', '.join(st.get('future_work') or []) or 'none'}.", '']
    out += ['One row is one dataset. A cell is units finished out of units planned, then '
            'the hours they took. "held" means the step may not start yet; the reason '
            'is in the last column.', '',
            '| dataset | papers | ' + ' | '.join(P.STEPS) + ' | held by job | why held |',
            '|---|---|' + '---|' * len(P.STEPS) + '---|---|']
    order = sorted(rows, key=lambda k: (rows[k]['priority'], k))
    for name in order:
        r = rows[name]
        cells, whys = [], []
        for sn in P.STEPS:
            s = L_mod.Ledger.stage_of(r, sn)
            if s is None:       # a run started before this step existed: extend --all
                cells.append('–')
                continue
            why = L.hold_reason(r, s) if (L and st) else None
            if why and s['status'] not in CLOSED and s['units']:
                whys.append(f"{s['name']}: {why}")
            cells.append(_cell(s, why))
        c = r['claim']
        held = c['job'].split(':')[1] if c and ':' in c['job'] else (c['job'] if c else '')
        gated = [w.split(': ', 1)[0] for w in whys if 'not open' in w]
        other = sorted(set(w.split(': ', 1)[1] for w in whys if 'not open' not in w))
        why = '; '.join(([f"not open: {', '.join(gated)}"] if gated else []) + other)
        papers = ', '.join(str(x) for x in r.get('papers', []))
        out.append(f"| {name} | {papers} | {' | '.join(cells)} | {held} | {why} |")
    out.append('')
    failed = [(n, s['name'], k, u) for n in order for s in rows[n]['stages']
              for k, u in s['units'].items() if u['status'] == 'failed']
    if failed:
        out += ['## Failed units', '',
                'A unit that exited non-zero runs once more; then it stays failed until '
                '`ledger.py reset DATASET UNIT`.', '',
                '| dataset | step | unit | attempts | exit | job | first error line | log |',
                '|---|---|---|---|---|---|---|---|']
        for n, sn, k, u in failed:
            err = (u.get('error') or '').strip().splitlines()
            err = (err[0] if err else '')[:160].replace('|', '/')
            job = (u.get('job') or '').split(':')
            out.append(f"| {n} | {sn} | {k} | {u.get('attempts')} | {u.get('rc')} | "
                       f"{job[1] if len(job) > 1 else ''} | {err} | `{u.get('log') or ''}` |")
        out.append('')
    return '\n'.join(out) + '\n'


# ── Command line ───────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--run', default=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest='cmd', required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *x, **k: _add(*x, parents=[common], **k)
    p = sub.add_parser('init')
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--cache-dir', default=None,
                   help='the run cache (default <run>/cache)')
    p.add_argument('--shared-cache', default=os.environ.get('KIRBY_CACHE_DIR'),
                   help='the thesis cache whose representation files are linked '
                        '(default $KIRBY_CACHE_DIR)')
    p.add_argument('--force', action='store_true')
    p = sub.add_parser('extend', help='add datasets, or new units and steps to existing ones; '
                                      'changes no setting and no finished unit')
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--all', action='store_true', help='every dataset already in the run')
    sub.add_parser('replan', help='rebuild rows and settings from the current code; no worker may be running')
    sub.add_parser('progress')
    sub.add_parser('status')
    sub.add_parser('settings')
    p = sub.add_parser('show')
    p.add_argument('dataset')
    p = sub.add_parser('reset')
    p.add_argument('dataset')
    p.add_argument('unit')
    p.add_argument('--stage', default=None)
    p = sub.add_parser('release-claim')
    p.add_argument('dataset')
    sub.add_parser('sync')
    p = sub.add_parser('set-priority')
    p.add_argument('dataset')
    p.add_argument('priority', type=int)
    a = ap.parse_args(argv)

    if a.cmd == 'init':
        if not a.run:
            sys.exit('init needs --run PATH')
        root = resolve_run(a.run)
        cache = Path(a.cache_dir) if a.cache_dir else root / 'cache'
        settings = default_settings(cache, a.shared_cache)
        linked, fresh = prepare_cache(cache, a.shared_cache)
        for t in P.SEARCHES:
            prepare_cache(search_cache(settings, t), a.shared_cache)
        rows = P.build_rows(a.datasets)
        L = PapersLedger(root)
        L.init(rows, settings, force=a.force)
        print(f'papers run {root}: {len(rows)} dataset(s)')
        print(f'cache {cache}: {len(linked)} entries linked from {a.shared_cache}; '
              f'starting empty: {", ".join(fresh) or "none"}')
        print(f'settings {root / SETTINGS_NAME}: no step past reps is open until you set it')
        print(f'CURRENT_PAPERS -> {root}')
        return

    root = resolve_run(a.run)
    if root is None:
        sys.exit('no papers run: pass --run, set KIRBY_PAPERS_RUN, or run init')
    L = PapersLedger(root)
    if a.cmd == 'progress':
        L.load()
        print(render_progress(L.state))
    elif a.cmd == 'status':
        L.load()
        print(render_md(L.state, L))
    elif a.cmd == 'settings':
        print(json.dumps(L.settings(), indent=1))
        for t in P.SEARCHES:
            print(f'search_{t} environment:', settings_env(L.settings(), t))
    elif a.cmd == 'show':
        L.load()
        r = L.row(a.dataset)
        for s in r['stages']:
            why = L.hold_reason(r, s)
            print(f"== {s['name']}: {s['status']}" + (f'  (held: {why})' if why else ''))
            for k, u in s['units'].items():
                extra = f" {u['error'][:120]}" if u.get('error') else ''
                print(f"   {k:32s} {u['status']:8s} att {u['attempts']} {_h(u.get('seconds'))}{extra}")
    elif a.cmd == 'replan':
        L_mod.refresh_live_jobs(force=True)
        with L.locked():
            busy = [f"{n}/{s['name']}/{k}" for n, r in L.state['rows'].items() for s in r['stages']
                    for k, u in s['units'].items()
                    if u['status'] == 'running' and L_mod.job_alive(u.get('job'), u.get('heartbeat'))]
            if busy:
                sys.exit(f'a worker is still running, stop it first: {busy[:5]}')
            old = read_settings(root)
            new = default_settings(old['cache_dir'], old.get('shared_cache'))
            new['future_work'] = old.get('future_work') or []
            new['meta_learner'] = old.get('meta_learner') or new['meta_learner']
            new['notes'] = old.get('notes') or []
            (root / SETTINGS_NAME).write_text(json.dumps(new, indent=1) + '\n')
            n = L.replan()
        for t in P.SEARCHES:
            prepare_cache(search_cache(new, t), new.get('shared_cache'))
        print(f'{n} rows replanned; steps now {", ".join(P.STEPS)}; every step past reps is closed')
    elif a.cmd == 'extend':
        if not a.datasets and not a.all:
            sys.exit('extend needs --datasets NAME ... or --all')
        with L.locked():
            names = a.datasets or list(L.state['rows'])
            print(f'added: {L.extend(names) or "nothing"}')
        (root / P.META_DIR).mkdir(exist_ok=True)
    elif a.cmd == 'reset':
        with L.locked():
            L.reset_unit(a.dataset, a.unit, a.stage)
        print(f'{a.dataset} {a.unit}: pending')
    elif a.cmd == 'release-claim':
        with L.locked():
            L.release(a.dataset)
        print(f'{a.dataset}: released')
    elif a.cmd == 'sync':
        L.load()
        found = L.scan_disk()
        with L.locked():
            n = L.apply_scan(found)
        print(f'{n} unit(s) marked ok from files on disk')
    elif a.cmd == 'set-priority':
        with L.locked():
            L.row(a.dataset)['priority'] = a.priority
        print(f'{a.dataset}: priority {a.priority}')


if __name__ == '__main__':
    main()
