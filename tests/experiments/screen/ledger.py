#!/usr/bin/env python3
"""The screening ledger: which views of which candidate dataset have been
built and scored, where, and how long each took.

It is the experiments' ledger (tests/experiments/orchestrate/ledger.py) with
a different plan: the same lock, claims, liveness (one squeue call; a job gone
from squeue is stale and its unit runs again), retry rules (a unit that exits
non-zero runs once more, a unit whose job died up to four times), heartbeats
and append-only event log. Only these differ: the rows (datasets/specs in
datasets.py), the file that proves a unit finished, what a unit waits for,
the document, and the pointer file (tests/results/CURRENT_SCREEN, so the
experiments' CURRENT_RUN is never touched).

Files under the run root:
    ledger.json  ledger.lock  events.jsonl  SCREEN.md  logs/<row>__<stage>__<unit>.log
    <dataset>/   the screen's results (summary.md is the one to read)

    python tests/experiments/screen/ledger.py init --run /data/stat-ecr/scat9264/screen/run_2026_11_01 \\
        --datasets dronerf matbench
    python tests/experiments/screen/ledger.py status
    python tests/experiments/screen/ledger.py progress
    python tests/experiments/screen/ledger.py views dronerf
    python tests/experiments/screen/ledger.py show dronerf
    python tests/experiments/screen/ledger.py extend --datasets matbench --models matbench=rf,modnet
    python tests/experiments/screen/ledger.py skip-view soap_avg --reason "too slow" --dry-run
    python tests/experiments/screen/ledger.py reset dronerf spectrum__rf__10__grouped
    python tests/experiments/screen/ledger.py release-claim dronerf
    python tests/experiments/screen/ledger.py sync
    python tests/experiments/screen/ledger.py set-priority matbench 0
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
from experiments.screen import datasets as D          # noqa: E402

CURRENT_SCREEN_FILE = TESTS / 'results' / 'CURRENT_SCREEN'
now = L_mod.now


def resolve_run(run=None):
    """The run root from the argument, KIRBY_SCREEN, or CURRENT_SCREEN."""
    cand = run or os.environ.get('KIRBY_SCREEN')
    if not cand and CURRENT_SCREEN_FILE.exists():
        cand = CURRENT_SCREEN_FILE.read_text().strip() or None
    if not cand:
        return None
    p = Path(cand)
    return p if p.is_absolute() else REPO / p


class ScreenLedger(L_mod.Ledger):
    def __init__(self, run_root):
        super().__init__(run_root)
        self.md = self.root / 'SCREEN.md'

    # ── what differs from the experiments' ledger ─────────────────────────

    def init(self, rows, run_name=None, force=False):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not force:
            raise FileExistsError(f'{self.path} exists; use extend to add datasets or units')
        self.state = {'run': run_name or self.root.name, 'root': str(self.root),
                      'created': now(), 'kind': 'screen', 'rows': rows}
        (self.root / 'logs').mkdir(exist_ok=True)
        self.save()
        self.event(what='init', rows=list(rows))
        CURRENT_SCREEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_SCREEN_FILE.write_text(str(self.root) + '\n')

    def save(self):
        tmp = self.path.with_suffix('.json.tmp.%d' % os.getpid())
        tmp.write_text(json.dumps(self.state, separators=(',', ':')))
        os.replace(tmp, self.path)
        try:
            self.md.write_text(render_md(self.state))
        except Exception as exc:          # the document must never block a run
            print(f'screen ledger: SCREEN.md not rendered: {exc}', file=sys.stderr)

    def open_stages(self, row):
        """Stages a job may work in, evaluate first: a job scores views that
        are ready before it builds more (otherwise every job would drain the
        long per-fold CNN builds while finished views sat unscored). An
        evaluate unit only becomes runnable once its view is built."""
        order = {D.EVALUATE: 0, D.BUILD: 1, D.SUMMARISE: 2}
        live = [s for s in row['stages'] if s['status'] not in ('done', 'done_with_failures', 'empty')]
        return sorted(live, key=lambda s: order.get(s['name'], 9))

    def deps_state(self, row, stage, unit_key):
        """'ok' when every unit this one waits for (in any stage of the row)
        is ok or skipped, 'blocked' when one has failed for good, else
        'waiting'. A dependency skipped by skip-view counts as blocked: an
        evaluate unit cannot score a view nobody built."""
        state = 'ok'
        for d in D.spec(row).deps(unit_key):
            u = None
            for s in row['stages']:
                if d in s['units']:
                    u = s['units'][d]
                    break
            if u is None or u['status'] == 'ok':
                continue
            if u['status'] == 'skipped':
                return 'blocked'
            if u['status'] == 'failed' and not self.unit_runnable(u):
                return 'blocked'
            state = 'waiting'
        return state

    def finish_unit(self, name, stage_name, unit_key, status, seconds=None,
                    rc=None, error=None, log=None):
        if status == 'ok' and os.environ.get('KW_NO_FILE_CHECK') != '1':
            # Exit 0 is not proof: the unit's file decides, as in the
            # experiments' ledger since 2026-10-08.
            status, error = D.spec(self.row(name)).check(self.row(name), unit_key)
        super().finish_unit(name, stage_name, unit_key, status, seconds=seconds,
                            rc=rc, error=error, log=log)

    def scan_disk(self):
        """Units whose result file exists although the ledger does not have
        them as ok (a unit run by hand, files copied in, a job that died after
        writing). No lock needed; apply_scan (inherited) marks them."""
        found = {}
        for name, row in self.state['rows'].items():
            sp = D.spec(row)
            for stage in row['stages']:
                for k, u in stage['units'].items():
                    if u['status'] in ('ok', 'skipped'):
                        continue
                    st, _ = sp.check(row, k)
                    if st == 'ok':
                        found[(name, stage['name'], k)] = ('ok', None, None, None)
        return found

    # ── screening-only commands ───────────────────────────────────────────

    def extend(self, names, models_by_ds, run_root):
        """Add rows for datasets not in the run, and units a row lacks (new
        models, new views in datasets.py). Existing units keep their marks."""
        added = {}
        for name in names:
            rows = self.state['rows']
            models = models_by_ds.get(name)
            if name not in rows:
                rows[name] = D.build_row(name, run_root, models)
                added[name] = 'new row'
                continue
            row = rows[name]
            if models:
                row['models'] = sorted(set(row['models']) | set(models),
                                       key=D.SPECS[name].all_models.index)
            fresh = D.build_row(name, run_root, row['models'], row['priority'])
            n = 0
            for fs in fresh['stages']:
                s = self.stage_of(row, fs['name'])
                for k, u in fs['units'].items():
                    if k not in s['units']:
                        s['units'][k] = u
                        n += 1
                if n and s['status'] in ('done', 'done_with_failures', 'empty'):
                    s['status'], s['ended'] = 'running', None
            if n:
                added[name] = f'{n} new unit(s)'
        self.event(what='extend', added=added)
        return added


# ── The document ───────────────────────────────────────────────────────────

_h = L_mod._h


def _counts(units):
    c = Counter(u['status'] for u in units)
    parts = [f"{c[k]} {k}" for k in ('ok', 'running', 'failed', 'skipped', 'pending') if c[k]]
    return f"{c['ok']}/{len(units)}: " + ', '.join(parts) if units else 'none'


def render_md(state):
    """The same page as the experiments' ledger: one row per dataset, then
    failed units, running units and the time of each finished unit."""
    rows = state['rows']
    order = sorted(rows, key=lambda k: (rows[k]['priority'], k))
    out = [f"# Run {state['run']}", '',
           f"Root: `{state['root']}`. Rewritten after every change; "
           f"{len(rows)} rows shown. Created {state['created']}.", '',
           'One row is one dataset. A cell is units finished / units planned, what is '
           'running, how many failed, and the hours spent so far. A build unit is one '
           'view\'s feature file (a view trained per fold: one unit per fold). An evaluate '
           'unit is one view x one model x one task (DroneRF: x one split), all its folds.', '']
    skipped = state.get('skipped_experiments') or {}
    if skipped:
        out += ['Views taken out of the run: '
                + '; '.join(f'{v} ({why})' for v, why in skipped.items()) + '.', '']
    out += ['| dataset | mem | build | evaluate | summarise | held by |', '|---|---|---|---|---|---|']
    for name in order:
        r = rows[name]
        cells = [L_mod._cell(L_mod.Ledger.stage_of(r, sn)) for sn in (D.BUILD, D.EVALUATE, 'summarise')]
        c = r['claim']
        held = f"{c['job']} since {c['since'][5:16]}" if c else ''
        out.append(f"| {name} | {r['mem']} | {' | '.join(cells)} | {held} |")
    out += ['', 'Per view: `ledger.py views DATASET`. Results when finished: '
            + ', '.join(f"`{D.spec(rows[n]).summary_file(rows[n])}`" for n in order) + '.']

    def units_with(status):
        return [(n, s['name'], k, u) for n in order for s in rows[n]['stages']
                for k, u in s['units'].items() if u['status'] in status]

    out += ['', '## Failed units', '']
    fails = units_with(('failed',))
    if not fails:
        out.append('none')
    else:
        out += ['| row | stage | unit | attempts | rc | job | log | error |', '|---|---|---|---|---|---|---|---|']
        for n, sn, k, u in fails:
            err = (u.get('error') or '').replace('|', '/').replace('\n', ' ')[:160]
            out.append(f"| {n} | {sn} | {k} | {u['attempts']} | {u.get('rc')} | {u.get('job') or ''} | "
                       f"{u.get('log') or ''} | {err} |")

    out += ['', '## Running now', '']
    runs = units_with(('running',))
    if not runs:
        out.append('none')
    else:
        out += ['| row | stage | unit | job | host | started | log |', '|---|---|---|---|---|---|---|']
        for n, sn, k, u in runs:
            out.append(f"| {n} | {sn} | {k} | {u.get('job')} | {u.get('host')} | {u.get('started')} | "
                       f"{u.get('log') or ''} |")

    out += ['', '## Time per unit (seconds, finished units)', '',
            '| row | stage | unit | status | seconds | attempts | job |', '|---|---|---|---|---|---|---|']
    for n, sn, k, u in units_with(L_mod.TERMINAL):
        if u.get('seconds') is not None:
            out.append(f"| {n} | {sn} | {k} | {u['status']} | {u['seconds']} | {u['attempts']} | "
                       f"{u.get('job') or ''} |")
    out.append('')
    return '\n'.join(out)


def render_progress(state):
    """One plain-text table, one row per dataset: `ledger.py progress`. Build,
    evaluate and summary are units finished / units planned; a skipped unit
    counts as finished and is also counted in its own column."""
    rows = state['rows']
    head = ['dataset', 'build', 'evaluate', 'summary', 'running', 'failed', 'skipped', 'hours']
    table = [head]
    for name in sorted(rows, key=lambda k: (rows[k]['priority'], k)):
        r = rows[name]
        cells, all_units = [], []
        for sn in (D.BUILD, D.EVALUATE, 'summarise'):
            us = list(L_mod.Ledger.stage_of(r, sn)['units'].values())
            all_units += us
            cells.append(f"{sum(u['status'] in ('ok', 'skipped') for u in us)}/{len(us)}")
        count = Counter(u['status'] for u in all_units)
        secs = sum(u.get('seconds') or 0 for u in all_units)
        table.append([name] + cells + [str(count['running']), str(count['failed']),
                                       str(count['skipped']), f'{secs / 3600:.1f}'])
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    return '\n'.join('  '.join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in table)


def render_views(state, names=None):
    """One row per view, one column per model: `ledger.py views DATASET`."""
    rows = state['rows']
    out = []
    for name in names or sorted(rows, key=lambda k: (rows[k]['priority'], k)):
        r = rows[name]
        ev = next(s for s in r['stages'] if s['name'] == D.EVALUATE)
        bd = next(s for s in r['stages'] if s['name'] == D.BUILD)
        by_view = {}
        for k, u in ev['units'].items():
            v, m = k.split('__')[:2]
            by_view.setdefault(v, {}).setdefault(m, []).append(u)
        builds = {}
        for k, u in bd['units'].items():
            builds.setdefault(k.split('__')[0], []).append(u)
        out += [f"## {name} ({r['section']})", '',
                'One row is one view. Build: its feature files. A model cell: units ok out of '
                'units planned (one unit per task' + (' and split' if name == 'dronerf' else '') +
                (', two on reactions: the view alone and with cgr_bond_changes appended'
                 if name == 'reactions' else '') +
                '), then the hours they took.', '',
                '| view | build | ' + ' | '.join(r['models']) + ' |',
                '|---|---|' + '---|' * len(r['models'])]
        for v in list(dict.fromkeys(list(builds) + list(by_view))):
            if v in ('targets', 'extract'):
                continue
            bcell = _counts(builds[v]) if v in builds else 'in extract'
            mcells = []
            for m in r['models']:
                us = by_view.get(v, {}).get(m, [])
                secs = sum(u.get('seconds') or 0 for u in us)
                ok = sum(u['status'] == 'ok' for u in us)
                bad = sum(u['status'] == 'failed' for u in us)
                run = sum(u['status'] == 'running' for u in us)
                cell = f'{ok}/{len(us)}'
                if run:
                    cell += f', {run} running'
                if bad:
                    cell += f', {bad} failed'
                if secs:
                    cell += f', {_h(secs)}'
                mcells.append(cell)
            out.append(f"| {v} | {bcell} | {' | '.join(mcells)} |")
        for k in ('extract', 'targets'):
            if k in builds:
                out.append('')
                out.append(f"`{k}` units: {_counts(builds[k])}.")
        out.append('')
    return '\n'.join(out)


# ── Command line ───────────────────────────────────────────────────────────

def _models_arg(items):
    out = {}
    for it in items or []:
        ds, _, ms = it.partition('=')
        out[ds] = [m for m in ms.split(',') if m]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None)
    # --run is accepted after the command too (`init --run PATH`), as in the
    # experiments' ledger.py
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--run', default=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest='cmd', required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *x, **k: _add(*x, parents=[common], **k)
    p = sub.add_parser('init')
    p.add_argument('--datasets', nargs='+', default=list(D.SPECS))
    p.add_argument('--models', nargs='*', help='DATASET=m1,m2 (default: each spec\'s models)')
    p.add_argument('--force', action='store_true')
    p = sub.add_parser('extend')
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--models', nargs='*')
    sub.add_parser('status')
    sub.add_parser('progress')
    p = sub.add_parser('views')
    p.add_argument('dataset', nargs='?')
    p = sub.add_parser('show')
    p.add_argument('dataset')
    p = sub.add_parser('reset')
    p.add_argument('dataset')
    p.add_argument('unit')
    p = sub.add_parser('release-claim')
    p.add_argument('dataset')
    sub.add_parser('sync')
    p = sub.add_parser('skip-view')
    p.add_argument('views', help='comma list of view names')
    p.add_argument('--reason', required=True)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('set-priority')
    p.add_argument('dataset')
    p.add_argument('priority', type=int)
    a = ap.parse_args(argv)

    if a.cmd == 'init':
        if not a.run:
            sys.exit('init needs --run PATH')
        root = resolve_run(a.run)
        models = _models_arg(a.models)
        rows = {n: D.build_row(n, root, models.get(n)) for n in a.datasets}
        L = ScreenLedger(root)
        L.init(rows, force=a.force)
        L.load()
        found = L.scan_disk()
        with L.locked():
            n = L.apply_scan(found)
        print(f'screen run {root}: {len(rows)} dataset(s); {n} unit(s) already on disk marked ok')
        print(f'CURRENT_SCREEN -> {root}')
        return

    root = resolve_run(a.run)
    if root is None:
        sys.exit('no screen run: pass --run, set KIRBY_SCREEN, or run init')
    L = ScreenLedger(root)

    if a.cmd == 'status':
        L.load()
        print(render_md(L.state))
    elif a.cmd == 'progress':
        L.load()
        print(render_progress(L.state))
    elif a.cmd == 'views':
        L.load()
        print(render_views(L.state, [a.dataset] if a.dataset else None))
    elif a.cmd == 'show':
        L.load()
        r = L.row(a.dataset)
        for s in r['stages']:
            print(f"== {s['name']}: {s['status']}")
            for k, u in s['units'].items():
                extra = f" {u['error'][:120]}" if u.get('error') else ''
                print(f"   {k:48s} {u['status']:8s} att {u['attempts']} {_h(u.get('seconds'))}{extra}")
    elif a.cmd == 'extend':
        with L.locked():
            added = L.extend(a.datasets, _models_arg(a.models), root)
        L.load()
        found = L.scan_disk()
        with L.locked():
            n = L.apply_scan(found)
        print(f'added: {added or "nothing"}; {n} unit(s) already on disk marked ok')
    elif a.cmd == 'reset':
        with L.locked():
            L.reset_unit(a.dataset, a.unit)
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
    elif a.cmd == 'skip-view':
        views = [v for v in a.views.split(',') if v]
        if a.dry_run:
            L.load()
            for name, r in L.state['rows'].items():
                n = sum(1 for s in r['stages'] for k, u in s['units'].items()
                        if k.split('__')[0] in views and u['status'] not in ('ok', 'skipped'))
                if n:
                    print(f'{name}: {n} unit(s) would be skipped')
            return
        with L.locked():
            out = L.skip_experiments(views, a.reason)
        print(f'skipped: {out or "nothing"}')
    elif a.cmd == 'set-priority':
        with L.locked():
            L.row(a.dataset)['priority'] = a.priority
        print(f'{a.dataset}: priority {a.priority}')


if __name__ == '__main__':
    main()
