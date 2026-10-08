"""The ledger: one JSON file per run that says, for every dataset, which stage
it is at, which experiment is running where, and how long each one took.

Files under the run root (<repo>/tests/results/<run>/):
    ledger.json    the state; every change goes through `locked()`
    ledger.lock    the flock file (fcntl.flock, the lock f38 used across sruns)
    events.jsonl   one line per change, append-only, for the record
    LEDGER.md      the document she reads; rewritten after every change

Rows, stages and units are built by plan.py. Unit statuses:
    pending -> running -> ok | failed | skipped
A failed unit is run once more by the next job (attempts < 2), then left
failed; the stage closes as done_with_failures so later stages still run.
A running unit whose job is no longer alive is stale and runs again.

Command line (python tests/experiments/orchestrate/ledger.py ...):
    init --run tests/results/run_2026_10_06 [--datasets a b]
    status | show DS | reset DS UNIT | release-claim DS | sync
    add-planned --datasets a b              units plan.py gained since init
    promote EXP[,EXP] [--datasets a b]      noise experiments to all folds
    prioritise DS DS ...                    these datasets before every other
    set-priority DS N
"""
import contextlib
import fcntl
import json
import re
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_TESTS = _Path(__file__).resolve().parent.parent.parent
for _p in (_TESTS.parent / 'src', _TESTS, _TESTS / 'experiments'):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from experiments.orchestrate import plan  # noqa: E402

MAX_ATTEMPTS_FAILED = 2      # a unit that exited non-zero runs this many times
MAX_ATTEMPTS_STALE = 4       # a unit whose job died (wall, OOM) runs this many
HEARTBEAT_STALE_S = 6 * 3600
LOCK_TIMEOUT_S = 1800
CURRENT_RUN_FILE = plan.TESTS / 'results' / 'CURRENT_RUN'
TERMINAL = ('ok', 'failed', 'skipped')


def now():
    return datetime.now().isoformat(timespec='seconds')


# ── Where the run is ───────────────────────────────────────────────────────

def resolve_run(run=None):
    """The run root as a Path, from the argument, KIRBY_RUN, or the
    CURRENT_RUN pointer file. None when nothing names a run."""
    cand = run or os.environ.get('KIRBY_RUN')
    if not cand and CURRENT_RUN_FILE.exists():
        cand = CURRENT_RUN_FILE.read_text().strip() or None
    if not cand:
        return None
    p = Path(cand)
    if not p.is_absolute():
        p = plan.REPO / p
    return p


class Ledger:
    def __init__(self, run_root):
        self.root = Path(run_root)
        self.path = self.root / 'ledger.json'
        self.lock_path = self.root / 'ledger.lock'
        self.events = self.root / 'events.jsonl'
        self.md = self.root / 'LEDGER.md'
        self.state = None

    # ── lock + io ────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def locked(self, write=True):
        """Hold the ledger lock; the state is loaded on entry and, when
        write=True and the block did not raise, saved on exit."""
        self.root.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, 'a+')
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() - t0 > LOCK_TIMEOUT_S:
                    fh.close()
                    raise TimeoutError(f'ledger lock not free after {LOCK_TIMEOUT_S}s: {self.lock_path}')
                time.sleep(0.5)
        try:
            self.load()
            yield self.state
            if write:
                self.save()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(f'no ledger at {self.path}; run: ledger.py init --run {self.root}')
        self.state = json.loads(self.path.read_text())
        return self.state

    def save(self):
        tmp = self.path.with_suffix('.json.tmp.%d' % os.getpid())
        tmp.write_text(json.dumps(self.state, separators=(',', ':')))
        os.replace(tmp, self.path)
        try:
            self.md.write_text(render_md(self.state))
        except Exception as exc:          # the document must never block a run
            print(f'ledger: LEDGER.md not rendered: {exc}', file=sys.stderr)

    def event(self, **rec):
        rec = {'t': now(), **rec}
        with open(self.events, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')

    # ── init ──────────────────────────────────────────────────────────────

    def init(self, rows, run_name=None, force=False):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not force:
            raise FileExistsError(f'{self.path} exists; use --force to rebuild (claims and marks are lost)')
        self.state = {
            'run': run_name or self.root.name,
            'root': str(self.root),
            'created': now(),
            'rows': rows,
        }
        for sub in plan.CATEGORY_DIR.values():
            (self.root / sub).mkdir(exist_ok=True)
        (self.root / 'logs').mkdir(exist_ok=True)
        self.save()
        self.event(what='init', rows=len(rows))
        CURRENT_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_RUN_FILE.write_text(str(self.root) + '\n')

    # ── helpers on the state (call inside locked()) ───────────────────────

    def row(self, name):
        try:
            return self.state['rows'][name]
        except KeyError:
            raise KeyError(f'no row {name!r} in the ledger (rows: dataset names, opig:<name>, cross:<name>)')

    @staticmethod
    def stage_of(row, stage_name):
        for s in row['stages']:
            if s['name'] == stage_name:
                return s
        return None

    @staticmethod
    def current_stage(row):
        """The first stage that is not closed, or None."""
        for s in row['stages']:
            if s['status'] not in ('done', 'done_with_failures', 'empty'):
                return s
        return None

    def searches_ready(self, row):
        """True when every per-fold search unit of the complexity stage is ok
        or skipped (or the stage is closed). The later stages read the stored
        per-fold searches and nothing from c06, c07, c13 ..., so they open
        here rather than when complexity closes (2026-10-07, Adelaide)."""
        c = self.stage_of(row, 'complexity')
        if c is None or c['status'] in ('done', 'done_with_failures', 'empty'):
            return True
        searches = [u for k, u in c['units'].items() if plan.SEARCH_UNIT_RE.match(k)]
        return bool(searches) and all(u['status'] in ('ok', 'skipped') for u in searches)

    def open_stages(self, row):
        """The stages a job may work in, in stage order: every stage not yet
        closed, except that on a dataset row the stages after complexity wait
        for searches_ready."""
        out = []
        for i, s in enumerate(row['stages']):
            if s['status'] in ('done', 'done_with_failures', 'empty'):
                continue
            if i > 0 and row['kind'] == 'dataset' and not self.searches_ready(row):
                continue
            out.append(s)
        return out

    def work_stage(self, row, mem_gb=None, stage_name=None, only_unit=None):
        """(stage, runnable unit keys): the first open stage with a unit this
        job could start; when none has one, the first open stage and [] (so
        the caller can say what it is waiting for); (None, []) when every
        stage is closed."""
        if stage_name:
            s = self.stage_of(row, stage_name)
            return (s, self.runnable_units(row, s, mem_gb, only_unit)) if s else (None, [])
        stages = self.open_stages(row)
        for s in stages:
            units = self.runnable_units(row, s, mem_gb, only_unit)
            if units:
                return s, units
        first = self.current_stage(row)
        return (first, []) if first else (None, [])

    def stage_done(self, ds, stage_name):
        row = self.state['rows'].get(ds)
        if row is None:
            return True          # a dataset not in this run never blocks
        s = self.stage_of(row, stage_name)
        return s is None or s['status'] in ('done', 'done_with_failures', 'empty')

    def after_satisfied(self, stage):
        return all(self.stage_done(ds, st) for ds, st in stage.get('after', []))

    @staticmethod
    def unit_exp(unit_key):
        return unit_key.split('__', 1)[0]

    def deps_state(self, row, stage, unit_key):
        """'ok' when every unit this one waits for is ok/skipped, 'blocked'
        when one has failed for good, else 'waiting'."""
        deps = plan.unit_deps(stage['name'], unit_key) if row['kind'] == 'dataset' else []
        state = 'ok'
        for d in deps:
            matches = [u for k, u in stage['units'].items()
                       if k == d or self.unit_exp(k) == d]
            for u in matches:
                if u['status'] in ('ok', 'skipped'):
                    continue
                if u['status'] == 'failed' and not self.unit_runnable(u):
                    return 'blocked'
                state = 'waiting'
        return state

    def unit_runnable(self, u):
        st = u['status']
        if st == 'pending':
            return True
        if st == 'failed':
            return u['attempts'] < MAX_ATTEMPTS_FAILED
        if st == 'running':
            return (not job_alive(u.get('job'), u.get('heartbeat'))
                    and u['attempts'] < MAX_ATTEMPTS_STALE)
        return False

    def runnable_units(self, row, stage, mem_gb=None, only_unit=None):
        """Unit keys of this stage a job could start now, in stage order."""
        if only_unit:
            # Named by hand (--experiment): run it whatever its dependencies
            # or the job's memory class; she knows what she asked for.
            u = stage['units'].get(only_unit)
            return [only_unit] if u is not None and self.unit_runnable(u) else []
        if not self.after_satisfied(stage):
            return []
        if mem_gb is not None and plan.MEM_GB[row['mem']] > mem_gb:
            return []
        out = []
        for k, u in stage['units'].items():
            if not self.unit_runnable(u):
                continue
            if self.deps_state(row, stage, k) != 'ok':
                continue
            out.append(k)
        return out

    def row_has_work(self, row, mem_gb=None):
        return bool(self.work_stage(row, mem_gb)[1])

    # ── claims ────────────────────────────────────────────────────────────

    def claim(self, job, host, dataset=None, mem_gb=None):
        """Take a row. Generic (dataset=None): the lowest-priority row with
        work that no live job holds and whose memory class fits. Targeted:
        that row, joined even when another job holds it."""
        rows = self.state['rows']
        if dataset:
            row = self.row(dataset)
            if row['claim'] is None or not job_alive(row['claim']['job'], row['claim'].get('heartbeat')):
                row['claim'] = {'job': job, 'host': host, 'since': now(), 'heartbeat': now()}
            self.event(what='claim', row=dataset, job=job, targeted=True)
            return dataset
        order = sorted(rows, key=lambda k: (rows[k]['priority'], k))
        # First pass: a row nobody holds. Second pass: a held row that still
        # has units free, joined rather than left idle (seen 2026-10-06: an
        # srun held herg_ki through c06 while twenty experiments sat
        # runnable and every generic worker walked past them).
        for join_held in (False, True):
            for name in order:
                row = rows[name]
                c = row['claim']
                held = bool(c and job_alive(c['job'], c.get('heartbeat')))
                if held != join_held:
                    continue
                if not self.row_has_work(row, mem_gb):
                    continue
                if not held:
                    row['claim'] = {'job': job, 'host': host, 'since': now(), 'heartbeat': now()}
                self.event(what='claim', row=name, job=job, targeted=False, joined=held)
                return name
        return None

    def release(self, name, job=None):
        row = self.row(name)
        if row['claim'] and (job is None or row['claim']['job'] == job):
            row['claim'] = None
            self.event(what='release', row=name, job=job)

    def heartbeat(self, name, job):
        row = self.row(name)
        if row['claim'] and row['claim']['job'] == job:
            row['claim']['heartbeat'] = now()

    # ── units ─────────────────────────────────────────────────────────────

    def start_unit(self, name, stage_name, unit_key, job, host, log=None):
        """Mark a unit running. Returns False when it is already ok/skipped
        or running in a live job (nothing to do)."""
        row = self.row(name)
        stage = self.stage_of(row, stage_name)
        if stage is None:
            raise KeyError(f'{name} has no stage {stage_name}')
        u = stage['units'].get(unit_key)
        if u is None:
            # A unit this run did not plan (an srun with --experiments it has
            # no row for): add it so the time is still recorded.
            u = stage['units'][unit_key] = plan._unit()
        if u['status'] in ('ok', 'skipped'):
            return False
        if u['status'] == 'running' and job_alive(u.get('job'), u.get('heartbeat')):
            return False
        u.update(status='running', job=job, host=host, started=now(), ended=None,
                 heartbeat=now(), log=log, rc=None, error=None)
        u['attempts'] = int(u.get('attempts') or 0) + 1
        if stage['status'] == 'pending':
            stage['status'] = 'running'
            stage['started'] = now()
        self.heartbeat(name, job)
        self.event(what='start', row=name, stage=stage_name, unit=unit_key, job=job,
                   attempt=u['attempts'])
        return True

    def unit_heartbeat(self, name, stage_name, unit_key, job):
        """Stamp the unit. Returns False when the row has been cut since the
        unit started, so the worker stops it."""
        row = self.row(name)
        u = self.stage_of(row, stage_name)['units'].get(unit_key)
        if u and u.get('job') == job:
            u['heartbeat'] = now()
        self.heartbeat(name, job)
        return not row.get('cut')

    def finish_unit(self, name, stage_name, unit_key, status, seconds=None,
                    rc=None, error=None, log=None):
        row = self.row(name)
        stage = self.stage_of(row, stage_name)
        u = stage['units'][unit_key]
        if row.get('cut'):
            # Cut while it ran: it stays skipped, only its time is kept.
            status, error = 'skipped', u.get('error') or f"cut: {row['cut']}"
        u.update(status=status, ended=now(), rc=rc, error=error)
        if seconds is not None:
            u['seconds'] = round(float(seconds), 1)
        if log:
            u['log'] = log
        self.event(what='finish', row=name, stage=stage_name, unit=unit_key,
                   status=status, seconds=u['seconds'], rc=rc)
        self.close_stage_if_done(name, stage)

    def close_stage_if_done(self, name, stage):
        units = stage['units']
        if not units:
            stage['status'] = 'empty'
            return
        row = self.row(name)
        # Units that can never run because a dependency failed for good.
        for k, u in units.items():
            if u['status'] not in TERMINAL and self.deps_state(row, stage, k) == 'blocked':
                u.update(status='skipped', ended=now(),
                         error='not run: a unit it depends on failed')
        open_units = [k for k, u in units.items()
                      if u['status'] not in TERMINAL or self.unit_runnable(u)]
        if open_units:
            return
        failed = [k for k, u in units.items() if u['status'] == 'failed']
        stage['status'] = 'done_with_failures' if failed else 'done'
        stage['ended'] = now()
        secs = [u['seconds'] for u in units.values() if u.get('seconds')]
        stage['seconds'] = round(sum(secs), 1) if secs else None
        self.event(what='stage_closed', row=name, stage=stage['name'],
                   status=stage['status'], failed=failed)

    def reset_unit(self, name, unit_key, stage_name=None):
        row = self.row(name)
        hits = 0
        for s in row['stages']:
            if stage_name and s['name'] != stage_name:
                continue
            if unit_key in s['units']:
                s['units'][unit_key] = plan._unit()
                if s['status'] in ('done', 'done_with_failures'):
                    s['status'] = 'running'
                    s['ended'] = None
                hits += 1
        if not hits:
            raise KeyError(f'{name} has no unit {unit_key}')
        self.event(what='reset', row=name, unit=unit_key)

    # ── promotion of noise experiments to all folds ───────────────────────

    def promote(self, experiments, datasets=None):
        """Add (experiment, condition) units for these noise experiments to
        every dataset row's noise_full stage. The worker runs them with
        KIRBY_NOISE_PROMOTE=<experiment>, so every fold runs; the one-fold
        file is not read back (the folds run are in its fingerprint)."""
        added = 0
        for name, row in self.state['rows'].items():
            if row['kind'] != 'dataset' or (datasets and name not in datasets):
                continue
            full = self.stage_of(row, 'noise_full')
            if full is None:
                continue
            exps = plan.HERG_IC50_EXPERIMENTS if name == 'herg_ic50' else None
            for key, exp, _c in plan.noise_units(name, experiments=exps, promoted=True):
                if exp in experiments and key not in full['units']:
                    full['units'][key] = plan._unit()
                    added += 1
            if full['units'] and full['status'] == 'empty':
                full['status'] = 'pending'
        self.event(what='promote', experiments=list(experiments), added=added)
        return added

    # ── units the plan gained after init ─────────────────────────────────

    def add_planned(self, planned_rows, names=None):
        """Add what plan.py now plans and this ledger lacks, keeping every
        existing unit, stage status, claim, priority and memory class as it
        is. A missing row is added whole; a missing stage is inserted at the
        plan's position; a missing unit is added pending. A closed stage that
        gains a unit opens again. Nothing is removed or reset. Returns
        [(row, stage, unit)] added."""
        added = []
        for name, prow in planned_rows.items():
            if names and name not in names:
                continue
            row = self.state['rows'].get(name)
            if row is None:
                self.state['rows'][name] = prow
                for st in prow['stages']:
                    added += [(name, st['name'], k) for k in st['units']]
                continue
            have = {st['name']: st for st in row['stages']}
            stages = []
            for pst in prow['stages']:
                st = have.pop(pst['name'], None)
                if st is None:
                    stages.append(pst)
                    added += [(name, pst['name'], k) for k in pst['units']]
                    continue
                for k in pst['units']:
                    if k not in st['units']:
                        st['units'][k] = plan._unit()
                        added.append((name, st['name'], k))
                        if st['status'] in ('done', 'done_with_failures', 'empty'):
                            st['status'] = 'pending' if st['status'] == 'empty' else 'running'
                            st['ended'] = None
                stages.append(st)
            # stages the ledger has and the plan does not: kept, at the end
            stages += list(have.values())
            row['stages'] = stages
        if added:
            self.event(what='add-planned', added=len(added),
                       units=[f'{r}/{s}/{u}' for r, s, u in added][:200])
        return added

    # ── sync from result files on disk ───────────────────────────────────

    def scan_disk(self):
        """Read result files for every unit that is not yet ok/skipped and
        return {(row, stage, unit): (status, ended, error, seconds)}. Needs
        no lock: call it on a loaded state before locked(), then apply_scan
        inside. The suite modules are imported here (result folder names),
        which is slow the first time and must not happen under the lock."""
        found = {}
        for name, row in self.state['rows'].items():
            if row['kind'] != 'dataset':
                continue
            for stage in row['stages']:
                for k, u in stage['units'].items():
                    if u['status'] in ('ok', 'skipped'):
                        continue
                    f = plan.result_file(self.root, stage['name'], name, k)
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
                            st, d.get('timestamp'), None if st == 'ok' else d.get('reason'),
                            d.get('seconds'))
        return found

    def apply_scan(self, found):
        """Mark the units scan_disk found, unless they are now running in a
        live job. Call inside locked()."""
        changed = 0
        touched = set()
        for (name, sn, k), (st, ts, err, secs) in found.items():
            row = self.state['rows'].get(name)
            stage = self.stage_of(row, sn) if row else None
            u = stage['units'].get(k) if stage else None
            if u is None or u['status'] in ('ok', 'skipped'):
                continue
            if u['status'] == 'running' and job_alive(u.get('job'), u.get('heartbeat')):
                continue
            u.update(status=st, ended=u.get('ended') or ts or now(), error=err)
            if u.get('seconds') is None and secs is not None:
                u['seconds'] = secs
            changed += 1
            touched.add((name, sn))
        for name, sn in touched:
            self.close_stage_if_done(name, self.stage_of(self.row(name), sn))
        if changed:
            self.event(what='sync', changed=changed)
        return changed

    def sync_from_disk(self):
        """scan_disk + apply_scan in one go (inside locked()); fine for a
        command line call, not for a worker among many."""
        return self.apply_scan(self.scan_disk())

    def cut_rows(self, names, reason, detach=True):
        """Take dataset rows out of a live run. Every unit not yet ok is
        marked skipped (nothing is deleted; finished results stay), the claim
        is released, and the rows are dropped from the OPIG and cross rows'
        waits and from the dataset lists the cross scripts are given.
        attempts is raised past every retry limit, so a worker still running
        older code that records a killed unit as failed never retries it.
        Returns {row: units skipped}."""
        out = {}
        for name in names:
            row = self.row(name)
            if row['kind'] != 'dataset':
                raise SystemExit(f'{name} is not a dataset row')
            row['cut'] = reason
            n = 0
            for st in row['stages']:
                for k, u in st['units'].items():
                    if u['status'] == 'ok':
                        continue
                    u.update(status='skipped', ended=now(), error=f'cut: {reason}',
                             attempts=max(int(u.get('attempts') or 0), 99))
                    n += 1
                if st['units']:
                    self.close_stage_if_done(name, st)
            row['claim'] = None
            out[name] = n
        cut = set(names) if detach else set()
        for r in self.state['rows'].values():
            if r['kind'] == 'dataset':
                continue
            if 'datasets' in r:
                r['datasets'] = [d for d in r['datasets'] if d not in cut]
            for st in r['stages']:
                st['after'] = [a for a in st.get('after', []) if a[0] not in cut]
        self.event(what='cut', rows=list(names), reason=reason, skipped=out)
        return out

    def redo_rows(self, names, reason, move=True):
        """Run dataset rows again from the start: every result file the run
        holds for them (final JSON, partial sidecar, the experiments' own
        checkpoints) is MOVED, never deleted, to
        <run>/superseded/<reason tag>/<category>/<experiment dir>/, and every
        unit goes back to pending with its attempts cleared. Call inside
        locked(). Returns {row: (units reset, files moved)}."""
        tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', reason)[:60]
        out = {}
        for name in names:
            row = self.row(name)
            if row['kind'] != 'dataset':
                raise SystemExit(f'{name} is not a dataset row')
            moved = 0
            if move:
                for cat in set(plan.CATEGORY_DIR.values()):
                    base = self.root / cat
                    if not base.is_dir():
                        continue
                    for expdir in base.iterdir():
                        if not expdir.is_dir():
                            continue
                        for f in list(expdir.iterdir()):
                            n = f.name
                            stem = n.lstrip('.')
                            if not (stem == f'{name}.json' or stem.startswith(f'{name}__')
                                    or stem.startswith(f'{name}.')
                                    or stem.endswith(f'_{name}.json')):
                                continue
                            dest = self.root / 'superseded' / tag / cat / expdir.name / n
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            f.rename(dest)
                            moved += 1
            n_units = 0
            for st in row['stages']:
                for k in st['units']:
                    st['units'][k] = plan._unit()
                    n_units += 1
                st.update(status='pending' if st['units'] else 'empty',
                          started=None, ended=None, seconds=None)
            row['claim'] = None
            row.pop('cut', None)
            out[name] = (n_units, moved)
        self.event(what='redo', rows=list(names), reason=reason,
                   result={k: list(v) for k, v in out.items()})
        return out

    def _move_result_files(self, name, exp, tag):
        """Move this row's files out of every <category>/<exp>_*/ folder to
        <run>/superseded/<tag>/... (never deleted). Returns the count."""
        moved = 0
        for cat in set(plan.CATEGORY_DIR.values()):
            base = self.root / cat
            if not base.is_dir():
                continue
            for expdir in base.glob(f'{exp}_*'):
                if not expdir.is_dir():
                    continue
                for f in list(expdir.iterdir()):
                    stem = f.name.lstrip('.')
                    if not (stem == f'{name}.json' or stem.startswith(f'{name}__')
                            or stem.startswith(f'{name}.')
                            or stem.endswith(f'_{name}.json')):
                        continue
                    dest = self.root / 'superseded' / tag / cat / expdir.name / f.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    f.rename(dest)
                    moved += 1
        return moved

    def redo_units(self, experiments, rows=None, reason='redo'):
        """Run these experiments again on these dataset rows (default: every
        dataset row not cut): result files moved aside, units back to
        pending. A unit running in a live job is left alone and reported.
        Call inside locked(). Returns (n reset, n files moved, [running])."""
        tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', reason)[:60]
        exps = set(experiments)
        names = rows or [n for n, r in self.state['rows'].items()
                         if r['kind'] == 'dataset' and not r.get('cut')]
        n_reset = n_moved = 0
        busy = []
        for name in names:
            row = self.row(name)
            for st in row['stages']:
                for k, u in list(st['units'].items()):
                    if self.unit_exp(k) not in exps:
                        continue
                    if u['status'] == 'running' and job_alive(u.get('job'), u.get('heartbeat')):
                        busy.append(f'{name}/{k} ({u.get("job")})')
                        continue
                    st['units'][k] = plan._unit()
                    n_reset += 1
                    if st['status'] in ('done', 'done_with_failures'):
                        st.update(status='running', ended=None)
            for e in exps:
                n_moved += self._move_result_files(name, e, tag)
        self.event(what='redo-units', experiments=sorted(exps), rows=list(names),
                   reason=reason, reset=n_reset, moved=n_moved, busy=busy)
        return n_reset, n_moved, busy

    def ok_without_file(self):
        """[(row, stage, unit)] marked ok whose result file does not exist
        (neither the final JSON nor its partial file). Search units have no
        result file and are left out. Needs no lock."""
        out = []
        for name, row in self.state['rows'].items():
            if row['kind'] != 'dataset' or row.get('cut'):
                continue
            for st in row['stages']:
                for k, u in st['units'].items():
                    if u['status'] != 'ok' or plan.SEARCH_UNIT_RE.match(k):
                        continue
                    f = plan.result_file(self.root, st['name'], name, k)
                    if f is None:
                        continue
                    part = Path(str(f)[:-5] + '.partial.jsonl')
                    if not f.exists() and not part.exists():
                        out.append((name, st['name'], k))
        return out

    def close_all(self):
        for name, row in self.state['rows'].items():
            for stage in row['stages']:
                if stage['status'] not in ('done', 'done_with_failures', 'empty'):
                    self.close_stage_if_done(name, stage)


# ── Job identity and liveness ──────────────────────────────────────────────

def this_job():
    """'<cluster>:<jobid>:<host>:<pid>' under SLURM, else 'local:<host>:<pid>'.
    The pid is there because several worker processes can share one SLURM
    job (an srun shell); each is its own claimant."""
    jid = os.environ.get('SLURM_JOB_ID')
    if jid:
        cl = os.environ.get('SLURM_CLUSTER_NAME') or 'slurm'
        return f'{cl}:{jid}:{socket.gethostname()}:{os.getpid()}'
    return f'local:{socket.gethostname()}:{os.getpid()}'


def this_mem_gb():
    """Memory this job was given, in GB, or None outside SLURM."""
    mb = os.environ.get('SLURM_MEM_PER_NODE')
    if mb:
        return int(mb) / 1024
    per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
    cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE')
    if per_cpu and cpus:
        return int(per_cpu) * int(cpus) / 1024
    return None


_ALIVE_CACHE = {}
_LIVE = {'t': 0.0, 'jobs': None}      # {'arc': {'123', ...}, 'htc': {...}} or None
LIVE_TTL_S = 20


def refresh_live_jobs(force=False):
    """Every SLURM job of this user on both clusters, one squeue call, cached
    LIVE_TTL_S seconds. None when squeue is unusable (the heartbeat decides
    then). Call it BEFORE taking the ledger lock."""
    if not force and _LIVE['jobs'] is not None and time.time() - _LIVE['t'] < LIVE_TTL_S:
        return _LIVE['jobs']
    user = os.environ.get('USER') or os.environ.get('LOGNAME') or ''
    cmd = ['squeue', '-h', '-u', user, '-o', '%i %T', '-M', 'arc,htc']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        _LIVE.update(t=time.time(), jobs=None)
        return None
    if r.returncode != 0:
        # -M may be refused where only one cluster is known: try without it
        try:
            r = subprocess.run(cmd[:-2], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            r = None
        if r is None or r.returncode != 0:
            _LIVE.update(t=time.time(), jobs=None)
            return None
    jobs, cluster = {}, 'slurm'
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith('CLUSTER:'):
            cluster = ln.split(':', 1)[1].strip()
            jobs.setdefault(cluster, set())
            continue
        jid, _, state = ln.partition(' ')
        if state.strip() in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'):
            jobs.setdefault(cluster, set()).add(jid.split('_')[0])
    _LIVE.update(t=time.time(), jobs=jobs)
    _ALIVE_CACHE.clear()
    return jobs


def job_alive(job, heartbeat=None, ttl=LIVE_TTL_S):
    """Is this job still running? SLURM jobs are looked up in the one cached
    squeue listing of this user's jobs; when that is unusable the heartbeat
    decides. A pid on this host is asked of the kernel; elsewhere the
    heartbeat decides."""
    if not job:
        return False
    hit = _ALIVE_CACHE.get(job)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    alive = _job_alive(job, heartbeat)
    _ALIVE_CACHE[job] = (time.time(), alive)
    return alive


def _heartbeat_fresh(heartbeat):
    if not heartbeat:
        return False
    try:
        age = time.time() - datetime.fromisoformat(heartbeat).timestamp()
    except Exception:
        return False
    return age < HEARTBEAT_STALE_S


def _pid_alive_here(host, pid):
    """True/False when the pid is on this host, None when it is elsewhere."""
    if host != socket.gethostname():
        return None
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True          # exists, not ours to signal


def _job_alive(job, heartbeat):
    parts = job.split(':')
    if parts[0] == 'local':
        here = _pid_alive_here(parts[1], parts[2])
        return _heartbeat_fresh(heartbeat) if here is None else here
    cluster, jid = parts[0], parts[1]
    if len(parts) >= 4:
        here = _pid_alive_here(parts[2], parts[3])
        if here is False:
            return False
    live = refresh_live_jobs()
    if live is None:
        return _heartbeat_fresh(heartbeat)
    if cluster in live:
        return jid in live[cluster]
    if cluster == 'slurm':                       # cluster name unknown at stamp time
        return any(jid in s for s in live.values())
    return _heartbeat_fresh(heartbeat)           # a cluster squeue did not list


# ── The document ───────────────────────────────────────────────────────────

def _h(seconds):
    if not seconds:
        return ''
    return f'{seconds / 3600:.1f}h' if seconds >= 600 else f'{int(seconds)}s'


def _cell(stage):
    units = stage['units']
    if not units:
        return '-' if stage['status'] == 'empty' else 'none'
    n = len(units)
    ok = sum(u['status'] in ('ok', 'skipped') for u in units.values())
    failed = [k for k, u in units.items() if u['status'] == 'failed']
    running = [k for k, u in units.items() if u['status'] == 'running']
    secs = sum(u['seconds'] or 0 for u in units.values())
    if stage['status'] == 'done':
        return f'done {_h(secs)}'
    if stage['status'] == 'done_with_failures':
        return f'done, {len(failed)} failed ({", ".join(failed[:3])}) {_h(secs)}'
    parts = [f'{ok}/{n}']
    if running:
        parts.append('running ' + ', '.join(running[:2]))
    if failed:
        parts.append(f'{len(failed)} failed')
    if secs:
        parts.append(_h(secs))
    if not ok and not running and not failed:
        return 'pending'
    return ' '.join(parts)


def render_md(state):
    # Cut datasets (ledger.py cut) are left out of every table; one line names them.
    cut = sorted(k for k, r in state['rows'].items() if r.get('cut'))
    rows = {k: r for k, r in state['rows'].items() if not r.get('cut')}
    out = [f"# Run {state['run']}", '',
           f"Root: `{state['root']}`. Rewritten after every change; "
           f"{len(rows)} rows shown. Created {state['created']}.", '',
           'One row is one dataset. A cell is one stage: units finished / '
           'units planned, what is running, how many failed, and the hours '
           'spent so far (sum over its experiments). A unit is one experiment '
           '(noise: one experiment under one noise condition).', '']
    if cut:
        out += [f"Cut, not shown: {', '.join(cut)}.", '']
    stage_names = ['complexity', 'federated', 'similarity', 'noise', 'noise_full']
    out.append('| dataset | mem | ' + ' | '.join(stage_names) + ' | held by |')
    out.append('|---|---|' + '---|' * len(stage_names) + '---|')
    ds_rows = sorted((k for k, r in rows.items() if r['kind'] == 'dataset'),
                     key=lambda k: (rows[k]['priority'], k))
    for k in ds_rows:
        r = rows[k]
        cells = []
        for sn in stage_names:
            s = Ledger.stage_of(r, sn)
            cells.append(_cell(s) if s else '')
        c = r['claim']
        held = f"{c['job']} since {c['since'][5:16]}" if c else ''
        out.append(f"| {k} | {r['mem']} | " + ' | '.join(cells) + f' | {held} |')
    other = [k for k, r in rows.items() if r['kind'] != 'dataset']
    if other:
        out += ['', '## Scripts that read several datasets', '',
                'They start when the stage named in `after` is closed on every dataset they read.', '',
                '| row | mem | status | waits for | held by |', '|---|---|---|---|---|']
        for k in sorted(other):
            r = rows[k]
            s = r['stages'][0]
            u = s['units'][list(s['units'])[0]]
            pend = [f'{d}/{st}' for d, st in s.get('after', [])
                    if not _stage_closed(rows, d, st)]
            waits = (f'{len(pend)} stages: ' + ', '.join(pend[:4]) + (' ...' if len(pend) > 4 else '')) if pend else 'ready'
            c = r['claim']
            held = f"{c['job']}" if c else ''
            out.append(f"| {k} | {r['mem']} | {u['status']} {_h(u['seconds'])} | {waits} | {held} |")
    # Failures
    fails = []
    for k, r in rows.items():
        for s in r['stages']:
            for uk, u in s['units'].items():
                if u['status'] == 'failed':
                    fails.append((k, s['name'], uk, u))
    out += ['', '## Failed units', '']
    if not fails:
        out.append('none')
    else:
        out.append('| row | stage | unit | attempts | rc | job | log | error |')
        out.append('|---|---|---|---|---|---|---|---|')
        for k, sn, uk, u in fails:
            err = (u.get('error') or '').replace('|', '/').replace('\n', ' ')[:160]
            out.append(f"| {k} | {sn} | {uk} | {u['attempts']} | {u.get('rc')} | {u.get('job') or ''} | {u.get('log') or ''} | {err} |")
    # Running
    runs = []
    for k, r in rows.items():
        for s in r['stages']:
            for uk, u in s['units'].items():
                if u['status'] == 'running':
                    runs.append((k, s['name'], uk, u))
    out += ['', '## Running now', '']
    if not runs:
        out.append('none')
    else:
        out.append('| row | stage | unit | job | host | started | log |')
        out.append('|---|---|---|---|---|---|---|')
        for k, sn, uk, u in runs:
            out.append(f"| {k} | {sn} | {uk} | {u.get('job')} | {u.get('host')} | {u.get('started')} | {u.get('log') or ''} |")
    # Times
    out += ['', '## Time per unit (seconds, finished units)', '',
            '| row | stage | unit | status | seconds | attempts | job |', '|---|---|---|---|---|---|---|']
    for k in ds_rows + sorted(other):
        r = rows[k]
        for s in r['stages']:
            for uk, u in s['units'].items():
                if u['status'] in TERMINAL and u.get('seconds') is not None:
                    out.append(f"| {k} | {s['name']} | {uk} | {u['status']} | {u['seconds']} | {u['attempts']} | {u.get('job') or ''} |")
    out.append('')
    return '\n'.join(out)


def _stage_closed(rows, ds, st):
    r = rows.get(ds)
    if r is None:
        return True
    s = Ledger.stage_of(r, st)
    return s is None or s['status'] in ('done', 'done_with_failures', 'empty')


# ── Command line ───────────────────────────────────────────────────────────

def main(argv=None):
    import argparse
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: a subcommand's own --run must not overwrite one given before it
    common.add_argument('--run', default=argparse.SUPPRESS, help='run root (default: KIRBY_RUN or tests/results/CURRENT_RUN)')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[common])
    sub = ap.add_subparsers(dest='cmd', required=True)
    P = dict(parents=[common])
    p = sub.add_parser('init', **P); p.add_argument('--datasets', nargs='*'); p.add_argument('--force', action='store_true')
    p.add_argument('--no-opig', action='store_true'); p.add_argument('--no-cross', action='store_true')
    sub.add_parser('status', **P)
    p = sub.add_parser('show', **P); p.add_argument('row')
    p = sub.add_parser('reset', **P); p.add_argument('row'); p.add_argument('unit'); p.add_argument('--stage')
    p = sub.add_parser('release-claim', **P); p.add_argument('row')
    sub.add_parser('sync', **P)
    p = sub.add_parser('add-planned', **P); p.add_argument('--datasets', nargs='+', required=True)
    p = sub.add_parser('promote', **P); p.add_argument('experiments'); p.add_argument('--datasets', nargs='*')
    p = sub.add_parser('set-priority', **P); p.add_argument('row'); p.add_argument('priority', type=int)
    p = sub.add_parser('set-mem', **P); p.add_argument('row'); p.add_argument('mem', choices=sorted(plan.MEM_GB))
    p = sub.add_parser('skip-condition', **P); p.add_argument('condition'); p.add_argument('--stage', default='noise')
    p = sub.add_parser('prioritise', **P); p.add_argument('rows', nargs='+',
                                                          help='these rows go first, in this order, before every other row')
    sub.add_parser('render', **P)
    p = sub.add_parser('cut', **P); p.add_argument('rows', nargs='*',
                                                   help='dataset rows (default: plan.CUT_DATASETS)')
    p.add_argument('--reason', default='cut by Adelaide 2026-10-08')
    p = sub.add_parser('redo-units', **P); p.add_argument('experiments', help='comma list, e.g. c18,c22')
    p.add_argument('--rows', nargs='*', default=None, help='dataset rows (default: every row not cut)')
    p.add_argument('--reason', required=True)
    p = sub.add_parser('reset-missing', **P)
    p.add_argument('--apply', action='store_true', help='reset them (default: only list)')
    p = sub.add_parser('hold', **P); p.add_argument('rows', nargs='+',
                                                    help='dataset rows to stop until `redo` (OPIG/cross waits kept)')
    p.add_argument('--reason', required=True)
    p = sub.add_parser('redo', **P); p.add_argument('rows', nargs='+',
                                                    help='dataset rows to run again from the start')
    p.add_argument('--reason', required=True, help='short tag, also the folder the old files move to')
    a = ap.parse_args(argv)
    a.run = getattr(a, 'run', None)

    if a.cmd == 'init':
        root = resolve_run(a.run)
        if root is None:
            ap.error('init needs --run <dir>')
        L = Ledger(root)
        rows = plan.build_rows(a.datasets or None, with_opig=not a.no_opig, with_cross=not a.no_cross)
        L.init(rows, force=a.force)
        print(f'ledger: {L.path}  rows {len(rows)}  document {L.md}')
        print(f'pointer: {CURRENT_RUN_FILE} -> {root}')
        return
    root = resolve_run(a.run)
    if root is None:
        ap.error('no run: pass --run, set KIRBY_RUN, or init one')
    L = Ledger(root)
    if a.cmd == 'status':
        L.load(); print(render_md(L.state))
    elif a.cmd == 'render':
        with L.locked():
            pass
        print(L.md)
    elif a.cmd == 'show':
        L.load(); r = L.row(a.row)
        print(json.dumps({k: v for k, v in r.items() if k != 'stages'}, indent=1))
        for s in r['stages']:
            print(f"\n[{s['name']}] {s['status']} after={s.get('after') or '-'}")
            for uk, u in s['units'].items():
                print(f"  {uk:28s} {u['status']:8s} att={u['attempts']} s={u['seconds']} job={u.get('job')} log={u.get('log') or ''}")
                if u.get('error'):
                    print(f"      {str(u['error'])[:200]}")
    elif a.cmd == 'reset':
        with L.locked():
            L.reset_unit(a.row, a.unit, a.stage)
        print(f'reset {a.row} {a.unit}')
    elif a.cmd == 'release-claim':
        with L.locked():
            L.release(a.row)
        print(f'released {a.row}')
    elif a.cmd == 'sync':
        with L.locked():
            n = L.sync_from_disk(); L.close_all()
        print(f'sync: {n} unit(s) marked from result files')
    elif a.cmd == 'add-planned':
        # Built before the lock: plan.py imports the suites (slow).
        # Named rows only: a run initialised with --datasets must not gain
        # every other dataset, and a row's other plan changes since init
        # (conditions, experiments) are not added behind her back.
        planned = plan.build_rows(a.datasets, with_opig=False, with_cross=False)
        with L.locked():
            added = L.add_planned(planned, set(a.datasets))
        for r, st, u in added:
            print(f'added {r} {st} {u}')
        print(f'add-planned: {len(added)} unit(s) added; existing units unchanged')
    elif a.cmd == 'promote':
        exps = [e.strip() for e in a.experiments.split(',') if e.strip()]
        with L.locked():
            n = L.promote(exps, a.datasets or None)
        print(f'promoted {exps}: {n} unit(s) added to noise_full')
    elif a.cmd == 'prioritise':
        with L.locked():
            lowest = min(r['priority'] for r in L.state['rows'].values())
            for i, name in enumerate(a.rows):      # ahead of everything, earlier calls included
                L.row(name)['priority'] = lowest - len(a.rows) + i
            L.event(what='prioritise', rows=list(a.rows))
        print('first, in order: ' + ', '.join(a.rows))
    elif a.cmd == 'set-priority':
        with L.locked():
            L.row(a.row)['priority'] = a.priority
        print(f'{a.row} priority {a.priority}')
    elif a.cmd == 'skip-condition':
        # A kind of damage dropped from the screen after init: its pending
        # units are marked skipped (never deleted), so the stage can close.
        n = 0
        with L.locked():
            for name, row in L.state['rows'].items():
                st = L.stage_of(row, a.stage)
                if not st:
                    continue
                for k, u in st['units'].items():
                    if k.endswith('__' + a.condition) and u['status'] == 'pending':
                        u.update(status='skipped', ended=now(),
                                 error=f'not run: {a.condition} dropped from the {a.stage} screen')
                        n += 1
                if st['status'] not in ('done', 'done_with_failures', 'empty'):
                    L.close_stage_if_done(name, st)
            L.event(what='skip-condition', stage=a.stage, condition=a.condition, skipped=n)
        print(f'{n} pending {a.stage} unit(s) under {a.condition} marked skipped')
    elif a.cmd == 'cut':
        names = a.rows or [d for d in plan.CUT_DATASETS]
        with L.locked():
            names = [d for d in names if d in L.state['rows']]
            running = {d: [(s['name'], k, u.get('job')) for s in L.row(d)['stages']
                           for k, u in s['units'].items() if u['status'] == 'running']
                       for d in names}
            done = L.cut_rows(names, a.reason)
        for d in names:
            print(f'{d}: {done[d]} unit(s) skipped')
            for sn, k, job in running[d]:
                print(f'    was running: {sn}/{k} in job {job}')
    elif a.cmd == 'redo-units':
        exps = [e.strip() for e in a.experiments.split(',') if e.strip()]
        with L.locked():
            n, m, busy = L.redo_units(exps, a.rows or None, a.reason)
        print(f'{n} unit(s) of {",".join(exps)} back to pending; {m} result file(s) moved to superseded/{a.reason}/')
        for b in busy:
            print(f'    still running, NOT reset (run this again when it ends): {b}')
    elif a.cmd == 'reset-missing':
        L.load()
        missing = L.ok_without_file()
        for r, sn, k in missing:
            print(f'  {r:28s} {sn:12s} {k}')
        print(f'{len(missing)} unit(s) marked ok with no result file')
        if a.apply and missing:
            with L.locked():
                for r, sn, k in missing:
                    L.reset_unit(r, k, sn)
                L.event(what='reset-missing', units=[f'{r}/{sn}/{k}' for r, sn, k in missing])
            print('reset to pending')
        elif missing:
            print('nothing changed; add --apply to reset them')
    elif a.cmd == 'hold':
        # Like cut, but the rows stay in the OPIG/cross scripts' waits: they
        # come back with `redo`. A unit running in a job that pulled this code
        # stops at its next heartbeat; older jobs need stop_cut_searches.py.
        with L.locked():
            running = {d: [(s['name'], k, u.get('job')) for s in L.row(d)['stages']
                           for k, u in s['units'].items() if u['status'] == 'running']
                       for d in a.rows}
            done = L.cut_rows(a.rows, f'hold: {a.reason}', detach=False)
        for d in a.rows:
            print(f'{d}: {done[d]} unit(s) held')
            for sn, k, job in running[d]:
                print(f'    was running: {sn}/{k} in job {job}')
    elif a.cmd == 'redo':
        with L.locked():
            done = L.redo_rows(a.rows, a.reason)
        for d, (n, m) in done.items():
            print(f'{d}: {n} unit(s) back to pending, {m} result file(s) moved to superseded/')
    elif a.cmd == 'set-mem':
        # Which jobs may take the row (plan.MEM_GB). The class is stamped at
        # init, so a change to plan.BIG_RE reaches a live run through this.
        with L.locked():
            L.row(a.row)['mem'] = a.mem
            L.event(what='set-mem', row=a.row, mem=a.mem)
        print(f'{a.row} mem {a.mem}: taken by jobs of {plan.MEM_GB[a.mem]} GB or more')


if __name__ == '__main__':
    main()
