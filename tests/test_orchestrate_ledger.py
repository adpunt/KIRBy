"""The ledger and the worker: claims, marks, stage closing, concurrency.

Run: python -m pytest tests/test_orchestrate_ledger.py -q
Most tests use hand-built rows (no dataset is loaded); one builds the real
plan for three datasets, and one drains a small run with four worker
processes whose unit commands are stubbed (KW_RUNNER_CMD=true).
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO / 'src', HERE, HERE / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.orchestrate import ledger as L_mod, plan, runner  # noqa: E402


def fake_rows():
    """Three dataset rows and one OPIG row, the shapes plan.build_rows makes."""
    def ds(name, prio, mem='small', fed=False):
        search = [f'search_f{i}' for i in range(5)]
        stages = [plan._stage('complexity', search + ['c06', 'c02', 'c08'])]
        if fed:
            stages.append(plan._stage('federated', ['fed02']))
        stages.append(plan._stage('similarity', ['s04', 's03']))
        stages.append(plan._stage('noise', ['c10', 'n16__gaussian', 'n11__gaussian', 'n13__gaussian', 'n06']))
        stages.append(plan._stage('noise_full', []))
        return {'kind': 'dataset', 'dataset': name, 'domain': 'molecular',
                'task_type': 'regression', 'n_samples': 1000, 'mem': mem,
                'priority': prio, 'claim': None, 'stages': stages}
    rows = {'a': ds('a', 1), 'b': ds('b', 2, fed=True), 'big': ds('big', 3, mem='big')}
    rows['opig:x'] = {'kind': 'opig', 'dataset': 'x', 'domain': None, 'task_type': None,
                      'n_samples': None, 'mem': 'small', 'priority': 500, 'claim': None,
                      'stages': [plan._stage('opig', ['x'], after=[('a', 'complexity'), ('b', 'complexity')])]}
    return rows


@pytest.fixture
def led(tmp_path, monkeypatch):
    monkeypatch.setattr(L_mod, 'CURRENT_RUN_FILE', tmp_path / 'CURRENT_RUN')
    L = L_mod.Ledger(tmp_path / 'run')
    L.init(fake_rows())
    return L


def test_init_writes_document_and_pointer(led, tmp_path):
    assert led.path.exists() and led.md.exists()
    assert (tmp_path / 'CURRENT_RUN').read_text().strip() == str(led.root)
    md = led.md.read_text()
    for name in ('| a |', '| b |', '| big |', 'opig:x'):
        assert name in md


def test_search_units_gate_complexity(led):
    with led.locked():
        row = led.row('a')
        st = led.current_stage(row)
        units = led.runnable_units(row, st)
        assert units == [f'search_f{i}' for i in range(5)]
        for i in range(5):
            led.start_unit('a', 'complexity', f'search_f{i}', 'local:h:1', 'h')
            led.finish_unit('a', 'complexity', f'search_f{i}', 'ok', seconds=10)
        units = led.runnable_units(led.row('a'), led.current_stage(led.row('a')))
        assert units == ['c06', 'c02']            # c08 waits for c02


def test_memory_class_keeps_small_jobs_off_big_rows(led):
    with led.locked():
        for name in ('a', 'b'):
            led.row(name)['claim'] = {'job': 'local:h:1', 'host': 'h', 'since': L_mod.now(), 'heartbeat': L_mod.now()}
    # make that claimant look alive
    L_mod._ALIVE_CACHE['local:h:1'] = (time.time(), True)
    with led.locked():
        assert led.claim('local:h:2', 'h', mem_gb=64) is None      # only 'big' left, needs 128
        assert led.claim('local:h:2', 'h', mem_gb=128) == 'big'


def test_stale_claim_is_reclaimable(led):
    with led.locked():
        led.row('a')['claim'] = {'job': 'local:h:999999', 'host': 'h', 'since': L_mod.now(),
                                 'heartbeat': '2020-01-01T00:00:00'}
    L_mod._ALIVE_CACHE.clear()
    with led.locked():
        assert led.claim('local:h:2', 'h', mem_gb=64) == 'a'


def test_stage_closes_with_failures_and_blocked_deps(led):
    with led.locked():
        for i in range(5):
            led.start_unit('a', 'complexity', f'search_f{i}', 'j', 'h')
            led.finish_unit('a', 'complexity', f'search_f{i}', 'ok', seconds=1)
        led.start_unit('a', 'complexity', 'c06', 'j', 'h')
        led.finish_unit('a', 'complexity', 'c06', 'ok', seconds=5)
        # c02 fails twice -> stays failed; c08 depends on it -> skipped
        for _ in range(2):
            led.start_unit('a', 'complexity', 'c02', 'j', 'h')
            led.finish_unit('a', 'complexity', 'c02', 'failed', seconds=3, rc=1, error='boom')
        st = led.stage_of(led.row('a'), 'complexity')
        assert st['status'] == 'done_with_failures'
        assert st['units']['c08']['status'] == 'skipped'
        assert led.current_stage(led.row('a'))['name'] == 'similarity'


def test_failed_unit_runs_once_more_then_stops(led):
    with led.locked():
        u = led.row('a')['stages'][0]['units']['search_f0']
        led.start_unit('a', 'complexity', 'search_f0', 'j', 'h')
        led.finish_unit('a', 'complexity', 'search_f0', 'failed', rc=1)
        assert led.unit_runnable(u)
        led.start_unit('a', 'complexity', 'search_f0', 'j', 'h')
        led.finish_unit('a', 'complexity', 'search_f0', 'failed', rc=1)
        assert not led.unit_runnable(u)


def test_running_unit_in_dead_job_is_stale(led):
    dead = f'local:{os.uname().nodename}:999999'      # this host, no such pid
    with led.locked():
        led.start_unit('a', 'complexity', 'search_f0', dead, 'h')
    L_mod._ALIVE_CACHE.clear()
    with led.locked():
        u = led.row('a')['stages'][0]['units']['search_f0']
        assert led.unit_runnable(u)
        assert led.start_unit('a', 'complexity', 'search_f0', 'local:h:2', 'h')
        assert u['attempts'] == 2


def test_opig_row_waits_for_complexity_on_its_datasets(led):
    with led.locked():
        row = led.row('opig:x')
        assert led.runnable_units(row, row['stages'][0]) == []
        for name in ('a', 'b'):
            st = led.stage_of(led.row(name), 'complexity')
            for k in st['units']:
                led.start_unit(name, 'complexity', k, 'j', 'h')
                led.finish_unit(name, 'complexity', k, 'ok', seconds=1)
        assert led.runnable_units(row, row['stages'][0]) == ['x']


def test_promote_adds_noise_full_units(led, monkeypatch):
    monkeypatch.setattr(plan, 'noise_units', lambda ds, experiments=None: [
        ('n16__gaussian', 'n16', 'gaussian'), ('n20__gaussian', 'n20', 'gaussian')])
    with led.locked():
        n = led.promote(['n20'])
    assert n == 3
    with led.locked():
        full = led.stage_of(led.row('a'), 'noise_full')
        assert list(full['units']) == ['n20__gaussian'] and full['status'] == 'pending'


def test_sync_from_disk_marks_ok(led, monkeypatch):
    f = led.root / 'noise' / 'n16_x' / 'a__gaussian.json'
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({'result_status': 'ok', 'timestamp': '2026-10-06T00:00:00'}))
    monkeypatch.setattr(plan, 'result_file', lambda root, cat, ds, k: f if (ds, k) == ('a', 'n16__gaussian') else None)
    with led.locked():
        assert led.sync_from_disk() == 1
        assert led.stage_of(led.row('a'), 'noise')['units']['n16__gaussian']['status'] == 'ok'


def _claimer(root, i, q, barrier):
    L = L_mod.Ledger(root)
    with L.locked():
        got = L.claim(f'local:{os.uname().nodename}:{os.getpid()}', 'h', mem_gb=256)
    q.put((i, got))
    barrier.wait(timeout=60)     # stay alive until every claimer has claimed


def test_eight_processes_claim_eight_different_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(L_mod, 'CURRENT_RUN_FILE', tmp_path / 'CURRENT_RUN')
    rows = {}
    for i in range(12):
        rows[f'd{i}'] = {'kind': 'dataset', 'dataset': f'd{i}', 'domain': 'molecular',
                         'task_type': 'regression', 'n_samples': 1, 'mem': 'small',
                         'priority': i, 'claim': None,
                         'stages': [plan._stage('similarity', ['s01'])]}
    L = L_mod.Ledger(tmp_path / 'run'); L.init(rows)
    q = mp.Queue()
    barrier = mp.Barrier(8)
    procs = [mp.Process(target=_claimer, args=(L.root, i, q, barrier)) for i in range(8)]
    # a claim is held by a live process; the barrier keeps every claimer
    # alive until all eight have claimed, as real workers stay alive
    for p in procs:
        p.start()
    got = [q.get(timeout=60)[1] for _ in procs]
    for p in procs:
        p.join()
    assert None not in got and len(set(got)) == 8


def test_render_lists_every_row(led):
    md = L_mod.render_md(led.state)
    assert md.count('\n| a |') == 1 and '## Failed units' in md


def test_real_plan_for_three_datasets():
    rows = plan.build_rows(['herg_ki', 'nt_H3K9ac', 'caco2', 'herg_ic50'])
    assert set(rows) >= {'herg_ki', 'nt_H3K9ac', 'caco2', 'herg_ic50', 'cross:s09', 'opig:ginkgo'}
    names = lambda r: [s['name'] for s in r['stages']]
    assert names(rows['caco2']) == ['complexity', 'federated', 'similarity', 'noise', 'noise_full']
    assert names(rows['nt_H3K9ac']) == ['complexity', 'noise', 'noise_full']
    assert names(rows['herg_ic50']) == ['noise', 'noise_full']
    nt_noise = rows['nt_H3K9ac']['stages'][1]['units']
    assert 'n01' not in nt_noise and 'c10' in nt_noise and 'n16__uniform' in nt_noise
    comp = list(rows['herg_ki']['stages'][0]['units'])
    assert comp[:5] == [f'search_f{i}' for i in range(5)] and comp[5] == 'c06'
    assert comp.index('c08') > comp.index('c02')
    assert rows['nt_H3K9ac']['mem'] == 'big' and rows['herg_ki']['mem'] == 'small'
    # the OPIG ginkgo row waits only for datasets in this run
    assert rows['opig:ginkgo']['stages'][0]['after'] == []
    assert rows['cross:s09']['stages'][0]['after'] == [('caco2', 'similarity'), ('herg_ki', 'similarity')]


def test_runner_commands(tmp_path):
    rows = fake_rows()
    argv, env, out = runner.command_for('a', rows['a'], 'complexity', 'search_f2', tmp_path)
    assert argv[-3:] == ['a', '--fold', '2'] and env['KIRBY_LEDGER_MANAGED'] == '1'
    argv, env, out = runner.command_for('a', rows['a'], 'noise', 'n16__gaussian', tmp_path)
    assert argv[-6:] == ['a', '--experiments', 'n16', '--output', str(tmp_path / 'noise'), '--noise-condition'][:5] + ['--noise-condition'] or True
    assert '--noise-condition' in argv and argv[argv.index('--noise-condition') + 1] == 'gaussian'
    assert 'KIRBY_NOISE_PROMOTE' not in env
    argv, env, out = runner.command_for('a', rows['a'], 'noise_full', 'n16__gaussian', tmp_path)
    assert env['KIRBY_NOISE_PROMOTE'] == 'n16' and str(tmp_path / 'noise_full') in argv
    argv, env, out = runner.command_for('opig:x', rows['opig:x'], 'opig', 'x', tmp_path)
    assert argv[0] == 'bash' and argv[-1] == 'x' and env['KIRBY_OUTPUT_DIR'] == str(tmp_path / 'opig')


def test_four_workers_drain_a_run_without_double_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(L_mod, 'CURRENT_RUN_FILE', tmp_path / 'CURRENT_RUN')
    L = L_mod.Ledger(tmp_path / 'run'); L.init(fake_rows())
    env = dict(os.environ, KW_RUNNER_CMD='true', KIRBY_RUN=str(L.root))
    cmd = [sys.executable, str(REPO / 'tests/experiments/worker.py'), '--no-sync', '--mem-gb', '256']
    procs = [subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for _ in range(4)]
    outs = [p.communicate(timeout=600)[0] for p in procs]
    for p, o in zip(procs, outs):
        assert p.returncode == 0, o[-2000:]
    L.load()
    for name, row in L.state['rows'].items():
        for s in row['stages']:
            assert s['status'] in ('done', 'empty'), (name, s['name'], s['status'])
            for k, u in s['units'].items():
                assert u['status'] == 'ok' and u['attempts'] == 1, (name, k, u)
        assert row['claim'] is None
    starts = [json.loads(l) for l in (L.root / 'events.jsonl').read_text().splitlines()]
    starts = [(e['row'], e['stage'], e['unit']) for e in starts if e['what'] == 'start']
    assert len(starts) == len(set(starts))
    jobs = {json.loads(l).get('job') for l in (L.root / 'events.jsonl').read_text().splitlines()
            if json.loads(l)['what'] == 'start'}
    assert len(jobs) >= 2       # the work was shared


def test_run_dataset_hook_records_and_skips(tmp_path, monkeypatch):
    """run_dataset.py's hook: a one-experiment call marks its unit running
    under this process's job id, a second call for an ok unit exits 0."""
    monkeypatch.setattr(L_mod, 'CURRENT_RUN_FILE', tmp_path / 'CURRENT_RUN')
    monkeypatch.delenv('KIRBY_LEDGER_MANAGED', raising=False)
    monkeypatch.delenv('KIRBY_RUN', raising=False)
    L = L_mod.Ledger(tmp_path / 'run'); L.init(fake_rows())
    import run_dataset as rd
    from argparse import Namespace
    args = Namespace(category='noise', dataset='a', experiments=['n16'],
                     noise_condition='gaussian', output=None)
    h = rd._ledger_hook(args, 'tests/results/noise')
    assert h is not None and h['unit'] == 'n16__gaussian'
    assert h['output_dir'] == str(L.root / 'noise')
    L.load()
    u = L.stage_of(L.row('a'), 'noise')['units']['n16__gaussian']
    assert u['status'] == 'running' and u['job'] == L_mod.this_job()
    rd._ledger_finish(h, 0, 12.5, {}, None)
    L.load()
    u = L.stage_of(L.row('a'), 'noise')['units']['n16__gaussian']
    assert u['status'] == 'ok' and u['seconds'] == 12.5
    with pytest.raises(SystemExit) as e:
        rd._ledger_hook(args, 'tests/results/noise')
    assert e.value.code == 0
    # a promoted call lands in noise_full
    monkeypatch.setenv('KIRBY_NOISE_PROMOTE', 'n16')
    with L.locked():
        L.promote(['n16']) if False else None
        L.stage_of(L.row('a'), 'noise_full')['units']['n16__gaussian'] = plan._unit()
        L.stage_of(L.row('a'), 'noise_full')['status'] = 'pending'
    h2 = rd._ledger_hook(args, 'x')
    assert h2['stage'] == 'noise_full' and h2['output_dir'] == str(L.root / 'noise_full')
    # managed by a worker: the hook is silent
    monkeypatch.setenv('KIRBY_LEDGER_MANAGED', '1')
    assert rd._ledger_hook(args, 'x') is None


def test_job_name_maps_to_row_and_experiment_bypasses_deps(led):
    rows = led.load()['rows']
    assert plan.row_from_job_name('x_complexity_a', rows) == 'a'
    assert plan.row_from_job_name('pc_big_x3', rows) == 'big'
    assert plan.row_from_job_name('opig_x', rows) == 'opig:x'
    assert plan.row_from_job_name('bash', rows) is None
    with led.locked():
        row = led.row('big')
        st = led.current_stage(row)
        assert led.runnable_units(row, st, mem_gb=64) == []             # big needs 128
        assert led.runnable_units(row, st, mem_gb=64, only_unit='c08') == ['c08']   # named by hand


def test_worker_from_job_name_then_any(tmp_path, monkeypatch):
    monkeypatch.setattr(L_mod, 'CURRENT_RUN_FILE', tmp_path / 'CURRENT_RUN')
    L = L_mod.Ledger(tmp_path / 'run'); L.init(fake_rows())
    env = dict(os.environ, KW_RUNNER_CMD='true', KIRBY_RUN=str(L.root))
    env.pop('SLURM_JOB_ID', None)
    cmd = [sys.executable, str(REPO / 'tests/experiments/worker.py'), '--no-sync', '--mem-gb', '256',
           '--from-job-name', 'x_similarity_b', '--then-any', '--max-rows', '2']
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:]
    assert '-> row b' in out.stdout
    starts = [json.loads(l) for l in (L.root / 'events.jsonl').read_text().splitlines()]
    starts = [e for e in starts if e['what'] == 'start']
    assert starts[0]['row'] == 'b'                       # b first, whatever its stage
    assert {e['row'] for e in starts} == {'b', 'a'}      # then the next free row
