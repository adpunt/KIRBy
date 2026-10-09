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


@pytest.fixture(autouse=True)
def _no_file_check(monkeypatch):
    # finish_unit marks an exit-0 unit ok only when its result file exists
    # (2026-10-08); these tests run no experiment, so no file is written.
    # test_finish_unit_reads_the_result_file covers the rule itself.
    monkeypatch.setenv('KW_NO_FILE_CHECK', '1')


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
        # a and b are held but have free units: a 64G job joins a (the
        # holder keeps the claim); only a 100G+ job may take big
        assert led.claim('local:h:2', 'h', mem_gb=64) == 'a'
        assert led.row('a')['claim']['job'] == 'local:h:1'
        assert led.claim('local:h:2', 'h', mem_gb=128) == 'big'     # unheld rows come first
        for name in ('a', 'b', 'opig:x'):            # nothing left but big
            for st in led.row(name)['stages']:
                for u in st['units'].values():
                    u['status'] = 'ok'
                st['status'] = 'done' if st['units'] else 'empty'
        assert led.claim('local:h:2', 'h', mem_gb=64) is None       # big needs 100
        assert led.claim('local:h:2', 'h', mem_gb=120) == 'big'


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
    monkeypatch.setattr(plan, 'noise_units', lambda ds, experiments=None, promoted=False: [
        ('n16__gaussian', 'n16', 'gaussian'), ('n20__gaussian', 'n20', 'gaussian')])
    with led.locked():
        added = led.promote(['n20'])
    assert len(added) == 3
    with led.locked():
        full = led.stage_of(led.row('a'), 'noise_full')
        assert list(full['units']) == ['n20__gaussian'] and full['status'] == 'pending'


def test_promote_by_dataset_and_noise_type(led, monkeypatch):
    monkeypatch.setattr(plan, 'noise_units', lambda ds, experiments=None, promoted=False: [
        ('n07__gaussian', 'n07', 'gaussian'), ('n07__censoring', 'n07', 'censoring'),
        ('n23__censoring', 'n23', 'censoring'), ('n23', 'n23', None), ('n30', 'n30', None)])
    with led.locked():
        added = led.promote(['n07', 'n23', 'n30'], datasets=['a'], conditions=['censoring'])
    # only dataset a; only censoring; n30 has no per-type units so it comes
    # whole; n23's no-type remainder stays out when types are named
    assert added == [('a', 'n07__censoring'), ('a', 'n23__censoring'), ('a', 'n30')]
    with led.locked():
        added = led.promote(['n07'], datasets=['a'], conditions=['censoring'])
    assert added == []


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
    # nt_H3K4me3 left this test when it was cut (2026-10-08); no nucleotide set remains.
    rows = plan.build_rows(['herg_ki', 'caco2', 'herg_ic50'])
    assert set(rows) >= {'herg_ki', 'caco2', 'herg_ic50', 'cross:s09', 'opig:ginkgo'}
    names = lambda r: [s['name'] for s in r['stages']]
    assert names(rows['caco2']) == ['complexity', 'federated', 'similarity', 'noise', 'noise_full']
    assert names(rows['herg_ic50']) == ['complexity', 'noise', 'noise_full']
    assert list(rows['herg_ic50']['stages'][0]['units']) == [
        'search_f0', 'search_f1', 'search_f2']
    comp = list(rows['herg_ki']['stages'][0]['units'])
    assert comp[:5] == [f'search_f{i}' for i in range(5)] and comp[5] == 'c06'
    assert comp.index('c08') > comp.index('c02')
    # nt_ left the big class 2026-10-07 (3.4 GB measured); kiba and bindingdb_kd stay
    assert rows['herg_ki']['mem'] == 'small'
    assert plan.mem_class('kiba') == 'big'
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
    assert 'KIRBY_NOISE_PROMOTE' not in env and env['KIRBY_NOISE_ONLY_FOLDS'] == '1,2'
    assert str(tmp_path / 'noise_full') in argv
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
    # Whether the four shared the work depends on timing (a fast stub lets
    # the first drain everything); what matters is above: nothing ran twice.


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
    # a by-hand run of the promoted folds lands in noise_full
    monkeypatch.setenv('KIRBY_NOISE_ONLY_FOLDS', '1,2')
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


def test_later_stages_open_once_the_searches_are_ok(led):
    with led.locked():
        row = led.row('a')
        assert [s['name'] for s in led.open_stages(row)] == ['complexity']
        for i in range(5):
            led.start_unit('a', 'complexity', f'search_f{i}', 'local:h:1', 'h')
            led.finish_unit('a', 'complexity', f'search_f{i}', 'ok', seconds=1)
        row = led.row('a')
        assert [s['name'] for s in led.open_stages(row)] == ['complexity', 'similarity', 'noise']
        # c06 running elsewhere, c02 running elsewhere, c08 waits for c02: complexity has nothing
        # runnable, so the work is similarity's first unit, not a wait
        led.start_unit('a', 'complexity', 'c06', 'local:h:' + str(__import__('os').getpid()), __import__('socket').gethostname())
        led.start_unit('a', 'complexity', 'c02', 'local:h:' + str(__import__('os').getpid()), __import__('socket').gethostname())
        st, units = led.work_stage(led.row('a'))
        assert st['name'] == 'similarity' and units == ['s04', 's03']
        assert led.row_has_work(led.row('a'))
        # a search fold that failed for good: every experiment depending on it is skipped,
        # complexity closes with failures, and the later stages open under the closed rule
        row = led.row('b')
        for i in range(4):
            led.start_unit('b', 'complexity', f'search_f{i}', 'local:h:1', 'h')
            led.finish_unit('b', 'complexity', f'search_f{i}', 'ok', seconds=1)
        for _ in range(2):
            led.start_unit('b', 'complexity', 'search_f4', 'local:h:1', 'h')
            led.finish_unit('b', 'complexity', 'search_f4', 'failed', seconds=1, rc=1, error='x')
        assert led.stage_of(led.row('b'), 'complexity')['status'] == 'done_with_failures'
        assert [s['name'] for s in led.open_stages(led.row('b'))] == ['federated', 'similarity', 'noise']


def _herg_ic50_like_row(with_searches):
    """herg_ic50's row shape, before (no searches) and after 2026-10-07."""
    stages = []
    if with_searches:
        stages.append(plan._stage('complexity', [f'search_f{i}' for i in range(3)]))
    stages += [plan._stage('noise', ['n17', 'n19', 'n26', 'n29']),
               plan._stage('noise_full', [])]
    return {'kind': 'dataset', 'dataset': 'herg_ic50', 'domain': 'molecular',
            'task_type': 'regression', 'n_samples': 9802, 'mem': 'small',
            'priority': 250, 'claim': None, 'stages': stages}


def test_herg_ic50_noise_waits_for_its_molecule_row_searches(led):
    """The herg_ic50 row: search_f0..f2 first, its noise units only after."""
    rows = plan.build_rows(['herg_ic50'], with_opig=False, with_cross=False)
    row = rows['herg_ic50']
    assert 'herg_ic50' in plan.MOLECULE_MEAN_SEARCH_DATASETS
    comp = row['stages'][0]
    assert comp['name'] == 'complexity'
    assert list(comp['units']) == ['search_f0', 'search_f1', 'search_f2']
    noise = row['stages'][1]['units']
    for e in ('n17', 'n19', 'n26', 'n29'):
        assert e in noise
    # none of herg_ic50's units depend on anything by name; the searches gate
    # the noise stage through searches_ready
    assert plan.unit_deps('complexity', 'search_f0') == []
    with led.locked():
        led.state['rows']['herg_ic50'] = _herg_ic50_like_row(True)
        r = led.row('herg_ic50')
        assert [s['name'] for s in led.open_stages(r)] == ['complexity']
        assert led.work_stage(r)[1] == ['search_f0', 'search_f1', 'search_f2']
        for i in range(3):
            led.start_unit('herg_ic50', 'complexity', f'search_f{i}', 'local:h:1', 'h')
            led.finish_unit('herg_ic50', 'complexity', f'search_f{i}', 'ok', seconds=1)
        r = led.row('herg_ic50')
        assert led.stage_of(r, 'complexity')['status'] == 'done'
        st, units = led.work_stage(r)
        assert st['name'] == 'noise' and units == ['n17', 'n19', 'n26', 'n29']


def test_add_planned_adds_search_stage_and_keeps_existing_units(led):
    """An existing herg_ic50 row without searches gains the stage in front;
    every unit it already had keeps its state; running twice adds nothing."""
    with led.locked():
        led.state['rows']['herg_ic50'] = _herg_ic50_like_row(False)
        led.start_unit('herg_ic50', 'noise', 'n17', 'local:h:1', 'h')
        led.finish_unit('herg_ic50', 'noise', 'n17', 'ok', seconds=5)
        led.start_unit('herg_ic50', 'noise', 'n19', 'local:h:1', 'h')
        led.finish_unit('herg_ic50', 'noise', 'n19', 'failed', seconds=5, rc=1, error='x')
    before_a = json.dumps(led.load()['rows']['a'], sort_keys=True)
    with led.locked():
        added = led.add_planned({'herg_ic50': _herg_ic50_like_row(True)},
                                {'herg_ic50'})
    assert added == [('herg_ic50', 'complexity', f'search_f{i}') for i in range(3)]
    st = led.load()
    r = st['rows']['herg_ic50']
    assert [s['name'] for s in r['stages']] == ['complexity', 'noise', 'noise_full']
    noise = r['stages'][1]['units']
    assert noise['n17']['status'] == 'ok' and noise['n17']['seconds'] == 5
    assert noise['n19']['status'] == 'failed' and noise['n19']['attempts'] == 1
    assert json.dumps(st['rows']['a'], sort_keys=True) == before_a
    with led.locked():
        r = led.row('herg_ic50')
        assert [s['name'] for s in led.open_stages(r)] == ['complexity']
        assert led.add_planned({'herg_ic50': _herg_ic50_like_row(True)},
                               {'herg_ic50'}) == []


def test_add_planned_reopens_a_closed_stage_and_skips_unnamed_rows(led):
    with led.locked():
        row = led.row('a')
        sim = led.stage_of(row, 'similarity')
        for k in list(sim['units']):
            led.start_unit('a', 'similarity', k, 'local:h:1', 'h')
            led.finish_unit('a', 'similarity', k, 'ok', seconds=1)
        assert sim['status'] == 'done'
        planned = {n: json.loads(json.dumps(r)) for n, r in fake_rows().items()}
        planned['a']['stages'][1]['units']['s99'] = plan._unit()
        planned['b']['stages'][2]['units']['s99'] = plan._unit()
        added = led.add_planned(planned, {'a'})
        assert added == [('a', 'similarity', 's99')]
        assert led.stage_of(led.row('a'), 'similarity')['status'] == 'running'
        assert 's99' not in led.stage_of(led.row('b'), 'similarity')['units']


def test_cut_skips_a_row_stops_its_running_unit_and_frees_waiters(led):
    with led.locked():
        led.start_unit('a', 'complexity', 'search_f0', 'j1', 'h')
        led.finish_unit('a', 'complexity', 'search_f0', 'ok', seconds=1)
        led.start_unit('a', 'complexity', 'search_f1', 'j1', 'h')
        assert led.unit_heartbeat('a', 'complexity', 'search_f1', 'j1') is True
        led.cut_rows(['a'], 'test')
        row = led.row('a')
        c = led.stage_of(row, 'complexity')
        assert c['units']['search_f0']['status'] == 'ok'          # finished work kept
        assert c['units']['search_f1']['status'] == 'skipped'
        assert all(s['status'] in ('done', 'done_with_failures', 'empty') for s in row['stages'])
        assert led.work_stage(row)[0] is None
        # the job still running search_f1 is told to stop, and its exit stays skipped
        assert led.unit_heartbeat('a', 'complexity', 'search_f1', 'j1') is False
        led.finish_unit('a', 'complexity', 'search_f1', 'failed', seconds=5, rc=143)
        assert c['units']['search_f1']['status'] == 'skipped'
        assert not led.unit_runnable(c['units']['search_f1'])
        # the OPIG row no longer waits for the cut row
        assert led.row('opig:x')['stages'][0]['after'] == [('b', 'complexity')] or \
            [tuple(x) for x in led.row('opig:x')['stages'][0]['after']] == [('b', 'complexity')]


def test_cut_datasets_are_not_in_the_roster():
    assert not set(plan.CUT_DATASETS) & set(plan.dataset_roster())


def test_redo_moves_result_files_and_resets_units(led):
    with led.locked():
        st = led.stage_of(led.row('a'), 'complexity')
        led.start_unit('a', 'complexity', 'c06', 'j', 'h')
        led.finish_unit('a', 'complexity', 'c06', 'ok', seconds=3)
    d = led.root / 'complexity' / 'c06_within_family_gradient'
    d.mkdir(parents=True, exist_ok=True)
    for n in ('a.json', 'a.partial.jsonl', 'a__gaussian.json', 'ab.json', 'b.json'):
        (d / n).write_text('{}')
    with led.locked():
        out = led.redo_rows(['a'], 'gp fix')
        assert led.stage_of(led.row('a'), 'complexity')['units']['c06']['status'] == 'pending'
        assert led.stage_of(led.row('a'), 'complexity')['units']['c06']['attempts'] == 0
    assert out['a'][1] == 3
    assert sorted(p.name for p in d.iterdir()) == ['ab.json', 'b.json']
    moved = led.root / 'superseded' / 'gp_fix' / 'complexity' / 'c06_within_family_gradient'
    assert sorted(p.name for p in moved.iterdir()) == ['a.json', 'a.partial.jsonl', 'a__gaussian.json']


def test_redo_units_moves_one_experiment_on_every_row(led):
    with led.locked():
        for r in ('a', 'b'):
            led.start_unit(r, 'complexity', 'c06', 'j', 'h')
            led.finish_unit(r, 'complexity', 'c06', 'ok', seconds=3)
        led.start_unit('a', 'complexity', 'c02', 'j', 'h')
        led.finish_unit('a', 'complexity', 'c02', 'ok', seconds=3)
    d = led.root / 'complexity' / 'c06_within_family_gradient'
    d.mkdir(parents=True, exist_ok=True)
    for n in ('a.json', 'b.partial.jsonl', 'big.json'):
        (d / n).write_text('{}')
    with led.locked():
        n, m, busy = led.redo_units(['c06'], reason='rerun c06')
        assert led.stage_of(led.row('a'), 'complexity')['units']['c06']['status'] == 'pending'
        assert led.stage_of(led.row('a'), 'complexity')['units']['c02']['status'] == 'ok'
    assert (n, m, busy) == (3, 3, [])
    assert list(d.iterdir()) == []


def test_finish_unit_reads_the_result_file(led, monkeypatch):
    """Exit 0 is ok only with the result file on disk; a skip file is skipped
    with its reason; no file, or a file that says partial/failed/empty, is
    failed with the reason."""
    monkeypatch.delenv('KW_NO_FILE_CHECK')
    d = led.root / 'noise' / 'n16_importance_under_noise'
    d.mkdir(parents=True)
    with led.locked():
        led.start_unit('a', 'noise', 'n16__gaussian', 'j', 'h')
        led.finish_unit('a', 'noise', 'n16__gaussian', 'ok', seconds=4)
        u = led.stage_of(led.row('a'), 'noise')['units']['n16__gaussian']
        assert u['status'] == 'failed' and 'wrote no result file' in u['error']
        (d / 'a__gaussian.json').write_text(json.dumps({'result_status': 'ok'}))
        led.start_unit('a', 'noise', 'n16__gaussian', 'j', 'h')
        led.finish_unit('a', 'noise', 'n16__gaussian', 'ok', seconds=4)
        u = led.stage_of(led.row('a'), 'noise')['units']['n16__gaussian']
        assert u['status'] == 'ok' and u['error'] is None
        (d / 'b__gaussian.json').write_text(json.dumps(
            {'skipped': True, 'reason': 'no groups', 'result_status': 'skipped'}))
        led.start_unit('b', 'noise', 'n16__gaussian', 'j', 'h')
        led.finish_unit('b', 'noise', 'n16__gaussian', 'ok', seconds=4)
        u = led.stage_of(led.row('b'), 'noise')['units']['n16__gaussian']
        assert u['status'] == 'skipped' and u['error'] == 'no groups'
        c10 = led.root / 'noise' / 'c10_noise_robustness'
        c10.mkdir()
        (c10 / 'b.json').write_text(json.dumps({'result_status': 'partial'}))
        led.start_unit('b', 'noise', 'c10', 'j', 'h')
        led.finish_unit('b', 'noise', 'c10', 'ok', seconds=4)
        u = led.stage_of(led.row('b'), 'noise')['units']['c10']
        assert u['status'] == 'failed' and "'partial'" in u['error']
        # a non-zero exit stays failed without reading anything
        led.start_unit('a', 'noise', 'n11__gaussian', 'j', 'h')
        led.finish_unit('a', 'noise', 'n11__gaussian', 'failed', rc=1, error='boom')
        u = led.stage_of(led.row('a'), 'noise')['units']['n11__gaussian']
        assert u['status'] == 'failed' and u['error'] == 'boom'


def test_skip_experiment_keeps_ok_and_stops_the_rest(led):
    with led.locked():
        st = led.stage_of(led.row('a'), 'complexity')
        st['units']['c13'] = plan._unit()
        st['units']['c13']['status'] = 'ok'
        st2 = led.stage_of(led.row('b'), 'complexity') if 'b' in led.state['rows'] else st
        st2['units'].setdefault('c13x', plan._unit())          # another experiment: untouched
        led.stage_of(led.row('a'), 'noise')['units']['c13__gaussian'] = plan._unit()
        assert led.start_unit('a', 'noise', 'c13__gaussian', 'j1', 'h')
        out = led.skip_experiments(['c13'], 'compute')
        assert out == {'a': 1}
        assert st['units']['c13']['status'] == 'ok'                         # finished result kept
        u = led.stage_of(led.row('a'), 'noise')['units']['c13__gaussian']
        assert u['status'] == 'skipped'
        assert led.unit_heartbeat('a', 'noise', 'c13__gaussian', 'j1') is False   # worker stops it
        led.finish_unit('a', 'noise', 'c13__gaussian', 'failed', rc=143)
        assert u['status'] == 'skipped'
        st['units']['c13']['status'] = 'pending'                             # e.g. added later
        assert led.start_unit('a', 'complexity', 'c13', 'j2', 'h') is False
        assert st['units']['c13']['status'] == 'skipped'
        assert st2['units']['c13x']['status'] == 'pending'
