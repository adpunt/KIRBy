"""queue_ledgers.py and the setup.sh redirect, against a fake sbatch, squeue and scontrol.

Run: python -m pytest tests/test_queue_ledgers.py -q
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / 'tests/slurm_scripts/queue_ledgers.py'

FAKE_SBATCH = r'''#!/bin/bash
echo "sbatch $*" >> "$CALLS"
case "$*" in
  *--test-only*)
    case "$*" in
      *clusters=arc*medium,long*47:59:00*) d=2026-11-07T10:00:00 ;;
      *clusters=arc*medium,long*23:59:00*) d=2026-11-07T08:00:00 ;;
      *clusters=arc*short*)                d=2026-11-06T10:00:00 ;;
      *clusters=htc*medium,long*47:59:00*) d=2026-11-06T09:00:00 ;;
      *clusters=htc*medium,long*23:59:00*) d=2026-11-06T08:00:00 ;;
      *clusters=htc*short*)                echo "sbatch: error: invalid partition" >&2; exit 1 ;;
    esac
    echo "sbatch: Job 1 to start at $d using 8 processors on nodes n1 in partition p" >&2 ;;
  *) echo 555 ;;
esac
'''
FAKE_SQUEUE = r'''#!/bin/bash
case "$*" in
  *clusters=arc*)
    echo "CLUSTER: arc"
    echo "101|x_complexity_qm9|PD|JobHeldUser|128G|(null)"
    echo "102|opig_ginkgo|PD|JobHeldUser|64G|afterany:101(unfulfilled)"
    echo "103|x_similarity_logd|PD|JobHeldUser|128G|(null)"
    echo "104|x_complexity_logd|PD|Priority|128G|(null)"
    echo "105|bash|R|None|32G|(null)" ;;
  *) echo "CLUSTER: htc"; echo "201|kp_qm9|PD|Priority|64G|(null)" ;;
esac
'''
FAKE_SCONTROL = '#!/bin/bash\necho "scontrol $*" >> "$CALLS"\n'


def fake(tmp_path, *args):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    for name, body in (('sbatch', FAKE_SBATCH), ('squeue', FAKE_SQUEUE), ('scontrol', FAKE_SCONTROL)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / 'calls'
    calls.write_text('')
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls), USER='me')
    r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    return r.stdout, calls.read_text().splitlines()


def test_check_submits_nothing_and_shows_estimates(tmp_path):
    out, calls = fake(tmp_path, 'check')
    assert all('--test-only' in c for c in calls)
    assert '2026-11-06 10:00:00' in out and 'invalid partition' in out
    assert 'arc: 1 running, 1 waiting, 3 held' in out


def test_submit_splits_clusters_and_names_no_dataset_or_account(tmp_path):
    _, calls = fake(tmp_path, 'submit', '5')
    jobs = [c for c in calls if '--test-only' not in c]
    assert len(jobs) == 5
    arc = [c for c in jobs if 'clusters=arc' in c]
    htc = [c for c in jobs if 'clusters=htc' in c]
    assert len(arc) == 3 and len(htc) == 2
    # arc: short starts 24 hours sooner than the full wall, so short.
    assert all('-p short --time=11:59:00' in c for c in arc)
    # htc: 23:59:00 starts only 1 hour sooner than the full wall, so the full wall.
    assert all('-p medium,long --time=47:59:00' in c and '--exclude=htc-g011' in c for c in htc)
    assert all('htc-g011' not in c for c in arc)
    for c in jobs:
        assert '--account' not in c and '--dataset' not in c
        assert '--cpus-per-task=8' in c and '--mem=64G' in c and '--job-name=kl' in c
        assert c.endswith('ledgers_worker.sh')


def test_release_takes_only_the_number_asked_and_touches_only_old_names(tmp_path):
    _, calls = fake(tmp_path, 'release', '2')
    cmds = [c for c in calls if c.startswith('scontrol')]
    assert sorted(c for c in cmds if ' release ' in c) == ['scontrol -M arc release 101', 'scontrol -M arc release 102']
    assert 'scontrol -M arc update JobId=101 MinMemoryNode=65536' in cmds
    assert 'scontrol -M arc update JobId=102 Dependency=' in cmds
    assert not any('JobId=102 MinMemoryNode' in c for c in cmds)    # already 64G
    assert not any('103' in c for c in cmds)                        # not taken
    assert 'scontrol -M arc update JobId=104 MinMemoryNode=65536' in cmds   # released earlier
    assert not any(' release 104' in c for c in cmds)
    assert not any(re.search(r'\b(105|201)\b', c) for c in cmds)    # srun shell, other cluster's kp job


def test_release_dry_run_changes_nothing(tmp_path):
    out, calls = fake(tmp_path, 'release', 'all', '--dry-run')
    assert not any(c.startswith('scontrol') for c in calls)
    assert 'scontrol -M arc release 103' in out


def redirect(tmp_path, job_name, **env):
    """Run only setup.sh's redirect function, with a stand-in for ledgers_worker.sh."""
    text = (REPO / 'setup.sh').read_text()
    fn = re.search(r'^_kirby_ledger_redirect\(\) \{.*?^\}$', text, re.S | re.M).group(0)
    worker = tmp_path / 'tests/slurm_scripts/ledgers_worker.sh'
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text('echo WORKER_RAN\n')
    script = f'SCRIPT_DIR={tmp_path}\n{fn}\n_kirby_ledger_redirect\necho FROZEN_BODY_RAN\n'
    full = {k: v for k, v in os.environ.items() if not k.startswith('KIRBY_')}
    full.update(SLURM_JOB_NAME=job_name, SLURM_JOB_ID='7', **env)
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=full).stdout


def test_old_names_run_the_two_ledgers_and_never_their_frozen_body(tmp_path):
    for name in ('x_complexity_qm9', 'opig_ginkgo', 'pc_qm9_2'):
        out = redirect(tmp_path, name)
        assert 'WORKER_RAN' in out and 'FROZEN_BODY_RAN' not in out


def test_other_jobs_and_managed_jobs_are_not_redirected(tmp_path):
    assert 'WORKER_RAN' not in redirect(tmp_path, 'bash')
    assert 'WORKER_RAN' not in redirect(tmp_path, 'kl')
    assert 'WORKER_RAN' not in redirect(tmp_path, 'x_complexity_qm9', KIRBY_LEDGER_MANAGED='1')


def test_job_script_drops_stale_settings_and_runs_both_ledgers(tmp_path):
    scripts = tmp_path / 'tests/slurm_scripts'
    scripts.mkdir(parents=True)
    src = (REPO / 'tests/slurm_scripts/ledgers_worker.sh').read_text()
    (scripts / 'ledgers_worker.sh').write_text(src)
    for ledger in ('papers', 'screen'):
        (scripts / f'{ledger}_worker.sh').write_text(
            f'echo RAN_{ledger} stale=${{KIRBY_SEARCH_REVERSE:-none}} managed=$KIRBY_LEDGER_MANAGED\n')
    env = dict(os.environ, KIRBY_REPO=str(tmp_path), KIRBY_SEARCH_REVERSE='1', SLURM_JOB_ID='8', SLURM_JOB_NAME='x_a')
    out = subprocess.run(['bash', str(scripts / 'ledgers_worker.sh')], capture_output=True, text=True, env=env).stdout
    assert out.index('RAN_papers') < out.index('RAN_screen')
    assert 'stale=none managed=1' in out and 'stale=1' not in out
    env['SLURM_JOB_ID'] = '9'
    out = subprocess.run(['bash', str(scripts / 'ledgers_worker.sh')], capture_output=True, text=True, env=env).stdout
    assert out.index('RAN_screen') < out.index('RAN_papers')


def ledger_row(stage, **units):
    return {'stages': [{'name': stage, 'units': {
        k: {'status': v[0], 'seconds': v[1]} for k, v in units.items()}}]}


def test_estimate_prefers_this_run_then_the_old_run_and_ignores_other_datasets():
    sys.path.insert(0, str(SCRIPT.parent))
    import queue_ledgers as Q
    rows = {'a': ledger_row('search_full', search_f0=('ok', 3600), search_f1=('pending', None)),
            'b': ledger_row('search_full', search_f0=('pending', None)),
            'c': ledger_row('paper2', c01=('pending', None), c02=('pending', None), c03=('skipped', None))}
    old = {'b': ledger_row('complexity', search_f0=('ok', 7200)),
           'c': ledger_row('complexity', c01=('ok', 1800)),
           'other': ledger_row('complexity', c02=('ok', 999999))}
    steps = Q.estimate_steps(rows, old)
    # a's fold from a's own finished fold, b's from the old run.
    assert steps['search_full'] == {'left': 2, 'seconds': 10800.0, 'no_time': 0, 'longest': 7200}
    # c01 from the old run, c02 only ran on a dataset not in this ledger, c03 is done.
    assert steps['paper2'] == {'left': 2, 'seconds': 1800.0, 'no_time': 1, 'longest': 1800}
