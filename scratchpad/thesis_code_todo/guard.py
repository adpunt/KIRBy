"""Run one command at a time with a memory ceiling.

    python guard.py [--max-gb 4] [--timeout 3600] -- cmd args...

- A global lock: only one guarded command runs on this machine at once; others wait.
- Threads capped at 2 (OMP/MKL/OpenBLAS/numexpr/torch), tokenizers parallelism off.
- Polls the whole process tree every 0.5 s. Kills it if its RSS passes --max-gb,
  if free system memory falls under 2.5 GB, or on --timeout.
Exit code 137 plus a line on stderr says why it was killed.
"""
import fcntl, os, subprocess, sys, time
import psutil

args = sys.argv[1:]
max_gb, timeout = 4.0, 3600
while args and args[0] != '--':
    k, v = args[0], args[1]; args = args[2:]
    if k == '--max-gb': max_gb = float(v)
    elif k == '--timeout': timeout = float(v)
cmd = args[1:]
lock = open('/private/tmp/claude-501/thesis_guard.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX)
env = dict(os.environ)
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
          'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    env[k] = '2'
env.update(TOKENIZERS_PARALLELISM='false', KIRBY_CACHE_DIR='/Volumes/seagate/kirby_cache',
           KIRBY_LGBM_THREADS='1', PYTHONUNBUFFERED='1')
env.pop('KIRBY_LOCAL_MAX_FEATURES_PER_REP', None)
p = subprocess.Popen(cmd, env=env)
pp = psutil.Process(p.pid); t0 = time.time(); peak = 0.0; why = None
while p.poll() is None:
    try:
        procs = [pp] + pp.children(recursive=True)
        rss = sum(q.memory_info().rss for q in procs if q.is_running()) / 1e9
    except psutil.Error:
        rss = 0.0
    peak = max(peak, rss)
    avail = psutil.virtual_memory().available / 1e9
    if rss > max_gb: why = f'RSS {rss:.1f} GB > {max_gb} GB'
    elif avail < 2.5: why = f'system free memory {avail:.1f} GB < 2.5 GB'
    elif time.time() - t0 > timeout: why = f'timeout {timeout}s'
    if why:
        for q in pp.children(recursive=True) + [pp]:
            try: q.kill()
            except psutil.Error: pass
        p.wait()
        print(f'[guard] KILLED: {why}; peak {peak:.2f} GB', file=sys.stderr, flush=True)
        sys.exit(137)
    time.sleep(0.5)
print(f'[guard] exit {p.returncode}; peak {peak:.2f} GB; {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
sys.exit(p.returncode)
