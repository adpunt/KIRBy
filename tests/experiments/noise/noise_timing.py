#!/usr/bin/env python3
"""How long every step of a noise experiment took.

Decided by Adelaide, 2026-11-04: "yea just record the times of everything".
No limit is set, nothing is flagged and nothing is dropped. This only records.

One log per (experiment, dataset, noise kind), next to its results:
    <output_dir>/<exp_id>/<dataset>__<kind>.timing.jsonl.gz

One line per step. A line holds:
    what    'detector', 'cleanup', 'result', 'search', 'inject',
            'build_model_inputs', 'piece', and so on;
            'cache:<kind>' is one computation that the noise cache stored
    s       seconds
    status  what those seconds were:
              computed  ran, and at least one step inside it was not stored yet
              no_cache  ran, with no stored step involved
              cached    read back from the noise cache, so it says nothing
                        about how long the work takes
              refused   ran and returned nothing (gp_smooth above its size
                        limit, a detector that gave no score)
              not_run   skipped (a weighted training set on a model whose fit
                        takes no per-molecule weight)
              raised    stopped on an error
              cache_off ran with the noise cache switched off
    at      when it ended, seconds since 1970
    and the fold, level, draw, model, model input, detector, cleanup, setting
    it belongs to, where those exist. `prep_s` on a 'result' line is the
    seconds since the previous result ended, which is the time spent making
    this cleaned training set.

Cache reads are counted, not written one per line: a piece reads the same two
model fits (noisy labels, correct labels) thousands of times.

The log is gzip, written in whole blocks, so a session killed at its time limit
loses at most the last minute. Read it with `python noise_timing.py <folder>`.
"""
import atexit
import functools
import gzip
import json
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

FLUSH_LINES = 2000
FLUSH_SECONDS = 60.0

_lock = threading.Lock()
_path = None
_pid = None
_buf = []
_last_flush = 0.0
_ctx = {}
_hits = 0
_misses = 0
_last_result_end = None


def timing_path(output_dir, exp_id, dataset_name, condition):
    """The log of one (experiment, dataset, noise kind)."""
    return (Path(output_dir) / exp_id /
            f'{dataset_name}__{condition}.timing.jsonl.gz')


def flush():
    """Write what is buffered, as one gzip block."""
    global _buf, _last_flush
    with _lock:
        if not _buf or _path is None or os.getpid() != _pid:
            _buf = [] if os.getpid() != _pid else _buf
            return
        data, _buf = '\n'.join(_buf) + '\n', []
        path = _path
        _last_flush = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'ab') as fh:
        fh.write(gzip.compress(data.encode('utf-8')))
        fh.flush()


atexit.register(flush)


@contextmanager
def log_to(path, **context):
    """Send everything inside this block to the log at `path`, with `context`
    on every line. path None records nothing."""
    global _path, _pid, _ctx, _last_flush
    flush()
    saved = (_path, _pid, _ctx)
    _path = None if path is None else Path(path)
    _pid = os.getpid()
    _ctx = {k: v for k, v in context.items() if v is not None}
    _last_flush = time.time()
    try:
        yield
    finally:
        flush()
        _path, _pid, _ctx = saved


def set_context(**fields):
    """Put these on every later line. A value of None takes the field off."""
    global _ctx
    new = dict(_ctx)
    for k, v in fields.items():
        if v is None:
            new.pop(k, None)
        else:
            new[k] = v
    _ctx = new


def event(what, seconds, status, **fields):
    """Write one line."""
    if _path is None or os.getpid() != _pid:
        return
    line = dict(_ctx)
    line.update({k: v for k, v in fields.items() if v is not None})
    line.update(what=what, s=round(float(seconds), 4), status=status,
                at=round(time.time(), 1))
    text = json.dumps(line, default=str, separators=(',', ':'))
    with _lock:
        _buf.append(text)
        due = (len(_buf) >= FLUSH_LINES
               or time.time() - _last_flush >= FLUSH_SECONDS)
    if due:
        flush()


def note_cache_hit():
    global _hits
    _hits += 1


def note_cache_miss():
    global _misses
    _misses += 1


def since_last_result():
    """Seconds since the last 'result' step ended; None before the first."""
    return (None if _last_result_end is None
            else round(time.time() - _last_result_end, 4))


class _Step:
    status = None


@contextmanager
def timed(what, **fields):
    """Time the block and write one line. Set `.status` on what it yields to
    say the block refused; left alone, the status says whether the block's
    stored steps were computed or read back."""
    global _ctx, _last_result_end
    step = _Step()
    hits0, misses0 = _hits, _misses
    saved = _ctx
    _ctx = dict(saved)
    _ctx.update({k: v for k, v in fields.items() if v is not None})
    t0 = time.perf_counter()
    try:
        yield step
    except BaseException:
        step.status = 'raised'
        raise
    finally:
        seconds = time.perf_counter() - t0
        _ctx = saved
        if step.status is None:
            step.status = ('computed' if _misses > misses0
                           else 'cached' if _hits > hits0 else 'no_cache')
        event(what, seconds, step.status, **fields)
        if what == 'result':
            _last_result_end = time.time()


def timed_call(what):
    """Decorator form of timed(), for a function called from many places."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            with timed(what):
                return fn(*a, **k)
        return inner
    return wrap


# ---------------------------------------------------------------------------
# reading the logs
# ---------------------------------------------------------------------------

def read_log(path):
    """Every line of one log. A last block cut off by a killed session is
    dropped."""
    out = []
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except (EOFError, OSError):
        pass
    return out


# Seconds that say how long the work takes. 'cached' and 'refused' take almost
# no time and would pull a detector's or cleanup's typical time down.
_REAL = ('computed', 'no_cache', 'cache_off')


def _table(title, rows):
    print(f'\n{title}')
    print('One row is one name. Only steps that really ran count towards the '
          'seconds.')
    print(f"{'name':<46}{'total s':>11}{'runs':>8}{'median s':>10}"
          f"{'max s':>9}{'cached':>8}{'refused':>8}{'raised':>7}")
    for name, st in sorted(rows.items(),
                           key=lambda kv: -sum(kv[1]['real'])):
        r = st['real']
        print(f"{str(name)[:45]:<46}{sum(r):>11.1f}{len(r):>8}"
              f"{(statistics.median(r) if r else 0.0):>10.3f}"
              f"{(max(r) if r else 0.0):>9.2f}{st['cached']:>8}"
              f"{st['refused']:>8}{st['raised']:>7}")


def summarise(lines):
    """Print the slowest detectors, cleanups, model fits and other steps."""
    def blank():
        return {'real': [], 'cached': 0, 'refused': 0, 'raised': 0}
    groups = {k: defaultdict(blank) for k in
              ('detector', 'cleanup', 'model fit', 'other step')}
    for ln in lines:
        what = ln.get('what')
        if what == 'detector':
            g, name = 'detector', ln.get('detector')
        elif what == 'cleanup':
            g, name = 'cleanup', ln.get('cleanup')
        elif what == 'cache:fit_predict':
            g, name = 'model fit', ln.get('model')
        elif what == 'result':
            continue                # the fits inside it are counted above
        else:
            g, name = 'other step', what
        st = groups[g][name]
        status = ln.get('status')
        if status in _REAL:
            st['real'].append(float(ln.get('s') or 0.0))
        elif status in ('cached', 'refused', 'raised'):
            st[status] += 1
    for g, rows in groups.items():
        if rows:
            _table(f'{g.upper()}S, slowest first', rows)
    results = [ln for ln in lines if ln.get('what') == 'result']
    n_by = defaultdict(int)
    for ln in results:
        n_by[ln.get('status')] += 1
    print(f'\nResults written: {len(results)}. By status: {dict(n_by)}')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print('usage: python noise_timing.py <folder or one .timing.jsonl.gz>')
        return 2
    p = Path(argv[0])
    files = ([p] if p.is_file() else sorted(p.rglob('*.timing.jsonl.gz')))
    lines = []
    for f in files:
        lines.extend(read_log(f))
    print(f'{len(files)} logs, {len(lines)} lines')
    summarise(lines)
    return 0


if __name__ == '__main__':
    sys.exit(main())
