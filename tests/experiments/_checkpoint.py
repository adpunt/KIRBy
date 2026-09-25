"""Partial-result sidecars: save each finished unit of an experiment as it
finishes, and read them back so a killed or failed run resumes.

Added 2026-09-25. Before this, every similarity experiment except s11, every
federated experiment and most OPIG scripts held all their results in memory
and wrote one JSON at the very end. A job killed by the wall-clock limit, a
SIGKILL from the memory limit, or one exception in the last model x seed run
threw away every unit that had already finished.

What is stored
--------------
One JSON-lines file next to the experiment's output, named
``<output stem>.partial.jsonl``. It does not end in ``.json``, so the result
readers that glob ``*.json`` never see it.

Line 1 is a header holding the run's IDENTITY: the experiment, the dataset,
the noise condition where there is one, and the input fingerprint
(_shared.input_fingerprint: row count, label hash, group hash, representation
names and widths, metric, distance) when the caller has it. Every later line is
one finished unit: its KEY (fold, model, seed, target, ... -- whatever the
experiment loops over) and its PAYLOAD (exactly the values the loop body would
have stored for that unit).

Resume rule
-----------
- A file whose header identity differs from the current run is moved aside to
  ``<name>.stale`` and the run starts empty. Nothing computed under other labels,
  groups, representations or metric is ever reused.
- ``force_rerun`` moves the file aside the same way. When the caller does not say,
  it is read from ``KIRBY_FORCE_RERUN=1`` (submit_dataset.sh:202) or a
  ``--force-rerun`` on the command line (run_dataset.py:82), so an experiment's
  ``run()`` that is not handed the flag still honours it. It is applied once per
  file per process.
- A unit whose key is present is not recomputed; its stored payload is used.

Exactness
---------
Payloads are encoded so that decoding gives back the same Python/numpy objects:
numpy arrays keep their dtype and shape, numpy scalars keep their type, tuples
stay tuples, dict keys that are not strings (ints, floats, tuples) stay what
they were, and float64 values round-trip exactly (json writes repr()). So the
code after the loop sees the same objects whether a unit was computed now or
read back, and the final JSON is byte-for-byte what an uninterrupted run
writes (apart from its timestamp). An object the encoder does not know (a
fitted model, a blueprint class) raises TypeError: that unit is not saved and
the run carries on, recomputing it on the next attempt.

Atomicity
---------
The header is written to a temporary file and renamed. Each unit is appended
as one line in a single write() on an O_APPEND descriptor, then fsync'd. A kill
mid-append can leave only a partial LAST line; the loader drops a line that
does not parse and rewrites the file without it (again via rename).
"""
import json
import os
import sys
import tempfile

import numpy as np

_TAG = '__ckpt__'
_RESET_THIS_PROCESS = set()


# ── encoding ────────────────────────────────────────────────────────────────

def encode(obj):
    """Turn obj into JSON-safe data that `decode` turns back into obj."""
    # numpy scalars first: np.float64 is a subclass of float and would
    # otherwise come back as a plain float.
    if isinstance(obj, np.generic):
        return {_TAG: 'npscalar', 'dtype': obj.dtype.str, 'value': obj.item()}
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            data = [encode(x) for x in obj.ravel().tolist()]
        else:
            data = obj.ravel().tolist()
        return {_TAG: 'ndarray', 'dtype': obj.dtype.str if obj.dtype != object
                else 'object', 'shape': list(obj.shape), 'data': data}
    if isinstance(obj, tuple):
        return {_TAG: 'tuple', 'items': [encode(x) for x in obj]}
    if isinstance(obj, (set, frozenset)):
        return {_TAG: 'set', 'items': [encode(x) for x in sorted(obj, key=repr)]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj) and _TAG not in obj:
            return {k: encode(v) for k, v in obj.items()}
        return {_TAG: 'dict',
                'items': [[encode(k), encode(v)] for k, v in obj.items()]}
    raise TypeError(f'_checkpoint cannot store a {type(obj).__name__}')


def decode(obj):
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    tag = obj.get(_TAG)
    if tag is None:
        return {k: decode(v) for k, v in obj.items()}
    if tag == 'ndarray':
        if obj['dtype'] == 'object':
            flat = np.empty(len(obj['data']), dtype=object)
            for i, x in enumerate(obj['data']):
                flat[i] = decode(x)
            return flat.reshape(obj['shape'])
        return np.asarray(obj['data'], dtype=np.dtype(obj['dtype'])).reshape(
            obj['shape'])
    if tag == 'npscalar':
        return np.dtype(obj['dtype']).type(obj['value'])
    if tag == 'tuple':
        return tuple(decode(x) for x in obj['items'])
    if tag == 'set':
        return set(decode(x) for x in obj['items'])
    if tag == 'dict':
        return {_hashable(decode(k)): decode(v) for k, v in obj['items']}
    raise ValueError(f'unknown _checkpoint tag {tag!r}')


def _hashable(k):
    return tuple(_hashable(x) for x in k) if isinstance(k, list) else k


def _canon(obj):
    """Canonical JSON text: equal keys/identities give equal strings."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(',', ':'))


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ── paths and identity ──────────────────────────────────────────────────────

def sidecar_for(output_json_path):
    """``<dir>/<stem>.json`` -> ``<dir>/<stem>.partial.jsonl``."""
    p = str(output_json_path)
    if p.endswith('.json'):
        p = p[:-5]
    return p + '.partial.jsonl'


def experiment_sidecar(output_dir, experiment_dir, dataset, suffix=None):
    """Sidecar next to save_experiment_results' file
    ``<output_dir>/<experiment_dir>/<dataset>[__suffix].json``."""
    stem = f'{dataset}__{suffix}' if suffix else f'{dataset}'
    return os.path.join(output_dir, experiment_dir, stem + '.partial.jsonl')


def fingerprint_of(data, config=None):
    """_shared.input_fingerprint, or {} if it cannot be computed here."""
    try:
        from experiments._shared import input_fingerprint
        return input_fingerprint(data, config) or {}
    except Exception:
        return {}


# Environment variables that name a place, a thread count, a log switch or the
# rerun flag itself. They do not change a computed number, so a resubmission
# that sets them differently still resumes. Every other KIRBY_* variable (the
# per-experiment sweeps such as KIRBY_S05_K_VALUES, KIRBY_FED_FOLDS,
# KIRBY_SMOKE_N, KIRBY_DISTANCE_METRIC, ...) goes into the identity, so a run
# under different settings starts its own units instead of reusing these.
_ENV_NOT_IDENTITY = ('DIR', 'PATH', 'THREADS', 'JOBS', 'VERBOSE', 'FORCE_RERUN',
                     'CACHE', 'CKPT', 'LOG', 'REPO', 'ROOT', 'OUT', 'SCRATCH',
                     'BATCH', 'PULL', 'BAK', 'STDERR', 'WARN', 'REPORT',
                     'TRACE', 'STACKDUMP', 'RESULT_DB', 'INSTALL', 'DOWNLOAD',
                     'HANDOFF')


def env_identity():
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith('KIRBY_')
            and not any(t in k[len('KIRBY_'):] for t in _ENV_NOT_IDENTITY)}


def _plain_only(obj):
    """obj restricted to JSON-plain values (callables, arrays, objects
    dropped). Used for a config, whose non-plain entries would print a memory
    address and so never compare equal across processes."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = _plain_only(v)
            if pv is not _DROP:
                out[str(k)] = pv
        return out
    if isinstance(obj, (list, tuple)):
        out = [_plain_only(v) for v in obj]
        return [v for v in out if v is not _DROP]
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return _DROP


_DROP = object()


def run_identity(experiment, dataset, data=None, config=None, **extra):
    """What a sidecar's units were computed under: experiment, dataset, the
    input fingerprint of `data`, the plain part of `config`, the KIRBY_*
    settings (env_identity) and anything in `extra` (noise condition, target,
    strategy)."""
    ident = {'experiment': str(experiment), 'dataset': str(dataset),
             'fingerprint': fingerprint_of(data, config) if data is not None
             else {},
             'config': _plain_only(config) if config is not None else {},
             'env': env_identity()}
    for k, v in extra.items():
        ident[k] = _plain_only(v)
    return ident


def force_rerun_requested(force_rerun=None):
    if force_rerun is not None:
        return bool(force_rerun)
    if os.environ.get('KIRBY_FORCE_RERUN', '').strip() == '1':
        return True
    return '--force-rerun' in sys.argv


# ── the checkpoint ──────────────────────────────────────────────────────────

class Checkpoint:
    """Load / done / get / put over one sidecar file.

    ``ck = Checkpoint(path, identity={...})``
    ``if ck.done(fold=0, model='rf'): r = ck.get(fold=0, model='rf')``
    ``else: r = compute(); ck.put(r, fold=0, model='rf')``

    ``put`` never raises: a unit that cannot be saved prints why and the run
    continues (it is simply recomputed next time). A payload that is a dict
    with a top-level ``'error'`` key is NOT saved: a unit that failed is
    retried on the next attempt, after whatever broke it has been fixed,
    instead of being read back as finished.
    """

    def __init__(self, path, identity, force_rerun=None, verbose=True):
        self.path = str(path)
        self.identity = _plain(identity)
        self.verbose = verbose
        self._records = {}
        self.n_resumed = 0
        force = force_rerun_requested(force_rerun)
        try:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        except OSError:
            pass
        if force and self.path not in _RESET_THIS_PROCESS:
            _RESET_THIS_PROCESS.add(self.path)
            if os.path.exists(self.path):
                self._move_aside('force_rerun')
        self._load()
        if not os.path.exists(self.path):
            self._write_all()
        if self._records and self.verbose:
            print(f'  [partial] {len(self._records)} finished unit(s) read '
                  f'back from {self.path}')

    # -- public --
    def done(self, **key):
        return _canon(key) in self._records

    def get(self, **key):
        k = _canon(key)
        if k not in self._records:
            raise KeyError(key)
        self.n_resumed += 1
        return decode(self._records[k][1])

    def put(self, payload, **key):
        if isinstance(payload, dict) and 'error' in payload:
            return False
        try:
            enc = encode(payload)
            line = json.dumps({'key': _plain(key), 'payload': enc}) + '\n'
        except (TypeError, ValueError) as e:
            print(f'  [partial] not saved {key}: {e}')
            return False
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                         0o644)
            try:
                os.write(fd, line.encode('utf-8'))
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as e:
            print(f'  [partial] could not write {self.path}: {e}')
            return False
        self._records[_canon(key)] = (key, enc)
        return True

    def keys(self):
        return [dict(k) for k, _ in self._records.values()]

    def __len__(self):
        return len(self._records)

    # -- internal --
    def _move_aside(self, why):
        try:
            os.replace(self.path, self.path + '.stale')
            if self.verbose:
                print(f'  [partial] {why}: moved {self.path} -> .stale')
        except OSError as e:
            print(f'  [partial] could not move {self.path} aside: {e}')

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding='utf-8') as fh:
                lines = fh.read().splitlines()
        except OSError as e:
            print(f'  [partial] could not read {self.path}: {e}')
            self._move_aside('unreadable')
            return
        header = None
        bad = 0
        recs = {}
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except ValueError:
                bad += 1
                continue
            if header is None:
                header = obj
                continue
            if isinstance(obj, dict) and 'key' in obj and 'payload' in obj:
                recs[_canon(obj['key'])] = (obj['key'], obj['payload'])
            else:
                bad += 1
        if (not isinstance(header, dict)
                or _canon(header.get('identity')) != _canon(self.identity)):
            self._move_aside('identity changed (experiment, dataset, '
                             'condition or input fingerprint)')
            return
        self._records = recs
        if bad:
            print(f'  [partial] dropped {bad} unreadable line(s) in {self.path}')
            self._write_all()

    def _write_all(self):
        d = os.path.dirname(self.path) or '.'
        try:
            fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps({'identity': self.identity}) + '\n')
                for key, enc in self._records.values():
                    fh.write(json.dumps({'key': _plain(key),
                                         'payload': enc}) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as e:
            print(f'  [partial] could not write {self.path}: {e}')


def cached(ck, compute, **key):
    """Return ck's stored payload for key, else compute(), store it, return it.

    ck may be None (no checkpoint), in which case this is just compute().
    """
    if ck is None:
        return compute()
    if ck.done(**key):
        return ck.get(**key)
    val = compute()
    ck.put(val, **key)
    return val


def resumable_run(experiment_dir, compute, data, config, output_dir,
                  dataset_name, key, **extra_identity):
    """One whole call of an experiment as one unit.

    Used where the loop over units is OUTSIDE the experiment file, as in
    federated_helpers.run_all, which calls each fed experiment's run() once per
    (FLuID model, seed). The call's return value is stored under `key`; the next
    attempt returns it without recomputing. The sidecar sits next to the file
    run_all writes, ``<output_dir>/<experiment_dir>/<dataset>.partial.jsonl``.
    """
    # The key's own fields (fed_model, fed_seed) are also in the config each
    # call receives; left in, every call would carry a different identity and
    # move the previous call's units aside.
    _cfg = ({k: v for k, v in config.items() if k not in key}
            if isinstance(config, dict) else config)
    ck = Checkpoint(
        experiment_sidecar(output_dir, experiment_dir, dataset_name),
        run_identity(experiment_dir, dataset_name, data, _cfg,
                     **extra_identity))
    if ck.done(**key):
        print(f'  [partial] {experiment_dir} {key}: finished in an earlier '
              f'attempt, read back from {ck.path}')
        return ck.get(**key)
    val = compute()
    ck.put(val, **key)
    return val


# ── OPIG scripts: one sidecar per output file, one unit per item ─────────────

def script_checkpoint(script, output_path, args=None, exclude=(),
                      force_rerun=None, **extra_identity):
    """Checkpoint for a standalone script that loops over targets / datasets /
    conditions and writes one JSON at the end.

    The sidecar is ``sidecar_for(output_path)``. The identity is the script
    name, every command-line argument except those in `exclude` (the list of
    items to loop over belongs there, so a run over a subset reuses what an
    earlier run over more items finished, and the output path itself), the
    KIRBY_* settings and `extra_identity`. `force_rerun` defaults to the
    script's own ``--force-rerun`` flag when `args` has one.
    """
    a = dict(vars(args)) if args is not None else {}
    if force_rerun is None and a.get('force_rerun'):
        force_rerun = True
    for k in tuple(exclude) + ('force_rerun',):
        a.pop(k, None)
    ident = {'experiment': str(script), 'args': _plain_only(a),
             'env': env_identity()}
    for k, v in extra_identity.items():
        ident[k] = _plain_only(v)
    return Checkpoint(sidecar_for(output_path), ident, force_rerun=force_rerun)


def is_failed_item(result):
    """A per-item result that recorded a failure instead of a measurement."""
    return isinstance(result, dict) and 'error' in result


def exit_if_failed(failed, what='item'):
    """After everything has been written: exit 1 naming the failed items."""
    if failed:
        print(f"\n{len(failed)} {what}(s) FAILED: {', '.join(map(str, failed))}"
              f" -- results for the rest were written; exiting 1.")
        sys.exit(1)
