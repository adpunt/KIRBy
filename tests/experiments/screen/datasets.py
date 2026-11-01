"""What each candidate dataset's screen consists of: its units, the command
that runs each one, the file that proves it finished, and what it waits for.

A screen asks one question per candidate dataset: how good is each view
(representation) on its own, under the models the dataset's papers used, and
how much do the views overlap. It feeds the choice of up to ten views per
dataset in NEW_DATASETS.md.

Every dataset row has three stages:
    build      make the views (feature files); some views are trained per
               fold and so are built per (task, split, fold)
    evaluate   one unit per (view, model, task[, split]): every outer fold of
               the published protocol, out-of-fold predictions kept
    summarise  one unit: the dataset's summary.md (scores, overlap between
               views); opens when evaluate is closed

Unit keys start with the view name (`spectrum__rf__10__grouped`,
`soap_avg__build__matbench_phonons`), so `ledger.py skip-view soap_avg`
stops every unit of a view on every task. `extract`, `targets` and `summary`
are the keys that are not views.

A dataset's screen script follows one contract (b06_dronerf_views.py and
b03_matbench_views.py do): every unit is one command; a unit writes its
result file last, by rename or after everything else; an evaluate result is
JSON with result_status 'ok'; a unit killed part way resumes from what it
wrote (folds, chunks). Paths come from environment variables the row stamps
at init (row['paths']), so the same ledger works on the laptop and the
server.

Adding a dataset: write its screen script to the contract, add a Spec below,
then `ledger.py extend --datasets <name>` on a live run.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
BENCH = TESTS / 'experiments' / 'benchmarks'

BUILD, EVALUATE, SUMMARISE = 'build', 'evaluate', 'summarise'


def _need_env(var, why):
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f'set {var} before init: {why}')
    return v


def _json_ok(path, want_debug=False):
    """(status, error) for an evaluate result file."""
    p = Path(path)
    if not p.exists():
        part = p.with_name(p.stem + '.partial.json')
        extra = ' (a partial file exists: it resumes)' if part.exists() else ''
        return 'failed', f'exited 0 but wrote no result file {p}{extra}'
    try:
        d = json.loads(p.read_text())
    except Exception as exc:
        return 'failed', f'result file does not parse: {type(exc).__name__}: {exc}'
    if d.get('result_status') != 'ok':
        return 'failed', f"result file says result_status={d.get('result_status')!r}"
    if bool(d.get('debug')) != want_debug:
        return 'failed', 'result file is from a --debug run'
    return 'ok', None


def _file_ok(path):
    p = Path(path)
    return ('ok', None) if p.exists() else ('failed', f'exited 0 but {p} is missing')


# ── DroneRF ────────────────────────────────────────────────────────────────

class DroneRF:
    """NEW_DATASETS.md section 6. Object: one 100,000-sample chunk. 22,700
    chunks, 227 recordings. Tasks 2, 4, 10 classes. Splits: published (10-fold
    over chunks, seed 1) and grouped (10-fold by recording, seed 1).
    Script: tests/experiments/benchmarks/b06_dronerf_views.py."""
    name = 'dronerf'
    section = 'NEW_DATASETS.md section 6'
    script = BENCH / 'b06_dronerf_views.py'
    hand_views = ['spectrum', 'spectrum_low', 'spectrum_high', 'welch_psd',
                  'spectral_scalars', 'stats']
    image_views = ['psd_resnet50', 'spectrogram_resnet50', 'histogram_resnet50', 'gaf_resnet50']
    fold_views = ['rfuavnet_emb', 'alemadi_emb']
    # Every vector model a DroneRF paper used (owner 2026-10-30). svm is not
    # in the default: Kilic's grid (2^-15..2^15 for C and gamma, inner CV) is
    # about 460,000 fits and the owner has not chosen a smaller one yet.
    models = ['knn', 'rf', 'logreg', 'xgboost', 'dense']
    all_models = models + ['svm']
    tasks = ['2', '4', '10']
    splits = ['published', 'grouped']
    n_folds = 10
    mem = 'small'

    def views(self):
        return self.hand_views + self.image_views + self.fold_views

    def paths(self, run_root):
        data = _need_env('KIRBY_DRONERF_DIR', 'the folder holding raw/ (the 454 CSVs) and views/')
        return {'data': data, 'results': str(Path(run_root) / self.name)}

    def build_units(self):
        out = ['extract'] + [f'{v}__build' for v in self.image_views]
        for v in self.fold_views:
            for t in self.tasks:
                for s in self.splits:
                    out += [f'{v}__build__{t}__{s}__f{k}' for k in range(self.n_folds)]
        return out

    def evaluate_units(self, models):
        return [f'{v}__{m}__{t}__{s}' for v in self.views() for m in models
                for t in self.tasks for s in self.splits]

    def deps(self, key):
        if key in ('extract', 'summary'):
            return []
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            return ['extract']
        if v in self.hand_views:
            return ['extract']
        if v in self.image_views:
            return [f'{v}__build']
        t, s = parts[2], parts[3]
        return [f'{v}__build__{t}__{s}__f{k}' for k in range(self.n_folds)]

    def env(self, row):
        p = row['paths']
        return {'KIRBY_DRONERF_DIR': p['data'], 'KIRBY_BENCH_RESULTS': p['results']}

    def command(self, row, key):
        py = os.environ.get('KIRBY_SCREEN_PY') or sys.executable
        base = [py, '-u', str(self.script)]
        if key == 'extract':
            return base + ['--stage', 'extract']
        if key == 'summary':
            return base + ['--stage', 'summarise']
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            if v in self.image_views:
                return base + ['--stage', 'images', '--views', v]
            t, s, f = parts[2], parts[3], parts[4][1:]
            return base + ['--stage', 'embed', '--views', v, '--tasks', t, '--splits', s, '--folds', f]
        m, t, s = parts[1], parts[2], parts[3]
        return base + ['--stage', 'evaluate', '--views', v, '--models', m, '--tasks', t, '--splits', s]

    def check(self, row, key):
        data = Path(row['paths']['data']) / 'views'
        res = Path(row['paths']['results']) / 'b06_views'
        if key == 'extract':
            return _file_ok(data / 'extract_done.json')
        if key == 'summary':
            return _file_ok(res / 'summary.md')
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            if v in self.image_views:
                return _file_ok(data / f'{v}.npy')
            t, s, f = parts[2], parts[3], parts[4][1:]
            return _file_ok(data / 'fold_views' / s / t / f'{v}_fold{f}.npy')
        m, t, s = parts[1], parts[2], parts[3]
        return _json_ok(res / s / t / f'{v}__{m}.json')

    def summary_file(self, row):
        return Path(row['paths']['results']) / 'b06_views' / 'summary.md'


# ── Matbench ───────────────────────────────────────────────────────────────

class Matbench:
    """NEW_DATASETS.md section 3. Object: one compound. Six regression tasks,
    Matbench's own 5 folds. Views: the 13 representations of
    src/kirby/representations/materials.py (expt_gap: the 4 composition ones).
    Script: tests/experiments/benchmarks/b03_matbench_views.py; caches built
    by tests/tools/matbench_build_reps.py --worker TASK REP."""
    name = 'matbench'
    section = 'NEW_DATASETS.md section 3'
    script = BENCH / 'b03_matbench_views.py'
    builder = TESTS / 'tools' / 'matbench_build_reps.py'
    tasks = ['matbench_jdft2d', 'matbench_phonons', 'matbench_expt_gap',
             'matbench_dielectric', 'matbench_log_gvrh', 'matbench_perovskites']
    composition_views = ['magpie', 'stoichiometry', 'valence_orbital', 'ion_property']
    structure_views = ['sine_coulomb', 'soap_avg', 'site_stats_fingerprint', 'density_features',
                       'rdf', 'xrd', 'global_symmetry', 'chgnet', 'mace_mp']
    # The two leaderboard models. modnet is not in the default: it needs its
    # own python 3.8 environment (modnet 0.1.12), which is not on the server
    # yet. Add it with `extend --models matbench=rf,modnet` once
    # KIRBY_SCREEN_PY_MODNET points at that interpreter.
    models = ['rf']
    all_models = ['rf', 'modnet']
    mem = 'small'

    def views_for(self, task):
        return self.composition_views + ([] if task == 'matbench_expt_gap' else self.structure_views)

    def views(self):
        return self.composition_views + self.structure_views

    def paths(self, run_root):
        return {'reps': os.environ.get('KIRBY_MATBENCH_REPS') or '/Volumes/seagate/matbench/reps',
                'data': os.environ.get('MATMINER_DATA') or '/Volumes/seagate/matbench/data',
                'protocol': (os.environ.get('KIRBY_MATBENCH_PROTOCOL')
                             or '/Volumes/seagate/benchmarks_protocol/matbench'),
                'results': str(Path(run_root) / self.name)}

    def build_units(self):
        out = []
        for t in self.tasks:
            out.append(f'targets__build__{t}')
            out += [f'{v}__build__{t}' for v in self.views_for(t)]
        return out

    def evaluate_units(self, models):
        return [f'{v}__{m}__{t}' for t in self.tasks for v in self.views_for(t) for m in models]

    def deps(self, key):
        if key == 'summary':
            return []
        parts = key.split('__')
        if parts[1] == 'build':
            return []
        v, t = parts[0], parts[2]
        return [f'{v}__build__{t}', f'targets__build__{t}']

    def env(self, row):
        p = row['paths']
        return {'KIRBY_MATBENCH_REPS': p['reps'], 'MATMINER_DATA': p['data'],
                'KIRBY_MATBENCH_PROTOCOL': p['protocol'], 'KIRBY_BENCH_RESULTS': p['results']}

    def command(self, row, key):
        py = os.environ.get('KIRBY_SCREEN_PY_MATBENCH') or sys.executable
        if key == 'summary':
            return [py, '-u', str(self.script), '--stage', 'summarise']
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            t = parts[2]
            if v == 'targets':
                return [py, '-u', str(self.script), '--stage', 'targets', '--tasks', t]
            return [py, '-u', str(self.builder), '--worker', t, v]
        m, t = parts[1], parts[2]
        if m == 'modnet':
            py = os.environ.get('KIRBY_SCREEN_PY_MODNET')
            if not py:
                raise RuntimeError('modnet units need KIRBY_SCREEN_PY_MODNET (the python 3.8 '
                                   'environment with modnet 0.1.12)')
        return [py, '-u', str(self.script), '--stage', 'evaluate', '--tasks', t,
                '--views', v, '--models', m]

    def check(self, row, key):
        reps = Path(row['paths']['reps'])
        res = Path(row['paths']['results']) / 'b03_views'
        if key == 'summary':
            return _file_ok(res / 'summary.md')
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            t = parts[2]
            return _file_ok(reps / t / ('_targets.npz' if v == 'targets' else f'{v}.npy'))
        m, t = parts[1], parts[2]
        return _json_ok(res / t / f'{v}__{m}.json')

    def summary_file(self, row):
        return Path(row['paths']['results']) / 'b03_views' / 'summary.md'


SPECS = {s.name: s for s in (DroneRF(), Matbench())}


def spec(row):
    return SPECS[row['dataset']]


def _unit():
    return {'status': 'pending', 'attempts': 0, 'seconds': None, 'job': None,
            'host': None, 'started': None, 'ended': None, 'log': None,
            'error': None, 'rc': None}


def _stage(name, keys, after=None):
    keys = list(keys)
    return {'name': name, 'status': 'pending' if keys else 'empty', 'started': None,
            'ended': None, 'seconds': None, 'after': after or [],
            'units': {k: _unit() for k in keys}}


def build_row(name, run_root, models=None, priority=None):
    sp = SPECS[name]
    models = list(models or sp.models)
    bad = [m for m in models if m not in sp.all_models]
    if bad:
        raise SystemExit(f'{name}: unknown model(s) {bad}; known: {sp.all_models}')
    return {'kind': 'screen', 'dataset': name, 'mem': sp.mem,
            'priority': priority if priority is not None else list(SPECS).index(name) + 1,
            'claim': None, 'models': models, 'paths': sp.paths(run_root),
            'section': sp.section,
            'stages': [_stage(BUILD, sp.build_units()),
                       _stage(EVALUATE, sp.evaluate_units(models)),
                       _stage(SUMMARISE, ['summary'], after=[(name, EVALUATE)])]}


def command_for(row_name, row, stage_name, unit_key, run_root):
    """runner.command_for's signature: (argv, env, result_dir)."""
    sp = spec(row)
    env = dict(os.environ)
    env.pop('KIRBY_RUN', None)                 # the experiments' run; not ours
    env.update({'KIRBY_SCREEN': str(run_root), 'PYTHONUNBUFFERED': '1',
                'PYTHONWARNINGS': env.get('PYTHONWARNINGS', 'ignore')})
    env.update(sp.env(row))
    return sp.command(row, unit_key), env, None
