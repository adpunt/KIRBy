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

A dataset's screen script follows one contract (b06_dronerf_views.py,
b03_matbench_views.py and b10_reaction_views.py do): every unit is one command; a unit writes its
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


# ── Reactions ──────────────────────────────────────────────────────────────

class Reactions:
    """The three reaction-yield datasets as one row, each dataset a task
    (like Matbench's). Object: one reaction. KIRBy's own outer folds
    (nested_cv_outer_folds, 5 folds, seed 42, the loader's groups). Views:
    the dataset's pool (complexity_helpers.DATASET_REPS[ds]['general'], 10,
    cgr_morgan among them) plus the CGR views below; every view but
    cgr_bond_changes is also scored with cgr_bond_changes appended. Models:
    decided_grid.decided_models(ds, 'general'), read when the row is planned.
    Script: tests/experiments/benchmarks/b10_reaction_views.py."""
    name = 'reactions'
    section = 'NEW_DATASETS.md section 7'
    script = BENCH / 'b10_reaction_views.py'
    tasks = ['buchwald_hartwig', 'bh_doyle_plates', 'suzuki']
    # Built once per dataset (cgr_morgan is in every pool already).
    cgr_views = ['cgr_smiles_lm', 'chytorch_graphormer', 'cgr_atompair', 'cgr_torsion',
                 'cgr_bond_changes']
    # Trained on each outer fold's training rows: one build per fold. DOPtools'
    # cgr_circus and cgr_chyline dropped 2026-11-02 (chython conflict with chytorch).
    fold_views = ['chemprop_cgr']
    aug_view = 'cgr_bond_changes'
    plus = 'plus_bond_changes'
    n_folds = 5
    mem = 'small'

    def _pool(self, task):
        from experiments.complexity.complexity_helpers import DATASET_REPS
        return list(DATASET_REPS[task]['general'])

    def _models(self, task):
        from experiments.decided_grid import decided_models
        return list(decided_models(task, 'general'))

    @property
    def models(self):
        out = []
        for t in self.tasks:
            out += [m for m in self._models(t) if m not in out]
        return out

    @property
    def all_models(self):
        return self.models

    def fixed_views_for(self, task):
        p = self._pool(task)
        return p + [v for v in self.cgr_views if v not in p]

    def views_for(self, task):
        return self.fixed_views_for(task) + self.fold_views

    def views(self):
        out = []
        for t in self.tasks:
            out += [v for v in self.views_for(t) if v not in out]
        return out

    def paths(self, run_root):
        cache = _need_env('KIRBY_CACHE_DIR', 'the KIRBy cache root; the pool views are read from '
                          '<KIRBY_CACHE_DIR>/rep_cache/<dataset>/ and the reaction files from '
                          '<KIRBY_CACHE_DIR>/reaction_yields/')
        return {'cache': cache,
                'cgr_checkpoint': os.environ.get('KIRBY_CGR_CHECKPOINT') or '',
                'results': str(Path(run_root) / self.name)}

    def build_units(self):
        out = []
        for t in self.tasks:
            out += [f'{v}__build__{t}' for v in self.fixed_views_for(t)]
            for v in self.fold_views:
                out += [f'{v}__build__{t}__f{k}' for k in range(self.n_folds)]
        return out

    def evaluate_units(self, models):
        out = []
        for t in self.tasks:
            ms = [m for m in models if m in self._models(t)]
            for v in self.views_for(t):
                for m in ms:
                    out.append(f'{v}__{m}__{t}')
                    if v != self.aug_view:
                        out.append(f'{v}__{m}__{t}__{self.plus}')
        return out

    def _view_builds(self, v, t):
        if v in self.fold_views:
            return [f'{v}__build__{t}__f{k}' for k in range(self.n_folds)]
        return [f'{v}__build__{t}']

    def deps(self, key):
        if key == 'summary':
            return []
        parts = key.split('__')
        if parts[1] == 'build':
            return []
        v, t = parts[0], parts[2]
        out = self._view_builds(v, t)
        if len(parts) > 3 and parts[3] == self.plus:
            out += self._view_builds(self.aug_view, t)
        return out

    def env(self, row):
        p = row['paths']
        e = {'KIRBY_CACHE_DIR': p['cache'], 'KIRBY_BENCH_RESULTS': p['results']}
        if p.get('cgr_checkpoint'):
            e['KIRBY_CGR_CHECKPOINT'] = p['cgr_checkpoint']
        return e

    def command(self, row, key):
        py = os.environ.get('KIRBY_SCREEN_PY') or sys.executable
        base = [py, '-u', str(self.script)]
        if key == 'summary':
            return base + ['--stage', 'summarise']
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            cmd = base + ['--stage', 'build', '--tasks', parts[2], '--views', v]
            if len(parts) > 3:
                cmd += ['--folds', parts[3][1:]]
            return cmd
        m, t = parts[1], parts[2]
        cmd = base + ['--stage', 'evaluate', '--tasks', t, '--views', v, '--models', m]
        if len(parts) > 3 and parts[3] == self.plus:
            cmd.append('--plus-bond-changes')
        return cmd

    def check(self, row, key):
        res = Path(row['paths']['results']) / 'b10_views'
        if key == 'summary':
            return _file_ok(res / 'summary.md')
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            t = parts[2]
            if len(parts) > 3:
                return _file_ok(res / 'fold_views' / t / f'{v}_fold{parts[3][1:]}.npz')
            return _json_ok(res / 'built' / t / f'{v}.json')
        m, t = parts[1], parts[2]
        tail = f'__{self.plus}' if len(parts) > 3 and parts[3] == self.plus else ''
        return _json_ok(res / t / f'{v}__{m}{tail}.json')

    def summary_file(self, row):
        return Path(row['paths']['results']) / 'b10_views' / 'summary.md'


# ── UCR Crop ───────────────────────────────────────────────────────────────

class UCRCrop:
    """NEW_DATASETS.md section "UCR Crop". Object: one pixel's 46-value NDVI
    series; 24 classes. The archive's fixed split (7,200 training, 16,800
    test rows), one fit per unit. Every view is scored alone and with each
    small block in `aug_blocks` appended. The `published` units run the
    published classifiers end to end (aeon 1.3; InceptionTime ported to
    PyTorch). Script: tests/experiments/benchmarks/b11_ucr_crop_views.py."""
    name = 'ucr_crop'
    section = 'NEW_DATASETS.md section "UCR Crop"'
    script = BENCH / 'b11_ucr_crop_views.py'
    view_list = ['raw', 'diff1', 'diff2', 'periodogram', 'harmonic2', 'harmonic3', 'catch24',
                 'tsfresh', 'rocket', 'minirocket', 'multirocket', 'hydra', 'quant', 'rdst',
                 'weasel2', 'random_intervals', 'supervised_intervals', 'ed_medoid',
                 'dtw_medoid', 'moment', 'mantis', 'chronos2', 'ts2vec', 'inceptiontime']
    # Phenology's three settings (smoother, Savitzky-Golay window in time
    # steps, threshold as a fraction of the amplitude) have no value. While
    # this is None the view and its units are not in the row.
    phenology_settings = None
    # Small blocks appended to every other view, each one in turn.
    aug_blocks = ['harmonic2', 'harmonic3', 'catch24']
    published = ['multirocket', 'minirocket', 'rocket', 'hydra', 'multirocket_hydra', 'quant',
                 'rdst', 'weasel2', 'catch22', 'freshprince', 'drcif', 'rstsf', 'tsf',
                 'knn1_ed', 'knn1_dtw', 'inceptiontime']
    models = ['ridge_cv', 'extra_trees', 'rf', 'rotation_forest', 'logreg', 'svm', 'knn1']
    all_models = models
    mem = 'small'

    def views(self):
        return self.view_list + (['phenology'] if self.phenology_settings else [])

    def blocks(self):
        return self.aug_blocks + (['phenology'] if self.phenology_settings else [])

    def paths(self, run_root):
        return {'data': os.environ.get('KIRBY_CROP_DIR') or '/Volumes/seagate/benchmarks_protocol/ucr_crop',
                'ts2vec': os.environ.get('KIRBY_TS2VEC_DIR') or '',
                'results': str(Path(run_root) / self.name)}

    def build_units(self):
        return ['extract'] + [f'{v}__build' for v in self.views()]

    def evaluate_units(self, models):
        out = [f'published__{p}' for p in self.published]
        for v in self.views():
            for m in models:
                out.append(f'{v}__{m}')
                out += [f'{v}__{m}__plus_{b}' for b in self.blocks() if b != v]
        return out

    def deps(self, key):
        if key in ('extract', 'summary'):
            return []
        parts = key.split('__')
        if parts[0] == 'published':
            return ['extract'] + (['inceptiontime__build'] if parts[1] == 'inceptiontime' else [])
        if parts[1] == 'build':
            return ['extract']
        out = [f'{parts[0]}__build']
        if len(parts) > 2:
            out.append(f'{parts[2][len("plus_"):]}__build')
        return out

    def env(self, row):
        p = row['paths']
        e = {'KIRBY_CROP_DIR': p['data'], 'KIRBY_BENCH_RESULTS': p['results']}
        if p.get('ts2vec'):
            e['KIRBY_TS2VEC_DIR'] = p['ts2vec']
        return e

    def command(self, row, key):
        py = os.environ.get('KIRBY_SCREEN_PY') or sys.executable
        base = [py, '-u', str(self.script)]
        if key == 'extract':
            return base + ['--stage', 'data']
        if key == 'summary':
            return base + ['--stage', 'summarise']
        parts = key.split('__')
        if parts[0] == 'published':
            return base + ['--stage', 'published', '--models', parts[1]]
        if parts[1] == 'build':
            cmd = base + ['--stage', 'build', '--views', parts[0]]
            if parts[0] == 'phenology':
                s, w, t = self.phenology_settings
                cmd += ['--pheno-smoother', str(s), '--pheno-window', str(w),
                        '--pheno-threshold', str(t)]
            return cmd
        cmd = base + ['--stage', 'evaluate', '--views', parts[0], '--models', parts[1]]
        if len(parts) > 2:
            cmd += ['--plus', parts[2][len('plus_'):]]
        return cmd

    def check(self, row, key):
        data = Path(row['paths']['data'])
        res = Path(row['paths']['results']) / 'b11_views' / 'full'
        if key == 'extract':
            return _file_ok(data / 'crop.npz')
        if key == 'summary':
            return _file_ok(res / 'summary.md')
        parts = key.split('__')
        if parts[0] == 'published':
            return _json_ok(res / 'published' / f'{parts[1]}.json')
        if parts[1] == 'build':
            return _file_ok(data / 'views' / f'{parts[0]}.npy')
        tail = f'__{parts[2]}' if len(parts) > 2 else ''
        return _json_ok(res / f'{parts[0]}__{parts[1]}{tail}.json')

    def summary_file(self, row):
        return Path(row['paths']['results']) / 'b11_views' / 'full' / 'summary.md'


# ── LSST (UEA) ─────────────────────────────────────────────────────────────

class LSST:
    """NEW_DATASETS.md section "LSST". Object: one light curve, 6 passbands x
    36 flux values; 2,459 training and 2,466 test curves, 14 classes. Split:
    the archive's default train/test files. Resamples r1 ... r29 (Ruiz et al.
    2021) exist in the script and are not in the row until chosen.
    Script: tests/experiments/benchmarks/b12_lsst_views.py."""
    name = 'lsst'
    section = 'NEW_DATASETS.md section "LSST"'
    script = BENCH / 'b12_lsst_views.py'
    unfitted_views = ['raw', 'band_stats', 'band_colours', 'index_periodogram', 'catch24',
                      'tsfresh', 'mantis', 'mantisv2', 'moment', 'chronos2']
    fitted_views = ['rocket', 'minirocket', 'multirocket', 'hydra', 'quant', 'rdst', 'ts2vec']
    # Every classifier a paper used on one of these views (b12 docstring).
    models = ['ridgecv', 'rotf', 'rf', 'extratrees', '1nn', 'logreg', 'svm']
    all_models = models
    splits = ['default']
    mem = 'small'

    def views(self):
        return self.unfitted_views + self.fitted_views

    def paths(self, run_root):
        data = _need_env('KIRBY_LSST_DIR', 'the folder for LSST raw/ and views/')
        return {'data': data, 'results': str(Path(run_root) / self.name),
                'ts2vec': os.environ.get('KIRBY_TS2VEC_DIR') or ''}

    def build_units(self):
        return ['data'] + [f'{v}__build__{s}' for v in self.views() for s in self.splits]

    def evaluate_units(self, models):
        return [f'{v}__{m}__{s}' for v in self.views() for m in models for s in self.splits]

    def deps(self, key):
        if key in ('data', 'summary'):
            return []
        parts = key.split('__')
        if parts[1] == 'build':
            return ['data']
        return [f'{parts[0]}__build__{parts[2]}']

    def env(self, row):
        p = row['paths']
        e = {'KIRBY_LSST_DIR': p['data'], 'KIRBY_BENCH_RESULTS': p['results']}
        if p.get('ts2vec'):
            e['KIRBY_TS2VEC_DIR'] = p['ts2vec']
        return e

    def command(self, row, key):
        py = os.environ.get('KIRBY_SCREEN_PY') or sys.executable
        base = [py, '-u', str(self.script)]
        if key == 'data':
            return base + ['--stage', 'data']
        if key == 'summary':
            return base + ['--stage', 'summarise']
        parts = key.split('__')
        if parts[1] == 'build':
            return base + ['--stage', 'build', '--views', parts[0], '--splits', parts[2]]
        return base + ['--stage', 'evaluate', '--views', parts[0], '--models', parts[1],
                       '--splits', parts[2]]

    def check(self, row, key):
        data = Path(row['paths']['data'])
        res = Path(row['paths']['results']) / 'b12_lsst_views'
        if key == 'data':
            return _file_ok(data / 'lsst.npz')
        if key == 'summary':
            return _file_ok(res / 'summary.md')
        parts = key.split('__')
        if parts[1] == 'build':
            return _file_ok(data / 'views' / parts[2] / f'{parts[0]}.npz')
        return _json_ok(res / parts[2] / f'{parts[0]}__{parts[1]}.json')

    def summary_file(self, row):
        return Path(row['paths']['results']) / 'b12_lsst_views' / 'summary.md'


# ── WiSig ──────────────────────────────────────────────────────────────────

class WiSig:
    """NEW_DATASETS.md section "WiSig". Object: one WiFi packet, 256 complex
    samples. Label: the transmitter. The authors' protocols
    (b13_wisig_published.py): ntx_<N> on ManyTx, nrx_<K> on ManyRx (5 receiver
    orders = 5 folds), ndays_<D> and nsig_<K> on ManySig, each with an
    equalisation flag. Script: tests/experiments/benchmarks/b13_wisig_views.py.

    Two OPEN settings with no default: KIRBY_WISIG_PROTOCOLS at init (comma
    list of PROTOCOL:EQ, e.g. ntx_150:0,nrx_25:1), stamped into the row, and
    the models (`--models wisig=...`). Init refuses without either."""
    name = 'wisig'
    section = 'NEW_DATASETS.md section "WiSig"'
    script = BENCH / 'b13_wisig_views.py'
    packet_views = ['iq', 'polar', 'phase_increment', 'cfo_compensated_iq', 'impairments',
                    'fft_magnitude', 'welch_psd', 'spectral_stats', 'stft_spectrogram',
                    'channel_independent_spectrogram', 'stf_response', 'preamble_error', 'dctf',
                    'hos_cumulants']
    fold_views = ['cnn_emb']
    sequence_views = ['iq', 'cfo_compensated_iq', 'polar']     # what wisig_cnn takes
    all_models = ['wisig_cnn', 'knn', 'rf', 'logreg', 'xgboost']
    subset_of = {'ntx': 'ManyTx', 'nrx': 'ManyRx', 'ndays': 'ManySig', 'nsig': 'ManySig'}
    n_folds_of = {'ntx': 1, 'nrx': 5, 'ndays': 1, 'nsig': 1}
    mem = 'small'

    @property
    def models(self):
        raise SystemExit('wisig: the models are an open setting with no default; pass '
                         '--models wisig=m1,m2 (candidates: ' + ','.join(self.all_models) + ')')

    def _kind(self, protocol):
        return protocol.rsplit('_', 1)[0]

    def _settings(self, row=None):
        raw = row['paths']['protocols'] if row else os.environ.get('KIRBY_WISIG_PROTOCOLS', '')
        if not raw:
            raise SystemExit('set KIRBY_WISIG_PROTOCOLS before init: the protocol settings are an '
                             'open setting with no default, e.g. ntx_150:0,nrx_25:0,nrx_25:1')
        out = []
        for s in raw.split(','):
            p, eq = s.split(':')
            if self._kind(p) not in self.subset_of or eq not in ('0', '1'):
                raise SystemExit(f'wisig: bad protocol setting {s!r}')
            out.append((p, eq))
        return out

    def _subset_flags(self, settings):
        out = []
        for p, eq in settings:
            k = (self.subset_of[self._kind(p)], eq)
            if k not in out:
                out.append(k)
        return out

    def views(self):
        return self.packet_views + self.fold_views

    def paths(self, run_root):
        data = _need_env('KIRBY_WISIG_DIR', 'the folder holding compact/<subset>.pkl')
        return {'data': data, 'protocols': ','.join(f'{p}:{e}' for p, e in self._settings()),
                'results': str(Path(run_root) / self.name)}

    def build_units(self):
        st = self._settings()
        out = []
        for sub, eq in self._subset_flags(st):
            out.append(f'packets__build__{sub}__eq{eq}')
            out += [f'{v}__build__{sub}__eq{eq}' for v in self.packet_views]
        for p, eq in st:
            out += [f'cnn_emb__build__{p}__eq{eq}__f{k}' for k in range(self.n_folds_of[self._kind(p)])]
        return out

    def evaluate_units(self, models):
        return [f'{v}__{m}__{p}__eq{eq}' for p, eq in self._settings() for v in self.views()
                for m in models if m != 'wisig_cnn' or v in self.sequence_views]

    def deps(self, key):
        if key == 'summary':
            return []
        parts = key.split('__')
        if parts[1] == 'build':
            if parts[0] == 'packets':
                return []
            if parts[0] in self.fold_views:
                return [f'packets__build__{self.subset_of[self._kind(parts[2])]}__{parts[3]}']
            return [f'packets__build__{parts[2]}__{parts[3]}']
        v, p, eq = parts[0], parts[2], parts[3]
        if v in self.fold_views:
            return [f'{v}__build__{p}__{eq}__f{k}' for k in range(self.n_folds_of[self._kind(p)])]
        return [f'{v}__build__{self.subset_of[self._kind(p)]}__{eq}']

    def env(self, row):
        p = row['paths']
        return {'KIRBY_WISIG_DIR': p['data'], 'KIRBY_BENCH_RESULTS': p['results']}

    def command(self, row, key):
        py = os.environ.get('KIRBY_SCREEN_PY') or sys.executable
        base = [py, '-u', str(self.script)]
        if key == 'summary':
            return base + ['--stage', 'summarise', '--protocols', row['paths']['protocols']]
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            if v == 'packets':
                return base + ['--stage', 'packets', '--subset', parts[2], '--equalized', parts[3][2:]]
            if v in self.fold_views:
                return base + ['--stage', 'embed', '--protocols', f'{parts[2]}:{parts[3][2:]}',
                               '--folds', parts[4][1:]]
            return base + ['--stage', 'views', '--subset', parts[2], '--equalized', parts[3][2:],
                           '--views', v]
        m, p, eq = parts[1], parts[2], parts[3][2:]
        return base + ['--stage', 'evaluate', '--protocols', f'{p}:{eq}', '--views', v, '--models', m]

    def check(self, row, key):
        data = Path(row['paths']['data'])
        res = Path(row['paths']['results']) / 'b13_wisig_views'
        if key == 'summary':
            return _file_ok(res / 'summary.md')
        parts = key.split('__')
        v = parts[0]
        if parts[1] == 'build':
            if v == 'packets':
                return _file_ok(data / 'packets' / f'{parts[2]}_{parts[3]}.meta.json')
            if v in self.fold_views:
                return _file_ok(data / 'views' / 'fold_views' / f'{parts[2]}__{parts[3]}'
                                / f'{v}_fold{parts[4][1:]}.npy')
            return _file_ok(data / 'views' / f'{parts[2]}_{parts[3]}' / f'{v}.npy')
        m, p, eq = parts[1], parts[2], parts[3]
        return _json_ok(res / f'{p}__{eq}' / f'{v}__{m}.json')

    def summary_file(self, row):
        return Path(row['paths']['results']) / 'b13_wisig_views' / 'summary.md'


# ── BEANS bats ─────────────────────────────────────────────────────────────

class BeansBats:
    """NEW_DATASETS.md section "BEANS bats". Object: one call, 250 kHz WAV.
    Label: which of 10 bats made it. Split: the BEANS 6,000 / 2,000 / 2,000
    calls (`published`); grouped split not built (how to group is open).
    Script: tests/experiments/benchmarks/beans_bats_views.py.

    `extract` writes calls.csv and mfcc_stats in KIRBy's environment. Every
    other view is one build unit run by the interpreter in
    KIRBY_SCREEN_PY_BEANS_EMBED (avex, multiband-audio, transformers 4.45),
    stamped into the row at init.

    The models are an OPEN setting with no default (`--models beans_bats=...`);
    the candidates built are the BEANS five (lr, svm, decisiontree, gbdt,
    xgboost). Init refuses without them."""
    name = 'beans_bats'
    section = 'NEW_DATASETS.md section "BEANS bats"'
    script = BENCH / 'beans_bats_views.py'
    all_models = ['lr', 'svm', 'decisiontree', 'gbdt', 'xgboost']
    splits = ['published']
    mem = 'small'

    def _views_module(self):
        import importlib.util
        sp = importlib.util.spec_from_file_location('beans_bats_views', str(self.script))
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        return mod

    @property
    def models(self):
        raise SystemExit('beans_bats: the models are an open setting with no default; pass '
                         '--models beans_bats=m1,m2 (candidates: ' + ','.join(self.all_models) + ')')

    def hand_views(self):
        return list(self._views_module().HAND_VIEWS)

    def embed_views(self):
        return list(self._views_module().EMBED_VIEWS)

    def views(self):
        return self.hand_views() + self.embed_views()

    def paths(self, run_root):
        data = _need_env('KIRBY_BEANS_DIR', 'the folder holding egyptian_fruit_bats.zip or audio/ '
                         '(tests/experiments/benchmarks/beans_bats_published.py --stage fetch)')
        py = _need_env('KIRBY_SCREEN_PY_BEANS_EMBED', 'the python with avex, multiband-audio and '
                       'transformers that builds the embedding views')
        return {'data': data, 'embed_py': py, 'results': str(Path(run_root) / self.name)}

    def build_units(self):
        return ['extract'] + [f'{v}__build' for v in self.embed_views()]

    def evaluate_units(self, models):
        return [f'{v}__{m}__{s}' for v in self.views() for m in models for s in self.splits]

    def deps(self, key):
        if key in ('extract', 'summary'):
            return []
        parts = key.split('__')
        if parts[1] == 'build':
            return ['extract']
        v = parts[0]
        return ['extract'] if v in self.hand_views() else [f'{v}__build']

    def env(self, row):
        p = row['paths']
        return {'KIRBY_BEANS_DIR': p['data'], 'KIRBY_BENCH_RESULTS': p['results']}

    def command(self, row, key):
        py = os.environ.get('KIRBY_SCREEN_PY') or sys.executable
        base = [py, '-u', str(self.script)]
        if key == 'extract':
            return base + ['--stage', 'extract']
        if key == 'summary':
            return base + ['--stage', 'summarise']
        parts = key.split('__')
        if parts[1] == 'build':
            return [row['paths']['embed_py'], '-u', str(self.script), '--stage', 'build', '--views', parts[0]]
        return base + ['--stage', 'evaluate', '--views', parts[0], '--models', parts[1], '--splits', parts[2]]

    def check(self, row, key):
        data = Path(row['paths']['data']) / 'views'
        res = Path(row['paths']['results']) / 'beans_bats_views'
        if key == 'extract':
            return _file_ok(data / 'mfcc_stats.npy')
        if key == 'summary':
            return _file_ok(res / 'summary.md')
        parts = key.split('__')
        if parts[1] == 'build':
            return _file_ok(data / f'{parts[0]}.npy')
        return _json_ok(res / parts[2] / f'{parts[0]}__{parts[1]}.json')

    def summary_file(self, row):
        return Path(row['paths']['results']) / 'beans_bats_views' / 'summary.md'


SPECS = {s.name: s for s in (DroneRF(), Matbench(), Reactions(), UCRCrop(), LSST(), WiSig(), BeansBats())}


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
    models = list(models) if models else list(sp.models)
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
