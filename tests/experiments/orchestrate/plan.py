"""Which rows, stages and units a run holds.

A ROW is one dataset (41, plus herg_ic50 for the six noise experiments that
read its repeat measurements) or one multi-dataset script (the 11 OPIG rows
of submit_everything.sh's OPIG_JOBS, and the three cross-dataset scripts
s09, c03, c14).

A dataset row's STAGES, in the order Adelaide set (2026-10-05):
    complexity   per-fold searches (one unit per outer fold) then the
                 complexity experiments
    federated    lhasa_herg, caco2, logd only
    similarity   every non-nucleotide dataset
    noise        the one-fold screen: a many-method noise experiment runs
                 fold 0 only (noise_helpers.one_fold_screen_config); the 8
                 descriptive ones run every fold
    noise_full   empty until she promotes experiments (ledger promote ...);
                 a promoted unit runs with KIRBY_NOISE_PROMOTE=<exp>
A UNIT is one experiment of a stage. Noise units are (experiment, condition)
pairs, built the way stage1_prelim_noise.build_tasks builds its tasks.

Everything here is read from the suites' own lists: SINGLE_DATASET_EXPERIMENTS
and _experiment_full_name of each *_helpers module, the nucleotide and
large-n skip sets in _shared, the OPIG table in submit_everything.sh. Nothing
is listed twice.
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CATEGORIES = ('complexity', 'federated', 'similarity', 'noise', 'noise_full')
# Where a category's experiments write: <run root>/<this>/<experiment>/<ds>.json
CATEGORY_DIR = {'complexity': 'complexity', 'federated': 'federated',
                'similarity': 'similarity', 'noise': 'noise',
                'noise_full': 'noise_full', 'opig': 'opig', 'cross': 'cross'}
HELPER_MODULES = {
    'complexity': 'experiments.complexity.complexity_helpers',
    'similarity': 'experiments.similarity.similarity_helpers',
    'noise': 'experiments.noise.noise_helpers',
    'federated': 'experiments.federated.federated_helpers',
}

# Not in the roster: chembl_qsar is reached through the OPIG qsar script only,
# davis is parked (same exclusion as submit_everything.sh and
# stage1_prelim_noise.py).
EXCLUDED_DATASETS = ('chembl_qsar', 'davis')
# noise_helpers.REP_LIST_FROM maps herg_ic50 to herg_ki's representations; it
# is in no DATASET_REPS loop, so it is a row of its own with one stage.
HERG_IC50_EXPERIMENTS = ('n17', 'n19', 'n22', 'n26', 'n29', 'n36')
FEDERATED_DATASETS = ('lhasa_herg', 'caco2', 'logd')   # submit_all_stages.sh FED_FULL
C14_DATASETS = ('logd', 'caco2', 'herg', 'flab_thermo_tresanco', 'tcga_luad_vs_lusc')

# Shortest expected time first, as submit_dataset.sh resolve_experiments ran
# them (sources in runbook/SUBMIT_TONIGHT.md). An experiment not listed runs
# after the listed ones, in suite order.
RUN_ORDER = {
    'complexity': ['c06', 'c19', 'c22', 'c07', 'c18', 'c01', 'c02', 'c08',
                   'c15', 'c04', 'c13', 'c11', 'c12', 'c16', 'c09', 'c24',
                   'c05', 'c23', 'c33', 'c17', 'c21'],
    'similarity': ['s04', 's03', 's01', 's12', 's06', 's13', 's02', 's10',
                   's05', 's08', 's11'],
    'federated':  ['fed02', 'fed03', 'fed04', 'fed05', 'fed01'],
}
# An experiment that reads another's output runs after it (submit_dataset.sh).
AFTER = {'c08': ['c02'], 'c09': ['c02', 'c06'], 'n06': ['n11', 'n13']}

# Memory classes. 'big' needs 128G (submit_all_stages.sh BIG_RE, the 128G
# rows of submit_all_datasets.sh); everything else runs in 64G.
BIG_RE = re.compile(r'^(nt_|gb_|tcga_|kiba$|bindingdb_kd$|qm9$)')
MEM_GB = {'small': 64, 'big': 128}

# Size order for the default priority: small first so stages close early,
# then by molecule count, the large ones last (run_precompute_dataset.sh's
# lists, 2026-09-25).
SMALL_FIRST = ('tcga_brca_pam50 flab_thermo_tresanco humanness_therapeutic herg_ki '
               'ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_titer ginkgo_pr_cho '
               'abcov_kd abagym_Ang2 abagym_VEGF abagym_EGFR ddg_combined '
               'cdr_flexibility').split()
LARGE_LAST = ('nt_H3K27me3 nt_H3K9me3 nt_H3K9ac nt_H3K4me3 gb_human_enhancers_cohn '
              'nt_promoter_tata qm9 herg logd flab_expression_koenig tz_her2_binary '
              'tz_her2_multiclass bindingdb_kd kiba').split()

SEARCH_UNIT = 'search_f{fold}'
SEARCH_UNIT_RE = re.compile(r'^search_f(\d+)$')


# ── Dataset facts ──────────────────────────────────────────────────────────

def dataset_roster():
    from experiments.complexity.complexity_helpers import DATASET_REPS
    return [d for d in sorted(DATASET_REPS) if d not in EXCLUDED_DATASETS]


def dataset_info(ds):
    """(domain, task_type, n_samples or None) from the registry."""
    from kirby.registry import lookup
    info = lookup(ds)
    if info is None:
        return 'molecular', 'regression', None
    return (getattr(info, 'domain', 'molecular') or 'molecular',
            getattr(info, 'task_type', 'regression') or 'regression',
            getattr(info, 'n_samples', None))


def mem_class(row_name):
    ds = row_name.split(':', 1)[-1]
    if row_name.startswith(('opig:', 'cross:')):
        return 'big' if row_name in ('opig:graphinity', 'opig:ginkgo',
                                     'cross:c03') else 'small'
    return 'big' if BIG_RE.match(ds) else 'small'


def default_priority(ds):
    """Lower runs first."""
    if ds in SMALL_FIRST:
        return 10 + SMALL_FIRST.index(ds)
    if ds in LARGE_LAST:
        return 300 + LARGE_LAST.index(ds)
    _, _, n = dataset_info(ds)
    return 100 + (int(n) // 100 if n else 50)


def n_outer_folds():
    from kirby.find_best_hybrid import DEFAULT_N_FOLDS
    return int(DEFAULT_N_FOLDS)


# ── Experiment lists per category ──────────────────────────────────────────

def helpers(category):
    import importlib
    return importlib.import_module(HELPER_MODULES[category])


def ordered(category, exps):
    """RUN_ORDER then AFTER applied, as submit_dataset.sh did it."""
    exps = list(exps)
    if category in RUN_ORDER:
        rank = {e: i for i, e in enumerate(RUN_ORDER[category])}
        exps = sorted(exps, key=lambda e: (rank.get(e, len(rank)), exps.index(e)))
    changed = True
    while changed:
        changed = False
        for e, deps in AFTER.items():
            if e not in exps:
                continue
            last = max((exps.index(d) for d in deps if d in exps), default=-1)
            if exps.index(e) < last:
                exps.remove(e)
                exps.insert(last, e)
                changed = True
    return exps


def category_experiments(category, ds):
    """The experiment ids run_dataset.py would run for this category on ds,
    minus the ones should_skip_experiment would mark skipped by policy."""
    from experiments._shared import (NUCLEOTIDE_SKIP_EXPERIMENTS,
                                     LARGE_N_OOM_SKIP_EXPERIMENTS,
                                     LARGE_N_SKIP_THRESHOLD)
    H = helpers(category if category != 'noise_full' else 'noise')
    exps = list(H.SINGLE_DATASET_EXPERIMENTS)
    if category == 'complexity' and ds.startswith('tcga_') and 'c12' not in exps:
        exps.insert(exps.index('c11') + 1 if 'c11' in exps else len(exps), 'c12')
    domain, _, n = dataset_info(ds)
    if domain == 'nucleotide':
        exps = [e for e in exps if e not in NUCLEOTIDE_SKIP_EXPERIMENTS]
    if n and int(n) > LARGE_N_SKIP_THRESHOLD:
        exps = [e for e in exps if e not in LARGE_N_OOM_SKIP_EXPERIMENTS]
    return ordered(category, exps)


def noise_units(ds, experiments=None):
    """[(unit_key, experiment, condition or None)] in run order: the task list
    stage1_prelim_noise.build_tasks builds, one unit per (experiment,
    condition), the no-condition remainder sweeps after, n06 last."""
    H = helpers('noise')
    _, task_type, _ = dataset_info(ds)
    conds = H.conditions_for_task({'task_type': task_type})
    takes = H.HONOURS_NOISE_CONDITION | H.SWEEPS_ALL_CONDITIONS
    remainder = ('n21', 'n23', 'n32', 'n40', 'n41')   # stage1_prelim_noise REMAINDER_SWEEP
    exps = category_experiments('noise', ds)
    if experiments is not None:
        exps = [e for e in exps if e in experiments]
    units = []
    for e in exps:
        if e == 'n06':
            continue
        if e in takes:
            lim = getattr(H, 'CONDITION_LIMITED_TO', {}).get(e)
            for c in conds:
                if lim and c not in lim:
                    continue
                units.append((f'{e}__{c}', e, c))
            if e in remainder:
                units.append((e, e, None))
        else:
            units.append((e, e, None))
    if 'n06' in exps:
        units.append(('n06', 'n06', None))
    return units


def unit_deps(category, unit_key):
    """Units of the same stage this one waits for."""
    exp = unit_key.split('__', 1)[0]
    if category == 'complexity' and not SEARCH_UNIT_RE.match(unit_key):
        deps = [SEARCH_UNIT.format(fold=i) for i in range(n_outer_folds())]
    else:
        deps = []
    for d in AFTER.get(exp, []):
        deps.append(d)
    return deps


def result_file(run_root, category, ds, unit_key):
    """Where run_dataset.py writes this unit's result, or None (search units)."""
    if SEARCH_UNIT_RE.match(unit_key):
        return None
    exp = unit_key.split('__', 1)[0]
    cond = unit_key.split('__', 1)[1] if '__' in unit_key else None
    H = helpers('noise' if category == 'noise_full' else category)
    full = H._experiment_full_name(exp)
    name = f'{ds}__{cond}.json' if cond else f'{ds}.json'
    return Path(run_root) / CATEGORY_DIR[category] / full / name


# ── OPIG and cross rows ────────────────────────────────────────────────────

def opig_table():
    """[(name, mem, cmd, [datasets])] from submit_everything.sh OPIG_JOBS,
    read the way runbook/opig_one.sh reads it."""
    se = REPO / 'tests' / 'slurm_scripts' / 'submit_everything.sh'
    rows, on = [], False
    for line in se.read_text().splitlines():
        if line.startswith("OPIG_JOBS='"):
            on = True
            continue
        if on and line.startswith("'"):
            break
        if on and line.strip():
            name, mem, cmd, ds = (line.split('|') + ['', '', '', ''])[:4]
            rows.append((name.strip(), mem.strip(), cmd.strip(), ds.split()))
    return rows


# ── Building the rows ──────────────────────────────────────────────────────

def _unit(status='pending'):
    return {'status': status, 'attempts': 0, 'seconds': None, 'job': None,
            'host': None, 'started': None, 'ended': None, 'log': None,
            'error': None, 'rc': None}


def _stage(name, unit_keys, after=None):
    unit_keys = list(unit_keys)
    return {'name': name, 'status': 'pending' if unit_keys else 'empty',
            'started': None, 'ended': None,
            'seconds': None, 'after': after or [],
            'units': {k: _unit() for k in unit_keys}}


def build_dataset_row(ds):
    domain, task_type, n = dataset_info(ds)
    nf = n_outer_folds()
    stages = []
    search = [SEARCH_UNIT.format(fold=i) for i in range(nf)]
    stages.append(_stage('complexity', search + category_experiments('complexity', ds)))
    if ds in FEDERATED_DATASETS:
        stages.append(_stage('federated', category_experiments('federated', ds)))
    if domain != 'nucleotide':
        stages.append(_stage('similarity', category_experiments('similarity', ds)))
    stages.append(_stage('noise', [k for k, _, _ in noise_units(ds)]))
    stages.append(_stage('noise_full', []))
    return {'kind': 'dataset', 'dataset': ds, 'domain': domain,
            'task_type': task_type, 'n_samples': n,
            'mem': mem_class(ds), 'priority': default_priority(ds),
            'claim': None, 'stages': stages}


def build_herg_ic50_row():
    ds = 'herg_ic50'
    units = [k for k, _, _ in noise_units(ds, experiments=HERG_IC50_EXPERIMENTS)]
    return {'kind': 'dataset', 'dataset': ds, 'domain': 'molecular',
            'task_type': 'regression', 'n_samples': 9802,
            'mem': 'small', 'priority': 250, 'claim': None,
            'stages': [_stage('noise', units), _stage('noise_full', [])]}


def build_opig_rows(roster):
    rows = {}
    for name, mem, cmd, dss in opig_table():
        after = [(d, 'complexity') for d in dss if d in roster]
        row = {'kind': 'opig', 'dataset': name, 'domain': None,
               'task_type': None, 'n_samples': None,
               'mem': 'big' if mem.upper().startswith('128') else 'small',
               'priority': 500, 'claim': None,
               'stages': [_stage('opig', [name], after=after)]}
        rows[f'opig:{name}'] = row
    return rows


def build_cross_rows(roster):
    sim_ds = [d for d in roster if dataset_info(d)[0] != 'nucleotide']
    rows = {
        'cross:s09': ('similarity', sim_ds),
        'cross:c03': ('complexity', list(roster)),
        'cross:c14': ('complexity', [d for d in C14_DATASETS if d in roster]),
    }
    out = {}
    for key, (stage, dss) in rows.items():
        name = key.split(':')[1]
        out[key] = {'kind': 'cross', 'dataset': name, 'domain': None,
                    'task_type': None, 'n_samples': None,
                    'mem': mem_class(key), 'priority': 600, 'claim': None,
                    'datasets': dss,
                    'stages': [_stage('cross', [name],
                                      after=[(d, stage) for d in dss])]}
    return out


def build_rows(datasets=None, with_opig=True, with_cross=True,
               with_herg_ic50=True):
    roster = dataset_roster()
    if datasets:
        unknown = sorted(set(datasets) - set(roster) - {'herg_ic50'})
        if unknown:
            raise SystemExit(f'not in the roster: {unknown}')
        roster = [d for d in roster if d in datasets]
    rows = {ds: build_dataset_row(ds) for ds in roster}
    if with_herg_ic50 and (not datasets or 'herg_ic50' in datasets):
        rows['herg_ic50'] = build_herg_ic50_row()
    if with_opig:
        rows.update(build_opig_rows(roster))
    if with_cross:
        rows.update(build_cross_rows(roster))
    return rows


JOB_NAME_RE = re.compile(r'^(?:x_(?:complexity|similarity|noise|federated)_(?P<x>.+)'
                         r'|opig_(?P<opig>.+)'
                         r'|pc_(?P<pc>.+?)(?:_x?\d+|_general.*)?)$')


def row_from_job_name(name, rows):
    """The ledger row a pre-ledger SLURM job name was meant for, or None.
    x_<category>_<dataset> and pc_<dataset>_x<N> name a dataset row;
    opig_<name> names opig:<name>."""
    m = JOB_NAME_RE.match(name or '')
    if not m:
        return None
    if m.group('opig'):
        key = f"opig:{m.group('opig')}"
        return key if key in rows else None
    ds = m.group('x') or m.group('pc')
    if ds in rows:
        return ds
    # pc_<dataset>_<N>: the greedy tail may have eaten part of a dataset name
    for cand in sorted(rows, key=len, reverse=True):
        if rows[cand]['kind'] == 'dataset' and (ds == cand or ds.startswith(cand + '_')):
            return cand
    return None
