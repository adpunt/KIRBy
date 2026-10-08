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
    noise_full   empty until she promotes (ledger promote EXP --datasets ..
                 --conditions ..); a promoted unit runs outer folds 1 and 2
                 (PROMOTED_FOLDS) at the screen's levels, so with the
                 screen's fold 0 it makes 3 folds (Adelaide 2026-10-08)
herg_ic50's row has no complexity experiments; its 'complexity' stage holds
only search_f0..search_f2, built on one row per molecule
(MOLECULE_MEAN_SEARCH_DATASETS), and its noise stages wait for them.
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
# Outer folds a promoted noise unit runs (Adelaide 2026-10-08): fold 0 is the
# screen's, in <run>/noise; these two go to <run>/noise_full, 3 folds in all.
PROMOTED_FOLDS = (1, 2)
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
# Cut 2026-10-08 (Adelaide): their per-fold searches had run 1-36 h without
# finishing. tcga_luad_vs_lusc for reasons of her own; the other eight on their
# single-representation scores (validity_summary.md, 23 July). Not in the
# roster, so a new ledger never plans them; `ledger.py cut` marks them in a
# live run.
CUT_DATASETS = ('tcga_luad_vs_lusc', 'nt_H3K27me3', 'nt_H3K9ac', 'nt_H3K9me3',
                'gb_human_enhancers_cohn', 'bindingdb_kd', 'kiba',
                'flab_expression_adams', 'flab_binding_adams',
                # Cut 2026-10-08 (Adelaide): 12 ligand groups of 480; on outer
                # folds 0 and 1 the inner split holds out one ligand alone and
                # every search configuration lost that fold (refused, 4 of 5).
                'suzuki',
                # Cut 2026-10-08 (Adelaide): lhasa_herg searches 27-30 h each
                # and only fed01-fed05 read it; bh_doyle_plates is a re-plated
                # subset of buchwald_hartwig; the two nucleotide sets fit few
                # noise or similarity methods; tz_her2_binary and _multiclass
                # hold out 13 and 14 rows in one fold.
                'lhasa_herg', 'bh_doyle_plates', 'nt_H3K4me3', 'nt_promoter_tata',
                'tz_her2_binary', 'tz_her2_multiclass')
# noise_helpers.REP_LIST_FROM maps herg_ic50 to herg_ki's representations; it
# is in no DATASET_REPS loop, so it is a row of its own with one stage.
HERG_IC50_EXPERIMENTS = ('n17', 'n19', 'n22', 'n26', 'n29', 'n36')
# Datasets whose search units search one row per molecule, labelled with its
# mean over all its measurements (noise_helpers.molecule_mean_data), filed
# under noise_helpers.MOLECULE_MEAN_SEARCH_SUFFIX (Adelaide 2026-10-07).
# n17, n19, n26 and n29 read those searches. n22 searches the 12,008
# measurement rows and n36 each seed's single-measurement labels; both still
# build their own searches.
MOLECULE_MEAN_SEARCH_DATASETS = ('herg_ic50',)
# Which outer folds of the 5-fold layout get a herg_ic50 search unit: 0, 1
# and 2. herg_ic50 is not in _shared.ALL_FOLDS_DATASETS, so n26 runs folds
# 0-2 (OUTER_FOLDS_RUN) and n17, n19 and n29 run fold 0 under the one-fold
# screen; nothing on herg_ic50 reads folds 3 or 4 until an experiment is
# promoted to every fold, and a promoted experiment builds folds 3 and 4
# itself under the same name, so later readers find them.
HERG_IC50_SEARCH_FOLDS = 3
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

# Memory classes. 'big' needs a 128G job (submit_all_stages.sh BIG_RE, the
# 128G rows of submit_all_datasets.sh); everything else runs in 64G. The
# numbers here are the LEAST a job must have to take the class: a big
# dataset is taken by a job of 100G or more, so a 120G job (12 cores at
# htc's 10 GB per core) qualifies and a 64G job never does.
# Measured 2026-10-07: the jobs running tcga_brca_pam50's c05 and c07 peaked
# at 1.2 GB (sstat MaxRSS 1187624K, 1243404K) and nt_H3K9ac's search_f0 at
# 3.4 GB 40 min in (3400132K), so tcga_, nt_, gb_ and qm9 left the class.
# kiba and bindingdb_kd, the two with the most rows, are unmeasured and stay;
# `ledger.py set-mem <row> small` moves a row in a live run.
BIG_RE = re.compile(r'^(kiba$|bindingdb_kd$)')
MEM_GB = {'small': 0, 'big': 100}

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
    return [d for d in sorted(DATASET_REPS)
            if d not in EXCLUDED_DATASETS and d not in CUT_DATASETS]


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


def noise_units(ds, experiments=None, promoted=False):
    """[(unit_key, experiment, condition or None)] in run order: the task list
    stage1_prelim_noise.build_tasks builds, one unit per (experiment,
    condition), the no-condition remainder sweeps after, n06 last.
    promoted=False is the screen's condition list (SCREEN_DROPPED_CONDITIONS
    left out); True is the full list, for the noise_full stage."""
    H = helpers('noise')
    _, task_type, _ = dataset_info(ds)
    conds = H.conditions_for_task({'task_type': task_type}, promoted=promoted)
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
    """herg_ic50: a 'complexity' stage holding only its search units
    (search_f0..search_f2 on the molecule rows, MOLECULE_MEAN_SEARCH_DATASETS),
    then noise and noise_full. Holding them in a stage named 'complexity'
    lets Ledger.searches_ready hold the noise stages until the searches are
    stored, as on every other dataset row; no complexity experiment runs on
    herg_ic50."""
    ds = 'herg_ic50'
    units = [k for k, _, _ in noise_units(ds, experiments=HERG_IC50_EXPERIMENTS)]
    search = [SEARCH_UNIT.format(fold=i) for i in range(HERG_IC50_SEARCH_FOLDS)]
    return {'kind': 'dataset', 'dataset': ds, 'domain': 'molecular',
            'task_type': 'regression', 'n_samples': 9802,
            'mem': 'small', 'priority': 250, 'claim': None,
            'stages': [_stage('complexity', search), _stage('noise', units),
                       _stage('noise_full', [])]}


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
