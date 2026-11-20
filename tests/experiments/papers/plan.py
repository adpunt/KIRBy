"""What the Papers 2 and 3 run holds: its datasets, the steps of each, the
units of each step, what each unit waits for, and the command it runs.

Built 2026-11-03 on Adelaide's plan (HANDOFF_PAPERS_2_3_LEDGER.md (in git history), and her
answers of 2026-11-03). It runs on the experiments' ledger code
(tests/experiments/orchestrate/ledger.py) the way the screening ledger does:
same lock, claims, liveness, retries and heartbeats; only the plan differs.

One row per dataset. Its steps, in order:

    reps            build and check the dataset's representation files
                    (tests/precompute_reps.py DATASET)
    search_pruned   the full search space with calibrated pruning
    search_fast     the engine's fast tier
    search_full     the full search space with no pruning
    paper2          the Paper 2 run list's experiments that run on one dataset.
                    One of them, the meta-learner over models with their own
                    hybrids (meta_learner_own_hybrids.py), has one unit per outer
                    fold, and each is held until Adelaide names the dataset in
                    RUN_SETTINGS.json, chosen from results and runtime (2026-11-05)
    paper3          the Paper 3 run list's experiments that run on one dataset

Each search step has one unit per outer fold. A unit runs the three kinds of
search the experiments read back (_shared.fold_neighbour_space_recipes:
predicting the label, the distance-based evaluators, the five nearest
neighbours). Adelaide, 2026-11-04: all three searches run on most datasets,
then she picks the one the experiments use. So each has its own cache and its
own empty score store, and no search reuses another's scores: the hours the
ledger records per fold are that search's real time.

The GATE is the run's settings file, <run>/RUN_SETTINGS.json, which only
Adelaide edits. Its "open" block names which steps may start at all. A step
that is not open is held on every dataset and starts by itself at the next
claim after she sets it open. "experiments_search" names the search the
experiments read: paper2 and paper3 stay held until it is set, and then wait
for that search's folds on the dataset. "future_work" lists datasets she moved
out: every unit not yet ok is held (never deleted).

A dataset is HELD until its two models are recorded: it has a row in
tests/experiments/decided_grid.json, which build_decided_grid.py writes from
DECISIONS.md. The check is made at every claim, so rebuilding the grid file
releases the dataset with no ledger command. A NEW_DATASETS.md dataset joins
the run with `ledger.py extend` once its loader and DATASET_REPS entry exist.
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

from experiments.orchestrate import plan as OP   # noqa: E402

SEARCHES = ('pruned', 'fast', 'full')
SEARCH_STEPS = tuple(f'search_{t}' for t in SEARCHES)
# The meta-learner's units in the paper2 step: one per outer fold.
META_UNIT = 'meta_f{fold}'
META_UNIT_RE = re.compile(r'^meta_f(\d+)$')
META_DIR = 'meta_learner'
STEPS = ('reps',) + SEARCH_STEPS + ('paper2', 'paper3')
# Steps the settings file can hold. reps is never held.
GATED_STEPS = SEARCH_STEPS + ('paper2', 'paper3')
# Steps whose units each write one result file the ledger checks.
RESULT_STEPS = ('paper2', 'paper3')
SEARCH_UNIT = OP.SEARCH_UNIT
SEARCH_UNIT_RE = OP.SEARCH_UNIT_RE

# ── Datasets ───────────────────────────────────────────────────────────────
# From the outlines, read 2026-11-03: PAPER_2_OUTLINE.md "Datasets" (lines
# 85-131) and PAPER_3_OUTLINE.md "Datasets (Adelaide, 2026-11-01)" (56-66).
# Names are the loader's. Patient data is never listed.
PAPER3_DATASETS = (
    'qm9', 'qm9_mu', 'qm9_alpha', 'herg_ki', 'herg_ic50', 'caco2', 'logd',
    'adme_rlm', 'adme_hlm', 'adme_mdr1', 'herg',
    'chembl_qsar_d2', 'chembl_qsar_fxa', 'chembl_qsar_mpro',
    'flab_expression_koenig', 'flab_binding_koenig', 'ddg_combined',
    'abagym_EGFR', 'abagym_Ang2', 'ginkgo_titer', 'ginkgo_acsins',
    'ginkgo_colloidal', 'ginkgo_hic', 'cdr_flexibility_h3',
    'cdr_flexibility_l3', 'abcov_kd', 'tz_her2_binary', 'tz_her2_multiclass',
)
PAPER2_ONLY = (
    'humanness_therapeutic', 'flab_expression_adams', 'flab_binding_adams',
    'nt_H3K27me3', 'nt_H3K4me3', 'nt_H3K9ac', 'nt_H3K9me3',
    'nt_promoter_tata', 'gb_human_enhancers_cohn',
    'buchwald_hartwig', 'bh_doyle_plates',
)
PAPER2_DATASETS = PAPER3_DATASETS + PAPER2_ONLY

# ── Experiments per paper, from the run lists ─────────────────────────────
# PAPER_2_OUTLINE.md "Experiments to run" (line 588). Only those that run on
# one dataset at a time are units of a dataset row. c03 and the OPIG scripts
# read several datasets and are not planned yet.
PAPER2_EXPERIMENTS = {
    'complexity': ('c01', 'c02', 'c04', 'c06', 'c08', 'c09', 'c11', 'c16',
                   'c17', 'c18', 'c19', 'c21', 'c22', 'c23', 'c24', 'c33'),
    'similarity': ('s03',),
    # meta_learner_own_hybrids.py, one unit per outer fold.
    'meta_learner': None,
}
# PAPER_3_OUTLINE.md "Experiments to run" (line 311), plus the noise
# experiments of HANDOFF_NOISE_RESTRUCTURE.md (in git history): n43-n47, n15, n20, n36
# (noise_helpers.SINGLE_DATASET_EXPERIMENTS). s09 and the two OPIG scripts
# read several datasets and are not planned yet.
PAPER3_EXPERIMENTS = {
    'similarity': ('s01', 's02', 's05', 's08', 's10'),
    'noise': None,      # None: every experiment of the noise suite
    # The activity-cliff study (HANDOFF_AC_STUDY.md (in git history), her outline of
    # 2026-11-04). None: every unit of the category, in dependency order.
    'cliffs': None,
}
# The five noise experiments her Paper 3 noise grid covers. They run that
# grid, not the one-fold screen: all five folds (Adelaide, 2026-10-30), four
# levels on datasets of 5,000 molecules or fewer and two on larger ones
# ("LARGE DATASETS EXCLUDING QM9 RUN LESS"; QM9 loads 5,000 molecules, so the
# size rule keeps its four), and hetero_seen ("Keep hetero_seen") wherever the
# experiment runs it. PAPER_3_NOISE_EXPERIMENTS.md section 1. The worker
# promotes each of these units by its own name (KIRBY_NOISE_PROMOTE); n15, n20
# and n36 keep the screen, their grids being open.
PAPER3_FULL_GRID_NOISE = ('n43', 'n44', 'n45', 'n46', 'n47')
# The noise kinds of that grid: the decided regression kinds left after her
# 2026-11-04 cuts, plus hetero_seen. A kind an experiment runs by hand that is
# outside this list (n45's outlier_only_p05) gets no ledger unit.
PAPER3_KIND_ORDER = ('gaussian', 'outlier_p05', 'grouped_shifted', 'censoring',
                     'shift_only_p10', 'hetero_seen')
_NOISE_MODULES = {'n43': 'n43_noise_aware_training', 'n44': 'n44_detect_and_clean',
                  'n45': 'n45_detector_grading', 'n46': 'n46_structure_specific_fixes',
                  'n47': 'n47_noise_measurements'}


def paper3_noise_kinds(exp, task_type):
    """The kinds one of the five experiments runs as ledger units: its own
    list (what it runs by hand under KIRBY_NOISE_PROMOTE), kept to her grid.
    None for a classification dataset: its flip kinds are planned as before."""
    if task_type == 'classification':
        return None
    import importlib
    from experiments.noise import noise_helpers as nh
    mod = importlib.import_module(f'experiments.noise.{_NOISE_MODULES[exp]}')
    data = {'task_type': task_type}
    own = set(nh.without_paper3_cuts(nh.conditions_for_task(data, promoted=True)))
    own |= set(getattr(mod, 'MODEL_ERROR_CONDITIONS', ()))
    return [c for c in PAPER3_KIND_ORDER if c in own]


def _paper3_noise_units(ds, exps):
    """orchestrate.plan.noise_units, with each of the five experiments' kinds
    replaced by paper3_noise_kinds. A unit with no kind (n46's remainder) is
    kept where it was."""
    _, task_type, _ = dataset_info(ds)
    base = OP.noise_units(ds, experiments=exps)
    out, done = [], set()
    for k, e, c in base:
        if e not in PAPER3_FULL_GRID_NOISE or c is None:
            out.append(k)
            continue
        kinds = paper3_noise_kinds(e, task_type)
        if kinds is None:
            out.append(k)
            continue
        if e in done:
            continue
        done.add(e)
        out += [f'{e}__{kind}' for kind in kinds]
    return out


# Units in one step that read another unit's result (orchestrate.plan.AFTER).
AFTER = {'c08': ('c02',), 'c09': ('c02', 'c06')}
# Where a unit's result goes: <run>/<category>/<exp dir>/<ds>[__<cond>].json
CATEGORY_DIR = {'complexity': 'complexity', 'similarity': 'similarity',
                'noise': 'noise', 'cliffs': 'cliffs', 'meta_learner': META_DIR}


#: Which category each unit key belongs to. Read from the category's own list
#: rather than from the first letter of the name: the activity-cliff units are
#: named for what they compute, and `cliff_sets` begins with a c.
_BY_NAME = None


def experiment_category(exp):
    global _BY_NAME
    if META_UNIT_RE.match(exp):
        return 'meta_learner'
    if _BY_NAME is None:
        from experiments.cliffs.cliffs_helpers import (
            SINGLE_DATASET_EXPERIMENTS as CLIFF_UNITS)
        _BY_NAME = {name: 'cliffs' for name in CLIFF_UNITS}
    if exp in _BY_NAME:
        return _BY_NAME[exp]
    return {'c': 'complexity', 's': 'similarity', 'n': 'noise'}[exp[0]]


def dataset_info(ds):
    return OP.dataset_info(ds)


def _suite_filter(category, ds, exps):
    """The suite's own skips (nucleotide, large-n), as the old ledger applied
    them: orchestrate.plan.category_experiments, in the suite's order."""
    allowed = OP.category_experiments(category, ds)
    if exps is None:
        return allowed
    return [e for e in allowed if e in exps]


def paper_units(paper, ds):
    """[(unit_key, category)] for one paper on one dataset, in run order."""
    table = PAPER2_EXPERIMENTS if paper == 2 else PAPER3_EXPERIMENTS
    domain, _, _ = dataset_info(ds)
    out = []
    for category, exps in table.items():
        if category == 'meta_learner':
            out += [(META_UNIT.format(fold=i), category)
                    for i in range(int(OP.n_outer_folds()))]
            continue
        if category == 'similarity' and domain == 'nucleotide':
            continue
        if category == 'cliffs' and domain != 'molecular':
            # Adelaide, 2026-11-04 (PAPER_3_OUTLINE.md): "Scope of the activity
            # cliff work: small molecules only, while the paper as a whole
            # keeps both small molecules and antibodies." Until 2026-11-18 this
            # also gave cliff units to antibody datasets. Nucleotide and
            # reaction sets were never given them: no cliff criterion exists.
            continue
        if category == 'noise':
            if ds in OP.NOISE_SKIP_DATASETS:
                continue
            if paper == 3:
                out += [(k, 'noise') for k in _paper3_noise_units(ds, exps)]
            else:
                out += [(k, 'noise') for k, _, _ in OP.noise_units(ds, experiments=exps)]
            continue
        out += [(e, category) for e in _suite_filter(category, ds, exps)]
    return out


def unit_category(unit_key):
    return experiment_category(unit_key.split('__', 1)[0])


def build_row(ds, n_folds=None):
    n = int(n_folds or OP.n_outer_folds())
    domain, task_type, n_samples = dataset_info(ds)
    papers = [2] if ds in PAPER2_DATASETS else []
    if ds in PAPER3_DATASETS:
        papers.append(3)
    stages = [OP._stage('reps', ['reps'])]
    stages += [OP._stage(st, [SEARCH_UNIT.format(fold=i) for i in range(n)])
               for st in SEARCH_STEPS]
    stages.append(OP._stage('paper2', [k for k, _ in paper_units(2, ds)]
                            if 2 in papers else []))
    stages.append(OP._stage('paper3', [k for k, _ in paper_units(3, ds)]
                            if 3 in papers else []))
    return {'kind': 'dataset', 'dataset': ds, 'domain': domain,
            'task_type': task_type, 'n_samples': n_samples, 'papers': papers,
            'mem': OP.mem_class(ds), 'priority': OP.default_priority(ds),
            'claim': None, 'stages': stages}


def build_rows(datasets=None):
    names = list(datasets) if datasets else list(PAPER2_DATASETS)
    unknown = [d for d in names if d not in PAPER2_DATASETS]
    if unknown and not datasets:
        raise SystemExit(f'not in either paper: {unknown}')
    return {d: build_row(d) for d in names}


# ── Holds ──────────────────────────────────────────────────────────────────

_GRID_SEEN = {'mtime': None, 'names': frozenset()}


def models_recorded(ds):
    """True when decided_grid.json has a row for ds (its two models are
    recorded). Re-read whenever the file changes, so rebuilding it releases
    a held dataset at the next claim."""
    import json
    from experiments.decided_grid import GRID_FILE, ROW_ALIASES
    try:
        mt = os.path.getmtime(GRID_FILE)
        if mt != _GRID_SEEN['mtime']:
            _GRID_SEEN['names'] = frozenset(json.loads(Path(GRID_FILE).read_text())['datasets'])
            _GRID_SEEN['mtime'] = mt
    except Exception:
        return False
    return ROW_ALIASES.get(ds, ds) in _GRID_SEEN['names']


def unit_deps(stage_name, unit_key):
    return list(AFTER.get(unit_key.split('__', 1)[0], ()))


# ── Commands ───────────────────────────────────────────────────────────────

def result_file(run_root, stage_name, ds, unit_key):
    """The unit's final result file, or None (reps and search units write
    into the cache, not the run)."""
    if stage_name not in ('paper2', 'paper3'):
        return None
    if META_UNIT_RE.match(unit_key):
        return Path(run_root) / META_DIR / f'{ds}__{unit_key}.json'
    exp = unit_key.split('__', 1)[0]
    cond = unit_key.split('__', 1)[1] if '__' in unit_key else None
    name = f'{ds}__{cond}.json' if cond else f'{ds}.json'
    base = Path(run_root) / CATEGORY_DIR[unit_category(unit_key)]
    dirs = sorted(d for d in base.glob(f'{exp}_*') if d.is_dir()) if base.is_dir() else []
    for d in dirs:
        if (d / name).exists():
            return d / name
    return (dirs[0] if dirs else base / f'{exp}_?') / name
