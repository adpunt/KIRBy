"""Canonical per-dataset metric / task / direction registry.

Single source of truth for:
- task_type ('classification' or 'regression')
- metric name (must match a key in find_best_hybrid.BUILTIN_METRICS)
- lower_is_better (derived from BUILTIN_METRICS; stored here for readability)
- domain ('molecular', 'antibody', 'dta', 'tcga', 'reaction', 'nucleotide')

Every consumer reads from here. There is no other metric-direction table in
the codebase. If you need to override per-experiment, pass `metric=` explicitly
to find_best_hybrid — but that will bypass the registry and force a mismatch
check on load.

Adding a new dataset:
1. Add an entry here with the four required fields.
2. Confirm the loader sets `data['task_type']` identically (load_data cross-
   checks this on every call and raises loudly on mismatch).
3. If the metric isn't in BUILTIN_METRICS, add it there first.
4. Run `verify_registry()` — it's called at module import and at the start of
   every precompute run, so broken entries fail immediately.
"""

from dataclasses import dataclass
from typing import Dict, Optional


# Valid values — any value outside these sets fails verify_registry().
_VALID_TASK_TYPES = ('classification', 'regression')
_VALID_DOMAINS = ('molecular', 'antibody', 'dta', 'tcga', 'reaction',
                  'nucleotide')


@dataclass(frozen=True)
class DatasetInfo:
    """Per-dataset metadata. Immutable on purpose — drift must be a code edit."""

    name: str
    task_type: str
    metric: str
    lower_is_better: bool
    domain: str
    n_samples: Optional[int] = None
    notes: str = ''


# =============================================================================
# REGISTRY
# =============================================================================
#
# Conventions used when populating:
# - Molecular regression → 'rmse' (lower_is_better=True).
# - Antibody regression → 'spearman' (lower_is_better=False).
#     Exception: graphinity_* uses 'pearson' to match the paper's reported
#     numbers (see opig_graphinity_ddg.py benchmark table).
# - DTA regression → 'ci' (concordance index, lower_is_better=False).
# - TCGA binary classification → 'auc'.
#     TCGA multi-class → 'f1_weighted'.
# - Reaction regression → 'rmse'.
# - Nucleotide classification → 'mcc' (Matthews correlation coefficient).
# - Antibody binary classification → 'auc'.
#
# When the experiment requires a different metric (e.g., MAE on chembl_qsar),
# the registry reflects that — the registry is descriptive of the dataset,
# not prescriptive of a family default.

DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    # ── Molecular regression ─────────────────────────────────────────────
    'herg_ki':      DatasetInfo('herg_ki',      'regression', 'rmse',     True,  'molecular', n_samples=1412, notes='ChEMBL Ki, pKi target. Not the TDC binary herg.'),
    'caco2':        DatasetInfo('caco2',        'regression', 'rmse',     True,  'molecular'),
    'logd':         DatasetInfo('logd',         'regression', 'rmse',     True,  'molecular'),
    'qm9':          DatasetInfo('qm9',          'regression', 'rmse',     True,  'molecular', notes='Subsampled to 10k per group decision.'),
    'adme_hlm':     DatasetInfo('adme_hlm',     'regression', 'rmse',     True,  'molecular', notes='Polaris ADME-Fang.'),
    'adme_rlm':     DatasetInfo('adme_rlm',     'regression', 'rmse',     True,  'molecular', notes='Polaris ADME-Fang.'),
    'adme_mdr1':    DatasetInfo('adme_mdr1',    'regression', 'rmse',     True,  'molecular', notes='Polaris ADME-Fang.'),

    # ── Molecular classification ─────────────────────────────────────────
    'herg':         DatasetInfo('herg',         'classification', 'auc',  False, 'molecular', notes='TDC/Lhasa binary hERG blocker, ~10K. Distinct from herg_ki. Used by federated experiments.'),

    # ── Antibody regression, inter-family ────────────────────────────────
    'flab_thermo_tresanco':         DatasetInfo('flab_thermo_tresanco',         'regression', 'spearman', False, 'antibody'),
    'flab_pk_jain_heprt':           DatasetInfo('flab_pk_jain_heprt',           'regression', 'spearman', False, 'antibody'),
    'flab_pk_jain_fcrnrt':          DatasetInfo('flab_pk_jain_fcrnrt',          'regression', 'spearman', False, 'antibody'),
    'flab_polyreact_jain_cic':      DatasetInfo('flab_polyreact_jain_cic',      'regression', 'spearman', False, 'antibody'),
    'flab_aggregation_jain_hicrt':  DatasetInfo('flab_aggregation_jain_hicrt',  'regression', 'spearman', False, 'antibody'),
    'flab_thermo_jain':             DatasetInfo('flab_thermo_jain',             'regression', 'spearman', False, 'antibody'),
    'abcov_kd':                     DatasetInfo('abcov_kd',                     'regression', 'spearman', False, 'antibody'),
    'flab_aggregation_jain':        DatasetInfo('flab_aggregation_jain',        'regression', 'spearman', False, 'antibody'),
    'flab_immunogenicity_marks':    DatasetInfo('flab_immunogenicity_marks',    'regression', 'spearman', False, 'antibody'),
    'flab_polyreact_jain_bvp':      DatasetInfo('flab_polyreact_jain_bvp',      'regression', 'spearman', False, 'antibody'),

    # ── Antibody regression, DMS / mutation-scan ─────────────────────────
    'ddg_combined':                 DatasetInfo('ddg_combined',                 'regression', 'spearman', False, 'antibody', notes='AB-Bind + SKEMPI combined DDG.'),
    'flab_binding_koenig':          DatasetInfo('flab_binding_koenig',          'regression', 'spearman', False, 'antibody'),
    'flab_expression_koenig':       DatasetInfo('flab_expression_koenig',       'regression', 'spearman', False, 'antibody'),
    'flab_binding_adams':           DatasetInfo('flab_binding_adams',           'regression', 'spearman', False, 'antibody'),
    'flab_expression_adams':        DatasetInfo('flab_expression_adams',        'regression', 'spearman', False, 'antibody'),
    'flab_binding_phillips_h3':     DatasetInfo('flab_binding_phillips_h3',     'regression', 'spearman', False, 'antibody'),
    'abagym_Ang2':                  DatasetInfo('abagym_Ang2',                  'regression', 'spearman', False, 'antibody', n_samples=397),
    'abagym_EGFR':                  DatasetInfo('abagym_EGFR',                  'regression', 'spearman', False, 'antibody', n_samples=659),
    'abagym_HER2':                  DatasetInfo('abagym_HER2',                  'regression', 'spearman', False, 'antibody', n_samples=146),
    'abagym_lysozyme':              DatasetInfo('abagym_lysozyme',              'regression', 'spearman', False, 'antibody', n_samples=627),
    'abagym_VEGF':                  DatasetInfo('abagym_VEGF',                  'regression', 'spearman', False, 'antibody', n_samples=418),

    # ── Antibody classification ──────────────────────────────────────────
    'tz_her2_binary':         DatasetInfo('tz_her2_binary',         'classification', 'auc',         False, 'antibody', n_samples=5000, notes='Trastuzumab-HER2 binary. Stratified subsample from 524K.'),
    'tz_her2_multiclass':     DatasetInfo('tz_her2_multiclass',     'classification', 'f1_weighted', False, 'antibody', n_samples=5000, notes='Trastuzumab-HER2 multi-class. Winner blosum+lightgbm=0.7083 F1w.'),
    'humanness_therapeutic':  DatasetInfo('humanness_therapeutic',  'classification', 'auc',         False, 'antibody', n_samples=564,  notes='IMGT/mAb-DB via BioPhi. Winner antiberty_cdr+rf=0.9623 AUROC.'),
    'humanness_species':      DatasetInfo('humanness_species',      'classification', 'auc',         False, 'antibody', notes='OAS FASTA (Zenodo 10562968). Human vs non-human. Survey pending.'),

    # ── Antibody regression, Ginkgo developability (B5) ──────────────────
    # 2026-04-17: aggregation, purity, tm2 dropped from registry (surveys
    # confirmed all three near-random — best ρ < 0.22). Graphinity_* also
    # dropped (best ρ ≤ 0.079, all R² negative across foldx/rosetta/aug).
    # Pre-computed 5-fold hierarchical cluster splits from GDPa1.
    'ginkgo_acsins':        DatasetInfo('ginkgo_acsins',      'regression', 'spearman', False, 'antibody', n_samples=242),
    'ginkgo_colloidal':     DatasetInfo('ginkgo_colloidal',   'regression', 'spearman', False, 'antibody', n_samples=242),
    'ginkgo_heparin':       DatasetInfo('ginkgo_heparin',     'regression', 'spearman', False, 'antibody', n_samples=94),
    'ginkgo_hic':           DatasetInfo('ginkgo_hic',         'regression', 'spearman', False, 'antibody', n_samples=242),
    'ginkgo_pr_cho':        DatasetInfo('ginkgo_pr_cho',      'regression', 'spearman', False, 'antibody', n_samples=197),
    'ginkgo_titer':         DatasetInfo('ginkgo_titer',       'regression', 'spearman', False, 'antibody', n_samples=239),

    # ── TCGA classification ──────────────────────────────────────────────
    'tcga_luad_vs_lusc':    DatasetInfo('tcga_luad_vs_lusc',  'classification', 'auc',         False, 'tcga', n_samples=1016, notes='LUAD vs LUSC binary.'),
    'tcga_brca_pam50':      DatasetInfo('tcga_brca_pam50',    'classification', 'f1_weighted', False, 'tcga', n_samples=900,  notes='BRCA PAM50 4-class.'),

    # ── Reaction regression ──────────────────────────────────────────────
    'buchwald_hartwig':     DatasetInfo('buchwald_hartwig',   'regression', 'rmse', True, 'reaction'),
    'bh_doyle_plates':      DatasetInfo('bh_doyle_plates',    'regression', 'rmse', True, 'reaction', notes='Doyle HTE plates.'),

    # ── Nucleotide classification ────────────────────────────────────────
    # NT Revised + Genomic Benchmarks. Gena_lm corrected 2026-04-17 and
    # dropped universally as near-random at 1024D. Dnabert2 dropped
    # per-dataset where it underperforms (most; retained on saturated
    # tasks). Surveys DONE for the 15 below. Remaining 4 NT + humanness
    # species pending on server (gena_refix jobs).
    #
    # nt_splice_sites_all DROPPED — only nt_v2_* reps above MCC 0.28.
    'nt_H2AFZ':                     DatasetInfo('nt_H2AFZ',                    'classification', 'mcc', False, 'nucleotide', n_samples=33000, notes='Winner mistral_dna+ridge=0.4323.'),
    'nt_H3K27ac':                   DatasetInfo('nt_H3K27ac',                  'classification', 'mcc', False, 'nucleotide', n_samples=31616, notes='Winner kmer6+ridge=0.4564.'),
    'nt_H3K27me3':                  DatasetInfo('nt_H3K27me3',                 'classification', 'mcc', False, 'nucleotide', n_samples=33000, notes='Winner mistral_dna+ridge=0.5487.'),
    'nt_H3K36me3':                  DatasetInfo('nt_H3K36me3',                 'classification', 'mcc', False, 'nucleotide', n_samples=33000, notes='Winner mistral_dna+mlp=0.5310.'),
    'nt_H3K4me1':                   DatasetInfo('nt_H3K4me1',                  'classification', 'mcc', False, 'nucleotide', n_samples=33000, notes='Winner grover_dna+ridge=0.4207.'),
    'nt_H3K4me2':                   DatasetInfo('nt_H3K4me2',                  'classification', 'mcc', False, 'nucleotide', n_samples=32138, notes='Winner mistral_dna+ridge=0.5054.'),
    'nt_H3K4me3':                   DatasetInfo('nt_H3K4me3',                  'classification', 'mcc', False, 'nucleotide', n_samples=18244, notes='Winner grover_dna+ridge=0.6519. Saturated.'),
    'nt_H3K9ac':                    DatasetInfo('nt_H3K9ac',                   'classification', 'mcc', False, 'nucleotide', n_samples=24278, notes='Winner kmer6+ridge=0.5082. Saturated.'),
    'nt_H3K9me3':                   DatasetInfo('nt_H3K9me3',                  'classification', 'mcc', False, 'nucleotide', n_samples=28288, notes='Winner mistral_dna+ridge=0.3482 (hardest nt task).'),
    'nt_H4K20me1':                  DatasetInfo('nt_H4K20me1',                 'classification', 'mcc', False, 'nucleotide', n_samples=32270, notes='Winner hyenadna+mlp=0.6041.'),
    'nt_promoter_all':              DatasetInfo('nt_promoter_all',             'classification', 'mcc', False, 'nucleotide', n_samples=31584, notes='Winner mistral_dna+ridge=0.7281. Saturated.'),
    'nt_promoter_tata':             DatasetInfo('nt_promoter_tata',            'classification', 'mcc', False, 'nucleotide', n_samples=5274,  notes='Winner mistral_dna+ridge=0.6743. Smallest nt dataset.'),
    'nt_promoter_no_tata':          DatasetInfo('nt_promoter_no_tata',         'classification', 'mcc', False, 'nucleotide', n_samples=31372, notes='Winner mistral_dna+ridge=0.7497. Saturated.'),
    'gb_human_nontata_promoters':   DatasetInfo('gb_human_nontata_promoters',  'classification', 'mcc', False, 'nucleotide', n_samples=36131, notes='GB dataset, 5 groups. Winner kmer5+mlp=0.8917. Classicals dominate.'),
    'gb_human_enhancers_cohn':      DatasetInfo('gb_human_enhancers_cohn',     'classification', 'mcc', False, 'nucleotide', n_samples=27791, notes='GB dataset, 5 groups. Winner grover_dna+ridge=0.4853.'),

    # ── DTA regression ───────────────────────────────────────────────────
    'davis':            DatasetInfo('davis',        'regression', 'ci', False, 'dta'),
    'kiba':             DatasetInfo('kiba',         'regression', 'ci', False, 'dta'),
    'bindingdb_kd':     DatasetInfo('bindingdb_kd', 'regression', 'ci', False, 'dta', notes='TDC BindingDB_Kd, ~52K pairs.'),

    # ── OPIG meta-entries ────────────────────────────────────────────────
    # These cover experiment "families" that use the same underlying loader
    # with a per-variant suffix. Use condition_suffix= on find_best_hybrid
    # to keep cache entries distinct while sharing metric/task semantics.
    'chembl_qsar':      DatasetInfo('chembl_qsar',     'regression',     'mae', True, 'molecular', notes='Parent registry entry for chembl_qsar_{target}. Call find_best_hybrid(dataset_name="chembl_qsar", condition_suffix="{target}").'),
    'cdr_flexibility':  DatasetInfo('cdr_flexibility', 'classification', 'auc', False, 'antibody', notes='Parent registry entry for cdr_flexibility_{cdr_types}. Use condition_suffix={cdr_types}.'),
}


# =============================================================================
# LOOKUP + VERIFICATION
# =============================================================================

def lookup(dataset_name: str) -> Optional[DatasetInfo]:
    """Return the DatasetInfo for `dataset_name`, or None if not registered.

    Returning None (rather than raising) lets callers decide whether the miss
    is fatal. `find_best_hybrid(cache=True)` requires a registered dataset
    when `metric=` isn't passed explicitly.
    """
    return DATASET_REGISTRY.get(dataset_name)


def require(dataset_name: str) -> DatasetInfo:
    """Like lookup but raises if the dataset is not registered."""
    info = DATASET_REGISTRY.get(dataset_name)
    if info is None:
        raise KeyError(
            f"Dataset '{dataset_name}' is not in DATASET_REGISTRY. "
            f"Add an entry to src/kirby/registry.py or pass metric= "
            f"and task_type= explicitly to find_best_hybrid."
        )
    return info


def crosscheck_task_type(dataset_name: str, loader_task_type: str) -> None:
    """Raise if a loader's task_type disagrees with the registry.

    Use this at the top of any `load_dataset`-style wrapper that
    doesn't route through `_shared.load_data` (which already does the
    cross-check). Silent drift between loader and registry produces
    metric-direction bugs that are expensive to find — fail loud.

    Returns None on match (or unregistered dataset). Raises ValueError
    on mismatch.
    """
    info = DATASET_REGISTRY.get(dataset_name)
    if info is None:
        return  # unregistered is fine — caller is on their own
    if loader_task_type not in ('classification', 'regression'):
        raise ValueError(
            f"crosscheck_task_type({dataset_name!r}): loader returned "
            f"task_type={loader_task_type!r}. Must be 'classification' "
            f"or 'regression'."
        )
    if loader_task_type != info.task_type:
        raise ValueError(
            f"Registry/loader mismatch for {dataset_name!r}: "
            f"loader returned task_type={loader_task_type!r}, but "
            f"DATASET_REGISTRY says {info.task_type!r}. Fix one side — "
            f"they must agree."
        )


def verify_registry() -> None:
    """Assert every entry is internally consistent.

    Called once at module import. Also safe to call from tests /
    precompute scripts as a pre-flight check.

    Raises ValueError on the first offending entry.
    """
    # Import lazily to avoid a hard circular dependency with find_best_hybrid.
    from kirby.find_best_hybrid import BUILTIN_METRICS

    for name, info in DATASET_REGISTRY.items():
        if name != info.name:
            raise ValueError(
                f"Registry key '{name}' does not match info.name '{info.name}'"
            )
        if info.task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"Dataset '{name}': invalid task_type {info.task_type!r}. "
                f"Must be one of {_VALID_TASK_TYPES}."
            )
        if info.domain not in _VALID_DOMAINS:
            raise ValueError(
                f"Dataset '{name}': invalid domain {info.domain!r}. "
                f"Must be one of {_VALID_DOMAINS}."
            )
        if info.metric not in BUILTIN_METRICS:
            raise ValueError(
                f"Dataset '{name}': metric {info.metric!r} not in BUILTIN_METRICS. "
                f"Add it there before registering the dataset."
            )
        _, builtin_lib = BUILTIN_METRICS[info.metric]
        if info.lower_is_better != builtin_lib:
            raise ValueError(
                f"Dataset '{name}': lower_is_better={info.lower_is_better} "
                f"does not match BUILTIN_METRICS[{info.metric!r}]={builtin_lib}. "
                f"Direction must agree."
            )


def all_datasets() -> Dict[str, DatasetInfo]:
    """Return a copy of the registry. Useful for tests / diagnostics."""
    return dict(DATASET_REGISTRY)


def datasets_by_domain(domain: str) -> Dict[str, DatasetInfo]:
    """Return all registry entries in `domain`."""
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}. Expected {_VALID_DOMAINS}.")
    return {k: v for k, v in DATASET_REGISTRY.items() if v.domain == domain}


# Verify at import — if someone edits the registry into an inconsistent state,
# every entry point fails loudly on first import, not later at runtime.
verify_registry()
