"""Dataset-specific cost/skip policy and the dataset registry.

These live in the experiment layer, NOT in the published framework.
Importing this module registers the policy + registry with kirby via
injection hooks.
"""

from typing import List, Optional

from kirby.find_best_hybrid import (
    register_model_skip_policy,
    register_budget_policy,
)
from kirby.registry import DatasetInfo, register_datasets


# =============================================================================
# MODEL SKIP POLICY
# =============================================================================
#
# Datasets / conditions where a model is verified unfinishable under the
# SLURM wall. Filenames are unchanged; cache metadata reflects the reduced
# list, so old caches built with the skipped model present will be
# invalidated and rebuilt.

_DOMAIN_MODEL_SKIPS = {
    # Ridge at n=20k-33k x wide rep x 1080 configs runs 22-37h per condition
    # on nucleotide. Tree/MLP models cover the linear-baseline role adequately.
    # knn dropped: 13% of time per condition, weakest contributor to nucleotide
    # hybrids in prior surveys. xgboost dropped: near-redundant with lightgbm
    # (both gradient-boosted trees); keeping lightgbm as the faster of the two.
    # Combined cut ~38% wall reduction on n>20k nucleotide.
    'nucleotide': ['ridge', 'knn', 'xgboost'],
}

# Per (domain, strategy) extra skips, applied on top of _DOMAIN_MODEL_SKIPS.
# Use this when the same model-pruning rule applies to every dataset in a
# domain for a specific augmentation strategy (avoids enumerating 20+ rows
# in _DATASET_STRATEGY_SKIPS).
_DOMAIN_STRATEGY_SKIPS = {
    # Nucleotide capped at n=20k-100k blew the wall. Domain already skips
    # ridge/knn/xgboost. svm auto-skipped at n>10000. Dropping mlp leaves
    # [rf, lightgbm] -- both tree-based, hybrid stays methodologically
    # defensible while capped becomes feasible (~40% cut).
    ('nucleotide', 'capped'): ['mlp'],
    # uncapped mirrors capped: uncapped runs the A/B refit loop (>= as
    # expensive per row as capped) and currently left mlp in -- drop it so
    # nucleotide uncapped also leaves [rf, lightgbm].
    ('nucleotide', 'uncapped'): ['mlp'],
}

_DATASET_STRATEGY_SKIPS = {
    # Augmentation reps blow these up under the wall.
    ('flab_thermo_tresanco', 'uncapped'): ['gp', 'mlp'],
    ('tcga_luad_vs_lusc',    'uncapped'): ['gp'],
    ('ddg_combined',         'uncapped'): ['gp'],
    # tcga_brca_pam50 n=821 sits under the n>1000 GP cap, but GP still
    # dominated 72% of general/uncapped (44h of 60h). Skip across all
    # strategies to make capped fit the wall.
    ('tcga_brca_pam50',      'none'):     ['gp'],
    ('tcga_brca_pam50',      'uncapped'): ['gp'],
    ('tcga_brca_pam50',      'capped'):   ['gp'],

    # --- Capped restrictions for large-n datasets ---
    # Capped at n>=2k did not finish under the wall. Per-model timing on
    # adme_hlm general/capped (7 models): rf 30-50%, xgboost 16-25%,
    # lightgbm 17%, mlp 15%, svm 12%, knn 13%. Dropping mlp/svm/knn keeps
    # tree-based ensemble + ridge baseline. Both precompute and
    # experiment-time find_best_hybrid calls route through the skip policy
    # so cache keys match -- no on-the-fly recompute. Direct
    # create_hybrid(..., 'capped', ...) callers use a single model for
    # augmentation selection (default RF), so they are unaffected.

    # n=2k-5k molecular/reaction: skip [mlp, svm, knn]
    ('adme_hlm',          'capped'): ['mlp', 'svm', 'knn'],
    ('adme_mdr1',         'capped'): ['mlp', 'svm', 'knn'],
    ('adme_rlm',          'capped'): ['mlp', 'svm', 'knn'],
    ('caco2',             'capped'): ['mlp', 'svm', 'knn'],
    ('logd',              'capped'): ['mlp', 'svm', 'knn'],
    ('bh_doyle_plates',   'capped'): ['mlp', 'svm', 'knn'],

    # n=5k-10k molecular: also skip rf (slowest per config at this scale)
    ('herg',              'capped'): ['mlp', 'svm', 'knn', 'rf'],
    ('qm9',               'capped'): ['mlp', 'svm', 'knn', 'rf'],

    # DTA n>50k: svm auto-skipped at n>10000 by _max_samples.
    # Keep gradient boosters + ridge only.
    ('bindingdb_kd',      'capped'): ['mlp', 'rf', 'knn'],
    ('kiba',              'capped'): ['mlp', 'rf', 'knn'],
    ('davis',             'capped'): ['mlp', 'rf', 'knn'],  # n~29k, mirrors kiba

    # --- uncapped mirrors capped for the large datasets ---
    # uncapped is the A/B-refit loop (>= capped's per-row cost) but
    # previously dropped far fewer models than capped, so it stalled on the
    # same big sets. Mirror each dataset's capped row into uncapped. svm is
    # already auto-cut at n>5000, so it is not re-listed here. Scoped to the
    # genuinely large sets only -- the 2k-5k molecular/reaction sets finish
    # uncapped fine and are deliberately omitted.
    # Large molecular (n>5k): keep the existing [mlp,knn,rf] cut AND add ridge
    # (ridge is single-threaded and was 25-33% of uncapped — the thread-fix
    # does nothing for it). Leaves [xgboost, lightgbm] only, both multi-core.
    # (2026-06-21 cost-cut plan, "wall-safe" choice. logd added to match.)
    ('herg',              'uncapped'): ['mlp', 'knn', 'rf', 'ridge'],
    ('qm9',               'uncapped'): ['mlp', 'knn', 'rf', 'ridge'],
    ('logd',              'uncapped'): ['mlp', 'knn', 'rf', 'ridge'],
    # bindingdb_kd / kiba now skip uncapped ENTIRELY (UNCAPPED_SKIP_DATASETS in
    # precompute_hybrids.py); these rows are harmless leftovers. davis keeps
    # uncapped (just budget-capped) with the original [mlp,rf,knn] cut.
    ('bindingdb_kd',      'uncapped'): ['mlp', 'rf', 'knn'],
    ('kiba',              'uncapped'): ['mlp', 'rf', 'knn'],
    ('davis',             'uncapped'): ['mlp', 'rf', 'knn'],  # n~29k, mirrors kiba

    # Antibody-large (Group D): drop the single-threaded models (svm/gp/mlp/
    # ridge) from uncapped — the thread-fix can't help them and they dominate
    # the tail. Leaves [xgboost, lightgbm, rf, knn], all multi-core.
    # (2026-06-21 cost-cut plan, Lever 1.)
    ('flab_binding_koenig',     'uncapped'): ['svm', 'gp', 'mlp', 'ridge'],
    ('flab_expression_koenig',  'uncapped'): ['svm', 'gp', 'mlp', 'ridge'],
    ('flab_binding_adams',      'uncapped'): ['svm', 'gp', 'mlp', 'ridge'],
    ('flab_expression_adams',   'uncapped'): ['svm', 'gp', 'mlp', 'ridge'],
    ('flab_binding_phillips_h3','uncapped'): ['svm', 'gp', 'mlp', 'ridge'],
}


def _domain_for_skip(dataset_name: Optional[str]) -> Optional[str]:
    """Cheap inline copy of precompute_hybrids._get_domain for skip lookups."""
    if not dataset_name:
        return None
    if dataset_name.startswith(('nt_', 'gb_')):
        return 'nucleotide'
    return None  # other domains have no current skips


def resolve_model_skips(dataset_name: Optional[str],
                        strategy: Optional[str]) -> List[str]:
    """List of model names to skip for this (dataset, strategy) combo."""
    skip: set = set()
    domain = _domain_for_skip(dataset_name)
    if domain and domain in _DOMAIN_MODEL_SKIPS:
        skip.update(_DOMAIN_MODEL_SKIPS[domain])
    if domain and strategy:
        ds_key = (domain, strategy)
        if ds_key in _DOMAIN_STRATEGY_SKIPS:
            skip.update(_DOMAIN_STRATEGY_SKIPS[ds_key])
    if dataset_name and strategy:
        key = (dataset_name, strategy)
        if key in _DATASET_STRATEGY_SKIPS:
            skip.update(_DATASET_STRATEGY_SKIPS[key])
    return list(skip)


register_model_skip_policy(resolve_model_skips)


# =============================================================================
# BUDGET POLICY  (Lever 2 — cap the budget axis at <=500)
# =============================================================================
#
# On large datasets the 1000/2000 budgets dominated wall time without adding
# wins (see find_best_hybrid.py "budgets above ~250 contributed dominant wall
# time"). Cap them at <=500. Applied to ALL strategies/tiers so precompute and
# experiments request the same rows (find_best_hybrid routes both through this).
# NOTE: this truncates the budget axis for these datasets' experiment analyses
# at 500 — intended (2026-06-21 cost-cut plan).

_BUDGET_CAP_500 = {
    # molecular mid (Group B)
    'adme_hlm', 'adme_mdr1', 'adme_rlm', 'caco2', 'herg', 'herg_ki', 'logd', 'qm9',
    # antibody-large (Group D)
    'flab_binding_koenig', 'flab_expression_koenig', 'flab_binding_adams',
    'flab_expression_adams', 'flab_binding_phillips_h3',
    # DTA: kiba/bindingdb_kd skip uncapped entirely; davis keeps uncapped + caps.
    'davis',
}


def resolve_budget_cap(dataset_name, budgets):
    """Cap budgets to <=500 for the large datasets in _BUDGET_CAP_500 and the
    whole nucleotide domain (nt_*/gb_*, Group C). Other datasets unchanged."""
    if not dataset_name or not budgets:
        return budgets
    capped = (dataset_name in _BUDGET_CAP_500
              or dataset_name.startswith(('nt_', 'gb_')))
    if not capped:
        return budgets
    return [b for b in budgets if b <= 500]


register_budget_policy(resolve_budget_cap)


# =============================================================================
# DATASET REGISTRY
# =============================================================================
#
# Conventions used when populating:
# - Molecular regression -> 'rmse' (lower_is_better=True).
# - Antibody regression -> 'spearman' (lower_is_better=False).
#     Exception: graphinity_* uses 'pearson' to match the paper's reported
#     numbers (see opig_graphinity_ddg.py benchmark table).
# - DTA regression -> 'ci' (concordance index, lower_is_better=False).
# - TCGA binary classification -> 'auc'.
#     TCGA multi-class -> 'f1_weighted'.
# - Reaction regression -> 'rmse'.
# - Nucleotide classification -> 'mcc' (Matthews correlation coefficient).
# - Antibody binary classification -> 'auc'.
#
# When the experiment requires a different metric (e.g., MAE on chembl_qsar),
# the registry reflects that -- the registry is descriptive of the dataset,
# not prescriptive of a family default.

DATASET_REGISTRY_DATA = {
    # -- Molecular regression --
    'herg_ki':      DatasetInfo('herg_ki',      'regression', 'rmse',     True,  'molecular', n_samples=1412, notes='ChEMBL Ki, pKi target. Not the TDC binary herg.'),
    'caco2':        DatasetInfo('caco2',        'regression', 'rmse',     True,  'molecular'),
    'logd':         DatasetInfo('logd',         'regression', 'rmse',     True,  'molecular'),
    'qm9':          DatasetInfo('qm9',          'regression', 'rmse',     True,  'molecular', notes='Subsampled to 10k per group decision.'),
    'adme_hlm':     DatasetInfo('adme_hlm',     'regression', 'rmse',     True,  'molecular', notes='Polaris ADME-Fang.'),
    'adme_rlm':     DatasetInfo('adme_rlm',     'regression', 'rmse',     True,  'molecular', notes='Polaris ADME-Fang.'),
    'adme_mdr1':    DatasetInfo('adme_mdr1',    'regression', 'rmse',     True,  'molecular', notes='Polaris ADME-Fang.'),

    # -- Molecular classification --
    'herg':         DatasetInfo('herg',         'classification', 'auc',  False, 'molecular', notes='TDC/Lhasa binary hERG blocker, ~10K. Distinct from herg_ki. Used by federated experiments.'),

    # -- Antibody regression, inter-family --
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

    # -- Antibody regression, DMS / mutation-scan --
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

    # -- Antibody classification --
    'tz_her2_binary':         DatasetInfo('tz_her2_binary',         'classification', 'auc',         False, 'antibody', n_samples=5000, notes='Trastuzumab-HER2 binary. Stratified subsample from 524K.'),
    'tz_her2_multiclass':     DatasetInfo('tz_her2_multiclass',     'classification', 'f1_weighted', False, 'antibody', n_samples=5000, notes='Trastuzumab-HER2 multi-class.'),
    'humanness_therapeutic':  DatasetInfo('humanness_therapeutic',  'classification', 'auc',         False, 'antibody', n_samples=564,  notes='IMGT/mAb-DB via BioPhi.'),
    'humanness_species':      DatasetInfo('humanness_species',      'classification', 'auc',         False, 'antibody', notes='OAS FASTA (Zenodo 10562968). Human vs non-human.'),

    # -- Antibody regression, Ginkgo developability --
    # Pre-computed 5-fold hierarchical cluster splits from GDPa1.
    'ginkgo_acsins':        DatasetInfo('ginkgo_acsins',      'regression', 'spearman', False, 'antibody', n_samples=242),
    'ginkgo_colloidal':     DatasetInfo('ginkgo_colloidal',   'regression', 'spearman', False, 'antibody', n_samples=242),
    'ginkgo_heparin':       DatasetInfo('ginkgo_heparin',     'regression', 'spearman', False, 'antibody', n_samples=94),
    'ginkgo_hic':           DatasetInfo('ginkgo_hic',         'regression', 'spearman', False, 'antibody', n_samples=242),
    'ginkgo_pr_cho':        DatasetInfo('ginkgo_pr_cho',      'regression', 'spearman', False, 'antibody', n_samples=197),
    'ginkgo_titer':         DatasetInfo('ginkgo_titer',       'regression', 'spearman', False, 'antibody', n_samples=239),

    # -- TCGA classification --
    'tcga_luad_vs_lusc':    DatasetInfo('tcga_luad_vs_lusc',  'classification', 'auc',         False, 'tcga', n_samples=1016, notes='LUAD vs LUSC binary.'),
    'tcga_brca_pam50':      DatasetInfo('tcga_brca_pam50',    'classification', 'f1_weighted', False, 'tcga', n_samples=900,  notes='BRCA PAM50 4-class.'),

    # -- Reaction regression --
    'buchwald_hartwig':     DatasetInfo('buchwald_hartwig',   'regression', 'rmse', True, 'reaction'),
    'bh_doyle_plates':      DatasetInfo('bh_doyle_plates',    'regression', 'rmse', True, 'reaction', notes='Doyle HTE plates.'),

    # -- Nucleotide classification --
    # NT Revised + Genomic Benchmarks.
    'nt_H2AFZ':                     DatasetInfo('nt_H2AFZ',                    'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'nt_H3K27ac':                   DatasetInfo('nt_H3K27ac',                  'classification', 'mcc', False, 'nucleotide', n_samples=31616),
    'nt_H3K27me3':                  DatasetInfo('nt_H3K27me3',                 'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'nt_H3K36me3':                  DatasetInfo('nt_H3K36me3',                 'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'nt_H3K4me1':                   DatasetInfo('nt_H3K4me1',                  'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'nt_H3K4me2':                   DatasetInfo('nt_H3K4me2',                  'classification', 'mcc', False, 'nucleotide', n_samples=32138),
    'nt_H3K4me3':                   DatasetInfo('nt_H3K4me3',                  'classification', 'mcc', False, 'nucleotide', n_samples=18244, notes='Saturated.'),
    'nt_H3K9ac':                    DatasetInfo('nt_H3K9ac',                   'classification', 'mcc', False, 'nucleotide', n_samples=24278, notes='Saturated.'),
    'nt_H3K9me3':                   DatasetInfo('nt_H3K9me3',                  'classification', 'mcc', False, 'nucleotide', n_samples=28288, notes='Hardest nt task.'),
    'nt_H4K20me1':                  DatasetInfo('nt_H4K20me1',                 'classification', 'mcc', False, 'nucleotide', n_samples=32270),
    'nt_promoter_all':              DatasetInfo('nt_promoter_all',             'classification', 'mcc', False, 'nucleotide', n_samples=31584, notes='Saturated.'),
    'nt_promoter_tata':             DatasetInfo('nt_promoter_tata',            'classification', 'mcc', False, 'nucleotide', n_samples=5274,  notes='Smallest nt dataset.'),
    'nt_promoter_no_tata':          DatasetInfo('nt_promoter_no_tata',         'classification', 'mcc', False, 'nucleotide', n_samples=31372, notes='Saturated.'),
    'gb_human_nontata_promoters':   DatasetInfo('gb_human_nontata_promoters',  'classification', 'mcc', False, 'nucleotide', n_samples=36131, notes='GB dataset, 5 groups. Classicals dominate.'),
    'gb_human_enhancers_cohn':      DatasetInfo('gb_human_enhancers_cohn',     'classification', 'mcc', False, 'nucleotide', n_samples=27791, notes='GB dataset, 5 groups.'),
    'nt_enhancers':                 DatasetInfo('nt_enhancers',                'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'nt_enhancers_types':           DatasetInfo('nt_enhancers_types',          'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'nt_splice_sites_acceptors':    DatasetInfo('nt_splice_sites_acceptors',   'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'nt_splice_sites_donors':       DatasetInfo('nt_splice_sites_donors',      'classification', 'mcc', False, 'nucleotide', n_samples=33000),
    'gb_demo_coding_vs_intergenomic_seqs': DatasetInfo('gb_demo_coding_vs_intergenomic_seqs', 'classification', 'mcc', False, 'nucleotide', n_samples=100000, notes='GB demo, 5 groups. Saturated task (n=100K).'),
    'gb_demo_human_or_worm':        DatasetInfo('gb_demo_human_or_worm',       'classification', 'mcc', False, 'nucleotide', n_samples=100000, notes='GB demo, 5 groups. Classicals win this task.'),

    # -- DTA regression --
    'davis':            DatasetInfo('davis',        'regression', 'ci', False, 'dta'),
    'kiba':             DatasetInfo('kiba',         'regression', 'ci', False, 'dta'),
    'bindingdb_kd':     DatasetInfo('bindingdb_kd', 'regression', 'ci', False, 'dta', notes='TDC BindingDB_Kd, ~52K pairs.'),
    'metz':             DatasetInfo('metz',         'regression', 'ci', False, 'dta', notes='TDC Metz kinase Ki, ~35K pairs. TDC name "Metz".'),

    # -- OPIG meta-entries --
    # These cover experiment "families" that use the same underlying loader
    # with a per-variant suffix. Use condition_suffix= on find_best_hybrid
    # to keep cache entries distinct while sharing metric/task semantics.
    'chembl_qsar':      DatasetInfo('chembl_qsar',     'regression',     'mae', True, 'molecular', notes='Parent registry entry for chembl_qsar_{target}. Call find_best_hybrid(dataset_name="chembl_qsar", condition_suffix="{target}").'),
    'cdr_flexibility':  DatasetInfo('cdr_flexibility', 'classification', 'auc', False, 'antibody', notes='Parent registry entry for cdr_flexibility_{cdr_types}. Use condition_suffix={cdr_types}.'),
}


register_datasets(DATASET_REGISTRY_DATA)
