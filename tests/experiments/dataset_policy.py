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
    # Ridge is INTENTIONALLY retained for nucleotide (user decision 2026-07-18):
    # classical linear models are competitive with trees on nucleotide one-hot/
    # k-mer reps, and ridge is allowed ONLY here (removed everywhere else). It is
    # injected by _resolve_models for nt_*/gb_* datasets; do not re-add it to this
    # skip list. NOTE the cost: ridge at n=20k-33k x wide rep x 1080 configs runs
    # ~22-37h per condition on the largest nucleotide sets -- budget for it.
    # knn dropped: 13% of time per condition, weakest contributor to nucleotide
    # hybrids in prior surveys. xgboost dropped: near-redundant with lightgbm
    # (both gradient-boosted trees); keeping lightgbm as the faster of the two.
    'nucleotide': ['knn', 'xgboost'],
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
# BUDGET POLICY  (REMOVED 2026-09-10 — owner's decision)
# =============================================================================
#
# There is no budget cap. `_BUDGET_CAP_MAX = 1000` and `resolve_budget_cap`
# stood here until 2026-09-10 and dropped every budget above 1000 features on
# 35 of the 70 datasets: adme_hlm, adme_mdr1, adme_rlm, caco2, herg, herg_ki,
# logd, qm9, flab_binding_koenig, flab_expression_koenig, flab_binding_adams,
# flab_expression_adams, flab_binding_phillips_h3, davis, and every nt_*/gb_*.
# It was raised 500 -> 1000 earlier the same day and then removed outright.
#
# WHY IT WENT. It was a cost cut (2026-06-21 plan), justified by "budgets above
# ~250 contributed dominant wall time" — a statement about hours, not about the
# budget axis being degenerate, broken or undefined there. It also removed
# exactly the measurement c33 exists to make: on herg_ki it dropped 1412, 1695,
# 2000, 2260, 4000 and 8000, every budget above that dataset's training row
# count, and herg_ki is the one local dataset with the power to show a small
# drop above the line. logd, caco2, nt_promoter_tata and flab_expression_koenig
# lost their above-the-line budgets the same way.
#
# WHAT IT COSTS TO HAVE REMOVED IT. The budget list is part of the search cache
# key, so every search on those 35 datasets now asks for a key it has never
# written. Nothing cached is wrong or lost; none of it is reused either. Those
# searches re-run, and they re-run over a longer budget axis than before.
#
# The `register_budget_policy` mechanism in find_best_hybrid is untouched and
# still works. Nothing registers a policy now, so `_apply_budget_policy` is a
# pass-through. Registering one again re-creates this defect — c33 checks for
# it explicitly and prints what a policy dropped (`budget_policy_cut`).


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
    # adams-binding and phillips are CENSORED assays (65.6% clamped at the Tite-Seq
    # ceiling; 88.4% on the SPR floor), so they are scored as binder/non-binder.
    # The loader binarises them; these entries must agree or the metric dispatch
    # will ask for a Spearman on a 0/1 label.
    'flab_binding_adams':           DatasetInfo('flab_binding_adams',           'classification', 'auc', False, 'antibody'),
    'flab_expression_adams':        DatasetInfo('flab_expression_adams',        'regression', 'spearman', False, 'antibody'),
    'flab_binding_phillips_h3':     DatasetInfo('flab_binding_phillips_h3',     'classification', 'auc', False, 'antibody'),
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
    # The remaining three of the nine properties opig_ginkgo_developability
    # sweeps. They were absent, so _resolve_dataset_metric raised on them, the
    # `except Exception` around the search swallowed it, and the run finished
    # and wrote a results file with no hybrid rows for those three. Same source
    # file, same loader, same task and metric as the six above — see
    # GINKGO_PROPERTIES in src/kirby/datasets/ginkgo_dev.py. n_samples is left
    # unset rather than guessed: the six above carry counts somebody measured,
    # and it is only used for reporting and the model caps.
    'ginkgo_tm2':           DatasetInfo('ginkgo_tm2',         'regression', 'spearman', False, 'antibody'),
    'ginkgo_purity':        DatasetInfo('ginkgo_purity',      'regression', 'spearman', False, 'antibody'),
    'ginkgo_aggregation':   DatasetInfo('ginkgo_aggregation', 'regression', 'spearman', False, 'antibody'),

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


# =============================================================================
# NOISE LEVELS
# =============================================================================
#
# NOISE_DESIGN.md section 6.4. Four of these datasets are endpoints the literature
# gives a measured error for, so their levels are chosen in the label's own units and
# bracket one unit of that error, running to roughly twice it. Every other dataset has
# no published assay error to anchor to, so a fraction of the label SD is the only
# honest axis -- the same ruling QM9 gets, and for the same reason. Both quantities are
# recorded on every row either way (kirby.noise_spec.noise_provenance_row).
#
# Which axis a dataset uses is _NOISE_DOSE_UNITS; the grids are _NOISE_LEVELS.
#
# One unit of real error, each with its source:
#   herg_ki  0.54 pKi units -- Kramer et al. 2012, J Med Chem 55(11):5165-5173. A
#            ChEMBL-wide figure used as a labelled STAND-IN: hERG has no
#            target-specific published standard deviation. Bracketed by Sato et al.
#            2018 (cross-assay-format RMSD 0.737, n=209) and Alvarez Baron et al. 2025
#            (standardised-protocol residual 0.18).
#   caco2    ~0.35 log10 (~2.2-fold) -- COMPUTED from 11 laboratories' digoxin efflux
#            ratios, Bentz et al. 2013, Drug Metab Dispos 41(7):1347-1366, Table 6.
#            Corroborated by Larregieu & Benet 2013 (median 0.33) and O'Hagan & Kell
#            2015 ("within a factor of 2-5" = 0.30-0.70).
#   logd     0.48 log units -- repeat-test error after curation, Niu et al. 2024
#            (PharmaBench, Table 5). Open access and source-verified. Chosen over
#            Wenlock et al. 2011's 0.15, which is a WITHIN-one-laboratory figure and
#            does not describe an aggregated public set, and over Bruneau & McElroy
#            2006's 0.27, which failed source verification. Measured on KIRBy's own
#            logd (OpenADMET, n=5028): at 0.15 log units XGBoost/ECFP4 moves Spearman
#            by 0.0006 against a fold spread of +-0.020 -- a level that produces a row
#            saying nothing. At 0.48 it moves 0.0202.
#   qm9      NONE. A quantum mechanical calculation has no random experimental error,
#            so the noise there is a controlled perturbation and k is the only honest
#            axis (Kolmar & Grulke).
#
# Two numbers that must NOT appear here: 0.68 is pIC50, not pKi, and the two get
# swapped constantly; and Bruneau & McElroy's 0.27 for logD failed source verification.
#
# SETTLED 2026-08-27 by the author. NOISE_DESIGN.md section 4a left the logd anchor
# open between its anchor table (Wenlock 0.15 / OECD 0.5) and its verified-quote layer
# (Niu et al. 2024, 0.48). Niu was taken: it is open access, source-verified, covers
# the endpoint directly, and it is the only one of the three that lands where the
# experiment has signal.

# ONE grid, every dataset, in fractions of the clean label SD. Author's decision,
# 2026-08-27. auc_norm is a summary of the curve of score against noise level, so it is
# only comparable between two datasets when both were swept over the SAME level axis
# meaning the SAME thing. Per-dataset grids in log units gave neither.
#
# The consequence, stated rather than buried: the published assay errors below are no
# longer grid entries. They stay as the realism anchor for the Methods, and
# reference_noise_level() reports each one as a fraction of that dataset's spread so
# the figure can mark where real error falls on the shared axis.
_NOISE_DOSE_UNITS = {}
_DEFAULT_NOISE_DOSE_UNITS = 'spread'

_NOISE_LEVELS = {}

# Fraction of the clean label SD. This is NOISE_DESIGN.md section 6.4's logD row,
# adopted for every dataset.
_DEFAULT_NOISE_LEVELS = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)

# Censoring is swept on its own axis -- the fraction of labels clipped -- because it has
# no variance parameter and is not zero-mean. Same grid for every dataset. Set by the
# range-finding run: no sharp knee, damage becoming serious around 20% and severe past
# 30%. The 0 entry is the negative control for the one condition that biases labels.
CENSORING_LEVELS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50)


def noise_dose_units(dataset_name: Optional[str]) -> str:
    """'label' where the literature anchors the endpoint, 'spread' otherwise."""
    return _NOISE_DOSE_UNITS.get(dataset_name, _DEFAULT_NOISE_DOSE_UNITS)


def noise_levels(dataset_name: Optional[str], strategy: Optional[str] = None):
    """The noise levels this dataset is swept over, read through noise_dose_units().

    Pass strategy='censoring' for the censored-fraction axis instead.
    """
    if strategy == 'censoring':
        return CENSORING_LEVELS
    return _NOISE_LEVELS.get(dataset_name, _DEFAULT_NOISE_LEVELS)


def noise_specs_for(dataset_name: Optional[str], strategies=None, noise_seed: int = 0,
                    levels=None):
    """Every NoiseSpec for one dataset: the conditions crossed with their level grid.

    Censoring is crossed with the censored-fraction grid instead of the dose grid, so
    the two axes stay separate as NOISE_DESIGN.md section 6.4 requires.
    """
    from kirby.noise_spec import NoiseSpec, REGRESSION_STRATEGIES
    strategies = tuple(strategies) if strategies is not None else REGRESSION_STRATEGIES
    units = noise_dose_units(dataset_name)
    out = []
    for s in strategies:
        grid = levels if levels is not None else noise_levels(dataset_name, strategy=s)
        for lvl in grid:
            out.append(NoiseSpec(strategy=s, target_effective_noise=float(lvl),
                                 noise_seed=int(noise_seed),
                                 dose_units=units))
    return out


_NOISE_GROUP_CACHE = {}


def noise_groups_for(data):
    """Scaffold groups for grouped noise, from the SMILES this dataset loaded.

    Returns (groups, n_groups, largest_group_share), or (None, 0, 0.0) when the
    dataset has no SMILES -- an antibody, nucleotide or expression set. The two
    grouped conditions are then dropped rather than run over invented groups, which
    would be uniform noise wearing a grouped name.

    Cached per label column: building Murcko scaffolds for 30k molecules is not free
    and every experiment in a wave asks for the same assignment.
    """
    import numpy as np
    from kirby.noise_spec import assign_noise_groups

    smiles = data.get('smiles') if isinstance(data, dict) else None
    if smiles is None or len(smiles) == 0:
        return None, 0, 0.0
    labels = data.get('labels')
    key = (id(data), len(smiles), None if labels is None else float(np.sum(labels)))
    if key not in _NOISE_GROUP_CACHE:
        _NOISE_GROUP_CACHE[key] = assign_noise_groups(list(smiles))
    return _NOISE_GROUP_CACHE[key]


# One unit of real error, in the label's own units. Not a grid entry any more -- the
# grid is shared and in fractions of the SD -- but still the realism anchor, and the
# point a figure should mark on the shared axis.
REFERENCE_ERROR_LABEL_UNITS = {
    'herg_ki': 0.54,   # pKi, Kramer et al. 2012, a labelled stand-in
    'caco2':   0.35,   # log10 efflux ratio, computed from Bentz et al. 2013 Table 6
    'logd':    0.48,   # Niu et al. 2024, PharmaBench Table 5, post-curation
}


def reference_error_as_fraction_of_spread(dataset_name, labels) -> Optional[float]:
    """Where one unit of real error falls on the shared axis, for this dataset.

    Needs the labels because it divides by their spread. Returns None where no assay
    error has been published -- QM9, the reaction sets, the antibody sets -- for which
    there is nothing to anchor to and the level is a controlled perturbation.
    """
    import numpy as np
    tau = REFERENCE_ERROR_LABEL_UNITS.get(dataset_name)
    if tau is None:
        return None
    sd = float(np.std(np.asarray(labels, dtype=float).ravel()))
    return tau / sd if sd > 0 else None


def reference_noise_level(dataset_name: Optional[str]) -> float:
    """The single level to use where an experiment needs one rather than a sweep.

    0.5 of the label SD -- the middle of the shared grid. It is deliberately NOT the
    published assay error any more: that figure is in the label's own units and the
    grid is in fractions of the spread, so the two cannot be the same number.
    Use reference_error_as_fraction_of_spread() to place the real error on this axis.
    """
    return 0.5
