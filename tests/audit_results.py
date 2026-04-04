#!/usr/bin/env python3
"""Audit all survey JSON results and report gaps.

Usage:
    python tests/audit_results.py                    # full report
    python tests/audit_results.py --domain molecular # one domain
    python tests/audit_results.py --summary          # compact summary only
"""
import json
import os
import sys
import argparse
from pathlib import Path

# ── Expected rep lists per domain ──────────────────────────────────────────

MOLECULAR_BASE_REPS = [
    'ecfp4', 'maccs', 'avalon', 'atom_pair', 'top_torsion', 'pdv', 'mordred',
    'coulomb_matrix', 'usr', 'usrcat', 'pharmacophore3d', 'tda', 'e3fp',
    'mol2vec', 'chemberta', 'molformer', 'selformer', 'chembert', 'molclr',
    'graphmvp', 'smited', 'mhg_gnn', 'grover', 'graph_kernel',
    'clamp', 'rmat', 'map4', 'unimol_v2', 'minimol', 'gp_molformer', 'mole',
    'rdkfp',
]
MOLECULAR_FACTORY_REPS = [
    'pdv_binary_mean', 'pdv_rank', 'sns',
    'chemberta_pca50', 'molformer_pca50', 'selformer_pca50',
    'molclr_pca50', 'smited_pca50', 'mhg_gnn_pca50', 'grover_pca50',
    'mordred_pca50',
]
MOLECULAR_DATASETS = [
    'logd', 'caco2', 'herg_ki', 'qm9',
    'adme_hlm', 'adme_rlm', 'adme_mdr1', 'adme_solubility',
]

REACTION_REPS = [
    'drfp_2048', 'product_ecfp4', 'reactant_ecfp4', 'product_maccs',
    'reactant_maccs', 'product_pdv', 'reactant_pdv', 'rxnfp', 'reactiont5',
    'onehot_components', 'diff_ecfp4', 'diff_maccs',
    'component_descriptors', 'ligand_descriptors',
]
REACTION_DATASETS = ['buchwald_hartwig', 'suzuki', 'bh_doyle_plates']

TCGA_REPS = [
    # Expression foundation models
    'gene_geneformer', 'gene_geneformer_v2', 'gene_bulkrnabert',
    'gene_scfoundation', 'gene_scgpt', 'gene_uce',
    # Expression-derived
    'gene_stats', 'gene_chromosome_arm',
    # Expression factory reps
    'gene_hvg2000', 'gene_pca100', 'gene_nmf50', 'gene_ica50',
    'gene_sparse_pca50', 'gene_autoencoder50', 'gene_quantile_rank',
    'gene_corr_modules', 'gene_immune',
    # Pathway factory reps
    'gene_hallmark', 'gene_kegg', 'gene_reactome', 'gene_oncogenic',
    'gene_progeny', 'gene_dorothea', 'gene_combined_pathway',
    # Multi-omic
    'cnv_gistic', 'cnv_pca50', 'methyl_hvg', 'methyl_pca50',
    'mirna_top200', 'mutation_matrix', 'rppa', 'clinical',
    'mut_tmb', 'mut_driver', 'mut_pathway', 'mut_pca50',
    # PanCanAtlas
    'immune_cibersort', 'immune_xcell', 'stemness_mrnasi',
    'purity_consensus', 'heterogeneity_math', 'msi_status',
    'hrd_score', 'cimp_status',
    # Mutational signatures
    'sig_sbs', 'sig_dbs', 'sig_id',
    # Tier 2 multi-omic
    'splicing_psi50', 'lncrna_pca50', 'gsva_hallmark', 'promoter_meth',
    # WSI
    'wsi_ctranspath', 'wsi_phikon', 'wsi_hoptimus0', 'wsi_hoptimus1',
    'wsi_gigapath', 'wsi_moco', 'wsi_titan', 'wsi_uni2h',
    'wsi_virchow2', 'wsi_conch',
]
TCGA_DATASETS = ['tcga_luad_vs_lusc', 'tcga_brca_pam50', 'tcga_brca_survival']

ANTIBODY_REPS = [
    'onehot', 'blosum', 'aac', 'paratope',
    'esm2', 'esmc', 'ablang2', 'igbert', 'igbert_cls', 'antiberty',
    'prot_t5', 'amplify', 'igt5', 'esm3', 'esm3_cdr', 'saprot', 'saprot_cdr',
    'esm2_cdr', 'esmc_cdr', 'antiberty_cdr', 'prot_t5_cdr', 'amplify_cdr',
    'esm2_midlayers', 'esm2_layer24', 'esm2_early', 'esm2_concat', 'esm2_cdr3',
    'abroberta', 'abroberta_cdr',
    'structural', 'mutation_features', 'esm2_mutsite', 'esm2_wt_delta',
    'esm2_llr', 'esmc_llr', 'antiberty_llr',
    'esm2_entropy_top20', 'esm2_per_cdr', 'esm2_per_region',
    'mint', 'aido_protein', 'xtrimopglm',
]
ANTIBODY_DMS_DATASETS = [
    'flab_binding_koenig', 'flab_expression_koenig',
    'flab_binding_phillips_h3', 'flab_binding_adams', 'flab_expression_adams',
    'abagym_HER2', 'abagym_VEGF', 'abagym_EGFR', 'abagym_lysozyme',
    'abagym_Ang2', 'abagym_all',
]
ANTIBODY_INTER_DATASETS = [
    'flab_thermo_tresanco', 'flab_pk_jain_heprt', 'flab_pk_jain_fcrnrt',
    'flab_polyreact_jain_cic', 'flab_aggregation_jain_hicrt',
    'flab_thermo_jain', 'abcov_kd', 'flab_aggregation_jain',
    'flab_immunogenicity_marks', 'flab_polyreact_jain_bvp',
    'flab_thermo_hutchinson', 'flab_aggregation_jain_sas',
    'ddg_combined', 'ddg_combined_full',
]
ANTIBODY_CLF_DATASETS = [
    'tz_her2_binary', 'tz_her2_multiclass',
    'humanness_species', 'humanness_therapeutic',
]
GINKGO_DATASETS = [
    'ginkgo_hic', 'ginkgo_tm2', 'ginkgo_titer', 'ginkgo_pr_cho',
    'ginkgo_acsins', 'ginkgo_purity', 'ginkgo_aggregation',
    'ginkgo_colloidal', 'ginkgo_heparin',
]
DDG_SYNTH_DATASETS = [
    'graphinity_foldx', 'graphinity_rosetta', 'graphinity_augmented',
]

DTA_DRUG_REPS = [
    'drug_ecfp4', 'drug_maccs', 'drug_pdv', 'drug_map4', 'drug_mol2vec',
    'drug_mordred', 'drug_avalon', 'drug_chemberta', 'drug_molformer',
    'drug_selformer', 'drug_clamp', 'drug_rmat', 'drug_unimol_v2',
    'drug_minimol', 'drug_gp_molformer', 'drug_mole',
]
DTA_PROTEIN_REPS = [
    'protein_aac', 'protein_descriptors', 'protein_kmer',
    'protein_esm2', 'protein_esmc', 'protein_esm2_midlayers',
    'protein_amplify', 'protein_mint', 'protein_esmfold',
    'protein_alphafold_db', 'protein_esm3', 'protein_saprot',
    'protein_aido', 'protein_xtrimopglm',
    'protein_contact_graph', 'protein_usr_shape', 'protein_pocket',
    'protein_esm2_pocket', 'protein_pocket_composition',
]
DTA_DATASETS = ['davis', 'kiba', 'bindingdb_kd']


def load_json_reps(filepath):
    """Load a JSON result file and return the set of rep names in baselines."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    baselines = data.get('baselines', {})
    if not baselines:
        # DTA older format or different structure
        for key in ['standalone', 'paired']:
            if key in data and isinstance(data[key], dict):
                baselines.update(data[key])
    return set(baselines.keys())


def find_json(base_dirs, dataset_name):
    """Search for a dataset JSON across multiple directories."""
    for d in base_dirs:
        candidate = os.path.join(d, f'{dataset_name}.json')
        if os.path.exists(candidate):
            return candidate
    return None


def audit_domain(domain_name, datasets, expected_reps, search_dirs, summary_only=False):
    """Audit a single domain and print gap report."""
    print(f"\n{'='*70}")
    print(f"  {domain_name}")
    print(f"{'='*70}")

    all_missing = {}  # rep -> list of datasets missing it
    dataset_results = {}

    for ds in datasets:
        path = find_json(search_dirs, ds)
        if path is None:
            dataset_results[ds] = None
            print(f"\n  {ds}: NO JSON FILE FOUND")
            for r in expected_reps:
                all_missing.setdefault(r, []).append(ds)
            continue

        found_reps = load_json_reps(path)
        if found_reps is None:
            dataset_results[ds] = None
            print(f"\n  {ds}: JSON PARSE ERROR")
            continue

        dataset_results[ds] = found_reps
        missing = sorted(set(expected_reps) - found_reps)
        present_count = len(set(expected_reps) & found_reps)
        total = len(expected_reps)

        if not summary_only:
            print(f"\n  {ds}: {present_count}/{total} reps")
            if missing:
                print(f"    Missing ({len(missing)}): {', '.join(missing)}")
            else:
                print(f"    COMPLETE")

        for r in missing:
            all_missing.setdefault(r, []).append(ds)

    # Summary: which reps are missing everywhere
    n_ds = len(datasets)
    print(f"\n  --- Gap Summary ({domain_name}) ---")
    universally_missing = [r for r, ds_list in all_missing.items()
                           if len(ds_list) == n_ds]
    partially_missing = {r: ds_list for r, ds_list in all_missing.items()
                         if 0 < len(ds_list) < n_ds}
    complete = [r for r in expected_reps if r not in all_missing]

    if universally_missing:
        print(f"  Missing from ALL {n_ds} datasets ({len(universally_missing)} reps):")
        for r in sorted(universally_missing):
            print(f"    - {r}")

    if partially_missing and not summary_only:
        print(f"  Partially missing ({len(partially_missing)} reps):")
        for r in sorted(partially_missing, key=lambda x: -len(partially_missing[x])):
            ds_list = partially_missing[r]
            print(f"    - {r}: missing from {len(ds_list)}/{n_ds} "
                  f"({', '.join(ds_list[:5])}{'...' if len(ds_list) > 5 else ''})")

    print(f"  Complete across all datasets: {len(complete)}/{len(expected_reps)} reps")

    # Datasets with no file
    missing_files = [ds for ds in datasets if dataset_results[ds] is None]
    if missing_files:
        print(f"  Datasets with NO results file: {', '.join(missing_files)}")

    return all_missing, dataset_results


def main():
    parser = argparse.ArgumentParser(description='Audit survey result gaps')
    parser.add_argument('--domain', type=str, default=None,
                        help='Audit one domain only')
    parser.add_argument('--summary', action='store_true',
                        help='Compact summary only')
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    results = root / 'tests' / 'results'

    p5 = results / 'phase5_baselines'
    opig = results / 'opig'

    domains = {}

    if args.domain in (None, 'molecular'):
        domains['Molecular'] = (
            MOLECULAR_DATASETS,
            MOLECULAR_BASE_REPS + MOLECULAR_FACTORY_REPS,
            [str(p5 / 'molecular')],
        )
    if args.domain in (None, 'reaction'):
        domains['Reaction'] = (
            REACTION_DATASETS,
            REACTION_REPS,
            [str(p5 / 'reaction')],
        )
    if args.domain in (None, 'tcga'):
        domains['TCGA'] = (
            TCGA_DATASETS,
            TCGA_REPS,
            [str(p5 / 'tcga')],
        )
    if args.domain in (None, 'antibody_dms'):
        domains['Antibody DMS'] = (
            ANTIBODY_DMS_DATASETS,
            ANTIBODY_REPS,
            [str(p5 / 'antibody')],
        )
    if args.domain in (None, 'antibody_inter'):
        domains['Antibody Inter-Family'] = (
            ANTIBODY_INTER_DATASETS,
            ANTIBODY_REPS,
            [str(p5 / 'antibody'), str(opig / 'ddg')],
        )
    if args.domain in (None, 'antibody_clf'):
        domains['Antibody CLF'] = (
            ANTIBODY_CLF_DATASETS,
            ANTIBODY_REPS,
            [str(opig / 'clf')],
        )
    if args.domain in (None, 'ginkgo'):
        domains['Ginkgo Developability'] = (
            GINKGO_DATASETS,
            ANTIBODY_REPS,
            [str(opig / 'ginkgo')],
        )
    if args.domain in (None, 'ddg_synth'):
        domains['Graphinity DDG'] = (
            DDG_SYNTH_DATASETS,
            ANTIBODY_REPS,
            [str(opig / 'ddg')],
        )
    if args.domain in (None, 'dta'):
        domains['DTA'] = (
            DTA_DATASETS,
            DTA_DRUG_REPS + DTA_PROTEIN_REPS,
            [str(p5 / 'dta'), str(results / 'dta_survey')],
        )

    print("KIRBy Survey Results Audit")
    print(f"Results directory: {results}")

    total_gaps = 0
    for name, (datasets, expected, dirs) in domains.items():
        missing, _ = audit_domain(name, datasets, expected, dirs, args.summary)
        total_gaps += sum(len(ds_list) for ds_list in missing.values())

    print(f"\n{'='*70}")
    print(f"  TOTAL GAPS: {total_gaps} (rep × dataset cells with no data)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
