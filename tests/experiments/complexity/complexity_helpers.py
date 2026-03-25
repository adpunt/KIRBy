"""Shared configuration and runner for per-dataset complexity experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from the pipeline readiness tiers.
Cross-dataset experiments (C03, C14) stay in their own files and
consume the per-dataset JSON outputs.

Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search.
"""

import sys
import os
import time
from pathlib import Path

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

import hashlib
import pickle

import numpy as np

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy,
)


# ── Cached create_hybrid wrapper ───────────────────────────────────────────
# Experiments that call create_hybrid directly (C09, C10, C11, C15) should
# use this wrapper so repeated runs don't recompute identical hybrids.
# TODO: create_hybrid API is not finalised — allocation_method, importance_method,
#       and budget semantics may change after framework experiments complete.
#       When they do, delete the cache dir (~/.cache/kirby/create_hybrid_cache/).

def _create_hybrid_cache_key(reps, labels, allocation_method, importance_method, budget):
    """Hash inputs to create_hybrid for disk caching."""
    h = hashlib.sha256()
    for name in sorted(reps.keys()):
        h.update(name.encode())
        h.update(reps[name].tobytes()[:4096])  # first 4KB per rep for speed
        h.update(str(reps[name].shape).encode())
    h.update(labels.tobytes()[:4096])
    h.update(str(len(labels)).encode())
    h.update(f'{allocation_method}_{importance_method}_{budget}'.encode())
    return h.hexdigest()[:24]


def cached_create_hybrid(reps, labels, allocation_method='greedy',
                         importance_method='treeshap', budget=100,
                         cache_dir=None):
    """Disk-cached wrapper around create_hybrid.

    Returns (X_hybrid, feature_info) — same as create_hybrid.
    """
    from kirby.hybrid import create_hybrid

    if cache_dir is None:
        cache_dir = Path(os.environ.get(
            'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby'))
        ) / 'create_hybrid_cache'
    else:
        cache_dir = Path(cache_dir)

    key = _create_hybrid_cache_key(reps, labels, allocation_method,
                                   importance_method, budget)
    cache_path = cache_dir / f'{key}.pkl'

    if cache_path.exists():
        try:
            result = pickle.loads(cache_path.read_bytes())
            print(f"    [cached create_hybrid: alloc={allocation_method}, "
                  f"imp={importance_method}, b={budget}]")
            return result
        except Exception:
            pass  # corrupted, recompute

    X_hybrid, feature_info = create_hybrid(
        reps, labels,
        allocation_method=allocation_method,
        importance_method=importance_method,
        budget=budget,
    )

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps((X_hybrid, feature_info)))
    except Exception:
        pass  # non-fatal

    return X_hybrid, feature_info


# ── Preliminary experiment config ──────────────────────────────────────────
# Screening-level: 3 models, 1 fold, fast search. Results inform which
# experiments are worth expanding to full mode.

PRELIMINARY = {
    'n_splits': 1,
    'models': ['rf', 'lightgbm', 'mlp'],
    'budgets': [25, 50, 100, 200, 500],
    'allocations': ['performance_weighted', 'softmax_T5'],
    'importance_methods': ['treeshap', 'lightgbm_gain'],
    'search_speed': 'fast',
}


# ── Per-dataset rep lists (from PIPELINE_READINESS_TODO.md tiers) ──────────
# General tier = reps used as base reps in experiments.
# Top tier is a subset used when an experiment needs a "best rep" reference.
#
# Dead antibody reps (excluded everywhere): aido_protein, igbert, igbert_cls,
# mint, xtrimopglm — NaN or near-zero on all datasets.
#
# TODO: mutation_features, esm2_wt_delta, antiberty_llr are in MUTATION_REPS
#       (src/kirby/representations/mutation.py) but _shared.py load_data() only
#       checks ANTIBODY_REPS — these reps get silently skipped. The baseline
#       surveys (antibody_baseline_survey.py) have a two-pass loader that handles
#       MUTATION_REPS separately. Until load_data is fixed to also check
#       MUTATION_REPS, these reps are EXCLUDED from configs below.
#       Affected datasets: all antibody/DMS/ginkgo datasets that had
#       mutation_features in their pipeline readiness tiers.

# Common antibody general set (viable on most datasets, dead reps excluded).
# Per-dataset overrides narrow this when a dataset has few viable reps.
_AB_GENERAL = [
    'esm2', 'esmc', 'ablang2', 'antiberty', 'prot_t5', 'igt5',
    'onehot', 'blosum', 'aac',
    'esm2_cdr', 'antiberty_cdr', 'prot_t5_cdr',
    'esm2_midlayers', 'esm2_layer24', 'esm2_early', 'esm2_concat',
    'esm2_entropy_top20',
    'abroberta', 'abroberta_cdr',
    'esm3', 'saprot']

DATASET_REPS = {
    # ── Molecular (7 datasets) ──────────────────────────────────────────
    'herg': {
        'general': [
            'mhg_gnn', 'mordred', 'graph_kernel', 'avalon', 'grover', 'pdv',
            'mol2vec', 'ecfp4', 'molformer', 'chemberta', 'atom_pair',
            'chembert', 'smited', 'top_torsion', 'graphmvp'],
        'augmentation': ['maccs', 'pharmacophore3d', 'coulomb_matrix', 'usr', 'usrcat'],
        'top_tier': [
            'mhg_gnn', 'mordred', 'graph_kernel', 'avalon', 'grover', 'pdv',
            'mol2vec', 'ecfp4'],
    },
    'caco2': {
        'general': [
            'graph_kernel', 'mol2vec', 'mhg_gnn', 'ecfp4', 'pdv', 'grover',
            'avalon', 'mordred', 'top_torsion', 'smited', 'atom_pair', 'molformer'],
        'augmentation': ['maccs', 'pharmacophore3d'],
        'top_tier': [
            'graph_kernel', 'mol2vec', 'mhg_gnn', 'ecfp4', 'pdv', 'grover',
            'avalon', 'mordred'],
    },
    'logd': {
        'general': [
            'mhg_gnn', 'mordred', 'grover', 'mol2vec', 'pdv', 'ecfp4',
            'smited', 'molformer', 'avalon', 'atom_pair', 'chemberta',
            'top_torsion', 'chembert', 'graph_kernel'],
        'augmentation': [
            'selformer', 'rdkfp', 'graphmvp', 'maccs', 'pharmacophore3d',
            'molclr', 'usrcat'],
        'top_tier': [
            'mhg_gnn', 'mordred', 'grover', 'mol2vec', 'pdv', 'ecfp4',
            'smited', 'molformer'],
    },
    'qm9': {
        'general': [
            'mordred', 'pdv', 'grover', 'mhg_gnn', 'avalon', 'rdkfp',
            'atom_pair', 'smited', 'ecfp4', 'selformer', 'maccs', 'chemberta',
            'mol2vec', 'chembert', 'graphmvp', 'molformer'],
        'augmentation': ['top_torsion', 'pharmacophore3d', 'usrcat', 'coulomb_matrix'],
        'top_tier': ['mordred', 'pdv', 'grover', 'mhg_gnn', 'avalon'],
    },
    'adme_hlm': {
        'general': [
            'mordred', 'pdv', 'mhg_gnn', 'smited', 'molformer', 'grover',
            'mol2vec', 'chemberta', 'atom_pair', 'chembert', 'graph_kernel',
            'avalon', 'top_torsion', 'ecfp4', 'selformer', 'graphmvp'],
        'augmentation': ['pharmacophore3d', 'maccs', 'usrcat', 'coulomb_matrix', 'usr'],
        'top_tier': ['mordred', 'pdv', 'mhg_gnn', 'smited', 'molformer', 'grover'],
    },
    'adme_rlm': {
        'general': [
            'pdv', 'mhg_gnn', 'smited', 'mordred', 'molformer', 'grover',
            'mol2vec', 'chemberta', 'chembert', 'graph_kernel', 'atom_pair',
            'selformer', 'avalon', 'ecfp4', 'top_torsion', 'graphmvp'],
        'augmentation': ['pharmacophore3d', 'maccs', 'usrcat', 'coulomb_matrix', 'usr'],
        'top_tier': ['pdv', 'mhg_gnn', 'smited', 'mordred', 'molformer', 'grover'],
    },
    'adme_mdr1': {
        'general': [
            'mordred', 'pdv', 'mhg_gnn', 'avalon', 'grover', 'smited',
            'molformer', 'atom_pair', 'graph_kernel', 'maccs', 'chemberta',
            'ecfp4', 'chembert', 'selformer', 'top_torsion', 'rdkfp'],
        'augmentation': ['pharmacophore3d', 'maccs'],
        'top_tier': ['mordred', 'pdv', 'mhg_gnn', 'avalon', 'grover'],
    },

    # ── Antibody Inter-Family (11 datasets) ─────────────────────────────
    'flab_thermo_tresanco': {
        'general': [
            'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
            'prot_t5_cdr', 'esm2_early', 'esm2', 'ablang2', 'antiberty',
            'esm2_concat', 'esm2_midlayers', 'abroberta', 'abroberta_cdr',
            'antiberty_cdr', 'esm2_layer24', 'esm2_entropy_top20', 'aac'
        ],
        'augmentation': ['structural'],
        'top_tier': [
            'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
            'prot_t5_cdr', 'esm2_early'],
    },
    'flab_pk_jain_heprt': {
        'general': [
            'igt5', 'esm2_concat', 'ablang2', 'prot_t5',
            'prot_t5_cdr', 'esm2_early'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': [
            'igt5', 'esm2_concat', 'ablang2', 'prot_t5',
            'prot_t5_cdr'],
    },
    'flab_pk_jain_fcrnrt': {
        'general': [
            'ablang2', 'igt5', 'esm2_early', 'prot_t5', 'saprot', 'antiberty', 'esmc', 'esm2', 'aac',
            'abroberta'],
        'augmentation': ['aac'],
        'top_tier': ['ablang2', 'igt5'],
    },
    'flab_polyreact_jain_cic': {
        'general': [
            'prot_t5', 'prot_t5_cdr', 'esm2_early', 'ablang2', 'igt5',
            'esm2', 'esmc', 'antiberty', 'saprot', 'esm2_concat',
            'esm2_midlayers'
        ],
        'augmentation': [],
        'top_tier': ['prot_t5', 'prot_t5_cdr', 'esm2_early'],
    },
    'flab_aggregation_jain_hicrt': {
        'general': [
            'prot_t5', 'prot_t5_cdr', 'esm2_early', 'ablang2', 'igt5',
            'esm2', 'esmc', 'antiberty', 'saprot', 'esm2_concat',
            'esm2_midlayers', 'abroberta', 'esm2_layer24'
        ],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['prot_t5', 'prot_t5_cdr', 'esm2_early'],
    },
    'flab_thermo_jain': {
        'general': [
            'ablang2', 'abroberta', 'igt5', 'esm3', 'prot_t5', 'prot_t5_cdr',
            'esm2_early', 'esmc', 'esm2', 'antiberty', 'saprot',
            'esm2_concat', 'esm2_midlayers', 'esm2_layer24',
            'esm2_entropy_top20'
        ],
        'augmentation': ['aac'],
        'top_tier': ['ablang2', 'abroberta', 'igt5', 'esm3'],
    },
    'abcov_kd': {
        'general': list(_AB_GENERAL),
        'augmentation': [],
        'top_tier': [
            'esm2_entropy_top20', 'prot_t5', 'blosum', 'antiberty',
            'esm2_midlayers', 'abroberta', 'esm2', 'ablang2'],
    },
    'flab_aggregation_jain': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['esm2_concat', 'ablang2', 'igt5'],
    },
    'flab_immunogenicity_marks': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': [],
        'top_tier': [
            'antiberty_cdr', 'antiberty', 'esm2_entropy_top20', 'igt5',
            'prot_t5', 'prot_t5_cdr', 'abroberta_cdr', 'abroberta'],
    },
    'flab_polyreact_jain_bvp': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': [],
        'top_tier': ['ablang2', 'igt5', 'antiberty', 'antiberty_cdr'],
    },
    'ddg_combined': {
        'general': [ 'prot_t5', 'esm2_midlayers', 'esm2', 'esmc',
            'ablang2', 'igt5', 'esm2_early', 'esm2_concat', 'esm2_layer24',
            'antiberty', 'prot_t5_cdr', 'saprot', 'esm3', 'abroberta',
            'esm2_entropy_top20', 'blosum', 'aac'],
        'augmentation': ['aac'],
        'top_tier': [ 'prot_t5', 'esm2_midlayers'],
    },

    # ── AbAgym DMS (5 datasets) ─────────────────────────────────────────
    'abagym_Ang2': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': [],
        'top_tier': ['saprot', 'esm3', 'esm2_midlayers', 'esm2_layer24'],
    },
    'abagym_EGFR': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': [],
        'top_tier': [
            'antiberty', 'antiberty_cdr', 'esm2_layer24', 'esm2_early',
            'esm2_midlayers'],
    },
    'abagym_HER2': {
        'general': [ 'prot_t5_cdr', 'prot_t5', 'esm2_early',
            'esm2', 'esmc', 'ablang2', 'igt5', 'antiberty', 'esm2_midlayers',
            'esm2_layer24', 'saprot', 'aac'],
        'augmentation': [],
        'top_tier': [ 'prot_t5_cdr', 'prot_t5'],
    },
    'abagym_lysozyme': {
        'general': [ 'aac', 'blosum'],
        'augmentation': ['structural'],
        'top_tier': [],
    },
    'abagym_VEGF': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': [],
        'top_tier': [ 'aac', 'prot_t5_cdr', 'prot_t5'],
    },

    # ── Antibody CLF (2 datasets) ───────────────────────────────────────
    'tz_her2_binary': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': [],
        'top_tier': ['blosum', 'onehot'],
    },
    'humanness_therapeutic': {
        'general': [
            'antiberty', 'igt5', 'abroberta', 'ablang2', 'blosum',
            'prot_t5', 'esm2', 'esmc', 'onehot', 'aac'],
        'augmentation': [],
        'top_tier': ['antiberty', 'igt5', 'abroberta', 'ablang2', 'blosum'],
    },

    # ── Ginkgo Developability (6 datasets) ──────────────────────────────
    'ginkgo_acsins': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['igt5', 'ablang2'],
    },
    'ginkgo_colloidal': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['prot_t5', 'prot_t5_cdr'],
    },
    'ginkgo_heparin': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['esm2_concat', 'prot_t5', 'prot_t5_cdr', 'ablang2'],
    },
    'ginkgo_hic': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['esm2_concat', 'prot_t5', 'prot_t5_cdr'],
    },
    'ginkgo_pr_cho': {
        'general': _AB_GENERAL + ['paratope', 'structural'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['prot_t5', 'prot_t5_cdr'],
    },
    'ginkgo_titer': {
        'general': _AB_GENERAL + ['paratope', 'structural', 'saprot_cdr'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['saprot', 'saprot_cdr'],
    },

    # ── TCGA (2 datasets) ──────────────────────────────────────────────
    'tcga_luad_vs_lusc': {
        'general': [
            'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
            'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
            'methyl_pca50', 'cnv_pca50', 'mirna_top200', 'wsi_gigapath',
            'rppa', 'immune_xcell', 'mut_driver', 'stemness_mrnasi'],
        'augmentation': [],
        'top_tier': [
            'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
            'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
            'methyl_pca50', 'cnv_pca50'],
    },
    'tcga_brca_pam50': {
        'general': [
            'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
            'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
            'methyl_pca50', 'cnv_pca50', 'mirna_top200', 'wsi_gigapath',
            'rppa', 'immune_xcell', 'mut_driver', 'stemness_mrnasi'],
        'augmentation': [
            'hrd_score', 'mut_tmb', 'purity_consensus', 'gene_stats',
            'gene_chromosome_arm', 'immune_xcell'],
        'top_tier': [
            'gene_geneformer_v2', 'gene_geneformer', 'gene_bulkrnabert'],
    },
}


# ── Single-dataset experiment list ─────────────────────────────────────────
# In dependency order. C12 (TCGA-only) excluded from default.
# C03 and C14 are cross-dataset — run separately.

SINGLE_DATASET_EXPERIMENTS = [
    'c01', 'c02',           # standalone, C02 produces data for C08/C09
    'c04', 'c05', 'c06',   # standalone, C06 produces data for C09
    'c07',                  # standalone
    'c08',                  # reads C02
    'c09',                  # reads C02 + C06
    'c10', 'c11',           # standalone
    'c13',                  # standalone (forward compat)
    'c15',                  # standalone (ML theory)
    'c16',                  # feature selection as regularization
    'c17',                  # information-theoretic decomposition
    'c18',                  # diversity vs depth
    'c19',                  # importance-weighted training
]


def _import_experiment(exp_id: str):
    """Dynamically import a complexity experiment module by ID (e.g. 'c01')."""
    module_map = {
        'c01': 'c01_rep_characterization',
        'c02': 'c02_complexity_grid',
        'c04': 'c04_importance_as_metric',
        'c05': 'c05_learning_curves',
        'c06': 'c06_within_family_gradient',
        'c07': 'c07_data_complexity_partitions',
        'c08': 'c08_calibration',
        'c09': 'c09_formal_analysis',
        'c10': 'c10_noise_robustness',
        'c11': 'c11_ensemble_vs_hybrid',
        'c12': 'c12_modality_ablation',
        'c13': 'c13_forward_compatibility',
        'c15': 'c15_ml_theory_depth',
        'c16': 'c16_feature_selection_as_regularization',
        'c17': 'c17_information_theoretic_decomposition',
        'c18': 'c18_diversity_vs_depth',
        'c19': 'c19_importance_weighted_training',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.complexity.{mod_name}')


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None):
    """Load data once and run all single-dataset complexity experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: PRELIMINARY).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
    """
    if config is None:
        config = dict(PRELIMINARY)
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)
        # C12 (modality ablation) is TCGA-only — auto-include for TCGA datasets
        if dataset_name.startswith('tcga_') and 'c12' not in experiments:
            # Insert after c11
            idx = experiments.index('c11') + 1 if 'c11' in experiments else len(experiments)
            experiments.insert(idx, 'c12')

    # Resolve rep list
    if rep_names is None:
        ds_reps = DATASET_REPS.get(dataset_name)
        if ds_reps is None:
            raise ValueError(
                f"No rep list for '{dataset_name}'. Pass rep_names explicitly "
                f"or add it to DATASET_REPS in complexity_helpers.py."
            )
        rep_names = ds_reps['general']

    print(f"\n{'=' * 70}")
    print(f"COMPLEXITY EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}, speed={config['search_speed']}")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    # ── Load data ONCE ──
    t0 = time.time()
    print(f"\nLoading {dataset_name}...")
    data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Run experiments in order ──
    results = {}
    for exp_id in experiments:
        print(f"\n{'─' * 70}")
        t_exp = time.time()

        try:
            mod = _import_experiment(exp_id)
        except (ImportError, KeyError) as e:
            print(f"  SKIP {exp_id}: import failed ({e})")
            continue

        # Build kwargs for experiments that need upstream results dirs
        kwargs = {}
        if exp_id == 'c08':
            kwargs['c02_results_dir'] = output_dir
        elif exp_id == 'c09':
            kwargs['c02_results_dir'] = output_dir
            kwargs['c06_results_dir'] = output_dir

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name, **kwargs)
            results[exp_id] = exp_result

            # Save per-experiment JSON
            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(exp_result, output_dir, exp_full_name, dataset_name)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e)}

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} — {len(results)} experiments in {total:.0f}s")
    print(f"{'=' * 70}")

    return results


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        'c01': 'c01_rep_characterization',
        'c02': 'c02_complexity_matching_grid',
        'c04': 'c04_importance_as_metric',
        'c05': 'c05_learning_curves',
        'c06': 'c06_within_family_gradient',
        'c07': 'c07_data_complexity_partitions',
        'c08': 'c08_calibration',
        'c09': 'c09_formal_analysis',
        'c10': 'c10_noise_robustness',
        'c11': 'c11_ensemble_vs_hybrid',
        'c13': 'c13_forward_compatibility',
        'c15': 'c15_ml_theory_depth',
        'c16': 'c16_feature_selection_as_regularization',
        'c17': 'c17_information_theoretic_decomposition',
        'c18': 'c18_diversity_vs_depth',
        'c19': 'c19_importance_weighted_training',
    }
    return name_map.get(exp_id, exp_id)
