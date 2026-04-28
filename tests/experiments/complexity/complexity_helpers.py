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
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
)


# ── Cached create_hybrid wrapper ───────────────────────────────────────────
# Experiments that call create_hybrid directly (C09, C10, C11, C15) should
# use this wrapper so repeated runs don't recompute identical hybrids.
# TODO: create_hybrid API is not finalised — allocation_method, importance_method,
#       and budget semantics may change after framework experiments complete.
#       When they do, delete the cache dir (~/.cache/kirby/create_hybrid_cache/).

# cached_create_hybrid DELETED (2026-03-29) — operated on full data with
# random internal split, causing data leakage. Replaced by precompute_hybrids
# system that operates on proper per-fold train splits.


# ── Experiment config is derived from SPEED_TIERS (fast/full) ─────────────
# No standalone config dict here. run_all() builds config from SPEED_TIERS
# via get_config() or directly from the tier dict.


# ── Per-dataset rep lists (from PIPELINE_READINESS_TODO.md tiers) ──────────
# General tier = reps used as base reps in experiments.
# Top tier is a subset used when an experiment needs a "best rep" reference.
#
# Dead antibody reps (excluded everywhere): aido_protein, igbert, igbert_cls,
# mint, xtrimopglm — NaN or near-zero on all datasets.
#
# FIXED: _shared.py load_data() now handles MUTATION_REPS (two-pass loader).
# mutation_features added back to DMS datasets that have mutation metadata.

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
    # ── Molecular (8 datasets) — 7 regression + 1 classification ───────
    'herg': {  # FLuID classification (Lhasa). n=7,772; 3-fold/3-model screening.
        # Survey numbers from schnet_e2e/herg.json — pre-publication tier set.
        # Caveats:
        #   - graph_kernel kept on prior-evidence grounds (top-tier on
        #     herg_ki/caco2/etc); its herg score is from a stale-merge survey
        #     bug (regression metrics) so don't trust the exact value.
        #   - unimol_v2 collapsed to majority class (acc=0.5330, MCC=0) — drop.
        #   - lora_chemberta excluded per domain policy (chemberta FT excluded everywhere).
        #   - ft_schnet not competitive vs frozen on herg (tracker line 455).
        # Precompute + experiments must run at full 5-fold/5-model.
        'general': [
            'mhg_gnn', 'mordred', 'avalon', 'ecfp4', 'grover', 'atom_pair',
            'pdv', 'graph_kernel', 'mol2vec', 'top_torsion', 'maccs', 'smited',
            'molformer', 'chembert', 'rdkfp', 'selformer', 'chemberta', 'graphmvp'],
        'augmentation': ['usrcat', 'coulomb_matrix', 'molclr'],
        'top_tier': [
            'mhg_gnn', 'mordred', 'avalon', 'ecfp4', 'grover', 'atom_pair',
            'pdv', 'graph_kernel'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
        },
    },
    'herg_ki': {
        'general': [
            'mhg_gnn', 'mordred', 'graph_kernel', 'avalon', 'grover', 'pdv',
            'mol2vec', 'ecfp4', 'molformer', 'chemberta', 'atom_pair',
            'chembert', 'smited', 'top_torsion', 'graphmvp'],
        'augmentation': ['maccs', 'pharmacophore3d', 'coulomb_matrix', 'usr', 'usrcat'],
        'top_tier': [
            'mhg_gnn', 'mordred', 'graph_kernel', 'avalon', 'grover', 'pdv',
            'mol2vec', 'ecfp4'],
        'ft_reps': {
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'caco2': {
        'general': [
            'graph_kernel', 'mol2vec', 'mhg_gnn', 'ecfp4', 'pdv', 'grover',
            'avalon', 'mordred', 'top_torsion', 'smited', 'atom_pair', 'molformer'],
        'augmentation': ['maccs', 'pharmacophore3d', 'usrcat', 'coulomb_matrix', 'usr'],
        'top_tier': [
            'graph_kernel', 'mol2vec', 'mhg_gnn', 'ecfp4', 'pdv', 'grover',
            'avalon', 'mordred'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
        },
    },
    'logd': {
        'general': [
            'mhg_gnn', 'mordred', 'grover', 'mol2vec', 'pdv', 'ecfp4',
            'smited', 'molformer', 'avalon', 'atom_pair', 'chemberta',
            'top_torsion', 'chembert', 'graph_kernel', 'graphmvp'],
        'augmentation': [
            'selformer', 'rdkfp', 'maccs', 'pharmacophore3d',
            'molclr', 'usrcat'],
        'top_tier': [
            'mhg_gnn', 'mordred', 'grover', 'mol2vec', 'pdv', 'ecfp4',
            'smited', 'molformer'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
            'contrastive_molformer': {'factory': 'molecular_contrastive', 'base_model': 'molformer'},
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'qm9': {
        'general': [
            'mordred', 'pdv', 'grover', 'mhg_gnn', 'avalon', 'rdkfp',
            'atom_pair', 'smited', 'ecfp4', 'selformer', 'maccs', 'chemberta',
            'mol2vec', 'chembert', 'graphmvp', 'molformer'],
        'augmentation': ['top_torsion', 'pharmacophore3d', 'usrcat', 'coulomb_matrix'],
        'top_tier': ['mordred', 'pdv', 'grover', 'mhg_gnn', 'avalon'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
        },
    },
    'adme_hlm': {
        'general': [
            'mordred', 'pdv', 'mhg_gnn', 'smited', 'molformer', 'grover',
            'mol2vec', 'chemberta', 'atom_pair', 'chembert', 'graph_kernel',
            'avalon', 'top_torsion', 'ecfp4', 'selformer', 'graphmvp'],
        'augmentation': ['pharmacophore3d', 'maccs', 'usrcat', 'coulomb_matrix', 'usr'],
        'top_tier': ['mordred', 'pdv', 'mhg_gnn', 'smited', 'molformer', 'grover'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
            'contrastive_molformer': {'factory': 'molecular_contrastive', 'base_model': 'molformer'},
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'adme_rlm': {
        'general': [
            'pdv', 'mhg_gnn', 'smited', 'mordred', 'molformer', 'grover',
            'mol2vec', 'chemberta', 'chembert', 'graph_kernel', 'atom_pair',
            'selformer', 'avalon', 'ecfp4', 'top_torsion', 'graphmvp'],
        'augmentation': ['pharmacophore3d', 'maccs', 'usrcat', 'coulomb_matrix', 'usr'],
        'top_tier': ['pdv', 'mhg_gnn', 'smited', 'mordred', 'molformer', 'grover'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
            'contrastive_molformer': {'factory': 'molecular_contrastive', 'base_model': 'molformer'},
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'adme_mdr1': {
        'general': [
            'mordred', 'pdv', 'mhg_gnn', 'avalon', 'grover', 'smited',
            'molformer', 'atom_pair', 'graph_kernel', 'maccs', 'chemberta',
            'ecfp4', 'chembert', 'selformer', 'top_torsion', 'rdkfp'],
        'augmentation': ['pharmacophore3d', 'coulomb_matrix'],
        'top_tier': ['mordred', 'pdv', 'mhg_gnn', 'avalon', 'grover'],
        'ft_reps': {
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },

    # ── Antibody Inter-Family (11 datasets) ─────────────────────────────
    'flab_thermo_tresanco': {
        'general': [
            'esm3', 'esm3_cdr', 'saprot_cdr', 'saprot', 'blosum', 'esmc',
            'esmc_cdr', 'prot_t5', 'igt5', 'onehot', 'prot_t5_cdr',
            'esm2_early', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_wt_delta', 'esm2_per_cdr', 'esm2_entropy_top20', 'esm2',
            'esm2_per_region', 'esm2_midlayers',
            'antiberty', 'antiberty_cdr', 'aac', 'esm2_layer24',
            'esm2_mutsite', 'paratope', 'structural'],
        'augmentation': ['mutation_features'],
        'top_tier': [
            'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
            'prot_t5_cdr', 'esm2_early'],
    },
    'flab_pk_jain_heprt': {
        'general': [
            'igt5', 'mutation_features', 'esm2_concat', 'ablang2', 'prot_t5',
            'prot_t5_cdr', 'aac', 'esm2_cdr', 'esm2_cdr3', 'esm2_per_cdr',
            'esm2_per_region', 'esm2_mutsite', 'esm2', 'saprot', 'saprot_cdr',
            'esmc', 'esmc_cdr', 'esm2_wt_delta', 'esm2_early',
            'esm2_midlayers'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': [
            'igt5', 'mutation_features', 'esm2_concat', 'ablang2', 'prot_t5',
            'prot_t5_cdr'],
    },
    'flab_pk_jain_fcrnrt': {
        'general': [
            'ablang2', 'igt5', 'esm2_early', 'prot_t5', 'saprot', 'saprot_cdr',
            'mutation_features', 'prot_t5_cdr', 'esm2_concat', 'esm2_mutsite',
            'antiberty_cdr'],
        'augmentation': ['aac'],
        'top_tier': ['ablang2', 'igt5'],
    },
    'flab_polyreact_jain_cic': {
        'general': [
            'prot_t5', 'prot_t5_cdr', 'esm2_early', 'igt5', 'esm2_concat',
            'ablang2', 'saprot', 'saprot_cdr', 'esm2_midlayers', 'aac',
            'esm2_layer24', 'mutation_features', 'esm2_per_region',
            'esm2_per_cdr', 'esm2_cdr', 'esm2_cdr3', 'esm2',
            'esm2_wt_delta', 'esmc', 'esmc_cdr', 'esm2_mutsite'],
        'augmentation': [],
        'top_tier': ['prot_t5', 'prot_t5_cdr', 'esm2_early'],
    },
    'flab_aggregation_jain_hicrt': {
        'general': [
            'prot_t5', 'prot_t5_cdr', 'esm2_early', 'esm2_mutsite',
            'esm2_concat', 'esmc', 'esmc_cdr', 'esm3', 'esm3_cdr',
            'esm2_layer24', 'ablang2', 'saprot', 'saprot_cdr',
            'esm2_per_region', 'esm2_midlayers', 'blosum', 'esm2_wt_delta',
            'esm2_cdr', 'esm2_cdr3', 'aac', 'esm2', 'esm2_per_cdr',
            'onehot'],
        'augmentation': ['paratope', 'structural', 'mutation_features'],
        'top_tier': ['prot_t5', 'prot_t5_cdr', 'esm2_early'],
    },
    'flab_thermo_jain': {
        'general': [
            'ablang2', 'abroberta', 'abroberta_cdr', 'igt5', 'esm3',
            'esm3_cdr', 'esm2_cdr', 'esm2_cdr3', 'esm2_per_region',
            'esm2_wt_delta', 'esm2_per_cdr', 'antiberty', 'esm2', 'saprot',
            'saprot_cdr', 'esmc', 'esmc_cdr', 'esm2_entropy_top20',
            'esm2_layer24', 'antiberty_cdr', 'prot_t5', 'prot_t5_cdr'],
        'augmentation': ['aac'],
        'top_tier': ['ablang2', 'abroberta', 'igt5', 'esm3'],
    },
    'abcov_kd': {
        'general': [
            'igt5', 'esm2_per_cdr', 'prot_t5', 'esm2_per_region', 'antiberty',
            'esm2_midlayers', 'esm2_cdr3', 'esm2_cdr', 'abroberta',
            'esmc_cdr', 'esm2', 'blosum', 'ablang2', 'onehot',
            'abroberta_cdr', 'saprot', 'esm2_layer24', 'prot_t5_cdr',
            'saprot_cdr', 'esm3', 'esmc', 'antiberty_cdr', 'aac',
            'esm3_cdr', 'esm2_early', 'esm2_concat', 'esm2_entropy_top20'],
        'augmentation': [],
        'top_tier': [
            'esm2_entropy_top20', 'prot_t5', 'blosum', 'antiberty',
            'esm2_midlayers', 'abroberta', 'esm2', 'ablang2'],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },
    'flab_aggregation_jain': {
        'general': [
            'esm2_concat', 'ablang2', 'igt5', 'prot_t5_cdr', 'prot_t5',
            'saprot', 'saprot_cdr', 'esm2_early', 'mutation_features', 'esmc',
            'esmc_cdr', 'antiberty', 'esm2', 'esm2_cdr', 'esm2_cdr3',
            'antiberty_cdr', 'esm2_entropy_top20',
            'abroberta', 'abroberta_cdr', 'esm2_mutsite', 'esm3', 'esm3_cdr',
            'esm2_wt_delta'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['esm2_concat', 'ablang2', 'igt5'],
    },
    'flab_immunogenicity_marks': {
        'general': [
            'antiberty_cdr', 'antiberty', 'igt5', 'prot_t5', 'prot_t5_cdr',
            'abroberta_cdr', 'esm2_early', 'abroberta', 'ablang2', 'onehot',
            'saprot_cdr', 'saprot', 'esm2_midlayers', 'blosum', 'esm2',
            'esm2_cdr3', 'esm2_cdr', 'esm2_layer24', 'esmc_cdr', 'esmc',
            'aac', 'esm2_concat', 'esm3_cdr', 'esm3', 'esm2_per_cdr',
            'esm2_per_region', 'esm2_entropy_top20'],
        'augmentation': [],
        'top_tier': [
            'antiberty_cdr', 'antiberty', 'esm2_entropy_top20', 'igt5',
            'prot_t5', 'prot_t5_cdr', 'abroberta_cdr', 'abroberta'],
    },
    'flab_polyreact_jain_bvp': {
        'general': [
            'ablang2', 'igt5', 'antiberty', 'antiberty_cdr', 'esm2_concat',
            'esm2_mutsite', 'abroberta', 'abroberta_cdr', 'esm2_per_region',
            'prot_t5_cdr', 'prot_t5', 'esmc', 'esmc_cdr', 'esm2_cdr',
            'esm2_cdr3', 'esm2', 'blosum', 'esm2_per_cdr', 'onehot',
            'paratope', 'structural', 'esm2_early', 'esm2_wt_delta'],
        'augmentation': [],
        'top_tier': ['ablang2', 'igt5', 'antiberty', 'antiberty_cdr'],
    },
    'ddg_combined': {
        'general': [
            'esm2_wt_delta', 'prot_t5', 'esm2_midlayers', 'esm2_concat',
            'onehot', 'prot_t5_cdr', 'antiberty_cdr', 'esmc',
            'mutation_features', 'saprot', 'esmc_cdr', 'antiberty',
            'esm2_cdr', 'esm2_cdr3', 'esm2', 'esm2_layer24', 'saprot_cdr',
            'esm2_per_cdr'],
        'augmentation': ['aac'],
        'top_tier': ['esm2_wt_delta', 'prot_t5', 'esm2_midlayers'],
    },

    # ── Antibody DMS / Large DMS (5 datasets) ──────────────────────────
    'flab_binding_koenig': {
        'general': [r for r in _AB_GENERAL
                    if r not in ('aac', 'blosum', 'onehot', 'antiberty_cdr')] + [
            'esm2_cdr3', 'esm2_per_cdr', 'esm2_per_region', 'esm2_wt_delta',
            'saprot_cdr', 'esmc_cdr', 'esm3_cdr'],
        'augmentation': ['esm2_llr', 'mutation_features', 'antiberty_llr', 'aac'],
        'top_tier': [
            'ablang2', 'saprot', 'saprot_cdr', 'esm3', 'esm3_cdr',
            'prot_t5', 'prot_t5_cdr', 'igt5'],
    },
    'flab_expression_koenig': {
        'general': [r for r in _AB_GENERAL
                    if r not in ('aac', 'blosum', 'onehot')] + [
            'esm2_cdr3', 'esm2_per_cdr', 'esm2_per_region', 'esm2_wt_delta',
            'esm2_mutsite', 'saprot_cdr', 'esmc_cdr', 'esm3_cdr'],
        'augmentation': ['mutation_features', 'esm2_llr', 'aac'],
        'top_tier': [
            'esm3', 'esm3_cdr', 'prot_t5_cdr', 'prot_t5', 'saprot',
            'saprot_cdr', 'igt5'],
    },
    'flab_binding_adams': {
        'general': [r for r in _AB_GENERAL
                    if r not in ('aac', 'antiberty', 'antiberty_cdr')] + [
            'esm2_cdr3', 'esm2_per_cdr', 'esm2_per_region', 'esm2_wt_delta',
            'esm2_mutsite', 'saprot_cdr', 'esmc_cdr', 'esm3_cdr'],
        'augmentation': ['aac', 'paratope'],
        'top_tier': [
            'blosum', 'onehot', 'esm3', 'esm3_cdr', 'saprot',
            'saprot_cdr', 'esm2_cdr', 'esm2_cdr3'],
    },
    'flab_expression_adams': {
        'general': [r for r in _AB_GENERAL if r != 'aac'] + [
            'esm2_cdr3', 'esm2_per_cdr', 'esm2_per_region', 'esm2_wt_delta',
            'esm2_mutsite', 'saprot_cdr', 'esmc_cdr', 'esm3_cdr'],
        'augmentation': ['esm2_llr'],
        'top_tier': [
            'blosum', 'saprot', 'saprot_cdr', 'esm3', 'esm3_cdr',
            'prot_t5', 'prot_t5_cdr'],
    },
    'flab_binding_phillips_h3': {
        'general': [r for r in _AB_GENERAL if r != 'aac'] + [
            'esm2_cdr3', 'esm2_per_cdr', 'esm2_per_region', 'esm2_wt_delta',
            'esm2_mutsite', 'saprot_cdr', 'esmc_cdr', 'esm3_cdr'],
        'augmentation': ['mutation_features', 'paratope', 'aac'],
        'top_tier': ['onehot', 'blosum', 'igt5', 'esm2_early'],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },

    # ── AbAgym DMS (5 datasets) ─────────────────────────────────────────
    # Surveys: tests/results/antibody_ft_fill/abagym_*.json (32 reps).
    # DMS-special reps (mutation_features, structural) not in survey.
    # Dead reps excluded: aido_protein, igbert, igbert_cls, mint, xtrimopglm.
    'abagym_Ang2': {  # n=397, ρ > 0.28, General (24)
        'general': [
            'aac', 'abroberta', 'abroberta_cdr', 'antiberty', 'antiberty_cdr',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_early', 'esm2_layer24',
            'esm2_midlayers', 'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr', 'igt5',
            'prot_t5', 'prot_t5_cdr', 'saprot', 'saprot_cdr',
            'mutation_features', 'structural'],
        'augmentation': [],
        'top_tier': ['saprot', 'esm3', 'esm2_midlayers', 'esm2_layer24'],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },
    'abagym_EGFR': {  # n=659, ρ > 0.18, General (26)
        'general': [
            'aac', 'ablang2', 'abroberta', 'abroberta_cdr',
            'antiberty', 'antiberty_cdr',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_early', 'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr', 'igt5',
            'prot_t5', 'prot_t5_cdr', 'saprot', 'saprot_cdr',
            'mutation_features', 'structural'],
        'augmentation': [],
        'top_tier': [
            'antiberty', 'antiberty_cdr', 'esm2_layer24', 'esm2_early',
            'esm2_midlayers'],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },
    'abagym_HER2': {  # n=146, ρ > 0.24, General (13)
        'general': [
            'prot_t5_cdr', 'prot_t5', 'esm2_early',
            'esm2', 'esmc', 'ablang2', 'igt5', 'antiberty', 'esm2_midlayers',
            'esm2_layer24', 'saprot', 'aac', 'mutation_features'],
        'augmentation': [],
        'top_tier': ['mutation_features', 'prot_t5_cdr', 'prot_t5'],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },
    'abagym_lysozyme': {  # n=627, ρ > 0.15, General (3)
        'general': ['mutation_features', 'aac', 'blosum'],
        'augmentation': ['structural', 'antiberty_llr'],
        'top_tier': ['mutation_features'],
    },
    'abagym_VEGF': {  # n=418, ρ > 0.39, General (24)
        'general': [
            'aac', 'ablang2', 'abroberta', 'abroberta_cdr',
            'antiberty', 'antiberty_cdr',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_early',
            'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'prot_t5', 'prot_t5_cdr', 'saprot', 'saprot_cdr',
            'mutation_features', 'structural'],
        'augmentation': [],
        'top_tier': ['mutation_features', 'aac', 'prot_t5_cdr', 'prot_t5'],
    },

    # ── Antibody CLF (2 datasets) ───────────────────────────────────────
    # tz_her2: tests/results/opig/clf/tz_her2_binary.json (37 reps).
    # humanness: tests/results/antibody_ft_fill/humanness_therapeutic.json (31 reps).
    'tz_her2_binary': {  # n=5000, AUROC > 0.80, General (29)
        'general': [
            'aac', 'ablang2', 'abroberta', 'abroberta_cdr',
            'antiberty', 'blosum',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_early', 'esm2_layer24', 'esm2_midlayers',
            'esm2_mutsite', 'esm2_per_cdr', 'esm2_per_region',
            'esm2_wt_delta', 'esm2_entropy_top20',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'igt5', 'mutation_features', 'onehot',
            'prot_t5', 'prot_t5_cdr', 'saprot', 'saprot_cdr'],
        'augmentation': [],
        'top_tier': ['blosum', 'onehot'],
    },
    'tz_her2_multiclass': {  # n=5000, F1w > 0.65, General (16). FULL survey 2026-04-16.
        # Winner: blosum+lightgbm=0.7083 F1w. Augmentations: none help.
        # 4 reps broken on this survey (xtrimopglm, mint, igbert_cls, igbert).
        'general': [
            'blosum', 'onehot', 'esm2_concat',
            'esm2_per_region', 'esm2_per_cdr', 'esm2_wt_delta',
            'esm2', 'esm2_cdr3', 'esm2_cdr', 'esm2_midlayers',
            'igt5', 'esm2_early', 'esm2_layer24',
            'abroberta', 'abroberta_cdr', 'esm2_entropy_top20'],
        'augmentation': [],
        'top_tier': ['blosum', 'onehot', 'esm2_concat'],
    },
    'humanness_therapeutic': {  # n=564, AUROC > 0.92, General (26). FULL survey 2026-04-16.
        # Winner: antiberty_cdr+rf=0.9623 AUROC. Saturated task; many reps ≥0.90.
        # 5 reps broken (igbert_cls, igbert, xtrimopglm, mint, aido_protein).
        'general': [
            'antiberty_cdr', 'antiberty', 'igt5', 'abroberta_cdr', 'abroberta',
            'ablang2', 'blosum', 'onehot', 'esm2_per_cdr', 'esm2_cdr3',
            'esm2_cdr', 'esm2_concat', 'esm2', 'esm2_per_region',
            'esm2_midlayers', 'esmc_cdr', 'esmc', 'esm2_entropy_top20',
            'prot_t5', 'prot_t5_cdr', 'esm2_layer24',
            'esm3_cdr', 'esm3', 'saprot_cdr', 'saprot', 'esm2_early'],
        'augmentation': [],
        'top_tier': [
            'antiberty_cdr', 'antiberty', 'igt5', 'abroberta_cdr', 'abroberta',
            'ablang2', 'blosum', 'onehot', 'esm2_per_cdr', 'esm2_cdr3',
            'esm2_cdr', 'esm2_concat', 'esm2', 'esm2_per_region',
            'esm2_midlayers', 'esmc_cdr', 'esmc', 'esm2_entropy_top20',
            'prot_t5', 'prot_t5_cdr', 'esm2_layer24'],
    },

    # ── Ginkgo Developability (6 datasets) ──────────────────────────────
    # Surveys: tests/results/opig/ginkgo/ginkgo_*.json (34 reps each).
    # paratope/structural in augmentation for all; also in general when
    # they pass the dataset cutoff (heparin, hic).
    'ginkgo_acsins': {  # n=242, ρ > 0.24, General (26)
        'general': [
            'aac', 'ablang2', 'abroberta', 'abroberta_cdr',
            'antiberty', 'antiberty_cdr', 'blosum',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_early', 'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'igt5', 'prot_t5', 'prot_t5_cdr',
            'saprot', 'saprot_cdr', 'esm2_entropy_top20'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['igt5', 'ablang2'],
    },
    'ginkgo_colloidal': {  # n=242, ρ > 0.20, General (23)
        'general': [
            'ablang2', 'antiberty', 'antiberty_cdr', 'blosum',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_early', 'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'igt5', 'prot_t5', 'prot_t5_cdr',
            'saprot', 'saprot_cdr', 'esm2_entropy_top20'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['prot_t5', 'prot_t5_cdr'],
    },
    'ginkgo_heparin': {  # n=94, ρ > 0.36, General (29)
        'general': [
            'aac', 'ablang2', 'abroberta', 'abroberta_cdr',
            'antiberty', 'antiberty_cdr', 'blosum',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_early', 'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'igt5', 'onehot',
            'prot_t5', 'prot_t5_cdr',
            'saprot', 'saprot_cdr', 'esm2_entropy_top20'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['esm2_concat', 'prot_t5', 'prot_t5_cdr', 'ablang2'],
    },
    'ginkgo_hic': {  # n=242, ρ > 0.35, General (23)
        'general': [
            'aac', 'ablang2',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_early', 'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'igt5',
            'prot_t5', 'prot_t5_cdr',
            'saprot', 'saprot_cdr', 'esm2_entropy_top20'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['esm2_concat', 'prot_t5', 'prot_t5_cdr'],
    },
    'ginkgo_pr_cho': {  # n=197, ρ > 0.22, General (23)
        'general': [
            'aac', 'ablang2', 'blosum',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_early', 'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'igt5', 'onehot',
            'prot_t5', 'prot_t5_cdr',
            'saprot', 'saprot_cdr', 'esm2_entropy_top20'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['prot_t5', 'prot_t5_cdr'],
    },
    'ginkgo_titer': {  # n=239, ρ > 0.19, General (23)
        'general': [
            'ablang2', 'abroberta', 'abroberta_cdr',
            'antiberty', 'antiberty_cdr',
            'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat',
            'esm2_layer24', 'esm2_midlayers',
            'esm2_per_cdr', 'esm2_per_region',
            'esm3', 'esm3_cdr', 'esmc', 'esmc_cdr',
            'igt5', 'prot_t5', 'prot_t5_cdr',
            'saprot', 'saprot_cdr', 'esm2_entropy_top20'],
        'augmentation': ['paratope', 'structural'],
        'top_tier': ['saprot', 'saprot_cdr'],
    },

    # ── TCGA (2 datasets) ──────────────────────────────────────────────
    'tcga_luad_vs_lusc': {
        'general': [
            'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
            'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
            'methyl_pca50', 'cnv_pca50', 'mirna_top200', 'wsi_gigapath',
            'wsi_hoptimus0', 'cnv_gistic', 'wsi_ctranspath', 'wsi_moco',
            'wsi_phikon', 'rppa', 'mutation_matrix', 'mut_driver',
            'mut_pca50', 'mut_pathway'],
        'augmentation': [
            'immune_xcell', 'gene_stats', 'stemness_mrnasi',
            'immune_cibersort', 'cimp_status', 'mut_tmb',
            'hrd_score', 'purity_consensus'],
        'top_tier': [
            'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
            'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
            'methyl_pca50', 'cnv_pca50'],
    },
    'tcga_brca_pam50': {
        'general': [
            'gene_geneformer_v2', 'gene_geneformer', 'gene_bulkrnabert',
            'rppa', 'cnv_pca50', 'wsi_titan',
            'methyl_pca50', 'wsi_hoptimus0', 'wsi_gigapath', 'mirna_top200',
            'methyl_hvg', 'wsi_moco', 'wsi_phikon', 'cnv_gistic',
            'wsi_ctranspath', 'mutation_matrix', 'mut_pathway',
            'mut_driver', 'mut_pca50', 'stemness_mrnasi',
            'immune_cibersort'],
        'augmentation': [
            'hrd_score', 'mut_tmb', 'purity_consensus', 'gene_stats',
            'gene_chromosome_arm', 'immune_xcell'],
        'top_tier': [
            'gene_geneformer_v2', 'gene_geneformer', 'gene_bulkrnabert'],
    },

    # ── Reaction (2 datasets) ─────────────────────────────────────────
    'buchwald_hartwig': {
        'general': [
            'reactant_ecfp4', 'diff_ecfp4', 'drfp_2048', 'reactiont5',
            'onehot_components', 'reactant_pdv', 'diff_maccs', 'rxnfp'],
        'augmentation': [],
        'top_tier': ['reactant_ecfp4', 'diff_ecfp4', 'drfp_2048'],
    },
    'bh_doyle_plates': {
        'general': [
            'reactant_avalon', 'diff_avalon', 'diff_mhg_gnn', 'diff_ecfp4',
            'reactant_grover', 'reactant_ecfp4', 'reactant_mhg_gnn',
            'diff_grover', 'drfp_2048', 'reactiont5', 'diff_mol2vec',
            'onehot_components', 'reactant_mol2vec', 'rxnfp', 'diff_maccs',
            'reactant_mordred', 'diff_mordred'],
        'augmentation': [
            'product_maccs', 'reactant_maccs',
            'product_pharmacophore3d', 'reactant_pharmacophore3d'],
        'top_tier': [
            'reactant_avalon', 'diff_avalon', 'diff_mhg_gnn', 'diff_ecfp4',
            'reactant_grover', 'reactant_ecfp4'],
    },
    # --- Nucleotide (NT Revised + Genomic Benchmarks, chromosome/natural groups) ---
    # Tier tables below come from the 2026-04-17 gena_refix survey round.
    # Across all nt_* and gb_*, gena_lm was corrected to 1024D hidden
    # states (was 32000D LM-head logits) and reconfirmed near-random on
    # every dataset — excluded universally. dnabert2 dropped per-dataset
    # where it underperforms; kept only on saturated tasks where it sits
    # within 94-96% of the winner.
    'nt_H2AFZ': {  # 33,000 samples. Winner: mistral_dna+ridge=0.4323 MCC.
        'general': [
            'mistral_dna', 'hyenadna', 'grover_dna', 'kmer6',
            'nt_v2_250m', 'nt_v2_500m', 'tnf_canonical', 'kmer4', 'kmer5',
            'tetranuc_freq', 'kmer3'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'hyenadna', 'grover_dna', 'kmer6'],
    },
    'nt_H3K27ac': {  # 31,616 samples. Winner: kmer6+ridge=0.4564 MCC.
        'general': [
            'kmer6', 'grover_dna', 'mistral_dna', 'kmer5', 'tnf_canonical',
            'nt_v2_500m', 'nt_v2_250m', 'kmer4', 'hyenadna', 'tetranuc_freq'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['kmer6', 'grover_dna', 'mistral_dna', 'kmer5', 'tnf_canonical'],
    },
    'nt_H3K27me3': {  # 33,000 samples. Winner: mistral_dna+ridge=0.5487 MCC.
        'general': [
            'mistral_dna', 'grover_dna', 'hyenadna', 'nt_v2_250m', 'nt_v2_500m',
            'tnf_canonical', 'kmer4', 'kmer3', 'kmer5', 'tetranuc_freq', 'kmer6'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'grover_dna', 'hyenadna', 'nt_v2_250m', 'nt_v2_500m'],
    },
    'nt_H3K36me3': {  # 33,000 samples. Winner: mistral_dna+mlp=0.5310 MCC.
        'general': [
            'mistral_dna', 'hyenadna', 'nt_v2_250m', 'nt_v2_500m', 'grover_dna',
            'kmer3', 'kmer4', 'kmer5', 'kmer6', 'tetranuc_freq', 'tnf_canonical'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'hyenadna', 'nt_v2_250m', 'nt_v2_500m', 'grover_dna'],
    },
    'nt_H3K4me1': {  # 33,000 samples. Winner: grover_dna+ridge=0.4207 MCC.
        'general': [
            'grover_dna', 'mistral_dna', 'hyenadna', 'nt_v2_250m', 'kmer6',
            'kmer5', 'nt_v2_500m', 'tnf_canonical', 'kmer4', 'tetranuc_freq', 'kmer3'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['grover_dna', 'mistral_dna', 'hyenadna', 'nt_v2_250m', 'kmer6'],
    },
    'nt_H3K4me2': {  # 32,138 samples. Winner: mistral_dna+ridge=0.5054 MCC.
        'general': [
            'mistral_dna', 'hyenadna', 'grover_dna', 'kmer6', 'nt_v2_500m',
            'nt_v2_250m', 'kmer5', 'tnf_canonical', 'kmer4', 'tetranuc_freq', 'kmer3'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'hyenadna', 'grover_dna', 'kmer6', 'nt_v2_500m'],
    },
    'nt_H3K4me3': {  # 18,244 samples. Winner: grover_dna+ridge=0.6519 MCC. dnabert2 kept (saturated task).
        'general': [
            'grover_dna', 'mistral_dna', 'hyenadna', 'nt_v2_500m', 'tnf_canonical',
            'kmer6', 'nt_v2_250m', 'kmer5', 'kmer3', 'kmer4', 'tetranuc_freq', 'dnabert2'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['grover_dna', 'mistral_dna', 'hyenadna', 'nt_v2_500m', 'tnf_canonical', 'kmer6'],
    },
    'nt_H3K9ac': {  # 24,278 samples. Winner: kmer6+ridge=0.5082 MCC. dnabert2 kept.
        'general': [
            'kmer6', 'kmer5', 'mistral_dna', 'grover_dna', 'nt_v2_500m',
            'nt_v2_250m', 'hyenadna', 'tnf_canonical', 'kmer4', 'tetranuc_freq',
            'kmer3', 'dnabert2'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['kmer6', 'kmer5', 'mistral_dna', 'grover_dna'],
    },
    'nt_H3K9me3': {  # 28,288 samples. Winner: mistral_dna+ridge=0.3482 MCC (hardest nt task).
        'general': [
            'mistral_dna', 'hyenadna', 'grover_dna', 'nt_v2_250m', 'nt_v2_500m',
            'kmer4', 'kmer6', 'tnf_canonical', 'tetranuc_freq', 'kmer5', 'kmer3'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'hyenadna', 'grover_dna'],
    },
    'nt_H4K20me1': {  # 32,270 samples. Winner: hyenadna+mlp=0.6041 MCC.
        'general': [
            'hyenadna', 'mistral_dna', 'grover_dna', 'nt_v2_250m', 'nt_v2_500m',
            'kmer3', 'kmer4', 'tetranuc_freq', 'tnf_canonical', 'kmer6', 'kmer5'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['hyenadna', 'mistral_dna', 'grover_dna'],
    },
    'nt_promoter_all': {  # 31,584 samples. Winner: mistral_dna+ridge=0.7281 MCC. dnabert2 kept (saturated).
        'general': [
            'mistral_dna', 'hyenadna', 'grover_dna', 'nt_v2_250m', 'kmer6',
            'kmer5', 'nt_v2_500m', 'kmer4', 'tnf_canonical', 'tetranuc_freq',
            'kmer3', 'dnabert2'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'hyenadna', 'grover_dna'],
    },
    'nt_promoter_tata': {  # 5,274 samples (smallest nt). Winner: mistral_dna+ridge=0.6743 MCC.
        'general': [
            'mistral_dna', 'kmer5', 'hyenadna', 'grover_dna', 'kmer4',
            'nt_v2_250m', 'kmer6', 'tnf_canonical', 'tetranuc_freq', 'nt_v2_500m'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'kmer5'],
    },
    'nt_promoter_no_tata': {  # 31,372 samples. Winner: mistral_dna+ridge=0.7497 MCC. dnabert2 kept.
        'general': [
            'mistral_dna', 'hyenadna', 'grover_dna', 'nt_v2_500m', 'kmer6',
            'nt_v2_250m', 'kmer5', 'kmer4', 'tnf_canonical', 'tetranuc_freq',
            'kmer3', 'dnabert2'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['mistral_dna', 'hyenadna', 'grover_dna', 'nt_v2_500m'],
    },
    # --- Genomic Benchmarks (different task layout, 5 natural groups) ---
    'gb_human_nontata_promoters': {  # 36,131 samples, 5 groups. Winner: kmer5+mlp=0.8917 MCC.
        # Classical k-mer dominance here — NT-v2 + hyenadna all underperform.
        # dnabert2 retained (within top half of general).
        'general': [
            'kmer5', 'kmer6', 'kmer4', 'tetranuc_freq', 'grover_dna',
            'tnf_canonical', 'dnabert2', 'mistral_dna', 'kmer3'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['kmer5', 'kmer6', 'kmer4', 'tetranuc_freq'],
    },
    'gb_human_enhancers_cohn': {  # 27,791 samples, 5 groups. Winner: grover_dna+ridge=0.4853 MCC.
        'general': [
            'grover_dna', 'kmer6', 'kmer5', 'mistral_dna', 'tnf_canonical',
            'hyenadna', 'nt_v2_250m', 'kmer4', 'nt_v2_500m', 'tetranuc_freq'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
        'top_tier': ['grover_dna', 'kmer6', 'kmer5', 'mistral_dna'],
    },
    'gb_demo_human_or_worm': {  # 100,000 samples, 5 groups. Winner: tetranuc_freq+mlp=0.9135 MCC.
        # Saturated task — top 8 reps within 98.7% of best. Classicals dominate.
        # dnabert2 dropped (91.5% of best, pattern consistent with gb_demo_coding sibling).
        # gena_lm dropped (5% of best — near-random at 1024D post-refix, all-nucleotide pattern).
        # Survey gap: gc_content, dinuc_freq, complexity, dna_shape did not run.
        'general': [
            'tetranuc_freq', 'tnf_canonical', 'kmer4', 'kmer5', 'grover_dna',
            'kmer3', 'nt_v2_500m', 'nt_v2_250m', 'mistral_dna', 'kmer6',
            'hyenadna', 'codon_usage'],
        'augmentation': ['codon_usage'],
        'top_tier': ['tetranuc_freq', 'tnf_canonical', 'kmer4', 'kmer5',
                     'grover_dna', 'kmer3', 'nt_v2_500m', 'nt_v2_250m'],
    },
}


# ── Augmentation group categories (for C21 single-rep augmentation) ────────
# Domain-aware rep groupings for testing whether augmentation value comes from
# diversity of signal type vs depth within one type.
# Each domain defines named groups; C21 tests each group as an augmentation pool.

AUG_GROUPS = {
    'molecular': {
        'structural': ['pharmacophore3d', 'coulomb_matrix', 'usr', 'usrcat', 'pdv'],
        'fingerprint': ['maccs', 'rdkfp', 'ecfp4', 'atom_pair', 'avalon',
                        'top_torsion'],
        'learned': ['mol2vec', 'chemberta', 'chembert', 'molformer', 'selformer',
                    'grover', 'graphmvp', 'molclr', 'mhg_gnn'],
        'descriptor': ['mordred', 'smited'],
    },
    'antibody': {
        'structural': ['paratope', 'structural', 'esm2_cdr', 'esm2_cdr3',
                       'esm2_per_cdr', 'esm2_per_region'],
        'sequence': ['aac', 'blosum', 'onehot', 'mutation_features'],
        'plm': ['esm2', 'esmc', 'prot_t5', 'igt5', 'ablang2', 'antiberty',
                'esm3', 'saprot', 'abroberta', 'esm2_concat', 'esm2_early',
                'esm2_layer24', 'esm2_midlayers'],
        'mutation_signal': ['esm2_wt_delta', 'esm2_mutsite', 'esm2_llr',
                            'antiberty_llr', 'esm2_entropy_top20'],
    },
    'dta': {
        'drug_structural': ['drug_ecfp4', 'drug_maccs', 'drug_avalon',
                            'drug_pharmacophore3d'],
        'drug_learned': ['drug_chemberta', 'drug_molformer', 'drug_grover',
                         'drug_mol2vec'],
        'protein': ['prot_esm2', 'prot_prost', 'prot_t5', 'prot_aac'],
        'interaction': ['drug_sns', 'mordred'],
    },
    'tcga': {
        'expression': ['rnaseq_pca50', 'rnaseq_hallmark', 'rnaseq_kegg',
                        'rnaseq_reactome', 'gsva_hallmark'],
        'mutation': ['mutation_binary', 'mutation_burden', 'mutation_signature'],
        'epigenetic': ['methylation_pca50', 'promoter_meth'],
        'structural': ['cnv_arm', 'cnv_focal', 'splicing_psi50'],
        'immune': ['immune_cibersort', 'immune_xcell', 'gene_immune'],
        'histology': ['wsi_titan', 'wsi_uni2h'],
    },
    'reaction': {
        'fingerprint': ['reactant_ecfp4', 'diff_ecfp4', 'reactant_avalon',
                        'diff_avalon', 'reactant_maccs', 'diff_maccs'],
        'learned': ['reactant_grover', 'reactant_chemberta', 'diff_mhg_gnn',
                    'rxnfp'],
        'component': ['onehot_components'],
    },
}


def get_aug_groups(dataset_name, available_reps):
    """Return augmentation group dict for a dataset, filtered to available reps.

    Args:
        dataset_name: Dataset name (used to determine domain).
        available_reps: Set/list of rep names that are actually loaded.

    Returns:
        Dict of {group_name: [rep_names]} with only reps in available_reps.
        Also includes 'mixed' group with one rep from each non-empty group.
    """
    # Determine domain
    ds_config = DATASET_REPS.get(dataset_name, {})
    if not ds_config:
        return {}

    # Detect domain from dataset name patterns
    if dataset_name.startswith(('tcga_', 'luad', 'brca')):
        domain = 'tcga'
    elif dataset_name.startswith(('davis', 'kiba', 'bindingdb')):
        domain = 'dta'
    elif dataset_name.startswith(('buchwald', 'doyle', 'suzuki', 'bh_')):
        domain = 'reaction'
    elif any(dataset_name.startswith(p) for p in (
            'flab_', 'abagym_', 'abcov_', 'ddg_', 'tz_', 'humanness',
            'ginkgo_', 'graphinity_')):
        domain = 'antibody'
    else:
        domain = 'molecular'

    domain_groups = AUG_GROUPS.get(domain, {})
    available = set(available_reps)

    # Filter each group to available reps
    groups = {}
    for gname, rep_list in domain_groups.items():
        filtered = [r for r in rep_list if r in available]
        if filtered:
            groups[gname] = filtered

    # Build 'mixed' group: first available rep from each non-empty group
    mixed = []
    for gname, reps in groups.items():
        if reps:
            mixed.append(reps[0])
    if len(mixed) >= 2:
        groups['mixed'] = mixed

    return groups


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
    'c21',                  # single-rep augmentation (merged C20)
    'c22',                  # hybrid explainability
    'c23',                  # evaluator-model matching
    'c24',                  # blueprint transfer (cross-fold)
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
        'c21': 'c21_single_rep_augmentation',
        'c22': 'c22_explainability',
        'c23': 'c23_evaluator_model_matching',
        'c24': 'c24_blueprint_transfer',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.complexity.{mod_name}')


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None,
            pruning_mode: str = 'default_pruning'):
    """Load data once and run all single-dataset complexity experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: fast tier from SPEED_TIERS).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
        pruning_mode: One of PRUNING_MODES from _shared.py. Default 'default_pruning'
            reproduces the legacy single-stage adaptive_pruning=0.15 behaviour.
            'aggressive_pruning' applies td=0.5 at b=50 to gp/svm/mlp ONLY when
            n_samples >= 1000 (gate applied automatically per dataset).
    """
    if config is None:
        from kirby.find_best_hybrid import SPEED_TIERS
        from experiments._shared import SCREENING_BUDGETS
        tier = SPEED_TIERS['full']
        config = {
            'n_splits': tier['n_folds'],
            'search_speed': 'full',
            'allocations': tier['allocation_methods'],
            'importance_methods': tier['importance_methods'],
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': SCREENING_BUDGETS,
            'models': DOWNSTREAM_MODELS,
        }
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

    # NaN guard: some reps produce NaN for certain sequences. Replace with 0 and warn.
    reps = data.get('reps', {})
    for rname, X in list(reps.items()):
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            pct = 100 * nan_count / X.size
            print(f"  WARNING: {rname} has {nan_count} NaN values ({pct:.1f}%) — imputing 0")
            reps[rname] = np.nan_to_num(X, nan=0.0)

    # ── Load augmentation reps ──
    aug_rep_names = DATASET_REPS.get(dataset_name, {}).get('augmentation', [])
    if aug_rep_names:
        missing = [r for r in aug_rep_names if r not in reps]
        if missing:
            print(f"  Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing)
            aug_reps = {}
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_reps[rn] = arr
            data['augmentations'] = aug_reps
        else:
            data['augmentations'] = {r: reps[r] for r in aug_rep_names}
        print(f"  Augmentations ({len(data['augmentations'])}): "
              f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
    else:
        data['augmentations'] = None

    # ── Resolve pruning_mode and apply n_samples gate ──
    from experiments._shared import resolve_pruning_mode, apply_pruning_gate
    _sched, _model_ovr_template, _min_n = resolve_pruning_mode(pruning_mode)
    _model_overrides = apply_pruning_gate(
        _model_ovr_template, len(data['labels']), _min_n)
    print(f"\n  Pruning mode: {pruning_mode}")
    if _model_overrides is not None:
        print(f"    overrides active for: {sorted(_model_overrides.keys())}")
    elif _model_ovr_template is not None:
        print(f"    overrides gated off (n={len(data['labels'])} < {_min_n})")

    # ── Precompute hybrid searches (general reps) ──
    if len(data['reps']) >= 2:
        from experiments._shared import precompute_hybrid_searches
        print(f"\n  Precomputing hybrid searches (general, {len(data['reps'])} reps)...")
        _task_type = data['task_type']
        precompute_results = precompute_hybrid_searches(
            reps=data['reps'],
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name, tier='general',
            adaptive_pruning_schedule=_sched,
            model_pruning_aggressiveness=_model_overrides,
        )
        data['precomputed_searches'] = precompute_results
    else:
        data['precomputed_searches'] = {}

    # ── Precompute hybrid searches (top-tier reps) ──
    top_tier_names = DATASET_REPS.get(dataset_name, {}).get('top_tier', [])
    top_tier_reps = {k: v for k, v in data['reps'].items() if k in top_tier_names}
    data['top_tier_reps'] = top_tier_reps
    if len(top_tier_reps) >= 2:
        print(f"\n  Precomputing hybrid searches (top-tier, {len(top_tier_reps)} reps)...")
        precompute_top = precompute_hybrid_searches(
            reps=top_tier_reps,
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name, tier='top',
            adaptive_pruning_schedule=_sched,
            model_pruning_aggressiveness=_model_overrides,
        )
        data['precomputed_searches_top'] = precompute_top
    else:
        data['precomputed_searches_top'] = {}

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
        'c21': 'c21_single_rep_augmentation',
        'c22': 'c22_explainability',
        'c23': 'c23_evaluator_model_matching',
        'c24': 'c24_blueprint_transfer',
    }
    return name_map.get(exp_id, exp_id)
