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
        'augmentation': ['maccs', 'pharmacophore3d', 'usrcat', 'coulomb_matrix', 'usr'],
        'top_tier': [
            'graph_kernel', 'mol2vec', 'mhg_gnn', 'ecfp4', 'pdv', 'grover',
            'avalon', 'mordred'],
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
        'augmentation': ['pharmacophore3d', 'coulomb_matrix'],
        'top_tier': ['mordred', 'pdv', 'mhg_gnn', 'avalon', 'grover'],
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
    },
    'abagym_HER2': {  # n=146, ρ > 0.24, General (13)
        'general': [
            'prot_t5_cdr', 'prot_t5', 'esm2_early',
            'esm2', 'esmc', 'ablang2', 'igt5', 'antiberty', 'esm2_midlayers',
            'esm2_layer24', 'saprot', 'aac', 'mutation_features'],
        'augmentation': [],
        'top_tier': ['mutation_features', 'prot_t5_cdr', 'prot_t5'],
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
    'humanness_therapeutic': {  # n=564, AUROC > 0.83, General (10)
        'general': [
            'antiberty', 'igt5', 'abroberta', 'ablang2', 'blosum',
            'prot_t5', 'esm2', 'esmc', 'onehot', 'aac'],
        'augmentation': [],
        'top_tier': ['antiberty', 'igt5', 'abroberta', 'ablang2', 'blosum'],
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
    'c21',                  # single-rep augmentation (merged C20)
    'c22',                  # hybrid explainability
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
        config: Config dict override (default: fast tier from SPEED_TIERS).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
    """
    if config is None:
        from kirby.find_best_hybrid import SPEED_TIERS
        from experiments._shared import SCREENING_BUDGETS
        tier = SPEED_TIERS['fast']
        config = {
            'n_splits': tier['n_folds'],
            'search_speed': 'fast',
            'allocations': tier['allocation_methods'],
            'importance_methods': tier['importance_methods'],
            'internal_evaluators': tier['internal_evaluators'],
            'budgets': SCREENING_BUDGETS,
            'models': ['rf', 'lightgbm', 'mlp'],
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

    # ── Precompute hybrid searches (general reps) ──
    if len(data['reps']) >= 2:
        from experiments._shared import precompute_hybrid_searches
        print(f"\n  Precomputing hybrid searches (general, {len(data['reps'])} reps)...")
        _task_type = data.get('task_type', 'regression')
        precompute_results = precompute_hybrid_searches(
            reps=data['reps'],
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
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
    }
    return name_map.get(exp_id, exp_id)
