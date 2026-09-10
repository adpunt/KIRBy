"""Shared configuration and runner for per-dataset complexity experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from DATASET_REPS[ds]['general'].
Cross-dataset experiments (C03, C14) stay in their own files and
consume the per-dataset JSON outputs.

Config (run_all default): the DEFAULT_* constants — 5-fold scaffold CV, 6
downstream models (DOWNSTREAM_MODELS = rf / xgboost / svm / mlp /
lightgbm), INTERNAL_EVALUATORS = ['match', 'rf', 'svm', 'lightgbm', 'mlp'],
and budgets = FULL_BUDGETS ([10, 25, 50, 100, 250, 500, 1000, 2000]).

NOTE on budgets: FULL_BUDGETS is the only budget list the default config
uses. The precompute itself (precompute_hybrid_searches) also runs
FULL_BUDGETS, so the precompute cache and pool lookups are built at the
same 8-budget grid the experiments sweep.
"""

import sys
import os
import json
import time

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

import numpy as np

from experiments._shared import (
    load_data, save_experiment_results,
    input_fingerprint, fingerprint_mismatch,
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
)


def rep_dim_summary(reps):
    """Per-rep feature dimensions + total, stamped into every complexity
    experiment's output JSON so wall-time can be attributed to representation
    dimensionality — the cost driver behind CX-9/11/12 and the per-dataset
    dimension dump UA-1 needs (it was previously recorded nowhere). Pure
    metadata: does NOT change any computed score or cache key."""
    if not isinstance(reps, dict):
        return {}
    dims = {}
    for name, X in reps.items():
        try:
            dims[name] = int(np.asarray(X).shape[1])
        except (IndexError, AttributeError, TypeError, ValueError):
            continue
    return {
        'per_rep': dims,
        'total_dim': int(sum(dims.values())),
        'n_reps': len(dims),
    }


# ── (historical note) cached_create_hybrid was removed ────────────────────
# A `cached_create_hybrid` wrapper used to live here. DELETED (2026-03-29)
# because it operated on full data with a random internal split, causing data
# leakage. It is replaced by the precompute_hybrids system (find_best_hybrid
# with cache=True), which operates on proper per-fold train splits. No
# create_hybrid wrapper exists in this module any more.


# ── Experiment config is derived from the DEFAULT_* constants ─────────────
# No standalone config dict here. run_all() builds config from the
# find_best_hybrid DEFAULT_* constants via get_config() or directly.


# ── Per-dataset rep pools ─────────────────────────────────────────────────
# DATASET_REPS[ds]['general'] = the one pool of <=10 complementary reps used as
# base reps in every experiment (see DATASET_STATUS.md / REP_PRUNING_LOG.md).
# There are NO tiers (top/general retired 2026-08-24).
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

# Structure-tuned similarity recipes (molecular only). These are the nine
# stacked-block architectures in kirby.similarity.recipes.RECIPE_ARCHITECTURES
# plus the two selected blocks on their own as controls. They are per-fold
# factory reps, dispatched by find_best_hybrid_molecular.get_rep_factories.
#
# They live under their OWN key, never inside 'general'. 'general' holds the
# pruned-to-10 pool the user signed off per dataset (REP_PRUNING_LOG.md);
# dropping eleven treatments into it would change the pool every experiment in
# the framework runs on. Ask for them explicitly via `similarity_rep_names()`.
#
# Before this key existed the recipes were registered and dispatchable but
# appeared in no dataset's rep list, so no experiment could request one — they
# were unreachable code.
SIMILARITY_RECIPE_REPS = [
    'sim_count',
    'sim_count_desc', 'sim_count_desc_half',
    'sim_count_desc208', 'sim_count_desc208_half',
    'sim_count_shape', 'sim_count_shape_half',
    'sim_all', 'sim_all_half',
    # The two selected blocks alone — controls for "did the selection help?"
    'sim_saboteur_drop', 'sim_fine2d',
]


def similarity_rep_names(dataset_name, include_general=True):
    """Rep names for a similarity run on `dataset_name`.

    Returns the dataset's signed-off 'general' pool plus its structure-tuned
    recipes, so a similarity experiment can pass the result straight to
    load_data(rep_names=...). Datasets with no 'similarity' key (everything
    non-molecular — the recipes need SMILES, 3D conformers and MCS pairs) get
    their 'general' pool back unchanged.

    include_general=False returns the recipes alone, for a recipes-only arm.
    """
    entry = DATASET_REPS.get(dataset_name) or {}
    recipes = list(entry.get('similarity') or [])
    if not include_general:
        return recipes
    return list(entry.get('general') or []) + recipes


DATASET_REPS = {
    # ── Molecular (8 datasets) — 7 regression + 1 classification ───────
    'herg': {  # FLuID classification (Lhasa). n=7,772.
        # Pruned 18 -> 10 complementary reps 2026-08-24 (REP_PRUNING_LOG.md, user call).
        # First real rep_prune score (AUROC), after the graph_kernel bond-less-salt fix
        # (commit 9665eab) unblocked the run. Saturated fingerprint pool (all solo 0.80-0.86);
        # cut redundant fingerprints + weak pretrained tail. Cut: pdv, selformer, chemberta,
        # graphmvp, rdkfp, chembert, maccs, smited.
        'general': [
            'mhg_gnn', 'mordred', 'grover', 'atom_pair', 'avalon', 'graph_kernel',
            'ecfp4', 'mol2vec', 'top_torsion', 'molformer'],
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': ['usrcat', 'coulomb_matrix', 'molclr'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
        },
    },
    'herg_ki': {
        # Pruned 15 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut: mol2vec, molformer, chemberta, top_torsion, graphmvp.
        'general': [
            'mhg_gnn', 'mordred', 'graph_kernel', 'avalon', 'grover', 'pdv',
            'ecfp4', 'atom_pair', 'chembert', 'smited'],
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': ['maccs', 'pharmacophore3d', 'coulomb_matrix', 'usr', 'usrcat'],
        'ft_reps': {
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'caco2': {
        # Pruned 12 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut: mordred, mol2vec.
        'general': [
            'graph_kernel', 'mhg_gnn', 'ecfp4', 'pdv', 'grover', 'avalon',
            'top_torsion', 'smited', 'atom_pair', 'molformer'],
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': ['maccs', 'pharmacophore3d', 'usrcat', 'coulomb_matrix', 'usr'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
        },
    },
    'logd': {
        # Pruned 15 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut: pdv, mol2vec, chemberta, graph_kernel, atom_pair.
        'general': [
            'mhg_gnn', 'mordred', 'grover', 'ecfp4', 'smited', 'molformer',
            'avalon', 'top_torsion', 'chembert', 'graphmvp'],
        # 2026-08-21: dropped selformer (768), rdkfp (2048) and molclr (512).
        # This list was created in 9e6039c as the "did not make general" bucket,
        # not as small supplementary blocks, so three full-size reps sat in it —
        # 3,328 of its 3,604 columns, 0.75x the width of the widest base rep.
        # That conflates "add a few supplementary columns" with "concatenate a
        # second whole representation". The remaining three are genuine small
        # blocks (167 + 49 + 60 = 276 columns).
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': [
            'maccs', 'pharmacophore3d', 'usrcat'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
            'contrastive_molformer': {'factory': 'molecular_contrastive', 'base_model': 'molformer'},
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'qm9': {
        # Pruned 16 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Saturated target; redundancy is in the strong fingerprints (2 tight clusters),
        # not the pretrained tail. Kept 1 rep per prediction-cluster (5 structural + 5 learned).
        # Cut: pdv, grover, mhg_gnn, rdkfp, atom_pair, maccs.
        'general': [
            'mordred', 'avalon', 'smited', 'ecfp4', 'selformer', 'chemberta',
            'mol2vec', 'chembert', 'graphmvp', 'molformer'],
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': ['top_torsion', 'pharmacophore3d', 'usrcat', 'coulomb_matrix'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
        },
    },
    'adme_hlm': {
        # Pruned 16 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut: ecfp4, top_torsion, chemberta, mol2vec, graphmvp, selformer.
        'general': [
            'mordred', 'pdv', 'mhg_gnn', 'smited', 'molformer', 'grover',
            'atom_pair', 'chembert', 'graph_kernel', 'avalon'],
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': ['pharmacophore3d', 'maccs', 'usrcat', 'coulomb_matrix', 'usr'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
            'contrastive_molformer': {'factory': 'molecular_contrastive', 'base_model': 'molformer'},
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'adme_rlm': {
        # Pruned 16 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut: chemberta, chembert, top_torsion, mol2vec, graphmvp, selformer.
        'general': [
            'pdv', 'mhg_gnn', 'smited', 'mordred', 'molformer', 'grover',
            'graph_kernel', 'atom_pair', 'avalon', 'ecfp4'],
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': ['pharmacophore3d', 'maccs', 'usrcat', 'coulomb_matrix', 'usr'],
        'ft_reps': {
            'lora_molformer': {'factory': 'molecular_lora', 'base_model': 'molformer'},
            'contrastive_molformer': {'factory': 'molecular_contrastive', 'base_model': 'molformer'},
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },
    'adme_mdr1': {
        # Pruned 16 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut: molformer, chembert, chemberta, rdkfp, top_torsion, selformer.
        'general': [
            'mordred', 'pdv', 'mhg_gnn', 'avalon', 'grover', 'smited',
            'atom_pair', 'graph_kernel', 'maccs', 'ecfp4'],
        # Structure-tuned recipes — their own key, NOT in 'general'.
        'similarity': SIMILARITY_RECIPE_REPS,
        'augmentation': ['pharmacophore3d', 'coulomb_matrix'],
        'ft_reps': {
            'gnn_graphmvp': {'factory': 'molecular_gnn', 'base_model': 'graphmvp'},
        },
    },

    # ── Antibody Inter-Family (11 datasets) ─────────────────────────────
    'flab_thermo_tresanco': {
        # Pruned 28 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 (CDR twins incl. esm3_cdr/esmc_cdr which are byte-identical dup
        # caches here, see project_esm3_esmc_cdr_cache_duplicate; + esm2 family + paratope/structural
        # 1.00 pair) then user cuts; onehot restored (most distinct of the tail).
        'general': [
            'saprot', 'esm3', 'esmc', 'esm2_entropy_top20', 'esm2_cdr',
            'esm2_early', 'esm2_wt_delta', 'prot_t5_cdr', 'igt5', 'onehot'],
        'augmentation': ['mutation_features'],
    },
    'abcov_kd': {
        # Pruned 27 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 (CDR twins + esm2 family) then user cut redundant/weak reps.
        #
        'general': [
            'igt5', 'antiberty_cdr', 'abroberta_cdr', 'blosum', 'abroberta',
            'saprot_cdr', 'ablang2', 'onehot', 'esm3', 'esmc_cdr'],
        'augmentation': [],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },
    # flab_immunogenicity_marks CUT 2026-07-18 (user decision).
    'ddg_combined': {
        # Pruned 18 -> 8 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Weak target (6 reps had NEGATIVE solo). Greedy cut corr>=0.95 then cut everything with
        # solo < 0.1 (user: <10 is fine here).
        'general': [
            'mutation_features', 'onehot', 'antiberty_cdr', 'esm2_layer24',
            'prot_t5_cdr', 'esm2_concat', 'esm2_wt_delta', 'antiberty'],
        'augmentation': ['aac'],
    },

    # ── Antibody DMS / Large DMS (5 datasets) ──────────────────────────
    'flab_binding_koenig': {
        # Pruned -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 (CDR twins) then user cuts.
        'general': [
            'esm3', 'ablang2', 'prot_t5', 'igt5', 'esmc_cdr', 'saprot',
            'esm2_early', 'esm2_entropy_top20', 'esm2', 'esm2_concat'],
        'augmentation': ['esm2_llr', 'mutation_features', 'antiberty_llr', 'aac'],
    },
    'flab_expression_koenig': {
        # Pruned -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 then user cuts.
        'general': [
            'prot_t5', 'esm3_cdr', 'saprot', 'igt5', 'esm2_early', 'esmc',
            'ablang2', 'esm2_concat', 'esm2', 'esm2_midlayers'],
        'augmentation': ['mutation_features', 'esm2_llr', 'aac'],
    },
    'flab_binding_adams': {
        # Pruned -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # AUROC target, saturated (all solo ~0.58). Greedy cut corr>=0.95 then user cuts.
        #
        'general': [
            'esm2_mutsite', 'abroberta_cdr', 'esm2_concat', 'blosum', 'igt5',
            'esm2_layer24', 'esm2_early', 'esm3_cdr', 'ablang2', 'saprot'],
        'augmentation': ['aac', 'paratope'],
    },
    'flab_expression_adams': {
        # Pruned -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Weak target (best solo 0.32). Greedy cut corr>=0.95 then cut bottom by solo.
        #
        'general': [
            'blosum', 'onehot', 'prot_t5', 'esm3_cdr', 'esm2_entropy_top20',
            'esm2_midlayers', 'saprot', 'esm2_layer24', 'esm2_early', 'esmc'],
        'augmentation': ['esm2_llr'],
    },
    # flab_binding_phillips_h3 CUT 2026-07-18 (user): single-WT combinatorial
    # library (all seqs 94-99% identical) — no clean grouped split, and saturated
    # (aac hit AUROC 0.98 even while leaking). No headroom for the rep thesis.

    # ── AbAgym DMS (5 datasets) ─────────────────────────────────────────
    # Surveys: tests/results/antibody_ft_fill/abagym_*.json (32 reps).
    # DMS-special reps (mutation_features, structural) not in survey.
    # Dead reps excluded: aido_protein, igbert, igbert_cls, mint, xtrimopglm.
    'abagym_Ang2': {  # n=397, ρ > 0.28, General (24)
        # Pruned 24 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut CDR-variant twins (saprot_cdr, esm3_cdr, prot_t5_cdr, esmc_cdr, esm2_cdr3 all ≡ base
        # at 1.00) + next-tier redundant esm2 family + weak aac/structural.
        'general': [
            'saprot', 'esm2_midlayers', 'esm2_layer24', 'antiberty', 'esm3',
            'prot_t5', 'esmc', 'abroberta_cdr', 'esm2_cdr', 'mutation_features'],
        'augmentation': [],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },
    'abagym_EGFR': {  # n=659, ρ > 0.18, General (26)
        # Pruned 26 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Weak-signal target (best solo 0.28); cut CDR-variant twins + the redundant esm2 family
        # and low-solo tail.
        'general': [
            'mutation_features', 'esm2_concat', 'esm3', 'abroberta', 'abroberta_cdr',
            'ablang2', 'aac', 'saprot_cdr', 'esmc', 'esm2_early'],
        'augmentation': [],
        'ft_reps': {
            'antibody_lora_esm2': {'factory': 'antibody_lora', 'base_model': 'esm2'},
        },
    },
    'abagym_VEGF': {  # n=418, ρ > 0.39, General (24)
        # Pruned 24 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut CDR-variant twins + redundant esm2 family.
        'general': [
            'mutation_features', 'saprot', 'prot_t5', 'structural', 'antiberty_cdr',
            'esm2_layer24', 'abroberta_cdr', 'esmc', 'esm2_early', 'ablang2'],
        'augmentation': [],
    },

    # ── Antibody CLF (2 datasets) ───────────────────────────────────────
    # tz_her2: tests/results/opig/clf/tz_her2_binary.json (37 reps).
    # humanness: tests/results/antibody_ft_fill/humanness_therapeutic.json (31 reps).
    'tz_her2_binary': {  # n=5000, AUROC > 0.80, General (29)
        # Pruned 29 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Saturated AUROC target (all solo 0.81-0.92). Greedy cut corr>=0.95 then user cuts.
        #
        'general': [
            'blosum', 'esm2_per_cdr', 'esm2_midlayers', 'prot_t5', 'esm3',
            'esmc', 'abroberta_cdr', 'antiberty', 'mutation_features', 'ablang2'],
        'augmentation': [],
    },
    'tz_her2_multiclass': {  # n=5000, F1w > 0.65, General (16). FULL survey 2026-04-16.
        # Pruned 16 -> 10 complementary reps 2026-08-23 (REP_PRUNING_LOG.md, user call).
        # Multiclass (AUROC-ovr rep-screen after the multiclass fix, commit 5974a61).
        # Greedy cut corr>=0.95 lands exactly at 10.
        'general': [
            'blosum', 'onehot', 'esm2_midlayers', 'esm2_layer24', 'esm2_concat',
            'esm2_per_cdr', 'esm2_early', 'esm2_entropy_top20', 'abroberta_cdr', 'igt5'],
        'augmentation': [],
    },
    'humanness_therapeutic': {  # n=564, AUROC. FULL survey 2026-04-16.
        # Winner: antiberty_cdr+rf=0.9623 AUROC. Saturated task; many reps ≥0.90.
        # 5 reps broken (igbert_cls, igbert, xtrimopglm, mint, aido_protein).
        # top_tier RECUT 2026-07-11 from 21 -> 5: the old >93%-of-best cutoff on
        # a saturated task admitted almost the whole general pool (21/26), which
        # is not a top tier. c01 baselines show a clean break — 5 reps ≥0.90
        # AUROC (antiberty_cdr/antiberty 0.914, igt5 0.913, abroberta_cdr/
        # abroberta 0.900), then a drop to ablang2 0.889. general unchanged.
        # Pruned 26 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Saturated AUROC target, small-n (p≫n). Greedy cut corr>=0.95 then user cuts.
        #
        'general': [
            'igt5', 'antiberty_cdr', 'abroberta', 'ablang2', 'blosum',
            'esmc', 'esm2_per_cdr', 'esm2_early', 'prot_t5', 'esm2_midlayers'],
        'augmentation': [],
    },

    # ── Ginkgo Developability (6 datasets) ──────────────────────────────
    # Surveys: tests/results/opig/ginkgo/ginkgo_*.json (34 reps each).
    # paratope/structural in augmentation for all; also in general when
    # they pass the dataset cutoff (heparin, hic).
    'ginkgo_acsins': {  # n=242, ρ > 0.24, General (26)
        # Pruned 26 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Small-n (242) high-dim embeddings — greedy cut corr>=0.95 then user cuts.
        'general': [
            'igt5', 'ablang2', 'antiberty_cdr', 'esm2_concat', 'abroberta',
            'saprot', 'aac', 'esmc', 'esm2_early', 'prot_t5'],
        'augmentation': ['paratope', 'structural'],
    },
    'ginkgo_colloidal': {  # n=242, ρ > 0.20, General (23)
        # Pruned 23 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Weak target (best solo 0.23). Greedy cut corr>=0.95 then cut bottom 5 by solo.
        'general': [
            'prot_t5', 'esm2_concat', 'esm2_entropy_top20', 'esm3', 'igt5',
            'esm2_early', 'ablang2', 'esmc', 'saprot', 'antiberty'],
        'augmentation': ['paratope', 'structural'],
    },
    'ginkgo_hic': {  # n=242, ρ > 0.35, General (23)
        # Pruned 23 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 then cut bottom 3 by solo.
        'general': [
            'esm2_concat', 'aac', 'esm3', 'prot_t5', 'esm2_early', 'igt5',
            'esm2_entropy_top20', 'esmc', 'ablang2', 'esm2_midlayers'],
        'augmentation': ['paratope', 'structural'],
    },
    'ginkgo_pr_cho': {  # n=197, ρ > 0.22, General (23)
        # Pruned 23 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 then user cuts.
        'general': [
            'esm2_concat', 'esmc', 'prot_t5', 'saprot', 'onehot', 'aac',
            'esm2_per_region', 'igt5', 'esm2_entropy_top20', 'ablang2'],
        'augmentation': ['paratope', 'structural'],
    },
    'ginkgo_titer': {  # n=239, ρ > 0.19, General (23)
        # Pruned 23 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Weak target (best solo 0.345). Greedy cut corr>=0.95 then cut bottom 4 by solo.
        #
        'general': [
            'saprot', 'igt5', 'esm2', 'esmc', 'esm2_entropy_top20',
            'esm2_concat', 'esm2_midlayers', 'ablang2', 'prot_t5', 'esm3'],
        'augmentation': ['paratope', 'structural'],
    },

    # ── TCGA (2 datasets) ──────────────────────────────────────────────
    'tcga_luad_vs_lusc': {
        # Pruned 20 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Multi-modal omics (least-redundant pool in the study; nothing >=0.95). Kept a spread
        # across every modality: gene(3) + methyl(2) + wsi(2) + cnv(1) + mirna(1) + rppa(1).
        #
        'general': [
            'gene_geneformer', 'methyl_hvg', 'gene_bulkrnabert', 'wsi_titan',
            'methyl_pca50', 'gene_chromosome_arm', 'cnv_pca50', 'mirna_top200',
            'wsi_ctranspath', 'rppa'],
        'augmentation': [
            'immune_xcell', 'gene_stats', 'stemness_mrnasi',
            'immune_cibersort', 'cimp_status', 'mut_tmb',
            'hrd_score', 'purity_consensus'],
    },
    'tcga_brca_pam50': {
        # Pruned 21 -> 10 complementary reps 2026-08-23 (REP_PRUNING_LOG.md, user call).
        # Multiclass PAM50 (AUROC-ovr rep-screen after the multiclass fix, commit 5974a61).
        # Multi-modal omics, least-redundant (greedy cut 0); user kept a spread across modalities.
        #
        'general': [
            'gene_geneformer_v2', 'gene_bulkrnabert', 'rppa', 'cnv_pca50', 'wsi_titan',
            'wsi_hoptimus0', 'wsi_gigapath', 'wsi_ctranspath', 'wsi_phikon', 'mirna_top200'],
        'augmentation': [
            'hrd_score', 'mut_tmb', 'purity_consensus', 'gene_stats',
            'gene_chromosome_arm', 'immune_xcell'],
    },

    # ── Reaction (2 datasets) ─────────────────────────────────────────
    'buchwald_hartwig': {
        # 2026-07-18 (user decision): mirror bh_doyle_plates' full applicable rep
        # set now that the reactant_/diff_ MOLECULAR-rep variants are wired into
        # _shared._rxn_generators (via create_reaction_rep_variant). cgr_morgan is
        # the proof-of-concept best single rep; the reactant_/diff_ variants are
        # the likely source of the top-5 complementarity behind the hybrid win.
        # All these matrices are already cached.
        # Pruned 19 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # reactant_/diff_ fingerprint variants are near-twins (0.94-0.98) -> kept one per family +
        # the distinct reaction-level reps. Cut: diff_ecfp4, reactant_mhg_gnn, reactant_pdv,
        # diff_mol2vec, diff_grover, diff_mordred, reactant_avalon, drfp_2048, reactant_ecfp4.
        #
        'general': [
            'cgr_morgan', 'reactant_grover', 'diff_avalon', 'reactant_mordred',
            'diff_mhg_gnn', 'reactiont5', 'onehot_components', 'reactant_mol2vec',
            'rxnfp', 'diff_maccs'],
        'augmentation': [
            'product_maccs', 'reactant_maccs',
            'product_pharmacophore3d', 'reactant_pharmacophore3d'],
    },
    'bh_doyle_plates': {
        # Pruned 18 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Cut: diff_ecfp4, diff_grover, reactant_mhg_gnn, reactant_avalon, reactant_mol2vec,
        # diff_maccs, diff_mordred, drfp_2048.
        'general': [
            'reactant_ecfp4', 'reactant_grover', 'diff_mhg_gnn', 'cgr_morgan',
            'onehot_components', 'diff_avalon', 'reactiont5', 'diff_mol2vec',
            'reactant_mordred', 'rxnfp'],
        'augmentation': [
            'product_maccs', 'reactant_maccs',
            'product_pharmacophore3d',
            'reactant_pharmacophore3d'],
    },

    # ── DTA (drug-target affinity, cold-protein splits) ───────────────
    # Reviewed 2026-04-28 from phase5 baseline re-survey under commit 506e4bb
    # (per-fold importance filter for high-dim reps; MLP dropped from
    # protein_* candidate pool — every protein_*+mlp had severely negative
    # R² before the fix).
    #
    # Universal drops across all DTA: drug_unimol_v2, protein_aido,
    # protein_mint, protein_xtrimopglm — all collapsed to CI=0.5 (random).
    # protein_esm2_lora kept in ft_reps for C02 even where FT doesn't beat
    # frozen, so the frozen+FT condition has a candidate to test.
    #
    # davis: tier review PARKED pending davis_invest job 7568428 (drug-rep
    # ceiling at CI=0.7567 may be per-drug-mean degeneracy; investigation
    # tests this hypothesis on the same 5-fold splits).
    'davis': {  # added 2026-06-10 (§G). Mirrors kiba's DTA rep set so precompute
        # stops skipping it (loader+registry already support davis; it was just
        # missing a DATASET_REPS key). Same drug+protein reps as kiba.
        'general': [
            # drug
            'drug_mordred', 'drug_pdv', 'drug_mol2vec', 'drug_chemberta',
            'drug_ecfp4', 'drug_molformer', 'drug_avalon', 'drug_maccs',
            'drug_selformer',
            # protein
            'protein_saprot', 'protein_esm2_pocket', 'protein_esmc',
            'protein_esm2_midlayers', 'protein_esm3', 'protein_esm2',
            'protein_kmer'],
        'augmentation': ['protein_usr_shape', 'protein_pocket', 'protein_contact_graph'],
        'ft_reps': {
            'protein_esm2_lora': {'factory': 'dta_protein_lora', 'base_model': 'esm2'},
        },
    },
    'kiba': {  # n=30,000, 229 protein groups. Winner: drug_mordred+xgboost=0.6924 CI.
        # Drug-side near-tied (spread 0.0006 across 9 reps) — partly degenerate
        # under cold-protein with only 229 groups. Top tier limited to 2 drug
        # + 1 protein. protein_aac dropped at 0.5496 (below 0.55 viability).
        # Pruned 16 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Two-block target (redundant drug blob + distinct protein reps) -> thin the drug side,
        # keep most of the protein side (it carries the target). Cut: drug_molformer, drug_mol2vec,
        # drug_chemberta, drug_selformer, protein_saprot, protein_esm2_pocket.
        'general': [
            # drug (5)
            'drug_mordred', 'drug_pdv', 'drug_ecfp4', 'drug_avalon', 'drug_maccs',
            # protein (5)
            'protein_esm2_midlayers', 'protein_esm2', 'protein_esmc',
            'protein_esm3', 'protein_kmer'],
        'augmentation': ['protein_usr_shape', 'protein_pocket', 'protein_contact_graph'],
        'ft_reps': {
            'protein_esm2_lora': {'factory': 'dta_protein_lora', 'base_model': 'esm2'},
        },
    },
    'bindingdb_kd': {  # n=30,000, 1,219 protein groups. Winner: drug_pdv+rf=0.7260 CI.
        # Real drug-side ranking (spread 0.0125) — 1,219 protein groups gives
        # enough diversity that per-drug-mean stops winning. Mixed-modality
        # top tier of 4 (2 drug + 2 protein) so top-tier-only experiments can
        # form genuine drug+protein hybrids.
        # Pruned 16 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Two-block target -> thin the redundant drug blob, keep most of the distinct protein side.
        # Cut: drug_pdv, drug_molformer, drug_avalon, drug_mol2vec, protein_esm3, protein_kmer.
        #
        'general': [
            # drug (5)
            'drug_mordred', 'drug_maccs', 'drug_chemberta', 'drug_selformer', 'drug_ecfp4',
            # protein (5)
            'protein_esm2_midlayers', 'protein_esm2', 'protein_esm2_pocket',
            'protein_saprot', 'protein_esmc'],
        'augmentation': ['protein_aac', 'protein_pocket', 'protein_contact_graph'],
        'ft_reps': {
            'protein_esm2_lora': {'factory': 'dta_protein_lora', 'base_model': 'esm2'},
        },
    },

    # --- Nucleotide (NT Revised + Genomic Benchmarks, chromosome/natural groups) ---
    # Tier tables below come from the 2026-04-17 gena_refix survey round.
    # Across all nt_* and gb_*, gena_lm was corrected to 1024D hidden
    # states (was 32000D LM-head logits) and reconfirmed near-random on
    # every dataset — excluded universally. dnabert2 dropped per-dataset
    # where it underperforms; kept only on saturated tasks where it sits
    # within 94-96% of the winner.
    'nt_H3K27me3': {  # 33,000 samples. Winner: mistral_dna+ridge=0.5487 MCC.
        # Pruned 11 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 (kmer4).
        'general': [
            'mistral_dna', 'grover_dna', 'hyenadna', 'nt_v2_250m', 'nt_v2_500m',
            'tnf_canonical', 'kmer3', 'kmer5', 'tetranuc_freq', 'kmer6'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
    },
    'nt_H3K4me3': {  # 18,244 samples. Winner: grover_dna+ridge=0.6519 MCC. dnabert2 kept (saturated task).
        # Pruned 12 -> 8 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Small pool of near-twin k-mers; greedy cut corr>=0.95 (kmer3/4/5, tetranuc) -> 8 (<10 OK).
        #
        'general': [
            'grover_dna', 'mistral_dna', 'tnf_canonical', 'nt_v2_250m', 'nt_v2_500m',
            'hyenadna', 'kmer6', 'dnabert2'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
    },
    'nt_H3K9ac': {  # 24,278 samples. Winner: kmer6+ridge=0.5082 MCC. dnabert2 kept.
        # Pruned 12 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Greedy cut corr>=0.95 (tetranuc) then user cut kmer5.
        'general': [
            'tnf_canonical', 'grover_dna', 'nt_v2_500m', 'mistral_dna', 'kmer4',
            'nt_v2_250m', 'kmer3', 'hyenadna', 'kmer6', 'dnabert2'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
    },
    'nt_H3K9me3': {  # 28,288 samples. Winner: mistral_dna+ridge=0.3482 MCC (hardest nt task).
        # Pruned 11 -> 10 complementary reps 2026-08-21 (REP_PRUNING_LOG.md, user call).
        # Hardest/least-redundant nt task (best solo 0.72, nothing >=0.81). Greedy cut 0; user cut
        # tetranuc_freq.
        'general': [
            'mistral_dna', 'grover_dna', 'hyenadna', 'nt_v2_250m', 'nt_v2_500m',
            'tnf_canonical', 'kmer5', 'kmer4', 'kmer3', 'kmer6'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
    },
    'nt_promoter_tata': {  # 5,274 samples (smallest nt). Winner: mistral_dna+ridge=0.6743 MCC.
        # Pruned 10 -> 9 complementary reps 2026-08-21 (REP_PRUNING_LOG.md, user call).
        # Small pool; greedy cut tetranuc_freq -> 9 (<10 OK).
        'general': [
            'mistral_dna', 'grover_dna', 'hyenadna', 'kmer6', 'kmer5',
            'tnf_canonical', 'kmer4', 'nt_v2_500m', 'nt_v2_250m'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
    },
    # --- Genomic Benchmarks (different task layout, 5 natural groups) ---
    'gb_human_enhancers_cohn': {  # 27,791 samples, 5 groups. Winner: grover_dna+ridge=0.4853 MCC.
        # Pruned 10 -> 9 complementary reps 2026-08-21 (REP_PRUNING_LOG.md, user call).
        # Small pool; greedy cut kmer4 -> 9 (<10 OK).
        'general': [
            'grover_dna', 'mistral_dna', 'tnf_canonical', 'tetranuc_freq', 'hyenadna',
            'kmer5', 'nt_v2_500m', 'nt_v2_250m', 'kmer6'],
        'augmentation': ['codon_usage', 'complexity', 'dinuc_freq', 'dna_shape', 'gc_content'],
    },

    # ── FIX 4c: register precompute-seeded datasets (rep lists copied from
    # the named domain/rep-set sibling so the search seeds them instead of
    # cold-recomputing under an unregistered name). ──────────────────────
    # Graphinity ΔΔG (synthetic antibody-protein ΔΔG) ← copy from ddg_combined.
    # Ginkgo developability (antibody) ← copy from ginkgo_hic.
    # ginkgo_tm2 / ginkgo_aggregation / ginkgo_purity CUT 2026-08-20 (user decision):
    # near-dead targets — best base-rep solo ρ was tm2 +0.092, aggregation +0.179,
    # purity +0.151 with most reps NEGATIVE (no signal even under the honest split).
    # See DATASET_STATUS.md. No headroom for the rep thesis -> removed from the roster.
    # Humanness species (antibody clf) ← copy from humanness_therapeutic.
    # ChEMBL QSAR activity-cliff (molecular regression) ← copy from caco2.
    # CDR flexibility (antibody clf, sequence reps) ← copy from humanness_therapeutic.
    'cdr_flexibility': {
        # Pruned 26 -> 10 complementary reps 2026-08-20 (REP_PRUNING_LOG.md, user call).
        # Saturated AUROC target, small-n (p≫n). Greedy cut corr>=0.95 then user cuts.
        #
        'general': [
            'antiberty', 'esm2_concat', 'esmc', 'ablang2', 'abroberta',
            'esm2_early', 'esm3', 'esm2_layer24', 'igt5', 'onehot'],
        'augmentation': [],
    },
}

# NB (2026-08-24): the 'top' tier is fully retired — there is ONE 10-rep pool
# per dataset (`general`). The old `top_tier` lists and the whole `top`/`top_distance`
# cache lineage were removed; `capped` augmentation now rides on `general`. Do not
# reintroduce a separate top list or a `top_tier` key.


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
                            'drug_pdv'],
        'drug_learned': ['drug_chemberta', 'drug_molformer', 'drug_selformer',
                         'drug_mol2vec'],
        'drug_descriptor': ['drug_mordred'],
        'protein': ['protein_esm2', 'protein_saprot', 'protein_esmc',
                    'protein_esm2_midlayers', 'protein_esm3',
                    'protein_esm2_pocket', 'protein_kmer'],
    },
    'tcga': {
        'expression': ['gene_geneformer_v2', 'gene_bulkrnabert',
                       'gene_geneformer', 'gene_stats',
                       'gene_chromosome_arm'],
        'mutation': ['mutation_matrix', 'mut_driver', 'mut_pca50',
                     'mut_pathway', 'mut_tmb'],
        'epigenetic': ['methyl_pca50', 'methyl_hvg'],
        'cnv': ['cnv_pca50', 'cnv_gistic'],
        'immune': ['immune_cibersort', 'immune_xcell', 'stemness_mrnasi',
                   'purity_consensus', 'hrd_score', 'cimp_status'],
        'histology': ['wsi_titan', 'wsi_gigapath', 'wsi_hoptimus0',
                      'wsi_ctranspath', 'wsi_moco', 'wsi_phikon'],
        'mirna': ['mirna_top200'],
        'protein': ['rppa'],
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
    # C21 runs early, right after C02, because its question gates the others:
    # if a top-tier rep plus a few individual features beats the C02 hybrid,
    # every downstream experiment should be comparing against that, not only
    # against the hybrid. C21 reads no other experiment's output; its hybrid
    # reference arms reuse the tier='top' precompute run_all builds up front.
    'c21',                  # single-rep augmentation (merged C20; C20 deleted in 5565666)
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
    'c22',                  # hybrid explainability
    'c23',                  # evaluator-model matching
    'c24',                  # blueprint transfer (cross-fold)
    'c33',                  # budget vs molecule count (standalone)
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
        'c33': 'c33_budget_overfitting',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.complexity.{mod_name}')


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None,
            pruning_mode: str = 'default_pruning',
            force_rerun: bool = False,
            c11_recipe_search: str = 'per_fold'):
    """Load data once and run all single-dataset complexity experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: built from the DEFAULT_* constants).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
        pruning_mode: One of PRUNING_MODES from _shared.py. Default 'default_pruning'
            reproduces the legacy single-stage adaptive_pruning=0.15 behaviour.
            'aggressive_pruning' applies F16-validated multi-stage pruning
            (checkpoints=[25,50,100], td=0.5) globally to all models with no
            n_samples gate. F16 (caco2): ~58% fold-0 eval reduction at zero loss.
        c11_recipe_search: where c11 chooses the hybrid's recipe.
            'per_fold' (default) searches inside each outer training fold, so
            the hybrid arm pays the same tax the ensemble arm already pays and
            the ensemble-versus-hybrid delta is a comparison. It costs one pool
            of searches per outer fold on this dataset — five without
            augmentation representations, fifteen with. 'whole_data' takes one
            recipe from a search over every row: free, but c11 then writes the
            head-to-head under names saying the hybrid saw rows the ensemble
            did not, and does not write auto_minus_hybrid at all. Set this per
            dataset: 'per_fold' wherever the number gets quoted.
    """
    if config is None:
        from kirby.find_best_hybrid import (
            DEFAULT_N_FOLDS, DEFAULT_ALLOCATION_METHODS,
            DEFAULT_IMPORTANCE_METHODS)
        from experiments._shared import FULL_BUDGETS
        config = {
            'n_splits': DEFAULT_N_FOLDS,
            'allocations': DEFAULT_ALLOCATION_METHODS,
            'importance_methods': DEFAULT_IMPORTANCE_METHODS,
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': FULL_BUDGETS,
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
    print(f"Config:      {config['n_splits']} fold, {config['models']}")
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
        aug_reps = {}
        if missing:
            print(f"  Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing)
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_reps[rn] = arr
        # Merge in any aug reps that were already loaded as base reps.
        for rn in aug_rep_names:
            if rn in reps and rn not in aug_reps:
                aug_reps[rn] = reps[rn]
        # If an aug rep was ONLY supposed to be an augmentation (not a base
        # rep), drop it from data['reps'] so find_best_hybrid does not see it
        # in both base and augmentation sets (double-counting / cache pollution).
        _general = DATASET_REPS.get(dataset_name, {}).get('general', [])
        for rn in aug_rep_names:
            if rn in reps and rn not in _general:
                reps.pop(rn, None)
        data['augmentations'] = aug_reps
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
    # precompute_hybrid_searches catches a per-aug_condition exception, stores
    # None under that key, and appends one record to `.failures`. Those records
    # are copied into data['precompute_errors'][tier] here and written to
    # _precompute_errors/<dataset>.json below, so a failed aug_condition is
    # named in the output instead of existing only as a printed line.
    precompute_errors = data.setdefault('precompute_errors', {})
    from experiments._shared import precompute_failures, PrecomputeResults
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
        precompute_errors['general'] = precompute_failures(precompute_results)
    else:
        data['precomputed_searches'] = {}
        precompute_errors['general'] = []

    # ── Precompute distance-tier searches (for c03_cross_dataset_transfer) ──
    # c03 is the only complexity experiment that uses the distance pool. Populate
    # so the pool helper can read in-memory dicts when distance=True.
    from experiments._shared import DISTANCE_EVALUATORS, DISTANCE_BUDGETS
    _dist_cfg = {**config, 'budgets': DISTANCE_BUDGETS}
    if len(data['reps']) >= 2:
        print(f"\n  Precomputing distance-tier (general_distance, {len(data['reps'])} reps)...")
        try:
            data['precomputed_searches_distance'] = precompute_hybrid_searches(
                reps=data['reps'],
                labels=data['labels'],
                groups=data.get('groups'),
                config=_dist_cfg,
                augmentations=data.get('augmentations'),
                task_type=_task_type,
                domain=data.get('domain', 'molecular'),
                dataset_name=dataset_name, tier='general_distance',
                internal_evaluators=DISTANCE_EVALUATORS,
                budgets=DISTANCE_BUDGETS,
                adaptive_pruning_schedule=_sched,
                model_pruning_aggressiveness=_model_overrides,
            )
            precompute_errors['general_distance'] = precompute_failures(
                data['precomputed_searches_distance'])
        except Exception as e:
            print(f"  general_distance precompute failed: {e}")
            # The whole pass raised, so no aug_condition key exists to attach
            # this to: record it with aug_condition None, same six fields.
            _empty = PrecomputeResults()
            _empty.record_failure(e, tier='general_distance',
                                  dataset=dataset_name, aug_condition=None)
            data['precomputed_searches_distance'] = _empty
            precompute_errors['general_distance'] = precompute_failures(_empty)
    else:
        data['precomputed_searches_distance'] = {}
        precompute_errors['general_distance'] = []

    # ── Record precompute failures ──
    # Written before any experiment runs, so the record survives a run that
    # dies part-way. Each record carries tier, dataset, aug_condition,
    # error_type, error and traceback.
    precompute_error_audit = {
        'dataset': dataset_name,
        'n_failures': sum(len(v) for v in precompute_errors.values()),
        'by_tier': {k: list(v) for k, v in precompute_errors.items()},
    }
    if precompute_error_audit['n_failures']:
        print(f"  Precompute failures recorded: "
              f"{precompute_error_audit['n_failures']}")
    save_experiment_results(precompute_error_audit, output_dir,
                            '_precompute_errors', dataset_name)

    # Metric direction for this dataset, resolved once. Passed to every
    # save below so the JSON is stamped with metric_name/lower_is_better even
    # when the dataset isn't in DATASET_REGISTRY — save_experiment_results only
    # re-looks-up the registry (and silently stamps nothing if it misses), so
    # experiments like c02 that don't carry direction in their result dict
    # would otherwise lose it for unregistered datasets. Fall back to None on
    # any failure (save then reverts to its registry-lookup path — no worse
    # than today).
    try:
        from experiments._shared import get_domain_metric
        _stamp_metric_name, _, _stamp_lower = get_domain_metric(data)
    except Exception as _e:
        print(f"  (metric direction not resolved for stamping: {_e})")
        _stamp_metric_name, _stamp_lower = None, None

    # What every result file written below is stamped with, and what a resumed
    # run compares its existing files against.
    _fingerprint = input_fingerprint(data, config)

    # ── Run experiments in order ──
    results = {}
    for exp_id in experiments:
        print(f"\n{'─' * 70}")

        # ── Resume: skip an experiment whose output JSON already exists ──
        # Without this, a job that walls mid-sequence re-runs c01.. from
        # scratch every resubmit and can never reach the back half (c05/c07
        # are uncached and multi-day). Skipping completed experiments lets
        # each resubmit advance. Downstream experiments (c08<-c02, c09<-c02+c06)
        # read upstream JSON from disk, so skipping an upstream is safe.
        # Pass force_rerun=True to recompute regardless (e.g. after a code change).
        #
        # The file existing is not enough. It says nothing about WHICH data it
        # was computed from, so a run resumed after the fold grouping, the
        # labels or the representation roster changed kept the old file and
        # reported it as this run's answer. Each file now carries an
        # input_fingerprint and it has to match what was just loaded.
        _out_json = os.path.join(
            output_dir, _experiment_full_name(exp_id), f"{dataset_name}.json")
        if not force_rerun and os.path.exists(_out_json):
            try:
                with open(_out_json) as _f:
                    _prev = json.load(_f)
                _stale = fingerprint_mismatch(
                    _prev.get('input_fingerprint'), _fingerprint)
            except (ValueError, OSError) as _e:
                _stale = f'could not be read ({_e})'
            if _stale:
                print(f"  RERUN {exp_id}: {_stale} — recomputing ({_out_json})")
            else:
                print(f"  RESUME {exp_id}: already done — skipping ({_out_json})")
                results[exp_id] = {'resumed': True}
                continue

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
        elif exp_id == 'c11':
            kwargs['recipe_search'] = c11_recipe_search

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name, **kwargs)
            results[exp_id] = exp_result

            # Cost-attribution metadata (UA-1 input): stamp the per-rep feature
            # dimensions for this dataset into the experiment output. Pure
            # metadata — never changes a score or cache key. Experiments that
            # already set their own '_rep_dims' are left untouched.
            if isinstance(exp_result, dict) and '_rep_dims' not in exp_result:
                exp_result['_rep_dims'] = rep_dim_summary(data.get('reps'))

            # Save per-experiment JSON
            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(exp_result, output_dir, exp_full_name,
                                    dataset_name,
                                    metric_name=_stamp_metric_name,
                                    lower_is_better=_stamp_lower,
                                    fingerprint=_fingerprint)
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

    # Underscore key so it does not collide with an experiment ID.
    results['_precompute_errors'] = precompute_error_audit
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
        'c33': 'c33_budget_overfitting',
    }
    return name_map.get(exp_id, exp_id)
