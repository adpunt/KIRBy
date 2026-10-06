#!/usr/bin/env python3
"""C13: Forward Compatibility — do new reps benefit from hybridization (and vice versa)?

Paper: Complexity
Conclusion: Quantifies the symbiotic value of new representations within the
    KIRBy hybrid framework, WITH the selection-on-same-data bias controlled.
    A new rep is only credited when the hybrid WITH it beats the hybrid built
    around a random-Gaussian matrix of the SAME dimensionality (D - D_random),
    not merely when it beats the smaller hybrid without it (D - C, which is a
    selection-inflated upper bound).
Dependencies: None (standalone).

Condition ablation per new embedding model r, per dataset d. Aligned
2026-10-05 (handoff 2.4): every condition is scored per outer fold, under each
of the dataset's two decided models, and every hybrid for an outer fold comes
from a search on that fold's training rows (fold_standard_selection: per
model, the 2 best single representations and the 2 best hybrids).
  B: New rep r alone, under each decided model.
     (Condition A, r under one fixed "probe" model, was removed 2026-10-05:
      ridge is removed everywhere, and with both decided models scored A
      repeated B. B no longer picks the best of five models on the folds it is
      reported on.)

  Two without-sides, both run and reported side by side (decided 2026-09-12):

  Side 1 — against the dataset's other representations. Every other embedding
  model stays in, so on flab_thermo_tresanco the without-side for esm3 still
  holds saprot, esmc and igt5.
  C:        Hybrids over the dataset's other representations, r left out
  D:        Hybrids over those plus r
  D_random: D's hybrids with r's selected features replaced by the same
            number of random-Gaussian features (forced share, DECIDED
            2026-10-06; no search of its own)

  Side 2 — against the established representations only. Every OTHER embedding
  model comes out of the without-side too, so the subtraction reads as one new
  model against the established set rather than against its own siblings.
  C_established:        Hybrids over the established representations
  D_established:        Hybrids over those plus r
  D_random_established: D_established's hybrids, r's features replaced
                        the same way

  Each of these is reported for hybrid1 and hybrid2, the two best hybrids of
  that fold's search for that model; the 2 best single representations of the
  same search are in the result file too.

  (Condition E is gone. It was once a duplicate of D and carried no
   information; the "does the hybrid help r itself" question is now answered
   honestly by D - B, below. No E / E - B is emitted any more.)

Controls per new rep r, under each decided model:
  - Random Gaussian matrix (same dims as r), alone
  - Row-shuffled r, alone: tests content vs dimensionality

Key metrics (all sign-normalised: positive = improvement, any metric direction),
per decided model and hybrid slot, fold by fold on the same outer folds:
  1. D - C          : new-rep contribution, IN-SAMPLE UPPER BOUND ONLY. D searches
                      a superset of C on the same data, so this is biased positive
                      by construction. Never present it as an unbiased effect.
  2. D - D_random   : the same hybrid with r's features against the same
                      number of random-Gaussian features; only the content of
                      those features differs. Since 2026-10-06 the Gaussian is
                      forced to r's feature share (in the 2026-09-29 run its
                      own search gave it 0 features on every fold, so
                      D - D_random was D - C). A fold where r got 0 features
                      in D gives D_random = D there.
  3. D - B          : does the hybrid improve on r alone under the same model?
  4. Features from r: subject_features_D, per fold
  5. Win counts     : how many embedding models have D - D_random and D - B
                      above 0, out of those with a score.
  6. The same three deltas again on side 2: D_established - C_established,
     D_established - D_random_established, D_established - B. Same units, same
     sign convention; they answer the same questions with every other embedding
     model held out of the without-side.

No significance testing is reported. Wilcoxon on <=5 non-independent CV folds is
underpowered and Friedman/Nemenyi need many datasets per call; a p-value here would
be misleading. Present effect sizes (the deltas above) with fold spread, not p-values.

Where the subjects come from:
    Two places, and every subject is scored one at a time.
    1. The dataset's own representations, whichever of them the rule below calls
       a learned embedding model (see classify_rep).
    2. forward_compat_reps.py, one block per domain. Each entry names a published
       embedding model, its weights URL and the number that put it on the list.
       Nothing on that list is in any dataset's pool, so for one of those the
       side-1 without-side is the whole pool.

Usage:
    # Every embedding model in the pool, plus every candidate this domain has:
    python c13_forward_compatibility.py --dataset logd --reps all --mode full

    # A subset of them, pool members and candidates alike:
    python c13_forward_compatibility.py --dataset logd --reps all --mode full \\
        --new-reps chemeleon,clamp

    # Intermediary (quick, 1-fold):
    python c13_forward_compatibility.py --dataset logd --reps all --mode intermediary
"""
import sys
import os
import time as _time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
# Ranking metrics need graded scores: an AUROC read off hard 0/1 labels
# collapses to the mean of sensitivity and specificity (hybrid.py:2661).
from kirby.find_best_hybrid import (_predict_for_metric,
                                    nested_cv_outer_folds,
                                    nested_cv_cache_namespace)
# 0/1 features pass through unscaled, the rest are standardised (2026-10-05,
# handoff 2.3 item 17). Replaces StandardScaler at both of c13's own fits.
from kirby.scaling import FeatureScaler

# 2026-10-05 (handoff 2.4): get_best_search_for_downstream, top_hybrids,
# cv_split, DISTANCE_MODELS and RULE_N_* are no longer imported. Every hybrid
# now comes from a per-fold search (fold_standard_selection) on the dataset's
# decided grid, and the distance hybrid is gone (2.3 item 1).
from experiments._shared import (
    first_outer_folds,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection, fold_searches,
    parse_rep_args,
    _is_tcga_dataset, _is_antibody_dataset)
from experiments.decided_grid import decided_models, decided_aug_conditions


# =============================================================================
# THE FORWARD-COMPATIBILITY CANDIDATES
# =============================================================================
# These are NOT the dataset's pool. The pool — DATASET_REPS['general'], the
# pruned ten — is what a candidate is added TO. The candidates live in
# forward_compat_reps.py, one block per domain, each entry carrying its paper,
# its weights URL and the published number that put it there.
#
# Until 2026-09-10 this file instead carried 108 legacy rep names typed inline and
# called a rep "new" when its name was absent from that list. That list sorted by
# when a name entered the repository, so it split the pool in two rather than
# adding anything to it: on the eight molecular datasets all ten pool members were
# on it, _detect_new_reps returned nothing and c13 exited writing
# {'error': 'no_new_reps'}; on flab_thermo_tresanco eight of ten were called new,
# leaving a two-rep pool to compare against. Both are gone with the list.

from experiments.complexity.forward_compat_reps import (
    forward_compat_rep_names, forward_compat_entry)


# =============================================================================
# THE ONE CONDITION LIST THIS FILE CUTS FOR A LAPTOP RUN
# =============================================================================
# ⛔ NOTHING HERE CUTS A ROW. Every fold still sees all 672 antibodies on
# flab_thermo_tresanco, and the outer fold count is untouched. What is cut is
# how many SUBJECTS c13 scores — how many embedding models get the A/B/C/D/
# D_random ablation run on them.
#
# WHAT IT CONTROLS. `new_reps` has two sources: the dataset's own pool members
# the rule calls learned embedding models, and the candidate list in
# forward_compat_reps.py. This knob decides whether the candidate list is
# loaded at all. The pool subjects are never touched by it.
#
# WHY. Measured 2026-09-25 on flab_thermo_tresanco, the whole dataset, on this
# laptop: from the candidate load starting to the kill was 26 minutes and no
# hybrid search had begun. The antibody block holds 9 candidates. ism and
# igbert came back from cache in seconds. mint, abcdr_esmc and abmpnn raised
# (packages absent, or no builder written). antifold called sys.exit. Those
# five raising forced the batch load to fall back to one load_data call per
# name, and the two that then had to be BUILT dominated everything: p_iggen
# took about 14 minutes to embed 556 unique heavy chains on CPU and antiberta2
# was still building at the 26-minute mark. Adding the five that do arrive
# also takes c13 from 5 subjects to 10, and each candidate subject costs 3
# more live hybrid searches (its C, D and D_random) against the pool
# subjects' 2.
#
# WHAT CUTTING IT COSTS. c13 then reports forward compatibility for the
# dataset's own embedding models only. The question the candidate list exists
# to answer — does a model that is NOT in the pool earn budget when it is
# added — is not answered on this run, and `candidates_requested` in the
# output is 0 rather than 9. Nothing about the five pool subjects changes: the
# pool is the same pool, the without-sides are the same representations and
# the numbers are the ones a full run would print for those five.
#
# ⛔ A COMPUTE CUT FOR A LAPTOP, NOT A FINDING. Nothing run under it says
# anything about whether a new embedding model helps.
#
# Default: the whole candidate block on the server (KIRBY_LOCAL_GRID unset),
# no candidates under the laptop grid. KIRBY_C13_CANDIDATES overrides either
# way: 'all', 'none', or a comma-separated list of candidate names.

def _c13_local_grid():
    """True when the laptop grid in env.sh is active (KIRBY_LOCAL_GRID)."""
    return bool(os.environ.get('KIRBY_LOCAL_GRID'))


def _c13_candidate_policy():
    """('all'|'none'|list-of-names, where the choice came from)."""
    raw = (os.environ.get('KIRBY_C13_CANDIDATES') or '').strip()
    if raw:
        low = raw.lower()
        if low in ('all', 'none'):
            return low, f'KIRBY_C13_CANDIDATES={low}'
        picked = [x.strip() for x in raw.split(',') if x.strip()]
        if picked:
            return picked, f'KIRBY_C13_CANDIDATES={raw}'
    if _c13_local_grid():
        return 'none', 'KIRBY_LOCAL_GRID'
    return 'all', 'shipped'


# How many embedding models get the ablation run on them, at most. Shipped: all
# of them. Under the laptop grid: 1.
#
# WHY 1 AND NOT 2. Measured 2026-09-25 on flab_thermo_tresanco, whole dataset,
# 672 antibodies, 3 folds, on this laptop while 810 other processes were
# queued on its 8 cores. One subject's three hybrid conditions took
# 425 s, 260 s and 396 s, so 18 minutes, and the dataset load plus the
# precompute that runs before c13 took 6 more. A second subject repeats all
# three (its condition-B winner was rf where the first's was svm, so the
# whole-pool search is not shared) and lands the run at about 48 minutes,
# past the 40-minute wall. At 1 the run is about 26 minutes.
#
# WHAT ONE SUBJECT COSTS, counted in live hybrid searches. Side 1 runs three
# conditions per subject: the pool without it (C), the pool with it (D) and the
# pool with a random-Gaussian matrix of its width (D_random). D carries the
# suffix "c13_whole_pool" for every pool subject, so all of them share one
# search; C and D_random are named after the subject and are its own. So the
# 5 embedding models in flab_thermo_tresanco's pool cost 5 + 1 + 5 = 11
# searches, and each additional subject costs 2 more.
# Since 2026-10-06 D_random runs no search (it is D's hybrids with the
# subject's features replaced by Gaussian ones), so each subject costs one
# search fewer than counted above.
#
# WHAT CUTTING IT COSTS. The win rates at the bottom of the summary count
# subjects: "D > D_random" over 2 subjects is 2 numbers, not 5, and the
# per-subject rows for the ones left out are absent from the table and from
# per_rep in the output. Nothing changes for the subjects that do run — same
# pool, same without-sides, same folds, same rows.
#
# ⛔ A COMPUTE CUT FOR A LAPTOP, NOT A FINDING. Which embedding models survive
# the cut is alphabetical order, not a ranking.
def _c13_max_subjects():
    """(limit or None, where the choice came from)."""
    raw = (os.environ.get('KIRBY_C13_MAX_SUBJECTS') or '').strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = -1
        if n > 0:
            return n, f'KIRBY_C13_MAX_SUBJECTS={n}'
        if n == 0:
            return None, 'KIRBY_C13_MAX_SUBJECTS=0'
    if _c13_local_grid():
        return 1, 'KIRBY_LOCAL_GRID'
    return None, 'shipped'


# =============================================================================
# REUSING ONE ROW SET'S FEATURE-IMPORTANCE VECTORS ACROSS c13's CONDITIONS
# =============================================================================
# ⛔ THIS CHANGES NO NUMBER. It hands build_fold_hybrid the vectors it would
# otherwise have computed itself, over the same rows and by the same method.
#
# WHY IT IS SAFE. compute_feature_importance fits ONE model per representation
# and never looks at the others: every branch of it is a
# `for rep_name, X in base_reps.items()` loop (hybrid.py:4683 onwards), so
# saprot's gain vector on a given set of rows is the same vector whether it was
# asked for alongside three representations or four. c13 asks for it many
# times over: on flab_thermo_tresanco the 11 conditions each rebuild the hybrid
# on 3 folds over 4 or 5 representations, which is about 145 importance fits
# over only 30 distinct (rows, representation) pairs.
#
# ⛔ THE ROWS MATTER, AND SO DOES THE CONTENT. create_hybrid carves an internal
# holdout out of the rows it is given and ranks features on the TRAINING side
# of that carve, so the cache computes over exactly that side. And the
# D_random condition swaps a representation's matrix for Gaussian noise of the
# SAME NAME and SAME WIDTH, so a key made of the name alone would hand the real
# representation's importance to the noise arm and destroy the null. The key
# therefore carries a hash of the matrix itself.
#
# Same pattern and same reasoning as c24's KIRBY_C24_REUSE_FOLD_WORK and c16's
# KIRBY_C16_REUSE_FOLD_WORK. Default: ON under the laptop grid, OFF everywhere
# else, so a cluster run recomputes exactly as it did before.
_C13_REUSE_FOLD_WORK = os.environ.get(
    'KIRBY_C13_REUSE_FOLD_WORK',
    '1' if os.environ.get('KIRBY_LOCAL_GRID') else '0') == '1'


def _c13_rep_key(name, method, X):
    """A key that separates a representation from noise wearing its name."""
    import hashlib
    a = np.ascontiguousarray(X)
    return (method, name, a.shape,
            hashlib.md5(a.tobytes()).hexdigest())


def _c13_fold_importance(cache, train_reps, y_train, groups_train, blueprint):
    """Importance vectors for this fold's representations, computed once.

    Returns None when reuse is off, the recipe names no importance method, or
    any representation could not be scored — in which case build_fold_hybrid
    computes its own exactly as before.
    """
    if cache is None or blueprint is None:
        return None
    method = getattr(blueprint, 'importance_method', None)
    if not method:
        return None
    try:
        from kirby.hybrid import (compute_feature_importance,
                                  internal_holdout_split)
        inner_tr, _ = internal_holdout_split(
            len(y_train), y_train, groups_train)
        out, missing = {}, {}
        for name, X in train_reps.items():
            Xi = X[inner_tr]
            key = _c13_rep_key(name, method, Xi)
            if key in cache:
                out[name] = cache[key]
            else:
                missing[name] = (key, Xi)
        if missing:
            fresh = compute_feature_importance(
                {n: v[1] for n, v in missing.items()},
                y_train[inner_tr], method=method)
            for n, (key, _Xi) in missing.items():
                if n in fresh:
                    cache[key] = fresh[n]
                    out[n] = fresh[n]
        return out if len(out) == len(train_reps) else None
    except Exception as e:
        print(f"      importance reuse off for this fold ({type(e).__name__}: "
              f"{e}); build_fold_hybrid computes its own")
        return None


# =============================================================================
# WHICH OF THE DATASET'S OWN REPRESENTATIONS c13 IS ABOUT
# =============================================================================
# DECIDED 2026-09-12, owner. c13's subject is a newly published embedding model —
# not a fingerprint, not a descriptor block, not a hand-built feature set. The
# typed 108-name list is deleted and nothing that sorts by arrival date replaces
# it. What decides now is how the features are produced:
#
#   SUBJECT     — the numbers are read out of a pretrained neural model's
#                 weights (a protein or chemical language model, a pretrained
#                 GNN, a pathology or single-cell foundation model, including a
#                 named slice or layer average of one).
#   ESTABLISHED — everything else: substructure fingerprints, computed
#                 descriptor blocks, graph kernels, hand-built encodings
#                 (one-hot, BLOSUM, amino-acid composition, k-mer counts),
#                 measured omics and clinical quantities, transforms fitted on
#                 this dataset (PCA, NMF, an autoencoder), and model SCORES that
#                 are not an embedding block (a 3-number log-likelihood ratio).
#
# Two domains already mark their own: ANTIBODY_LM_REPS and NUCLEOTIDE_LM_REPS in
# src/kirby/representations/. Those are read here, never edited. The rest of the
# names are grouped below by what builds them.
#
# A name this rule does not know is NOT promoted to subject — that silent
# promotion is what produced "8 of the 10 loaded representations are called new".
# It is counted with the established side and reported by name in
# results.summary.reps_unclassified, so an unknown name is visible, not silent.

# Learned embeddings that src does not already mark. Grouped by what builds them.
_LEARNED_EMBEDDING_REPS = {
    # Molecular — pretrained chemical LMs and pretrained GNN encoders, frozen
    # extraction (the PRETRAINED (Frozen) block of representations/molecular.py).
    'mol2vec', 'chemberta', 'chembert', 'molformer', 'selformer', 'molbert',
    'molclr', 'graphmvp', 'smited', 'mhg_gnn', 'grover', 'unimol_v2',
    'minimol', 'gp_molformer', 'mole', 'clamp', 'rmat',
    # Protein / antibody LMs used outside the antibody loader (dta targets,
    # mutation reps). Every esm2_* name is a pooling or layer choice over the
    # same ESM-2 weights, so each one is a readout of a pretrained model.
    'esm2', 'esm2_cdr', 'esm2_cdr3', 'esm2_concat', 'esm2_early',
    'esm2_entropy_top20', 'esm2_layer24', 'esm2_midlayers', 'esm2_per_cdr',
    'esm2_per_region', 'esm2_mutsite', 'esm2_wt_delta', 'esm2_pocket',
    'esm2_ft', 'esmc', 'esmc_cdr', 'esm3', 'esm3_cdr', 'saprot', 'saprot_cdr',
    'ablang', 'ablang2', 'ablang2_ft', 'antiberty', 'antiberty_cdr',
    'abroberta', 'abroberta_cdr', 'prot_t5', 'prot_t5_cdr', 'protbert',
    'prot_albert', 'prot_xlnet', 'prot_electra', 'igt5', 'amplify',
    'amplify_cdr', 'prosst', 'igbert', 'mint', 'ism', 'abcdr_esmc',
    'antifold', 'p_iggen', 'antiberta2', 'currab',
    # Reaction — pretrained transformers over reaction SMILES.
    'rxnfp', 'reactiont5',
    # TCGA — pretrained pathology slide encoders and bulk/single-cell
    # expression foundation models.
    'wsi_ctranspath', 'wsi_phikon', 'wsi_hoptimus0', 'wsi_hoptimus1',
    'wsi_gigapath', 'wsi_moco', 'wsi_titan', 'wsi_uni2h', 'wsi_virchow2',
    'wsi_conch',
    'gene_geneformer', 'gene_geneformer_v2', 'gene_bulkrnabert', 'gene_scgpt',
}

# Everything else that appears in a dataset's pool, grouped by what builds it.
_ESTABLISHED_REPS = {
    # Substructure fingerprints — an enumerator hashed into bits or counts.
    'ecfp4', 'maccs', 'avalon', 'atom_pair', 'top_torsion', 'rdkfp', 'map4',
    'e3fp', 'drfp', 'drfp_2048', 'cgr_morgan', 'ecfp4_components',
    'maccs_components', 'mol2vec_components', 'mordred_components',
    # Computed descriptor blocks and 3D shape/pharmacophore descriptors.
    'mordred', 'mordred_pca50', 'pdv', 'padel', 'rdkit_desc', 'erg',
    'coulomb_matrix', 'usr', 'usrcat', 'usr_shape', 'pharmacophore3d', 'tda',
    # Graph statistics and kernels, and transforms fitted on this dataset.
    'graph_kernel', 'sali_cliff', 'sns', 'pdv_binary_mean', 'pdv_rank',
    'onehot_pca50',
    # Structure-tuned similarity recipes — hand-built selections of the count,
    # descriptor and shape blocks above.
    'sim_count', 'sim_count_desc', 'sim_count_desc_half', 'sim_count_desc208',
    'sim_count_desc208_half', 'sim_count_shape', 'sim_count_shape_half',
    'sim_all', 'sim_all_half', 'sim_saboteur_drop', 'sim_fine2d',
    # Hand-built sequence encodings and residue-level property tables.
    'onehot', 'onehot_components', 'blosum', 'aac', 'zscale', 'kidera',
    'paratope', 'structural', 'mutation_features', 'contact_graph', 'pocket',
    'kmer',
    # Antibody geometry — measured angles, shapes and contacts off a structure.
    'vhvl_angles', 'cdr_shape', 'cdr_canonical', 'paratope_charge',
    'paratope_hydrophobicity', 'cdr_contacts', 'vhvl_interface',
    'all_geometry',
    # Model SCORES, not embedding blocks: three numbers per antibody.
    'esm2_llr', 'esmc_llr', 'antiberty_llr',
    # Nucleotide counts and computed sequence properties.
    'kmer3', 'kmer4', 'kmer5', 'kmer6', 'gc_content', 'dinuc_freq',
    'tetranuc_freq', 'tnf_canonical', 'dna_shape', 'complexity',
    'codon_usage',
    # TCGA — measured omics, pathway scores, deconvolutions, clinical fields,
    # and PCA/NMF/ICA/autoencoder views fitted on this dataset.
    'gene_hvg2000', 'gene_pca100', 'gene_hallmark', 'gene_progeny', 'gene_kegg',
    'gene_dorothea', 'gene_reactome', 'gene_oncogenic',
    'gene_combined_pathway', 'gene_nmf50', 'gene_ica50', 'gene_sparse_pca50',
    'gene_autoencoder50', 'gene_corr_modules', 'gene_quantile_rank',
    'gene_immune', 'gene_stats', 'gene_chromosome_arm',
    'methyl_pca50', 'methyl_hvg', 'promoter_meth', 'mutation_matrix',
    'mut_tmb', 'mut_driver', 'mut_pathway', 'mut_pca50', 'sig_sbs', 'sig_dbs',
    'sig_id', 'cnv_gistic', 'cnv_pca50', 'mirna_top200', 'rppa', 'clinical',
    'immune_cibersort', 'immune_xcell', 'purity_consensus', 'stemness_mrnasi',
    'heterogeneity_math', 'msi_status', 'hrd_score', 'cimp_status',
    'lncrna_pca50', 'splicing_psi50', 'gsva_hallmark',
}

# The two domain sets src already keeps. Read-only: this file does not own them.
for _src_module, _src_set in (
        ('kirby.representations.antibody', 'ANTIBODY_LM_REPS'),
        ('kirby.representations.nucleotide', 'NUCLEOTIDE_LM_REPS')):
    try:
        _mod = __import__(_src_module, fromlist=[_src_set])
        _LEARNED_EMBEDDING_REPS |= set(getattr(_mod, _src_set))
    except Exception as _e:  # a missing optional dependency must not stop c13
        print(f"  [c13] could not read {_src_set} from {_src_module} ({_e}); "
              f"falling back to the names listed in this file")

# A rep name can carry the side of the pair it was built from, or the per-fold
# transform applied to it. Strip those to reach the model name underneath.
_REP_NAME_PREFIXES = ('drug_', 'protein_', 'reactant_', 'product_', 'diff_',
                      'concat_')
_REP_NAME_SUFFIXES = ('_pca50',)


def _name_forms(rep_name):
    """The name as loaded, then with the pair-side prefix and the per-fold
    transform suffix stripped. The full name is tried first, so methyl_pca50
    (a PCA of measured methylation) is never mistaken for a transform of a
    representation called methyl."""
    forms = [rep_name]
    for suffix in _REP_NAME_SUFFIXES:
        if rep_name.endswith(suffix):
            forms.append(rep_name[:-len(suffix)])
    for prefix in _REP_NAME_PREFIXES:
        if rep_name.startswith(prefix):
            forms.extend(f[len(prefix):] for f in list(forms))
            break
    return forms


def classify_rep(rep_name):
    """Is this representation a learned embedding model's output?

    Returns ('embedding_model' | 'established' | 'unclassified', matched_name).
    matched_name is the form of the name that matched, so a caller can print
    which one did it (protein_esm2 matches on esm2).
    """
    for form in _name_forms(rep_name):
        if form in _LEARNED_EMBEDDING_REPS:
            return 'embedding_model', form
        if form in _ESTABLISHED_REPS:
            return 'established', form
    return 'unclassified', rep_name


def split_pool_by_rule(reps):
    """Split the dataset's loaded representations into subjects and the rest.

    Returns (subject_names, established_names, unclassified_names), each sorted.
    An unclassified name is returned separately and counted with the established
    side by the caller: a name the rule does not know is never promoted to
    subject.
    """
    subjects, established, unknown = [], [], []
    for name in sorted(reps):
        kind, _matched = classify_rep(name)
        if kind == 'embedding_model':
            subjects.append(name)
        elif kind == 'established':
            established.append(name)
        else:
            unknown.append(name)
    return subjects, established, unknown


def load_forward_compat_reps(dataset_name, domain, n_rows, rep_names=None,
                             delta=False, augment=False, pool_labels=None,
                             reasons=None):
    """Load the forward-compatibility candidates for this dataset.

    reasons: a dict filled with {candidate: why it did not come back}, for the
    result file (results audit C13d, 2026-10-05: record unloadable candidates).

    Separate from the pool load, the way augmentation reps get their own
    load_data call in complexity_helpers. Returns (reps, missing) where `missing`
    is the candidate names that did not come back — no builder written, weights
    absent, or the build raised. Every domain loader warns and skips on a name it
    cannot build, so a missing candidate never stops the run.

    n_rows guards the size gate: run() may have subsampled the pool, and a
    candidate matrix that does not match row-for-row cannot be compared to it.
    """
    wanted = list(rep_names) if rep_names else forward_compat_rep_names(domain)
    if not wanted:
        return {}, []

    # Every candidate on this list is a neural embedding model, so building one
    # loads torch into this process. LightGBM then fits inside the hybrid search
    # in the same process, and on macOS those link two different OpenMP runtimes
    # (torch libiomp5, LightGBM libomp). The second one to start its pool kills
    # the process with SIGSEGV and no traceback — exit 139. KIRBY_LGBM_THREADS=1
    # caps LightGBM's own threads and is NOT enough; the trigger is the OpenMP
    # pool size, which only OMP_NUM_THREADS controls. Measured 2026-09-02 and
    # 2026-09-03; f30_scorers.assert_omp_safe guards the importance arms the same
    # way. Raise a readable error rather than let a run die silently.
    if sys.platform == 'darwin' and os.environ.get('OMP_NUM_THREADS') not in ('1', None):
        raise RuntimeError(
            f"c13 builds {len(wanted)} torch-backed candidate representations and "
            f"would SIGSEGV when LightGBM fits after them: "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, needs 1. "
            f"Re-run with OMP_NUM_THREADS=1 KIRBY_LGBM_THREADS=1.")

    print(f"\n  Loading {len(wanted)} forward-compatibility candidates "
          f"for domain '{domain}'...")

    def _load(names):
        return load_data(dataset_name, 'all', rep_names=list(names),
                         delta=delta, augment=augment)

    # Ask for all of them at once, then fall back to one at a time if that
    # raises. Most candidates on this list have no builder yet, and a domain
    # loader can raise on the whole batch when none of the names it was asked to
    # GENERATE worked — get_representations does exactly that ("No
    # representations could be generated!") even when other names came back from
    # cache. Measured 2026-09-10 on caco2: asking for clamp (cached) plus
    # chemeleon and mol_jepa (no builder) raised, and catching that around the
    # batch threw away the clamp that had already loaded. One name per call
    # cannot lose a working candidate to a broken neighbour.
    # ⛔ SystemExit AS WELL AS Exception. A third-party candidate builder can
    # call sys.exit() instead of raising: AntiFold's loader prints "CSV file
    # requires columns 'pdb, Hchain, Lchain'" and exits, which `except
    # Exception` does not catch, so it took the whole c13 run down with exit=1
    # on flab_thermo_tresanco (2026-09-24 02:16). mint and abcdr_esmc, missing
    # in the same way on the same run, raised ImportError and were skipped with
    # a WARNING. A candidate that cannot be built is a candidate that cannot be
    # built, whichever way its library says so.
    fc_data = None
    try:
        fc_data = _load(wanted)
    except (Exception, SystemExit) as e:
        print(f"  Batch load raised ({e}); retrying one candidate at a time so "
              f"a name with no builder cannot take out the ones that work.")
        merged, labels_ref = {}, None
        for name in wanted:
            try:
                one = _load([name])
            except (Exception, SystemExit) as _e:
                print(f"    {name}: cannot be built ({type(_e).__name__}: "
                      f"{_e}); skipped")
                if reasons is not None:
                    reasons[name] = f'{type(_e).__name__}: {_e}'
                continue
            got = one.get('reps', {}) or {}
            if name in got:
                merged[name] = got[name]
                if labels_ref is None:
                    labels_ref = one.get('labels')
        if not merged:
            if reasons is not None:
                for n in wanted:
                    reasons.setdefault(n, 'the domain loader returned no '
                                          'matrix for it')
            return {}, wanted
        fc_data = {'reps': merged, 'labels': labels_ref}

    # Row COUNT matching is not proof the rows are the SAME rows in the SAME
    # order. Both calls go through one loader and both subsamples are
    # deterministic, so they should agree exactly — comparing the labels the two
    # calls returned is the cheap check that says so. A mismatch here means a
    # candidate would be scored against another molecule's label, so stop.
    fc_labels = fc_data.get('labels')
    if pool_labels is not None and fc_labels is not None:
        if len(fc_labels) != len(pool_labels) or not np.allclose(
                np.asarray(fc_labels, dtype=float),
                np.asarray(pool_labels, dtype=float),
                rtol=0, atol=0, equal_nan=True):
            raise RuntimeError(
                f"The candidate load returned different rows than the pool load: "
                f"{len(fc_labels)} labels against the pool's {len(pool_labels)}, "
                f"and they do not match element for element. Every candidate would "
                f"be scored against the wrong molecule's label, so nothing is run.")

    reps = fc_data.get('reps', {}) or {}
    # Drop anything that is not row-aligned with the pool.
    aligned = {}
    for name, X in reps.items():
        if X.shape[0] != n_rows:
            print(f"  WARNING: candidate '{name}' has {X.shape[0]} rows, pool has "
                  f"{n_rows} — dropped (rerun without the size gate to include it)")
            if reasons is not None:
                reasons[name] = (f'{X.shape[0]} rows against the pool\'s '
                                 f'{n_rows}')
            continue
        aligned[name] = X

    missing = [n for n in wanted if n not in aligned]
    if reasons is not None:
        for n in missing:
            reasons.setdefault(n, 'the domain loader returned no matrix for it '
                                  '(no builder written, or weights absent)')
    if aligned:
        print(f"  Loaded {len(aligned)}: "
              f"{', '.join(f'{k}({v.shape[1]}D)' for k, v in sorted(aligned.items()))}")
    if missing:
        print(f"  Not loaded ({len(missing)}): {', '.join(sorted(missing))}")
        print(f"    (no builder written, weights absent, or the build raised — "
              f"see forward_compat_reps.py for which)")
    return aligned, missing


def _whole_group_draw(groups, n_rows, k, seed=42):
    """Row indices of up to k rows made of whole groups, and a record.

    DECIDED 2026-10-06 (Adelaide delegated): above 5,000 rows c13 draws its
    2,000 rows by whole scaffold groups. Groups are taken in a fixed random
    order (RandomState(seed)); a group is added when it fits within k rows and
    skipped when it would overshoot, until k rows are reached or the groups
    run out. Without groups it raises: a random row draw is not allowed.
    """
    if groups is None:
        raise ValueError(
            f"c13: {n_rows} rows needs a draw of {k} rows by whole groups, "
            f"and this dataset has no groups array.")
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.random.RandomState(seed).permutation(len(uniq))
    keep, total = [], 0
    for gi in order:
        if total + sizes[gi] <= k:
            keep.append(gi)
            total += int(sizes[gi])
            if total == k:
                break
    idx = np.sort(np.flatnonzero(np.isin(inv, keep)))
    return idx, {'rule': ('whole groups in RandomState(%d) order, each added '
                          'if it fits within %d rows' % (seed, k)),
                 'n_rows_asked': int(k), 'n_rows_drawn': int(len(idx)),
                 'n_groups_drawn': int(len(keep)),
                 'n_groups_total': int(len(uniq))}


def _subsample(reps, idx):
    """Take the same rows out of every rep matrix."""
    return {k: v[idx] for k, v in reps.items()}


# =============================================================================
# PER-FOLD EVALUATION (aligned 2026-10-05, handoff 2.4)
# =============================================================================
# Every condition is scored the same way, fold by fold. The outer folds come
# from kirby.find_best_hybrid.nested_cv_outer_folds, and their count from
# config['n_splits'] alone. For each outer fold, one search runs on that fold's
# TRAINING rows only, over the condition's representations, on the dataset's
# decided grid with pruning off (fold_standard_selection). Per decided model
# it names the 2 best single representations and the 2 best hybrids; a second
# hybrid whose selected features equal the first's is skipped for the next.
# Each is fitted on the fold's training rows and scored on its held-out rows.
#
# Until 2026-10-05 each condition ran ONE search over every row c13 held
# (speed='fast', one model: condition B's winner) and only rebuilt that
# recipe per fold, so the held-out rows had helped choose the allocation
# method, budget, importance method, evaluator and representation count. The
# rebuild also ranked representations by scores measured over every row
# (dirty-run bug 1). A third hybrid came from a distance search; c13 measures
# no neighbours or distances, so that slot is gone (handoff 2.3 item 1).

SEED = 42
# Stored in the checkpoint identity and the result file, so a partial file
# written by the whole-row version of c13 is never read back as this one's.
# 2026-10-06: hybrid rows carry selected indices; D_random is forced from D;
# rows above 5,000 are drawn by whole groups.
SELECTION = 'fold_standard_selection_2026_10_06'
ONE_FOLD_NOTE = 'one fold, no spread'
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count')


def _fit_score(X_train, X_val, y_train, y_val, factory, metric_fn):
    """Fit one model on a fold's training rows; its metric on the held-out rows.

    FeatureScaler (handoff 2.3 item 17, 2026-10-05): each 0/1 feature passes
    through unscaled, every other feature is standardised, fitted on the
    training rows only. The features are not converted to float64 (item 18).
    Returns None when the fit raised, and prints the error.
    """
    try:
        scaler = FeatureScaler()
        Xtr = scaler.fit_transform(np.asarray(X_train))
        Xva = scaler.transform(np.asarray(X_val))
        model = factory()
        model.fit(Xtr, y_train)
        return float(metric_fn(y_val,
                               _predict_for_metric(model, Xva, metric_fn)))
    except Exception as e:
        print(f"      [fold fit failed: {type(e).__name__}: {e}]")
        return None


def _finite(x):
    return x is not None and np.isfinite(x)


def _spread(scores):
    """Mean and spread over the folds that produced a score (None = not
    scored). The spread is the standard deviation over folds; with one scored
    fold it is None and 'spread_note' says so."""
    ok = [float(s) for s in scores if _finite(s)]
    out = {'fold_scores': [float(s) if _finite(s) else None for s in scores],
           'score_mean': float(np.mean(ok)) if ok else float('nan'),
           'score_spread': float(np.std(ok)) if len(ok) > 1 else None,
           'n_folds_scored': len(ok)}
    if len(ok) == 1:
        out['spread_note'] = ONE_FOLD_NOTE
    return out


def _eval_single_rep_on_folds(X, labels, folds, factory, metric_fn):
    """One representation, one model: the held-out score of every outer fold
    (None where the fit failed). The same folds the hybrids are scored on."""
    return [_fit_score(X[tr], X[va], labels[tr], labels[va], factory,
                       metric_fn)
            for tr, va in folds]


def _search_failures(reps, labels, groups, tr, k, n_outer, dataset_name,
                     augmentations, task_type, metric_name, suffix):
    """The recorded failures of the search fold_standard_selection just ran.

    Reads the in-process memo with the arguments fold_standard_selection
    passes, so nothing is searched again.
    """
    try:
        srch = fold_searches(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            augmentations=augmentations, task_type=task_type,
            metric=metric_name, seed=SEED, condition_suffix=suffix)
        return [{kk: (vv if isinstance(vv, (int, float, str, type(None)))
                      else str(vv)) for kk, vv in f.items()}
                for f in (getattr(srch, 'failures', None) or [])]
    except Exception as e:  # reading the memo must not stop the fold
        return [{'error': f'could not read the failure record: '
                          f'{type(e).__name__}: {e}'}]


def _eval_condition_per_fold(reps, labels, groups, folds, factories, *,
                             dataset_name, metric, task_type, suffix,
                             augmentations=None, subject=None,
                             importance_cache=None, ck=None,
                             n_outer_folds=None):
    """One condition (one set of representations), scored per outer fold.

    reps: the condition's representations. Two or more: per outer fold, the
        standard selection (fold_standard_selection) on that fold's training
        rows; each of the 2 base representations and 2 hybrids per decided
        model is fitted on the training rows and scored on the held-out rows.
        One: that representation alone under each decided model, no search
        (find_best_hybrid needs two); 'is_hybrid' is then False.
    suffix: the condition_suffix. It keeps this condition's per-fold searches
        apart from every other condition's in the store and in the in-process
        memo: the random-Gaussian null carries the subject's NAME and WIDTH,
        and representation content is not part of the stored-result key.
    subject: the representation this condition adds (the embedding model, or
        the Gaussian matrix wearing its name). Used for the printed lines only;
        each hybrid row records the features of EVERY representation
        (features_by_rep), since one condition can serve several subjects.
    ck: the checkpoint. Each finished fold is stored under (condition, fold)
        and read back on the next attempt.

    Returns {'condition_suffix', 'reps_searched', 'is_hybrid', 'per_fold',
    'by_model': {model: {slot: _spread(...)}}, 'search_failures',
    'seconds'}. Slots are base1, base2, hybrid1, hybrid2, or 'single'.
    """
    metric_name, metric_fn, lower_is_better = metric
    n_outer = n_outer_folds or len(folds)     # the layout's count, not the folds run
    is_hybrid = len(reps) >= 2
    t0 = _time.time()
    per_fold = []
    for k, (tr, va) in enumerate(folds):
        ck_key = dict(condition=suffix, fold=int(k))
        if ck is not None and ck.done(**ck_key):
            per_fold.append(ck.get(**ck_key))
            print(f"      fold {k}: finished in an earlier attempt, read back")
            continue
        y_tr, y_va = labels[tr], labels[va]
        tr_reps = {r: X[tr] for r, X in reps.items()}
        va_reps = {r: X[va] for r, X in reps.items()}
        g_tr = groups[tr] if groups is not None else None
        fold_rec = {
            'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
            'cache_namespace': nested_cv_cache_namespace(k, n_outer, SEED),
            'condition_suffix': suffix,
            'representations': sorted(reps),
            'selection_scope': 'fold_train_rows',
            'metric': metric_name,
            'lower_is_better': bool(lower_is_better),
            'search_failures': [], 'shortfall': [], 'models': {}}

        if not is_hybrid:
            (rep,) = list(reps)
            for m, factory in factories.items():
                s = _fit_score(tr_reps[rep], va_reps[rep], y_tr, y_va,
                               factory, metric_fn)
                fold_rec['models'][m] = {
                    'single': {'slot': 'single', 'rep': rep,
                               'heldout_score': s}}
        else:
            sel = fold_standard_selection(
                reps, labels, groups, train_idx=tr, fold_i=k,
                n_outer_folds=n_outer, dataset_name=dataset_name,
                models=list(factories), augmentations=augmentations,
                task_type=task_type, metric=metric_name, seed=SEED,
                condition_suffix=suffix)
            failed = sorted(m for m, s in sel.items() if s is None)
            if failed:
                errs = _search_failures(
                    reps, labels, groups, tr, k, n_outer, dataset_name,
                    augmentations, task_type, metric_name, suffix)
                for m in failed:
                    fold_rec['search_failures'].append({
                        'dataset': dataset_name, 'condition': suffix,
                        'fold': int(k), 'model': m, 'errors': errs})
                    print(f"      fold {k} {m}: SEARCH FAILED, slots skipped: "
                          f"{[e.get('error') or e.get('error_type') for e in errs]}")
            for m, s in sel.items():
                if s is None:
                    continue
                factory = factories[m]
                bd = s.get('base_detail') or {}
                n_base = len(bd.get('selected') or [])
                if n_base < 2 or len(s['hybrids']) < 2:
                    fold_rec['shortfall'].append({
                        'model': m, 'n_base': n_base,
                        'n_hybrids': len(s['hybrids']),
                        'reason': bd.get('shortfall_reason')})
                rows = {}
                for i, e in enumerate(bd.get('selected') or [], 1):
                    rep = e['rep']
                    rows[f'base{i}'] = {
                        'slot': f'base{i}', 'rep': rep,
                        # Mean over the search's inner folds, inside this
                        # outer fold's training rows.
                        'training_rows_score': e.get('score'),
                        'heldout_score': _fit_score(
                            tr_reps[rep], va_reps[rep], y_tr, y_va, factory,
                            metric_fn)}
                for i, e in enumerate(s['hybrids'], 1):
                    bp = e['blueprint']
                    aug_cond = e.get('aug_condition') or 'none'
                    row = {'slot': f'hybrid{i}',
                           **{f: e.get(f) for f in HYBRID_FIELDS},
                           'aug_condition': aug_cond,
                           'training_rows_score': e.get('score_mean')}
                    use_augs = (augmentations if aug_cond != 'none'
                                else None)
                    try:
                        X_tr_h, X_va_h, fi = build_fold_hybrid(
                            tr_reps, va_reps, y_tr, bp,
                            augmentations_train=(
                                {a: X[tr] for a, X in use_augs.items()}
                                if use_augs else None),
                            augmentations_val=(
                                {a: X[va] for a, X in use_augs.items()}
                                if use_augs else None),
                            groups=g_tr,
                            precomputed_importance=_c13_fold_importance(
                                importance_cache, tr_reps, y_tr, g_tr, bp),
                            # This fold's search measured these on this
                            # fold's training rows (dirty-run bug 1).
                            precomputed_alloc_scores=getattr(
                                bp, 'rep_scores', None),
                            **fold_hybrid_scoring_kwargs(
                                bp, model=factory(), metric_fn=metric_fn,
                                lower_is_better=lower_is_better))
                    except Exception as ex:
                        row['error'] = f'{type(ex).__name__}: {ex}'
                        row['heldout_score'] = None
                        print(f"      fold {k} {m} hybrid{i} build failed: "
                              f"{row['error']}")
                        rows[f'hybrid{i}'] = row
                        continue
                    row['features_by_rep'] = {
                        r: int(len(info['selected_indices']))
                        for r, info in fi.items()
                        if isinstance(info, dict)
                        and 'selected_indices' in info}
                    row['n_features'] = int(np.asarray(X_tr_h).shape[1])
                    # The selected feature indices per representation, in the
                    # hybrid's column order. The forced random-feature control
                    # (_forced_null_condition, 2026-10-06) rebuilds this
                    # hybrid from them with the subject's features swapped
                    # for Gaussian ones.
                    row['selected_indices_by_rep'] = {
                        r: [int(x) for x in np.asarray(
                            info['selected_indices']).ravel()]
                        for r, info in fi.items()
                        if isinstance(info, dict)
                        and 'selected_indices' in info}
                    row['has_added_features'] = bool(
                        fi.get('augmentation_info'))
                    # Per representation, not per subject: one condition's
                    # folds are shared by every subject whose rep set it is
                    # (the memo in run()), so nothing here may depend on which
                    # subject asked first.
                    row['modality_novelty_by_rep'] = {
                        r: _modality_novelty(
                            fi, r, tr_reps[r],
                            {o: X for o, X in tr_reps.items() if o != r})
                        for r, c in row['features_by_rep'].items()
                        if c > 0 and r in tr_reps}
                    row['heldout_score'] = _fit_score(
                        X_tr_h, X_va_h, y_tr, y_va, factory, metric_fn)
                    rows[f'hybrid{i}'] = row
                rows['hybrids_skipped_identical_features'] = [
                    x.get('label') for x in
                    (s.get('hybrids_skipped_identical_features') or [])]
                fold_rec['models'][m] = rows
        for m, rows in fold_rec['models'].items():
            for slot in ('single', 'base1', 'base2', 'hybrid1', 'hybrid2'):
                r = rows.get(slot)
                if r is None:
                    continue
                v = r.get('heldout_score')
                extra = (f", {r['features_by_rep'].get(subject, 0)} features "
                         f"from {subject}"
                         if subject is not None and 'features_by_rep' in r
                         else '')
                print(f"      fold {k} {m:8s} {slot:8s} "
                      f"{(r.get('rep') or r.get('label') or ''):28.28s} "
                      f"{metric_name}="
                      f"{('%.4f' % v) if _finite(v) else 'not scored'}"
                      f"{extra}")
        per_fold.append(fold_rec)
        # A fold whose search failed is in the result file but not stored, so
        # the next attempt searches it again.
        if ck is not None and not fold_rec['search_failures']:
            ck.put(convert_numpy(fold_rec), **ck_key)

    slots = ('single',) if not is_hybrid else ('base1', 'base2', 'hybrid1',
                                               'hybrid2')
    by_model = {}
    for m in factories:
        by_model[m] = {}
        for slot in slots:
            by_model[m][slot] = _spread([
                ((fr.get('models') or {}).get(m) or {}).get(slot, {})
                .get('heldout_score') for fr in per_fold])
    return {
        'condition_suffix': suffix,
        'reps_searched': sorted(reps),
        'is_hybrid': is_hybrid,
        'selection_scope': 'fold_train_rows',
        'per_fold': per_fold,
        'by_model': by_model,
        'search_failures': [f for fr in per_fold
                            for f in fr.get('search_failures', [])],
        'seconds': float(_time.time() - t0),
    }


def _slot_scores(cond, model, slot, n_folds):
    """Held-out score per fold for one model and slot of a condition. A
    condition with one representation has only 'single', which then stands in
    for every hybrid slot ('is_hybrid' False on the condition says so)."""
    if not cond:
        return [None] * n_folds
    if not cond.get('is_hybrid'):
        slot = 'single'
    ent = ((cond.get('by_model') or {}).get(model) or {}).get(slot)
    return list(ent['fold_scores']) if ent else [None] * n_folds


def _paired_delta(a, b, sign):
    """sign * (a - b) fold by fold, on the folds where both scored. Positive
    is an improvement in any metric direction."""
    d = [sign * (x - y) if (_finite(x) and _finite(y)) else None
         for x, y in zip(a, b)]
    ok = [v for v in d if v is not None]
    out = {'fold_deltas': d,
           'mean': float(np.mean(ok)) if ok else float('nan'),
           'spread': float(np.std(ok)) if len(ok) > 1 else None,
           'n_folds': len(ok)}
    if len(ok) == 1:
        out['spread_note'] = ONE_FOLD_NOTE
    return out


def _forced_null_condition(d_cond, with_reps, subject, X_null, labels, folds,
                           factories, *, metric, suffix):
    """The random-feature control, forced to the subject's feature share.

    DECIDED 2026-10-06 (Adelaide delegated): the random-Gaussian control gets
    the same number of features the embedding model got, and that number is
    recorded. Until then the control had its own per-fold search, and that
    search gave the Gaussian 0 features on every fold of the 2026-09-29 run,
    so D_random was the without-side hybrid and D - D_random equalled D - C.

    Per outer fold, decided model and hybrid slot: take D's hybrid (the
    without-side plus the subject), keep every other representation's selected
    features, and replace the subject's selected features with the Gaussian
    matrix's features at the same indices (the Gaussian has the subject's
    width and its features are independent draws, so which indices are taken
    does not matter). Fit on the fold's training rows, score the held-out rows.
    So D and D_random differ in one thing only: the content of the subject's
    features. No search runs here.

    with_reps: D's representations (the without-side plus the subject).
    Returns a condition dict shaped like _eval_condition_per_fold's, with the
    hybrid slots only.
    """
    from kirby.hybrid import apply_feature_selection
    metric_name, metric_fn, lower_is_better = metric
    src = {**with_reps, subject: X_null}
    slots = ('hybrid1', 'hybrid2')
    per_fold = []
    for k, (tr, va) in enumerate(folds):
        d_fold = next((fr for fr in (d_cond or {}).get('per_fold') or []
                       if fr.get('fold') == k), None)
        fold_rec = {'fold': int(k), 'condition_suffix': suffix,
                    'forced_from': (d_cond or {}).get('condition_suffix'),
                    'selection_scope': 'fold_train_rows',
                    'metric': metric_name, 'models': {}}
        for m, factory in factories.items():
            d_rows = ((d_fold or {}).get('models') or {}).get(m) or {}
            rows = {}
            for slot in slots:
                d_row = d_rows.get(slot)
                row = {'slot': slot,
                       'label': (d_row or {}).get('label'),
                       'forced_feature_share': True,
                       'heldout_score': None}
                sel = (d_row or {}).get('selected_indices_by_rep')
                if d_row is None or d_row.get('error'):
                    row['not_run'] = ("D has no built hybrid in this slot: "
                                      f"{(d_row or {}).get('error')}")
                elif not sel:
                    row['not_run'] = 'D row carries no selected feature indices'
                elif d_row.get('has_added_features'):
                    row['not_run'] = ("D's hybrid carries added features; "
                                      "the forced control rebuilds the base "
                                      "features only")
                if 'not_run' in row:
                    rows[slot] = row
                    continue
                n_sub = len(sel.get(subject) or [])
                n_tot = sum(len(v) for v in sel.values())
                fi = {r: {'selected_indices': np.asarray(idx, dtype=int)}
                      for r, idx in sel.items()}
                try:
                    X_tr = apply_feature_selection(
                        {r: src[r][tr] for r in sel}, fi, base_only=True)
                    X_va = apply_feature_selection(
                        {r: src[r][va] for r in sel}, fi, base_only=True)
                except Exception as ex:
                    row['error'] = f'{type(ex).__name__}: {ex}'
                    rows[slot] = row
                    continue
                row.update({
                    'features_by_rep': {r: len(v) for r, v in sel.items()},
                    'gaussian_features': int(n_sub),
                    'subject_features_in_D': int(n_sub),
                    'gaussian_feature_share': (float(n_sub) / n_tot
                                               if n_tot else None),
                    'n_features': int(n_tot),
                    'heldout_score': _fit_score(X_tr, X_va, labels[tr],
                                                labels[va], factory,
                                                metric_fn)})
                rows[slot] = row
                v = row['heldout_score']
                print(f"      fold {k} {m:8s} {slot:8s} forced null: "
                      f"{n_sub} of {n_tot} features Gaussian, {metric_name}="
                      f"{('%.4f' % v) if _finite(v) else 'not scored'}")
            fold_rec['models'][m] = rows
        per_fold.append(fold_rec)
    by_model = {m: {slot: _spread([
        ((fr['models'].get(m) or {}).get(slot) or {}).get('heldout_score')
        for fr in per_fold]) for slot in slots} for m in factories}
    return {'condition_suffix': suffix,
            'reps_searched': [],
            'forced_from': (d_cond or {}).get('condition_suffix'),
            'is_hybrid': True,
            'selection_scope': 'fold_train_rows',
            'per_fold': per_fold, 'by_model': by_model,
            'search_failures': []}


# =============================================================================
# CONTROL CONDITIONS
# =============================================================================

def _make_random_gaussian(X, seed=42):
    """Random Gaussian matrix with same shape as X — permutation null."""
    rng = np.random.RandomState(seed)
    return rng.randn(*X.shape).astype(np.float32)


def _modality_novelty(feature_info, rep_name, X_new, other_reps):
    """Cosine distance from the new representation's selected features to the
    nearest other representation's selected features, inside the hybrid that
    selected them. NaN when that hybrid selected nothing from the new one.

    Lifted out of the condition-D block unchanged on 2026-09-12 so both
    without-sides can use it; other_reps is the side's own set.
    """
    try:
        from scipy.spatial.distance import cosine as _cosine_dist
        if not feature_info or rep_name not in feature_info:
            return float('nan')
        new_selected = feature_info[rep_name].get('selected_indices', [])
        if not new_selected:
            return float('nan')
        new_vec = X_new[:, new_selected].mean(axis=0)
        min_dist = float('inf')
        for ename, edata in feature_info.items():
            if ename == rep_name:
                continue
            e_sel = edata.get('selected_indices', [])
            if e_sel and ename in other_reps:
                e_vec = other_reps[ename][:, e_sel].mean(axis=0)
                # Pad to same length for cosine
                maxlen = max(len(new_vec), len(e_vec))
                nv = np.zeros(maxlen); nv[:len(new_vec)] = new_vec
                ev = np.zeros(maxlen); ev[:len(e_vec)] = e_vec
                d = _cosine_dist(nv, ev)
                if d < min_dist:
                    min_dist = d
        return float(min_dist) if min_dist < float('inf') else float('nan')
    except Exception:
        return float('nan')


def _make_row_shuffled(X, seed=42):
    """Row-shuffled version of X — tests content vs dimensionality."""
    rng = np.random.RandomState(seed)
    X_shuffled = X.copy()
    for col in range(X_shuffled.shape[1]):
        rng.shuffle(X_shuffled[:, col])
    return X_shuffled

# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run(data, config, output_dir, dataset_name, new_rep_names=None, **kwargs):
    """C13: Forward Compatibility — the two-sided ablation per new rep.

    Args:
        data: Loaded dataset dict from load_data().
        config: Config dict (fast or full). Only 'n_splits' is read: the number
            of outer folds. 1 is a single held-out split.
        output_dir: Output directory for results JSON (and the per-fold
            checkpoint beside it).
        dataset_name: Name of the dataset (the real name; it looks up the
            decided grid).
        new_rep_names: Candidate rep names to load and score. If None, the whole
            forward-compatibility block for this dataset's domain is used.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C13: FORWARD COMPATIBILITY")
    print("=" * 70)

    # 2026-10-05: the KIRBY_SMOKE_N block that shrank _shared.FULL_BUDGETS to
    # [25] here is gone. The per-fold searches take the dataset's decided
    # budgets and never read FULL_BUDGETS, and a "does it run" check runs the
    # normal settings.

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    augs = data.get('augmentations')
    task_type = data['task_type']

    # Size-gate (2026-06-28): above the cutoff, subsample (reps/labels/groups/
    # augs jointly) to K rows. Fixed seed = stable.
    # DECIDED 2026-10-06 (Adelaide delegated): the rows are drawn by whole
    # scaffold groups, not at random (_whole_group_draw).
    from experiments._shared import LARGE_DATASET_N, LARGE_SUBSAMPLE_K
    _n_full = len(labels)
    _idx = None
    _draw_record = None
    if len(labels) > LARGE_DATASET_N:
        _idx, _draw_record = _whole_group_draw(groups, len(labels),
                                               LARGE_SUBSAMPLE_K, seed=42)
        reps = _subsample(reps, _idx)
        labels = labels[_idx]
        groups = groups[_idx] if groups is not None else None
        if augs is not None:
            augs = _subsample(augs, _idx)
        print(f"  [size-gate] n>{LARGE_DATASET_N}: drew {len(_idx)} rows "
              f"({_draw_record['n_groups_drawn']} whole groups) for c13")

    # Outer folds: one setting, config['n_splits'] (handoff 2.4 step 14).
    n_folds = int(config['n_splits'])

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    sign = -1 if lower_is_better else 1

    # Models (R1, 2026-10-05): the dataset's two decided models, each scored
    # on every condition. Until then c13 fitted rf for condition A and both
    # controls, picked condition B's winner out of five models on the same
    # folds it then reported, and searched and scored every hybrid for that
    # one winner only.
    model_names = [m for m in decided_models(dataset_name, 'general')
                   if m != 'knn']
    models = get_model_factories(model_names, task_type=task_type,
                                 n_samples=len(labels))
    _no_factory = [m for m in model_names if m not in models]
    if _no_factory:
        print(f"  Decided models with no factory at n={len(labels)}, not "
              f"scored: {_no_factory}")
    if not models:
        return {'error': 'no_decided_model_available',
                'decided_models': model_names}

    # Augmentation (2.3 item 4, Step 8): only the decided conditions of the
    # general search. c13 compares no augmentation condition; the per-fold
    # search runs whichever conditions decided_aug_conditions keeps ('none'
    # on every dataset's general search).
    aug_conditions = decided_aug_conditions(dataset_name, 'general')
    print(f"  Augmentation conditions the general search runs here: "
          f"{aug_conditions}")

    # ── The pool, the rule, and the candidates ────────────────────────────
    # existing_reps is the dataset's whole pool: DATASET_REPS[ds]['general']
    # only (2.4 step 9). A run started with --reps naming others drops them
    # here, by name. It is the side-1 without-side (minus the subject).
    from experiments.complexity.complexity_helpers import DATASET_REPS
    _general = (DATASET_REPS.get(dataset_name) or {}).get('general')
    if _general is not None:
        _off_list = sorted(set(reps) - set(_general))
        if _off_list:
            print(f"  Not in this dataset's pruned representation list, left "
                  f"out of the pool ({len(_off_list)}): {', '.join(_off_list)}")
        existing_reps = {k: v for k, v in reps.items() if k in _general}
    else:
        existing_reps = dict(reps)

    # One outer split for every condition, single representation and control
    # below, from the same function the per-fold store is keyed on.
    folds = nested_cv_outer_folds(existing_reps or reps, labels, groups,
                                  n_folds, SEED, task_type)
    n_folds = len(folds)
    # n_folds stays the layout's count (search names, namespaces, the run
    # identity); the folds run may be the first few of it (OUTER_FOLDS_RUN,
    # 2026-10-07).
    folds = first_outer_folds(folds, 'c13')
    n_folds_run = len(folds)

    # DECIDED 2026-09-12: which of the dataset's own representations is a
    # newly published embedding model is decided by how its features are
    # produced, not by when the name arrived. Subjects are the pool members
    # whose numbers come out of a pretrained model's weights; the established
    # names are the rest and are the side-2 without-side.
    subject_names, established_names, unclassified_names = split_pool_by_rule(
        existing_reps)
    print(f"\n  Learned embedding models in this dataset's {len(existing_reps)} "
          f"representations ({len(subject_names)}): "
          f"{', '.join(subject_names) if subject_names else 'none'}")
    print(f"  Established representations ({len(established_names)}): "
          f"{', '.join(established_names) if established_names else 'none'}")
    if unclassified_names:
        print(f"  Not named by the rule ({len(unclassified_names)}): "
              f"{', '.join(unclassified_names)}")
        print("    These are counted with the established side, NOT as subjects. "
              "Add them to _LEARNED_EMBEDDING_REPS or _ESTABLISHED_REPS in this "
              "file to say which they are.")

    # --new-reps names which subjects to run. A named pool member is a pool
    # subject; anything else is looked for in forward_compat_reps.py. Before
    # 2026-09-12 naming a pool member here did nothing.
    requested = list(new_rep_names) if new_rep_names else None
    pool_subjects = {n: existing_reps[n] for n in subject_names}
    if requested is not None:
        pool_subjects = {n: X for n, X in pool_subjects.items() if n in requested}
        candidate_names = [n for n in requested if n not in existing_reps]
    else:
        candidate_names = None  # the whole block for this domain

    # Which candidates this run loads, and why (see _c13_candidate_policy).
    # --new-reps naming a candidate always wins: an explicit request is not a
    # default and is never overridden by the knob.
    _cand_policy, _cand_why = _c13_candidate_policy()
    if candidate_names is None and isinstance(_cand_policy, list):
        candidate_names = list(_cand_policy)

    # Candidates load at the dataset's FULL row count, so the size-gate index has
    # to be applied to them too or every one of them lands row-misaligned.
    missing_reasons = {}
    if requested is not None and not candidate_names:
        candidate_reps, missing_candidates = {}, []
    elif _cand_policy == 'none' and requested is None:
        candidate_reps, missing_candidates = {}, []
        print(f"\n  Candidate embedding models: none loaded ({_cand_why}). "
              f"Only this dataset's own {len(subject_names)} learned embedding "
              f"model(s) are scored. Set KIRBY_C13_CANDIDATES=all for the "
              f"whole '{domain}' block.")
    else:
        candidate_reps, missing_candidates = load_forward_compat_reps(
            dataset_name, domain, n_rows=_n_full,
            rep_names=candidate_names,
            delta=data.get('_delta', False), augment=data.get('_augment', False),
            pool_labels=data['labels'], reasons=missing_reasons)
        if _idx is not None and candidate_reps:
            candidate_reps = _subsample(candidate_reps, _idx)

    # A candidate name that is also a pool name is already a subject; the pool's
    # own matrix is the one every without-side was built from, so it wins.
    _both = sorted(set(candidate_reps) & set(pool_subjects))
    if _both:
        print(f"  In the pool AND on the candidate list ({len(_both)}): "
              f"{', '.join(_both)} — the pool's matrix is used.")
    new_reps = dict(pool_subjects)
    new_reps.update({k: v for k, v in candidate_reps.items()
                     if k not in new_reps})

    # How many of them actually get the ablation run (see _c13_max_subjects).
    _max_subjects, _max_why = _c13_max_subjects()
    if _max_subjects is not None and len(new_reps) > _max_subjects:
        _kept = sorted(new_reps)[:_max_subjects]
        _dropped_subjects = sorted(set(new_reps) - set(_kept))
        new_reps = {k: new_reps[k] for k in _kept}
        print(f"\n  Embedding models scored: {len(_kept)} of "
              f"{len(_kept) + len(_dropped_subjects)} ({_max_why}). "
              f"Scored: {', '.join(_kept)}. Not scored: "
              f"{', '.join(_dropped_subjects)}. The pool itself is unchanged — "
              f"every one of them is still on the without-side of the ones that "
              f"do run.")
    else:
        _dropped_subjects = []

    if not new_reps:
        print("\n  No embedding model to test on this dataset.")
        print(f"  The rule found {len(subject_names)} learned embedding model(s) "
              f"among the {len(existing_reps)} loaded representations, and "
              f"{len(missing_candidates)} candidate(s) were asked for with none "
              f"coming back.")
        print("  A candidate needs a builder in the domain's rep registry before "
              "it can be scored — see "
              "tests/experiments/complexity/forward_compat_reps.py for which are "
              "written and which are not.")
        return {
            'error': 'no_embedding_models',
            'domain': domain,
            'pool_reps': sorted(existing_reps.keys()),
            'pool_subject_reps': subject_names,
            'pool_established_reps': established_names,
            'reps_unclassified': unclassified_names,
            'candidates_requested': sorted(missing_candidates),
            'candidates_not_loaded_reasons': missing_reasons,
        }

    if len(existing_reps) < 2:
        print(f"\n  Pool has {len(existing_reps)} rep(s); the hybrid search needs "
              f"at least 2, so conditions C, D and D_random cannot run.")
        return {
            'error': 'pool_too_small',
            'domain': domain,
            'pool_reps': sorted(existing_reps.keys()),
        }

    # Side 2's without-side: the established representations, every other
    # embedding model held out. The same set for every subject on this dataset.
    established_reps = {n: existing_reps[n]
                        for n in established_names + unclassified_names}

    print(f"  Domain:       {domain}")
    print(f"  Metric:       {metric_name} ({'lower is better' if lower_is_better else 'higher is better'})")
    print(f"  n_samples:    {len(labels)}")
    print(f"  Outer folds:  {n_folds_run} run of a {n_folds}-fold layout"
          f"{' (a single held-out split)' if n_folds == 1 else ''}")
    print(f"  Models:       {list(models.keys())} (the dataset's decided models)")
    if missing_candidates:
        print(f"  Candidates not loaded: {len(missing_candidates)} "
              f"(named above; they need a builder before they can be scored)")
    print(f"\n  Embedding models to test ({len(new_reps)}): "
          f"{', '.join(f'{k}({v.shape[1]}D)' for k, v in sorted(new_reps.items()))}")
    print(f"  Side-1 without-side, the whole pool minus the one being tested "
          f"({len(existing_reps)}): "
          f"{', '.join(f'{k}({v.shape[1]}D)' for k, v in sorted(existing_reps.items()))}")
    print(f"  Side-2 without-side, the established representations "
          f"({len(established_reps)}): "
          f"{', '.join(f'{k}({v.shape[1]}D)' for k, v in sorted(established_reps.items())) or 'none'}")
    if len(established_reps) < 2:
        # Results audit C13c (owner, 2026-10-05: keep). With one established
        # representation, C_established is that representation alone and
        # D_established is a hybrid, so D_established - C_established changes
        # two things at once: one representation added, and a single
        # representation replaced by a hybrid.
        print(f"    Fewer than 2, so the side-2 without-side is scored as a "
              f"single representation, not as a hybrid. D_established - "
              f"C_established then changes two things at once.")

    # Per-fold checkpoint (2026-10-05). Every condition's finished folds are
    # stored as they finish, under (condition, fold). The identity carries the
    # selection, so partial files of the whole-row version are never read.
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity)
    ck = None
    if output_dir:
        ck = Checkpoint(
            experiment_sidecar(output_dir, 'c13_forward_compatibility',
                               dataset_name),
            run_identity('c13_forward_compatibility', dataset_name, data,
                         config, selection=SELECTION, models=list(models),
                         n_outer_folds=n_folds,
                         subsample_rows=(int(len(_idx)) if _idx is not None
                                         else None)))

    per_rep_results = {}

    # One condition per set of representations. Side 2's without-side is the
    # same set for every subject on this dataset, and so is side 1's with-side
    # whenever the subject came out of the pool, so the memo returns the
    # identical numbers instead of searching them again. The key is the
    # condition_suffix, which names the rep set.
    _cond_memo = {}
    # One importance cache for the whole dataset. Keyed by method, name, shape
    # and a hash of the matrix, so the Gaussian null never reads the real
    # representation's vector. Off unless _C13_REUSE_FOLD_WORK.
    _importance_cache = {} if _C13_REUSE_FOLD_WORK else None

    def _condition(rep_dict, suffix, subject):
        if suffix not in _cond_memo:
            print(f"    condition {suffix}: {len(rep_dict)} representation(s), "
                  f"{n_folds} outer fold(s)")
            _cond_memo[suffix] = _eval_condition_per_fold(
                rep_dict, labels, groups, folds, models,
                dataset_name=dataset_name, metric=metric,
                task_type=task_type, suffix=suffix, augmentations=augs,
                subject=subject, importance_cache=_importance_cache, ck=ck,
                n_outer_folds=n_folds)
        return _cond_memo[suffix]

    hybrid_slots = ('hybrid1', 'hybrid2')

    for rep_name, X_new in sorted(new_reps.items()):
        print(f"\n{'='*60}")
        print(f"  NEW REP: {rep_name} (dim={X_new.shape[1]})")
        print(f"{'='*60}")

        rep_result = {
            'dim': X_new.shape[1],
            # Where this embedding model came from (2026-09-12): one of the
            # dataset's own representations, or the candidate list. A pool one
            # is inside the side-1 with-side's search either way; a candidate
            # one is added to the whole pool.
            'source': 'pool' if rep_name in existing_reps else 'candidate_list',
            'conditions': {},
            'controls': {},
            'deltas': {},
        }

        # =================================================================
        # Condition B: new rep r alone, under each decided model.
        # 2026-10-05: condition A (r under one fixed "probe" model, rf in
        # practice since ridge is in no model list) is gone. With both decided
        # models scored, A would repeat B's fit under the first model. B no
        # longer picks the best model on the folds it is reported on.
        # =================================================================
        print(f"\n  Condition B: {rep_name} alone, each decided model")
        b_scores = {}
        for m, factory in models.items():
            b_scores[m] = _eval_single_rep_on_folds(
                X_new, labels, folds, factory, metric_fn)
            _b = _spread(b_scores[m])
            print(f"    {m:8s} {metric_name} = {_b['score_mean']:.4f}"
                  f"{'' if _b['score_spread'] is None else ' +/- %.4f' % _b['score_spread']}")
        rep_result['conditions']['B'] = {
            'description': f'{rep_name} alone',
            'by_model': {m: _spread(s) for m, s in b_scores.items()},
        }

        # =================================================================
        # THE TWO WITHOUT-SIDES. DECIDED 2026-09-12: both get run, and both
        # get reported, side by side.
        #   Side 1 — the dataset's other representations. Every other embedding
        #     model stays in, so testing esm3 on flab_thermo_tresanco leaves
        #     saprot, esmc and igt5 on the without-side.
        #   Side 2 — the established representations only. Every other embedding
        #     model comes out too, so the subtraction reads as one new model
        #     against the established set and not against its own siblings.
        # Each side runs three conditions: the without-side, the without-side
        # plus r, and the without-side plus a random-Gaussian matrix shaped like
        # r. The Gaussian keeps its role as the selection-bias null on both.
        # =================================================================
        X_null = _make_random_gaussian(X_new, seed=1234)

        # When r came out of the pool, side 1's with-side IS the pool — the same
        # representations for every subject on this dataset. One condition_suffix
        # for that set, so the memo and the search store both see one rep set.
        in_pool = rep_name in existing_reps
        sides = [
            {
                'keys': ('C', 'D', 'D_random'),
                'label': "the dataset's other representations",
                'without': {k: v for k, v in existing_reps.items()
                            if k != rep_name},
                # A candidate's side-1 without-side is the whole pool, the
                # same representations as a pool subject's with-side, so it
                # reads that condition instead of searching it again under a
                # second name (2026-10-05; same searches, same numbers).
                'suffixes': (
                    f"c13_without_{rep_name}" if in_pool else "c13_whole_pool",
                    "c13_whole_pool" if in_pool else f"c13_with_{rep_name}",
                    f"c13_randomnull_{rep_name}"),
            },
            {
                'keys': ('C_established', 'D_established',
                         'D_random_established'),
                'label': 'the established representations only',
                'without': {k: v for k, v in established_reps.items()
                            if k != rep_name},
                'suffixes': (
                    "c13_established",
                    f"c13_est_with_{rep_name}",
                    f"c13_est_randomnull_{rep_name}"),
            },
        ]

        conds = {}
        for side in sides:
            c_key, d_key, dr_key = side['keys']
            without_reps = side['without']
            without_suffix, with_suffix, null_suffix = side['suffixes']
            names_line = ', '.join(sorted(without_reps)) or 'none'

            print(f"\n  Condition {c_key}: {side['label']} — "
                  f"{len(without_reps)} representation(s): {names_line}")

            if not without_reps:
                print(f"    SKIPPED: nothing left on this without-side, so "
                      f"{c_key}, {d_key} and {dr_key} do not run.")
                for k in side['keys']:
                    rep_result['conditions'][k] = {
                        'description': f'{k} (skipped, this without-side is empty)',
                        'reps_searched': [], 'by_model': {}, 'per_fold': [],
                        'is_hybrid': False}
                    conds[k] = None
                continue

            # A one-representation without-side is scored as that representation,
            # not as a hybrid — find_best_hybrid needs two. is_hybrid records
            # which happened so the number is never read as a hybrid score.
            if len(without_reps) < 2:
                print(f"    Only 1 representation here, so this side is scored "
                      f"as that representation alone, not as a hybrid.")

            c = _condition(without_reps, without_suffix, None)
            conds[c_key] = c
            rep_result['conditions'][c_key] = dict(
                c, description=(f'{side["label"]} ({len(without_reps)} reps), '
                                f'{rep_name} left out'))

            # ── that side's without-reps PLUS r ──────────────────────────
            with_reps = {**without_reps, rep_name: X_new}
            print(f"\n  Condition {d_key}: those {len(without_reps)} plus "
                  f"{rep_name}")
            d = _condition(with_reps, with_suffix, rep_name)
            conds[d_key] = d
            rep_result['conditions'][d_key] = dict(
                d, description=f'{side["label"]} plus {rep_name}')

            # ── that side's without-reps PLUS the Gaussian matrix ────────
            # The with-side searches a superset of the without-side on the SAME
            # rows, so it can win by selection alone. A random matrix of
            # identical width is meant to earn exactly that inflation and
            # nothing else. The distinct condition_suffix gives it its own
            # stored rows (it rides on `tier`, find_best_hybrid.py:8801), so it
            # never reads the real representation's search. Verified 2026-09-23:
            # loops/c13/nullcheck.py.
            # ⛔ DIRTY RUN 2026-09-29: on herg_ki and buchwald_hartwig the
            # Gaussian got 0 features on every fold from its own search, so
            # this condition built the same hybrid as the without-side and
            # D - D_random equalled D - C.
            # DECIDED 2026-10-06 (Adelaide delegated): the Gaussian is forced
            # to the subject's feature share and the share is recorded. D's
            # own hybrid per fold, model and slot is rebuilt with the
            # subject's selected features replaced by Gaussian features at the
            # same indices (_forced_null_condition). The Gaussian's own
            # search no longer runs.
            print(f"\n  Condition {dr_key}: D's hybrids with {rep_name}'s "
                  f"features replaced by the same number of random-Gaussian "
                  f"features ({X_new.shape[1]}-feature matrix)")
            dr = _forced_null_condition(
                d, with_reps, rep_name, X_null, labels, folds, models,
                metric=metric, suffix=null_suffix)
            conds[dr_key] = dr
            rep_result['conditions'][dr_key] = dict(
                dr, description=(f'{d_key} hybrids with {rep_name} features '
                                 f'replaced by the same number of '
                                 f'random-Gaussian features'))

        # =================================================================
        # Deltas, per decided model and per hybrid slot, fold by fold on the
        # same outer folds (sign-normalised: positive = improvement).
        # =================================================================
        def _feat_counts(cond, m, slot):
            """Features r (or the Gaussian in its place) got, per fold."""
            out = []
            for fr in (cond or {}).get('per_fold') or []:
                r = ((fr.get('models') or {}).get(m) or {}).get(slot) or {}
                fb = r.get('features_by_rep')
                out.append(None if fb is None else int(fb.get(rep_name, 0)))
            return out

        for m in models:
            rep_result['deltas'][m] = {}
            b = b_scores[m]
            for slot in hybrid_slots:
                s = {k: _slot_scores(conds.get(k), m, slot, n_folds_run)
                     for k in ('C', 'D', 'D_random', 'C_established',
                               'D_established', 'D_random_established')}
                g1 = _feat_counts(conds.get('D_random'), m, slot)
                g2 = _feat_counts(conds.get('D_random_established'), m, slot)
                rep_result['deltas'][m][slot] = {
                    # ── Side 1: against the dataset's other representations ──
                    # In-sample upper bound: D searches a superset of C.
                    'D_minus_C': _paired_delta(s['D'], s['C'], sign),
                    # What a same-width random matrix earns.
                    'D_random_minus_C': _paired_delta(s['D_random'], s['C'], sign),
                    'D_minus_D_random': _paired_delta(s['D'], s['D_random'], sign),
                    # The hybrid against r alone under the same model.
                    'D_minus_B': _paired_delta(s['D'], b, sign),
                    # ── Side 2: against the established representations only ──
                    'D_established_minus_C_established': _paired_delta(
                        s['D_established'], s['C_established'], sign),
                    'D_random_established_minus_C_established': _paired_delta(
                        s['D_random_established'], s['C_established'], sign),
                    'D_established_minus_D_random_established': _paired_delta(
                        s['D_established'], s['D_random_established'], sign),
                    'D_established_minus_B': _paired_delta(
                        s['D_established'], b, sign),
                    # The features the subject got in D, and the Gaussian got
                    # in D_random, per fold. Equal since 2026-10-06 (forced
                    # share); a fold where the subject got 0 features in D
                    # has D_random equal to D.
                    'subject_features_D': _feat_counts(conds.get('D'), m, slot),
                    'gaussian_features_D_random': g1,
                    'n_folds_gaussian_got_no_features': sum(
                        1 for x in g1 if x == 0),
                    'subject_features_D_established': _feat_counts(
                        conds.get('D_established'), m, slot),
                    'gaussian_features_D_random_established': g2,
                    'n_folds_gaussian_got_no_features_established': sum(
                        1 for x in g2 if x == 0),
                }

        print(f"\n  --- DELTAS in {metric_name}, positive = improvement, mean "
              f"over outer folds ---")
        for m in models:
            for slot in hybrid_slots:
                dd = rep_result['deltas'][m][slot]
                print(f"    {m} {slot}: D-C {dd['D_minus_C']['mean']:+.4f}  "
                      f"Drand-C {dd['D_random_minus_C']['mean']:+.4f}  "
                      f"D-Drand {dd['D_minus_D_random']['mean']:+.4f}  "
                      f"D-B {dd['D_minus_B']['mean']:+.4f}  | side 2: "
                      f"D-C {dd['D_established_minus_C_established']['mean']:+.4f}  "
                      f"D-Drand {dd['D_established_minus_D_random_established']['mean']:+.4f}  "
                      f"D-B {dd['D_established_minus_B']['mean']:+.4f}  | "
                      f"folds where the Gaussian got 0 features: "
                      f"{dd['n_folds_gaussian_got_no_features']} of "
                      f"{sum(1 for x in dd['gaussian_features_D_random'] if x is not None)}")

        # =================================================================
        # Controls: random Gaussian and row-shuffled r, alone, each decided
        # model, the same folds as B.
        # =================================================================
        print(f"\n  Controls for {rep_name}:")
        X_random = _make_random_gaussian(X_new)
        X_shuffled = _make_row_shuffled(X_new)
        for ctrl_key, X_ctrl, desc in (
                ('random_gaussian', X_random,
                 f'random Gaussian ({X_new.shape[1]}D)'),
                ('row_shuffled', X_shuffled, f'row-shuffled {rep_name}')):
            by_m = {}
            for m, factory in models.items():
                sc = _eval_single_rep_on_folds(X_ctrl, labels, folds, factory,
                                               metric_fn)
                by_m[m] = dict(_spread(sc),
                               signal_above=_paired_delta(b_scores[m], sc, sign))
                print(f"    {desc:30s} {m:8s} {metric_name} = "
                      f"{by_m[m]['score_mean']:.4f}  ({rep_name} alone minus "
                      f"this, positive = better: "
                      f"{by_m[m]['signal_above']['mean']:+.4f})")
            rep_result['controls'][ctrl_key] = {'description': desc,
                                                'by_model': by_m}

        per_rep_results[rep_name] = rep_result

    # =====================================================================
    # Cross-rep summary, per decided model and hybrid slot
    # =====================================================================
    print(f"\n{'='*70}")
    print("  FORWARD COMPATIBILITY SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Dataset: {dataset_name} | Metric: {metric_name} | "
          f"Domain: {domain} | outer folds: {n_folds}")
    print(f"  {'lower is better' if lower_is_better else 'higher is better'}")

    # One row per embedding model tested on this dataset. Every score is in
    # {metric_name}, the mean over outer folds of the held-out score. Dim is
    # that representation's number of features. Feats is the mean number of
    # features the hybrid took from it per fold.
    win_rates = {}
    for m in models:
        win_rates[m] = {}
        for slot in hybrid_slots:
            counts = {}
            for side_tag, keys in (
                    ('side1', ('C', 'D', 'D_random', 'D_minus_D_random',
                               'D_minus_B', 'subject_features_D')),
                    ('side2', ('C_established', 'D_established',
                               'D_random_established',
                               'D_established_minus_D_random_established',
                               'D_established_minus_B',
                               'subject_features_D_established'))):
                c_k, d_k, dr_k, ddr_k, db_k, feat_k = keys
                print(f"\n  {m}, {slot}, "
                      f"{'side 1 (the dataset' + chr(39) + 's other representations)' if side_tag == 'side1' else 'side 2 (the established representations only)'}")
                print(f"  {'New Rep':<20} {'Dim':>5} {'B(alone)':>9} {c_k[:9]:>9} "
                      f"{d_k[:9]:>9} {dr_k[:9]:>9} {'D-Drand':>8} {'D-B':>8} "
                      f"{'Feats':>6}")
                n_ddr = n_db = w_ddr = w_db = 0
                for rep_name, rr in sorted(per_rep_results.items()):
                    cs = rr['conditions']
                    dd = rr['deltas'][m][slot]

                    def _mean(k):
                        return float(np.nanmean([x for x in _slot_scores(
                            cs.get(k), m, slot, n_folds_run) if _finite(x)] or
                            [np.nan]))
                    b_val = cs['B']['by_model'][m]['score_mean']
                    ddr = dd[ddr_k]['mean']
                    db = dd[db_k]['mean']
                    feats = [x for x in dd[feat_k] if x is not None]
                    if np.isfinite(ddr):
                        n_ddr += 1
                        w_ddr += int(ddr > 0)
                    if np.isfinite(db):
                        n_db += 1
                        w_db += int(db > 0)
                    print(f"  {rep_name:<20} {rr['dim']:>5} {b_val:>9.4f} "
                          f"{_mean(c_k):>9.4f} {_mean(d_k):>9.4f} "
                          f"{_mean(dr_k):>9.4f} {ddr:>+8.4f} {db:>+8.4f} "
                          f"{(np.mean(feats) if feats else float('nan')):>6.1f}")
                counts[side_tag] = {
                    'n_scored_D_vs_D_random': n_ddr,
                    'n_D_beats_D_random': w_ddr,
                    'n_scored_D_vs_B': n_db,
                    'n_D_beats_B': w_db,
                }
                print(f"    D beats D_random on {w_ddr} of the {n_ddr} embedding "
                      f"models with a score; D beats B on {w_db} of {n_db}.")
            win_rates[m][slot] = counts

    # Build output
    output = make_structured_output(
        experiment_id='c13_forward_compatibility',
        dataset=dataset_name,
        data=data,
        config=config,
        results={
            'per_rep': per_rep_results,
            'selection': SELECTION,
            'selection_scope': 'fold_train_rows',
            'metric': metric_name,
            'lower_is_better': bool(lower_is_better),
            'models': list(models.keys()),
            'decided_models_without_factory': _no_factory,
            'n_outer_folds': n_folds,
            'n_outer_folds_run': n_folds_run,
            'outer_split': {'splitter': 'nested_cv_outer_folds', 'seed': SEED,
                            'n_outer_folds': n_folds,
                            'fold_sizes': [[int(len(tr)), int(len(va))]
                                           for tr, va in folds]},
            'aug_conditions_general': list(aug_conditions),
            'subsampled_rows': (int(len(_idx)) if _idx is not None
                                else None),
            'subsample_draw': _draw_record,
            'summary': {
                'new_reps': sorted(new_reps.keys()),
                'existing_reps': sorted(existing_reps.keys()),
                # What the rule made of this dataset's own representations
                # (decided 2026-09-12). reps_unclassified are names the rule
                # does not know; they were counted with the established side and
                # were NOT tested as subjects.
                'pool_subject_reps': subject_names,
                'pool_established_reps': established_names,
                'reps_unclassified': unclassified_names,
                'subject_reps_from_pool': sorted(pool_subjects.keys()),
                'subject_reps_from_candidate_list': sorted(
                    k for k in candidate_reps if k not in pool_subjects),
                # Embedding models the rule named as subjects that this run did
                # NOT score, and why. Empty on a shipped run. Without this a
                # win count over 2 subjects reads as though the dataset had 2.
                'subject_reps_not_scored': list(_dropped_subjects),
                'subject_limit': _max_subjects,
                'subject_limit_source': _max_why,
                'candidate_policy': _cand_why,
                # The two without-sides, by name.
                'without_side_1': 'the dataset\'s other representations',
                'without_side_2_reps': sorted(established_reps.keys()),
                # Results audit C13c (owner, 2026-10-05: keep). True when the
                # side-2 without-side is one representation scored alone.
                'without_side_2_is_single_rep': len(established_reps) == 1,
                # Without these, a result carrying one candidate reads as
                # "the list holds one" rather than "one of N was buildable
                # here". Results audit C13d (2026-10-05): each unloadable
                # candidate is recorded with the reason its build gave.
                'candidates_not_loaded': sorted(missing_candidates),
                'candidates_not_loaded_reasons': missing_reasons,
                'candidates_requested': len(candidate_reps) + len(missing_candidates),
                # {model: {hybrid slot: {side1|side2: counts}}}. Each count is
                # over the embedding models whose delta has a score.
                'win_counts': win_rates,
                'total_new_reps_tested': len(per_rep_results),
                'condition_A_note': (
                    'Condition A (one fixed probe model) was removed '
                    '2026-10-05: ridge is removed everywhere, and with both '
                    'decided models scored it repeated condition B.'),
                'metric_note': (
                    'D-C is a selection-biased upper bound. D_random is D\'s '
                    'own hybrid with the subject\'s features replaced by the '
                    'same number of random-Gaussian features (forced share, '
                    '2026-10-06), so D - D_random differs in feature content '
                    'only; gaussian_features_D_random records the share. The '
                    'same numbers exist on side 2, where every other '
                    'embedding model is out of the without-side. Per outer '
                    'fold, every hybrid comes from a search on that fold\'s '
                    'training rows. No significance tests are reported.'),
            },
        },
        primary_metric=metric_name,
        lower_is_better=lower_is_better)

    return output


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = create_experiment_parser(
        'C13: Forward Compatibility — do new reps benefit from hybridization?',
        has_delta=True,
        has_augment=True)
    parser.add_argument(
        '--new-reps', type=str, default=None,
        help='Comma-separated list of rep names to test, one at a time. '
             'A name that is one of the dataset\'s own representations is '
             'tested from the pool; any other name is looked for in this '
             'domain\'s block of forward_compat_reps.py and loaded separately. '
             'If not specified: every representation in the pool the rule calls '
             'a learned embedding model, plus every candidate in the block. '
             '(Before 2026-09-12 naming a pool member here did nothing.)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode:    {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps:    {args.reps}")
    print(f"Models:  {config['models']}")
    print(f"Folds:   {config['n_splits']}")

    # Parse new-reps argument
    new_rep_names = None
    if args.new_reps:
        new_rep_names = [r.strip() for r in args.new_reps.split(',') if r.strip()]
        print(f"New reps (user-specified): {new_rep_names}")

    # Honor a comma-separated --reps list (e.g. "aac,blosum,onehot") as an
    # explicit rep_names override. Without this, load_data treats the whole
    # string as a preset name, falls through to rep_names=None, and then
    # auto-fills the FULL DATASET_REPS 'general' list (foundation models
    # included) — which is unloadable locally and defeats --reps entirely.
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset,
                     rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    # The candidate load is a SECOND load_data call inside run(), and it has to
    # be given the same delta/augment settings or the candidates come back built
    # on different rows than the pool.
    data['_delta'] = getattr(args, 'delta', False)
    data['_augment'] = getattr(args, 'augment', False)
    results = run(data, config, args.output, args.dataset,
                  new_rep_names=new_rep_names)
    save_experiment_results(results, args.output, 'c13_forward_compatibility',
                            args.dataset)
