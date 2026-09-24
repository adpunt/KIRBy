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

Condition ablation per new embedding model r, per dataset d:
  A: New rep r + one fixed model (raw embedding quality; a linear probe when
     ridge is in the model set, otherwise the first model in it — the stored
     conditions.A.probe_model says which ran)
  B: New rep r + best model      (best-case single-rep)

  Two without-sides, both run and reported side by side (decided 2026-09-12):

  Side 1 — against the dataset's other representations. Every other embedding
  model stays in, so on flab_thermo_tresanco the without-side for esm3 still
  holds saprot, esmc and igt5.
  C:        Best hybrid over the dataset's other representations, r left out
  D:        Best hybrid over those plus r
  D_random: Best hybrid over those plus a random-Gaussian matrix shaped like r
            (the selection-bias null for D)

  Side 2 — against the established representations only. Every OTHER embedding
  model comes out of the without-side too, so the subtraction reads as one new
  model against the established set rather than against its own siblings.
  C_established:        Best hybrid over the established representations
  D_established:        Best hybrid over those plus r
  D_random_established: Best hybrid over those plus the same Gaussian matrix

  (Condition E is gone. It was once a duplicate of D and carried no
   information; the "does the hybrid help r itself" question is now answered
   honestly by D - B, below. No E / E - B is emitted any more.)

Controls per new rep r:
  - Random Gaussian matrix (same dims as r): single-rep permutation null (cond A)
  - Row-shuffled r: tests content vs dimensionality

Key metrics (all sign-normalised: positive = improvement, any metric direction):
  1. D - C          : new-rep contribution, IN-SAMPLE UPPER BOUND ONLY. D searches
                      a superset of C on the same data, so this is biased positive
                      by construction. Never present it as an unbiased effect.
  2. D - D_random   : the DEBIASED new-rep contribution. Subtracts the selection
                      inflation that any same-dim matrix earns. If D - C is not
                      clearly above D_random - C (i.e. D - D_random <= 0), the
                      apparent gain is selection noise, not the rep.
  3. D - B          : does the hybrid improve the new rep beyond its own best
                      single-model score? (the honest "hybrid helps r" metric)
  4. Allocation share: features_from_r / total_budget
  5. Win rate       : fraction of datasets where the DEBIASED effect (D - D_random)
                      and D - B are positive.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    build_fold_hybrid,
    get_best_search_for_downstream,
    parse_rep_args,
    _is_tcga_dataset, _is_antibody_dataset)


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
                             delta=False, augment=False, pool_labels=None):
    """Load the forward-compatibility candidates for this dataset.

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
                continue
            got = one.get('reps', {}) or {}
            if name in got:
                merged[name] = got[name]
                if labels_ref is None:
                    labels_ref = one.get('labels')
        if not merged:
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
            continue
        aligned[name] = X

    missing = [n for n in wanted if n not in aligned]
    if aligned:
        print(f"  Loaded {len(aligned)}: "
              f"{', '.join(f'{k}({v.shape[1]}D)' for k, v in sorted(aligned.items()))}")
    if missing:
        print(f"  Not loaded ({len(missing)}): {', '.join(sorted(missing))}")
        print(f"    (no builder written, weights absent, or the build raised — "
              f"see forward_compat_reps.py for which)")
    return aligned, missing


def _subsample(reps, idx):
    """Take the same rows out of every rep matrix."""
    return {k: v[idx] for k, v in reps.items()}


# =============================================================================
# PER-FOLD EVALUATION HELPERS
# =============================================================================

def _eval_single_rep_per_fold(X, labels, groups, model_factory, n_folds, mode,
                               metric_fn):
    """Evaluate a single rep with a single model, returning per-fold scores."""
    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels, mode=mode):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = model_factory()
        model.fit(X_train, labels[train_idx])
        scores.append(metric_fn(labels[val_idx], model.predict(X_val)))

    return scores


def _eval_hybrid_per_fold(reps, labels, groups, model_factory, n_folds, mode,
                           metric_fn, augmentations=None,
                           metric_name='r2', dataset_name=None,
                           task_type='regression', condition_suffix=None,
                           downstream_model=None):
    """Run find_best_hybrid and evaluate the best config per fold.

    Returns (per_fold_scores, search_result_dict).
    search_result_dict contains best_by_model results, baselines, allocation info.

    `downstream_model` is the model the recipe is searched FOR, and it is the
    same model `model_factory` builds to score that recipe. Without it the
    recipe came from whichever of five downstream models scored best inside
    the search, while the scoring model was condition B's winner — two
    independent choices, so the recipe was often built for a model that never
    ran here. Left None (no caller does), the old five-model loop runs.
    """
    if len(reps) < 2:
        # find_best_hybrid needs >= 2 reps; fall back to single-rep eval
        rep_name = list(reps.keys())[0]
        X = reps[rep_name]
        scores = _eval_single_rep_per_fold(
            X, labels, groups, model_factory, n_folds, mode, metric_fn)
        return scores, {
            'best_result': {'score_mean': float(np.mean(scores)) if scores else float('nan')},
            'baselines': {},
            'allocation': {rep_name: X.shape[1]},
            'n_features': X.shape[1],
        }

    # Whose recipe this is. One name when the caller said which model scores
    # it; otherwise the study's default model set (2026-09-04 tiers), of which
    # the best-scoring one is kept. That list used to read
    # ['rf', 'svm', 'lightgbm', 'mlp', 'knn'] — mlp and knn moved to tier
    # 'extra' that day, so two of the five could never return a blueprint, and
    # the bare `except Exception: continue` below meant they dropped out with
    # no message.
    best = None
    best_search = None
    best_meta = None
    _candidates = ([downstream_model] if downstream_model
                   else ['rf', 'lightgbm', 'xgboost', 'svm', 'dnn'])
    _dropped = []
    for candidate_model in _candidates:
        try:
            # condition_suffix is enough on its own. It looks at first read as
            # though it only names the search pickle -- it appears nowhere in
            # result_cache.py -- but find_best_hybrid.py:8801 folds it into
            # `tier`, and `tier` is IDENTITY_COLS[0]. So two arms differing
            # only by suffix get different row identities.
            #
            # That matters most for the random-Gaussian control, which replaces
            # a representation with noise of the SAME NAME and SAME WIDTH:
            # _rep_pool_key carries names and feature counts only, and
            # representation CONTENT is excluded on purpose because neural
            # embeddings are not bit-reproducible. The suffix is the only thing
            # keeping the two apart. Checked 2026-09-23 on flab with two
            # representations at budget 25, sharing one result database: the
            # real arm scored 0.440540 spearman and the noise arm 0.354353.
            # See loops/c13/nullcheck.py.
            _extra = {'condition_suffix': condition_suffix} if condition_suffix else {}
            # ⛔ speed='fast' SINCE 2026-09-18, OWNER'S DECISION.
            #
            # Every search c13 runs is built live. Each condition carries a
            # condition_suffix -- "with representation r" against "without" --
            # so each is a distinct search identity that no precompute covers.
            # There is no stored grid to miss by asking for a shorter one.
            #
            # Full is 5 allocation methods x 3 importance methods x 4
            # representation counts x 8 budgets x 17 (model, evaluator) pairs =
            # 8,160 configurations per search. c13 runs 24 of them per dataset
            # at 4 pool subjects, and up to 75 if every candidate
            # representation builds, so it was about 19 million model fits per
            # dataset.
            #
            # ⛔ THIS DOES NOT SHORTEN THE MODEL LIST. The filter that drops
            # xgboost and svm under fast runs only when models=None
            # (find_best_hybrid.py:8221-8232), and _candidates above is an
            # explicit list. So the (model, evaluator) pair count stays at 17
            # and the grid lands at 2,160 configurations, not 432 -- about 9
            # times fewer, not 19. Shortening _candidates as well would take it
            # the rest of the way, and that is a decision about which models
            # c13 reports, not a compute setting.
            #
            # Nothing c13 reads is lost: it takes best_config and its
            # score_mean, and a fast search produces both.
            s, m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=candidate_model, budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name,
                task_type=task_type, metric=metric_name,
                return_fitted=True,
                speed='fast',
                **_extra)
        except Exception as e:
            _dropped.append((candidate_model, f'{type(e).__name__}: {e}'))
            continue
        if not m:
            _dropped.append((candidate_model, 'no search metadata returned'))
            continue
        bc = m['best_config']
        score = bc.get('score_mean')
        if score is None:
            _dropped.append((candidate_model, 'best_config carried no score_mean'))
            continue
        lib = s.lower_is_better
        is_better = (best is None) or (
            score < best['score_mean'] if lib else score > best['score_mean'])
        if is_better:
            best = bc
            best_search = s
            best_meta = m
    if _dropped:
        print(f"      {len(_dropped)} of {len(_candidates)} downstream models "
              f"gave no blueprint:")
        for _name, _why in _dropped:
            print(f"        {_name}: {_why}")
    if best is None or best_search is None:
        if downstream_model:
            print(f"      no recipe searched for {downstream_model}, the model "
                  f"that scores it - this condition is not scored")
        else:
            print(f"      pool helper found no usable blueprint from any of "
                  f"the {len(_candidates)} models")
        return [], {'error': 'no blueprint found via pool',
                    'downstream_model': downstream_model,
                    'models_dropped': dict(_dropped)}
    search = best_search

    blueprint = best.get('blueprint')
    if blueprint is None:
        return [], {'error': 'no blueprint in best_config'}

    scores = []
    allocation_dicts = []
    last_feature_info = None
    sample_X = next(iter(reps.values()))

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
        val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

        try:
            X_train, X_val, feature_info = build_fold_hybrid(
                train_reps, val_reps, y_train, blueprint,
                augmentations_train=train_augs, augmentations_val=val_augs,
                groups=(groups[train_idx] if groups is not None else None),
                model=model_factory(),
                metric_fn=metric_fn)
        except Exception:
            continue

        # Extract allocation from feature_info
        alloc = feature_info.get('allocation', {})
        allocation_dicts.append(alloc)
        last_feature_info = feature_info

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        scores.append(metric_fn(y_val, model.predict(X_val_s)))

    # Average allocation across folds
    avg_alloc = {}
    if allocation_dicts:
        all_keys = set()
        for ad in allocation_dicts:
            all_keys.update(ad.keys())
        for k in sorted(all_keys):
            avg_alloc[k] = float(np.mean([ad.get(k, 0) for ad in allocation_dicts]))

    result_info = {
        # The model this recipe was searched for. It is the model that scored
        # it too, whenever the caller named one.
        'downstream_model': downstream_model or (best_meta or {}).get('model'),
        'best_result': convert_numpy(best),
        'baselines': convert_numpy(search.baselines),
        'allocation': avg_alloc,
        'n_features': best.get('n_features', sum(avg_alloc.values()) if avg_alloc else 0),
        'feature_info': last_feature_info or {},  # consumed by modality_novelty
        'blueprint': {
            'allocation_method': blueprint.allocation_method,
            'importance_method': blueprint.importance_method,
            'budget': blueprint.budget,
            'internal_evaluator': blueprint.internal_evaluator,
        },
    }

    return scores, result_info


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
    """C13: Forward Compatibility — 5-condition ablation per new rep.

    Args:
        data: Loaded dataset dict from load_data().
        config: Config dict (fast or full).
        output_dir: Output directory for results JSON.
        dataset_name: Name of the dataset.
        new_rep_names: Candidate rep names to load and score. If None, the whole
            forward-compatibility block for this dataset's domain is used.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C13: FORWARD COMPATIBILITY")
    print("=" * 70)

    # Smoke-test shrink (gated on KIRBY_SMOKE_N; no-op on server/full runs).
    # precompute_hybrid_searches reads the module-level _shared.FULL_BUDGETS and
    # IGNORES config['budgets'], so the only way to make the local hybrid search
    # tractable is to shrink FULL_BUDGETS here before any precompute fires.
    if os.environ.get('KIRBY_SMOKE_N', '').strip():
        import experiments._shared as _shared_mod
        if len(_shared_mod.FULL_BUDGETS) > 1:
            _shared_mod.FULL_BUDGETS = [25]
            print("  [SMOKE] FULL_BUDGETS shrunk to [25] for local hybrid search")

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    augs = data.get('augmentations')

    # Size-gate (2026-06-28): c13 runs uncached find_best_hybrid searches with
    # FULL budgets (it ignores config['budgets']), so it blows the wall on big
    # datasets. Above the cutoff, subsample (reps/labels/groups/augs jointly) to
    # K rows — same approach c17 already uses internally. Fixed seed = stable.
    from experiments._shared import LARGE_DATASET_N, LARGE_SUBSAMPLE_K
    _n_full = len(labels)
    _idx = None
    if len(labels) > LARGE_DATASET_N:
        _rng = np.random.RandomState(42)
        _idx = np.sort(_rng.choice(len(labels), LARGE_SUBSAMPLE_K, replace=False))
        reps = _subsample(reps, _idx)
        labels = labels[_idx]
        groups = groups[_idx] if groups is not None else None
        if augs is not None:
            augs = _subsample(augs, _idx)
        print(f"  [size-gate] n>{LARGE_DATASET_N}: subsampled to "
              f"{LARGE_SUBSAMPLE_K} rows for c13")

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    models = get_model_factories(
        config['models'],
        task_type=data['task_type'],
        n_samples=len(labels))

    # ── The pool, the rule, and the candidates ────────────────────────────
    # existing_reps is the dataset's whole pool. It is the side-1 without-side
    # (minus whichever representation is the subject).
    existing_reps = reps

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

    # Candidates load at the dataset's FULL row count, so the size-gate index has
    # to be applied to them too or every one of them lands row-misaligned.
    if requested is not None and not candidate_names:
        candidate_reps, missing_candidates = {}, []
    else:
        candidate_reps, missing_candidates = load_forward_compat_reps(
            dataset_name, domain, n_rows=_n_full,
            rep_names=candidate_names,
            delta=data.get('_delta', False), augment=data.get('_augment', False),
            pool_labels=data['labels'])
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
    print(f"  n_folds:      {n_folds}")
    print(f"  Models:       {list(models.keys())}")
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
        print(f"    Fewer than 2, so the side-2 without-side is scored as a "
              f"single representation, not as a hybrid.")

    # Find best model for each condition via CV
    # We use all available models, then pick the best for conditions A/B
    per_rep_results = {}

    # One hybrid search per (rep set, downstream model). Side 2's without-side is
    # the same set of representations for every subject on this dataset, and so
    # is side 1's with-side whenever the subject came out of the pool, so the
    # memo returns the identical numbers instead of searching them again. The
    # key is the condition_suffix, which names the rep set, plus the model.
    _hybrid_memo = {}

    def _hybrid(rep_dict, model_factory, model_name, suffix):
        key = (suffix, model_name)
        if key not in _hybrid_memo:
            _hybrid_memo[key] = _eval_hybrid_per_fold(
                rep_dict, labels, groups, model_factory,
                n_folds, mode, metric_fn,
                # The recipe is searched for the same model that scores it:
                # model_factory builds model_name, and the search is asked for
                # model_name. The memo key already carries it.
                downstream_model=model_name,
                augmentations=augs, metric_name=metric_name,
                # Real dataset_name + condition_suffix so the cache key remains
                # content-validated (the rep_shapes hash already distinguishes
                # "with r" from "without r"); a synthetic dataset name bypasses
                # the precompute cache and forces a live find_best_hybrid that
                # sees all labels in its internal CV.
                dataset_name=dataset_name,
                condition_suffix=suffix,
                task_type=data['task_type'])
        return _hybrid_memo[key]

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
        # Condition A: new rep r under one fixed model (raw embedding quality).
        # Intended as a linear probe, which is how embedding quality is reported
        # in the literature. ridge is in NO model list any caller passes — the
        # default set is rf/lightgbm/xgboost/svm/dnn — so this fell through to
        # the first model in that set while the printed line and the STORED
        # description both still said "ridge". Both now name the model that ran,
        # and probe_model records it, so a reader can never mistake a random
        # forest probe for a linear one.
        # =================================================================
        ridge_factory = models.get('ridge')
        if ridge_factory is not None:
            probe_model = 'ridge'
        else:
            probe_model = list(models.keys())[0]
            ridge_factory = models[probe_model]
        print(f"\n  Condition A: {rep_name} + {probe_model}"
              f"{'' if probe_model == 'ridge' else '  (ridge not in this model set)'}")

        a_scores = _eval_single_rep_per_fold(
            X_new, labels, groups, ridge_factory, n_folds, mode, metric_fn)
        a_mean = float(np.mean(a_scores)) if a_scores else float('nan')
        a_std = float(np.std(a_scores)) if a_scores else float('nan')
        rep_result['conditions']['A'] = {
            'description': f'{rep_name} + {probe_model}',
            'probe_model': probe_model,
            'probe_is_linear': probe_model == 'ridge',
            'score_mean': a_mean,
            'score_std': a_std,
            'fold_scores': [float(s) for s in a_scores],
        }
        print(f"    {metric_name} = {a_mean:.4f} +/- {a_std:.4f}")

        # =================================================================
        # Condition B: new rep r + best model
        # =================================================================
        print(f"\n  Condition B: {rep_name} + best model")
        best_b_mean = -np.inf if not lower_is_better else np.inf
        best_b_model = None
        best_b_scores = []

        for model_name, model_factory in models.items():
            b_scores = _eval_single_rep_per_fold(
                X_new, labels, groups, model_factory, n_folds, mode, metric_fn)
            b_mean = float(np.mean(b_scores)) if b_scores else float('nan')

            if np.isnan(b_mean):
                continue

            is_better = (b_mean > best_b_mean) if not lower_is_better else (b_mean < best_b_mean)
            if is_better:
                best_b_mean = b_mean
                best_b_model = model_name
                best_b_scores = b_scores

        b_std = float(np.std(best_b_scores)) if best_b_scores else float('nan')
        rep_result['conditions']['B'] = {
            'description': f'{rep_name} + {best_b_model}',
            'best_model': best_b_model,
            'score_mean': float(best_b_mean) if not np.isinf(best_b_mean) else float('nan'),
            'score_std': b_std,
            'fold_scores': [float(s) for s in best_b_scores],
        }
        print(f"    Best model: {best_b_model}")
        print(f"    {metric_name} = {best_b_mean:.4f} +/- {b_std:.4f}")

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
        model_factory = models.get(best_b_model, ridge_factory)
        X_null = _make_random_gaussian(X_new, seed=1234)

        # When r came out of the pool, side 1's with-side IS the pool — the same
        # representations for every subject on this dataset. One condition_suffix
        # for that set, so the memo and the search cache both see one rep set.
        in_pool = rep_name in existing_reps
        sides = [
            {
                'keys': ('C', 'D', 'D_random'),
                'label': "the dataset's other representations",
                'without': {k: v for k, v in existing_reps.items()
                            if k != rep_name},
                'suffixes': (
                    f"c13_without_{rep_name}",
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

        side_means = {}
        for side in sides:
            c_key, d_key, dr_key = side['keys']
            without_reps = side['without']
            without_suffix, with_suffix, null_suffix = side['suffixes']
            names_line = ', '.join(sorted(without_reps)) or 'none'

            print(f"\n  Condition {c_key}: hybrid over {side['label']} — "
                  f"{len(without_reps)} representation(s): {names_line}")

            if not without_reps:
                print(f"    SKIPPED: nothing left on this without-side, so "
                      f"{c_key}, {d_key} and {dr_key} do not run.")
                for k in side['keys']:
                    rep_result['conditions'][k] = {
                        'description': f'{k} (skipped, this without-side is empty)',
                        'score_mean': float('nan'),
                        'score_std': float('nan'),
                        'fold_scores': [],
                        'reps_searched': [],
                    }
                    side_means[k] = float('nan')
                continue

            # A one-representation without-side is scored as that representation,
            # not as a hybrid — find_best_hybrid needs two. is_hybrid records
            # which happened so the number is never read as a hybrid score.
            is_hybrid = len(without_reps) >= 2
            if not is_hybrid:
                print(f"    Only 1 representation here, so this side is scored "
                      f"as that representation alone, not as a hybrid.")

            c_scores, c_info = _hybrid(without_reps, model_factory,
                                       best_b_model, without_suffix)
            c_mean_side = float(np.mean(c_scores)) if c_scores else float('nan')
            c_std_side = float(np.std(c_scores)) if c_scores else float('nan')
            rep_result['conditions'][c_key] = {
                'description': (f'hybrid over {side["label"]} '
                                f'({len(without_reps)} reps), {rep_name} left out'),
                'score_mean': c_mean_side,
                'score_std': c_std_side,
                'fold_scores': [float(s) for s in c_scores],
                'hybrid_info': c_info,
                'reps_searched': sorted(without_reps),
                'is_hybrid': is_hybrid,
            }
            side_means[c_key] = c_mean_side
            print(f"    {metric_name} = {c_mean_side:.4f} +/- {c_std_side:.4f}")
            if c_info.get('allocation'):
                print(f"    Allocation: {c_info['allocation']}")

            # ── that side's without-reps PLUS r ──────────────────────────
            with_reps = {**without_reps, rep_name: X_new}
            print(f"\n  Condition {d_key}: those {len(without_reps)} plus "
                  f"{rep_name}")
            d_scores, d_info = _hybrid(with_reps, model_factory,
                                       best_b_model, with_suffix)
            d_mean_side = float(np.mean(d_scores)) if d_scores else float('nan')
            d_std_side = float(np.std(d_scores)) if d_scores else float('nan')
            alloc = d_info.get('allocation', {})
            total_budget = sum(alloc.values()) if alloc else 1
            alloc_share = (alloc.get(rep_name, 0) / total_budget
                           if total_budget > 0 else 0)
            rep_result['conditions'][d_key] = {
                'description': f'hybrid over {side["label"]} plus {rep_name}',
                'score_mean': d_mean_side,
                'score_std': d_std_side,
                'fold_scores': [float(s) for s in d_scores],
                'hybrid_info': d_info,
                'reps_searched': sorted(with_reps),
                'allocation_share': float(alloc_share),
                'features_from_new_rep': float(alloc.get(rep_name, 0)),
                'modality_novelty': _modality_novelty(
                    d_info.get('feature_info', {}), rep_name, X_new,
                    without_reps),
            }
            side_means[d_key] = d_mean_side
            print(f"    {metric_name} = {d_mean_side:.4f} +/- {d_std_side:.4f}")
            if alloc:
                print(f"    Allocation: {alloc}")
                print(f"    Features from {rep_name}: {alloc.get(rep_name, 0)} "
                      f"({alloc_share:.1%} of budget)")

            # ── that side's without-reps PLUS the Gaussian matrix ────────
            # The with-side searches a superset of the without-side on the SAME
            # data, so it wins almost always by selection alone. A random matrix
            # of identical width earns exactly that inflation and nothing else,
            # so the null minus the without-side measures the bias and the
            # representation's own contribution is the with-side minus the null.
            # The distinct condition_suffix gives it its own cache file AND
            # its own row identity (it rides on `tier`,
            # find_best_hybrid.py:8801), so it is computed live on the random
            # content and never collides. Verified 2026-09-23, not assumed:
            # loops/c13/nullcheck.py.
            null_reps = {**without_reps, rep_name: X_null}
            print(f"\n  Condition {dr_key}: those {len(without_reps)} plus a "
                  f"random-Gaussian matrix, {X_new.shape[1]} features, the same "
                  f"width as {rep_name}")
            dr_scores, dr_info = _hybrid(null_reps, model_factory,
                                         best_b_model, null_suffix)
            dr_mean_side = float(np.mean(dr_scores)) if dr_scores else float('nan')
            dr_std_side = float(np.std(dr_scores)) if dr_scores else float('nan')
            dr_alloc = dr_info.get('allocation', {})
            dr_total = sum(dr_alloc.values()) if dr_alloc else 1
            dr_share = (dr_alloc.get(rep_name, 0) / dr_total
                        if dr_total > 0 else 0)
            rep_result['conditions'][dr_key] = {
                'description': (f'hybrid over {side["label"]} plus a '
                                f'random-Gaussian matrix ({X_new.shape[1]}D)'),
                'score_mean': dr_mean_side,
                'score_std': dr_std_side,
                'fold_scores': [float(s) for s in dr_scores],
                'reps_searched': sorted(null_reps),
                'allocation_share': float(dr_share),
            }
            side_means[dr_key] = dr_mean_side
            print(f"    {metric_name} = {dr_mean_side:.4f} +/- {dr_std_side:.4f}")
            print(f"    The Gaussian matrix took {dr_share:.1%} of the budget")

        c_mean = side_means.get('C', float('nan'))
        d_mean = side_means.get('D', float('nan'))
        d_random_mean = side_means.get('D_random', float('nan'))
        c_est_mean = side_means.get('C_established', float('nan'))
        d_est_mean = side_means.get('D_established', float('nan'))
        dr_est_mean = side_means.get('D_random_established', float('nan'))

        # =================================================================
        # Compute deltas (all sign-normalised: positive = improvement)
        # =================================================================
        sign = -1 if lower_is_better else 1
        delta_dc = sign * (d_mean - c_mean)               # in-sample upper bound
        delta_drc = sign * (d_random_mean - c_mean)       # selection-bias null
        delta_ddr = sign * (d_mean - d_random_mean)       # DEBIASED contribution
        delta_db = sign * (d_mean - best_b_mean) if not np.isinf(best_b_mean) else float('nan')
        # The same three subtractions on side 2 (decided 2026-09-12): every
        # other embedding model is out of the without-side.
        delta_dc_est = sign * (d_est_mean - c_est_mean)
        delta_drc_est = sign * (dr_est_mean - c_est_mean)
        delta_ddr_est = sign * (d_est_mean - dr_est_mean)
        delta_db_est = (sign * (d_est_mean - best_b_mean)
                        if not np.isinf(best_b_mean) else float('nan'))

        def _interp(delta, pos, neg):
            if np.isnan(delta):
                return 'insufficient data'
            return pos if delta > 0 else neg

        rep_result['deltas'] = {
            # ── Side 1: against the dataset's other representations ────────
            # Biased upper bound — keep for transparency, never present as effect.
            'D_minus_C': float(delta_dc) if not np.isnan(delta_dc) else float('nan'),
            # The selection inflation any same-dim matrix earns.
            'D_random_minus_C': float(delta_drc) if not np.isnan(delta_drc) else float('nan'),
            # The debiased new-rep contribution — THIS is the headline effect.
            'D_minus_D_random': float(delta_ddr) if not np.isnan(delta_ddr) else float('nan'),
            # Does the hybrid improve the new rep beyond its own best single model?
            'D_minus_B': float(delta_db) if not np.isnan(delta_db) else float('nan'),
            'D_minus_D_random_interpretation': _interp(
                delta_ddr,
                'new rep HELPS hybrid beyond a same-dim noise matrix',
                'apparent gain is selection noise (rep no better than random dims)'),
            'D_minus_B_interpretation': _interp(
                delta_db,
                'hybrid HELPS the new rep beyond its best single model',
                'hybrid does NOT beat the new rep alone'),
            # ── Side 2: against the established representations only ───────
            'D_established_minus_C_established': (
                float(delta_dc_est) if not np.isnan(delta_dc_est) else float('nan')),
            'D_random_established_minus_C_established': (
                float(delta_drc_est) if not np.isnan(delta_drc_est) else float('nan')),
            'D_established_minus_D_random_established': (
                float(delta_ddr_est) if not np.isnan(delta_ddr_est) else float('nan')),
            'D_established_minus_B': (
                float(delta_db_est) if not np.isnan(delta_db_est) else float('nan')),
            'D_established_minus_D_random_established_interpretation': _interp(
                delta_ddr_est,
                'new rep HELPS the established hybrid beyond a same-dim noise matrix',
                'apparent gain is selection noise (rep no better than random dims)'),
            'D_established_minus_B_interpretation': _interp(
                delta_db_est,
                'hybrid HELPS the new rep beyond its best single model',
                'hybrid does NOT beat the new rep alone'),
        }

        print(f"\n  --- DELTAS in {metric_name}, positive = improvement ---")
        print(f"    Side 1, against the dataset's other representations:")
        print(f"      D - C        (upper bound, biased):      {delta_dc:+.4f}")
        print(f"      D_random - C (selection-bias null):      {delta_drc:+.4f}")
        print(f"      D - D_random (DEBIASED new-rep gain):    {delta_ddr:+.4f}  "
              f"({rep_result['deltas']['D_minus_D_random_interpretation']})")
        print(f"      D - B        (hybrid helps new rep?):    {delta_db:+.4f}  "
              f"({rep_result['deltas']['D_minus_B_interpretation']})")
        print(f"    Side 2, against the established representations only:")
        print(f"      D_est - C_est        (upper bound, biased):  {delta_dc_est:+.4f}")
        print(f"      D_rand_est - C_est   (selection-bias null):  {delta_drc_est:+.4f}")
        print(f"      D_est - D_rand_est   (DEBIASED gain):        {delta_ddr_est:+.4f}  "
              f"({rep_result['deltas']['D_established_minus_D_random_established_interpretation']})")
        print(f"      D_est - B            (hybrid helps new rep?):{delta_db_est:+.4f}  "
              f"({rep_result['deltas']['D_established_minus_B_interpretation']})")

        # =================================================================
        # Controls: random Gaussian and row-shuffled
        # =================================================================
        print(f"\n  Controls for {rep_name}:")

        # Random Gaussian (same dims)
        X_random = _make_random_gaussian(X_new)
        ctrl_random_scores = _eval_single_rep_per_fold(
            X_random, labels, groups, ridge_factory, n_folds, mode, metric_fn)
        ctrl_random_mean = float(np.mean(ctrl_random_scores)) if ctrl_random_scores else float('nan')
        rep_result['controls']['random_gaussian'] = {
            'description': f'random Gaussian ({X_new.shape[1]}D)',
            'score_mean': ctrl_random_mean,
            'score_std': float(np.std(ctrl_random_scores)) if ctrl_random_scores else float('nan'),
            'fold_scores': [float(s) for s in ctrl_random_scores],
        }
        print(f"    Random Gaussian ({X_new.shape[1]}D): {metric_name} = {ctrl_random_mean:.4f}")

        # Row-shuffled
        X_shuffled = _make_row_shuffled(X_new)
        ctrl_shuffled_scores = _eval_single_rep_per_fold(
            X_shuffled, labels, groups, ridge_factory, n_folds, mode, metric_fn)
        ctrl_shuffled_mean = float(np.mean(ctrl_shuffled_scores)) if ctrl_shuffled_scores else float('nan')
        rep_result['controls']['row_shuffled'] = {
            'description': f'row-shuffled {rep_name}',
            'score_mean': ctrl_shuffled_mean,
            'score_std': float(np.std(ctrl_shuffled_scores)) if ctrl_shuffled_scores else float('nan'),
            'fold_scores': [float(s) for s in ctrl_shuffled_scores],
        }
        print(f"    Row-shuffled {rep_name}: {metric_name} = {ctrl_shuffled_mean:.4f}")

        # Signal above null (positive = rep is better than noise)
        if lower_is_better:
            signal_above_random = ctrl_random_mean - a_mean
            signal_above_shuffled = ctrl_shuffled_mean - a_mean
        else:
            signal_above_random = a_mean - ctrl_random_mean
            signal_above_shuffled = a_mean - ctrl_shuffled_mean
        rep_result['controls']['signal_above_random'] = float(signal_above_random)
        rep_result['controls']['signal_above_shuffled'] = float(signal_above_shuffled)
        # Both sides of each subtraction used this model, so the difference is
        # clean; naming it stops the number being read as a linear probe when it
        # is not one.
        rep_result['controls']['probe_model'] = probe_model
        print(f"    Signal above random: {signal_above_random:+.4f}")
        print(f"    Signal above shuffled: {signal_above_shuffled:+.4f}")

        per_rep_results[rep_name] = rep_result

    # =====================================================================
    # Cross-rep summary
    # =====================================================================
    print(f"\n{'='*70}")
    print("  FORWARD COMPATIBILITY SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Dataset: {dataset_name} | Metric: {metric_name} | "
          f"Domain: {domain} | n_folds: {n_folds}")
    print(f"  {'lower is better' if lower_is_better else 'higher is better'}")

    # One row per embedding model tested on this dataset. Every score is in
    # {metric_name}. Dim is that representation's number of features; Alloc% is
    # the share of the hybrid's feature budget the search took from it.
    # * D-Dr = D - D_random, the DEBIASED contribution. D - C is deliberately NOT
    #   in the win-rate summary: it is selection-biased positive and would
    #   overstate the effect.
    win_count_ddr = 0
    win_count_db = 0
    win_count_ddr_est = 0
    win_count_db_est = 0
    n_scored_ddr_est = 0
    n_scored_db_est = 0
    total_reps_tested = 0

    print(f"\n  Side 1 — the without-side is the dataset's other "
          f"representations, every other embedding model kept in.")
    print(f"  {'New Rep':<20} {'Dim':>5} {'B(best)':>9} {'C(no r)':>9} "
          f"{'D(+r)':>9} {'Drand':>9} {'D-Dr*':>8} {'D-B':>8} {'Alloc%':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")

    for rep_name, rr in sorted(per_rep_results.items()):
        conds = rr['conditions']
        deltas = rr['deltas']
        b_val = conds['B']['score_mean']
        c_val = conds['C']['score_mean']
        d_val = conds['D']['score_mean']
        dr_val = conds.get('D_random', {}).get('score_mean', float('nan'))
        ddr = deltas['D_minus_D_random']
        db = deltas['D_minus_B']
        alloc_pct = conds['D'].get('allocation_share', 0) * 100

        # Win counting on the DEBIASED effect. `ddr` and `db` are already
        # sign-normalised (positive = improvement, any metric direction), so
        # re-applying lower_is_better would invert RMSE/MAE/logloss domains.
        total_reps_tested += 1
        if not np.isnan(ddr) and ddr > 0:
            win_count_ddr += 1
        if not np.isnan(db) and db > 0:
            win_count_db += 1
        # Side 2 has its own denominators: on a dataset whose representations
        # are all embedding models the side-2 without-side is empty and these
        # are NaN, and counting those as losses would read as a result.
        ddr_est = deltas['D_established_minus_D_random_established']
        db_est = deltas['D_established_minus_B']
        if not np.isnan(ddr_est):
            n_scored_ddr_est += 1
            if ddr_est > 0:
                win_count_ddr_est += 1
        if not np.isnan(db_est):
            n_scored_db_est += 1
            if db_est > 0:
                win_count_db_est += 1

        print(f"  {rep_name:<20} {rr['dim']:>5} {b_val:>9.4f} {c_val:>9.4f} "
              f"{d_val:>9.4f} {dr_val:>9.4f} {ddr:>+8.4f} {db:>+8.4f} "
              f"{alloc_pct:>6.1f}%")

    # The same rows again with the second without-side (decided 2026-09-12).
    print(f"\n  Side 2 — the without-side is the established representations "
          f"only, every other embedding model held out.")
    print(f"  The established representations here ({len(established_reps)}): "
          f"{', '.join(sorted(established_reps)) or 'none'}")
    print(f"  {'New Rep':<20} {'Dim':>5} {'B(best)':>9} {'C_est':>9} "
          f"{'D_est(+r)':>9} {'Drand_est':>9} {'D-Dr*':>8} {'D-B':>8} {'Alloc%':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")
    for rep_name, rr in sorted(per_rep_results.items()):
        conds = rr['conditions']
        deltas = rr['deltas']
        b_val = conds['B']['score_mean']
        c_val = conds.get('C_established', {}).get('score_mean', float('nan'))
        d_val = conds.get('D_established', {}).get('score_mean', float('nan'))
        dr_val = conds.get('D_random_established', {}).get('score_mean', float('nan'))
        ddr = deltas['D_established_minus_D_random_established']
        db = deltas['D_established_minus_B']
        alloc_pct = conds.get('D_established', {}).get('allocation_share', 0) * 100
        print(f"  {rep_name:<20} {rr['dim']:>5} {b_val:>9.4f} {c_val:>9.4f} "
              f"{d_val:>9.4f} {dr_val:>9.4f} {ddr:>+8.4f} {db:>+8.4f} "
              f"{alloc_pct:>6.1f}%")

    if total_reps_tested > 0:
        print(f"\n  Win rates on the DEBIASED effect, not the biased D-C. The "
              f"bottom of each fraction counts the embedding models tested on "
              f"this dataset.")
        print(f"    Side 1, against the dataset's other representations:")
        print(f"      D > D_random (rep beats same-dim noise): {win_count_ddr}/{total_reps_tested} "
              f"({win_count_ddr/total_reps_tested:.0%})")
        print(f"      D > B (hybrid helps the new rep):        {win_count_db}/{total_reps_tested} "
              f"({win_count_db/total_reps_tested:.0%})")
        print(f"    Side 2, against the established representations only. The "
              f"bottom counts the embedding models this side produced a score "
              f"for, which is 0 when the dataset has no established "
              f"representation left:")
        if n_scored_ddr_est:
            print(f"      D_est > D_random_est:                    {win_count_ddr_est}/{n_scored_ddr_est} "
                  f"({win_count_ddr_est/n_scored_ddr_est:.0%})")
        else:
            print(f"      D_est > D_random_est:                    not scored on "
                  f"any of the {total_reps_tested} embedding models here")
        if n_scored_db_est:
            print(f"      D_est > B:                               {win_count_db_est}/{n_scored_db_est} "
                  f"({win_count_db_est/n_scored_db_est:.0%})")
        else:
            print(f"      D_est > B:                               not scored on "
                  f"any of the {total_reps_tested} embedding models here")

    # Controls summary
    print(f"\n  Controls:")
    print(f"  {'New Rep':<20} {'A(real)':>9} {'Gaussian':>9} {'Shuffled':>9} "
          f"{'vs Random':>10} {'vs Shuf':>10}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")
    for rep_name, rr in sorted(per_rep_results.items()):
        a_val = rr['conditions']['A']['score_mean']
        rand_val = rr['controls']['random_gaussian']['score_mean']
        shuf_val = rr['controls']['row_shuffled']['score_mean']
        sig_r = rr['controls']['signal_above_random']
        sig_s = rr['controls']['signal_above_shuffled']
        print(f"  {rep_name:<20} {a_val:>9.4f} {rand_val:>9.4f} {shuf_val:>9.4f} "
              f"{sig_r:>+10.4f} {sig_s:>+10.4f}")

    # Build output
    output = make_structured_output(
        experiment_id='c13_forward_compatibility',
        dataset=dataset_name,
        data=data,
        config=config,
        results={
            'per_rep': per_rep_results,
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
                # The two without-sides, by name.
                'without_side_1': 'the dataset\'s other representations',
                'without_side_2_reps': sorted(established_reps.keys()),
                # Without these two, a result carrying one candidate reads as
                # "the list holds one" rather than "one of N was buildable here".
                'candidates_not_loaded': sorted(missing_candidates),
                'candidates_requested': len(candidate_reps) + len(missing_candidates),
                # Headline win rates are on the DEBIASED effect. D-C is retained
                # per-rep in deltas as an in-sample upper bound only.
                'win_rate_D_vs_D_random': win_count_ddr / total_reps_tested if total_reps_tested > 0 else None,
                'win_rate_D_vs_B': win_count_db / total_reps_tested if total_reps_tested > 0 else None,
                'win_rate_D_established_vs_D_random_established': (
                    win_count_ddr_est / n_scored_ddr_est if n_scored_ddr_est > 0 else None),
                'win_rate_D_established_vs_B': (
                    win_count_db_est / n_scored_db_est if n_scored_db_est > 0 else None),
                # The side-2 win rates count only the embedding models side 2
                # produced a score for; these say how many that was.
                'n_scored_D_established_vs_D_random_established': n_scored_ddr_est,
                'n_scored_D_established_vs_B': n_scored_db_est,
                'total_new_reps_tested': total_reps_tested,
                'metric_note': (
                    'D-C is a selection-biased upper bound; the unbiased new-rep '
                    'effect is D - D_random. The same pair of numbers exists on '
                    'side 2 as D_established - C_established and D_established - '
                    'D_random_established, where every other embedding model is '
                    'out of the without-side. No significance tests are reported '
                    '(underpowered on non-independent CV folds).'
                ),
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
