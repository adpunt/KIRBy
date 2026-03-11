# KIRBy Splitting Strategy Audit

**Date:** 2026-03-10
**Purpose:** Verify every domain's splitting strategy against published best practices. Random splits are forbidden by the research group -- every domain must use a defensible non-random split.

---

## Table of Contents

1. [Molecular QSAR](#1-molecular-qsar)
2. [Antibody DMS (Intra-family)](#2-antibody-dms-intra-family)
3. [Antibody Inter-family](#3-antibody-inter-family)
4. [DDG (Binding Affinity Change)](#4-ddg-binding-affinity-change)
5. [DTA (Drug-Target Affinity)](#5-dta-drug-target-affinity)
6. [TCGA (Cancer Genomics)](#6-tcga-cancer-genomics)
7. [Reaction Yields](#7-reaction-yields)
8. [Summary of Required Changes](#8-summary-of-required-changes)

---

## 1. Molecular QSAR

### KIRBy's current split
Murcko scaffold `GroupKFold` via `assign_scaffold_groups()` in `tests/find_best_hybrid_molecular.py`. Each molecule gets a group ID based on its Bemis-Murcko scaffold. All molecules sharing a scaffold stay in the same fold.

### Published approaches

#### A. Scaffold split (DeepChem / MoleculeNet standard)
- **Repo:** [deepchem/deepchem/splits/splitters.py](https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py)
- **Code:** `ScaffoldSplitter` computes Bemis-Murcko scaffolds, sorts scaffold groups by size (largest first), and fills train/val/test buckets in order. This is a single 80/10/10 split, not cross-validated.
- **Implementation:** KIRBy's `GroupKFold` variant is *more rigorous* than DeepChem's single split because it does K-fold CV with scaffold groups, meaning every molecule is tested exactly once while respecting scaffold boundaries.

#### B. Butina clustering split
- **Repo:** [PatWalters/useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils) -- `split_utils.py`
- **Code:** `get_butina_clusters(smiles_list)` computes Morgan fingerprints, runs RDKit's Butina clustering, returns cluster IDs. These cluster IDs feed into `GroupKFoldShuffle`.
- **Why it matters:** Pat Walters' [blog post (Nov 2024)](http://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html) argues Butina splits are his current preference. Butina clusters based on Tanimoto similarity rather than scaffold identity, catching cases where different scaffolds are structurally very similar.
- **Data from Ballester group:** Average max Tanimoto similarity between test and nearest training neighbor across 60 NCI-60 datasets: random=0.717, scaffold=0.625, Butina=0.417, UMAP=0.375. Scaffold is much closer to random than to Butina/UMAP.

#### C. UMAP clustering split
- **Repo:** [Rong830/UMAP_split_for_VS](https://github.com/Rong830/UMAP_split_for_VS)
- **Paper:** [UMAP-based clustering split for rigorous evaluation](https://link.springer.com/article/10.1186/s13321-025-01039-8)
- **Code:** Morgan FP -> UMAP 2D projection -> HDBSCAN clustering -> GroupKFold on cluster IDs.
- **Key finding:** With UMAP split, only GEM achieved a non-zero hit rate (11.9%) while LR and RF dropped to 0%. This is the hardest split but also the most realistic.

#### D. "Scaffold Splits Overestimate Virtual Screening Performance" (2024)
- **Paper:** [arxiv.org/abs/2406.00873](https://arxiv.org/abs/2406.00873)
- **Repo:** [ScaffoldSplitsOverestimateVS](https://github.com/ScaffoldSplitsOverestimateVS/ScaffoldSplitsOverestimateVS)
- **Key finding:** Bemis-Murcko scaffolds can be almost identical while being technically different. The scaffold split gives a false sense of out-of-distribution testing. This paper recommends Butina or UMAP splits instead.

#### E. Greg Landrum's RDKit blog (May 2024)
- **Post:** [The problem(s) with scaffold splits, part 1](https://greglandrum.github.io/rdkit-blog/posts/2024-05-31-scaffold-splits-and-murcko-scaffolds1.html)
- **Key point:** Looking across 7,148 Ki assays, the median number of Murcko scaffolds per assay is only 12. Many scaffolds differ by trivial substitutions. A Murcko decomposition is "likely to be less wrong when partitioning datasets of diverse compounds that don't contain SAR series."

#### F. Temporal split
- **Paper:** [Sheridan 2013, JCIM](https://pubs.acs.org/doi/abs/10.1021/ci400084k) -- "Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction"
- **Key finding:** Time-split provides the most realistic estimate of prospective prediction. Data is split by measurement date into K folds. Not applicable for most public benchmarks (no timestamps), but gold standard for pharma.

### Assessment for KIRBy

**Current status: ACCEPTABLE but not optimal.**

Murcko scaffold GroupKFold is the field standard and defensible for a PhD thesis. It is used by MoleculeNet, TDC, and nearly every published QSAR benchmark.

**Known weakness:** Scaffold splits overestimate performance vs. Butina/UMAP splits (Ballester 2024, Landrum 2024). However, this is well-known and your research group likely expects scaffold splitting.

**Recommendation:** Keep Murcko scaffold GroupKFold as the primary split. Optionally add Butina clustering as a supplementary comparison if a reviewer asks. No change required.

---

## 2. Antibody DMS (Intra-family)

### KIRBy's current split
Identity `GroupKFold` on heavy chain sequence hash. In `tests/antibody_baseline_survey.py::_identity_groups()`:
```python
def _identity_groups(seqs, min_groups=3):
    unique = {}
    groups = np.zeros(len(seqs), dtype=int)
    for i, s in enumerate(seqs):
        if s not in unique:
            unique[s] = len(unique)
        groups[i] = unique[s]
    n_groups = len(unique)
    # Fallback: deterministic hash bucketing if too few groups
    if n_groups < min_groups:
        n_buckets = max(min_groups, len(seqs) // 5)
        groups = np.array([
            int(hashlib.sha256(s.encode()).hexdigest(), 16) % n_buckets
            for s in seqs
        ])
```

### THE PROBLEM: Identity split is BROKEN for single-mutation DMS data

In a DMS experiment (e.g., FLAb intra-family datasets like `flab_expression_koenig`, `flab_binding_koenig`), every variant differs from the wildtype by exactly 1 mutation. This means:

- **Every single-mutant has a unique sequence** -> every variant gets its own group
- **n_groups == n_samples** -> `GroupKFold` degenerates into `KFold` with random assignment
- **Result:** The "identity-based" split is effectively a RANDOM split

This is exactly the scenario the hash-bucketing fallback catches (when `n_groups < min_groups`), but the fallback is just SHA256 hash bucketing -- still essentially random with respect to biology.

### Published approaches: ProteinGym and FLIP

#### A. ProteinGym supervised benchmark (NeurIPS 2023)
- **Repo:** [OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym)
- **Paper:** [ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/)
- **Three splitting strategies for supervised DMS prediction:**
  1. **`fold_random_5`**: Random 5-fold. Easiest. Used as baseline.
  2. **`fold_modulo_5`**: Every 5th residue position goes to the same fold. Tests generalization across scattered positions.
  3. **`fold_contiguous_5`**: Protein split into 5 contiguous segments. Tests generalization to entirely new sequence regions.
- **Key insight:** ProteinGym provides pre-computed fold assignments in `cv_folds_singles_substitutions.zip`. The modulo and contiguous splits are biologically motivated: they test whether a model can predict fitness of mutations at *positions it has never seen*.

#### B. FLIP benchmark (NeurIPS 2021)
- **Repo:** [J-SNACKKB/FLIP](https://github.com/J-SNACKKB/FLIP)
- **Paper:** [FLIP: Benchmark tasks in fitness landscape inference for proteins](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2b44928ae11fb9384c4cf38708677c48-Abstract-round2.html)
- **Split types:**
  1. **Sampled:** Random sampling (for discourse only, "rarely observed in practice")
  2. **Position-based:** Train/test on variants with mutations at different positions
  3. **Low vs. high:** Train on low-fitness, test on high-fitness
  4. **Two vs. many:** Train on double mutants, test on higher-order
- **Key quote:** Position splits "test ability to generalize to previously unperturbed positions targeted in subsequent engineering rounds"

#### C. Kermut (NeurIPS 2024 Spotlight)
- **Repo:** [petergroth/kermut](https://github.com/petergroth/kermut)
- **Paper:** [Kermut: Composite kernel regression for protein variant effects](https://proceedings.neurips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf)
- **Uses ProteinGym's three splits.** Reports results across all three. Shows "significant gains in the challenging modulo and contiguous settings."

### Assessment for KIRBy

**Current status: BROKEN for DMS data. The identity split degenerates to random for single-mutation datasets.**

For intra-family DMS datasets (most FLAb datasets), every variant has a unique sequence, so identity grouping gives n_groups == n_samples, which means GroupKFold == random KFold.

**Recommendation: MUST CHANGE.** Implement ProteinGym-style position-based splitting:

1. **`fold_modulo_5`**: Assign mutations to folds based on `mutation_position % 5`. Every 5th position in the sequence gets the same fold. This ensures that mutations at position 27 (fold 2) are never in the same fold as mutations at position 28 (fold 3).

2. **`fold_contiguous_5`**: Split the protein into 5 contiguous segments. All mutations in segment 1 go to fold 1, etc.

3. For **multi-mutant** or **inter-family** datasets (e.g., `flab_thermo_jain`, `flab_aggregation_jain`), identity grouping is fine because different antibodies do have different sequences.

**Implementation sketch:**
```python
def assign_position_groups_modulo(mutations, n_folds=5):
    """ProteinGym-style modulo split for DMS data."""
    groups = np.array([mut[0]['pos'] % n_folds for mut in mutations])
    return groups

def assign_position_groups_contiguous(mutations, seq_len, n_folds=5):
    """ProteinGym-style contiguous split for DMS data."""
    segment_size = seq_len / n_folds
    groups = np.array([int(mut[0]['pos'] / segment_size) for mut in mutations])
    return groups
```

The key question is: does KIRBy's FLAb loader expose mutation positions? Currently `flab.py` returns full sequences but does not parse which position was mutated. This would need to be added by aligning each variant to the wildtype.

---

## 3. Antibody Inter-family

### KIRBy's current split
Identity `GroupKFold` on heavy chain sequences. For inter-family datasets (e.g., `flab_thermo_jain`, `flab_aggregation_jain`, `abcov_kd`), different antibodies have genuinely different sequences.

### Assessment

**Current status: ACCEPTABLE.** Unlike DMS data, inter-family datasets contain antibodies from different lineages/germlines. Identity grouping on the full VH sequence ensures that identical antibodies (or antibodies with very minor differences) stay together. This is analogous to scaffold splitting for molecules.

**Potential improvement:** Could group by germline V-gene rather than exact sequence identity, which would be a stricter split. But this is not standard practice and the current approach is defensible.

**Recommendation:** No change required.

---

## 4. DDG (Binding Affinity Change)

### KIRBy's current split
PDB complex `GroupKFold` via `assign_complex_groups()` in `ddg_combined.py`. All mutations from the same PDB complex stay in the same fold.

### Published approaches

#### A. Bushuiev et al. 2024 -- "Revealing data leakage in protein interaction benchmarks"
- **Paper:** [arxiv.org/abs/2404.10457](https://arxiv.org/abs/2404.10457)
- **Repo:** [anton-bushuiev/PPIRef](https://github.com/anton-bushuiev/PPIRef)
- **Notebook:** [revealing_data_leakage_iclr_gem_2024.ipynb](https://github.com/anton-bushuiev/PPIRef/blob/main/notebooks/revealing_data_leakage_iclr_gem_2024.ipynb)

**The 56% leakage claim:**
When using PDB code-based splits on SKEMPI, iDist (a scalable interface structural similarity algorithm) detects that **56% of test protein-protein interfaces have near-duplicates in the training set**. The problem is that different PDB codes can contain structurally near-identical interfaces (e.g., the same antibody-antigen pair crystallized under different conditions, or homologous complexes).

When Bushuiev et al. use domain-expertise grouping based on 3D interface structural similarity, leakage drops to **0%**.

**Why PDB-based splitting leaks:**
1. Same complex crystallized multiple times -> different PDB IDs
2. Homologous complexes (e.g., same antibody with different antigens from same family)
3. Engineered variants of the same complex

**Recommended fix:** Split based on 3D structural similarity of interfaces using iDist or similar.

#### B. CATH-ddG (Bioinformatics 2025)
- **Repo:** [ak422/CATH-ddG](https://github.com/ak422/CATH-ddG)
- **Paper:** [CATH-ddG: towards robust mutation effect prediction on PPI out of CATH homologous superfamily](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i362/8199350)

**Three levels of data leakage identified:**
1. **Mutation-level:** Same mutation appearing in train and test
2. **Protein-level:** Same protein (different mutations) in both splits
3. **Complex-level:** Same complex structure in both splits

CATH-ddG proposes splitting by CATH homologous superfamily, ensuring that no protein in the test set shares a CATH superfamily with any protein in the training set. This is much stricter than PDB-based splitting.

#### C. DDGb_bias (Pucci et al.)
- **Repo:** [3BioCompBio/DDGb_bias](https://github.com/3BioCompBio/DDGb_bias)
- Shows that once leakage is corrected, deep learning predictors lag behind physics-based methods like Flex ddG.

### Assessment for KIRBy

**Current status: AT RISK but likely acceptable given dataset size.**

KIRBy's PDB complex GroupKFold is the same strategy that Bushuiev et al. showed has 56% leakage on full SKEMPI. However:

1. KIRBy filters SKEMPI to Ab-Ag only (~hundreds of entries, not thousands)
2. KIRBy deduplicates with AB-Bind using normalized mutation keys
3. The resulting dataset has ~20 unique PDB complexes, which is already quite small for cross-validation

**The real question:** Do any of KIRBy's ~20 Ab-Ag PDB complexes share CATH superfamilies or have structurally similar interfaces? Given that they are all antibody-antigen complexes, the answer is almost certainly yes -- antibody Fabs share the immunoglobulin fold.

**Recommendation:** Acknowledge this limitation in the thesis. The dataset is too small to implement proper CATH superfamily splitting (would likely leave 1-2 samples per fold). Document that PDB-based splitting may overestimate performance by ~56% (Bushuiev 2024). Consider reporting both PDB-grouped results and leave-one-complex-out results (already effectively what GroupKFold does with ~20 groups and 5 folds).

**If a reviewer challenges this:** Point to the deduplication and Ab-Ag filtering as mitigations. The 56% figure applies to the full SKEMPI with all protein types, not just Ab-Ag.

---

## 5. DTA (Drug-Target Affinity)

### KIRBy's current split
Cold-target `GroupKFold` via `assign_cold_target_groups()` in `dta.py`. Groups by protein ID so no protein appears in both train and test.

### Published approaches

#### A. TDC (Therapeutics Data Commons)
- **Docs:** [tdcommons.ai/functions/data_split](https://tdcommons.ai/functions/data_split/)
- **Repo:** [mims-harvard/TDC](https://github.com/mims-harvard/TDC)
- **Three cold-start splits:**
  1. **Cold drug:** Group by drug molecule. Tests generalization to unseen drugs.
  2. **Cold target:** Group by protein target. Tests generalization to unseen proteins.
  3. **Cold pair:** Neither drug nor target in test appears in training.

#### B. DeepPurpose
- **Repo:** [kexinhuang12345/DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose)
- **Code in `utils.py`:**
  ```python
  data_process(X_drugs, X_targets, y, split_method='cold_drug')
  data_process(X_drugs, X_targets, y, split_method='cold_protein')
  ```
- Cold drug: Davis split into 54/7/7 non-overlapping drugs
- Cold target: Davis split into 354/44/44 non-overlapping proteins

#### C. DeepDTA (Bioinformatics 2018)
- **Paper:** [DeepDTA: deep drug-target binding affinity prediction](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245)
- Uses 5-fold cross-validation with no explicit cold-start enforcement (essentially random within folds). This is the *wrong* way to do it.

### Assessment for KIRBy

**Current status: GOOD.** Cold-target GroupKFold is the standard approach recommended by TDC and used by most recent DTA papers. It tests the hardest practical scenario: can the model predict binding to a protein it has never seen?

- Davis: 442 proteins -> plenty of groups for 5-fold
- KIBA: similar structure

**Recommendation:** No change required. KIRBy's cold-target split matches TDC's recommended strategy.

**Optional enhancement:** Could also report cold-drug results (group by drug molecule) as a supplementary analysis. This tests a different generalization axis.

---

## 6. TCGA (Cancer Genomics)

### KIRBy's current split
TSS (Tissue Source Site) `GroupKFold` via `_extract_tss_groups()` in `tcga.py`. Groups by the 2-character TSS code from the TCGA barcode (positions 5-6), which identifies the contributing institution/hospital.

### Rationale
TSS grouping prevents batch-effect leakage: samples from the same hospital were processed together, using the same protocols, equipment, and personnel. If a model memorizes hospital-specific technical artifacts rather than biological signal, TSS grouping will expose this.

### Published support

#### A. Batch effect literature
- [Assessing and mitigating batch effects in large-scale omics studies (Genome Biology 2024)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03401-9)
- In a clinical trial, batch effects from a change in RNA-extraction solution caused incorrect classification for 162 patients, 28 of whom received wrong chemotherapy.

#### B. TCGA acquisition site bias
- [Biased data, biased AI: deep networks predict the acquisition site of TCGA images (Diagnostic Pathology 2023)](https://diagnosticpathology.biomedcentral.com/articles/10.1186/s13000-023-01355-3)
- CNNs can identify which hospital a TCGA image came from without being explicitly trained to do so, demonstrating strong site-specific artifacts.

### Assessment for KIRBy

**Current status: GOOD.** TSS GroupKFold is the correct approach for TCGA data. It directly addresses the most common source of leakage in genomics: batch effects from different sample processing sites.

KIRBy also applies ComBat batch correction across TSS batches (line 250 of `tcga.py`), which is an additional mitigation.

**Recommendation:** No change required.

---

## 7. Reaction Yields

### KIRBy's current split
**RANDOM 70/30 x 10 splits.** Both Buchwald-Hartwig and Suzuki-Miyaura use random train/test partitions.

From `reaction_yields.py` lines 145-151:
```python
# Random 70/30 splits (matches original protocol)
splits = []
for i in range(n_splits):
    rng = np.random.RandomState(42 + i)
    perm = rng.permutation(n)
    n_train = int(0.7 * n)
    splits.append((perm[:n_train], perm[n_train:]))
```

### **THIS IS THE PROBLEM. MUST CHANGE.**

### Published approaches

#### A. Original Doyle/Dreher paper (Science 2018)
- **Paper:** [Predicting reaction performance in C-N cross-coupling using machine learning](https://pubmed.ncbi.nlm.nih.gov/29449509/)
- **Dataset:** 4608 BH reactions: 15 aryl halides x 4 ligands x 3 bases x 23 isoxazole additives (+ 2 controls)
- **Original split:** Random 70/30. This was criticized.

#### B. On the Use of Real-World Datasets (Doyle group, Chem. Sci. 2023)
- **Paper:** [On the use of real-world datasets for reaction yield prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10189898/)
- **Key finding:** Leave-one-group-out analysis for BH dataset shows "modest degradation" when each additive is left out. They performed **leave-one-additive-out** testing across all 23 isoxazoles.
- **Multiple out-of-sample partitions tested:** leave-one-aryl-halide-out, leave-one-base-out, leave-one-ligand-out, leave-one-additive-out.

#### C. Schwaller et al. (Yield-BERT, ML:ST 2021)
- **Repo:** [rxn4chemistry/rxn_yields](https://github.com/rxn4chemistry/rxn_yields)
- **Split:** Random 70/30 x 10 for both BH and Suzuki. This is what KIRBy copies.
- **Problem:** This is the same random split that has been criticized.

#### D. "When Yield Prediction Does Not Yield Prediction" (JCIM 2024)
- **Paper:** [When Yield Prediction Does Not Yield Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01524)
- **Repo:** [v-in-cube/YieldnotYield](https://github.com/v-in-cube/YieldnotYield)
- **11 different data partitioning schemes for BH dataset, including 4 out-of-sample partitions based on additives.**
- Tested leave-one-additive-out, leave-one-base-out, leave-one-ligand-out, leave-one-aryl-halide-out.
- RMSE of 17.4% and R^2 of 0.51 in leave-one-additive-out (vs R^2 ~0.92 with random split).

#### E. Bradshaw et al. (ACS Cent. Sci. 2025)
- **Repo:** [john-bradshaw/rxn-splits](https://github.com/john-bradshaw/rxn-splits)
- **Paper:** [Challenging reaction prediction models to generalize to novel chemistry](https://pubs.acs.org/doi/10.1021/acscentsci.5c00055)
- For general reaction prediction (not yield), they use time splits and author splits. Shows random splits are "overly optimistic."

### Which component should be the group key?

The BH dataset has 4 variable components: aryl halide (15), ligand (4), base (3), additive (23).

| Group key | N groups | Practical meaning |
|-----------|----------|-------------------|
| Additive  | 23       | Can we predict yield with a new isoxazole? Most groups, best for CV |
| Aryl halide | 15     | Can we predict yield with a new substrate? Chemically meaningful |
| Ligand    | 4        | Too few groups for 5-fold CV |
| Base      | 3        | Too few groups for 5-fold CV |

**The additive is the best group key for the BH dataset** because:
1. 23 groups is enough for robust 5-fold GroupKFold
2. The original paper's key claim was that the model could predict which additives would poison the reaction
3. Leave-one-additive-out is the most established out-of-sample test for this dataset

For the **Suzuki dataset** (5760 reactions: 15 electrophiles x 12 ligands x 8 bases x 4 solvents):
- The Suzuki dataset structure is a full factorial of substrates x conditions
- **Ligand** (12 groups) or **electrophile** (15 groups) are the best group keys
- The AbbVie Suzuki library paper tested **core-based splits** (grouping by shared reaction core) and **monomer-based splits**

### Assessment for KIRBy

**Current status: UNACCEPTABLE. Random splitting is used and must be changed.**

**Recommendation: MUST CHANGE.**

**For Buchwald-Hartwig:**
- Primary: Group by **additive** (23 groups, 5-fold GroupKFold)
- The `components` dict already contains `'additive'` key per reaction
- Implementation:
  ```python
  additives = [c['additive'] for c in data['components']]
  unique_add = sorted(set(additives))
  add_to_id = {a: i for i, a in enumerate(unique_add)}
  groups = np.array([add_to_id[a] for a in additives])
  ```

**For Suzuki-Miyaura:**
- Primary: Group by **ligand** or by **electrophile** (whichever has more unique values)
- Need to parse components from reaction SMILES (the `_parse_rxn_components` function already does this, but the Suzuki loader doesn't expose named component keys like BH does)
- Alternative: Group by product scaffold (Murcko scaffold of the product molecule)

---

## 8. Summary of Required Changes

| Domain | Current split | Status | Action needed |
|--------|--------------|--------|---------------|
| Molecular QSAR | Murcko scaffold GroupKFold | ACCEPTABLE | None |
| Antibody DMS (intra-family) | Identity GroupKFold | **BROKEN** | **Implement position-based splitting (modulo/contiguous)** |
| Antibody inter-family | Identity GroupKFold | ACCEPTABLE | None |
| DDG | PDB complex GroupKFold | AT RISK | Acknowledge 56% leakage risk in thesis; dataset too small to fix |
| DTA | Cold-target GroupKFold | GOOD | None |
| TCGA | TSS GroupKFold | GOOD | None |
| Reaction yields | **Random 70/30 x 10** | **UNACCEPTABLE** | **Change to leave-one-component-out GroupKFold** |

### Priority 1 (must fix before submission):
1. **Reaction yields:** Replace random splits with additive-grouped (BH) and ligand-grouped (Suzuki) GroupKFold
2. **Antibody DMS:** Replace identity grouping with position-based splitting for intra-family datasets

### Priority 2 (should acknowledge):
3. **DDG:** Document the Bushuiev 56% leakage finding. Note that KIRBy's Ab-Ag-only filtering and deduplication partially mitigate this.

### Priority 3 (optional enhancements):
4. **Molecular:** Could add Butina clustering as supplementary split comparison
5. **DTA:** Could add cold-drug results alongside cold-target

---

## 9. Specific Investigative Answers

### Q: Does identity GroupKFold on sequence hash work for DMS data where sequences differ by 1 mutation?

**No.** In DMS data, every single-mutant variant has a unique sequence. Identity grouping creates N groups for N samples, which means GroupKFold degenerates to random KFold. The hash-bucketing fallback in `_identity_groups()` (activated when `n_groups < min_groups`) is also essentially random. This is why ProteinGym and FLIP use position-based splits instead.

### Q: What is the 56% leakage claim for SKEMPI PDB-based splitting?

Bushuiev et al. (ICLR GEM Workshop 2024) developed iDist, a scalable algorithm for comparing 3D protein-protein interfaces. They found that when SKEMPI is split by PDB code, **56% of test protein-protein interfaces have structurally near-identical counterparts in the training set** (identified by iDist with 99% precision and 97% recall vs. iAlign). When domain-expertise grouping (based on interface structural similarity) is used instead, leakage drops to 0%. This means that PDB-code-based splitting allows models to memorize structurally redundant interfaces rather than learning generalizable patterns.

### Q: For BH reactions, which component should be the group key?

**Additive (isoxazole)** is the best group key because:
- 23 unique additives = enough groups for robust 5-fold CV
- The original Doyle/Dreher paper's central claim was predicting which additives poison the reaction
- Leave-one-additive-out is the most established OOS test (Doyle 2023, YieldnotYield 2024)
- Performance drops dramatically: R^2 ~0.92 (random) -> R^2 ~0.51 (leave-additive-out)

Ligand (4 groups) and base (3 groups) have too few unique values for meaningful cross-validation.

### Q: Is there a universal non-random split strategy?

**No.** Each domain requires its own splitting strategy based on the specific leakage risks:
- Molecules: scaffold/cluster-based (prevent memorizing SAR series)
- Proteins/DMS: position-based (prevent memorizing position-specific effects)
- DDG: interface similarity-based (prevent structural redundancy)
- DTA: cold-target/cold-drug (prevent memorizing target-specific patterns)
- Genomics: batch/site-based (prevent memorizing technical artifacts)
- Reactions: component-based (prevent memorizing reagent-specific effects)

The common principle is: **group by the axis along which you want to generalize**, then use GroupKFold to ensure no group appears in both train and test.

---

## Sources

- [DeepChem splitters documentation](https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html)
- [DeepChem splitters.py](https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py)
- [Pat Walters -- Some Thoughts on Splitting Chemical Datasets (Nov 2024)](http://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html)
- [Pat Walters -- useful_rdkit_utils split_utils](https://useful-rdkit-utils.readthedocs.io/en/latest/split_utils.html)
- [Greg Landrum -- The problem(s) with scaffold splits, part 1 (May 2024)](https://greglandrum.github.io/rdkit-blog/posts/2024-05-31-scaffold-splits-and-murcko-scaffolds1.html)
- [Scaffold Splits Overestimate Virtual Screening Performance (2024)](https://arxiv.org/abs/2406.00873)
- [UMAP-based clustering split (J Cheminform 2025)](https://link.springer.com/article/10.1186/s13321-025-01039-8)
- [ProteinGym (NeurIPS 2023)](https://github.com/OATML-Markslab/ProteinGym)
- [FLIP (NeurIPS 2021)](https://github.com/J-SNACKKB/FLIP)
- [Kermut (NeurIPS 2024 Spotlight)](https://github.com/petergroth/kermut)
- [Bushuiev et al. -- Revealing data leakage in protein interaction benchmarks (ICLR GEM 2024)](https://arxiv.org/abs/2404.10457)
- [PPIRef repository with leakage analysis notebook](https://github.com/anton-bushuiev/PPIRef)
- [CATH-ddG (Bioinformatics 2025)](https://github.com/ak422/CATH-ddG)
- [TDC -- Dataset Splits](https://tdcommons.ai/functions/data_split/)
- [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose)
- [Doyle/Dreher -- Predicting reaction performance (Science 2018)](https://pubmed.ncbi.nlm.nih.gov/29449509/)
- [On the use of real-world datasets for reaction yield prediction (Chem Sci 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10189898/)
- [rxn4chemistry/rxn_yields](https://github.com/rxn4chemistry/rxn_yields)
- [When Yield Prediction Does Not Yield Prediction (JCIM 2024)](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01524)
- [Bradshaw et al. -- rxn-splits for OOD evaluation](https://github.com/john-bradshaw/rxn-splits)
- [Sheridan 2013 -- Time-Split Cross-Validation (JCIM)](https://pubs.acs.org/doi/abs/10.1021/ci400084k)
- [Biased data, biased AI: deep networks predict the acquisition site of TCGA images (Diagn Pathol 2023)](https://diagnosticpathology.biomedcentral.com/articles/10.1186/s13000-023-01355-3)
