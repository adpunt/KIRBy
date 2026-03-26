# OPIG Collaboration Roadmap

Roadmap for integrating OPIG lab publications into KIRBy as new representations,
datasets/tasks, and model/evaluation enhancements. Each entry links back to the
original paper/authors for reference. Implementation will use published code where
available.

Source document: OPIG molecular-ML catalog (2018-2026), 122 entries across 14 sections.

---

## Category A: New Representations (just add the rep)

These are self-contained representation methods that slot into KIRBy's existing
rep→model pipeline with minimal architectural changes.

### A1. Sort & Slice (collision-free ECFPs)

| | |
|---|---|
| **Paper** | *Sort & Slice: A Simple and Superior Alternative to Hash-Based Folding for ECFPs* (#31 in catalog) |
| **Authors** | Markus Dablander, Hanser, Lambiotte, Garrett Morris |
| **Year** | 2024 |
| **What it is** | An improved pooling method for ECFPs. Replaces hash-based folding (which causes bit collisions) with frequency-based substructure selection. Sorts all detected ECFP substructures by training-set frequency, slices to keep the top L. Output: L-dimensional binary fingerprint, same format as standard ECFP, zero collisions. |
| **Input** | SMILES (same as existing ECFP pipeline) |
| **Output dim** | Configurable (e.g. 1024, 2048) |
| **Code** | [github.com/oxpig/ECFP-Sort-and-Slice](https://github.com/oxpig/ECFP-Sort-and-Slice) |
| **KIRBy domains** | Molecular, DTA (ligand side), Reaction (substrate/product side) |
| **Scope** | Small — drop-in replacement for hashed ECFPs. Needs per-fold fitting (frequency counts are training-set-dependent, like SNS). Wire into rep_factory. |
| **Status** | **Molecular: DONE** (Phase 3B). **DTA: DONE** — `drug_sns` wired into `dta_baseline_survey.py` (factory rep with per-fold fitting on unique training drugs) and `find_best_hybrid_dta.py` (in `FACTORY_REPS` + `full` preset, 40 reps total). **Reaction: SKIPPED** — all fingerprint reps produce R²≈-0.03 under leave-ligand-out for Suzuki; adding another fingerprint won't help. |
| **Notes** | Should have been included already. Consistently outperforms hash-based folding across models, splits, and tasks in the original paper. Natural hybrid component with mordred, learned reps. |

### A2. CaLM (codon-level protein language model)

| | |
|---|---|
| **Paper** | *Codon Language Embeddings Provide Strong Signals for Protein Engineering* (#62 in catalog) |
| **Authors** | Carlos Outeiral, Charlotte Deane |
| **Year** | 2024 (Nature Machine Intelligence) |
| **What it is** | BERT-style transformer (86M params, 12 layers) trained on ~9M non-redundant coding DNA sequences at the **codon** level. Produces per-token and per-sequence embeddings that capture information inaccessible to amino acid-level PLMs (codon usage bias, expression signals). Outperforms ESM, ProtT5, ProtBERT on abundance, species, melting point tasks. |
| **Input** | DNA coding sequences (NOT amino acid sequences — different input modality) |
| **Output dim** | 768D (BERT hidden size, per-sequence mean-pooled) |
| **Code** | [github.com/oxpig/CaLM](https://github.com/oxpig/CaLM) |
| **KIRBy domains** | Protein (if coding DNA available), Antibody (if coding DNA available) |
| **Scope** | Medium — requires DNA coding sequences as input. Need to check whether KIRBy's current protein/antibody datasets have corresponding coding DNA, or whether reverse-translation is viable. If coding DNA is available for OAS/SAbDab entries, this is straightforward. If not, scope grows. |
| **Status** | **DEFERRED** — cDNA availability investigation complete (2026-03-14). See findings below. |
| **Notes** | Genuinely novel modality. Captures codon usage bias, which amino acid LMs fundamentally cannot. The hybrid value proposition is strong: CaLM captures expression/abundance signals, ESM captures structural/evolutionary signals — combining them tests whether orthogonal information sources help. |

#### A2 cDNA Availability Investigation (2026-03-14)

**Input format:** CaLM takes raw DNA codon strings (`model.embed_sequence('ATGGTATAGAGGCATTGA')`). No reverse-translation or cDNA retrieval tools provided. Install: `git clone + setup.py install`.

**DTA proteins (Davis/KIBA/BindingDB):**
- **KIBA** (229 proteins): Uses UniProt accessions directly → EMBL/ENA CDS mapping straightforward. **~95-100% coverage.**
- **DAVIS** (442 kinases): Uses gene names (incl. mutant forms like `ABL1(E255K)`, ~300 unique genes). Wildtype → UniProt → EMBL feasible. **~90-95% for wildtype**, but mutant/phospho variants would use wildtype CDS (codon bias unchanged by point mutations, so this is acceptable).
- **BindingDB_Kd** (~1,400 proteins): Mixed IDs (some UniProt, some NaN/hash). **~70-85% coverage** depending on ID quality.
- **Feasible** but requires building a cDNA retrieval pipeline (UniProt REST API → EMBL CDS → cache). CaLM's own paper uses exactly this pipeline.

**Antibody datasets (AbAgym, SAbDab, SKEMPI, FLAb, AbBind, AbCov):**
- All PDB-derived — amino acid sequences only. **~0% coverage** without manual curation.
- OAS (Observed Antibody Space) stores nucleotide sequences (it's NGS data — nucleotide reads are the raw input), but KIRBy has no OAS loader. If OAS were added, CaLM would be a natural fit.
- **Not feasible** with current datasets.

**Decision: DEFER.** Rationale:
1. Antibody domain (where codon usage bias would be most interesting for orthogonal hybridization) lacks nucleotide data entirely.
2. DTA is feasible but requires a non-trivial cDNA retrieval pipeline.
3. Better ROI from other OPIG items (B1 Trastuzumab, B2 Graphinity) first.
4. Revisit if: (a) OAS loader is added to KIRBy, or (b) a cDNA retrieval utility becomes available.

---

## Category B: New Datasets and Tasks

These introduce new prediction targets, datasets, or evaluation benchmarks that KIRBy
can run its existing rep→model pipeline on.

### B1. Trastuzumab-HER2 Binding (antibody binding classification)

| | |
|---|---|
| **Paper** | *Baselining the Buzz: Trastuzumab-HER2 Affinity, and Beyond* (#66 in catalog) |
| **Authors** | Lewis Chinery, Alissa Hummer, Mehta, ..., Jeliazkov, Charlotte Deane |
| **Year** | 2024 |
| **What it is** | 524K trastuzumab CDRH3 variant dataset with binding/non-binding labels (from Mason et al. DMS data). Benchmarks BLOSUM, AbLang, ESM, ProteinMPNN for binder enrichment. Experimentally validated 700 designs via BLI. |
| **Task** | Binary classification: binder vs. non-binder. Metric: binder enrichment rate. |
| **Data** | Public — [github.com/lewis-chinery/Tz_her2_affinity_and_beyond](https://github.com/lewis-chinery/Tz_her2_affinity_and_beyond) |
| **KIRBy domains** | Antibody |
| **Scope** | Medium — new dataset loader, new task (classification, not regression). Reps tested in the paper (BLOSUM, AbLang, ESM, ProteinMPNN log-likelihoods) mostly overlap with existing KIRBy antibody reps. The interesting question: does hybridizing AbLang + ESM + structural reps beat any individual rep for enrichment? |
| **Status** | **DONE** (2026-03-15). Dataset loader: `src/kirby/datasets/trastuzumab.py`. Two presets: `tz_her2_binary` (binder/non-binder), `tz_her2_multiclass` (low/mid/high). 524K→5K stratified subsample by (label, edit_distance). Full VH reconstructed from CDRH3 + trastuzumab framework. Classification support added to `antibody_baseline_survey.py` (StratifiedGroupKFold, accuracy/F1/AUROC metrics). Groups by edit distance. Initial results: BLOSUM ridge 80.7% acc, RF 84.9% acc (binary, 5-fold). |
| **Notes** | This paper is practically asking for KIRBy. The authors compared 4 reps independently — KIRBy's whole point is to combine them. ~26-30% enrichment with individual reps vs ~33% with DMS. Can hybrid close the gap? 524K is a large dataset by antibody standards. |

### B2. Graphinity Synthetic DDG Data (ΔΔG augmentation)

| | |
|---|---|
| **Paper** | *Investigating the Volume and Diversity of Data Needed for Generalizable Antibody-Antigen ΔΔG Prediction* (#65 in catalog) |
| **Authors** | Alissa Hummer, Constantin Schneider, Lewis Chinery, Charlotte Deane |
| **Year** | 2025 (Nature Computational Science) |
| **What it is** | ~943K FoldX-computed and ~21K Rosetta Flex ddG-computed synthetic ΔΔG values from mutating Ab-Ag interfaces in SAbDab structures. Experimental test sets: SKEMPI 2.0 and AB-Bind (645 single-point mutations). |
| **Task** | ΔΔG regression (same as existing SKEMPI domain) |
| **Data** | Public — [Zenodo](https://doi.org/10.5281/zenodo.15384945), [opig.stats.ox.ac.uk](https://opig.stats.ox.ac.uk/data/downloads/affinity_dataset/) |
| **KIRBy domains** | DDG / Antibody |
| **Scope** | Medium-Large — the Graphinity model itself is an end-to-end EGNN (takes structures, not pre-computed reps), so it doesn't fit KIRBy's rep→model paradigm directly. BUT the synthetic DDG dataset is valuable for: (1) augmenting KIRBy's existing SKEMPI DDG training data, (2) adding AB-Bind as a new test set, (3) testing whether pre-training on synthetic DDG + fine-tuning on experimental DDG improves KIRBy's DDG predictions. |
| **Status** | **DONE** (2026-03-15). Dataset loader: `src/kirby/datasets/graphinity.py`. Three presets: `graphinity_foldx` (~943K FoldX, subsampled to 50K), `graphinity_rosetta` (~21K Flex ddG), `graphinity_augmented` (experimental SKEMPI + synthetic). CSVs downloaded from Zenodo 15384945. Mutations parsed from `complex` column (`{pdb}_{wt_aa}_{chain}_{pos}_{mut_aa}`). PDB sequences extracted via `pdb_utils` and cached. Groups by PDB complex (leave-one-complex-out). |
| **Survey status** | **IN PROGRESS** (2026-03-26). Full antibody survey (41 reps) running via `submit_missing_htc.sh`. Jobs 7331380/7331381 (graphinity_foldx/rosetta) hit ProstT5 bottleneck: ~40K duplicate sequences processed individually with no dedup/cache → >24h for saprot alone. **Fix applied**: `_predict_3di_with_prostt5()` now deduplicates (graphinity has ~200 PDB structures → ~400 unique seqs vs ~40K duplicates = 100x speedup) and caches 3Di to disk (`~/.cache/kirby/prostt5_3di/`). saprot_cdr reuses cached 3Di from saprot → instant. Needs re-deploy + resubmit. |
| **Notes** | Key finding from the paper: current experimental DDG data is insufficient — orders of magnitude more needed. The synthetic data could help KIRBy's DDG domain substantially. Graphinity achieves Pearson r ~0.87. |

### B3. Hu-mAb (antibody humanness classification)

| | |
|---|---|
| **Paper** | *Humanization of Antibodies Using a Machine Learning Approach* (#71 in catalog) |
| **Authors** | Claire Marks, Alissa Hummer, Chin, Charlotte Deane |
| **Year** | 2021 |
| **What it is** | RF classifier for predicting antibody humanness from sequence features. Trained on OAS human sequences. Outputs humanness score per VH/VL domain. |
| **Task** | Classification: human vs. non-human antibody (humanness score). New task for KIRBy. |
| **Data** | OAS (already partially available in KIRBy ecosystem) |
| **KIRBy domains** | Antibody (new task) |
| **Scope** | Medium — need to construct a humanness classification dataset (human sequences from OAS vs. non-human/therapeutic sequences from Thera-SAbDab). The prediction task is sequence-based, so existing antibody reps (AbLang, ESM, AntiBERTy, etc.) all apply directly. |
| **Status** | **DONE** (2026-03-15). Dataset loader: `src/kirby/datasets/humanness.py`. Two presets: `humanness_species` (human VH vs mouse+rhesus+camelid from OAS, 10K balanced), `humanness_therapeutic` (human vs non-human therapeutics from Thera-SAbDab/IMGT). Data from Zenodo 10562968 (species FASTA files) with Thera-SAbDab fallback (therapeutics). Binary classification (label=1 human, 0 non-human). Identity groups for CV. |
| **Notes** | Interesting as a classification task (most KIRBy antibody work is regression). Tests whether hybrid reps can distinguish human from non-human antibodies better than individual LM embeddings. |

### B4. Polyspecificity Prediction (antibody risk classification)

| | |
|---|---|
| **Paper** | *Prediction of Polyspecificity from Antibody Sequence Data by Machine Learning* (#74 in catalog) |
| **Authors** | Éliás et al. (incl. OPIG members) |
| **Year** | 2024 |
| **What it is** | ML classifiers predicting polyspecificity risk from antibody sequence. Polyspecificity = tendency to bind non-target proteins (a developability liability). |
| **Task** | Classification: polyspecific vs. non-polyspecific. New task for KIRBy. |
| **Data** | Custom dataset — need to check availability from the paper. |
| **KIRBy domains** | Antibody (new task) |
| **Scope** | Medium — depends on data availability. If the dataset is public, this is a straightforward new classification benchmark. |
| **Notes** | Developability prediction is a practical application where hybrid reps could have real impact. Pairs well with B3 (Hu-mAb) as part of a "developability suite." |

### B5. Ginkgo Developability Competition (antibody property prediction)

| | |
|---|---|
| **Paper** | *Ginkgo Datapoints Antibody Developability Competition Outcomes* (#79 in catalog) |
| **Authors** | van Niekerk et al. (incl. OPIG members) |
| **Year** | 2026 |
| **What it is** | Competition-style benchmark for predicting antibody developability properties (expression, aggregation, etc.) from sequence. Multiple ML methods benchmarked. |
| **Task** | Multiple prediction targets (expression, aggregation, etc.) |
| **Data** | Competition dataset — need to check if public post-competition. |
| **KIRBy domains** | Antibody (new tasks) |
| **Scope** | Medium — depends on data availability. Competition datasets tend to be clean with well-defined splits. |
| **Status** | **DONE** (2026-03-15). Dataset loader: `src/kirby/datasets/ginkgo_dev.py`. 9 regression properties from GDPa1 v1.2 (242 antibodies): HIC, Tm2, Titer, PR_CHO, AC-SINS_pH7.4, Purity, SEC %Monomer, SMAC, HAC. CSV from GitHub `ginkgobioworks/abdev-benchmark`. Pre-computed 5-fold hierarchical cluster splits (IgG isotype-stratified). VH+VL paired sequences. Each property = separate dataset (e.g. `ginkgo_hic`). Sample sizes 94-242 depending on NaN coverage. |
| **Notes** | Competition setting = clean benchmarks with established leaderboards. Good for demonstrating hybrid rep improvements against published baselines. |

---

## Category C: Models and Evaluation Protocols

These are not reps or datasets — they are modeling approaches or evaluation methodologies
that could enhance KIRBy's downstream evaluation or reveal new insights about reps.

### C1. GAUCHE Tanimoto Kernel (GP kernel for fingerprints)

| | |
|---|---|
| **Paper** | *GAUCHE: A Library for Gaussian Processes in Chemistry* (#28 in catalog) |
| **Authors** | Griffiths, Leo Klarner, Moss et al. |
| **Year** | 2023 (NeurIPS) |
| **What it is** | GP library with 30+ molecular kernels. The Tanimoto kernel on binary fingerprints is the most relevant piece for KIRBy — it's a kernel specifically designed for comparing bit vectors (ECFPs, MACCS, Sort & Slice output). |
| **Code** | [github.com/leojklarner/gauche](https://github.com/leojklarner/gauche) |
| **KIRBy relevance** | KIRBy already has GP in its model set. The question is whether GAUCHE's Tanimoto kernel improves GP performance on fingerprint-based reps vs. the default RBF kernel. |
| **Scope** | Small — swap kernel in existing GP pipeline for fingerprint reps. Benchmark against current GP setup. |
| **Status** | **DONE** (2026-03-15). `TanimotoKernel` class added to `src/kirby/find_best_hybrid.py` as a custom sklearn `Kernel` subclass. Pure numpy, no GAUCHE/GPyTorch dependency. Added as 5th option in GP HP search grid (`_HP_SEARCH_SPACES['gp']['kernel']`). Automatically tried during `--tune N` runs. Default GP still uses `RBF() + WhiteKernel()` for baseline surveys. |
| **Notes** | Low priority unless current GP performance on fingerprints is suspiciously poor. The Tanimoto kernel is mathematically well-motivated for binary vectors (it's the kernel analogue of the Tanimoto similarity coefficient). |

### C2. Q-SAVI Covariate Shift Evaluation Protocol

| | |
|---|---|
| **Paper** | *Drug Discovery under Covariate Shift with Domain-Informed Prior Distributions* (#29 in catalog) |
| **Authors** | Leo Klarner, Rudner, Reutlinger, Schindler, Garrett Morris, Charlotte Deane, Teh |
| **Year** | 2023 (ICML) |
| **What it is** | Evaluation protocol using multiple splitting strategies (random, scaffold, molecular weight, spectral) to induce and quantify covariate shift. Also introduces a Bayesian model with domain-informed priors. |
| **Code** | [github.com/leojklarner/Q-SAVI](https://github.com/leojklarner/Q-SAVI) |
| **KIRBy relevance** | The multi-split evaluation protocol is the interesting part. KIRBy already uses scaffold splitting for molecular — Q-SAVI's protocol tests whether hybrid reps are robust across MULTIPLE types of distribution shift, not just scaffold. |
| **Scope** | Medium — adding MW-based and spectral splits alongside scaffold. The Bayesian model itself is specialized and probably not worth adding to KIRBy's model set. |
| **Status** | **DONE** (2026-03-15). Three split strategies implemented: `assign_scaffold_groups` (existing), `assign_mw_groups` (MW percentile bins), `assign_spectral_groups` (Tanimoto spectral clustering) in `tests/find_best_hybrid_molecular.py`. Wired into `run_baseline_survey.py` via `--split-strategy scaffold|mw|spectral` CLI arg (molecular-only). Dedicated experiment: `tests/experiments/covariate_shift.py` evaluates all reps x models x split strategies and prints comparison table. |
| **Notes** | Interesting for a "robustness" experiment: do hybrid reps maintain their advantage under scaffold split AND MW split AND spectral split? If so, that's a much stronger claim than showing improvement under a single split type. |

### C3. AEV-PLIG / FEP Gap (model + overlapping data)

| | |
|---|---|
| **Paper** | *Narrowing the Gap Between ML Scoring Functions and FEP Using Augmented Data* (#22 in catalog) |
| **Authors** | Isak Valsson, Warren, Magarkar, Biggin, Charlotte Deane |
| **Year** | 2025 |
| **What it is** | Attention-based GNN using Atomic Environment Vectors (AEVs) for binding affinity prediction. Trained on PDBbind v2020, augmented with BindingNet + BindingDB. Claims ~400,000x faster than FEP with approaching accuracy. |
| **KIRBy relevance** | The model operates on 3D protein-ligand complexes (AEVs), which doesn't fit KIRBy's rep→model paradigm directly. BUT: (1) BindingDB overlaps with KIRBy's existing BindingDB_Kd dataset, (2) PDBbind is a standard benchmark KIRBy could add, (3) AEV representations could theoretically be extracted and used as a rep type. |
| **Scope** | Large — AEVs require 3D structures. Adding PDBbind as a dataset is feasible. Extracting AEV embeddings as a rep is a bigger lift. |
| **Status** | **DONE — PDBBind fix + LP splits** (2026-03-15). PDBBind loader fixed: `drug_ids`, `protein_ids`, `n_drugs`, `n_proteins` added to metadata (was KeyError). LP-PDBBind leak-proof grouping implemented: `assign_lp_groups()` clusters proteins at 30% sequence identity via connected components (Volkov et al. 2022). Wired into all three DTA consumers (`run_baseline_survey.py`, `dta_baseline_survey.py`, `find_best_hybrid_dta.py`). Clustering cached to disk. **AEV as a rep: DEFERRED** — requires 3D complex structures (only PDBBind has them), mean-pooling AEVs discards spatial info, and single-dataset reps can't participate in cross-dataset hybrid comparisons. |
| **Notes** | The data overlap (BindingDB) makes this relevant even if the model/rep doesn't fit. The question is whether KIRBy's fingerprint/LM-based reps on BindingDB can approach AEV-PLIG's structure-based performance — that would be a powerful demonstration. |

### C4. Imrie Transfer Learning Protocol

| | |
|---|---|
| **Paper** | *Protein Family-Specific Models Using Deep Neural Networks and Transfer Learning Improve Virtual Screening* (#13 in catalog) |
| **Authors** | Fergus Imrie, Bradley, van der Schaar, Charlotte Deane |
| **Year** | 2018 |
| **What it is** | Transfer learning protocol: pre-train DNNs on all ChEMBL bioactivity data, fine-tune on specific protein families. Uses ECFPs + protein descriptors. Showed TL improves VS performance, especially for low-data targets. |
| **KIRBy relevance** | The TL protocol (pre-train broadly, fine-tune narrowly) could be applied to KIRBy's DNN model. The question is whether hybrid reps + TL is better than either alone. Also, the ECFP + protein descriptor input could be swapped for hybrid reps. |
| **Scope** | Large — implementing TL in KIRBy's model pipeline is a significant addition. Also needs ChEMBL pre-training data. |
| **Status** | **DONE** (2026-03-15). Self-contained comparison experiment, NOT a core pipeline change. Files: `src/kirby/datasets/chembl_bioactivity.py` (bulk ChEMBL IC50 loader, top-200 targets, InChIKey dedup per target, leakage exclusion), `src/kirby/models/dnn_pretrained.py` (sklearn-compatible DNN with pre-trained weight loading + layer freezing), `src/kirby/models/pretrain_chembl_dnn.py` (standalone pre-training script, 6 fast reps), `tests/experiments/imrie_transfer_learning.py` (5 conditions × 6 reps × 3 datasets: dnn_scratch, dnn_tl_full, dnn_tl_frozen, ridge, gp). SLURM: `imrie_pretrain.sh` (128G, 4h) + `imrie_eval.sh` (64G, 2h). Analysis: per-rep TL delta, rep ranking preservation (Spearman ρ), model-vs-rep trade-off. |
| **Notes** | TL is a model-side improvement, not a rep-side one. KIRBy's thesis is about the input mattering more than the model. This experiment answers: "Does TL change the conclusion that reps matter?" Either way is thesis-relevant: if TL helps all reps equally → model improvement orthogonal; if TL collapses ranking → model matters more; if hybrid + TL > best single + TL → hybrid value persists. |

---

## Implementation Priority & Dependencies

### Quick wins (can start now, no dependencies)
1. ~~**A1 (Sort & Slice)**~~ — **DONE** (molecular Phase 3B, DTA 2026-03-14). Reaction skipped (fingerprints collapse under leave-ligand-out).
2. ~~**C1 (GAUCHE Tanimoto kernel)**~~ — **DONE** (2026-03-15). TanimotoKernel in GP HP grid.

### Next tier (need data acquisition/verification)
3. ~~**B1 (Trastuzumab)**~~ — **DONE** (2026-03-15). Dataset loader + classification survey support.
4. ~~**B2 (Graphinity DDG data)**~~ — **DONE** (2026-03-15). 3 presets (foldx/rosetta/augmented).
5. ~~**B3 (Hu-mAb)**~~ — **DONE** (2026-03-15). 2 presets (species/therapeutic).

### Requires investigation
6. ~~**A2 (CaLM)**~~ — **INVESTIGATED, DEFERRED** (2026-03-14). cDNA not available for antibody datasets; DTA feasible but low ROI vs other items.
7. **B4 (Polyspecificity)** — need to check data availability (raw NGS data — may not be feasible)
8. ~~**B5 (Ginkgo competition)**~~ — **DONE** (2026-03-15). 9 developability properties from GDPa1.
9. ~~**C2 (Q-SAVI splits)**~~ — **DONE** (2026-03-15). MW + spectral splits, experiment script.

### Larger scope (defer)
10. ~~**C3 (AEV-PLIG / FEP)**~~ — **DONE** (2026-03-15). PDBBind metadata fix + LP-PDBBind splits. AEV rep deferred (3D structure requirement).
11. ~~**C4 (Imrie TL)**~~ — **DONE** (2026-03-15). Self-contained experiment: 5 conditions × 6 reps × 3 datasets.

### Deferred (user decision)
- **Activity cliffs (#30, Dablander)** — central to research but held for later experiments

---

## Cross-references

| This roadmap | ISSUES.md phase | Relationship |
|---|---|---|
| A1 (Sort & Slice) | Phase 3B (molecular reps) | New molecular fingerprint rep |
| A2 (CaLM) | Phase 4 (forward compat) | New protein LM, novel modality |
| B1 (Trastuzumab) | Phase 3C (antibody) | New antibody dataset + task |
| B2 (Graphinity DDG) | Phase 3C (antibody/DDG) | DDG data augmentation |
| B3 (Hu-mAb) | Phase 3C (antibody) | New antibody classification task |
| B4 (Polyspecificity) | Phase 3C (antibody) | New antibody classification task |
| B5 (Ginkgo) | Phase 3C (antibody) | New antibody prediction tasks |
| C1 (GAUCHE kernel) | Phase 2 (model expansion) | GP kernel improvement |
| C2 (Q-SAVI splits) | Phase 1 (eval overhaul) | Additional split strategies |
| C3 (AEV-PLIG) | Phase 3D (DTA) | Data overlap, model comparison |
| C4 (Imrie TL) | Phase 2 (model expansion) | Training protocol addition |
