# KIRBy Paper Outline — Nature Machine Intelligence

**KIRBy = Knowledge Integration by Representation**

**Status:** Draft outline. Results pending — structure is flexible. Word limits are targets, not constraints. Room for change based on what the data shows.

---

## Working titles
- "Learned representation fusion improves molecular and biological prediction across domains"
- "Task-adaptive representation selection: matching feature complexity to model capacity"
- "KIRBy: Knowledge Integration by Representation — a forward-compatible framework for hybrid representation learning"

---

## Core claims

**Claim 1: Most features in any single representation are noise for a given task.**
Individual reps (ECFP4 at 2048D, ESM-2 at 1280D, Mordred at 1613D) are bloated for any specific prediction task. A small, task-relevant subset is sufficient — and finding that subset is not trivial because it depends on the downstream model.

**Claim 2: Complementary features from different representations capture orthogonal signal.**
No single rep captures everything. An importance-weighted combination of the best features from multiple reps consistently outperforms the best individual rep. This is true across domains (molecular, antibody, TCGA, reactions).

**Claim 3: This combination framework is forward-compatible.**
When a new foundation model drops, its embeddings slot directly into the search — no retraining, no architecture changes. The hybrid search automatically determines whether the new rep adds value and how to allocate features. Fine-tuned model variants are just another "new rep" source.

---

## Positioning vs literature

### Key NMI references and how KIRBy relates

**Boiko et al. (2025, SIMGs) & Fang et al. (2022, GEM):**
Both argue "the representation is the bottleneck, not the model." Each proposes ONE new representation. KIRBy proposes a method to combine ALL of them — and any future ones.

**Steyaert et al. (2023, multimodal fusion for cancer):**
Lays out early/intermediate/late fusion. Concludes "no decisive evidence that one fusion strategy is superior" and "the choice is task-dependent." KIRBy's answer: *learn* which strategy works, per task, automatically.

**Dong & Rudin (2020, Variable Importance Cloud):**
Feature importance is model-dependent. A feature important for one good model may be irrelevant for another. Supports KIRBy's matched evaluator design and ANOVA decomposition experiments.

**Ektefaie et al. (2024, SPECTRA):**
Standard benchmarks give incomplete views of generalizability. Performance degrades with decreasing cross-split overlap. KIRBy's similarity experiments and strict no-random-split methodology address this directly.

**Zeng et al. (2022, ImageMol):**
Self-supervised molecular image rep. Achieves SOTA on some tasks, not others. A perfect example of a rep KIRBy could absorb — sometimes its features get budget, sometimes not.

### Gap in the literature
Everyone is building better individual representations. Nobody is systematically studying how to *combine* them, how to *match* representation complexity to model capacity, or how to make this *forward-compatible* with representations that don't exist yet.

### Comparison to existing combination approaches

**Naive concatenation:** Stack all features. Works up to ~500-1000D, then curse of dimensionality hits. Tree models and kNN degrade. Full concatenation often underperforms budget-constrained hybrids.

**PCA on concatenated features:** Maximizes variance, not predictive power. High-variance noise dimensions get preserved. C17 compares hybrid selection vs PCA at matched dimensionality.

**CCA / partial least squares:** Limited to two sources, requires paired data, loses interpretability. KIRBy handles N sources natively.

**Multi-kernel learning (MKL):** Learns kernel weights across modalities. Operates at kernel level, not feature level. Doesn't produce interpretable feature subsets.

**MOGONET / intermediate fusion networks:** Learn cross-modal interactions via GNNs or attention. Requires end-to-end training, not forward-compatible. New modality = retrain entire network.

**KIRBy's differentiators:** (a) works at the feature level with frozen embeddings, (b) forward-compatible, (c) produces interpretable, portable blueprints.

### Comparison to model ensembles (C11)

- **Ensemble:** Train separate models on separate reps, average predictions. Each model still navigates all the noise in its rep. Ensembling averages *predictions*, not signal.
- **Hybrid:** Select the useful features from each rep, train *one* model on clean features. Denser, higher-SNR input.
- **Why ensembles multiply noise:** Each member independently overfits to its rep's noise. Averaging predictions reduces variance but bias from noisy features persists.
- **Computational cost:** Ensemble of K reps x M models = K*M trained models. Hybrid = 1 model on selected features.
- **Framing:** "Model ensembles combine predictions; KIRBy combines information. When the bottleneck is noisy features rather than model capacity, operating at the feature level is more efficient."

### Feature selection overfitting

Reviewers will ask. Mitigations already in the pipeline:

1. **Per-fold importance computation.** Importance scores computed on training data only within each CV fold. Test fold never informs selection.
2. **Budget as implicit regularization.** Constrained budget (e.g. 100 features from 5000+) acts as a hard capacity limit, analogous to L2 but more interpretable. C16 tests this directly.
3. **Why less bad than model overfitting.** Model overfitting happens in continuous, high-dimensional parameter space. Feature selection operates over a discrete, much smaller space. And features are *frozen embeddings*, not learned — choosing which pre-existing knowledge to trust, not fitting new parameters.
4. **Cross-fold stability.** If selection were overfit, feature sets would differ wildly across folds. C10 measures Jaccard similarity of selected features across folds. Stable = not overfit.
5. **Blueprint transfer.** Blueprints applied to held-out data without re-selection. If they work, selection generalised.

**Framing for paper:** "Feature selection operates in the representation space, not the parameter space. The capacity of the selection procedure is bounded by the budget, not by the expressiveness of a trainable model."

---

## Outline

### 1. Introduction (~800w)
- Problem: representations are the bottleneck, but no single rep is best
- Existing approaches and why they fall short (concat, PCA, ensemble, MKL, intermediate fusion)
- KIRBy: task-adaptive, budget-constrained feature selection across representations
- Forward-compatible by design

### 2. Results

#### 2a. The hybrid representation (~400w, 1 schematic figure)
- Framework overview — how it works at a high level
- Comparison to concat / PCA / ensemble at matched dimensions
- Feature selection overfitting controls

#### 2b. Selective case studies: where hybrids help and where they don't (~600w, 1 figure)
- Curated datasets across domains — interesting cases, not exhaustive
- Strong gains, moderate gains, near-ceiling (minimal improvement where signal was already strong), and honest negatives
- What predicts gain magnitude? (complexity metrics, dataset size, rep diversity)
- Number of datasets TBD based on results; full tables in supplementary

#### 2c. Why it works: complexity matching (~500w, 1 figure)
- The core mechanism — budget x capacity interaction
- Within-family gradient (C06)
- Whichever complexity experiment result is most compelling
- *Flexible: results will determine emphasis. ANOVA (C09), cross-dataset transfer (C03), feature-as-regularization (C16), information-theoretic (C17) — pick the most striking.*

#### 2d. Downstream benefits beyond accuracy (~400w, 1 figure)
- Similarity / applicability domain / calibration / noise robustness — pick 2-3 strongest results
- OPIG case studies where relevant (case-by-case basis)
- "Hybrids produce more reliable predictions, not just higher scores"

#### 2e. Forward compatibility and fine-tuning (~400w, 1 figure)
- New models (ESM3, Geneformer V2, SaProt, MINT, etc.) slot in without retraining
- Fine-tuned variants (LoRA MolFormer, ESM-2 LoRA) as another "new rep" source
- The blueprint: a portable, transferable recipe

#### 2f. [Flexible slot] (~300w)
- Whatever surprising result emerges from full runs
- Could be explainability (C22), modality ablation (C12), learning curves (C05), or something from noise/similarity experiments

### 3. Discussion (~800w)
- Positioning vs fusion literature (early/intermediate/late)
- Feature selection overfitting — why the controls work
- Limitations (see below)
- Future directions (active learning in hybrid space, federated learning, new domains)

### 4. Methods (~2000w, separate section)

#### 4a. Search tiers and configuration
Two F12-validated speed tiers control the search space:
- **Fast (screening):** 3 allocation strategies (entropy_1.0, top_3, top_2), 2 importance methods (treeshap, lightgbm_gain), matched internal evaluator, 3-fold CV. For rapid iteration and experiment prototyping.
- **Full (publication):** 13 allocation strategies (top_2/3/5, performance_weighted, min_floor, entropy_1.0/2.0, score_gap_0.2, power_2, rank_linear, thresh_0.3, rank_exp_d0.5, greedy_fill_s5), 2 importance methods (deeplift, lightgbm_gain), matched evaluator, 5-fold CV.

Per-model budget caps prevent curse-of-dimensionality on sensitive models: GP/kNN <= 2000, SVM/MLP/DNN/BNN/CNN <= 3000, tree models uncapped.

Budget grids are user-specified per experiment, not baked into tiers. Typical screening: [10, 25, 50, 100, 200, 500, 1K, 2K]. Full: [10, 25, 50, 100, 200, 300, 500, 750, 1K, 1.5K, 2K].

#### 4b. Augmentation strategies
After base hybrid construction, optional augmentation adds supplementary features:
- **none:** No augmentation.
- **uncapped (default):** Rank augmentation features by importance, A/B test each against base hybrid one at a time, keep if it improves validation score. No budget limit — keep adding while improving. Default step_size=5, patience=20.
- **capped:** Phase A adds features up to budget_limit. Phase B replaces worst base features with best augmentation features.
- **greedy_ablation:** Test each augmentation rep independently at the rep level (not individual features), keep only those that improve.
- **all:** Include all augmentation features as baseline comparison.

Augmentation reps are separate from the base rep pool — typically small, specialised feature sets (e.g. developability scores, humanness, pharmacophore descriptors) that complement the main hybrid.

#### 4c. Remaining methods
- Pipeline details: importance scoring (8 methods, 3 validated clusters from F05), allocation strategies (13 F12-validated from 31 original), budget search
- Search engine (find_best_hybrid): deduplication, fold-aware caching, sequential evaluation
- Blueprint system: portable JSON recipe for reproducing a hybrid on new data
- Datasets and splits (scaffold/group-aware, no random splits, per-domain splitting strategies)
- Statistical framework (Demsar, Nadeau-Bengio corrected t-test, Wilcoxon signed-rank, Friedman-Nemenyi)
- Computational infrastructure

### Extended Data / Supplementary

This is now a substantial section since the main Methods defers most technical detail here.

#### Supplementary Tables

| Table | Content | Source |
|-------|---------|--------|
| ST1 | Per-dataset details: sample sizes, rep counts, task types, splitting strategies, primary metrics | DATASET_REPS + ANALYSIS.md |
| ST2 | Per-dataset representation pools with dimensionalities and sources (all 5 domains) | DATASET_REPS registry |
| ST3 | Downstream model hyperparameters (all 13), per-model budget caps, sample-size guards | DEFAULT_MODELS in find_best_hybrid.py |
| ST4 | Internal evaluator hyperparameters (all 5) | find_best_hybrid.py evaluator code |
| ST5 | All 13 allocation strategy definitions with formulas | NAMED_ALLOCATIONS in hybrid.py |
| ST6 | Feature importance method hyperparameters (LightGBM gain, TreeSHAP, DeepLift) | find_best_hybrid.py importance code |

#### Supplementary Methods

- **Preprocessing details:** clipping (±10¹⁰), NaN imputation (train median), all-NaN column drop, binary detection, StandardScaler (train-fold only)
- **Importance method selection:** 8 candidate methods → cluster analysis → 3 retained (1 per cluster). F05 experiment results.
- **Allocation strategy selection:** 31 candidates → performance evaluation → 13 retained. F12 experiment results.
- **Internal evaluator selection:** F07/F13 experiment results. Matched vs fixed evaluator comparison.
- **Speed tier configurations:** Fast (3 allocs, 2 importance, 3-fold) and Full (13 allocs, 3 importance, 5-fold). Exact allocations per tier.
- **Augmentation procedure:** Sequential testing with RF importance ranking. Step size = 5, patience = 20. F08 hyperparameter sensitivity results.
- **Computational cost analysis:** Importance pre-computation, allocation score caching, feature selection deduplication (index hashing), disk caching (deterministic hash), factory pre-computation (fine-tuned reps once per fold).

#### Supplementary Results

- Full per-dataset results tables (all 40 datasets × all conditions)
- Learning curves (C05)
- Data complexity partitions (C07)
- Ensemble vs hybrid full comparison (C11)
- ML theory depth (C15) if interesting
- Information-theoretic decomposition (C17)
- Diversity vs depth (C18)
- Framework validation details (F05 importance method clustering, F08 augmentation hyperparams, F12/F13 methodology)

---

## Limitations to address

1. **Small datasets (n<200).** F10/F11 show the hybrid search cannot reliably improve on best single rep for very small datasets. The greedy allocation uses ~22-sample internal validation — too noisy. This is a structural limitation, not fixable by tuning. Honest negative.

2. **Representation engineering and version management.** Importing, computing, and caching representations across many foundation models and many CV folds is heavy-duty engineering. Each model has its own dependencies, version requirements, and GPU needs — and these frequently conflict (e.g. different PyTorch versions, CUDA requirements, or HuggingFace API changes across models). Managing 30-50+ representation sources as a coherent pipeline requires significant version control discipline. Representations only need to be computed once per dataset with proper fold-aware caching, but the initial integration and maintenance burden is non-trivial and scales with the number of models in the pool. This is a practical barrier to adoption and reproducibility.

3. **Computational cost of exhaustive search.** The full search over models x budgets x allocations x importance methods is expensive. Speed tiers (fast/full) mitigate this, and caching amortises repeated searches, but the first run on a new dataset is slow.

4. **Frozen embeddings assumption.** KIRBy operates on frozen representations. It cannot discover features that no existing representation captures. If the true signal requires end-to-end learning (e.g. QM9 3D properties where SchNet E2E massively beats frozen reps), the hybrid approach has a ceiling.

5. **Importance method sensitivity.** F05 shows importance methods cluster into 3 families with high intra-family correlation. The choice between families matters less than expected, but there's still some sensitivity. Mitigated by using multiple methods in the full search.

---

## Key figures (6)

1. **Framework overview** — schematic of KIRBy pipeline (reps -> importance -> allocation -> hybrid -> search -> blueprint)
2. **Case studies** — selected datasets showing strong/moderate/minimal/negative improvement. Grouped by domain.
3. **Complexity matching** — budget x capacity interaction. Within-family gradient or heatmap. Most compelling complexity result.
4. **Downstream benefits** — panels for 2-3 of: similarity, calibration, noise robustness, OPIG case studies
5. **Forward compatibility** — new models slotting in. 5-condition ablation or before/after comparison.
6. **[Flexible]** — whatever the results demand. Could be explainability, modality ablation, or a striking negative result.

---

## Experiments that map to each section

| Section | Primary experiments | Backup / flexible |
|---------|-------------------|-------------------|
| 2a (framework) | C02 (grid), C11 (vs ensemble), C17 (vs PCA) | — |
| 2b (case studies) | C14 (cross-domain), precompute results | — |
| 2c (why it works) | C06 (within-family), C09 (ANOVA) | C03 (transfer), C16 (regularization), C15 (theory) |
| 2d (downstream) | S02 (similarity), C08 (calibration), C10 (noise) | OPIG experiments, S04 (cliffs), S06 (AD) |
| 2e (forward compat) | C13 (forward compat), FT results | — |
| 2f (flexible) | TBD based on results | C05, C07, C12, C22, anything surprising |

---

## Experiment gap analysis

Maps each paper section to existing experiments, identifies what's missing, and specifies whether the gap requires a **new experiment** (code + compute) or **analysis-level work** (post-hoc on existing outputs).

### Section 2a gaps (framework overview)

Existing coverage: C02 (concat via full-dim budget), C17 (PCA/mRMR/RFE/random), C11 (ensemble vs hybrid), C10 Phase 3 (cross-fold Jaccard + Nogueira stability).

| Gap | Level | Description |
|-----|-------|-------------|
| Concat framing in C02 | ANALYSIS | C02 includes full-dim budget but doesn't label it "concatenation baseline." Post-hoc script to extract and compute delta. |
| Matched-dimension concat | ANALYSIS | C17 has "random" strategy — verify it serves as random-features-from-concat control. |
| ~~Stability summary~~ | ~~DONE~~ | Nogueira index now computed in C10 Phase 3 and C24. |

### Section 2b gaps (case studies)

Existing coverage: C14 (cross-domain summary), precompute results (240 SLURM jobs running).

| Gap | Level | Description |
|-----|-------|-------------|
| Gain predictor analysis | ANALYSIS (script) | Correlate delta with dataset properties (n, n_reps, erank, domain) from C14 + C03 outputs. |
| Curated case selection | ANALYSIS (manual) | Pick 8-12 interesting cases once precompute results arrive. Editorial decision. |
| Near-ceiling detection | ANALYSIS | Filter C14/precompute for datasets where best single rep >0.9 and hybrid delta is small. |

### Section 2c gaps (complexity matching)

Existing coverage: C02 (heatmap), C06 (within-family), C09 (ANOVA/MI/rank/bias-variance), C16 (regularization), C03 (cross-dataset transfer), C17 (info-theoretic).

| Gap | Level | Description |
|-----|-------|-------------|
| Multi-dataset C06 | EXPERIMENT (run) | Run C06 on 3-4 datasets to show within-family gradient is general. Code supports it — just SLURM. |
| C09 on multiple datasets | EXPERIMENT (run) | C09 reads C02 JSON. Run C02 on 3-4 representative datasets first. |
| Cross-dataset C06 summary | ANALYSIS (script) | Aggregate capacity-budget correlations across datasets. Report median/range per family. |

### Section 2d gaps (downstream benefits)

Existing coverage: S02 (SAC), S04 (cliffs), S06 (AD), S09 (cross-dataset similarity), C08 (calibration — now with classification), C10 (noise NDS), N04 (GP-LOO).

| Gap | Level | Description |
|-----|-------|-------------|
| S02 on multiple datasets | EXPERIMENT (run) | Need 3-4 datasets for S09 Wilcoxon. Verify SLURM scripts include this. |
| C08 on multiple datasets | EXPERIMENT (run) | Run on 2-3 datasets for generality. |
| ~~C08 classification~~ | ~~DONE~~ | Conformal prediction sets now implemented. |
| Summary figure script | ANALYSIS (script) | Pull S02 + C08 + C10 into multi-panel figure. |

### Section 2e gaps (forward compatibility)

Existing coverage: C13 (5-condition ablation with controls and Wilcoxon).

| Gap | Level | Description |
|-----|-------|-------------|
| LEGACY_REPS verification | EXPERIMENT (config) | Verify C13's LEGACY_REPS list is correctly defined. |
| FT as forward compat | EXPERIMENT (config) | Run C13 separately with FT reps (lora_molformer, esm2_lora, graphmvp_ft) as "new" reps. |
| Cross-dataset C13 | EXPERIMENT (run) | Run on 2-3 datasets (molecular + antibody minimum). Same code, multiple invocations. |
| Temporal framing | ANALYSIS (editorial) | Note release dates of each new rep relative to KIRBy development. |

### Section 3 gaps (discussion — overfitting defense)

Existing coverage: C10 Phase 3 (Jaccard + Nogueira), C16 (budget as regularization), C24 (blueprint transfer).

| Gap | Level | Description |
|-----|-------|-------------|
| ~~Nogueira stability~~ | ~~DONE~~ | Now computed in C10 Phase 3 and C24. |
| ~~Blueprint transfer~~ | ~~DONE~~ | C24 implemented. Transfer matrix + aggregate delta + Nogueira index. |

### Supplementary experiments — status

| Experiment | Status | Action needed |
|-----------|--------|---------------|
| C05 (learning curves) | EXISTS | Run on 2-3 datasets |
| C07 (data complexity partitions) | EXISTS | Run on 2-3 datasets |
| C12 (TCGA modality ablation) | EXISTS | Run on tcga_luad_vs_lusc and tcga_brca_pam50 |
| C15 (ML theory depth) | EXISTS | Run on 1-2 datasets; assess interest |
| C18 (diversity vs depth) | EXISTS | Run on 2-3 datasets |
| C22 (explainability) | EXISTS | Requires precomputed search results |
| C23 (evaluator-model matching) | EXISTS | Run on 2-3 datasets |
| C24 (blueprint transfer) | NEW | Run on 2-3 datasets |

No new code needed for supplementary — they all exist and just need runs.

### Priority summary

**P0 (must-have):** Run C14 on all ready datasets, run C13 on 2-3 datasets, unified concat/PCA/ensemble comparison script (A2), run C06 on 3-4 datasets, run C08 on 2-3 datasets.

**P1 (strengthens paper):** FT reps in C13 (config change), gain predictor analysis (A4), run S02 on 3-4 datasets, multi-dataset C09 aggregate (A5).

**P2 (nice-to-have):** Summary figure script (A7), cross-dataset C06 aggregation (A6), near-ceiling detection (A8).

### Analysis TODOs (post-hoc on existing outputs, no compute)

| # | Task | Input | Output | Section |
|---|------|-------|--------|---------|
| A1 | Extract concat baseline from C02 full-dim, compute delta | C02 JSON | concat R² vs hybrid R² per model | 2a |
| A2 | Merge C11+C17+C02 into unified 5-method comparison | C11, C17, C02 JSONs | Table/figure at matched dims | 2a |
| A3 | ~~Nogueira stability from C10~~ | ~~DONE — now in C10 code~~ | | |
| A4 | Correlate delta with dataset properties (n, erank, domain) | C14 + C03 JSONs | Scatter/regression | 2b |
| A5 | Aggregate C09 ANOVA across datasets | Multiple C09 JSONs | Median eta² per factor | 2c |
| A6 | Aggregate C06 capacity-budget correlations | Multiple C06 JSONs | Median Spearman per family | 2c |
| A7 | Multi-panel figure: S02 (SAC) + C08 (cal) + C10 (NDS) | S02, C08, C10 JSONs | 3-panel figure | 2d |
| A8 | Near-ceiling datasets (best single >0.9, small delta) | C14/precompute JSONs | List for honest reporting | 2b |
| A9 | Temporal framing of new reps | Manual | Table: rep, release date, lag | 2e |

---

## Open questions for after results

- How much space does "why it works" (2c) deserve vs "what you can do with it" (2d/2e)?
- Which OPIG experiments fit as case studies vs supplementary?
- Is the modality ablation (C12, TCGA) interesting enough for main text or supplementary?
- Does the ensemble comparison (C11) go in 2a (framework) or get its own panel?
- Title: should it emphasise the method (hybrid/fusion) or the result (forward compatibility)?
