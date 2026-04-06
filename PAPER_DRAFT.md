# KIRBy: Knowledge Integration by Representation for task-adaptive prediction across biological domains

---

**Draft — Nature Machine Intelligence**

*Note: Citation numbers are placeholders [X]. Final numbering will be assigned during manuscript preparation. All citations below are traceable to PAPER_REFERENCES.md or the lit review.*

---

## Introduction

A comparison of 62,820 models across molecular property prediction benchmarks showed that classical RDKit2D descriptors matched or outperformed learned representations from graph neural networks and pretrained language models on the majority of datasets^[Deng2023]. Descriptor-based models occupied 73% of top-three positions across 11 benchmarks^[Jiang2021], and in 1,310 ChEMBL assays, feedforward networks on ECFP fingerprints outperformed graph convolutions and SMILES-based recurrent models^[Mayr2018]. The choice of input representation — not the model — is often the primary determinant of predictive accuracy. The same pattern holds outside small-molecule chemistry. No single protein language model consistently achieves the highest performance across tasks^[Kenlay2024,Schmirler2024]; for antibodies, general-purpose models sometimes outperform domain-specific ones^[Kenlay2024]. In computational pathology, no individual foundation model dominates across clinical endpoints^[ElNahhas2025], and in cancer genomics, integrating additional omics layers can negatively impact performance when the data types are not properly combined^[Steyaert2023].

For small molecules alone, the landscape now spans extended-connectivity fingerprints, physicochemical descriptor sets exceeding 1,600 dimensions, graph neural network embeddings^[Yang2019,Fang2022], language models trained on billions of SMILES strings^[Ross2022], image-based encodings (molecules rendered as 2D images)^[Zeng2022], and stereoelectronics-infused molecular graphs^[Boiko2025]. The same is true for proteins (ESM-2^[Lin2023], ProtTrans^[Elnaggar2022]), antibodies (AntiBERTy^[Ruffolo2021], AbLang^[Olsen2022]), histopathology^[ElNahhas2025], and multi-omics cancer profiles^[Steyaert2023]. A recent review counted over 40 molecular foundation models published between 2020 and 2024^[Song2026]. More appear every year^[Bommasani2021]. Different representations likely encode complementary aspects of the same biology — single data modalities can be noisy or incomplete, but when combined with complementary signals from other modalities they produce more sensitive and robust predictions^[Steyaert2023]. El Nahhas et al. found that an ensemble of frozen embeddings from the top four pathology foundation models achieved a mean AUROC 1.2% higher than the best individual model across 31 clinical tasks^[ElNahhas2025].

The most common strategy for combining representations is concatenation — early fusion — which requires only a single downstream model^[Stahlschmidt2022]. It works when feature sets are homogeneous. When they are not, concatenation degrades^[Stahlschmidt2022], and Li and Deng et al. called the practice "misleading" when the goal is to assess the contribution of individual representation types^[Deng2023]. Late fusion trains separate models per source and aggregates their predictions (typically by averaging or stacking), avoiding the dimensionality problem but discarding interactions between features from different sources^[Steyaert2023]. Intermediate fusion — MOGONET^[Wang2021], cross-attention architectures^[Zhang2024] — can capture cross-modal interactions, but at the cost of requiring end-to-end training on all sources; adding a new representation type means retraining from scratch. Multiple kernel learning^[Gonen2011] operates at the kernel level and produces no interpretable feature subsets. As Steyaert et al. noted, no decisive evidence favours one of these strategies over the others^[Steyaert2023].

In practice, most research groups default to one representation and ignore the rest. A cheminformatics lab reaches for ECFP fingerprints; a deep learning group uses MoLFormer or ChemBERTa; a structural biology group extracts ESM-2 embeddings. These may encode quite different aspects of the same underlying biology. But they are rarely compared, let alone combined — and when groups do attempt combination, the most accessible strategy is concatenation. It works at modest scales. Concatenating just three standard sources, however (ECFP4 at 2,048 dimensions, ESM-2 at 1,280, Mordred descriptors at ~1,600), produces a ~5,000-dimensional input for a dataset that may contain only a few hundred labelled samples. At these sample-to-feature ratios, accuracy peaks and then degrades as more features are added^[Hughes1968,Bellman1961], and distance-sensitive models such as kNN and SVM are affected most severely^[FernandezDelgado2014]. The community keeps producing better individual representations — perhaps faster than we can evaluate them — but we do not have a principled method for combining them without the dimensionality penalty. Which features from which sources should be kept? Does the answer change with the dataset, the endpoint, the downstream model?

Over 40 molecular foundation models were published between 2020 and 2024^[Song2026], with comparable growth in protein language models^[Lin2023,Elnaggar2022] and domain-specific encoders for antibodies^[Ruffolo2021,Olsen2022], histopathology^[ElNahhas2025], and gene expression. New models emerge as architectures, training corpora, and pretraining objectives evolve^[Bommasani2021]. If a new protein language model appears next month, must a practitioner retrain from scratch to determine whether it adds value? Any practical combination method must be forward-compatible — able to incorporate new sources without retraining or architectural modification. Deep fusion approaches fail this requirement. Adding a new modality means retraining the entire network. If, however, representations are treated as independent frozen feature sets, a new model's output can be added to the pool and evaluated alongside existing sources with no changes to the combination procedure.

Here, we present KIRBy (Knowledge Integration by Representation), a framework that takes a collection of frozen representation matrices for a dataset and produces a customised hybrid feature set optimised for a specific prediction task. Given any number of pre-computed representations — fingerprints, physicochemical descriptors, foundation model embeddings, gene expression signatures, histopathology features — KIRBy searches over feature budgets, allocation strategies, importance methods, and downstream models to select a compact subset of features drawn from across the available sources. The search is run once per dataset. The result — a portable Blueprint — can be applied to new data without repeating the search. We evaluate KIRBy across five biological domains — molecular property prediction, antibody engineering, drug-target interaction, cancer genomics, and chemical reactions — spanning 40 datasets and up to 57 representation sources per domain. We find that most features in any individual representation are uninformative for a given task. Features drawn from multiple representations capture complementary signal and consistently outperform the best single representation. The framework is also forward-compatible: when new foundation model embeddings become available, they are added to the representation pool and the search determines whether and how to use them.

---

## Results

### The hybrid representation

*Framework overview — how KIRBy works at a high level.*

- Pipeline schematic: representations -> importance scoring -> allocation across reps -> budget-constrained feature selection -> hybrid construction -> search over configurations -> blueprint output
- Comparison to concatenation (full-dim budget = concat baseline), PCA at matched dimensionality, model ensemble (separate models per rep, averaged predictions), and existing feature selection methods (mRMR, RFE, random)
- Key result: hybrid outperforms all alternatives at matched feature dimensions
- Feature selection overfitting controls: per-fold importance, cross-fold stability (Jaccard + Nogueira index), blueprint transfer test

*Primary experiments: C02 (budget x model grid), C11 (vs ensemble), C17 (vs PCA/mRMR/RFE/random), C10 Phase 3 (stability), C24 (blueprint transfer).*

**[Figure 1: Framework overview schematic]**

---

### Hybrid gains across domains

*Curated datasets across domains — interesting cases, not exhaustive. Full tables in supplementary.*

- Strong gains: [TBD — datasets where hybrid >> best single rep]
- Moderate gains: [TBD — consistent but smaller improvement]
- Near-ceiling: [TBD — best single rep already >0.9, hybrid adds little]
- Honest negatives: [TBD — cases where hybrid does not help or hurts]
- What predicts gain magnitude? Correlate delta with dataset properties (n, number of reps, effective rank, domain)

*Primary experiments: C14 (cross-domain summary), precompute results (40 datasets x 6 conditions). Analysis: gain predictor regression (A4), near-ceiling detection (A8).*

**[Figure 2: Case study panel — grouped by domain, showing delta over best single rep]**

---

### Feature budget and model capacity interact

*The core mechanism — budget x model capacity interaction.*

- Within-family gradient (C06): for a fixed dataset, models within the same family (e.g. RF with varying n_estimators/max_depth) prefer different budgets. More constrained models prefer smaller budgets.
- Budget x capacity heatmap (C02): systematic interaction between feature budget and model type. Tree models tolerate high budgets; distance-sensitive models (kNN, SVM, GP) degrade.
- ANOVA decomposition (C09): which factor contributes most to performance variance — model, budget, allocation, importance method, or their interactions?
- *Flexible: emphasis will shift based on which result is most compelling. Could feature C16 (budget as regularization), C03 (cross-dataset transfer), C15 (ML theory depth), or C17 (information-theoretic decomposition).*

*Primary experiments: C06 (within-family gradient), C02 (heatmap), C09 (ANOVA/MI/rank). Backup: C03, C15, C16, C17.*

**[Figure 3: Complexity matching — budget x capacity interaction, within-family gradient or heatmap]**

---

### Hybrid representations improve calibration and noise robustness

*Hybrids produce more reliable predictions, not just higher scores.*

- Similarity / applicability domain: hybrid representations change the geometry of the descriptor space, potentially resolving activity cliffs that appear in one representation but not another. SAC (similarity-accuracy correlation) comparison between single reps and hybrids.
- Calibration: conformal prediction intervals (regression) and prediction sets (classification). Are hybrid-based models better calibrated? Does the matched model-budget configuration improve calibration?
- Noise robustness: noise degradation slope — how quickly does performance degrade as label noise increases? Hybrids with controlled redundancy from multiple reps may degrade more gracefully.
- OPIG case studies where relevant (antibody developability, binding prediction, transfer learning)

*Primary experiments: S02 (SAC), C08 (calibration), C10 (noise NDS). Backup: S04 (cliffs), S06 (AD), OPIG experiments.*

**[Figure 4: Downstream benefits — 2-3 panels from similarity, calibration, noise robustness]**

---

### Forward compatibility and fine-tuning

*New models slot in without retraining. The blueprint is portable.*

- 5-condition ablation (C13): (A) legacy reps only, (B) legacy + new rep, (C) new rep only, (D) full pool, (E) random control. Wilcoxon signed-rank test across datasets.
- New foundation model embeddings tested: ESM3, SaProt, Boltz-2, MINT, Ab-RoBERTa (depending on domain)
- Fine-tuned model variants (LoRA MolFormer, ESM-2 LoRA, GraphMVP-FT) as another "new rep" source — fine-tuning produces a different embedding, which KIRBy treats identically to any other rep
- Temporal framing: note release dates of each new rep relative to KIRBy development to illustrate the forward-compatibility claim

*Primary experiments: C13 (forward compatibility ablation). Config variants: FT reps as "new" reps.*

**[Figure 5: Forward compatibility — ablation showing performance with/without new reps]**

---

### [Flexible slot]

*Whatever surprising result emerges from full runs. Candidates:*

- Explainability (C22): which features from which reps are selected, and do they align with known biology?
- Modality ablation (C12): TCGA — systematically remove one omics modality at a time from the hybrid
- Learning curves (C05): how does hybrid advantage scale with dataset size?
- Fast vs full tier comparison (C25): does the fast tier converge to the same best configuration as the full tier, and at what computational savings? Run across all datasets.
- A striking negative result that deserves honest discussion

*This section may be absorbed into another section or dropped entirely depending on results.*

**[Figure 6: Flexible — TBD based on results]**

---

## Discussion

*[TODO: 1-2 sentence summary of key findings once results are available — e.g., "Across X datasets spanning five domains, hybrid representations outperformed the best individual representation in Y% of cases, with a median improvement of Z. The advantage was largest for datasets where multiple representations contributed complementary features, and smallest where a single source already captured most of the relevant signal."]*

What does this mean in practice? When a new foundation model is published — as happens every few months in molecular, protein, and antibody modelling — a researcher using KIRBy does not need to decide whether to replace their current representation or retrain their pipeline. They compute the new model's embeddings, add the matrix to the pool, and re-run the search. The framework determines whether the new embedding contributes useful features and, if so, how many and which ones. In a field where representation choice is often determined by familiarity or lab convention rather than by systematic comparison, this is a substantive change: it replaces a single bet on one representation with a principled, task-specific combination that is updated as the pool grows.

Concatenation — the most common method for combining representations — produces the full feature set that KIRBy starts from, but on biological sample sizes it can degrade rather than help^[Bellman1961,Stahlschmidt2022]. Late fusion sidesteps the dimensionality problem by training separate models per representation and aggregating predictions, though it loses interactions between features from different sources^[Steyaert2023]. Intermediate fusion (MOGONET^[Wang2021], cross-attention^[Zhang2024]) captures those interactions — at the cost of retraining the entire model whenever a new representation is added. Multiple kernel learning^[Gonen2011] learns kernel-level weights, as in pairwiseMKL for drug bioactivity^[Cichonska2018], but produces no interpretable feature-level selections. Consensus methods^[Valsecchi2020] combine predictions, not features. They cannot exploit cases where one representation's noise is another's signal. KIRBy works at the feature level instead, selecting complementary features that span representation boundaries while remaining modular — any new encoder's output is just another matrix in the pool. Perhaps the closest analogy in the optimisation literature is the CASH problem^[Thornton2013] (the joint selection of algorithm and hyperparameters), extended here to include the input representation as an additional search dimension. Tishby and Zaslavsky^[Tishby2015] showed why compression helps: the information bottleneck principle states that optimal representations maximise mutual information with the target while minimising mutual information with the input. In effect, the budget forces the model to keep only what matters, with the feature count controlling the trade-off between informativeness and dimensionality. *[TODO: Reference complexity matching results — e.g., "The budget-capacity interaction we observed (Fig. X) is consistent with this principle: distance-sensitive models preferred lower budgets, while tree ensembles tolerated larger feature sets."]*

Ambroise and McLachlan^[Ambroise2002] demonstrated the extreme case of feature selection overfitting: an SVM achieved near-zero cross-validated error on random labels when gene selection was performed on the full dataset before cross-validation, but when selection was nested inside each fold the median error was 11 out of 40 — close to the 50% expected by chance. Vabalas et al.^[Vabalas2019] showed that the bias introduced by pre-selection can exceed the bias from hyperparameter tuning, and Cawley and Talbot^[Cawley2010] argued that feature selection is itself a form of model selection whose complexity cost must be accounted for. KIRBy follows the standard remedy: all importance scores are computed within each cross-validation fold on training data only. But the problem of "search degrees of freedom"^[Tibshirani2015] remains — searching over many possible feature subsets adds complexity even when each individual selection is unbiased. The budget acts as a hard capacity constraint: selecting 100 features from a pool of 5,000 limits the space of possible selections far more tightly than L1 regularisation over the same pool, because the feasible set grows polynomially in the budget while the continuous parameter space of a regularised model is unbounded^[Ng2004]. The features being selected are frozen embeddings produced by external encoders, not parameters being fitted. Choosing which pre-existing knowledge to trust is a less expressive operation than learning new features, and Kumar et al.^[Kumar2022] showed that frozen embeddings fed through a linear probe can outperform full fine-tuning by up to 10% in out-of-distribution accuracy precisely because fine-tuning distorts pretrained structure. And stability can be measured directly: the Nogueira index^[Nogueira2018] (a stability metric for feature selection) quantifies the consistency of selected feature sets across folds. If selection were overfit, feature sets would differ wildly from fold to fold. *[TODO: Report stability and transfer results — e.g., "The mean Nogueira index was X (range Y-Z), and Blueprint transfer incurred a median performance change of W."]*

We want to be clear about the framework's limitations. For small datasets — below approximately 200 samples — the hybrid search cannot reliably outperform the best single representation. The internal validation set within each fold shrinks to roughly 20 samples, and at that size the evaluator cannot distinguish genuine improvements from noise. *[TODO: Cite F10/F11 experimental evidence.]* This seems to be a structural constraint, not one that tuning can fix. At the other extreme, frozen embeddings have a ceiling. On tasks where the true signal requires end-to-end learning from raw inputs — certain 3D quantum-mechanical properties, for example, where SchNet outperforms any frozen embedding by a wide margin — the framework cannot improve on what the representations themselves contain. Integrating 20 to 50 foundation models into a single pipeline, each with its own dependencies, GPU requirements, and version constraints, is genuinely difficult engineering. Representations need only be computed once per dataset and everything is cached, but the initial integration burden is real and scales with the number of models in the pool. Finally, on endpoints with high experimental noise, even a well-constructed hybrid may not improve on simpler baselines. Kramer et al.^[Kramer2012] documented that experimental uncertainty in heterogeneous public K_i data is approximately 0.54 log units, setting a natural accuracy ceiling of R² ≈ 0.81 regardless of model sophistication — the budget constraint provides a guard against overshoot, but it cannot push past the noise floor.

*[TODO: Future directions — 2-3 sentences grounded in demonstrated results. Candidates: (1) Active learning in hybrid space — use hybrid uncertainty to guide data acquisition, since the framework naturally identifies which regions of descriptor space are sparsely covered. (2) Federated learning — Blueprint portability suggests applicability to privacy-preserving settings where raw data cannot be pooled but representations can be exchanged. (3) Meta-learning across tasks — learn which features tend to be selected for similar endpoints to warm-start the search on new datasets. Link each to specific experimental evidence once available.]*

---

## Methods

The KIRBy framework treats pre-computed representations as independent frozen feature sets and searches for the combination of features across these sets that best predicts a given endpoint. The search is task-specific: the same set of input representations may produce different hybrids for different prediction targets, different metrics, or different downstream models.

### Hybrid search framework

**Pipeline.** Given *K* frozen representation matrices **R**₁, ..., **R**_K for the same *n* samples, a label vector **y**, and optional group assignments for cross-validation, the search identifies which features from which representations maximise predictive performance for a specified metric and downstream model. The output is a hybrid feature matrix of *b* columns, where *b* is the feature budget, together with a Blueprint that records all parameters needed to reproduce the identical feature selection on new data. The search first evaluates each representation independently, then scores feature importance on training data within each cross-validation fold, allocates the budget across representations according to baseline performance, and scores candidate hybrid configurations via cross-validation. The best configuration is retained for each combination of downstream model and budget. Every selected feature can be traced to its source representation and ranked by importance, providing a complete provenance record.

**Preprocessing.** Because representations from different sources differ in scale, distribution, and missingness patterns, all preprocessing is performed within each cross-validation fold to prevent information leakage from the validation set. Missing values are imputed with per-feature medians computed on the training fold; columns that are entirely missing in the training set are dropped. Binary feature matrices such as molecular fingerprints are left unstandardised. All other features are standardised to zero mean and unit variance using training-fold statistics only. Details are provided in Supplementary Methods.

**Feature importance and budget allocation.** Three importance methods score each feature within each representation: a tree-based method (LightGBM gain), an explanation-based method (TreeSHAP), and a neural attribution method (DeepLift). These were selected from an initial pool of eight methods spanning tree-based, gradient-based, and model-agnostic families, retaining one representative per cluster after redundancy analysis across datasets (Supplementary Fig. X). An allocation strategy then distributes the total feature budget across representations. We evaluated 31 candidate strategies and retained 13 that appeared in top-performing configurations across multiple datasets and model types (Supplementary Fig. X). The 13 retained strategies fall into three families. Score-based strategies distribute the budget in proportion to transformed baseline scores — for example, restricting features to the top *k* representations, or penalising redundancy between correlated representations. A floor-based variant guarantees every representation a minimum allocation, while a greedy forward approach iteratively adds features from whichever representation most improves the internal evaluator, with early stopping. Full definitions are given in Supplementary Table X. Once the allocation is determined, the top features from each representation are selected by importance score and concatenated to form the hybrid.

**Internal evaluation.** Each candidate configuration is scored by an internal evaluator fitted on the training fold and scored on the held-out validation fold. Five evaluators are used: a matched evaluator that uses the same model class as the downstream model under evaluation, and four fixed evaluators spanning different inductive biases (random forest, SVM, LightGBM, MLP). The matched evaluator ensures that features are selected jointly with the model that will use them — because variable importance is model-dependent^[Dong2020]. Evaluator hyperparameters are listed in Supplementary Table 4.

**Search configuration.** The full search evaluates all combinations of budgets, allocations, importance methods, downstream models, and internal evaluators under 5-fold cross-validation — approximately 15,000 configurations for a typical dataset with 7 models. A reduced fast tier using 3 allocations, 2 importance methods, and 3-fold cross-validation is available for exploratory analysis (Supplementary Methods). Models sensitive to high dimensionality are capped at 2,000 or 3,000 features depending on model type (Supplementary Table 3). The complete search result is cached to disk. Once computed, subsequent requests for the same dataset return immediately.

**Blueprint.** Each search produces a Blueprint recording all parameters — representation names, allocation strategy, importance method, budget, evaluator, metric — required to reproduce the hybrid on new data without re-running the search.

**Augmentation.** Some features are too low-dimensional or too specialised to participate in the budget allocation stage — MACCS fingerprints (167 features), for instance, or log-likelihood ratios (3 features). These augmentation features are evaluated after the base hybrid is constructed: they are pooled, ranked by random forest feature importance, and appended sequentially to the hybrid only if they improve the internal evaluator score (Supplementary Methods).

### Evaluation metrics

The primary metric is domain-specific: RMSE for molecular property prediction, Spearman rank correlation for antibody regression and developability, AUROC for binary classification (including antibody classification and tumour subtype), weighted F1 for multiclass classification, concordance index for drug-target affinity, and R² for reaction yield. Ten metrics are supported in total: R², RMSE, MAE, accuracy, AUROC, Spearman ρ, Pearson r, concordance index, weighted F1, and Matthews correlation coefficient. Every Blueprint and search result carries a direction flag indicating whether lower or higher values are better, so that all comparisons — within the search, across experiments, and in the Blueprint — use a consistent ordering.

### Representations

For each domain, the representation pool spans classical descriptors and fingerprints, graph neural network embeddings, pretrained language model embeddings, and domain-specific features. The molecular pool contains 20 to 30 representations per dataset, ranging from fingerprints (ECFP4, MACCS, Avalon) and physicochemical descriptors (Mordred, ~1,600 dimensions) to neural embeddings from graph networks (GraphMVP, GROVER, MoLCLR) and language models (MoLFormer, ChemBERTa). Antibody representations draw on protein language models with multiple extraction strategies (ESM-2 full-sequence, CDR-only, CDR3-only, per-region, and several layer variants), antibody-specific models (AntiBERTy, AbLang2, Ab-RoBERTa, IgT5), and sequence features (BLOSUM substitution matrices, one-hot encodings). For cancer genomics, representations span gene expression, copy-number variation, somatic mutations, methylation, protein expression, and pre-computed signatures from the PanCanAtlas. Per-dataset representation pools are configured in a central registry. Complete representation lists with dimensionalities are provided in Supplementary Table 2.

### Downstream models

Thirteen models are available, spanning tree ensembles (random forest, XGBoost, LightGBM, CatBoost, NGBoost), kernel methods (SVM, Gaussian process), nearest-neighbour (kNN), linear models (Ridge regression with cross-validated regularisation for regression, logistic regression with cross-validated regularisation for classification), and neural networks (single-layer MLP, three-layer DNN, MC-Dropout BNN (dropout kept active at inference) for uncertainty quantification, and a multi-scale 1D CNN). Standard experiments use seven of these: random forest, XGBoost, LightGBM, kNN, SVM, Gaussian process, and MLP. The full set of thirteen is used for model-matching and noise experiments. Models with cubic or higher scaling are skipped automatically for large datasets — Gaussian processes above 6,000 samples, SVMs above 10,000. All models use fixed hyperparameters across experiments; no per-dataset tuning is performed. Hyperparameters are listed in Supplementary Table 3.

### Datasets and splitting

We evaluate KIRBy across five biological domains: molecular property prediction, antibody engineering, drug-target interaction, cancer genomics, and chemical reactions. Molecular datasets cover ADMET endpoints (Caco-2 permeability, hERG inhibition, LogD, microsomal stability, MDR1 efflux) and quantum-mechanical properties from QM9. For antibodies, tasks span thermostability, pharmacokinetics, polyreactivity, aggregation, immunogenicity, binding affinity, and developability. Drug-target interaction datasets focus on kinase and GPCR binding affinity, while cancer genomics tasks include tumour subtype classification and survival-associated phenotypes. Reaction datasets cover yield prediction from high-throughput experimentation plates. Dataset sizes range from 94 to approximately 50,000 samples, and representation dimensionalities from 3 features (log-likelihood ratios) to 2,048 (extended-connectivity fingerprints), with up to 57 representation sources per domain. Per-dataset details are provided in Supplementary Table 1.

No random splits are used in any experiment. Splits match domain conventions: Murcko scaffold splits for molecular datasets, sequence-cluster or edit-distance groups with GroupKFold for antibodies, cold-drug splits for drug-target interaction (evaluating generalisation to unseen compounds), stratified group splits for cancer genomics, and leave-plate-out splits for reaction datasets. All experiments use 5-fold cross-validation.

### Statistical analysis

Within-dataset comparisons use the Nadeau-Bengio corrected paired t-test, which adjusts for the dependence between training and test sets introduced by k-fold cross-validation^[NadeauBengio2003]. Cross-dataset comparisons — for example, whether the hybrid outperforms the best single representation across all molecular datasets — use the Wilcoxon signed-rank test on paired per-dataset scores. When ranking more than two methods, the Friedman test is applied with Nemenyi post-hoc correction^[Demsar2006]. This controls the family-wise error rate. Effect sizes are reported as rank-biserial correlation for Wilcoxon tests and Kendall's W for Friedman tests.

### Computational infrastructure

All experiments were run on the Advanced Research Computing (ARC) cluster at the University of Oxford, managed via the SLURM scheduler. Representation computation (pretrained language models, graph neural networks) used GPU nodes; hybrid search and downstream model evaluation used CPU nodes with 8 cores per job. The full search is run once per dataset and the result cached to disk. Subsequent analyses — applying a Blueprint, evaluating downstream models, computing stability indices — retrieve the cached result and complete in seconds.

### Data and code availability

The KIRBy framework is available as an open-source Python package at [URL]. All datasets used in this study are publicly available; sources and download instructions are listed in Supplementary Table 1. Pre-computed representation matrices and search results for all 40 datasets will be deposited at [URL].

---

## Extended Data / Supplementary

**Supplementary Table 1:** Per-dataset details (sample sizes, representation counts, task types, splitting strategies, primary metrics).

**Supplementary Table 2:** Per-dataset representation pools with dimensionalities and sources across all five domains.

**Supplementary Table 3:** Downstream model hyperparameters for all 13 models, including per-model budget caps and sample-size guards.

**Supplementary Table 4:** Internal evaluator hyperparameters for all 5 evaluators.

**Supplementary Methods:** Preprocessing details (clipping, imputation, binary detection, standardisation). Feature importance method hyperparameters and selection procedure (8 candidate methods, cluster analysis, 3 retained). All 13 allocation strategy definitions with formulas (31 candidates, pruning criteria, 13 retained). Speed tier configurations (fast and full). Per-model budget caps. Augmentation procedure details (sequential testing, step size, patience). Computational cost analysis (importance pre-computation, allocation score caching, feature selection deduplication via index hashing, disk caching with deterministic hashing, factory pre-computation for fine-tuned representations). Internal evaluator selection and tuning.
