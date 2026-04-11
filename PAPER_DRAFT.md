# KIRBy: Knowledge Integration by Representation for task-adaptive prediction across biological domains

Pretrained models for molecules, proteins, antibodies, and gene expression now number in the dozens, yet no single representation consistently gives the best predictions. Combining their outputs should help: different encodings reflect different aspects of the underlying biology. But the most common combination strategy, concatenation, yields feature sets that are too large for biological sample sizes. Performance degrades rather than improves. Here we present KIRBy (Knowledge Integration by Representation), a method that takes frozen embeddings from any number of sources and selects a compact set of features, drawn from multiple sources, that is tailored to a specific prediction task. We test KIRBy on [X] datasets in five domains (molecular properties, antibody engineering, drug-target affinity, cancer genomics, and chemical reactions) with up to 57 sources per domain. Hybrids outperform the best individual source on [Y] of [X] datasets (median improvement: [Z]). The feature budget interacts with model capacity: distance-sensitive models such as kNN and SVM benefit most from dimensionality reduction, while tree ensembles tolerate larger feature sets. Because all inputs are frozen, new foundation model embeddings can be added and tested without retraining.

---

**Draft — Nature Machine Intelligence**

*Note: Citation numbers are placeholders [X]. Final numbering will be assigned during manuscript preparation. All citations below are traceable to PAPER_REFERENCES.md or the lit review.*

---

## Introduction

A comparison of 62,820 models in molecular property prediction benchmarks showed that classical RDKit2D descriptors matched or outperformed learned features from graph neural networks and pretrained language models on the majority of datasets^[Deng2023]. Descriptor-based models occupied 73% of top-three positions on 11 benchmarks^[Jiang2021], and in 1,310 ChEMBL assays, feedforward networks on ECFP fingerprints outperformed graph convolutions and SMILES-based recurrent models^[Mayr2018]. The input often matters more than the model. Yet we still lack a systematic way to exploit this. No single representation wins. The same pattern holds outside small-molecule chemistry: no single protein language model consistently achieves the highest performance on all tasks^[Kenlay2024,Schmirler2024]; for antibodies, general-purpose models sometimes outperform those built for the domain^[Kenlay2024]. In computational pathology, no individual foundation model dominates on clinical endpoints^[ElNahhas2025], and in cancer genomics, integrating additional omics layers can negatively impact performance when the data types are not properly combined^[Steyaert2023].

For small molecules alone, the landscape now includes extended-connectivity fingerprints, physicochemical descriptor sets exceeding 1,600 dimensions, graph neural network embeddings^[Yang2019,Fang2022], language models trained on billions of SMILES strings^[Ross2022], image-based encodings (molecules rendered as 2D images)^[Zeng2022], and stereoelectronics-infused molecular graphs^[Boiko2025]. The same is true for proteins (ESM-2^[Lin2023], ProtTrans^[Elnaggar2022]), antibodies (AntiBERTy^[Ruffolo2021], AbLang^[Olsen2022]), histopathology^[ElNahhas2025], and multi-omics cancer profiles^[Steyaert2023]. A recent review counted over 40 molecular foundation models published between 2020 and 2024^[Song2026]. Increasingly, new models appear faster than we can compare them^[Bommasani2021]. The choice is not getting simpler. Different sources presumably encode distinct aspects of the same biology. Noisy or incomplete on their own, single data modalities somehow give more sensitive and robust predictions when combined with independent signals from other modalities^[Steyaert2023]. An ensemble of frozen embeddings from the top four pathology foundation models, El Nahhas et al. found, achieved a mean AUROC 1.2% higher than the best individual model on 31 clinical tasks^[ElNahhas2025].

So far, the most common strategy for combining feature sets has been concatenation (early fusion), which requires only a single downstream model^[Stahlschmidt2022]. It works when feature sets are homogeneous. When they are not, concatenation degrades^[Stahlschmidt2022], and Li and Deng et al. called the practice "misleading" when the goal is to measure the contribution of individual representation types^[Deng2023]. Late fusion trains a separate model per source and aggregates predictions^[Steyaert2023], avoiding the dimensionality problem. But cross-source feature interactions are lost. To recover cross-source interactions requires intermediate fusion (MOGONET^[Wang2021], cross-attention^[Zhang2024]), but at the cost of end-to-end training on all sources; adding a new source means retraining from scratch. No interpretable feature subsets emerge from multiple kernel learning^[Gonen2011], which operates entirely at the kernel level. There remains, as Steyaert et al. noted, no decisive evidence that any of these strategies is broadly superior^[Steyaert2023].

In practice, most research groups default to one representation and ignore the rest. A cheminformatics lab reaches for ECFP fingerprints; a deep learning group uses MoLFormer or ChemBERTa; a structural biology group extracts ESM-2 embeddings. These may encode quite different aspects of the same underlying biology. But they are rarely compared, let alone combined. This leaves signal on the table. When groups do attempt combination, the most accessible strategy is concatenation. It works at modest scales. Concatenating just three standard sources, however (ECFP4 at 2,048 dimensions, ESM-2 at 1,280, Mordred descriptors at ~1,600), gives a ~5,000-dimensional input for a dataset that may contain only a few hundred labeled samples. At these sample-to-feature ratios, more features mean worse predictions^[Hughes1968,Bellman1961]. Models sensitive to distance, such as kNN and SVM, are affected most severely^[FernandezDelgado2014]. New encodings appear, perhaps faster than we can test them, and no principled method exists for combining them without the dimensionality penalty. Which features from which sources should be kept? Does the answer change with the dataset, the endpoint, the downstream model?

The pace of new model development shows no sign of slowing, and extends well beyond small molecules into protein language models^[Lin2023,Elnaggar2022], encoders built for antibodies^[Ruffolo2021,Olsen2022], histopathology foundation models^[ElNahhas2025], and gene expression. If a new protein language model appears next month, must we retrain from scratch to determine whether it adds value? Forward compatibility is therefore essential: any practical method must incorporate new sources without retraining or architectural modification. Deep fusion approaches fail this requirement, since adding a new modality means retraining the entire network. But treat encodings as independent frozen feature sets, and the problem disappears: a new model's output joins the pool and is tested alongside existing sources with no changes to the procedure.

Here, we present KIRBy (Knowledge Integration by Representation), a method that takes a collection of frozen feature matrices for a dataset and returns a customized hybrid feature set optimized for a specific prediction task. The inputs can be anything that gives a feature vector per sample: fingerprints, descriptor sets, foundation model embeddings, gene expression signatures, histopathology features. From these sources, KIRBy selects a compact feature subset, searching over budgets, allocation strategies, importance methods, and downstream models. The search runs once per dataset. A portable Blueprint stores the result; applying it to new data does not require repeating the search. We test KIRBy on datasets in five biological domains (molecular property prediction, antibody engineering, drug-target interaction, cancer genomics, and chemical reactions) with up to 57 sources per domain. Most features in any individual source turn out to be uninformative for a given task. So the budget discards them. Features drawn from multiple sources carry complementary signal and outperform the best single source on the majority of datasets tested *[TODO: insert fraction and median improvement once results are final]*. KIRBy is also forward-compatible: when new foundation model embeddings become available, they are added to the pool and the search determines whether and how to use them. Below we open with complexity-matching results, then benchmark KIRBy against published methods on six antibody tasks. The paper closes with forward-compatibility experiments: encodings released after the method was built slot in and sometimes lift the performance ceiling.

---

## Results

### The hybrid representation

*What the hybrid looks like in practice — concrete example, not pipeline description (pipeline is in Methods/Fig. 1).*

- Example dataset walkthrough: which sources contribute features, how many from each, which features are selected, what the final hybrid looks like
- Feature selection overfitting controls: per-fold importance, cross-fold stability (Jaccard + Nogueira index), blueprint transfer test
- Most features in any individual source are uninformative for a given task; the budget discards the majority

*[Editorial: experiments — stability, blueprint transfer.]*

**[Figure 1: Framework overview schematic + example hybrid composition]**

---

### Cross-domain hybrid gains

*Curated datasets across domains — interesting cases, not exhaustive. Full tables in supplementary.*

- Strong gains: [TBD — datasets where hybrid >> best single rep]
- Moderate gains: [TBD — consistent but smaller improvement]
- Near-ceiling: [TBD — best single rep already >0.9, hybrid adds little]
- Honest negatives: [TBD — cases where hybrid does not help or hurts]
- What predicts gain magnitude? Correlate delta with dataset properties (n, number of reps, effective rank, domain)
- Comparison to alternatives at matched dimensionality: concatenation (full-dim budget = concat baseline), PCA, model ensemble (separate models per rep, averaged predictions), and existing feature selection methods (mRMR, RFE, random)

*[Editorial: experiments — cross-domain summary, precompute (40 datasets × 6 conditions), gain predictor regression, near-ceiling detection, vs ensemble, vs PCA/mRMR/RFE/random.]*

**[Figure 2: Case study panel — grouped by domain, showing delta over best single rep and alternative combination methods]**

---

### Feature budget and model capacity interact

*The core mechanism — budget x model capacity interaction.*

- Within-family gradient: for a fixed dataset, models within the same family (e.g. RF with varying n_estimators/max_depth) prefer different budgets. More constrained models prefer smaller budgets.
- Budget × capacity heatmap: systematic interaction between feature budget and model type. Tree models tolerate high budgets; distance-sensitive models (kNN, SVM, GP) degrade.
- Variance decomposition: which factor contributes most to performance variance (model, budget, allocation, importance method, or their interactions)?
- *Flexible: emphasis will shift based on which result is most compelling. Could feature budget as regularization, cross-dataset transfer, ML theory depth, or information-theoretic decomposition.*

*[Editorial: experiments — within-family gradient, budget×capacity heatmap, ANOVA/MI/rank decomposition. Backup: cross-dataset transfer, ML theory, budget-as-regularization, info-theoretic.]*

**[Figure 3: Complexity matching — budget x capacity interaction, within-family gradient or heatmap]**

---

### OPIG benchmark validations

*Antibody-domain case studies against published OPIG benchmarks. Tests whether sequence-based KIRBy hybrids are competitive with task-specific methods.*

- Trastuzumab-HER2 binding (B1): binary classification on Chinery et al. DMS library (~5K subsample). KIRBy supervised classification vs paper's generative library design; different tasks, but tests whether LM-based reps find binders
- Graphinity DDG prediction (B2): Hummer et al. synthetic ΔΔG. Sequence-only reps vs paper's 3D EGNN. Tests whether sequence encodings can match structure-based methods for mutation effect prediction
- Humanness classification (B3): Prihoda et al. BioPhi. Human vs non-human VH classification. KIRBy reps vs pre-computed humanness scores
- CDR flexibility prediction (B4): Spoendlin et al. ITsFlexible. Very small samples (n~34–200 per CDR after dedup). Tests sequence-only baselines on a task the original paper solved with 3D structures
- Ginkgo developability (B5): van Niekerk et al. 9 regression properties (HIC, Tm, titer, polyreactivity, etc.) on 242 antibodies. Pre-computed hierarchical cluster splits. Compares KIRBy sequence-based reps with benchmark's 16 models (many structure-based)
- Activity cliff QSAR (B6): Dablander et al. Compares KIRBy hybrids vs paper's RF/KNN/MLP/GIN. Key difference: paper uses random splits + hyperparameter tuning; KIRBy uses scaffold splits + fixed defaults

*[Editorial: experiments — B1 trastuzumab, B2 graphinity, B3 humanness, B4 CDR flexibility, B5 ginkgo, B6 activity cliff QSAR.]*

**[Figure 4: OPIG benchmarks — performance comparison panels across B1–B6]**

---

### Forward compatibility and fine-tuning

*New models slot in without retraining. The blueprint is portable.*

- 5-condition ablation (C13): (A) legacy reps only, (B) legacy + new rep, (C) new rep only, (D) full pool, (E) random control. Wilcoxon signed-rank test across datasets
- New foundation model embeddings tested: ESM3, SaProt, Boltz-2, MINT, Ab-RoBERTa (depending on domain)
- Fine-tuned model variants (LoRA MolFormer, ESM-2 LoRA, GraphMVP-FT) as another "new rep" source: fine-tuning yields a different embedding, which KIRBy treats identically to any other rep
- Temporal framing: note release dates of each new rep relative to KIRBy development to illustrate the forward-compatibility claim

*[Editorial: experiments — C13 forward compatibility ablation (5 conditions). Fine-tuning comparisons (molecular, antibody, DTA).]*

**[Figure 5: Forward compatibility — ablation showing performance with/without new reps]**

---

### [Flexible slot]

*Whatever surprising result emerges from full runs. Candidates:*

- Noise robustness (C10): noise degradation slope: do hybrids degrade more gracefully than single reps as label noise increases?
- Calibration (C08): conformal prediction intervals: are hybrid-based models better calibrated?
- Explainability (C22): which features from which reps are selected, and do they align with known biology?
- Modality ablation (C12): TCGA: systematically remove one omics modality at a time from the hybrid
- Learning curves (C05): how does hybrid advantage scale with dataset size?
- Fast vs full tier convergence (C25): does the reduced search tier find the same best configuration as the full tier, and at what computational savings?
- A striking negative result that deserves honest discussion

*This section may be absorbed into another section or dropped entirely depending on results.*

**[Figure 6: Flexible — TBD based on results]**

---

## Discussion

*[TODO: 1-2 sentence summary of key findings once results are available — e.g., "Across X datasets spanning five domains, hybrid representations outperformed the best individual representation in Y% of cases, with a median improvement of Z. The advantage was largest for datasets where multiple representations contributed complementary features, and smallest where a single source already captured most of the relevant signal."]*

What does this mean in practice? A new foundation model appears every few months in molecular, protein, or antibody modeling. A researcher using KIRBy does not need to decide whether to replace their current encoding or retrain their pipeline; they compute the new model's embeddings, add the matrix to the pool, and re-run the search. The search determines whether the new embedding contributes useful features and, if so, how many and which ones. In a field where encoding choice is often determined by familiarity or lab convention rather than by systematic comparison, this is a substantive change. It replaces a single bet with a principled combination, tailored to each task, that updates as the pool grows.

Concatenation, the most common method for combining feature sets, gives the full pool that KIRBy starts from, but on biological sample sizes it can degrade rather than help^[Bellman1961,Stahlschmidt2022]. What late fusion gains by training separate models per source^[Steyaert2023] it loses in cross-source feature interactions. Those interactions can be recovered through intermediate fusion (MOGONET^[Wang2021], cross-attention^[Zhang2024]), but only at the cost of retraining whenever a new source appears. At the kernel level, multiple kernel learning^[Gonen2011] (e.g., pairwiseMKL for drug bioactivity^[Cichonska2018]) can learn source weights, but no interpretable feature-level selections result. Consensus methods^[Valsecchi2020] combine predictions, not features, and so cannot exploit cases where one source's noise is another's signal. Instead, KIRBy works at the feature level: it selects features that span source boundaries, and because any new encoder's output is just another matrix in the pool, the approach is modular by construction. Perhaps the closest analogy in the optimization literature is the CASH problem^[Thornton2013] (the joint selection of algorithm and hyperparameters), extended here to include the input encoding as an additional search dimension. Tishby and Zaslavsky^[Tishby2015] showed why compression helps: the information bottleneck principle states that optimal representations maximize mutual information with the target while minimizing mutual information with the input. In effect, the budget forces the model to keep only what matters. The feature count controls the trade-off between informativeness and dimensionality. *[TODO: Reference complexity matching results — e.g., "The budget-capacity interaction we observed (Fig. X) is consistent with this principle: distance-sensitive models preferred lower budgets, while tree ensembles tolerated larger feature sets."]*

Ambroise and McLachlan^[Ambroise2002] showed the extreme case of feature selection overfitting: an SVM achieved near-zero cross-validated error on random labels when gene selection was performed on the full dataset before cross-validation, but when selection was nested inside each fold the median error was 11 out of 40, close to the 50% expected by chance. Vabalas et al.^[Vabalas2019] showed that the bias introduced by pre-selection can exceed the bias from hyperparameter tuning, and Cawley and Talbot^[Cawley2010] argued that feature selection is itself a form of model selection whose complexity cost must be accounted for. KIRBy follows the standard remedy: we compute all importance scores within each cross-validation fold, on training data only. But the problem of "search degrees of freedom"^[Tibshirani2015] remains. Searching over many possible feature subsets adds complexity even when each individual selection is unbiased. The budget acts as a hard capacity constraint: selecting 100 features from a pool of 5,000 limits the space of possible selections far more tightly than L1 regularization over the same pool, because the feasible set grows polynomially in the budget while the continuous parameter space of a regularized model is unbounded^[Ng2004]. Important here is a distinction: the features being selected are frozen embeddings, not parameters being fitted. Choosing which pre-existing knowledge to trust is, we think, a less expressive operation than learning new features, and Kumar et al.^[Kumar2022] showed that frozen embeddings fed through a linear probe can outperform full fine-tuning by up to 10% in out-of-distribution accuracy precisely because fine-tuning distorts pretrained structure. And stability can be measured directly: the Nogueira index^[Nogueira2018] (a stability metric for feature selection) quantifies the consistency of selected feature sets between folds. If selection were overfit, feature sets would differ wildly from fold to fold. *[TODO: Report stability and transfer results — e.g., "The mean Nogueira index was X (range Y-Z), and Blueprint transfer incurred a median performance change of W."]*

We want to be clear about KIRBy's limitations. For small datasets (below about 200 samples), the hybrid search cannot reliably outperform the best single source. We suspect this is because the internal validation set within each fold shrinks to roughly 20 samples, and at that size the evaluator cannot distinguish genuine improvements from noise. *[TODO: Cite F10/F11 experimental evidence.]* Tuning cannot fix this. And at the other extreme, frozen embeddings have a ceiling. No signal can be added beyond what the inputs already contain, and on tasks where the true signal requires end-to-end learning from raw inputs (certain 3D quantum-mechanical properties, for example, where SchNet outperforms any frozen embedding by a wide margin), this ceiling binds. Although the methodology naturally accommodates embeddings from sequence-trained models (ESM-2, ProtBERT, etc.), encoding raw sequences directly as one-hot or k-mer features disperses the positional and compositional information that makes sequences useful in the first place. Integrating 20 to 50 foundation models into a single pipeline, each with its own dependencies, GPU requirements, and version constraints, is genuinely difficult engineering. Embeddings need only be computed once per dataset and everything is cached. Yet the initial integration burden is real and scales with the number of models in the pool. Finally, on endpoints with high experimental noise, even a well-constructed hybrid may not improve on simpler baselines. Kramer et al.^[Kramer2012] documented that experimental uncertainty in heterogeneous public K_i data is approximately 0.54 log units. This sets a natural accuracy ceiling of R² ≈ 0.81 regardless of model sophistication; the budget constraint plausibly guards against overshoot, but it cannot push past the noise floor.

*[TODO: Future directions — 2-3 sentences, each grounded in a specific Results finding. Candidates: (1) Forward compatibility as a living benchmark — the C13 ablation shows new reps can be tested without retraining, so the framework could serve as a continuous evaluation harness as new foundation models appear (link to forward compatibility results). (2) Meta-learning across tasks — the OPIG benchmarks and cross-domain results show that certain representation families recur in top hybrids; learning which features tend to be selected for similar endpoints could warm-start the search on new datasets (link to OPIG and cross-domain patterns). (3) Budget-capacity matching as model selection — the complexity matching results suggest the optimal budget encodes information about dataset difficulty; this could inform automated model selection beyond feature selection (link to C02/C06/C09 results). Each must cite a specific figure or table from this paper.]*

---

## Methods

KIRBy treats pre-computed feature sets as independent frozen inputs and searches for the combination of features from these sets that best predicts a given endpoint. The search runs separately for each task: the same set of input sources may yield different hybrids for different prediction targets, different metrics, or different downstream models. Below we describe the search pipeline and preprocessing, the feature scoring and budget allocation procedure, the internal evaluation strategy, the metrics and models used, and the datasets and splitting conventions. Exact hyperparameters, allocation strategy formulas, and importance method details are in Supplementary Methods.

### Hybrid search pipeline

Treating each encoding as a frozen feature set, rather than learning a joint embedding end-to-end, makes the approach modular: any new encoder's output can be added to the pool without retraining, and the selection process itself is decoupled from how the features were produced. Given *K* frozen representation matrices **R**₁, ..., **R**_K for the same *n* samples, a label vector **y**, and optional group assignments for cross-validation, the search finds which features from which sources best predict for a specified metric and downstream model (Fig. 1). The output is a hybrid feature matrix of *b* columns, where *b* is the feature budget, together with a Blueprint that records all parameters needed to reproduce the identical feature selection on new data. The search first scores each source independently, then scores feature importance on training data within each cross-validation fold, allocates the budget among sources according to baseline performance, and scores candidate hybrid configurations via cross-validation. We retain the best configuration for each combination of downstream model and budget. Every selected feature traces back to its source and importance rank: a complete provenance record.

Inputs from different sources differ in scale, distribution, and missingness patterns, and naively pooling them before splitting would leak validation-set statistics into the training procedure, a well-documented source of optimistic bias in feature selection studies^[Ambroise2002,Vabalas2019]. All preprocessing therefore runs within each cross-validation fold. We impute missing values with per-feature medians computed on the training fold and drop columns that are entirely missing. Binary feature matrices (molecular fingerprints, for instance) skip standardization, since their {0,1} encoding already carries a consistent scale. For all other features, we standardize to zero mean and unit variance using training-fold statistics only. Details are in Supplementary Methods.

### Feature scoring and budget allocation

Not all features in a given source are equally informative for a given task, and, for reasons that are not entirely clear, different importance methods disagree about which features matter. Tree-based measures favor features with high split gain, while gradient-based attributions reflect sensitivity in a neural network's loss landscape. To avoid committing to a single notion of importance, we score features using three methods, one from each family: LightGBM gain (tree-based), TreeSHAP (explanation-based), and DeepLift (neural attribution). We selected these three from an initial pool of eight methods after redundancy analysis on datasets showed that members of the same family gave highly correlated rankings (Supplementary Fig. X). How to divide the total feature budget among sources is the role of the allocation strategy. We tested 31 candidates and retained 13 that appeared in top-performing configurations on multiple datasets and model types (Supplementary Fig. X). The 13 retained strategies fall into three families. Score-based strategies divide the budget in proportion to transformed baseline scores: restricting features to the top *k* sources, for example, or penalizing redundancy between correlated sources. A floor-based variant guarantees every source a minimum allocation, while a greedy forward approach iteratively adds features from whichever source most improves the internal evaluator, with early stopping. Supplementary Table X gives full definitions. Once we have the allocation, we select the top features from each source by importance score and concatenate them to form the hybrid.

Some features are too low-dimensional or too specialized to participate in the budget allocation stage: MACCS fingerprints (167 features), for instance, or log-likelihood ratios (3 features). Excluding them from the allocation would lose potentially useful signal; including them would distort the budget distribution because they cannot contribute many features. We handle these as augmentation: after the base hybrid is constructed, we pool the augmentation features, rank them by random forest feature importance, and append them only if they improve the internal evaluator score (Supplementary Methods).

### Search configuration and evaluation

Features that improve predictions under a random forest may not help a kNN model, and vice versa. This is because variable importance depends fundamentally on the model, as Dong and Rudin showed^[Dong2020]. To address this, the internal evaluator scores each candidate configuration with a model whose inductive bias matches the downstream model. We use five evaluators: a matched evaluator that uses the same model class as the downstream model, and four fixed evaluators with different inductive biases (random forest, SVM, LightGBM, MLP). We fit each on the training fold and score it on the held-out fold. The matched evaluator ensures that features are selected jointly with the model that will use them; the interaction between budget and model capacity is examined in Fig. 3. Evaluator hyperparameters are listed in Supplementary Table 4.

The full search tests all combinations of budgets, allocations, importance methods, downstream models, and internal evaluators under 5-fold cross-validation: about 15,000 configurations per dataset when using 7 downstream models. This combinatorial design is deliberate: it allows us to measure which factors contribute most to performance variance (model choice, budget, allocation, or their interactions) rather than assuming any single axis dominates. We cap models sensitive to high dimensionality at 2,000 or 3,000 features depending on model type (Supplementary Table 3). The complete search result is cached to disk. The search runs once; everything after that is a lookup. Each search yields a Blueprint recording all parameters (source names, allocation strategy, importance method, budget, evaluator, metric) required to reproduce the hybrid on new data without re-running the search. We test Blueprint portability in the forward compatibility experiments (Fig. 5).

### Evaluation metrics

Different biological domains have established different conventions for what constitutes a good prediction, and applying a single metric everywhere would misrepresent performance in at least some of them. To this end, we assign each task a primary metric matched to its domain: RMSE for molecular property prediction, Spearman rank correlation for antibody regression and developability, AUROC for binary classification (including antibody classification and tumor subtype), weighted F1 for multiclass classification, concordance index for drug-target affinity, and R² for reaction yield. We support ten metrics in total (Supplementary Table X); the primary metric for each task follows domain convention.

### Representations

A central claim of this work is that combining features from diverse sources encodes distinct signal. To test this credibly, the representation pool for each domain must cover qualitatively different encoding strategies, not just minor variants of the same approach. Each domain therefore draws on three broad categories: classical descriptors and fingerprints, neural embeddings from graph networks and pretrained language models, and features particular to each domain. The molecular pool (20–30 sources per dataset) includes bit-vector fingerprints, physicochemical descriptor sets exceeding 1,600 dimensions, and embeddings from both graph-based and SMILES-based foundation models. Antibody encodings combine general-purpose protein language models (with several extraction strategies: full-sequence, CDR-only, per-region, and multiple layer depths), language models trained on antibodies, and sequence-derived features. The cancer genomics pool covers six data modalities (gene expression, copy-number variation, somatic mutations, methylation, protein expression, and pre-computed PanCanAtlas signatures), for a total of up to 57 sources. Drug-target and reaction domains use ligand and protein encodings jointly. The complete list with dimensionalities is in Supplementary Table 2.

### Downstream models

For a fair comparison, the downstream model pool must cover a range of inductive biases, from methods that degrade in high dimensions (kNN, SVM, GP) to tree ensembles that tolerate irrelevant features. Thirteen models are available: tree ensembles (random forest, XGBoost, LightGBM, CatBoost, NGBoost), kernel methods (SVM, Gaussian process), nearest-neighbor (kNN), linear models (Ridge regression for regression, logistic regression for classification, both with cross-validated regularization), and neural networks (single-layer MLP, three-layer DNN, MC-Dropout BNN for uncertainty quantification, and a multi-scale 1D CNN). Standard experiments use seven of these: random forest, XGBoost, LightGBM, kNN, SVM, Gaussian process, and MLP. We use the full set for model-matching and noise experiments, and skip models with cubic or higher scaling for large datasets: Gaussian processes above 6,000 samples, SVMs above 10,000. No per-dataset tuning is performed: all models use fixed hyperparameters throughout, so performance differences reflect the input, not the model configuration. Hyperparameters are listed in Supplementary Table 3.

### Datasets and splitting

We test KIRBy in five biological domains: molecular property prediction, antibody engineering, drug-target interaction, cancer genomics, and chemical reactions. Two criteria guided the selection: each domain should present a different kind of encoding diversity (fingerprints vs. language model embeddings vs. multi-omics), and the datasets within each domain should cover a range of sample sizes and difficulty levels to avoid testing only favorable cases. Molecular datasets cover ADMET endpoints (Caco-2 permeability, hERG inhibition, LogD, microsomal stability, MDR1 efflux) and quantum-mechanical properties from QM9. For antibodies, tasks include thermostability, pharmacokinetics, polyreactivity, aggregation, immunogenicity, binding affinity, and developability. Drug-target interaction datasets focus on kinase and GPCR binding affinity, while cancer genomics tasks include tumor subtype classification and survival-associated phenotypes. Reaction datasets cover yield prediction from high-throughput experimentation plates. Dataset sizes range from 94 to about 50,000 samples, and input dimensionalities from 3 features (log-likelihood ratios) to 2,048 (extended-connectivity fingerprints), with up to 57 sources per domain. Fig. 2 summarizes the cross-domain results; per-dataset details are in Supplementary Table 1.

No random splits are used. Random splitting inflates performance estimates for molecular data by placing structurally similar compounds on both sides of the train/test boundary^[Yang2019,Wallach2018], and position-level leakage in deep mutational scanning data allows models to memorise site-specific effects rather than learning generalisable sequence-fitness relationships^[Notin2023,Vieira2026]. Splits therefore match domain conventions and use 5-fold GroupKFold throughout:

- **Molecular:** Murcko scaffold groups^[Yang2019]. All molecules sharing a Bemis-Murcko scaffold are assigned to the same fold, preventing near-analogue leakage. This is the established standard in QSAR benchmarking (MoleculeNet^[Yang2019], Therapeutics Data Commons^[Huang2022], DeepChem). Scaffold splits are known to overestimate performance relative to fingerprint-based clustering^[Guo2024] or temporal splits^[Sheridan2013], but remain the accepted minimum for method comparison.
- **Antibody inter-family:** Grouped by unique VH sequence identity. Each antibody in these panels is a distinct molecule from a different lineage; mean pairwise sequence identity across datasets ranges from 32% to 43%, with fewer than 1% of pairs exceeding 90% identity. At this level of diversity, identity-based grouping is equivalent to ungrouped splitting, consistent with the approach taken by AbPROP^[Tennenhouse2023] for panels where pairwise similarity is below 93%.
- **Deep mutational scanning (DMS):** Grouped by mutation position modulo 5, following the ProteinGym benchmark^[Notin2023]. This prevents mutations at the same residue from appearing in both training and test folds, which would otherwise allow models to exploit position-specific effects^[Vieira2026]. Kermut^[Groth2024] and VespaG^[Marquet2024] evaluate under the same scheme.
- **Binding ΔΔG (SKEMPI):** Grouped by PDB complex identifier. All mutations from the same antibody-antigen complex stay in the same fold. This is the minimum standard for DDG prediction; stricter approaches based on CATH superfamily^[Kang2025] or 3D interface structural similarity^[Bushuiev2024] exist but require structural metadata not available for all entries.
- **Ginkgo developability:** Pre-computed hierarchical cluster splits provided by the benchmark authors^[Arsiwala2025], constructed via MMseqs2 pairwise identity clustering at 90% with isotype stratification.
- **Reaction:** Grouped by reaction template to prevent the same transformation scaffold from appearing in both train and test.
- **Cancer genomics:** Grouped by patient identifier with stratification by cancer type or subtype.

The hybrid construction step uses a nested internal train/validation split within each training fold to score representations for feature allocation. This internal split is group-aware, using GroupShuffleSplit with the same grouping variable as the outer cross-validation, to prevent the allocation evaluator from seeing related samples on both sides of its split.

### Statistical analysis

Comparing models tested under k-fold cross-validation requires care, because the training sets overlap between folds and a standard paired t-test overstates significance^[NadeauBengio2003]. Within-dataset comparisons therefore use the Nadeau-Bengio corrected paired t-test, which adjusts for this dependence. Cross-dataset comparisons (e.g., the domain-level tests in Fig. 2) use the Wilcoxon signed-rank test on paired per-dataset scores, since performance differences need not be normally distributed over heterogeneous tasks. When ranking more than two methods, we apply the Friedman test with Nemenyi post-hoc correction^[Demsar2006] to control the family-wise error rate. We report effect sizes as rank-biserial correlation for Wilcoxon tests and Kendall's W for Friedman tests.

### Data availability

All datasets used in this study are publicly available; sources, accession numbers, and download instructions are listed in Supplementary Table 1. Pre-computed representation matrices and cached search results for all datasets are deposited at [DOI/URL].

### Code availability

KIRBy is available as an open-source Python package at [URL] under the [LICENSE] license. The version used in this study is archived at [DOI]. Scripts to reproduce all figures and statistical tests are included in the repository.

---

## Extended Data / Supplementary

**Supplementary Table 1:** Per-dataset details (sample sizes, representation counts, task types, splitting strategies, primary metrics).

**Supplementary Table 2:** Per-dataset representation pools with dimensionalities and sources for all five domains.

**Supplementary Table 3:** Downstream model hyperparameters for all 13 models, including per-model budget caps and sample-size guards.

**Supplementary Table 4:** Internal evaluator hyperparameters for all 5 evaluators.

**Supplementary Methods:** Preprocessing details (clipping, imputation, binary detection, standardization). Feature importance method hyperparameters and selection procedure (8 candidate methods, cluster analysis, 3 retained). All 13 allocation strategy definitions with formulas (31 candidates, pruning criteria, 13 retained). Speed tier configurations (fast and full). Per-model budget caps. Augmentation procedure details (sequential testing, step size, patience). Adaptive pruning: allocation strategies are evaluated on fold 0 and pruned if they appear in the top-3 in fewer than a threshold fraction of (importance, evaluator, model, budget) groups. The primary hybrid searches (no augmentation and uncapped augmentation across general and top-tier representation sets) use a 15% threshold. Capped augmentation searches and standalone experiment-specific searches use a more aggressive 25% threshold to reduce computational cost; this higher threshold prunes more aggressively from the candidate allocation strategies without affecting the quality of the surviving strategies (F15-validated: pruned strategies had zero best-wins across all test datasets). Computational cost analysis (importance pre-computation, allocation score caching, feature selection deduplication via index hashing, disk caching with deterministic hashing, factory pre-computation for fine-tuned representations). Internal evaluator selection and tuning.
