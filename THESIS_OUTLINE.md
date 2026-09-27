DPhil THESIS OUTLINE: Adelaide Punt

Working title: Noise and Uncertainty in Molecular Property Prediction: Robust Models and Adaptive Representations

Department of Statistics, University of Oxford. Supervisor: Garrett Morris.
Collaborations: Lhasa Limited (Thierry Hanser, Stephane Werner); OPIG.

THIS IS THE ONE OUTLINE. It lives at KIRBy/THESIS_OUTLINE.md, next to thesis.tex.
- It is the 10 June outline (then at ~/Documents/Theses/THESIS_OUTLINE.md, the version thesis.tex
  was assembled from), updated on 2026-09-27. The May outline that used to be at this path, the
  KIRBy-first nine-chapter plan, was deleted; it is in git history (commit 147293c) if needed.
- The 10 June original was renamed ~/Documents/Theses/THESIS_OUTLINE_2026-06-10_original.md.

CHANGES ON 2026-09-27 (noise half only; Chapters 4 to 7 changed only in their numbers)
- The uncertainty chapter (old Chapter 4) is merged into Chapter 3, as its Section 3.8. One
  paper, one chapter. Old Chapters 5, 6, 7, 8 are now 4, 5, 6, 7, and every chapter, section,
  figure and table number inside them was shifted to match.
- Chapter 3 now follows the Results order of qsar_qm_models/paper.tex: variance decomposition,
  performance against robustness, noise conditions, assay datasets, probabilistic against
  deterministic counterparts, uncertainty. Its bullets report what paper.tex says (line numbers
  given), replacing the old-study numbers (NDS, six strategies, Kendall's W = 0.92, GP + SNS
  rho = 0.56), none of which are in the current paper.
- Background and Methods split: Chapter 2 keeps what each model family is and the seven noise
  conditions (the KIRBy noise experiments inject the same conditions). Everything specific to the
  NoiseInject study moves to a new Methods section, 3.2: model settings, noise levels, which
  conditions ran on which pairs, the QM9 precompute pipeline, the out-of-fold uncertainty
  protocol, AUC_norm, the exclusion gate, the ANOVA. The same move was made in thesis.tex;
  every changed spot there carries a "% [SPLIT 2026-09-27]" comment.
- Appendix B is now the thesis_appendix/ system (see APPENDICES).


FRONT MATTER
- Title page.
- Acknowledgements (Garrett Morris; the Lhasa supervisors Thierry Hanser and Stephane
  Werner; the EPSRC Centre for Doctoral Training, grant EP/S024093/1; OPIG; group; family).
- Abstract (about one page; the noise problem in molecular property prediction, then one
  clause per chapter in order, then the open-source NoiseInject note).
- Contents.
- List of Figures.
- List of Tables.
- Abbreviations (QSAR, ECFP4, SNS, PDV, MHG-GNN, GNN, MLP, RF, QRF, XGBoost, LightGBM,
  NGBoost, GP, SVM, BNN, VBLL, LNL, AUC_norm, ECE, ANOVA, ICC, RMSE, ROC-AUC, F1, PCA, mRMR,
  RFE, RBF, DFT, ADME, and the rest, spelled out once here so the chapters can use them bare).
  NDS and OHE belong to the earlier study and drop out unless a chapter still uses them.
- Declaration (the thesis includes the jointly authored NoiseInject paper and the Lhasa
  and OPIG collaborations; a short statement of the author's own contribution). A signed
  Statement of Authorship for the jointly authored NoiseInject paper is included, as the
  Oxford requirement for joint work, covering Chapter 3.
- Data and Code Availability (a one-page statement covering the three distinct software
  artifacts and the datasets, with per-dataset and per-repository detail deferred to the
  availability appendix). The three artifacts are: (1) NoiseInject, the open-source
  benchmarking framework (MIT, github.com/adpunt/noise_inject); (2) a standalone Rust and
  Python chemical-processing and precompute pipeline (from the qsar_qm work behind Chapter 3,
  to be simplified and released separately); and (3) the KIRBy research codebase
  (pure Python, the adaptive hybrid representation engine and experiments). KIRBy contains
  no Rust.
- List of Publications: the NoiseInject paper (Punt, Hanser, Werner and Morris, in
  preparation for the Journal of Cheminformatics); the two planned hybrid-representation
  papers (Chapters 5 and 6); the UK-QSAR conference presentation; the OPIG Blopig posts;
  and the open-source software (NoiseInject; the standalone Rust and Python
  chemical-processing pipeline; and the KIRBy codebase).


CHAPTER 1  INTRODUCTION  (about 8 to 10 pages)

1.1  Molecular property prediction and the two-component model
- Virtual screening and the role of fast property prediction in the drug discovery
  pipeline (Lill 2007; Neves et al. 2018).
- A predictor is a molecular representation paired with a model; the representation sets
  what chemical information is available, the model sets how it is used (Sabando et al.
  2021 on the static-versus-learned representation split).

1.2  Label noise in molecular data
- Experimental endpoints carry noise from assay variability, measurement error, data
  aggregation, and curation; this noise is rarely the homoscedastic Gaussian usually
  assumed.
- Noisy labels inflate apparent error, can corrupt the very test sets used to judge a
  model, and place a ceiling on achievable accuracy; once model error approaches the
  experimental measurement error, more training data yields diminishing returns (Jörner et
  al. 2021).

1.3  The questions this thesis asks
- Under label noise, which matters more for robustness, the representation or the model
  architecture, and can probabilistic models recover usable uncertainty estimates that
  track that noise?
- Building on those answers: can the representation itself be adapted to make prediction
  more robust, and to make molecular similarity and noise estimation more reliable?
- thesis.tex 1.3 still quotes the earlier paper's three questions. paper.tex now asks four
  (paper.tex:197): how representation and architecture divide the variance in performance and
  in retained performance; what the choice of model provides, deterministic against
  probabilistic; whether the type of noise matters or only its amount; and what label noise does
  to a model's uncertainty estimates.

1.4  From noise robustness to adaptive representations
- The first half establishes that, under noise, the model dominates robustness while the
  representation and the model contribute comparably to clean performance, and describes what
  probabilistic models' uncertainty does and does not register about the noise.
- An exploratory study of learning-with-noisy-labels methods then showed that the
  approaches relying on molecular similarity fail under noise, which motivated the second
  half: an adaptive hybrid representation that builds a compact, informative representation
  from several others, and in doing so repairs the notion of similarity those methods
  depend on.

1.5  Outline of the thesis and contributions
A per-chapter statement of what each chapter does, what it finds (for the completed
chapters) or will test (for the planned ones), and where the work is published or intended.
- Chapter 2 sets out the representations, models, noise model, and evaluation that the
  rest of the thesis uses, and introduces NoiseInject, the open-source benchmarking
  framework.
- Chapter 3 is the NoiseInject chapter (adapted from Punt, Hanser, Werner and Morris, in
  preparation for the Journal of Cheminformatics). It decomposes the variance in performance
  and in noise robustness across representations and model architectures; compares seven noise
  conditions at a matched amount of noise; repeats the study on three assay datasets; compares
  each model with its probabilistic counterpart; and asks what label noise does to predicted
  uncertainty. paper.tex's headline: model architecture is the main factor in noise robustness,
  while architecture and its pairing with the representation contribute comparably to
  performance; the pattern of the noise matters more than its shape; uncertainty rises with
  noise but does not pick out which labels were corrupted.
- Chapter 4 introduces the adaptive hybrid representation, with the exploratory
  learning-with-noisy-labels work as its motivation.
- Chapter 5 is the first of the two planned method papers (intended for submission;
  venue to be confirmed). It will test when and why the hybrid representation helps, that
  it is explainable, and that it extends to new embeddings and to federated data. Results
  are pending a full re-run (see Chapter 4).
- Chapter 6 is the second planned method paper (intended for submission; venue to be
  confirmed). It will return to molecular similarity and label noise, testing whether the
  hybrid representation gives a more property-aware similarity and improves noise
  estimation. Results are pending a full re-run.
- Chapter 7 draws the argument together and then develops the main open directions.

1.6  Collaborations and funding
- The EPSRC Centre for Doctoral Training (grant EP/S024093/1); Lhasa Limited (label-noise
  framing and the federated work); OPIG (antibody and protein datasets and the Blopig
  collaboration).


CHAPTER 2  BACKGROUND  (about 30 to 35 pages)
The shared-knowledge chapter. It introduces, once, the representations, models, noise
model, and evaluation that every later chapter refers back to, and ends by introducing
NoiseInject. A few central ideas are stated as numbered definitions so later chapters can
cite them. Source: confirmation report Chapters 1 and 2, and the NoiseInject paper.
The split with Chapter 3 (agreed 2026-09-27): this chapter says what a model family, a noise
condition or a metric IS. Which settings, levels, pairings and tests the NoiseInject study used
belongs in Chapter 3's Methods (3.2). Background is written in general voice, not "we fitted".

2.1  Molecular representations
- Definition 2.1 (molecular representation / featurization): the map from a molecule to a
  fixed-length numeric vector a model can consume.
- Each representation is given a short account of what it captures and where it is weak,
  ending in a comparison table (Table 2.1).
- String representations: SMILES, and one-hot-encoded (OHE) SMILES (canonical and
  randomized; Bertoni et al. 2020).
- Fingerprints: MACCS keys (Durant et al. 2002); ECFP4 and its bit collisions (Rogers and
  Hahn 2010); Avalon; and Sort & Slice (SNS), the collision-free alternative for pooling
  ECFP substructures (Dablander, Hanser, Lambiotte and Morris 2024).
- Physicochemical descriptor vectors (PDVs) and mordred descriptors (Moriwaki et al. 2018).
- Learned embeddings: mol2vec (Jaeger et al. 2018), ChemBERTa (Chithrananda et al. 2020)
  and MolBERT, MolFormer (Ross et al. 2022).
- Molecular graphs and message-passing GNNs; MHG-GNN as a fixed-length pretrained graph
  embedding (Kishimoto et al. 2023). For broader framing, David et al. 2020 review
  molecular representations and Cherkasov et al. 2014 the QSAR modelling perspective.
- Protein and antibody language-model embeddings used in the second half (ESM-2, ESM-C,
  ESM3, SaProt, AbLang2, AntiBERTy, IgT5), introduced here so Chapter 5 can refer back.
- Figure 2.1: the molecule O=C1C=CN(C1)C=O shown as an ECFP4 fingerprint, a tokenized
  one-hot SMILES, and a molecular graph (reuse the confirmation report's Figure 1).

2.2  Models for molecular property prediction
- What each family is, how it uses the features, and why it might resist or absorb label
  noise. No study settings here (tree counts, network widths, the Gaussian-process training
  cap, kernel choices per representation): those are in 3.2.
- Tree ensembles: random forests (Svetnik et al. 2003), XGBoost (Mustapha and Saeed 2016;
  Tian et al. 2022), LightGBM (Ke et al. 2017), and the probabilistic quantile regression
  forest (QRF, Meinshausen 2006) and NGBoost (Duan et al. 2020).
- Support vector machines (Vapnik 1995).
- Gaussian processes (Rasmussen and Williams 2006), the Tanimoto kernel for binary inputs
  (Ralaivola et al. 2005; Moss et al. 2020) and the RBF kernel, both stated as equations; the
  GAUCHE framework for chemistry (Griffiths et al. 2023); and the cubic scaling that limits
  them (Quiñonero-Candela and Rasmussen 2005 for sparse approximations). Obrezanova et al.
  2007 are an early QSAR/ADME precedent for GPs.
- Figure 2.2: the deterministic-versus-Bayesian neural-network node schematic (a fixed weight
  giving a point prediction versus a distribution-valued node giving a prediction histogram),
  reused from the confirmation report's Figure 3.
- Neural networks (feed-forward, MLP) and the Bayesian transformations in general: fully
  Bayesian layers, a mean-and-variance output head (Kendall and Gal 2017), and Variational
  Bayesian Last Layer (VBLL, Harrison et al. 2024). MC dropout (Gal and Ghahramani 2016) and
  deep ensembles (Lakshminarayanan et al. 2017) are the non-Bayesian uncertainty baselines.
  In thesis.tex this general account is now missing: the only description of the
  transformations was one paragraph mixed with the NN-alpha/beta settings, and it moved to 3.2.
- Aleatoric versus epistemic uncertainty, defined here (the definitions stay in 2.2 in
  thesis.tex; which models the study decomposed moved to 3.2).
- Two distinct uncertainty toolkits run in this thesis, and they should not be conflated.
  The NoiseInject study (Chapter 3) uses QRF, NGBoost, the variational and Bayesian networks,
  and GAUCHE Gaussian processes. KIRBy's own conformal uncertainty (Chapter 6) uses MAPIE split
  and cross-conformal prediction and a Mondrian applicability-domain conformal regressor binned
  by distance to the nearest training compound, and its GP uses an sklearn RBF kernel rather
  than a Tanimoto kernel. KIRBy does not reuse the NoiseInject probabilistic models.

2.3  Molecular similarity and activity cliffs
- Definition 2.2 (the similarity assumption): similar molecules tend to have similar
  properties (Johnson and Maggiora 1990; van Tilborg et al. 2022). This holds only relative
  to a similarity measure, which is fixed by the representation, and the whole second half
  turns on that dependence; Maggiora et al. 2014 stress that similarity is measure-dependent
  and easily over-interpreted.
- Definition 2.3 (activity cliff): a pair of structurally similar molecules with a large
  property difference (Stumpfe and Bajorath 2012; van Tilborg et al. 2022), which
  distance-based reasoning readily mistakes for noise.
- Why high-dimensional, largely uninformative representations make pairwise distances
  unreliable (Aggarwal et al. 2001).

2.4  Noise in molecular data
- Sources of label noise: assay variability (plate effects, carryover, reagent
  degradation; Kalliokoski et al. 2013; Kramer et al. 2012), instrument detection limits,
  biological variability, aggregation across databases (ChEMBL, Gaulton et al. 2017 and
  Zdrázil et al. 2023; BindingDB, PubChem; Landrum and Riniker 2024 quantify the noise from
  combining values across sources), federated data (Lhasa's Effiris), and curation and
  annotation errors (Fourches et al. 2010). All of this raises the likelihood of
  encountering noise in the data (Fang et al. 2022), and the training representations
  themselves may carry errors (Walters 2023). Frénay and Verleysen 2014 survey the sources
  and consequences of label noise.
- Why real noise is neither independent, identically distributed, nor symmetric (Heid et al.
  2023): its shape (Kruger et al. 2012 fit a Laplace to repeat bioactivity differences), shared
  offsets between laboratories (Bentz et al. 2013: 62% of the variance in Caco-2 efflux is
  between laboratories), and censoring at assay limits (Svensson et al. 2025).
- A caveat carried from the confirmation report: not all label noise is harmful, since mild
  noise and scaffold splitting can act as a form of regularization that limits overfitting
  (Ullah and Welsh 2024; Camuto et al. 2021).
- The artificial noise model (thesis.tex label sec:noise_injection): the clean label y_i is
  replaced by y_i + epsilon_i. Seven regression conditions: three shapes (Gaussian, Student-t
  with nu = 5, Laplace), two that concentrate the error (outlier: a random 10% of molecules;
  grouped-wider: whole scaffold families covering 20% of the split, each at three times the
  scale), grouped-shifted (one shared offset per scaffold family, carrying a share rho = 0.62
  of the variance), and censoring (labels past a limit set to the limit). All but censoring
  deliver the same expected amount of noise at a given level (the dose-matching equation), so
  a difference between conditions is a difference in pattern, not amount. Where each parameter
  comes from: lambda = 3 (Avdeef 2019), 10% (Hampel 2001), rho = 0.62 (Bentz 2013), the shapes
  (Kruger 2012), the contamination model (Huber 1964). Figure 2.3 (F1_noise_conditions).
  The noise levels used and which conditions ran on which model and representation pairs are
  study settings and are in 3.2. The classification strategies that NoiseInject also provides
  are named in 2.6 only.

2.5  Learning with noisy labels
- The Song et al. 2022 taxonomy (architecture, regularization, loss, sample reweighting,
  sample selection); the No Free Lunch result (Wolpert and Macready 1997) and its
  chemistry-domain analogue under simulated noise (Cortes-Ciriano, Bender and Malliavin
  2015); and why methods that lean on molecular similarity are exposed once similarity is
  unreliable (Wilson 1972 for the neighbour-editing root; Dablander et al. 2023 for the
  activity-cliff confound), returned to and tested in Chapter 6.

2.6  NoiseInject: a benchmarking framework
- NoiseInject 1.0.0, the open-source, model-agnostic framework that injects controlled label
  noise and reports robustness and uncertainty metrics for regression and classification
  (MIT, github.com/adpunt/noise_inject, archived on Zenodo). How the regression injector
  combines a shape with a targeting rule; the dose check; the classification strategies; the
  robustness metrics (retention curve, AUC_norm, the Weibull fit); the uncertainty metrics
  (rank correlation with error, coverage at 1 and 2 standard deviations, interval width, ECE,
  and the rank correlation with injected noise, within a noise level); the optional wrappers.
- It stays in Background, not Chapter 3, because the KIRBy noise experiments use it as well.
  The QM9 precompute pipeline, used by Chapter 3 alone, moved to 3.2 (thesis.tex: the
  subsection "The QM9 precompute pipeline" now sits in Chapter 3 Methods; its figure is out of
  date, noted in a TODO there).

2.7  Evaluation methodology
- Scaffold and group-aware splitting (no random splitting; Bemis-Murcko scaffolds, each
  acyclic molecule its own group). Predictive metrics by task (RMSE, R2; accuracy, F1,
  ROC-AUC). The general definitions of the uncertainty metrics, if they are not already in
  2.6 (thesis.tex has them twice; a TODO marks it).
- The study-specific evaluation (AUC_norm over the seven levels, the clean R2 < 0.3 gate, the
  Wilcoxon test on QM9 only, the Type I ANOVA with leave-one-replicate-out bands, the
  redundancy screen) is in 3.2.
- With these in place, Chapter 3 puts them to work, asking whether the representation or
  the model architecture governs how badly a predictor degrades under label noise.


CHAPTER 3  NOISE ROBUSTNESS ACROSS MOLECULAR REPRESENTATIONS AND MODEL ARCHITECTURES
(about 30 to 38 pages; title to revisit now that uncertainty is part of it)
Completed work; the NoiseInject chapter. An opening note records that it is adapted from Punt,
Hanser, Werner and Morris, in preparation for the Journal of Cheminformatics, and that a signed
Statement of Authorship for this jointly authored paper is included in the front matter.
Source: qsar_qm_models/paper.tex (read 2026-09-27). The findings below are what paper.tex
states, with its line numbers; where paper.tex has a placeholder, the bullet says so.
In thesis.tex: the Introduction and the new Methods section (3.2) exist. The results sections
3.3 to 3.8 below are the target; thesis.tex still carries the earlier study's results under
their old headings, followed by the merged uncertainty section.

3.1  Introduction
- A predictor is a representation paired with a model (Chapter 2). The gap: comparisons of how
  representations respond to label noise are limited (Deng et al. 2023 on clean data), and
  Heid et al. 2023 found no difference between error distributions with a matched mean and
  standard deviation; this study adds noise concentrated in scaffold groups or on a sparse
  subset (paper.tex:195).
- The four questions (paper.tex:197), as listed in 1.3.
- Review of existing approaches: Cortes-Ciriano, Bender and Malliavin 2015, Kolmar and Grulke
  2021, and Jörner et al. 2021 (thesis.tex TODO: pull them in here).

3.2  Methods
- Datasets: QM9 HOMO-LUMO gap, ten replicates of 10,000 molecules split 8,000 / 1,000 / 1,000;
  LogD (5,039), Caco-2 efflux (2,161) and hERG Ki (1,415, curated from ChEMBL), each in five
  scaffold-grouped folds.
- Representations: ECFP4, Avalon, Sort & Slice, PDV, ChemBERTa, MHG-GNN.
- Models: thirteen base models (RF, QRF, XGBoost, LightGBM, NGBoost, SVM, GP, NN-alpha, BNN-alpha,
  VBLL-alpha, NN-beta, BNN-beta, VBLL-beta) and six variant models; their settings;
  hyperparameters in Appendix B (hyperparameters_default, hyperparameters_tuned).
- Noise conditions and levels: the seven conditions from 2.4; levels 0, 0.2, 0.3, 0.5, 0.75,
  1.0, 1.5 of the clean training label spread; censoring by fraction clipped (0 to 0.5);
  Gaussian, grouped-wider and grouped-shifted on every pair, the other four on selected pairs.
- The QM9 precompute pipeline (the Rust and Python pipeline; moved from 2.6).
- Uncertainty: the aleatoric/epistemic split for each model (paper.tex:285-310); scored out of
  fold on the training molecules, because no held-out label carries noise.
- Evaluation: AUC_norm and its median over replicates or folds; replicates with clean
  R2 < 0.3 excluded (Appendix B, excluded_configurations: its content still uses the old 0.6
  gate); rank correlations; Wilcoxon on QM9 only; two-way Type I ANOVA per condition with the
  redundancy screen (Appendix B, model_redundancy to model_icc).
- In thesis.tex the [PAPER] paragraphs of 3.2 are current; the [MOVED] paragraphs that
  follow them in "Uncertainty" and "Evaluation" are from the earlier study and are marked for
  deletion.

3.3  Variance decomposition
- Under Gaussian noise, model architecture explains 49.5% of the variance in AUC_norm and the
  representation 9.2% (interaction 20.4%, residual 20.8%). For R2 at a noise level of 1.0 the
  two are comparable: 29.1% and 26.6% (paper.tex:398). Grouped-wider looks the same.
- Under grouped-shifted the residual takes 56.3% of the variance in AUC_norm and 76.6% in R2
  (paper.tex:429).
- Placeholder in paper.tex: the clean-label decomposition on all four datasets (F2b, T3b not
  yet generated).
- Floats: T3 (variance decomposition), F2.

3.4  Predictive performance and noise robustness
- Clean R2 and AUC_norm rank the thirteen base models at a Spearman correlation of -0.18
  (p = 0.57), Gaussian, ECFP4 (paper.tex:461).
- The plain and Bayesian networks are among the most accurate on clean labels (NN-beta
  R2 = 0.851) and are the four least robust (paper.tex:464).
- NGBoost has the highest AUC_norm on all six representations under Gaussian noise and the
  lowest clean accuracy on four of them; on PDV it starts lowest and ends highest
  (paper.tex:467).
- The networks' AUC_norm spans 0.039 to 0.063 across representations, against at most 0.030 for
  every other model (paper.tex:469).
- Floats: T4 (robustness by model and condition), F4a, R16, F3.

3.5  Noise conditions
- The shape of the noise made little difference; no change exceeds the replicate spread on QM9
  (paper.tex:540), in line with Heid et al. 2023.
- Grouped-shifted lowered the AUC_norm of every base model on every representation; censoring
  gave the largest drop, median AUC_norm 0.813 against 0.951 under Gaussian noise
  (paper.tex:542).
- The best representation for a model changes between Gaussian and grouped-shifted noise for
  five models (paper.tex:544).
- Model ranks move with the noise level: RF from seventh on clean labels to first by level 1.0,
  SVM from first to fourth (paper.tex:549).
- Floats: R19, T12 (not yet generated), F4b, R15.

3.6  Robustness on the three assay datasets
- RF ranks first to eighth for AUC_norm on every dataset and representation; the GP is the
  most accurate on clean labels on all three assay datasets; LightGBM ranks tenth to
  thirteenth on Caco-2 and hERG Ki (paper.tex:623).
- Grouped-shifted is the hardest condition on every dataset (paper.tex:618).
- paper.tex marks this section for rework: the paragraph on non-Gaussian shapes needs a table,
  and the claims about representations other than ECFP4 have no figure behind them yet
  (paper.tex:595, 607-609, 620).
- Floats: F8 (one panel per dataset), R9, F2b/T3b.

3.7  Probabilistic and deterministic counterparts
- Bayesian transformations raised the networks' AUC_norm at the cost of clean R2, on every
  dataset: BNN-beta kept more AUC_norm than NN-beta in most combinations on every dataset,
  BNN-alpha in most on Caco-2 and logD and about half on hERG Ki (paper.tex:713).
- Placeholders in paper.tex: RF at 300 trees against QRF; the variance-head networks against
  their Bayesian bases (the two run at different settings).
- Floats: R17b, R17c, T11 (QM9 in the text; the three assay tables go to the appendix as
  counterparts_assay, not built yet).

3.8  Uncertainty under label noise   (was Chapter 4)
- Mean predicted uncertainty rises with Gaussian noise for every model and representation on
  QM9, and for most on the assay datasets (paper.tex:763).
- The split into aleatoric and epistemic parts: for the variance-head BNNs the epistemic part
  grows by at most 0.43 of the aleatoric growth; for QRF it grows 0.63 to 2.05 times as much,
  depending on the representation (paper.tex:767-769). Grouped-shifted raises the epistemic
  share of the heteroscedastic GP and of QRF everywhere (paper.tex:771).
- The aleatoric part rises 0.36 to 0.99 units per unit of added noise, so it can compare noise
  between datasets but does not estimate a dataset's own error (paper.tex:773).
- Per molecule, uncertainty against injected noise stays between -0.16 and 0.16 under six of
  the seven conditions. Under censoring the correlation is large and negative: models are most
  certain on the labels that were clipped (paper.tex:781-783).
- Floats: T9 (not yet built), T6, F6, F6b. paper.tex:761 cites tab:top_unc_noise, a label from
  the earlier study that no longer exists.
- In thesis.tex this is Section 3.8 "Uncertainty under label noise", holding the old chapter's
  text and subsections unchanged (earlier study; to be rewritten from paper.tex:759-783).

3.9  Conclusion
- paper.tex:853-871: architecture governs robustness; systematic noise (grouped-shifted,
  censoring) costs most; RF the most consistent across the four datasets; Bayesian networks
  trade clean accuracy for robustness; no model's per-molecule uncertainty identifies the
  corrupted labels.
- The bridge to Chapter 4 is open. The old one ("if the representation governs whether
  uncertainty and similarity are meaningful ...") rested on the retracted uncertainty result.
  Material in paper.tex that could carry it: the representation decides whether QRF's
  uncertainty split works (3.8), and the best representation changes with the noise condition
  (3.5). Or the bridge rests on the similarity-based LNL failure alone (4.1). Your call.
- thesis.tex: the old uncertainty chapter's conclusions were appended here, unchanged.


CHAPTER 4  A METHOD FOR BUILDING ADAPTIVE HYBRID REPRESENTATIONS  (about 26 to 32 pages)
The method chapter for the second half, introduced once and referred back to thereafter.
Source: the method's implementation and the design experiments behind its defaults.

4.1  Motivation: the failure of similarity-based noise handling
- An exploratory study adapted twelve learning-with-noisy-labels methods to regression
  (tested over 221 configurations across methods, loss functions, distance metrics, and
  representations) and found that those relying on molecular distance or similarity were the
  worst performers; every one of the twenty worst-performing configurations for noise
  retention used a distance metric, across standard metrics (Euclidean, Manhattan, cosine,
  Mahalanobis) and the chemistry-aware Tanimoto coefficient, and molecular representation
  choice explained even less variance under these methods.
- The likely cause, in the terms of Definition 2.2: a single representation carries many
  uninformative dimensions, so pairwise distances are dominated by noise and the
  similarity assumption breaks down. This is the motivation for the method; it is
  exploratory, not a reported result, and the noisy-label question is measured properly in
  Chapter 6.

4.2  The adaptive hybrid representation
- We define the adaptive hybrid representation (thereafter the hybrid representation, or
  the method where unambiguous): rather than choosing one representation, it assembles a
  single feature set by selecting the most informative features from several
  representations at once, chosen for the dataset and model in hand, before the final model
  is trained. It prepares better inputs for standard downstream models rather than
  introducing a new model.
- The idea of combining representations is not itself new; the contribution is that the
  method poses this combination as a search-space optimisation problem and solves it
  automatically. The space is the product of component representations, feature-importance
  methods, allocation methods, internal evaluators, downstream models, and feature budgets,
  and per-fold caching is what makes the search over it affordable. The formulation is not
  specific to chemistry: it applies to any supervised learning task for which several
  representations of the same inputs are available.
- Review of existing approaches: multi-view learning shows that naive concatenation of
  views overfits on small samples (Xu, Tao and Xu 2013); prior QSAR fusion either combines
  predictions (Valsecchi et al. 2020), concatenates whole fingerprints (Xie et al. 2020),
  or hybridizes feature selection and feature learning only when subsets are complementary
  (Ponzoni et al. 2017). The hybrid representation generalizes these by selecting features
  across many representations under a budget.

4.3  The method as a search
- The search ranges over six axes: the component representations; the feature-importance
  method used to rank features within each representation; the allocation method that
  divides a feature budget across representations; the internal evaluator used to score
  candidate feature sets during the search; the downstream model; and the feature budget
  itself. The contribution is not the notion of a combined representation but the automated
  search over this large combinatorial space, together with the per-fold caching that makes
  it affordable; the same machinery serves any supervised task with multiple representations.
  Figure 4.1 lays out the pipeline.
- The searchable downstream models are random forest, XGBoost, ridge, k-nearest
  neighbours, LightGBM, MLP, SVM, and the Gaussian process, with CatBoost, DNN, CNN, BNN,
  and NGBoost available when requested by name. Feature budgets span roughly 10 to 2,000
  features, filtered per model by a maximum-budget cap (the GP is capped at 1,000, SVM and
  the neural models lower). The allocation methods are greedy forward selection,
  performance-weighted (with softmax, power-law, top-k, rank, score-gap, and
  diversity-penalised entropy weighting schemes), dominant-greedy, fixed equal split,
  two-stage, and stability-selection. The internal evaluators are match-the-downstream-model
  or an explicit ridge, RF, SVM, LightGBM, or MLP.
- The feature-importance methods are SHAP-family and gradient or permutation importances:
  treeshap, xgboost_gain, lightgbm_gain, permutation, kernelshap, deeplift,
  integrated_gradients, and gradientshap. Mutual information was implemented but removed for
  poor recall; mRMR and recursive feature elimination are not implemented. The classical
  feature-selection literature the approach is positioned against includes mRMR's
  relevance-versus-redundancy criterion (Peng, Long and Ding 2005) and SVM-RFE (Guyon et al.
  2002); the importance methods used here build on model-based importance (Breiman 2001),
  with Guyon and Elisseeff 2003 for the goals of selection (performance, cost,
  interpretability).
- The search also supports capabilities beyond the six core axes: augmentation strategies
  (uncapped, capped, greedy-ablation, all, none) that append small auxiliary feature sets
  beyond the budget; fine-tuned per-fold representation factories (rep_factories) that fold
  learned or LoRA embeddings into the search with their own per-fold embedding cache; and a
  two-phase refinement of step-size-sensitive greedy allocations.

4.4  Blueprints and explainability
- A completed search is stored as a compact, reusable recipe, the blueprint, holding exactly:
  the representations, the allocation method, the importance method, the feature budget, the
  internal evaluator, the step size, the number of folds, the seed, the metric, the task
  type, the augmentation strategy, and the per-representation scores. Because the blueprint
  stores the recipe rather than the data, it can be saved and reapplied to new molecules.
- The blueprint does not store the selected feature indices; those are recovered when the
  blueprint is refitted on data, which re-runs selection deterministically from the recipe
  and seed (a saved blueprint is the recipe, the fitted selection is a separate object).
- Named-feature attribution is real: a feature-name registry resolves each selected column
  back to its representation and a named feature, including ECFP and MACCS bits mapped to
  SMARTS substructures, so a prediction traces back to specific, named features. Learned
  embedding dimensions, where the registry has no chemical name, fall back to the bare
  representation name plus the dimension index, so attribution is only as rich as the
  registry. Figure 4.2.

4.5  Default settings
- The design experiments that fix the defaults: which allocation methods and
  feature-importance methods to keep (Altmann et al. 2010 document the bias of naive
  permutation importance, motivating careful choice), whether the internal evaluator should
  match the downstream model, and how aggressively the search can be pruned. The current
  defaults are a top-3 performance-weighted allocation, lightgbm_gain importance, a step
  size of 5, and 5-fold cross-validation; pruning is handled by an adaptive multi-stage
  schedule that drops the worst allocation-importance pairs per model after the first fold,
  with a floor that keeps a best configuration for every model, rather than by separate
  speed tiers. Results to be added as these are finalized; Table 4.1.

4.6  Implementation
- Caching and reuse that make the search tractable, and reproducibility.
- Results note: the second-half results (Chapters 4 to 6) are pending a full re-run.
  Earlier cached results were invalidated by a caching defect, since fixed, that combined
  non-deterministic data splits, malformed cache keys, and unsafe concurrent cache writes;
  the experiments are being re-run from scratch and the results will be added once the
  re-run completes.
- With the method defined, Chapter 5 asks the first empirical questions: when does the
  hybrid representation beat the best single representation, and why.

4.7  Conclusion
- The chapter has defined the adaptive hybrid representation, the search that builds it,
  and the blueprint that stores it. Chapter 5 turns to when and why it helps.


CHAPTER 5  WHY HYBRID REPRESENTATIONS HELP: COMPLEXITY, EXPLAINABILITY, AND EXTENSION
(about 28 to 32 pages)
The first of the two method papers, building directly on Chapter 4. (Results pending a full
re-run, as set out in Chapter 4.)

5.1  Introduction
- When does a hybrid representation beat the best single representation, what is the
  mechanism, can its feature choices be explained and shown stable, and does it keep
  working as new embeddings arrive and when data is held by separate parties.
- Review of existing approaches: Niazi and Mariam 2023 (high-dimensional redundancy and PCA
  as the standard response), Xie et al. 2020 (conjoint fingerprints), and Valsecchi et al.
  2020 (prediction-level consensus).

5.2  When it helps
- The core question: does the hybrid beat the best single representation, and does the
  optimal feature budget vary by downstream model? A model-by-budget grid over the cached
  search answers both, with a two-way ANOVA over model, budget, and their interaction.
  Figure 5.1 (model-by-budget heatmap) and Table 5.1 (best-per-model delta against the best
  single representation).
- The headline is the cross-domain summary: across small-molecule, antibody, and
  gene-expression tasks, does the hybrid win consistently? Each task contributes a paired
  per-fold delta against the best single representation, combined with a Wilcoxon test and
  per-domain win rates. Figure 5.2 (forest of per-task deltas with the Wilcoxon result) and
  Table 5.2.

5.3  Why it helps
- The mechanism comparison isolates "more features chosen well" from "any compression to the
  same width" by running one model on three matched-dimensionality inputs: the best single
  representation, a PCA-of-concatenation, and the hybrid. The PCA control is the ablation
  reviewers will expect. Figure 5.3.
- The more interesting angle is information-theoretic: the mutual information with the target
  per selected dimension as the budget grows, against random selection and a
  matched-dimensionality null. If the hybrid packs more target-relevant information per
  dimension than the alternatives, that is the mechanism. mRMR (Peng, Long and Ding 2005) and
  RFE (Guyon et al. 2002) are named as the classical alternatives the approach is positioned
  against, not as implemented baselines. Figure 5.4.

5.4  Explainability and stability
- SHAP on the best hybrid, mapped through the Chapter 4 feature-name registry, aggregates
  importance by source representation and traces predictions to named features, with the
  caveat that learned-embedding dimensions resolve only to a name plus a dimension index.
  Figure 5.5.
- A feature-set stability analysis (Nogueira's index with a confidence interval, plus
  fold-to-fold Jaccard) checks that the selection is consistent across folds. It is an
  experiment-level analysis, not a core-engine output. Table 5.3.

5.5  Breadth across domains
- To make the "applied well beyond molecular similarity" claim concrete, the method is
  carried across small molecules, antibodies (Trastuzumab-HER2 binding, Ginkgo
  developability), protein affinity (Graphinity antibody-antigen ddG), and gene-expression
  (the TCGA foundation-model comparison), each under its own splitting protocol. Whether the
  advantage holds is summarised in one aggregate comparison. Figure 5.6.

5.6  Forward compatibility: absorbing new embeddings
- The mechanism is light: a new embedding is added by registering an embed function and one
  dispatch line, after which the search gives it budget only if its features are selected.
  The interesting test is whether a new embedding earns budget above two
  matched-dimensionality null controls, a same-dimensionality random Gaussian matrix and a
  row-shuffled copy, which separate genuine signal from dimensionality and content;
  significance via Wilcoxon per dataset and Friedman across datasets. A second strand tests
  whether a saved blueprint transfers across folds and, where dimensions line up, across
  datasets. The symmetric direction, whether the hybrid improves a weak new embedding, is
  future work (Section 7.2). Figure 5.7 and Table 5.4.

5.7  Federated data (with Lhasa)
- A smaller, applied section: is sharing only which features matter competitive with
  ensembling each party's separate model on a shared hERG benchmark, including under unequal
  per-party noise? The shareable anchor splits hERG into clustered silos with label noise
  rising across silos, and contrasts privacy-preserving feature-importance sharing against
  similarity-weighted, performance-weighted, and distillation ensembles. A real-source variant
  repeats the contrast on public ChEMBL plus two proprietary Lhasa silos, reported under a
  data-availability caveat (the Lhasa data is not redistributable and the comparison depends
  on Lhasa's FLuID code). A demonstration, not a new method. Figure 5.8 and Tables 5.5 and 5.6.
- Review of existing approaches: McMahan et al. 2017 (FedAvg, sharing model updates not raw
  data), the MELLODDY consortium (Heyndrickx et al. 2023, cross-pharma federated QSAR), and
  FL-QSAR (Chen et al. 2020).

5.8  Conclusion
- Having tested when the hybrid representation helps, why, whether it can be explained, and
  how it extends, Chapter 6 returns to the problem that began the second half.

CHAPTER 6  MOLECULAR SIMILARITY AND LABEL NOISE  (about 24 to 28 pages)
The second method paper, closing the loop opened in Chapters 3 and 4. (Results pending a
full re-run, as set out in Chapter 4.)

6.1  Introduction
- Recalling Definition 2.2 and the motivation of Chapter 4, this chapter measures how much
  the notion of similarity depends on the representation, and whether the hybrid
  representation gives a more property-aware similarity, then uses that to revisit the
  noisy-label question Chapter 3 left open.
- Review of existing approaches: van Tilborg et al. 2022 and Maggiora et al. 2014
  (similarity depends on the representation), Aggarwal et al. 2001 (high-dimensional
  distance concentration), and, for noise cleaning, Northcutt et al. 2021 (Confident
  Learning, a model-confidence contrast to neighbour-based estimators), Brodley and Friedl
  1999 (instance-selection filtering), and Wilson 1972 (edited nearest neighbours).

6.2  Similarity and the representation
- The premise is that similarity is representation-dependent: for each representation, the
  similarity-activity correlation (Spearman of pairwise distance against absolute property
  difference, with a permutation test) measures how well "similar" tracks "similar property".
  Figure 6.1 (spread of the correlation across representations).
- The central positive claim is that the hybrid's importance-weighted distance gives a more
  property-aware similarity than the best single representation or a plain concatenation,
  reported as a delta in that correlation and aggregated across datasets with a Wilcoxon test.
  Figure 6.2 and Table 6.1.

6.3  Noise estimation and mitigation inside the method
- The flagship validates the noise estimators against real assay noise rather than injected
  noise: on BindingDB, the per-(drug,target) replicate standard deviation of measured affinity
  is the ground-truth noise signal, and the question is whether each estimator ranks samples in
  agreement with that real disagreement, in the hybrid space versus the alternatives. This is
  the chapter's external-validity anchor. Figure 6.3.
- Supporting experiments: a Gaussian-process leave-one-out test of whether the hybrid space
  separates genuine noise from activity cliffs (Figure 6.4); a remediation experiment measuring
  how much clean-data performance is recovered by downweighting, label smoothing, and neighbour
  smoothing (Figure 6.5); and the corrected learning-with-noisy-labels survey, the matured,
  oracle-leak-free version of the exploratory Chapter 4 study, re-searching the feature mix per
  noise level (Table 6.2).
- The conceptual hinge between the chapter's two halves: neighbour-based noise detectors built
  on the similarity assumption mistake activity cliffs for noise, and the question is whether
  the hybrid distance reduces that contamination relative to single-representation baselines.
  Table 6.3.
- Calibration of the hybrid across feature budgets would tie the uncertainty thread of
  Chapter 3 to the method; the conformal machinery exists (MAPIE split and cross-conformal, and
  a Mondrian applicability-domain regressor) but the budget-sweep figure is planned, not yet
  wired. Figure 6.6.

6.4  Conclusion
- Honest limits: where comparisons are best-case rather than fully blind, stated plainly,
  with the fair comparison foregrounded. This closes the loop opened in Chapter 3; the
  concluding chapter develops the open directions and draws the argument together.


CHAPTER 7  DISCUSSION AND FUTURE DIRECTIONS  (about 11 to 14 pages)
The single closing chapter: it draws the argument together, recaps the contributions, then
develops the main open directions. The back-matter Summary of Research Contributions
follows separately.

7.1  Conclusions
- The single line of the thesis: under label noise the model governs robustness and the
  representation governs clean performance and the meaningfulness of uncertainty and
  similarity; adapting the representation then makes prediction more robust, similarity
  more property-aware, and noise estimation more reliable, while staying interpretable and
  extendable.
- Contributions, across three distinct software artifacts and their findings: the
  NoiseInject benchmarking framework (released, MIT) and the noise-robustness and
  uncertainty findings of Chapter 3; the standalone Rust and Python
  chemical-processing and precompute pipeline for high-throughput feature extraction and
  serialization, the apparatus behind those experiments, to be simplified and released
  separately; and the KIRBy research codebase (pure Python), comprising the adaptive hybrid
  representation and its blueprint, and the results on when and why it helps, on its
  extension to new embeddings and federated data, and on similarity and label noise.
- Closing remarks on what this means for choosing representations and models in noisy
  settings.

7.2  Future directions
- The directions that remain fall into a few groups, building on the same method.
- Future work could make forward compatibility symmetric: testing whether the hybrid
  representation improves a weak new embedding, not only whether the embedding improves the
  hybrid set. This is a proposed direction, not a result of this thesis.
- Carrying a blueprint across domains, and across noise distributions beyond Gaussian.
- A fuller federated deployment with Lhasa, and extending the noise-estimation work from
  regression to classification endpoints.


SUMMARY OF RESEARCH CONTRIBUTIONS
- A short closing list (following the peer theses): the NoiseInject paper and the planned
  hybrid-representation papers, the UK-QSAR conference presentation, the OPIG Blopig posts,
  and the three distinct open-source software artifacts (NoiseInject, MIT, released; the
  standalone Rust and Python chemical-processing and precompute pipeline, to be simplified
  and released separately; and the KIRBy research codebase, pure Python).


BIBLIOGRAPHY
- Author-year style, about 180 to 220 entries.


APPENDICES
- A: datasets and curation (Chapters 3 to 6): QM9, LogD, Caco-2, hERG, and the second-half
  datasets, with per-dataset provenance, accession numbers, and DOIs.
- B: the supplementary tables and figures for Chapter 3. Each is one file in
  KIRBy/thesis_appendix/noise/items/, listed in thesis_appendix/noise/INDEX.md, and placed in
  thesis.tex as its own section with the label app:noise:sec:<name>. They were copied on
  2026-09-27 from the paper's Additional files (qsar_qm_models/supplementary/) and are edited
  separately from the paper from now on. Rules: thesis_appendix/README.md.
  In it now, by name: hyperparameters_default, hyperparameters_tuned, pdv_descriptors (current);
  model_redundancy, representation_redundancy, model_icc, excluded_configurations,
  ecfp4_overview, clean_r2_vs_robustness, bayesian_transformation_tests,
  uncertainty_metrics_ecfp4, assay_variance_decomposition, rf_vs_qrf_assay (stale: from the
  earlier NDS study, to be replaced or dropped).
  Cited by the paper text but not built yet: variant_models_assay (the six variant models on the
  three assay datasets) and counterparts_assay (the T11 counterpart tables for logD, Caco-2 and
  hERG Ki).
- C: the method's full search space and the design-experiment tables behind its defaults
  (Chapter 4). The full vocabulary: downstream models (rf, xgboost, ridge, knn, lightgbm,
  mlp, svm, gp, with optional catboost, dnn, cnn, bnn, ngboost); feature budgets (roughly 10
  to 2,000, per-model capped); allocation methods (greedy, performance-weighted with its
  weighting schemes, dominant-greedy, fixed, two-stage, stability-selection); importance
  methods (treeshap, xgboost_gain, lightgbm_gain, permutation, kernelshap, deeplift,
  integrated_gradients, gradientshap); internal evaluators (match, or rf/svm/lightgbm/mlp);
  augmentation strategies; and the exact blueprint fields.
- D: extended result tables for the second-half chapters (Chapters 5 to 6), organised by the
  featured experiments: the cross-domain summary (per-task hybrid-versus-best-single deltas
  with the Wilcoxon test and per-domain win rates); the information-theoretic decomposition
  (mutual information per dimension against feature budget, with the PCA, random, and
  matched-dimensionality-null controls); the explainability and attribution tables (per-
  representation SHAP importance and named top features, with the feature-set stability index
  and fold-to-fold Jaccard); the forward-compatibility tables (per-new-representation
  contribution deltas against the random-Gaussian and row-shuffled nulls, with the
  blueprint-transfer ladder); the federated comparator tables (feature-importance sharing
  versus the similarity-weighted, performance-weighted, and distillation ensembles, under
  unequal per-party noise, plus the real-source three-silo comparison under its data-
  availability caveat); the similarity-activity correlation tables (per-representation
  correlation and the hybrid-versus-best-single delta with cross-dataset significance); the
  BindingDB ground-truth-noise table (estimator-by-space agreement with replicate-variance
  ground truth); and the remediation and noise-estimation tables (recovery of clean-data
  performance per estimator, representation, and noise level).
- E: data and code availability (all chapters). The full statement covers three distinct
  software artifacts. (1) NoiseInject: the open-source benchmarking framework, MIT, at
  github.com/adpunt/noise_inject, archived on Zenodo; released. (2) The standalone Rust and
  Python chemical-processing and precompute pipeline (from the qsar_qm work behind Chapters
  3 and 4): a standalone crate with optional Python bindings, a more complex version of
  which the author intends to simplify and release separately under [MIT / Apache-2.0] at
  [GitHub URL], archived at [Zenodo DOI]. (3) The KIRBy research codebase: pure Python (the
  adaptive hybrid representation engine, representation generators, dataset loaders, and the
  UQ and federated experiment code), to be released under [MIT / Apache-2.0] at [GitHub URL],
  archived at [Zenodo DOI], on [paper acceptance / thesis deposit]. KIRBy contains no Rust;
  it operates on precomputed numpy feature matrices. The datasets (QM9 and the OpenADMET
  LogD and Caco-2 sets; hERG Ki regenerable from ChEMBL by the protocol in Chapter 3; and
  the second-half antibody, protein, and gene-expression datasets) are publicly available
  under their upstream licenses, with provenance in Appendix A. The Lhasa federated data
  (Effiris) is proprietary and not redistributable; the federated demonstration also depends
  on Lhasa's FLuID code being importable, so it does not run standalone, and as a documented
  simulation compromise it uses the training set as the shared transfer pool rather than a
  separate public pool. Per-repository Springer-style itemised blocks (project name, home
  page, archived version, OS, language, requirements, license, restrictions) are included
  for all three artifacts. Cross-referenced from the front-matter Data and Code Availability
  statement.
