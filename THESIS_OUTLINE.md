DPhil THESIS OUTLINE: Adelaide Punt

Working title: Noise and Uncertainty in Molecular Property Prediction: Robust Models and Adaptive Representations

Department of Statistics, University of Oxford. Supervisor: Garrett Morris.
Collaborations: Lhasa Limited (Thierry Hanser, Stephane Werner); OPIG.

THIS IS THE ONE OUTLINE. It lives at KIRBy/THESIS_OUTLINE.md, next to thesis.tex.
- It is the 10 June outline (then at ~/Documents/Theses/THESIS_OUTLINE.md, the version thesis.tex
  was assembled from), updated on 2026-09-27. The May outline that used to be at this path, the
  KIRBy-first nine-chapter plan, was deleted; it is in git history (commit 147293c) if needed.
- The 10 June original was renamed ~/Documents/Theses/THESIS_OUTLINE_2026-06-10_original.md.

RESULT TABLES (generated from the server results; references only, not thesis text)
All in /Volumes/seagate/claude_analysis_2026-10-08/answers/. Rebuild every one from a new rsync:
  bash /Volumes/seagate/claude_analysis_2026-10-08/answers/_scripts/rebuild_all.sh /Volumes/seagate/<results folder>
- <exp>.md: one page per experiment: summary, yes/no/n/a per dataset, full sweep detail.
- SUCCESSES.md: every yes from every page; ends with results that have no page yet.
- DETECTOR_BY_MITIGATION.md: noise identification x mitigation, per dataset and noise type, base
  and hybrid; ends with the noise methods that do not fit that grid and where to read them.
- CLEANING_BY_METHOD.md: each cleaning method, what noise it was tested on and worked on.
- MEETING_METHODS.md: DivideMix, kNN label smoothing and replacement, detect-and-remove,
  censored-label correction: single representation against hybrid.
- NOISE_BY_DATASET.md, NOISE_METHODS.md, OVERVIEW.md, FAULTS.md, RERUN_2026_10_08.md.

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


CHANGES ON 2026-09-27 (second half: Chapters 4 to 7, rewritten from brainDumpOutline.txt)
- Old Chapters 4 (method), 5 (why hybrids help) and 6 (similarity and label noise) are replaced
  by three new chapters: 4, the flaws of the similarity assumption (a motivation chapter, no new
  method); 5, building adaptive hybrid representations (method, datasets, complexity, OPIG and
  federated results); 6, hybrid representations applied to similarity and label noise. Chapter 7
  keeps its place and now carries the closing argument from the brain dump.
- Old 4.1 (the LNL motivation) moves to 4.4. Old 4.2 to 4.6 move into 5.2 and 5.3. Old 5.2 to
  5.7 fold into 5.5 to 5.8. Old 6.2 and 6.3 fold into 6.2 to 6.5.
- Every Chapter 4 fact about the engine was re-checked against src/ on 2026-09-27. The old
  outline was wrong in 15 places (greedy, mlp, ridge, per-model budget caps, "top-3", evaluator
  list, pruning, blueprint contents); the corrected values are in 5.2.
- Section 1.5 (the other chat's half) still describes the old Chapters 4 to 7 and needs the new
  four. So do 1.3 (no questions for Chapters 4 to 6) and the Abbreviations list (new terms:
  budget, allocation method, internal evaluator, downstream model, blueprint, objective, LNL).

MARKERS USED BELOW (grep for them)
- [DIAGRAM ...]   a figure for Adelaide to draw, with what it needs to show.
- TODO(lit)       literature still to find; the claim is hers, the sources are not yet in refs.bib.
- TODO(prelim)    a preliminary number from a laptop run on a cut-down search. Every one of these
                  will be replaced by the server results. The method next to it is final.
- CHECK           two sources disagree (a file against the code, or the "Promising experiments"
                  file against its own logs). Says what each side says; not resolved.

Writing rules for the results (from the brain dump, applied to every TODO(prelim) section):
name the pattern, give the numbers in the dataset's own metric, no size adjectives in either
direction ("small", "large", "only", "modest"). The preliminary searches were cut to 5
representations, 3 folds, 1 importance method, 1 internal evaluator, budgets 100 and 250, rf and
svm (/Volumes/seagate/kirby_localrun/env.sh, KIRBY_LOCAL_GRID). The full runs search the whole
space; the text says the preliminary grid was cut, and does not say the effects are small.


THE SHAPE OF THE ARGUMENT, AND WHY THE HYBRID IS ON TOPIC
Added 2026-10-08. The full statement is Chapter 8, 8.2, and it is the thesis's claim. In short:
each step was forced by the previous one failing. Noise, then the noisy-label methods failing, then
the similarity they rest on failing, then the features themselves, then feature selection, then
back up the same path proving each level is repaired. The hybrid is not a second topic; it is the
answer to the question label noise produced. Chapters 1 and 8 both carry it. Every chapter's
introduction says which step of the chain it is.

TABLE AND FIGURE RULES (author, 2026-10-08; apply to every chapter and to every existing table)
- EXPERIMENT CODES NEVER APPEAR IN THE THESIS. No c02, s05, n18, fed03 in text, caption, table or
  figure. Address everything by what it does, in words. The codes in square brackets in this
  outline are file pointers for finding the results, and they stop at the outline's edge.
- MIXED METRICS NEED AN ARROW. Any table with a metric column, where the rows do not share one
  metric, puts an arrow next to the metric: a down arrow when lower is better, an up arrow when
  higher is better. RMSE and MAE get a down arrow. Spearman, AUROC, MCC, weighted F1 and R-squared
  get an up arrow. The caption says the arrow gives the direction.
- SELECTIVE BOLD. Bold a few cells in some tables, not every table and never a whole column.
  Two rules, and the caption says which one the table uses:
  (a) the best value in the row, for a table whose point is a comparison between named methods;
  (b) the counter-example, for a table whose point is that the answer changes with the dataset.
  Rule (b) is the one that fits the central claim, so prefer it in Chapters 6 and 7, and rule (a)
  in Chapter 5 where the hybrid is being compared with named alternatives.
  Settings, definition and roster tables get no bold at all: the noise conditions, the importance
  methods, the allocation methods, the search space, the per-dataset grid, the dataset table, the
  five representation tables and the per-dataset representation table.
- TODO(bold): the specific cells cannot be chosen for a results table until the full runs land.
  Every results section below carries a [BOLD] note saying what the bolded cell should show, not
  which cell it is.
- Existing tables that CAN be bolded now, with the cells named:
  - Chapter 4, all-pairs correlation and five-neighbour prediction per dataset
    (thesis.tex tab:flaws:distance_summary): bold QM9's all-pairs 0.19 and its five-neighbour 0.88.
    QM9 is the one dataset where distance tracks the label, and it is a computed property rather
    than an assay. Bold HLM clearance's -0.00 as the floor. Rule (b); caption says so.
  - Chapter 4, agreement between representations (tab:flaws:neighbor_overlap): bold the highest
    five-neighbour overlap, 0.52 on hERG Ki between ECFP4 and Mordred, and the lowest, 0.13 on QM9.
    The point is that even at best two representations share about half of a molecule's five
    nearest neighbours. Rule (b); caption says so.
  - Every other existing results table waits on the full runs.
- A FIGURE IS A DIRECT DISPLAY OF STORED NUMBERS. No derived statistic in a figure unless the
  outline names it. The analysis stays in the text.
- THE FIGURE SLOTS, with the shape fixed (details in the sections below):
  F1  hybrid against the best single representation, one dot per dataset, three panels by metric
      [CHECKED 2026-10-09 against the script: F1 is now a LaTeX table, not dots, on the author's
      word (commit f84364e). One row is one dataset and model, fold means over 5 folds, better of
      the pair in bold in every row. Its label is tab:hybrid_vs_best_single; the thesis.tex notes
      ask for tab:hybrid_vs_single with per-fold scores. Captions do not yet say what the arrow or
      the bold means. Source: scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md §0 and A.3 F1.]
  F2a the best number of features, per dataset and model
      [CHECKED 2026-10-09: built, 40 marks, 5 folds. No c09 file for 10 dataset-and-model pairs:
      abagym_Ang2, adme_hlm, adme_rlm, flab_binding_koenig, flab_expression_koenig, both models
      each. Source: FIGURES_AND_STYLE.md A.3 F2a.]
  F2b the best number of base representations, per dataset and model, beside F2a
      [CHECKED 2026-10-09: removed from the script in commit 493526a. The loader load_f2b still
      exists. Source: FIGURES_AND_STYLE.md A.3 F2b.]
      [DECIDE: bring F2b back into the script, or drop it from this list and from 5.5. Bringing it
      back is a script change only; dropping it leaves F2a alone in 5.5.]
  F3  blueprints, one horizontal bar per dataset, plus a deeper panel for one small-molecule set
      [CHECKED 2026-10-09: the bars are built, outer fold 0, first decided model. The deeper panel
      cannot be built from disk. c22 and c24 store only a count of features per representation,
      and the search cache stores only a hash of the selected feature indices. Naming the
      features means re-fitting the blueprint with tests/analysis/blueprint_trace.py, which costs
      compute. Source: FIGURES_AND_STYLE.md A.3 F3.]
  F5  distance against activity difference, single representation against hybrid, one dot per set
      [CHECKED 2026-10-09: now a LaTeX table, not dots (commit f84364e). 5 folds. Source:
      FIGURES_AND_STYLE.md A.3 F5.]
  F6  prediction from nearest neighbours, single representation against hybrid, one dot per set
      [CHECKED 2026-10-09: now a LaTeX table, not dots (commit f84364e). 5 folds. Source:
      FIGURES_AND_STYLE.md A.3 F6.]
  F8  did the cleaning work: two grids side by side, rows are named methods, shade is the share of
      the 8 model-and-configuration pairs improved. ROWS ARE BLOCKED BY ACTION -- delete, count
      less, move the label -- with the move rows sub-ordered by target, and each row labelled
      "detector, action". A horizontal rule and a block label between the three blocks. No row is
      averaged with any other; the blocking is only the row order, so the figure shows whether one
      action wins without any arithmetic across methods.
      [CHECKED 2026-10-09: the script's F8 differs from this in two ways at once. It writes one
      file per experiment, 10 files, titled by experiment. Its rows are grouped by detector under
      "By <detector>" headers, not blocked by action. It also drops the reweighting and
      training-side methods, so it has no "count less" block. Fold 0. Source:
      FIGURES_AND_STYLE.md §0 and A.3 F8.]
      [CHECKED 2026-10-09: F8 does not filter datasets. Its labels-as-delivered file has 4
      small-molecule columns and 8 others (7 antibody, 1 protein), against the domain-balance
      rule below. Four F8 files are wider than the page on the 9 Oct results: 10.0, 9.6, 7.4 and
      7.2 inches, against 6.5. Source: FIGURES_AND_STYLE.md §0 items 4 and 5.]
  F9  the censoring ladder, four marks on one axis, one panel per dataset
      [CHECKED 2026-10-09: built as specified, Caco-2 and hERG Ki, fold 0, one noise draw.
      Source: FIGURES_AND_STYLE.md A.3 F9.]
  F10 detection against action, one dot per detector ("repair" renamed; see NAMES FOR THE NOISE
      METHODS)
      [CHECKED 2026-10-09: the y axis still reads "After repair" and the files are named
      detection_vs_repair. On Caco-2 it draws unfinished checkpoint units; the CSV says so, the
      figure does not. Seven graded detectors drive no action, so they get no y value. That
      includes the Shapley-value detector, so F10 cannot show the 7.6 AUROC claim. Source:
      FIGURES_AND_STYLE.md A.3 F10.]
      [DECIDE: ITLM trims high-loss molecules inside the training loop. Under THE TEST below
      that may be noise-robust training, not a detector. F10 plots it as a detector. Keeping it
      changes nothing in the script; moving it removes the three ITLM settings from F10. Source:
      FIGURES_AND_STYLE.md A.3 F10.]
  F11 score against damage level, tentative on the full folds. Marks are grouped by the three
      groups, with noise-robust training kept visibly apart from mitigation, because the two are
      not alternatives to each other on the same data.
      [CHECKED 2026-10-09: the script does not follow this yet. Its five default methods are no
      cleaning, label_smooth, oof_smooth, remove@flag_share0.1 and js_shrink. None is noise-robust
      training, and colours are per method, not per group. hERG Ki, network on the hybrid, fold 0.
      Source: FIGURES_AND_STYLE.md A.3 F11.]
  Kept as TABLES on purpose: admitting extra features and the decoy counts; where importance
  weighting helps; the hybrid against ensembles; every settings and roster table.
- All figures come out of one script, run on the server. Its handoff prompt is written. It reuses
  the style of qsar_qm_models/scripts/generate_paper_figures_v2.py: 6.5 inch width, 300 dpi,
  DejaVu Sans at 11 point, seaborn "ticks", no legend frame, colour by model family with the
  marker separating variants within a family.
  [CHECKED 2026-10-09: the script is tests/analysis/thesis_figures.py. Where it now differs:
  - The thesis text width is 5.91 inches (main.tex:41, 21 cm minus 2 x 3 cm). The script sizes
    for 6.5. Every figure shrinks to 91%, so 7 pt text in F8 prints at about 6.4 pt.
  - The generated tables need the adjustbox package. Root main.tex does not load it. The stage1
    preamble does.
  - No Chapter 5-7 output of the script is in thesis.tex yet, and figures_ch5_7/ is not on the
    graphicspath. The files now in figures_ch5_7/ are stale: F1, F5 and F6 there are PNG dot
    plots, and the F8 files there hold 2.5 to 4.7 times fewer marks than the 9 Oct results.
  - The default results folder is the 8 Oct one. Pass --results-root
    /Volumes/seagate/kirby_results_2026-10-09.
  - By default F1, F2a, F5 and F6 show 7 datasets: caco2, herg, herg_ki, logd, ddg_combined,
    ginkgo_colloidal and ginkgo_titer. QM9, the ADME sets and reactions are left
    out. The script's docstring calls this filter a placeholder.
  - Printed text uses British "neighbour" in table labels and F8 row names.
  Source: scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md §0, A.1, A.5, A.6.]
  [DECIDE: which datasets the default filter shows. Keeping the 7 leaves QM9 out of F1, F2a, F5
  and F6, against "QM9 = clean noise control" and the domain-balance rule. Widening it is a
  script change only. This waits on the dataset list TODO below.]

WHICH DATASETS ARE SHOWN, AND HOW MUCH WEIGHT EACH DOMAIN GETS (author, 2026-10-09)
- BALANCE ACROSS DOMAINS, NOT BY COUNT. Many small antibody datasets must not outweigh a few large
  small-molecule ones just by number. No table, figure or paragraph lets one domain dominate
  because it has more datasets.
- SMALL MOLECULES ARE THE MAIN FOCUS.
- THE ANTIBODY EXCEPTION GETS ITS OWN DISCUSSION. On some antibody datasets the methods work far
  better than elsewhere. That is a use case to discuss properly, and those datasets must be named.
  TODO: identify them from the rebuilt analysis pages. Already recorded: SMAC colloidal stability
  (ginkgo_colloidal) under cleaning, 7.8.
- OTHER DOMAINS (reactions, proteins, genomics) are included in smaller parts, to show
  the methods are not domain-dependent and work as a general machine-learning technique.
- A dataset with missing results is mentioned only if what it has shows something; its missing
  results are then a limitation (7.14).
- TODO: the list of which datasets appear for which experiments is still to be decided with the
  author.
- CHAPTERS 5 AND 6 DATASETS, SIGNED OFF (author, 2026-10-09). This list replaces "THE LIVE
  DATASETS" in 5.3.1 for deciding what appears. The figure script's default dataset list is this
  one (THESIS_DATASETS in tests/analysis/thesis_figures.py).
  HOW IT WAS DECIDED. Each Chapter 5 experiment's yes/no answers per dataset were read from the
  analysis pages rebuilt on 9 Oct (17 experiments for Chapter 5, 12 for Chapter 6), counted as
  yes with both models / one model / neither / no result. Each antibody dataset then had a sanity
  check: is the task useful, and how do the scores compare with the literature on a comparable
  split (scratchpad/thesis_ch5_8/antibody_check/, one file per group, 9 Oct; a reading by Claude,
  not a measured result).
  - Small molecules, Chapters 5 and 6: qm9, caco2, herg_ki, logd, herg.
  - Small molecules, Chapter 5 only (no similarity results): adme_mdr1, adme_rlm.
  - OUT: adme_hlm. It has results in 5 of the 17 Chapter 5 experiments.
  - Reaction, Chapters 5 and 6: buchwald_hartwig. Never plotted as a single point in a by-domain
    figure; its result is said in the text.
  - Antibodies, Chapters 5 and 6: flab_thermo_tresanco, abagym_EGFR, ddg_combined, ginkgo_titer,
    ginkgo_pr_cho, ginkgo_colloidal, ginkgo_acsins.
  - Antibodies, Chapter 5 only, until their similarity results exist: flab_expression_koenig (it
    has 4 of 11 similarity experiments; the clearest antibody hybrid gain in Chapter 5),
    flab_binding_koenig (kept only if its runs finish; last in the queue), ginkgo_hic (a usable
    negative: the hybrid loses with both models on the benchmark's own folds).
  - Antibodies used for ONE result each, described briefly, never carrying the chapter:
    - abagym_Ang2, Chapter 5 only: one line on the hybrid against the single representation, with
      its caveats (labels read off a heatmap, the gp to be replaced, the win on 3 of 5 folds).
    - humanness_therapeutic: one named example of a task one simple representation already
      solves (AUROC 0.95 to 0.97; germline identity alone 0.968), where the hybrid has nothing to
      add. It never carries a hybrid-helps claim.
  - ginkgo_pr_cho also gets its own experiment on real censoring (58 of 197 labels at the assay
    floor; HANDOFF_CENSORING_PRCHO.md), the censoring counterpart of QM9 as the clean control.
    The floor is 0, the low reference antibody; that values below it are clipped to 0 is inferred
    from the labels, not stated by Ginkgo. Design and result slot: 7.7.
  - OUT, with the reason:
    - abcov_kd: labels from different papers, antigen not recorded, identical sequences differ by
      up to 1.26 log units.
    - cdr_flexibility: the split leaks (67 of 81 antibodies have their heavy and light loops in
      different folds); its one notable result is under label flips with a model that is not one
      of its two decided models.
    - abagym_VEGF: 285 of its 418 mutants are also in flab_binding_koenig.
  - CAVEATS TO STATE when these are written up: the gp numbers on abagym_Ang2, ginkgo_hic,
    ginkgo_acsins and ginkgo_pr_cho change when the gp is replaced; the ddg_combined svm gain is
    one fold (the network gain holds on 4 of 5); flab_thermo_tresanco pools five assays in one
    label; ginkgo_pr_cho has 58 labels at the assay floor.
  - herg_ic50 (noise only, repeat measurements): keeps running, never cut (author, 2026-10-09).
  - OPIG AND CROSS ROWS (author, 2026-10-09). Reported: opig_qsar (Dablander et al. 2023 setup,
    ChEMBL D2, factor Xa, Mpro; 5.7), opig_shared_vectors_remedy and opig_blueprint_transfer (one
    line each, 5.7), opig_shared_vectors_census (5.7 and Chapter 4). Not in the thesis and stopped
    or skipped: opig_ginkgo (c02 covers the Ginkgo sets on the competition folds), opig_graphinity
    (ddg_combined covers the task), opig_cgr_comparison (reactions are text-only),
    opig_trastuzumab (datasets cut 2026-10-08), opig_humanness (humanness_therapeutic covers it),
    opig_scaffold_hybrid and opig_scaffold_similarity (wait on flab_binding_koenig), cross c14
    (repeats c02), cross s09 (the figure script makes the same summary). cross c03 (do the best
    hybrid settings follow dataset properties): KEPT, reported as the conditional-conclusions
    result for Chapter 5. Its wait list (plan.py:374, every dataset in the run) is re-pointed to the
    signed-off Chapter 5 datasets WITHOUT flab_binding_koenig, so it can start once those finish
    their complexity runs.
  - CHAPTER 7 datasets: not yet decided. Per dataset, the author decides whether its noise
    experiments continue (see HANDOFF_LEDGER_CUTS.md).

NAMES FOR THE NOISE METHODS (author, 2026-10-09; applies to every chapter, 2 and 4 to 8)
Settled because the same methods were being called different things in different chapters. Use
these words and no others. The point is that a reader can tell which of three things a method is
without being told twice.

THE THREE GROUPS
- A DETECTOR scores every molecule on how wrong its label looks. It does nothing else. It changes
  no label and fits no final model. High score means the label looks wrong. The number it gives is
  that molecule's SUSPICION SCORE. Formal name for the group: noise detection.
  A small split inside it, worth one sentence and no more: most detectors only rank molecules,
  while a few return a number in the label's own units and so can feed inverse-variance weighting
  or a noise budget. Those are the ones built on a model's own predicted spread.
- A NOISE MITIGATION METHOD acts on the training data using a detector's scores. Three actions
  only: delete the molecule, make it count less, or change its label. Formal name: noise
  mitigation. Every one of these is a detector plus an action, and the thesis always names both.
- NOISE-ROBUST TRAINING leaves the data alone and changes how the model trains: the loss, the
  kernel, a noise term in the model, regularisation, or reweighting inside the training loop. It
  has no detector. That missing half is what makes it a separate group, not anything about being
  robust. The author's own name for this group is "noise-aware learning"; "noise-robust training"
  is the established term and is already in the thesis in sec:lnl_background, so prefer it and say the other
  as a gloss if wanted. [DECIDE: keep noise-robust training, or switch to noise-aware learning.]

WHAT SAYS WHICH MOLECULES -- THREE KINDS, NOT ONE (added 2026-10-09 after sorting all 85 methods)
A detector is not the only thing that can say which molecules to act on. There are three, and
this was the missing piece: 22 of the 85 methods in the noise work had no home without it.
- A DETECTOR infers the suspicion score from the features and the labels. 42 of 85.
- A KNOWN FLAG is recorded in the data and nothing is inferred: the label sits at the assay
  limit, the molecule was measured so many times, a named source reported it, it belongs to a
  scaffold group. 22 of 85.
- EVERY MOLECULE, with no selection at all -- moving every label part of the way toward an
  out-of-fold prediction, mixing pairs, a power transform on all the labels. 6 of 85.
WHY THIS IS NOT JUST VOCABULARY. Correction works far better when you KNOW which labels are
damaged than when you have to infer it. The censoring corrections are the strongest results in
the whole noise programme and every one of them is flag-driven. The old grouping hid that,
because it filed them as though a detector had found them. Chapters 7 and 8 both say it.
And a method that acts on every label cannot be compared with a detector-driven cleanup on equal
terms; it is a different claim and the tables must not put them in one block.

THE TEST, FOR ANY METHOD
What says which molecules -- a detector, a known flag, every molecule, or nothing?
  Nothing -> noise-robust training. The loss, the kernel, a noise term or the loop changes.
  One of the first three -> then what is done to them: delete, count less, or move the label.
         That is noise mitigation, and the thesis names both halves every time.
  A detector with nothing done afterwards -> it is only a detector. It belongs in the detection
         results and not the mitigation ones.
This is the same test the code already uses: n07_remediation.py:362 keeps huber_weight out of the
label-repair list because "it sets a row weight and never touches the label".

[OPEN POINT, one decision] The author's wording for mitigation was "modifying or removing the
label". Making a molecule count less does neither, so down-weighting has no home under that
wording. Recommended fix: widen mitigation from "the label" to "the training data", with the three
actions above. That keeps three groups and leaves nothing orphaned. The alternative is a fourth
group for weighting, which is what the 2026-10-09 analysis page did, and which left 10 of its 29
sections marked "mixed".

ONE ACTION, NOT THIRTEEN
Thirteen method names in the repo all do the same thing: they move a label. label_smooth,
knn_smooth, oof_smooth, gp_smooth, prox_smooth, js_shrink, quantile_clip, constant_lift,
move_confidence, move_proportional, winsor, censored_em, truncated_normal_mean. They differ in
WHERE THE LABEL MOVES TO and HOW FAR. Targets used: the neighbours' mean, an out-of-fold
prediction, a Gaussian process prediction, the scaffold group's mean, a quantile bound, the assay
limit. Write them that way. Chapter 7's methods section is then a few detectors crossed with three
actions, not a list of thirty names, and the comparison between the movers becomes a comparison of
targets, which is the interesting question anyway.
So a full description of a mitigation method is four things: the detector, how many molecules it
acts on, which of the three actions, and for a move, the target and the distance.
The code keeps its own spellings (CLEANUPS, CORRECTIONS, OPERATIONS, ACTIONS, REPAIR). Those are
five words for one thing and they stop at the edge of the code. None appears in the thesis.

THE TERM "LNL"
LNL names the PROBLEM, not the mechanism: you have noisy labels and want a good model. It is not
one of the three groups and must never be used as a group name. Use it sparingly, in Chapters 2
and 4 only, where the literature is being described. Two reasons to keep it narrow: the LNL
literature is overwhelmingly deep networks and classification, while most of this thesis is trees
and regression; and Song2022's five categories are all training-time, so a detector that runs
once, deletes, and hands a smaller dataset to any model has no slot in them. Cite Song2022 for
noise-robust training and not for mitigation.

THE TWELVE, AND WHAT CHAPTER 4's SECTION IS CALLED
Four of the twelve have a detector and are mitigation: small-loss selection, confident learning,
distance-based selection and multistage cleaning. They score molecules and drop them before a
refit; multistage cleaning does this three times, 10% per round. The other eight have no detector
and are noise-robust training. The code makes the same split in src/kirby/utils/lnl_methods.py as
DATA_LEVEL_METHODS against the rest (multistage cleaning moved into it 2026-10-09).
- "LNL survey" was a name the author gave this work about 1.5 years ago and it carries no meaning
  for a reader. It does not appear in the thesis.
- Chapter 4's section is "a small sweep of published noise-handling methods". "Noise-handling"
  because the twelve span two of the three groups, so neither "mitigation" nor "noise-robust
  training" covers them on its own. Chapter 7 is the expansion of that sweep, and each chapter
  says so in one sentence.
- TODO thesis.tex, not yet applied: rename \subsection{Exploratory study of learning with noisy
  labels} (sec:lnl_motivation) and \section{Results: Learning with Noisy Labels by Noise Type} and
  \subsection{Gaussian Noise: Methods for Learning with Noisy Labels}; rewrite the categories
  paragraph in sec:lnl_background; and work through the remaining uses of "LNL" and "learning with
  noisy labels" in the body text and in the comment blocks. Comment blocks count: they are what
  the drafting works from. Nothing in thesis.tex has been changed for this yet.

WHAT THIS CHANGES IN THE FIGURES AND THE ANALYSIS
The groups are how the results should be grouped, not just how they are described. The experiment
code is not a unit of analysis -- n07 means nothing to a reader and nothing to the author either;
what is IN n07 is a detector crossed with eleven actions, and that is the thing worth plotting.
- F8, F10 and F11 are respecified above. F8 is the one that changes most: its rows block by action.
  [CHECKED 2026-10-09: none of the three follows this yet. F8 is one file per experiment with rows
  grouped by detector, and it drops the count-less methods. F10 still says "repair". F11 colours
  per method, not per group. Details under THE FIGURE SLOTS above. Source:
  scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md A.3.]
- The figure script's handoff prompt (scratchpad HANDOFF_FIGURE_SCRIPT.md) carries this. The script
  is not written yet, so this costs nothing now and would be expensive later.
  [CHECKED 2026-10-09: stale. The script has existed since commit 58bbd3f (2026-10-09 00:36), as
  tests/analysis/thesis_figures.py. So applying the grouping above is now a change to written
  code. Source: FIGURES_AND_STYLE.md §0 item 1.]
- Any new figure or table about noise is organised by group first, then by detector and action.
  Never by experiment, and never by averaging the methods inside a group -- blocking the rows gives
  the group comparison without any arithmetic across methods.
- The 2026-10-09 analysis page is organised by experiment with a group label bolted on top. It
  should be organised by group, then action, then detector, with the experiment only as a file
  pointer. That is a rebuild of answers/_scripts/meeting_methods.py and meeting_readable.py, not a
  thesis change. [OFFERED, NOT AUTHORISED]

THE SORTED LIST OF EVERY METHOD -- DONE
/Volumes/seagate/claude_analysis_2026-10-08/answers/_ref/METHOD_GROUPS.md holds all 85, each with
what selects the molecules, the action, what it does in one line, and where it ran. 63 are noise
mitigation and 22 are noise-robust training. Nothing is "mixed". It is a reading of the code by
Claude on 2026-10-09, not a measured result, and three rows are marked CHECK for the author.
Rebuild it with scratchpad/build_method_groups.py, which holds the classification as a dict, so a
correction is a one-line edit and new methods are added there.

STILL TO DO
- Settle the two open decisions above (down-weighting's home; the third group's name).
- Check the three CHECK rows in METHOD_GROUPS.md.
- Point meeting_readable.py at the table instead of keyword-matching section titles
  (meeting_readable.py:436-452 holds the old three groups and the keyword rule). Until that is
  done the analysis page and this outline disagree, and the page's own "noise mitigation" means
  something narrower than this one's.
- Fix the language in chapters 1 to 4. Handoff prompt written and ready to paste:
  ~/repos/KIRBy/HANDOFF_CH14_NOISE_NAMES.md. It carries the Chapter 2 terms table (tab:noise_terms)
  as verbatim LaTeX, the per-chapter line inventory, and the instruction to keep both section
  labels so the 18 cross-references survive. Chapter 3 has no uses and is left alone.

CHAPTER 4  THE FLAWS OF THE SIMILARITY ASSUMPTION  (about 15 to 20 pages)
A motivation chapter. It does not introduce a method. It argues, with small measurements, that
(i) standard base representations carry many features that do not vary on a given dataset and
cannot tell some molecules apart, (ii) similarity computed over those representations inherits
both problems, and (iii) methods that rely on that similarity, including the learning-with-
noisy-labels (LNL) methods from the confirmation report, fail as a consequence. Every result in
this chapter is smaller than Chapter 3. The chapter ends by stating what a representation would
need to fix this, which Chapter 5 builds.
Opening note (Dablander's convention): 4.4 reuses the confirmation report's LNL study; say so and
cite it.

4.1  Introduction
- Bridge from Chapter 3: under noise, the model governed robustness and the representation
  governed clean performance. This chapter asks what the representation contributes to methods
  that compare molecules to one another rather than predict from them.
- Research questions, stated as bullets at the end of the introduction:
  (1) On a given dataset, how many of a standard representation's features vary, and how many
      molecules does it map to the same vector?
  (2) How well does distance in each representation track differences in the property?
  (3) Do methods that rely on that distance (noise detection by neighbours, distance-based LNL
      methods) do better than methods that do not?
- Related work on similarity and structure-activity landscapes goes here, not in Chapter 2:
  Chapter 2.3 already defines the similarity assumption and activity cliffs. This section adds
  the modelability literature: MODI (Golbraikh et al.), Sheridan et al. "Experimental error,
  kurtosis, activity cliffs, and methodology", van Tilborg et al. 2022 (MoleculeACE),
  Medina-Franco "Activity cliffs: facts or artifacts?" (activity cliffs are sometimes dismissed as
  measurement error; removing them loses information). TODO(lit): only van Tilborg 2022 is cited
  in thesis.tex so far; the other three are not in refs.bib.

4.2  Flaws in standard base representations
Lead with this (brain dump). One subsection per flaw, each with one small measurement.

4.2.1  Most features do not vary on a given dataset
- The argument: a 2048-bit ECFP4 is built for all of chemistry. On one dataset from one chemical
  series most bits are off in almost every molecule. The features that matter for the task are a
  small subset, and which subset depends on the task.
- Evidence already measured (counted from /Volumes/seagate/kirby_cache/rep_cache/<ds>/<rep>.npy
  on 2026-09-27; CHECK: the pipeline reads a second cache, reps/<sha>, first, so these matrices may
  not be byte-identical to what the runs used). Each row is one dataset and representation;
  "constant" counts features with one value in every molecule; ">=95%" counts features where one
  value covers at least 95% of the molecules:
    herg_ki (1,412 molecules): ecfp4 2048 features, 105 constant, 1,815 at >=95%
                               top_torsion 2048, 1,012 constant, 1,747 at >=95%
                               atom_pair 2048, 201 constant, 977 at >=95%
                               mordred 1613, 242 constant, 336 at >=95%
                               pdv 200, 21 constant, 55 at >=95%
                               rdkfp 2048, 0 constant, 18 at >=95%
    caco2 (2,158):  ecfp4, 194 constant, 1,826 at >=95%
    logd (5,028):   ecfp4, 39 constant, 1,834 at >=95%
    qm9 (9,978):    ecfp4, 0 constant, 1,997 at >=95%
- Planned statistic (not run; about 20 lines, same masks as
  src/kirby/utils/feature_filtering.py:14): the same three counts for every representation in
  every molecular pool (herg_ki, caco2, logd, qm9, adme_rlm, adme_mdr1), on the reps/<sha> cache
  the pipeline reads. Table 4.1: one row per dataset, one column per representation.
- Literature this puts numbers under: Dablander's thesis says, without a count, that "in common
  real-world molecular data sets, the least frequent substructures usually only appear in a few
  compounds" (p. 112; also pp. 121-122). His count is of substructures before hashing; this one is
  of the 2048 bits after hashing.
- Caveat to state in the text: a constant feature does not change Tanimoto (bits off in both
  molecules are ignored) and adds zero to a Euclidean or Manhattan distance on standardised
  features. What does distort those distances is near-constant features: after standardisation
  they take large values on the few molecules that have them. So the argument is about the
  >=95% column, not the constant column.

4.2.2  Hash collisions: one bit, several substructures
- ECFP folds each substructure identifier into 2048 bits, so bits are shared by unrelated
  substructures (Rogers and Hahn 2010; Dablander et al. 2024, Sort & Slice; Gütlein and Kramer
  2016, filtered circular fingerprints, refs.bib key Gtlein2016).
- Planned statistic (not run; no code in KIRBy): median number of bits set per molecule, and the
  share of occupied bits that more than one distinct substructure identifier hashes into,
  computed from the unfolded Morgan identifiers. On herg_ki, caco2, logd.

4.2.3  Molecules the representation cannot tell apart
- Evidence already measured: tests/experiments/opig_shared_vectors_census.py, results
  tests/results/comparisons/opig_shared_vectors_census.csv (16 Sep). It fits no model and splits
  every tie into repeated input, all-zero row, or genuine merge.
  - herg_ki (1,412 molecules after InChIKey deduplication): ecfp4, atom_pair, pdv and mordred
    each give 56 molecules an identical vector, and all 56 differ only in stereochemistry. maccs
    gives 377 molecules (26.7%) an identical vector. chembert, grover, mhg_gnn and the 3D
    representations give 0.
  - logd (5,028): ecfp4 merges 641 molecules (633 stereo-only); maccs merges 2,343 (46.6%).
  - caco2 (2,158): ecfp4 merges 179 (177 stereo-only); maccs merges 804 (37.3%).
  - The cost on herg_ki: the best score any model could reach is R2 0.998 on ecfp4 and 0.978 on
    maccs; the widest label spread inside one tied group is 0.57 pKi (ecfp4) and 0.78 pKi (maccs).
  - Learned models merge too: the ChemBERTa-77M-MTR tokeniser gives 72 of 1,415 herg_ki
    molecules identical token ids (~/repos/qsar_qm_models/results/chemberta_collisions.json).
  - CHECK: smited ties 56 molecules in the 16 Sep file and 52 in the 18 Sep re-run.
- Literature: GNN expressivity is bounded by the 1-WL test and some non-isomorphic graphs map to
  the same vector (Xu et al.; Morris et al.; Dablander thesis pp. 35-38).
- Planned extension (not run): the same census on qm9 and adme_mdr1.

4.2.4  Learned embeddings cannot be read chemically (a short point)
- An embedding feature has no chemical name; the feature-name registry
  (src/kirby/importance/hybrid.py:259-296) resolves it only to "representation name + dimension
  index". Dablander calls pooled GNN vectors "obscure real numbers that cannot be directly
  interpreted" (p. 41).
- Planned statistic (not run; no code exists): for each embedding feature, the largest absolute
  Spearman correlation with any of the 200 PDV descriptors; report how many features exceed 0.8.

4.2.5  Different representations carry different information
- This is the reason to combine them (sets up Chapter 5). Evidence already measured, from the
  f26 redundancy results (tests/experiments/results/f26_redundancy_penalty_*.json): the
  Spearman correlation between two representations' pairwise distances over the same molecule
  pairs. Median over all representation pairs: herg_ki 0.315, logd 0.388,
  flab_thermo_tresanco 0.509. Least alike on herg_ki: graph_kernel and
  mordred, 0.019. Most alike: mordred and pdv, 0.720.
- CHECK before quoting: the f26 pools predate the signed-off 10-representation pools (herg_ki
  there includes chemberta, maccs and molformer, and omits atom_pair and chembert).
- Table 4.2 (Dablander's Table 2.4 as a model): one row per representation family, what it
  encodes, and its weaknesses on these measurements. Covers small molecules only here; the other
  domains' representations are in 5.4.

4.3  Flaws in similarity and other pairwise computations
- The argument: a distance compares every feature. If most features are irrelevant to the task,
  most of the distance is irrelevant to the task, and a fixed measure cannot know which features
  matter for which property. The similarity assumption holds only relative to a measure
  (Maggiora et al. 2014), and the measure is fixed by the representation.
- Replace the confirmation report's "exponential growth in the number of irrelevant pairwise
  interactions" (paper.tex:784, copied to thesis.tex:1265): no result or citation supports it.
  The literature statement is distance concentration in high dimensions (Beyer et al. 1999;
  Aggarwal et al. 2001), which the planned statistic below measures. TODO(lit): Beyer 1999 is not
  in refs.bib.
- Distance definitions: the eight metrics already written with formulas in thesis.tex:1665-1681
  (Euclidean, Manhattan, cosine, Chebyshev, Canberra, correlation, Bray-Curtis, Mahalanobis) move
  here, plus Tanimoto (Bajusz et al. 2015, in refs.bib).
- Evidence already measured:
  - Distance against property difference. s01 (tests/experiments/similarity/s01_similarity_
    baseline.py; results tests/results_local_sim/s01_similarity_baseline/, 18 Sep): Spearman of
    Manhattan distance against absolute activity difference over 5,000 sampled molecule pairs.
    herg_ki: ecfp4 0.049, pdv 0.089, atom_pair 0.072, maccs 0.030. caco2: ecfp4 0.163, pdv 0.193,
    atom_pair 0.225. adme_mdr1: ecfp4 0.231, pdv 0.227, atom_pair 0.280.
  - Local against global. On herg_ki, predicting each molecule's activity from its 5 nearest
    neighbours under ECFP4-count Tanimoto reaches Spearman 0.709
    (tests/experiments/similarity_metrics/aggregate_experiment.json), against 0.05 for all pairs.
    The text must say which of the two the argument is about: nearest neighbours carry signal
    that the all-pairs correlation does not show.
  - Fingerprint distance against structure. The same ECFP4-count Tanimoto agrees with
    maximum-common-substructure distance at Spearman 0.043 on herg_ki; the RDKit fingerprint at
    -0.334. On logd, 0.37 (ECFP4-count) and 0.46 (torsion).
  - Activity cliffs. At ECFP4 Tanimoto >= 0.55 and an activity gap of at least twice the labels'
    median absolute deviation (similarity_helpers.py:695 with calibration off): herg_ki 451 of
    1,412 molecules (31.9%) sit in 710 cliff pairs; caco2 1,625 of 2,158 (75.3%); adme_mdr1 125 of
    2,641 (4.7%).
  - CHECK: SIMILARITY_AUTORESEARCH_STATUS.md:377-399 implies the best importance-weighted single
    representation on herg_ki scores about 0.33 on the s01 statistic; s01 gives 0.03-0.09 for
    unweighted single representations. Not established what explains the gap.
- Planned statistics (not run):
  - Distance concentration: relative contrast (d_max - d_min)/d_min per query molecule, and the
    skew of how often each molecule appears among others' 5 nearest neighbours (Radovanovic et al.
    2010). Code exists at c15_ml_theory_depth.py:199 but has run only on antibody datasets; it
    standardises first, so also compute it on raw Tanimoto.
  - Neighbour disagreement between representations: Jaccard overlap of each molecule's 5 nearest
    neighbours under ecfp4, pdv and one learned embedding. No code exists.
  - Activity cliffs at the MoleculeACE thresholds (Tanimoto >= 0.9, gap >= 1 log unit).
- [DIAGRAM D4.1] Two panels on one dataset (herg_ki). Left: activity difference against Tanimoto
  similarity for sampled pairs, binned (the Patterson 1996 similarity-property plot). Right: the
  same pairs under a second representation. Shows that "similar" pairs differ in property as much
  as dissimilar ones, and that the two representations disagree about which pairs are similar.

4.4  Consequences for methods that rely on similarity
The LNL results from the confirmation report are the negative example.

4.4.1  The exploratory LNL study (confirmation report)
- Source: /Users/apunt/Downloads/Confirmation_Report/paper.tex:666-775. QM9 HOMO-LUMO gap, 10,000
  molecules, scaffold splits; Gaussian noise sigma 0 to 1.0 on the training labels only; five
  representations (ECFP4, Sort & Slice, PDV, canonical and randomised SMILES); 12 LNL methods
  adapted to regression, 221 configurations.
- Finding (paper.tex:721): methods that use a distance metric did worst. All 20 configurations
  with the lowest noise retention used one; none reached both clean R2 > 0.6 and retention > 80%.
  The methods that did not use distance did better: DivideMix retained 86.7%, small-loss 84.2%,
  the uncertainty curriculum 82.7%, heteroscedastic GP 79.0% at sigma = 0.6.
- Existing prose: thesis.tex:1250-1266 (moves here from old Chapter 4.1). Fix the doubled full
  stop at 1255.
- CHECK, three things before quoting:
  (1) Which distance metrics ran. paper.tex:676 says Tanimoto on ECFP4 and Euclidean on PDV;
      paper.tex:721 lists Euclidean, Manhattan, cosine, Mahalanobis and Tanimoto; the job names in
      ~/Downloads/experiments/slurm_scripts/ contain only euclidean, mahalanobis and tanimoto.
  (2) Sample size. The 8 Mahalanobis commands used -n 5000, not 10,000.
  (3) The 221 count and the top-20 claim cannot be recomputed locally: the raw CSVs were on the
      server ($DATA/results/comprehensive_noise_study/, unverified).
- The ANOVA line ("distance metric usage is the dominant factor in variance in R2") has no effect
  size anywhere in the report.
- HOW THIS STUDY COMPARES WITH CHAPTER 7's SURVEY: the same twelve methods, no shared dataset,
  and the four optional components crossed here (uncertainty, distance, kernel, loss function)
  are NOT swept in Chapter 7. So this study holds the only measurement of the distance axis and
  cannot be replaced by Chapter 7. Full comparison in Chapter 7, 7.11.

4.4.2  Noise detection by neighbours
- Confirmation report Chapter 5 (paper.tex:1028-1146): 8 of 860 cleaning configurations improved
  R2 at both sigma = 0.3 and 0.6; paper.tex:1144 "the concept of using a sample's neighbors to
  determine its ground truth is flawed". Caveat: its hybrid rows used greedy allocation, which is
  cut, so only the single-representation rows can be cited.
- Sample-removal file tests/results_server/noise_robustness_v3/qm9/all_results.csv (816 rows;
  pdv, mhggnn, and a 100-feature hybrid of the two; 8 distance metrics). Recovery = (R2 after
  cleaning - R2 noisy)/(R2 clean - R2 noisy). Neighbour-based removal: 157 of 768 configurations
  had positive recovery; median recovery at sigma = 0.3 was -2.06 (pdv), -2.15 (mhggnn), -2.12
  (hybrid). The z-score filter, which reads no distance, gave the only positive non-neighbour
  rows (pdv +0.12 at sigma = 0.3).
- Current code, same pattern on an antibody dataset: in n09's preliminary run
  (flab_thermo_tresanco, onehot input), distance_selection is the lowest of the 12 LNL methods at
  both noise levels (Spearman 0.1790 at 0.5x against 0.4534 for the plain network). TODO(prelim).
- Neighbour-based detection of injected noise on herg_ki: detection AUROC 0.646-0.669 averaged
  over the noise grid, about 0.59-0.62 at sigma = 0.3, across 6 single and 6 fused distances;
  fusing metrics never leaves the single-metric range (aggregate_experiment.json).
- Planned control (not run; one new argument): DistanceBasedSelection
  (src/kirby/utils/lnl_methods.py:722) with neighbours from ECFP4, from PDV, and from random
  molecules via its precomputed_neighbors hook. If ECFP4 neighbours do no better than random
  neighbours, that is the chapter's cleanest single result.
- CENSORING INVERTS THE DETECTION SCORE (written into thesis.tex 2026-10-05, paragraph
  "Censoring inverts the detection score"; numbers from f38, herg_ki/caco2/adme_mdr1 complete).
  Clipped labels are ranked the LEAST suspicious by the neighbour gap: detection AUROC 0.2768
  herg_ki, 0.3151 caco2, 0.3936 adme_mdr1 at k=10 with every neighbour counted equally. Of the
  rows measured (one row = one feature space, fold and model), 0 of 2550, 8 of 2580 and 200 of
  2520 score above 0.5. Replacing a clipped label with its neighbours' mean was worse than
  leaving it in every row. Four TODO(delve) items are listed in the thesis.tex comment above
  that paragraph.

4.4.3  Other standard uses of similarity that the same flaw reaches
- k-nearest-neighbour prediction; applicability domain (distance to the training set; s06 gives
  Spearman of distance against error +0.30 on herg_ki, +0.21 caco2, +0.38 adme_mdr1 at shipped
  settings, from SIMILARITY_AUTORESEARCH_STATUS.md:455-475, a dated document, re-check);
  activity-cliff detection; diversity selection and clustering for splits; nearest-neighbour
  weighting in federated ensembles (FLuID weights each site's prediction by fingerprint distance,
  5.8). TODO(lit): one citation each.

4.5  What a representation would need
- Short. From 4.2-4.4: features chosen for the task, from several representations that disagree,
  with the irrelevant ones removed before any distance is taken. That is the adaptive hybrid
  representation of Chapter 5.
- Brain dump's origin story, as the bridge: the first version selected features only to compute
  similarity; the selected sets also worked as ordinary model inputs, which made the selection
  itself the method.

4.6  Conclusion
- One paragraph per research question, then the bridge.


CHAPTER 5  BUILDING ADAPTIVE HYBRID REPRESENTATIONS  (about 40 to 50 pages)
Step 5 and 6 of the chain (see THE SHAPE OF THE ARGUMENT above): select features per dataset, and
show the resulting representation predicts better. 5.1 and 5.2 are accurate and current as of
2026-09-27; 5.3 onward were rewritten 2026-10-08.
The method chapter and the first set of results. Planned as the first of the two hybrid papers.
Results are preliminary; the methods are final (they are what the code does).

5.1  Introduction
5.1.1  Combining representations: what exists
- TODO(lit), the main literature gap in the second half. Adelaide is doing the review. Recorded
  here from what thesis.tex already cites, so nothing is lost:
  - Concatenation and fingerprint fusion: Xie2020 (conjoint fingerprints), Laufktter2019
    (structural plus bioactivity fingerprints), Turkina2025 (fusion across six fingerprints by
    feature importance; the closest prior work). All in refs.bib. thesis.tex:1309-1315.
  - Early, late and intermediate fusion, multiple kernel learning: Stahlschmidt2022, Deng2023,
    Steyaert2023, Wang2021 (MOGONET), Zhang2024, Gonen2011, Cichonska2018 (thesis.tex:1319,
    1623). Only Deng2023 is in refs.bib.
  - "No single representation wins": Deng2023 (62,820 models), Jiang2021, Mayr2018, Kenlay2024,
    Schmirler2024, ElNahhas2025 (thesis.tex:1520-1523). Only Deng2023 in refs.bib.
  - Named in the old outline but never cited: Xu, Tao and Xu 2013 (multi-view learning overfits
    on small samples), Valsecchi et al. 2020 (consensus of predictions), Ponzoni et al. 2017
    (hybrid feature selection and learning).
  - 39 keys cited in the old Chapters 4 to 7 have no bib entry anywhere (thesis.tex header points
    to KIRBy/PAPER_REFERENCES.md, which does not exist). List in the thesis.tex audit of
    2026-09-27.
- The brain dump's claim: most prior combination is simple concatenation, and a concatenated
  representation scores between its components because it adds as much irrelevant information as
  relevant. The argument is to combine and filter at the same time.
  - Evidence that exists: c11's pca_matched arm (every representation concatenated, then PCA to
    the hybrid's width) is below the best single representation in 53 of 54 (dataset, model)
    rows over 9 datasets and 6 models (tests/results/c11_ensemble_vs_hybrid/, May 2026, old pools,
    includes mlp). [CHECKED 2026-10-09 against the 9 Oct results: on the current pools this no
    longer holds everywhere. With dnn, all representations joined and reduced by PCA beats the
    hybrid on LogD, Caco-2 and adme_mdr1, 5 folds. Numbers in 5.4. c11's PCA is now to each
    decided budget, not to the hybrid's width. Source: scratchpad/thesis_ch5_8/CH5_RESULTS.md §0
    item 1, §2.] The antibody geometry survey (7 structure-geometry groups concatenated onto
    the best single representation) went down on 7 of 8 datasets, median -0.0528
    (GEOMETRY_SURVEY_RESULT_2026_09_11.md, a dated document; result files not on this machine).
  - Gap: no plain two-representation concatenation test exists on the current pools (the
    rep-pruning "concat test" in REP_PRUNING_LOG.md:22-23 has no per-pair results). Planned small
    local test: for each molecular dataset, concatenate each pair of its top 3 representations,
    score with rf and dnn, and report where the concatenation lands between its two parts.

5.1.2  Other ways to use several representations
- Dimensionality reduction that merges features rather than selects them (PCA). TODO(lit):
  Niazi and Mariam 2023 named in the old outline, not in refs.bib.
- Ensembling: one model per representation, predictions pooled. Compared directly in 5.4 (c11).
- Classical feature selection (mRMR, Peng et al. 2005; RFE, Guyon et al. 2002; Guyon and
  Elisseeff 2003). Compared directly in 5.5 (c17). TODO(lit): none in refs.bib.
- Why this method differs from pooling all features into one selection: the budget is split per
  representation first. Evidence in 5.2.3 (allocation method and representation count): some
  datasets are best with 2 or 3 representations, others with all 10.
- Selecting features also shrinks the input, and models differ in how many features they want and
  which. Evidence in 5.5.
- Prior art on label-driven selection to address head-on: Dablander (thesis p. 138) found that
  unsupervised Sort & Slice beat two label-based bit-selection methods and suggested label-based
  selection may overfit. The hybrid ranks features by importance on training labels, so 5.9
  (overfitting) answers this.

5.1.3  The hybrid, and two ways to use it for similarity
- Definition 5.1 (adaptive hybrid representation): a feature set assembled by selecting features
  from several base representations, under a budget (the maximum number of features), chosen for
  the dataset and downstream model by a search. It prepares inputs for standard models; it is not
  a model. NAMING (decided 2026-09-27): "adaptive hybrid representation", usually just
  "hybrid". "KIRBy" only when referring to the repository.
- Its intended use: representations that say different things about the molecule (PDV, ECFP4,
  graph embeddings, sequence embeddings), with the features for this task picked from all of them.
- Two uses for similarity, both returned to in Chapter 6: (a) build an ordinary hybrid and compute
  distances in it; (b) change what the search optimises, so the hybrid is chosen for how well each
  molecule's activity is predicted from its 5 nearest neighbours (objective='similarity'; the
  "similarity hybrid").
- Research questions for the chapter, as bullets.

5.1.4  Blueprints and what they tell a chemist
- A finished search is saved as a blueprint (5.2.6; results in 5.6). Features from PDV and substructure
  fingerprints are named, so a blueprint says which chemistry the model uses. That works both ways:
  it can point to parts of the molecule involved in binding, or confirm that the selection is
  picking chemically sensible features. Embedding features have no chemical name (4.2.4).
- Other applications that could use a task-specific similarity: TODO, list (nearest-neighbour
  prediction, applicability domain, noise detection, clustering for splits, federated weighting,
  generative model inputs).

5.1.5  Beyond small molecules
- The selection needs only supervised learning with several vector representations of the same
  inputs: tabular data in general. One-hot sequence encodings work but are sparse; sequence
  embeddings are recommended over one-hot (onehot worked on antibodies).
- Applied here to antibodies (strong preliminary results; 5.7), reactions and nucleotides
  (proof of concept).
- Requirements for a dataset: enough samples, and base representations whose features carry
  importance. On tasks where every representation already scores poorly there were not enough
  useful features to improve on (brain dump; TODO: point to the specific runs).

5.2  Methods: the hybrid search (find_best_hybrid)
Every fact below is from src/ as of 2026-09-27. Line numbers are src/kirby/find_best_hybrid.py
unless marked hybrid.py.

5.2.1  From base representations to a hybrid
- Inputs: one matrix per base representation (one row per molecule), labels, groups for
  splitting, the metric (required), the task type, the budgets.
- Folds: group-aware, 5 by default (DEFAULT_N_FOLDS, 3664); cached per dataset.
- Per fold, on the fold's training rows only: an 80/20 group-aware internal split
  (internal_holdout_split, hybrid.py:5662). Feature importance per representation and each
  representation's standalone score are computed on the internal training part.
- Allocation: the budget is divided across representations in proportion to weights computed from
  their standalone scores (allocate_performance_weighted, hybrid.py:1602), by largest remainder.
  Scores are first rescaled so zero means "adds nothing over predicting the training mean"
  (hybrid.py:1767-1770). Each representation then contributes its top-k features by importance,
  stopping at the last feature with importance above zero, so a hybrid can be narrower than its
  budget (hybrid.py:436-441). The global-pool variant (allocate_weighted_global, hybrid.py:2195)
  lets each representation offer candidates in proportion to its weight and ranks the pooled
  candidates once.
- Each configuration is scored: the downstream model is fitted on the fold's training rows and
  scored on its held-out rows. Result: best configuration per (downstream model, budget).
- Reported comparison: nested_cv_evaluate (10397) runs the whole search inside each outer
  training fold and scores the winner and the single representations on the untouched outer fold.
- Fold-local preprocessing (median imputation, standardising on training-fold statistics only):
  existing prose thesis.tex:1362 (Ambroise2002, Vabalas2019).
- [DIAGRAM D5.1] The pipeline: several base representations -> per fold: importance per
  representation, standalone score per representation -> allocation splits the budget -> top
  features from each -> hybrid -> downstream model -> held-out score. Show the outer folds, the
  internal 80/20 split inside each training fold, and where the blueprint is saved. (Old TODO at
  thesis.tex:1432.)

5.2.2  The search space: every axis, as a product
State it the way Dablander does: axes, values, and the total count of configurations.
- Base representations: whatever the dataset's pool holds; 10 per dataset for 14 of the 16
  datasets (DATASET_REPS, tests/experiments/complexity/complexity_helpers.py:136).
  [CHECKED 2026-10-09: on the 25 live datasets, every pool holds 10 except ddg_combined with 8.
  Source: scratchpad/thesis_ch5_8/CH5_METHODS.md §3.]
- Representation count: 2, 3, 5 or the whole pool (DEFAULT_REP_COUNTS, 3028). Replaced the
  retired top_2/top_3/top_5 names.
- Importance methods: deeplift, lightgbm_gain, treeshap (3633). Eight are implemented
  (hybrid.py:4629); five others (permutation, kernelshap, xgboost_gain, integrated_gradients,
  gradientshap) are not searched.
- Allocation methods: performance_weighted (linear weights), power_2 (squared), softmax_T1,
  rank_exp_d0.3 (rank with exponential decay), wglobal_power2 (squared weights, global pool)
  (3642-3648). Greedy is cut (hybrid.py:461-485). Two-stage and stability-selection exist but are
  never dispatched.
- Internal evaluators (the model that scores candidate hybrids inside the search): match (the
  downstream model itself), rf, lightgbm, dnn (3663).
- Downstream models: rf, lightgbm, dnn, xgboost, svm (2679-2927). knn, gp, ngboost available by
  name only. mlp, ridge, catboost, cnn, bnn are off. mlp was cut 2026-09-18 and replaced by dnn.
- Budgets: 10, 25, 50, 100, 250, 500, 1000, 2000 features (tests/experiments/_shared.py:494).
  Each budget is a maximum. No per-model feature cap (MODEL_MAX_BUDGET is empty, 3728); svm and
  gp have caps on training rows (10,000 and 1,000).
- Objective: predictive (default), similarity, predictive_corr, struct_fid, or a weighted blend
  (1795-1860). Chosen per call, not swept inside one call.
- Augmentation strategy: none, uncapped, capped (5.2.4). A separate call each.
- Step size and two-phase refinement: read only by the greedy family, so they have no effect on
  any default search. The blueprint still stores step size.
- Total: configurations per (dataset, fold, objective, augmentation strategy) = count of each
  axis multiplied. FRAMEWORK_FINALIZATION.md §7.2 (dated document) gives 162 configurations at
  the fast tier and 3,060 at full over three budgets; dataset_policy.py:650-655 gives 816 against
  8,160 per condition. CHECK: recompute from the code for the text, and state the number of fits.
  [CHECKED 2026-10-09 from the code (_expected_config_count, find_best_hybrid.py:609-634).
  Configurations = (model, evaluator) pairs x allocation methods x importance methods x budgets x
  representation counts. Full default grid: 17 pairs x 5 x 3 x 8 x 4 = 8,160. Same as thesis.tex.
  The 816 is not reproduced by any live dataset's grid. The 162 and 3,060 were not recomputed.
  Under the current decided_grid.json a dataset's general search has 63 (logd) to 3,360
  (humanness_therapeutic) configurations per outer fold. Summed over 25 datasets and 5 outer
  folds, about 540,525 model fits for the general search alone. That fit count is an estimate
  from the loop structure, not a run log. Per-dataset table: scratchpad/thesis_ch5_8/
  CH5_METHODS.md §3.]
  [CHECKED 2026-10-09: the stored results did not all run this grid. The representation-count
  cuts landed on 7 Oct, 22:20-22:35. Most c02 files were written 6-7 Oct. 7 of 27 c02 files
  scored a different number of configurations than decided_grid.json now gives. Example: logd
  scored 252 per fold (counts 2, 3, 5, all); the current grid gives 63 (all only). The others:
  adme_hlm 630 against 315, abagym_EGFR 800 against 600, flab_thermo_tresanco 1,680 against
  1,260, cdr_flexibility 1,120 against 840, ginkgo_acsins
  1,451 against 1,512. thesis.tex tab:decided_grid prints the current counts. Source:
  CH5_METHODS.md §0 item 3, §3.]
  [DECIDE: which grid the text and tab:decided_grid state for these 7 datasets. Print the
  counts the stored results ran (text matches the numbers in 5.4 to 5.8), or the current grid
  (text matches the code; those 7 datasets' results need a re-run to match).]
- Table 5.1: one row per axis, its values, the fast-tier values, and where it is set.

5.2.3  Why each axis is swept
The brain dump asks for the reason for each axis, with results showing that datasets and models
disagree. Source: the f33 pruning record, tests/results/framework/f33_pruning/{flab_thermo_
tresanco, herg_ki, humanness_therapeutic, logd}.json: 5,100 configurations each,
[PATIENT DATA REMOVED 2026-10-09: the f33 record also holds one patient dataset. Its numbers are
cut below; any total over "five datasets" here includes it and needs recounting over four.]
5 folds, budgets 25/50/100/150/300, 10 representations, no augmentation. A cell is one
(downstream model, budget), 25 per dataset; the cell's winner is the best mean over 5 folds.
- Downstream model and budget: the best configuration is dnn at budget 50 on flab_thermo_
  tresanco (Spearman 0.6696), svm at 300 on herg_ki (RMSE 0.5589 pKi), lightgbm at 50 on
  humanness_therapeutic (AUROC 0.9723), dnn at 300 on logd (RMSE 0.4170). Within flab_thermo_tresanco, svm peaks at 300 features and
  dnn, lightgbm and rf at 50.
- Allocation method, cells won out of 25 (performance_weighted / power_2 / softmax_T1 /
  rank_exp_d0.3 / wglobal_power2): flab 3/0/2/6/14; herg_ki 3/6/1/2/13; humanness 4/7/3/6/5;
  logd 2/2/0/1/20.
- Representation count, cells won out of 25 (2 / 3 / 5 / all 10): flab 2/1/5/17; herg_ki
  0/2/13/10; humanness 7/14/4/0; logd 17/5/0/3. If only one count had been
  searched, the best score falls short: flab 0.0256 Spearman (count 2) or 0.0190 (all 10);
  herg_ki 0.0125 RMSE (count 2).
- Importance method, cells won out of 25 (lightgbm_gain / treeshap / deeplift): flab 15/10/0;
  herg_ki 14/11/0; humanness 12/12/1; logd 8/0/17.
- Internal evaluator, most frequent winner: dnn on flab (12 of 25), rf on herg_ki (14) and
  humanness (14), lightgbm on logd (10).
- Pooling all features into one selection against a quota per representation. Two records
  disagree:
  - c32 (/Volumes/seagate/kirby_cache/c32_allocation_cross/, 9 datasets, lightgbm/rf/catboost,
    budgets 500 and 1,000; 54 cells, 32 strategies each): the winner used a per-representation
    quota in 38 of 54 cells and the global pool in 16.
  - f33 (above): wglobal_power2, the only global method of five, wins 66 of 125 cells.
  - The two differ in three things at once: budgets (500/1,000 against 25-300), models (c32 has
    catboost, no dnn or svm), and how many global arms (16 of 32 against 1 of 5).
- [DIAGRAM D5.2] One small-multiple per axis (allocation, representation count, importance,
  evaluator), datasets on one side, cells won as bars. Shows that no setting wins on every
  dataset.

5.2.4  Augmentation: improving one dominant representation
- Use case: one representation already dominates and a full hybrid adds little. Keep it at full
  width and admit a few features from the others.
- How it works (hybrid.py:6036 onward): candidates from other representations are ranked by how
  well each correlates with what a model on the base representation gets wrong (out-of-fold
  residuals of a 50-tree forest; residual_correlation, the default, pinned 2026-09-10). The top 50
  candidates are offered. Blocks of 0, 5, 10, 20 or 40 are scored on the internal split; 0 wins
  ties. uncapped appends the block outside the budget; capped replaces the base's weakest features
  so the total stays within the budget.
- Wrapper find_best_single_rep_augmented (8046) exists but has no caller; f24 and c21 build the
  same arrangement themselves.
- No admission gate. One was built (a near-copy test and a shuffle test) and removed on
  2026-09-28 (920bbb8): the near-copy test did not improve scores and the shuffle test did
  not stop noise features.
- Decoys used to test admission (tests/experiments/_shared.py:4911): __gaussian__ (pure noise, as
  wide as all real donors) and __echo__ (64 base features with 5% jitter). The right count taken
  is 0.

5.2.5  Fast tier and adaptive pruning
- Fast tier (speed='fast'; lists 3020-3031): downstream models rf, lightgbm, dnn; importance
  lightgbm_gain only; allocation performance_weighted, power_2, rank_exp_d0.3; representation
  counts 3 and all. Same lists for every dataset. A fast run can be topped up to a full one.
  Evidence behind the lists (f33, from the code comments): dropping deeplift changed only logd
  (0.4905 to 0.4930 RMSE) and deeplift was about half the run time; adding dnn moved logd from
  0.4930 to 0.4273; adding svm moved herg_ki from 0.5776 to 0.5599; adding xgboost changed nothing;
  wglobal_power2 was 71.0% of logd's 14.45 hours.
- Adaptive pruning (DEFAULT_PRUNING_SCHEDULE, 3747-3848; off by default since 2026-09-13):
  - Every configuration is scored on folds 0 and 1 first.
  - At budget 100, within each (model, internal evaluator, representation count), the worst half
    of (allocation, importance) pairs are dropped at all higher budgets; at least one pair is kept.
  - Then votes, one axis at a time: a value that is among the best in fewer than a cut-off share
    of groups is dropped from the remaining folds (allocation: best 2, 20%; representation count:
    best 1, 10%; importance: best 2, 20%; internal evaluator: best 3, 10%). A vote can never delete
    every value of an axis.
  - On the f33 record it cut 30% of the hours (130.4 to 91.5 over five datasets, one of them
    patient data now removed; recount over four) and changed no (model, budget) best score.
    Voting after fold 0 alone was not safe on one dataset, since removed (patient data).
- CHECK: find_best_hybrid.py:3733-3738 says every pruning number is uncalibrated; the comments at
  3761-3845 say they were set from f33.
- [DIAGRAM D5.3] The pruning schedule as a funnel: all configurations on folds 0-1 -> checkpoint
  at budget 100 drops half the pairs -> per-axis votes -> survivors on folds 2-4.

5.2.6  Blueprints
- What is stored (Blueprint, 1397-1561): the representation pool, allocation method, importance
  method, budget, internal evaluator, fold count, seed, metric and direction, task type,
  objective, representation count, all augmentation settings; and as data, each representation's
  score, the augmentation features kept per fold, and each representation's feature count per
  fold.
- Rebuilding (apply_blueprint, 9570) re-runs the selection on the rows given, and needs labels.
  On different molecules it can select different features. To apply to unlabelled molecules,
  reuse the features a labelled fit chose (apply_feature_selection).
- Feature names (FEATURE_NAME_REGISTRY, importance/hybrid.py:259-296): pdv and other descriptor
  sets are named; ECFP bits resolve to an atom environment; MACCS keys are named; embeddings
  resolve to representation plus dimension index. atom_pair, avalon, graph_kernel, mordred and
  every learned molecular representation fall through to dimension indices. Decision for
  Adelaide: add names for mordred and atom_pair before the blueprint results (5.6).

5.3  Methods: experimental setup
5.3.1  Datasets
- Table 5.2 (Dablander's Table 5.1 as the model): one row per dataset, grouped by domain.
  Fields: source and citation; measurement and unit, and label transform; task type (class
  balance for classification); n after curation; metric and direction; splitting and groups;
  representation pool; one sentence of chemical or biological rationale. Curation (deduplication
  rules, counts before and after) goes in Appendix A.
- THE LIVE DATASETS, updated 2026-10-09 (author asked). 26 rows in the run, none cut. [2026-10-09: 25 after the patient dataset was removed.] Counts are
  the molecules (or antibodies, reactions) across the 5 outer folds, read from each
  dataset's representation-characterisation result file on 2026-10-09
  (/Volumes/seagate/kirby_results_2026-10-09/complexity/c01_rep_characterization/<ds>.json,
  _meta.outer_split). "Models" are the two the precompute decided for the dataset. Domain is
  tests/experiments/dataset_policy.py. Sources are the loader docstrings; anything not checked
  says so. The previous list ("The 16", 2026-09-27) still named datasets cut since and missed
  twelve live ones.
  BALANCE (author, 2026-10-09): 9 small-molecule rows, 15 antibody rows, 1 reaction set, 1
  gene-expression set. Counting rows would let antibodies dominate; see WHICH DATASETS ARE SHOWN.

  Small molecules (9):
    qm9         9,978, HOMO-LUMO gap, RMSE, regression; computed labels, the clean control;
                10,000 drawn (_shared.py:540; 5,000 was tried and reverted 2026-10-06);
                models dnn, lightgbm
    herg_ki     1,412, pKi, RMSE; ChEMBL CHEMBL240 Ki; models svm, dnn
    caco2       2,158, log10 efflux ratio, RMSE; OpenADMET ExpansionRx; models dnn, svm
    logd        5,028, LogD, RMSE; OpenADMET ExpansionRx; models dnn, svm
    adme_hlm    3,086, log HLM CLint, RMSE; Polaris ADME-Fang; models svm, dnn
    adme_rlm    3,053, log RLM CLint, RMSE; same source; models svm, dnn
    adme_mdr1   2,641, log MDR1-MDCK efflux ratio, RMSE; same source; models svm, dnn
    herg        7,772, binary hERG blocker, AUROC; FLuID (Lhasa) by the loader's default
                (herg.py:270), registry note says TDC/Lhasa; models lightgbm, xgboost
    herg_ic50   pIC50, one row per measurement with repeats kept (herg.py:503); noise-only row,
                runs only the repeat-measurement experiments; molecule count not checked
  Antibodies (15):
    flab_expression_koenig  4,275, expression enrichment, Spearman; FLAb, Koenig 2017 G6 scan;
                            models svm, dnn
    flab_binding_koenig     4,275, binding Kd, Spearman; FLAb, same G6 scan; models rf, svm
    flab_thermo_tresanco    672, melting temperature, Spearman; FLAb, Tresanco; models dnn, svm
    abagym_EGFR             659, deep mutational scan of cetuximab against EGFR, Spearman;
                            AbAgym; models svm, dnn
    abagym_VEGF             418, the same, G6 against VEGF; models gp, svm
    abagym_Ang2             397, the same, G6 against Ang2; models gp, svm
    humanness_therapeutic   564, binary human or not, AUROC; BioPhi; models xgboost, rf
    ddg_combined            429, binding free-energy change (DDG), Spearman; SKEMPI 2.0 +
                            AB-Bind, antibody-antigen complexes only; models svm, dnn
    cdr_flexibility         328, binary CDR flexibility, AUROC; ITsFlexible (OPIG), CDRH3 and
                            CDRL3 loops, grouped by 80% CDR3-loop identity; models xgboost,
                            lightgbm
    abcov_kd                141, binding -log Kd against SARS-CoV-2, Spearman; Ab-CoV (Rawat
                            2022) via FLAb; models lightgbm, xgboost
    ginkgo_colloidal        242, SMAC (colloidal stability), Spearman; Ginkgo GDPa1, Ginkgo's
                            own 5 cluster folds; models dnn, svm
    ginkgo_hic              242, HIC retention, Spearman; same antibodies; models gp, dnn
    ginkgo_acsins           242, AC-SINS pH 7.4, Spearman; same antibodies; models lightgbm, gp
    ginkgo_titer            239, expression titer, Spearman; same antibodies; models dnn, rf
    ginkgo_pr_cho           197, polyreactivity (PR_CHO), Spearman; same antibodies; models gp,
                            dnn
  Reactions (1):
    buchwald_hartwig  3,955 reactions, yield %, RMSE; Ahneman et al. 2018; grouped by additive;
                      models rf, lightgbm
  Cut, not in the thesis (plan.CUT_DATASETS): nt_H3K27me3, nt_H3K9ac,
  nt_H3K9me3, nt_H3K4me3, nt_promoter_tata, gb_human_enhancers_cohn, bindingdb_kd, kiba,
  flab_expression_adams, flab_binding_adams, suzuki, lhasa_herg, bh_doyle_plates,
  tz_her2_binary, tz_her2_multiclass.
  NOT CHECKED in this update: grouping for the abagym, flab, abcov, humanness and ddg rows (the
  CHECK list below covers the old rows); the ginkgo "same 242 antibodies" claim beyond equal
  counts; herg_ic50's count.
- Classification datasets use StratifiedGroupKFold; regression uses GroupKFold; no random-split
  fallback (tests/experiments/_shared.py:2034-2103).
- CHECK, disagreements to settle in the text:
  - Chapter 3 gives LogD 5,039, Caco-2 2,161, hERG 1,415; the KIRBy loaders give 5,028, 2,158,
    1,412 (InChIKey deduplication, probably; unverified). Appendix A must say which curation each
    chapter used.
  - thesis.tex:1474 says R2 for reaction yield; the code uses RMSE.
  - thesis.tex:1480 says "5-fold GroupKFold throughout"; classification uses StratifiedGroupKFold.
  - thesis.tex:1484 TODO says reactions by template, antibodies by VH identity;
    the code groups reactions by additive, and antibodies by heavy and
    light chain together at 95% identity.
- Best single representation per dataset and model: the f26 files carry some
  (tests/experiments/results/f26_redundancy_penalty_*.json) but used pre-pruning pools. Fill
  Table 5.2's last column from the server runs. TODO(prelim).
  [CHECKED 2026-10-09: fill it from c02 instead, the single representation each fold's search
  ranked first on training molecules (most frequent pick over 5 folds). qm9 mordred (both);
  herg_ki svm mhg_gnn, dnn grover (3 of 5); caco2 dnn mhg_gnn (3 of 5), svm pdv (4 of 5); logd
  dnn mordred (4 of 5), svm mhg_gnn; adme_hlm and adme_rlm svm pdv, dnn mhg_gnn (4 of 5);
  adme_mdr1 pdv (both); herg mhg_gnn (3 of 5, both). Antibodies often pick a different
  representation on each fold. Per-fold lists: scratchpad/thesis_ch5_8/CH5_RESULTS.md §2.]

5.3.2  Representations by domain
- Not a list: for each domain, what the representation families encode and why they were
  chosen, then the pool. Each pool of 10 came from a pruning step recorded in REP_PRUNING_LOG.md
  (2026-07-18 to 08-23; a dated document).
  - Small molecules: hashed substructure fingerprints (ecfp4, avalon, atom_pair, top_torsion),
    predefined keys (maccs), graph-kernel histograms (graph_kernel), computed descriptors (pdv,
    mordred), pretrained graph encoders (grover, mhg_gnn, graphmvp), pretrained SMILES and SELFIES
    language models (molformer, smited, chembert, chemberta, selformer), mol2vec; 3D shape and
    pharmacophore features (usr, usrcat, pharmacophore3d, coulomb_matrix) as augmentation only.
  - Antibodies: hand-built encodings (onehot, blosum, aac), protein language models (esm2, esmc,
    esm3, prot_t5, saprot), antibody language models (antiberty, ablang2, abroberta, igt5),
    variants that pool a model differently (esm2_midlayers, esm2_early, esm2_per_cdr, *_cdr), and
    mutation features.
  - Reactions: condensed graph of reaction (cgr_morgan), reaction language models (rxnfp,
    reactiont5), component identity (onehot_components), and any molecular representation applied
    to reactants, products or their difference.
  - Nucleotides: k-mer counts, tetranucleotide frequency, DNA foundation models (dnabert2,
    nt_v2_250m, nt_v2_500m, hyenadna, grover_dna, mistral_dna).
- How alike the representations are: f26 distance agreement (4.2.5) and the pruning step's
  correlation of out-of-fold predictions (REP_PRUNING_LOG.md; logd's top 9 are all mutually
  above 0.90; every antibody *_cdr representation predicts
  identically to its base). The brain dump asks for these here.
- CHECK: chembert (256 features, jonghyunlee/ChemBERT_ChEMBL_pretrained) is not chemberta (384,
  DeepChem/ChemBERTa-77M-MTR). chembert is in 4 pools (herg_ki, logd, qm9, adme_hlm); its cache
  has never been checked against its builder.
- CHECK: gene_geneformer and gene_geneformer_v2 have the same 1,152 features and distance
  correlation 0.940; the v2 builder may have fallen back to v1 on the server.
- [NEW 2026-10-09, author agreed] Why the features depend on the model, from each representation
  scored alone. The best single representation is the same under both decided models on 7 of 8
  small-molecule datasets, but on only 5 of 15 antibody datasets. [file pointer: c01. Counts from
  the page's yes/no table, rebuilt 9 Oct; not yet re-read from the raw files.]
- Table 5.3: one row per representation, domain, what it encodes, number of features, source
  model or citation. Pretraining corpora and citations are missing in code for most learned
  models; TODO(lit).

5.3.3  Models, metrics and error bars
- Downstream models and their settings (a hyperparameter table, Dablander Tables 4.1/5.2 as the
  model); internal evaluators and their settings (rf evaluator: 50 trees, depth 10; rf_exact:
  the downstream forest's own 100 trees, max_features sqrt; hybrid.py:2851-2865).
- Metrics per dataset as in 5.3.1. Results are the per-fold scores and their spread. No
  significance tests, within or across datasets, and no confidence intervals (decided
  2026-09-27). thesis.tex:1492 (Nadeau-Bengio, Wilcoxon, Friedman/Nemenyi) is to be deleted
  when the thesis is restructured.
- Compute: hours per dataset for the full search, and the hardware (Dablander gives both).
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH5_METHODS.md §1] gp is a decided model on five
  live datasets: abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho
  (decided_grid.json). thesis.tex says "all but the Gaussian process are used". That is now wrong.
- [NEW 2026-10-09, defects in the stored results, from CH5_RESULTS.md §0 items 10-12 and
  CH5_METHODS.md §0 items 9-11]
  - dnn predictions are clipped to the training-label range only since 8 Oct 13:04 (c18ee30).
    Most stored c02 files were written before that. Which code each later file ran is unverified.
  - gp numbers on the five gp datasets are to be replaced (FAULTS.md A2). The gp length-scale
    fixes landed 8 Oct 11:03 and 15:59. The per-fold searches that chose those hybrids ran before.
    c23's svm near-family evaluator is gp, so every c23 svm comparison used the old gp.
  - Held-out features are clipped to the training range in c01, c02, c05, c07, c08 and c18, and
    not in c11, c17, c19, c21-c24, c33, OPIG or federated. thesis.tex states no rule.
  - The code limits ten experiments to the first 3 outer folds, with 13 datasets exempt. The
    limit arrived 6-7 Oct, so most stored files have 5 folds. thesis.tex says four experiments.
  [DECIDE: re-run the pre-fix dnn and gp results before the tables are drafted, or report them
  with the date and the fix named. Re-running changes the numbers in 5.4-5.5 for every dnn row
  and the five gp datasets; not re-running means a caveat beside each.]

5.3.4  Complexity experiments (methods, final)
Each is a script in tests/experiments/complexity/. Stated as coded.
- c11, hybrid against ensembles. Hybrid: a find_best_hybrid search on each outer fold's
  training rows. Ensembles: one model per representation (or each representation's own best
  model), membership top 3 or all, pooled by mean, median, trimmed, IQR- or MAD-filtered,
  weighted, or top-3 mean, chosen on an inner 3-fold split. Controls: best single
  representation, pca_matched (every representation concatenated, then PCA to the hybrid's
  width), oracle_single (reads the held-out fold; an upper bound only). Reported comparison: for
  each held-out fold, the budget with the best mean on the other folds, so the hybrid and the best
  single representation pay the same selection cost (fold_reselection, _shared.py:1734).
  [CHECKED 2026-10-09 against the code: was fold reselection; now no fold reselection (removed
  2026-10-05). Each outer fold runs one search on its training molecules; the 2 best single
  representations and 2 best hybrids per model are picked there and scored once held out.
  pca_matched is PCA to each decided budget, not to the hybrid's width. oracle_single is not an
  upper bound: an ensemble can beat it. thesis.tex already says all three. Source:
  scratchpad/thesis_ch5_8/CH5_METHODS.md §2 c11.]
- c05, learning curves. Subsamples of 10, 25, 50, 75 and 100% of the molecules, drawn as whole
  scaffold groups, 3 seeds. A search on each subsample, budgets capped at the training rows.
  Delta = hybrid minus best single representation.
- c17, hybrid against feature reduction at the same number of features. Five arms: the hybrid
  (the stored rf recipe at that budget, rebuilt per fold), PCA, mRMR (capped at 200 features),
  RFE (50-tree, depth-5 forest), random. All fitted on the fold's training rows; one downstream
  model, a 100-tree random forest. Regression datasets only. CHECK: the recipe itself is chosen on
  all rows (recipe_selected_on: 'all rows').
  [CHECKED 2026-10-09 against the code: was one 100-tree forest and a recipe chosen on all rows;
  now the dataset's two decided models and a per-fold search, so the CHECK is resolved. Per fold,
  budget and model: the 2 best hybrids at that budget against PCA, mRMR, RFE and random. Source:
  CH5_METHODS.md §2 c17.]
  [CHECKED 2026-10-09: mRMR stops at 200 features. Every small-molecule decided budget starts at
  250 or 500. So mRMR never ran on qm9, herg_ki, caco2, logd, adme_mdr1 or buchwald_hartwig. It
  ran on 4 of 15 regression datasets: abcov_kd, ddg_combined, flab_thermo_tresanco,
  ginkgo_acsins. Source: CH5_METHODS.md §0 item 8; CH5_RESULTS.md §0 item 3.]
  [CHECKED 2026-10-09: c17's PCA is not c02's or c11's PCA. c02 and c11 fit the model on the
  principal components directly. c17 rescales every component to unit variance first. So the two
  differ in one way. Size: hERG Ki svm at 2,000 features, c17's PCA 0.8961 RMSE, c11's 0.5806.
  c17's PCA gets worse as components are added on every small-molecule table read; c02's and
  c11's get better. Source: CH5_RESULTS.md §0 item 2.]
  [DECIDE: which PCA the chapter uses. Keep c17's rescaled PCA (5.5 "beats PCA" stands, but
  against a weaker PCA than 5.4's), or switch c17 to fit on unscaled components like c02 and c11
  (c17 needs a re-run; 5.5's verdict may change on LogD and Caco-2 with dnn).]
- c23, does the internal evaluator need to match the downstream model? nested_cv_evaluate; each
  outer fold searches with one evaluator. Conditions: same family, near family (rf with
  lightgbm, svm with gp, dnn with a smaller dnn), far family, and the downstream rf's exact
  settings (rf_exact). Deltas paired per outer fold. Widths must be matched.
  [CHECKED 2026-10-09 against the code: there is no rf_exact condition in c23. Conditions now: A
  matched; B near family (rf-lightgbm, lightgbm-rf, xgboost-lightgbm, svm-gp, gp-svm, dnn by a
  64-32 dnn); C far family; D dnn scored by a wider 512-256-128 dnn; E the dataset's decided
  evaluator list. Each condition picks its own budget, so widths differ; they are stored, not
  matched. thesis.tex matches the code. Source: CH5_METHODS.md §2 c23.]
- c19, importance used during training. Each model's own recipe at budget 100. Conditions:
  unweighted; features multiplied by the square root of importance, normalised to mean 1; each
  molecule weighted by how far its label is from its 10 neighbours' mean; both; features above
  the median importance only. Models rf, lightgbm, xgboost.
  [CHECKED 2026-10-09 against the code: was each model's recipe at budget 100 with rf, lightgbm
  and xgboost; now the dataset's two decided models, and per fold the 2 best single
  representations and the 2 best hybrids, each under the five conditions. Molecule weighting is
  not run for gp or on nominal labels. Source: CH5_METHODS.md §2 c19.]
  CITATIONS (added by Claude 2026-10-09 from the code; not reviewed by the author). Weighting a
  molecule by its neighbour-label gap is the reliability weight of n02 (n02:255-362, weight
  1/(1 + 3 x gap), via weights_from_scores, noise_estimation.py:2178).
  [CHECKED 2026-10-09: c19 does not use n02's weight. c19 computes 1 - |label - mean label of 10
  nearest neighbours| / (label range), clipped to 0.1-1 (c19_importance_weighted_training.py:
  100-126). Neighbours are found in the importance-weighted, scaled space. thesis.tex states
  c19's formula correctly. Source: CH5_METHODS.md §0 item 5.] The code cites no source,
  and none was found in the 2026-10-08 search. Yours if used. Multiplying features by importance
  before a distance is feature-weighted kNN; Wettschereck, Aha and Mohri 1997 is the usual
  reference, not yet verified or in refs.bib.
- c21, augmenting one representation, with decoys. One base representation at full width,
  candidates ranked by residual correlation from a pool of 50, blocks of 5/10/20/40, decoys
  planted. Counts how many of each kind are admitted.
  [CHECKED 2026-10-09 against the code and results: the pool of 50 and blocks of 5/10/20/40
  describe the engine's admission step, not what c21 varies. c21 conditions: baseline; curated
  augmentation sets; top 9 features of every other pool representation; both; matched random;
  and the multi-representation hybrid. The extra-feature conditions ran only on ginkgo_colloidal
  and ginkgo_hic (uncapped). The other 21 files hold the baseline and the hybrid only. So c21
  does not test admission on any other dataset. Source: CH5_METHODS.md §2 c21; CH5_RESULTS.md
  §3 5.5a.]
- c33, does the search overfit? For each budget, the recipe is scored on untouched outer folds;
  bias = inner-search score minus held-out score (positive means the search over-promised).
  Budgets up to 8,000 and ratios 0.5-2.0 of the training rows.
  [CHECKED 2026-10-09: same. Budgets 10-2,000, plus 4,000 and 8,000, plus 0.5 to 2 times the
  smallest outer training set; herg_ki has 16 budgets. Two decided models. 5 folds on 15 of 16
  files, 3 on humanness_therapeutic. Source: CH5_METHODS.md §2 c33.]
- c24, stability: Nogueira's stability index of the selected features across folds.
  [CHECKED 2026-10-09: incomplete. c24's main measurement is not stability. The own-fold cell is
  the standard per-fold search. Each cross cell is a fresh search on the molecules outside both
  the target fold and one source fold, so it trains on one fold fewer. Nothing is transferred.
  Stability is a second output. 18 files, 5 folds each. Source: CH5_METHODS.md §2 c24.]
- [NEW 2026-10-09, from CH5_METHODS.md §2.0] What every complexity experiment shares, as coded:
  outer folds from nested_cv_outer_folds, seed 42, 5 in full mode. One search per outer fold on
  that fold's training molecules, grid from decided_grid.json, pruning off. 5 inner folds, each
  with an 80/20 group-aware split for importance and standalone scores. Per decided model, the 2
  best single representations and 2 best hybrids are picked on inner scores, rebuilt on the outer
  training molecules and scored once held out. No fold reselection except c16 and c12. No
  augmentation on the general search.

5.3.5  Applied settings: OPIG and federated (methods, final)
- OPIG experiments (tests/experiments/opig_*.py). Each needs a paragraph of background on the task.
  - opig_blueprint_transfer: can a hybrid found on a large dataset be reused on a small one?
    Source logd (5,028 molecules), target herg_ki (1,412). Pool: representations in both
    datasets' pools with the same width (6: atom_pair, avalon, ecfp4, maccs, mordred, pdv). Four
    arms per target fold: best single representation (searched on that fold's training rows),
    native hybrid (searched on that fold), recipe transfer (logd's allocation, importance,
    budget and evaluator; features re-selected on herg_ki), exact-feature transfer (logd's feature
    indices applied unchanged). 5 group folds, RMSE, 5 downstream models.
    [CHECKED 2026-10-09 against the result file: was 6 shared representations and 5 models; now
    7 (avalon, chembert, ecfp4, grover, mhg_gnn, mordred, smited) and 2 models (svm, dnn).
    thesis.tex already says seven. chembert's cache was never checked against its builder
    (unverified). The source recipes come from one search on every LogD molecule. Source:
    scratchpad/thesis_ch5_8/CH5_METHODS.md §0 item 6.]
  - opig_scaffold_hybrid: the code's question is whether two mutation-difference representations
    (dphyschem, 5 features; dphyschem_pos, 1,140) earn a place in the pool on antibody deep
    mutational scans (flab_expression_koenig and three other FLAb sets). Evaluated with nested
    cross-validation.
  - Other OPIG experiments, one line each in the text: activity-cliff QSAR (replicates Dablander
    et al. 2023 on D2, factor Xa, Mpro), CGR reaction representations, Ginkgo developability,
    Graphinity DDG, humanness, trastuzumab enrichment, scaffold similarity, the shared-vector
    census and remedy (4.2.3). opig_cdr_flexibility and opig_kingsmith_component_splits were cut
    on 2026-09-23.
- Federated, with Lhasa (tests/experiments/federated/).
  - The standard process, as Lhasa's FLuID implements it (~/repos/lhasa/FLuID_POC/preprocessing/
    FLuID.py): sites are k-means clusters of one dataset's fingerprints, with a share of molecules
    reassigned at random; one teacher model per site trained on its own molecules; every teacher
    labels a shared public set, each prediction weighted by the molecule's distance to that
    teacher's data; the weighted labels train one student model. Only predictions on public
    molecules cross between sites. TODO(lit): McMahan et al. 2017 (FedAvg), Heyndrickx et al. 2023
    (MELLODDY), and Bassani2023 and Hanser2023 (Lhasa, both in refs.bib, not yet cited).
  - KIRBy's federated feature selection (federated_feature_selection, federated_helpers.py:1328):
    each site runs create_hybrid on its own molecules and reports feature indices, importances and
    one validation score; a coordinator splits the total budget across sites in proportion to
    score and drops features already taken. No molecule or raw feature crosses.
  - fed03 Part A: three real hERG sources as sites (ChEMBL herg_ki, pKi >= 6.0 as blocker; Lhasa
    training set, 7,772; Lhasa test set, 4,383), deduplicated across sites by InChIKey keeping the
    ChEMBL copy. 5 scaffold-group folds over the pooled molecules. Search metric AUROC; reported
    metric MCC. Arms: best single site; FLuID similarity ensemble; FLuID quality ensemble;
    federated selection then one model on pooled molecules (C4); one model per site on the
    fingerprint, predictions combined by FLuID's distance weighting (C8); the same on the selected
    features (C9, added 2026-09-24, the arm that keeps molecules at their sites); naive pool;
    hybrid on ecfp4 alone; one teacher per representation.
    [CHECKED 2026-10-09: Part A has no result and cannot run. It runs only on lhasa_herg, cut on
    8 Oct (plan.py:74-80). fed03's results are Part B only, on caco2 and logd: three sites, one
    per representation family (structural, physicochemical, learned). Arms: C1 best family; C2
    one selection over all families; C3 one selection per family at an even share of the budget.
    Budgets 500-2,000, dnn and svm, 3 seeds, RMSE. The budgets 100-2,000 and LightGBM/XGBoost in
    thesis.tex are lhasa_herg's. Source: CH5_METHODS.md §0 item 1, §2 fed03; CH5_RESULTS.md §0
    item 6.]
    [DECIDE: what 5.3.5 describes for fed03. Replace Part A with Part B (methods then match the
    5.8 results; the hERG-sources design, the C8/C9 CHECK and D5.4's bottom row go), or keep Part
    A as a design with no result (needs lhasa_herg back in the roster and a run).]
  - CHECK: C8 fits on one 2,048-bit Morgan fingerprint; C9 fits on 100 features drawn from 6
    representations. So C8 and C9 differ in two ways at once, width and source. The "Promising
    experiments" file says C8 uses "all 6,000 numbers per molecule"; the code says 2,048.
    [CHECKED 2026-10-09: C8 and C9 are Part A arms, so they have no results. See the DECIDE
    above.]
  - The Lhasa data is not redistributable and the comparison needs Lhasa's FLuID code; the
    simulation uses the training molecules as the public set (a stated compromise).
  - [DIAGRAM D5.4] Two rows. Top: standard FLuID (sites -> teachers -> public set labelled ->
    student). Bottom: fed03 Part A C9 (sites -> each site's hybrid -> coordinator splits the
    budget -> each site's model on the shared feature list -> distance-weighted predictions). Mark
    what crosses between sites in each.
    [CHECKED 2026-10-09: the bottom row draws Part A, which has no result. Source: CH5_METHODS.md
    §0 item 1.]
  - [NEW 2026-10-09, from CH5_METHODS.md §0 item 2] fed05 (antibody sites) cannot run. Its caco2
    and logd files are skip records that say "computed once in the lhasa_herg job". thesis.tex
    still lists it as the fourth separate-site experiment. The author already cut it (5.8).

5.4  Results: when the hybrid helps
Every number here is replaced by the full runs. The verdicts come from the per-experiment pages of
the 2026-10-08 analysis (/Volumes/seagate/claude_analysis_2026-10-08/answers/), fold 0 for the
noise work and up to 5 folds for the complexity work.
- The hybrid against the best single representation, per dataset and downstream model. It wins on
  the small-molecule datasets for both models: Caco-2, LogD, hERG Ki, hERG, and the three ADME
  sets. On LogD the network goes 0.4497 to 0.4089 RMSE and the support vector regressor 0.4930 to
  0.4302. It loses for both models on nanobody thermostability, SMAC colloidal stability and HIC
  retention. [FIGURE F1. file pointer: c02, c22]
  [CHECKED 2026-10-09 against the 9 Oct results: same, 5 folds. QM9 also wins for both models:
  dnn 0.3619 to 0.3155, lightgbm 0.3720 to 0.3463 RMSE. Of the six losses, three are under 0.002
  Spearman: ginkgo_colloidal svm 0.3485 to 0.3480, flab_thermo_tresanco svm 0.6901 to 0.6899,
  ginkgo_hic dnn 0.5311 to 0.5293. Fold SD there is 0.05 to 0.16, so these are no gain rather than
  a loss. The hybrid also loses with one model on abcov_kd lightgbm, ginkgo_pr_cho gp,
  humanness_therapeutic rf, cdr_flexibility lightgbm and buchwald_hartwig rf. Use c02 for F1, not
  c22: c22 has only partial files for logd (2 folds), adme_rlm and flab_expression_koenig.
  flab_binding_koenig has a c02 partial file only (fold 0). Source: scratchpad/thesis_ch5_8/
  CH5_RESULTS.md §2, §3 5.4 bullet 1.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH5_RESULTS.md §2] THE CORE TABLE. One row per
  dataset and decided model: best single representation and best hybrid, each picked on the
  fold's training molecules, scored once on the held-out fold, 5 folds, per-fold scores and SD.
  The full table is CH5_RESULTS.md §2. By domain:
  - Small molecules, 8 datasets, 16 dataset-and-model pairs: the hybrid's mean is better in all
    16. On the seven RMSE datasets it is better on 70 of 70 folds. RMSE gains run from Caco-2 svm
    0.3036 to 0.2862 up to LogD svm 0.4930 to 0.4302. herg: 4 of 5 folds per model, lightgbm
    0.8555 to 0.8666 AUROC, xgboost 0.8532 to 0.8644.
  - Antibodies, 14 datasets, 28 pairs: the hybrid's mean is better in 18 and worse in 10. Fold
    wins are often 2 or 3 of 5. Fold SDs reach 0.30 Spearman (abagym_Ang2 gp single 0.3022).
  - Reaction, buchwald_hartwig: rf 14.4056 to 15.2383 RMSE in yield % (worse, 2 of 5 folds);
    lightgbm 16.1679 to 15.6211 (better, 4 of 5). Fold SD is about 3.0 to 3.3, larger than either
    difference.
  [NEW 2026-10-09, from CH5_RESULTS.md §2 "The antibody exception" and scratchpad/thesis_ch5_8/
  PATTERNS.md §2.3] THE ANTIBODY FINDING. Only one antibody pair has a gain larger than its fold SD
  and better on every fold: flab_expression_koenig svm, 0.7029 to 0.7729 Spearman, 5 of 5 folds,
  single SD 0.0268. The other large antibody gains come with fold SDs of 0.12 to 0.27, often
  carried by one fold. Example: ddg_combined svm 0.0819 to 0.2178, where fold 3 alone contributes
  +0.6992. Against the single representation that scored best on the held-out molecules (an upper
  bound, picked on the answer), the hybrid wins all 16 small-molecule pairs and 6 of 28 antibody
  pairs. The training molecules picked the held-out-best single on every fold for 8 of 16
  small-molecule pairs and 4 of 28 antibody pairs. So on antibodies much of "hybrid beats single"
  is the hybrid beating a poorly chosen single representation.
  [DECIDE: how 5.4 names the antibody exception. Say the hybrid is consistent on small molecules
  and spread in both directions on antibodies, naming flab_expression_koenig svm as the one clean
  antibody gain; or keep "unusually well on antibodies", which the results support only for the
  size of a few means, not for consistency (CH8.md §3.1 has the same point for 8.2).]
- Splitting the budget over the top two or three representations against taking it all from the
  top one. hERG Ki with the support vector regressor at 500 features: 0.6006 to 0.5672 RMSE.
  [file pointer: c18]
  [CHECKED 2026-10-09: same, 5 folds. Not in the outline: in the same cell an even split over all
  10 representations gives 0.5569, and the search's own allocation 0.5593. Both are better than
  the top 3 (0.5672). On herg_ki the search's allocation loses to the even split with both
  models. On adme_mdr1, adme_rlm, herg and logd it wins with both. c18 has no adme_hlm file;
  flab_expression_koenig (2 folds) and flab_binding_koenig (3 folds) are partial. Source:
  CH5_RESULTS.md §3 5.4 bullet 2; CH5_METHODS.md §2 c18.]
- The hybrid against the two ensembles. It beats both on human microsomal clearance, on hERG Ki
  with the support vector regressor. Both ensembles beat it on
  Angiopoietin-2 binding, anti-coronavirus affinity and CHO polyreactivity. [TABLE, not a figure.
  file pointer: c11]
  [CHECKED 2026-10-09 against the 9 Oct results, 5 folds. Human microsomal clearance: NOT FOUND,
  c11 has no adme_hlm file (waiting on the dnn re-run, FAULTS A1). Angiopoietin-2: NOT FOUND, no
  abagym_Ang2 file. The rest same. hERG Ki svm: hybrid 0.5596, ensembles 0.5819 and 0.5828 RMSE.
  abcov_kd lightgbm: hybrid 0.1654, ensembles 0.4186 and 0.4532 Spearman. ginkgo_pr_cho gp: hybrid
  0.4434, ensembles 0.4853 and 0.5078 (gp to be replaced). Now also: both ensembles beat the
  hybrid with both models on ginkgo_colloidal, ginkgo_titer and herg (lightgbm 0.8719 and 0.8724
  against 0.8666 AUROC). c11 is missing adme_hlm, adme_rlm, flab_binding_koenig, abagym_Ang2 and
  ginkgo_hic. Source: CH5_RESULTS.md §2 "Domain balance", §3 5.4 bullet 3.]
  [DECIDE: drop human microsomal clearance and Angiopoietin-2 from this bullet, or keep them
  marked pending until c11 runs on adme_hlm and abagym_Ang2.]
  RESTATE BY DOMAIN (author agreed 2026-10-09). The ensemble result is conditional on domain, not
  three named exceptions. One model per representation, averaged, beats the hybrid on 10 of 12
  antibody datasets with the first model and 8 of 12 with the second. On small molecules it beats
  the hybrid on 1 of 6 with the first model and 4 of 6 with the second. [Counts from the c11 page's
  yes/no table, rebuilt 9 Oct; not yet re-read from the raw files.]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH5_RESULTS.md §0 item 1, §2] Matched-width PCA
  beats the hybrid with dnn on LogD and Caco-2. All representations joined, then PCA to the
  budget, model fitted on the components. Two experiments agree, 5 folds. LogD dnn: PCA at 1,000
  and 2,000 components 0.3923 and 0.3920 RMSE; hybrid 0.4089. Caco-2 dnn: PCA at 1,000 components
  0.2727; hybrid 0.2775. adme_mdr1 dnn too: c02 PCA at 1,000 components 0.4411, hybrid 0.4438.
  With svm the hybrid beats PCA on all three. [file pointer: c02 pca_baselines, c11
  pca_matched]
  HOW THE PCA COMPARISON IS COMPUTED (agreed 2026-10-09). Per outer fold, the hybrid the search
  picked on that fold's training molecules against PCA at that hybrid's own budget on the same
  fold, from c02's pca_baseline_fold_scores. Nothing is picked on the held-out fold, and no
  re-run is needed. c11's pca_matched fold scores are a cross-check on the same folds: the svm
  numbers agree to about 0.001, the dnn ones differ by run-to-run network spread. Limit: where the
  budget exceeds the training molecules, PCA is capped at the number of training molecules
  (ginkgo_titer gives the same PCA score at every budget), so the widths are not matched there.
  Small molecules read this way (5 folds, own metric): the hybrid beats PCA on every fold with the
  svm, lightgbm and xgboost models on every dataset; with dnn it beats PCA on qm9 (0.3155 against
  0.3321 RMSE) and loses on logd (0.4089 against 0.3920, 0 of 5 folds); caco2, herg_ki, adme_mdr1
  and adme_rlm with dnn are within a fold or two either way.
- The hybrid inside an easy and a hard half of one dataset. LogD's hard half by scaffold
  frequency, network: 0.6698 single against 0.5735 RMSE on the half's own search. [file pointer:
  c07]
  [CHECKED 2026-10-09: same, 5 folds. The text must say c07 searches 2,000 of LogD's 5,028
  molecules, cut by whole scaffold groups, seed 42. The hard half is 649 of those 2,000. On the
  same half the hybrid searched on all 2,000 molecules gives 0.5830. c07 has no qm9 or adme_hlm
  file; the five ginkgo sets are skipped (5 groups). Source: CH5_RESULTS.md §3 5.4 bullet 4;
  CH5_METHODS.md §2 c07.]
- Where the search overfits. Hybrids wider than the number of training molecules are better on
  expression titer and SMAC colloidal stability for both models, and worse on hERG Ki and LogD for
  both. Searching with one fold fewer of training molecules is generally worse: hERG Ki with the
  support vector regressor goes 0.5596 to 0.5931 RMSE. [file pointer: c33, c24]
  [CHECKED 2026-10-09: same. ginkgo_titer and ginkgo_colloidal yes for both models; herg_ki and
  logd no for both. Now also yes for both on ginkgo_acsins and abcov_kd. The c33 comparison takes
  the best held-out mean among budgets on each side, so both sides are picked on held-out scores.
  Example: logd dnn 0.4066 RMSE at 3,016 features against 0.4157 at 8,044. The one-fold-fewer
  number is c24's, 5 target folds x 4 source folds; LogD dnn goes 0.4174 to 0.4400. The own and
  cross sides also pick different budgets, so they differ in two ways at once. c33 is missing qm9,
  the three ADME sets, both flab_koenig sets, abagym_VEGF, abagym_Ang2 and ginkgo_hic. Source:
  CH5_RESULTS.md §3 5.4 bullet 5.]
  [DECIDE: whether c33's wide-hybrid verdict is reported with the held-out budget pick and a
  caveat, or re-read with the budget each fold picked on inner scores (c33 stores that pick
  beside the held-out-best budget; no re-run, but the verdict is not yet known).]
- CUT: whether the lead grows as the dataset shrinks (c05). Removed from the chapter 2026-10-08.

5.5  Results: why the hybrid helps
- Against principal components, minimum-redundancy maximum-relevance selection, recursive feature
  elimination and random features at the same number of features. It beats all three on human and
  rat microsomal clearance, Caco-2, hERG Ki and LogD with both models. LogD at 1,000 features,
  network: 0.4099 against 0.4146 RMSE for recursive elimination. It beats none of the three on
  Angiopoietin-2 binding, Buchwald-Hartwig yield, nanobody thermostability or CHO
  polyreactivity. [FIGURE F1 variant or TABLE. file pointer: c17, c15, c09]
  [STALE 2026-10-09: the HLM and RLM clearance answers are now n/a until their network results
  are re-run; the rebuilt page adds QM9, where the hybrid beats PCA and recursive elimination
  (mRMR did not run on QM9). Re-read c17 before drafting.]
  [CHECKED 2026-10-09 against the 9 Oct results. Human and rat microsomal clearance: NOT FOUND,
  c17 has no adme_hlm or adme_rlm file. Angiopoietin-2: NOT FOUND, no abagym_Ang2 file. Caco-2,
  hERG Ki and LogD with both models: same. Buchwald-Hartwig, nanobody thermostability, CHO
  polyreactivity: same (none). LogD 1,000 features, dnn: same, 0.4099 against 0.4146; PCA 0.4627;
  the hybrid beat the best other method on 2 of 5 folds there. Now also yes for both models on
  qm9 and adme_mdr1. Three limits on "beats all three":
  - mRMR never ran at a small-molecule budget (capped at 200 features), so on small molecules it
    is "beats recursive elimination, random and c17's PCA".
  - c17's PCA rescales each component and is weaker than c02's and c11's (see 5.3.4 and the
    matched-width PCA bullet in 5.4).
  DECIDED (author, 2026-10-09; replaces the earlier "re-run c17" decision): c17 is NOT re-run
  (it would cost about 49 job-hours over 15 datasets, 18.2 hours wall clock for
  flab_expression_koenig alone). c17's PCA arm is left out of the thesis. This bullet reports the
  hybrid against mRMR, recursive elimination and random features only. The comparison with PCA at
  the same width comes from c02 and c11 alone (5.4). On small molecules that leaves recursive
  elimination and random features, because mRMR never ran there.
  - On hERG Ki dnn, recursive elimination beats the hybrid at 1,000 features (0.5794 against
    0.6112 RMSE). Random features beat it at 500, 1,000 and 2,000 (0.5656, 0.5705, 0.5813 against
    0.5751, 0.6112, 0.6200).
  c15 does not compare against PCA, mRMR or RFE, so its pointer here is wrong. c09 compares
  against PCA and random on effective rank only. c17 skips the four classification datasets.
  Source: scratchpad/thesis_ch5_8/CH5_RESULTS.md §0 items 2-4, §3 5.5 bullet 1;
  CH5_METHODS.md §2 c15, c09.]
- The best number of features differs by dataset and by downstream model. [FIGURE F2a. file
  pointer: c06, c09, c16]
  [2026-10-09, author agreed: c06, c09 and c15 appear in this outline only as file pointers. Each
  gets a sentence saying what it found, or comes out. c15's "hybrid beats PCA at the same number
  of features" (4 of 4 small-molecule datasets) disagrees with c02 and c11, where PCA beats the
  hybrid with the network on LogD and Caco-2. Look at the three PCA comparisons together before any
  of them goes in. None of these is its own section: see WEIGHTING in the handoff rules.]
- The best number of base representations differs by dataset and by downstream model. [FIGURE F2b,
  side by side with F2a. file pointer: c18, the rep_counts field of decided_grid.json]
  [CHECKED 2026-10-09: data for F2a and F2b are in c02 (best_by_model, and the budget and
  representation count each fold's hybrid1 picked). Table: scratchpad/thesis_ch5_8/
  CH5_RESULTS.md §6. The best budget differs by dataset and model on 17 datasets and not on 7
  (abagym_Ang2, abagym_VEGF, abcov_kd, ddg_combined, flab_expression_koenig, ginkgo_pr_cho, qm9).
  LogD dnn, adme_hlm dnn and flab_expression_koenig (both models) picked 2,000 features on every
  fold. That is the top of the grid, so the best width may lie above it. c33's wider search on
  LogD dnn gives 0.4066 RMSE at 3,016 features, picked on held-out scores. Most small-molecule
  folds use all representations. buchwald_hartwig picks 2 on every fold with both models; qm9
  lightgbm picks 3 on every fold. Source: CH5_RESULTS.md §2, §3 5.5 bullets 2-3, §6.]
- Scoring candidate hybrids with the downstream model itself does not generally help. It wins only
  on SMAC colloidal stability, and each condition picks its own number of features, so the
  conditions differ in two ways at once. [file pointer: c23]
  [CHECKED 2026-10-09 against the 9 Oct results: was SMAC colloidal stability only; now the
  matched evaluator beats every mismatched one with both models on ginkgo_colloidal and
  abagym_EGFR. ginkgo_colloidal svm: 0.3806 Spearman against 0.3501 and 0.3626. qm9 has 3 of 5
  folds; flab_expression_koenig is partial. Every svm comparison used the gp evaluator from
  before the 8 Oct length-scale fix. Source: CH5_RESULTS.md §3 5.5 bullet 4.]
- Feature-importance weights during training: at least one of four weighting conditions beats
  unweighted fitting on most datasets, and which one wins changes with the dataset, the model and
  the feature set. A yes needs only one of four, so the claim is weak by construction. [file
  pointer: c19]
  [CHECKED 2026-10-09: same. Yes for at least one model on 23 of 24 datasets with results; no for
  both on adme_mdr1; adme_hlm has no file. Sizes are small on small molecules: qm9 dnn 0.3163 to
  0.3134, LogD dnn 0.4086 to 0.4077 RMSE. The largest is ginkgo_colloidal dnn, 0.2607 to 0.3571
  Spearman. Source: CH5_RESULTS.md §3 5.5 bullet 5.]
  RESTATE BY DOMAIN (author agreed 2026-10-09). Weighting beats unweighted fitting on the first
  hybrid on 13 of 15 antibody datasets, with each model. On small molecules it is mixed: 4 of 7
  datasets with the first model, 3 of 7 with the second. [Counts from the c19 page's yes/no table,
  rebuilt 9 Oct; not yet re-read from the raw files.]
- [NEW 2026-10-09, author agreed] Fewer features against the model's own regularisation setting.
  A smaller hybrid never beats the full feature set with the regularisation setting tuned on small
  molecules: 0 of 5 datasets, for each of the two smaller hybrids. On antibodies it does: 4 of 7 and
  5 of 7 datasets. A conditional result by domain. See the caveat on this experiment in 5.10.
  [file pointer: c16. Counts from the c16 page's yes/no table, rebuilt 9 Oct; not yet re-read from
  the raw files.]
- [NEW 2026-10-09, author agreed, one line in the text] Adding a newly released embedding to the
  pool helps the hybrid on 4 of 4 small-molecule datasets with the first model and 2 of 4 with the
  second, and on 5 of 8 antibody datasets with each model. See the caveat on this experiment in
  5.10 (the LogD file ran on 2,000 molecules). [file pointer: c13. Counts from the page's yes/no
  table, rebuilt 9 Oct; not yet re-read from the raw files.]
- [NEW 2026-10-09, author agreed, one line in the text] Calibration of conformal intervals. The
  hybrid's interval is better calibrated than the best single representation's on 5 of 8
  small-molecule datasets with the first model and 6 of 8 with the second, and on 2 and 3 of 14
  antibody datasets. Conditional by domain. [file pointer: c08. Counts from the page's yes/no table,
  rebuilt 9 Oct; not yet re-read from the raw files.]
- Extra features added to one strong representation, with two planted decoys. See 5.5a.

5.5a  Results: admitting extra features, and the decoy test
Written 2026-10-08. The results exist and were found at
tests/experiments/results/framework/f24_augmentation_admission_*.json. Five datasets, 3 outer
folds, 3 seeds per fold. Two planted decoy donors: a pure-noise donor, and a near-copy of the base
representation roughed up by 5 per cent of each feature's own standard deviation. A rule that keeps
either has not won anything.
[CHECKED 2026-10-09: was five datasets; now three with results (buchwald_hartwig, caco2,
flab_expression_koenig; a fourth was patient data, removed). herg_ki has one single-representation rf file only.
abagym_VEGF, abcov_kd, flab_thermo_tresanco and logd have timing files only. 3 folds and 3 seeds:
same. Models are rf and lightgbm, not each dataset's two decided models. The files are local,
dated 7 Sep, and not in git. They predate the 18 Sep representation-cache fix, the 28 Sep removal
of the admission gate (920bbb8) and the 5-6 Oct per-fold redesign. The full-run c21 does not test
admission except on ginkgo_colloidal and ginkgo_hic. So this table can only come from these
7 Sep files. Source: scratchpad/thesis_ch5_8/CH5_METHODS.md §0 item 13; CH5_RESULTS.md §3 5.5a,
§5.]
[DECIDE: report f24 as a dated framework result on 4 datasets with the code changes named, or
re-run it on the current code and decided models before quoting it (the counts below may
change).]
- The rule with no test at all, which takes every donor's top features, took decoy features in
  every configuration on every dataset and both models: 0 of 36 clean, eight times over.
  [NOT RECHECKED 2026-10-09: this count and "a fifth to two thirds" below were not recomputed.
  Source: CH5_RESULTS.md §3 5.5a.]
- The two real rules take decoy features in between a fifth and two thirds of their
  configurations, and the decoy they take is the near-copy, not the pure noise.
- Caco-2 with the random forest, extras appended outside the budget: 208 of 407 configurations
  beat no extras, 185 beat the matched random donor, 160 took no decoy features.
- Buchwald-Hartwig with LightGBM, extras displacing base features: 108 of 259 beat no extras, 50
  beat the matched random donor, 59 took no decoy features.
  [CHECKED 2026-10-09: 407 and 160 on Caco-2, 259 and 59 on Buchwald-Hartwig are the same. 208,
  185, 108 and 50 were not recomputed. Source: CH5_RESULTS.md §3 5.5a.]
- LIMIT ON QUOTING: on Buchwald-Hartwig the fold spread is 1.6198 RMSE with LightGBM and 2.1040
  with the random forest, and every delta in the file is smaller than that. Three folds. So the
  counts rank the rules against each other and no delta is quotable.
  [NOT RECHECKED 2026-10-09: the Buchwald-Hartwig spreads were not recomputed. Source:
  CH5_RESULTS.md §3 5.5a.]
- [TABLE, not a figure. One row per dataset, model and admission rule. Three count columns.]
- This is a controlled negative result and it is the reason augmentation is off in the full runs.
  State it as such, and carry it to the limitations and the future work.

5.6  Results: blueprints on specific datasets
- What a blueprint is, read on named datasets: which base representation each selected feature came
  from, and what chemistry that names. [FIGURE F3. One horizontal bar per dataset, segments by
  representation. A second, deeper panel for one small-molecule dataset: segments broken down to
  named feature groups rather than whole representations.]
- Feature-set stability across the five folds. The overlap is low: the agreement coefficient
  reaches 0.4 only on Caco-2 with the support vector regressor, hERG with LightGBM, and LogD with
  both models. [file pointer: c24]
  [CHECKED 2026-10-09: same, 5 folds. caco2 svm 0.424, herg lightgbm 0.447, logd dnn 0.420, logd
  svm 0.497; caco2 dnn 0.399; all others below 0.33. The sets compared differ in size from fold
  to fold, because each fold picks its own budget (herg_ki svm: 1,000, 250, 250, 500 and 500
  features). The text should say the coefficient is over sets of different sizes. c24 is missing
  qm9, the three ADME sets, flab_binding_koenig, abagym_Ang2 and ginkgo_hic. Source:
  scratchpad/thesis_ch5_8/CH5_METHODS.md §2 c24; CH5_RESULTS.md §3 5.6.]
- TODO: the deeper small-molecule panel needs the per-feature names from the blueprint, which
  nothing has collected into one file yet. Reading only, no compute.
  [CHECKED 2026-10-09: c24's feature_sets_per_fold holds (representation, feature index) pairs
  per fold, without feature names. TODO stays. Source: CH5_RESULTS.md §3 5.6.]

5.7  Results: OPIG
- The hybrid against the single representation the same search ranked first, on three ChEMBL
  targets. It wins on every target and model except one. The network on factor Xa goes 0.7367 to
  0.6612 MAE in log units on all 5 folds. [file pointer: opig_qsar]
  [CHECKED 2026-10-09: same, 5 folds. The exception is factor Xa with lightgbm, 0.6854 to 0.7144.
  Each target built 8 of 10 representations; grover and smited are missing. Source:
  scratchpad/thesis_ch5_8/CH5_METHODS.md §2 opig_qsar; CH5_RESULTS.md §3 5.7.]
- Molecules that a representation cannot tell apart. On hERG Ki with the support vector regressor,
  the hybrid takes the graph-kernel representation from 0.6109 to 0.5715 RMSE, lower on all 5
  folds, and the molecules sharing an identical feature row fall from 105 to 0. [file pointer:
  opig_shared_vectors_remedy]
  [CHECKED 2026-10-09: same, 5 folds; the pair that adds ecfp4 and mhg_gnn. Only herg_ki ran.
  Source: CH5_METHODS.md §2 opig_shared_vectors_remedy.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH5_METHODS.md §2 opig_shared_vectors_remedy] The
  other pair adds ecfp4 and pdv, both blind to stereochemistry. It also improves svm, 0.6109 to
  0.5755, lower on 5 of 5 folds, while 56 molecules still share a row. So the score gain does not
  need the shared rows broken.
- How many molecules share an identical feature row, per representation. MACCS keys merge 804 of
  2,158 Caco-2 molecules; on two antibody sets a representation merges most of them. [file
  pointer: opig_shared_vectors_census]
  [CHECKED 2026-10-09: same. aac on abagym_EGFR 432 of 659; structural on abagym_VEGF 332 of 418.
  maccs is in Caco-2's augmentation list, not its base pool. The census is a partial file
  covering only the 10 datasets up to ddg_combined alphabetically; herg_ki, logd, qm9, herg and
  the ginkgo and flab sets are not counted. Source: CH5_METHODS.md §2; CH5_RESULTS.md §3 5.7.]
- Transferring a recipe from LogD to hERG Ki. It beats hERG Ki's own best single representation
  with both models, and loses to a hybrid searched on hERG Ki. [file pointer:
  opig_blueprint_transfer]
  [CHECKED 2026-10-09: same, 5 folds. svm: best single 0.5888, native hybrid 0.5631, recipe
  transfer 0.5665, exact-feature transfer 0.5692 RMSE. dnn: 0.6035, 0.5666, 0.5889, 0.5926.
  Source: CH5_METHODS.md §2 opig_blueprint_transfer.]
- Antibody developability on the Ginkgo competition data: neither hybrid beats the single best
  representation on HIC retention. [file pointer: opig_ginkgo]
  [CUT 2026-10-09, author: opig_ginkgo is not in the thesis; c02 covers the Ginkgo sets on the
  competition folds, and ginkgo_hic is reported from c02.]
  [CHECKED 2026-10-09: same, but on 4 of 5 folds, ginkgo_hic only. gp: best single 0.6173,
  hybrids 0.5537 and 0.5072 Spearman. dnn: esm2_concat 0.5529, hybrid 0.4728. The other four
  Ginkgo properties have no result. Folds are the competition's own cluster folds. Source:
  CH5_METHODS.md §2 opig_ginkgo; CH5_RESULTS.md §3 5.7.]

5.8  Results: features chosen across separate sites
Confirmed as a section of this chapter, 2026-10-08. Only Caco-2 and LogD have federated results,
which is the reason it is a section and not a chapter.
- One feature selection over all representation families beats both the best single family and the
  best single representation, on Caco-2 and LogD, with both models. Splitting the budget evenly
  across families scores lower, but also builds a different number of features, so those two
  differ in two ways at once. [file pointer: fed03]
  [CHECKED 2026-10-09 against the 9 Oct results. First sentence: same, yes on all 8 entries. "Scores
  lower" is the wrong way round: in RMSE lower is better. The even split has the lower RMSE than
  one selection over all families on Caco-2 with both models and LogD with dnn, 3 of 4 pairs. On
  LogD svm (one finished seed) it is lower in 2 of 6 configurations. The width caveat stands. These
  are Part B results (families as sites), not the hERG design in 5.3.5. Source:
  scratchpad/thesis_ch5_8/CH5_RESULTS.md §0 item 5, §3 5.8.]
  [DECIDE: rewrite the second sentence as "the even split scores better on 3 of 4 pairs, but
  builds a different number of features", or drop the even-split comparison until widths are
  matched.]
- One hybrid over all five site representations beats the best single site and the best single
  representation on Caco-2 and LogD with both models. It beats the FLuID ensembles except with
  the Caco-2 network, where averaging five models gives 0.2720 against 0.2858 RMSE. [file
  pointer: fed02]
  [CONFLICT 2026-10-09: the two findings files read different fields. CH5_RESULTS.md §3 5.8: 0.2720
  and 0.2858 are seed 0 (per_seed[0]), SAME; per-cluster selection is also 0.2858 there. CH5_METHODS.md
  §2 fed02: the mean over 3 seeds is FLuID quality 0.2721, similarity 0.2730, hybrid 0.2837. The
  direction is the same either way. Say which the text quotes. Sites on Caco-2: ecfp4,
  graph_kernel, molformer, pdv, avalon; 5 folds, 3 seeds.]
- THE HONEST LIMIT. Selecting features inside each site beats the FLuID ensembles at every label
  flip level, and never beats the same selection made on all molecules pooled. Federation still
  costs something and choosing features across sites recovers part of it. [file pointer: fed01]
  [CHECKED 2026-10-09: same for Caco-2 only, from a partial file (3 dnn seeds, 2 svm seeds; no
  caco2.json). LogD has no finished seed: NOT YET RUN. Labels are two-class, metric MCC, 5 k-means
  sites. Source: CH5_RESULTS.md §0 item 7, §3 5.8.]
  [NEW 2026-10-09, from CH5_METHODS.md §2 fed01] A defect in the arm that removes flagged
  molecules at each site. On Caco-2, 240 site selections in that arm and 12 in the arm without
  removal were refused. A site's held-out block had one group, under ten molecules or one class.
  So the two arms differ in two ways at once: the removal, and which sites contributed.
- A null worth one line: selecting features per chemical-space cluster matches one global
  selection, and a separate model per cluster is worse. The clusters disagree about which
  representations matter more than random parts of the same size do: Caco-2 network, 0.5675
  against 0.6567 mean Spearman at 500 features. [file pointer: fed04]
  [CHECKED 2026-10-09: 0.5675 and 0.6567 same in caco2.json, dnn seed 0. The raw field read is the
  hybrid1 aggregate, so "at 500 features" is unverified. Per-cluster above global: Caco-2 yes for
  both models, LogD no/yes, by at most 0.0088 MCC. LogD has partial files only. Source:
  CH5_RESULTS.md §3 5.8.]
- CUT 2026-10-09 (author: no time): the antibody-site experiment (fed05). It was computed inside
  the lhasa_herg job, and lhasa_herg was cut on 8 Oct.
  [CHECKED 2026-10-09: same. Its caco2 and logd files are skip records. Source: CH5_RESULTS.md §3
  5.8.]

5.9  Discussion
- Does the search overfit. Both sides, with the numbers of 5.4 and 5.6.
- Where the hybrid does not help, stated as a pattern and not as a count: a single dominant
  representation, or several representations carrying the same information.

5.10  Limitations
- Compute and run time; any change to the data means a new search.
- The exact features do not transfer; only the recipe does, and on one dataset pair.
- A simpler selection may suffice on some datasets, and 5.5 names which.
- Admitting extra features was not solved: see 5.5a.
- CAVEATS TO STATE IF EITHER RESULT IS REPORTED (author, 2026-10-09: "if I report either of
  those results in the thesis I'll need to mention that"). Both are facts read from the result
  files, cause not known:
  - Adding a new embedding to the hybrid [c13]: the LogD file ran on 2,000 of LogD's 5,028
    molecules. Every other finished file for that experiment ran on all molecules.
  - Fewer features against the model's regularisation setting [c16]: the support vector
    regressor gives identical held-out scores at C = 10 and C = 100 on most HLM clearance results.
    [2026-10-09: a patient dataset was removed from this bullet; recheck "most".]

5.11  Conclusion
One paragraph per research question, then the bridge to Chapter 6.


CHAPTER 6  SIMILARITY IN AN ADAPTIVE HYBRID REPRESENTATION  (about 20 to 25 pages)
Rewritten 2026-10-08. The author's decision: this chapter is SIMILARITY ONLY. Label noise leaves
it entirely and goes to Chapter 7. Chapter 4 already sets up why similarity matters for noise, and
the noise step is left as a next step at the end of this chapter.
What is written in thesis.tex for this chapter is NOT the basis for the rewrite. Its noise half
(thesis.tex:3705-3975 and 4102-4667, about 6,166 words and 9 tables) moves to Chapter 7. Its
similarity half is 1,698 words, so most of this chapter is new writing from experiments that
thesis.tex has never named.
[CHECKED 2026-10-09 against thesis.tex: the line numbers above are stale. The noise half is now at
3738-4012 and 4139-4704. Every similarity number in thesis.tex 4013-4138 comes from three-fold
local runs on hERG Ki only, and all are superseded. Its sentence that the similarity hybrid is
skipped on two of 16 datasets is wrong now. Source: scratchpad/thesis_ch5_8/CH6.md §C.]
[CHECKED 2026-10-09: missing results, from the 9 Oct result folders. flab_expression_koenig has no
result in s02, s05, s06, s08, s10, s11 and s13. qm9 has none in s08, s10 and s11; those are
running. s11 ran on herg_ki and buchwald_hartwig only. s04 has real results on five datasets only
(caco2, herg, herg_ki, logd, qm9); the other 13 files point to s12. Eight datasets that c04 has
are in no similarity run: abagym_Ang2, abagym_VEGF, adme_mdr1, adme_rlm, bh_doyle_plates,
ginkgo_hic, tz_her2_binary, tz_her2_multiclass. Source: scratchpad/thesis_ch5_8/CH6.md §D.]

6.1  Introduction
- The bridge from Chapter 4: distance in a standard representation barely tracks activity, and
  every method built on that distance inherits the problem.
- The two ways a hybrid is used for similarity: as the space a distance is measured in, and as the
  space a search optimises for predicting each molecule's activity from its nearest neighbours.
- Research questions, two now rather than three. The third, on noise mitigation, moves to
  Chapter 7.

6.2  Methods
- Which feature space the distance is measured in: the best single representation, the second
  best, each with extra features added, four hybrids, raw ECFP4 bits under Tanimoto, forest
  proximity, and the model's own input matrix. Which distance formula. How many neighbours. How
  much each neighbour counts. Whether a molecule's own group is excluded.
  [CHECKED 2026-10-09 against the code: this is the noise experiments' list
  (noise/similarity_space.py), not this chapter's. Forest proximity, representations with extra
  features added, and the model's own input matrix appear in no similarity experiment. The spaces
  compared here, each picked per outer fold from searches on that fold's training molecules:
  kNN's best and second-best single representation; each decided model's best two single
  representations; each decided model's two best hybrids, so four hybrids; kNN's top hybrid
  (the distance hybrid); the top hybrid of the similarity objective; principal components; a
  random projection to 50 features. Raw ECFP4 bits under Tanimoto appear only as a Bayesian
  optimisation surrogate and in the cliff rule. Source: scratchpad/thesis_ch5_8/CH6.md §A.1.]
  [CHECKED 2026-10-09: distances are Manhattan, cosine and Minkowski p = 0.5 on datasets of 5,000
  molecules or fewer. Above 5,000 (herg, logd, qm9) only Manhattan and cosine. Neighbours: 5, 10,
  20. How much each counts: 1/rank or 1/d². Gaussian and 1/d are used in no similarity experiment.
  Source: CH6.md §A.2.]
  [CHECKED 2026-10-09: no similarity experiment excludes a molecule's own group from its
  neighbours. Every result file stores scaffold_group_excluded: false. Only the molecule itself is
  left out. The one place a group is excluded is inside the similarity objective's own score,
  through GroupKFold. Source: CH6.md §A.2, §A.3.]
  [DECIDE: what 6.2 lists. Either list only the spaces above and move forest proximity, extra
  features and the model's own input matrix to Chapter 7, which changes text only. Or keep them
  here, which needs them added to the similarity experiments and run.]
- Where the similarity objective is undefined and why: SKEMPI and AB-Bind binding free energy,
  because one protein complex holds 69 of the 429 rows; and all five Ginkgo developability sets,
  because the competition's own five folds are the five groups, so no neighbour exists outside a
  held-out block.
  [CHECKED 2026-10-09 against tests/experiments/dataset_policy.py: was "all five Ginkgo sets"; the
  code lists eight Ginkgo sets (hic, tm2, titer, pr_cho, acsins, purity, aggregation, colloidal).
  Four of them ran in these experiments: acsins, colloidal, pr_cho, titer. Also undefined, and
  missing above: tz_her2_multiclass, because they have more than 2 classes and
  the binary score is AUROC. tz_her2_binary is also undefined: its folds are cut on edit-distance
  shells from the wild-type CDR-H3, so a held-out block can hold one shell. Source:
  scratchpad/thesis_ch5_8/CH6.md §A.5.]
  [CHECKED 2026-10-09: binding free energy (ddg_combined) is skipped by name on all 5 outer folds.
  The one-complex problem refuses every configuration on folds 1 to 4 only. Fold 0 could have been
  scored. Source: CH6.md §A.5.]
  [DECIDE: binding free energy fold 0. Keep skipping it on all 5 folds, so the dataset has no
  similarity hybrid at all. Or score fold 0 alone, which gives a one-fold result and needs s02
  onwards re-run on that dataset.]
- Weighting features by their importance before the distance is taken, and the exponent that
  controls it.
  [CHECKED 2026-10-09 against the code: three formulas are in use, not one. s02 and s08 rescale
  the weights to mean 1, then raise them to 1 under Manhattan, 0.5 under cosine and Euclidean, 2
  under Minkowski p = 0.5. s04, s05, s06, s10 and s13 raise them to 0.5 under every distance, then
  rescale to mean 1. c04 takes the square root and rescales to sum 1. The exponent is swept only
  in f38 and n12, in no experiment of this chapter. thesis.tex's "x ⊙ w under Manhattan, x ⊙ √w
  under Euclidean" matches s02 and s08 only. The similarity objective's own distance has no
  weighting. Source: scratchpad/thesis_ch5_8/CH6.md §A.3, §A.4.]
  [DECIDE: the weighting formula. Describe the three formulas as they are, which changes text only.
  Or make the code use one formula, which needs the experiments on the other formulas re-run (s02
  and s08, or s04, s05, s06, s10, s13 and c04).]
- CITATIONS AND WHAT IS YOURS (added by Claude 2026-10-09 from the code; not reviewed by the
  author). Keys marked * were added to refs.bib on 2026-10-09, each checked against Crossref or
  the publisher.
  - Predicting a molecule from its k nearest neighbours: classical kNN regression. The
    task-specific version, choosing the descriptors to maximise that prediction, is Zheng2000 and
    Itskowitz2005. Both are already cited in Chapter 4. Yours: the features are chosen across
    several representations, not within one descriptor set.
  - How much each neighbour counts. 1/d is Dudani1976; inverse powers of d are Shepard 1968*
    (shepard1968two). The 1/rank scheme and the Gaussian weight scaled by each molecule's median
    neighbour distance have no source in code. None was found
    (src/kirby/similarity/neighbour_weighting.py:71-72, 259-272).
    [CHECKED 2026-10-09: 1/d is used only inside the similarity objective's own score, not in the
    similarity experiments. 1/d² and 1/rank are the two schemes in every similarity experiment.
    Gaussian is used in none (f34 only). The cited lines are the name list and weight_array; the
    schemes themselves are at lines 200-212 (Gaussian) and 214 onward (1/rank). Source:
    scratchpad/thesis_ch5_8/CH6.md §A.6.]
  - Forest proximity, 1 minus the share of trees in which two molecules share a leaf: Breiman2001
    (noise_estimation.py:2234-2277). Two details differ from the source. The forest is fitted on
    the training molecules' own labels, and every tree counts, not only the trees that left a
    molecule out. Earlier molecular use: Li et al. 2005 (J. Chem. Inf. Model. 45:952,
    doi 10.1021/ci050049u) used it to cluster chemicals, not to predict from neighbours.
    [CHECKED 2026-10-09: the lines match (2232-2276); 100 trees, 5 molecules per leaf. Forest
    proximity is used in no similarity experiment, only in noise experiments and f36/f38. It
    belongs with Chapter 7. Li et al. 2005 is not in refs.bib. Source:
    scratchpad/thesis_ch5_8/CH6.md §A.6.]
  - Importance-weighted features before the distance: feature-weighted kNN (Wettschereck, Aha and
    Mohri 1997, not yet verified). Yours: the importance comes from the hybrid search, and the
    exponent is swept.
    [CHECKED 2026-10-09: Wettschereck, Aha and Mohri 1997 is not in refs.bib. "The exponent is
    swept" is not true of any experiment in this chapter; see the weighting note above. Source:
    scratchpad/thesis_ch5_8/CH6.md §A.4, §A.6.]
  - Excluding a molecule's own scaffold group from its neighbours (noise_estimation.py:204-262):
    no source in code.
    [CHECKED 2026-10-09: that function excludes repeat measurements of the same drug-target pair
    (BindingDB), not scaffold groups. No experiment in this chapter excludes anything. The
    similarity objective excludes a molecule's group through GroupKFold. Source: CH6.md §A.6.]
  - [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH6.md §A.3, §A.6] Sources the chapter's code
    relies on that this block does not name. In refs.bib: Golbraikh2014 (MODI, named in the
    similarity objective's docstring), Guha2008 (SALI), van Tilborg 2022 (cliffs), Beyer1999
    (distance concentration, s11), Vovk2005, Shafer2008 and Lei2018 (split conformal), Kraskov2004
    (the s03 estimator). The s04 result files cite Stumpfe & Bajorath 2019 for the cliff threshold;
    refs.bib has Stumpfe2012 only. Not in refs.bib: Levina and Bickel 2005 (s08), Snoek 2012 (s11),
    a q-RASAR source (s10), a k-means source (s13).

6.3  Results: does distance in a hybrid track activity better
- In every standard representation, distance between two held-out molecules rises with their
  activity difference, and the correlations are small. The highest on Caco-2 in any single
  representation is 0.2292 Spearman. On SMAC colloidal stability the highest is 0.0155. On eight
  datasets some representations sit at or below zero. [file pointer: s01]
  RESTATE BY DOMAIN (author agreed 2026-10-09). Distance follows activity in every representation
  (Manhattan) on 5 of 5 small-molecule datasets. On 7 of 11 antibody datasets at least one
  representation sits at or below zero. [Counts from the page's yes/no table, rebuilt 9 Oct; not yet re-read from the raw files.]
  [CHECKED 2026-10-09 against the 9 Oct results, all 18 datasets [17 after the patient dataset was removed], 5 folds: Caco-2 0.2292 and SMAC
  0.0155 are the same. Was "five datasets"; now eight have at least one representation and
  distance at or below zero. They are abagym_EGFR, cdr_flexibility, flab_expression_koenig,
  ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho and humanness_therapeutic. None
  is a small-molecule set. Small molecules, highest single representation: caco2 0.2292, qm9
  0.2789, logd 0.1470, herg_ki 0.1170, herg 0.0412. The fold spread is large: herg_ki mhg_gnn
  under Manhattan runs 0.241 to 0.020 across the five folds. Source:
  scratchpad/thesis_ch5_8/CH6.md §B 6.3 bullet 1.]
- In the hybrid built for distance it rises more steeply, on every dataset that ran except
  anti-coronavirus affinity (abcov_kd). hERG Ki goes
  0.0729 to 0.2625 Spearman, Manhattan distance, no limit on the number of features. The models'
  own predictive hybrids do not always do this. [FIGURE F5. file pointer: s02, s08]
  [CHECKED 2026-10-09 against the 9 Oct results, 5 folds: hERG Ki 0.0729 to 0.2625 is the same.
  Was "on every dataset that ran"; now 16 of the 17 that ran. On abcov_kd the
  distance hybrid is lower: 0.0901 to 0.0376 unweighted, 0.0715 to 0.0621 weighted. Small
  molecules all rise: qm9 0.1298 to 0.3779, logd 0.0765 to 0.2687, caco2 0.1517 to 0.1969, herg
  0.0257 to 0.0552. "The models' own predictive hybrids do not always do this" is the same: at least
  one model's hybrid is lower on abagym_EGFR, abcov_kd, buchwald_hartwig, caco2, ginkgo_colloidal,
  ginkgo_pr_cho and ginkgo_titer. Source: scratchpad/thesis_ch5_8/CH6.md §B 6.3 bullet 2.]
  [CHECKED 2026-10-09: two things the comparison does not say. The single representation compared
  is kNN's top-ranked one for prediction, not the one whose distance best follows activity.
  Against s01's best single representation, picked on the held-out molecules, the distance hybrid
  is lower on caco2 (0.1969 against 0.2292), abcov_kd and ginkgo_titer. The similarity hybrid on
  caco2 is higher, at 0.2769. Second, importance weighting alone closes much of the gap. On logd
  the weighted single representation scores 0.2762, above the unweighted hybrid's 0.2687; the
  weighted hybrid scores 0.3094. On qm9: weighted single 0.3286, unweighted hybrid 0.3779,
  weighted hybrid 0.4596. Source: CH6.md §B 6.3 bullet 2.]
  [CHECKED 2026-10-09: five s02 files hold unweighted distances only: caco2, ginkgo_colloidal,
  ginkgo_titer, herg and herg_ki. They were written before weighted distances were restored on
  2026-10-07. The other twelve hold both. A weighted comparison on these five needs the same
  command run again. flab_expression_koenig: NOT YET RUN. Source: CH6.md §B 6.3 bullet 2.]
  [CHECKED 2026-10-09: F5 is now a table, not a figure (tab:distance_tracks_activity, commit
  f84364e). By default the figure script keeps 7 datasets with noise results (caco2, herg,
  herg_ki, logd, ddg_combined, ginkgo_colloidal, ginkgo_titer). So QM9 and the reaction sets
  are left out of F5 and F6. The script calls this filter a placeholder. Source:
  scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md §0 items 1-3, §A.2.]
  [DECIDE: which datasets the F5 and F6 tables show. The current 7, which leaves out QM9 and
  under-counts small molecules. Or every dataset with a result, split by domain, which changes
  only the script's dataset filter.]
- The number of features at which a model predicts best is generally not the number at which
  distance tracks activity best, and prediction mostly peaks at more features. They coincide only
  for the support vector regressor on Caco-2 at 500 features, LightGBM on CDR flexibility at
  50, and LightGBM on anti-coronavirus affinity at 500. [file pointer: s08]
  WEIGHT (author agreed 2026-10-09): one line in the text, not its own section.
  [CHECKED 2026-10-09 against the 9 Oct results, 16 datasets, 5 folds: Caco-2 and CDR flexibility
  are the same. Was those two only; now also LightGBM on abcov_kd at 500 features (Manhattan and
  Minkowski, unweighted). Partial coincidences, under some distances only: abagym_EGFR network at
  250, ddg_combined network at 100 and support vector regressor at 500, flab_thermo_tresanco both
  models at 250. Over all 184 model, distance and weighting combinations, prediction peaks at
  more features in 158, at the same number in 22, at fewer in 4. Source:
  scratchpad/thesis_ch5_8/CH6.md §B 6.3 bullet 3.]
  [CHECKED 2026-10-09: "prediction peaks at more features" is partly built into the design.
  Prediction is swept over the general budgets and distance over the distance budgets. On caco2,
  logd, buchwald_hartwig and ginkgo_colloidal the prediction budgets are 500, 1000 and 2000
  features. The distance budgets are 100, 250 and 500. They share only 500, so prediction cannot
  peak below 500 there. On herg_ki the two share 250 and 500. Many prediction peaks are ties:
  caco2 network 0.2785 RMSE at 1000 features, 0.2783 at 2000. qm9 and flab_expression_koenig: NOT
  YET RUN. Source: CH6.md §B 6.3 bullet 3.]
  [DECIDE: the two budget grids. Keep them and state that the overlap limits where each can peak,
  which changes text only. Or sweep prediction and distance over the same budgets, which needs s08
  re-run.]

6.4  Results: prediction from nearest neighbours
- Predicting each molecule from its nearest training molecules scores better in the hybrid's
  features than in the best single representation's, on every dataset except binding free energy,
  Buchwald-Hartwig and AC-SINS self-association.
  CHO polyreactivity goes 0.2059 to 0.3591 Spearman at the setting fixed before the run; binding
  free energy falls 0.1424 to 0.0899. [FIGURE F6. file pointer: s05]
  RESTATE BY DOMAIN (author agreed 2026-10-09). At the setting fixed before the run the hybrid
  wins on 5 of 5 small-molecule datasets and 8 of 10 antibody datasets; with the setting re-picked
  per fold, 5 of 5 and 4 of 10. [Counts from the page's yes/no table, rebuilt 9 Oct; not yet re-read from the raw files.]
  [CHECKED 2026-10-09 against the 9 Oct results, 5 folds: CHO polyreactivity is the same. Binding
  free energy was 0.0864; now 0.0899. Was one losing dataset; now three. buchwald_hartwig goes
  14.4572 to 16.1811 RMSE. ginkgo_acsins goes 0.3912 to 0.3884 Spearman. The fixed setting is the
  distance hybrid against kNN's best single representation, unweighted, Manhattan, 5 neighbours,
  1/rank. All five small-molecule sets are better: caco2 0.3386 to 0.3246, herg_ki 0.6280 to
  0.5822, logd 0.6766 to 0.5831, qm9 0.6692 to 0.4205 RMSE; herg 0.8143 to 0.8335 AUROC. Largest
  antibody gain: ginkgo_colloidal 0.1268 to 0.2418. Binding free energy per fold, hybrid minus
  single: 0.212, -0.329, -0.176, 0.071, -0.039. Source: scratchpad/thesis_ch5_8/CH6.md §B 6.4
  bullet 1; FIGURES_AND_STYLE.md §A.3 F6.]
  [CHECKED 2026-10-09: F6 is now a table, not a figure (tab:nearest_neighbour_prediction). Its
  similarity-hybrid rows are empty on every Ginkgo set and binding free energy, for the
  reasons in 6.2. Its labels use British "neighbour"; the thesis uses American spelling. Source:
  FIGURES_AND_STYLE.md §A.3 F6.]
- With the setting re-picked on each fold the hybrid loses on seven datasets, and that comparison
  picks from a different set of settings on each side. State that.
  [CHECKED 2026-10-09 against the 9 Oct results, 5 folds: was four datasets; now seven. abcov_kd
  0.2897 to 0.1638, buchwald_hartwig 13.8442 to 14.0299 RMSE, cdr_flexibility 0.8734 to 0.8553
  AUROC, flab_thermo_tresanco 0.5717 to 0.5605, ginkgo_acsins 0.4559 to 0.4410, ginkgo_pr_cho
  0.3501 to 0.2600, ginkgo_titer 0.3221 to 0.2482. All seven are antibody sets or the reaction set.
  All five small-molecule sets win, for example logd 0.6384 to 0.5648 and qm9 0.6167 to 0.4221.
  The two sides pick from 216 hybrid settings against 36 single-representation settings. On herg,
  logd and qm9 it is 144 against 24. Where the similarity hybrid is undefined it is 180 against
  36. ginkgo_pr_cho flips sign: +0.1532 at the fixed setting, -0.0901 re-picked. Source:
  scratchpad/thesis_ch5_8/CH6.md §B 6.4 bullet 2.]
- Multiplying features by their importance generally helps prediction from nearest neighbours, and
  generally does not help the Gaussian process or the support vector regressor. On LogD it goes
  0.6512 to 0.5911 RMSE on the first-ranked single representation. [TABLE, not a figure. file
  pointer: s10, c04]
  [CHECKED 2026-10-09 against the 9 Oct results, 16 datasets, 5 folds: the direction is the same.
  One unit here is one feature space at one setting. kNN: weighting better in 1,409 of 2,100
  units. GP: 41 of 145. Support vector regressor: 24 of 91. Where weighting mostly hurts kNN:
  ginkgo_titer 41 of 126, humanness_therapeutic 40 of 144, abcov_kd 67 of 144. LogD 0.6512 to
  0.5911 is the same, but it is one of 12 settings (cosine, 10 neighbours, 1/d²). The setting
  fixed before the run gives 0.6766 to 0.6065. GP and the regressor run on regression sets only;
  GP is skipped on logd, which is over 5,000 molecules. qm9 and flab_expression_koenig: NOT YET
  RUN. Source: scratchpad/thesis_ch5_8/CH6.md §B 6.4 bullet 3.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH6.md §B 6.4 bullet 3] q-RASAR, the fourth model
  that reads a distance, is not in this bullet. Weighting is better in 558 of 1,212 units, about
  even.
  [CHECKED 2026-10-09, c04: all representations joined, 10 neighbours, 5 folds. Weighting helps
  for at least one importance method on all 20 regression datasets that ran. It loses for some
  method on five: abagym_Ang2, abcov_kd, caco2, ginkgo_colloidal, ginkgo_hic. Small molecules:
  logd 0.6962 to 0.6693, qm9 0.5749 to 0.4841, herg_ki 0.6422 to 0.6198, caco2 0.3425 to 0.3365
  RMSE. Classification sets are skipped. No experiment has a shuffled-weights control. Source:
  CH6.md §B 6.4 bullet 3, §C.]
- Picking which molecules to measure next, in a hybrid space. hERG Ki and Buchwald-Hartwig: on
  hERG Ki the second-ranked network hybrid, rebuilt every 25 picks, reaches 410.59 against 426.25
  for raw ECFP4 bits, in pKi times picks, lower is better. [file pointer: s11]
  WEIGHT (author agreed 2026-10-09): one line in the text, not its own section. Two datasets
  only: the hybrid wins on herg_ki and loses on buchwald_hartwig.
  [CHECKED 2026-10-09 against the 9 Oct results, 3 seed sets, not folds: was 410.63 against
  426.63; now 410.59 against 426.25. Others on hERG Ki: ECFP4 under the RBF kernel 428.03, random
  picking 431.70, principal components 415.00, random projection 417.34. The first network hybrid
  with rebuilding scores 417.55, worse than principal components. Bayesian-network and NGBoost
  surrogates on hybrids are at or worse than random picking, up to 435.83. Source:
  scratchpad/thesis_ch5_8/CH6.md §B 6.4 bullet 4.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH6.md §B 6.4 bullet 4] It also ran on
  Buchwald-Hartwig, hunting high yield, higher is better. The best is the Gauche GP on the second
  LightGBM hybrid: 9992.31. The GP with the RBF kernel on the first LightGBM hybrid scores 9871.61.
  That is below random picking at 9901.52. Random projection scores 9849.84. The Tanimoto
  condition was skipped: no ECFP4 representation. Skipped datasets: caco2 and logd (labels
  clipped at the 99th percentile), herg (classification), all antibody sets. qm9 is
  running.

6.5  Results: activity cliffs
- Where cliff molecules could be scored, the hybrid mostly beats the single representation on them
  and on the other molecules alike. The gain is larger on cliff molecules on caco2, herg_ki and
  herg, not on logd or qm9. LogD with the network:
  0.4299 to 0.3876 RMSE on cliff molecules, 0.4650 to 0.4174 on the rest. [file pointer: s04, s12]
  RESTATE BY DOMAIN, AND COMBINE (author agreed 2026-10-09). s12 carries this section. The hybrid
  is better on cliff molecules on 4 of 5 small-molecule datasets and 1 of 4 antibody datasets. s04
  (cliffs defined in ECFP4, small molecules only) becomes one sentence inside the s12 paragraph,
  not a result of its own. [Counts from the page's yes/no table, rebuilt 9 Oct; not yet re-read from the raw files.]
  [CHECKED 2026-10-09 against the 9 Oct results, 5 folds: the LogD numbers are the same. Was "the
  gain is larger on the cliff molecules"; on this LogD example it is 0.0423 RMSE on cliff molecules
  and 0.0476 on the rest. Larger on cliffs: caco2 (network 0.3396 to 0.2881 cliff, 0.2890 to
  0.2720 rest), herg_ki (network 0.5696 to 0.5266, 0.6095 to 0.5788), herg (LightGBM 0.7930 to
  0.8082, 0.8883 to 0.8954 AUROC). qm9 is worse on cliffs: network 0.2422 to 0.2798, LightGBM
  0.2612 to 0.2666 RMSE, on 2 of 5 folds, 48 cliff molecules held out in all. Other cases where the
  hybrid is worse on cliffs: buchwald_hartwig random forest (worse on both kinds),
  humanness_therapeutic XGBoost,
  flab_expression_koenig network, abagym_EGFR (1 fold). s04 has real results on the five
  small-molecule sets only; s12 ran on all 18 [17 after the patient dataset was removed]. Source: scratchpad/thesis_ch5_8/CH6.md §B 6.5.]
  [CHECKED 2026-10-09: the cliff rule aims at 15% of training molecules. Held-out molecules come
  out at about 40%: logd folds 39.1% to 44.5%, herg 38% to 46%, caco2 28% to 33%, herg_ki 19% to
  27%. The held-out rule is "at least two similar training molecules differ by more than the gap".
  It is looser than the training rule. So "cliff molecule" means a different thing in training and
  held-out molecules. Source: CH6.md §B 6.5.]
  [DECIDE: the held-out cliff rule. Keep it and state the two shares, which changes text only. Or
  apply the training rule to held-out molecules, which needs s04 and s12 re-run.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH6.md §B 6.5] An antibody gain on cliff molecules:
  flab_expression_koenig with the support vector regressor, 0.4884 to 0.6020 Spearman on cliff
  molecules and 0.7123 to 0.7866 on the rest, Manhattan.
- THE CONDITIONAL CASE, and it belongs in the text. On CDR flexibility the hybrid is worse on cliff
  molecules for both models, LightGBM 0.8437 to 0.8332 and the gradient-boosted trees 0.8642 to
  0.8467 AUROC, while on the non-cliff molecules XGBoost goes 0.8393 to 0.8827 and LightGBM falls
  0.8811 to 0.8665. Counted
  inside that one dataset, the hybrid won none of its 18 settings on cliff molecules and 9 of 18 on
  the rest.
  [CHECKED 2026-10-09 against the 9 Oct results, concatenation partition, Manhattan, 5 folds: the
  cliff numbers are the same. Was "the same model goes 0.8393 to 0.8827"; those are XGBoost's
  numbers, not LightGBM's. For LightGBM the hybrid is worse on both kinds of molecule. The "18
  settings" count is the kNN count (3 distances, 3 neighbour counts, 2 schemes), taken from an
  answer page and UNVERIFIED. Source: scratchpad/thesis_ch5_8/CH6.md §B 6.5.]
- NO DATA, state it as no data and not as no effect: on binding free energy, nanobody
  thermostability, anti-coronavirus affinity and all four Ginkgo sets no fold held the 10 cliff
  molecules a score needs.
  [CHECKED 2026-10-09: the list is the same, plus abcov_kd, which has 10 cliff molecules over 141
  held out. Thermostability has one fold with 3 cliff molecules under cosine. Source:
  scratchpad/thesis_ch5_8/CH6.md §B 6.5.]

6.6  Results: other uses of a distance
- Distance to the nearest training molecules against prediction error. It rises with error but
  weakly, 0.1241 Spearman on Caco-2 with the support vector regressor's errors. On three datasets
  it does not rise at all. The hybrid tracks error worse for both models on six datasets. An
  interval that is narrower while keeping its coverage appears on hERG Ki with the network and on
  Buchwald-Hartwig with both models: on hERG Ki, width 2.1086 to
  1.9476 at coverage 0.9129. [file pointer: s06]
  REMOVE THE INTERVAL HALF (author agreed 2026-10-09). The interval has one width for every
  molecule in a fold, so it measures the model's error in each space, not distance. Keep only
  distance to the nearest training molecules against prediction error.
  [CHECKED 2026-10-09 against the 9 Oct results, Manhattan, 5 neighbours, 5 folds. Errors come
  from each model fitted on all representations joined. Caco-2 0.1241 is the same; herg_ki 0.1996
  and logd 0.2163 with the same model. The three that do not rise are the same: buchwald_hartwig,
  ddg_combined and ginkgo_titer, at or below zero for both models. Was "better except on three";
  now worse for both models on five: abcov_kd (XGBoost 0.1865 to 0.0723), buchwald_hartwig,
  ginkgo_colloidal, herg (LightGBM 0.1390 to 0.0185) and humanness_therapeutic. Mixed on abagym_EGFR. Small molecules: better on caco2, herg_ki,
  logd and qm9; worse on herg. Source: scratchpad/thesis_ch5_8/CH6.md §B 6.6 bullet 1.]
  [CHECKED 2026-10-09: the hERG Ki interval numbers are the same, coverage 0.9044 to 0.9129. Was
  "hERG Ki with the network alone". Under the same rule, narrower at coverage 0.9 or above,
  buchwald_hartwig also qualifies: random forest 63.16 to 54.80 at 0.9066, LightGBM 65.85 to 60.87
  at 0.9226. Classification sets have no interval. Source: CH6.md §B 6.6 bullet 1.]
  [CHECKED 2026-10-09 against s06_applicability_domain.py:73-75: the interval has one width for
  every held-out molecule in a fold. It is built from the plain absolute error, and each space
  refits the model. So a narrower interval means smaller model errors in that space. It does not
  use distance. Source: CH6.md §B 6.6 bullet 1.]
  [DECIDE: the interval. Move it to prediction, as a statement about model error in each space,
  which changes text only. Or scale each molecule's interval by its distance to the training
  molecules, which needs s06 changed and re-run.]
- Retrieval and clustering. Hybrid spaces retrieve nearest neighbours with closer labels than the
  best single representation on most datasets, and beat the principal-component control on most.
  Clustering by label flips between cluster counts on seven datasets. LogD wins both: coherence
  0.1002 to 0.2252 at 10 clusters. Anti-coronavirus affinity loses both. [file pointer: s13]
  RESTATE BY DOMAIN (author agreed 2026-10-09). Retrieval: the hybrid wins on 5 of 5
  small-molecule datasets and 6 of 10 antibody datasets, and beats the PCA control on 5 of 5 and
  7 of 10. [Counts from the page's yes/no table, rebuilt 9 Oct; not yet re-read from the raw files.]
  [CHECKED 2026-10-09 against the 9 Oct results, 17 datasets, 5 folds: LogD is the same.
  Retrieval on LogD wins 12 of 12 settings; re-picked label gap 0.5932 to 0.5434. Was "retrieves
  with closer labels" everywhere. The re-picked label gap to the nearest neighbours rises, which is
  worse, on abcov_kd (1.0933 to 1.2040), buchwald_hartwig (12.013 to 12.149), cdr_flexibility
  (0.2039 to 0.2082) and ddg_combined (1.4386 to 1.5072). Was "beats the principal-component
  control". The control is better on abagym_EGFR (0.2610 against 0.2696), abcov_kd (1.0865
  against 1.2040) and cdr_flexibility (0.1831 against 0.2082). It ties on ginkgo_pr_cho (0.1546).
  Source: scratchpad/thesis_ch5_8/CH6.md §B 6.6 bullet 2.]
  [CHECKED 2026-10-09: was five datasets that flip; now seven. buchwald_hartwig, caco2,
  ddg_combined, ginkgo_acsins, ginkgo_titer, herg_ki, and humanness_therapeutic (by 0.0002 at 3
  clusters). Was "SMAC colloidal stability loses both". Its clustering loses at 3, 5 and 10
  clusters (0.5369 to 0.4608 at 10). Its retrieval loses 13 of 18 settings, but the re-picked
  retrieval wins, 0.4586 to 0.4241. abcov_kd loses both: clustering at all three counts, retrieval
  at 16 of 18 settings. Small molecules: retrieval wins every setting on all five. Clustering
  wins every count on herg, logd and qm9 and flips on caco2 and herg_ki. Clustering is k-means
  under squared Euclidean distance, not the distances swept elsewhere. Source: CH6.md §B 6.6
  bullet 2.]
- How much label information the kept features carry, against a random subset of the same size and
  against principal components of the same size. [file pointer: s03]
  WEIGHT (author agreed 2026-10-09): one line in the text. The kept features carry no more label
  information than PCA on most datasets.
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH6.md §B 6.6 bullet 3] s03 has results now, 18
  datasets. One unit is one fold, model and representation. The kept features carry more label
  information than a random subset of the same size in about half the units on most datasets:
  caco2 52 of 100, logd 53 of 100, herg 20 of 51, ginkgo_acsins 24 of 77, ginkgo_colloidal 17 of
  62. Clearly above half only on qm9 50 of 58, herg_ki 72 of 99, flab_expression_koenig 62 of 98
  and flab_thermo_tresanco 44 of 70. Against principal components of the same size they carry
  more in under half the units on 15 of 18 datasets [one of the 18 was patient data, now removed; recount over 17]. The three at or above half: buchwald_hartwig
  15 of 20, qm9 40 of 58, flab_expression_koenig 53 of 98. The score is a sum of one-component
  estimates over at most 50 principal components, not the mutual information of the set. This goes
  against the framing that the kept features concentrate label information. State it as found.

6.7  Conclusion
- What a hybrid fixes about a distance and what it does not.
- The step left as a next step: using that distance for label noise, which is Chapter 7.
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH6.md §0, §B] The small-molecule picture on its
  own. The similarity runs hold 11 antibody sets and 5 small-molecule sets (caco2, herg, herg_ki,
  logd, qm9). So every "on N datasets" count above leans on antibodies unless split by domain. On
  all five small-molecule sets the distance hybrid tracks activity better (s02). It predicts from 5
  nearest neighbours better, at the fixed and at the re-picked setting (s05). It retrieves
  neighbours with closer labels at every setting (s13). Distance tracks error better on four of
  the five; herg is worse (s06). Clustering wins on herg, logd and qm9 and flips on caco2 and
  herg_ki (s13). On cliff molecules qm9 is worse (s12, 2 of 5 folds).


CHAPTER 7  LABEL NOISE: DETECTING IT AND DOING SOMETHING ABOUT IT  (about 45 to 55 pages)
Written 2026-10-08 from the author's own outline of the chapter plus the 2026-10-08 analysis.
The author's decision: ONE chapter, however long. The methods and the setup take a lot of space and
that is accepted, because it is all one topic. Do not split it.
What thesis.tex has for this material today sits inside Chapter 6 and is preliminary. It is not the
basis for this chapter.

STANDING CAVEATS, to state once at the top of the chapter and not repeat
- Every number is outer fold 0, with one exception: the removal-cost and label-smoothing experiment
  has all five folds on Caco-2 and hERG Ki.
  [CHECKED 2026-10-09 against the 9 Oct results: OUT OF DATE. The removal-cost experiment has 5
  folds on five datasets: buchwald_hartwig, caco2, ginkgo_colloidal, herg_ki, logd. Six more
  experiments have 3 to 5 folds on some datasets. Folds 1 and 2 exist for parts of six others.
  Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §6 and item 10. [file pointer: n27; c10, n14,
  n15, n16, n26, n37 have 3-5 folds; promoted folds in n07, n20, n23, n24, n25, n30]]
  [CONFLICT: PATTERNS.md and FIGURES_AND_STYLE.md §A.2 say every noise result they read is fold 0.
  The figure loaders found no other folds. Those read the answer pages and the figure inputs, not
  the promoted runs.]
- A control exists for only some treatments. Deleting molecules must beat a matched random deletion
  of the same count at five seeds. Cleaning the labels as delivered must beat the same detector
  with its scores shuffled. Putting cliff molecules back must beat putting random molecules back.
  Raising clipped labels per molecule must beat the same lifts dealt to the wrong molecules.
  Everything else has no control, and a win without a control is a weaker claim.
  [CHECKED 2026-10-09: the four controls above ran; same. Three more ran: random trimming,
  random two-learner selection and random winsorising, same count, 3 seeds [file pointer: n32,
  n38, n41]. No other control has run. The generic random and shuffled twins exist in code but
  are switched off unless KIRBY_NOISE_CONTROLS=1. 108 result files across 13 experiments record
  controls_run false. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §3.4.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_RESULTS.md §6: the random put-back control is
  usable only on hERG Ki. On Caco-2, hERG and LogD the random pool was smaller than the cliff set
  on every pass. Cliff molecules are 54 to 86 per cent of the training molecules there.]
- On Caco-2 any gain below about 0.014 RMSE sits inside the score spread of a cache fault, now
  fixed.
- A hybrid's own clean-label score moves with the damage level, because the hybrid is rebuilt from
  that draw's damaged labels. So no share-of-damage-recovered number is readable for a hybrid.
  [CHECKED 2026-10-09 against the code: HALF TRUE. In most experiments the per-fold search picks
  the two base representations and the hybrid recipe on the CLEAN training labels. Only the
  hybrid's features are then re-ranked, and the budget re-split, on the damaged labels. The
  exceptions are the twelve-method survey, which searches on damaged labels, and the two
  experiments that compare clean-searched with damaged-searched. Source:
  scratchpad/thesis_ch5_8/CH7_METHODS.md §1.6. [file pointer: noise_helpers.py:2459-2461,
  n07:3230-3235; exceptions n09, n15, n20]]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_RESULTS.md §0] Two more caveats on every
  count. Where a method has several settings, the best one is picked on the held-out score. The
  single representation and the hybrid in each comparison are also picked after treatment.
- [NEW 2026-10-09, same source] The network's untreated score moves between jobs that should be
  identical. On QM9's second hybrid it is 0.3484, 0.3568 or 0.3618 RMSE. lightgbm gives one score
  in every job. So small network gains can be fit-to-fit spread.

7.1  Introduction
- The question: on labels that already carry assay error, and on labels with damage injected on
  purpose, does finding and changing suspect labels improve a held-out score.
- An assay dataset as delivered is not clean. A method that improves the fit on the labels as
  delivered MAY be removing the assay's own error. It may also be changing the model in a way
  that helps on any labels. QM9 separates the two (author, 2026-10-09; see 7.4a): its labels are
  computed, so a method that improves QM9 with no noise added is not removing noise.
  [REWRITTEN 2026-10-09. Before, this bullet said such a gain "is not a warning sign".]

7.2  Methods: what the noise is
- The damage kinds, one row per kind, as a table: Gaussian, outliers in 5 per cent of molecules,
  a shift in 10 per cent, a sparse spread increase in 10 per cent, an offset per scaffold group,
  a wider spread per scaffold group, clipping at a detection limit, and the six label-flip kinds
  for the classification datasets.
  [CHECKED 2026-10-09 against the code: the decided grid has six regression kinds, not seven.
  Three entries above are wrong. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §1.1.
  [file pointer: noise_helpers.py:2118-2120]
  - "Outliers in 5 per cent": EVERY training label gets Gaussian error. A random 5 per cent get it
    at 3 times the scale. Those 5 per cent carry 32 per cent of the squared error.
  - "A sparse spread increase in 10 per cent": it is an upward SHIFT of 10 per cent of labels by
    one fixed amount. The amount is the level times the spread of a forest's out-of-fold residual.
  - "A wider spread per scaffold group" is not in the decided grid. No result file has it.
  - The six kinds are: Gaussian, outliers in 5 per cent, offset per scaffold group, clipping,
    shift in 10 per cent, shift in 10 per cent sized by the model's error.]
  [NEW 2026-10-09, same source §1.4: the results hold damage kinds this list never names.
  Outliers in 5 per cent with every other label left alone; Gaussian error sized by the model's
  own error; simulated several sources per molecule with correlated errors and a bias per source.
  Plus three regimes with nothing injected: labels as delivered, real repeat disagreement on
  hERG IC50, and no damage at all. [file pointer: n31, n40, n39; n18/n20/n23, n17/n19/n36]]
  [NEW 2026-10-09, same source §1.2: of the six flip kinds, five are distinct on a two-class
  dataset. Confusion-directed flips with no matrix given are the same rule as uniform flips.]
  [NEW 2026-10-09, same source §1.2: the scaffold-group offset needs SMILES. It is dropped on
  antibodies, proteins and reactions.]
- The damage level is a fraction of the spread of the clean training labels. Censoring is a share
  of labels clipped.
  [CHECKED 2026-10-09: true for Gaussian, outliers and the group offset. The two shift kinds use
  the level as the size of a fixed shift. One is in units of the label spread, the other in units
  of the forest residual's spread. Flips use a share of labels flipped. Censoring: same; the limit
  is a quantile of the clean training labels. Source: CH7_METHODS.md §1.1, §7.]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_METHODS.md §1.3] Levels and draws. Three draws
  per non-zero level; one under censoring, because the clip reads no seed. Level 0 runs once.
  The seed holds neither the level nor the fold, so levels nest: the error at 1.0 is twice the
  error at 0.5 on every molecule. A molecule carries the same damage in every fold it trains in.
- [NEW 2026-10-09, same source item 5] One damage draw can never succeed. A draw outside the dose
  band raises an error, and the fixed seed makes a retry fail the same way. This killed one
  Gaussian unit on flab_thermo_tresanco twice. [file pointer: n12]
- Only the training labels are damaged. Held-out labels are left as delivered.
  [CHECKED 2026-10-09: same, for the outer held-out fold. Where a per-fold search runs on damaged
  labels, its inner folds are scored on damaged labels too. Source: CH7_METHODS.md §1.5.
  [file pointer: n09, n15, n20]]

7.3  Methods: finding the suspect labels
Group the detectors by what they read, not by their code names.
- The residual of a molecule's label against a prediction: an out-of-fold forest prediction, and
  an out-of-bag residual.
- The gap between a molecule's label and its neighbours' labels, with and without the neighbours'
  own spread, and with the cliff-aware correction.
- A model's own predicted spread per molecule: the Gaussian process, the heteroscedastic Gaussian
  process, the variance-head network, and the natural-gradient boosted model.
- A molecule's value to a nearest-neighbour predictor, as a Shapley value.
- Conformal p-values with a false-discovery-rate rule.
- Which labels sit at the assay limit.
- Which of these read a label at all, and which read only features. State it, because a detector
  that reads no label cannot find a wrong one.
  [CHECKED 2026-10-09 against the code: three read only features. They are the heteroscedastic
  Gaussian process, the variance-head network and natural-gradient boosting. Each predicts a
  spread from features with a model that never saw that molecule. Every other detector reads the
  molecule's own label. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §2.1.]
- CITATIONS AND WHAT IS YOURS, one line per detector, used only if the detector stays in (added
  by Claude 2026-10-09 from the code; not reviewed by the author). * = added to refs.bib on
  2026-10-09. "Yours" lists only what the code does differently. "Precedent"
  is what the 2026-10-08 literature search found. Those searches were not a systematic review.
  Two changes apply to every detector, so state them once, not per line:
  - Every prediction a detector reads is out of fold, with scaffold groups kept together in the
    inner folds (noise_estimation.py:141-197; noisy_labels/core.py:135-186).
    [CHECKED 2026-10-09 against the code: WRONG for four. The out-of-bag residual is plain
    out-of-bag, with no scaffold grouping. cleanlab's internal folds are random. Confident
    learning and small-loss selection use in-sample predictions. The Gaussian process subsamples
    each inner fold to 1,000 molecules at random. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md
    §2.2.]
  - Every neighbour detector works in the space Chapter 6 chooses: neighbour count 5, 10 or 20,
    weighting by rank or 1/d squared, picked by predicting a held-out fifth of the training
    molecules (n07:1906-1981).
    [CHECKED 2026-10-09: same. Rows carry k 5, 10, 20 and both weightings. Two exceptions. The
    Shapley value is outside the weighting sweep by decision. The default detector of the main
    cleanup experiment is the forest residual, so there the screen picks neighbours for the label
    moves, not the detector. Source: CH7_METHODS.md §2.2. [file pointer: n32:1115-1125, n07:550]]
  - Out-of-fold forest residual (noise_estimation.py:635-725). Regression: the cross-validated
    filter of Brodley1999, as a residual. Classification: 1 minus the out-of-fold probability of
    the recorded class, the self-confidence score of Northcutt2021, without its per-class
    thresholds. The code cites neither. Precedent in QSAR: Zhao, Wang, Sedykh and Zhu 2017
    (ACS Omega 2:2805, doi 10.1021/acsomega.7b00274). They removed molecules with large
    cross-validation error, which did not improve prediction of external molecules.
  - Out-of-bag residual (noise_estimation.py:780-843): Breiman2001 out-of-bag prediction. The code
    calls it leave-one-out.
  - Neighbour-label gap, with and without the neighbours' own spread (noise_estimation.py:361-497).
    Classification: the edited-nearest-neighbour rule of Wilson1972 as a score. Regression:
    Martin2021 adapt nearest-neighbour filters to regression on general benchmarks. Dividing by
    the neighbours' spread has no source. Precedent on molecules: none found.
  - Cliff-aware correction (noise_estimation.py:557-628): the gap times 1/(1 + neighbour spread /
    label spread). No source; the formula is yours. The framing is Maggiora2006 and Stumpfe2012.
    The docstring formula at :565 differs from the code at :621. Fix one before describing it.
    [CHECKED 2026-10-09: same, but the code line is :623. Source: CH7_METHODS.md §2.2.]
  - Gaussian process predicted spread (noise_estimation.py:914-1080): Rasmussen2006. The score is
    the posterior spread plus the absolute residual. That sum has no source; it is yours.
    [CHECKED 2026-10-09: same. Add that each inner training fold is capped at 1,000 molecules by
    random subsample. Source: CH7_METHODS.md §2.1. [file pointer: noise_estimation.py:986-993]]
  - Heteroscedastic Gaussian process (noise_estimation.py:1838-2021): ported from qsar_qm_models.
    No paper is named. Closest is kersting2007most, which feeds the learned noise back into the
    GP. This code does not, so cite it as "after", not as the method.
  - Variance-head network: Seitzer2022 beta-NLL, as published. Natural-gradient boosting: Duan2020,
    at 50 stages instead of 500.
  - Conformal p-values with Benjamini-Hochberg (n41:248-360): bates2023testing* and
    benjamini1995controlling*. Two details differ from the source. The scores come from models
    that saw each other's molecules, which breaks the exchangeability the p-values assume. And a
    calibration set of molecules with 3 or more consistent repeats (n41:322-348) is yours.
    Precedent on molecules: none found for conformal outlier tests with FDR control. Conformal
    QSAR itself is xu2024conformal*, which is the "ACS Omega 2024" named at
    noise_estimation.py:1213.
    [CHECKED 2026-10-09: same. Add: some arms calibrate only on molecules the injection did not
    move. That needs the answer, so it is not a method a user can run. Its rows are named
    remove_oracle_clean_* and winsorise_oracle_clean_*. Do not report them as a method. Source:
    CH7_METHODS.md §2.2. [file pointer: n41:306-319]]
  - Shapley value to a nearest-neighbour predictor (n32:995-1075): jia2019efficient*, from
    cooperative game theory via ghorbani2019data*. Yours: the neighbours are found in the hybrid
    and forest-proximity spaces, and each training molecule is its own test point. Jia et al.
    already give an exact recursion for regression (their Theorem 6, utility = minus the squared
    error of the neighbour mean). The code uses minus the mean squared gap per neighbour instead,
    and its comment at n32:1012-1015 wrongly says Jia's recursion needs that. Precedent on
    molecules: Hesse, Boldini and Sieber 2024 (J. Chem. Inf. Model. 64:8142,
    doi 10.1021/acs.jcim.4c01547), on screening classification. None on regression.
  - Cleanlab regression score (noise_helpers.py:762-779): zhou2024detecting*, through the cleanlab
    package. Its internal folds are random, not scaffold-grouped.
  - Labels at the assay limit (n30:293-311): the largest training label, if it holds at least 1
    per cent of the molecules. The rule is yours; no source.
    [CHECKED 2026-10-09: same; the rule also needs at least 2 molecules on that value. Source:
    CH7_METHODS.md §2.1. [file pointer: n30:293-300]]

7.4  Methods: what to do about them
- Changing a label: toward its neighbours' mean, toward an out-of-fold prediction, toward a
  Gaussian-process posterior mean, toward a forest-proximity neighbours' mean, shrunk toward an
  out-of-fold prediction, replaced by its neighbours' mean, clipped onto a leaf quantile range, or
  raised off the assay limit.
  [CHECKED 2026-10-09 against the code and the result files. Two items differ. Source:
  scratchpad/thesis_ch5_8/CH7_METHODS.md §3.1.
  - "Clipped onto a leaf quantile range" has NO results. It was cut on 2026-10-05.
    [file pointer: n07:182-186]
  - "Toward a forest-proximity neighbours' mean" moves EVERY label a quarter of the way, with no
    detector. Its rows exist only in two experiments. [file pointer: n20, n25]
  The rest: same.]
- Deleting molecules.
- Reweighting molecules, which is reported separately because no reweighting method has a control.
  [CHECKED 2026-10-09: no longer true. Down-weighting on the labels as delivered has the
  shuffled-score control, and passes it in 8 rows. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md
  §1d. [file pointer: n18]]
  [DECIDE, the same choice as the OPEN POINT under NAMES FOR THE NOISE METHODS: keep reweighting
  as its own group, or file it under noise mitigation as "count less". Separate keeps this bullet.
  Merged adds the page's 993 reweighting rows to noise mitigation in the overall table (see 7.4a).
  Source: CH7_RESULTS.md §1c.]
- Subtracting an offset per scaffold group.
- Changing the model instead of the labels: averaging ten network seeds, stacking both models,
  adding noise to the targets and bagging, and robust loss functions.
  [CHECKED 2026-10-09: robust losses mostly have NO results. Only the network's Huber loss,
  C-Mixup and one natural-gradient-boosting Student-t pair ran. Tukey, expectile, fair, median and
  pseudo-Huber losses ran on no dataset. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §4.2.
  [file pointer: n33; results on buchwald_hartwig, caco2, herg_ki, logd only]]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_METHODS.md §3.2] The method grouping page
  (answers/_ref/METHOD_GROUPS.md) has wrong rows and is incomplete. Do not copy it into the
  thesis unchecked.
  - Confident learning does not use out-of-fold probabilities. Its ensemble predicts on the
    molecules it was trained on.
  - The model-uncertainty cleaning rows are filed "delete". That experiment also moves labels.
    [file pointer: n23]
  - "Search and fit on cleaned labels" is filed "delete". It ran nine cleanups; seven move a
    label. [file pointer: n20]
  - Putting cliff molecules back is filed "a detector, delete". It undoes part of another cleanup.
  - Two-learner selection is filed noise-robust training. It is a detector plus delete.
  - Missing: DivideMix, the forest-proximity move, and at least 25 other methods with results.
  - "85 methods" counts the analysis page's sections, not methods with results.
  - The three CHECK rows: the constant lift is "a known flag, move the label", but the flag is
    read off the labels' own top value. The measurement-count method weights AND adds a feature;
    its weight half is mitigation. [file pointer: n18, n07, n19]
- [NEW 2026-10-09, same source item 7] The QM9 control page files the removal-cost experiment as
  "a detector, delete". It is random deletion with no detector. Its QM9 rows belong beside the
  random-deletion control. [file pointer: n27:17-24; answers/_scripts/qm9_control.py:52]
- Cleaning before the feature search, which no cited source does. See 7.9.
- CITATIONS AND WHAT IS YOURS, one line per treatment, used only if it stays in (added by Claude
  2026-10-09 from the code; not reviewed by the author). Same conventions as 7.3.
  - Toward the neighbours' mean, and replacing a flagged label by it (n07:1103-1168): kNN
    smoothing (Nadaraya-Watson; nadaraya1964estimating* and watson1964smooth*, BibTeX checked
    against OpenAlex only). The replacement form is edited nearest neighbour (Wilson1972) adapted
    to regression. Yours: the neighbour space. Precedent on bioactivity labels: none found.
  - Toward an out-of-fold prediction (n07:1247-1264): closest is the bootstrapping target blend of
    Reed et al. (reed2014training*, arXiv only). The step is scaled by the detector score.
    Precedent on molecules: none found.
  - Toward a Gaussian-process posterior mean (n07:1266-1343): Rasmussen2006 as a denoiser,
    out of fold. Precedent on bioactivity labels: none found.
  - Toward the mean of the forest-proximity neighbours (n07:1170-1237): Breiman2001 proximity
    used as the smoothing kernel. No source for the smoothing step; yours.
    [CHECKED 2026-10-09: it reads no detector and moves every label. Rows exist only in n20 and
    n25; n07 leaves it out by decision of 2026-10-06. Source: CH7_METHODS.md §3.1.]
  - "Shrunk toward an out-of-fold prediction" (js_shrink, n07:1049-1101). The code cites
    james1961estimation* and efron1975data*. DO NOT DESCRIBE IT AS JAMES-STEIN. The variance it
    estimates cancels, so the fraction moved is 0.5s/(1 + 0.5s), where s is the detector score
    scaled to 0 to 1 (n07:1094-1098). It is a score-scaled blend, the same family as the
    out-of-fold blend above. n19's shrink_toward (n19:600-621) is a real James-Stein form with a
    model prediction as the prior.
  - Clipped onto a leaf quantile range (n07:1383-1400): the ranges are Meinshausen2006 quantile
    forests. Clipping each label to its own range has no source; yours.
    [CHECKED 2026-10-09: NO RESULTS; cut 2026-10-05. Drop this line unless it is re-run. Source:
    CH7_METHODS.md §3.1.]
  - Raised off the assay limit (constant_lift, n07:1345-1381; n30:331-394): a Tobit imputation
    (Tobin1958). The lift is the limit + sd x inverse Mills ratio. That is larger than the textbook
    truncated-normal mean by (limit - prediction), as n30:91-97 says. If kept, describe it as a
    variant and say how it differs. The iterated version (censored_em) is after
    buckley1979linear*. Precedent: Tobit on censored drug-discovery labels is Svensson et al. 2025
    (Artif. Intell. Life Sci. 7:100128, doi 10.1016/j.ailsci.2025.100128).
    [PLANNED 2026-10-09] For a LOWER limit (ginkgo_pr_cho, 7.7) the same moves run on the negated
    labels, so the floor becomes a pile-up at the top and n30's code runs unchanged; the moved
    labels are negated back. Held-out labels at the floor are censored too, so that experiment
    adds Harrell's C, a concordance that skips pairs whose order the floor hides [CITATION TO
    CHECK: Harrell et al. 1982 or 1996; not in refs.bib].
  - Deleting molecules: Brodley1999 filtering, tested in QSAR by Zhao et al. 2017 (see 7.3).
    Yours: the matched random-deletion control.
  - Reweighting by 1/(1 + score) (n07:966-971): no source. Huber weights (n07:973-998):
    Huber1964, one pass, with the residual taken from a random forest rather than from the model
    being fitted. The comment at n07:974 calls Huber's weight "redescending". It is not; that word
    belongs to Tukey's.
    [CHECKED 2026-10-09: Huber weights have rows only in n24, n25 and n42, not n07. Source:
    CH7_METHODS.md §3.1.]
  - Weighting by measurement count or by each molecule's error variance (n07:1000-1047;
    n19:496-699): inverse-variance weighting, hedges1985meta*. Moderating each molecule's variance
    toward the pooled one (n19:664-689) is limma-style empirical Bayes, smyth2004limma*.
    Precedent for weighting molecules this way in QSAR: none found.
  - Subtracting an offset per scaffold group (n34:223-300): the random-intercept version is MERF
    (hajjem2014mixed*) and GPBoost (sigrist2022gaussian*). Yours: the residuals come from an
    out-of-fold model, and the offsets are removed before the downstream model is refit. The
    per-document version (n29) follows the annotator-bias model of raykar2010learning*.
    Precedent: no mixed-model assay or document offset on ChEMBL-type data found. Landrum and
    Riniker 2024 (landrum2024) measure the disagreement between sources without modelling it.
  - Robust losses (n33): Huber1964; Tukey's biweight beaton1974fitting*; Student-t likelihood
    lange1989robust*; expectiles newey1987asymmetric*; C-Mixup yao2022cmixup*. Yours: the
    threshold is swept in units of the residual MAD. Precedent in QSAR: robust linear regression
    (Konovalov et al. 2008, J. Chem. Inf. Model. 48:2081, doi 10.1021/ci800209k). No robust loss
    on bioactivity labels found. C-Mixup has drug-affinity precedent (MixingDTA 2025).
    [CHECKED 2026-10-09: Tukey, expectiles and Student-t are cited for methods that did not run.
    Student-t ran once, as one natural-gradient-boosting pair. Only the network's Huber loss ran
    among the loss changes. Source: CH7_METHODS.md §4.2.]
  - Iterative trimmed loss (n32:1214-1241): shen2019learning*. Precedent in QSAR: none found.
  - Ten seeds, stacking, target-noise bagging (n33, n40): standard ensembling. Breiman's stacked
    regressions (1996) and randomized outputs (2000) are the usual references; neither is verified
    or in refs.bib yet.
    [CHECKED 2026-10-09: target-noise bagging is in n33; ten seeds and stacking are in n40.
    n40 results: caco2, herg_ki, logd only. Source: CH7_METHODS.md §4.3.]
  - Putting cliff molecules back after cleaning (n25:868-944): no source found; yours. The cliff
    definition is Maggiora2006 and Stumpfe2012.
  - Cleaning before the feature search (n20): no source found; yours.
  - PROVENANCE TO STATE OR AVOID. Several of these treatments were added from AI-written
    literature reports, named in the code at n07:187, n29:13, n30:21, n31:6, n33:9 and n40:7.
    The reports are not citable. Every method above is cited to its original paper instead.

7.4a  Methods: how each group is tested, and QM9 as the clean control
Added 2026-10-09. The author approved the control scheme below; items marked TENTATIVE are to be
tried and looked at, and dropped if QM9 turns out to be a bad comparison for some reason.
- WHY QM9. The QM9 HOMO-LUMO gap labels are computed, not measured, so they carry no assay
  error. It is the only dataset in the noise work with clean labels, and the same dataset
  Chapter 3 rests on. A noise method that improves QM9 with no noise added is not removing
  noise. It is changing the model in a way that would help on any labels, or the model is
  fitting something else. QM9 was added to the noise runs in this round; 14 noise jobs had
  finished and 47 were running on 9 Oct.
  [CHECKED 2026-10-09 against the ledger at 13:10: same, if the 3 not-applicable records count as
  finished. 11 have results. 5 more were skipped. Experiments needing repeats, documents or curator
  flags will never run on QM9. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §5.1.
  [file pointer: not applicable n22, n29, n36; skipped n17, n19, n26, n32 x2]]
- EACH GROUP HAS ITS OWN TEST. The three groups (NAMES FOR THE NOISE METHODS) do not share one.
  - Detection: is the suspicion score high on the molecules that were damaged? Readable only
    where damage was added, because only then is it known which molecules were damaged. On QM9
    with no noise added, every flag is a false flag.
  - Mitigation chosen by a detector: held-out score after the action against the same model on
    the same features with no action. Control: the same action on the same number of molecules
    chosen at random.
  - Mitigation chosen by a known flag: the same comparison and the same random control. QM9 has
    no real flags (no assay limit, one value per molecule), so these are tested on QM9 only under
    added damage.
    [CHECKED 2026-10-09: no random control of this kind has run for a known flag. The only
    clipped-label control deals the same lifts to other clipped molecules. No experiment raises
    the same number of unclipped labels. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §2c.]
    [PLANNED 2026-10-09: ginkgo_pr_cho is the one test on a real flag, the 58 labels at the PR_CHO
    floor of 0. It carries the first control of this kind: the same move on the same number of
    labels not at 0. Replace the CHECKED note above when its results exist. See 7.7.]
  - Mitigation on every molecule: compared with no treatment. No random-selection control is
    possible, so QM9 with no noise added is the only check these methods have.
  - Noise-robust training: compared with the same model trained normally on the same labels. Its
    gain on QM9 with no noise added is a gain from the training change alone; set it beside the
    gain under damage.
- THE QM9 ROW IN THE RESULTS (author: yes, one of the most important cases, because the labels
  are clean). Every results table carries each method's QM9 score with no noise added, before
  and after the method. A separate table gives one row per method with that QM9 number.
- TENTATIVE: a gain on a damaged dataset is shown beside the same method's QM9 gain with no noise
  added. Both numbers are printed; neither is subtracted from the other.
- TENTATIVE: the neural network's results are shown beside its random-deletion result, to show
  how far its score moves between fits. Check this before adopting it.
- FIRST LOOK, 9 Oct, QM9 fold 0, no noise added, held-out RMSE. A reading by Claude of the
  per-row result files, on /Volumes/seagate/claude_analysis_2026-10-08/answers/QM9_CONTROL.md. A
  setting is one configuration of a method on one representation.
  - Deleting the molecules a detector flags never helped [n07]: 0 of 12 neural-network settings
    on the best hybrid, 0 of 6 lightgbm settings.
  - Moving a flagged label toward its neighbours' mean [n07] helped the neural network on the
    best hybrid in 10 of 12 settings, at best 0.3568 to 0.3395.
  - Deleting molecules at random [n24] helped the neural network on the best hybrid in 9 of 90
    settings, at best 0.3568 to 0.3440.
  - The neural network's untreated score is not the same in every job: on the second hybrid it is
    0.3484, 0.3568 or 0.3618 depending on the job. lightgbm's is the same in every job (0.3759).
    A neural-network gain smaller than that spread cannot be read on QM9.
  [CORRECTED 2026-10-09: an earlier version of this bullet counted each no-noise setting once per
  damage-kind file, which stores its own copy, so its counts were too high.]
  [CHECKED 2026-10-09 against the raw rows: all four counts same. Add lightgbm's first hybrid,
  0.3719 in every job. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §5.2.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_RESULTS.md §5] Label smoothing helps the
  network's hybrid by a median of about 0.003 RMSE. The job-to-job spread is up to 0.013. The best
  setting, 0.3568 to 0.3395, is picked on the test score.
  [NEW 2026-10-09, same source §5] Deleting flagged molecules in the delete-or-keep experiment:
  0 of 18 settings improved the hybrid, both models. [file pointer: n24]
  [NEW 2026-10-09, same source §5] Still missing on QM9 with no noise added: n11, n20, n23, n25,
  n31, n33, n34, n35, n39 and n40 have no QM9 row.
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_METHODS.md §5.3] QM9 with no noise added, the
  twelve published methods, network, fold 0. Ten of the twelve are never better than the plain
  network on the hybrid. DivideMix is the worst: 0.4517 against 0.3550 RMSE on the hybrid.
  Confident learning beats the plain network on 4 of 5 feature spaces in one job and 1 of 5 in
  another. The labels are the same; the per-fold search differs. [file pointer: n09]
  [CONFLICT: CH7_RESULTS.md §5 reads the final files and says all twelve make QM9 worse, single and
  hybrid, in that run. A re-run gives four of them gains of 0.001 to 0.021 RMSE, inside the
  network's spread. [file pointer: n09, n13]]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_RESULTS.md §1] THE "DOES IT WORK, BY GROUP"
  TABLE (author: put it in the chapter; highlight its noise mitigation row most). Source page:
  /Volumes/seagate/claude_analysis_2026-10-08/answers/MEETING_METHODS.md, built 9 Oct 13:45.
  - One row of the source page is one dataset, one noise type and one model. Each compares the
    best single representation and the best hybrid, before and after the method.
  - "Yes" means the method improved both the single representation and the hybrid. Where a
    control exists, it also beat the control. There is no minimum size.
  - The page's own counts, recounted and reproduced exactly. Rows by verdict:
    - training with noisy labels: 653 yes (hybrid better 507, worse 146), 297 single only, 304
      hybrid only, 477 no; 1,731 rows; 143 strong.
    - reweighting: 544 yes (396 / 148), 109, 156, 184; 993 rows; 93 strong.
    - noise mitigation: 691 yes (544 / 147), 163, 190, 490; 1,534 rows; 187 strong.
  - "Strong" means the smaller of the two gains is at least 5 per cent of the starting RMSE, +0.05
    Spearman, or +0.01 AUROC. That column is not on the page; it was added from the same parse.
  - Under the outline's three groups, rows move. All 993 reweighting rows join noise mitigation as
    "count less". 506 rows move from training with noisy labels to noise mitigation: the four
    selection methods of the twelve, shrinking flagged labels, the Huber-teacher blend, Yeo-Johnson
    and the measurement-count shrink. None move the other way.
    - noise mitigation: 1,374 yes (hybrid better 1,043, worse 331), 359 single only, 426 hybrid
      only, 874 no; 3,033 rows; 313 strong.
    - noise-robust training: 514 yes (404 / 110), 210, 224, 277; 1,225 rows; 110 strong.
  - The regrouping uses answers/_scripts/method_groups.json, a Claude reading of the code not
    checked against the code. Four sections had no entry and were classified by hand.
  - CAVEATS, to print with the table:
    - The best setting of each method is picked on the held-out score. So is the single
      representation and the hybrid in each row. Every count leans optimistic.
    - Fold 0 only.
    - In 585 of the 1,374 noise mitigation "yes" rows, the smaller gain is under 1 per cent of
      the starting RMSE, or under 0.01 Spearman or AUROC.
    - 114 rows with no score count as "no"; 84 of them are in the page's noise mitigation group.
    - "Hybrid worse" includes 5 exact ties.
    - Only 9 mitigation methods have a control column with values. 68 rows pass both the method
      and its control. Three control columns are empty.
  [DECIDE: which version of the table goes in the chapter. The page's three groups match the
  source page as built. The outline's three groups match NAMES FOR THE NOISE METHODS, but rest on
  an unchecked regrouping. The noise mitigation row is 1,534 rows in the first and 3,033 in the
  second.]
  [DECIDE: print a "strong" column, a "below 1 per cent" column, both, or neither. Without them,
  a 0.228 to 0.228 rounded gain counts as yes.]

7.5  Results: changing a label against deleting a molecule
- Moving a flagged label toward its neighbours' mean or toward an out-of-fold prediction worked on
  the labels as delivered and under damage, on Caco-2, hERG, hERG Ki and LogD. hERG Ki with the
  network on the hybrid: 0.5989 to 0.5671 RMSE.
  [CHECKED 2026-10-09 against the 9 Oct results: number same, verdict CHANGED. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §3, 7.5 bullet 1. [file pointer: n24, n18]
  - The 0.5989 to 0.5671 is in two damage-kind files. The censoring file's own level-0 copy
    reads 0.5972 to 0.5706.
  - On the labels as delivered the single representation did not improve: 0.5892 to 0.5965 at
    best. So this row is "hybrid only".
  - Against shuffled detector scores on the labels as delivered: hERG Ki passes, network only.
    Caco-2 and LogD pass by 0.012 RMSE or less, network only. hERG passes with gradient-boosted
    trees, by 0.009 and 0.005 AUROC.
  - Under damage it passes for the network on Caco-2, hERG Ki and LogD, and for both tree models
    under every hERG flip kind. The support vector regressor moves RMSE by 0.0053 or less.]
- Deleting works in narrow cases. Under the matched random-deletion control it fails on the labels
  as delivered of Caco-2, hERG, hERG Ki and LogD. It passes under hERG's label flips, where the
  gradient-boosted trees on the hybrid go 0.8028 to 0.8375 AUROC against 0.8179 for the best random
  deletion, and under LogD's censoring and outliers.
  [CHECKED 2026-10-09: labels as delivered, fails on all four: same. hERG 0.8028 to 0.8375
  against 0.8179: same, 3 draws. LogD CHANGED to "hybrid only": censoring 0.622 to 0.609,
  outliers 0.585 to 0.546, network on the hybrid. The single representation fails or loses to
  random deletion. hERG Ki under censoring passes with the support vector regressor, but only
  0.760 to 0.757 and 0.746 to 0.745. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3, 7.5 bullet 2.
  [file pointer: n24]]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_RESULTS.md §2c result 5] Buchwald-Hartwig,
  labels as delivered: deleting flagged reactions beats random deletion. lightgbm on the hybrid,
  20.0823 to 18.0417 RMSE against 19.8986 for the best random deletion. On the second base
  representation, 20.2242 to 18.3284 against 19.9173. Neighbour-label gap, 20 per cent flagged,
  fold 0. Random deletion alone helps the first base representation, 19.4875 to 18.8663, so the
  score moves between fits here too. [file pointer: n24]
- DO NOT CLAIM A WINNER BETWEEN THE TWO FAMILIES. Counting the 198 picked methods, deleting
  improved 171 of a possible 264 model-and-configuration pairs on the labels as delivered, and
  changing a label 813 of a possible 1,320. Those rates are 0.648 and 0.616 per pair. What
  separates the families is that deleting is the only one with a control.
  [CHECKED 2026-10-09: NOT FOUND. No current page or script produces the 198, 171 of 264 or 813
  of 1,320. How "picked" was defined is not written down. Nearest counts, small molecules, labels
  as delivered, rows that helped both: delete 33 of 123, move the label 29 of 154. All datasets
  and noise types: a detector then delete, 413 of 986; a detector then move, 284 of 750. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §1c, §2d. "No clear winner" still holds.]
  [CHECKED 2026-10-09: "deleting is the only one with a control" is out of date. Moves and
  down-weighting on the labels as delivered have the shuffled-score control. Source: same, §1d.]
- The exception to everything above: on the Ginkgo antibody sets, removing flagged antibodies is
  the strongest result in the chapter. See 7.8.
  [CHECKED 2026-10-09: CHANGED in wording. It is the strongest CONTROLLED result: SMAC colloidal
  stability, support vector regressor, 0.2740 to 0.4034 Spearman against 0.2782 shuffled. Larger
  gains on SMAC and titer come from the detector-by-action grid and from cleaning before the
  search, neither with a control. Source: CH7_RESULTS.md §3, 7.5 bullet 4. [file pointer: n18,
  n11, n20]]
- [FIGURE F8, two grids side by side. Rows are selected methods, named in words. Columns are
  datasets. Left grid is the labels as delivered, right grid is injected damage. Shade is the share
  of the 8 model-and-configuration pairs the method improved. NOT a yes or no, and NOT a share of
  damage recovered, because the standing caveat above makes recovery unreadable for a hybrid.]
  [CHECKED 2026-10-09 against thesis_figures.py: the current F8 differs in two ways at once. It
  writes one file per experiment, and its rows are grouped by detector, not by action. Four of
  its ten files are wider than the page with the 9 Oct results. It does not compute the 171 of
  264 or 813 of 1,320. Source: scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md §0 item 1, §A.3.]

7.6  Results: which detectors find the suspect labels
- The out-of-fold forest residual and the neighbour-label gap drive almost every success.
  [CHECKED 2026-10-09: CHANGED. Partly by construction: four of the main cleanup experiments offer
  only these two detectors. Where others were offered, they also pass: the four model-uncertainty
  detectors pass in 31 to 38 of 52 rows each. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3,
  7.6 bullet 1. [file pointer: n07, n18, n24, n25; n23]]
  [CHECKED 2026-10-09: CHANGED in emphasis. PATTERNS.md says that where both ran side by side
  with a control on small molecules, the forest residual beat the control less often in all
  three experiments. Pairs that improved and beat the control, neighbour gap against forest
  residual: 51 of 168 against 21 of 168 (cliff put-back); 28 of 144 against 18 of 144 (labels as
  delivered); 155 of 480 against 53 of 160 (delete or keep; the denominators differ). The gap
  damped by the neighbours' spread did best in the cliff experiment, 64 of 168. Source:
  scratchpad/thesis_ch5_8/PATTERNS.md §4.6. [file pointer: n25, n18, n24]]
- A Gaussian process's own predicted spread worked on hERG and hERG Ki.
  [CHECKED 2026-10-09: CHANGED, wider. Rows that helped both: Caco-2 9 of 14, hERG 12 of 14,
  hERG Ki 8 of 12, LogD 7 of 12. 13 rows are strong. Labels as delivered: yes on 1 hERG row and
  1 hERG Ki row. No control. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3, 7.6 bullet 2.
  [file pointer: n23]]
- The three detectors that predict the error from the features alone work only when the damage is
  tied to structure, which is the per-scaffold-group offset. Under damage drawn at random they are
  undefined, and the files say so.
  [CHECKED 2026-10-09: NOT FOUND on any current page. Under the scaffold-group offset every
  molecule moves, so no detection score exists there either. The Caco-2 file records an empty
  detection block. So "work" here can only mean the cleaning result, not detection. The
  "detection_undefined" marks in the conformal experiment are for a different reason: those
  kinds move every label. Source: scratchpad/thesis_ch5_8/CH7_METHODS.md §2.1; CH7_RESULTS.md §3,
  7.6 bullet 3. [file pointer: n23, n41]]
- FINDING THEM IS NOT FIXING THEM. On Caco-2, the Shapley-value detector under forest proximity
  with 10 neighbours reached 0.8106 to 0.8921 AUROC at a clipped share of 0.25, against 0.4666 to
  0.5475 for the plain out-of-fold residual. At a flip rate of 0.1 it reached 0.9577 to 0.9681
  against 0.6205 to 0.7274. No removal or shrinking method beat the damaged labels on half the
  pairs under any damage kind. [FIGURE F10]
  [CHECKED 2026-10-09: AUROC ranges same, but read from a checkpoint; no final Caco-2 censoring
  file exists. "At a flip rate of 0.1" is WRONG: Caco-2 has no flips. It is a clipped share of
  0.1. "No method beat the damaged labels" is CHANGED; it is now true of Caco-2 only. Shrinking
  passes under damage on buchwald_hartwig, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal,
  ginkgo_titer, herg_ki and logd. Trimmed refits pass on buchwald_hartwig, flab_thermo_tresanco,
  ginkgo_colloidal and logd. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3, 7.6 bullet 4.
  [file pointer: n32]]
  [CHECKED 2026-10-09: F10 reads unfinished checkpoints. It cannot show the Shapley numbers,
  because no action reads that detector. Its axis still says "repair". Source:
  scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md §A.3.]
- Conformal p-values with a false-discovery-rate rule flag too few molecules to act on, and the
  floor on how few it can flag is derived and written into each record.
  [CHECKED 2026-10-09: "too few" is CHANGED. It flags at q 0.05 or 0.1 on buchwald_hartwig, caco2
  and logd, not herg_ki. Acting on the flags fails on all four: same. "The floor in each record"
  is NOT FOUND on the page. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3, 7.6 bullet 5.
  [file pointer: n41]]

7.7  Results: by noise type
- [OPEN, author 2026-10-09: "the noise mitigation methods that work on censoring should work on
  this"] REAL CENSORING IN THE LABELS AS DELIVERED. ginkgo_pr_cho has 58 of 197 labels exactly at
  the assay floor (antibody_check/GINKGO.md). n30 never ran on it, handles injected censoring only,
  and flags only a pile-up at the TOP label (n30_censored_likelihood.py:136-146, 287-300). The
  same file names adme_hlm (958 of 3,087 labels at the minimum) and adme_rlm (346 of 3,054) as
  possible real floors, undecided. Needed to test it: a lower-limit side, a real-labels mode with
  no injection, and a held-out score not corrupted by the same floor (score the held-out molecules
  above the floor as well as all of them).
  DESIGN, approved by the author 2026-10-09 (HANDOFF_CENSORING_PRCHO.md):
  - THE FLOOR. 58 of 197 PR_CHO labels are exactly 0. None is below 0; the next lowest is 0.0171.
    The PROPHET-Ab preprint (biorxiv 10.1101/2025.05.01.651684) says the bead MFI values "were
    normalized between 0 to 1 based on three reference antibodies", with elotuzumab as the low
    reference. It does not say that values below the low reference are set to 0, and gives no
    detection limit. So 0 is treated as a lower limit: "no more polyreactive than the low
    reference". The clipping is inferred from the labels, not stated by Ginkgo. PR_Ova, from the
    same assay against ovalbumin, has 51 of 197 at 0.
  - HELD-OUT ANTIBODIES AT 0, by outer fold: 21 of 45, 15 of 40, 11 of 36, 4 of 35, 7 of 41.
  - TREATMENTS, each fitted on the labels as delivered and scored against no treatment on the same
    held-out antibodies. The labels at 0 are moved below 0 by constant lift, censored EM,
    truncated-normal mean, and per-molecule lift with its shuffled-lift control. Also included:
    winsorising toward an out-of-fold prediction, and deleting the antibodies at 0.
  - NEW CONTROL: the same downward move applied to the same number of training labels NOT at 0,
    chosen at random (3 seeds). It tests whether knowing which labels sit at the limit is what
    produces the gain.
  - The censored-likelihood XGBoost (AFT) does not run: xgboost is not one of this dataset's two
    decided models. Censored EM covers the idea with gp and dnn.
  - FEATURES: per outer fold and model, the best and second-best single representation and the
    first and second general hybrid, from that fold's search. MODELS: gp (searches re-run with the
    gp fixed on 8 Oct) and dnn. All 5 outer folds.
  - SCORES, per fold with the spread across folds: Spearman on all held-out antibodies; Spearman
    on the held-out antibodies above 0; Harrell's C with antibodies at 0 treated as censored
    [CITATION TO CHECK].
  - adme_hlm and adme_rlm: not run. adme_hlm is out of the thesis.
  [RESULTS PENDING: scratchpad/thesis_ch5_8/PRCHO_REAL_CENSORING.md. Per treatment: single
  representation before and after, hybrid before and after, for gp and dnn, under each of the
  three scores. Then set beside the injected-censoring result.]
- Clipping at a detection limit is the most fixable. Raising the clipped labels worked on Caco-2,
  hERG Ki and LogD. hERG Ki with the network on the second base representation: 0.7485 to 0.6662
  RMSE at a clipped share of 0.25. [FIGURE F9, the four-step ladder: the labels as delivered, after
  clipping, clipped then corrected, clipped molecules deleted. One panel for Caco-2 and one for
  hERG Ki.]
  [CHECKED 2026-10-09: 0.7485 to 0.6662 same, fold 0, one draw. Wider: the censored-likelihood
  correction is strong on both models on caco2, herg_ki, logd, qm9, flab_thermo_tresanco, and
  buchwald_hartwig with the forest. It fails on ginkgo_colloidal and ddg_combined. LogD, network
  on the hybrid: 0.4059 before clipping, 0.6049 clipped, 0.4604 after the constant lift. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §2c result 1. [file pointer: n30]]
  [CHECKED 2026-10-09: "most fixable" holds only if the selection is named. Pooled over every
  method, small molecules, clipping is helped less often than Gaussian or outliers: 120 of 317
  rows against 128 of 233 and 158 of 265 (PATTERNS.md §4.1, all methods). Mitigation only: 84 of
  234 against 86 of 161 and 104 of 169 (CH7_RESULTS.md §2d). Moving labels by the clip flag: 38
  of 38 (PATTERNS.md §4.1). The two files count different pages; both give the same direction.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_RESULTS.md §2c] The per-molecule lift passes
  its control in 1 of 22 rows. The control deals the same lifts to other clipped molecules. On
  every clipped row of caco2, herg_ki, logd and qm9 the shuffled lifts score better. So the gain
  comes from knowing which labels are clipped and raising them by about the right average amount.
  No experiment raises the same number of unclipped labels, so the flag itself is untested.
  [file pointer: n30]
- Why it is the most fixable: no detector is needed. The clipped labels are visible, because they
  share the dataset's largest value. Contrast this with Chapter 3, where models were MOST certain
  about exactly the labels that had been clipped.
- An offset per scaffold group is the next most fixable. Subtracting a shrunk offset worked on
  Caco-2 and hERG Ki. Caco-2 with the network on the hybrid: 0.3328 to 0.2866 RMSE. These methods
  did nothing on the labels as delivered, which is the right answer, because nothing had shifted a
  group.
  [CHECKED 2026-10-09: number CHANGED, gain bigger. Random intercept, network, 3 draws: hybrid
  0.3423 to 0.2800; second hybrid 0.3329 to 0.2798. Clean labels give 0.2374. Results exist on
  caco2, herg_ki and logd only. "Did nothing on the labels as delivered" is WRONG: they made
  Caco-2 and LogD WORSE. Caco-2, network on the hybrid: 0.238 to 0.263. LogD: 0.395 to 0.483.
  One exception: centring groups of 3 or more on hERG Ki, network, 0.638 to 0.618 single and
  0.612 to 0.589 hybrid, no control. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §2c result 3,
  §3 7.7 bullet 3. [file pointer: n34]]
- Gaussian damage and outliers were fixed less often, and most of their passes come from moving a
  label rather than deleting a molecule.
  [CHECKED 2026-10-09: CHANGED. Small molecules, mitigation, rows that helped both: Gaussian 86 of
  161 (delete 36, move 29, count less 21); outliers 104 of 169 (delete 43, move 39); clipping 84 of
  234. So they are fixed less STRONGLY, not less often: strong rows 7 and 10 against 30 for
  clipping. Deleting passes about as often as moving. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §3, 7.7 bullet 4.]
  [CHECKED 2026-10-09: F11's five default methods are all mitigation. None is noise-robust
  training, and colours are per method, not per group. Source: FIGURES_AND_STYLE.md §A.3.]
- [FIGURE F11, tentative on the full folds. Score against damage level, two panels, four or five
  lines per panel.]

7.8  Results: by dataset
- SMAC colloidal stability is the standout on the labels as delivered. Removing or relabelling the
  flagged antibodies beat both no cleaning and the same detector with its scores shuffled on 7 or 8
  of the 8 model-and-configuration pairs. The support vector regressor on the first base
  representation goes 0.2740 to 0.4034 Spearman against 0.2782 for the best shuffled control. 112
  settings beat the shuffled control. Several of its starting scores are near zero, and each fold
  holds out about 50 antibodies. Under damage it also passes four damage kinds.
  [CHECKED 2026-10-09: 0.2740 to 0.4034 against 0.2782 same, forest-residual detector, 20 per cent
  flagged, fold 0. 112 same. Fold 0 holds out 53 of 242 antibodies. "7 or 8 of 8 pairs" NOT
  FOUND: at 20 per cent flagged, removal beats the shuffled control on 5 of 8 pairs. The summary
  page picks 5 per cent flagged for this dataset and shows 0.274 to 0.352. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.8 bullet 1, §6 item 2. [file pointer: n18]]
  [NEW 2026-10-09, same source §4a] The network's untreated score on SMAC moves a lot between
  jobs. Single representation 0.065 or 0.097 Spearman. The shuffled control alone lifts the
  network's single representation from 0.065 to 0.199.
- THE PAIR THAT CARRIES THE CHAPTER'S ARGUMENT. SMAC colloidal stability and HIC retention hold
  the SAME 242 antibodies, measured on different endpoints. On SMAC, cleaning is the strongest
  result in the chapter. On HIC, neither hybrid beats the single best representation and there is
  no cleaning result at all. Same molecules, same representations, opposite answer.
  [CHECKED 2026-10-09: ginkgo_hic has no noise result file of any kind. So the "opposite answer"
  cannot be made for cleaning yet. It rests on HIC's hybrid result alone. The cleaning half is NOT
  YET RUN. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.8 bullet 2; PATTERNS.md §5.]
  [DECIDE: run cleaning on HIC before writing this pair, or write the pair on the hybrid result
  only. Running it lets the sentence stand as written. Not running it means dropping "cleaning is
  the strongest result" from the HIC side.]
- hERG Ki is the most consistent dataset. Relabelling cleaned the labels as delivered: the network
  on the first base representation goes 0.6380 to 0.5775 RMSE. Putting activity-cliff molecules
  back after cleaning beat putting random molecules back on all 8 pairs, which is a distinct and
  controlled result.
  [CHECKED 2026-10-09: relabelling same, 0.638 to 0.578 single and 0.599 to 0.578 hybrid, against
  shuffled 0.6419 / 0.5877. It is the only controlled gain over 0.012 RMSE on small-molecule real
  labels. Cliff put-back, network: 0.607 to 0.591 single, 0.596 to 0.579 hybrid, against random
  0.6005 / 0.5949. "All 8 pairs" NOT RE-DERIVED. "Most consistent" is not supported: mitigation
  yes share 0.42 for hERG Ki, 0.41 LogD, 0.44 Caco-2. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.8 bullet 3, §4b. [file pointer: n18, n25]]
- hERG: deleting and relabelling work under every flip kind and beat random deletion. 397 settings
  beat the random-deletion control and 15 did not.
  [CHECKED 2026-10-09: deleting same, 397 of the 412 settings that beat no cleaning. Relabelling
  has NO random-deletion control; its passes are uncontrolled. Source: CH7_RESULTS.md §2c
  result 2. [file pointer: n24]]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/PATTERNS.md §4.2] hERG under flips: down-weighting
  the molecules a detector flags helped both in 93 of 104 rows. Deleting them helped in 58 of 153.
  No control. [file pointer: n21, n12]
- Caco-2: many passes, most gains small. Anything under about 0.014 RMSE sits inside the cache
  fault's spread.
  [CHECKED 2026-10-09: same. 237 of 533 rows help both, 23 strong. Labels as delivered: 0 strong.
  Source: CH7_RESULTS.md §3 7.8 bullet 5.]
- Binding free energy and expression titer: large gains on the labels as delivered, and no
  shuffled-score control has run on either. State the missing control in the same sentence as the
  gain.
  [CHECKED 2026-10-09: CHANGED. The shuffled-score control now exists for both, and some cleanups
  pass it. Binding free energy, network, label smoothing: 0.341 to 0.484 single, 0.259 to 0.287
  hybrid, against 0.4595 / 0.2759. Titer, forest, move toward a Gaussian-process prediction: 0.308
  to 0.364 and 0.224 to 0.390, against 0.3116 / 0.2861. On both, the single representation beats
  the hybrid in most cleaned rows. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.8 bullet 6,
  §4a. [file pointer: n18]]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_RESULTS.md §4a and PATTERNS.md §5] THE
  ANTIBODY EXCEPTION, named. Share of mitigation rows that are strong, labels as delivered in
  brackets: cdr_flexibility 0.25 (7 of 35), ginkgo_pr_cho 0.27 (2 of 6), abcov_kd 0.25 (3 of 20),
  ginkgo_titer 0.24 (9 of 48), ginkgo_colloidal 0.12 (18 of 73). For comparison: caco2 0.04,
  herg_ki 0.07, logd 0.07.
  - With controls: ginkgo_colloidal, the strongest. ginkgo_titer, cdr_flexibility and abcov_kd
    are partly controlled. ddg_combined passes the shuffled control with four cleanups.
  - Without a control: ginkgo_pr_cho.
  - Not exceptions (CH7_RESULTS.md): humanness_therapeutic, flab_thermo_tresanco, ginkgo_acsins.
  - [CONFLICT: PATTERNS.md §5 counts pairs that beat the control and lists flab_thermo_tresanco
    (42 of 96) among the controlled exceptions. It lists ginkgo_acsins as an uncontrolled one,
    down-weighting 12 of 13 rows. CH7_RESULTS.md counts strong shares and lists neither.]
  - Caveat, to sit beside any "far better" sentence: the strong rule asks +0.05 Spearman whatever
    the held-out set's size. With about 53 held-out antibodies one fold's Spearman moves by more
    than that between fits. So the antibody strong shares are inflated by the rule itself.
  - Domain and size are confounded: every antibody set has 141 to 672 molecules, every
    small-molecule set 1,412 to 9,978 (PATTERNS.md §4.5).
  - No damage result with a control exists for any antibody dataset (PATTERNS.md §4.5).
  [DECIDE: which antibody datasets the chapter names as the exception. Using strong shares names
  cdr_flexibility, pr_cho, abcov_kd, titer and colloidal. Using pairs beating the control names
  colloidal, flab_thermo_tresanco, titer and ddg_combined.]

7.9  Results: cleaning inside the feature search
This question has no cited precedent anywhere, and it is the most original result in the chapter.
- Four conditions crossing which labels the features are chosen from with which labels the model is
  fitted on.
- All three of its questions are yes for both models on all five datasets that ran: Caco-2,
  binding free energy, SMAC colloidal stability, expression titer and hERG Ki. The largest is SMAC
  colloidal stability with the network on the second hybrid, -0.0143 to 0.5596 Spearman, from
  shrinking labels toward an out-of-fold prediction.
  [CHECKED 2026-10-09: CHANGED, wider. 9 datasets have results now. buchwald_hartwig,
  cdr_flexibility, herg and logd are added, on the labels as delivered only. 83 of 84 answered
  cells are yes. The one no: buchwald_hartwig, "search on cleaned labels too", second model.
  -0.0143 to 0.5596 is on the page; it is the "search on cleaned labels, fit on the labels as they
  are" question, cleaned in the second base representation. Its starting score is near zero.
  Caveat: a dataset is yes if any one of about 40 settings wins on half the configurations. So
  the rule passes easily. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.9 bullet 1.
  [file pointer: n20]]
  [CONFLICT: CH7_METHODS.md §6 says n20 files exist for 8 datasets; CH7_RESULTS.md says 9.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_METHODS.md §6] A separate experiment compares
  a hybrid searched on clean labels with one searched on damaged labels, on 6 datasets, 3 to 5
  folds. The outline does not use it. It bears directly on this section. [file pointer: n15]
- NO CONTROL HAS RUN. Say so in the same sentence as every number. [file pointer: n20]
  [CHECKED 2026-10-09: same. The page says "random-flag controls exist in the code but were not
  run". Source: CH7_RESULTS.md §3 7.9 bullet 2.]
- A related fact that belongs here: a sample weight reaches no part of the feature selection, so a
  reweighting cleanup cannot affect the search even in principle.
  [CHECKED 2026-10-09 against the code: same. The only two mentions are the support vector
  regressor's fit signature, which passes the weight to the model. Source: CH7_METHODS.md §3.3.]

7.10  Results: not cleaning at all
- Averaging ten network seeds and stacking both models beat the plain model on the labels as
  delivered and under damage, on Caco-2 and hERG Ki. 0.7184 to 0.6557 RMSE.
  [CHECKED 2026-10-09: same, and wider. 0.7184 to 0.6557 is hERG Ki, Gaussian at 1.0, network on
  the second hybrid. LogD is added. Seed averaging helps both in 19 of 22 rows; stacking in 39 of
  44. Labels as delivered: seed averaging helps Caco-2 and LogD, and only the hybrid on hERG Ki.
  Stacking against a support vector regressor baseline compares a two-model stack with the weaker
  single model. Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.10 bullet 1. [file pointer:
  n40]]
- Adding noise to the targets and bagging did the same.
  [CHECKED 2026-10-09: PARTLY. It helps both in 15 of 32 rows. The support vector regressor fails
  on Caco-2. It is weaker than seed averaging and stacking. Source: CH7_RESULTS.md §3 7.10
  bullet 2. [file pointer: n33]]
- Weighting by each source's error variance beat a plain mean on Caco-2, hERG Ki and LogD.
  [CHECKED 2026-10-09: same in direction, small. 22 of 54 and 25 of 54 rows help both, 0 strong.
  The sources are simulated. By the outline's own grouping this is noise mitigation, a known flag
  then count less, not "not cleaning". Source: CH7_RESULTS.md §3 7.10 bullet 3. [file pointer:
  n39]]
- Retuning the model on the damaged labels did not help.
  [CHECKED 2026-10-09: REVERSED. Retuning helps both in 34 of 44 rows on caco2, herg_ki and logd.
  Under censoring it helped on only 1 or 2 of 8 pairs. The old bullet came from a stale summary.
  Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.10 bullet 4, §6 item 5. [file pointer:
  n40]]
- None of these has a control.
  [CHECKED 2026-10-09: same.]

7.11  Results: model and representation
- The network benefits from cleaning far more than the support vector regressor does. The support
  vector regressor often fails on Caco-2, hERG Ki and LogD.
  [CHECKED 2026-10-09: same on caco2, herg_ki and logd ONLY. Mitigation rows that help both,
  network against regressor: caco2 159 of 291 against 78 of 242; herg_ki 132 of 259 against 66
  of 217; logd 98 of 203 against 53 of 161 (CH7_RESULTS.md §3 7.11 bullet 1). Paired on method,
  dataset and damage: 421 against 260 of 711 on small molecules. The other model leads on
  herg_ic50, ddg_combined (regressor 50 against 41 of 75) and ginkgo_titer (forest 51 against 30
  of 62) (PATTERNS.md §4.3). Part of the network's lead may be its job-to-job spread.]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/PATTERNS.md §4.2] Noise-robust training rarely
  helps on the classification datasets. Rows that help both: hERG 3 of 70, humanness 0 of 30,
  cdr_flexibility 23 of 70. Against LogD 91 of 158 and Caco-2 94 of 202. Those rows are the
  network, and neither decided model for the classification sets is a network.
- The hybrid keeps its lead after cleaning. Cleaned hybrids beat cleaned single representations
  nearly everywhere, and often gain more. The exception is SMAC colloidal stability with the
  support vector regressor.
  [CHECKED 2026-10-09: CHANGED. Small molecules: the cleaned hybrid wins in 95 per cent of 1,704
  rows, but it already won in 92 per cent before cleaning. Mitigation flips the winner in
  126 rows, 93 toward the hybrid. Other domains: close to a coin toss before and after; 256 of
  1,215 flip, both ways equally. "Often gains more" is not supported: 669 rows against 625.
  SMAC with the regressor is no longer the exception, 89 better against 30 worse. The exceptions
  now: ginkgo_titer, humanness_therapeutic, flab_thermo_tresanco, abcov_kd, where the single
  representation wins in 71, 89, 81 and 68 per cent of rows. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §7, §3 7.11 bullet 2.]
- The best learning-with-noisy-labels method on a hybrid is NEVER the best method on either single
  representation, on all 9 datasets that ran. This is the sentence that joins the chapter to
  Chapter 4: Chapter 4 showed that similarity-based noise handling fails in a standard
  representation; this shows that the ranking of noise-handling methods is a property of the
  feature space and not of the method. [file pointer: n09, question 2]
  [CHECKED 2026-10-09: CHANGED. "Never" is false on 3 datasets. The same method is top on the
  hybrid and on both single representations on caco2 (10 per cent shift, level 1.0),
  cdr_flexibility (level 0 in all 6 flip files, and 0.1 in one) and herg (0.1, two flip kinds).
  Source: scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.11 bullet 3.]
  [DECIDE: keep this as the sentence that joins the chapter to Chapter 4, weakened to "on most
  datasets", or drop it. Keeping it needs the three exceptions named beside it.]
- [NEW 2026-10-09] The twelve-method comparison now has 13 datasets, not 9.
  [CONFLICT: CH7_RESULTS.md §3 says 13. CH7_METHODS.md §6 says 11 have files, including qm9, after the patient dataset is removed. [file pointer: n09]]
- The twelve published learning-with-noisy-labels methods, by dataset. Which methods pass differs
  by dataset. On three Ginkgo sets every method passes. On binding free energy only one does.
  Report the reversal honestly: distance-based selection was the worst of the twelve on QM9 in
  Chapter 4, and here it is the only one of the twelve that passes on hERG at a flip rate of 0.1.
  [CHECKED 2026-10-09: methods passing, of 12, labels as delivered / under damage: ginkgo_titer
  12 / 12; ginkgo_pr_cho 11 / 12; ginkgo_colloidal 10 / 12; ddg_combined 1 (heteroscedastic GP)
  / 1 (DivideMix); buchwald_hartwig, new, 10 / 11. Same under damage and for binding free energy.
  "Every method" on the labels as delivered is not true for colloidal or PR_CHO. The hERG
  reversal: same, 1 of 10 methods under damage, distance-based selection. On hERG's labels as
  delivered, uncertainty curriculum, DivideMix and MentorNet pass. QM9: 1 of 12 pass on clean
  labels, 8 of 12 under damage, by the survey page's rule. Source:
  scratchpad/thesis_ch5_8/CH7_RESULTS.md §3 7.11 bullets 4-6. [file pointer: n09]]
- CITATIONS FOR THE TWELVE (added by Claude 2026-10-09 from src/kirby/utils/lnl_methods.py; not
  reviewed by the author). "Published" above is too strong for most of them, because most keep
  only part of the cited method. Per method:
  - small_loss (L:568): cites Han2018 co-teaching. Uses one network and selects once, so it is
    small-loss filtering, not co-teaching.
  - confident_learning (L:659): cites Northcutt2021. Uses in-sample probabilities, with no
    per-class thresholds and no confident joint.
  - distance_selection (L:748): no citation; Wilson1972-style, removing a fixed 15 per cent.
  - multistage_cleaning (L:840): cites zhao2025. WRONG PAPER. zhao2025 is MANTRA, a framework for
    code models that removes samples once, by a mixture fitted to their loss trajectories. The code
    removes the worst 10 per cent in three rounds. No matching source found.
  - early_learning (L:929): cites zhang2025. WRONG PAPER. That paper uses an additive update
    driven by similarity to a clean subset. The code multiplies above-median-loss weights by 0.95
    each epoch. Not liu2020early* (early-learning regularisation) either.
  - uncertainty_curriculum (L:1077): cites Bengio2009. Ranks by the model's own loss, which is
    self-paced learning, kumar2010selfpaced*. Not ranked by uncertainty.
  - dividemix (L:1241): li2020dividemix without its semi-supervised half.
  - meta_weight_net (L:1443): shu2019metaweightnet, with the meta set carved from noisy labels.
    The source needs a clean set.
  - mentornet (L:1633): jiang2018mentornet, predefined-curriculum variant, no LSTM.
  - contrast_to_divide (L:1842): cites Zheltonozhskii 2021. WRONG DESCRIPTION. The code scores
    prediction change under input perturbation, which is not in that paper.
  - heteroscedastic_gp (L:2019): cites liu2020. WRONG PAPER for the code, which fits a GP, then a
    network to its residuals, then a weighted network. Closest is kersting2007most.
  - evidential_kernel (L:2154): amini2020deep, as published.
  [CHECKED 2026-10-09 against the code: all twelve lines same. The first four are the data-level
  methods in the code, matching the outline's split. All twelve train the same network as the
  plain model; only distance-based selection's neighbour setting is swept. Source:
  scratchpad/thesis_ch5_8/CH7_METHODS.md §4.1.]
  Precedent on molecules for this family: none found for co-teaching, DivideMix, Meta-Weight-Net
  or MentorNet on molecular property data. Cleanlab only in a 2025 preprint on physical
  properties (Marimuthu and McGuire, arXiv 2505.08688).
- HOW CHAPTER 4's LNL STUDY AND THIS CHAPTER'S SURVEY COMPARE (added 2026-10-09; verified against
  both sources in session). This is the comparison the chapter has to make explicitly, because a
  reader will ask whether Chapter 7 simply repeats Chapter 4.
  - THE SAME TWELVE METHODS, name for name. Chapter 4's table
    (~/Downloads/Confirmation_Report/paper.tex:680-712) and LNL_REGISTRY
    (src/kirby/utils/lnl_methods.py:2309-2326) hold the same set: meta weight net, small-loss
    selection, confident learning, distance-based selection, MentorNet, uncertainty curriculum,
    DivideMix, multistage cleaning, early learning, contrast-to-divide, heteroscedastic Gaussian
    process, evidential kernel.
  - NOT THE SAME CODE. The KIRBy file's header says "12 methods adapted from the confirmation
    report LNL survey", so this is a reimplementation. The thesis must NOT say Chapter 7 re-ran
    Chapter 4's experiment. It ran the same twelve methods. Which parts of each published method
    survive in the KIRBy version is the citation block above.
  - WIDER IN FOUR WAYS.
    Datasets: QM9 alone at 10,000 molecules, against nine datasets here. QM9 is not among the
      nine, so the two studies share no dataset at all. Say that.
      [OUT OF DATE 2026-10-09: QM9 now runs in the noise stage, on the same 9,978 molecules as
      the other KIRBy chapters (10,000 drawn; _shared.py:540); on 9 Oct
      the twelve-method sweep had finished on QM9 under censoring and grouped shift. Re-check
      this paragraph when its QM9 jobs finish.]
      [CHECKED 2026-10-09: still out of date. The survey now has 12 or 13 datasets including QM9
      (see the CONFLICT above). On clean QM9, comparing the best single representation and the
      best hybrid, the n09 run makes all twelve worse. Source:
      CH7_RESULTS.md §5; CH7_METHODS.md §5.3.]
    Noise: Gaussian at levels 0 to 1.0, against six damage kinds on the regression datasets and
      six label-flip kinds on the classification ones.
    Feature space: Chapter 4's axis is WHICH SINGLE REPRESENTATION (ECFP4, Sort & Slice, PDV,
      canonical and randomised SMILES). This chapter's axis is SINGLE AGAINST HYBRID: the
      network's best and second-best single representation, its best and second-best hybrid, and
      an importance-weighted copy of the first hybrid, each picked per fold by a search
      (n09_lnl_survey.py:119-121).
    Baseline: Chapter 4 ranked the methods against each other and reported a noise-retention
      percentage. This chapter compares each method against the PLAIN network on the same damaged
      labels in the same feature space, in the dataset's own metric. That is the stricter test.
  - NARROWER IN ONE WAY, AND THIS IS WHY CHAPTER 4 CANNOT BE DROPPED. Chapter 4's 221
    configurations came from crossing the twelve methods with four optional components:
    uncertainty-awareness, a molecular distance, an alternative kernel, and an alternative loss
    function. This chapter sweeps none of the four. n09_lnl_survey.py:102-123 sweeps the methods,
    the five feature spaces, the damage kinds, the levels and three noise draws, and the only
    survivor is a neighbour-space sweep for distance-based selection alone. So the ANOVA that said
    distance-metric usage was the dominant factor in the variance in R2 has NO counterpart here
    and cannot be reproduced from these runs. That measurement is step 3 of the chain, and the
    scope defence rests on it.
  - WHERE THEY AGREE. The heteroscedastic Gaussian process and DivideMix keep the same role. On
    binding free energy, the hardest dataset here, exactly one method passes on the labels as
    delivered and it is the heteroscedastic Gaussian process; exactly one passes under injected
    damage and it is DivideMix. Chapter 4 had them at 79.0 and 86.7 per cent noise retention at
    sigma 0.6.
    [CHECKED 2026-10-09: the binding free energy counts are same. Source: CH7_RESULTS.md §3 7.11
    bullet 4.]
  - THE ONE REVERSAL, to report plainly. Distance-based selection was the worst of the twelve on
    QM9 regression. On hERG under label flips at a rate of 0.1 it is the ONLY one of the twelve
    that passes. Same method, opposite verdict, different task and different data. Do not smooth
    this over: it is the central claim appearing inside a single method.
  - WHAT CHAPTER 4 COULD NOT SEE. On all 9 datasets, the method that scores best on the hybrid is
    not the method that scores best on both single representations.
    [CHECKED 2026-10-09: false on caco2, cdr_flexibility and herg; see the first 7.11 bullet on
    the twelve. Source: CH7_RESULTS.md §3 7.11 bullet 3.] Chapter 4 showed that
    similarity-based noise handling fails in a standard representation. This shows the RANKING of
    noise-handling methods is a property of the feature space and not of the method. That is step
    8 of the chain closing on step 2.
  - [BOLD] rule (b): in the per-method table, bold the distance-based-selection row on hERG, which
    is the reversal.

7.12  What to confirm with the full folds, in order
Carried from the author's own list, 2026-10-08.
1. SMAC colloidal stability: removal and relabelling, on the labels as delivered and under damage.
2. hERG Ki: relabelling, and putting cliff molecules back.
3. hERG: deleting and relabelling under the flip kinds.
4. The clipped-label lifts.
5. The per-scaffold-group offsets, as the result for structured noise.

7.13  Discussion
- Is there a general method when the noise is unknown. The answer the data supports: no single
  detector and no single treatment passes everywhere, and the two that come closest are the
  out-of-fold residual and the neighbour-label gap.
  [CHECKED 2026-10-09: see 7.6. Most experiments offer only these two. With a control on small
  molecules the neighbour gap leads the forest residual. Source: CH7_RESULTS.md §3 7.6;
  PATTERNS.md §4.6.]
- Improving an experimental dataset with nothing injected, as a use case in its own right.
- How this sits with Chapter 3. Chapter 3 found that no model's per-molecule uncertainty
  identifies the corrupted labels, and that models are most certain about clipped labels. This
  chapter agrees: the residual detectors beat the model-uncertainty detectors, and clipping is
  fixable precisely because it needs no detector.
  [CHECKED 2026-10-09: "residual detectors beat the model-uncertainty detectors" is not shown by
  any finding. The four model-uncertainty detectors pass in 31 to 38 of 52 rows each, with no
  control and no side-by-side comparison. The clipping half holds. Source: CH7_RESULTS.md §2b,
  §2c result 1. [file pointer: n23, n30]]

7.14  Limitations
- One outer fold for almost everything.
  [CHECKED 2026-10-09: still true for the results used in 7.5 to 7.11. Some experiments now have
  3 to 5 folds; see the STANDING CAVEATS note. Source: CH7_METHODS.md §6.]
- No control for most treatments, and which ones.
  [CHECKED 2026-10-09: the controls that ran are listed in the STANDING CAVEATS note. The generic random
  and shuffled twins never ran. No damage condition on an antibody dataset has a control. Source:
  CH7_METHODS.md §3.4; PATTERNS.md §4.5.]
- [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH7_METHODS.md §6] Experiments with results that
  the chapter does not use: c10, n01, n11, n12, n13, n14, n15, n16, n17, n21, n26, n35, n36, n37,
  n38, n42, noise_ceiling. Two measure the size of the noise itself, which the assay error floor
  below needs. [file pointer: n26, n37]
- Correcting a label with no ground truth.
- The assay error floor per dataset, which no model can beat.
- Only one QM9 property was tested: the HOMO-LUMO gap (author, 2026-10-09). QM9 is the only clean
  control, so the control rests on one computed property. Testing more QM9 properties would have
  shown whether the control holds beyond that one target. Carry to 8.6 and 8.7.
- Datasets with complexity or similarity results but no noise results. Status on 9 Oct, from the
  run's job list; it changes as jobs finish:
  - no noise and no similarity results, complexity still finishing: abagym_Ang2, abagym_VEGF,
    adme_hlm, adme_mdr1, adme_rlm, flab_binding_koenig, ginkgo_hic.
  - similarity partly done (4 of 11 experiments), no noise results: flab_expression_koenig.
  - complexity and similarity done, no noise results: abagym_EGFR.
  [DECIDE per dataset when the runs settle: a dataset is mentioned only if its results show
  something, same rules as always. Where it is mentioned, its missing noise results are named
  here as a limitation.]

7.15  Conclusion
The bridge to Chapter 8.


CHAPTER 8  DISCUSSION AND CONCLUSIONS  (about 14 to 18 pages)
Was Chapter 7 before 2026-10-08. The author's central argument, below, is what this chapter has to
end on.
[CHECKED 2026-10-09: thesis.tex Chapter 8 (lines 4740-5103) predates this outline. It has no
eight-step chain, no scope defence, no D8.1 and no central claim. Its cross-references still treat
similarity and noise as one chapter. "LNL" appears 28 times. It holds 22 [RESULT NEEDED] markers
and 19 lines saying "preliminary". Its numbers come from preliminary runs. Source:
scratchpad/thesis_ch5_8/CH8.md §1.]

8.1  Summary
One paragraph per chapter.

8.2  THE SHAPE OF THE ARGUMENT  (the author's words, 2026-10-08)
Each step was forced by the previous one failing. Nothing here was a change of topic. State the
chain in this order, in Chapter 1 as the plan and in Chapter 8 as what happened.

  1. Investigate label noise.                                              Chapter 3
  2. Try the published methods for learning with noisy labels. They fail.  Chapter 4, 4.4.1
  3. Look at the similarity those methods rest on. It fails too.           Chapter 4, 4.3-4.4
  4. Look at the representations the similarity is computed in, and call
     the features themselves into question.                               Chapter 4, 4.2
  5. Select features more intelligently, per dataset.                      Chapter 5
  6. Show the resulting representation predicts better.                    Chapter 5, 5.4-5.5
  7. Show that makes similarity better.                                    Chapter 6
  8. Show that makes the noise methods better.                             Chapter 7, 7.11

Steps 1 to 4 narrow: each one asks why the step above it failed, and the answer is always one level
further down, ending at the features. Steps 5 to 8 widen back out along the same path, in reverse,
and each one is tested where the corresponding step failed.

THE SCOPE DEFENCE, for Chapter 1 and for the viva. One supervisor cautioned that the hybrid strays
from the topic. The answer is step 4: the thesis is about label noise, and the reason it reaches
feature selection is that label noise could not be handled without it. The evidence that the chain
is real and not a reconstruction after the fact:
- Step 2 is measured, not asserted: of the methods tried in the exploratory study, all 20
  configurations with the lowest noise retention used a distance (Chapter 4, 4.4.1).
  [CHECKED 2026-10-09: the only source on this machine is one sentence of the confirmation
  report, ~/Downloads/Confirmation_Report/paper.tex:721. The raw CSVs were on the server under
  $DATA/results/comprehensive_noise_study/ and are not here, so the count cannot be recomputed.
  The ANOVA beside it has no effect size anywhere. Source: scratchpad/thesis_ch5_8/CH8.md §2.]
  [MISMATCH in this outline: here the distance evidence supports step 2. In 7.11 the ANOVA on
  distance usage is called "step 3 of the chain". Source: CH8.md §2.]
  [DECIDE: say step 2 rests on the confirmation report's sentence, or retrieve the server CSVs
  and recount. The first costs nothing. The second turns it into a number read from data.]
- Step 3 is measured: distance in a standard representation tracks activity weakly, with the
  highest Caco-2 correlation in any single representation at 0.2292 Spearman (Chapter 6, 6.3).
  [CHECKED 2026-10-09: same, 5 folds. It is top_torsion under Minkowski p=0.5. It is the highest
  on Caco-2 only. Higher bests elsewhere: humanness_therapeutic 0.3885,
  qm9 0.2789, flab_expression_koenig 0.2498. Chapter 4 (4.3) quotes an older local Caco-2 run,
  atom_pair 0.225 under Manhattan; Chapters 4, 6 and 8 must quote one run. The step sits in
  Chapter 4, but its number sits in Chapter 6. Source: CH8.md §2 step 3. [file pointer: s01]]
- Step 4 is measured: features that do not vary, hash collisions, and molecules a representation
  cannot tell apart (Chapter 4, 4.2).
  [CHECKED 2026-10-09: two of the three parts are measured. Hash collisions are NOT MEASURED.
  4.2.2 lists that statistic as planned, and no code for it exists in KIRBy. Molecules a
  representation cannot tell apart: on hERG Ki, ecfp4, atom_pair, pdv and mordred each give 56 of
  1,412 molecules an identical feature vector, and maccs gives 377. That census fits no model, so
  it has no folds. Features that do not vary were counted 2026-09-27 and not re-checked here.
  Source: CH8.md §2 step 4; tests/results/comparisons/opig_shared_vectors_census.csv.]
  [DECIDE: drop "hash collisions" from this bullet, or run the planned statistic. Dropping costs
  nothing. Running it needs new code.]
- Step 8 closes it: the best learning-with-noisy-labels method on a hybrid is never the best on
  either single representation, on all 9 datasets (Chapter 7, 7.11).
  [CHECKED 2026-10-09: false as written. The count is now 13 datasets, not 9, all fold 0. "Never
  the best on either" holds on 1 of 13: ginkgo_acsins. On cdr_flexibility the hybrid's top method
  equals the second single representation's in 10 of 12 combinations of damage kind and level. In
  4 combinations it equals the top method on both singles: caco2 once, cdr_flexibility once, herg
  twice. The weaker claim holds on all 13: in at least half the combinations, the hybrid's top
  method is not the top method on both singles. Source: CH8.md §2 step 8, recomputed from the
  noise_view n09 folder. [file pointer: n09]]
  [DECIDE: use the weaker wording, or drop step 8 from 8.2 and 8.2a. Weaker wording, from CH8.md
  §2: "On 12 of 13 datasets, the noise method that scores best with the hybrid's features is
  sometimes also the best on one of the two single representations. On none of the 13 is it
  usually the best on both. The ranking of noise-handling methods moves when the features move,
  but not every time. One fold." Dropping it leaves step 8 with no evidence bullet. Note the
  weaker form is about ranking, not about the methods getting better.]
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/PATTERNS.md §4.6: the nearest result to "the
  hybrid makes the noise methods better". Down-weighting with neighbours found in the first hybrid
  improved a larger share of model-and-configuration pairs than neighbours found in the pair's own
  single representation, on all 4 small-molecule datasets that ran. caco2 220 of 530 against 40 of
  112. herg_ki 149 of 278 against 19 of 54. logd 127 of 238 against 16 of 48. herg 226 of 246
  against 42 of 52. Antibodies are split. The comparison is unequal: hybrid-space rows cover 8
  pairs, single-space rows cover 2. No control, fold 0. [file pointer: n12]]
Written this way, the hybrid is not a second topic. It is the answer to the question the first
topic produced.
- [DIAGRAM D8.1] The chain above, drawn once. Eight boxes in a vertical line, numbered as above.
  The arrow down the left side is labelled "why did that fail?" and runs from box 1 to box 4. The
  arrow up the right side is labelled "does fixing the features fix it?" and runs from box 5 to
  box 8, with box 5 beside box 4, box 6 beside box 3, box 7 beside box 2 and box 8 beside box 1, so
  each repair sits level with the failure it answers. The chapter number goes on each box.

8.2a  THE CENTRAL CLAIM  (the author's words, 2026-10-08; this is what the thesis argues)
- There is no single best model, representation, similarity measure, learning-with-noisy-labels
  method or noise mitigation method that fits every dataset.
- Everything depends on the data, the features, the true labels, and the noise those labels carry.
- Every dataset carries its own noise, as outliers and activity cliffs, and it is not predictable.
- So the honest answer is not a winner. It is a broad sweep, which this thesis runs repeatedly,
  plus hybrid featurisation: a procedure that adaptively builds a feature selection for each
  dataset.
- That is not a one-size-fits-all method. It is an example of something that ADAPTS to unknown
  data corruption, and it adapts further when used alongside noise mitigation.
- The claim is not "this method works". It is "no single method can work, so manipulate the data
  in terms of its features instead".
- The chain to state: noise, then similarity, then the finding that standard representations do
  not support similarity, then adaptive feature selection as the answer to that, and then back to
  noise. The hybrids also work on their own for plain prediction.
- The strongest single piece of evidence for the argument: SMAC colloidal stability and HIC
  retention hold the same 242 antibodies with different endpoints, and the answer is opposite
  (7.8).
  [CHECKED 2026-10-09: NOT FOUND as stated. Same antibodies: yes, by the c02 input fingerprint
  (242 antibodies, same groups hash); sequences were not compared directly. But HIC has no noise
  result in any experiment, so the cleaning half compares a result with an absence. On the one
  question both have answered, the answer is the same: the hybrid loses to the best single
  representation with both models on both, 5 folds. SMAC dnn -0.0088, svm -0.0005 Spearman. HIC
  gp -0.0486, dnn -0.0018. Source: scratchpad/thesis_ch5_8/CH8.md §5; PATTERNS.md §5.]
  [CHECKED 2026-10-09: where the two do give opposite answers. Adding the augmentation
  representations in the distance search makes SMAC worse with dnn (0.3224 to 0.1593) and svm
  (0.3837 to 0.3430) and better with kNN (0.1918 to 0.2056). It makes HIC better with all three
  (gp 0.4980 to 0.5151, dnn 0.4538 to 0.4600, kNN 0.3301 to 0.3612). 5 folds. These come from the
  answer page and were not traced to one raw key. Across the complexity answer pages the two
  differ in 31 of 79 yes/no answers, and many of those flips are tiny. Source: CH8.md §5.2.
  [file pointer: c02 Q5]]
  [DECIDE: keep this as the strongest evidence and wait for HIC's noise results, or replace it
  now with "same antibodies, the hybrid loses on both, and the endpoints disagree on about two in
  five of the complexity questions". Waiting needs the HIC noise runs to finish. Replacing changes
  the claim from cleaning to the hybrid search's choices. [file pointers: n07, n09, n18]]
- The second strongest: the best learning-with-noisy-labels method on a hybrid is never the best
  on either single representation, on all 9 datasets (7.11).
  [CHECKED 2026-10-09: false as written; 13 datasets now, and it holds strictly on 1. See the
  step 8 bullet in 8.2 for the numbers and the weaker wording. Source: CH8.md §2 step 8.]

WHAT THE 9 OCT RESULTS SUPPORT  (added 2026-10-09 by Claude, from the findings files; the
strongest supportable form of each claim above, each with its source; none is a verdict)
- No single method fits every dataset. Whether noise handling helps the network more than the
  second model depends on the dataset. Small-molecule regression sets: the network gains in 421
  and the support vector regressor in 260 of 711 paired conditions. ddg_combined: the regressor
  leads, 50 against 41 of 75. ginkgo_titer: rf leads, 51 against 30 of 62. Fold 0. Source:
  PATTERNS.md §4.3.
- The hybrid builds a different feature selection for each dataset. The median number of
  representations given features is 10 on caco2 and herg_ki, 3 on herg and qm9, and 2 on
  buchwald_hartwig. 5 folds, 2 models. Source: CH8.md §3.2. [file pointer: c22]
- The hybrids work on their own for plain prediction, on small molecules. The hybrid beats the
  single representation picked on training molecules in 78 of 80 fold-and-model pairs, over 8
  datasets and 5 folds. It also beats the single representation that scored best on the held-out
  molecules in 16 of 16 dataset-and-model pairs. Source: PATTERNS.md §2.1, §2.3. [file pointer:
  c02, c01]
- On antibodies the same comparison is 83 of 140 fold-and-model pairs, and 6 of 28 against the
  held-out-best single. Source: PATTERNS.md §2.2, §2.3.
- Correction works far better when the damaged labels are known. Small molecules, 25% of labels
  clipped: moving the flagged labels helped both the single representation and the hybrid in 38
  of 38 rows. Detector-chosen actions helped in 51 of 168. Noise-robust training helped in 25 of
  87. Fold 0, no control counted. Source: PATTERNS.md §4.1.
- The hybrid stays ahead under noise handling on small molecules. It is ahead of the single
  representation in 2,033 of 2,245 rows before treatment and 2,100 after. On antibodies it is
  ahead in about half, 696 and 675 of 1,323. Fold 0. Source: PATTERNS.md §4.4.
- The clearest controlled noise result: hERG under label flips, 160 of 192 model-and-configuration
  pairs improved and beat the control. The clean control behaves: on QM9, detector deletion helped
  in 0 of 11 rows. Fold 0. Source: PATTERNS.md §4.2, §4.5.
- "Adapts further alongside noise mitigation": the nearest result is the neighbour-space one under
  step 8 in 8.2. It has no control.
- Real censoring (ginkgo_pr_cho, 58 of 197 labels at 0): whether the injected-censoring result
  carries over to labels censored as delivered, with the single representation and the hybrid.
  [RESULTS PENDING: see 7.7.]

8.3  Handling noise realistically
- Model the noise before choosing a method, and what the noise types of Chapter 7 imply.

8.4  No single method fits every dataset
- The conditional results across all chapters, named per dataset, never counted across datasets.
- THE ANTIBODY HYPOTHESIS (the author's): antibodies are larger molecules, so there is more room
  for distinct featurisation, so the hybrid should help more on them.
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH8.md §3 and PATTERNS.md §2-3: what the results
  show against it. Her hypothesis is kept as hers; these are the numbers beside it.]
  - Prediction. Small molecules: the hybrid is better on 5 of 5 folds for both models on 7
    datasets, and 4 of 5 on herg. Antibodies: one dataset and model reaches 5 of 5,
    flab_expression_koenig with svm, 0.7029 to 0.7729 Spearman. The hybrid loses with both models
    on flab_thermo_tresanco, ginkgo_colloidal and ginkgo_hic. Some antibody mean gains are large,
    ddg_combined dnn 0.1317 to 0.3542, but the folds disagree in sign: ddg_combined svm ranges
    from -0.121 to +0.699. Source: CH8.md §3.1. [file pointer: c02]
  - What the hybrids draw on. Effective number of representations by feature share, median over
    10 hybrids: caco2 9.92, herg_ki 9.95. Antibodies range from 9.23 (ginkgo_pr_cho) to 2.94
    (abcov_kd). Source: CH8.md §3.2. [file pointer: c22]
  - Language models of the sequence give 69 to 100% of the antibody hybrids' features. They carry
    at least 90% of the permutation drop on 11 of 14 antibody sets. Small-molecule hybrids draw on
    fingerprints, descriptors, pretrained graph models and SMILES models, each 10 to 47% of the
    features. Source: PATTERNS.md §3. [file pointer: c22]
  - The antibody pools are narrow. Most hold 2 to 4 variants of ESM-2 and at most 2 hand-built
    representations; 4 pools hold none. Small-molecule pools hold 3 to 7 hand-built ones. Source:
    PATTERNS.md §3.
  - Agreement between representations' pairwise distances, median Spearman: ginkgo_hic 0.42,
    flab_thermo_tresanco 0.509, against herg_ki 0.315 and logd 0.388. Higher means the
    representations agree more. These are local runs of 8 Sep on older pools. Source: CH8.md §3.2.
    [file pointer: f26]
  - Many antibody sets are deep mutational scans of one parent antibody: koenig G6, the abagym
    sets, the mutation sets inside ddg_combined. There most of the molecule does not vary. Source:
    PATTERNS.md §3, from the 5.3.1 descriptions; loaders not re-read.
  - Confound. Molecule size and dataset size move together. Antibody sets hold 141 to 672
    antibodies, except the two Koenig sets at 4,275. Small-molecule sets hold 1,412 to 9,978.
    These results cannot separate the two. The Koenig sets are the only large antibody sets, and
    there the hybrid wins with both models. Source: CH8.md §3.2.
  [DECIDE: keep the hypothesis as stated with these numbers beside it as a counterpoint; or
  restate it about the pool rather than the molecule ("the hybrid helps where the pool holds
  representations of different kinds, or there are many molecules"); or drop it. The second is a
  reading of 3 datasets (ddg_combined, flab_expression_koenig, ginkgo_acsins), not a measured
  result. Source: PATTERNS.md §3, §7.]
- WHERE THE ANTIBODY EXCEPTION DOES HOLD: the noise methods, not the hybrid.
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH8.md §3.1 and PATTERNS.md §4.5, §5]
  - On ginkgo_titer every one of the 12 published methods passes in 24 of 24 cells.
    ginkgo_pr_cho is 23 of 24 and ginkgo_colloidal 22 of 24. Fold 0. [file pointer: n09]
  - SMAC colloidal stability, svm on the first single representation, removing the 20% the
    out-of-fold forest residual flags: 0.2740 to 0.4034 Spearman. Fold 0. [file pointer: n18]
  - On the labels as delivered, detector-chosen cleaning improved and beat its control in 22 to
    51% of model-and-configuration pairs on antibodies, against 7 to 21% on small molecules.
    Fold 0. Same size confound as above. Source: PATTERNS.md §4.5.
  - No antibody damage condition has a control. In the one experiment run only on SMAC and titer,
    all 14 detectors work, which a detector carrying no information would also give. Source:
    PATTERNS.md §4.6, §5. [file pointer: n11]
  - Fold 0 of ginkgo_colloidal holds 53 antibodies. A Spearman on about 50 molecules moves by
    roughly 0.14 (CH8.md §3.1) or 0.15 (PATTERNS.md §4.5) by chance. That is arithmetic, not a
    measured spread. [CONFLICT: same arithmetic, rounded differently in the two files.]
  [DECIDE: does "works unusually well on antibodies" refer to the noise methods (supported, fold 0,
  no control under damage) or to the hybrid (not supported for consistency, 5 folds)? The first
  keeps the antibody section in Chapter 7. The second needs the wording changed in 5.4 as well.
  Source: CH8.md §7.]
- WHAT EXISTS PER DOMAIN, for "the methods are not domain-dependent".
  [NEW 2026-10-09, from scratchpad/thesis_ch5_8/CH8.md §4]
  - Reactions, buchwald_hartwig: mixed. rf 14.4056 to 15.2383 RMSE in yield %, better on 2 of 5
    folds. lightgbm 16.1679 to 15.6211, better on 4 of 5. The fold spread is larger than either
    mean. Prediction from 5 nearest neighbours is worse in the hybrid at 0 of 18 settings better.
    [file pointers: c02, s05]
  - DNA sequences: cut 2026-10-08. One fold of nt_promoter_tata exists: lightgbm mistral_dna
    0.6268 to hybrid 0.7060 MCC. Usable at most as "the search runs on DNA sequences".
    [file pointer: c22 partial]
  - Spinoffs (finance, autonomous vehicles, cross-species, ROGII): none has a
    hybrid-search result. Finance has a single-representation survey only. Cross-species stored no
    hybrid score.
  - Supportable form: in the thesis runs the hybrid was tested on small molecules, antibodies, one
    reaction set and one gene-expression set. "Not limited to molecules" rests on one expression
    set. "Any domain" is not supported.
  [DECIDE: mention the spinoffs at all, or leave them out. Mentioning them adds no hybrid result.
  Leaving them out loses nothing measured. Source: CH8.md §7.]

8.5  Simple is often better
- Trees against networks under noise; fewer features; prediction from nearest neighbours on a small
  dataset; the methods that had no effect.

8.6  Limitations
- Correcting labels without ground truth; the search's cost; one fold for most of Chapter 7; the
  datasets whose task carries one example rather than a general claim; only one QM9 property (the
  HOMO-LUMO gap) as the clean control (see 7.14); datasets with no noise results (see 7.14).
  [CHECKED 2026-10-09: every noise result is fold 0 only (folds_run [0] in all 55 n09 files
  read). No noise result exists for adme_hlm, adme_rlm, adme_mdr1, the abagym sets, the Koenig
  sets or ginkgo_hic; flab_thermo_tresanco has no n09 json. No antibody
  damage condition has a control. Source: scratchpad/thesis_ch5_8/CH8.md §1.3, §4.1;
  PATTERNS.md §8.]

8.7  Future directions
- More QM9 properties as clean controls for the noise methods (author, 2026-10-09).
  [CHECKED 2026-10-09: the loader already supports 12 QM9 properties (src/kirby/datasets/qm9.py,
  property_idx; the gap is index 4). Only the gap is registered as a dataset. Each new property
  needs a dataset entry, its representations and its runs. No results exist for any other
  property. Source: CH8.md §6.]
- Classification and label flipping.
  [CHECKED 2026-10-09: partly done. Six kinds of label flip already run on herg (7,772),
  cdr_flexibility (328) and humanness_therapeutic (564), fold 0. Source: CH8.md §6.
  [file pointer: n09; also n07, fed01]]
  [DECIDE: keep "classification and label flipping", or reword to what is still missing: flips
  that depend on the molecule near a class threshold, more folds, more classification datasets.
  The reword stops the thesis listing done work as future work.]
- Admitting extra features: the decoy test of 5.5a is a designed negative, and the rule where an
  extra feature must displace a base one has results on 3 folds only, with augmentation off in the
  full runs.
  [CHECKED 2026-10-09: was "has no results"; that contradicts 5.5a. Result files for the
  displacing rule exist for buchwald_hartwig, caco2, flab_expression_koenig and
  herg_ki. 5.5a quotes one: Buchwald-Hartwig with LightGBM, 108 of 259 configurations beat no
  extras. Source: CH8.md §6. [file pointer: f24, capped]]
- Hybrid features for generative models.
  [CHECKED 2026-10-09: built, not run. A REINVENT 4 scorer steers the generator toward high
  predicted hERG pKi, with four feature sets: ecfp4, the hybrid's blueprint, a random matrix of
  the same width, and the best single representation. No result files. Source: CH8.md §6.
  [file pointer: tests/experiments/opig_generator_scoring_swap.py, untracked]]
- A faster search.
  [CHECKED 2026-10-09: partly done. On hERG Ki the cut grid keeps 224 configurations and 1.4% of
  the fitting time and keeps the best configuration; UNVERIFIED here, quoted from thesis.tex:2288.
  Still open: a warm start from a similar dataset's blueprint, and measured hours for the full
  search against the cut grid. Source: CH8.md §6.]
- A fuller federated deployment.
  [CHECKED 2026-10-09: stands. Current results are Caco-2 and LogD only, as simulated silos on
  public data. The Lhasa hERG sites went with lhasa_herg on 8 Oct. The antibody sites were cut on
  9 Oct. So "fuller" now also means restoring real-site data. Source: CH8.md §6.
  [file pointers: fed01-fed04]]
- [DECIDE: thesis.tex Chapter 8 has two future directions this list lacks. One is transformed
  features, graph-distance features, KernelSHAP and permutation importance. The other is external
  benchmarks: activity cliffs, Ginkgo competition splits, Hummer2025. Add them here, or delete
  them from thesis.tex. Source: CH8.md §1.3.]


SUMMARY OF RESEARCH CONTRIBUTIONS
- A short closing list (following the peer theses): the NoiseInject paper and the planned
  hybrid-representation papers (Chapters 5 to 7), the UK-QSAR conference presentation, the OPIG
  Blopig posts, and the three distinct open-source software artifacts (NoiseInject, MIT,
  released; the standalone Rust and Python chemical-processing and precompute pipeline, to be
  simplified and released separately; and the KIRBy research codebase, pure Python).
- thesis.tex:1911 carries the old second paper title; typos "prepration" (1919, 1922) and "Posted
  presented" (1925).


BIBLIOGRAPHY
- Author-year style, about 180 to 220 entries.
- 39 keys cited in thesis.tex's second half have no entry in refs.bib or sn_bibliography.bib.
  Fang2022 is cited at thesis.tex:1523 for GNN embeddings but is the CVPR federated-noise paper;
  that sentence needs Fang et al. 2022, GEM (Nature Machine Intelligence).


APPENDICES
- A: datasets and curation (Chapters 3 to 7): QM9, LogD, Caco-2, hERG, and the 26 live datasets
  of Chapters 5 to 7 (43 in tests/experiments/decided_grid.json, 16 cut 2026-10-08 in
  tests/experiments/orchestrate/plan.py:62-75, chembl_qsar reached only by the OPIG script), with per-dataset provenance, curation counts before and after, accession numbers,
  and DOIs. Must say which curation each chapter used (Chapter 3's 5,039/2,161/1,415 against
  KIRBy's 5,028/2,158/1,412).
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
- C: the hybrid search's full space and the design-experiment tables behind its defaults
  (Chapter 5): the values in 5.2.2, the f33 cells-won tables of 5.2.3, the c32 allocation cross,
  the fast-tier evidence, the pruning schedule, and the exact blueprint fields. thesis.tex's
  Appendix C comment still lists greedy, dominant-greedy and mlp.
- D: extended result tables for Chapters 4 to 7: Chapter 4's per-representation feature counts,
  tie census and distance-activity tables; Chapter 5's per-dataset hybrid-against-best-single
  tables, c17 per budget, c23, c19, c21 with decoy counts, blueprint transfer per fold, fed03 arms;
  Chapter 6's per-dataset distance and nearest-neighbour tables; Chapter 7's per-method
  learning-with-noisy-labels tables, the per-cleanup tables on the labels as delivered, and the
  clipped-label tables per treatment and clipped share.
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
  the second-half antibody and protein datasets) are publicly available
  under their upstream licenses, with provenance in Appendix A. The Lhasa federated data
  (Effiris) is proprietary and not redistributable; the federated demonstration also depends
  on Lhasa's FLuID code being importable, so it does not run standalone, and as a documented
  simulation compromise it uses the training set as the shared transfer pool rather than a
  separate public pool. Per-repository Springer-style itemised blocks (project name, home
  page, archived version, OS, language, requirements, license, restrictions) are included
  for all three artifacts. Cross-referenced from the front-matter Data and Code Availability
  statement.
