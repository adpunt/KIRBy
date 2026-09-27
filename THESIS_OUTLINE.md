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
  flab_thermo_tresanco 0.509, tcga_luad_vs_lusc 0.106. Least alike on herg_ki: graph_kernel and
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
    includes mlp). The antibody geometry survey (7 structure-geometry groups concatenated onto
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
  (proof of concept), gene expression.
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
- Table 5.1: one row per axis, its values, the fast-tier values, and where it is set.

5.2.3  Why each axis is swept
The brain dump asks for the reason for each axis, with results showing that datasets and models
disagree. Source: the f33 pruning record, tests/results/framework/f33_pruning/{flab_thermo_
tresanco, herg_ki, humanness_therapeutic, logd, tcga_brca_pam50}.json: 5,100 configurations each,
5 folds, budgets 25/50/100/150/300, 10 representations, no augmentation. A cell is one
(downstream model, budget), 25 per dataset; the cell's winner is the best mean over 5 folds.
- Downstream model and budget: the best configuration is dnn at budget 50 on flab_thermo_
  tresanco (Spearman 0.6696), svm at 300 on herg_ki (RMSE 0.5589 pKi), lightgbm at 50 on
  humanness_therapeutic (AUROC 0.9723), dnn at 300 on logd (RMSE 0.4170), dnn at 150 on
  tcga_brca_pam50 (weighted F1 0.9097). Within flab_thermo_tresanco, svm peaks at 300 features and
  dnn, lightgbm and rf at 50.
- Allocation method, cells won out of 25 (performance_weighted / power_2 / softmax_T1 /
  rank_exp_d0.3 / wglobal_power2): flab 3/0/2/6/14; herg_ki 3/6/1/2/13; humanness 4/7/3/6/5;
  logd 2/2/0/1/20; pam50 2/4/2/3/14.
- Representation count, cells won out of 25 (2 / 3 / 5 / all 10): flab 2/1/5/17; herg_ki
  0/2/13/10; humanness 7/14/4/0; logd 17/5/0/3; pam50 0/3/8/14. If only one count had been
  searched, the best score falls short: flab 0.0256 Spearman (count 2) or 0.0190 (all 10);
  herg_ki 0.0125 RMSE (count 2); pam50 0.0086 weighted F1 (all 10).
- Importance method, cells won out of 25 (lightgbm_gain / treeshap / deeplift): flab 15/10/0;
  herg_ki 14/11/0; humanness 12/12/1; logd 8/0/17; pam50 21/3/1.
- Internal evaluator, most frequent winner: dnn on flab (12 of 25), rf on herg_ki (14) and
  humanness (14), lightgbm on logd (10), lightgbm then dnn on pam50 (9, 7).
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
  - On the f33 record it cut 30% of the hours (130.4 to 91.5 over five datasets) and changed no
    (model, budget) best score. Voting after fold 0 alone was not safe (pam50's fold 0 takes four
    distinct values across 5,100 configurations).
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
- The 16, from the loaders and tests/experiments/dataset_policy.py (n from running the loaders
  locally on 2026-09-27 unless marked):
  Small-molecule ADME and toxicity, regression, RMSE, Murcko scaffold GroupKFold, 5 folds:
    herg_ki     1,412 molecules, pKi (4.07-9.41); ChEMBL CHEMBL240 Ki, binding assays, median
                pChEMBL per compound, compounds with SD > 1.0 dropped; 769 scaffolds
    caco2       2,158, log10 efflux ratio; OpenADMET ExpansionRx; labels clipped at their 1st and
                99th percentiles by the loader; 1,086 scaffolds
    logd        5,028, LogD; OpenADMET ExpansionRx; same clipping; 2,413 scaffolds
    adme_rlm    3,053, log RLM CLint; Polaris biogen/adme-fang-v1 (Fang et al.); 2,625 scaffolds
    adme_mdr1   2,641, log MDR1-MDCK efflux ratio; same source; 2,318 scaffolds
  Quantum, regression, RMSE:
    qm9         10,000 drawn (9,978 in DATASET_STATUS.md, unverified), HOMO-LUMO gap; Murcko
                scaffold GroupKFold 5 on one seed-42 draw (the noise chapter's ten replicates are
                a different pipeline)
  Reactions, regression (yield %), RMSE, grouped by additive:
    buchwald_hartwig  3,955 reactions, Ahneman et al. 2018; 22 additive groups
    bh_doyle_plates   4,312 (287 of 4,599 dropped for a missing aryl halide); 24 groups. The
                      name says plates but the code groups by additive, not plate
  Antibodies:
    flab_expression_koenig  4,275, expression enrichment, Spearman; FLAb, Koenig et al. 2017 (G6
                            anti-VEGF scan); grouped by first mutation position mod 5
    flab_thermo_tresanco    672, melting temperature (39.0-98.2 C), Spearman; FLAb, Tresanco
                            2023 (NbThermo, unverified); 95% sequence-identity clusters (309)
    humanness_therapeutic   564, binary (198 human, 366 non-human), AUROC; BioPhi (Prihoda et
                            al. 2022); 95% identity clusters. Loader docstring says 549
    tz_her2_binary          5,000 drawn from 524K, binary (1,639 binders), AUROC; Chinery et al.
                            2024; edit-distance shells from the wild-type CDRH3 (10)
  Nucleotides, binary, MCC, grouped by chromosome:
    nt_H3K4me3       18,244 (registry, unverified); NT Revised (InstaDeep)
    nt_promoter_tata 5,274 (registry, unverified); same source
  Gene expression and other TCGA data, grouped by tissue source site, StratifiedGroupKFold 5:
    tcga_luad_vs_lusc  1,017 (registry says 1,016), binary, AUROC; UCSC Xena/Toil
    tcga_brca_pam50    821 in the local cache (registry says 900), 4 classes (LumA, LumB,
                       Basal, Her2), weighted F1
- Classification datasets use StratifiedGroupKFold; regression uses GroupKFold; no random-split
  fallback (tests/experiments/_shared.py:2034-2103).
- CHECK, disagreements to settle in the text:
  - Chapter 3 gives LogD 5,039, Caco-2 2,161, hERG 1,415; the KIRBy loaders give 5,028, 2,158,
    1,412 (InChIKey deduplication, probably; unverified). Appendix A must say which curation each
    chapter used.
  - thesis.tex:1474 says R2 for reaction yield; the code uses RMSE.
  - thesis.tex:1480 says "5-fold GroupKFold throughout"; classification uses StratifiedGroupKFold.
  - thesis.tex:1484 TODO says reactions by template, TCGA by patient, antibodies by VH identity;
    the code groups reactions by additive, TCGA by tissue source site, and antibodies by heavy and
    light chain together at 95% identity.
- Best single representation per dataset and model: the f26 files carry some
  (tests/experiments/results/f26_redundancy_penalty_*.json) but used pre-pruning pools. Fill
  Table 5.2's last column from the server runs. TODO(prelim).

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
  - TCGA: expression foundation models (geneformer, bulkrnabert), methylation, copy number,
    miRNA, protein array, slide-image embeddings.
- How alike the representations are: f26 distance agreement (4.2.5) and the pruning step's
  correlation of out-of-fold predictions (REP_PRUNING_LOG.md; logd's top 9 are all mutually
  above 0.90; tcga has no pair at or above 0.95; every antibody *_cdr representation predicts
  identically to its base). The brain dump asks for these here.
- CHECK: chembert (256 features, jonghyunlee/ChemBERT_ChEMBL_pretrained) is not chemberta (384,
  DeepChem/ChemBERTa-77M-MTR). chembert is in 4 pools (herg_ki, logd, qm9, adme_hlm); its cache
  has never been checked against its builder.
- CHECK: gene_geneformer and gene_geneformer_v2 have the same 1,152 features and distance
  correlation 0.940; the v2 builder may have fallen back to v1 on the server.
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
- c05, learning curves. Subsamples of 10, 25, 50, 75 and 100% of the molecules, drawn as whole
  scaffold groups, 3 seeds. A search on each subsample, budgets capped at the training rows.
  Delta = hybrid minus best single representation.
- c17, hybrid against feature reduction at the same number of features. Five arms: the hybrid
  (the stored rf recipe at that budget, rebuilt per fold), PCA, mRMR (capped at 200 features),
  RFE (50-tree, depth-5 forest), random. All fitted on the fold's training rows; one downstream
  model, a 100-tree random forest. Regression datasets only. CHECK: the recipe itself is chosen on
  all rows (recipe_selected_on: 'all rows').
- c23, does the internal evaluator need to match the downstream model? nested_cv_evaluate; each
  outer fold searches with one evaluator. Conditions: same family, near family (rf with
  lightgbm, svm with gp, dnn with a smaller dnn), far family, and the downstream rf's exact
  settings (rf_exact). Deltas paired per outer fold. Widths must be matched.
- c19, importance used during training. Each model's own recipe at budget 100. Conditions:
  unweighted; features multiplied by the square root of importance, normalised to mean 1; each
  molecule weighted by how far its label is from its 10 neighbours' mean; both; features above
  the median importance only. Models rf, lightgbm, xgboost.
- c21, augmenting one representation, with decoys. One base representation at full width,
  candidates ranked by residual correlation from a pool of 50, blocks of 5/10/20/40, decoys
  planted. Counts how many of each kind are admitted.
- c33, does the search overfit? For each budget, the recipe is scored on untouched outer folds;
  bias = inner-search score minus held-out score (positive means the search over-promised).
  Budgets up to 8,000 and ratios 0.5-2.0 of the training rows.
- c24, stability: Nogueira's stability index of the selected features across folds.

5.3.5  Applied settings: OPIG and federated (methods, final)
- OPIG experiments (tests/experiments/opig_*.py). Each needs a paragraph of background on the task.
  - opig_blueprint_transfer: can a hybrid found on a large dataset be reused on a small one?
    Source logd (5,028 molecules), target herg_ki (1,412). Pool: representations in both
    datasets' pools with the same width (6: atom_pair, avalon, ecfp4, maccs, mordred, pdv). Four
    arms per target fold: best single representation (searched on that fold's training rows),
    native hybrid (searched on that fold), recipe transfer (logd's allocation, importance,
    budget and evaluator; features re-selected on herg_ki), exact-feature transfer (logd's feature
    indices applied unchanged). 5 group folds, RMSE, 5 downstream models.
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
  - CHECK: C8 fits on one 2,048-bit Morgan fingerprint; C9 fits on 100 features drawn from 6
    representations. So C8 and C9 differ in two ways at once, width and source. The "Promising
    experiments" file says C8 uses "all 6,000 numbers per molecule"; the code says 2,048.
  - The Lhasa data is not redistributable and the comparison needs Lhasa's FLuID code; the
    simulation uses the training molecules as the public set (a stated compromise).
  - [DIAGRAM D5.4] Two rows. Top: standard FLuID (sites -> teachers -> public set labelled ->
    student). Bottom: fed03 Part A C9 (sites -> each site's hybrid -> coordinator splits the
    budget -> each site's model on the shared feature list -> distance-weighted predictions). Mark
    what crosses between sites in each.

5.4  Results: when the hybrid helps
All TODO(prelim). The method is a search; the preliminary grids searched 2 to 6 configurations
per model against 16 to 48 on the earlier larger grid.
- Hybrid against ensembles (c11). flab_thermo_tresanco, Spearman, 5 folds, 32 configurations per
  outer search (16 per model): lightgbm hybrid 0.6246, ensemble 0.6023, best single 0.4809; rf
  hybrid 0.6364, ensemble 0.5984, best single 0.5896. On the small grid (3 folds): herg_ki rf
  hybrid 0.6114 RMSE, ensemble 0.6146; svm 0.5736 against 0.6001 (best single 0.5979); caco2 svm
  0.2827 against 0.2954; flab rf 0.6304 against 0.6182; flab lightgbm 0.5535 against 0.6323. The
  hybrid is ahead of the ensemble in 5 of 6 (model, dataset) pairs on the small grid.
- Hybrid against the best single representation by sample size (c05, flab_thermo_tresanco).
  Delta, Spearman, at 67 / 168 / 336 / 504 / 672 molecules: lightgbm, larger grid +0.2204,
  +0.0133, -0.0671, +0.0403, +0.0964; small grid n/a, -0.0343, -0.0994, -0.0548, +0.0443. rf,
  larger grid -0.1512, -0.0077, +0.0154, +0.0090, +0.0224; small grid -0.0698, -0.0448, -0.0463,
  +0.0345, +0.0320. At 67 molecules the training rows (about 45) are below every swept budget.
  herg_ki (small grid): rf +0.0171, +0.0042, +0.0008; svm +0.0201, +0.0118, +0.0256. caco2: rf
  -0.0037, +0.0022, +0.0086; svm +0.0105, +0.0112, +0.0149 (RMSE, positive = hybrid better).
- CHECK: the "Promising experiments" file says the larger grid is positive at 3 of 8 points and
  the small grid negative at 7 of 8. The files give 5 of 8 and 6 of 7.
- Where the hybrid does not help: datasets where one representation dominates. That is expected;
  the search should not manufacture a gain where selection has nothing to add. TODO(prelim):
  name the datasets from the server runs.
- The pattern to name: the hybrid's margin over the ensemble and the best single representation
  grows with how many configurations the search covers. CHECK before writing it as one claim: the
  "larger versus smaller search" comparison changes configuration count, fold count (5 against 3)
  and representation width together, and the ensemble and single-representation arms get no
  search, so part of the gap is the hybrid taking a maximum over more tries. c11 reselects the
  budget with one fold held out, which charges for that on the budget axis only.

5.5  Results: why the hybrid helps
- Against feature reduction at the same number of features (c17), flab_thermo_tresanco, Spearman,
  5 folds, 5 representations (6,528 features): budget 10: hybrid 0.5945, PCA 0.4860, RFE 0.4682,
  random 0.2145; budget 50: 0.6131, 0.5225, 0.5689, 0.5094; budget 250: 0.6029, 0.4824, 0.6003,
  0.5636. herg_ki at budget 10: hybrid 0.6304 RMSE, PCA 0.6776, hybrid ahead on all 5 folds. The
  pattern: the hybrid is ahead of PCA at every budget; RFE reaches the hybrid at 250 features. At
  budget 1,000 the arms are not width-matched (hybrid 376-434 features, PCA 536-537), so it is
  not reported. TODO(prelim).
- Does the internal evaluator need to match the downstream model (c23)? Small grid, held-out,
  budgets 100 and 250: flab Spearman shipped rf evaluator 0.6132, rf_exact 0.6128; herg_ki RMSE
  0.6124 and 0.6078. Configuration by configuration on inner-search scores, the exact match is
  better in 7 of 18 on flab and 8 of 18 on herg_ki. Suite run at budget 250: flab rf, same family
  0.6098, near family 0.6220, far family 0.6172. The pattern: the evaluator that best predicts
  what the downstream model will do with a feature set is not always the downstream model itself.
  TODO(prelim). CHECK: the "+0.0138" for exact match on the larger grid compares different
  widths ([250, 250, 250, 1000, 380] against [1000, 1000, 250, 401, 250] features per fold).
  Adelaide's note: this is informative rather than a negative; the text should say what it
  implies for the default (match is one of four evaluators searched, so the search already covers
  it).
- Importance used during training (c19). For tree models the feature-weighting condition cannot
  change anything: lightgbm_gain gives exactly zero importance to 66-78 of a 250-feature hybrid's
  features on herg_ki, and a feature multiplied by zero is a constant no tree can split on, so on a
  forest the condition is "delete about 27% of the features and rescale the rest". Suite run: flab
  rf 0.6373 to 0.6360; herg_ki rf 0.6055 to 0.6097 RMSE; caco2 lightgbm 0.3006 to 0.2981. Where a
  distance is used (k-nearest neighbours), weighting helps: herg_ki 0.6196 to 0.6111 RMSE, flab
  0.4302 to 0.5081 Spearman, and shuffling the same weights across features makes it worse (0.6371,
  0.3872). svm gets worse on both (herg_ki 0.5678 to 0.5861; flab 0.6356 to 0.6190). TODO(prelim).
  CHECK: the kNN numbers come from a separate harness (loops/c19/arms.py), not c19's script, and
  one fold carries the flab gain (fold 0, 0.3137 to 0.7226). Decision: kNN is not in c19's model
  list; add it to the script if this goes in the thesis. The distance half of this result
  belongs with c04 in 6.4.
- Augmenting one representation (c21). flab_thermo_tresanco, saprot base, Spearman: rf 0.5683 to
  0.5762, svm 0.5716 to 0.5830. Admitted per fold under rf: near-copy decoys 21, 11, 21; noise
  decoys 7, 2, 1; real mutation features 12, 7, 18. herg_ki (mhg_gnn base): rf 0.6117 to 0.6124,
  svm 0.5979 to 0.5983 RMSE. caco2 (graph_kernel base): rf 0.3188 to 0.3119, svm 0.3207 to 0.2975,
  no noise decoy admitted. The pattern: the score moves, and more than half of what is admitted
  on flab is decoy. TODO(prelim). CHECK: the
  "Promising experiments" file says up to 20 are admitted; the code's blocks go to 40, and flab rf
  fold 0 admitted 40.
- Tree ensembles and neural networks want different numbers of features and different
  representations (the brain dump's "small blueprint investigation"):
  - Budget, from stored inner-search scores (result_db, 10-representation pools, 5 folds): on caco2
    dnn is best at 2,000 features and still improving, rf at 250 and worse above it, lightgbm and
    svm at 500; on flab_thermo_tresanco lightgbm and svm are best at 50, dnn and rf at 2,000 (409
    realised). Held-out (c33, rf and svm only): herg_ki rf best at 250, svm at 941; flab rf 448,
    svm 100. TODO(prelim).
  - Representations, from f29 (each whole representation fed to each model, 5 folds): the rank
    correlation of the 10 representations' scores between rf and dnn is 0.09 on
    buchwald_hartwig, 0.22 caco2, 0.52 logd, 0.83 flab_expression_koenig, 0.95 tcga_brca_pam50;
    between rf and xgboost 0.84-0.99. On caco2 dnn's best representation is mhg_gnn while all
    four tree models pick graph_kernel or pdv.
  - Planned: which features a tree hybrid and a dnn hybrid select on the same dataset (Jaccard
    of selected features, by source representation). Nothing held-out exists yet.

5.6  Results: blueprints on specific datasets
- Deep dives: take the best hybrid on a dataset and ask whether its features match the
  chemistry (explain_hybrid, importance/hybrid.py:474-560). Choose datasets where the selected
  features are named (pdv, ecfp, maccs), so herg_ki, caco2 or logd. TODO(prelim).
- Stability of the selection across folds (c24): Nogueira index 0.074 on flab_thermo_tresanco,
  0.275 herg_ki, 0.447 caco2. TODO(prelim).
- [DIAGRAM D5.5] For one herg_ki blueprint: the budget split across representations as a
  stacked bar, and the top named features with their substructures drawn (for ECFP bits, the
  atom environment on an example molecule).

5.7  Results: OPIG
- Blueprint transfer (logd to herg_ki), RMSE in pKi, positive delta = arm beat herg_ki's best
  single representation: svm best single 0.6616, recipe 0.5678 (+0.0938), exact features 0.5824
  (+0.0791); dnn 0.6594, 0.5943 (+0.0651), 0.6815 (-0.0222); lightgbm 0.6318, 0.6019 (+0.0300),
  0.6005 (+0.0314); rf 0.5994, 0.6030 (-0.0037), 0.6197 (-0.0203); xgboost 0.6226, 0.6447
  (-0.0222), 0.6277 (-0.0052). Native hybrid for reference: rf 0.6040, xgboost 0.6411, svm 0.5693,
  dnn 0.6064, lightgbm 0.5925. The pattern: re-running the recipe transfers to the smaller dataset
  on 3 of 5 models; copying the exact features beats the best single representation on 2 of 5.
  That supports publishing a recipe found on a large dataset, not a feature list. TODO(prelim).
  CHECK: the "Promising experiments" table puts "wins 3/5" under the exact-features column; the
  run log (scratchpad/opig_verify/blueprint.log) gives recipe 3 of 5, exact features 2 of 5.
  Per-fold spread of the exact-feature delta for dnn is 0.1823.
- Hybrid against best single representation on an antibody deep mutational scan
  (opig_scaffold_hybrid, flab_expression_koenig, 4,275 antibodies, 3 outer folds, reduced search):
  delta in Spearman, dnn +0.0286, lightgbm +0.0431, rf +0.0366, svm +0.0287, xgboost +0.0257; best
  single was prot_t5 on every model; the winning hybrids mix prot_t5, igt5, ablang2 and esm2 at
  roughly 30/25/25/20. The mutation-difference representations this experiment was written for
  were not selected by 9 of 15 winning recipes, so report it as a hybrid-against-best-single
  result, not a result about those representations. TODO(prelim).
- For each OPIG task included: background (what the task is and how it is used), the hybrid
  result, and which features were selected.

5.8  Results: federated
- fed03 Part A, preliminary probe (herg_ki sites, 3 folds, seed 0, budget 100): one model per site
  with distance-weighted predictions, on the fingerprint 0.4711 MCC, on the selected features
  0.4981, ahead on all 3 folds; the same selection with the molecules pooled (C4) 0.5426; naive
  pool 0.4995. The pattern: the selection gain survives keeping the molecules at their sites, at
  about 60% of its pooled size (+0.027 against +0.043). TODO(prelim): single budget, one seed.
  CHECK: the fingerprint arm and the selected arm differ in width and source at once (5.3.5);
  add an arm with one model per site on all six representations to separate them.
- Other Part A results after the metric fix (a dated memory note, re-check): C4 0.5355 against the
  same selection computed on pooled molecules 0.5335; allocation rules within 0.004 MCC of one
  another; budget 400 0.5485, 100 0.5426, 50 0.5287.
- Attempts that did not work, briefly (brain dump: "failed attempts"):
  - fed01, different label noise per site: removing even every truly flipped molecule gains at
    most +0.0073 to +0.0677 MCC; every vote threshold for removing flagged molecules loses 0.03 to
    0.11 (RESULTS.md, dated).
  - fed02, each site holds one representation of the same molecules: on herg_ki the best single
    site beats every combination; caco2 and logd each have a different winner (3 folds; fold spread
    about ten times the gaps). The server run never completed.
  - fed04, per-region feature importance on k-means sites: +0.0004 over 40 paired folds.
  - fed05, antibody sites: the noisy site receives 8 features against 46 for each clean site.
    CHECK: its C2 arm moves features between sites by raw array index, which RESULTS.md flags.

5.9  Discussion
- Does the search overfit? Arguments both ways (brain dump). For: features are chosen on the same
  training labels the model sees; Dablander found label-driven bit selection did worse than an
  unsupervised one. Against: c33 scores each budget's recipe on untouched outer folds, and the
  inner search under-promised at every budget (bias -0.0095 to -0.0354 on flab, -0.0338 to
  -0.0511 on herg_ki); c11 charges the hybrid and the best single representation the same cost
  for choosing a budget. The robustness side: a model that learns from strong features drawn from
  several sources is not relying on one representation's view. Existing prose:
  thesis.tex:1626 (Ambroise2002, Cawley2010, Nogueira2018 and others, not in refs.bib).
  TODO(prelim) for the c33 numbers.
- Where hybrids do not help (one representation dominates): a sign that the search is not
  inflating results where selection has nothing to add.

5.10  Limitations
- Compute and run time. Hours per dataset for the full search; wglobal_power2 alone was 71% of
  logd's 14.45 hours on the f33 record.
- A change to the data (different splits), a new representation or a new model means repeating
  the search.
- High cost before the gain is known; some datasets are not helped, and a standout
  representation in the pool is a hint but not a guarantee.
- Reproducibility is harder: the blueprint stores the recipe, and a rebuild on other molecules
  can pick other features.
- Simpler feature selection may be enough for some tasks.
- Existing prose: thesis.tex:1865-1869 (small datasets, frozen-embedding ceiling, sparse
  sequence encodings, engineering burden, the noise ceiling from Kramer2012). Re-check the
  "no gain under about 200 molecules" claim; it predates the small-dataset allocation work.

5.11  Conclusion
- One paragraph per research question; bridge to Chapter 6.


CHAPTER 6  HYBRID REPRESENTATIONS FOR SIMILARITY AND LABEL NOISE  (about 25 to 30 pages)
Planned as the second hybrid paper. Results are preliminary; methods are final.

6.1  Introduction
- Short. Chapter 4 showed where the similarity assumption fails; Chapter 5 built a
  representation chosen for the task. This chapter asks whether that representation repairs
  what Chapter 4 found broken: similarity, noise detection and noise mitigation.
- Research questions as bullets.
- TODO(lit): Northcutt et al. 2021 (Confident Learning; northcutt2021confident in refs.bib, not
  yet cited), Brodley and Friedl 1999, Wilson 1972 (not in refs.bib).

6.2  Methods
6.2.1  Similarity in a hybrid
- Two spaces: an ordinary hybrid (objective='predictive'), and a similarity hybrid
  (objective='similarity': for each held-out molecule, its 5 nearest neighbours by distance in the
  hybrid, taken from other scaffold groups; the molecule's activity predicted as their
  distance-weighted mean; scored by Spearman against the truth, or 2(AUROC - 0.5) for
  classification; no downstream model fitted; _readacross_objective_score, find_best_hybrid.py:
  2079-2186; Manhattan distance by default).
- Undefined on tcga_brca_pam50 (4 classes), tz_her2_multiclass, and tz_her2_binary (a held-out
  block can hold one edit-distance shell, so no molecule has neighbours from another group);
  dataset_policy.py:606-645. Similarity recipes exist only for the six molecular datasets.
- CHECK: tests/precompute_hybrids.py:757-761 forces the whole pool for every non-predictive
  objective, so the similarity precomputes never search representation count.
- Importance-weighted distance: each feature scaled by the square root of its importance before
  the distance (existing prose thesis.tex:1683-1687; CHECK the weight convention against the code).
- [DIAGRAM D6.1] The two objectives side by side: predictive (hybrid -> downstream model ->
  held-out score) and similarity (hybrid -> 5 nearest neighbours from other groups -> predicted
  activity -> Spearman). Same search, different score.

6.2.2  Noise conditions
- The conditions are defined once in Chapter 2.4 (the NoiseInject noise model). Table 6.1 (a
  Dablander-style table the brain dump lacks): one row per noise type used in this chapter, what is
  injected, its parameters, and which datasets. Covers Gaussian, outlier, grouped-shifted,
  censoring, and real labels with nothing injected (src/kirby/noise_spec.py:96-134).

6.2.3  The three noise-mitigation experiments (final methods)
Brain dump: three that look like they work, method by method. Chosen: detection-limit correction
(n30), the twelve LNL methods (n09), cleaning labels as they come (n18).
- Every experiment compares the hybrid against each downstream model's own top 2 single
  representations (base_1, base_2), per the standing rule. CHECK: the preliminary runs below used
  a fixed list instead (pick_noise_base_rep, deleted in 57dc506): onehot on flab_thermo_tresanco
  and ecfp4 on herg_ki and caco2. The commit message says onehot is the worst of
  flab_thermo_tresanco's 10 representations (0.3471 Spearman with rf). So every preliminary
  "hybrid beats onehot/ecfp4" number is against a weak single representation, not the best one.
- n30, correcting labels clipped at a detection limit (tests/experiments/noise/n30_censored_
  likelihood.py). Condition: censoring (training labels past a limit set to the limit; test
  labels untouched). Label arms: clean, clipped, constant_lift (every clipped label raised by one
  amount estimated from the non-clipped training molecules), per_molecule_lift and its shuffled
  control, truncated_normal_mean, drop_clipped, winsorising at 1.0/1.25/1.5. Model swaps:
  xgboost_aft (a censored likelihood), rf_l1, lightgbm_huber. Spaces: base_1, base_2, and the hybrid
  rebuilt per fold from the clipped labels. Recovery = (clipped - arm)/(clipped - clean), refused
  when the gap is under 2% of the clean score.
  CHECK: the probe's lift is limit + sd*phi(a)/(1-Phi(a)); the mean of a normal above the limit is
  p + sd*phi(a)/(1-Phi(a)), smaller by (limit - p). n30 keeps both (constant_lift reproduces the
  probe; truncated_normal_mean is the textbook mean).
- n09, twelve LNL methods (n09_lnl_survey.py; methods in src/kirby/utils/lnl_methods.py:
  2272-2288): small_loss, confident_learning, distance_selection, multistage_cleaning,
  early_learning, uncertainty_curriculum, dividemix, meta_weight_net, mentornet,
  contrast_to_divide, heteroscedastic_gp, evidential_kernel. Conditions: plain dnn and each method,
  on the single representations, the hybrid, and the importance-weighted hybrid. The hybrid is
  re-searched per noise level and fold on the noisy labels. dnn: layers 256/128/64, 200 epochs,
  dropout 0.2. Table 6.2: one line per method saying what it does (the "Promising experiments"
  file has these).
- n18, cleaning labels as they come, nothing injected (n18_clean_the_real_labels.py). Cleanups:
  remove, downweight, label_smooth, knn_smooth, oof_smooth, gp_smooth, quantile_clip. Flag rates
  1%, 5%, 20%. Suspicion scores: neighbor_residual, cross_val. Spaces: base_1, base_2, the hybrid,
  the similarity hybrid, each built once per fold on the uncleaned labels. QM9 is the control: its
  labels come from a calculation. CHECK: the preliminary run swept 5 suspicion scores; the script
  defines 2.
- Existing prose to reuse: sample removal, noise estimation, label smoothing and weighting,
  recovery rate (thesis.tex:1713-1817). Needs its "TODO: get sources" (1732) and the informal
  first-person lines (1716) fixed.

6.3  Results: similarity
- Importance-weighted distance for k-nearest-neighbour prediction (c04, from the complexity suite,
  moved here because it is a distance result). All representations concatenated; features scaled
  by the square root of their importance, so zero-importance features drop out. k = 10, R2
  weighted minus unweighted: flab_thermo_tresanco +0.2164 (unweighted -0.0375, weighted 0.1790),
  caco2 +0.0546, herg_ki +0.0487 (10 representations, 5 folds). On the later 5-representation,
  3-fold run: +0.0629, +0.0687, +0.0556. The pattern: on all three datasets, weighting by
  importance gives a better distance than the raw features. TODO(prelim).
- The kNN half of c19 (5.5) belongs with this.
- The similarity hybrid as a predictor: in n18 the similarity hybrid is worse than the ordinary
  hybrid for a random forest on all three datasets with no cleaning (herg_ki 0.6235 against 0.6029
  RMSE; caco2 0.3226 against 0.3090; flab 0.6087 against 0.6279 Spearman). Expected: it is chosen
  for neighbour prediction, not for a forest. TODO(prelim).
- TODO: decide whether to add an s-series result (s02 weighting and s10 exponent-and-voting are
  complete on three datasets per recent commits). "Promising experiments" has none.
- [DIAGRAM D6.2] Same layout as D4.1 but in the similarity hybrid: activity difference against
  distance for sampled pairs, single representation against hybrid.

6.4  Results: noise detection and estimation
- "Promising experiments" has no detection or estimation result. Evidence that exists (dated
  files, re-check): on herg_ki with outlier noise on 10% of labels (chance precision 0.095), ten
  single representations detect the noisy labels at average precision 0.4036-0.4326 and every
  combination of them lands inside that range (/Volumes/seagate/local_noise_run/loop_headroom/
  FINDINGS.md §4). Decision for Adelaide: include, or wait for n11/n21/n23.
- The paradox to state: identifying which labels are noisy, and estimating how noisy, is possible
  for outliers and systematic offsets; correcting them is a different problem (6.5).

6.5  Results: noise mitigation, by noise type
Method by method first, then by noise type. All TODO(prelim).
6.5.1  Censoring: correcting labels clipped at a detection limit
- herg_ki, top 10% of training labels (94 of 941) clipped at 6.86 pKi, RMSE on test molecules,
  mean of 3 folds (rf / lightgbm / knn / svm): never clipped 0.6032 / 0.6087 / 0.6871 / 0.6841;
  clipped 0.6588 / 0.6499 / 0.7158 / 0.7148; clipped then corrected 0.6118 / 0.6248 / 0.6937 /
  0.6847; clipped molecules deleted 0.7706 / 0.7631 / 0.8037 / 0.8162. Also herg_ki at 25% and 40%
  clipped and caco2 at 25%: the corrected labels beat the clipped labels in 48 of 48 comparisons
  (every model, fold and condition). Share of the clipping damage recovered: 45-98% across 4
  conditions and 4 models. Raising every clipped label by the same amount does as well as a
  per-molecule amount (dealing the per-molecule amounts to the wrong molecules does as well too).
  Deleting the clipped molecules is worse than keeping them clipped, on every model.
- Source: the Seagate probe (loop_headroom/censoring_probe.py), on ecfp4. n30 is the suite
  version and has not run.
- Note for the text: real herg_ki is fetched with relation "=", so censored ChEMBL records are
  filtered out; this mechanism is studied by injecting it. caco2 and logd are already clipped at
  their 1st and 99th percentiles by the loader.

6.5.2  Gaussian noise: twelve LNL methods (n09)
- flab_thermo_tresanco, 672 antibodies, Spearman, mean of 3 folds. Noise standard deviation 0.5x
  and 1.5x the labels' own (5.18 and 15.54 C against a spread of 10.12 C).
  - Plain network, no added noise: onehot 0.4167, hybrid 0.5418, importance-weighted hybrid
    0.5857.
  - At 0.5x: neither input lost accuracy (onehot 0.4534, hybrid 0.5991); 0 of 12 methods beat the
    plain network on either input.
  - At 1.5x: onehot 0.2729, hybrid 0.4859, weighted hybrid 0.5118; 8 of 12 methods beat the plain
    network on each input. The best method on onehot (small_loss, 0.3568) is below the plain
    network on the hybrid (0.4859).
  - The hybrid is ahead of onehot at every noise level, for the plain network and for all twelve
    methods.
  - The pattern: at the noise level where the methods help, the input changes the score more than
    the method does. Per-method tables go to Appendix D.
  - CHECK: onehot is the worst single representation on this dataset (6.2.3). The plain onehot
    network ranges 0.257-0.644 across folds with no noise.

6.5.3  Real labels, nothing injected: cleaning (n18)
- Random forest, 3 folds, hybrid budget 50; each cell the mean over all settings run.
  - herg_ki, RMSE: no cleaning ecfp4 0.6032, hybrid 0.6029, similarity hybrid 0.6235. Every cleanup
    on every space is at or above its no-cleaning score (downweight on ecfp4 0.6040).
  - caco2, RMSE: no cleaning 0.3242 / 0.3090 / 0.3226; downweight equals no cleaning on the hybrid
    (0.3090); every other cleanup is above it.
  - flab_thermo_tresanco, Spearman: no cleaning onehot 0.4068, hybrid 0.6279, similarity hybrid
    0.6087. On the hybrid, every cleanup is above no cleaning (remove 0.6429, downweight 0.6407);
    on onehot, every cleanup is below (remove 0.3498).
- The pattern: on the two small-molecule datasets, cleaning real labels did not improve on leaving
  them; on the antibody dataset the direction depends on the space. TODO(prelim).

6.5.4  Outliers and grouped offsets
- No preliminary result in "Promising experiments". Experiments that cover them: n07 (seven
  corrections after detection, outlier_p05), n24 (delete or keep flagged molecules), n34 (one
  offset per scaffold group, grouped_shifted), n29 (one offset per ChEMBL document). Decision for
  Adelaide: which one, if any, goes in the first draft.

6.5.5  By noise type (the discussion the brain dump asks for)
- Gaussian noise is the standard assumption and the hardest to fix: every label carries some
  error of unknown size and direction; a detector that compares a molecule to its neighbours
  compares it to labels that are also noisy; even a correctly flagged label has no known
  correction. Cleaned datasets often give worse models than the noisy ones they came from
  (confirmation report; 4.4.2). Better similarity does not remove that problem.
- Outliers are easier to find (a point that sticks out). Censoring has a known direction and a
  known location, which is why one constant recovers most of the damage (6.5.1).
- Practical takeaways (brain dump): what kind of noise do you suspect? Assays pooled from many
  laboratories (shifted or systematic offsets); molecules from very different domains; a single
  controlled experiment with a detection limit (censoring). If the source and amount are unknown,
  what should you do?
- [DIAGRAM D6.3] One panel per noise type (Gaussian, outlier, grouped-shifted, censoring): true
  label against observed label, with the detectable part and the correctable part marked. Shows why
  censoring is fixable and Gaussian is not.

6.6  Conclusion
- Short; limits stated plainly. Bridge to Chapter 7.


CHAPTER 7  DISCUSSION AND FUTURE DIRECTIONS  (about 12 to 15 pages)
The closing argument from the brain dump, then limitations and next steps. Each chapter's own
conclusions carry the detail (Dablander points readers to them the same way).

7.1  Summary
- One paragraph per chapter: what it asked and what it found.

7.2  Handling noise realistically
- Model the noise before choosing a method: noise is not only Gaussian, and if it is, that is the
  hardest case.

7.3  No single method fits every dataset
- Methods for learning with noisy labels exist; mitigation is harder; both depend on the data,
  the amount and type of noise, the number of samples, and whether the features describe the data.
- Count, across the thesis, the results where datasets preferred different settings: allocation
  method, representation count, importance method, internal evaluator, downstream model, budget
  (5.2.3); the best representation changing with the noise condition (Chapter 3, 3.5).
- What the thesis provides is a way to test: methods that might work, and a procedure that gives
  a supervised task its best chance by choosing the features for that task, instead of a new
  architecture tuned to one problem.

7.4  Simple is often better
- Chapter 3: the most consistently noise-robust models were random forests and other tree
  ensembles; neural networks varied (standout pairs exist, such as NGBoost on PDV). TODO: re-check
  the NGBoost/PDV example against Chapter 3's current text (3.4: NGBoost has the highest AUC_norm on
  all six representations under Gaussian noise).
- Chapter 5: the best hybrids often use fewer features than the largest budget; k-nearest
  neighbours with a small, task-relevant feature set does better than on raw features (6.3).
- Complex noise-mitigation methods that had no effect: say which, without method-level detail
  (n09 at 0.5x noise: 0 of 12 beat the plain network; fed01; the confirmation report's
  distance methods). Why methods that work on images or financial data do not carry to molecular
  assay data. TODO(lit).
- Methods tuned to one problem do not generalise; simple ones do not either (feature selection
  needs tuning per dataset). The range of tasks (binding affinity, ADME, antibodies) is why models
  are used at all; over-optimising to one loses generality.

7.5  Limitations
- Noise mitigation: with unknown noise on every label and no ground truth, correcting labels is
  hard; outlier and systematic-error detection are possible.
- Recommendation: learning with noisy labels over direct mitigation. Models can adapt to noise
  during training more reliably than a per-sample guess at the error's direction. They can be paired
  with uncertainty estimation where it is reliable: per-sample uncertainty tracks noise weakly
  (Chapter 3, 3.8: per-molecule correlation between -0.16 and 0.16 under six of seven conditions),
  while the overall level of uncertainty rises with the amount of noise.
- The search's cost and reproducibility (5.10).

7.6  Future directions
- Classification and label flipping.
- Other uses of hybrid features: they worked, indirectly, in the federated setting (5.8); as
  inputs to generative models, and what the search would optimise for that task.
- A faster search: it is too heavy to be practical as it stands.
- From the old outline, kept: forward compatibility made symmetric (does the hybrid improve a weak
  new embedding?); carrying a blueprint across domains; a fuller federated deployment with Lhasa.
- Existing prose: thesis.tex:1871-1896 (the confirmation report's next steps, partly done; the
  "what this means in practice" paragraph).


SUMMARY OF RESEARCH CONTRIBUTIONS
- A short closing list (following the peer theses): the NoiseInject paper and the planned
  hybrid-representation papers (Chapters 5 and 6), the UK-QSAR conference presentation, the OPIG
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
- A: datasets and curation (Chapters 3 to 6): QM9, LogD, Caco-2, hERG, and the sixteen datasets
  of Chapter 5, with per-dataset provenance, curation counts before and after, accession numbers,
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
- D: extended result tables for Chapters 4 to 6: Chapter 4's per-representation feature counts,
  tie census and distance-activity tables; Chapter 5's per-dataset hybrid-against-best-single
  tables, c17 per budget, c23, c19, c21 with decoy counts, blueprint transfer per fold, fed03 arms;
  Chapter 6's per-method n09 tables at both noise levels, n18 per cleanup and setting, and n30 per
  arm and clipped fraction.
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
