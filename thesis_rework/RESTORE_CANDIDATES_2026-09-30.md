# What the sentence-simplification pass cut that may be worth putting back

Nothing here has been applied. Each item is something that pass cut which a reviewer judged to be content rather than wordiness.
★ marks my recommendation: findings, qualifiers, your framing and cited claims in the main text. Unstarred items are mostly settings and seeds.
AF items are in the Additional files now (the former appendices), so putting them back costs the thesis no pages.
Reply with what you want, e.g. "the starred ones", "Ch1 all, Ch4 ★ only", or specific IDs.

## Chapter 1

**★ Ch1-1. Introduction (opening, questions paragraph)**: The thesis statement that opens the paragraph.

> This thesis addresses the challenge of conducting molecular property prediction when the underlying experimental data contain noisy labels.

**★ Ch1-2. Introduction (opening, questions paragraph)**: The claim that the gap is about the relative importance of representation versus architecture. The shortened sentence says no one has compared them at all, which is a broader claim.

> No work has systematically compared the relative importance of representation versus architecture choice for noise robustness

**★ Ch1-3. Introduction (opening, questions paragraph)**: How the third question is answered: noise is generated and added in several ways, both to clean data and to data with experimental noise. The pass folded the second half into the question, so the question now reads oddly.

> Third, does the type of label noise matter, or only its quantity? To answer this, we generate and add label noise in several ways, both to clean data and to data with inherent experimental noise.

**★ Ch1-4. Introduction (opening, questions paragraph)**: The sentence that says what the four questions add up to: what, if anything, makes a QSAR model robust to noise.

> Together, these questions aim to identify what, if anything, makes a QSAR model robust to noise.

**★ Ch1-6. Outline of the Thesis and Contributions (Chapter 4 paragraph)**: The caveat that Chapter 4 introduces no method and that its measurements are smaller in scale than Chapter 3's.

> It introduces no method of its own, and its measurements are on a smaller scale than those of Chapter~\ref{chap:robustness}.

**★ Ch1-7. Outline of the Thesis and Contributions (Chapter 6 paragraph)**: The word "selected". Without it, the reader does not learn that the hybrid representations are chosen for this prediction task.

> hybrid representations selected for predicting each molecule's activity from its five nearest neighbors


## Chapter 4

**★ Ch4-1. Chapter 4 opening paragraph**: The words-only opening you approved earlier today. It gave the chapter's argument in order: few features vary, stereoisomers get the same vector, distance is diluted, distance-based noise methods did worst.

> This chapter asks whether the standard ways of comparing molecules suit the dataset and task at hand. Because general-purpose representations are built for all of chemistry, we first count how many of their features vary on a single dataset. On hERG $K_i$, most ECFP4 features take a single value in almost every molecule. Some representations also give different molecules the same vector, mostly stereoisomers, so that no model can tell those molecules apart. Because a distance compares every feature, features irrelevant to the task dilute it, and different representations disagree about which molecules are close. In an exploratory study of learning with noisy labels, the configurations with the lowest retention under noise all used a distance, and noise detection by neighbors usually made the model worse.

**★ Ch4-2. Constant and Near-Constant Features**: The sentence saying the useful features are a small subset of the varying ones, and that which subset depends on the task.

> The features that matter for a task form a small subset of those that vary, and which subset matters depends on the task.

**Ch4-3. Constant and Near-Constant Features**: The counts showing that bits which are on in every molecule do occur: on Caco-2, LogD and MACCS on hERG Ki.

> whereas 1 of the 194 constant ECFP4 bits is on for Caco-2 and 1 of 39 for LogD. On hERG $K_i$, 2 of the 24 constant MACCS features are on.

**Ch4-4. Constant and Near-Constant Features**: The whole Dablander paragraph: his cited claim that rare substructures appear in few compounds, how it differs from our count after folding, and bit collisions making a selected bit hard for a chemist to read.

> \citet[p.~112]{Dablander2024} states, without giving a count, that the least frequent substructures usually appear in only a few compounds. His statement concerns substructures before hashing, whereas our count is of bits after folding to 2,048. Bit collisions (Chapter~\ref{chap:background}) add a further difficulty, because one bit can stand for several substructures, which makes a selected fingerprint bit harder for a chemist to read than a named descriptor. This point returns in Chapter~\ref{chap:method}, in the discussion of blueprints and what they tell a chemist.

**Ch4-6. Flaws in Similarity (distance methods paragraph)**: That the 2,000-molecule sample is fixed, and its seed.

> a fixed random sample of 2,000 molecules (seed 0)

**★ Ch4-7. Flaws in Similarity (all-pairs vs five-neighbor results)**: The hERG Ki ECFP4 example: all-pairs correlation 0.06 against five-neighbor Spearman 0.72, RMSE 0.63 against a label standard deviation of 0.92 pKi.

> On hERG $K_i$, ECFP4 gives an all-pairs correlation of 0.06, while its five-neighbor prediction reaches 0.72, with an RMSE of 0.63 p$K_i$ against a label standard deviation of 0.92 p$K_i$.

**Ch4-8. Flaws in Similarity (MCS paragraph)**: The seed, and the MCS matching settings (ring atoms match only ring atoms; only complete rings kept).

> (seed 0) ... matching ring atoms only to ring atoms, keeping only complete rings and allowing each pair 2 seconds

**★ Ch4-9. Flaws in Similarity (learned similarity paragraph)**: The sentence linking this section to the next: every similarity-based method inherits the dilution just measured.

> Every method in Section~\ref{sec:flaws:consequences} takes a similarity as input and inherits the dilution described here.

**★ Ch4-10. Exploratory Study of Learning with Noisy Labels, second paragraph**: That LNL has been studied in QSAR, and the Cortes2015 result on which algorithms tolerated noise worst. Chapter 2 cites Cortes2015 only for its setup, not these findings.

> , and such methods have been studied across machine learning, including QSAR. In the study of \citet{Cortes2015}, gradient boosting tolerated noise worst, SVM and GP models with polynomial kernels were sensitive at high noise in some cases, and the other algorithms degraded smoothly and linearly with the noise level.

**Ch4-11. Exploratory Study of Learning with Noisy Labels (methods that held up)**: What the 15 configurations are: 3 kernels and 5 representations.

> (3 kernels and 5 representations)

**★ Ch4-12. Exploratory Study of Learning with Noisy Labels (limitations)**: That Chapter 6 repeats the question on more datasets and noise types.

> and Chapter~\ref{chap:similarity} repeats the question on more datasets and noise types

**★ Ch4-13. Exploratory Study of Learning with Noisy Labels (two reasons)**: The support for the first reason: each representation encodes a different kind of information. This sentence carried the kishimoto2023 citation.

> PDVs encode global molecular properties, ECFP4 contains substructure information and MHG-GNNs encode the molecular graph \citep{kishimoto2023}, and any one of these sources alone may not be sufficient for robust similarity estimation.

**★ Ch4-14. Noise Detection by Neighbors (exploratory cleaning study)**: The caveat on what the recovery table contains: only configurations on a single representation. The hybrid ones used an allocation method no longer in the thesis.

> Table~\ref{tab:code_input_neighbor_recovery} holds only the configurations built on a single representation, because those built on a hybrid used an allocation method that is no longer part of the thesis.

**Ch4-15. Noise Detection by Neighbors (sample weighting paragraph)**: That the best configuration (+17.2% at sigma 0.6) also recovered +11.4% at sigma 0.3.

> The best configuration described above, which used exponential sample weighting, also recovered +11.4\% at $\sigma = 0.3$.

**★ Ch4-16. Conclusions (first finding)**: The two numbers behind this finding: 1,815 of 2,048 ECFP4 features near-constant on 1,412 hERG Ki molecules, and 56 hERG Ki molecules sharing a vector.

> On hERG $K_i$, 1,815 of the 2,048 ECFP4 features take one value in at least 95\% of the 1,412 molecules ... ECFP4 gives 56 of the 1,412 hERG $K_i$ molecules a vector shared with another molecule, all of them stereoisomers (``Molecules with identical feature vectors'').

**★ Ch4-17. Conclusions (third finding)**: The caveat that neighbor-based judging did not always fail: one neighbor-based configuration gave the best recovery at the higher noise level.

> It did not always fail, since one neighbor-based configuration gave the best recovery at the higher noise level.


## Chapter 5 opening and Introduction

**★ Ch5a-1. Combining Representations and the Alternatives**: The Mayr2018 result: on 1,310 ChEMBL assays, feedforward networks on ECFP beat graph convolutions and SMILES recurrent models. Mayr2018 is now cited nowhere in the thesis.

> and in 1,310 ChEMBL assays, feedforward networks on ECFP fingerprints outperformed graph convolutions and SMILES-based recurrent models \citep{Mayr2018}

**Ch5a-2. Combining Representations and the Alternatives**: Two cited findings outside small molecules. For antibodies, general-purpose models sometimes beat domain-specific ones. In cancer genomics, adding omics layers can lower performance when the data types are not properly combined.

> For antibodies, general-purpose models sometimes outperform those built for the domain \citep{Kenlay2024}. In computational pathology, no individual foundation model dominates on clinical endpoints \citep{ElNahhas2025}, and in cancer genomics, integrating additional omics layers can reduce performance when the data types are not properly combined \citep{Steyaert2023}.

**Ch5a-3. Combining Representations and the Alternatives**: The cited claim that new models appear faster than they can be compared, so choosing one representation is not getting easier. Bommasani2021 is now cited nowhere in the thesis.

> Increasingly, new models appear faster than we can compare them \citep{Bommasani2021}, so the choice is not getting simpler.

**Ch5a-4. Combining Representations and the Alternatives**: The cited claim that single modalities give more sensitive and robust predictions when combined with independent signals from other modalities.

> Different sources encode distinct aspects of the same biology. Single data modalities, noisy or incomplete on their own, give more sensitive and robust predictions when combined with independent signals from other modalities \citep{Steyaert2023}.

**Ch5a-5. Combining Representations and the Alternatives**: The cited ensemble result: in computational pathology, an ensemble of the four best foundation models had a mean AUROC 1.2% above the best single model, over 31 clinical tasks.

> In computational pathology, an ensemble built on the four best foundation models reached a mean AUROC 1.2\% above the best single model over 31 clinical tasks \citep{ElNahhas2025}.

**Ch5a-6. Combining Representations and the Alternatives**: The cited QSAR use of dimensionality reduction: turning many correlated descriptors into fewer uncorrelated ones. Niazi2023 is now cited nowhere in the thesis.

> and in QSAR it turns many correlated descriptors into fewer uncorrelated ones \citep{Niazi2023}

**★ Ch5a-7. Combining Representations and the Alternatives**: The step saying why PCA's dependence on units matters in this thesis: fingerprint bits, descriptors and embeddings are joined.

> and this matters when fingerprint bits, descriptors and embeddings are joined

**★ Ch5a-8. Combining Representations and the Alternatives**: The advantage of late fusion: it avoids the dimensionality problem. Only its cost is left.

> which avoids the dimensionality problem but loses interactions between features from different sources

**★ Ch5a-9. Combining Representations and the Alternatives**: The reason a new downstream model gets its own search: it may prefer a different feature set.

> A new downstream model may prefer a different feature set, and it therefore receives its own search.

**Ch5a-10. The Adaptive Hybrid Representation**: The conclusion of the paragraph: therefore we built a method that selects the best features for one dataset and model, which should filter out noisy features.

> With this in mind, we decided to develop a method that selects the best features for a given dataset and model. In principle, such a method should filter out noisy features and isolate the features that carry the most relevant information.

**★ Ch5a-11. The Adaptive Hybrid Representation**: What the two uses are: distances in an ordinary hybrid, or a search that picks features by how well each molecule's activity is predicted from its five nearest neighbors.

> A hybrid can also be used to measure similarity in two ways. Distances can be computed in an ordinary hybrid, or the search can pick features by how well each molecule's activity is predicted from its five nearest neighbors. Both approaches are defined in the section ``Similarity in a hybrid'' of Chapter~\ref{chap:similarity}.

**★ Ch5a-12. Blueprints and Their Chemical Interpretation**: The cited claim that explainability is a recognized need in drug discovery. JimenezLuna2020 is now cited nowhere in the thesis.

> Explainability is a recognized need in drug discovery \citep{JimenezLuna2020}

**★ Ch5a-13. Beyond Small Molecules**: The qualifier that one-hot encodings can still be useful despite being sparse.

> One-hot encodings can nevertheless still be useful.

**★ Ch5a-14. Combining Representations and the Alternatives (Table tab:hybrid_alternatives caption)**: The caption's definitions of the table columns. In particular, "Several representations" means one model sees several representations at once.

> Methods for combining or reducing representations, compared by whether they use the labels, keep the original features, let one model see several representations, and must refit to add a representation.


## Chapter 5 Methods: The Hybrid Search

**Ch5b-1. Methods: The Hybrid Search (preprocessing paragraph)**: How a single representation is preprocessed when it is scored on the outer test fold. This is a reproducibility detail for the comparator score.

> whereas a single representation scored on the outer test fold is standardized like the hybrid.

**★ Ch5b-2. The Search Space (six settings paragraph)**: The reason the importance method is a searched setting, with its citation. The downstream model, budget and representation count each still have their own reason. The importance method now has none.

> The importance method is LightGBM gain, TreeSHAP or DeepLIFT. It is searched because importance depends on the model as well as on the data \citep{Dong2020}. The three methods come from different model families and rank features differently, and a different ranking builds a different hybrid.

**★ Ch5b-3. The Search Space (representation count paragraph)**: The contrast with pooled feature selection over all representations, which has no representation-count setting.

> and pooled feature selection over all representations has no such setting.

**Ch5b-4. Feature Importance Scoring**: How the features are preprocessed before each importance fit. The preprocessing paragraph covers the hybrid and the standalone scoring, but not the importance fits, and the importance table does not give it either.

> Before fitting, missing values are set to 0 and each feature is standardized to mean 0 and variance 1 on the same molecules.

**Ch5b-5. Augmentation (donors paragraph)**: The sentence linking the Bahia citation to the design choice. It said the literature supports offering 3D donors.

> Evidence from the literature suggests that 3D donors are worth offering.

**★ Ch5b-6. Augmentation (conditions paragraph)**: Her reason for the second use of augmentation: a chemist keeps the representation they trust and learns which few outside features help.

> This use matters for adoption, because a chemist can keep the representation they already trust and learn which few outside features help, and how many.


## Chapter 5 Experimental Setup to end of chapter

**★ Ch5c-1. Methods: Experimental Setup (opening)**: Two points. The comparator is refit and scored on the same held-out samples as the hybrid. Choosing it on held-out samples would give it an unfair advantage.

> then refit on those samples and scored on the same held-out samples as the hybrid. ... Choosing it on the held-out samples would give it an advantage that the hybrid does not get.

**★ Ch5c-2. Methods: Experimental Setup (opening)**: The reason both widths are recorded: part of any score difference is a difference in width.

> since a difference between the two scores is partly a difference in width as well as in method.

**★ Ch5c-3. Methods: Experimental Setup (opening)**: How the sign works for RMSE.

> (for RMSE the sign is flipped)

**★ Ch5c-4. Datasets and Splitting (Groups by domain)**: The warning that Chapter 3 used a different scaffold key and different curation for the same four datasets. Chapter 3 treats each acyclic molecule as its own group; this chapter puts them all in one group.

> Chapter~\ref{chap:robustness} used a different scaffold key and different curation for its versions of LogD, Caco-2, hERG $K_i$ and QM9, and Appendix~\ref{app:datasets} lists the differences.

**Ch5c-5. Datasets and Splitting (Groups by domain)**: Two things. The pointer to where the identity measure, cluster sizes and humanness labels are given. And what an unseen trastuzumab fold contains.

> after \citet{Kenlay2024}, and the identity measure, the cluster sizes and the humanness labels are given in Appendix~\ref{app:datasets}. ... A held-out fold therefore contains variants at mutation counts not seen in training.

**Ch5c-6. Models and metrics**: The kernel used on humanness. The evidence number (44 of 60 hours). The fact that the two datasets were excluded under different settings for different reasons: PAM50 under every augmentation setting, nanobody because its searches did not finish.

> In practice the Gaussian process runs on humanness only, and because humanness is a classification task, the Gaussian process there uses an RBF kernel. ... PAM50 subtype is left out under every augmentation setting because of run time. In the search with augmentation features added beyond the budget, the Gaussian process took 44 of its 60 hours. Nanobody thermostability is left out under augmentation beyond the budget, where its searches did not finish within the time limit. Because a dataset that drops the Gaussian process under any setting receives it under none, nanobody thermostability is left out everywhere.

**Ch5c-7. Models and metrics**: The molecule counts that explain why the two counts fall on either side of the 5,000-molecule cap.

> A search inside one outer fold is given the training molecules of that fold, about 80\% of the dataset. ... LogD has 5,028 molecules and TATA promoter 5,274

**★ Ch5c-8. Models and metrics**: The reason for AUROC and the reason for MCC. RMSE and Spearman keep their reasons, and these two lost theirs.

> Binary classification uses AUROC, which does not depend on a decision threshold. The two nucleotide datasets are the exception and use MCC, which is high only when all four entries of the confusion matrix are good \citep{Chicco2020}.

**Ch5c-9. Comparators, Controls and Ablations**: The argument for why the budget does not set the parameter count, model by model.

> The DNN and the random forest have more fitted parameters than training molecules at every budget. The SVM is fitted with at most one coefficient per training molecule, so it never can, and the capacity of the boosted trees is set by the number and depth of their trees.

**★ Ch5c-10. Comparators, Controls and Ablations**: Why the molecule-scaled budgets are there: they put a budget where features equal molecules.

> which place a budget where the number of features equals the number of molecules

**Ch5c-11. Comparators, Controls and Ablations**: The seven pooling methods themselves (the reader can no longer count seven), the 20% trim, the two outlier rules, the top-3 mean, and the majority-vote option.

> Three are the mean, the median and a 20\% trimmed mean, and two are means after removing outliers found by interquartile range or by median absolute deviation. The last two are a mean weighted by the inner score of each member and the mean of the top 3 members. Classification ensembles pool predicted probabilities, and where the metric reads class labels, a majority vote is also offered.

**Ch5c-12. Comparators, Controls and Ablations**: The pointer saying where a result in which the ensemble matches the hybrid at lower cost is discussed.

> Because an ensemble is the obvious simpler alternative, a result in which it matches the hybrid at lower cost belongs in Chapter~\ref{chap:discussion}, ``Simple is often better''.

**Ch5c-13. Comparators, Controls and Ablations**: The settings of the comparators: the PCA component cap, the mRMR 2,000-feature pre-filter, the RFE forest, and the size of the scoring forest. The later fixes pass kept the list of methods but not these settings.

> PCA is then fitted on those molecules with $B$ components, capped at the total number of features and at one fewer than the number of training molecules. ... When the representations hold more than 2,000 features in total, mRMR first keeps the 2,000 with the highest mutual information with the label. Recursive feature elimination uses a random forest of 50 trees of depth at most 5. ... a random forest of 100 trees

**★ Ch5c-14. Comparators, Controls and Ablations**: Why the experiment matters: whether one cheap evaluator is enough or the search must be rerun per model.

> The answer determines whether one cheap evaluator is enough or whether the search must be rerun for each model.

**Ch5c-15. Comparators, Controls and Ablations**: The matched evaluator's settings (a table pointer), the exact tree pairings, and the DNN layer widths for the narrower and wider evaluators.

> and uses the lighter evaluator settings of Table~\ref{tab:add:model_settings}. The near-family setting scores the random forest with LightGBM and the reverse, and XGBoost with LightGBM. It also scores the SVM with a Gaussian process, and the DNN with a narrower DNN of 64 and 32 units. ... A fourth setting scores the DNN with a wider DNN of 512, 256 and 128 units.

**★ Ch5c-16. Comparators, Controls and Ablations**: Why stability is measured at all.

> Unstable feature sets would weaken any chemical reading of the blueprints.

**Ch5c-17. Comparators, Controls and Ablations**: The link to the Chapter 3 finding that the model, not the representation, explained most of the variance in the normalized AUC. It is the reason the model is fixed here.

> In Chapter~\ref{chap:robustness} most of the variance in \aucnorm was explained by the model rather than by the representation, whereas here the model is fixed, so only the representation changes.

**★ Ch5c-18. Comparators, Controls and Ablations**: What the outcome would mean for the argument.

> A concatenation that lands between its two parts supports the case for combining and filtering at once.

**★ Ch5c-19. Transfer, Mutational Scans and Separate Sites**: The Meta-QSAR result itself.

> It beat random forest on a fingerprint by up to 13\% on average

**Ch5c-20. Transfer, Mutational Scans and Separate Sites**: A second reported quantity: the two wild-type-difference representations are also scored alone, and their rank among all representations is reported.

> The two wild-type-difference representations are also scored as single representations, and each run reports their rank among all representations.

**Ch5c-21. Transfer, Mutational Scans and Separate Sites**: How closeness is computed: 8 nearest neighbors among a random 20% of the site's training molecules, per-site rescaling to 0 to 1, the (1 - distance)^2 similarity, and normalization over sites.

> Closeness is the mean fingerprint distance from the query molecule to its 8 nearest neighbors among a random 20\% of the site's training molecules. ... For each site, these distances are rescaled to the range 0 to 1 over the query molecules. Each is turned into a similarity as $(1 - \text{distance})^2$, and the similarities are normalized to sum to 1 over the sites.

**Ch5c-22. Transfer, Mutational Scans and Separate Sites**: Three things. The 80/20 scaffold split each site uses. The rule when every site scores 0. The link to the allocation method.

> choosing features on 80\% of them and scoring them on the other 20\%, split by scaffold group ... If every site scores 0, the budget is split equally. ... Splitting the budget between sites is also the same question that the allocation method answers between representations.

**Ch5c-23. Transfer, Mutational Scans and Separate Sites**: The identity key used for deduplication.

> Molecules held by more than one source are deduplicated on InChIKey, keeping the public copy.

**★ Ch5c-24. Transfer, Mutational Scans and Separate Sites**: One of the four cells of the two-by-two comparison: pooled molecules with selected features.

> Pooled molecules with selected features measure what the selection is worth on its own

**Ch5c-25. Transfer, Mutational Scans and Separate Sites**: Two things. Why the budget sweep does not fix the difference in source. The link to Chapter 6 on label noise with a known source.

> The budget sweep varies only the width of the selected set and cannot separate the source difference. ... Two curators disagreeing on one molecule constitute label noise with a known source (Chapter~\ref{chap:similarity}).

**★ Ch5c-26. Transfer, Mutational Scans and Separate Sites**: That the comparison is made at several noise levels.

> compared, at several noise levels,

**Ch5c-27. Transfer, Mutational Scans and Separate Sites**: How sub-sites are made in the antibody version.

> Sub-sites within a site are made by clustering language-model embeddings instead of fingerprints.

**★ Ch5c-28. Transfer, Mutational Scans and Separate Sites**: That held-out molecules are kept out of the student's training set, and that this is a compromise.

> play that role as a compromise, and held-out molecules are excluded from it.

**★ Ch5c-29. Discussion (overfitting through feature selection)**: The Cawley and Talbot claim, with its citation, that feature selection is model selection with a complexity cost. And the qualifier that search degrees of freedom remain even when each selection is unbiased.

> Cawley and Talbot \citep{Cawley2010} argued that feature selection is itself a form of model selection whose complexity cost must be accounted for. ... even when each individual selection is unbiased.

**★ Ch5c-30. Limitations**: The claim that tuning cannot fix the small-sample problem.

> and tuning cannot fix this.

**Ch5c-31. Limitations**: The mitigation that embeddings are computed once per dataset and cached, so the burden is in the initial integration.

> Embeddings need to be computed only once per dataset and are cached, but the initial integration burden is real and scales with the number of models in the pool.

**★ Ch5c-32. Limitations**: The concession that the budget plausibly guards against overshoot.

> The budget constraint plausibly guards against overshoot, but it cannot push past this noise floor.


## Chapter 6

**★ Ch6-3. Methods (paragraph on which labels the hybrid saw)**: Which experiment has the unfair advantage. The survey avoids it and the censoring experiment does not. The kept sentence now does not say which result is affected. The word "clean" before "unclipped" was also dropped, and that is what linked the censoring search to this caveat.

> Settings searched on clean labels give the hybrid information that a real user would not have, an advantage that the survey avoids and the clipped-label experiment does not.

**★ Ch6-4. Similarity in a Hybrid**: The statement that the two searches differ only in the score that ranks candidates, which is the design point of the comparison.

> and differ only in the score that ranks candidates.

**★ Ch6-5. Similarity in a Hybrid**: The specific finding from Chapter 4 that motivates the similarity hybrid. The pass replaced it with a vague "the problems of Chapter 4".

> Chapter~\ref{chap:flaws} showed that the representation determines which molecules count as neighbors.

**Ch6-6. Similarity in a Hybrid (distance paragraph)**: The condition in the placeholder: which experiment has to finish before the choice can be confirmed.

> \textbf{[RESULT NEEDED: confirm Manhattan distance and one-over-distance neighbor weighting once the design experiment crossing distance, neighbor weighting and number of neighbors has run in full]}

**★ Ch6-7. Similarity in a Hybrid (neighbor score paragraph)**: Two things. First, the rule for a neighbor at distance 0, which the equation leaves undefined. Second, the literature comparison with the modelability index of Golbraikh2014, with its citation: it uses one nearest neighbor and a hard class match, and ours keeps the graded fraction.

> A neighbor at distance 0 receives all of the weight. ... The modelability index of \citet{Golbraikh2014} (Chapter~\ref{chap:flaws}) is a related published measure, but it uses a single nearest neighbor and a hard class match, whereas our score retains the graded fraction of neighbors. The score thus measures the extent to which the similar property principle

**★ Ch6-8. Noise Conditions (censoring paragraph)**: The reason censoring is regression-only.

> because a class label has no value to clip.

**★ Ch6-9. Noise Conditions (Gaussian paragraph)**: The framing that Gaussian noise is the standard assumption in noise-robustness studies. This is why the survey uses it.

> and is also the standard assumption in studies of noise robustness

**★ Ch6-10. Noise Magnitude Estimation**: Two things. Her framing that estimating the size of the noise is the central part of the noise mitigation work. The contrast with removing labels outright, which is the point of the chapter's move from removal to weighting and smoothing.

> The central part of the noise mitigation work is the estimation of noise magnitude, which allows noisy labels to be handled in proportion to their error instead of being removed outright.

**★ Ch6-11. Evaluation (recovery rate)**: The reason recovery is left undefined below the 2% gap.

> the noise did too little damage to be recovered, and the recovery rate is left undefined.


## Chapter 7

**★ Ch7-1. Summary**: The whole closing Summary paragraph went. It held the only statement in Chapter 7 of what the thesis contributes, and the only pointer from Chapter 7 to the Summary of Research Contributions.

> Chapter~\ref{chap:robustness} tells a reader which model to choose when labels are noisy, and Chapters~\ref{chap:flaws} to~\ref{chap:similarity} tell them which features to give that model. The contributions of this thesis are the NoiseInject package, the adaptive hybrid representation and its blueprints, and evidence that distance-based noise handling on standard representations usually made models worse. The resulting papers and software are listed in the Summary of Research Contributions.

**Ch7-2. Dataset Dependence of the Best Method**: A Chapter 3 finding: model ranks also moved with the amount of noise. On ECFP4 under Gaussian noise, random forest went from mid-ranking on clean labels to the top at higher noise.

> Model ranks also moved with the amount of noise. On ECFP4 under Gaussian noise, for instance, random forest went from mid-ranking on clean labels to the top at higher noise levels.

**★ Ch7-3. Simple Models and Small Feature Sets**: Two pointers to named sections of Chapter 6 went. One pointed to where the twelve-method comparison under Gaussian noise is shown, the other to where cleaning of real labels is shown. Only a bare chapter reference is left.

> (``Gaussian noise: methods for learning with noisy labels'') ... (``Real labels: cleaning without injected noise'')

**★ Ch7-4. Simple Models and Small Feature Sets**: A step in her argument about why generality matters: the tasks span binding affinity, ADME, quantum-chemical properties and antibodies, and that range is why learned models are used at all.

> The tasks in this thesis span binding affinity, ADME, quantum-chemical properties and antibodies, and this range is the reason learned models are used at all.

**★ Ch7-5. Limitations**: The premise was dropped: every experimental label carries an unknown amount of noise. The kept sentence depends on it.

> Every experimental label carries an unknown amount of noise, so there

**Ch7-6. Future Directions**: A third missing importance method went: importance measured by dropping one feature and refitting. The code comment above this paragraph records that it is not implemented.

> importance measured by dropping one feature and refitting is not implemented

**Ch7-7. Future Directions**: The test design for the weak-embedding question went. The hybrid is scored against the embedding alone with its best downstream model, and against a hybrid where a random matrix of the same width replaces the embedding.

> This would be tested by scoring the hybrid that includes the embedding against the embedding alone with its best downstream model. The same hybrid would also be scored against a hybrid in which a random matrix of the same width replaces the embedding.


## Summary of Research Contributions and the former Appendices A and B (the appendices now live in additional_files_body.tex, headings one level lower)

**★ AF-1. Datasets and curation / Small Molecules (hERG K_i)**: The caveat that repeats are merged on ChEMBL's own SMILES before standardization. So two ChEMBL entries that standardize to the same molecule are not median-merged; the InChIKey step keeps only the first of them.

> Because repeats are merged on ChEMBL's own SMILES before standardization, two ChEMBL entries that standardize to the same molecule are not merged by median; the InChIKey step keeps the first of them.

**★ AF-2. Datasets and curation / Small Molecules (hERG K_i)**: The date the experiments' file was pulled (1 February 2026). That the ChEMBL release was not recorded at the time. That release 37 was a later re-run which gave a byte-identical file.

> The file read by the experiments was pulled on 1 February 2026; the ChEMBL release was not recorded at the time. Re-running the same query in September 2026 against ChEMBL release 37 gives a byte-identical file of 1,415 compounds.

**★ AF-3. Datasets and curation / Small Molecules (QM9)**: The seed (42). The qualifier that this is a random draw of molecules, not a random split.

> From it, 10,000 molecules were drawn once at random with seed 42, so the subsample is a random draw of molecules, not a random split.

**★ AF-4. Datasets and curation / Small Molecules (hERG IC50)**: The number of records from each assay type that the unfiltered set mixes. Binding 10,219, toxicity 1,543, functional 217, ADME 31, out of 12,010 records.

> Of the 12,010 records, 10,219 come from binding assays, 1,543 from toxicity assays, 217 from functional assays and 31 from ADME assays, as ChEMBL classifies them.

**★ AF-5. Datasets and curation / Antibodies**: The commits that identify the exact files used (FLAb c41e56e and 005e6c1, BioPhi c0797f2), their dates, and the check that the local copies match those commits.

> Each file has been changed only once. The Koenig and nanobody files were changed in FLAb commits \texttt{c41e56e} and \texttt{005e6c1}, both on 6 August 2025, and the two humanness files in BioPhi commit \texttt{c0797f2} on 24 August 2021. The local copies of the Koenig and humanness files are identical to those commits.

**★ AF-6. Datasets and curation / Antibodies (Humanness)**: One sequence carries two different labels across three antibodies. All three sit in one identity cluster, so the conflict never crosses a split.

> One sequence is shared by three antibodies with different labels, two humanized and one human. All three sit in one identity cluster, so they never fall on both sides of a split.

**★ AF-7. Datasets and curation / Antibodies (Trastuzumab--HER2 binding)**: The seed of the 5,000-variant draw.

> From these, 5,000 variants were drawn once with seed 42.

**★ AF-8. Datasets and curation / Nucleotide Sequences**: The qualifier that the download is not pinned to a revision; 851f994 was the latest revision when the thesis was written.

> The download is not pinned to a revision; the files used are those of revision \texttt{851f994}, dated 30 June 2025, the latest at the time of writing.

**★ AF-9. Datasets and curation / Nucleotide Sequences (H3K4me3)**: The dataset card lists 30,000 training sequences, which disagrees with the files. The count given comes from the files.

> The dataset card lists 30,000 training sequences; the count here is that of the files.

**★ AF-10. Datasets and curation / Cancer Genomics (TCGA)**: Missing expression values are set to 0.

> Missing expression values are set to 0.

**★ AF-11. Datasets and curation / Cancer Genomics (TCGA)**: The second half of the argument against batch correction: a correction fitted on all sites would already have seen the test samples.

> and a correction fitted on all sites would already have seen the test samples.


## Left out of this list because you approved those cuts

- Ch1-5: you approved cutting this sentence (last sentence of the "We hypothesized" paragraph)
- Ch4-5: you approved cutting the twice-the-median-absolute-deviation sentence
- Ch6-1: you approved cutting the comparison with Chapter 5's eight budgets
- Ch6-2: you approved cutting the fallback sentence