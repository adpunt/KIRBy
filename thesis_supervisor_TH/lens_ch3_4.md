# Supervisor-lens proposals: chap:robustness and chap:flaws (new items only)

Source read in full: Repo/thesis.tex lines 395-1260 (2026-10-10). Every paragraph in both
chapters is her text except one table row (line 1254), which no proposal touches. So every edit
below is minimal. For chap:flaws, nothing here repeats a comment in comments_ch4.md. Each item is a
place where one of his concerns comes back in text he did not annotate.

Line numbers are from my own read today.

---------------------------------------------------------------------------------------------------
## HIGH
---------------------------------------------------------------------------------------------------

### X-robustness-1
Location: line 451, Noise conditions and levels; Owner: hers
Quote: "On the three assay datasets, \aucnorm{} under censoring was computed for the same five pairings, ..."
His comment: AUC_norm appears here for the first time but is only defined later, in Evaluation. What is it at this point?
Lens item: 4
Proposed change:
old_string: `On the three assay datasets, \aucnorm{} under censoring was computed for the same five pairings`
new_string: `On the three assay datasets, robustness under censoring (\aucnorm, defined in the Evaluation subsection below) was computed for the same five pairings`
Priority: high

### X-robustness-2
Location: line 534, Variance decomposition; Owner: hers
Quote: "This pattern is observed across noise conditions and makes sense, as label noise does not impact the features."
His comment: True for fixed fingerprints and for frozen embeddings. A representation learned on the noisy labels would be impacted. Say which kind you have here.
Lens item: 9
Proposed change: all six representations are computed without labels. ChemBERTa and MHG-GNN are frozen pretrained models: Table tab:reps_smallmol, line 1735 onward, lists both as "frozen". Sort & Slice fits its vocabulary on the training molecules, not their labels (line 422).
old_string: `This pattern is observed across noise conditions and makes sense, as label noise does not impact the features.`
new_string: `This pattern is observed across noise conditions and makes sense, as label noise does not change the features. All six representations are computed without the labels, and the two embeddings are frozen rather than trained on the task.`
Priority: high

### X-robustness-3
Location: line 528 (Evaluation) against lines 860 and 862 (Aleatoric and epistemic attribution); Owner: hers
Quote (Methods): "We divided the increase of mean epistemic uncertainty between the highest level of noise and clean labels by the mean of the aleatoric component"
Quote (Results): "the epistemic component grows by at most 0.43 of the aleatoric growth"; "the epistemic growth is 0.61 to 1.14 times the aleatoric growth"
His comment: Is the ratio divided by the aleatoric value or by the aleatoric growth? Methods says one, Results the other.
Lens item: 17, 4
Proposed change: the next Methods sentence ("a ratio of 1 or above means the epistemic component grew as much or more") reads as if the denominator is the aleatoric growth. NEEDS HER: check the denominator in the analysis code. If it is the growth:
old_string: `We divided the increase of mean epistemic uncertainty between the highest level of noise and clean labels by the mean of the aleatoric component to determine how each responds to label noise.`
new_string: `We divided the increase in mean epistemic uncertainty, from clean labels to the highest noise level, by the increase in the mean aleatoric component over the same range.`
If it is the aleatoric mean, change "aleatoric growth" at lines 860 and 862 instead.
Priority: high

### X-robustness-4
Location: line 904, Conclusions; Owner: hers
Quote: "the BNNs with a variance head, the heteroscedastic GP and the heteroscedastic VBLL networks correctly attributed error from the injected noise as aleatoric, while QRF read that error as epistemic."
His comment: Results say the heteroscedastic GP fails on ECFP4 under grouped-shifted, and the VBLLs not on every representation. The conclusion is stronger than the results.
Lens item: 17
Proposed change: line 858 says the heteroscedastic VBLLs separate the components "not on every dataset and representation". Line 862 says the heteroscedastic GP fails on ECFP4 under grouped-shifted noise. Line 864 says grouped-shifted noise raises its epistemic share. Line 860 says the BNN ratio stays at or below 0.43 everywhere.
old_string: `Of the models whose aleatoric and epistemic components can be split per molecule, the BNNs with a variance head, the heteroscedastic GP and the heteroscedastic VBLL networks correctly attributed error from the injected noise as aleatoric, while QRF read that error as epistemic.`
new_string: `Of the models whose aleatoric and epistemic components can be split per molecule, the BNNs with a variance head attributed the injected noise to the aleatoric component wherever the ratio could be computed. The heteroscedastic GP and the heteroscedastic VBLL networks did so on most representations, but not all, and grouped-shifted noise raised the epistemic share of the GP. QRF read that error as epistemic.`
Priority: high

### X-robustness-5
Location: lines 448-451, Noise conditions and levels; Owner: hers
Quote: "We ran Gaussian, grouped-wider and grouped-shifted noise on every model and representation pairing. The remaining four conditions ran on selected pairs." [then a long list of which models and representations each subset used]
His comment: A table summarizing which condition was run on which models, representations and datasets would let the reader see where the comparisons come from.
Lens item: 15
Proposed change: a new table placed after line 451, built only from facts already in lines 448-451. No new numbers. Suggested rows:

| Noise conditions | Models | Representations | Datasets |
|---|---|---|---|
| Gaussian, grouped-wider, grouped-shifted | all 13 base and 6 variant models | all six | all four |
| Laplace, Student-t, outlier | RF, NGBoost, SVM, GP, GP (het.), BNN-α (var. head), VBLL-α (het.), VBLL-β (het.) | ECFP4, PDV, ChemBERTa | QM9; the assay datasets without VBLL-α (het.) |
| Censoring, robustness | RF and GP (het.) on ECFP4; NGBoost, GP and BNN-α (var. head) on PDV | as listed | all four; not BNN-α (var. head) on hERG K_i |
| Censoring, uncertainty (assay) | QRF, NGBoost, GP, GP (het.), BNN-α (var. head), BNN-β (var. head), VBLL-α | ECFP4, PDV, ChemBERTa | LogD, Caco-2, hERG K_i |

Caption: "Noise conditions run on each model, representation and dataset." NEEDS HER: approve. A flow figure of the QM9 pipeline (lines 453-464, Python to Rust and back) would also fit this lens. That figure NEEDS HER, if she wants it.
Priority: high

### X-robustness-6
Location: lines 754-773, Robustness on the three assay datasets; Owner: hers
Quote (example, line 758): "RF is the most consistent, among the 13 base models under Gaussian noise it ranks between first and seventh ... NGBoost ... ranks first to third on LogD, second to tenth on Caco-2 and sixth to ninth on hERG K_i. LightGBM falls furthest, ranking 10th to 13th ..."
His comment: Eight paragraphs of ranks. The reader is flooded with numbers. What is the take-home of this section?
Lens item: 2
Proposed change: add one closing paragraph after line 773, before Figure fig:curves_all. Every clause restates her lines 756, 761 and 769.
old_string: `Yet on Caco-2 and hERG $K_i$ it is among the flattest models.`
new_string: `Yet on Caco-2 and hERG $K_i$ it is among the flattest models.

In short, on the three assay datasets the forests were the most reliably robust. The boosted trees lost the most on Caco-2 and hERG $K_i$. The networks changed the most from one dataset to the next.`
Priority: high

### X-robustness-7
Location: line 438 (Models) and line 408 (Methods); Owner: hers
Quote: "Each network and its Bayesian counterpart were tuned to the same set of parameters so that NN-α against BNN-α and NN-β against BNN-β differ by the Bayesian layers alone."
His comment: Tuned on which data? If tuning used the test molecules, or the clean labels of the molecules later scored, the comparison is optimistic.
Lens item: 8
Proposed change: NEEDS HER. Add one sentence after line 408 that states three things: which models were tuned, on which molecules (training and validation only?), and whether the labels were clean or noisy. Additional file app:noise:sec:hyperparameters_tuned says "Only the Bayesian and variational networks are tuned", once per model per dataset, for all six representations. That table also lists NN-α rows on QM9. So that sentence in the Additional file disagrees with its own table, and it needs her check too.
Priority: high

### X-flaws-1
Location: line 964 (Constant and near-constant features) against Chapter robustness line 414; Owner: hers
Quote: "On hERG K_i (1,412 molecules) 1,815 of the 2,048 features are near-constant."
His comment: Chapter 3 had 1,415 hERG K_i molecules, 5,039 LogD and 2,161 Caco-2. Here it is 1,412, 5,028 and 2,158. Why do they differ?
Lens item: 17
Proposed change: the old build (p90) had this explanation, and the rewrite dropped it.
old_string: `On hERG $K_i$ (1,412 molecules) 1,815 of the 2,048 features are near-constant.`
new_string: `On hERG $K_i$ (1,412 molecules; the datasets of this chapter are curated as in Chapter~\ref{chap:method}, so their sizes differ slightly from Chapter~\ref{chap:robustness}) 1,815 of the 2,048 features are near-constant.`
NEEDS HER: the old build pointed to Additional file S.5.2. Add that file's current label if it still exists.
Priority: high

### X-flaws-2
Location: lines 962 and 983-985, Constant and near-constant features / Molecules with identical feature vectors; Owner: hers
Quote: "their base representations are those with counts in Additional file ..."; "for each dataset's base representations and for its augmentation features, the small feature sets added to the hybrid after it is built"
His comment: (as on old p104) We haven't explained hybrid representations yet. What are "base representations" and "augmentation features" at this point?
Lens item: 4, 11
Proposed change: the old build defined base representations at this spot ("the representations the hybrid of Chapter 5 selects features from"). The rewrite dropped that. Define both terms without the hybrid.
Edit 1:
old_string: `We counted constant and near-constant features from a set of representations (described in Table~\ref{tab:dataset_reps}) across`
new_string: `We counted constant and near-constant features for each dataset's base representations, the representations that Chapter~\ref{chap:method} later selects features from (Table~\ref{tab:dataset_reps}), across`
Edit 2:
old_string: `for its augmentation features, the small feature sets added to the hybrid after it is built (Section~\ref{sec:method:augmentation})`
new_string: `for its augmentation features, small extra feature sets such as the MACCS keys and 3D shape descriptors that Chapter~\ref{chap:method} can add after selection (Section~\ref{sec:method:augmentation})`
Priority: high

### X-flaws-3
Location: lines 981 and 985, Molecules with identical feature vectors; Owner: hers
Quote: "The tokenizer of the ChemBERTa-77M-MTR checkpoint ... gives 72 of the 1,415 hERG K_i molecules ... the same token sequence as another molecule." / "The 3D augmentation features and the learned embeddings ChemBERT, GROVER, MHG-GNN, MoLFormer and GraphMVP tie none"
His comment: ChemBERTa ties 72 molecules but ChemBERT ties none. Are these the same model? And are the 72 stereoisomers?
Lens item: 4, 6
Proposed change: Table tab:reps_smallmol (line 1735) lists ChemBERT as a separate 256-feature frozen SMILES model, and ChemBERTa-2 as 384 features.
old_string: `the learned embeddings ChemBERT, GROVER, MHG-GNN, MoLFormer and GraphMVP tie none`
new_string: `the learned embeddings ChemBERT (a 256-feature SMILES model, not ChemBERTa), GROVER, MHG-GNN, MoLFormer and GraphMVP tie none`
NEEDS HER: line 981 never says whether the 72 molecules with a shared ChemBERTa token sequence are all stereoisomers. Add that.
Priority: high

### X-flaws-4
Location: lines 1185 and 1190, A small sweep of published noise-handling methods; Owner: hers
Quote: "All 20 configurations with the lowest retention used a distance" / "Four more methods add a distance only in some configurations."
His comment: How many of the 221 configurations used a distance at all? If most did, the bottom 20 using one says little. For the four methods where the distance is optional, the clean test is the same method with and without the distance.
Lens item: 8
Proposed change: NEEDS HER. Two numbers are needed. (a) How many of the 221 configurations used a distance. (b) Retention with and without the distance for small-loss selection, confident learning, MentorNet and contrast-to-divide. The TODO at line 1189 says the raw sweep results are not on this machine. If they cannot be found, add one sentence saying the with-and-without comparison was not made.
Priority: high

### X-flaws-5
Location: lines 964, 1192 and 1233; Owner: hers
Quote: "The learned embeddings, which are taken from the last layers of a NN, have none except MHG-GNN" / "A pairwise distance metric sums over every feature, so when only a few features matter for the property, the rest dominate the sum."
His comment: (his general comment from p88) Distinguish task-agnostic representations from representations learned against the target. The embeddings here are pretrained elsewhere and frozen. The conclusion should say which kind it covers.
Lens item: 9
Proposed change: every embedding in Table tab:reps_smallmol is marked "frozen".
Edit 1 (line 964):
old_string: `The learned embeddings, which are taken from the last layers of a NN, have none except MHG-GNN`
new_string: `The learned embeddings, which are taken from the last layers of a frozen pretrained network that never saw these labels, have none except MHG-GNN`
Edit 2 (line 1233):
old_string: `A pairwise distance metric sums over every feature, so when only a few features matter for the property, the rest dominate the sum.`
new_string: `A pairwise distance metric sums over every feature, so when only a few features matter for the property, the rest dominate the sum. Every representation tested here is task-agnostic, so none was built to make the distance follow the property.`
Priority: high

---------------------------------------------------------------------------------------------------
## LOW
---------------------------------------------------------------------------------------------------

### X-robustness-8
Location: line 622, Noise conditions on the HOMO--LUMO gap; Owner: hers
Quote: "The size of the shift stays small, which is why these results coincide with \citet{Heid2023}"
His comment: Small compared to what? Give the size.
Lens item: 6
Proposed change: the numbers come from Table tab:assay_shape, in the Laplace, Student-t and Outlier columns.
old_string: `The size of the shift stays small, which is why these results coincide with \citet{Heid2023}`
new_string: `Apart from BNN-$\alpha$ (var.\ head) on Caco-2, which moves by $-0.110$ and $+0.092$, no shift exceeds 0.069 of \aucnorm. These shifts are small, which is why these results coincide with \citet{Heid2023}`
Priority: low

### X-robustness-9
Location: line 444, Noise conditions and levels; Owner: hers
Quote: "The seven noise conditions are defined in Section~\ref{sec:noise_injection}."
His comment: Easier for the reader to restate them in one line here than to send them back to Chapter 2.
Lens item: 5
Proposed change: the facts come from lines 189-235 and Table tab:noise_conditions.
old_string: `The seven noise conditions are defined in Section~\ref{sec:noise_injection}.`
new_string: `The seven noise conditions are defined in Section~\ref{sec:noise_injection}. Gaussian, Laplace and Student-$t$ noise differ only in the shape of the error distribution. Outlier noise triples the error on a random 10\% of molecules, and grouped-wider noise triples it on whole scaffold families. Grouped-shifted noise gives each scaffold family a shared offset, and censoring clips labels above a limit.`
Priority: low

### X-robustness-10
Location: line 712, start of Robustness on the three assay datasets; Owner: hers
Quote: "So far, we have only discussed results coming on HOMO--LUMO gap from QM9, ..."
His comment: A short recap of the QM9 findings before moving to the assay datasets.
Lens item: 3
Proposed change: the recap restates lines 534, 622 and 624.
old_string: `So far, we have only discussed results coming on HOMO--LUMO gap from QM9, a computed property whose labels carry no measurement error.`
new_string: `So far, we have only discussed results coming on HOMO--LUMO gap from QM9, a computed property whose labels carry no measurement error. There, model family set robustness more than representation did, systematic noise cost the most, and the shape of the noise distribution changed little.`
Priority: low

### X-robustness-11
Location: line 808, Probabilistic and deterministic counterparts; Owner: hers
Quote: "QRF had lower \aucnorm{} on QM9 under Gaussian and grouped-shifted noise on every representation"
His comment: RF and QRF have the same trees. Why is QRF less robust?
Lens item: 18
Proposed change: NEEDS HER. Add one sentence giving QRF's point prediction (the median of the leaf labels, or their mean?). If it is the median, that is the one mechanical difference from RF.
Priority: low

### X-robustness-12
Location: line 499 (Evaluation) and line 915 (Conclusions); Owner: hers
Quote: "While we tested other metrics of robustness, we found \aucnorm{} to be less directly correlated to clean-label $R^2$ predictions."
His comment: Which other metrics, and how correlated were they? The metric was chosen on the same results it then reports.
Lens item: 8
Proposed change: NEEDS HER. Name the other metrics and give each one's Spearman correlation with clean R^2, next to AUC_norm's -0.10 (line 584).
Priority: low

### X-robustness-13
Location: line 520, Evaluation; Owner: hers
Quote: "using Type I (sequential) sums of squares, entered in the order replicate, model family, model architecture within family, molecular representation, interaction."
His comment: With sequential sums of squares the shares can depend on the order. Is the design balanced after the excluded replicates?
Lens item: 8
Proposed change: NEEDS HER. If every model, representation and replicate cell is present, add: "Every combination is present, so the order does not change the shares." If not, give the shares under another order, or say how many cells are missing (Additional file app:noise:sec:excluded_configurations).
Priority: low

### X-robustness-14
Location: lines 810 and 902; Owner: hers
Quote: "produced a slightly larger \aucnorm, but only because they lowered predictive performance on clean labels" / "increased \aucnorm{} on the assay datasets only by lowering clean $R^2$"
His comment: "Only because" is a causal claim. Was this shown, or is it the likely reason?
Lens item: 19
Proposed change:
old_string: `produced a slightly larger \aucnorm, but only because they lowered predictive performance on clean labels.`
new_string: `produced a slightly larger \aucnorm, but they also lowered predictive performance on clean labels, which can explain the gain.`
Same hedge at line 902. old_string: `increased \aucnorm{} on the assay datasets only by lowering clean $R^2$` new_string: `increased \aucnorm{} on the assay datasets alongside a lower clean $R^2$`
Priority: low

### X-robustness-15
Location: line 713; Owner: hers
Quote: "so a noise level of 0.6 of that spread is about one unit of that estimated laboratory error."
His comment: 0.6 is not one of the levels tested. Which tested levels bracket it?
Lens item: 21
Proposed change:
old_string: `so a noise level of 0.6 of that spread is about one unit of that estimated laboratory error.`
new_string: `so a noise level of 0.6 of that spread, between the tested levels 0.5 and 0.75, is about one unit of that estimated laboratory error.`
Priority: low

### X-robustness-16
Location: line 909 or line 923, Conclusions; Owner: hers
Quote: "When training on noisy experimental data in the context of QSAR, it is important to keep in mind ..."
His comment: Link the conclusion to drug safety: hERG is a toxicity endpoint.
Lens item: 14
Proposed change: add after the first sentence of line 909. The facts come from lines 416 and 906.
old_string: `oftentimes with respect to the model.`
new_string: `oftentimes with respect to the model. This matters most in drug safety assessment, where toxicity endpoints such as hERG binding are pooled from several laboratories and their noise is unknown.`
Priority: low

### X-robustness-17
Location: Tables tab:censoring, tab:condition_cost, tab:assay_shape; Owner: hers
His comment: Bold the best results.
Lens item: 16
Proposed change: at most four bold cells per table. In tab:censoring, bold the highest censoring value (RF, ECFP4, 0.817). In tab:condition_cost, bold the smallest loss per representation column; that needs a choice, since six columns exceed four. NEEDS HER: which comparison to bold, and whether these tables come from a script. If they do, the script is the place to change.
Priority: low

### X-flaws-6
Location: lines 937 and 1080; Owner: hers
Quote: "... or by smoothing their labels, inherits this choice, as do the noise estimation and mitigation methods of this thesis." / "Every method in Section~\ref{sec:flaws:consequences} that takes a similarity as input inherits the dilution described here."
His comment: (as on p89) A method is not given the flaw. It is negatively impacted by it.
Lens item: 19
Proposed change:
old_string: `Every method in Section~\ref{sec:flaws:consequences} that takes a similarity as input inherits the dilution described here.`
new_string: `Every method in Section~\ref{sec:flaws:consequences} that takes a similarity as input is negatively impacted by the dilution described here.`
old_string: `or by smoothing their labels, inherits this choice,`
new_string: `or by smoothing their labels, depends on this choice and is negatively impacted by its flaws,`
Priority: low

### X-flaws-7
Location: lines 1142 and 1200; Owner: hers
Quote: "If this measure of similarity, or equivalently distance, is flawed" / "A noise detection method requires a distance that picks the right neighbors"
His comment: (as on p90) "... depends on the distance metric". Here the neighbors depend on the representation and the metric together.
Lens item: 19
Proposed change:
old_string: `A noise detection method requires a distance that picks the right neighbors,`
new_string: `A noise detection method requires a representation and a distance metric that together pick the right neighbors,`
Priority: low

### X-flaws-8
Location: lines 1063-1065, Activity cliffs; Owner: hers
Quote: "On hERG K_i the number of cliff molecules then ranges from 138 (atom pairs) to 267 (WL graph kernel). Of the pairs that ECFP4 calls similar, only 57% to 74% are also among another representation's closest pairs, and on QM9 only 1% to 24%"
His comment: (as on p99) What is the take-home here?
Lens item: 2
Proposed change: add after the paragraph (line 1065).
old_string: `and on QM9 only 1\% to 24\% (Section~\ref{app:add:flaws} of the Additional files).`
new_string: `and on QM9 only 1\% to 24\% (Section~\ref{app:add:flaws} of the Additional files). Which molecules are cliffs therefore depends on the representation as well as on the labels.`
Priority: low

### X-flaws-9
Location: line 1078, Agreement with shared structure; Owner: hers
Quote: "For 500 random pairs per dataset we found the maximum common substructure (MCS) with RDKit ..."
His comment: Why MCS? One opening line saying what it is for.
Lens item: 12
Proposed change:
old_string: `For 500 random pairs per dataset we found the maximum common substructure (MCS)`
new_string: `The maximum common substructure (MCS) of two molecules does not depend on any fingerprint, so it gives a structural reference to check ECFP4 against. For 500 random pairs per dataset we found the MCS`
Priority: low

### X-flaws-10
Location: line 931, Introduction; Owner: hers
Quote: "Since common fingerprints do not encode stereochemistry \citep{orsi2024map4c}, they are often unable to differentiate stereoisomers"
His comment: (as on p93) The 2D graph does express stereochemistry. ECFP can include it. It is off by default.
Lens item: 7
Proposed change:
old_string: `Since common fingerprints do not encode stereochemistry \citep{orsi2024map4c},`
new_string: `Since common fingerprints do not encode stereochemistry in their default settings \citep{orsi2024map4c},`
Priority: low

### X-flaws-11
Location: line 1187; Owner: hers
Quote: "Under these methods, the choice of representation explained even less variance than in Chapter~\ref{chap:robustness}."
His comment: How much less? "Even less" without a number leaves the reader guessing.
Lens item: 6
Proposed change: NEEDS HER. Give the share of variance explained by representation in the sweep, next to Chapter 3's 8.1% (line 1149).
Priority: low

### X-flaws-12
Location: line 1060, Flaws in similarity; Owner: hers
Quote: "Binary representations were compared by Tanimoto distance, and all others by Manhattan distance on standardized features"
His comment: The ECFP4 columns and the descriptor columns in Table tab:flaws:distance_summary use different metrics. Are they comparable?
Lens item: 10
Proposed change: add one sentence.
old_string: `on standardized features (Section~\ref{sec:flaws:methods}), with constant features removed.`
new_string: `on standardized features (Section~\ref{sec:flaws:methods}), with constant features removed. Both measures are Spearman correlations, so they do not depend on the scale of the distance, but a difference between columns mixes representation and distance metric.`
Priority: low

### X-flaws-13
Location: line 1231, Conclusions; Owner: hers
Quote: "ECFP4 gives 56 of the 1,412 hERG K_i molecules a vector shared with another molecule, all of them stereoisomers."
His comment: Link to drug safety. On a toxicity endpoint this has a concrete cost.
Lens item: 14
Proposed change: the facts come from line 993 and Figure fig:flaws:stereo_pair.
old_string: `ECFP4 gives 56 of the 1,412 hERG $K_i$ molecules a vector shared with another molecule, all of them stereoisomers.`
new_string: `ECFP4 gives 56 of the 1,412 hERG $K_i$ molecules a vector shared with another molecule, all of them stereoisomers. For a toxicity endpoint, this means an ECFP4 model gives the same predicted hERG binding to two stereoisomers measured 1.14 p$K_i$ apart (Figure~\ref{fig:flaws:stereo_pair}).`
Priority: low
