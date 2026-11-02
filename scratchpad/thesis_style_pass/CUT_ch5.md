# Chapter 5 (chap:method) cutting pass, 2026-10-10

Word counts: prose words in Chapter 5, lines starting with % left out, measured with python3 -I. Table bodies and the blueprint listing count too, because they are not comment lines.

- Before, in the backup thesis.tex.before_style_apply_2026-10-10: 23,207 words
- Before this cut, in thesis.tex: 24,843 words
- After this cut: 22,411 words

The target was 22,047 words, 5% under the backup. The chapter ends 796 words (3.4%) under the backup and 2,432 words under where this pass started. It is 364 words short of the target. Most of what was added today is tables and paragraphs the author asked for: the tuning table, the short-names table, the significance paragraph, the 8 October corrections paragraph and the hours paragraph. Getting below 22,047 would mean cutting results, method details or limitations.

Every cut is in a paragraph marked "% TODO: manual review" or "% TODO: review, needs manual review". The one exception is an author paragraph that got a terminology fix with its own note. No number was changed. Some sentences that carried numbers were removed or shortened, keeping one number per point. No \label, \input, \includegraphics, table body or SOURCE comment was touched.

## What was cut, one line per paragraph

### Introduction
- One-hot caveat: cut "Such encodings are usually passed to a network...". The author's sentence below it now says "change in binding free energy ($\Delta\Delta G$)" at the first mention in the chapter, with an edit note.

### Methods
- Allocation, after the global pool: "The five methods were chosen in a preliminary comparison" cut. Its Additional file pointer moved into the preliminary design paragraph.
- Fixed-share allocation: cut two sentences that repeat the last column of Table tab:alloc_methods.
- Global pool: cut "The cost is one extra importance fit per configuration."
- Internal evaluator: cut the pointer sentence back to the allocation section.
- Search size: cut the 7 October representation-count sentence, which the footnote of Table tab:decided_grid already gives.
- Representation count paragraph: cut. It repeats the definitions in Sections sec:method:build and sec:method:searchspace.
- Paragraph on reusing importance scores (cost): cut.
- Preliminary design experiments: the hERG Ki rescaling example is shortened to one sentence without its four RMSE values. The one-line duplicate "Preliminary design experiments set the values..." is cut.
- Augmentation intro: one sentence shortened.
- Donors: cut the sentences listing donor types and where MACCS is a donor, which Table tab:dataset_reps already shows.
- Uncapped and capped modes: capped-mode mechanics shortened to one clause. Capped mode runs on no dataset.
- Augmentation conditions: cut "14 ran all three, while humanness and trastuzumab have no donors" (trastuzumab is a cut dataset). Also cut the "second use" and adoption sentences. The table pointer is merged in from its own one-line paragraph.
- hERG Ki cut walk-through: cut the sentences that restate Table tab:cut_walkthrough (71%, 224 configurations, 1.4%, 0.0051 to 0.0154). Kept the best configuration, its spread, and where the removed time came from.
- Adaptive pruning, which was not used: cut to two sentences. Removed the voting rules and the 79-to-44-hour figure.
- Evaluation protocol: cut the three-fold limit and its date, which the new author-requested paragraph states. Also cut "so each result states its number of folds".
- Comparator paragraph: cut the held-out-advantage sentence and merged the sign sentences. Cut the per-fold "two best" rule, which the Comparators subsection states. Its TODO comment is kept on its own line.
- "Chapter 7 reuses this protocol...": cut. Section sec:method:build already says held-out labels stay clean under label noise.
- Splitting rules: the random-splitting justification is replaced by a pointer to Section sec:evaluation (Chapter 2).
- Groups and consequences of the grouping: cut "results are read per dataset and never pooled", which is stated in the datasets intro. "Binding free energy set" is now "$\Delta\Delta G$ set".
- Datasets intro: cut "Folds keep each group whole...", which is stated in Splitting rules.
- Representations: cut "The first version of the method used three...". Cut the one-line pre-selection summary, which the author's paragraph below covers. Cut "We scored each representation on its own" and "This is one reason the search is run separately...". The augmentation sentences are merged into one. The one-line pointer to the representation tables is cut. "Binding free energy has eight" is now "The $\Delta\Delta G$ dataset has eight".
- Models: the Chapter 2 pointer sentence is cut and the pointer moved into the next sentence. Cut "Several models are used because...". Size limits of the GP and SVM now point to Table tab:search_tiers.
- Tuning test: cut the hERG Ki ECFP4 pair and the RF yield numbers, which are in Table tab:tuning_base_reps. Kept the LogD pair.
- Significance: cut "Each hybrid is paired by fold with the best single representation" and "on outer test folds".
- Comparators intro: cut the nested-CV sentence, which the Evaluation protocol states.
- Experiment paragraphs: cut the opening "This experiment asks whether..." sentences, which repeat the Question column of Table tab:experiment_register. This covers many or few, easy and hard halves, overfitting, feature reduction, evaluator, importance in training, tuning, new embedding and calibration.
- PCA method: cut the three sentences that repeat the main comparison.
- Ensembles method: cut "We also record the number of representations and models needed at prediction time". No result reports it.
- Feature reduction: cut "The comparison with PCA at the same width is the one described above".
- Evaluator method: cut "We report each setting against the matched evaluator, paired per outer fold".
- Importance in training and calibration: cut the "two best single representations and two best hybrids" restatements.
- Stability: cut "Unstable feature sets would weaken any chemical reading...".
- Transfer: cut "If settings transfer, the expensive search becomes a one-off cost" and the Meta-QSAR 13% clause.
- Site combination: cut "A site therefore counts most on chemistry it has seen".
- Three further site experiments: the not-reported flagged-molecule version is shortened to one sentence.
- Site limitations: the simulated-sites sentence now points to the chapter's limitations, which give it in full.

### Results
- Results intro: cut the three sentences that repeat the evaluation protocol.
- Small molecules against best single: cut the two sentences that repeat how the comparator is chosen.
- Hindsight and shuffling: merged the first three sentences. Cut "Fingerprints, descriptors... hold part of the signal" and "This is the premise of the thesis...", which the Conclusions state.
- Antibodies: cut the opening sentence and the two sentences listing the representations built for this thesis, which Table tab:reps_antibodies shows. "Binding free energy" is now "$\Delta\Delta G$".
- Antibodies where the hybrid adds nothing: cut "A few hundred antibodies did not support anything better" and the Ginkgo fold-size and Gaussian-process sentence. Both are stated in Methods and in the limitations.
- Angiopoietin-2: cut the Gaussian-process clause, which is in the limitations.
- Dataset properties: cut the per-model budget picks on LogD, Caco-2 and QM9, which the Dependence paragraph and the dots in the F2a figure show. Also cut the duplicate DNN-correction sentence and the closing restatement.
- Ensembles: cut the anti-VEGF single-representation reminder, the duplicate hERG blocking sentence and the cost sentence. The explanation is cut from five sentences to two.
- Many or few: merged the opening and cut "On some antibody datasets the split matters in both directions". Shortened the last sentence.
- Rare scaffolds: the description of the split is shortened, since Methods gives it.
- PCA: cut the LogD SVM pair, the Caco-2 DNN pair and the anti-VEGF SVM pair. "Binding free energy" is now "$\Delta\Delta G$".
- Overfitting: cut the opening definitions, which Methods gives, and "This is the same divide as...".
- Ambroise paragraph: the study is cut to one sentence. The Vabalas and Cawley sentence is cut; both are cited in Methods.
- One fold fewer: cut the closing "the hybrid is a method that needs data", which the limitations state.
- Sources of the gain: the signposting paragraph is cut. Its TODO line is kept.
- Selection at a fixed number of features: merged the opening. Cut the "2,048-bit fingerprint" sentence, "random features are further behind", "The hybrid ranks features within each representation first..." and the mRMR sentence (no mRMR result is reported).
- Dependence on the model: cut the representation-count sentences, which Methods and the reactions paragraph give.
- Evaluator: cut the width caveat, which Methods states.
- Importance in training: cut the list of conditions in the first sentence.
- New embedding: cut "Each embedding... in turn", the 2,000-molecule and random-control caveat (in Methods and the limitations) and "A practical consequence...".
- Calibration: merged the first three sentences. Cut "On small molecules the hybrid therefore narrows the intervals...". This also fixes the missing space in "error.On".
- Blueprints figure: the opening that restates the figure caption is cut. The Buchwald--Hartwig sentence is cut; the reactions results paragraph gives it.
- LogD blueprint: the logD background is shortened from three sentences to two.
- Stability: cut the index definition (given in Methods), "Each fold picks its own budget..." and "The low antibody values fit the rest of this chapter".
- Activity-cliff targets: the source sentence is cut (Methods gives it), as is the activity-cliff gloss (Chapter 2).
- Shared rows: cut the first sentence (Methods and Chapter 4) and "which are also blind to stereochemistry" (Methods).
- Transfer: cut "and we tested this on one pair of datasets only", which the limitations state.
- Sites: cut the restated design and merged the last two sentences. The flipped-labels opening is shortened.

### Conclusions
- First paragraph: cut the sentence that restates what the search searches.
- Simpler selection paragraph and antibody paragraph: "binding free energy" is now "$\Delta\Delta G$".
- Last paragraph: cut the sentence that restates Chapter 4.

### Terminology, in AI-marked text only
- "Scaffold group(s)" is now "scaffold families" or "grouped by scaffold family" in five places. The author's own "its scaffold group" in the inputs paragraph of Section sec:method:build is unchanged, because that paragraph is unmarked.
- The antibody dataset is "change in binding free energy ($\Delta\Delta G$)" at its first mention and "$\Delta\Delta G$" after that. Table cells and captions are unchanged.
