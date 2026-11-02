# Review: Chapter 8 (chap:discussion), after the style pass

Compared with thesis.tex.before_style_apply_2026-10-10 by diff (chapter at backup lines 4703-4883; current chapter starts at the `\chapter{Discussion and Conclusions}` line, about 5077 when read) and read in full as it stands on 2026-10-10. The chapter was rebuilt in this pass: every prose paragraph in it carries "% TODO: manual review", so all of it is AI text and may be rewritten. her_text_check.py has no hit inside the chapter. Every OLD block below was checked with python3 -I str.count to occur exactly once in thesis.tex. Every number in the NEW text is already in thesis.tex: 31.1% (Table tab:variance), 0.4930/0.4302 (tab_hybrid_vs_best_single), 0.0729/0.2625 (tab_distance_tracks_activity), 0.6354/0.4795/0.4314 (Chapter 7 comments and Figure fig:noise:censoring_ladder), 2,000 molecules of LogD and hERG blocking and "four Gaussian-process datasets" (Chapter 5, sec:method results text), twelve QM9 properties (current Ch8).

Checks that passed: every `\ref` in the chapter resolves (chap:*, sec:method:results_single, sec:noise:res:censoring, sec:noiseinject_pkg, plus the new ones planned below: tab:hybrid_vs_best_single, tab:distance_tracks_activity, fig:noise:censoring_ladder, all defined). Every old section label is kept. No "Results:" prefix, no claim titles, no Discussion-in-a-research-chapter issue. TCGA no longer named (gap 46 done). The Chapter 1 research questions are answered one by one (gap 20 done). "no single method can work" is gone (gap 37 done). Chemistry guide item 6 (activity cliffs called noise) is gone. Chapter length 1,737 prose words, against 1,634 (Buttenschoen), 1,377 (Dablander, plus a separate future-directions chapter) and 4,362 (Vost): in range.

Failures found, worst first:
1. The argument Chapter 1 and the abstract promise ("the noise in a dataset should be modeled before a method is chosen ... simple methods are often as good") is one hedged sentence ("it therefore helps to ask") in the eleventh paragraph of "Summary of findings". It has no section of its own. V11, V12.
2. The author's 2026-10-09 decision on antibodies (recorded in the chapter's own DECISIONS comment) was dropped in the rebuild: the chapter says only "On antibodies the gains depended on the dataset." Chapter 5 now carries her reading; Chapter 8 does not echo it. V7.
3. Factual and dangling text: "Bayesian neural networks ... never predicted better" contradicts Chapter 3 (V3); "That experiment has no result yet" has no antecedent in the printed text, and the 8 October caveat is printed twice in one paragraph, with "the correction" never said (V13); the DNA promoter result is reported nowhere else in the thesis, and Chapter 5 says nucleotides were cut (V14); "Chapter 7 found that they can, under the right conditions, and that the hybrid keeps its lead" overstates the antibody case (V9); "and by changing the features they run on" in the concluding remarks claims more than Chapter 7 found (V16).

## Fixes

### V1 — Chapter title in title case
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- title was: Discussion and Conclusions -- edited your text: sentence case, as in the reference theses and the other chapter titles (style guide 22)
- why: Every other chapter title is now in sentence case ("Detecting and handling label noise"). Her title, so it gets a "title was" note.
OLD:
```latex
\chapter{Discussion and Conclusions}\label{chap:discussion}
```
NEW:
```latex
% TODO: manual review -- title was: Discussion and Conclusions -- edited your text: sentence case, as in the reference theses and the other chapter titles (style guide 22)
\chapter{Discussion and conclusions}\label{chap:discussion}
```

### V2 — Opening: no link to Chapter 7, vague second sentence, does not announce the argument
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: Check 6 asks the chapter to open by linking to the previous chapter; it names only the thesis aim. "The published noise methods did poorly" introduces with "The" a study the reader has not been told about in this chapter; the thesis's agreed phrase is "an exploratory study carried out earlier in this project". "led us to the representation itself" is performative. The opening should also say the chapter argues for a way to choose a method, as Chapter 1 promises.
OLD:
```latex
This thesis set out to handle label noise in molecular data. The published noise methods did poorly, and asking why led us to the representation itself. This chapter answers the research questions of Chapter~\ref{chap:intro} in turn and lists the contributions of the thesis. It closes with the limitations, future work and concluding remarks.
```
NEW:
```latex
This thesis set out to handle label noise in molecular data. In an exploratory study carried out earlier in this project, published methods for learning with noisy labels did poorly on molecular data. We traced the failure to the representation (Chapter~\ref{chap:flaws}), and built the adaptive hybrid representation to address it. Chapter~\ref{chap:noise} then returned to label noise with the hybrid and the single representations side by side. This chapter answers the research questions of Chapter~\ref{chap:intro} in turn and lists the contributions of the thesis. It then sets out how we would choose a noise-handling method, and closes with the limitations, future work and concluding remarks.
```

### V3 — Bayesian networks: "never predicted better" is not what Chapter 3 found
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: Chapter 3 (subsection "Probabilistic and deterministic counterparts") says the Bayesian transformations "produced a slightly larger \aucnorm, but only because they lowered predictive performance on clean labels", on the assay datasets, and made no change on QM9. "Raised \aucnorm{} but never predicted better" drops the cause and the dataset scope.
OLD:
```latex
Bayesian neural networks raised \aucnorm{} but never predicted better than their deterministic counterparts.
```
NEW:
```latex
On the assay datasets, Bayesian neural networks kept slightly more \aucnorm{} than their deterministic counterparts, but only because they predicted worse on clean labels.
```

### V4 — Question 1 answer: "mattered more" with no quantity
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: The paragraph gives the shares for robustness but none for predictive performance, so "mattered more" has nothing to compare. 31.1% is in Table tab:variance (row None, Clean R^2).
OLD:
```latex
The representation mattered more for predictive performance. Its pairing
```
NEW:
```latex
The representation mattered more for predictive performance, where it explained 31.1\% of the variance in clean $R^2$. Its pairing
```

### V5 — Question 3 answer: conclusion broader than the result
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: Chapter 3 found that systematic injected noise (grouped-shifted, censoring) did more damage than Gaussian noise. "Label noise in molecular data therefore cannot be treated as Gaussian" is a claim about all real noise. Chapter 3's own wording is narrower: "we do not find that is the case for systematic label noise". The abstract has the same sentence; see "For the author".
OLD:
```latex
Label noise in molecular data therefore cannot be treated as Gaussian.
```
NEW:
```latex
A study that injects only Gaussian noise therefore leaves out the kinds of noise that did the most damage.
```

### V6 — Question 4 answer: the censoring result is half told
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: "Outside censoring, no model's uncertainty ... identified the corrupted labels" implies something did under censoring, and the next sentence does not say what. Chapter 3's Conclusions: "dividing a molecule's error by its uncertainty picked out the clipped labels better than the error alone."
OLD:
```latex
Under censoring the models were most confident about the clipped labels.

```
NEW:
```latex
Under censoring the models were most confident about the clipped labels. Dividing each molecule's error by its uncertainty picked them out better than the error alone.

```

### V7 — Chapter 4 paragraph: "the exploratory study", "little ... varies", one of the two problems missing
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: "configurations that used a distance" is vaguer than the thesis's "compared molecules by a distance". "little of a standard representation varies" is not what the number says (most features take one value in almost every molecule). Chapter 1 and Chapter 4 name two problems; this paragraph gives one and leaves out identical feature vectors. No new number is added.
OLD:
```latex
Chapter~\ref{chap:flaws} looked at why the configurations that used a distance did worst in the exploratory study of published noise-handling methods. On a single dataset, little of a standard representation varies.
```
NEW:
```latex
Chapter~\ref{chap:flaws} looked at why, in the exploratory study, the configurations that compared molecules by a distance did worst. On a single dataset, most features of a standard representation take the same value in almost every molecule.
```

### V8 — Chapter 4 paragraph: second problem
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: as V7.
OLD:
```latex
A distance sums over every feature, so the features that carry nothing dominate it.
```
NEW:
```latex
Some representations also give different molecules the same feature vector. A distance sums over every feature, so the features that carry nothing dominate it.
```

### V9 — Hybrid question 1: the author's antibody reading is missing
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: The chapter's DECISIONS comment (2026-10-09) records her reading: a method built for small molecules carried over to antibodies because the representations were what held the methods back, with the antibody representations built for this thesis named as hers. The rebuild cut it to "On antibodies the gains depended on the dataset." Chapter 5's Conclusions now state it (three datasets beat every single representation; on two, the best single representation is one built here); Chapter 8 should echo Chapter 5 without new numbers. Datasets are not named, to avoid repeating the "binding free energy" name (chemistry guide item 8). Also adds the table that holds the LogD numbers, since Chapter 5 now prints them only there.
OLD:
```latex
On LogD with the SVM, for example, it went from 0.4930 to 0.4302 RMSE. On antibodies the gains depended on the dataset.
```
NEW:
```latex
On LogD with the SVM, for example, it went from 0.4930 to 0.4302 RMSE (Table~\ref{tab:hybrid_vs_best_single}). On antibodies the gains depended on the dataset. On three antibody datasets the hybrid beat every single representation, and on two of them the best single representation was one built for this thesis. Those representations read a language model's embedding over the regions that vary, or describe each mutation, so that the few features that matter are not buried among many that do not. On the smaller antibody datasets an ensemble of one model per representation often did better.
```

### V10 — Hybrid question 2: no pointer to the table; "Prediction from neighbors" vague
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: Chapter 6 now prints 0.0729 and 0.2625 only in Table tab:distance_tracks_activity, so the reader needs the pointer. "Prediction from neighbors" is the thesis's phrase for predicting each label from its five nearest neighbors; say it once here.
OLD:
```latex
went from 0.0729 to 0.2625. On the antibody datasets and the reactions, distance also tracked the label more closely. Prediction from neighbors, however, was worse in the hybrid on several of them.
```
NEW:
```latex
went from 0.0729 to 0.2625 (Table~\ref{tab:distance_tracks_activity}). On the antibody datasets and the reactions, distance also tracked the label more closely. Predicting each label from its five nearest neighbors, however, was worse in the hybrid on several of them.
```

### V11 — Hybrid question 3: overstated and vague
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: "found that they can, under the right conditions, and that the hybrid keeps its lead" says the hybrid kept its lead without the small-molecule limit (gap item 36), then repeats the point with the limit two sentences later. "under the right conditions" and "on several datasets" are vague; Chapter 7's Conclusions name the datasets. "the data flagged the wrong labels" is replaced by Chapter 1's wording ("a known flag, such as an assay limit or a scaffold family"). The QM9 control sentence is from Chapter 7's Conclusions. Chapter 7 prints the LogD numbers only in Figure fig:noise:censoring_ladder.
OLD:
```latex
Chapter~\ref{chap:noise} found that they can, under the right conditions, and that the hybrid keeps its lead. Corrections worked best when the data flagged the wrong labels. With a quarter of LogD's training labels clipped, censored EM took the network on its best single representation from 0.6354 to 0.4795 RMSE, against 0.4314 before clipping. With a detector, cleaning helped under label flips and outliers, and beat random and shuffled controls on several datasets. On small molecules the hybrid stayed ahead of the single representation through every treatment.
```
NEW:
```latex
Chapter~\ref{chap:noise} found that they can in some settings. Corrections worked best when a known flag, such as an assay limit or a scaffold family, said which labels were wrong. With a quarter of LogD's training labels clipped, censored EM took the network on its best single representation from 0.6354 to 0.4795 RMSE, against 0.4314 before clipping (Figure~\ref{fig:noise:censoring_ladder}). With a detector, cleaning helped under label flips and outliers. It beat random deletion or shuffled detector scores on hERG, hERG $K_i$, Buchwald--Hartwig and three antibody datasets. On QM9 with no noise added, deletion never helped. On small molecules the hybrid stayed ahead of the single representation through every treatment.
```

### V12 — Hybrid question 4: "is what makes the hybrid work"; detection sentence unclear
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: "is what makes the hybrid work" is an appraisal with no result behind it in this paragraph. Chapter 5's Conclusions give the result: at the same number of features the hybrid beat recursive elimination and random features on all but one small-molecule dataset with a result. "importance-weighted features of a single representation did better than the hybrid's" does not say at what; Chapter 7 says it was the space in which neighbor detectors found their neighbors, and names the datasets.
OLD:
```latex
Ranking features by importance within each representation is what makes the hybrid work (Chapter~\ref{chap:method}). Using the scores again in training added little on small molecules. For noise detection alone, importance-weighted features of a single representation did better than the hybrid's (Chapter~\ref{chap:noise}).
```
NEW:
```latex
In construction, the hybrid ranks features by importance within each representation. At the same number of features it beat recursive feature elimination and random features on all but one of the small-molecule datasets with a result (Chapter~\ref{chap:method}). Using the importance scores again in training added little on small molecules. For noise detection, neighbors found in importance-weighted features of a single representation did best on Buchwald--Hartwig, Caco-2, hERG $K_i$, LogD and SMAC colloidal stability, taking the best setting of each (Chapter~\ref{chap:noise}). Averaged over the settings, neighbors found in the hybrid came out ahead more often.
```

### V13 — The chapter's argument has no section; move three labels to it
- file: thesis.tex
- kind: ai-text (structure)
- todo_line: existing "% TODO: manual review" kept on the Summary heading's first paragraph; new heading gets "% TODO: manual review -- new section: holds the chapter's argument on choosing a noise method, which Chapter 1 promises; labels sec:discussion:noise, sec:discussion:conditional and sec:discussion:simple moved here from Summary of findings"
- why: Failure 1 above. The labels move to the section that now holds their content (old sections "Handling noise realistically", "No single method fits every dataset", "Simple is often better"). sec:discussion:antibodies stays on Summary, where V9 puts the antibody content. Step 1 of 2: the heading.
OLD:
```latex
\section{Summary of findings}\label{sec:discussion:summary}\label{sec:discussion:noise}\label{sec:discussion:antibodies}\label{sec:discussion:conditional}\label{sec:discussion:simple}
```
NEW:
```latex
\section{Summary of findings}\label{sec:discussion:summary}\label{sec:discussion:antibodies}
```

### V14 — The argument paragraph: new section, split, recommendation stated plainly
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept above the first paragraph; "% TODO: manual review" added above the second
- why: Failure 1 above. Step 2 of 2. The paragraph mixes the evidence (which method worked where) with the recommendation, and states the recommendation as "it therefore helps to ask", which is weaker than Chapter 1 ("should be modeled before a method is chosen") and the abstract ("We recommend choosing a noise method by how the labels were made"). Split into the evidence and the recommendation. "one constant lift for every clipped label" was given with no dataset; Chapter 7 says it beat censored EM on hERG $K_i$ only (censored EM was best on LogD and QM9). The sentence on pooled data and reporting limits is from the backup's Chapter 8 ("Data pooled from several sources or laboratories, assays with a reporting limit ... all carry such a flag"). No new numbers.
OLD:
```latex
Label noise was handled, but no single method handled it everywhere. Which method worked depended on the prediction task, the data, the model and the type of noise. In Chapter~\ref{chap:robustness}, NGBoost was the most robust model under Gaussian noise on the HOMO--LUMO gap, but not on Caco-2 or hERG $K_i$. In Chapter~\ref{chap:noise}, the forest residual detector fell to chance under censoring, and acting on the known flag did best. On Caco-2's labels as delivered, subtracting scaffold-family offsets made the model worse. Deleting the molecules the network fits worst almost never helped on small molecules, but helped on the reactions. The ranking of noise methods also changed between the hybrid and the single representations. Before choosing a method, it therefore helps to ask where the labels came from and what noise they are likely to carry. Simple methods were often enough, such as random forests under label noise and one constant lift for every clipped label.
```
NEW:
```latex
Label noise was handled, but no single method handled it on every dataset. Which method worked depended on the prediction task, the data, the model and the type of noise. In Chapter~\ref{chap:robustness}, NGBoost was the most robust model under Gaussian noise on the HOMO--LUMO gap, but not on Caco-2 or hERG $K_i$. In Chapter~\ref{chap:noise}, the forest residual detector fell to chance under censoring, and acting on the known flag did best. On Caco-2's labels as delivered, subtracting scaffold-family offsets made the model worse. Deleting the molecules the network fits worst almost never helped on small molecules, but helped on the reactions. The ranking of noise-handling methods also changed between the hybrid and the single representations.

% TODO: manual review
We therefore recommend modeling the noise in a dataset before choosing a method. That means asking where the labels came from and what noise they are likely to carry. Data pooled from several laboratories can carry an offset shared within each source, and an assay with a reporting limit gives clipped labels. Both come with a flag that a correction can use. A method that assumes noise that is not there does harm, as the scaffold-family offsets did on Caco-2. Simple methods were often as good as complex ones. Random forests were the most consistent model under label noise, and on hERG $K_i$ one constant lift for every clipped label beat censored EM.
```
Then insert the heading directly above the "% TODO: manual review" line that precedes "% Links to Chapter 3's conclusion" (the comment lines stay with the paragraph). Insert:
OLD:
```latex
% TODO: manual review
% Links to Chapter 3's conclusion (grouped-shifted and censoring hardest)
```
NEW:
```latex
% TODO: manual review -- new section: holds the chapter's argument on choosing a noise method, which Chapter 1 promises; labels sec:discussion:noise, sec:discussion:conditional and sec:discussion:simple moved here from Summary of findings
\section{Choosing a noise-handling method}\label{sec:discussion:noise}\label{sec:discussion:conditional}\label{sec:discussion:simple}

% TODO: manual review
% Links to Chapter 3's conclusion (grouped-shifted and censoring hardest)
```

### V15 — Limitations: dangling "That experiment", caveat printed twice, "the correction" undefined, inline TODO
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept; the inline "% TODO: manual review" at the end of the prose line moves to its own line above the second paragraph (not deleted)
- why: "That experiment has no result yet" refers to nothing in the printed text; the experiment is named only in a comment. "the correction of 8 October 2026" never says what was corrected; Chapter 5 says the DNN now clips each prediction to the range of its training labels, and the Gaussian process's starting length scale was changed. "the Gaussian-process results will change when that model is replaced" does not match Chapter 5's description (the hybrids of the four Gaussian-process datasets were chosen before the change). The 8 October caveat is printed twice in one paragraph. "moved the picked settings" is loose. Split into two paragraphs. The SOURCE comment is kept unchanged.
OLD:
```latex
Most DNN results were computed before the correction of 8 October 2026, and the Gaussian-process results will change when that model is replaced. % TODO: manual review
```
NEW:
```latex
Most DNN results were computed before 8 October 2026, when the DNN began clipping each prediction to the range of its training labels. The hybrids of the four Gaussian-process datasets were chosen before that model's starting length scale was changed.

% TODO: manual review
```
and
OLD:
```latex
That experiment has no result yet, so no dataset property has been tested as a predictor (Section~\ref{sec:method:results_single}). In the results we have, the downstream model moved the picked settings as much as the dataset did. On LogD the DNN picked 2,000 features on all 5 folds, and the SVM picked 1,000 on 4 of 5. These DNN results were computed before the DNN correction of 8 October 2026.
```
NEW:
```latex
We have not yet run the experiment that tests whether the best hybrid settings follow measurable properties of a dataset (Section~\ref{sec:method:results_single}). In the results we have, the downstream model changed the chosen settings as much as the dataset did. On LogD the DNN picked 2,000 features on all 5 folds, and the SVM picked 1,000 on 4 of 5.
```

### V16 — Future work: DNA result reported nowhere else; reactions wording differs from Chapter 5
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: "On DNA promoter sequences the hybrid beat the two highest-ranked single representations on each finished fold" is a result with no source in the thesis: Chapter 5's dataset note says nucleotides were cut, and the number of finished folds is not given (the backup said three, the outline says one). A result first stated in the closing chapter cannot be checked by the reader. The DECISIONS comment keeps nucleotide sequences as a next step, so the mention stays, without the result. "Results on reactions were mixed" differs from Chapter 5's Conclusions ("it gains nothing on the additive split"). "natural next domains" is an appraisal; give the reason instead.
OLD:
```latex
Results on reactions were mixed. On DNA promoter sequences the hybrid beat the two highest-ranked single representations on each finished fold. Gene-expression profiles and financial time series are natural next domains.
```
NEW:
```latex
On reactions it gained nothing on the additive split of Chapter~\ref{chap:method}. It has also run on DNA promoter sequences, but those runs are incomplete and are not reported in this thesis. Gene-expression profiles and financial time series are the next domains to test, since each has many competing representations of the same samples.
```

### V17 — Future work: two topics in one paragraph; flips sentence loose
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept; "% TODO: manual review" added above the new second paragraph
- why: "The noise methods need wider controls" then turns to class-label flips, which are not a control. "real flips concentrate near it" does not say near what or why: measurement error moves a value across the threshold only when the value is close to it. The QM9 paragraph also loses what more properties would show; the backup said it.
OLD:
```latex
The noise methods need wider controls. QM9 offers twelve computed properties, and only the HOMO--LUMO gap was used. In QSAR a class label usually comes from cutting a continuous value at a threshold, so real flips concentrate near it. Flips of that kind are the next test for the detectors.
```
NEW:
```latex
The noise methods need more clean controls. QM9 offers twelve computed properties, and only the HOMO--LUMO gap was used. Running the noise methods on several more would show whether the QM9 results hold beyond one target.

% TODO: manual review
Chapter~\ref{chap:noise} flipped class labels on hERG blocking along a confusion matrix. In QSAR, a class label usually comes from cutting a continuous measurement at a threshold. Measurement error then flips mostly the molecules whose values lie near that threshold. Flips of that kind, on more classification datasets, are the next test for the detectors.
```

### V18 — Concluding remarks: claim larger than the results; "not X. It Y" contrast
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: "Label noise was handled in the end, by ... and by changing the features they run on" says the hybrid handled noise; Chapter 7 found that the hybrid stayed ahead and changed which method won, not that it removed noise. "The hybrid is not a one-size-fits-all method. It adapts ..." is the rhetorical contrast the brief bans. "it adapts further when the labels are cleaned before the search" rests on one dataset, one fold, with no control (Chapter 7: expression titer, "No random-flag control has run for this experiment"); say so. "Everything depends on the data" is the author's outline wording (8.2a) and is kept in substance.
OLD:
```latex
Noise mitigation in molecular data is hard, and for most of this project it did not work. Label noise was handled in the end, by methods built or changed for the noise molecular data carry, and by changing the features they run on. No single model, representation, similarity measure or noise-handling method was best on every dataset we tested. Everything depends on the data: the features, the true labels, and the noise those labels carry. The hybrid is not a one-size-fits-all method. It adapts to the data it is given, and it adapts further when the labels are cleaned before the search.
```
NEW:
```latex
Noise mitigation in molecular data is hard, and for most of this project it did not work. In the end, label noise was handled in some settings, mostly by methods built or changed for the noise that molecular data carry. No single model, representation, similarity measure or noise-handling method was best on every dataset we tested. The result depends on the data, meaning the features, the true labels and the noise those labels carry. The hybrid search builds a different feature set for each dataset and model. On expression titer, cleaning the labels before the search helped more than cleaning them after it, on one fold and with no control yet.
```

### V19 — Concluding remarks: 31-word sentence; claim without its scope
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: One sentence is 31 words. The claim that the search decides about a new embedding was tested only on 2,000 molecules of LogD and hERG blocking (Chapter 5 Conclusions); the paragraph's own comment says so for small molecules.
OLD:
```latex
They add the new embedding to the representations searched and run the search again, and the search decides whether the new embedding contributes features and, if so, how many and which.
```
NEW:
```latex
They add the new embedding to the representations searched and run the search again. The search then decides whether the new embedding contributes features and, if so, how many and which. Chapter~\ref{chap:method} tested this on 2,000 molecules of LogD and hERG blocking.
```

### V20 — Last sentence: support claimed for noise as well as similarity
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: "on small molecules the results support it" covers noise too. Chapter 7's own Results say "The detectors told a more mixed story" and the best method still depended on the dataset. Keep the personal "I" (style guide: B keeps two personal sentences in his closure).
OLD:
```latex
My hypothesis was that they were the root of the trouble with similarity and with noise, and on small molecules the results support it.
```
NEW:
```latex
My hypothesis was that they were the root of the trouble with similarity and with noise. On small molecules the results support it for similarity. For noise they support it in part, since the hybrid stayed ahead through every cleaning but the best method still depended on the dataset.
```

### V21 — Contributions item 4 says "against controls"; Limitations says most methods have none
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" kept
- why: Chapter 7's Conclusions: "Most methods have no control. Five controls ran, one per experiment." The contribution should not read as if every method was tested against one.
OLD:
```latex
    \item Noise-handling methods tested on single representations and hybrids side by side, against controls: QM9 with no noise added, random deletion of the same size, and shuffled detector scores (Chapter~\ref{chap:noise}).
```
NEW:
```latex
    \item Noise-handling methods tested on single representations and hybrids side by side, with three kinds of control where they ran: QM9 with no noise added, random deletion of the same size, and shuffled detector scores (Chapter~\ref{chap:noise}).
```

### V22 — Stale comment in the DECISIONS block
- file: thesis.tex
- kind: comment
- todo_line: none (comment added, nothing deleted)
- why: The comment says the antibody permutation shares sit in Section sec:discussion:antibodies; they are no longer in this chapter, and Chapter 5's SOURCE says the 0.68 could not be reproduced. A note keeps the next reader from looking for them.
OLD:
```latex
% Every number below is copied from Chapters 5 to 7 as written, with that chapter's SOURCE comment carried over, except the antibody permutation shares in Section sec:discussion:antibodies, which were re-read from the raw c22 files on 2026-10-09 (SOURCE given there).
```
NEW:
```latex
% Every number below is copied from Chapters 5 to 7 as written, with that chapter's SOURCE comment carried over, except the antibody permutation shares in Section sec:discussion:antibodies, which were re-read from the raw c22 files on 2026-10-09 (SOURCE given there).
% [2026-10-10 review] The antibody permutation shares are no longer printed in this chapter. Chapter 5 gives 0.83 (SVM) with its own SOURCE and says the 0.68 (DNN) was not reproduced.
```

## For the author

- Abstract, last paragraph: "Label noise in molecular data therefore cannot be treated as Gaussian" claims more than Chapter 3 measured (systematic injected noise did the most damage). V5 changes Chapter 8; the abstract is another unit and needs the same change to keep the two in step.
- Chapter structure: your outline has 8.3 "Handling noise realistically", 8.4 "No single method fits every dataset" and 8.5 "Simple is often better" as separate sections, and asks for 14 to 18 pages. The rebuild cut the chapter to about 1,700 words with these folded into one paragraph. V13-V14 restore one section for the argument; whether you want the three back is yours.
- Your outline 8.4 "antibody hypothesis" (more room for distinct featurization on larger molecules) has a [DECIDE] still open; Chapter 8 now gives only the Chapter 5 reading (V9).
- DNA promoter sequences: the backup said three finished folds, the outline says one fold and "cut 2026-10-08". V16 keeps the mention without a result. Say whether any DNA result should be reported.
- "binding free energy" is the dataset name throughout Chapters 5 to 7, but the label is the change on mutation (ΔΔG) (chemistry guide item 8). Chapter 8 avoids the name after V9; a rename is thesis-wide.
- "scaffold family" (30 uses) against "scaffold group" (15) (chemistry guide item 14). Chapter 8 uses "scaffold family"; one term thesis-wide is your choice.
- Contribution 7, "the ranking of noise-handling methods depends on the feature space", rests on fold 0. Your outline's CHECKED note (2026-10-09) found the strong form held on one dataset. Confirm it should stay a listed contribution.
- Contribution 6 keeps its "originality claim pending literature check" TODO.
- Two [RESULT NEEDED]/[RESULT PENDING] slots remain: CHO polyreactivity real censoring (Limitations) and the dataset-property experiment c03.
- Summary of Research Contributions: one planned method paper or two is still marked DECISION FOR ADELAIDE.

## Applied
- V1 applied: chapter title in sentence case, "title was" TODO added.
- V2 applied as planned.
- V3 applied as planned (checked against Chapter 3, Bayesian transformations paragraph).
- V4 applied as planned.
- V5 applied as planned. Abstract not touched (listed under For the author).
- V6 applied as planned.
- V7 applied as planned.
- V8 applied as planned.
- V9 applied; the 31-word "Those representations ..." sentence split in two.
- V10 applied as planned.
- V11 applied as planned (checked against Chapter 7 Conclusions).
- V12 applied; the 31-word detection sentence split in two ("we compared the feature spaces in which the neighbor detectors find their neighbors" + "Taking the best setting of each, ...").
- V13 applied: Summary heading keeps sec:discussion:summary and sec:discussion:antibodies.
- V14 applied: paragraph split, recommendation paragraph with its own TODO, new section "Choosing a noise-handling method" carrying sec:discussion:noise, :conditional, :simple.
- V15 applied: both parts; inline TODO moved to its own line; SOURCE comment unchanged.
- V16 applied as planned.
- V17 applied as planned, with the new second paragraph's TODO.
- V18 applied with one change: Chapter 7 says cleaning both the search and the fit did best on expression titer, so the sentence now reads "cleaning the labels before both the search and the fit helped more than cleaning them before the fit alone. That result is on one fold, with no control yet."
- V19 applied as planned (checked against Chapter 5: LogD and hERG blocking runs used 2,000 molecules).
- V20 applied as planned.
- V21 applied with "where run" in place of "where they ran".
- V22 applied, corrected: Chapter 5's SOURCE gives 0.83 (SVM) and 0.81 (DNN) on binding free energy; the 0.68 was not reproduced.
