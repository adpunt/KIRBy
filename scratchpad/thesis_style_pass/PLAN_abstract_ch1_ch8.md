# Plan: abstract_ch1_ch8

Scope: the abstract in main.tex, Chapter 1 and Chapter 8 in thesis.tex. Source of the takeaways: scratchpad/thesis_style_pass/TAKEAWAYS.md. Every OLD block below was checked to be an exact, unique substring of the file on 2026-10-10. Every number in NEW text already appears in thesis.tex.

## Edits

### E1 — Abstract rewritten in chapter order, no numbers
- file: main.tex
- kind: ai-text
- why: Task brief and style guide item 13/14 (front matter): about 400 words, paragraphs in chapter order (problem; part one; the chain; the hybrid; noise handled, conditionally; software), contributions named, "we", few or no numbers as in the three reference abstracts. The current draft (about 300 words) has no paragraph for the hybrid as a contribution, does not say the noise was handled conditionally, and does not name the software.
- todo_line: keep existing, plus one added comment line
OLD:
```latex
% TODO: manual review -- [CLAUDE 2026-10-09] abstract drafted from Chapters 3 to 8 as written in thesis.tex on 2026-10-09 (about 300 words). Numbers: 13 base models and six representations (Chapter 3); 76% of the lost RMSE won back by censored EM on LogD (Chapter 7, [N7-20], fold 0).
Machine learning models of molecular properties are trained on experimental labels, and those labels carry noise from measurement error, differences between laboratories and the limits of an assay. This thesis asks what makes a model robust to that noise, and how the noise can be handled. We first introduce NoiseInject, an open-source Python package for benchmarking noise robustness, and use it to compare 13 base models and six molecular representations under seven kinds of label noise, on QM9 and three assay datasets. Model family was the main factor in noise robustness, while the representation mattered more for predictive performance. Systematic error, a shared offset per scaffold family or labels clipped at an assay limit, did the most damage, and no model's uncertainty picked out the corrupted labels.

Published methods for learning with noisy labels did poorly in this setting. We trace the failure to the similarity many of them rely on. On a single dataset, most features of a standard representation do not vary or do not relate to the property, yet each one adds to the distance between molecules. We therefore build the adaptive hybrid representation, a search that selects features from several base representations for one dataset and model, within each fold. On every small-molecule dataset the hybrid predicts better than the best single representation, and a hybrid built for distance gives better nearest neighbors. The method carries over to antibodies, reactions and DNA sequences without redesign.

Returning to label noise, we test detection, noise-robust training and noise mitigation on single representations and hybrids side by side, with QM9 as a clean control. Mitigation works best when the data say which labels are wrong: correcting labels clipped at an assay limit won back 76\% of the lost error on LogD. On small molecules the hybrid stays ahead through every kind of cleaning, and the ranking of noise methods changes with the features they run on. No single model, representation or noise method fits every dataset, so we argue for adapting the features to the data instead.
```
NEW:
```latex
% TODO: manual review -- [CLAUDE 2026-10-09] abstract drafted from Chapters 3 to 8 as written in thesis.tex on 2026-10-09 (about 300 words). Numbers: 13 base models and six representations (Chapter 3); 76% of the lost RMSE won back by censored EM on LogD (Chapter 7, [N7-20], fold 0).
% TODO: manual review -- [CLAUDE 2026-10-10] rewritten from scratchpad/thesis_style_pass/TAKEAWAYS.md (about 450 words, six paragraphs in chapter order, no numbers except "seven" noise conditions); it now matches Chapter 1's questions and Chapter 8's answers.
Machine learning models of molecular properties learn from experimental labels, and those labels carry noise from measurement error, differences between laboratories and the limits of an assay. Most studies of this noise assume it is Gaussian, and compare either molecular representations or models, not both. This thesis asks what makes a quantitative structure--activity relationship (QSAR) model robust to label noise, and how that noise can be handled.

We first compare a broad set of models and representations under seven kinds of label noise, on computed labels from QM9 and on three assay datasets. Model family decided noise robustness more than the representation did, and the pairing of model and representation explained more than the representation alone. Random forests were the most consistent model. The shape of the noise distribution barely mattered, but systematic noise, an offset shared within a scaffold family or labels clipped at an assay limit, did the most damage. Bayesian versions of the networks never predicted better, and outside censoring no model's uncertainty pointed to the corrupted labels.

Published methods for learning with noisy labels then did poorly on molecular data, and those that compare molecules by a distance did worst. We trace this failure to the representation. On a single dataset, most features of a standard representation do not vary or do not relate to the label, yet each adds to the distance.

We therefore build the adaptive hybrid representation. It ranks the features of several representations, splits a fixed budget of features between them, and searches the selection settings for each dataset and model. On every small-molecule dataset the hybrid predicted better than the best single representation, and a hybrid built for distance gave better nearest neighbors. The same search runs on antibodies, for which we built new region-level and mutation-level representations, and on reactions and DNA sequences. A new embedding enters the search as one more representation, so the method keeps up with new models.

Returning to label noise, we test noise detection, mitigation and noise-robust training on single representations and hybrids side by side, against random and shuffled controls. Label noise was handled, but which method worked depended on the task, the data, the model and the type of noise. Corrections worked best when the data flag the wrong labels, as an assay limit or a scaffold family does. The hybrid kept its lead through every cleaning on small molecules, and the ranking of noise methods changed with the feature space they ran in.

Two software packages accompany this thesis: NoiseInject, for benchmarking noise robustness and uncertainty, and KIRBy, for the hybrid search. No single model, representation or noise method fits every dataset. We argue instead for adapting the features to the data, and for choosing a noise method by how the labels were made.
```

### E2 — Ch1 opening paragraph 2: comma
- file: thesis.tex
- kind: author-text-trivial
- why: Misplaced comma ("are of course, sensitive"); grammar slip.
- todo_line: none (trivial)
OLD:
```latex
QSAR models are of course, sensitive
```
NEW:
```latex
QSAR models are, of course, sensitive
```

### E3 — Ch1 paragraph 3: "I" to "we", spelling, broad term
- file: thesis.tex
- kind: author-text
- why: Style guide item 7 lists "I will also discuss" here as "I" in the chapters; "cheminformatcs" misspelled; the broad problem is learning with noisy labels, of which noise mitigation is one subset, and Chapter 4 tests the broad set.
- todo_line: % TODO: manual review -- edited your text: "I" and "my" changed to "we" and "our" in the last two sentences, "cheminformatcs" corrected, and "noise mitigation techniques" changed to "methods for learning with noisy labels", since Chapter 4 tests the broad set, not only mitigation.
OLD:
```latex
This thesis addresses the challenge of conducting molecular property prediction when the underlying experimental data, or ``labels'' are noisy. Most work on noise in QSAR modeling assumes Gaussian label noise and evaluates either representations or model architectures in isolation. \citet{Cortes2015} examined 12 algorithms across multiple noise levels but used fixed representations, and studies that compare representations typically use clean data. No work has systematically compared the relative importance of representation versus architecture choice for label noise robustness, asked whether probabilistic methods justify their computational overhead under noise, or tested whether findings generalize across different types of injected noise. This thesis fills that gap. I will also discuss noise mitigation, using the results of my noise robustness study. I explore why many noise mitigation techniques underperform in cheminformatcs data and methods to address this problem in the form of novel molecular representations.
```
NEW:
```latex
% TODO: manual review -- edited your text: "I" and "my" changed to "we" and "our" in the last two sentences, "cheminformatcs" corrected, and "noise mitigation techniques" changed to "methods for learning with noisy labels", since Chapter 4 tests the broad set, not only mitigation.
This thesis addresses the challenge of conducting molecular property prediction when the underlying experimental data, or ``labels'' are noisy. Most work on noise in QSAR modeling assumes Gaussian label noise and evaluates either representations or model architectures in isolation. \citet{Cortes2015} examined 12 algorithms across multiple noise levels but used fixed representations, and studies that compare representations typically use clean data. No work has systematically compared the relative importance of representation versus architecture choice for label noise robustness, asked whether probabilistic methods justify their computational overhead under noise, or tested whether findings generalize across different types of injected noise. This thesis fills that gap. We also discuss noise mitigation, using the results of our noise robustness study. We explore why many methods for learning with noisy labels underperform in cheminformatics data and methods to address this problem in the form of novel molecular representations.
```

### E4 — Ch1 bridge paragraph made self-contained; hybrid questions rewritten (your four questions kept)
- file: thesis.tex
- kind: author-text
- why: Follows the author TODOs "come back to this at the end and double-check the contents" (bridge paragraph) and "definitely modify these" (numbered list); the bridge opened "the fault lay in this distance" with no distance introduced before it; style guide item 8 (Chapter 6 and 7 work missing from the list). The four first-half questions in prose carry no TODO and are not wrong, so they are kept as written.
- todo_line: two lines, see NEW
OLD:
```latex
% TODO: come back to this at the end and double-check the contents, this is missing a lot because it was written prior to results in chapters 5+
We hypothesized that the fault lay in this distance. On a given dataset, most features of a general-purpose representation either do not vary or do not relate to the property, yet each adds a term to the distance, so together they dominate it (Chapter~\ref{chap:flaws}). We propose instead to compute distances over a hybrid representation: a fixed number of task-relevant features selected from several representations. A first version showed that the selected features also work as ordinary model inputs, so the question became how to choose them (Chapter~\ref{chap:method}).

The second half asks whether the representation itself can be adapted to the task:\label{sec:intro:hybrid_questions}
% TODO: definitely modify these
\begin{enumerate}
    \item How can the best set of features be selected for a given prediction task?
    \item Do hybrid representations retain more of their predictive performance under label noise than their base representations?
    \item Do the features selected for a hybrid representation help stabilize distance-based noise mitigation?
    \item Does weighting features by their importance improve the construction of hybrid representations, model training and noise mitigation?
    \item When sample removal is suboptimal, can sample weighting or label smoothing handle Gaussian label noise better?
\end{enumerate}
```
NEW:
```latex
% TODO: come back to this at the end and double-check the contents, this is missing a lot because it was written prior to results in chapters 5+
% TODO: manual review -- edited your text: two sentences added at the start to say which failure and which distance, since "this distance" had nothing before it to refer to; one sentence added at the end pointing to Chapter 7; your sentences kept.
The second half of this thesis grew out of a failure. In an exploratory study carried out earlier in this project, published methods for learning with noisy labels did poorly on QSAR data. Those that compared molecules by a distance did worst (Chapter~\ref{chap:flaws}). We hypothesized that the fault lay in this distance. On a given dataset, most features of a general-purpose representation either do not vary or do not relate to the property, yet each adds a term to the distance, so together they dominate it (Chapter~\ref{chap:flaws}). We propose instead to compute distances over a hybrid representation: a fixed number of task-relevant features selected from several representations. A first version showed that the selected features also work as ordinary model inputs, so the question became how to choose them (Chapter~\ref{chap:method}). We then return to label noise with the hybrid and the single representations side by side (Chapter~\ref{chap:noise}).

The second half asks whether the representation itself can be adapted to the task:\label{sec:intro:hybrid_questions}
% TODO: definitely modify these
% TODO: manual review -- edited your text: your questions 1 and 4 kept word for word and in the same places (Chapter 5's Conclusions answers "the fourth question"); question 2 now asks Chapter 6's question on distance; question 3 now asks Chapter 7's question and takes in your old questions 2, 3 and 5 (noise robustness of the hybrid, distance-based noise mitigation, sample weighting and label smoothing).
\begin{enumerate}
    \item How can the best set of features be selected for a given prediction task?
    \item Does distance in a hybrid representation track the label better than distance in the best single representation, and do methods built on nearest neighbors work better in it?
    \item Can noisy labels be found and corrected, by deletion, sample weighting, label smoothing or other means, and do hybrid representations keep their lead through that cleaning?
    \item Does weighting features by their importance improve the construction of hybrid representations, model training and noise mitigation?
\end{enumerate}
```

### E5 — Ch1 outline, Chapter 2 paragraph
- file: thesis.tex
- kind: author-text
- why: "artifical" misspelled; NoiseInject was named only in the old question paragraph, and Chapter 2 is where it is described (Section sec:noiseinject_pkg).
- todo_line: % TODO: manual review -- edited your text: "artifical" corrected, the last sentence split off, and NoiseInject named, since Chapter 2 is where it is described.
OLD:
```latex
Chapter~\ref{chap:background} introduces molecular representations, model families, molecular similarity and activity cliffs, and the sources of noise in molecular data. It defines the seven noise conditions used in Chapter~\ref{chap:robustness} and the three groups of methods for noisy labels (noise detection, noise mitigation and noise-robust training), and closes with artifical noise injection, and evaluation metrics.
```
NEW:
```latex
% TODO: manual review -- edited your text: "artifical" corrected, the last sentence split off, and NoiseInject named, since Chapter 2 is where it is described.
Chapter~\ref{chap:background} introduces molecular representations, model families, molecular similarity and activity cliffs, and the sources of noise in molecular data. It defines the seven noise conditions used in Chapter~\ref{chap:robustness} and the three groups of methods for noisy labels (noise detection, noise mitigation and noise-robust training). It closes with artificial noise injection, the NoiseInject package (Section~\ref{sec:noiseinject_pkg}) and the evaluation metrics.
```

### E6 — Ch1 outline, Chapter 3 paragraph
- file: thesis.tex
- kind: author-text
- why: Style guide item 11: tense mixes within the paragraph ("looks into ... were also run ... We compare ... We matched ... We measure ... We found ... matters"); the sentence "Label noise mean predicted uncertainty ..." is broken, and "only models with a variational head" disagrees with Chapter 3's Conclusions (BNNs with a variance head, the heteroscedastic GP and the heteroscedastic VBLL networks); "made-up" is informal (chemistry guide item 12, and the glossary entry for "censored label").
- todo_line: % TODO: manual review -- edited your text: tense made consistent (present for what the chapter does, past for what it found); the broken uncertainty sentence split in two and its model list matched to Chapter 3's Conclusions; "a made-up assay limit" changed to "a fixed value, in imitation of an assay limit".
OLD:
```latex
Chapter~\ref{chap:robustness} looks into what makes a QSAR model robust to label noise. The primary prediction target is the HOMO--LUMO gap in QM9, whose computed labels carry negligible measurement noise. The same experiments were also run on three assay datasets: LogD, Caco-2 and hERG $K_i$. We compare seven different types of label noise, referred to as noise conditions. We matched the quantity of noise to the training set label spread for all but one of the noise conditions. We measure noise robustness as the proportion of predictive performance retained as artificial label noise increases. We found that model family was the main factor in noise robustness, and choice of molecular representation matters more for predictive performance than noise robustness. Strong predictive performance on clean labels does not imply robustness. The models with the highest predictive performance, LightGBM, XGBoost and the plain neural networks lost the most with increasing label noise, while random forests were the most consistent model across datasets, representations, and noise conditions. We replicated systematic noise by adding it in disproportionate amounts to select scaffold families and by clipping labels at a made-up assay limit. Models struggled the most when training on this systematic noise. Label noise mean predicted uncertainty for the probabilistic models which estimate it, however, only models with a variational head correctly attributed the rise in uncertainty to the data rather than to the model itself.
```
NEW:
```latex
% TODO: manual review -- edited your text: tense made consistent (present for what the chapter does, past for what it found); the broken uncertainty sentence split in two and its model list matched to Chapter 3's Conclusions; "a made-up assay limit" changed to "a fixed value, in imitation of an assay limit".
Chapter~\ref{chap:robustness} looks into what makes a QSAR model robust to label noise. The primary prediction target is the HOMO--LUMO gap in QM9, whose computed labels carry negligible measurement noise. The same experiments are also run on three assay datasets: LogD, Caco-2 and hERG $K_i$. We compare seven different types of label noise, referred to as noise conditions. We match the quantity of noise to the training set label spread for all but one of the noise conditions. We measure noise robustness as the proportion of predictive performance retained as artificial label noise increases. We found that model family was the main factor in noise robustness, and choice of molecular representation mattered more for predictive performance than noise robustness. Strong predictive performance on clean labels did not imply robustness. The models with the highest predictive performance, LightGBM, XGBoost and the plain neural networks lost the most with increasing label noise, while random forests were the most consistent model across datasets, representations, and noise conditions. We imitate systematic noise by adding it in disproportionate amounts to select scaffold families and by clipping labels at a fixed value, in imitation of an assay limit. Models struggled the most when training on this systematic noise. Mean predicted uncertainty rose with label noise for the probabilistic models which estimate it. However, only the networks with a variance head, the heteroscedastic GP and the heteroscedastic VBLL networks correctly attributed the rise in uncertainty to the data rather than to the model.
```

### E7 — Ch1 outline, Chapter 4 paragraph
- file: thesis.tex
- kind: author-text
- why: "None of the methods that incorporated the concept of molecular similarity showed any promising results" is contradicted by Chapter 4, which reports that one neighbor-based configuration gave the best recovery at the higher noise level; Chapter 4's own summary says the configurations that used a distance "did worst".
- todo_line: % TODO: manual review -- edited your text: "None of the methods that incorporated the concept of molecular similarity showed any promising results" changed to "did worst", since Chapter 4 reports one neighbor-based configuration as the best at the higher noise level.
OLD:
```latex
Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular similarity. The natural next step to Chapter~\ref{chap:robustness} is noise mitigation. However, a small sweep of twelve published noise-handling methods, four that remove molecules and retrain and eight that change the training, showed disappointing results. None of the methods that incorporated the concept of molecular similarity showed any promising results. On a given dataset, standard representations carry many features that do not vary, and some map different molecules to the same vector. Similarity computed over these representations inherits both of these problems, and methods that rely on similarity suffer as a result. The chapter ends with what a representation would need to mitigate these problems, which Chapter~\ref{chap:method} builds.
```
NEW:
```latex
% TODO: manual review -- edited your text: "None of the methods that incorporated the concept of molecular similarity showed any promising results" changed to "did worst", since Chapter 4 reports one neighbor-based configuration as the best at the higher noise level.
Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular similarity. The natural next step to Chapter~\ref{chap:robustness} is noise mitigation. However, a small sweep of twelve published noise-handling methods, four that remove molecules and retrain and eight that change the training, showed disappointing results. The configurations that incorporated the concept of molecular similarity did worst. On a given dataset, standard representations carry many features that do not vary, and some map different molecules to the same vector. Similarity computed over these representations inherits both of these problems, and methods that rely on similarity suffer as a result. The chapter ends with what a representation would need to mitigate these problems, which Chapter~\ref{chap:method} builds.
```

### E8 — Ch1 outline, Chapter 5 paragraphs: spelling, missing word, findings added
- file: thesis.tex
- kind: author-text
- why: Follows the author TODO "waiting on results, may need another pass" above the second paragraph; "The hybrid is compared the base representations" is missing "with"; "customises" (British) and "genereate" are spelling slips (fixed with no comment).
- todo_line: % TODO: manual review -- edited your text: (second paragraph only, see NEW)
OLD:
```latex
The adaptive hybrid search customises the settings of a feature selection process to a given dataset, model, and budget. It takes in several base representations and selects the most important features for that task by a fixed budget. This produces a tailor-made feature set, referred to as the hybrid representation. The actual output of the hybrid search is not the set of features, but rather the settings that define the feature selection method used to genereate that hybrid representation. 

% TODO: waiting on results, may need another pass
The hybrid is compared the base representations on the same data, an ensemble of one model per representation, principal component analysis, minimum-redundancy maximum-relevance selection, recursive feature elimination and random features of the same size. It also investigates properties of the hybrid representation, including whether the selected features are interpretable and stable, whether search settings transfer between datasets, and whether the method extends to other domains and to data held at separate sites.
```
NEW:
```latex
The adaptive hybrid search customizes the settings of a feature selection process to a given dataset, model, and budget. It takes in several base representations and selects the most important features for that task by a fixed budget. This produces a tailor-made feature set, referred to as the hybrid representation. The actual output of the hybrid search is not the set of features, but rather the settings that define the feature selection method used to generate that hybrid representation. 

% TODO: waiting on results, may need another pass
% TODO: manual review -- edited your text: "with" added to "The hybrid is compared the base representations"; two sentences of findings added at the end from Chapter 5's Conclusions, as your TODO above asked.
The hybrid is compared with the base representations on the same data, an ensemble of one model per representation, principal component analysis, minimum-redundancy maximum-relevance selection, recursive feature elimination and random features of the same size. It also investigates properties of the hybrid representation, including whether the selected features are interpretable and stable, whether search settings transfer between datasets, and whether the method extends to other domains and to data held at separate sites. On every small-molecule dataset the hybrid predicted better than the best single representation with both downstream models, including the representation that was best in hindsight. On the smaller antibody datasets its gains were inconsistent, and an ensemble of one model per representation often did better.
```

### E9 — Ch1 outline, Chapter 6 paragraph brought up to date and Chapter 7 paragraph added
- file: thesis.tex
- kind: author-text + new-paragraph
- why: Follows the author TODO "come back to all of this, it's all out of date"; style guide item 8: the outline has no paragraph for Chapter 7 and the Chapter 6 paragraph describes noise work that is now Chapter 7; "preliminary" results are now final in Chapter 6.
- todo_line: % TODO: manual review -- edited your text: (Chapter 6) / % TODO: manual review (Chapter 7, new)
OLD:
```latex
% TODO: come back to all of this, it's all out of date
Chapter~\ref{chap:similarity} asks whether a hybrid representation repairs the failures described in Chapter~\ref{chap:flaws}. It compares standard and hybrid representations selected for predicting each molecule's activity from its five nearest neighbors, asks whether neighbors in a hybrid representation identify corrupted labels better than neighbors in a standard one, and tests noise mitigation and noise-robust training methods one noise type at a time. Its results are also preliminary. On hERG $K_i$, predicting each molecule's activity from its five nearest neighbors was more accurate in a hybrid than in the best single representation, and, on ECFP4, correcting labels clipped at a limit by one shared amount recovered more than half of the lost performance, whereas deleting the clipped molecules gave a higher error than keeping them. Under Gaussian noise on nanobody thermostability, every one of twelve published noise-handling methods scored higher on the hybrid than any of them did on one-hot, the weakest input.
```
NEW:
```latex
% TODO: come back to all of this, it's all out of date
% TODO: manual review -- edited your text: the noise part moved to a new Chapter 7 paragraph below; "Its results are also preliminary" and the preliminary hERG $K_i$ and nanobody results replaced by Chapter 6's findings; "activity" changed to "label", since QM9, LogD and Caco-2 labels are not activities (chemistry guide item 17).
Chapter~\ref{chap:similarity} asks whether a hybrid representation repairs the failures of similarity described in Chapter~\ref{chap:flaws}. It builds hybrids for predicting each molecule's label from its five nearest neighbors, and compares their distances and neighbors with those of standard representations. On all five small-molecule datasets, distance in the hybrid tracked the label more closely and the neighbors predicted better. On the antibody and reaction datasets distance also tracked the label more closely, but prediction from neighbors was worse in the hybrid on several of them.

% TODO: manual review
Chapter~\ref{chap:noise} returns to label noise, with the single representation and the hybrid side by side. It tests noise detection, noise mitigation and noise-robust training, on labels damaged on purpose and on labels as delivered. QM9 with no noise added, random deletion and shuffled detector scores serve as controls. Noise mitigation worked best when a known flag, such as an assay limit or a scaffold family, said which labels were wrong. With a detector it helped under label flips and outliers, and on several datasets it beat the controls on the labels as delivered. Which method worked depended on the type of noise, the data and the model. The hybrid stayed ahead of the single representation on small molecules through every treatment, and the ranking of noise methods changed with the feature space.
```

### E11 — Ch1 contributions named
- file: thesis.tex
- kind: new-paragraph
- why: Style guide item 13: contributions should be named in the abstract, Chapter 1 and Chapter 8; the section is titled "Outline of the thesis and contributions" but names only the software. List bounded as Chapter 8 bounds it ("we found no published source"). Her sentence is kept unchanged; the list is inserted after it.
- todo_line: % TODO: manual review
OLD:
```latex
The software contributions are NoiseInject and the KIRBy repository, which implements the search for hybrid representations and the experiments (Additional file~\ref{app:availability}).
```
NEW:
```latex
The software contributions are NoiseInject and the KIRBy repository, which implements the search for hybrid representations and the experiments (Additional file~\ref{app:availability}).

% TODO: manual review
Beyond the software, the original contributions of this thesis are the following.
\begin{itemize}
    \item Seven noise conditions at a matched amount of noise, with parameters taken from published measurement errors where they exist. They include noise grouped by scaffold family and labels clipped at an assay limit (Chapter~\ref{chap:background}).
    \item A split of the variance in performance and in noise robustness between model family, representation and their pairing (Chapter~\ref{chap:robustness}).
    \item The adaptive hybrid representation, which ranks features within each representation and then splits a budget of features among them (Chapter~\ref{chap:method}).
    \item A search over the selection settings for each dataset and model, saved as a blueprint that traces each feature to its representation (Chapter~\ref{chap:method}).
    \item Hybrids built for distance rather than prediction (Chapter~\ref{chap:similarity}), and hybrids built where data are held at separate sites (Chapter~\ref{chap:method}).
    \item Antibody representations built for this thesis: readouts of the complementarity-determining regions (CDRs) and of each region, mutation features, and the change in embedding from the parent antibody (Chapter~\ref{chap:method}).
    \item Noise-handling methods for which we found no published source, and published methods with our own changes, tested against controls (Chapter~\ref{chap:noise}).
    \item The finding that the ranking of noise-handling methods depends on the feature space they run in (Chapter~\ref{chap:noise}).
\end{itemize}
```

### E12 — Ch1 Collaborations: Chapter 7 missing from the author's own work
- file: thesis.tex
- kind: author-text
- why: Wrong cross-reference range: "Chapters flaws to similarity" leaves out Chapter 7 (label noise), which is also the author's work.
- todo_line: % TODO: manual review -- edited your text: the range now ends at Chapter 7 (label noise) instead of Chapter 6, which left Chapter 7 out.
OLD:
```latex
The adaptive hybrid representation, its software and the experiments in Chapters~\ref{chap:flaws} to~\ref{chap:similarity} are the author's own work.
```
NEW:
```latex
% TODO: manual review -- edited your text: the range now ends at Chapter 7 (label noise) instead of Chapter 6, which left Chapter 7 out.
The adaptive hybrid representation, its software and the experiments in Chapters~\ref{chap:flaws} to~\ref{chap:noise} are the author's own work.
```

### E13 — Ch8 opening paragraph points to answers, not chapter summaries
- file: thesis.tex
- kind: ai-text
- why: Style guide item 8: Chapter 8 should answer Chapter 1's questions in order (D's recap form); the Summary section now does that.
- todo_line: keep existing
OLD:
```latex
This thesis set out to handle label noise in molecular data. Nobody asked me to look at the features. The noise methods kept failing, and every time we asked why, the answer sat one level further down, until it reached the representation itself. This chapter brings the three result chapters together and argues that following that chain was the right call. We first summarize each chapter (Section~\ref{sec:discussion:summary}), then lay out the chain of the argument (Section~\ref{sec:discussion:chain}) and the claim it supports (Section~\ref{sec:discussion:claim}). We then set out what is new in this work (Section~\ref{sec:discussion:contributions}) and what the noise results mean in practice (Section~\ref{sec:discussion:noise}). After that we discuss why the method carried over to antibodies (Section~\ref{sec:discussion:antibodies}), and when each method works (Sections~\ref{sec:discussion:conditional} and~\ref{sec:discussion:simple}). The chapter ends with the limitations, future directions and conclusions.
```
NEW:
```latex
This thesis set out to handle label noise in molecular data. Nobody asked me to look at the features. The noise methods kept failing, and every time we asked why, the answer sat one level further down, until it reached the representation itself. This chapter brings the three result chapters together and argues that following that chain was the right call. We first answer the research questions of Chapter~\ref{chap:intro} in order (Section~\ref{sec:discussion:summary}), then lay out the chain of the argument (Section~\ref{sec:discussion:chain}) and the claim it supports (Section~\ref{sec:discussion:claim}). We then set out what is new in this work (Section~\ref{sec:discussion:contributions}) and what the noise results mean in practice (Section~\ref{sec:discussion:noise}). After that we discuss why the method carried over to antibodies (Section~\ref{sec:discussion:antibodies}), and when each method works (Sections~\ref{sec:discussion:conditional} and~\ref{sec:discussion:simple}). The chapter ends with the limitations, future directions and conclusions.
```

### E14 — Ch8 Summary restructured: one answer per Chapter 1 question, in order
- file: thesis.tex
- kind: ai-text
- why: Style guide item 8 ("If Chapter 1 numbers its questions, Chapter 8 answers each by number"); TAKEAWAYS.md section 7. Follows Chapter 1 as E4 leaves it: the author's four first-half questions, the Chapter 4 bridge, then the four hybrid questions. The fourth hybrid answer copies Chapter 5's Conclusions and Chapter 8's contributions section. Numbers are the ones already in these paragraphs, with their SOURCE comments carried over; Chapter 3 answers stay in words, as the existing comment says; "distance also tracks activity" on antibodies and reactions changed to "label" (chemistry guide item 17).
- todo_line: keep existing on each paragraph; one [CLAUDE] comment added at the top
OLD:
```latex
% Words only; the numbers are in Chapter 3's Conclusions.
Chapter~\ref{chap:robustness} asked what makes a QSAR model robust to label noise. Model family was the main factor in noise robustness, while the representation mattered more for predictive performance than for robustness. Grouped-shifted noise and censoring, the two conditions where the error is systematic, did the most damage. The shape of the noise distribution made little difference. Random forests were the most consistent model across the four datasets. Mean predicted uncertainty rose with noise, but no model's uncertainty for a single molecule picked out which labels had been corrupted. Under censoring the models were most confident about the clipped labels.

% TODO: manual review
Chapter~\ref{chap:flaws} looked at why the published methods for learning with noisy labels did so poorly in that setting. On a single dataset, very little of a standard representation varies. On hERG $K_i$, 1,815 of the 2,048 ECFP4 features take one value in at least 95\% of the molecules, and 56 of the 1,412 molecules share their feature vector with another molecule. A distance sums over every feature, so the features that carry nothing dominate it. Methods that judge a label by its neighbors' labels inherit this, and in the exploratory study they usually made the model worse.

% TODO: manual review
% SOURCE (carried over from Chapter 5, Section sec:method:results_single): /Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/<ds>.json standard_selection_summary.<model>.{base1,hybrid1}.per_fold[].heldout_score, 5 folds. caco2 svm 0.3036 -> 0.2862; logd svm 0.4930 -> 0.4302; qm9 dnn 0.3619 -> 0.3155 (RERUN-SWAP-DNN); flab_expression_koenig svm 0.7029 -> 0.7729, 5/5 folds; ddg_combined dnn 0.1317 -> 0.3542, 4/5 (RERUN-SWAP-DNN).
Chapter~\ref{chap:method} built the adaptive hybrid representation. A search selects features from up to ten base representations for one dataset and downstream model, inside each fold. On every small-molecule dataset the hybrid beat the best single representation with both downstream models, and on the six regression datasets it did so on every fold. On LogD with the SVM it went from 0.4930 to 0.4302 RMSE, and on QM9 with the DNN from 0.3619 to 0.3155 eV. It also beat the single representation that was best in hindsight, which no user could pick in advance. On antibodies the gains depend on the dataset. On the anti-VEGF expression scan the hybrid went from 0.7029 to 0.7729 Spearman with the SVM, ahead on all five folds, and on binding free energy from 0.1317 to 0.3542 with the DNN.

% TODO: manual review
Chapter~\ref{chap:similarity} asked whether a hybrid built for distance gives better neighbors. On all five small-molecule datasets it does. On hERG $K_i$ the correlation between the distance of two molecules and the difference in their activity goes from 0.0729 to 0.2625 Spearman. Predicting each molecule from its five nearest neighbors goes from 0.6280 to 0.5822 RMSE. On the antibody datasets and the Buchwald--Hartwig reactions distance also tracks activity more closely, but prediction from neighbors is worse in the hybrid on several of them.

% TODO: manual review
Chapter~\ref{chap:noise} returned to label noise, with the single representation and the hybrid side by side. Noise mitigation works. With a quarter of LogD's training labels clipped at an assay limit, censored EM took the network from 0.6354 to 0.4795 RMSE, against 0.4314 before clipping. Under hERG's label flips, deleting the flagged molecules took gradient-boosted trees on the hybrid from 0.8028 to 0.8375 AUROC, against 0.8179 for random deletion of the same size. On the labels as delivered, label smoothing took the network on hERG $K_i$ from 0.6380 to 0.5775 RMSE, where shuffled detector scores stayed at 0.6419 or above. On SMAC colloidal stability, deleting the flagged antibodies took the SVM from 0.2740 to 0.4034 Spearman, against at most 0.2782 with shuffled scores.
```
NEW:
```latex
% [CLAUDE 2026-10-10] Restructured: one paragraph per research question of Chapter 1, in the same order (the four first-half questions, then Chapter 4, then the four hybrid questions). Numbers carried over from the five paragraphs this replaces, with their SOURCE comments; nothing new added. The section title "Summary" is left for the separate titles task.
% Words only; the numbers are in Chapter 3's Conclusions.
\textit{How much of the variance in predictive performance, and in noise robustness, do the representation and the model explain?} Model family was the main factor in noise robustness, while the representation mattered more for predictive performance than for robustness (Chapter~\ref{chap:robustness}). The pairing of model and representation explained more of the variance in robustness than the representation alone. So the two should be chosen together.

% TODO: manual review
\textit{What does the choice of model provide under label noise, and does probabilistic modeling help?} Random forests were the most consistent model across the four datasets. Strong performance on clean labels did not imply robustness. LightGBM and XGBoost predicted well on clean labels and lost the most as noise rose. Making a network Bayesian raised its robustness, but the Bayesian versions never predicted better at any noise level.

% TODO: manual review
\textit{Does the type of label noise matter, or only its amount?} The type matters. The shape of the noise distribution made little difference. Grouped-shifted noise and censoring, the two conditions where the error is systematic, did the most damage. Which model to prefer also changed with the noise condition (Table~\ref{tab:condition_choice}).

% TODO: manual review
\textit{What does label noise do to a model's uncertainty?} Mean predicted uncertainty rose with noise for almost every probabilistic model on QM9. Outside censoring, no model's uncertainty for a single molecule picked out which labels had been corrupted. Under censoring the models were most confident about the clipped labels.

% TODO: manual review
Chapter~\ref{chap:flaws} looked at why the published methods for learning with noisy labels did so poorly in that setting. On a single dataset, very little of a standard representation varies. On hERG $K_i$, 1,815 of the 2,048 ECFP4 features take one value in at least 95\% of the molecules. And 56 of its 1,412 molecules share their feature vector with another molecule. A distance sums over every feature, so the features that carry nothing dominate it. Methods that judge a label by its neighbors' labels inherit this, and in the exploratory study they usually made the model worse.

% TODO: manual review
% SOURCE (carried over from Chapter 5, Section sec:method:results_single): /Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/<ds>.json standard_selection_summary.<model>.{base1,hybrid1}.per_fold[].heldout_score, 5 folds. caco2 svm 0.3036 -> 0.2862; logd svm 0.4930 -> 0.4302; qm9 dnn 0.3619 -> 0.3155 (RERUN-SWAP-DNN); flab_expression_koenig svm 0.7029 -> 0.7729, 5/5 folds; ddg_combined dnn 0.1317 -> 0.3542, 4/5 (RERUN-SWAP-DNN).
\textit{How can the best set of features be selected for a given prediction task?} By a search for each dataset and model (Chapter~\ref{chap:method}). It selects features from up to ten base representations for one dataset and downstream model, inside each fold. On every small-molecule dataset the hybrid beat the best single representation with both downstream models, and on the six regression datasets it did so on every fold. On LogD with the SVM it went from 0.4930 to 0.4302 RMSE, and on QM9 with the DNN from 0.3619 to 0.3155 eV. It also beat the single representation that was best in hindsight, which no user could pick in advance. On antibodies the gains depend on the dataset. On the anti-VEGF expression scan the hybrid went from 0.7029 to 0.7729 Spearman with the SVM, ahead on all five folds. On binding free energy it went from 0.1317 to 0.3542 with the DNN.

% TODO: manual review
\textit{Does distance in a hybrid track the label better, and do methods built on nearest neighbors work better in it?} On all five small-molecule datasets, yes (Chapter~\ref{chap:similarity}). On hERG $K_i$ the correlation between the distance of two molecules and the difference in their activity goes from 0.0729 to 0.2625 Spearman. Predicting each molecule from its five nearest neighbors goes from 0.6280 to 0.5822 RMSE. On the antibody datasets and the Buchwald--Hartwig reactions distance also tracks the label more closely, but prediction from neighbors is worse in the hybrid on several of them.

% TODO: manual review
\textit{Can noisy labels be found and corrected, and does the hybrid keep its lead?} Yes, under conditions set by the noise, the data and the model (Chapter~\ref{chap:noise}). With a quarter of LogD's training labels clipped at an assay limit, censored EM took the network from 0.6354 to 0.4795 RMSE, against 0.4314 before clipping. Under hERG's label flips, deleting the flagged molecules took gradient-boosted trees on the hybrid from 0.8028 to 0.8375 AUROC, against 0.8179 for random deletion of the same size. On the labels as delivered, label smoothing took the network on hERG $K_i$ from 0.6380 to 0.5775 RMSE, where shuffled detector scores stayed at 0.6419 or above. On SMAC colloidal stability, deleting the flagged antibodies took the SVM from 0.2740 to 0.4034 Spearman, against at most 0.2782 with shuffled scores. The hybrid kept its lead over the single representation on small molecules through every treatment. The ranking of noise methods changed with the feature space they ran in (Section~\ref{sec:discussion:chain}).

% TODO: manual review
\textit{Does weighting features by their importance help?} In construction, ranking features by importance within each representation is what makes the hybrid work (Chapter~\ref{chap:method}). In training, using the importance scores again adds little on small molecules and helps on some small antibody datasets, with no condition that works everywhere. In noise mitigation, importance-weighted features are one of the spaces in which a neighbor detector can look for neighbors (Chapter~\ref{chap:noise}).
```

### E15 — Ch8 central claim: activity cliffs are not noise; claim bounded to what was measured; takeaways paragraph added
- file: thesis.tex
- kind: author-text + new-paragraph
- why: The paragraph has a "% TODO: manual review" line above it, but its own comment says "the author's words", so it is treated as her text: only the chemistry error is fixed (chemistry guide item 6: activity cliffs are real information, not noise; Chapter 8's own cliff-aware gap and cliff put-back rest on this). Her claim "no single method can work" is left as written (see For the author). TAKEAWAYS.md sections 1, 4 and 5 for the new paragraph.
- todo_line: keep existing; added "% TODO: manual review -- edited your text: ..." line (see NEW); new paragraph gets "% TODO: manual review"
OLD:
```latex
% Outline 8.2a, the author's words (2026-10-08).
There is no single best model, representation, similarity measure, method for learning with noisy labels or noise mitigation method that fits every dataset. Everything depends on the data: the features, the true labels, and the noise those labels carry. Every dataset carries its own noise, as outliers and activity cliffs, and it is not predictable in advance. So the honest answer is not a winner. It is a broad sweep, which this thesis runs again and again, together with hybrid featurization, a procedure that builds a feature selection for each dataset. The hybrid is not a one-size-fits-all method. It adapts to the data it is given, including data whose corruption is unknown, and it adapts further when used alongside noise mitigation. The claim is not ``this method works''. It is that no single method can work, so we should change the data, in terms of its features, instead.
```
NEW:
```latex
% Outline 8.2a, the author's words (2026-10-08).
% TODO: manual review -- edited your text: "carries its own noise, as outliers and activity cliffs" changed so that activity cliffs are not called noise, since they are real differences in activity and Chapter 7's cliff-aware methods rest on that.
There is no single best model, representation, similarity measure, method for learning with noisy labels or noise mitigation method that fits every dataset. Everything depends on the data: the features, the true labels, and the noise those labels carry. Every dataset carries its own noise, such as outliers, and its own activity cliffs, which a noise method can mistake for noise. Neither is predictable in advance. So the honest answer is not a winner. It is a broad sweep, which this thesis runs again and again, together with hybrid featurization, a procedure that builds a feature selection for each dataset. The hybrid is not a one-size-fits-all method. It adapts to the data it is given, including data whose corruption is unknown, and it adapts further when used alongside noise mitigation. The claim is not ``this method works''. It is that no single method can work, so we should change the data, in terms of its features, instead.

% TODO: manual review
Two things follow from the results. First, label noise in QSAR data was handled. Correcting clipped labels, removing scaffold-family offsets, deleting what a precise detector flags and label smoothing each improved held-out scores (Section~\ref{sec:discussion:noise}). Which of them worked depended on four conditions: the task, the data, the model and the type of noise (Section~\ref{sec:discussion:conditional}). Second, the hybrid works for both halves of the problem. On small molecules it predicts better than the best single representation, gives neighbors that predict better, and keeps its lead through every cleaning. It is also forward compatible. A new embedding is one more representation for the search, and on Caco-2 and hERG $K_i$ the features selected from every embedding beat random features put in the same places.
```

### E16 — Ch8 contributions: noise-robustness contributions added before the hybrid paragraph
- file: thesis.tex
- kind: new-paragraph
- why: Style guide item 13 and TAKEAWAYS.md section 6 items 1 to 3: the section names the hybrid and the Chapter 7 methods but not NoiseInject, the noise conditions or the variance split.
- todo_line: % TODO: manual review
OLD:
```latex
\section{What Is New in This Thesis}\label{sec:discussion:contributions}
```
NEW:
```latex
\section{What Is New in This Thesis}\label{sec:discussion:contributions}

% TODO: manual review
The first contributions concern noise robustness (Chapters~\ref{chap:background} and~\ref{chap:robustness}). NoiseInject is an open-source Python package for benchmarking noise robustness and uncertainty. It applies seven noise conditions at a matched amount of noise, with parameters taken from published measurement errors where they exist. Among them are noise grouped by scaffold family and labels clipped at an assay limit. With it we split the variance in performance and in robustness between model family, representation and their pairing. We found no earlier work that compared these for noise robustness (Chapter~\ref{chap:intro}).
```

### E17 — Ch8 contributions: closing paragraph naming the antibody representations and the feature-space finding
- file: thesis.tex
- kind: new-paragraph
- why: TAKEAWAYS.md section 6 items 7 and 11, which the section does not name; keeps the three lists (abstract, Chapter 1, Chapter 8) the same.
- todo_line: % TODO: manual review
OLD:
```latex
Just as new is the way they were tested. Every method was run on the single representation and the hybrid side by side, under seven kinds of damage, with QM9 as a control where the labels are clean. Cleanups on the labels as delivered were compared with the same cleanup driven by shuffled detector scores, and deletion with random deletion of the same size. This is what lets the thesis say that a gain is noise removed, and not just a model that got lucky.
```
NEW:
```latex
Just as new is the way they were tested. Every method was run on the single representation and the hybrid side by side, under seven kinds of damage, with QM9 as a control where the labels are clean. Cleanups on the labels as delivered were compared with the same cleanup driven by shuffled detector scores, and deletion with random deletion of the same size. This is what lets the thesis say that a gain is noise removed, and not just a model that got lucky.

% TODO: manual review
Two more contributions complete the list. The antibody representations built for this thesis, readouts of the complementarity-determining regions and of each region, mutation features and the change in embedding from the parent, are described in Section~\ref{sec:discussion:antibodies}. And the ranking of noise-handling methods depends on the feature space they run in (Section~\ref{sec:discussion:chain}), a finding that needs the single representation and the hybrid run side by side.
```

### E18 — Ch8 conditional section: opening paragraph naming the four conditions
- file: thesis.tex
- kind: new-paragraph
- why: TAKEAWAYS.md section 4: "Noise was handled. The conditions say where, not whether." The section gives the evidence but never states the conditions as a list; numbers 0.1957 and 0.2369 are from Chapter 7 (Section sec:noise:res:detectors).
- todo_line: % TODO: manual review
OLD:
```latex
\section{No Single Method Fits Every Dataset}\label{sec:discussion:conditional}
```
NEW:
```latex
\section{No Single Method Fits Every Dataset}\label{sec:discussion:conditional}

% TODO: manual review
% Numbers from Chapter 7, Section sec:noise:res:detectors (LogD, clipped share 0.25, average precision).
Noise was handled in this thesis, and the conditions say where, not whether. Four things decided which method worked: the type of noise, the data, the model and the task. Under censoring the forest residual fell below chance as a detector, 0.1957 average precision on LogD against a chance level of 0.2369. There a flag from the assay limit does better than a detector. A method that assumes damage that is not there does harm (Section~\ref{sec:discussion:noise}). The feature space matters as well. The noise method that does best on the hybrid is rarely the one that does best on both single representations (Section~\ref{sec:discussion:chain}). The paragraphs below give the evidence for the dataset, the model and the type of noise.
```

### E19 — Ch8 Conclusions: noise handled under named conditions; hybrid forward compatible
- file: thesis.tex
- kind: ai-text
- why: TAKEAWAYS.md sections 1 and 5: the conclusions should say which conditions decide the method and that the hybrid takes in new representations; the rest of both paragraphs kept.
- todo_line: keep existing
OLD:
```latex
Noise mitigation in molecular data is hard, and for most of this project it did not work. It works now. It works best when the data say which labels are wrong, as an assay limit or a scaffold family does. It works with a detector when the detector is precise enough for the damage at hand. It works on the labels as delivered too, on hERG $K_i$ and on three antibody datasets, against a control that rules out chance.

% TODO: manual review
Fixing the feature set was the right place to start. I spent a year on what looked like a separate problem, on the hypothesis that the representation was the root of the trouble with similarity and with noise. On small molecules the hybrid predicts better than any single representation, gives better neighbors, and keeps its lead through every kind of cleaning. It carried over to antibodies without redesign, and did best where the representations were built to see what varies. No single method fits every dataset, and the hybrid does not pretend to be one. It is a procedure that adapts the features to the data, and it adapts further when the labels are cleaned before it.
```
NEW:
```latex
Noise mitigation in molecular data is hard, and for most of this project it did not work. It works now, under conditions this thesis can name. It works best when the data say which labels are wrong, as an assay limit or a scaffold family does. It works with a detector when the detector is precise enough for the damage at hand. It works on the labels as delivered too, on hERG $K_i$ and on three antibody datasets, against a control that rules out chance. Which method to use depends on the task, the data, the model and the type of noise, so the first step is to ask where the labels came from.

% TODO: manual review
Fixing the feature set was the right place to start. I spent a year on what looked like a separate problem, on the hypothesis that the representation was the root of the trouble with similarity and with noise. On small molecules the hybrid predicts better than any single representation, gives better neighbors, and keeps its lead through every kind of cleaning. It carried over to antibodies without redesign, and did best where the representations were built to see what varies. No single method fits every dataset, and the hybrid does not pretend to be one. It is a procedure that adapts the features to the data. It adapts further when the labels are cleaned before it, and it takes in each new representation as it appears.
```

## For the author (not changed)
- Chapter 1, paragraph 2: "Repeat measurements of the same compound-target pair disagree by about $0.68$ log units for pIC$_{50}$ and $0.54$ log units for pK$_i$" -- chemistry guide item 7: elsewhere (Chapter 4) 0.44 is the mean difference and 0.54 the standard deviation for Kramer 2012; check the source before changing (not checked here).
- Chapter 1, paragraph 1: "QSAR models aim to predict the relationship between a molecule's structural and chemical features as encoded by a particular molecular representation and a desired experimental molecular property \citep{Sabando2021} of interest." -- the citation sits mid-phrase; style only, not changed.
- Chapter 1 NOTES block, Paragraph 3: "Say plainly that this study comes from the confirmation report." -- conflicts with the standing rule (say "an exploratory study carried out earlier in this project"); the plan follows the rule. The NOTES block is not edited.
- Chapter 1 NOTES block, Paragraph 2: "Adapted from the paper ... Say this in the paragraph." -- the plan does not mention the Chapter 3 paper (standing rule); decide whether Chapter 3's paragraph should name it and in what words.
- Chapter 5 Conclusions: "The fourth question asked whether weighting features by their importance improves the hybrid and model training." -- still matches Chapter 1, since E4 keeps the importance-weighting question as hybrid question 4.
- Chapter 5 Conclusions: "This chapter asked how the best set of features can be selected for a given prediction task (Section~\ref{sec:intro:hybrid_questions})." -- still matches hybrid question 1, kept word for word.
- Chapter 6 Introduction asks its own two numbered questions ("Does distance in a hybrid track activity better than distance in the best single representation?") -- they now map onto Chapter 1's hybrid question 2; "activity" there covers non-activity labels (chemistry guide item 17).
- "scaffold family" (used in the new text, as Chapter 8 uses it) against "scaffold group" elsewhere -- chemistry guide item 14 recommends one term; a thesis-wide choice, not made here.
- Chapter 8, Summary section title: with the restructure, a title such as "Answers to the Research Questions" would fit; left to the titles task.
- Chapter 8: "Nobody asked me to look at the features." and "I spent a year on what looked like a separate problem" -- kept as the two personal sentences (style guide item 7 allows this deliberately); decide whether to keep both.
- TAKEAWAYS.md "For the author": Chapter 3's Conclusions say RF ranks "between first and eighth" and Chapter 3's results say "first and seventh" (Gaussian only). Not used in the new text; check.
- main.tex abstract was not empty as the brief said: it held a 2026-10-09 draft of about 300 words (kept comment line, text replaced in E1).
- Chapter 1 Collaborations: "Lhasa also supplied the hERG data used in Chapters~\ref{chap:flaws} and~\ref{chap:method}" -- hERG classification results also appear in Chapter 7 (label flips on hERG); check whether Chapter 7 belongs in that list.
- Summary of Research Contributions does not list NoiseInject or KIRBy as software (style guide item 13); out of this plan's scope.
- Chapter 1 outline, Chapter 8 paragraph: "Chapter~\ref{chap:discussion} summarizes each chapter and argues that the noise in a dataset should be modeled before a method is chosen" -- not wrong, so not changed (the plan's E10 rewrite was deleted). Update it yourself if Chapter 8 is restructured as E13 and E14 propose.
- Chapter 8, "The Central Claim" (marked as your words): "It is that no single method can work, so we should change the data" -- stronger than the evidence (style guide item 9); the thesis shows that no single method worked on every dataset tested. Not changed because it is your claim.
- Chapter 1 outline, Chapter 4 paragraph: "Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular similarity." -- Chapter 8 argues "None of them was a change of topic." Kept as your phrasing; decide which framing you want.
- Chapter 1 outline, Chapter 3 paragraph: "The models with the highest predictive performance, LightGBM, XGBoost and the plain neural networks lost the most" -- Chapter 3's Conclusions name LightGBM and XGBoost as losing the most, and say the plain networks depended on the representation. Check the plain networks.
- Chapter 1, first-half questions: "does that division change when you instead consider" -- second person in the thesis body; not changed.
- Chapter 1 Collaborations: Chapter 3 (robustness) is not in the "author's own work" range, presumably because of the co-authored paper. Check that this is intended.

## Checker notes
Checked 2026-10-10 against thesis.tex and main.tex as they are now. Every OLD block was re-tested with python3 -I str.count and is an exact, unique substring (count 1). No number in any NEW block is absent from thesis.tex. No banned word was found in new text.
- E1 (abstract, ai-text): fixed. "the representation acted mostly through the model it was paired with" is not what Chapter 3 says; it now says the pairing explained more than the representation alone. "Probabilistic models did not help, and their uncertainty did not point to the corrupted labels" was too broad; Chapter 3 says the Bayesian networks never predicted better, and that uncertainty failed to find corrupted labels only outside censoring.
- E2: kept.
- E3: fixed. The rewrite of her last clause ("and address this problem with new molecular representations") was a style change, so her wording is restored; only I/my to we/our, the spelling and the broad term remain.
- E4: fixed. The plan deleted her four first-half questions, which carry no TODO and are not wrong, and merged everything into a list of seven. Her four questions are now kept unchanged and are not in the edit. Only the bridge paragraph (her TODO "come back to this") and the numbered list (her TODO "definitely modify these") are rewritten. Her list items 1 and 4 are kept word for word in their positions, so Chapter 5's Conclusions ("The fourth question asked whether weighting features...") still matches. The first new bridge sentence was split to stay under 30 words.
- E5: kept.
- E6: fixed. The plan reworded most of her sentences for style ("looks into" to "asks", "primary" to "main", reordering). Now only the tense slips, the broken uncertainty sentence and "made-up" are changed. The plan's added sentence "Even so, no model's uncertainty picked out which labels were corrupted" was dropped: Chapter 3 says this holds only outside censoring.
- E7: fixed. Removing "takes a turn to look at another topic" was taste (it is her framing), as was renaming the sweep and adding a sentence. Now only the factual sentence "None of the methods ... showed any promising results" is changed, to "did worst", which is what Chapter 4 says.
- E8: fixed. Merging her two paragraphs was taste, so it was undone and the TODO lines stay between them. Spelling fixes in the first paragraph need no comment; the "edited your text" line now sits above the second paragraph only. "On antibodies the gains depended on the dataset, and an ensemble often did better" narrowed to the smaller antibody datasets, as Chapter 5's Conclusions say.
- E9: kept (her TODO "it's all out of date"; claims checked against Chapters 6 to 8).
- E10: deleted. Her Chapter 8 outline paragraph is not wrong; the rewrite was an update for taste. Listed under For the author.
- E11: fixed. Kind changed from author-text to new-paragraph, since her sentence is kept unchanged; the list is now inserted after it with "% TODO: manual review". The NoiseInject item was dropped from the list because her sentence already names the software. The garbled item "Hybrids built for distance (...), for data held at separate sites and open to new embeddings (...)" was rewritten.
- E12: kept.
- E13: fixed. "the seven questions" changed to "the research questions", since E4 no longer makes seven.
- E14: fixed. Restructured to follow Chapter 1 as E4 now leaves it: her four first-half questions, the Chapter 4 paragraph, then the four hybrid questions, with an answer added for importance weighting from Chapter 5's Conclusions. Three carried-over sentences over 30 words were split. "Bayesian networks kept a slightly larger share, but only because they started lower" was replaced with what Chapter 3 says (raised robustness, never predicted better). "Outside censoring" added to the uncertainty answer, as in Chapter 3.
- E15: fixed. The paragraph is labeled in a comment as "the author's words", so it is treated as her text. The activity-cliff fix is a chemistry error and stays, now with an "edited your text" TODO line. The change of her central claim "no single method can work" was reverted and moved to For the author.
- E16, E17: kept.
- E18: fixed. The ranking of conditions ("comes first ... second ... third ... fourth") is not in the thesis and was removed. The link from "task" to detectors that never read a label was not supported and was removed. The feature-space sentence now copies Chapter 8's own wording.
- E19: kept.
