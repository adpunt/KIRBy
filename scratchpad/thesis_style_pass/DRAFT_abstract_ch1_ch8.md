# Draft: abstract, Chapter 1 and Chapter 8 (2026-10-10)

Drafts only; nothing in main.tex or thesis.tex was edited. Every OLD block below was checked with `python3 -I` `str.count(old) == 1` against the file as it stood when this draft was written. Other sessions are editing thesis.tex, so re-check each OLD block before applying.

## Reference measurements (python3 -I on the PDF text files; words = alphabetic tokens, so running heads and page numbers add a few)

| | Buttenschoen 2025 | Dablander 2023 | Vost 2025 |
|---|---|---|---|
| Abstract, words | 398 | 250 | 568 |
| Abstract, paragraphs | 6 | 2 | 5 |
| Abstract, sentences (mean / longest words) | 13 (30.4 / 52) | 10 (25.1 / 46) | 27 (21.0 / 41) |
| Introduction chapter, words | 4,218 (Ch1, 5 sections) | 2,158 (Ch1, one section "Outline of Thesis and Research Contributions") | 9,386 (Ch1, includes the background chapter's job) |
| Outline part of the introduction, words | 148 ("1.5 Structure of the thesis") | 1,061 ("1.1 Outline of Thesis and Research Contributions", one block per chapter with findings and novelty) | 438 ("1.5 Thesis outline") |
| Final chapter, words | 1,593 (Ch6 "Conclusion") | 896 (Ch7 "Conclusions and Further Thoughts"; plus Ch6 "Future Directions", 1,788) | 1,015 (Ch6 "Conclusions and outlook") |
| Final chapter section titles | 6.1 Key findings and contributions; 6.2 Limitations and recommendations for future research; 6.3 Closure | none (unsectioned; recap "(i) ... (ii) ... (iii)" then further thoughts) | 6.1 Interpretability in virtual screening; 6.2 Plausibility in diffusion-based generative design; 6.3 AlphaFold for cryo-EM model building; 6.4 Concluding remarks |

Line ranges used: Buttenschoen abstract 20-52, Ch1 275-712, Ch6 3607-3773; Dablander abstract 82-111, Ch1 285-509, Ch6 5374-5623, Ch7 5623-5715; Vost abstract 17-71, Ch1 612-1823, Ch6 4933-5066.

## New lengths (prose words, comment lines removed)

| | Before | After |
|---|---|---|
| Abstract (main.tex) | 338 | about 450 (six paragraphs) |
| Chapter 1 | 1,956 | 2,211 |
| Chapter 8 | 4,443 | about 1,700 |

Chapter 1 grows because the outline gains a Chapter 7 paragraph, findings for Chapter 5 and a short contributions paragraph, and her own paragraphs were not shortened (her rule: change her text only where it is wrong). At about 2,200 words it matches Dablander's introduction (2,158). Chapter 8 now sits between Vost (1,015) and Buttenschoen (1,593) plus the two result placeholders.

## How the three parts line up

The abstract, Chapter 1's outline and Chapter 8 tell one story in one order, with the same terms:
1. Label noise is not Gaussian; model family decides robustness more than representation; random forests most consistent; uncertainty does not find the noisy labels (Chapter 3).
2. Published methods for learning with noisy labels failed; those using a distance failed worst, because most features of a standard representation carry nothing for the task (Chapter 4).
3. The adaptive hybrid representation selects features for each dataset and model; it predicts better and gives better neighbors (Chapters 5 and 6).
4. Label noise was handled; which method works depends on the task, data, model and noise type; the hybrid keeps its lead; method rankings change with the feature space (Chapter 7).
5. Recommendation: choose a noise method by how the labels were made, and adapt the features to each dataset.

Chapter 8's "Summary of findings" answers her four first-half questions (kept word for word in Chapter 1) and then the four second-half questions of D4, in that order. Her first-half questions stay as prose, not a list, because changing them would change her text.

## For the author (not changed in this draft)

- Originality: thesis_provenance/LITERATURE_SEARCH_2026-10-08.md already has a 2026-10-10 entry for all four methods. It says the gap over neighbor spread is the same statistic as the meteorological "buddy check" (published outside machine learning and chemistry). The draft therefore claims "no earlier use on molecular data", not "no published source", for all four, with the pending-check TODO above that item.
- Chapter 6 Conclusions says "These losses fit the central claim of this thesis". Chapter 8 no longer has a section called the central claim; its content is in "Concluding remarks" (label sec:discussion:claim kept there).
- Chapter 1 outline, Chapter 3 paragraph: "LightGBM, XGBoost and the plain neural networks lost the most". Chapter 3's Conclusions name only LightGBM and XGBoost, and say the plain networks depended on the representation. Not changed; check.
- Chapter 1 Collaborations: Chapter 3 is outside the "author's own work" range, and Lhasa's hERG data is listed for Chapters 4 and 5 only; Chapter 7's label flips run on hERG. Check both.
- Chapter 1 NOTES block (comments): Paragraph 3 says "Say plainly that this study comes from the confirmation report", and Paragraph 2 asks to name the Chapter 3 paper. Both conflict with the standing rules; the draft follows the rules. The NOTES block is not edited.
- Chapter 8's chapter title "Discussion and Conclusions" is kept. The references use "Conclusion", "Conclusions and Further Thoughts" and "Conclusions and outlook"; "Conclusions and Outlook" would match the new contents. Not changed.
- Old Chapter 8 "% TODO: manual review" markers are not carried over one for one; every new paragraph has its own. Every other comment of the old chapter is carried (checked by script).
- Chapter 8 Future work no longer names The Cancer Genome Atlas; it says "gene-expression profiles" only.
- TAKEAWAYS "For the author": Chapter 3 says RF ranks "between first and eighth" in its Conclusions; the draft copies that.

## Changes

### D1 — Abstract replaced
- file: main.tex
- kind: ai-text
- why: The author asked for the abstract to be redone on the reference abstracts. Six paragraphs in chapter order: problem; noise robustness (Ch3); why LNL methods failed (Ch4); the hybrid (Ch5, Ch6); noise handled, conditionally (Ch7); closing recommendation.
- todo_line: existing TODO line kept; a second "% TODO: manual review -- [CLAUDE 2026-10-10] ..." line added
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
% TODO: manual review -- [CLAUDE 2026-10-10] rewritten from scratch on the pattern of the reference abstracts (Vost's problem-first opening; Buttenschoen's "This thesis investigates ... First, ... Next, ... Last, ..." order, in "we"). Six paragraphs in chapter order, about 430 words, no results numbers. Same story and terms as Chapter 1's outline and Chapter 8.
Machine learning models of molecular properties are trained on experimental labels, and those labels carry noise from measurement error, differences between laboratories and the limits of an assay. Most studies of label noise in quantitative structure--activity relationship (QSAR) modeling assume the noise is Gaussian, and evaluate either representations or model architectures in isolation. This thesis investigates what label noise does to QSAR models, and how it can be handled.

First, we introduced NoiseInject, an open-source Python package for benchmarking noise robustness, and compared 13 models and six molecular representations under seven kinds of label noise. Model family decided noise robustness more than the representation did, and random forests were the most consistent model. The shape of the noise distribution barely mattered. Systematic noise did the most damage, such as an offset shared within a scaffold family or labels clipped at an assay limit. Predicted uncertainty rose with noise, but outside censoring it did not point to the corrupted labels.

In an exploratory study, published methods for learning with noisy labels did poorly on molecular data, and those that compared molecules by a distance did worst. Next, we traced this failure to the representation. On a single dataset, most features of a standard representation do not vary or do not relate to the label, yet each adds to the distance between two molecules.

We then built the adaptive hybrid representation, which selects features from several representations under a fixed budget and searches the selection settings for each dataset and model. On every small-molecule dataset it predicted better than the best single representation, and a hybrid built for distance gave better nearest neighbors. The search also runs on antibodies, for which we built new representations, and on reactions. A newly published embedding is added as one more representation to search.

Last, we returned to label noise, testing noise detection, noise mitigation and noise-robust training on single representations and hybrids side by side, against random and shuffled controls. Label noise was handled, but which method worked depended on the task, the data, the model and the type of noise. Corrections worked best when the data flagged the wrong labels, as an assay limit or a scaffold family does. On small molecules the hybrid kept its lead through every cleaning, and the ranking of noise methods changed with the feature space.

Together, these studies show that label noise in molecular data is not Gaussian, and that the features a model is given limit how well the noise can be handled. No single model, representation or noise method was best on every dataset we tested. We therefore recommend choosing a noise method by how the labels were made, and adapting the features to each dataset.
```

### D2 — Ch1 paragraph 2: comma
- file: thesis.tex
- kind: author-text (trivial)
- why: Misplaced comma.
- todo_line: none (trivial grammar)
OLD:
```latex
QSAR models are of course, sensitive
```
NEW:
```latex
QSAR models are, of course, sensitive
```

### D3 — Ch1 paragraph 3: "I" to "we", spelling, broad term
- file: thesis.tex
- kind: author-text
- why: Wrong person ("I" for the work), spelling, and a term that names a subset (mitigation) where Chapter 4 tests the broad set.
- todo_line: % TODO: manual review -- edited your text: "I" and "my" changed to "we" and "our" in the last two sentences (the thesis uses "we" outside personal motivation); "cheminformatcs" corrected; "noise mitigation techniques" changed to "methods for learning with noisy labels", since Chapter 4 tests the broad set (LNL), of which noise mitigation is one group.
OLD:
```latex
This thesis addresses the challenge of conducting molecular property prediction when the underlying experimental data, or ``labels'' are noisy. Most work on noise in QSAR modeling assumes Gaussian label noise and evaluates either representations or model architectures in isolation. \citet{Cortes2015} examined 12 algorithms across multiple noise levels but used fixed representations, and studies that compare representations typically use clean data. No work has systematically compared the relative importance of representation versus architecture choice for label noise robustness, asked whether probabilistic methods justify their computational overhead under noise, or tested whether findings generalize across different types of injected noise. This thesis fills that gap. I will also discuss noise mitigation, using the results of my noise robustness study. I explore why many noise mitigation techniques underperform in cheminformatcs data and methods to address this problem in the form of novel molecular representations.
```
NEW:
```latex
% TODO: manual review -- edited your text: "I" and "my" changed to "we" and "our" in the last two sentences (the thesis uses "we" outside personal motivation); "cheminformatcs" corrected; "noise mitigation techniques" changed to "methods for learning with noisy labels", since Chapter 4 tests the broad set (LNL), of which noise mitigation is one group.
This thesis addresses the challenge of conducting molecular property prediction when the underlying experimental data, or ``labels'' are noisy. Most work on noise in QSAR modeling assumes Gaussian label noise and evaluates either representations or model architectures in isolation. \citet{Cortes2015} examined 12 algorithms across multiple noise levels but used fixed representations, and studies that compare representations typically use clean data. No work has systematically compared the relative importance of representation versus architecture choice for label noise robustness, asked whether probabilistic methods justify their computational overhead under noise, or tested whether findings generalize across different types of injected noise. This thesis fills that gap. We also discuss noise mitigation, using the results of our noise robustness study. We explore why many methods for learning with noisy labels underperform in cheminformatics data and methods to address this problem in the form of novel molecular representations.
```

### D4 — Ch1 bridge paragraph and second-half questions
- file: thesis.tex
- kind: author-text (both carry her TODOs)
- why: Her TODOs "come back to this ... missing a lot" and "definitely modify these". "this distance" referred to nothing. The old list asked questions Chapters 6 and 7 no longer answer as asked.
- todo_line: her two TODO lines kept; one "% TODO: manual review -- edited your text: ..." line under each
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
% TODO: manual review -- edited your text: as your TODO asked. Two sentences added at the start to say which failure and which distance, since "this distance" had nothing before it to refer to; one sentence added at the end pointing to Chapters 6 and 7, which the paragraph left out. Your four sentences in the middle are unchanged.
The second half of this thesis grew out of a failure. In an exploratory study carried out earlier in this project, published methods for learning with noisy labels did poorly on QSAR data. Those that compared molecules by a distance did worst (Chapter~\ref{chap:flaws}). We hypothesized that the fault lay in this distance. On a given dataset, most features of a general-purpose representation either do not vary or do not relate to the property, yet each adds a term to the distance, so together they dominate it (Chapter~\ref{chap:flaws}). We propose instead to compute distances over a hybrid representation: a fixed number of task-relevant features selected from several representations. A first version showed that the selected features also work as ordinary model inputs, so the question became how to choose them (Chapter~\ref{chap:method}). We then use the hybrid as a space for similarity (Chapter~\ref{chap:similarity}), and return to label noise with the hybrid and the single representations side by side (Chapter~\ref{chap:noise}).

The second half asks whether the representation itself can be adapted to the task:\label{sec:intro:hybrid_questions}
% TODO: definitely modify these
% TODO: manual review -- edited your text: as your TODO asked. Questions 1 and 4 kept word for word and in the same places, because Chapter 5's Conclusions answers "the fourth question" by number. Question 2 now asks Chapter 6's question on distance. Question 3 now asks Chapter 7's question and takes in your old questions 2, 3 and 5 (noise robustness of the hybrid, distance-based noise mitigation, sample weighting and label smoothing). Chapter 8 answers your four first-half questions and these four, in this order.
\begin{enumerate}
    \item How can the best set of features be selected for a given prediction task?
    \item Does distance in a hybrid representation track the label better than distance in the best single representation, and do methods built on nearest neighbors work better in it?
    \item Can noisy labels be found and corrected, by deletion, sample weighting, label smoothing or other means, and do hybrid representations keep their lead through that cleaning?
    \item Does weighting features by their importance improve the construction of hybrid representations, model training and noise mitigation?
\end{enumerate}
```

### D5 — Ch1 outline, Chapter 2 paragraph: spelling
- file: thesis.tex
- kind: author-text (trivial)
- why: Spelling and a stray comma.
- todo_line: none (trivial spelling)
OLD:
```latex
and closes with artifical noise injection, and evaluation metrics.
```
NEW:
```latex
and closes with artificial noise injection and evaluation metrics.
```

### D6 — Ch1 outline, Chapter 3 paragraph: tense, broken sentence, model list
- file: thesis.tex
- kind: author-text
- why: Tense slips; a broken sentence; "only models with a variational head" disagrees with Chapter 3's Conclusions; "made-up" is informal.
- todo_line: see first line of NEW
OLD:
```latex
Chapter~\ref{chap:robustness} looks into what makes a QSAR model robust to label noise. The primary prediction target is the HOMO--LUMO gap in QM9, whose computed labels carry negligible measurement noise. The same experiments were also run on three assay datasets: LogD, Caco-2 and hERG $K_i$. We compare seven different types of label noise, referred to as noise conditions. We matched the quantity of noise to the training set label spread for all but one of the noise conditions. We measure noise robustness as the proportion of predictive performance retained as artificial label noise increases. We found that model family was the main factor in noise robustness, and choice of molecular representation matters more for predictive performance than noise robustness. Strong predictive performance on clean labels does not imply robustness. The models with the highest predictive performance, LightGBM, XGBoost and the plain neural networks lost the most with increasing label noise, while random forests were the most consistent model across datasets, representations, and noise conditions. We replicated systematic noise by adding it in disproportionate amounts to select scaffold families and by clipping labels at a made-up assay limit. Models struggled the most when training on this systematic noise. Label noise mean predicted uncertainty for the probabilistic models which estimate it, however, only models with a variational head correctly attributed the rise in uncertainty to the data rather than to the model itself. 
```
NEW:
```latex
% TODO: manual review -- edited your text: tense made consistent (present for what the chapter does, past for what it found): "were also run" to "are also run", "We matched" to "We match", "matters" to "mattered", "does not imply" to "did not imply", "We replicated" to "We imitate". The broken sentence "Label noise mean predicted uncertainty ..." split in two, and "only models with a variational head" changed to the models Chapter 3's Conclusions names (BNNs with a variance head, the heteroscedastic GP and the heteroscedastic VBLL networks), with "in uncertainty" dropped to keep the sentence under 30 words. "a made-up assay limit" changed to "a fixed value, in imitation of an assay limit".
Chapter~\ref{chap:robustness} looks into what makes a QSAR model robust to label noise. The primary prediction target is the HOMO--LUMO gap in QM9, whose computed labels carry negligible measurement noise. The same experiments are also run on three assay datasets: LogD, Caco-2 and hERG $K_i$. We compare seven different types of label noise, referred to as noise conditions. We match the quantity of noise to the training set label spread for all but one of the noise conditions. We measure noise robustness as the proportion of predictive performance retained as artificial label noise increases. We found that model family was the main factor in noise robustness, and choice of molecular representation mattered more for predictive performance than noise robustness. Strong predictive performance on clean labels did not imply robustness. The models with the highest predictive performance, LightGBM, XGBoost and the plain neural networks lost the most with increasing label noise, while random forests were the most consistent model across datasets, representations, and noise conditions. We imitate systematic noise by adding it in disproportionate amounts to select scaffold families and by clipping labels at a fixed value, in imitation of an assay limit. Models struggled the most when training on this systematic noise. Mean predicted uncertainty rose with label noise for the probabilistic models which estimate it. However, only the BNNs with a variance head, the heteroscedastic GP and the heteroscedastic VBLL networks correctly attributed the rise to the data rather than to the model itself.
```

### D7 — Ch1 outline, Chapter 4 paragraph: one factual sentence
- file: thesis.tex
- kind: author-text
- why: Contradicted by Chapter 4 (one neighbor-based configuration gave the best recovery at the higher noise level).
- todo_line: % TODO: manual review -- edited your text: "None of the methods that incorporated the concept of molecular similarity showed any promising results" changed to "did worst", since Chapter 4's Conclusions reports one neighbor-based configuration as the best at the higher noise level.
OLD:
```latex
Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular similarity. The natural next step to Chapter~\ref{chap:robustness} is noise mitigation. However, a small sweep of twelve published noise-handling methods, four that remove molecules and retrain and eight that change the training, showed disappointing results. None of the methods that incorporated the concept of molecular similarity showed any promising results. On a given dataset, standard representations carry many features that do not vary, and some map different molecules to the same vector. Similarity computed over these representations inherits both of these problems, and methods that rely on similarity suffer as a result. The chapter ends with what a representation would need to mitigate these problems, which Chapter~\ref{chap:method} builds.
```
NEW:
```latex
% TODO: manual review -- edited your text: "None of the methods that incorporated the concept of molecular similarity showed any promising results" changed to "did worst", since Chapter 4's Conclusions reports one neighbor-based configuration as the best at the higher noise level.
Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular similarity. The natural next step to Chapter~\ref{chap:robustness} is noise mitigation. However, a small sweep of twelve published noise-handling methods, four that remove molecules and retrain and eight that change the training, showed disappointing results. The configurations that incorporated the concept of molecular similarity did worst. On a given dataset, standard representations carry many features that do not vary, and some map different molecules to the same vector. Similarity computed over these representations inherits both of these problems, and methods that rely on similarity suffer as a result. The chapter ends with what a representation would need to mitigate these problems, which Chapter~\ref{chap:method} builds.
```

### D8 — Ch1 outline, Chapter 5 paragraphs: spelling, missing word, findings
- file: thesis.tex
- kind: author-text (second paragraph carries her TODO "waiting on results")
- why: Her TODO "waiting on results, may need another pass"; missing "with"; two spelling slips.
- todo_line: her TODO kept; "% TODO: manual review -- edited your text: ..." added under it
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
% TODO: manual review -- edited your text: as your TODO above asked. "with" added to "The hybrid is compared the base representations"; two sentences of findings added at the end, from Chapter 5's Conclusions. ("customises" and "genereate" in the paragraph above were spelling slips, fixed.)
The hybrid is compared with the base representations on the same data, an ensemble of one model per representation, principal component analysis, minimum-redundancy maximum-relevance selection, recursive feature elimination and random features of the same size. It also investigates properties of the hybrid representation, including whether the selected features are interpretable and stable, whether search settings transfer between datasets, and whether the method extends to other domains and to data held at separate sites. On every small-molecule dataset, the hybrid predicted better than the best single representation with both downstream models. On antibodies the gains depended on the dataset, and on the smaller sets an ensemble of one model per representation often did better.
```

### D9 — Ch1 outline, Chapter 6 paragraph brought up to date; Chapter 7 paragraph added
- file: thesis.tex
- kind: author-text (her TODO "it's all out of date") + new
- why: Her TODO; the outline had no Chapter 7 paragraph and the Chapter 6 paragraph described work now in Chapter 7.
- todo_line: her TODO kept; edited-your-text line; "% TODO: manual review" above the new Chapter 7 paragraph
OLD:
```latex
% TODO: come back to all of this, it's all out of date
Chapter~\ref{chap:similarity} asks whether a hybrid representation repairs the failures described in Chapter~\ref{chap:flaws}. It compares standard and hybrid representations selected for predicting each molecule's activity from its five nearest neighbors, asks whether neighbors in a hybrid representation identify corrupted labels better than neighbors in a standard one, and tests noise mitigation and noise-robust training methods one noise type at a time. Its results are also preliminary. On hERG $K_i$, predicting each molecule's activity from its five nearest neighbors was more accurate in a hybrid than in the best single representation, and, on ECFP4, correcting labels clipped at a limit by one shared amount recovered more than half of the lost performance, whereas deleting the clipped molecules gave a higher error than keeping them. Under Gaussian noise on nanobody thermostability, every one of twelve published noise-handling methods scored higher on the hybrid than any of them did on one-hot, the weakest input.
```
NEW:
```latex
% TODO: come back to all of this, it's all out of date
% TODO: manual review -- edited your text: as your TODO asked. The noise part moved to a new Chapter 7 paragraph below; "Its results are also preliminary" and the preliminary hERG $K_i$, ECFP4 and nanobody results replaced by Chapter 6's findings; "activity" changed to "label", since QM9, LogD and Caco-2 labels are not activities.
Chapter~\ref{chap:similarity} asks whether a hybrid representation repairs the failures of similarity described in Chapter~\ref{chap:flaws}. It builds hybrids for predicting each molecule's label from its five nearest neighbors, and compares distances and neighbors in them with those in standard representations. On all five small-molecule datasets, distance in the hybrid tracked the label more closely and the neighbors predicted better. On the antibody and reaction datasets distance also tracked the label more closely, but prediction from neighbors was worse in the hybrid on several of them.

% TODO: manual review
Chapter~\ref{chap:noise} returns to label noise, with the single representations and the hybrid side by side. It tests noise detection, noise mitigation and noise-robust training on labels damaged on purpose and on labels as delivered. QM9 with no noise added, random deletion and shuffled detector scores serve as controls. Label noise was handled, but which method worked depended on the type of noise, the data and the model. Corrections worked best when a known flag, such as an assay limit or a scaffold family, said which labels were wrong. The hybrid stayed ahead of the single representation on small molecules through every treatment, and the ranking of noise methods changed with the feature space.
```

### D10 — Ch1 outline, Chapter 8 paragraph
- file: thesis.tex
- kind: author-text
- why: Becomes wrong once Chapter 8 is rebuilt (D-Ch8): the chapter no longer summarizes each chapter.
- todo_line: see first line of NEW
OLD:
```latex
Chapter~\ref{chap:discussion} summarizes each chapter and argues that the noise in a dataset should be modeled before a method is chosen, since no single method fits every dataset and simple methods are often as good as complex ones. It then gives limitations, future directions and conclusions.
```
NEW:
```latex
% TODO: manual review -- edited your text: "summarizes each chapter and argues" changed to "answers each research question in turn and lists the contributions. It argues", because Chapter 8 was rebuilt on 2026-10-10 around answers to the research questions and a contributions list. The rest is unchanged.
Chapter~\ref{chap:discussion} answers each research question in turn and lists the contributions. It argues that the noise in a dataset should be modeled before a method is chosen, since no single method fits every dataset and simple methods are often as good as complex ones. It then gives limitations, future directions and conclusions.
```

### D11 — Ch1 contributions named
- file: thesis.tex
- kind: new
- why: The section is titled "Outline of the thesis and contributions" but named only the software. Same order and terms as Chapter 8's Contributions section.
- todo_line: % TODO: manual review
OLD:
```latex
The software contributions are NoiseInject and the KIRBy repository, which implements the search for hybrid representations and the experiments (Additional file~\ref{app:availability}).
```
NEW:
```latex
The software contributions are NoiseInject and the KIRBy repository, which implements the search for hybrid representations and the experiments (Additional file~\ref{app:availability}).

% TODO: manual review
On label noise, the main contributions are NoiseInject and seven noise conditions compared at a matched amount of noise. They also include a split of the variance in robustness between model and representation, and noise-handling methods tested against controls. On the representation, the main contribution is the adaptive hybrid representation, built for prediction and for distance, with antibody representations built for it. Section~\ref{sec:discussion:contributions} lists the contributions in full.
```

### D12 — Ch1 Collaborations: antibody chapters
- file: thesis.tex
- kind: author-text (section carries her TODO "review for final checks")
- why: Chapter 7 uses the antibody datasets too.
- todo_line: see first line of NEW
OLD:
```latex
The Oxford Protein Informatics Group (OPIG) pointed us to the antibody and protein datasets used in Chapters~\ref{chap:method} and~\ref{chap:similarity}; all datasets are listed in Additional file~\ref{app:datasets}.
```
NEW:
```latex
% TODO: manual review -- edited your text: "Chapters 5 and 6" changed to "Chapters 5 to 7", since Chapter 7 also uses the antibody datasets (SMAC colloidal stability, binding free energy, expression titer).
The Oxford Protein Informatics Group (OPIG) pointed us to the antibody and protein datasets used in Chapters~\ref{chap:method} to~\ref{chap:noise}; all datasets are listed in Additional file~\ref{app:datasets}.
```

### D13 — Ch1 Collaborations: author's own work range
- file: thesis.tex
- kind: author-text (section carries her TODO "review for final checks")
- why: The range left out Chapter 7, which is also her work.
- todo_line: see first line of NEW
OLD:
```latex
The adaptive hybrid representation, its software and the experiments in Chapters~\ref{chap:flaws} to~\ref{chap:similarity} are the author's own work.
```
NEW:
```latex
% TODO: manual review -- edited your text: the range now ends at Chapter 7 (label noise) instead of Chapter 6, which left Chapter 7 out.
The adaptive hybrid representation, its software and the experiments in Chapters~\ref{chap:flaws} to~\ref{chap:noise} are the author's own work.
```

### D14 — Chapter 8 rebuilt
- file: thesis.tex
- kind: ai-text (whole chapter; the decision comments at the top kept verbatim)
- why: The author called Chapter 8 a mess and too long; titles "The Central Claim", "What Is New in This Thesis", "The Antibodies" removed. Rebuilt on the reference final chapters: Summary of findings (answers each Chapter 1 question in order), Contributions (numbered, noise first), Limitations, Future work, Concluding remarks.
- todo_line: every new paragraph has "% TODO: manual review"; the originality TODO sits above item 6 of the noise contributions
OLD:
```latex
\chapter{Discussion and Conclusions}\label{chap:discussion}
% DECISIONS WITH THE AUTHOR, 2026-10-09:
% - The chain section (outline 8.2, with diagram D8.1) was removed by the author on 2026-10-10. Backup: thesis.tex.before_ch8_chain_removed_2026-10-10.
% - The antibodies: a method built for small molecules that generalizes to antibodies, because the representations were what held the methods back; the hand-built antibody representations (CDR and region readouts, mutation features, the change from the parent) named as hers. Not "unusually well".
% - Original contributions named: the hybrid, and the noise methods that are new or changed from their sources (METHOD_PROVENANCE.md; the author's path scratchpad/provenance/ does not exist, the file read is /Volumes/seagate/claude_analysis_2026-10-08/answers/_ref/METHOD_PROVENANCE.md).
% - Next steps: reactions (tested) and nucleotide sequences (results exist), then other domains, including finance and gene-expression data. No patient data was used in this thesis, and none is reported.
% Every number below is copied from Chapters 5 to 7 as written, with that chapter's SOURCE comment carried over, except the antibody permutation shares in Section sec:discussion:antibodies, which were re-read from the raw c22 files on 2026-10-09 (SOURCE given there).

% TODO: manual review
This thesis set out to handle label noise in molecular data. Nobody asked me to look at the features. The noise methods kept failing, and every time we asked why, the answer sat one level further down, until it reached the representation itself. This chapter brings the three result chapters together and argues that looking at the features was the right call. We first summarize each chapter (Section~\ref{sec:discussion:summary}), then state the claim the thesis argues (Section~\ref{sec:discussion:claim}). Next comes what is new in this work (Section~\ref{sec:discussion:contributions}) and what the noise results mean in practice (Section~\ref{sec:discussion:noise}). After that we discuss why the method carried over to antibodies (Section~\ref{sec:discussion:antibodies}), and when each method works (Sections~\ref{sec:discussion:conditional} and~\ref{sec:discussion:simple}). The chapter ends with the limitations, future directions and conclusions.


\section{Summary}\label{sec:discussion:summary}

% TODO: manual review
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


\section{The Central Claim}\label{sec:discussion:claim}

% TODO: manual review
% Outline 8.2a, the author's words (2026-10-08).
There is no single best model, representation, similarity measure, method for learning with noisy labels or noise mitigation method that fits every dataset. Everything depends on the data: the features, the true labels, and the noise those labels carry. Every dataset carries its own noise, as outliers and activity cliffs, and it is not predictable in advance. So the honest answer is not a winner. It is a broad sweep, which this thesis runs again and again, together with hybrid featurization, a procedure that builds a feature selection for each dataset. The hybrid is not a one-size-fits-all method. It adapts to the data it is given, including data whose corruption is unknown, and it adapts further when used alongside noise mitigation. The claim is not ``this method works''. It is that no single method can work, so we should change the data, in terms of its features, instead.

% TODO: manual review
The hybrid builds a different selection for each dataset and model, and this is what the search is for. On Caco-2 it picked 1,000 to 2,000 features for the DNN on every fold, and 500 to 1,000 for the SVM. Most small-molecule folds drew on all ten representations, while the Buchwald--Hartwig searches drew on two on every fold, and QM9 with LightGBM on three. Across the configurations of one search, the downstream model accounted for 48.5\% of the variance in score on LogD and the budget for 1.4\%. A fixed recipe would have to get all of these right at once, for every dataset.

% TODO: manual review
The hybrid and noise mitigation also work together. Because the hybrid's features are chosen against the labels, the labels can be cleaned before the search instead of after it. On the expression titer labels as delivered, the network on the hybrid scores 0.1550 Spearman with no cleaning. Deleting the antibodies the forest residual flags raises it to 0.3899 when only the fit sees the cleaned labels, and to 0.5619 when the search sees them too. The hybrid can also be the space in which a detector finds its neighbors. On LogD under a scaffold-family shift, down-weighting by the neighbor gap took the network's single representation from 0.7051 RMSE to 0.7535 with neighbors in its own space, and to 0.6626 with neighbors in the hybrid. Both results are on one fold, and neither has a control yet, but they are what ``adapts further alongside noise mitigation'' looks like in practice.


\section{What Is New in This Thesis}\label{sec:discussion:contributions}

% TODO: manual review
The adaptive hybrid representation is entirely our own. Several parts of it have no counterpart in the methods it is closest to (Table~\ref{tab:hybrid_alternatives}). It ranks features within each base representation before it decides how many each one contributes, where flat feature selection and recursive elimination put every feature on one scale. It searches the settings of the selection itself per dataset and downstream model: the importance method, allocation method, internal evaluator, number of representations and budget. It returns them as a blueprint that can be rerun on new data. It can be built for distance as well as for prediction (Chapter~\ref{chap:similarity}). It runs where data cannot be pooled, with each site selecting its own features and sending only their indices (Chapter~\ref{chap:method}, Section~\ref{sec:method:results_sites}). And it accepts a new embedding as one more base representation, so the search, rather than habit, decides whether a new model earns a place.

% TODO: manual review
On the noise side, several of the methods in Chapter~\ref{chap:noise} are new, and many of the rest were changed from their published form. For the following we found no published source:
\begin{itemize}
    \item cleaning the labels before the hybrid search rather than after it, a question no published method has to answer, since none chooses its features against the labels it cleans;
    \item the cliff-aware gap, a detector that compares a label with its neighbors' labels but trusts a neighborhood less when its labels disagree, so that an activity cliff is not mistaken for noise;
    \item the gap over neighbor spread, a detector that scales a label's distance from its neighbors' mean by how much those neighbors disagree;
    \item putting activity-cliff molecules back after any cleanup, so that the most informative molecules are not deleted as noise.
\end{itemize}
Other methods follow a published idea with our own change. Censored EM is the Buckley--James estimator for labels clipped at a limit, run with any downstream model instead of a linear one. The random intercept for scaffold families is a mixed-effects forest fitted to out-of-fold residuals, with the scaffold family as the known group. Neighbor detectors search the space in which neighbors are found, among the base representations, the hybrid, importance-weighted features and forest proximity, and pick it on held-out training molecules. Every prediction used to score a label is made out of fold, with inner folds that keep scaffold groups together. Most of the published methods for learning with noisy labels were written for flipped classes, and we adapted them to regression.

% TODO: manual review
Just as new is the way they were tested. Every method was run on the single representation and the hybrid side by side, under seven kinds of damage, with QM9 as a control where the labels are clean. Cleanups on the labels as delivered were compared with the same cleanup driven by shuffled detector scores, and deletion with random deletion of the same size. This is what lets the thesis say that a gain is noise removed, and not just a model that got lucky.


\section{Handling Noise Realistically}\label{sec:discussion:noise}

% TODO: manual review
The strongest noise results in this thesis came where something other than a model says which labels are wrong. A clipped label is flagged by the assay limit itself. The correction then only has to decide how far to raise it, and no detector is needed. With a quarter of LogD's training labels clipped, censored EM won back 76\% of the RMSE that clipping cost the network, and on the hybrid it went from 0.6049 to 0.4610. The gain held on each of the four folds where it was run. On QM9, where the labels before clipping carry no assay error, LightGBM on the hybrid went from 0.5240 to 0.4301 against 0.3694 before clipping.

% TODO: manual review
Knowing which labels are clipped is what helps. Knowing how far each one should go is not. Raising each clipped label by its own estimated amount did worse than dealing the same amounts to the wrong clipped molecules. For the network on LogD's hybrid it gave 0.4988 RMSE, against 0.4714, 0.4673 and 0.4632 with the amounts shuffled. Deleting the clipped molecules, which uses the same flag but throws them away, was the worst choice of all, because the model then never sees a high label. Scaffold families behave the same way when each carries its own offset. The family is the flag, and subtracting each family's estimated offset took the network on Caco-2's hybrid from 0.3214 to 0.2796 RMSE on the first damaged draw, against 0.2385 on clean labels.

% TODO: manual review
When a detector has to infer which labels are wrong, mitigation still helps, but less, and only when the detector is precise enough. Under outliers, the forest residual's top 5\% are mostly damaged molecules, and acting on them pays off. On hERG $K_i$ with outliers, down-weighting the flagged molecules took the network on the hybrid from 0.6917 to 0.6388 RMSE, while down-weighting the same number of undamaged molecules made it worse. Under a shift in a few labels the residual's precision stayed below the point where acting on it pays off. And deletion never helped on clean labels. On QM9 with no noise added, the best deletion took the network's hybrid from 0.3550 to 0.3931 RMSE. So where deletion helps on damaged labels and beats random deletion, the gain is noise removed.

% TODO: manual review
% RESULT PENDING: ginkgo_pr_cho real censoring (Chapter 7, Section sec:noise:res:censoring, slot [N7-20 pending]; tests/experiments/noise/n30b_real_censoring.py). One sentence here once it lands: whether the corrections that worked on injected clipping also work on the 58 of 197 labels at CHO polyreactivity's floor, single representation and hybrid.
Every clipped-label result above is on injected clipping, where the limit is known. CHO polyreactivity is the one dataset in this thesis with a floor visible in its labels, with 58 of its 197 labels at 0. \textbf{[RESULT NEEDED: whether the same corrections help on CHO polyreactivity's labels at the floor, against the same move applied to labels not at the floor (Chapter~\ref{chap:noise}, Section~\ref{sec:noise:res:censoring})]}

% TODO: manual review
% Links to Chapter 3's conclusion (grouped-shifted and censoring hardest), Chapter 7's offsets on labels as delivered [N7-27] caco2 dnn hybrid 0.2333 -> 0.2669, logd 0.3977 -> 0.4872, fold 0.
This gives a practical order of work. Before choosing a method, ask where the labels came from and what shape their noise is likely to have. Chapter~\ref{chap:robustness} found that systematic error, a shared offset or a clipped range, does the most damage to a model. Chapter~\ref{chap:noise} found that the same systematic error is the easiest to undo, because it comes with a flag. Data pooled from several sources or laboratories, assays with a reporting limit, and series measured in batches all carry such a flag. A method that assumes damage that is not there does harm, though. Subtracting scaffold-family offsets from Caco-2's labels as delivered took the network's hybrid from 0.2333 to 0.2669 RMSE, because the offsets it removed were real differences between scaffolds. So the flag must come from how the data were made, not from the method.


\section{The Antibodies}\label{sec:discussion:antibodies}

% TODO: manual review
This work began with small molecules, and small molecules remain its focus. The hybrid search, though, takes any set of feature matrices for the same samples and a supervised task, and nothing in it is specific to chemistry. We ran it on antibodies with the same code and the same search, and on several antibody datasets it worked well. This was not luck. The hypothesis behind the hybrid is that the methods were held back by representations whose important features are few and buried among many that carry nothing. That holds for any domain where several representations describe the same samples. The adaptive search is what lets the method carry over without being redesigned.

% TODO: manual review
Most antibody representations in use read a language model's embedding of the whole sequence. We built a set of our own on top of these models. Several pool the embedding over the complementarity-determining regions (CDRs) only, over each CDR separately, or over each of the seven IMGT regions separately. One takes the change in the embedding from the parent antibody, and another describes each mutation by its residues and the change in hydrophobicity, weight and charge. Breaking the sequence down by region keeps the parts of the antibody that vary apart from the framework that mostly does not, which is the same problem Chapter~\ref{chap:flaws} found with general-purpose fingerprints.

% TODO: manual review
% SOURCE (carried over from Chapter 5, Section sec:method:results_single): c01_rep_characterization against c02, 5 folds: flab_expression_koenig held-out-best single ESM3 CDR, svm 0.7249, dnn 0.7243; hybrid svm 0.7729, dnn 0.7352 (RERUN-SWAP-DNN). ddg_combined held-out-best single: svm mutation features 0.2034, dnn ESM-2 shift from parent 0.2215; hybrid svm 0.2178, dnn 0.3542 (RERUN-SWAP-DNN).
Those representations carry the clearest antibody results. On the anti-VEGF expression scan, the best single representation in hindsight was our CDR readout of ESM3, at 0.7249 Spearman with the SVM, and the hybrid beat it at 0.7729. On binding free energy, the best single representations were our mutation features with the SVM and the change from the parent embedding with the DNN, at 0.2034 and 0.2215. The hybrid reached 0.2178 and 0.3542. We shuffled each representation's features in the binding free energy hybrid in turn. The representations built for this thesis account for 0.83 of the drop in score with the SVM and 0.68 with the DNN, averaged over the five folds. On the other antibody datasets they account for between 0.03 and 0.27, so this is where they matter most, not everywhere.

% TODO: manual review
The noise methods also carried over. On three antibody datasets, cleaning the labels as delivered passed the shuffled-score control, which on small molecules only label smoothing on hERG $K_i$ did by a clear margin. On SMAC colloidal stability, deletion took the SVM from 0.2740 to 0.4034 Spearman on the single representation and from 0.3108 to 0.3782 on the hybrid. On binding free energy, label smoothing took the network from 0.3414 to 0.4840, and on expression titer deletion took the random forest's hybrid from 0.2244 to 0.3168. These datasets were measured in a few experiments on a few hundred antibodies each, and their labels carry real noise that cleaning can find.

% TODO: manual review
The hybrid did not help on every antibody dataset, and the reasons are specific. On nanobody thermostability, SMAC colloidal stability, HIC retention and CHO polyreactivity it added nothing, for example 0.6899 against 0.6901 Spearman on thermostability with the SVM. These sets hold a few hundred antibodies, and their held-out folds 35 to 53. On so few, the search's own scores overstate what the chosen hybrid delivers, by 0.1571 Spearman on colloidal stability with the DNN. Several of them also carry label problems of their own: thermostability pools five assays into one label, and 58 of CHO polyreactivity's 197 labels sit at the assay floor. The hybrid needs data, and on the antibody sets with enough of it, or with a representation that sees what varies, it delivered.


\section{No Single Method Fits Every Dataset}\label{sec:discussion:conditional}

% TODO: manual review
% SOURCE (carried over from Chapter 5): c18 herg_ki svm B=250 depth_1 0.5990, natural 0.5642, uniform_all 0.5619; c11 flab_expression_koenig dnn ensemble 0.7854 vs hybrid 0.7348 (0/5 folds), qm9 dnn hybrid 0.3174 vs ensembles 0.3402; c17 herg_ki dnn B=1000 RFE 0.5794, random 0.5705, hybrid 0.6112; c06 SVM C 0.01 and 0.1 pick 500 on every fold (logd, adme_mdr1, adme_rlm, caco2). RERUN-SWAP-DNN for DNN numbers.
Which hybrid works depends on the dataset and the model. Drawing on several representations beat filling the budget from the top one alone on almost every dataset, 0.5642 against 0.5990 RMSE on hERG $K_i$ with the SVM at 250 features. How the budget was split among them mattered much less: an even split gave 0.5619 there. The strongly regularized SVM picked 500 features on every fold of four small-molecule datasets, while the DNN took 2,000 on every fold of LogD. On small molecules one model on the hybrid beat an ensemble of one model per representation, 0.3174 against 0.3402 RMSE on QM9 with the DNN. On antibodies the ensemble often won, 0.7854 against 0.7348 Spearman on the expression scan with the DNN. And on hERG $K_i$ with the DNN, recursive feature elimination and even random features beat the hybrid at 1,000 features. Whether the best settings follow measurable properties of a dataset is the question of one experiment still running.

% TODO: manual review
% RESULT PENDING: cross experiment c03 (do the best hybrid settings follow dataset properties), slot in Chapter 5 Section sec:method:results_single. One sentence here when it lands.
\textbf{[RESULT NEEDED: which dataset properties, if any, predict the best hybrid settings (Chapter~\ref{chap:method}, Section~\ref{sec:method:results_single})]}

% TODO: manual review
The same holds for similarity. On every small-molecule dataset the distance hybrid gave better neighbors. On nanobody thermostability, AC-SINS self-association, CHO polyreactivity, expression titer and the Buchwald--Hartwig reactions, prediction from neighbors was worse in the hybrid once its setting was picked again on each fold. Distance still tracked activity more closely on all of them, so the hybrid improved the space, but on those small sets a better space did not always become a better neighbor model.

% TODO: manual review
For noise, which method works depends on the kind of noise first. Under clipping, censored EM was best on LogD and QM9, while on hERG $K_i$ the simpler constant lift was better, 0.6662 against 0.7037 RMSE for the network on its second single representation. Under scaffold-family offsets, the random intercept was best on Caco-2 and LogD, and on hERG $K_i$ the plain family mean did as well. Among the detectors, the cliff-aware gap found shifted labels better than the forest residual, 0.2686 against 0.2023 average precision on Caco-2, while the residual was as good or better at outliers. A Huber loss helped the network under outliers, 0.7413 to 0.7269 RMSE on hERG $K_i$, and made Gaussian noise worse at every threshold. Selecting with two learners beat random deletion with tree models, as on Buchwald--Hartwig, where the random forest went from 20.3381 to 18.6007 RMSE, but not with the network.

% TODO: manual review
The model and the domain matter too. Deleting the molecules the network fits worst, the idea behind small-loss selection, almost never helped on small molecules, and helped on Buchwald--Hartwig under every damage kind it ran on (Table~\ref{tab:noise:res:sweep}). The network gained from cleaning more often than the SVM on Caco-2, hERG $K_i$ and LogD. On binding free energy the SVM gained more often, and on expression titer the random forest did. None of these is a ranking to carry to a new dataset. They are the reason a new dataset needs its own sweep.


\section{Simple Is Often Better}\label{sec:discussion:simple}

% TODO: manual review
Several of the most effective changes in this thesis were the simplest. Random forests were the most consistent model under label noise across the four datasets of Chapter~\ref{chap:robustness}. In Chapter~\ref{chap:noise}, averaging ten network seeds took the network on hERG $K_i$'s hybrid from 0.7184 to 0.6557 RMSE under Gaussian noise, against 0.6075 on clean labels. That is as large as the best mitigation results on the same dataset, and it needs no detector and changes no label. There is no QM9 result for it yet, so we cannot say how much of the gain is noise handled and how much is a better model.

% TODO: manual review
Simple corrections also beat elaborate ones. Under clipping, a constant lift for every clipped label did better than a lift estimated for each molecule. Of the twelve published methods for learning with noisy labels, every one made the network worse on clean QM9 labels, DivideMix the most, 0.4517 against 0.3550 RMSE on the hybrid. On the hybrid side, tuning the SVM on a wider hybrid beat a smaller hybrid at the default setting on all four small-molecule datasets with a result, 0.4114 against 0.4299 RMSE on LogD. The search is worth its cost where the representations disagree and the data are large enough to choose between them. Where one representation already solves the task, as for antibody humanness, there is nothing for it to add.


\section{Limitations}\label{sec:discussion:limitations}

% TODO: manual review
% Collected from the Limitations sections of Chapters 5 to 7 as written (sec:method:limitations, sec:similarity:conclusions, sec:noise:limitations).
Most noise results rest on the first outer fold. Further folds exist for the clipped-label corrections on LogD and Caco-2, deletion under hERG's flips and label smoothing on hERG $K_i$, and they mostly agree with fold 0. Most noise methods have no control, and no antibody dataset has one under injected damage. Changing the model, cleaning before the search and moving every label toward a Huber-fitted prediction have no QM9 result with no noise added. Their gains cannot yet be split into noise removed and a better model. QM9 is the only clean control, and only one of its properties, the HOMO--LUMO gap, was tested.

% TODO: manual review
There is no way to correct a label without ground truth and know it was corrected. On the labels as delivered, the shuffled-score control shows that a gain is more than chance, but not that the corrected labels are closer to the truth. Seven of the datasets of Chapters~\ref{chap:method} and~\ref{chap:similarity} have no noise results yet: adme\_mdr1, adme\_rlm, flab\_expression\_koenig, flab\_binding\_koenig, abagym\_EGFR, abagym\_Ang2 and ginkgo\_hic.

% TODO: manual review
The hybrid needs data, and most antibody datasets here hold a few hundred antibodies, so the antibody results say as much about dataset size as about antibodies. The search is costly and must be rerun when the data, the representations or the downstream model change, and each dataset searched a smaller grid cut from an earlier search. Most DNN results were computed before the correction of 8 October 2026, and the Gaussian-process results will change when that model is replaced.


\section{Future Directions}\label{sec:discussion:future}

% TODO: manual review
The hybrid search is not tied to molecules. It needs several feature matrices for the same samples and a supervised task. We ran it on reactions, where the Buchwald--Hartwig results were mixed. With the hybrid, LightGBM went from 16.17 to 15.62 RMSE in percentage points of yield, and the random forest from 14.41 to 15.24. We also ran it on DNA sequences, telling TATA-box promoters apart in 5,274 sequences. On each of the three folds that finished, the hybrid beat the two single representations the search ranked highest, with both models. With LightGBM it went from 0.6268 to 0.7060 MCC on the first fold. The natural next step is to test it in other domains with many competing representations of the same samples. Gene-expression profiles, for example from The Cancer Genome Atlas, are one. Financial time series, where many hand-built indicators describe the same market, are another.

% TODO: manual review
% CHECKED 2026-10-09: src/kirby/datasets/qm9.py load_qm9(property_idx=4), 12 properties (0-11); only the gap is used in this thesis.
QM9 should get more than one clean control. Its loader already offers twelve computed properties, and only the HOMO--LUMO gap was used here. Running the noise methods on several more would show whether the QM9 results hold beyond one target, and would give the methods without a clean control the result they need.

% TODO: manual review
% Outline 8.7 [DECIDE] reworded as what is still missing: six flip kinds already ran on herg (n09; Chapter 7).
Classification needs a closer look. Chapter~\ref{chap:noise} flipped class labels on hERG in six ways, but in QSAR a class label usually comes from cutting a continuous value at a threshold. Real flips then concentrate near the threshold and depend on the molecule. Flips of that kind, on more classification datasets and more folds, are the next test for the detectors and the published methods, most of which were designed for flipped classes.

% TODO: manual review
% CHECKED 2026-10-09: tests/experiments/opig_generator_scoring_swap.py and tests/experiments/opig_generator_scoring/ exist (REINVENT scorer); no result files (outline 8.7).
A hybrid could also be the input to the model that scores molecules for a generator. The scoring code for this exists and has not been run. What the search should optimize there is open, since accuracy on known molecules may not reward the right properties in new ones.

% TODO: manual review
The search should get faster. Settings found on LogD and reused on hERG $K_i$ beat hERG $K_i$'s best single representation, 0.5665 against 0.5888 RMSE with the SVM. A search on hERG $K_i$ itself still did better, at 0.5631. A search that starts from the blueprint of a similar dataset could reach a good hybrid at a fraction of the cost. A fuller federated deployment, with real sites that select features on their own data, would test the separate-site results of Chapter~\ref{chap:method} beyond one public dataset split into simulated sites.


\section{Conclusions}\label{sec:discussion:conclusions}

% TODO: manual review
Noise mitigation in molecular data is hard, and for most of this project it did not work. It works now. It works best when the data say which labels are wrong, as an assay limit or a scaffold family does. It works with a detector when the detector is precise enough for the damage at hand. It works on the labels as delivered too, on hERG $K_i$ and on three antibody datasets, against a control that rules out chance.

% TODO: manual review
Fixing the feature set was the right place to start. I spent a year on what looked like a separate problem, on the hypothesis that the representation was the root of the trouble with similarity and with noise. On small molecules the hybrid predicts better than any single representation, gives better neighbors, and keeps its lead through every kind of cleaning. It carried over to antibodies without redesign, and did best where the representations were built to see what varies. No single method fits every dataset, and the hybrid does not pretend to be one. It is a procedure that adapts the features to the data, and it adapts further when the labels are cleaned before it.

% TODO: manual review
% The author's "What does this mean in practice?" paragraph, kept from the old text; Chapter 5 (c13) supports it on small molecules.
A new foundation model appears every few months in molecular, protein and antibody modeling. With the hybrid search, a researcher does not need to choose between their current representation and a new one. They add the new embedding to the representations searched and run the search again, and the search decides whether the new embedding contributes features and, if so, how many and which. A choice usually set by habit or laboratory convention becomes one measured on each task, and it can be measured again every time a new representation appears.
```
NEW:
```latex
\chapter{Discussion and Conclusions}\label{chap:discussion}
% DECISIONS WITH THE AUTHOR, 2026-10-09:
% - The chain section (outline 8.2, with diagram D8.1) was removed by the author on 2026-10-10. Backup: thesis.tex.before_ch8_chain_removed_2026-10-10.
% - The antibodies: a method built for small molecules that generalizes to antibodies, because the representations were what held the methods back; the hand-built antibody representations (CDR and region readouts, mutation features, the change from the parent) named as hers. Not "unusually well".
% - Original contributions named: the hybrid, and the noise methods that are new or changed from their sources (METHOD_PROVENANCE.md; the author's path scratchpad/provenance/ does not exist, the file read is /Volumes/seagate/claude_analysis_2026-10-08/answers/_ref/METHOD_PROVENANCE.md).
% - Next steps: reactions (tested) and nucleotide sequences (results exist), then other domains, including finance and gene-expression data. No patient data was used in this thesis, and none is reported.
% Every number below is copied from Chapters 5 to 7 as written, with that chapter's SOURCE comment carried over, except the antibody permutation shares in Section sec:discussion:antibodies, which were re-read from the raw c22 files on 2026-10-09 (SOURCE given there).

% [2026-10-10] Chapter rebuilt on the shape and length of the reference final chapters (Buttenschoen: findings and contributions, limitations and future research, closure; Vost: short topic sections and concluding remarks). Content from scratchpad/thesis_style_pass/TAKEAWAYS.md. Every \label kept: the old section labels now sit on the section that holds their content. Comments of the old sections are carried below or at the end of the chapter.

% TODO: manual review
This thesis set out to handle label noise in molecular data. The published noise methods kept failing, and each time we asked why, the answer sat one level further down, until it reached the representation itself. This chapter answers the research questions of Chapter~\ref{chap:intro} in turn and lists the contributions of the thesis. It closes with the limitations, future work and concluding remarks.


\section{Summary of findings}\label{sec:discussion:summary}\label{sec:discussion:noise}\label{sec:discussion:antibodies}\label{sec:discussion:conditional}\label{sec:discussion:simple}

% TODO: manual review
% Words only; the numbers are in Chapter 3's Conclusions.
% [2026-10-10] Numbers now kept to one or two per answer. 47.2% and 8.1%: Chapter 3, variance decomposition table, row "Gaussian & \aucnorm{}".
Chapter~\ref{chap:intro} asked four questions about label noise, and Chapter~\ref{chap:robustness} answered them. The first asked how the representation and the model divide the variance in performance, and whether the division changes for noise robustness. It does. Model family explained 47.2\% of the variance in \aucnorm{} on QM9 under Gaussian noise, and the representation 8.1\%. The representation mattered more for predictive performance, and its pairing with the model explained more than the representation alone.

% TODO: manual review
The second question asked what the choice of model provides under label noise. Random forests were the most consistent model, ranking between first and eighth for \aucnorm{} on every dataset, representation and noise condition. LightGBM and XGBoost predicted well on clean labels and lost the most as noise rose. Bayesian networks raised \aucnorm{} but never predicted better than their deterministic counterparts.

% TODO: manual review
The third question asked whether the type of noise matters, or only its amount. The type matters. The shape of the noise distribution barely changed the damage, while grouped-shifted noise and censoring, where the error is systematic, did the most. Label noise in molecular data therefore cannot be treated as Gaussian.

% TODO: manual review
The fourth question asked what label noise does to estimated uncertainty. Mean predicted uncertainty rose with noise for almost every probabilistic model on QM9. Outside censoring, no model's uncertainty for a single molecule identified the corrupted labels, and under censoring the models were most confident about the clipped labels.

% TODO: manual review
Chapter~\ref{chap:flaws} explained why published methods for learning with noisy labels did poorly on molecular data, and why those using a distance did worst. On a single dataset, little of a standard representation varies. On hERG $K_i$, 1,815 of the 2,048 ECFP4 features take one value in at least 95\% of the molecules. A distance sums over every feature, so the features that carry nothing dominate it.

% TODO: manual review
% SOURCE (carried over from Chapter 5, Section sec:method:results_single): /Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/<ds>.json standard_selection_summary.<model>.{base1,hybrid1}.per_fold[].heldout_score, 5 folds. caco2 svm 0.3036 -> 0.2862; logd svm 0.4930 -> 0.4302; qm9 dnn 0.3619 -> 0.3155 (RERUN-SWAP-DNN); flab_expression_koenig svm 0.7029 -> 0.7729, 5/5 folds; ddg_combined dnn 0.1317 -> 0.3542, 4/5 (RERUN-SWAP-DNN).
The second half of Chapter~\ref{chap:intro} asked four questions about adapting the representation to the task. The first asked how the best set of features can be selected for a given prediction task. Chapter~\ref{chap:method} found that the best set depends on the dataset and the downstream model, and that a search can find it. The adaptive hybrid representation beat the best single representation on every small-molecule dataset with both downstream models, for example from 0.4930 to 0.4302 RMSE on LogD with the SVM. On antibodies the gains depended on the dataset.

% TODO: manual review
The second asked whether distance in a hybrid tracks the label better, and whether methods built on nearest neighbors work better in it. On all five small-molecule datasets, both were true (Chapter~\ref{chap:similarity}). On hERG $K_i$ the Spearman correlation between the distance of two molecules and the difference in their activity went from 0.0729 to 0.2625. On the antibody datasets and the reactions, distance also tracked activity more closely, but prediction from neighbors was worse in the hybrid on several of them.

% TODO: manual review
The third asked whether noisy labels can be found and corrected, and whether the hybrid keeps its lead through that cleaning. Chapter~\ref{chap:noise} found that both hold, under conditions. Corrections worked best when the data flagged the wrong labels. With a quarter of LogD's training labels clipped, censored EM took the network from 0.6354 to 0.4795 RMSE, against 0.4314 before clipping. With a detector, cleaning helped under label flips and outliers, and beat random and shuffled controls on several datasets. On small molecules the hybrid stayed ahead of the single representation through every treatment.

% TODO: manual review
The fourth asked whether weighting features by their importance improves the construction of hybrid representations, model training and noise mitigation. Ranking features by importance within each representation is what makes the hybrid work. Using the scores again in training added little on small molecules. For noise detection alone, importance-weighted features of a single representation did better than the hybrid's.

% TODO: manual review
% Links to Chapter 3's conclusion (grouped-shifted and censoring hardest), Chapter 7's offsets on labels as delivered [N7-27] caco2 dnn hybrid 0.2333 -> 0.2669, logd 0.3977 -> 0.4872, fold 0.
Label noise was handled, but no single method handled it everywhere. As in Chapter~\ref{chap:robustness}, which method worked depended on the prediction task, the data, the model and the type of noise. NGBoost was the most robust model under Gaussian noise on the HOMO--LUMO gap, but not on Caco-2 or hERG $K_i$. Under censoring the forest residual detector fell to chance, and the known flag did best. On Caco-2's labels as delivered, subtracting scaffold-family offsets made the model worse. Deleting the molecules the network fits worst almost never helped on small molecules, but helped on the reactions. The ranking of noise methods also changed between the hybrid and the single representations. So before choosing a method, ask where the labels came from and what noise they are likely to carry. Simple methods were often enough: random forests, and one constant lift for every clipped label.


\section{Contributions}\label{sec:discussion:contributions}

% TODO: manual review
% Sources: TAKEAWAYS.md section 6 and thesis_provenance/ (METHOD_PROVENANCE.md, LITERATURE_SEARCH_2026-10-08.md). Only the four methods in item 6 are claimed as without precedent on molecular data.
The contributions of this thesis on label noise are the following.
\begin{enumerate}
    \item NoiseInject, an open-source Python package for benchmarking noise robustness and uncertainty in QSAR models (Section~\ref{sec:noiseinject_pkg}).
    \item Seven noise conditions at a matched amount of noise, including noise shared within scaffold families and labels clipped at an assay limit (Chapter~\ref{chap:background}).
    \item A split of the variance in performance and in noise robustness between model family, representation and their pairing (Chapter~\ref{chap:robustness}).
    \item Noise-handling methods tested on single representations and hybrids side by side, against controls: QM9 with no noise added, random deletion of the same size, and shuffled detector scores (Chapter~\ref{chap:noise}).
    \item Published methods changed for QSAR data, such as censored EM with any downstream model, a random intercept per scaffold family, and regression versions of methods for flipped classes (Chapter~\ref{chap:noise}).
% TODO: manual review -- originality claim pending literature check (thesis_provenance/)
    \item Four methods for which we found no earlier use on molecular data (Chapter~\ref{chap:noise}). These are the cliff-aware gap, the gap over neighbor spread, cleaning the labels before the hybrid search, and putting activity-cliff molecules back after a cleanup.
    \item The finding that the ranking of noise-handling methods depends on the feature space they run in (Chapter~\ref{chap:noise}).
\end{enumerate}

% TODO: manual review
The contributions on the representation are the following.
\begin{enumerate}[resume]
    \item The adaptive hybrid representation, which ranks features within each representation, splits a budget of features among them, and searches the selection settings for each dataset and model (Chapter~\ref{chap:method}).
    \item Hybrids built for distance (Chapter~\ref{chap:similarity}), and hybrids built where data are held at separate sites (Chapter~\ref{chap:method}).
    \item Antibody representations built for this thesis: readouts of the complementarity-determining regions (CDRs) and of each region, mutation features, and the change in embedding from the parent antibody (Chapter~\ref{chap:method}).
    \item A search that takes in new embeddings. A newly published embedding is added as one more representation to search.
\end{enumerate}


\section{Limitations}\label{sec:discussion:limitations}

% TODO: manual review
% Collected from the Limitations sections of Chapters 5 to 7 as written (sec:method:limitations, sec:similarity:conclusions, sec:noise:limitations).
Most noise results rest on the first outer fold. Where further folds exist, they mostly agree with it. Most noise methods have no control, and QM9 is the only clean control, with one property tested. On the labels as delivered there is no ground truth, so a control can show that a gain is more than chance but not that a label was corrected.

% TODO: manual review
% RESULT PENDING: ginkgo_pr_cho real censoring (Chapter 7, Section sec:noise:res:censoring, slot [N7-20 pending]; tests/experiments/noise/n30b_real_censoring.py). One sentence here once it lands: whether the corrections that worked on injected clipping also work on the 58 of 197 labels at CHO polyreactivity's floor, single representation and hybrid.
Every clipped-label result is on injected clipping. CHO polyreactivity is the one dataset here with a floor visible in its labels. \textbf{[RESULT NEEDED: whether the same corrections help on CHO polyreactivity's labels at the floor, against the same move applied to labels not at the floor (Chapter~\ref{chap:noise}, Section~\ref{sec:noise:res:censoring})]}

% TODO: manual review
% RESULT PENDING: cross experiment c03 (do the best hybrid settings follow dataset properties), slot in Chapter 5 Section sec:method:results_single. One sentence here when it lands.
The hybrid needs data. Most antibody datasets here hold a few hundred antibodies, and on so few the search's own scores overstate what the chosen hybrid delivers. The search is costly and must be rerun when the data, the representations or the model change. Most DNN results were computed before the correction of 8 October 2026, and the Gaussian-process results will change when that model is replaced. \textbf{[RESULT NEEDED: which dataset properties, if any, predict the best hybrid settings (Chapter~\ref{chap:method}, Section~\ref{sec:method:results_single})]}


\section{Future work}\label{sec:discussion:future}

% TODO: manual review
% [2026-10-10] The Cancer Genome Atlas no longer named; the gene-expression mention kept in general terms (no patient data rule).
The hybrid search needs only several feature matrices for the same samples and a supervised task. Results on reactions were mixed, and on DNA promoter sequences the hybrid beat the two highest-ranked single representations on each finished fold. Gene-expression profiles and financial time series are natural next domains.

% TODO: manual review
% CHECKED 2026-10-09: src/kirby/datasets/qm9.py load_qm9(property_idx=4), 12 properties (0-11); only the gap is used in this thesis.
% Outline 8.7 [DECIDE] reworded as what is still missing: six flip kinds already ran on herg (n09; Chapter 7).
The noise methods need wider controls. QM9 offers twelve computed properties, and only the HOMO--LUMO gap was used. In QSAR a class label usually comes from cutting a continuous value at a threshold, so real flips concentrate near it. Flips of that kind are the next test for the detectors.

% TODO: manual review
% CHECKED 2026-10-09: tests/experiments/opig_generator_scoring_swap.py and tests/experiments/opig_generator_scoring/ exist (REINVENT scorer); no result files (outline 8.7).
A hybrid could also be the input to the model that scores molecules for a generative model; the code exists and has not been run. The search could start from the blueprint of a similar dataset to save compute, and a deployment across real sites would test the separate-site results.


\section{Concluding remarks}\label{sec:discussion:conclusions}\label{sec:discussion:claim}

% TODO: manual review
% Outline 8.2a, the author's words (2026-10-08).
% [2026-10-10] From the author's outline 8.2a, shortened. "no single method can work" changed to what was measured (no single method was best on every dataset tested), and the "The claim is not ... It is ..." contrast removed.
Noise mitigation in molecular data is hard, and for most of this project it did not work. Label noise was handled in the end, by methods built or changed for the noise molecular data carry, and by changing the features they run on. No single model, representation, similarity measure or noise-handling method was best on every dataset we tested. Everything depends on the data: the features, the true labels, and the noise those labels carry. The hybrid is not a one-size-fits-all method. It adapts to the data it is given, and it adapts further when the labels are cleaned before the search.

% TODO: manual review
% The author's "What does this mean in practice?" paragraph, kept from the old text; Chapter 5 (c13) supports it on small molecules.
A new foundation model appears every few months in molecular, protein and antibody modeling. With the hybrid search, a researcher does not need to choose between their current representation and a new one. They add the new embedding to the representations searched and run the search again, and the search decides whether the new embedding contributes features and, if so, how many and which. A choice usually set by habit or laboratory convention becomes one measured on each task, and it can be measured again every time a new representation appears.

% TODO: manual review
I spent much of this project on what looked like a separate problem, the features. My hypothesis was that they were the root of the trouble with similarity and with noise, and on small molecules the results support it.

% Comments carried over from the Chapter 8 text replaced on 2026-10-10 (their paragraphs were folded into the sections above or cut):
% SOURCE (carried over from Chapter 5, Section sec:method:results_single): c01_rep_characterization against c02, 5 folds: flab_expression_koenig held-out-best single ESM3 CDR, svm 0.7249, dnn 0.7243; hybrid svm 0.7729, dnn 0.7352 (RERUN-SWAP-DNN). ddg_combined held-out-best single: svm mutation features 0.2034, dnn ESM-2 shift from parent 0.2215; hybrid svm 0.2178, dnn 0.3542 (RERUN-SWAP-DNN).
% SOURCE (carried over from Chapter 5): c18 herg_ki svm B=250 depth_1 0.5990, natural 0.5642, uniform_all 0.5619; c11 flab_expression_koenig dnn ensemble 0.7854 vs hybrid 0.7348 (0/5 folds), qm9 dnn hybrid 0.3174 vs ensembles 0.3402; c17 herg_ki dnn B=1000 RFE 0.5794, random 0.5705, hybrid 0.6112; c06 SVM C 0.01 and 0.1 pick 500 on every fold (logd, adme_mdr1, adme_rlm, caco2). RERUN-SWAP-DNN for DNN numbers.
```
