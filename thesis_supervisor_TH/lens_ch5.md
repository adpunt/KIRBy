# Chapter 5 (chap:method): new items in the supervisor's lens

Source read: thesis.tex lines 1262-2660 in full (no \input tables in this chapter; all tables are inline).
Also read, for numbers only: thesis_rework/stage1_takeaways_bullets/additional_files_body.tex
(app:add:evaluator, app:add:conformal, app:add:pca_disagreement, app:add:hindsight_shuffle),
tests/experiments/complexity/c19_importance_weighted_training.py and c09_formal_analysis.py.
Nothing here repeats an item in comments_ch5.md. Where an item sits next to one of his existing
comments, the overlap is named.

Line numbers are from thesis.tex as read on 2026-10-10.

---------------------------------------------------------------------------------------------
## HIGH
---------------------------------------------------------------------------------------------

### X-ch5-1
Location: lines 1404, 1494, 1504, 1523, 1870; sec:method:searchspace, sec:method:decided_grid, sec:method:models. Owner: hers
Quote: "Each dataset's search space is hand-selected from the full search, made to save compute (Section~\ref{sec:method:decided_grid}). At worst, it finds a less informative hybrid than the full search would have." Also: "An earlier search of each whole dataset covered 16 datasets of an earlier list" (1504); "From earlier searches across each dataset, we identified two downstream models" (1523); "Each dataset uses the two that scored best in the earlier search of the whole dataset" (1870); base representations "chosen prior to the search ... based on each representation's relative performance on the prediction task" (1404).
His comment: The grid, the two models, the augmentation decision and maybe the 10 representations were chosen on the whole dataset. Did this earlier search see the samples that are held out later? If yes, it is the same data snooping risk as for the comparison, and "at worst less informative" is not true.
Lens item: 8
Proposed change: NEEDS HER. Which samples did the earlier whole-dataset searches use, and were the 10 base representations chosen from them? This is a different point from his TH-p128a (that one is about the best-single comparator).
old_string:
```
At worst, it finds a less informative hybrid than the full search would have. This decision was made due to time limitations.
```
new_string (if the earlier searches saw every sample):
```
This decision was made due to time limitations. The grids, the two downstream models and the augmentation decision were chosen from earlier searches of each whole dataset. Those searches saw the labels of samples that are held out later. The held-out scores in this chapter may therefore be optimistic, for the hybrid and for the single representations.
```
new_string (if they did not):
```
This decision was made due to time limitations. The earlier searches that chose each grid used only the training samples of each outer fold, so no held-out label shaped the grid.
```
Priority: high

### X-ch5-2
Location: lines 1601-1605 (sec:method:evaluation) and 1885 (sec:method:comparators). Owner: hers
Quote: "We report the results that exist, give the number of folds beside each one, and name the datasets where a result is missing." and "From each fold's search it takes, for each downstream model, the two single representations and the two hybrids with the best scores on the training samples."
His comment: The same hybrid gets a different score in each table. Nanobody thermostability with the DNN is 0.6764 (Table hybrid_vs_best_single), 0.6788 (ensembles), 0.623 (search optimism), 0.615 (one fold fewer) and 0.628 (importance during training). SMAC with the DNN is 0.2677, 0.265, 0.310, 0.317 and 0.261. Which one is "the hybrid", and why do they differ?
Lens item: 17 (same as his TH-p157a, but across this chapter's current tables)
Proposed change: NEEDS HER: the reasons the scores differ (each experiment runs its own search? the range limit used in three experiments only, line 1601? the budget range up to 8,000 in the optimism table? first or second of the two best hybrids?). Suggested sentence once the reasons are known:
old_string:
```
We report the results that exist, give the number of folds beside each one, and name the datasets where a result is missing. 
```
new_string:
```
We report the results that exist, give the number of folds beside each one, and name the datasets where a result is missing. Each experiment runs its own searches, so the same dataset and model can show a different hybrid score in different tables. [NEEDS HER: one sentence per cause, for example the range limit above, the budgets searched, the subset of samples.]
```
Priority: high

### X-ch5-3
Location: line 2555, sec:method:results_blueprints. Owner: hers
Quote: "The selected features do not vary much between folds on the same search. On the stability index of \citet{Nogueira2018}, the small-molecule hybrids score higher than the antibody and Buchwald--Hartwig hybrids, and all score well below 1 (Table~\ref{tab:method:one_fold_fewer})."
His comment: The first sentence says the features are stable, the table says the antibody hybrids are at 0.06-0.27 where 0 is random. Which is it? Give the numbers.
Lens item: 2, 19
Proposed change (numbers from tab:method:one_fold_fewer, lines 2324-2363):
old_string:
```
The selected features do not vary much between folds on the same search. On the stability index of \citet{Nogueira2018}, the small-molecule hybrids score higher than the antibody and Buchwald--Hartwig hybrids, and all score well below 1 (Table~\ref{tab:method:one_fold_fewer}).
```
new_string:
```
The selected features vary between folds of the same search. On the stability index of \citet{Nogueira2018}, the small-molecule hybrids score 0.273 to 0.577, the Buchwald--Hartwig hybrids 0.144 and 0.156, and the antibody hybrids 0.063 to 0.270 (Table~\ref{tab:method:one_fold_fewer}). An index of 1 means the same features on every fold, and about 0 means random selection.
```
Priority: high

### X-ch5-4
Location: line 2080, sec:method:results_ensembles. Owner: hers
Quote: "On small molecules the hybrid holds up better, the same-model ensemble beats it with both models only on hERG blocking, and on the Buchwald--Hartwig reactions. The hybrid is ahead of both ensembles on QM9 with the DNN and on hERG $K_i$ with the SVM, on every fold. The success of hybrids versus antibodies likely hinges on ..."
His comment: Look per model. With the SVM the hybrid wins on all five small-molecule datasets; with the DNN the ensemble wins on four of six. Why does the answer depend on the model? This is the "merge representations or merge predictions" question I asked.
Lens item: 18 (links to his TH-p108c, ensemble as baseline)
Proposed change: the counts are from tab:method:ensembles (lines 2097-2108). The explanation is NEEDS HER; only the pattern is added here. The second edit fixes "hybrids versus antibodies".
old_string:
```
The hybrid is ahead of both ensembles on QM9 with the DNN and on hERG $K_i$ with the SVM, on every fold.
```
new_string:
```
The hybrid is ahead of both ensembles on QM9 with the DNN and on hERG $K_i$ with the SVM, on every fold. With the SVM, the hybrid is ahead of the same-model ensemble on all five small-molecule datasets. With the DNN, the same-model ensemble is ahead on hERG $K_i$, Caco-2, rat microsomal clearance and MDR1--MDCK efflux. [NEEDS HER: one hedged sentence on why the DNN gains more from pooling predictions.]
```
old_string:
```
The success of hybrids versus antibodies likely hinges
```
new_string:
```
The success of hybrids against ensembles on antibodies likely hinges
```
Priority: high

### X-ch5-5
Location: line 2250, sec:method:results_overfit. Owner: hers
Quote: "With budgets searched up to 8,000 features, and at half to twice the number of training samples, the selection bias is small on the larger small-molecule datasets. On these datasets the folds also often pick a hybrid wider than their own training set, for example on every fold of EGFR binding with the SVM."
His comment: EGFR is an antibody dataset, not one of "these". And "small" bias: how small? Give one pair for small and one for large. Also, the held-out score is better than the search's own score on hERG Ki and LogD with the SVM; the reader will wonder how that happens.
Lens item: 6, 2, 21
Proposed change (numbers from tab:method:search_optimism, lines 2263-2294):
old_string:
```
the selection bias is small on the larger small-molecule datasets. On these datasets the folds also often pick a hybrid wider than their own training set, for example on every fold of EGFR binding with the SVM.
```
new_string:
```
the selection bias is small on the larger small-molecule datasets. On hERG $K_i$ with the DNN, for example, the search's own score is 0.591 RMSE and the held-out score 0.592. On the small antibody datasets the bias is large. On EGFR binding with the DNN the search's own score is 0.371 Spearman and the held-out score 0.152. On several datasets the folds pick a hybrid wider than their own training set, for example on every fold of EGFR binding with the SVM. [NEEDS HER: one sentence on why the held-out score beats the search's own score on hERG $K_i$ with the SVM, 0.584 against 0.559, and on LogD with both models.]
```
Priority: high

### X-ch5-6
Location: line 1352, sec:method:build. Owner: hers
Quote: "Because the feature selection process trains on the features and their labels, pooling all the features prior to selection would leak validation-level information into the training, which is a well-documented bias in feature selection  \citep{Ambroise2002, Vabalas2019}."
His comment: The leak comes from selecting on all the samples before the split, not from pooling features. As written, a reader thinks concatenating representations is the leak.
Lens item: 8, 19
Proposed change (old_string keeps the double space before \citep that is in the file):
old_string:
```
pooling all the features prior to selection would leak validation-level information into the training, which is a well-documented bias in feature selection  \citep{Ambroise2002, Vabalas2019}.
```
new_string:
```
selecting features on all samples before the split would let the held-out labels shape the selection, which is a well-documented bias in feature selection \citep{Ambroise2002, Vabalas2019}.
```
Priority: high

### X-ch5-7
Location: line 1696, sec:method:reps. Owner: hers
Quote: "These scores and relative rankings are later used in the concatenation step of the feature selection process."
His comment: Which concatenation step? Section 5.2 says the scores split the budget.
Lens item: 4, 19
Proposed change: NEEDS HER to confirm which scores are meant. Lines 1380-1382 say the budget split uses the internal evaluator's standalone score r_k on the 80/20 internal split. The baseline scores of Figure 5.1 may be different scores. If they are the same:
old_string:
```
These scores and relative rankings are later used in the concatenation step of the feature selection process. 
```
new_string:
```
These scores set how the budget is split among the representations, and which representations are kept (Section~\ref{sec:method:allocation}). 
```
Priority: high

### X-ch5-8
Location: line 2004, "Against the best single representation". Owner: hers
Quote: "It does not improve things globally. On Caco-2 with the DNN, for example, the best single representation in hindsight is PDV. ... but the hybrid beats both, 0.2775 against 0.298 RMSE."
His comment: "Does not improve things globally" and then the hybrid beats everything. What is meant?
Lens item: 6, 19
Proposed change: NEEDS HER to confirm the intended meaning. If it means the gain is not just picking a better representation:
old_string:
```
It does not improve things globally. On Caco-2 with the DNN
```
new_string:
```
The gain is not only from finding a better single representation. On Caco-2 with the DNN
```
Priority: high

### X-ch5-9
Location: line 2648, Conclusions. Owner: hers
Quote: "The gain is not simply from using fewer features, since the hybrid also beat other ways of selecting the same number."
His comment: Not everywhere. Line 2372 says random features and recursive elimination beat the hybrid on hERG Ki with the DNN at 1,000 and 2,000 features. Hedge it.
Lens item: 19
Proposed change:
old_string:
```
The gain is not simply from using fewer features, since the hybrid also beat other ways of selecting the same number.
```
new_string:
```
The gain is not simply from using fewer features, since the hybrid also beat other ways of selecting the same number on most small-molecule datasets. The exception is hERG $K_i$ with the DNN at 1,000 and 2,000 features.
```
Priority: high

### X-ch5-10
Location: lines 1507-1523, sec:method:pruning and sec:method:decided_grid. Owner: hers
Quote: "Adaptive pruning: every configuration is first scored on the first two folds. ..." then "It is more efficient to adaptively prune the search space (Section~\ref{sec:method:pruning}), or rely on hard-coded search settings for each dataset. Due to time constraints, we chose the latter as it gave us a bit more flexibility."
His comment: Was pruning used in the results of this chapter or not? The section describes it in detail and then says the other option was chosen.
Lens item: 4, 3
Proposed change: NEEDS HER: did any reported search also prune inside its fixed grid? (The rules in the section were checked against DEFAULT_PRUNING_SCHEDULE in src/kirby/find_best_hybrid.py per the comment on line 1509. Whether the reported runs used it is not checked here.)
old_string:
```
Due to time constraints, we chose the latter as it gave us a bit more flexibility.
```
new_string (pick one):
```
Due to time constraints, we chose the latter as it gave us a bit more flexibility. Adaptive pruning was also applied inside each dataset's grid.
```
```
Due to time constraints, we chose the latter as it gave us a bit more flexibility. No search reported in this chapter used adaptive pruning.
```
Priority: high

### X-ch5-11
Location: line 2624, sec:method:results_sites. Owner: AI
Quote: "On LogD with the SVM, for example, it is well ahead of the best family, and the best single representation is further behind. ... Each site can therefore select on its own and send only feature indices."
His comment: How far ahead? And the settings were chosen on all sites' data pooled (line 1980). Then the result is not fully federated; say so here, where the conclusion is drawn.
Lens item: 8, 6
Proposed change (numbers from tab:method:family_sites, lines 2632-2635; the pooled-settings fact is from line 1980; the author's own TODO DECIDE on line 1979 is still open):
old_string:
```
On Caco-2 and LogD, with both models, one selection over all families beats the best family and the best single representation, as seen in Table~\ref{tab:method:family_sites}. On LogD with the SVM, for example, it is well ahead of the best family, and the best single representation is further behind. Selecting at each site does as well as one selection over all families on both datasets and with both models, although it can build a different number of features. Each site can therefore select on its own and send only feature indices.
```
new_string:
```
On Caco-2 and LogD, with both models, one selection over all families beats the best family and the best single representation, as seen in Table~\ref{tab:method:family_sites}. On LogD with the SVM, for example, one selection over all families gives 0.438 RMSE. The best family gives 0.460, and the best single representation 0.493. Selecting at each site does as well as one selection over all families, on both datasets and with both models. It can, however, build a different number of features. The search settings used here still come from one search on the training molecules of all sites together (Section~\ref{sec:method:sites}). Only the feature selection after that search runs at each site. Once the settings are fixed, each site can therefore select on its own and send only feature indices.
```
Priority: high

### X-ch5-12
Location: every results table of the chapter: tab:method:ensembles, tab:method:top_rep_even_split, tab:method:scaffold_halves, tab:method:search_optimism, tab:method:one_fold_fewer, tab:method:variance_shares, tab:method:importance_training, tab:method:opig_qsar, tab:method:shared_rows, tab:method:settings_transfer, tab:method:family_sites. Owner: hers
Quote (examples): tab:method:importance_training bolds LogD SVM "\textbf{0.430} & \textbf{0.462}" (0.462 is the worse score); tab:method:shared_rows bolds "Graph kernel alone & \textbf{0.611}" (the worst row); tab:method:settings_transfer bolds 0.589 (the worst SVM score). None of these captions says what bold means. No table gives the spread over folds.
His comment: I read bold as "best". Here bold is sometimes on the worse number. Please bold the better result, or say in the caption what bold marks, and give the variability next to the means (as I asked for Table 5.15).
Lens item: 16 (recurrence of his TH-p150a in the other tables)
Proposed change: NEEDS HER (decision). The bold seems to mark the score pair the text cites. Option A: add to each caption "Bold: the pair cited in the text." Option B: bold the better score of the pair the text cites (at most four bold cells per table). Spread needs per-fold numbers: one extra column per table with the fold standard deviation, or the lowest and highest fold. These tables are typed inline in thesis.tex, so no generator script to change, except where a SOURCE comment names one.
Priority: high

---------------------------------------------------------------------------------------------
## LOW
---------------------------------------------------------------------------------------------

### X-ch5-13
Location: line 2382 and Table tab:method:variance_shares (2392-2430). Owner: AI
Quote: "Table~\ref{tab:method:variance_shares} splits the variance in score across a search's configurations into these two parts, their interaction and a residual. On most datasets with a result, the downstream model explains more of the variance than the budget. On LogD, for example, the model explains about half of it and the budget almost none."
His comment: What is in the residual? It is half the variance on many rows. And Caco-2 has 0% for the model but 44% interaction. What is the take-home?
Lens item: 4, 2, 21
Proposed change. The meaning of the residual was checked in tests/experiments/complexity/c09_formal_analysis.py (lines 308 and 326). The residual is the variance between configurations that share the same model and budget, so it comes from the other four settings. The counts below are from the table: the model share is larger than the budget share on 13 of the 18 rows.
old_string:
```
We then ask how much of the search's own score depends on the downstream model and how much on the budget. Table~\ref{tab:method:variance_shares} splits the variance in score across a search's configurations into these two parts, their interaction and a residual. On most datasets with a result, the downstream model explains more of the variance than the budget. On LogD, for example, the model explains about half of it and the budget almost none.
```
new_string:
```
We then ask how much of the search's own score depends on the downstream model and how much on the budget. Table~\ref{tab:method:variance_shares} splits the variance in score across a search's configurations into four parts. These are the downstream model, the budget, their interaction and a residual. The residual is the variance between configurations with the same model and budget. It comes from the other four settings: importance method, allocation method, internal evaluator and representation count. On most datasets with a result, the downstream model explains more of the variance than the budget. On LogD, for example, the model explains about half of it and the budget almost none. The budget explains more than the model on Caco-2, MDR1--MDCK efflux, hERG blocking, nanobody thermostability and antibody humanness. On Caco-2 the model alone explains none of it, but the interaction explains 43.6\%, so the two models do best at different budgets. In short, on most datasets the choice of downstream model changes the search's score more than the number of features does.
```
Priority: low

### X-ch5-14
Location: line 1482, sec:method:allocation (global pool). Owner: AI
Quote: "Second, the same importance method is fitted once on all the offered features joined together, and the hybrid keeps the top $B$ that score above 0."
His comment: In this joint ranking, does a fingerprint bit compete fairly with a continuous descriptor like logP? You said yourself (Strobl) that tree importance favors features with many split points.
Lens item: 10 (his TH-p114a asks the same for the budget; this is the one place where features of different representations are ranked against each other)
Proposed change (the Strobl fact and citation are already on line 1411):
old_string:
```
The global pool works in two steps. First, each representation offers a number of candidate features, set by its squared weight $\tilde r_k^{\,2}$ as a share of a pool of $2B$ candidates. Each offer is capped at the representation's width and holds only features that score above 0. Second, the same importance method is fitted once on all the offered features joined together, and the hybrid keeps the top $B$ that score above 0. How many features a representation contributes is therefore decided by this joint ranking, and it can be none. A feature's rank also depends on what the other representations offered.
```
new_string:
```
The global pool works in two steps. First, each representation offers a number of candidate features, set by its squared weight $\tilde r_k^{\,2}$ as a share of a pool of $2B$ candidates. Each offer is capped at the representation's width and holds only features that score above 0. Second, the same importance method is fitted once on all the offered features joined together, and the hybrid keeps the top $B$ that score above 0. How many features a representation contributes is therefore decided by this joint ranking, and it can be none. A feature's rank also depends on what the other representations offered. In this ranking, fingerprint bits compete directly with continuous descriptors and embedding values. Split-based tree importance favors features with many possible split points \citep{Strobl2007}. A binary bit can therefore rank below a continuous feature that carries no more information.
```
Priority: low

### X-ch5-15
Location: line 2456, "Dependence on the downstream model". Owner: AI
Quote: "On some antibody datasets the evaluator matters. The matched evaluator is best with both models on EGFR binding, HIC retention, expression titer and colloidal stability. On EGFR binding with the SVM, it is well ahead of a LightGBM evaluator."
His comment: "Well ahead" by how much? The table is now in the Additional files, so the reader cannot see it.
Lens item: 6, 5
Proposed change (numbers from tab:method:evaluator in additional_files_body.tex; the four-dataset claim was re-checked there):
old_string:
```
On some antibody datasets the evaluator matters. The matched evaluator is best with both models on EGFR binding, HIC retention, expression titer and colloidal stability. On EGFR binding with the SVM, it is well ahead of a LightGBM evaluator.
```
new_string:
```
On some antibody datasets the evaluator matters. The matched evaluator is best with both models on EGFR binding, HIC retention, expression titer and colloidal stability. On EGFR binding with the SVM, the matched evaluator gives 0.353 Spearman, against 0.299 with a LightGBM evaluator.
```
Priority: low

### X-ch5-16
Location: line 2071. Owner: hers
Quote: "Across the antibody datasets the hybrids draw on about one and a half to five representations by the same shuffling"
His comment: What is "one and a half representations"? How is it counted?
Lens item: 4
Proposed change (definition from the SOURCE comment of app:add:hindsight_shuffle: "effective number 1/sum(share^2)"):
old_string:
```
Across the antibody datasets the hybrids draw on about one and a half to five representations by the same shuffling (Additional file~\ref{app:add:hindsight_shuffle}).
```
new_string:
```
Across the antibody datasets the hybrids draw on an effective number of about one and a half to five representations by the same shuffling (Additional file~\ref{app:add:hindsight_shuffle}). The effective number is $1/\sum_k p_k^2$, where $p_k$ is representation $k$'s share of the drop in score. It equals the number of representations when all shares are equal.
```
Priority: low

### X-ch5-17
Location: lines 2241-2244 (PCA results). Owner: hers
Quote: "With the SVM, the hybrid is ahead on every fold of hERG $K_i$, Caco-2, LogD, ... With the DNN, PCA is ahead on 3 of 5 folds of the expression scan, and the hybrid on 4 of 5 of $\Delta\Delta G$, 0.354 against 0.239 Spearman."
His comment: Many fold counts, no table, no one-line summary. And why is the DNN the model where PCA catches up?
Lens item: 2, 16, 18
Proposed change: NEEDS HER for a PCA table (the experiment register, line 1896, lists PCA as part of the first question, but no table in the chapter or Additional files gives the PCA scores) and for the "why DNN" sentence. Suggested take-home at the end of the antibody paragraph:
old_string:
```
and the hybrid on 4 of 5 of $\Delta\Delta G$, 0.354 against 0.239 Spearman.
```
new_string:
```
and the hybrid on 4 of 5 of $\Delta\Delta G$, 0.354 against 0.239 Spearman. In short, on small molecules the hybrid is ahead of PCA with the SVM and the tree models, while with the DNN the two are close. On antibodies neither is ahead overall.
```
Priority: low

### X-ch5-18
Location: line 1919, PCA paragraph in "Comparisons against related tasks". Owner: hers
Quote: "We pool together the features across every base representation, standardize them on the training samples, and reduce the set by PCA \citep{Jolliffe2016} to each budget of the dataset's grid."
His comment: After standardization a 2,048-bit fingerprint weighs ten times more in the components than 200 descriptors. Is that intended?
Lens item: 10
Proposed change (2,048 and 200 are the widths in tab:reps_smallmol):
old_string:
```
to each budget of the dataset's grid. The downstream model is fit on the principal components as they are.
```
new_string:
```
to each budget of the dataset's grid. After standardization every feature has the same variance, so a wide representation weighs more in the components. A 2,048-bit ECFP4, for example, enters with about ten times the variance of the 200 PDV descriptors. The downstream model is fit on the principal components as they are.
```
Priority: low

### X-ch5-19
Location: line 2203 and tab:method:scaffold_halves. Owner: hers
Quote: "Out of the four small-molecule regression tasks tested, the hybrid also gains on rare scaffolds, such that each half is searched separately, as seen in Table~\ref{tab:method:scaffold_halves}."
His comment: Hard to read. And the QM9 errors here (0.53 to 0.68) are about twice those of Table 5.x (0.32 to 0.37). The reader will think these are different results.
Lens item: 1, 17, 21
Proposed change (checked in the table: the hybrid is ahead on the rare half on all four regression datasets with both models):
old_string:
```
Out of the four small-molecule regression tasks tested, the hybrid also gains on rare scaffolds, such that each half is searched separately, as seen in Table~\ref{tab:method:scaffold_halves}.
```
new_string:
```
When each half is searched separately, the hybrid is also ahead on the rare scaffolds of all four small-molecule regression datasets tested, with both models (Table~\ref{tab:method:scaffold_halves}). Each search here uses a subset of about 2,000 molecules, so its scores cannot be compared with those of Table~\ref{tab:hybrid_vs_best_single}.
```
Note: "half" (frequent 2,085 against rare 345 molecules on QM9) is not a half; consider "group".
Priority: low

### X-ch5-20
Location: line 2528 (calibration). Owner: hers
Quote: "These intervals are also narrower, for example on LogD with the SVM. On several small antibody datasets the hybrid's intervals cover too few held-out antibodies, fewer than the single representation's, for example on angiopoietin-2 binding and $\Delta\Delta G$ with the SVM."
His comment: How much narrower, how much too few? The table is in the Additional files.
Lens item: 6, 5
Proposed change (numbers from tab:method:conformal in additional_files_body.tex):
old_string:
```
These intervals are also narrower, for example on LogD with the SVM.
```
new_string:
```
These intervals are also narrower, for example 1.403 against 1.652 log units on LogD with the SVM.
```
old_string:
```
for example on angiopoietin-2 binding and $\Delta\Delta G$ with the SVM.
```
new_string:
```
for example on angiopoietin-2 binding and $\Delta\Delta G$ with the SVM. On angiopoietin-2 binding with the SVM they cover 0.720 of the held-out antibodies, against 0.804 for the single representation.
```
Priority: low

### X-ch5-21
Location: lines 2646-2659, Conclusions. Owner: hers
Quote: "On small molecules the hybrid works. With both models, it beat the best single representation on held-out molecules on every dataset, for example from 0.450 to 0.409 RMSE on LogD with the DNN."
His comment: Two datasets are drug-safety endpoints, hERG Ki and the Lhasa hERG blocking set. Say that the hybrid helps there.
Lens item: 14 (his TH-p112a asks this in the explainability paragraph; this is the chapter end)
Proposed change (numbers from tab:hybrid_vs_best_single):
old_string:
```
for example from 0.450 to 0.409 RMSE on LogD with the DNN.
```
new_string:
```
for example from 0.450 to 0.409 RMSE on LogD with the DNN. This includes two cardiac safety endpoints. On hERG $K_i$ with the SVM the error fell from 0.5881 to 0.5590 RMSE, and on hERG blocking with LightGBM the AUROC rose from 0.8555 to 0.8666.
```
Priority: low

### X-ch5-22
Location: line 1915 "\subsubsection{Comparisons against related tasks}" and line 2557 "\subsection{Extension studies}". Owner: hers
Quote: the two headings.
His comment: PCA and ensembles are related methods, not tasks. "Extension studies" does not say the aim.
Lens item: 11
Proposed change:
old_string:
```
\subsubsection{Comparisons against related tasks}
```
new_string:
```
\subsubsection{Comparisons against other ways of combining representations}
```
old_string:
```
\subsection{Extension studies}\label{sec:method:results_opig}
```
new_string:
```
\subsection{The hybrid on new targets, shared feature rows and reused settings}\label{sec:method:results_opig}
```
Priority: low

### X-ch5-23
Location: line 1314. Owner: hers
Quote: "Because only original features are kept, each traces back to one representation, which makes blueprints possible."
His comment: Blueprint has not been defined yet; it comes in the next section.
Lens item: 4
Proposed change:
old_string:
```
Because only original features are kept, each traces back to one representation, which makes blueprints possible.
```
new_string:
```
Because only original features are kept, each traces back to one representation. This makes possible the blueprints of the next section, which record how each hybrid was built.
```
Priority: low

### X-ch5-24
Location: line 1382. Owner: hers
Quote: "The global pool does not use these shares (see below)."
His comment: What is the global pool? It is introduced two pages later.
Lens item: 4
Proposed change:
old_string:
```
The global pool does not use these shares (see below).
```
new_string:
```
One of the five allocation methods, the global pool, does not use these shares (Section~\ref{sec:method:allocation}).
```
Priority: low

### X-ch5-25
Location: line 2463 and tab:method:importance_training. Owner: hers
Quote: "Feature scaling helps colloidal stability with the DNN most, and keeping the above-median features also helps $\Delta\Delta G$ with the SVM. Both seemingly carry a regularization effect."
His comment: Feature scaling helps SMAC with the DNN (0.261 to 0.357) but hurts it with the SVM (0.354 to 0.314), and hurts LogD with the SVM. Why the opposite effect per model? Also, are these importances computed per representation (not comparable, line 1411) or on the hybrid as one set?
Lens item: 18, 10
Proposed change: the second question is answered in code. tests/experiments/complexity/c19_importance_weighted_training.py line 90 computes importance on the hybrid's features as one matrix. Add to line 1943:
old_string:
```
Importance is computed on the training samples by the method the search picked.
```
new_string:
```
Importance is computed on the training samples by the method the search picked, in one fit on all of the hybrid's features together.
```
The model-dependence sentence is NEEDS HER (for example: for the SVM, scaling changes the kernel distance between molecules; for trees it changes nothing except dropping zero-importance features, line 1951).
Priority: low

### X-ch5-26
Location: line 1437, sec:method:allocation. Owner: hers
Quote: "All four assume that representations with stronger predictive performance contribute more valuable features. This assumption fails when a strong representation relies on a few select features."
His comment: Was this assumption measured? The even-split experiment seems to test it. Point to it.
Lens item: 10, 8
Proposed change:
old_string:
```
This assumption fails when a strong representation relies on a few select features.
```
new_string:
```
This assumption fails when a strong representation relies on a few select features. Section~\ref{sec:method:results_ensembles} tests it against an even split over the same representations.
```
Priority: low

### X-ch5-27
Location: line 2545, LogD blueprint. Owner: hers
Quote: "Among the bits are tertiary amine environments, in which a nitrogen is bonded to three carbons (\texttt{CN(C)C}), and aryl chloride environments."
His comment: An amide nitrogen is also bonded to three carbons, and it is not basic. Does this bit separate the basic amine from the amide?
Lens item: 7
Proposed change: NEEDS HER: check the atom environment of the bit (ECFP4 radius and the neighbor atoms) in the explainability output. If it covers only sp3 carbons, add "three sp3 carbons, which excludes amides". If not, soften "basicity ... from ECFP4 amine bits" in the same paragraph.
Priority: low

### X-ch5-28
Location: line 2559, activity-cliff targets. Owner: hers
Quote: "These targets have many activity cliffs, which is where a single similarity-based representation struggles most (Chapter~\ref{chap:flaws})."
His comment: The gain is small (0.519 to 0.514 MAE on dopamine D2) and factor Xa with LightGBM is worse. What is the take-home? And how were the grid and the eight representations chosen for these new targets?
Lens item: 2, 8
Proposed change: NEEDS HER: one sentence on how the grid was set for these three targets (from training molecules only?). Suggested take-home in place of the last sentence:
old_string:
```
These targets have many activity cliffs, which is where a single similarity-based representation struggles most (Chapter~\ref{chap:flaws}).
```
new_string:
```
These targets have many activity cliffs (Chapter~\ref{chap:flaws}). On them the hybrid lowers the error in five of six target and model pairs, mostly by a small amount.
```
Priority: low

### X-ch5-29
Location: line 1698, sec:method:reps. Owner: hers
Quote: "If an input cannot be parsed or embedded, it gets an all-zero row, so that every representation of a dataset keeps the same rows."
His comment: How many molecules got an all-zero row? They all share one row, so they look identical to every model and every distance.
Lens item: 21
Proposed change: NEEDS HER: the count per dataset and representation (or "none on any dataset"), added as one sentence after the quote.
Priority: low
