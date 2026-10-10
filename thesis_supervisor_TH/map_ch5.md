# Map of supervisor comments, old Chapter 5 (old PDF pp. 107-161)

Now `\chapter{Building adaptive hybrid representations}\label{chap:method}`, thesis.tex lines 1262-2660 (read 2026-10-10).
No Chapter 5 table is `\input`; every Chapter 5 table is written inline in thesis.tex. `figures_ch5_7/tab_hybrid_vs_best_single.tex` is the generator output (tests/analysis/thesis_figures.py:327-349) and has only 4 columns; the thesis copy at lines 2007-2065 has two more (Best in hindsight, Its score).
Highlight assignment: in comments_ch5.md, the highlights listed under a comment almost always belong to the NEXT comment (checked by y position and sense). The assignment below follows that.
No paragraph edited below carries the `% [AI text, revised 2026-10-10]` marker, so every edit is to her text and is kept minimal.

---

### TH-p107a
Comment: Here we could add one sentence to link it to the general topic of the thesis. For instance , will hybrid representation impact more the variance and benefit noise mitigation
Current location: lines 1266-1268, Section "Introduction" of chap:method
Owner: hers
Status: PARTLY
Reasoning: The noise link can be stated now: Chapter 7 asks whether noise handling helps a hybrid more than the best single representation (line 3399) and cleans the labels the search uses (sec:noise:res:search, line 4641). The variance part NEEDS HER: no experiment in thesis.tex measures whether a hybrid changes the variance split of Chapter 3 (searched "variance" in lines 2660-5000; only uncertainty-method hits).
Edit:
old_string:
```latex
Distance measures between hybrids are covered in Chapter~\ref{chap:similarity}, and noise detection and learning with noisy labels in Chapter~\ref{chap:noise}.
```
new_string:
```latex
Distance measures between hybrids are covered in Chapter~\ref{chap:similarity}, and noise detection and learning with noisy labels in Chapter~\ref{chap:noise}. Chapter~\ref{chap:noise} asks whether noise detection and noise mitigation work better in a hybrid than in the best single representation. It also cleans the labels that the search itself uses (Section~\ref{sec:noise:res:search}).
```

### TH-p108a
Comment: Maybe easier for the reader to restate the question here rather than force navigation.
Current location: line 1268, Section "Introduction" of chap:method (highlight p108 y=106-141 "The questions this chapter and Chapter 6 address are set out in Chapter 1")
Owner: hers
Status: APPLIES
Reasoning: The sentence still sends the reader to Chapter 1, and Chapter 1 (lines 14-24) lists only the four noise questions, not this chapter's. The new wording reuses the question as the Conclusions state it (line 2646) and Chapter 1's description of Chapter 6 (line 22).
Edit:
old_string:
```latex
The questions this chapter and Chapter~\ref{chap:similarity} address are set out in Chapter~\ref{chap:intro}.
```
new_string:
```latex
This chapter asks how to select the best set of features from several representations for a given dataset and model. It also asks whether the method carries over from small molecules to antibodies and reactions. Chapter~\ref{chap:similarity} asks whether distances computed in a hybrid are better than in a single representation.
```

### TH-p108b
Comment: This might be a bit contradicting the explainability goal as you previously mentioned that GNN might loose their link to structural features.
Current location: line 1276, last sentence, Subsection "Combining representations" (old highlight p108 y=398-434 "We also include GNN embeddings..."; that sentence is gone, replaced by the embeddings sentence quoted below). The trade-off is stated later, at line 1320: "This feature-level explainability fails when features are derived from learned embeddings."
Owner: hers
Status: PARTLY
Reasoning: Line 1320 admits the loss, but at line 1276, where embeddings are brought in, nothing names the cost. One added sentence there points forward to line 1320.
Edit:
old_string:
```latex
Although the features derived from embeddings are not as conceptually straightforward, they contain another axis of information.
```
new_string:
```latex
Although the features derived from embeddings are not as conceptually straightforward, they contain another axis of information. The cost is interpretability. A selected embedding feature can be traced to its representation, but not to a substructure or a descriptor, as discussed for blueprints below.
```

### TH-p108c
Comment: Indeed, the 'ensemble' of the models (with the target algorithm) should be a base line to compare with the hybrid-representation (on that same target algorithm) Is it better to merge the representation or the predictions?
Current location: lines 1921-1922 (Methods, "Ensembles"), lines 2078-2139 (sec:method:results_ensembles, tab:method:ensembles), line 2652 (Conclusions)
Owner: hers
Status: ALREADY ADDRESSED
Reasoning: The same-model ensemble is exactly the baseline he asks for, and the results answer his question per domain.
Edit: none. Current text: "The first ensemble uses the same downstream model on every representation, and the second uses each representation's own better model." (line 1922) and "We find that an ensemble of one model per representation beats the hybrid with both models on most antibody datasets tested, seen in Table~\ref{tab:method:ensembles}. ... On small molecules the hybrid holds up better" (line 2080).

### TH-p109a
Comment: That's an important point indeed and needs to be stressed as it is a driver of the added value of your contribution.
Current location: highlight p109 y=126-182 ("concatenation adds the irrelevant features of both parts along with the useful ones, so a method that joins representations has to combine and filter at the same time"). That sentence is gone. Its nearest home is line 1274 (dimensionality paragraph, Subsection "Combining representations").
Owner: hers
Status: APPLIES
Reasoning: The sentence he praised was cut. Putting it back at line 1274 stresses the point where the reader meets the problem. The old "[RESULT NEEDED: count of datasets]" claim is not restored; the Deng2023 count (9 of 24, line 1276) is the evidence already in the text.
Edit:
old_string:
```latex
With such ratios, more features mean weaker predictions \citep{Hughes1968, Bellman1961}.
```
new_string:
```latex
With such ratios, more features mean weaker predictions \citep{Hughes1968, Bellman1961}. Joining representations also adds the irrelevant features of every part along with the useful ones. So a method that joins representations has to combine and filter at the same time. The hybrid is built to do both.
```

### TH-p109b
Comment: Nice info !
Current location: line 1272, Subsection "Combining representations" (highlight p109 y=349-364 "Descriptor-based models held 73% of top-three positions on 11 benchmarks [158].")
Owner: hers
Status: PRAISE
Reasoning: Survived unchanged: "Descriptor-based models held 73\% of top-three positions on 11 benchmarks \citep{Jiang2021}."
Edit: none.

### TH-p109c
Comment: We could have investigated how many structure are merged across representation. This give an idea of the potential of breaking ties by using hybrid representations
Current location: line 1975 (Methods, sec:method:sites), lines 2579-2599 (sec:method:results_opig, tab:method:shared_rows). Duplicates his p106 comment.
Owner: hers
Status: ALREADY ADDRESSED
Reasoning: The tie count was run, and a hybrid built to break the ties was scored.
Edit: none. Current text: "None of the hybrids built on these datasets leaves two samples with the same row." (line 2579) and "Adding ECFP4 and PDV instead improves it almost as much while some molecules still share a row. Most of the gain therefore comes from the information the added representations give every molecule, and little from breaking the ties." (line 2582)

### TH-p109d
Comment: There is one aspect that we might not have covered here: Some representation fits better some ML algorithm and thus there maybe a value of pairing the best fit representation/learning-type and merge predictions (i.e. in such a case not mixing representation would be by design)
Current location: line 1922 (Methods, "Ensembles"); results lines 2080-2139
Owner: hers
Status: ALREADY ADDRESSED
Reasoning: The best-model ensemble pairs each representation with its own better model and pools the predictions, which is his proposal.
Edit: none. Current text: "The first ensemble uses the same downstream model on every representation, and the second uses each representation's own better model." (line 1922). Table tab:method:ensembles, column "Best-model ensemble".

### TH-p109e
Comment: Important distinction indeed!
Current location: old highlights p109 y=632-768 (ensembling combines after prediction; feature selection keeps original features unlike PCA). Now line 1280 ("Feature selection keeps a subset of the original features, unlike PCA or PLS which build new ones.") and line 1922 ("As the hybrid fits a single model and a fused feature vector, it could be considered a form of early fusion.")
Owner: hers
Status: PRAISE
Reasoning: Both distinctions survived. The old sentence "Consensus QSAR works this way \citep{Valsecchi2020}, so it cannot exploit cases where one representation's noise is another's signal" was cut; Valsecchi2020 survives only in Table tab:hybrid_alternatives.
Edit: none.

### TH-p110a
Comment: If we had time it would be nice to have a figure representing the different approaches in this paragraph and how hybrid representation are unique.
Current location: lines 1276-1306 (Subsection "Combining representations", Table tab:hybrid_alternatives)
Owner: hers
Status: NEEDS HER
Reasoning: A new figure. Table tab:hybrid_alternatives already compares the eight methods on four properties, so a figure would draw what the table lists. Her call whether to draw one.
Edit: none.

### TH-p111a
Comment: A pedant reviewer could argue that these are the same operations (selecting useful features implicitely means discarding useless features) ;-). The difference is in the selection criteria and they are complementary.
Current location: old sentence "Filtering out useless features matters as much as selecting useful ones" is gone. The same two-operation framing survives at line 1266 ("while filtering out the irrelevant ones").
Owner: hers
Status: PARTLY
Reasoning: Minimal change so the sentence presents keeping and leaving out as one choice.
Edit:
old_string:
```latex
we aim to select the most informative features across a larger set of representations, while filtering out the irrelevant ones.
```
new_string:
```latex
we aim to select the most informative features across a larger set of representations, and by the same choice to leave out the irrelevant ones.
```

### TH-p111b
Comment: Worth writing an actual definition of an adaptative hybrid representation. " Definition:…"
Current location: line 1314, first sentence, Subsection "The adaptive hybrid representation" (highlight p111 y=562-597 "Formally, an adaptive hybrid representation is ...")
Owner: hers
Status: APPLIES
Reasoning: The "Formally" sentence becomes a marked definition. Every fact in it is already in the text: budget as a maximum (lines 1350, 1402), search on training data (line 1312), per dataset and model (line 1314), traceability (line 1314). No definition environment is defined in preamble.tex, so the edit uses `\textit{Definition.}` in its own paragraph.
Edit:
old_string:
```latex
Formally, an adaptive hybrid representation is a set of features, capped at a fixed budget, selected from several base representations of the same molecules.
```
new_string:
```latex
\noindent\textit{Definition.} Take $K \ge 2$ base representations of the same samples, their labels, one downstream model and a budget $B$. An adaptive hybrid representation is a set of at most $B$ of the original features of these base representations. Its features are chosen by a hybrid search on the training samples, for that dataset and downstream model. Each kept feature stays linked to the base representation it came from.

```

### TH-p112a
Comment: "…especially in drug safety assessment when predicting and interpreting toxicity." (to reinforce the general context of this thesis)
Current location: old sentence "Explainability is a recognized need in drug discovery [175]" (highlight p112 y=621-656) is gone. Its home is now line 1322, Subsection "Blueprints and their chemical interpretation".
Owner: hers
Status: APPLIES
Reasoning: The old sentence is put back with his words added. Its citation was \citep{JimenezLuna2020} (thesis_rework/stage1_takeaways_bullets/thesis.tex.bak_before_abstract_2026-10-01). The key exists in refs.bib:1799 and is not cited anywhere else in thesis.tex now.
Edit:
old_string:
```latex
If the selected features make sense chemically for that endpoint, it's a good sign that the selection mechanism is finding real signal instead of fitting noise.
```
new_string:
```latex
If the selected features make sense chemically for that endpoint, it's a good sign that the selection mechanism is finding real signal instead of fitting noise. Explainability is a recognized need in drug discovery \citep{JimenezLuna2020}, especially in drug safety assessment when predicting and interpreting toxicity.
```

### TH-p114a
Comment: Does a bit feature in ECFP has the same budget weight as a phys-chem descriptor (e.g. logP) in representations like PDV?
Current location: line 1402 (budget), Subsection "The search space"; lines 1382-1384 (shares $b_k$ and top-$b_k$ per representation)
Owner: hers
Status: APPLIES
Reasoning: Yes. The budget counts features, one each. The allocation returns a feature count per representation (src/kirby/hybrid.py:1629-1659, `allocate_performance_weighted`, "Returns: dict: Allocation of {rep_name: n_features}", verified). Features are ranked only within their own representation (lines 1384, 1411), except under the global pool (line 1482). The text never says this outright.
Edit:
old_string:
```latex
The budget is the maximum number of features in the hybrid, taken from eight values (10, 25, 50, 100, 250, 500, 1,000 and 2,000 features).
```
new_string:
```latex
The budget is the maximum number of features in the hybrid, taken from eight values (10, 25, 50, 100, 250, 500, 1,000 and 2,000 features). Every feature counts once against the budget, whether it is an ECFP4 bit or a PDV descriptor such as logP. Bits and descriptors from different representations do not compete for the budget directly. Each representation's share is set first, and its features are ranked only against each other (Section~\ref{sec:method:allocation}). The global pool is the one exception.
```

### TH-p114b
Comment: Might be worth highlighting here, where the current validation and future test sets fits in this setup with respect to the data used for the hybrid search since we want to make sure there is no information leak,
Current location: Figure fig:hybrid_search_space, caption line 1344 (highlight p114 y=276-340, the Figure 5.1 caption). Partly answered by Figure fig:split_levels (lines 1357-1378: "Outer test fold: used once, to score the finished hybrid") and line 1386.
Owner: hers
Status: PARTLY
Reasoning: The image (thesis_rework/stage1_takeaways_bullets/figures_code_input/hybrid_search_space.png) shows only the search's five folds and no outer test fold. It still draws a "Pruning" box, but the per-fold searches run unpruned (tests/experiments/_shared.py:1419-1432, verified). The caption should say both. Redrawing the image NEEDS HER.
Edit:
old_string:
```latex
\caption[Schematic overview of the hybrid search.]{Schematic overview of the hybrid search. The inputs (left) are split into five cross-validation folds, and the single representations are scored as baselines. Each fold is searched over importance methods, allocation methods, budgets, internal evaluators and downstream models (center). The best hybrid for each downstream model is stored as a blueprint (right).}
```
new_string:
```latex
\caption[Schematic overview of the hybrid search.]{Schematic overview of the hybrid search on the training part of one outer fold. The outer test fold is set aside before the search starts and is not drawn (Figure~\ref{fig:split_levels}). The inputs (left) are split into five cross-validation folds, and the single representations are scored as baselines. Each fold is searched over importance methods, allocation methods, budgets, internal evaluators and downstream models (center). The best hybrid for each downstream model is stored as a blueprint (right). Pruning is drawn but was not used in this chapter's searches (Section~\ref{sec:method:pruning}).}
```

### TH-p120a
Comment: Very important reminder.
Current location: line 1435, Subsubsection "Feature allocation methods" (highlight p120 y=467-502)
Owner: hers
Status: PRAISE
Reasoning: Survived: "The allocation method decides how many of the $B$ features each representation contributes while the importance method decides which."
Edit: none.

### TH-p122a
Comment: Is this b_k in table 5.3?
Current location: line 1490 (highlight p122 y=292-307 "The internal evaluator's scores set how the budget is split"), also line 1486; Table 5.3 is now tab:alloc_methods (line 1459)
Owner: hers
Status: APPLIES
Reasoning: Yes. The internal evaluator's scores are the standalone scores $r_k$ (line 1380). They give $w_k$ and the share $b_k$ of Table tab:alloc_methods, and the representation count keeps the top N by them. Verified: per-representation allocation scores are keyed by internal evaluator (src/kirby/find_best_hybrid.py:4978-4983) and used for the top-N restriction (4985-4989); the allocation takes `internal_evaluator` and "Tests each representation individually ... then allocates budget" (src/kirby/hybrid.py:1631, 1640-1641). Each configuration is then scored with the downstream model (`_evaluate_hybrid_config`, find_best_hybrid.py:4738, fit at 5302, score at 5314). Lines 1486 and 1490 now say the evaluator scores "candidate feature sets" or "candidate hybrids". That disagrees with line 1380 and with the code, so three edits.
Edit 1:
old_string:
```latex
Within the search grid, we use an internal evaluator to fit on one candidate feature set at a time.
```
new_string:
```latex
Within the search grid, we use an internal evaluator to fit on one base representation at a time.
```
Edit 2:
old_string:
```latex
When the internal evaluator is set to \texttt{match}, the search scores each candidate hybrid on the same model type as the downstream model,
```
new_string:
```latex
When the internal evaluator is set to \texttt{match}, the search scores each base representation with the same model type as the downstream model,
```
Edit 3:
old_string:
```latex
The internal evaluator's scores determine how the budget is split and which representations are kept. The internal evaluator ranks candidate feature sets, it explicitly does not rank features within a representation. This process of selecting a feature set by fitting a model on it, and keeping the set that scores the best is known as the wrapper method \citep{Kohavi1997}.
```
new_string:
```latex
The internal evaluator's scores are the standalone scores $r_k$ (Section~\ref{sec:method:build}). They set each representation's share $b_k$ of the budget (Table~\ref{tab:alloc_methods}) and, through the representation count, which representations are kept. The internal evaluator does not rank features within a representation. Each finished hybrid is then scored by fitting the downstream model on it. Selecting a feature set by fitting a model on it, and keeping the set that scores best, is known as the wrapper method \citep{Kohavi1997}.
```

### TH-p122b
Comment: Augmentation may trigger the reader's 'Data augmentation' concept. Maybe the title could be related to the objective in something around "Handling small representations.."
Current location: line 1496 `\subsection{Augmentation}\label{sec:method:augmentation}`, lines 1498-1504 (highlight p122 y=720-732 "Augmentation")
Owner: hers
Status: APPLIES
Reasoning: The title changes and the label stays, so every `\ref` updates. "Small" alone is not accurate, because topological torsions (2,048 features) are a donor on QM9 (line 1842). One added sentence separates the idea from data augmentation. Mentions elsewhere: line 983 (Ch4, "augmentation features, the small feature sets added to the hybrid after it is built (Section~\ref{sec:method:augmentation})"), line 1350, and the caption of tab:dataset_reps at line 1829 all use `\ref` and read correctly with the new title. A comment in additional_files_body.tex:1662 names the old title ("Augmentation"). It is a comment only, so it needs no change. No other prose names the section title.
Edit 1:
old_string:
```latex
\subsection{Augmentation}\label{sec:method:augmentation}
```
new_string:
```latex
\subsection{Handling small and specialized feature sets}\label{sec:method:augmentation}
```
Edit 2:
old_string:
```latex
Therefore, on some datasets we add them as augmentation features, calling the finished hybrid $\mathbf{H}$ the base and the feature sets that supply extra features the donors.
```
new_string:
```latex
Therefore, on some datasets we add them to the hybrid as augmentation features. These are extra features for each sample, not the extra samples of data augmentation. We call the finished hybrid $\mathbf{H}$ the base and the feature sets that supply extra features the donors.
```

### TH-p123a
Comment: Why was the 2048 feature descriptor not part of the initial representation candidates?
Current location: highlight p123 y=288-323 (QM9 topological torsions as a donor). The sentence moved to Additional file app:add:augmentation (thesis_rework/stage1_takeaways_bullets/additional_files_body.tex:1666). The fact also stands in Table tab:dataset_reps, line 1842 (QM9 Augmentation column: "Topological torsions, ...").
Owner: hers
Status: NEEDS HER
Reasoning: Only she knows why torsions were a donor on QM9 but a base representation on Caco-2, LogD and hERG blocking (lines 1844, 1845, 1848). Line 1404 says the ten base representations were chosen "based on each representation's relative performance on the prediction task", and which data that was measured on is not stated (unverified). One fact for her: no search in this chapter uses augmentation (line 1504, caption line 1829), so torsions never enter a QM9 hybrid.
Edit: none until she gives the reason.

### TH-p124a
Comment: Have we measure the difference of the quality of the hybrid representation with and without donors? The augmention step feels a bit 'hacky' and might be challenged.
Current location: line 1504, sec:method:augmentation
Owner: hers
Status: ALREADY ADDRESSED
Reasoning: The comparison was made, and augmentation was dropped from this chapter's searches.
Edit: none. Current text: "An earlier search of each whole dataset covered 16 datasets of an earlier list. On none did either augmented condition beat no augmentation by more than 1\% (3\% on datasets under about 1,000 samples). We therefore ran the searches in this chapter without augmentation on all 20 datasets (Section~\ref{sec:method:decided_grid})."

### TH-p124b
Comment: Could the title be more related to the aim of this section (e.g. Pruning search space)
Current location: line 1507 `\subsection{Adaptive pruning}\label{sec:method:pruning}` (old title "Fast Tier and Adaptive Pruning"; the fast tier now lives in sec:method:decided_grid, line 1525)
Owner: hers
Status: APPLIES
Reasoning: The title changes and the label stays. Mentions elsewhere: line 1523 ("adaptively prune the search space (Section~\ref{sec:method:pruning})") is a `\ref` and reads correctly. The caption edit in TH-p114b adds one more `\ref`. Additional file text at additional_files_body.tex:1473 says "with adaptive pruning off", which is the method name, not the title, so no change. No prose anywhere names the old title.
Edit:
old_string:
```latex
\subsection{Adaptive pruning}\label{sec:method:pruning}
```
new_string:
```latex
\subsection{Pruning the search space}\label{sec:method:pruning}
```

### TH-p126a
Comment: This simple statement explains well the reason for the full/fast tiers and could be put as an introduction of the section.
Current location: the old sentence ("Configurations far behind after two folds would only confirm their loss on the other three, so pruning stops scoring them", highlight p124/p126) is gone. The section is lines 1507-1519, and her restored paragraph is line 1510.
Owner: hers
Status: APPLIES
Reasoning: His sentence opens the section. A second sentence says that this chapter's per-fold searches did not prune. Verified: `search_on_fold` drops any pruning arguments and runs unpruned ("Pruning is off in every per-fold search (Adelaide, 2026-10-05)", tests/experiments/_shared.py:1419-1432), and every Chapter 5 search goes through it (fold_standard_selection -> fold_searches -> search_on_fold, _shared.py:1750, 1522). Line 1523 already says the per-dataset grid was used instead. Her paragraph is unchanged.
Edit:
old_string:
```latex
Adaptive pruning: every configuration is first scored on the first two folds.
```
new_string:
```latex
Configurations far behind after two folds would only confirm their loss on the other three, so adaptive pruning stops scoring them. The per-fold searches reported in this chapter ran without it and used a smaller grid per dataset instead (Section~\ref{sec:method:decided_grid}).

Adaptive pruning: every configuration is first scored on the first two folds.
```

### TH-p128a
Comment: So this remains an internal validation using a comparison hybrid vs best base representation (in full). Given the numerous hyperparameter in the search space, is there not a risk of datasnooping since the hybrid-search operated on the same data as the comparison. Should we not have an external (to the hybrid-search) test dataset to run the full base representions baselines and compare with the selected hybrid-representation ? Unless I missintepreted figure 5.1 where the selection process is outside the CV box. We can discuss this point in the meeting (Maybe that was done and I missed it or maybe it comes late though)
Current location: line 1605 (sec:method:evaluation; old highlight p128 y=470-687), also lines 1352, 1386, 1601, 2301-2303, Figure fig:split_levels (1357-1378)
Owner: hers
Status: PARTLY
Reasoning: Facts from the code:
(1) Each search sees only its outer fold's training rows. Every matrix, label and group is sliced to `train_idx` before `find_best_hybrid` (tests/experiments/_shared.py:1438-1441, verified).
(2) The hybrid is chosen on training rows. It is the top configuration by the search's own score (_shared.py:1782-1798), then rebuilt on the training rows and scored once on the held-out rows (tests/experiments/complexity/c02_complexity_grid.py:291-303, 158-181, verified).
(3) The best single representation is chosen the same way. It is ranked by `search.baselines`, the per-fold search's own scores on training rows (_shared.py:1720-1725, 1774), then refit on the training rows and scored on the same held-out rows (c02_complexity_grid.py:277-287, 113-130, verified).
(4) No test set exists outside the outer folds. Every reported score is an outer-fold score (c02_complexity_grid.py:485-516, verified).
(5) One choice did see the held-out molecules: each dataset's grid, including its two downstream models, budgets, methods and evaluators. It came from a search of the whole dataset. DECISIONS.md:3 reads "Scores are from the whole-dataset search (leaky, accepted)" (scratchpad/precompute_cuts/DECISIONS.md, an uncommitted dated document). decided_grid.json is built from it (tests/experiments/build_decided_grid.py:1-20), and the whole-dataset search is the one not sliced to train_idx (_shared.py:1349-1352 docstring). Lines 1523 and 1870 say "earlier search(es) of the whole dataset", but they do not say that this search included the held-out molecules.
(6) How the ten base representations per dataset were picked (line 1404, "relative performance on the prediction task"), and on which molecules, is not stated. Unverified.
The text already says (1)-(3) at lines 1386 and 1605. It does not say (4) or (5), and Figure fig:hybrid_search_space shows no outer fold, which is what misled him (see TH-p114b). NEEDS HER: whether to hold out an external test set, and (6).
Edit:
old_string:
```latex
A positive value means the hybrid scored better, and for RMSE we flip the sign so that this still holds.
```
new_string:
```latex
A positive value means the hybrid scored better, and for RMSE we flip the sign so that this still holds.

No test set exists outside these outer folds. Each reported score is on an outer test fold that the search of that fold never saw. The hybrid's settings and the best single representation are both chosen from scores on the outer training samples only. One choice was made before these searches. Each dataset's grid, including its two downstream models, came from an earlier search of the whole dataset (Section~\ref{sec:method:decided_grid}). That earlier search included the samples that the outer folds later hold out.
```

### TH-p149a
Comment: What is a 'default hybrid' ?
Current location: the term is gone (no "default hybrid" in thesis.tex or additional_files_body.tex). The old 3-fold local runs on hERG Ki with one fixed configuration were replaced by the 5-fold searches. The successor of old Table 5.15 is tab:hybrid_vs_best_single (lines 2007-2065).
Owner: hers
Status: PARTLY
Reasoning: The term is gone. The new table's caption still does not say which hybrid it shows. The hybrid is `hybrid1`, the configuration with the best search score on each fold's training rows (tests/analysis/thesis_loader_f1.py:81, 98-100; _shared.py:1788-1798; c02_complexity_grid.py:292-303, verified). The same caption text is generated at tests/analysis/thesis_figures.py:342-344, so change it there too. The thesis copy is inline and has two more columns than the generator output.
Edit (thesis.tex, line 2063):
old_string:
```latex
\caption{Held-out score of the best single base representation and the hybrid, per dataset and model, mean over outer folds, in each dataset's own metric.
```
new_string:
```latex
\caption{Held-out score of the best single base representation and the hybrid, per dataset and model, mean over outer folds, in each dataset's own metric. On each fold, the hybrid is the configuration with the best score in that fold's search on the training samples. It is rebuilt on those samples and scored on the held-out fold. The best single base representation is picked and scored the same way.
```
Generator: in tests/analysis/thesis_figures.py:342-344, add the same three sentences after "in each dataset's own metric." in the caption string of `write_pair_table(..., 'tab_hybrid_vs_best_single', ...)`.

### TH-p150a
Comment: We can see that hybrid outperforms the best base representation in these combinations. We could add variability in the Mean and use bold for the better RMSE.
Current location: old Table 5.15 is gone. Its successor tab:hybrid_vs_best_single (thesis.tex lines 2007-2065, inline) gives means only. Generator: tests/analysis/thesis_figures.py:327-349 (fig_f1) with write_pair_table (293-324) and _bold_rows (269-284).
Owner: hers
Status: NEEDS HER
Reasoning: Bold: the four-bold-cells rule (MAX_BOLD = 4, thesis_figures.py:256) rules out bolding the better score in all 40 rows. The table now bolds the four largest gains. Variability: the data exist. c02 records, per model, the held-out min and max over folds (`_over_folds`, c02_complexity_grid.py:133-144, 633-636, verified). The loader keeps only the mean (thesis_loader_f1.py:92-108). The one-number-per-cell rule rules out a "min-max" cell. Her choice is between two extra columns (fold minimum, fold maximum) added through the loader and write_pair_table, and a per-fold table in an Additional file.
Edit: none until she chooses.

### TH-p152a
Comment: Nice summary table !
Current location: old Table 5.16 (48 allocation settings at budgets 500 and 1,000, local 3-fold hERG Ki runs) is gone from thesis.tex and from additional_files_body.tex (no "Joint ranking"/"Own ranking"). The nearest successors are Table tab:method:top_rep_even_split (lines 2144-2201) and Additional file app:add:top_rep_even_split.
Owner: hers
Status: PRAISE
Reasoning: The praised table did not survive.
Edit: none.

### TH-p153a
Comment: Landscape clipped
Current location: old Table 5.17 (representation count 2 or 3 x allocation settings) is gone. The only landscape table in the chapter now is tab:decided_grid (sidewaystable, lines 1552-1594).
Owner: hers
Status: GONE
Reasoning: The clipped table was removed. tab:decided_grid renders whole on page 118 of scratchpad/thesis_code_todo/build/main.pdf (built 2026-10-10 18:38; checked by rendering the page), and main.log has no overfull box for it. Its columns are tight, because `\setlength{\tabcolsep}{3pt}` (line 1555) leaves "Budgets" and "Importance" nearly touching.
Edit: none.

### TH-p155a
Comment: Interesting how the hybrid search reveals a representation that generalises better (better behaviour at low number of features)
Current location: old Table 5.20 (nanobody thermostability, searched hybrid vs PCA, recursive elimination and random features at 10, 50 and 250 features) is gone. The comparison now lives in Figure fig:best_number_of_features_antibodies (lines 2444-2449) and line 2372.
Owner: hers
Status: PRAISE
Reasoning: The praised numbers survive only as the thermostability panel of a figure, and the image was not checked here. Line 2372 says mRMR is "close to it on thermostability" and does not mention the low-budget behavior he noticed.
Edit: none.

### TH-p157a
Comment: Are these result compatible with table 5.15 where the best base in RF seemed to perform less well than the hybrid search? Here in RF and XGBoost the baest base seems to outperform the hybrid representation.
Current location: old Table 5.22 is now tab:method:settings_transfer (lines 2604-2616), SVM and DNN only. Old Table 5.15 is now tab:hybrid_vs_best_single (lines 2026-2027 for hERG Ki).
Owner: hers
Status: PARTLY
Reasoning: Random forest and XGBoost are in neither table now. In both tables, the hybrid beats the best single representation on hERG Ki with the SVM and the DNN: SVM 0.5590 against 0.5881 and 0.563 against 0.589; DNN 0.5691 against 0.6031 and 0.567 against 0.604. The numbers differ slightly between the two tables. The transfer searches use only the seven representations hERG Ki shares with LogD (line 1969; from the thesis text, not checked in code), whereas the main searches use all ten (line 1843). The caption should say so.
Edit:
old_string:
```latex
the LogD settings and features reused unchanged. Held-out RMSE in p$K_i$ units, lower is better, mean over five outer folds.}
```
new_string:
```latex
the LogD settings and features reused unchanged. Held-out RMSE in p$K_i$ units, lower is better, mean over five outer folds. These searches draw on the seven representations that hERG $K_i$ shares with LogD, so the scores differ from Table~\ref{tab:hybrid_vs_best_single}, whose searches draw on all ten.}
```

### TH-p158a
Comment: Nice!
Current location: old Table 5.26 (per-fold DNN RMSE for the LogD-settings transfer) is gone. Only the five-fold means remain, in tab:method:settings_transfer (lines 2604-2616).
Owner: hers
Status: PRAISE
Reasoning: The praised per-fold table did not survive.
Edit: none.
