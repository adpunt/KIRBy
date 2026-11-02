# Plan: Chapter 6 (Similarity in an Adaptive Hybrid Representation, chap:similarity)

Audited against THESIS_STYLE_GUIDE.md and CHEMISTRY_LANGUAGE_GUIDE.md on 2026-10-10. Every OLD block was checked with python3 -I to be an exact substring occurring once in thesis.tex.

Every prose paragraph and every caption in this chapter carries a "% TODO: manual review" marker, so all of it is AI text and no "edited your text" comment is needed. The existing TODO lines stay where they are, outside the OLD blocks. No section title, \paragraph heading or \label is changed.

Two choices run through the plan:

- "Activity" is changed to "label" (or "label difference") wherever a sentence covers QM9, LogD, Caco-2, antibodies or Buchwald--Hartwig (chemistry guide item 17). It stays where the sentence is about hERG only, in "activity cliff", and in the experiment names that repeat a \paragraph heading ("distance against activity").
- $K_i$ is left as it is in every NEW block. Changing it only in edited paragraphs would mix two forms in one chapter; the chapter-wide change is listed under "For the author" (chemistry guide item 10).

The chapter opening already looks back to Chapters 4 and 5 and states the two questions. The conclusions already end by pointing to Chapter 7. No transition edits are needed (style guide item 1 lists Chapter 6 as already linked).

## Edits

### E1 — Introduction: split a 32-word sentence; "activity" to "label"
- file: thesis.tex
- kind: ai-text
- why: Chapter 4 measured the label difference on datasets whose labels are not activities (chemistry guide item 17); one sentence over 30 words.
- todo_line: keep existing
OLD:
```latex
Chapter~\ref{chap:flaws} showed where the similarity assumption fails on standard representations. In most of them, the distance between two molecules barely rises with the difference in their activity. Any method that trusts a molecule's neighbors inherits this problem, from predicting a molecule's activity from its nearest neighbors to finding a noisy label by comparing it with its neighbors' labels.
```
NEW:
```latex
Chapter~\ref{chap:flaws} showed where the similarity assumption fails on standard representations. In most of them, the distance between two molecules barely rises with the difference in their labels. Any method that trusts a molecule's neighbors inherits this problem. Two examples are predicting a molecule's label from its nearest neighbors and finding a noisy label by comparing it with its neighbors' labels.
```

### E2 — Introduction: research question 1 says "label", not "activity"
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 17 names this question; it covers QM9, LogD, Caco-2, antibodies and reaction yield, which are not activities.
- todo_line: keep existing
OLD:
```latex
In the second, the search ranks each candidate hybrid by how well each molecule's activity is predicted from its five nearest neighbors in it, and no downstream model is fitted. We ask two research questions:
\begin{enumerate}
    \item Does distance in a hybrid track activity better than distance in the best single representation?
```
NEW:
```latex
In the second, the search ranks each candidate hybrid by how well each molecule's label is predicted from its five nearest neighbors in it, and no downstream model is fitted. We ask two research questions:
\begin{enumerate}
    \item Does distance in a hybrid track label differences better than distance in the best single representation?
```

### E3 — Introduction: datasets outnumber datasets, not antibodies molecules
- file: thesis.tex
- kind: ai-text
- why: "antibodies outnumber small molecules seven to five" compares dataset counts but names the samples; wrong as written.
- todo_line: keep existing
OLD:
```latex
Because antibodies outnumber small molecules seven to five, we give every result by domain rather than pooled over all datasets.
```
NEW:
```latex
Because antibody datasets outnumber small-molecule datasets seven to five, we give every result by domain rather than pooled over all datasets.
```

### E4 — Methods: say once that "molecule" also covers antibodies and reactions
- file: thesis.tex
- kind: ai-text
- why: The chapter writes "molecule" for every row of 13 datasets, seven of them antibody sets and one a reaction set (chemistry guide section 1 and item 19); one sentence fixes it without rewriting about 60 uses.
- todo_line: keep existing
OLD:
```latex
In the experiments, but not inside the searches, held-out values are also cut to the training range, so a single extreme molecule cannot dominate a distance.
```
NEW:
```latex
In the experiments, but not inside the searches, held-out values are also cut to the training range, so a single extreme molecule cannot dominate a distance. Where a statement covers all 13 datasets, ``molecule'' also stands for an antibody or a reaction.
```

### E5 — Feature spaces: split a 31-word sentence
- file: thesis.tex
- kind: ai-text
- why: Sentence over 30 words.
- todo_line: keep existing
OLD:
```latex
The second, the distance search, uses a k-nearest-neighbor model as its downstream model, with Euclidean distance and five equally weighted neighbors (fewer when a fold has under 25 training molecules). It has its own lists of importance methods, allocation methods, numbers of representations and budgets. Its budgets are capped at 500 features, and on some datasets its other lists differ from the general search's too. On hERG $K_i$, for example,
```
NEW:
```latex
The second, the distance search, uses a k-nearest-neighbor model as its downstream model, with Euclidean distance and five equally weighted neighbors. It uses fewer neighbors when a fold has under 25 training molecules. It has its own lists of importance methods, allocation methods, numbers of representations and budgets. Its budgets are capped at 500 features, and on some datasets its other lists differ from the general search's too. On hERG $K_i$, for example,
```

### E6 — Similarity search: "activity" to "label"
- file: thesis.tex
- kind: ai-text
- why: The score $q_5$ is computed on every dataset's label (chemistry guide item 17).
- todo_line: keep existing
OLD:
```latex
In the similarity search, $s(S) = q_5(S)$, how accurately each molecule's activity is predicted from its five nearest neighbors in $S$ (Equation~\ref{eq:similarity_objective}).
```
NEW:
```latex
In the similarity search, $s(S) = q_5(S)$, how accurately each molecule's label is predicted from its five nearest neighbors in $S$ (Equation~\ref{eq:similarity_objective}).
```

### E7 — Neighbor weights: split a 34-word sentence
- file: thesis.tex
- kind: ai-text
- why: Sentence over 30 words.
- todo_line: keep existing
OLD:
```latex
A later comparison on four datasets, at Manhattan distance with 5 neighbors, found them better than equal weights in all 48 dataset and representation pairs, but worse than inverse squared distance in 39 of the 48.
```
NEW:
```latex
A later comparison on four datasets used Manhattan distance with 5 neighbors. There, inverse-distance weights beat equal weights in all 48 pairs of dataset and representation, but lost to inverse squared distance in 39 of the 48.
```

### E8 — Ranking candidates: split a 31-word sentence; "scaffolds" to "scaffold groups"
- file: thesis.tex
- kind: ai-text
- why: Sentence over 30 words; the groups are Bemis--Murcko scaffold groups, and "scaffold group" is the term the chemistry guide asks for (item 14), as this chapter's own Methods use it.
- todo_line: keep existing
OLD:
```latex
To rank whole candidates, we compute the score on the held-out molecules of each fold of the search, inside the outer training fold, and these molecules act as each other's neighbors. Their groups (scaffolds, for molecules) are split into up to 5 parts, and each molecule's 5 nearest neighbors are drawn from the other parts. A neighbor with the same scaffold is an easy match and would inflate the score.
```
NEW:
```latex
To rank whole candidates, we compute the score on the held-out molecules of each fold of the search, inside the outer training fold. These molecules act as each other's neighbors. Their groups (scaffold groups, for small molecules) are split into up to 5 parts, and each molecule's 5 nearest neighbors are drawn from the other parts. A neighbor from the same scaffold group is an easy match and would inflate the score.
```

### E9 — Ginkgo datasets: antibodies, not molecules
- file: thesis.tex
- kind: ai-text
- why: The four Ginkgo datasets hold antibodies (chemistry guide item 19).
- todo_line: keep existing
OLD:
```latex
Inside an outer fold, each held-out block of the search is then a single group, so no molecule has a neighbor outside its own group.
```
NEW:
```latex
Inside an outer fold, each held-out block of the search is then a single group, so no antibody has a neighbor outside its own group.
```

### E10 — Caption of the nearest-neighbor settings table: define the symbols
- file: thesis.tex
- kind: caption
- why: Style guide item 16: a caption states its symbols; $\mathbf{s}$, $\bar{s}$ and $a$ appear in the table but are defined only in later text.
- todo_line: keep existing
OLD:
```latex
\caption[Nearest-neighbor settings]{Nearest-neighbor settings. One row is one setting, with its levels; every combination of levels is scored. Minkowski distance is left out on datasets above 5,000 molecules (hERG blocking, LogD and QM9).}
```
NEW:
```latex
\caption[Nearest-neighbor settings]{Nearest-neighbor settings. One row is one setting, with its levels; every combination of levels is scored. In the last row, $\mathbf{s}$ is the vector of feature importances, $\bar{s}$ its mean and $a$ an exponent set by the distance. Minkowski distance is left out on datasets above 5,000 molecules (hERG blocking, LogD and QM9).}
```

### E11 — Importance weighting: split a 33-word sentence
- file: thesis.tex
- kind: ai-text
- why: Sentence over 30 words.
- todo_line: keep existing
OLD:
```latex
Under Manhattan distance, a feature with four times the average importance counts four times as much as an average feature under the first formula, but only twice as much under the second.
```
NEW:
```latex
Take a feature with four times the average importance, under Manhattan distance. The first formula makes it count four times as much as an average feature, and the second only twice as much.
```

### E12 — Experiments table, first row: "activity" to "label"
- file: thesis.tex
- kind: ai-text
- why: The row covers all 13 datasets, most of whose labels are not activities (chemistry guide item 17). Only the words change; no number or column.
- todo_line: keep existing
OLD:
```latex
Does distance track activity? & All 13 & Feature space; distance; importance weighting & Up to 50,000 held-out pairs per fold & Spearman between pair distance and absolute activity difference &
```
NEW:
```latex
Does distance track label differences? & All 13 & Feature space; distance; importance weighting & Up to 50,000 held-out pairs per fold & Spearman between pair distance and absolute label difference &
```

### E13 — Distance against activity, definition: "activity" to "label"
- file: thesis.tex
- kind: ai-text
- why: The experiment is run on all 13 datasets (chemistry guide item 17).
- todo_line: keep existing
OLD:
```latex
For one feature space, this experiment estimates the Spearman correlation between the distance of two held-out molecules and the absolute difference in their activity. A positive value means that molecules further apart differ more in activity.
```
NEW:
```latex
For one feature space, this experiment estimates the Spearman correlation between the distance of two held-out molecules and the absolute difference in their labels. A positive value means that molecules further apart differ more in their labels.
```

### E14 — Prediction from nearest neighbors: "activity" to "label"; say what "counted by rank" means
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 17; "counted by rank" does not say that the neighbors are weighted by 1/rank (Table tab:knn_settings).
- todo_line: keep existing
OLD:
```latex
We predict each held-out molecule's activity from its nearest training molecules (Equation~\ref{eq:knn_weights}), and make two comparisons. The first was fixed before the run, so nothing in it is picked on the folds it is scored on. It compares the distance hybrid against the single representation, both unweighted, under Manhattan distance with 5 neighbors counted by rank.
```
NEW:
```latex
We predict each held-out molecule's label from its nearest training molecules (Equation~\ref{eq:knn_weights}), and make two comparisons. The first was fixed before the run, so nothing in it is picked on the folds it is scored on. It compares the distance hybrid against the single representation, both unweighted, under Manhattan distance with 5 neighbors weighted by the inverse of their rank.
```

### E15 — Importance weighting in four models: split a 37-word sentence
- file: thesis.tex
- kind: ai-text
- why: Sentence over 30 words.
- todo_line: keep existing
OLD:
```latex
These are prediction from nearest neighbors at every setting, a Gaussian process with an RBF kernel, a support vector regressor with an RBF kernel, and each downstream model given four extra features from its nearest training molecules, following q-RASAR \citep{Banerjee2022}.
```
NEW:
```latex
The first three are prediction from nearest neighbors at every setting, a Gaussian process with an RBF kernel and a support vector regressor with an RBF kernel. The fourth is each downstream model given four extra features from its nearest training molecules, following q-RASAR \citep{Banerjee2022}.
```

### E16 — Activity cliffs: say the term is kept for non-activity labels; define a cliff molecule
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 18 (QM9, LogD and Caco-2 labels are not activities, so say the term is borrowed) and section 3 (define "cliff molecule" as a molecule in at least one cliff pair); the last sentence was also 32 words.
- todo_line: keep existing
OLD:
```latex
An activity cliff is a pair of structurally similar molecules with very different activity \citep{Stumpfe2012,vanTilborg2022}. We decide which molecules are cliff molecules in each outer fold, from its training molecules. We keep the structural side of the rule of \citet{Hu2015}: two molecules are similar when their ECFP4 Tanimoto similarity is at least 0.55. We modify its activity side. A gap of two log units flags no molecule at all on Caco-2, so instead we solve for the gap that puts 15\% of the training molecules in a similar pair whose labels differ by more than it.
```
NEW:
```latex
An activity cliff is a pair of structurally similar molecules with very different activity \citep{Stumpfe2012,vanTilborg2022}. Only the two hERG datasets measure an activity, but we keep the term for the other three small-molecule datasets. We decide which molecules are cliff molecules in each outer fold, from its training molecules. We keep the structural side of the rule of \citet{Hu2015}: two molecules are similar when their ECFP4 Tanimoto similarity is at least 0.55. We modify its activity side. A gap of two log units flags no molecule at all on Caco-2. Instead, we choose the gap so that 15\% of the training molecules are cliff molecules. A cliff molecule is one that is in at least one similar pair whose labels differ by more than the gap.
```

### E17 — Retrieval: "activity" to "label"
- file: thesis.tex
- kind: ai-text
- why: Retrieval is scored on all 13 datasets (chemistry guide item 17).
- todo_line: keep existing
OLD:
```latex
Retrieval measures how close in activity a molecule's neighbors are.
```
NEW:
```latex
Retrieval measures how close a molecule's neighbors are to it in label.
```

### E18 — Three smaller checks: "activity" to "label"; name Bayesian optimization
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 17; the Introduction calls this experiment Bayesian optimization and the Methods never use that name, so the two do not visibly match.
- todo_line: keep existing
OLD:
```latex
At each budget we record, for that same hybrid, its prediction score and the correlation of distance against activity, and find the budget where each peaks. On hERG $K_i$ and Buchwald--Hartwig only, we simulate choosing which molecules to measure.
```
NEW:
```latex
At each budget we record, for that same hybrid, its prediction score and the correlation between distance and label difference, and find the budget where each peaks. On hERG $K_i$ and Buchwald--Hartwig only, we simulate choosing which molecules to measure, a form of Bayesian optimization.
```

### E19 — First results paragraph: "activity difference" to "label difference"
- file: thesis.tex
- kind: ai-text
- why: The sentence covers QM9, LogD and Caco-2 (chemistry guide item 17).
- todo_line: keep existing
OLD:
```latex
Distance in a standard representation does rise with activity difference, but only weakly.
```
NEW:
```latex
Distance in a standard representation does rise with the difference in labels, but only weakly.
```

### E20 — Informal phrase "worst offender"
- file: thesis.tex
- kind: ai-text
- why: Style guide item 2 lists "the worst offender" (this sentence) as a phrase none of the references would print.
- todo_line: keep existing
OLD:
```latex
SMAC colloidal stability is the worst offender.
```
NEW:
```latex
SMAC colloidal stability has the weakest correlations.
```

### E21 — Distance hybrid result: point to the table instead of listing all ten numbers
- file: thesis.tex
- kind: ai-text
- why: Task item 6 and style guide item 3: all ten values are in Table tab:distance_tracks_activity; keep the hERG $K_i$ pair, which the Conclusions repeat; chemistry guide item 17.
- todo_line: keep existing
OLD:
```latex
Building a hybrid for distance changes this. In the distance hybrid, distance tracks activity more closely on all five small-molecule datasets (Table~\ref{tab:distance_tracks_activity}). Under Manhattan distance with no weighting, QM9 goes from 0.1298 to 0.3779 Spearman, LogD from 0.0765 to 0.2687 and hERG $K_i$ from 0.0729 to 0.2625. The gain is smaller on Caco-2, 0.1517 to 0.1969, and on hERG blocking, 0.0257 to 0.0552.
```
NEW:
```latex
Building a hybrid for distance changes this. In the distance hybrid, distance tracks label differences more closely on all five small-molecule datasets (Table~\ref{tab:distance_tracks_activity}). On hERG $K_i$, under Manhattan distance with no weighting, the correlation goes from 0.0729 to 0.2625 Spearman. The gain is smaller on Caco-2 and hERG blocking than on the other three.
```

### E22 — Predictive hybrids: one name for them; "activity" to "label"
- file: thesis.tex
- kind: ai-text
- why: Methods define "predictive hybrids"; this paragraph calls them "model hybrids" and "some model hybrid" (inconsistent term); chemistry guide item 17; the second sentence split to stay under 30 words.
- todo_line: keep existing
OLD:
```latex
The predictive hybrids are built for each downstream model rather than for distance, and some of them track activity less closely than their model's best single representation. On Caco-2 the support vector machine's first hybrid scores 0.1794 against 0.1843, and some model hybrid does the same on EGFR binding, expression titer, SMAC colloidal stability and Buchwald--Hartwig. On CHO polyreactivity all four model hybrids are lower,
```
NEW:
```latex
The predictive hybrids are built for each downstream model rather than for distance, and some of them track label differences less closely than their model's best single representation. On Caco-2 the support vector machine's first hybrid scores 0.1794 against 0.1843. At least one predictive hybrid does the same on EGFR binding, expression titer, SMAC colloidal stability and Buchwald--Hartwig. On CHO polyreactivity all four predictive hybrids are lower,
```

### E23 — Hindsight comparison: split a 38-word sentence; "activity" to "label"
- file: thesis.tex
- kind: ai-text
- why: Sentence over 30 words; chemistry guide item 17.
- todo_line: keep existing
OLD:
```latex
Here it is the one the nearest-neighbor model ranked first for prediction, not the one whose distance best follows activity. If we instead pick that representation with hindsight on the held-out molecules, under any of the distances, the distance hybrid is lower on Caco-2, 0.1969 against 0.2292, although the similarity hybrid on Caco-2 is higher at 0.2769.
```
NEW:
```latex
Here it is the one the nearest-neighbor model ranked first for prediction, not the one whose distance best follows the label. Suppose we instead pick that representation with hindsight on the held-out molecules, under any of the distances. Then the distance hybrid is lower on Caco-2, 0.1969 against 0.2292, although the similarity hybrid is higher at 0.2769.
```

### E24 — Best number of features: remove a sentence that describes the old experiment
- file: thesis.tex
- kind: ai-text
- why: The Methods ("At each budget we record, for that same hybrid, its prediction score and the correlation …") and the source comment above them (changed 2026-10-09: both curves from the same hybrid on the same budgets, LogD Spearman 0.2315, 0.1875, 0.1535 at 500, 1,000, 2,000) say distance is measured on the prediction budgets; the deleted sentence says distance runs over 100 to 500, which was the earlier design. Not re-checked against s08 output in this session.
- todo_line: keep existing
OLD:
```latex
The number of features matters too, because a hybrid predicts best at more features than its distance tracks activity best. On LogD the network's hybrid goes from 0.4146 RMSE at 500 features to 0.4076 at 2,000, while distance in the same hybrid falls from 0.2315 to 0.1535 Spearman. Part of this gap is built into the experiment: on LogD prediction is swept over 500 to 2,000 features and distance over 100 to 500, so the two can only meet at 500.
```
NEW:
```latex
The number of features matters too, because a hybrid's prediction score peaks at more features than its distance correlation does. On LogD the network's hybrid goes from 0.4146 RMSE at 500 features to 0.4076 at 2,000, while distance in the same hybrid falls from 0.2315 to 0.1535 Spearman.
```

### E25 — Prediction from neighbors, fixed setting: point to the table
- file: thesis.tex
- kind: ai-text
- why: Task item 6 and style guide item 3: all ten values are in Table tab:nearest_neighbour_prediction; keep the hERG $K_i$ pair (repeated in the Conclusions) and the largest gain.
- todo_line: keep existing
OLD:
```latex
At the setting fixed before the run, predicting each molecule from its five nearest neighbors is more accurate in the distance hybrid on all five small-molecule datasets (Table~\ref{tab:nearest_neighbour_prediction}). QM9 goes from 0.6692 to 0.4205 RMSE, LogD from 0.6766 to 0.5831 and hERG $K_i$ from 0.6280 to 0.5822. Caco-2 goes from 0.3386 to 0.3246 RMSE, and hERG blocking from 0.8143 to 0.8335 AUROC. The hybrid is better in every fold on LogD, and worse in one fold of five on hERG $K_i$ and Caco-2.
```
NEW:
```latex
At the setting fixed before the run, predicting each molecule from its five nearest neighbors is more accurate in the distance hybrid on all five small-molecule datasets (Table~\ref{tab:nearest_neighbour_prediction}). On hERG $K_i$ the RMSE goes from 0.6280 to 0.5822, and the largest gain is on QM9, from 0.6692 to 0.4205. The hybrid is better in every fold on LogD, and worse in one fold of five on hERG $K_i$ and Caco-2.
```

## For the author (not changed)
- $K_i$ typography: all 137 uses of `$K_i$` in thesis.tex stay as they are; chemistry guide item 10 asks for `$K_\mathrm{i}$` everywhere through one macro, which is a whole-thesis change, not a Chapter 6 edit.
- Section title "Results: Does Distance in a Hybrid Track Activity Better" and the paragraph heading "Distance against activity." still say "activity" for all 13 datasets (chemistry guide item 17); titles are a separate task.
- Table caption of tab:distance_tracks_activity still says "their absolute activity difference"; the caption is generated by the figure script, so change it there.
- Table caption of tab:nearest_neighbour_prediction still says "neighbors counted by rank" (E14 changes the same phrase in the text to "weighted by the inverse of their rank").
- E24 removes the sentence "Part of this gap is built into the experiment: on LogD prediction is swept over 500 to 2,000 features and distance over 100 to 500, so the two can only meet at 500." The source comment above the Methods paragraph says both curves now come from the same hybrid on the same budgets; s08_optimal_dimension.py's docstring still also sweeps distance-search budgets, so check which of the two the LogD numbers come from.
- "Distance still tracked activity more closely on all of them" in Chapter 8 (chemistry guide item 17) is outside this chapter.

## Checker notes
- All 25 OLD blocks re-checked with python3 -I: each occurs exactly once in thesis.tex.
- All 25 edits sit in paragraphs, table rows or captions with a "% TODO: manual review" marker above them, so kind ai-text/caption is right; no author-text edits.
- No new number in any NEW block that is absent from thesis.tex; no banned phrasing; no \label changed.
- Reverted `$K_\mathrm{i}$` to `$K_i$` in E5, E18, E21, E25 and the header: changing it only in edited paragraphs would mix two forms in the chapter. Moved to "For the author".
- E16: the new last sentence was 33 words; split into two.
- E22: the second sentence grew to 31 words; split into two.
- E24: reworded the awkward clause "than those at which its distance best tracks the label" to "than its distance correlation does".
- E25: the plan file ended mid-sentence inside the NEW block, and had no "For the author" section. Completed E25 NEW (keeps the hERG $K_i$ pair, the QM9 pair and the fold sentence, all from the OLD text) and added the section above.
- Kept unchanged: E1-E4, E6-E15, E17, E19-E21, E23.
