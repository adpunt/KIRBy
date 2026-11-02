# Review: Chapter 6 (chap:similarity), after the style pass

Compared with thesis.tex.before_style_apply_2026-10-10 by a paragraph-level word diff, and read in full as it stands on 2026-10-10. Every prose paragraph, table and caption in the chapter carries a "% TODO: manual review" marker in the backup, so the whole chapter is AI-written; her text here is only the chapter, section and subsection titles (the title changes already carry "edited your text" notes). her_text_check.py has no hit in this chapter (its one hit mentioning Chapter 6 is outside it and has a note). Every number in the chapter's prose and in every NEW block below is in the backup thesis.tex. Every \ref in the chapter resolves (the two table labels exist twice only because the placeholder inside \IfFileExists repeats them). Every OLD block was checked with python3 -I str.count to occur exactly once in thesis.tex.

The style pass changed little here: "activity" to "label" in the Methods, sentence-case titles, a parent Results section, four sentence splits. The Results and Conclusions prose kept its rhetorical questions, colon reveals, slogan closers and "activity" for non-activity labels. Two factual errors in the results text were not caught (V12, V13).

## Fixes

### V1 — Introduction opens with Chapter 4, not the previous chapter
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: The chapter should open by linking to Chapter 5. The paragraph now opens with Chapter 4 and reaches Chapter 5 in sentence six. "The original idea behind it was simple:" is a colon reveal, and "This chapter returns to that idea." does not say what the chapter does.
OLD:
```latex
Chapter~\ref{chap:flaws} showed where the similarity assumption fails on standard representations. In most of them, the distance between two molecules barely rises with the difference in their labels. Any method that trusts a molecule's neighbors inherits this problem. Two examples are predicting a molecule's label from its nearest neighbors and finding a noisy label by comparing it with its neighbors' labels. Chapter~\ref{chap:method} introduced the adaptive hybrid representation, a set of features selected from several base representations for one task. The original idea behind it was simple: select the features relevant to the task first, then compute distances on those features alone. This chapter returns to that idea.
```
NEW:
```latex
Chapter~\ref{chap:method} introduced the adaptive hybrid representation, a set of features selected from several base representations for one task, and used it as input to a model. Chapter~\ref{chap:flaws} showed where the similarity assumption fails on standard representations. In most of them, the distance between two molecules barely rises with the difference in their labels. Any method that trusts a molecule's neighbors inherits this problem. Two examples are predicting a molecule's label from its nearest neighbors and finding a noisy label by comparing it with its neighbors' labels. The first idea behind the hybrid was to select the features relevant to the task and then compute distances on those features alone. This chapter tests that idea.
```

### V2 — Introduction: "this distance" refers to nothing
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: "this distance" has no antecedent in the paragraph (the same fault as style guide item 19 in Chapter 1). "is the step this chapter leaves open, and we take it up" is padding.
OLD:
```latex
Using this distance to find noisy labels is the step this chapter leaves open, and we take it up in Chapter~\ref{chap:noise}.
```
NEW:
```latex
Using distance in a hybrid to find noisy labels is left to Chapter~\ref{chap:noise}.
```

### V3 — Prediction from nearest neighbors (Methods): 31-word sentence with a colon reveal
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: Over 30 words; the colon introduces the reveal.
OLD:
```latex
So the two sides differ in two ways at once: the hybrid side picks from five or six times as many combinations, and only the hybrid side can use importance weighting.
```
NEW:
```latex
So the two sides differ in two ways at once. The hybrid side picks from five or six times as many combinations, and only the hybrid side can use importance weighting.
```

### V4 — Results subsection title still says "activity" for all 13 datasets
- file: thesis.tex
- kind: author-title
- todo_line: replaces the existing title note (shown in NEW)
- why: Chemistry guide item 17 names this title. Only the two hERG datasets measure an activity (the chapter says so itself in the cliff Methods paragraph). The text under the title already says "label differences".
OLD:
```latex
% TODO: manual review -- title was: Results: Does Distance in a Hybrid Track Activity Better -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; question (with no question mark) made a topic, in the words of the matching Methods paragraph "Distance against activity."
\subsection{Distance against activity difference}\label{sec:similarity:results_similarity}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Does Distance in a Hybrid Track Activity Better -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; question (with no question mark) made a topic, in the words of the matching Methods paragraph; "activity" changed to "label", because only the two hERG datasets measure an activity (QM9, LogD, Caco-2, the antibody sets and the reaction yields do not)
\subsection{Distance against label difference}\label{sec:similarity:results_similarity}
```

### V5 — Methods paragraph heading: same term as V4
- file: thesis.tex
- kind: ai-text
- todo_line: none needed (heading belongs to the AI-marked paragraph under it)
- why: Chemistry guide item 17; keeps the heading and the Results title the same.
OLD:
```latex
\paragraph{Distance against activity.}
```
NEW:
```latex
\paragraph{Distance against label difference.}
```

### V6 — Importance weighting (Methods): experiment name, first mention
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: Same experiment name as V4 and V5.
OLD:
```latex
Distance against activity refits LightGBM gain
```
NEW:
```latex
Distance against label difference refits LightGBM gain
```

### V7 — Importance weighting (Methods): experiment name, second mention
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: Same as V6.
OLD:
```latex
\item for distance against activity and the best number of features,
```
NEW:
```latex
\item for distance against label difference and the best number of features,
```

### V8 — First results paragraph: "disagree a lot"
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: Informal and vague. The numbers were cut to a table by another session, so none is re-added.
OLD:
```latex
The folds also disagree a lot, for example on hERG $K_i$.
```
NEW:
```latex
The value also varies widely between folds, for example on hERG $K_i$.
```

### V9 — Weak correlations: "a different story:" colon reveal
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: "a different story" is the informal register; the colon is a reveal. "sits" for a correlation is loose.
OLD:
```latex
None of the small-molecule representations sits at or below zero, and neither does any on Buchwald--Hartwig. The antibody datasets are a different story: on EGFR binding, CHO polyreactivity, AC-SINS self-association and SMAC colloidal stability, at least one representation does.
```
NEW:
```latex
On the small-molecule datasets and Buchwald--Hartwig, no representation has a correlation at or below zero. On EGFR binding, CHO polyreactivity, AC-SINS self-association and SMAC colloidal stability, at least one representation does.
```

### V10 — Distance hybrid result: one-line opener
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: "Building a hybrid for distance changes this." is a dramatic one-line sentence; the next sentence already says it.
OLD:
```latex
Building a hybrid for distance changes this. In the distance hybrid, distance tracks
```
NEW:
```latex
In the distance hybrid, distance tracks
```

### V11 — Predictive hybrids: slogan closer
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: "If the question is about distance, the hybrid has to be built for distance." is a slogan. Say which search the hybrid has to come from.
OLD:
```latex
If the question is about distance, the hybrid has to be built for distance.
```
NEW:
```latex
A hybrid used for distance therefore has to come from the distance search.
```

### V12 — Antibody prediction: four datasets change sign, not one
- file: thesis.tex
- kind: ai-text (factual error)
- todo_line: keep the existing markers
- why: The paragraph's own lists show four sign changes. At the fixed setting the hybrid loses on binding free energy and wins on nanobody thermostability, CHO polyreactivity and expression titer (Table tab:nearest_neighbour_prediction: 0.1424 to 0.0899; 0.5005 to 0.5458; 0.2059 to 0.3591; 0.2521 to 0.2622). With the setting picked again it wins on binding free energy and loses on the other three. Naming only CHO polyreactivity hides the other three. "less clear-cut" is vague.
OLD:
```latex
The other domains are less clear-cut. At the fixed setting
```
NEW:
```latex
On the antibody datasets and Buchwald--Hartwig, the result depends on the comparison. At the fixed setting
```

### V13 — Same paragraph: the sign-change sentence
- file: thesis.tex
- kind: ai-text (factual error)
- todo_line: covered by V12
- why: See V12.
OLD:
```latex
CHO polyreactivity therefore changes sign between the two comparisons.
```
NEW:
```latex
Binding free energy, nanobody thermostability, CHO polyreactivity and expression titer therefore change sign between the two comparisons.
```

### V14 — Picked-again comparison: "six times" contradicts the Methods
- file: thesis.tex
- kind: ai-text (factual error)
- todo_line: keep the existing markers
- why: The Methods give 216 against 36 and 144 against 24 (six times), but 180 against 36 where the similarity hybrid is undefined (five times), and the Methods sentence says "five or six times". "does favor" is emphatic; the sentence is 33 words.
OLD:
```latex
This comparison does favor the hybrid in two ways, since its side picks from six times as many combinations and only its side may use importance weighting (Section~\ref{sec:similarity:experiments}).
```
NEW:
```latex
This comparison favors the hybrid in two ways. Its side picks from five or six times as many combinations, and only its side may use importance weighting (Section~\ref{sec:similarity:experiments}).
```

### V15 — Opening question of the nearest-neighbor results
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: Rhetorical question as paragraph opener. The reference theses state the result.
OLD:
```latex
Does a better distance give better predictions from neighbors? At the setting fixed before the run,
```
NEW:
```latex
At the setting fixed before the run,
```

### V16 — Importance weighting result misses three losses shown in its own table
- file: thesis.tex
- kind: ai-text (factual error)
- todo_line: keep the existing markers
- why: Table tab:importance_weighting shows weighting also makes the hybrid worse on SMAC colloidal stability (0.2418 to 0.2339 Spearman), CHO polyreactivity (0.3591 to 0.3574) and Buchwald--Hartwig (16.1811 to 16.5264 RMSE). The text names only expression titer and the binding free energy single representation.
OLD:
```latex
It hurts on expression titer, and in the single representation of binding free energy.
```
NEW:
```latex
It hurts on expression titer, in the single representation of binding free energy, and in the hybrid on SMAC colloidal stability, CHO polyreactivity and Buchwald--Hartwig.
```

### V17 — Table tab:importance_weighting: bold misses three cells; Buchwald--Hartwig unit; "counted by rank"
- file: thesis.tex
- kind: ai-table (caption and cell formatting; no number changed)
- todo_line: keep "% TODO: manual review (table and caption)"
- why: The caption says bold marks where weighting makes prediction worse, but 0.2339, 0.3574 and 16.5264 are worse than the unweighted value beside them and are not bold. The caption gives no unit for the Buchwald--Hartwig RMSE (chemistry guide item 29; Chapter 5's caption at the tuning table uses "Buchwald--Hartwig in yield (\%)"). "neighbors counted by rank" was changed in the text to "weighted by the inverse of their rank" and should match.
OLD:
```latex
SMAC colloidal stability & Spearman $\uparrow$ & 0.1268 & 0.1706 & 0.2418 & 0.2339 \\
CHO polyreactivity & Spearman $\uparrow$ & 0.2059 & 0.2713 & 0.3591 & 0.3574 \\
```
NEW:
```latex
SMAC colloidal stability & Spearman $\uparrow$ & 0.1268 & 0.1706 & 0.2418 & \textbf{0.2339} \\
CHO polyreactivity & Spearman $\uparrow$ & 0.2059 & 0.2713 & 0.3591 & \textbf{0.3574} \\
```

### V18 — Same table: Buchwald--Hartwig row bold
- file: thesis.tex
- kind: ai-table
- todo_line: covered by V17
- why: See V17.
OLD:
```latex
Buchwald--Hartwig yield & RMSE $\downarrow$ & 14.4572 & 14.2584 & 16.1811 & 16.5264 \\
```
NEW:
```latex
Buchwald--Hartwig yield & RMSE $\downarrow$ & 14.4572 & 14.2584 & 16.1811 & \textbf{16.5264} \\
```

### V19 — Same table: caption
- file: thesis.tex
- kind: ai-caption
- todo_line: covered by V17
- why: See V17. The caption stays at four sentences after the change, with one added for the unit.
OLD:
```latex
One row is one dataset; values are means over five outer folds, Manhattan distance, neighbors counted by rank. The arrow gives the direction of the metric. Bold marks the datasets where weighting makes prediction worse.}
```
NEW:
```latex
One row is one dataset; values are means over five outer folds, Manhattan distance, neighbors weighted by the inverse of their rank. The arrow gives the direction of the metric; Buchwald--Hartwig in yield (\%). Bold marks a weighted value worse than the unweighted value beside it.}
```

### V20 — Joined representations: "works"
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: "works" says nothing about what was measured.
OLD:
```latex
Weighting also works on all representations joined.
```
NEW:
```latex
Weighting also improves prediction from all representations joined.
```

### V21 — "So when does the hybrid win?" rhetorical question
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: Rhetorical question opener; "binding free energy loses" makes a dataset the loser.
OLD:
```latex
So when does the hybrid win? It depends on the dataset much more than on how the neighbors are found. Averaged over the other settings, the gain on LogD is similar for each distance and number of neighbors, and binding free energy loses under all three distances.
```
NEW:
```latex
Whether the hybrid wins depends on the dataset much more than on how the neighbors are found. Averaged over the other settings, the gain on LogD is similar for each distance and number of neighbors, and the hybrid loses on binding free energy under all three distances.
```

### V22 — Same paragraph: "more say"; antibodies called molecules
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: "giving the features that matter more say" is idiom. Binding free energy and AC-SINS self-association are antibody datasets, so "training molecules" is wrong there (chemistry guide item 19).
OLD:
```latex
Part of what the hybrid does for a distance is therefore what weighting does, giving the features that matter more say. Where the hybrid loses, the spread across folds is larger than the gain, for example on binding free energy and AC-SINS self-association. These datasets have a few hundred training molecules, against one to eight thousand for the small-molecule datasets,
```
NEW:
```latex
Part of what the hybrid does for a distance is therefore what weighting does, giving more weight to the features that matter. Where the hybrid loses, the spread across folds is larger than the gain, for example on binding free energy and AC-SINS self-association. These datasets have a few hundred training antibodies, against one to eight thousand training molecules for the small-molecule datasets,
```

### V23 — Retrieval by setting: colon reveal and dataset as loser
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: "and that is the distance:" is a colon reveal; "binding free energy loses" as in V21.
OLD:
```latex
and binding free energy loses under all three distances. There too, the spread across folds is larger than the loss at the setting fixed before the run. One setting does matter on the antibody datasets, and that is the distance: cosine distance gives the hybrid its smallest lead.
```
NEW:
```latex
and the hybrid loses on binding free energy under all three distances. There too, the spread across folds is larger than the loss at the setting fixed before the run. On the antibody datasets the distance does matter. Cosine distance gives the hybrid its smallest lead.
```

### V24 — Bayesian optimization: "seed sets" against "starting sets"
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: The Methods and the experiments table say "three starting sets"; two words for one thing.
OLD:
```latex
on hERG $K_i$ and Buchwald--Hartwig only, over three seed sets.
```
NEW:
```latex
on hERG $K_i$ and Buchwald--Hartwig only, from three starting sets.
```

### V25 — Activity cliffs: antibodies called molecules; bare fraction
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: Chemistry guide item 19 names this sentence. "11 of 429" has no unit.
OLD:
```latex
On the antibody datasets too few held-out molecules count as cliff molecules for a score, for example 11 of 429 over five folds on binding free energy.
```
NEW:
```latex
On the antibody datasets too few held-out antibodies count as cliff antibodies for a score. On binding free energy, for example, 11 of the 429 antibodies held out over five folds do.
```

### V26 — QM9 cliffs: 34-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: Over 30 words, opens with "But".
OLD:
```latex
But in each model the hybrid is better in one of those two folds and worse in the other, and with 48 cliff molecules in all we do not read a conclusion from QM9.
```
NEW:
```latex
In each model, the hybrid is better in one of those two folds and worse in the other. With 48 cliff molecules in all, we draw no conclusion from QM9.
```

### V27 — Limitation paragraph inside Results; move it to the Conclusions
- file: thesis.tex
- kind: ai-text (moved)
- todo_line: keep "% TODO: manual review" and add the comment shown in NEW
- why: Decision: a chapter's limitations go inside its Conclusions. This paragraph sits in the cliff results; its first sentence is 33 words. Its content goes into the limitations paragraph in V31.
OLD:
```latex
A limitation runs through this section. Held-out molecules are called cliff molecules by a looser rule than training molecules, so the share of cliff molecules differs between the two, and on QM9 too few held-out molecules qualify to read. The hybrid and the single representation are scored on the same cliff molecules, so the comparison between them is not affected.
```
NEW:
```latex
% (2026-10-10: this paragraph was moved into the limitations paragraph of Section~\ref{sec:similarity:conclusions}.)
```

### V28 — Retrieval: colon question and "activity"
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: Colon reveal with a question; "activity" for all datasets (chemistry guide item 17).
OLD:
```latex
Retrieval asks a simpler question: are a molecule's nearest neighbors similar in activity?
```
NEW:
```latex
Retrieval measures whether a molecule's nearest neighbors have labels close to its own.
```

### V29 — Conclusions, first paragraph: "repairs much of"; "activity"; cliffs not answered
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing markers
- why: "repairs much of what ... found wrong" is appraisal, not a finding. "Distance tracks activity" covers QM9, LogD and Caco-2 (chemistry guide item 17). Research question 2 names activity cliffs, and the Conclusions never give the cliff result; the added sentence restates Section sec:similarity:results_cliffs with no new number.
OLD:
```latex
On the five small-molecule datasets, a hybrid built for distance repairs much of what Chapter~\ref{chap:flaws} found wrong with distance in standard representations. Distance tracks activity more closely on all five (Table~\ref{tab:distance_tracks_activity}).
```
NEW:
```latex
Chapter~\ref{chap:flaws} found that distance in standard representations barely rises with the difference in labels. On the five small-molecule datasets, distance in a hybrid built for distance tracks label differences more closely than in the single representation (Table~\ref{tab:distance_tracks_activity}).
```

### V30 — Conclusions: cliff result added after the retrieval sentence
- file: thesis.tex
- kind: ai-text
- todo_line: covered by V29
- why: See V29.
OLD:
```latex
Neighbors retrieved in the hybrid have closer labels on all five.
```
NEW:
```latex
Neighbors retrieved in the hybrid have closer labels on all five. On LogD, Caco-2, hERG $K_i$ and hERG blocking, each model's first hybrid also predicts cliff molecules better than its best single representation.
```

### V31 — Conclusions, second paragraph: "does not fix everything"; "activity"; QM9 cliffs moved
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: "The hybrid does not fix everything." is a dramatic one-liner. "tracks activity" as in V29. The QM9 cliff sentence moves to the limitations paragraph (V33).
OLD:
```latex
The hybrid does not fix everything. Weighting features by importance does part of the same work without any selection, and on LogD the weighted single representation tracks activity better than the unweighted hybrid. Distance in the hybrid tracks prediction error worse on hERG blocking, and the kept features carry no more label information than principal components of the same size. Activity cliffs on QM9 could not be judged at all, because only 48 of its held-out molecules are cliff molecules.
```
NEW:
```latex
Weighting features by importance does part of the same work without any selection. On LogD the weighted single representation tracks label differences better than the unweighted hybrid. Distance in the hybrid tracks prediction error worse on hERG blocking, and the kept features carry no more label information than principal components of the same size.
```

### V32 — Conclusions, third paragraph: "activity", 36-word sentence, colon and "not X" contrast
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: "Distance tracks activity" covers antibody properties and yields. Two sentences over 30 words. "the central claim of this thesis: ... not a fix that holds everywhere" is a colon reveal plus a rhetorical contrast.
OLD:
```latex
On the antibody datasets and Buchwald--Hartwig, the answer depends on the dataset. Distance tracks activity more closely on all of them. With the setting picked again, however, prediction from nearest neighbors is worse in the hybrid on nanobody thermostability, AC-SINS self-association, CHO polyreactivity, expression titer and Buchwald--Hartwig, and retrieval is worse on binding free energy and Buchwald--Hartwig. These losses fit the central claim of this thesis: the right input depends on the data, and a hybrid is an input chosen for one dataset, not a fix that holds everywhere.
```
NEW:
```latex
On the antibody datasets and Buchwald--Hartwig, the result depends on the dataset. Distance in the hybrid tracks label differences more closely on all of them. With the setting picked again, prediction from nearest neighbors is worse in the hybrid on nanobody thermostability, AC-SINS self-association, CHO polyreactivity, expression titer and Buchwald--Hartwig. Retrieval is worse on binding free energy and Buchwald--Hartwig. These losses agree with the central claim of this thesis, that the right input depends on the data. A hybrid is selected for one dataset, so its benefit has to be measured on each dataset.
```

### V33 — Limitations paragraph: take in the cliff limitation; "activity"
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review" and the TODO(missing result) comment
- why: Receives V27 and the QM9 cliff sentence from V31, so all limitations sit in the Conclusions. "distance against activity" as in V4.
OLD:
```latex
Three limitations apply. QM9 has no result for the best number of features, importance weighting in the four models, or choosing which molecules to measure next. The weighted comparison of distance against activity is missing on Caco-2, hERG blocking, hERG $K_i$, expression titer and SMAC colloidal stability. And the comparison with the setting picked again lets the hybrid side choose among more combinations and use importance weighting, so it favors the hybrid.
```
NEW:
```latex
Four limitations apply. QM9 has no result for the best number of features, importance weighting in the four models, or choosing which molecules to measure next. The weighted comparison of distance against label difference is missing on Caco-2, hERG blocking, hERG $K_i$, expression titer and SMAC colloidal stability. The comparison with the setting picked again lets the hybrid side choose among more combinations and use importance weighting, so it favors the hybrid. Held-out molecules are called cliff molecules by a looser rule than training molecules, so the share of cliff molecules differs between the two. The hybrid and the single representation are scored on the same cliff molecules, so their comparison is not affected. On QM9 only 48 held-out molecules are cliff molecules, too few to judge.
```

### V34 — Last paragraph: 38-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep the existing "% TODO: manual review"
- why: Over 30 words. The link to Chapter 7 stays.
OLD:
```latex
Methods that find noisy labels by comparing each label with its neighbors' labels rest on the same distance we studied here, so Chapter~\ref{chap:noise} tests whether the hybrid's distance makes noisy labels easier to find and to act on.
```
NEW:
```latex
Methods that find noisy labels by comparing each label with its neighbors' labels rest on the same distance studied here. Chapter~\ref{chap:noise} tests whether the hybrid's distance makes noisy labels easier to find and to act on.
```

## Checks that passed
- No separate Discussion section; the chapter runs Introduction, Methods, Results, Conclusions; titles in sentence case; no "Results:" prefix left.
- Conclusions point to Chapter 7 (V34 keeps it). Chapter 5's last paragraph points to this chapter.
- Captions of tab:knn_settings and tab:similarity_experiments open with a noun phrase, 2 to 4 sentences; neither reports a metric.
- No significance tests; no patient data; no new numbers.

## For the author
- The generated tables tab:distance_tracks_activity and tab:nearest_neighbour_prediction (figures_ch5_7/) have 14 rows, including MDR1--MDCK efflux. The chapter text says 13 datasets and "all five small-molecule datasets" throughout. Decide whether MDR1--MDCK belongs in Chapter 6; every "five" and "13" in the chapter and the experiments table depends on it.
- The same two generated captions say "between two samples" and "Bold: a dataset where the hybrid is lower, at most four"; they are written by the figure script, so the figures session has to change them.
- The placeholder captions inside \IfFileExists (print only if the table file is missing) still say "absolute activity difference" and "neighbors counted by rank". Left alone with the \input lines.
- "A second run with the small-molecule models flags the same cliff molecules" (cliff results): which models, and what the second run is, is not said anywhere in the chapter. Needs your wording or a code check.
- Manhattan choice: the text says "an earlier comparison of four distances", but the source comment above it lists three (Manhattan, cosine, Euclidean). "never much worse than second" is vague; on the comment's numbers Manhattan ranks second on all three scores. Check the fourth distance before rewording.
- $K_i$ typography (chemistry guide item 10) is a whole-thesis change through one macro, not a Chapter 6 edit.
- Style guide item 28 (numbers in prose, no figure): the TABLE NEEDED slots in this chapter remain; tables are not created in this pass.
- Chapter 8's "Distance still tracked activity more closely on all of them" (chemistry guide item 17) is outside this unit.

## Applied
- V1 applied as written.
- V2 applied as written.
- V3 applied as written.
- V4 applied as written (title and its note).
- V5 applied as written.
- V6 applied as written.
- V7 applied as written.
- V8 applied as written.
- V9 applied as written.
- V10 applied as written.
- V11 applied as written.
- V12 applied as written (checked against tab:importance_weighting unweighted columns: four sign changes).
- V13 applied as written.
- V14 applied as written.
- V15 applied as written.
- V16 applied as written (checked against tab:importance_weighting: 0.2418 to 0.2339, 0.3591 to 0.3574, 16.1811 to 16.5264).
- V17 applied as written.
- V18 applied as written.
- V19 applied, adapted: unit written as "Buchwald--Hartwig RMSE is in percentage points of yield", the wording of the Chapter 5 tuning caption (tab:tuning_base_reps) and Chapter 7; "Buchwald--Hartwig in yield (\%)" is not in thesis.tex.
- V20 applied as written.
- V21 applied as written.
- V22 applied as written.
- V23 applied as written.
- V24 applied as written.
- V25 applied as written.
- V26 applied as written.
- V27 applied as written ("% TODO: manual review" kept above the move comment).
- V28 applied as written.
- V29 applied as written.
- V30 applied as written.
- V31 applied as written.
- V32 applied as written.
- V33 applied as written.
- V34 applied as written.
