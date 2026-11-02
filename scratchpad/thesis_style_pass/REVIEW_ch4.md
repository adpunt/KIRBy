# Review: Chapter 4 (chap:flaws), after the style pass

Compared with thesis.tex.before_style_apply_2026-10-10 by diff (chapter at backup lines 1272-1765, current 1352-1875) and read in full as it stands on 2026-10-10. her_text_check.py on the chapter gives two hits: #1, the chapter title put in sentence case with no "title was" note (real, V1); #2, the recovery-rate pointer at current line 1790, which has its note one comment line higher (false positive). Every OLD block below was checked with python3 -I str.count to occur exactly once in thesis.tex. No NEW text adds a number: 0.55, 15%, 95% and the QM9/Gaussian facts are already in this chapter.

Checks that passed: the chapter runs Introduction, Methods, Results, Conclusions, with no Discussion section and no "Results:" prefix. Every section title is in sentence case. The opening links to the previous chapters and the Conclusions point to Chapter 5. Every number in the chapter's prose is in the backup. Representation's share "at most 8.1% on QM9" matches Table tab:variance (8.1, 7.0, 7.9 across the three conditions). Every \ref resolves; app:add:flaws resolves through the Additional files build, as in the backup. Chemistry guide items 21, 22 and the italic *R*/*S* half of 27 are done; item 18 (cliffs on QM9) is done by "We keep the term activity cliff ..."; item 7 is right in this chapter (0.44 mean, 0.54 standard deviation). Style guide gap items 22 (title case) and 43 (stale pointers to Chapter 6) are done. Each edit to her text made in this pass has an "edited your text" note, and each fixes something wrong (contraction, second person, stale pointer, stereochemistry default, SPP wording, AC undefined), except where listed below.

Not done: gap item 34 (no limitations paragraph in Conclusions, V8); gap item 42 (sections cited by quoted title at four places, V9-V12); gap item 27 (a 64-word sentence of hers, V13); chemistry guide item 26 (V15); one caption without a direction (V14).

## Fixes

### V1 — Chapter title changed with no note
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- title was: Limitations of the Similarity Assumption -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
- why: The pass changed "Similarity Assumption" to "similarity assumption" and added no note. Every other chapter title it changed carries a "title was" note (for example Chapter 5 at line 1875).
OLD:
```latex
\chapter{Limitations of the similarity assumption}\label{chap:flaws}
```
NEW:
```latex
% TODO: manual review -- title was: Limitations of the Similarity Assumption -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\chapter{Limitations of the similarity assumption}\label{chap:flaws}
```

### V2 — New Methods section has no label
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review -- new section: ..." line kept
- why: V3 needs a label to point at. The other Methods sections in the thesis carry one (sec:robustness:methods, sec:similarity:methods, sec:noise:methods).
OLD:
```latex
because each of those paragraphs also carries its numbers.
\section{Methods}
```
NEW:
```latex
because each of those paragraphs also carries its numbers.
\section{Methods}\label{sec:flaws:methods}
```

### V3 — "(next subsection)" now points backwards
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: review, needs manual review -- simplified rewrite ..." line kept
- why: The distance definitions moved from after this paragraph to the Methods section before it, so "next subsection" is wrong. The Methods section is two sections earlier.
OLD:
```latex
and all others by Manhattan distance on standardized features (next subsection), with constant features removed.
```
NEW:
```latex
and all others by Manhattan distance on standardized features (Section~\ref{sec:flaws:methods}), with constant features removed.
```

### V4 — Cliff gap: a claim the same paragraph contradicts
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" line kept
- why: The sentence says the gap "follows the spread of the labels". Three sentences later the gap is 0.06 times the median absolute deviation on QM9 and 0.50 to 0.60 times on the Biogen datasets. So it does not follow the spread. The sentence also has two "so" clauses in a row.
OLD:
```latex
We chose the gap on each dataset so that 15\% of the molecules are cliff molecules, so the gap follows the spread of the labels (Table~\ref{tab:flaws:distance_summary}).
```
NEW:
```latex
We chose the gap on each dataset so that 15\% of the molecules are cliff molecules (Table~\ref{tab:flaws:distance_summary}).
```

### V5 — "The useful information sits in the few closest pairs"
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" line kept
- why: "Useful information sits in" is a metaphor that says nothing measured. The paragraph measured label differences by similarity bin. Say that. The "whereas" contrast becomes two plain sentences.
OLD:
```latex
The useful information sits in the few closest pairs. The all-pairs correlation is set by the other 95\%, whereas the five-neighbor prediction uses only the closest pairs.
```
NEW:
```latex
Label differences are small only among the closest pairs. The all-pairs correlation is set mostly by the other 95\% of pairs. The five-neighbor prediction uses only the closest pairs.
```

### V6 — Neighbor overlap: 32-word sentence, bare fractions
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" line kept
- why: Over 30 words. "0.47 to 0.52 of each molecule's five neighbors" does not say it is a fraction, and the second number range has no subject.
OLD:
```latex
On hERG $K_i$ any two of the three share on average 0.47 to 0.52 of each molecule's five neighbors, and on the three Biogen datasets and QM9 only 0.13 to 0.21 (Table~\ref{tab:flaws:neighbor_overlap}).
```
NEW:
```latex
On hERG $K_i$, any two of the three representations share on average a fraction of 0.47 to 0.52 of each molecule's five neighbors (Table~\ref{tab:flaws:neighbor_overlap}). On the three Biogen datasets and QM9 the fraction is 0.13 to 0.21.
```

### V7 — Conclusions: two unrelated clauses joined by a semicolon
- file: thesis.tex
- kind: ai-text
- todo_line: existing "% TODO: manual review" line kept
- why: The first clause explains a result; the second announces Chapter 7. A semicolon makes the second read as following from the first. Also say what Chapter 7 does with the other noise conditions.
OLD:
```latex
That noise was Gaussian on every training label, so each molecule's neighbors were noisy too; Chapter~\ref{chap:noise} turns to the other noise conditions of Chapter~\ref{chap:background}.
```
NEW:
```latex
That noise was Gaussian on every training label, so each molecule's neighbors were noisy too. Chapter~\ref{chap:noise} tests neighbor-based detection under the other noise conditions of Chapter~\ref{chap:background}.
```

### V8 — No limitations paragraph in Conclusions; closing sentence is narrative
- file: thesis.tex
- kind: ai-text (new paragraph plus rewrite of an AI paragraph)
- todo_line: % TODO: manual review (above each paragraph)
- why: Decision above and style guide gap 34: a chapter's limitations go as a short paragraph inside its Conclusions. Chapter 4 has caveats in place (lines 1771 and 1818) but none in Conclusions. Every fact in the new paragraph is already stated in the chapter (lines 1612, 1728, 1771, 1795). "These flaws are where the thesis began" is a story line, not a statement of what Chapter 1 says.
OLD:
```latex
% TODO: manual review
These flaws are where the thesis began (Chapter~\ref{chap:intro}). Chapter~\ref{chap:method} builds the method they led to, the adaptive hybrid representation, and asks how the best set of features can be selected for a given prediction task.
```
NEW:
```latex
% TODO: manual review
Three limits apply to this chapter. The sweep of published methods and the neighbor-based cleaning used one dataset, the QM9 HOMO--LUMO gap, and Gaussian noise only. The sweep also added a distance to methods designed without one, so poor adaptation may explain part of their result. The cliff counts rest on one similarity threshold, 0.55, and on a gap set so that 15\% of the molecules are cliff molecules.

% TODO: manual review
Chapter~\ref{chap:intro} described how these flaws led to the adaptive hybrid representation. Chapter~\ref{chap:method} builds it and asks how the best set of features can be selected for a given prediction task.
```

### V9 — Section cited by its quoted title (her paragraph)
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: the pointer `(Chapter~\ref{chap:background}, ``Molecular similarity and activity cliffs'')` became `(Section~\ref{sec:mol_similarity})`, because the section has a label and the thesis cites sections by number (style guide gap 42).
- why: Style guide gap 42. sec:mol_similarity is the label on that section (line 270).
OLD:
```latex
Activity cliffs also look like noise to such a detector (Chapter~\ref{chap:background}, ``Molecular similarity and activity cliffs'').
```
NEW:
```latex
Activity cliffs also look like noise to such a detector (Section~\ref{sec:mol_similarity}).
```
Note: the todo_line goes directly above the paragraph that starts "The method needs a distance that picks the right neighbors". That paragraph has no comment above it now, so insert the line before its first word.

### V10 — "Molecules with identical feature vectors" has no label
- file: thesis.tex
- kind: author-text (markup only; wording unchanged)
- todo_line: none; the existing "title was ... wording unchanged" note stays true
- why: V11 and V12 cite this subsection; it is the only subsection in the chapter without a label. tab:flaws:ties is a table label, so sec:flaws:ties is free (str.count 0).
OLD:
```latex
\subsubsection{Molecules with identical feature vectors}
```
NEW:
```latex
\subsubsection{Molecules with identical feature vectors}\label{sec:flaws:ties}
```

### V11 — Conclusions cite a subsection by quoted title (her paragraph)
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: `(``Molecules with identical feature vectors'')` became `(Section~\ref{sec:flaws:ties})`, so the subsection is cited by number like the one before it (style guide gap 42).
- why: Style guide gap 42. The same paragraph cites the previous subsection as Section~\ref{sec:flaws:constant}.
OLD:
```latex
ECFP4 gives 56 of the 1,412 hERG $K_i$ molecules a vector shared with another molecule, all of them stereoisomers (``Molecules with identical feature vectors'').
```
NEW:
```latex
ECFP4 gives 56 of the 1,412 hERG $K_i$ molecules a vector shared with another molecule, all of them stereoisomers (Section~\ref{sec:flaws:ties}).
```
Note: the todo_line goes directly above the paragraph that starts "On a single dataset, often very little of a standard representation varies."

### V12 — Requirements table cites sections by quoted title
- file: thesis.tex
- kind: author-text (table cells)
- todo_line: % TODO: manual review -- edited your table: section titles in quotes became Section~\ref numbers where the section has a label; "Combining representations" and "Blueprints and their chemical interpretation" stay quoted because those Chapter 5 subsections have no label (style guide gap 42).
- why: Style guide gap 42. Labels used: sec:flaws:constant, sec:flaws:ties (V10), sec:flaws:similarity, sec:flaws:consequences, sec:flaws:representations, sec:method:importance (line 2118), sec:method:searchspace (2096), sec:similarity:hybrid_similarity (3471). Apply V10 first.
OLD:
```latex
Features that do not vary (``Constant and near-constant features'') & Features chosen for the task, within a budget & ``Feature importance scoring'' \\
Molecules merged into one vector (``Molecules with identical feature vectors'') & Features drawn from several representations & ``Combining representations'' \\
Distance dominated by irrelevant features (``Flaws in similarity and other pairwise computations'') & Compact, with the number of features searched & ``The search space'' \\
Poor performance of neighbor-based methods (``Consequences for similarity-based methods'') & Usable as a space for similarity & ``Similarity in a hybrid'' (Chapter~\ref{chap:similarity}) \\
% TODO: manual review
Features that are hard to read chemically (``Flaws in standard representations'') & Traceable,
```
NEW:
```latex
% TODO: manual review -- edited your table: section titles in quotes became Section~\ref numbers where the section has a label; "Combining representations" and "Blueprints and their chemical interpretation" stay quoted because those Chapter 5 subsections have no label (style guide gap 42).
Features that do not vary (Section~\ref{sec:flaws:constant}) & Features chosen for the task, within a budget & Section~\ref{sec:method:importance} \\
Molecules merged into one vector (Section~\ref{sec:flaws:ties}) & Features drawn from several representations & ``Combining representations'' \\
Distance dominated by irrelevant features (Section~\ref{sec:flaws:similarity}) & Compact, with the number of features searched & Section~\ref{sec:method:searchspace} \\
Poor performance of neighbor-based methods (Section~\ref{sec:flaws:consequences}) & Usable as a space for similarity & Section~\ref{sec:similarity:hybrid_similarity} \\
% TODO: manual review
Features that are hard to read chemically (Section~\ref{sec:flaws:representations}) & Traceable,
```

### V13 — Her 64-word sentence listing uses of nearest neighbors
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: the 64-word sentence listing nearest-neighbor uses became three sentences, and "identifying activity cliffs" was dropped because "detecting outliers, either ACs or noise" already lists it. In the old list the commas made "either ACs or noise" read as two more uses.
- why: Style guide gap 27 (sentences over 55 words); the list is ambiguous as written and repeats one item. Put the todo_line directly under the existing "We've" note, so both notes sit above the paragraph; delete neither.
OLD:
```latex
They are used within other processes like predicting a molecule's activity using its neighbors' labels, detecting outliers, either ACs or noise, defining an applicability domain by distance to the training set \citep{Tropsha2010}, grouping molecules into clusters for data splits \citep{Butina1999}, identifying activity cliffs, and, in federated learning, weighting each site's vote on a molecule by that molecule's distance to the site's own data \citep{Hanser2025}.
```
NEW:
```latex
They are used to predict a molecule's activity from its neighbors' labels and to detect outliers, whether ACs or noise. They also define an applicability domain by distance to the training set \citep{Tropsha2010} and group molecules into clusters for data splits \citep{Butina1999}. In federated learning, each site's vote on a molecule can be weighted by that molecule's distance to the site's own data \citep{Hanser2025}.
```

### V14 — Distance table caption gives no direction
- file: thesis.tex
- kind: author-text (caption)
- todo_line: % TODO: manual review -- edited your text: added that a higher correlation is better in both columns (style guide item 16, gap 39), as was done for the Chapter 3 captions.
- why: Check 7: a caption states the metric with its direction. The caption stays at four sentences. Table captions only; the figures session edits figure captions, not this table.
OLD:
```latex
and Five neighbors the Spearman correlation between each label and the mean label of its five nearest neighbors. Cliffs gives
```
NEW:
```latex
and Five neighbors the Spearman correlation between each label and the mean label of its five nearest neighbors; a higher correlation is better in both. Cliffs gives
```
Note: the todo_line goes on its own line directly above `\caption[All-pairs correlation, five-nearest-neighbor prediction`.

### V15 — "LogD (log units)" in the ties table
- file: thesis.tex
- kind: author-text (table cell)
- todo_line: % TODO: manual review -- edited your text: "LogD (log units)" became "LogD (log $D$)", because log D is itself a dimensionless logarithm (chemistry guide item 26).
- why: Chemistry guide item 26 names this row. The caption says RMSE and spread are "in the units beside the dataset name", so the RMSE of 0.044 now reads as 0.044 log D units.
OLD:
```latex
LogD (log units) & ECFP4 & 5,028
```
NEW:
```latex
% TODO: manual review -- edited your text: "LogD (log units)" became "LogD (log $D$)", because log D is itself a dimensionless logarithm (chemistry guide item 26).
LogD (log $D$) & ECFP4 & 5,028
```

## For the author

- Stereo pair figure (fig:flaws:stereo_pair): are the two molecules enantiomers or epimers? The chemistry guide (item 27) asks the caption to say which; only the figure file shows it.
- "Here the cliff is made by the representation rather than the chemistry" (SALI paragraph): the 1.14 p$K_i$ gap between two stereoisomers is a real difference in activity; only the infinite SALI comes from the representation. Keep or reword.
- Gap "2.90 log" for LogD in tab:flaws:distance_summary and "0.20 to 0.34 log units" for the Biogen datasets: same unit question as V15; left alone because the figures session may regenerate these tables.
- The TODO [2026-10-09] block above the sweep limits paragraph asks to say the later survey shares no dataset with this one (no QM9). The Chapter 7 source comment at sec:noise:res:sweep says "QM9 finals for censoring and grouped_shifted only". The two disagree, so nothing was added. Which is right?
- The same TODO block's point (7), distance-based selection being worst here but the only method to pass on hERG under label flips later, is not in the text. Add or drop.
- Conclusions paragraph 4 cites "(Chapter~\ref{chap:method}, ``Combining representations'')", and V12 keeps two quoted Chapter 5 titles. Those Chapter 5 subsections need labels; that belongs to the Chapter 5 unit.
- Table tab:flaws:requirements, last row: "Features that are hard to read chemically" points to "Flaws in standard representations", which does not discuss reading features chemically. Point it elsewhere or drop the parenthesis.
- Conclusions paragraph 2 has a "% CHECK" saying its numbers come from saved features, not the full runs. If the full runs differ, the 44-of-70 and all-70 statements change.
- Style guide item 465: the Chapter 1 notes call this chapter "The flaws of the similarity assumption". The title is "Limitations of the similarity assumption". Pick one.

## Applied
- V1: skipped, already in place (title note above \chapter at thesis.tex line 1372).
- V2: applied, \label{sec:flaws:methods} on the Methods section.
- V3: applied, "(next subsection)" became Section~\ref{sec:flaws:methods}.
- V4: applied, "so the gap follows the spread of the labels" removed.
- V5: applied as written.
- V6: applied as written.
- V7: applied as written.
- V8: applied; the existing sentence after the old paragraph ("The settings that method searches over ...") kept in place after the new second paragraph. Ch1 line 26 checked: it does describe the flaws leading to the hybrid.
- V9: applied, with the edited-your-text note above the paragraph.
- V10: applied, \label{sec:flaws:ties}.
- V11: applied, with the note above "On a single dataset ...".
- V12: applied, with the edited-your-table note above the first row.
- V13: applied, adapted to keep more of her phrasing ("used within other processes like predicting ... and detecting outliers, whether ACs or noise"); note placed under the existing "We've" note.
- V14: applied, note above \caption, "; a higher correlation is better in both" added.
- V15: applied, note above the row.
