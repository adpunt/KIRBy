# Chapter 5: benchmark edits (2026-10-10)

File: `thesis.tex`, Chapter 5 ("Building adaptive hybrid representations"). Backup: `thesis.tex.before_benchmark_edits_2026-10-10`.
Only paragraphs with a "% TODO: manual review" (or "review, needs manual review") line directly above them were edited. No number was added or changed in value. No table was built. No \label, \ref target, SOURCE comment, \includegraphics or \input was touched.

## Chapter prose word count (comment lines and inline comments removed; tables and captions included)

- Backup: 22,077 words
- Now: 21,410 words (667 fewer)

## Edited paragraphs

Words before -> after. "First eight words" are from the backup.

### Introduction and Methods

| First eight words | Before | After | What changed |
|---|---|---|---|
| Beyond simply combining features, there are a range | 88 | 67 | Opening cut to "Other methods handle ..."; "essentially", "rather than before", "never", "must" removed |
| On nanobody thermostability, budgets from 50 to 2,000 | 49 | 24 | Grid range dropped (it is in Table tab:decided_grid); per-fold picks kept as "four of five folds" and "250 or 500", with a pointer to the antibody budget figure |
| The fixed-share methods differ only in how | 79 | 75 | Worked-example pointer moved into the first sentence; "Since" clause made a main clause |
| Donors are set per dataset (Augmentation column | 71 | 69 | "is a donor, the only wide one" -> "is the only wide donor"; "found" -> "reported" |
| Blocks of the top k ranked features, | 90 | 87 | "must beat k=0, which wins ties" -> "and k=0 wins ties" |
| In uncapped mode, extras are appended outside | 81 | 77 | Capped-mode sentences rewritten without "must" |
| Augmentation is a separate condition rather than | 100 | 84 | "rather than" contrast removed; 35-word sentence split in two; exception sentence shortened |
| Searching every value of Table in every | 95 | 87 | Two sentences merged ("From an earlier search ... we kept"); "clearly worse" -> "well below the best" (the TODO above still asks you to state the rule exactly) |
| The search also has an adaptive pruning | 48 | 46 | Split into shorter sentences; "clearly behind" -> "far behind" |
| The hybrid experiments use 20 datasets (Table | 57 | 55 | "never pooled" -> "not pooled" |
| Each dataset is scored in the metric | 73 | 72 | "in the units of the label" -> "in the label's units"; "Metrics are never averaged" -> "We do not average metrics" |
| Summed over its 5 outer folds, the | 110 | 107 | Times ordered small molecules, reaction, antibodies in two sentences; the hERG Ki caveat moved after them |
| The hybrid is the best of many | 48 | 40 | "may be no more than the reward for" -> "may only reward"; last sentence shortened |
| The budgets are the standard list from 10 | 148 | 140 | "from 10 to 2,000 features" replaced by a pointer to Table tab:search_tiers; sign sentence shortened |

### Results

| First eight words | Before | After | What changed |
|---|---|---|---|
| On every small-molecule dataset the hybrid also beats, | 138 | 123 | Shuffling sentences merged ("none carrying a third of it"); last two sentences made one plus a short one |
| On antibodies the result differs from one | 97 | 90 | Opens with a pointer to Table tab:hybrid_vs_best_single; "with 4,275 variants" cut (n is in Table tab:datasets) |
| In an antibody representation, as in a | 234 | 228 | Shuffling sentence shortened |
| On nanobody thermostability, SMAC colloidal stability, HIC retention | 92 | 74 | Cut the sentence that re-described the table ("the two scores are almost equal"); label-problem sentences joined |
| On angiopoietin-2 binding the hybrid is higher | 52 | 40 | Cut "with a high AUROC for both ..." (the table shows it) |
| On the Buchwald--Hartwig reactions the two models | 82 | 74 | Three sentences on the two models made one |
| We have not yet run the experiment | 153 | 134 | Opens with a pointer to the two budget figures; closing sentence that repeated the opening cut |
| The ensemble of one model per representation | 156 | 145 | First two sentences merged |
| At the same budget, drawing on several | 184 | 172 | "even split" sentences shortened (exception kept: QM9 with the DNN) |
| Against PCA of all representations joined, at | 281 | 261 | Cross-check sentences shortened; DNN-correction caveat shortened (kept, because it bears on this result); antibody fold counts regrouped by model (same counts) |
| With budgets searched up to 8,000 features, | 101 | 95 | Last two sentences made one |
| Repeating each fold's search with one more | 77 | 68 | Last sentence rewritten: the old one claimed "the hybrid gains from more data" and in the same sentence said the cause is unknown; now "we cannot say which of the two causes the loss" |
| At the same number of features, drawn | 129 | 125 | Opens with a pointer to the budget figure (which draws elimination and random features); antibody pointer added; pooled-ranking sentence shortened |
| The number of features the search picks | 152 | 148 | Figure pointer moved to the first sentence |
| Across the configurations of a search, the | 138 | 133 | "when the SVM is not ($C = 10$)" was unclear; now "strongly regularized ($C$ of 0.01 or 0.1)" against "$C = 10$" |
| Scoring candidate hybrids with the downstream model | 114 | 113 | Antibody sentences reordered: dataset list first, then the EGFR pair |
| The hybrid can take in a new | 89 | 88 | Added, from the Conclusions it was cut from: the random control swaps features without a new search, and LogD ran on 2,000 molecules (your standing c13 caveat) |
| On one fold of each dataset (Figure | 57 | 56 | Opens with the figure pointer; 34-word sentence split |
| LogD can be read chemically (Figure ). | 148 | 146 | Two count sentences made one |
| The features picked are partly the same | 133 | 116 | Antibody and Buchwald-Hartwig ranges in one sentence; "does not mean ... It does mean ... rather than" rewritten |
| The small-molecule result carries over to the | 96 | 84 | "all but one" and "the exception is" merged into one sentence |
| Search settings found on LogD were reused | 61 | 61 | Two sentences joined with "but" |
| Feature selection can run where data cannot | 111 | 107 | First result sentence shortened |
| With two-class labels and up to 30% | 101 | 99 | Small word cuts |

### Conclusions

| First eight words | Before | After | What changed |
|---|---|---|---|
| This chapter asked how the best set of | 134 | 108 | "We found"; the "29%" and "every fold of the six regression datasets" numbers cut; findings in words |
| At the same number of features the hybrid | 121 | 121 | "So its gain is more than a reduction in width" added in place of the "two to all ten" clause; "We found no setting ..." |
| On antibodies the hybrid helps for the | 137 | 136 | "4,275-variant" cut; "one we built"; "We also ran the method on reactions" |
| The hybrid needs data. (plus five paragraphs merged into it) | 473 | 136 | Six limitation paragraphs (hybrid needs data; pool; cost; hours; simpler selection; provisional results) merged into one. Kept: data size, antibody pools, cost and cut grids, simpler selection, pre-correction DNN results, simulated sites. Cut: the 0.16 Spearman number, the hours, the frozen-embedding ceiling, the engineering burden, the one-pair transfer test, the matched-evaluator gap, the Gaussian-process and PCA notes (both stated in Methods and Results), the flipped-label note (stated in Results), the c13 caveat (moved to Results) |

Totals over the edited paragraphs: 4,978 words before, 4,311 after.

Paragraphs edited: 42 (41 single paragraphs, and one that merges six limitation paragraphs, five of which are now gone).
Tables built: 0. The one candidate (the PCA paragraph, 23 numbers) was not built: most of its numbers are fold tallies, which your rule keeps out of results tables, and the mean scores are mostly in its SOURCE comment, not in the text.

## Things to check

1. The "% TODO: manual review -- RERUN-SWAP-DNN" line belonged to the cut "Several results are provisional" paragraph, so it is gone. The merged limitations paragraph now holds a DNN sentence ("Most DNN results predate the correction ...") with no RERUN-SWAP-DNN tag. A grep for the tag will miss it.
2. The SOURCE comment for the search hours (min 0.41 h, max 38.58 h) now sits under the merged limitations paragraph with no sentence of its own. The same hours are still in Methods ("Summed over its 5 outer folds ..."), which has its own SOURCE comment.
3. Not edited, but it disagrees with Table tab:datasets (unverified which is right): the paragraph "Which representation is best depends on the downstream model" says the best representation is the same under both models on six of seven small-molecule datasets and three of twelve antibody datasets. The last column of Table tab:datasets shows the same one under both models on three small-molecule datasets (QM9, MDR1-MDCK, hERG blocking) and five antibody datasets. That paragraph also names SaProt and ProtT5 for the anti-VEGF binding scan, where the table says "first fold only". The two may count different things (best on held-out samples against the search's pick); the source of the paragraph was not re-read.
4. "clearly worse" in the search-grid paragraph is now "well below the best". The rule is still a summary; its TODO asks whether you want it stated exactly.
