# Chapter 5 cuts, carried out 2026-10-10

Decisions: `scratchpad/thesis_style_pass/CUTS_DECISIONS.md`, items 7, 8, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 30.
Table numbers were matched to labels with `scratchpad/thesis_code_todo/build/main.aux` (built 10 Oct 10:08) and confirmed by caption content.

Files changed:
- `thesis.tex` (Chapter 5).
- `thesis_rework/stage1_takeaways_bullets/additional_files_body.tex`: 14 new `\addfile` items, all inside section `app:add:method`, after "Worked example of fold counts" and before the "TODO: Additional files for Chapter 5" line. Each carries a `% [CUTS 2026-10-10, item N]` comment.

Everything moved was pasted unchanged, with its `% TODO: manual review` lines, SOURCE comments and labels. Moved Table 5.21, 5.25, 5.32 and 5.33 copies were checked against the thesis text with `diff` before the thesis copy was deleted.

Every new sentence in the thesis has `% TODO: manual review` above it.

---

## Item 7: Ch5 ensembling paragraph. Cut.

- Cut: the paragraph "Ensembling, a form of late fusion, fits one model per representation ..." (AI-written, `% TODO: manual review`), with its TODO line.
- Kept: the author's TODO above the concatenation paragraph ("I'm pretty sure I'm doing late fusion but calling it ensembling ..."). `Valsecchi2020` is still cited in Table `tab:hybrid_alternatives`.
- Labels moved: none. References repointed: none.

## Item 8: Tables 5.17 and 5.19 merged into Table 5.18 as two columns.

- Table 5.18 (`tab:hybrid_vs_best_single`) was `\input{figures_ch5_7/tab_hybrid_vs_best_single}`. The `\input` is replaced by the same table written inline, with two new last columns: "Best in hindsight" (representation) and "Its score".
  - The first four columns are copied from the generated file unchanged.
  - The two new columns are copied unchanged from Table 5.17 (`tab:method:hindsight_smallmol`) and Table 5.19 (`tab:method:hindsight_antibodies`), matched by dataset and model. All 40 rows have a match.
  - One bold pair was dropped: Caco-2 DNN in Table 5.17. Table 5.18 already has four bold cells, the maximum.
  - Caption: one sentence added: "Best in hindsight is the single representation best on the held-out samples, with its score."
  - A comment above the table says the generator (`tests/analysis/thesis_figures.py`, F1) does not write the hindsight columns. If F1 is re-run, its new numbers must be copied into the first four columns by hand.
- Tables 5.17 and 5.19 also have shuffling columns (effective number of representations, top share, share built here). The text cites them, and Table 5.18 has no room for them. Both tables were therefore moved unchanged to a new Additional file, so no number is lost:
  - `app:add:hindsight_shuffle`, "Representation best in hindsight, and how many representations the hybrid draws on". Labels kept: `tab:method:hindsight_smallmol`, `tab:method:hindsight_antibodies`. One new lead sentence.
- References repointed:
  - small-molecule hindsight sentence: `Table~\ref{tab:method:hindsight_smallmol}` became `Table~\ref{tab:hybrid_vs_best_single}`; "(Additional file~\ref{app:add:hindsight_shuffle})" added to the sentence that introduces the shuffling.
  - antibody hindsight sentence: `Table~\ref{tab:method:hindsight_antibodies}` became `Table~\ref{tab:hybrid_vs_best_single}`.
  - "about one and a half to five representations by the same shuffling": became `Additional file~\ref{app:add:hindsight_shuffle}`.

## Item 13: Ch5 greedy paragraph. One sentence.

- The paragraph was marked `% TODO: manual review -- fixed: FcRn ...`, which marks the author's text with a fix. The decision names this paragraph, so it was carried out. Flagged here for her to check.
- Its numbers (500 against 10 fits, 16th of 17, 0.360 against 0.453 Spearman on FcRn retention) are already in Additional file `app:add:design_experiments` ("Allocation methods", 0.3600 against 0.4526). Nothing was moved.
- Replacement sentence: "The first method of feature allocation we tried was a greedy forward selection \citep{Kohavi1997, Guyon2003}, which we removed from the search because it was more expensive and less accurate (Additional file~\ref{app:add:design_experiments})."
- "FcRn" no longer appears in the thesis text.

## Item 14: Ch5 Augmentation and Table 5.5. One paragraph; rest to the Additional files.

- Kept in the thesis, as one paragraph: the first paragraph (minus "Including them in the budget split would distort it ..."), the first sentence on donors, and the earlier-search result. New: one sentence defining uncapped and capped, and one pointer sentence.
- The author's TODO "I need a better list to describe augmentations" stays in the thesis, below the paragraph.
- Moved unchanged to `app:add:augmentation`, "How augmentation ranks and adds donor features":
  - the donors and Bahia2023 paragraph;
  - the residual ranking paragraph;
  - the block sizes paragraph;
  - the capped/uncapped paragraph;
  - the "Augmentation is not a searched setting" paragraph (its result is repeated in the thesis paragraph);
  - the limitations paragraph;
  - Table 5.5, label `tab:aug_conditions` kept.
- `tab:aug_conditions` had one reference, inside the moved paragraph. Nothing to repoint.
- New sentences in the thesis paragraph:
  - "Uncapped augmentation adds the extra features outside the budget, and capped augmentation puts them in place of the weakest base features."
  - "Additional file~\ref{app:add:augmentation} gives how donor features are ranked and added, and the three conditions."

## Item 15: Ch5 adaptive pruning. Cut.

- Cut: `\subsubsection{Adaptive pruning}\label{sec:method:pruning}` and its one AI paragraph.
- `sec:method:pruning` had no references. `Jamieson2016` and `Li2018Hyperband` are no longer cited in the thesis.
- The Additional files still mention adaptive pruning in two places, both untouched: "Size of the search" ("with adaptive pruning off") and "Fast tier" (a RESULT NEEDED on time saved).

## Item 16: Tables 5.8 and 5.15. One sentence each; tables to the Additional files.

- Table 5.8 (`tab:cut_walkthrough`): the walkthrough paragraph and the table were moved unchanged to `app:add:cut_walkthrough`, "What each cut removes from the hERG $K_i$ search". Label kept.
  - Replacement sentence: "Additional file~\ref{app:add:cut_walkthrough} follows hERG $K_i$ through these cuts, from 8,220 configurations to 224, and its final grid still contains the best configuration."
- Table 5.15 (`tab:tuning_base_reps`): the tuning paragraph, its RERUN-SWAP-DNN comment and the table were moved unchanged to `app:add:tuning_base_reps`, "Default and tuned model settings on single representations". Label kept.
  - Replacement sentence: "On one outer fold of five datasets, tuning helped the SVM most and made the random forest worse on Buchwald--Hartwig yield (Additional file~\ref{app:add:tuning_base_reps})."
  - The sentence after the table ("The Gaussian process and the SVM are skipped above ...") stays in the thesis. It now has its own `% TODO: manual review` line.
- Each table's only reference sat inside the moved paragraph. Nothing else to repoint.

## Item 17: Ch5 PCA paragraph. Keep who wins with each model; explanation to the Additional files.

- Kept: the first five sentences (SVM, both models, DNN, the within-0.0020 sentence) and the antibody paragraph after it.
- Moved unchanged to `app:add:pca_disagreement`, "Hybrid against PCA: three experiments that disagree":
  - the last five sentences (ensemble experiment agrees; third experiment disagrees at 0.409 against 0.477; its three differences; "cannot say which"; DNN results predate the correction);
  - a copy of the SOURCE comment and of the c15 TODO.
  - One new lead sentence, so the moved text has its subject.
- The c15 TODO and the RERUN-SWAP-DNN rule comment also stay in the thesis.
- Replacement sentence: "Additional file~\ref{app:add:pca_disagreement} compares this with two other experiments, one of which disagrees on LogD and Caco-2."

## Item 18: Ch5 separate sites, Tables 5.37 to 5.40. Keep 5.37 and its paragraph; 5.38 to 5.40 and their Methods detail to the Additional files.

- Kept: the paragraph on one family per site and Table 5.37 (`tab:method:family_sites`). Kept in Methods: the general paragraphs, including "All four separate-site experiments ..." and the limitations paragraph.
- Moved unchanged to `app:add:separate_sites`, "Three further experiments with separate sites":
  - Methods: the paragraph on how the site models' predictions are combined (distance and score weighting, student model). Only Tables 5.38 and 5.39 use these ensembles.
  - Methods: "In the first of three further experiments ..." and "In the second experiment ...".
  - Results: the three paragraphs and Tables 5.38 (`tab:method:site_reps`), 5.39 (`tab:method:flipped_sites`) and 5.40 (`tab:method:cluster_sites`), with their SOURCE and NOTE comments. Labels kept.
- Each table's references sat inside the moved paragraphs.
- Replacement sentences:
  - Methods: "Three further experiments, with flipped labels, with one representation per site and with features selected per cluster, are described in Additional file~\ref{app:add:separate_sites}."
  - Results: "Additional file~\ref{app:add:separate_sites} reports three further experiments: one hybrid over sites that each hold one representation, selection at sites with flipped labels, and selection per cluster of Caco-2."

## Item 20: Table 5.21. Subset (one budget per dataset and model); full table to the Additional files.

- Rule used: the smallest budget each dataset and model searched. 36 rows in total. Rows were copied unchanged.
- Why this budget: it keeps the table's two bold rows (QM9 DNN at 500 features, Nb-Tm DNN at 50 features). It also keeps the 50-feature Nb-Tm row the text names.
- Her call if she prefers another rule. The alternatives are the largest budget, or the budget the search picked most often.
- The thesis subset keeps the label `tab:method:top_rep_even_split`. The caption now says which budget each row is at, and that every budget is in the Additional file.
- Full table moved unchanged to `app:add:top_rep_even_split`. Its label was renamed to `tab:add:top_rep_even_split_all`, since two tables cannot share one label.
- Reference: "(Table~\ref{tab:method:top_rep_even_split}; every budget in Additional file~\ref{app:add:top_rep_even_split})".
- The claims in the paragraph ("at every budget searched", "at all but one budget") are about all budgets. They now rest on the Additional file.

## Item 21: Table 5.25. To the Additional files.

- `tab:method:feature_reduction` moved unchanged to `app:add:feature_reduction`, "Hybrid against other feature selections at the same number of features". Label kept.
- Reference repointed: "Figure~\ref{fig:best_number_of_features} and Additional file~\ref{app:add:feature_reduction} compare ...". Figures 5.3 and 5.4 stay.

## Item 22: Table 5.26. To the Additional files.

- `tab:method:budget_per_fold` moved unchanged to `app:add:budget_per_fold`, "Budget and number of representations each fold picked". Label kept.
- References repointed (2): "... and Additional file~\ref{app:add:budget_per_fold} lists it", and the Buchwald--Hartwig sentence "(Additional file~\ref{app:add:budget_per_fold})".

## Item 23: Tables 5.28 and 5.29. To the Additional files.

- `tab:method:budget_by_capacity` moved unchanged to `app:add:budget_by_capacity`. Label kept. Reference repointed: "(Additional file~\ref{app:add:budget_by_capacity})".
- `tab:method:evaluator` (a sidewaystable) moved unchanged to `app:add:evaluator`. Label kept. Reference repointed: "(Additional file~\ref{app:add:evaluator})".
- Both paragraphs stay.

## Item 24: Tables 5.32 and 5.33. To the Additional files; paragraphs stay.

- `tab:method:new_embeddings` moved unchanged to `app:add:new_embeddings`. Label kept. Reference repointed: "(Additional file~\ref{app:add:new_embeddings})".
- `tab:method:conformal` moved unchanged to `app:add:conformal`. Label kept. Reference repointed: "(Additional file~\ref{app:add:conformal})".
- Both paragraphs stay unchanged apart from the pointer.

## Item 30: Listing 5.1 (author's). To the Additional files.

- `lst:blueprint` moved unchanged to `app:add:blueprint_listing`, "Fields of a blueprint and their types". Label kept.
- Her sentence: only "Listing~\ref{lst:blueprint}" became "Additional file~\ref{app:add:blueprint_listing}". The rest is unchanged.
- The existing Additional file `app:add:blueprint_fields` (a table of the same fields) was left as it is.

---

## Not done or not checked

- No LaTeX build was run, because other agents are editing the same files. `\begin`/`\end` balance was checked in both files with a script. All moved labels now exist once, in the Additional files only. No thesis `\ref` points to a label that was removed.
- `Bahia2023`, `friedman2001greedy`, `efron2004lars`, `Jamieson2016` and `Li2018Hyperband` may now be cited only in the Additional files, or not at all. Not checked against refs.bib usage.
