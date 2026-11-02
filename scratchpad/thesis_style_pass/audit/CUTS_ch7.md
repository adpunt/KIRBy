# Chapter 7 cuts, items 3, 9, 26, 27, 31 (2026-10-10)

Decisions from `scratchpad/thesis_style_pass/CUTS_DECISIONS.md`. Table numbers mapped through
`scratchpad/thesis_code_todo/build/main.aux` and confirmed by caption and content.

New Additional files section, in `thesis_rework/stage1_takeaways_bullets/additional_files_body.tex`,
placed after the Chapter 6 section and before the former thesis appendices:
`\section{Detecting and handling label noise}\label{app:add:noise}`, with five `\addfile` entries:

| Entry label | Holds | Item |
|---|---|---|
| `app:add:noise_bygroup` | Table 7.6 (`tab:noise:res:bygroup`, `\input` of `figures_ch5_7/tab_noise_bygroup`) | 31 |
| `app:add:noise_mitigation_broad` | Table 7.12 (`tab:noise:res:mitigation_broad`) | 31 |
| `app:add:noise_detect_other` | Table 7.20 (`tab:noise:res:detect_other`) | 27 |
| `app:add:noise_censoring_moves` | Tables 7.23 and 7.24 in full (`tab:noise:res:censoring_moves`, `tab:noise:res:censoring_moves_other`) | 26 |
| `app:add:noise_robust_broad` | Table 7.35 (`tab:noise:res:robust_broad`) | 31 |

Every moved table keeps its label, caption, SOURCE comments and TODO lines. The three tables pasted
as text (7.20, 7.23, 7.24) were checked byte for byte against a copy of thesis.tex taken before the cut.
The three generated tables move as their `\input` line, written `../../figures_ch5_7/...` like the
other inputs in that file.

## Item 3. Ch7 redefines the three groups four times

Done. Each of the four places now points to Table 2.3 (`tab:noise_terms`) without restating it.
Chapter 2 still holds Table 2.3 (`\label{tab:noise_terms}`); six `\ref`s to it remain.

1. Introduction. Cut the list after the colon.
   - Was: "The methods fall into the three groups of Table~\ref{tab:noise_terms}: noise detection, noise mitigation and noise-robust training."
   - Now: "The methods fall into the three groups of Table~\ref{tab:noise_terms}."
2. Start of Noise detection. Cut the two defining sentences.
   - Was: "A detector gives every training molecule a suspicion score, a number that is high when the label looks wrong (Table~\ref{tab:noise_terms}). It changes no label and fits no final model. We group the detectors by what they read: ..."
   - Now: "We group the detectors (Table~\ref{tab:noise_terms}) by what they read: ..."
3. Start of Noise mitigation. Cut the definition of the two halves and the three actions.
   - Was: "A noise mitigation method has two halves (Table~\ref{tab:noise_terms}). Something selects the molecules: a detector, a known flag or every molecule. Then one of three actions follows: delete the molecule, make it count less, or move its label. Table~\ref{tab:noise:mitigation} describes each method this way: the methods reduce to a few selectors crossed with three actions."
   - Now: "Table~\ref{tab:noise:mitigation} describes each noise mitigation method by its two halves (Table~\ref{tab:noise_terms}). The methods reduce to a few selectors crossed with three actions."
4. Start of Noise-robust training. The paragraph was only the definition; replaced by one sentence so the subsection still has a lead-in.
   - Was: "A noise-robust training method selects no molecules and leaves the labels alone. It changes the loss, the model or the training loop instead (Table~\ref{tab:noise_terms})."
   - Now: "Two experiments test noise-robust training (Table~\ref{tab:noise_terms}): one changes the downstream model, and one reruns the twelve published methods."

All four paragraphs already carried `% TODO: manual review`.

## Item 9. Ch7 "Set side by side…"

Done. Cut the paragraph under `\subsection{Effect by noise type}` with its three comment lines
(two `% TODO: manual review`, one `% TODO: overlaps the paragraph above; merge them.`).
It had no label; nothing referred to it. The subsection now opens directly with
`\subsubsection{Labels clipped at an assay limit}`.

## Item 26. Tables 7.23 and 7.24

Done. Both full tables moved to `app:add:noise_censoring_moves`, labels unchanged.
In their place the thesis has one new table, `tab:noise:res:censoring_moves_main`, with one row per dataset,
the first hybrid, numbers copied unchanged:

- Network, RMSE: LogD, QM9, hERG $K_i$, Caco-2.
- Buchwald–Hartwig: no network row exists (its models are LightGBM and the random forest), so the random-forest first-hybrid row is shown, under its own heading.
- Network, Spearman: nanobody thermostability, SMAC colloidal stability, $\Delta\Delta G$.

The two bold cells that fall in these rows (17.6, 0.7164) keep their bold. Both SOURCE comments stay above the subset table.

References repointed:
- "Which of the three moves works best depends on the dataset (Table~\ref{tab:noise:res:censoring_moves})" now reads "(Table~\ref{tab:noise:res:censoring_moves_main}; every model and configuration in Additional file~\ref{app:add:noise_censoring_moves})".
- Buchwald–Hartwig sentence, "(Table~\ref{tab:noise:res:censoring_moves_other})" now "(Additional file~\ref{app:add:noise_censoring_moves})".
- "Censoring did not need fixing on two antibody sets" paragraph, same change.

Two numbers in the text are now only in the Additional file, not the subset: hERG $K_i$ second single 0.666 and 0.704, and $\Delta\Delta G$ SVR first single 0.119 and 0.248. Both sentences point to the Additional file.

## Item 27. Table 7.20

Done. Moved with its SOURCE line and TODO line to `app:add:noise_detect_other`, label `tab:noise:res:detect_other` unchanged.
The one reference, in the paragraph that opens "A Gaussian process's own predicted spread plus its residual…", now reads
"(Additional file~\ref{app:add:noise_detect_other})". That paragraph already says what the table shows,
so no second sentence was added where the table stood. The caption's pointer to `tab:noise:res:detect_residual` stays and resolves into the thesis through xr.

## Item 31. Counts in four forms

Done. Figures 7.3–7.5 (`fig:noise:did_cleaning_work_detector`, `_flag`, `_training`) stay in the thesis, untouched.

- Table 7.6 (`tab:noise:res:bygroup`). The `\input`, its two TODO lines and the `% TODO: decide:` line after it moved to `app:add:noise_bygroup`. The sentence "The table counts fold 0 only: …" also moved there unchanged, as a note to the table.
  Thesis now: "Additional file~\ref{app:add:noise_bygroup} counts how often each group of methods improved the held-out score on the first outer fold." (was "Table~\ref{tab:noise:res:bygroup} shows how often…"). The next sentence, "Mitigation helps most often under injected damage, and least often on the labels as delivered.", stays.
- Table 7.12 (`tab:noise:res:mitigation_broad`). The `\input`, its TODO lines and the paragraph after it ("Table~\ref{tab:noise:res:mitigation_broad} counts outer fold 0 only, …") moved to `app:add:noise_mitigation_broad`.
  Replacement sentence, with `% TODO: manual review` above it: "Additional file~\ref{app:add:noise_mitigation_broad} counts, for each broad noise mitigation method, the conditions where it improved the best single representation, the hybrid, both or neither."
- Table 7.35 (`tab:noise:res:robust_broad`). The `\input` and its TODO lines moved to `app:add:noise_robust_broad`. The sentence before it already said what the table shows; it now reads "The other noise-robust training methods are in Additional file~\ref{app:add:noise_robust_broad}."

## Checks

- No `\ref` in thesis.tex or the Additional files body still points at a moved label as "Table".
- The PDF was not rebuilt. The new `app:add:noise_*` labels resolve only after `additional_files.tex` is built and then the thesis, so its xr link sees them.
- No number, claim or table cell was changed, apart from the subset in item 26.
