# Chapter 3 against the benchmark table (audit, 2026-10-10)

Read-only audit. Nothing in `thesis.tex` was changed. Source: `thesis.tex` as of 08:26 today, lines 491–1093.

## Which text in Chapter 3 is AI-written

A paragraph counts as AI-written when a `% TODO: manual review` line sits directly above it without `-- fixed:` or `-- moved:`. Chapter 3 has seven such paragraphs and one table:

| tex line | Section | First words | Words |
|---|---|---|---|
| 499 | Introduction | "In this chapter, we inject the noise conditions" | 26 |
| 519 | Methods, Datasets | "Log $D$ is the logarithm of the distribution" | 158 |
| 546 | Methods, Models | "The thirteen base models are RF, QRF, XGBoost," | 60 |
| 550 | Methods, Noise conditions and levels | "The seven noise conditions used in this chapter" | 12 |
| 573 | Methods, QM9 data pipeline | "Rather than parse a CSV or Parquet file" | 97 |
| 576 | Methods, QM9 data pipeline | "The Rust program draws the noise for each" | 106 |
| 579 | Methods, QM9 data pipeline | "For every molecule, the Rust program records its" | 90 |
| 1046–1088 | Conclusions | Table `tab:condition_choice` and its caption | float |

The marker at line 652 sits above `\section{Results}`, not above a paragraph. The paragraph below the heading (line 657) has no marker, so it is the author's.

The four paragraphs marked `-- fixed:` (lines 543, 569, 878, 886) are the author's. Everything in Results, Discussion and Conclusions prose is the author's.

So the AI text in this chapter is one Introduction sentence and 523 words of Methods. No AI prose sits in Results, Discussion or Conclusions.

## Language measures

Computed with `compute_metrics.py` (`sent_metrics`, `normalise`) and the `extra_measures.py` functions. I re-segmented the current `thesis.tex` with `segment_full_author.py` first, written to the session scratchpad. The stored `full_segments_author_current.jsonl` (07:39) predates the 08:26 `thesis.tex`. Surprisal is GPT-2 small on every AI sentence (26).

"After fixes" is the same measure after fixes 1 to 7 below. They were applied to a scratch copy and computed over the same sentences. Fixes 8 to 10 were not included: fix 8 does not change the measures, and fixes 9 and 10 add new text or are in a table.

### Introduction (AI: one sentence, 26 words)

Not a target for any row. One sentence gives no mean, spread or share worth comparing. The paragraph is judged by reading instead: it is a one-sentence paragraph (fix 1).

### Methods (AI: 26 sentences, 523 words, 6 paragraphs)

Reference range = lowest to highest of Buttenschoen, Dablander, Vost. "Yours" = the author's Methods prose across the whole thesis, from the benchmark table.

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 22.3–28.0 | 20.1 | 20.1 | 21.6 | Not a target. Same as the author's mean; short sentences are her preference. |
| % sentences over 30 words | 21.1–32.7 | 12.2 | 15.4 | 16.7 | Not a target. Above the author's own share. |
| Dependent clauses per sentence | 0.46–0.95 | 0.50 | 0.65 | 0.79 | Done (inside). |
| Words per paragraph | 76.3–142.4 | 81.9 | 87.2 | 86.5 | Done (inside). The 12-word paragraph at line 550 is a fragment (fix 4). |
| Lexical density % | 58.8–59.7 | 56.6 | 58.8 | 58.1 | Done (lower edge). |
| Nominalizations per 100 words | 3.6–5.0 | 3.8 | 2.3 | 2.3 | Not a target. Few -tion/-ment nouns because the text names steps with verbs; that is the plain register she asked for. |
| % passive sentences | 40.8–45.6 | 28.2 | 38.5 | 41.7 | Done (near or inside). |
| we/our/I per 100 words | 0.83–1.6 | 1.1 | 0.19 | 0.19 | Not a target. The pipeline paragraphs name "the Rust program" as the actor, which is accurate. |
| Hedges per 100 words | 0.36–0.85 | 0.39 | 0.96 | 0.96 | Not a target. Two of the five counted hedges are false hits: "Rather than parse" and "would have received". Without them: 0.57, inside. |
| Boosters per 100 words | 0.14–0.26 | 0.10 | 0.19 | 0.19 | Done (inside). |
| AI-guide phrases per 100 words | 0.0–0.10 | 0.01 | 0.0 | 0.0 | Done. |
| Numbers per 100 words | 1.8–2.1 | 2.5 | 0.57 | 0.58 | Not a target. The AI paragraphs define endpoints and describe file handling; they carry no settings or counts. |
| % sentences with 3+ numbers | 3.7–4.7 | 6.0 | 0.0 | 0.0 | Not a target, same reason. |
| MTLD (lexical diversity) | 56.9–81.9 | 69.6 | 66.5 | 69.8 | Done (inside). |
| MATTR-50 % | 75.2–79.8 | 78.5 | 75.9 | 76.1 | Done (inside). |
| Sentence length SD (words) | 12.1–18.1 | 9.19 | 8.14 | 8.52 | Partly. Below the references, near the author's 9.19. Fixes 2, 5 and 6 merge or shorten the most uniform runs. |
| Sentence length coefficient of variation | 0.56–0.67 | 0.47 | 0.41 | 0.40 | Partly, same reason. Lengths run 6 to 37 words, so variety exists; the mean change between neighbours is the real check. |
| Mean change in length between consecutive sentences | 12.5–17.1 | 9.13 | 11.5 | 10.4 | Not a target. Above the author's own 9.13. |
| GPT-2 surprisal, bits per token | 7.04–7.42 | 7.67 | 7.42 | 7.40 | Done (inside). |
| Surprisal SD across sentences | 1.11–1.37 | 1.22 | 1.00 | 1.08 | Not a target at 26 sentences; the estimate is noisy. |
| Connectives per 100 words | 1.16–1.76 | 1.80 | 1.36 | 1.37 | Done (inside). "so" appears three times, two of them in the line-519 paragraph; acceptable. |
| Distinct connectives / connectives used | 0.15–0.28 | 0.17 | 0.71 | 0.71 | Done. Higher means more varied. |
| % sentences opening with a formulaic connector | 1.0–3.5 | 0.85 | 0.0 | 0.0 | Not a target. The author avoids them too. |
| Three-item lists per 100 sentences | 7.0–8.4 | 12.8 | 23.1 | 20.8 | Partly. Six hits in 26 sentences. Four are real content lists: the 13 base models, the fields of a record (twice), the fields written per molecule. One is a parallel triplet repeated in the next sentence (line 576); fix 6 removes the repeat. |
| Pronouns and demonstratives per 100 words | 3.6–5.5 | 5.04 | 4.47 | 4.50 | Done (inside). |

### Results, Discussion, Conclusion: rhetorical moves and language rows

Not a target. No AI-written prose sits in these sections in Chapter 3. That covers every move row: method restated, pointer to figure/table, result as numbers, result in words, comparison of results, interpretation, comparison with literature, claim of strength, limitation, link/transition, recommendation. The Conclusions prose, including its three limitation paragraphs (lines 1035–1041), is the author's. The one AI item in Conclusions is Table `tab:condition_choice`, covered under the structure rows.

## Structure rows that apply to Chapter 3

| Benchmark row | Status | Fix |
|---|---|---|
| Chapter opening links back | Done. Line 499 names Chapter 2 (`chap:background`). | Fix 1 changes only the paragraph break. |
| Chapter ends point forward | Done. Line 1092 (author's) names Chapter 4 and its question. | None. |
| Text between section title and first subsection | Not done for Results. `\section{Results}` (line 653) runs straight into `\subsection{Variance decomposition}`. Methods has a lead-in (line 508, author's). Introduction and Conclusions have no subsections. `sec:uq` has text before its first subsubsection. | Fix 9. |
| Title case | Done. Every heading is sentence case. | None. |
| Section titles | Done. Topic noun phrases, no questions or claims. | None. |
| Numbers in results text; results words per float; results pointers | Not a target. Author's text. Tables in Chapters 5–7 are in progress elsewhere; Chapter 3 has its own tables. | None. |
| Limitations | Not a target. Author's text (lines 1035–1041). | None. |
| Research questions | Done. Line 503 (author's) restates them from Chapter 1. | None. |
| Hedges | Not a target. See the Methods table: the corrected AI value is inside the range. | None. |
| Statistics | Not a target. The Wilcoxon test (line 625) and Tukey test (line 645) are in the author's text. | None. |
| Caption length | Done. The one AI caption (`tab:condition_choice`) is 25 words. That is below the reference means (49.8–73.5) but follows her rule: rows, metric, direction only. | None to the caption. Fix 10 changes two table cells. |
| Figure/table references | Done. The AI text uses `Figure~\ref` and `Section~\ref`, matching the rest. | None. |
| Decimals | Done for AI text; it has no decimals. The chapter's one four-decimal value is 0.0625 (line 626, author's), an exact p-value. | None. |
| Chemistry wording | Checked by reading line 519: log D vs log P, Caco-2 origin, efflux ratio direction (basolateral-to-apical over apical-to-basolateral), and hERG/QT are stated correctly. The TODOs above it (efflux-ratio label check, missing citations) stay open. | None. |
| "you" in body | Not a target. No "you" in AI text. The Conclusions use it in the author's text. | None. |

## Fixes

The 10 fixes are ordered by importance. Each one quotes the paragraph's first eight words and the exact old text. No meaning or number changes.

### Fix 1. Line 499, "In this chapter, we inject the noise conditions" (row: fragment paragraph; % sentences opening; chapter opening)

The opening is a one-sentence paragraph. The reference theses open with a full paragraph (Buttenschoen's unnumbered paragraph). Join it to the author's next paragraph as its first sentence. Her text is unchanged.

- Old: `...each pairing of representation and model keeps.` + blank line + `While there have been many investigations...`
- New: `...each pairing of representation and model keeps. While there have been many investigations...` (blank line removed)

Optional. The `% TODO: manual review` marker then sits above a paragraph that is mostly hers. She may prefer to keep the break.

### Fix 2. Line 519, "Log $D$ is the logarithm of the distribution" (row: sentence length SD; mean change in length)

Two short sentences describe one measurement. Merge them. The 8-word hERG opener stays short.

- Old: `Permeability across the layer is measured in both directions. The efflux ratio is the permeability from the basolateral to the apical side divided by that in the other direction.`
- New: `Permeability across the layer is measured in both directions, and the efflux ratio is the permeability from the basolateral to the apical side divided by that in the other direction.`

### Fix 3. Line 546, "The thirteen base models are RF, QRF, XGBoost," (row: three-item lists; clarity of a list)

The second sentence nests a two-name list inside a four-item list with commas. It reads as seven or eight models, not six. "the heteroscedastic GP (het.)" also says heteroscedastic twice. Use semicolons, with the short name first and its description after.

- Old: `The six variant models are RF (300 trees), the heteroscedastic GP (het.), the Bayesian networks with a variance head, BNN-$\alpha$ (var.\ head) and BNN-$\beta$ (var.\ head), and the heteroscedastic VBLL-$\alpha$ (het.) and VBLL-$\beta$ (het.).`
- New: `The six variant models are RF (300 trees); GP (het.), the heteroscedastic GP; BNN-$\alpha$ (var.\ head) and BNN-$\beta$ (var.\ head), the two Bayesian networks with a variance head; and VBLL-$\alpha$ (het.) and VBLL-$\beta$ (het.), the two heteroscedastic VBLL networks.`

### Fix 4. Line 550, "The seven noise conditions used in this chapter" (row: fragment paragraph; link sentence that only announces)

This is a 12-word paragraph that only points elsewhere. Make it the first sentence of the author's next paragraph. Drop "used in this chapter"; the subsection heading already says that.

Do not cut the sentence. Line 508 points to the same section, but the author's TODO at line 507 asks to remove part of that paragraph. This sentence may become the only pointer.

- Old: `The seven noise conditions used in this chapter are defined in Section~\ref{sec:noise_injection}.` + blank line + `We controlled noise quantity using noise levels,...`
- New: `The seven noise conditions are defined in Section~\ref{sec:noise_injection}. We controlled noise quantity using noise levels,...`

### Fix 5. Line 573, "Rather than parse a CSV or Parquet file" (row: sentence length SD; fragmentation)

The 6-word sentence and the one after it describe one hand-off between Python and Rust. Merge them. The paragraph then runs 36, 30, 22 and 11 words, instead of 36, 6, 22, 22 and 11.

- Old: `Python writes one file per split. The Rust program writes a copy in which the processed labels replace the raw ones, and Python reads that copy for training.`
- New: `Python writes one file per split, and the Rust program writes a copy of it in which the processed labels replace the raw ones; Python reads that copy for training.`

### Fix 6. Line 576, "The Rust program draws the noise for each" (row: three-item lists per 100 sentences)

Two sentences in a row list the same three things: amount, limit and families. Name them once.

- Old: `Validation labels get the same amount of noise in label units, the same limit and the same families, from an independent draw seeded from the training seed.`
- New: `Validation labels get all three, with the amount in label units, from an independent draw seeded from the training seed.`

### Fix 7. Line 579, "For every molecule, the Rust program records its" (row: lexical diversity; repeated sentence openings)

"The Rust program" is the subject in line 573, opens line 576, and appears again here. Optional, low priority.

- Old: `For every molecule, the Rust program records`
- New: `For every molecule, the program records`

### Fix 8. Line 519, "Log $D$ is the logarithm of the distribution" (row: connectives; connective repeated)

Optional. The same paragraph uses "so" twice for two different causal links. One is enough to vary.

- Old: `so a basic amine that is protonated at that pH lowers log $D$ below log $P$.`
- New: `and a basic amine that is protonated at that pH therefore has a log $D$ below its log $P$.`

### Fix 9. Lines 653–655, `\section{Results}` (row: text between section title and first subsection)

This adds a two-sentence lead-in. The text is new and AI-written, so the existing `% TODO: manual review` at line 652 covers it. It names the six subsections in order and gives no results.

- Old: `\section{Results}` + blank line + `\subsection{Variance decomposition}`
- New: `\section{Results}` + blank line + `We first split the variance in predictive performance and robustness between model and representation, then compare single models across noise conditions and datasets. The last two subsections compare probabilistic models with their deterministic counterparts and ask whether the uncertainty a model reports tracks the injected noise (Section~\ref{sec:uq}).` + blank line + `\subsection{Variance decomposition}`

### Fix 10. Table `tab:condition_choice`, lines 1072–1082 (row: captions and tables say only what the column says)

In two rows the "Avoid" column holds a statement about every model, not a model to avoid. Move each statement to "Extent of the evidence".

- Grouped-shifted, Avoid. Old: `The GP, which loses the most; every model loses \aucnorm{} on every dataset and representation`. New: `The GP, which loses the most`.
- Grouped-shifted, Extent. Old: `All four datasets; the hardest condition`. New: `All four datasets; the hardest condition, as every model loses \aucnorm{} on every dataset and representation`.
- Censoring, Avoid. Old: `Every pair tested loses more robustness than under Gaussian noise`. New: `---`.
- Censoring, Extent. Old: `Reported for the HOMO--LUMO gap only (Table~\ref{tab:censoring}), on the five model and representation pairs run under censoring`. New: `Reported for the HOMO--LUMO gap only (Table~\ref{tab:censoring}), on the five model and representation pairs run under censoring, each of which loses more robustness than under Gaussian noise`.

Not a language fix, unverified: the caption of `tab:censoring` says its two \aucnorm{} columns are on different scales. "Loses more robustness than under Gaussian noise" compares across those scales. The author's line 746 makes the same comparison (0.12 to 0.18), so I left the claim as it is.

## Not changed, on purpose

- The three-item lists of record fields (lines 573, 579) and the list of the 13 base models (line 546) are content, not rhythm. They stay.
- The 37-word sentence in line 579 ("The run halts if a record holds...") stays. The reference theses have 21–33% of Methods sentences over 30 words.
- Paragraphs 573, 576 and 579 may be cut for the page limit. That is the author's call, not a style question.
