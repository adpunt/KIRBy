# Benchmark after the benchmark edits

Written 2026-10-10. This re-measures the thesis after the edit pass that followed `BENCHMARK_REPORT.md`. No reference thesis was re-read; their numbers are copied from that report's run (`archive_first_run/metrics_sentences.jsonl`).

## What was compared

- **before** = `thesis.tex.before_benchmark_edits_2026-10-10`, the text the first report measured. Its numbers here equal the first report's "yours now" column.
- **now** = `thesis.tex`, read 10 Oct after this check. It includes nine sentence splits made during this check (listed under Verification).
- **reference range** = lowest and highest of Buttenschoen, Vost and Dablander for that section type. Vost has no discussion section, so the discussion range comes from two theses.
- **gap** = distance outside the range, in standard deviations of the reference paragraphs of that section type. Same definition as the first report.
- **closed** = outside before, inside now. **narrowed / widened** = outside both times, gap smaller / larger by more than 0.05 SD. **held** = gap moved by 0.05 SD or less. **opened** = inside before, outside now.

Scripts: `segment_full_author.py` (new version `before_edits`), `compute_metrics_after.py`, `build_after_tables.py`, `gen_after_report.py`. Per-measure CSV: `after_section_types.csv`. The first run's outputs are kept in `archive_first_run/`.

**Moves.** Sentences that did not change keep their old code (matched by exact text). The 276 sentences in results, discussion and conclusion sections that are new or reworded in this pass were coded by reading, with the same eleven moves (`moves_after_new_sentences.csv`). One coder, one pass, no agreement check. 53 sentences now and 62 before were uncoded in the first run and stay uncoded; they are left out of the move shares.

Sentences / paragraphs / words in the measured prose, before -> now:

| Section type | before | now |
|---|---|---|
| introduction | 281 / 56 / 5,764 | 284 / 56 / 5,726 |
| background | 253 / 56 / 5,111 | 253 / 56 / 5,111 |
| methods | 1145 / 245 / 21,935 | 1147 / 245 / 21,783 |
| results | 952 / 198 / 19,188 | 943 / 198 / 18,731 |
| discussion | 77 / 26 / 1,330 | 71 / 24 / 1,219 |
| conclusion | 228 / 48 / 4,276 | 186 / 37 / 3,629 |

## Summary

- **introduction**: 12 measures. held 6, inside both times 4, narrowed 2.
- **background**: 12 measures. held 7, inside both times 5.
- **methods**: 12 measures. held 6, inside both times 4, closed 1, widened 1.
- **results**: 24 measures. held 10, inside both times 9, narrowed 2, closed 2, widened 1.
- **discussion**: 24 measures. held 10, inside both times 6, widened 4, narrowed 3, crossed the range 1.
- **conclusion**: 24 measures. inside both times 8, narrowed 6, held 5, closed 3, widened 2.

## Introduction

One row is one measure. Moves are shares of coded sentences in that section type.

| Measure | reference range | before | now | gap |
|---|---|---|---|---|
| Words per sentence | 25.3 to 30.8 | 20.5 | 20.2 | held (below, 0.56 -> 0.60 SD) |
| Subordinate clauses per sentence | 0.71 to 0.85 | 0.77 | 0.75 | inside the range both times |
| Nominalizations per 100 words | 4.07 to 6.08 | 6.14 | 6.15 | held (above, 0.02 -> 0.02 SD) |
| Numbers per 100 words | 0.24 to 1.12 | 0.45 | 0.45 | inside the range both times |
| % sentences with 3+ numbers | 0.00 to 3.27 | 1.07 | 1.06 | inside the range both times |
| Lexical density % | 60.6 to 62.1 | 58.6 | 58.6 | held (below, 0.46 -> 0.46 SD) |
| Hedges per 100 words | 0.72 to 1.00 | 0.83 | 0.80 | inside the range both times |
| Boosters per 100 words | 0.31 to 0.36 | 0.42 | 0.37 | narrowed (above, 0.12 -> 0.02 SD) |
| AI-guide phrases per 100 words | 0.17 to 0.48 | 0.05 | 0.05 | held (below, 0.22 -> 0.22 SD) |
| % passive sentences | 24.8 to 30.8 | 19.9 | 19.4 | held (below, 0.18 -> 0.20 SD) |
| we/our/I per 100 words | 1.41 to 1.70 | 0.71 | 0.82 | narrowed (below, 0.38 -> 0.32 SD) |
| Words per paragraph | 112.0 to 145.7 | 102.9 | 102.2 | held (below, 0.10 -> 0.11 SD) |

## Background

One row is one measure. Moves are shares of coded sentences in that section type.

| Measure | reference range | before | now | gap |
|---|---|---|---|---|
| Words per sentence | 23.1 to 26.0 | 20.2 | 20.2 | held (below, 0.43 -> 0.43 SD) |
| Subordinate clauses per sentence | 0.59 to 0.81 | 0.62 | 0.62 | inside the range both times |
| Nominalizations per 100 words | 3.99 to 4.55 | 5.24 | 5.24 | held (above, 0.30 -> 0.30 SD) |
| Numbers per 100 words | 0.46 to 1.18 | 0.53 | 0.53 | inside the range both times |
| % sentences with 3+ numbers | 0.83 to 1.73 | 0.40 | 0.40 | held (below, 0.05 -> 0.05 SD) |
| Lexical density % | 61.2 to 62.4 | 61.8 | 61.8 | inside the range both times |
| Hedges per 100 words | 0.95 to 1.11 | 0.92 | 0.92 | held (below, 0.03 -> 0.03 SD) |
| Boosters per 100 words | 0.19 to 0.39 | 0.16 | 0.16 | held (below, 0.04 -> 0.04 SD) |
| AI-guide phrases per 100 words | 0.03 to 0.20 | 0.14 | 0.14 | inside the range both times |
| % passive sentences | 26.5 to 40.2 | 26.5 | 26.5 | held (below, 0.00 -> 0.00 SD) |
| we/our/I per 100 words | 0.29 to 0.85 | 0.49 | 0.49 | inside the range both times |
| Words per paragraph | 91.7 to 154.7 | 91.3 | 91.3 | held (below, 0.00 -> 0.00 SD) |

## Methods

One row is one measure. Moves are shares of coded sentences in that section type.

| Measure | reference range | before | now | gap |
|---|---|---|---|---|
| Words per sentence | 22.3 to 28.1 | 19.2 | 19.0 | held (below, 0.31 -> 0.33 SD) |
| Subordinate clauses per sentence | 0.46 to 0.95 | 0.50 | 0.49 | inside the range both times |
| Nominalizations per 100 words | 3.60 to 5.00 | 3.37 | 3.38 | held (below, 0.08 -> 0.08 SD) |
| Numbers per 100 words | 1.75 to 2.05 | 2.27 | 2.17 | held (above, 0.10 -> 0.05 SD) |
| % sentences with 3+ numbers | 3.68 to 4.70 | 5.15 | 4.62 | CLOSED (was 0.04 SD above) |
| Lexical density % | 58.8 to 59.7 | 55.6 | 55.7 | held (below, 0.59 -> 0.56 SD) |
| Hedges per 100 words | 0.36 to 0.85 | 0.41 | 0.38 | inside the range both times |
| Boosters per 100 words | 0.14 to 0.26 | 0.13 | 0.08 | widened (below, 0.02 -> 0.15 SD) |
| AI-guide phrases per 100 words | 0.00 to 0.10 | 0.00 | 0.00 | inside the range both times |
| % passive sentences | 40.8 to 45.6 | 27.8 | 27.1 | held (below, 0.41 -> 0.43 SD) |
| we/our/I per 100 words | 0.83 to 1.62 | 0.78 | 0.80 | held (below, 0.03 -> 0.02 SD) |
| Words per paragraph | 76.2 to 142.4 | 89.5 | 88.9 | inside the range both times |

## Results

One row is one measure. Moves are shares of coded sentences in that section type.

| Measure | reference range | before | now | gap |
|---|---|---|---|---|
| Words per sentence | 24.6 to 31.1 | 20.2 | 19.9 | held (below, 0.67 -> 0.71 SD) |
| Subordinate clauses per sentence | 0.58 to 1.02 | 0.50 | 0.49 | held (below, 0.14 -> 0.17 SD) |
| Nominalizations per 100 words | 3.92 to 4.42 | 3.62 | 3.64 | held (below, 0.10 -> 0.09 SD) |
| Numbers per 100 words | 1.17 to 5.92 | 2.85 | 2.82 | inside the range both times |
| % sentences with 3+ numbers | 3.12 to 20.4 | 7.46 | 7.10 | inside the range both times |
| Lexical density % | 59.3 to 62.9 | 54.8 | 55.1 | narrowed (below, 0.86 -> 0.80 SD) |
| Hedges per 100 words | 0.87 to 2.31 | 0.68 | 0.67 | held (below, 0.16 -> 0.17 SD) |
| Boosters per 100 words | 0.24 to 0.48 | 0.16 | 0.15 | held (below, 0.13 -> 0.14 SD) |
| AI-guide phrases per 100 words | 0.09 to 0.58 | 0.03 | 0.03 | held (below, 0.08 -> 0.08 SD) |
| % passive sentences | 26.2 to 30.6 | 12.6 | 11.7 | held (below, 0.55 -> 0.58 SD) |
| we/our/I per 100 words | 0.26 to 1.58 | 0.46 | 0.46 | inside the range both times |
| Words per paragraph | 99.2 to 177.7 | 96.9 | 94.6 | held (below, 0.03 -> 0.07 SD) |
| % M3 among M2+M3+M4 | 37.9 to 44.8 | 36.6 | 40.6 | CLOSED (was 0.03 SD below) |
| M1 method restated % | 8.57 to 33.7 | 13.0 | 13.0 | inside the range both times |
| M2 pointer to figure/table % | 3.11 to 18.2 | 1.43 | 2.00 | held (below, 0.08 -> 0.05 SD) |
| M3 result as numbers % | 7.86 to 24.8 | 13.0 | 16.5 | inside the range both times |
| M4 result in words % | 8.29 to 12.4 | 21.0 | 22.2 | widened (above, 0.49 -> 0.56 SD) |
| M5 comparison of results % | 18.1 to 23.6 | 24.8 | 20.3 | CLOSED (was 0.05 SD above) |
| M6 interpretation % | 7.44 to 27.1 | 12.5 | 12.8 | inside the range both times |
| M7 comparison with literature % | 0.00 to 0.83 | 1.65 | 1.66 | held (above, 0.24 -> 0.25 SD) |
| M8 claim of strength % | 0.83 to 3.63 | 1.87 | 1.77 | inside the range both times |
| M9 limitation % | 0.83 to 5.00 | 7.37 | 6.43 | narrowed (above, 0.43 -> 0.26 SD) |
| M10 link % | 1.65 to 6.43 | 2.42 | 2.44 | inside the range both times |
| M11 recommendation % | 0.00 to 4.29 | 0.99 | 1.00 | inside the range both times |

## Discussion

One row is one measure. Moves are shares of coded sentences in that section type.

| Measure | reference range | before | now | gap |
|---|---|---|---|---|
| Words per sentence | 24.8 to 34.2 | 17.3 | 17.2 | held (below, 1.03 -> 1.04 SD) |
| Subordinate clauses per sentence | 0.75 to 0.84 | 0.47 | 0.48 | held (below, 0.59 -> 0.57 SD) |
| Nominalizations per 100 words | 4.07 to 4.94 | 4.29 | 4.35 | inside the range both times |
| Numbers per 100 words | 0.24 to 1.16 | 1.05 | 0.25 | inside the range both times |
| % sentences with 3+ numbers | 1.98 to 2.08 | 2.60 | 0.00 | crossed: was 0.07 SD above, now 0.29 SD below |
| Lexical density % | 60.4 to 61.1 | 58.7 | 60.2 | narrowed (below, 0.32 -> 0.04 SD) |
| Hedges per 100 words | 1.22 to 1.48 | 0.68 | 0.57 | widened (below, 0.40 -> 0.47 SD) |
| Boosters per 100 words | 0.18 to 0.72 | 0.23 | 0.25 | inside the range both times |
| AI-guide phrases per 100 words | 0.18 to 0.24 | 0.00 | 0.00 | held (below, 0.32 -> 0.32 SD) |
| % passive sentences | 34.6 to 39.6 | 13.0 | 11.3 | widened (below, 0.72 -> 0.78 SD) |
| we/our/I per 100 words | 0.96 to 1.34 | 0.23 | 0.33 | narrowed (below, 0.41 -> 0.35 SD) |
| Words per paragraph | 92.8 to 126.2 | 51.1 | 50.8 | held (below, 0.70 -> 0.71 SD) |
| % M3 among M2+M3+M4 | 0.00 to 25.0 | 20.0 | 11.1 | inside the range both times |
| M1 method restated % | 9.72 to 50.0 | 1.37 | 1.43 | held (below, 0.25 -> 0.25 SD) |
| M2 pointer to figure/table % | 0.00 to 2.38 | 0.00 | 0.00 | inside the range both times |
| M3 result as numbers % | 0.00 to 5.56 | 5.48 | 2.86 | inside the range both times |
| M4 result in words % | 2.38 to 16.7 | 21.9 | 22.9 | widened (above, 0.30 -> 0.35 SD) |
| M5 comparison of results % | 0.00 to 9.72 | 13.7 | 14.3 | held (above, 0.29 -> 0.33 SD) |
| M6 interpretation % | 9.52 to 15.3 | 5.48 | 5.71 | held (below, 0.24 -> 0.22 SD) |
| M7 comparison with literature % | 4.17 to 9.52 | 0.00 | 0.00 | held (below, 0.24 -> 0.24 SD) |
| M8 claim of strength % | 4.76 to 9.72 | 19.2 | 20.0 | held (above, 0.43 -> 0.47 SD) |
| M9 limitation % | 0.00 to 9.72 | 13.7 | 11.4 | narrowed (above, 0.20 -> 0.09 SD) |
| M10 link % | 1.39 to 2.38 | 6.85 | 8.57 | widened (above, 0.70 -> 0.98 SD) |
| M11 recommendation % | 18.1 to 19.1 | 12.3 | 12.9 | held (below, 0.21 -> 0.19 SD) |

## Conclusion

One row is one measure. Moves are shares of coded sentences in that section type.

| Measure | reference range | before | now | gap |
|---|---|---|---|---|
| Words per sentence | 25.0 to 32.5 | 18.8 | 19.5 | narrowed (below, 0.83 -> 0.73 SD) |
| Subordinate clauses per sentence | 0.71 to 0.96 | 0.58 | 0.64 | narrowed (below, 0.22 -> 0.12 SD) |
| Nominalizations per 100 words | 4.22 to 6.86 | 4.47 | 4.71 | inside the range both times |
| Numbers per 100 words | 0.04 to 0.43 | 0.77 | 0.50 | narrowed (above, 0.56 -> 0.11 SD) |
| % sentences with 3+ numbers | 0.00 to 0.00 | 1.75 | 1.61 | held (above; references all 0, gap in SD undefined) |
| Lexical density % | 61.7 to 64.3 | 56.4 | 57.2 | narrowed (below, 1.12 -> 0.95 SD) |
| Hedges per 100 words | 0.95 to 2.16 | 0.98 | 1.05 | inside the range both times |
| Boosters per 100 words | 0.24 to 0.90 | 0.19 | 0.33 | CLOSED (was 0.07 SD below) |
| AI-guide phrases per 100 words | 0.43 to 0.97 | 0.00 | 0.00 | held (below, 0.54 -> 0.54 SD) |
| % passive sentences | 14.6 to 30.1 | 20.2 | 18.8 | inside the range both times |
| we/our/I per 100 words | 1.33 to 1.89 | 0.33 | 0.66 | narrowed (below, 0.66 -> 0.44 SD) |
| Words per paragraph | 95.3 to 138.9 | 89.1 | 98.1 | CLOSED (was 0.14 SD below) |
| % M3 among M2+M3+M4 | 0.00 to 18.2 | 14.3 | 12.8 | inside the range both times |
| M1 method restated % | 10.4 to 15.9 | 7.98 | 12.0 | CLOSED (was 0.12 SD below) |
| M2 pointer to figure/table % | 0.00 to 1.37 | 0.00 | 0.00 | inside the range both times |
| M3 result as numbers % | 0.00 to 3.17 | 3.29 | 3.43 | held (above, 0.03 -> 0.06 SD) |
| M4 result in words % | 6.85 to 14.3 | 19.7 | 23.4 | widened (above, 0.28 -> 0.48 SD) |
| M5 comparison of results % | 2.99 to 17.8 | 11.3 | 12.0 | inside the range both times |
| M6 interpretation % | 0.00 to 9.59 | 7.04 | 8.57 | inside the range both times |
| M7 comparison with literature % | 0.00 to 17.9 | 0.47 | 0.57 | inside the range both times |
| M8 claim of strength % | 15.1 to 29.9 | 3.76 | 2.86 | held (below, 0.35 -> 0.38 SD) |
| M9 limitation % | 0.00 to 5.48 | 28.2 | 18.3 | narrowed (above, 2.95 -> 1.66 SD) |
| M10 link % | 1.37 to 4.76 | 5.16 | 6.29 | widened (above, 0.05 -> 0.17 SD) |
| M11 recommendation % | 16.4 to 17.9 | 13.2 | 12.6 | held (below, 0.12 -> 0.14 SD) |

## Small samples

The discussion sections now hold 71 sentences and the conclusion sections 186. In discussion, one sentence is 1.4% of the sentences, so a single sentence can move a share or a move percentage across a narrow reference range. The discussion reference range for "% sentences with 3+ numbers" is 1.98 to 2.08; the thesis went from 2 of 77 sentences to 0 of 71.

## Words per chapter, before -> now

Count 1 is every word of the chapter in `thesis.tex`, comment lines and inline comments removed, tables and captions included (the count the edit logs used). Count 2 is words in the measured prose only (figures, tables, captions and equations removed; citations and cross-reference numbers not counted).

| Chapter | count 1 before | count 1 now | count 2 before | count 2 now |
|---|---|---|---|---|
| 1 | 2,024 | 2,018 | 1,908 | 1,902 |
| 2 | 7,756 | 7,752 | 6,521 | 6,517 |
| 3 | 13,050 | 13,017 | 10,533 | 10,498 |
| 4 | 7,549 | 7,505 | 5,229 | 5,184 |
| 5 | 22,077 | 21,410 | 16,128 | 15,452 |
| 6 | 6,017 | 5,811 | 5,181 | 4,975 |
| 7 | 12,631 | 12,376 | 10,582 | 10,263 |
| 8 | 1,532 | 1,417 | 1,522 | 1,408 |

Every chapter is lower on both counts.

## Verification

(a) **Paragraphs, captions and titles without an AI marker.** Every non-comment line of the backup was compared with `thesis.tex` (python3 -I, difflib over lines). A line counts as AI-marked when a `% TODO: manual review` or `% TODO: review, ...` comment sits above it in the same blank-line block. Changed lines outside marked blocks: 0, so nothing was reverted. Rounding-only changes are allowed by the rule; none were needed outside marked blocks. The checker was tested by inserting a change into an unmarked paragraph and a section title in a copy; it found both.

(b) **`\paragraph`**: 0 occurrences in `thesis.tex`.

(c) **Chapter word counts**: lower in all eight chapters (table above).

(d) **Sentences over 30 words in AI paragraphs edited in this pass.** 158 paragraphs changed. Nine AI-marked paragraphs had a sentence over 30 words (most were edited only by rounding, so the long sentence was older). Each was split in two during this check, with no number or claim changed:
- Ch1 Introduction: the list of baselines (31 words).
- Ch2 noise injector: Gaussian noise or a shift of 2, 5 or 10% of labels (38 words). Rewritten one word shorter so Chapter 2 stays below its backup count.
- Ch4 ties table: the definition of augmentation features (32 words).
- Ch4 earlier noise-handling study: what a configuration could add (32 words).
- Ch5 hERG K_i walk through the cuts: the 0.547 RMSE configuration and its fold spread (31 words).
- Ch5 conformal intervals: coverage and width (41 words).
- Ch7 cleaning before the search, SMAC with clipped labels (32 words).
- Ch7 retuning on damaged labels, 0.691 against 0.703 (33 words).
- Ch7 neighbors in the hybrid or the own space, LogD (32 words).

Two long sentences remain, both left alone on purpose:
- Ch5 greedy forward selection, "It was also less accurate: in preliminary runs on Jain antibody ..." (34 words). Its marker reads "manual review -- fixed:", which the first report counted as your paragraph with a fix note. This pass changed only the rounding of 0.3600 and 0.4526.
- Ch8 Limitations: a `[RESULT NEEDED: ...]` placeholder runs into the next sentence in the count. The sentence itself is 15 words.

(e) **Cross-references**: the build log has no undefined reference, no undefined citation and no multiply defined label.

## Build

`pdflatex`, `bibtex`, `pdflatex`, `pdflatex` on `main.tex` into `scratchpad/thesis_code_todo/build`. No LaTeX errors. BibTeX gave one warning: empty year in the `rdkit` entry. Output: 265 pages, copied to `scratchpad/thesis_code_todo/thesis.pdf`.

