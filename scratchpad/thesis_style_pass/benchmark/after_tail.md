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
