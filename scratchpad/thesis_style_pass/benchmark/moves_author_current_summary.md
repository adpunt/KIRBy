# Rhetorical moves, author_current (thesis.tex as of 2026-10-10)

Source: `thesis.tex`, Chapters 3 to 8. Segmented by `segment_author_current.py` into `segments_author_current.jsonl`; METHOD.md and the build-step segment files were not present in this folder, so the segmentation here is this script's own. Results = every `\section{Results}`; conclusion = every `\section{Conclusions}` in Chapters 3-7 plus Chapter 8 "Concluding remarks"; discussion = the other Chapter 8 sections (Chapters 3-7 have no discussion section). Figures, tables, equations and itemize wrappers removed; list items kept as their own paragraphs.

Moves coded by reading each sentence (scheme after Yang & Allison 2003 and Swales 1990), one move per sentence. `hedged` coded by reading. `contains_number` is mechanical: any digit or a spelled number word (one to thousand, half, third, quarter, tenth, first, second, fifth) outside cross-references. One coder, one pass, no second-coder agreement measured. Two sentences are `[RESULT NEEDED]` placeholders (AC1006 coded M3, AC1220 coded M9).

Sentences: 1246 (results 951, discussion 74, conclusion 221); paragraphs: 274.

## Move shares per section type

One row is one move; each cell is the percentage of that section type's sentences coded with it (sentence count in brackets).

| Move | results | discussion | conclusion | all |
|---|---|---|---|---|
| M1 background/method restatement | 12.7% (121) | 1.4% (1) | 7.7% (17) | 11.2% (139) |
| M2 pointer to figure/table/section | 1.5% (14) | 0.0% (0) | 0.0% (0) | 1.1% (14) |
| M3 result with numbers | 13.7% (130) | 6.8% (5) | 3.6% (8) | 11.5% (143) |
| M4 result in words | 20.7% (197) | 21.6% (16) | 20.4% (45) | 20.7% (258) |
| M5 comparing results | 24.9% (237) | 13.5% (10) | 11.3% (25) | 21.8% (272) |
| M6 interpreting/explaining | 12.4% (118) | 5.4% (4) | 6.8% (15) | 11.0% (137) |
| M7 comparing with literature | 1.6% (15) | 0.0% (0) | 0.5% (1) | 1.3% (16) |
| M8 evaluating/claiming strength | 1.8% (17) | 18.9% (14) | 4.1% (9) | 3.2% (40) |
| M9 limitation or caveat | 7.3% (69) | 13.5% (10) | 27.6% (61) | 11.2% (140) |
| M10 transition/linking | 2.4% (23) | 6.8% (5) | 5.0% (11) | 3.1% (39) |
| M11 recommendation or implication | 1.1% (10) | 12.2% (9) | 13.1% (29) | 3.9% (48) |

## Number and hedge tags per section type

Percentage of sentences tagged yes.

| Section type | contains_number | hedged |
|---|---|---|
| results | 58.6% | 6.1% |
| discussion | 37.8% | 1.4% |
| conclusion | 37.1% | 5.0% |

## Results move shares by chapter

Percentage of each chapter's results sentences (n in header).

| Move | Ch3 (n=206) | Ch4 (n=175) | Ch5 (n=196) | Ch6 (n=87) | Ch7 (n=287) |
|---|---|---|---|---|---|
| M1 | 18.0% | 28.6% | 4.6% | 3.4% | 7.7% |
| M2 | 1.5% | 1.7% | 0.5% | 0.0% | 2.4% |
| M3 | 17.5% | 18.3% | 10.7% | 9.2% | 11.5% |
| M4 | 24.3% | 5.1% | 23.0% | 24.1% | 25.1% |
| M5 | 15.0% | 5.7% | 33.7% | 50.6% | 30.0% |
| M6 | 11.2% | 24.0% | 15.8% | 1.1% | 7.3% |
| M7 | 1.5% | 5.1% | 1.0% | 0.0% | 0.3% |
| M8 | 1.5% | 2.9% | 1.0% | 0.0% | 2.4% |
| M9 | 3.4% | 4.0% | 7.1% | 9.2% | 11.5% |
| M10 | 5.8% | 4.6% | 0.0% | 1.1% | 0.7% |
| M11 | 0.5% | 0.0% | 2.6% | 1.1% | 1.0% |

## Commonest move bigrams within a paragraph, results

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 750 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M5 -> M5 | 104 | 13.9% |
| M1 -> M1 | 54 | 7.2% |
| M4 -> M4 | 46 | 6.1% |
| M4 -> M5 | 46 | 6.1% |
| M4 -> M3 | 43 | 5.7% |
| M6 -> M6 | 33 | 4.4% |
| M3 -> M3 | 31 | 4.1% |
| M5 -> M4 | 31 | 4.1% |
| M3 -> M4 | 30 | 4.0% |
| M5 -> M6 | 20 | 2.7% |
| M3 -> M6 | 19 | 2.5% |
| M5 -> M3 | 18 | 2.4% |

## Commonest move bigrams within a paragraph, discussion

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 49 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M9 -> M9 | 7 | 14.3% |
| M4 -> M5 | 4 | 8.2% |
| M11 -> M11 | 4 | 8.2% |
| M10 -> M4 | 3 | 6.1% |
| M4 -> M3 | 3 | 6.1% |
| M5 -> M4 | 3 | 6.1% |
| M4 -> M4 | 3 | 6.1% |
| M8 -> M8 | 3 | 6.1% |
| M3 -> M4 | 2 | 4.1% |
| M8 -> M4 | 2 | 4.1% |
| M5 -> M3 | 2 | 4.1% |
| M3 -> M5 | 2 | 4.1% |

## Commonest move bigrams within a paragraph, conclusion

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 173 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M9 -> M9 | 41 | 23.7% |
| M5 -> M4 | 12 | 6.9% |
| M4 -> M5 | 11 | 6.4% |
| M4 -> M4 | 10 | 5.8% |
| M1 -> M1 | 9 | 5.2% |
| M11 -> M11 | 9 | 5.2% |
| M5 -> M5 | 7 | 4.0% |
| M6 -> M6 | 6 | 3.5% |
| M4 -> M11 | 5 | 2.9% |
| M1 -> M4 | 3 | 1.7% |
| M4 -> M3 | 3 | 1.7% |
| M3 -> M4 | 3 | 1.7% |

## Commonest move bigrams within a paragraph, all

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 972 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M5 -> M5 | 113 | 11.6% |
| M1 -> M1 | 63 | 6.5% |
| M9 -> M9 | 63 | 6.5% |
| M4 -> M5 | 61 | 6.3% |
| M4 -> M4 | 59 | 6.1% |
| M4 -> M3 | 49 | 5.0% |
| M5 -> M4 | 46 | 4.7% |
| M6 -> M6 | 40 | 4.1% |
| M3 -> M4 | 35 | 3.6% |
| M3 -> M3 | 31 | 3.2% |
| M3 -> M6 | 22 | 2.3% |
| M5 -> M6 | 21 | 2.2% |

## First sentence of each paragraph

Share of paragraphs whose first sentence is a finding in words or a comparison (M4/M5), against a method restatement or a result with numbers (M1/M3). Single-sentence paragraphs included.

| Section type | paragraphs | M4/M5 first | M1/M3 first | M1 first | M3 first | other first |
|---|---|---|---|---|---|---|
| results | 201 | 48.3% (97) | 22.9% (46) | 42 | 4 | 28.9% (58) |
| discussion | 25 | 16.0% (4) | 0.0% (0) | 0 | 0 | 84.0% (21) |
| conclusion | 48 | 18.8% (9) | 12.5% (6) | 6 | 0 | 68.8% (33) |
| all | 274 | 40.1% (110) | 19.0% (52) | 48 | 4 | 40.9% (112) |

First-sentence moves of results paragraphs, all: M1 42, M2 11, M3 4, M4 58, M5 39, M6 15, M7 2, M8 4, M9 11, M10 14, M11 1.

