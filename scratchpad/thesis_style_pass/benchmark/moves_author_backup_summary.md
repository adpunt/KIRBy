# Rhetorical moves, author_backup (thesis.tex.before_style_apply_2026-10-10)

Source: `/Users/apunt/repos/KIRBy/thesis.tex.before_style_apply_2026-10-10`, Chapters 3 to 8. METHOD.md and a build-step segment file for author_backup were not in this folder, so `segment_author_backup.py` (a copy of `segment_author_current.py` with the section rule changed) wrote `segments_author_backup.jsonl`.

Section types. The backup has no single `Results` heading in Chapters 3 and 4, so every section there between Methods/Introduction and Conclusions is results. Chapters 5 to 7: sections titled `Results: ...` are results; `Discussion` and `Limitations` are discussion; `Conclusion(s)` is conclusion. Chapter 8: `Conclusions` is conclusion, every other section is discussion. The author_current file used a different rule (its Chapters 5 and 7 have no separate Limitations or Discussion sections), so section-type counts between the two files are not like for like.

Coding. One move per sentence, scheme after Yang & Allison (2003) and Swales (1990), coded by reading each sentence, in seven batches of 180 to 230 sentences. One coder, one pass, no second-coder agreement measured. `hedged` = yes when the writer withholds full commitment (may, likely, probably, seem, partly as an explanation, "in general", "broadly", "has the potential"); approximators on a measured number ("about 0.54", "almost every dataset") were not counted. `contains_number` is mechanical: a numeral outside a name (ECFP4, QM9, Caco-2, K_i do not count) or a spelled count or fraction (two to twelve, twenty, hundred, thousand, half, third, quarter, tenth); "one" and ordinals do not count.

Non-prose lines kept as sentences: five figure or table file paths that the segmenter read as text (AB0510, AB0610, AB0645, AB0752, AB0776; coded M2), three `[TABLE NEEDED]` lines (AB0890, AB0923, AB1138; M2) and eight sentences from six `[RESULT NEEDED]` placeholders (AB0543-0544, AB0571-0572, AB0708, AB1029, AB1328, AB1375; coded M9). One sentence's text is withheld in the CSV because it names a patient-data source; its move is kept.

Sentences: 1454 (results 1017, discussion 277, conclusion 160). Paragraphs: 299 (results 209, discussion 58, conclusion 32).

## Move shares per section type

One row is one move. Each cell is the percentage of that section type's sentences coded with it, sentence count in brackets.

| Move | results | discussion | conclusion | all |
|---|---|---|---|---|
| M1 background or method restatement | 16.2% (165) | 11.2% (31) | 10.6% (17) | 14.6% (213) |
| M2 pointer to figure, table or section | 2.2% (22) | 0.0% (0) | 0.0% (0) | 1.5% (22) |
| M3 result with numbers | 16.3% (166) | 13.7% (38) | 4.4% (7) | 14.5% (211) |
| M4 result in words | 20.1% (204) | 12.6% (35) | 20.0% (32) | 18.6% (271) |
| M5 comparing results | 20.6% (210) | 11.2% (31) | 16.9% (27) | 18.4% (268) |
| M6 interpreting or explaining | 11.2% (114) | 6.5% (18) | 6.9% (11) | 9.8% (143) |
| M7 comparing with literature | 0.9% (9) | 2.5% (7) | 0.6% (1) | 1.2% (17) |
| M8 evaluating or claiming strength | 1.8% (18) | 9.0% (25) | 6.9% (11) | 3.7% (54) |
| M9 limitation or caveat | 7.1% (72) | 20.9% (58) | 13.1% (21) | 10.4% (151) |
| M10 transition or link | 2.6% (26) | 1.8% (5) | 5.6% (9) | 2.8% (40) |
| M11 recommendation or implication | 1.1% (11) | 10.5% (29) | 15.0% (24) | 4.4% (64) |

## Number and hedge tags per section type

Percentage of that section type's sentences tagged yes.

| Section type | sentences | contains_number | hedged |
|---|---|---|---|
| results | 1017 | 40.0% | 3.2% |
| discussion | 277 | 31.4% | 1.4% |
| conclusion | 160 | 21.9% | 1.9% |

## Results move shares by chapter

Percentage of each chapter's results sentences; n in the header is that chapter's results sentence count.

| Move | Ch3 (n=206) | Ch4 (n=186) | Ch5 (n=205) | Ch6 (n=115) | Ch7 (n=305) |
|---|---|---|---|---|---|
| M1 | 19.4% | 39.8% | 6.8% | 7.0% | 9.5% |
| M2 | 1.5% | 3.2% | 2.0% | 1.7% | 2.3% |
| M3 | 18.4% | 18.8% | 15.6% | 16.5% | 13.8% |
| M4 | 23.3% | 4.3% | 16.1% | 21.7% | 29.5% |
| M5 | 16.0% | 4.8% | 26.3% | 36.5% | 23.6% |
| M6 | 9.7% | 18.8% | 16.1% | 2.6% | 7.5% |
| M7 | 1.0% | 0.5% | 2.4% | 0.0% | 0.3% |
| M8 | 1.9% | 0.5% | 2.4% | 0.9% | 2.3% |
| M9 | 3.4% | 2.7% | 8.3% | 10.4% | 10.2% |
| M10 | 4.9% | 5.4% | 1.0% | 1.7% | 0.7% |
| M11 | 0.5% | 1.1% | 2.9% | 0.9% | 0.3% |

## Commonest move bigrams within a paragraph, results

A bigram is one sentence's move followed by the next sentence's move in the same paragraph. 808 bigrams in all; top 12 shown.

| Bigram | count | share of bigrams |
|---|---|---|
| M1 → M1 | 73 | 9.0% |
| M5 → M5 | 68 | 8.4% |
| M4 → M3 | 47 | 5.8% |
| M4 → M4 | 47 | 5.8% |
| M5 → M3 | 42 | 5.2% |
| M4 → M5 | 40 | 5.0% |
| M3 → M4 | 35 | 4.3% |
| M3 → M5 | 33 | 4.1% |
| M3 → M3 | 31 | 3.8% |
| M5 → M4 | 31 | 3.8% |
| M6 → M6 | 30 | 3.7% |
| M1 → M3 | 22 | 2.7% |

## Commonest move bigrams within a paragraph, discussion

A bigram is one sentence's move followed by the next sentence's move in the same paragraph. 219 bigrams in all; top 12 shown.

| Bigram | count | share of bigrams |
|---|---|---|
| M9 → M9 | 36 | 16.4% |
| M3 → M3 | 12 | 5.5% |
| M4 → M3 | 11 | 5.0% |
| M1 → M1 | 11 | 5.0% |
| M5 → M5 | 10 | 4.6% |
| M4 → M4 | 9 | 4.1% |
| M4 → M5 | 8 | 3.7% |
| M5 → M3 | 8 | 3.7% |
| M8 → M8 | 7 | 3.2% |
| M3 → M5 | 7 | 3.2% |
| M1 → M4 | 6 | 2.7% |
| M8 → M6 | 5 | 2.3% |

## Commonest move bigrams within a paragraph, conclusion

A bigram is one sentence's move followed by the next sentence's move in the same paragraph. 128 bigrams in all; top 12 shown.

| Bigram | count | share of bigrams |
|---|---|---|
| M11 → M11 | 13 | 10.2% |
| M4 → M5 | 11 | 8.6% |
| M5 → M4 | 11 | 8.6% |
| M9 → M9 | 10 | 7.8% |
| M4 → M4 | 7 | 5.5% |
| M5 → M5 | 5 | 3.9% |
| M1 → M1 | 5 | 3.9% |
| M6 → M6 | 5 | 3.9% |
| M5 → M11 | 3 | 2.3% |
| M4 → M3 | 3 | 2.3% |
| M3 → M4 | 3 | 2.3% |
| M1 → M5 | 3 | 2.3% |

## Commonest move bigrams within a paragraph, all

A bigram is one sentence's move followed by the next sentence's move in the same paragraph. 1155 bigrams in all; top 12 shown.

| Bigram | count | share of bigrams |
|---|---|---|
| M1 → M1 | 89 | 7.7% |
| M5 → M5 | 83 | 7.2% |
| M4 → M4 | 63 | 5.5% |
| M9 → M9 | 62 | 5.4% |
| M4 → M3 | 61 | 5.3% |
| M4 → M5 | 59 | 5.1% |
| M5 → M3 | 50 | 4.3% |
| M5 → M4 | 47 | 4.1% |
| M3 → M3 | 43 | 3.7% |
| M3 → M5 | 42 | 3.6% |
| M3 → M4 | 41 | 3.5% |
| M6 → M6 | 38 | 3.3% |

## First sentence of each paragraph

One row is one section type. "Finding first" = the first sentence is M4 or M5. "Method or number first" = M1 or M3. Paragraphs with one sentence are counted in the first block and left out of the second.


**all paragraphs**

| Section type | paragraphs | finding first (M4/M5) | method or number first (M1/M3) | M1 alone | M3 alone | other first moves |
|---|---|---|---|---|---|---|
| results | 209 | 43.1% (90) | 27.3% (57) | 52 | 5 | M2 16, M6 13, M7 1, M8 5, M9 11, M10 16 |
| discussion | 58 | 12.1% (7) | 19.0% (11) | 11 | 0 | M6 1, M7 2, M8 10, M9 14, M10 4, M11 9 |
| conclusion | 32 | 18.8% (6) | 25.0% (8) | 8 | 0 | M6 3, M8 4, M9 6, M10 2, M11 3 |
| all | 299 | 34.4% (103) | 25.4% (76) | 71 | 5 | M2 16, M6 17, M7 3, M8 19, M9 31, M10 22, M11 12 |

**paragraphs of two or more sentences**

| Section type | paragraphs | finding first (M4/M5) | method or number first (M1/M3) | M1 alone | M3 alone | other first moves |
|---|---|---|---|---|---|---|
| results | 194 | 46.4% (90) | 26.8% (52) | 47 | 5 | M2 7, M6 12, M7 1, M8 5, M9 11, M10 16 |
| discussion | 48 | 14.6% (7) | 16.7% (8) | 8 | 0 | M6 1, M7 1, M8 10, M9 13, M10 4, M11 4 |
| conclusion | 32 | 18.8% (6) | 25.0% (8) | 8 | 0 | M6 3, M8 4, M9 6, M10 2, M11 3 |
| all | 274 | 37.6% (103) | 24.8% (68) | 63 | 5 | M2 7, M6 16, M7 2, M8 19, M9 30, M10 22, M11 7 |
