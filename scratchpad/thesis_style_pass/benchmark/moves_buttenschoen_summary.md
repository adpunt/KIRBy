# Rhetorical moves, reference: Buttenschoen (2025) DPhil thesis

Source: `~/Documents/ReferencePapers/Theses/text/Buttenschoen_2025_Deep_learning-based_proteinligand.txt` (pdftotext of the PDF next to it). Segmented by `segment_buttenschoen.py` into `segments_buttenschoen.jsonl`; METHOD.md and the build-step segment files were not in this folder, so the segmentation is this script's own. Sections taken: x.3 Results, x.4 Discussion and x.5 Conclusion of Chapters 2 to 5, plus the whole of Chapter 6 (Conclusion). Chapter 1 is the introduction and has none of these. Figure and table captions, table rows, page numbers and figure axis labels removed; paragraphs cut by a page break or a float rejoined.

Sample: the segmentation gives 366 sentences in 86 paragraphs, above the 250 cap, so `sample_buttenschoen.py` drew a stratified sample (strata = chapter x section type, each stratum gets its share of 250 in proportion to its size). The unit drawn is a whole paragraph, so the sequence of moves inside each kept paragraph is intact for the bigram and first-sentence counts; whole paragraphs make the total 262 sentences, not exactly 250. Sentence ids kept: `sample_buttenschoen_ids.txt`; seed 20261010.

Moves coded by reading each sentence (scheme after Yang & Allison 2003 and Swales 1990), one move per sentence, in five batches (one per chapter), re-reading the scheme before each. Codes: `codes_buttenschoen.txt`. `hedged` coded by reading (epistemic hedges and approximators: may, might, could, suggest, appear, seem, likely, possibly, perhaps, some evidence, about, around). `contains_number` is mechanical, the same rule as `build_moves_author_current.py` (any digit or spelled number word), after removing figure/table/section/chapter references, citation years, PDB identifiers and digits inside names (PDBbind 2020, CASF-2016, OpenFF 2.0.0, AlphaFold 3). One coder, one pass, no second-coder agreement measured.

Sentences: 262 (results 123, discussion 72, conclusion 67); paragraphs: 60.

## Move shares per section type

One row is one move; each cell is the percentage of that section type's sentences coded with it (sentence count in brackets).

| Move | results | discussion | conclusion | all |
|---|---|---|---|---|
| M1 background/method restatement | 12.2% (15) | 9.7% (7) | 10.4% (7) | 11.1% (29) |
| M2 pointer to figure/table/section | 18.7% (23) | 0.0% (0) | 0.0% (0) | 8.8% (23) |
| M3 result with numbers | 25.2% (31) | 5.6% (4) | 0.0% (0) | 13.4% (35) |
| M4 result in words | 12.2% (15) | 16.7% (12) | 13.4% (9) | 13.7% (36) |
| M5 comparing results | 20.3% (25) | 9.7% (7) | 3.0% (2) | 13.0% (34) |
| M6 interpreting/explaining | 7.3% (9) | 15.3% (11) | 0.0% (0) | 7.6% (20) |
| M7 comparing with literature | 0.8% (1) | 4.2% (3) | 17.9% (12) | 6.1% (16) |
| M8 evaluating/claiming strength | 0.8% (1) | 9.7% (7) | 29.9% (20) | 10.7% (28) |
| M9 limitation or caveat | 0.8% (1) | 9.7% (7) | 3.0% (2) | 3.8% (10) |
| M10 transition/linking | 1.6% (2) | 1.4% (1) | 4.5% (3) | 2.3% (6) |
| M11 recommendation or implication | 0.0% (0) | 18.1% (13) | 17.9% (12) | 9.5% (25) |

## Number and hedge tags per section type

Percentage of sentences tagged yes.

| Section type | contains_number | hedged |
|---|---|---|
| results | 53.7% | 8.9% |
| discussion | 27.8% | 13.9% |
| conclusion | 19.4% | 10.4% |

The mechanical number rule counts spelled words such as "seven methods", "first" and "two file formats". Coded by reading instead (a quantity or threshold stated in the sentence; spelled counts of data items yes, ordinals and counts of methods or file formats no), the share of sentences with a number is: results 48.8% (60 of 123 sentences); discussion 15.3% (11 of 72 sentences); conclusion 6.0% (4 of 67 sentences). The two tags disagree on 24 of 262 sentences, all of them mechanical yes, read no. The reading-based tag is in the third field of `codes_buttenschoen.txt`.

## Results move shares by chapter

Percentage of each chapter's results sentences (n in header).

| Move | Ch2 (n=35) | Ch3 (n=21) | Ch4 (n=40) | Ch5 (n=27) |
|---|---|---|---|---|
| M1 | 17.1% | 9.5% | 10.0% | 11.1% |
| M2 | 25.7% | 4.8% | 22.5% | 14.8% |
| M3 | 2.9% | 57.1% | 22.5% | 33.3% |
| M4 | 11.4% | 4.8% | 22.5% | 3.7% |
| M5 | 34.3% | 9.5% | 7.5% | 29.6% |
| M6 | 2.9% | 9.5% | 10.0% | 7.4% |
| M7 | 0.0% | 0.0% | 2.5% | 0.0% |
| M8 | 2.9% | 0.0% | 0.0% | 0.0% |
| M9 | 0.0% | 0.0% | 2.5% | 0.0% |
| M10 | 2.9% | 4.8% | 0.0% | 0.0% |
| M11 | 0.0% | 0.0% | 0.0% | 0.0% |

## Commonest move bigrams within a paragraph, results

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 98 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M3 -> M3 | 14 | 14.3% |
| M2 -> M5 | 8 | 8.2% |
| M5 -> M5 | 7 | 7.1% |
| M1 -> M2 | 6 | 6.1% |
| M3 -> M6 | 6 | 6.1% |
| M4 -> M2 | 5 | 5.1% |
| M2 -> M2 | 4 | 4.1% |
| M5 -> M4 | 3 | 3.1% |
| M4 -> M4 | 3 | 3.1% |
| M1 -> M1 | 3 | 3.1% |
| M1 -> M3 | 3 | 3.1% |
| M2 -> M4 | 3 | 3.1% |

## Commonest move bigrams within a paragraph, discussion

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 54 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M11 -> M11 | 4 | 7.4% |
| M4 -> M4 | 4 | 7.4% |
| M1 -> M1 | 3 | 5.6% |
| M5 -> M11 | 3 | 5.6% |
| M5 -> M6 | 3 | 5.6% |
| M6 -> M6 | 3 | 5.6% |
| M3 -> M6 | 3 | 5.6% |
| M6 -> M4 | 2 | 3.7% |
| M6 -> M9 | 2 | 3.7% |
| M9 -> M11 | 2 | 3.7% |
| M8 -> M5 | 2 | 3.7% |
| M4 -> M3 | 2 | 3.7% |

## Commonest move bigrams within a paragraph, conclusion

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 50 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M7 -> M7 | 6 | 12.0% |
| M8 -> M8 | 6 | 12.0% |
| M11 -> M11 | 4 | 8.0% |
| M4 -> M4 | 3 | 6.0% |
| M1 -> M8 | 3 | 6.0% |
| M1 -> M4 | 2 | 4.0% |
| M7 -> M8 | 2 | 4.0% |
| M4 -> M8 | 2 | 4.0% |
| M8 -> M1 | 2 | 4.0% |
| M4 -> M11 | 2 | 4.0% |
| M11 -> M7 | 2 | 4.0% |
| M8 -> M7 | 1 | 2.0% |

## Commonest move bigrams within a paragraph, all

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 202 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M3 -> M3 | 14 | 6.9% |
| M4 -> M4 | 10 | 5.0% |
| M3 -> M6 | 9 | 4.5% |
| M5 -> M5 | 8 | 4.0% |
| M2 -> M5 | 8 | 4.0% |
| M11 -> M11 | 8 | 4.0% |
| M1 -> M1 | 7 | 3.5% |
| M8 -> M8 | 7 | 3.5% |
| M1 -> M2 | 6 | 3.0% |
| M7 -> M7 | 6 | 3.0% |
| M4 -> M2 | 5 | 2.5% |
| M2 -> M2 | 4 | 2.0% |

## First sentence of each paragraph

Share of paragraphs whose first sentence is a finding in words or a comparison (M4/M5), against a method restatement or a result with numbers (M1/M3). Single-sentence paragraphs included.

| Section type | paragraphs | M4/M5 first | M1/M3 first | M1 first | M3 first | other first |
|---|---|---|---|---|---|---|
| results | 25 | 28.0% (7) | 48.0% (12) | 7 | 5 | 24.0% (6) |
| discussion | 18 | 27.8% (5) | 22.2% (4) | 4 | 0 | 50.0% (9) |
| conclusion | 17 | 17.6% (3) | 17.6% (3) | 3 | 0 | 64.7% (11) |
| all | 60 | 25.0% (15) | 31.7% (19) | 14 | 5 | 43.3% (26) |

First-sentence moves of results paragraphs, all: M1 7, M2 3, M3 5, M4 2, M5 5, M7 1, M10 2.

