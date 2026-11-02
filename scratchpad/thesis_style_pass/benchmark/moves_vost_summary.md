# Rhetorical moves, reference: Vost (2025) DPhil thesis

Source: `~/Documents/ReferencePapers/Theses/text/Vost_2025_Machine_learning_for.txt` (pdftotext -layout of the PDF next to it). METHOD.md and a build-step segment file for "vost" were not in this folder when this was run (10 Oct, 03:48 to about 04:15), so `segment_vost.py` here does its own segmentation, in the same form as `segment_buttenschoen.py`. Sections taken: 2.5 Results and 2.6 Conclusions; 3.4 Results and discussion and 3.5 Conclusions; 4.4 Results and 4.6 Conclusions; 5.4 Results and 5.5 Conclusions; all of Chapter 6 (Conclusions and outlook). Section 4.5 Related work is left out. Chapter 3 has no separate discussion section: its 3.4 Results and discussion is counted as results, and the thesis has no section titled Discussion anywhere, so the discussion column is empty. Figure and table captions, table rows, running page headers and page numbers removed. In this text a paragraph starts at an indented line, not a blank line; paragraphs cut by a page break or a float rejoined.

Sample: the segmentation gives 417 sentences in 110 paragraphs, above the 250 cap, so `sample_vost.py` (a copy of `sample_buttenschoen.py`) drew a stratified sample: strata = chapter x section type, each stratum gets its share of 250 in proportion to its size. The unit drawn is a whole paragraph, so the move sequence inside each kept paragraph is intact for the bigram and first-sentence counts; whole paragraphs make the total 258 sentences, not exactly 250. Sentence ids kept: `sample_vost_ids.txt`; seed 20261010.

Moves coded by reading each sentence (scheme after Yang & Allison 2003 and Swales 1990), one move per sentence, in five batches (one per chapter), re-reading the scheme before each. Codes: `codes_vost.txt`. Decision rules used where two moves fit: a sentence that sets two named methods, conditions, datasets or a before/after pair side by side is M5 even when it carries numbers; M3 is a number reported for one condition. A sentence that describes what was run, even inside a results section, is M1. A citation used as motivation for the next experiment is M1, not M7; M7 is kept for setting a result of this thesis against a published one. `hedged` coded by reading (may, might, could, suggest, appear, seem, likely, possibly, perhaps, some evidence, approximately, around, somewhat, at least partially). `contains_number` is mechanical, the same rule as `build_moves_buttenschoen.py` (any digit or spelled number word, including ordinals such as first and second), after removing figure/table/section references, citation years, PDB identifiers and digits inside names (QM9, CASF16, Core80, Boltz-2, GFN2, 3D). A second reading-based number tag is in the third field of `codes_vost.txt`. One coder, one pass, no second-coder agreement measured.

Sentences: 258 (results 195, discussion 0, conclusion 63); paragraphs: 70.

Reading-based number tag (a quantity or threshold stated in the sentence; ordinals and counts of methods no): results 30.8% (60 of 195 sentences); conclusion 6.3% (4 of 63 sentences).

## Move shares per section type

One row is one move; each cell is the percentage of that section type's sentences coded with it (sentence count in brackets).

| Move | results | discussion | conclusion | all |
|---|---|---|---|---|
| M1 background/method restatement | 33.3% (65) | none | 15.9% (10) | 29.1% (75) |
| M2 pointer to figure/table/section | 3.1% (6) | none | 0.0% (0) | 2.3% (6) |
| M3 result with numbers | 8.2% (16) | none | 3.2% (2) | 7.0% (18) |
| M4 result in words | 8.2% (16) | none | 14.3% (9) | 9.7% (25) |
| M5 comparing results | 18.5% (36) | none | 12.7% (8) | 17.1% (44) |
| M6 interpreting/explaining | 14.9% (29) | none | 6.3% (4) | 12.8% (33) |
| M7 comparing with literature | 0.0% (0) | none | 0.0% (0) | 0.0% (0) |
| M8 evaluating/claiming strength | 3.6% (7) | none | 25.4% (16) | 8.9% (23) |
| M9 limitation or caveat | 2.1% (4) | none | 0.0% (0) | 1.6% (4) |
| M10 transition/linking | 6.2% (12) | none | 4.8% (3) | 5.8% (15) |
| M11 recommendation or implication | 2.1% (4) | none | 17.5% (11) | 5.8% (15) |

## Number and hedge tags per section type

Percentage of sentences tagged yes.

| Section type | contains_number | hedged |
|---|---|---|
| results | 40.5% | 13.8% |
| discussion | none | none |
| conclusion | 20.6% | 15.9% |

## Results move shares by chapter

Percentage of each chapter's results sentences (n in header).

| Move | Ch2 (n=83) | Ch3 (n=48) | Ch4 (n=18) | Ch5 (n=46) |
|---|---|---|---|---|
| M1 | 37.3% | 29.2% | 27.8% | 32.6% |
| M2 | 4.8% | 4.2% | 0.0% | 0.0% |
| M3 | 8.4% | 10.4% | 5.6% | 6.5% |
| M4 | 6.0% | 6.2% | 5.6% | 15.2% |
| M5 | 20.5% | 14.6% | 22.2% | 17.4% |
| M6 | 10.8% | 18.8% | 33.3% | 10.9% |
| M7 | 0.0% | 0.0% | 0.0% | 0.0% |
| M8 | 4.8% | 2.1% | 0.0% | 4.3% |
| M9 | 2.4% | 0.0% | 0.0% | 4.3% |
| M10 | 3.6% | 12.5% | 5.6% | 4.3% |
| M11 | 1.2% | 2.1% | 0.0% | 4.3% |

## Commonest move bigrams within a paragraph, results

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 140 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M1 -> M1 | 40 | 28.6% |
| M5 -> M5 | 11 | 7.9% |
| M5 -> M6 | 9 | 6.4% |
| M6 -> M6 | 7 | 5.0% |
| M1 -> M3 | 5 | 3.6% |
| M3 -> M5 | 4 | 2.9% |
| M3 -> M6 | 4 | 2.9% |
| M4 -> M4 | 4 | 2.9% |
| M6 -> M5 | 3 | 2.1% |
| M1 -> M2 | 3 | 2.1% |
| M10 -> M10 | 3 | 2.1% |
| M5 -> M3 | 3 | 2.1% |

## Commonest move bigrams within a paragraph, conclusion

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 48 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M8 -> M8 | 7 | 14.6% |
| M1 -> M4 | 5 | 10.4% |
| M11 -> M11 | 4 | 8.3% |
| M1 -> M1 | 3 | 6.2% |
| M4 -> M5 | 3 | 6.2% |
| M5 -> M6 | 3 | 6.2% |
| M8 -> M11 | 3 | 6.2% |
| M5 -> M4 | 2 | 4.2% |
| M10 -> M8 | 2 | 4.2% |
| M11 -> M8 | 2 | 4.2% |
| M4 -> M1 | 1 | 2.1% |
| M4 -> M8 | 1 | 2.1% |

## Commonest move bigrams within a paragraph, all

One bigram is a sentence's move followed by the next sentence's move in the same paragraph; 188 bigrams in all.

| Bigram | count | share |
|---|---|---|
| M1 -> M1 | 43 | 22.9% |
| M5 -> M5 | 12 | 6.4% |
| M5 -> M6 | 12 | 6.4% |
| M6 -> M6 | 8 | 4.3% |
| M8 -> M8 | 7 | 3.7% |
| M1 -> M3 | 5 | 2.7% |
| M3 -> M5 | 5 | 2.7% |
| M4 -> M5 | 5 | 2.7% |
| M1 -> M4 | 5 | 2.7% |
| M6 -> M5 | 4 | 2.1% |
| M3 -> M6 | 4 | 2.1% |
| M4 -> M4 | 4 | 2.1% |

## First sentence of each paragraph

Share of paragraphs whose first sentence is a finding in words or a comparison (M4/M5), against a method restatement or a result with numbers (M1/M3). Single-sentence paragraphs included.

| Section type | paragraphs | M4/M5 first | M1/M3 first | M1 first | M3 first | other first |
|---|---|---|---|---|---|---|
| results | 55 | 38.2% (21) | 40.0% (22) | 20 | 2 | 21.8% (12) |
| conclusion | 15 | 20.0% (3) | 33.3% (5) | 5 | 0 | 46.7% (7) |
| all | 70 | 34.3% (24) | 38.6% (27) | 25 | 2 | 27.1% (19) |

First-sentence moves of results paragraphs, all: M1 20, M2 2, M3 2, M4 8, M5 13, M6 3, M8 2, M10 5.

