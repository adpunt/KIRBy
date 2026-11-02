# Benchmark audit: Chapter 1 and the abstract

Written 2026-10-10 by Claude. Read-only: nothing in thesis.tex or main.tex was changed.

## What counts as AI-written here

AI-written paragraphs in Chapter 1, by the rule (a plain `% TODO: manual review` line directly above, not `-- fixed:` / `-- moved:`):

| Where (thesis.tex line) | Starts with | Words |
|---|---|---|
| 15, last four sentences only | "We then test whether distance in the hybrid ..." | 63 |
| 19–24, the four numbered questions | "How can the best set of features be selected ..." | 94 |
| 39 | "We compare the hybrid with each base representation ..." | 118 |
| 42 | "Chapter 6 asks whether a hybrid representation repairs ..." | 95 |
| 45 | "Chapter 7 returns to label noise, with the single ..." | 81 |
| 48 | "Chapter 8 answers the research questions. It argues ..." | 47 |
| 55 (marker sits above the Collaborations heading) | "This work was supervised by Garrett Morris ..." | 61 |

Two placement notes:
- Line 15: the marker says "added the last four sentences ... The first five sentences are yours". An older copy (`thesis.tex.before_folds_statement_2026-10-10`) confirms the first five are her text. Only the last four are counted and touched.
- Lines 19–24: the marker sits above `\begin{enumerate}`, so the segmenter files the items as hers. They are counted as AI-written here.

All other Chapter 1 paragraphs are hers or `-- fixed:` notes. They are accepted as they are.

The abstract (main.tex lines 150–160) has its markers above the first paragraph only. The second of those markers says it rewrote "Six paragraphs". So all six paragraphs are counted as AI-written.

Sizes: Chapter 1 AI-written text is 31 sentences, 582 words. Her Chapter 1 text is 75 sentences, 1,568 words. The abstract is 20 sentences, 296 words by `wc -w`.

These samples are small. One sentence moves a percentage by about 3 points. Treat every value as a reason to read the paragraph, not as a result.

Method: same segmentation as `benchmark/segment_full_author.py`, run on the current thesis.tex and written to the session scratchpad. The script is `ch1audit/run.py` + `rerun.py` in the session scratchpad. Measures are `compute_metrics.sent_metrics` and `normalise`, plus the `extra_measures.py` code (GPT-2 small, the same lists and regexes). The abstract has no reference row in BENCHMARK_TABLE.md. It is compared with the Introduction ranges.

## Language table, Introduction (Chapter 1 AI-written text and the abstract)

Range = lowest to highest of Buttenschoen, Dablander and Vost.

| Row | Range | Yours (Ch1, now) | Ch1 AI now | Abstract now | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 25.3–30.8 | 20.9 | 18.8 | 15.0 | Not a target. Short sentences are her preference. The abstract is the shortest; the merges below take it up a little. |
| % sentences over 30 words | 22.0–43.2 | 14.7 | 6.5 (2 of 31) | 0 | Not done. Not a target in itself, but the abstract has none at all. Fixes 1, 3, 6, 9 create one or two. |
| Dependent clauses per sentence | 0.71–0.85 | 0.64 | 0.45 | 0.35 | Not done. Fixes 1, 4, 12 join clauses with "whose", "but", "although". |
| Words per paragraph | 112–146 | 121 | 58 (prose only: 81) | 50 | Not a target. The four numbered questions pull the Ch1 mean down; outline paragraphs are one per chapter, as in hers (her Ch2 paragraph is 54 words). |
| Lexical density % | 60.6–62.1 | 60.2 | 57.9 | 60.4 | Not a target. Abstract inside her value; Ch1 AI text slightly low, from short function-word clauses. |
| Nominalizations per 100 words | 4.1–6.1 | 6.5 | 5.8 | 5.4 | Done. Inside. |
| % passive sentences | 24.8–30.8 | 14.7 | 19.4 | 20.0 | Not a target. She writes in active "we"; the AI value is already above hers. |
| we/our/I per 100 words | 1.4–1.7 | 0.89 | 1.55 | 1.67 | Done. Inside. |
| Hedges per 100 words | 0.72–1.0 | 1.08 | 0.86 | 1.0 | Done. Inside. |
| Boosters per 100 words | 0.31–0.36 | 0.26 | 0.17 | 0.33 | Not a target. One booster in 582 words; adding boosters is not wanted. |
| AI-guide phrases per 100 words | 0.17–0.48 | 0 | 0 | 0 | Done. |
| Numbers per 100 words | 0.24–1.1 | 0.64 | 0 | 0.33 | Not a target. The outline gives results in words, which follows her rule. The abstract has one digit ("13 base models"), a count of models, not a result. |
| % sentences with 3+ numbers | 0–3.3 | 2.7 | 0 | 0 | Done. |

## Added measures, Introduction

| Row | Range | Yours (Ch1) | Ch1 AI now | Abstract now | Verdict |
|---|---|---|---|---|---|
| MTLD | 82.9–124.7 | 74.7 | 66.7–71.6 | 77.8 | Not done. Two repeated phrases: "distance in the hybrid tracked ... more closely" twice in consecutive sentences (line 42); "the single representations and the hybrid side by side" at line 15 and line 45. Fixes 5, 10, 12. |
| MATTR-50 % | 80.9–84.5 | 78.7 | 76.6 | 81.2 | Same cause as MTLD; same fixes. |
| Sentence length SD (words) | 8.9–15.0 | 8.9 | 6.7 | 4.0 | Not done. Main defect. Ch1 AI sentences sit between 12 and 23 words; abstract between 7 and 22, mostly 11–20. Fixes 1–4, 6, 9, 11. |
| Sentence length coefficient of variation | 0.36–0.51 | 0.43 | 0.36 | 0.27 | Ch1 at the lower edge. Abstract not done; fixes 1–4. |
| Mean change in length between consecutive sentences | 10.2–14.8 | 10.5 | 7.2 | 4.5 | Not done. Same fixes as SD. |
| GPT-2 surprisal, bits per token | 6.5–6.9 | 7.2 | 8.4 | 8.1 | Not a target. Higher means less predictable to GPT-2; her own text is also above the range. Likely from compressed terms ("hybrids for distance", "labels as delivered"). No fix proposed, since those terms are defined in later chapters. |
| Surprisal SD across sentences | 0.87–1.06 | 1.22 | 1.09 | 1.16 | Not a target. Close to range, below hers. |
| Connectives per 100 words | 1.99–2.39 | 1.68 | 2.08 | 2.03 | Done. Inside. "also" 4 times in 582 words (3 in prose, 1 in Collaborations); fix 7 removes one. |
| Distinct connectives / connectives used | 0.28–0.30 | 0.58 | 0.58 | 1.0 | Not a target. Above the range means more varied; samples this small always score high. |
| % sentences opening with a formulaic connector | 6.1–8.4 | 4.0 | 0 | 0 | Not a target. Adding "However," for its own sake is not wanted. |
| Three-item lists per 100 sentences | 7.3–11.1 | 17.3 | 19.4 (6 lists) | 20.0 (4 lists) | Partly a target. Six Ch1 lists read one by one: four carry content she defined elsewhere (the three groups of noise methods; Ch8's three sections; question 4's three uses of importance, answered in that order in Chapter 8; the comparison methods). One list is changed (fix 8). In the abstract two of four "lists" are the regex catching a four-item list and "built or adapted for molecular data"; not changed. |
| Pronouns and demonstratives per 100 words | 5.8–6.4 | 4.7 | 5.2 | 5.1 | Done. Within 1 per 100 words of the range, near hers. |

## Structure table rows that apply to Chapter 1 and the abstract

| Row | Status | Fix |
|---|---|---|
| Chapter opening links back | Not a target: Chapter 1 is the first chapter. | – |
| Chapter ends point forward | Done in effect. The outline section points to every later chapter. The chapter itself ends with Collaborations, a statement of who did what, which is normal there. | – |
| Text between section title and first subsection | Not applicable: Chapter 1 has no subsections. | – |
| Title case / section titles | Done. "Outline of the thesis and contributions", "Collaborations": sentence case, noun phrases. | – |
| Numbers in results text | Not applicable (no results section). The outline states findings in words, no score pairs. | – |
| Limitations | Not applicable in Chapter 1. The Ch8 outline paragraph only names the limitations section. | – |
| Research questions | Done. Line 12 (hers) states four questions in prose; lines 19–24 state four as a numbered list. Chapter 8 "Summary of findings" answers the first half, then the second half as First–Fourth, in the same order. | – |
| Abstract | Done. 296 words by `wc -w` (limit 300). Six paragraphs, as in Buttenschoen's. One digit ("13 base models"), a model count, not a result. After fixes 1–4 it is 297 words. | Fixes 1–4 (sentence length only). |
| Introduction length | Done. About 2,150 words (1,568 hers + 582 AI-written, Collaborations included); target is about Dablander's 2,158. | – |
| Hedges per 1,000 words | Done for Chapter 1 AI text (8.6 per 1,000; references 7.2–10). | – |
| Negative/positive results | Done. The outline states negatives plainly: "on the reactions the hybrid gained nothing"; "prediction from neighbors was worse on several of them". | – |
| Originality | Done. Line 10 (hers) states the gap; the outline names the methods introduced; line 50 (hers) names the software contributions. | – |
| Caption length, Figure/table references | Not applicable: no floats in Chapter 1 or the abstract. | – |
| Decimals | Done. Only her paragraph at line 8 has decimals (0.68, 0.54, 0.50, 0.27: two places, literature values). | – |
| "you" in body | One left in Chapter 1, in her first question at line 12 ("when you instead consider"). Her text, her choice; not changed. | – |
| Chemistry wording | Not assessed here (separate guide). | – |

## Fix list

Each fix quotes the paragraph's first eight words, the exact old text, and the new text. No meaning or numbers change. Word counts in brackets are sentence lengths before → after.

### Abstract (main.tex)

**Fix 1.** Row: sentence length SD / mean change; dependent clauses. Two short sentences with the same subject; merge with "whose".
Paragraph: "This thesis investigates label noise in quantitative structure--activity relationship"
Old: `This thesis investigates label noise in quantitative structure--activity relationship (QSAR) models. Their labels carry measurement error, differences between laboratories and assay limits.`
New: `This thesis investigates label noise in quantitative structure--activity relationship (QSAR) models, whose labels carry measurement error, differences between laboratories and assay limits.`
[12 + 11 → 23; word count unchanged]

**Fix 2.** Row: sentence length SD; fragment (7-word sentence). The random-forest sentence belongs to the robustness finding before it.
Paragraph: "First, we introduced NoiseInject, an open-source package for"
Old: `model family decided robustness more than the representation did. Random forests were the most consistent model.`
New: `model family decided robustness more than the representation did; random forests were the most consistent model.`
[21 + 7 → 28; word count unchanged]

**Fix 3.** Row: sentence length SD / mean change (19, 17 words in a row); % over 30 words.
Paragraph: "Next, we traced why published methods for learning with"
Old: `do not vary or do not relate to the label. Each still adds to the distance between molecules, so methods built on that distance inherit the problem.`
New: `do not vary or do not relate to the label, yet each adds to the distance between molecules, so methods built on that distance inherit the problem.`
[19 + 17 → 36; "still" replaced by "yet", word count unchanged]

**Fix 4.** Row: sentence length SD (four sentences of 10–14 words); dependent clauses.
Paragraph: "Last, we returned to label noise with single"
Old: `Which method worked depended on the task, the data, the model and the noise. On small molecules the hybrid stayed ahead through every treatment.`
New: `Which method worked depended on the task, the data, the model and the noise, but on small molecules the hybrid stayed ahead through every treatment.`
[14 + 10 → 25; +1 word, abstract 297]

Not changed: the one-sentence last paragraph ("The noise in a molecular dataset should therefore ..."). It is the thesis claim, set apart on purpose; Buttenschoen's abstract also has six paragraphs.

### Chapter 1 (thesis.tex)

**Fix 5.** Row: MTLD (the phrase "the single representations and the hybrid side by side" is repeated at line 45); sentence length (13, 17).
Paragraph (line 15): "In an early study with Gaussian noise added" (AI sentences only)
Old: `We then test whether distance in the hybrid tracks the label better (Chapter~\ref{chap:similarity}). Last, we return to noisy labels with the single representations and the hybrid side by side (Chapter~\ref{chap:noise}).`
New: `We then test whether distance in the hybrid tracks the label better (Chapter~\ref{chap:similarity}), and last return to noisy labels to test each method on both the single representations and the hybrid (Chapter~\ref{chap:noise}).`
[13 + 17 → 33]

**Fix 6.** Row: sentence length (11-word closing sentence); the "therefore" sentence only completes the one before it.
Paragraph (line 15): "In an early study with Gaussian noise added"
Old: `The same problem appears in antibody representations, where the features that matter are few and buried in representations of the whole sequence. We therefore test the method on antibodies, and on chemical reactions.`
New: `The same problem appears in antibody representations, where the features that matter are few and buried in representations of the whole sequence; we therefore test the method on antibodies, and on chemical reactions.`
[22 + 11 → 33]
If fixes 5 and 6 both feel too long together, take only one; fix 5 matters more (repetition).

**Fix 7.** Row: sentence length SD (17, 17, 19 words, three sentences opening "We"); connectives ("also").
Paragraph (line 39): "We compare the hybrid with each base representation"
Old: `We compare the hybrid with each base representation and with an ensemble of one model per representation. We also compare it with principal components, recursive feature elimination and random features of the same size.`
New: `We compare the hybrid with each base representation, with an ensemble of one model per representation and, at the same number of features, with principal components, recursive feature elimination and random features.`
[17 + 17 → 32; "of the same size" becomes "at the same number of features", same meaning]

**Fix 8.** Row: three-item lists. Antibodies and reactions are other kinds of data; data at separate sites is a different setting. Two groups, not three.
Paragraph (line 39): "We compare the hybrid with each base representation"
Old: `We read the features it selects, and test the method on antibodies, reactions and data held at separate sites.`
New: `We also read the features it selects, and test the method on antibodies and reactions, and on data held at separate sites.`
[19 → 22; "also" moves here from the sentence removed in fix 7]

**Fix 9.** Row: fragment sentence (8 words); three sentences in a row opening "On ..."; sentence length SD.
Paragraph (line 39): "We compare the hybrid with each base representation"
Old: `On antibodies the gain depended on the dataset. On two antibody datasets the best single representation was one we built to read only the regions or mutations that vary. On the smaller antibody datasets an ensemble often did better, and on the reactions the hybrid gained nothing.`
New: `On antibodies the gain depended on the dataset: on two of them the best single representation was one we built to read only the regions or mutations that vary, and on the smaller ones an ensemble often did better. On the reactions the hybrid gained nothing.`
[8 + 21 + 18 → 41 + 6. The short last sentence keeps the negative result plain]

**Fix 10.** Row: MTLD ("distance in the hybrid ... tracked ... more closely" twice in a row).
Paragraph (line 42): "Chapter~\ref{chap:similarity} asks whether a hybrid representation repairs the failures"
Old: `On the antibody datasets and the reactions, distance in the hybrid also tracked the label more closely.`
New: `On the antibody datasets and the reactions it also tracked the label more closely,`
(continues into fix 11)

**Fix 11.** Row: sentence length SD; the "Yet" sentence holds two claims.
Paragraph (line 42): same
Old (with fix 10): `On the antibody datasets and the reactions, distance in the hybrid also tracked the label more closely. Yet prediction from neighbors was worse on several of them, so the benefit of a hybrid has to be measured on each dataset.`
New: `On the antibody datasets and the reactions it also tracked the label more closely, yet prediction from neighbors was worse on several of them. The benefit of a hybrid therefore has to be measured on each dataset.`
[17 + 23 → 26 + 13. "it" refers to "distance in the hybrid" in the sentence before]

**Fix 12.** Row: sentence length (14, 12); dependent clauses; MTLD ("worked best" also in the sentence before; "single representation(s)" twice).
Paragraph (line 45): "Chapter~\ref{chap:noise} returns to label noise, with the single"
Old: `On small molecules the hybrid stayed ahead of the single representation through every treatment. Which method worked best changed between the single representations and the hybrid.`
New: `On small molecules the hybrid stayed ahead of the single representation through every treatment, although the best method changed between the two.`
[14 + 12 → 23]

### Read and left as they are

- The four numbered questions (lines 19–24). They match Chapter 8's four answers in order and wording; question 4's three-part list is answered in three parts there. Changing them would break that match.
- Line 48, "Chapter 8 answers the research questions. It argues ...". Sentences of 5, 33 and 9 words: already varied. Its three-item list is Chapter 8's section names.
- Line 55, Collaborations ("This work was supervised by Garrett Morris ..."). Factual; 26 and 35 words.

## Expected effect (estimated, not recomputed)

Fixes 1–4 move the abstract's sentence length SD from 4.0 toward 7–8 words; still below the references (8.9–15.0). Fixes 5–12 move Chapter 1's AI text from 6.7 toward about 9, which is her own value (8.9). Recompute after applying, with the same scripts.
