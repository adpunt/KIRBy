# Chapter 4 against the benchmark table (audit, 2026-10-10)

Read-only audit. Nothing in thesis.tex, main.tex or refs.bib was edited.
Chapter 4 is thesis.tex lines 1094–1514, "Limitations of the similarity assumption".

## Which paragraphs are AI-written

Twelve paragraphs carry a plain "% TODO: manual review" marker directly above them. One table row also does.

| tex line | Section | First words |
|---|---|---|
| 1131 | Methods | The later chapters use a different set |
| 1230 | Results | Two molecules are tied when a representation |
| 1315 | Results | The four measurements below use the six |
| 1411 | Results | This section reuses a small sweep of |
| 1417 | Results | The study tested 12 published noise-handling methods |
| 1421 | Results | We measured noise robustness by retention, $R^2$ |
| 1427 | Results | The study used one dataset, one property |
| 1434 | Results | Neighbor-based noise detection treats a label far |
| 1470 | Results | The study also found neighbors in an |
| 1482 | Conclusions | We found that methods built on a |
| 1489 | Conclusions | We tested the published methods and the |
| 1492 | Conclusions | Chapter~\ref{chap:method} builds the adaptive hybrid representation |
| 1507 | Table `tab:flaws:requirements`, last row | Features that are hard to read chemically |

Two markers sit on headings, not paragraphs: line 1113 above `\section{Methods}` and line 1133 above `\section{Results}`.

One marker is ambiguous. Line 1473 sits under `\section{Conclusions}`, with a blank line before the paragraph "On a single dataset, often very little of a standard representation varies". The rule needs the marker directly above, so that paragraph is counted as yours. Please confirm.

The paragraph at 1437 ("The neighbor-based noise scores, sample removal...") is marked "-- fixed:", so it is yours. The segmenter joins it to 1434 because only comment lines separate them; it was not edited.

No caption in Chapter 4 is AI-written. The one caption with a marker (line 1341) is marked "-- fixed:".

## How the numbers were made

- Segmentation: `benchmark/segment_full_author.py` run on the current thesis.tex, written to the session scratchpad, not over the shared `full_segments_author_current.jsonl`.
- Measures: `compute_metrics.sent_metrics` and `normalise`, and the code of `extra_measures.py` (MTLD, MATTR, sentence-length spread, GPT-2 surprisal, connectives, three-item lists, pronouns). Moves are joined from `moves_author_current.csv` by sentence text, as `split_hers_ai.py` does. Those move codes were made by an earlier session, not by me.
- "After" values: the fixes below applied to a scratch copy of thesis.tex, then measured the same way.
- Sizes are small. AI-written text in Chapter 4: Methods 4 sentences (71 words), Results 33 sentences (596 words), Conclusions 8 sentences (127 words). The reference tables need 20 sentences per cell for the added measures, so Methods and Conclusions values below are indications only.

## Structure table, rows that apply to Chapter 4

| Row | Status in Chapter 4 | Verdict and fix |
|---|---|---|
| Chapter opening links back | Your first sentence: "The previous chapters have been focused on the impact of label noise." | Done. |
| Chapter ends point forward | Last paragraph (AI, 1492) names Chapter 5 and its question, how to select features for a task. | Done. Keep as its own one-sentence paragraph. |
| Text between section title and first subsection | `\section{Methods}` goes straight to its only subsection. `\section{Results}` goes straight to `\subsection{Flaws in standard representations}`. | Not done. Fixes S1 and S2 below. |
| Section titles | Methods, Results, Conclusions; subsection titles are topic noun phrases in sentence case. | Done. |
| Numbers in results text | AI Results: 3.0 numbers per 100 words (references 1.2–5.9). One AI sentence carries more than one score pair (1470). | Mostly done. Option O3. |
| Results sentences pointing to a figure or table | AI Results: 2 of 32 coded sentences (6.3%; references 3.1–18.2%). | Done for the AI text. |
| Limitations | Conclusions: 6 of 24 sentences state a limitation, 3 in AI paragraph 1489 and 3 in your paragraph "Feature selection cannot fix every flaw". The three in 1489 repeat 1427 (Results) and your sentence at 1465. | Not done. Fix 13 cuts 1489; with fix 12 as well, 3 of 19 sentences remain limitations, all yours. |
| Hedges | AI Results 0.34 per 100 words (references 0.87–2.3). The weak-evidence claims are already hedged ("may", "probably"). | Not a target. See option O4 for the one claim whose strength two AI paragraphs state differently. |
| Statistics | No AI sentence makes a claim across folds. | Not a target. |
| Negative results | "We found that methods built on a distance did worst under label noise." Plain. | Done. |
| Caption length | Chapter 4 captions average 57 words (references 49.8–73.5). None is AI-written. | Not a target. |
| Figure/table references | The AI table row cites `Section~\ref{sec:flaws:representations}` while the other four rows cite quoted subsection names. That section also never shows the flaw the row names. | Not done. Fix 14. |
| Decimals | AI paragraphs: percentages at one decimal (8.1%, 44.4%, 27.8%, −0.9%, −3.0%); thresholds as set (0.6, 80%). | Done. Your paragraphs and the Chapter 4 tables give Spearman correlations at two decimals; not an AI target. |
| "you" in body | Two in your text (1138 "your dataset", 1303 "cannot tell you"). None in AI text. | Not a target. |

## Language tables, by section type

Values are for Chapter 4 AI-written paragraphs only. "Now" is the current text; "after" is with fixes 1–13 applied. The reference range is the lowest to highest of Buttenschoen, Dablander and Vost.

### Results (AI: 33 sentences, 596 words, 8 paragraphs; your Chapter 4 Results: 145 sentences)

| Measure | References | Yours (Ch4) | AI now | AI after | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 24.7–31.1 | 21.4 | 18.1 | 22.3 | Not a target in itself; short sentences are your preference. After the fixes the AI mean sits at yours. |
| % sentences over 30 words | 24.4–47.5 | 19.3 | 3.0 | 11.1 | Partly done. One long sentence per paragraph where the content allows it. |
| Dependent clauses per sentence | 0.58–1.0 | 0.59 | 0.30 | 0.37 | Not done. This is the fragmentation signal; fixes 2, 3, 6, 7, 10, 12 join clauses. |
| Words per paragraph | 99–178 | 86 | 75 | 75 | Not a target. Paragraph 1315 is a 38-word lead-in, which is its job. |
| Lexical density % | 59.3–62.9 | 56.1 | 62.0 | 62.2 | Inside. |
| Nominalizations per 100 words | 3.9–4.4 | 4.7 | 5.9 | 6.0 | Above, but the words are terms ("configurations", "selection", "detection", "retention"). Not a target. |
| % passive sentences | 26.2–30.6 | 17.2 | 12.1 | 14.8 | Not a target; you write in the active voice. |
| we/our/I per 100 words | 0.26–1.6 | 0.64 | 0.34 | 0.33 | Inside. |
| Hedges per 100 words | 0.87–2.3 | 0.71 | 0.34 | 0.33 | Below; not a target (see Hedges row above). |
| Boosters per 100 words | 0.24–0.48 | 0.10 | 0.17 | 0.17 | Below, at your level. Not a target. |
| AI-guide phrases per 100 words | 0.09–0.58 | 0.03 | 0.0 | 0.0 | Done. |
| Numbers per 100 words | 1.2–5.9 | 4.5 | 3.0 | 3.2 | Inside. |
| % sentences with 3+ numbers | 3.1–20.3 | 15.9 | 6.1 | 11.1 | Inside. The 1470 sentence is option O3. |
| Move: method/background restated | 8.6–33.7 | 22.1 | 53.1 | – | Above every reference. Four AI Results paragraphs describe how the exploratory study was run. Option O1 moves them to Methods. |
| Move: pointer to figure/table | 3.1–18.2 | 1.4 | 6.3 | – | Inside. |
| Move: result as numbers | 7.9–24.8 | 20.7 | 9.4 | – | Inside. |
| Move: result in words | 8.3–12.4 | 4.8 | 6.3 | – | Below, like yours. Not a target. |
| Move: comparison of results | 18.1–23.6 | 6.9 | 0.0 | – | Below. The AI paragraphs here describe a study rather than compare results; not a target. |
| Move: interpretation | 7.4–27.1 | 29.0 | 3.1 | – | Below; your paragraphs carry the interpretation. Not a target. |
| Move: comparison with literature | 0–0.83 | 4.1 | 9.4 | – | Above. All three sentences are in 1434. Option O2. |
| Move: claim of strength | 0.83–3.6 | 3.4 | 3.1 | – | Inside. |
| Move: limitation | 0.83–5.0 | 2.8 | 6.3 | – | Above. Both sentences are in 1427, which stays as the chapter's one statement of the sweep's limits once 1489 is cut (fix 13). |
| Move: link/transition | 1.7–6.4 | 4.8 | 3.1 | – | Inside. Fix 3 removes the one announcing sentence ("This section reuses..."). |
| Move: recommendation | 0–4.3 | 0.0 | 0.0 | – | Inside. |
| MTLD (higher = more varied) | 55.9–83.8 | 76.4 | 105.0 | 106.0 | Above, so more varied words than the references. Not a problem. |
| MATTR-50 % | 76.0–80.4 | 79.5 | 81.4 | 82.3 | Slightly above. Not a problem. |
| Sentence length SD (words) | 9.5–12.8 | 9.3 | 5.3 | 6.1 | Not done. The main AI marker here: almost every sentence is 12–24 words. The fixes raise it; the rest of the gap would need a few short sentences among the long, which only the content can justify. |
| Sentence length coefficient of variation | 0.42–0.44 | 0.46 | 0.31 | 0.29 | Not done, for the same reason. It falls after the fixes because the mean rises. |
| Mean change in length between consecutive sentences | 10.0–14.2 | 10.4 | 6.6 | 9.0 | Mostly done after the fixes. |
| GPT-2 surprisal, bits per token | 6.8–7.1 | 7.6 | 8.3 | 8.2 | Above, as for your text. Symbols and numbers drive it. Not a target. |
| Surprisal SD across sentences | 1.01–1.04 | 1.20 | 1.16 | 1.00 | Inside after the fixes. |
| Connectives per 100 words | 1.6–3.1 | 1.7 | 1.4 | 1.2 | Slightly below. Not a target. |
| Distinct connectives / connectives used | 0.21–0.32 | 0.34 | 0.50 | 0.57 | Above (few connectives, none repeated much: "also" 3, "so" 2). Not a problem. |
| % sentences opening with a formulaic connector | 5.2–8.8 | 1.4 | 0.0 | 0.0 | Below, like yours. Not a target. |
| Three-item lists per 100 sentences | 4.7–10.0 | 10.3 | 24.2 | 25.9 | Above. Most hits are real lists (datasets, distances, methods). One is a rhetorical triplet ("one dataset, one property and Gaussian noise"), removed by fix 8. One list is nested inside another, fixed by fix 5. The count stays high because the merges put more lists into fewer sentences. |
| Pronouns and demonstratives per 100 words | 2.7–5.8 | 5.7 | 4.1 | 4.0 | Inside. |

### Conclusions (AI: 8 sentences, 127 words, 3 paragraphs; too few for the added measures)

| Measure | References | Yours (Ch4) | AI now | AI after | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 25.0–32.5 | 19.1 | 15.9 | 22.0 | Not a target in itself. |
| % sentences over 30 words | 25.8–42.9 | 6.3 | 0.0 | 33.3 | Done after fix 12 (one sentence of 33 words). |
| Dependent clauses per sentence | 0.71–0.96 | 0.56 | 0.25 | 0.33 | Below. Not a target with three sentences left. |
| Words per paragraph | 95–139 | 76.5 | 42.3 | 33.0 | Below. 1492 is one sentence by design (forward pointer). Not a target. |
| we/our/I per 100 words | 1.3–1.9 | 0.0 | 1.6 | 1.5 | Inside. |
| Hedges per 100 words | 0.95–2.2 | 3.6 | 1.6 | 1.5 | Inside. |
| Numbers per 100 words | 0.04–0.43 | 3.3 | 0.0 | 0.0 | Done. |
| Move: claim of strength | 15.1–29.9 | 0.0 | 0.0 | – | Below. 1482 opens with the finding stated with "we" ("We found that methods built on a distance did worst under label noise"), coded as a result in words. A stronger claim would be an AI verdict. Not a target. |
| Move: limitation | 0–5.5 | 18.8 | 37.5 | 0.0 | Not done now; done after fix 13. |
| Move: link/transition | 1.4–4.8 | 0.0 | 25.0 | 50.0 | Above. Fix 12 cuts the link inside 1482. What remains is 1492, the chapter's forward pointer, which the structure table asks for. Done in substance. |
| Move: recommendation/implication | 16.4–17.9 | 37.5 | 0.0 | – | Below for AI text. Your paragraph "A representation that avoids these flaws would need five properties" carries it. Not a target. |
| Three-item lists per 100 sentences | 9.4–14.3 | 6.3 | 0.0 | 0.0 | Below. Not a target. |
| Sentence length SD (words) | 8.0–12.6 | 6.6 | 5.7 | 8.4 | Inside after the fixes. |

### Methods (AI: 1 paragraph, 4 sentences, 71 words)

Too small to compare with the references. Sentence lengths are 11, 10, 29 and 21 words. Fix 1 joins the two short ones. The three-item list hits are the distance names, a real list. Not a target otherwise.

### Introduction, Background, Discussion

Chapter 4 has no AI-written paragraphs in its Introduction and no Background or Discussion section.

## Fixes

Each fix quotes the paragraph's first eight words, the exact old text and the new text. No meaning or number changes.

Note on fixes 6 and 7: an earlier pass (`BENCH_EDITS_ch4.md`) split these sentences. The benchmark shows that the splits made the sentence lengths more uniform, so these fixes join them again.

**Fix 1.** Row: sentence length SD, change between consecutive sentences (Methods). Two sentences of 11 and 10 words in a row.
Paragraph: "The later chapters use a different set, chosen" (line 1131).
Old: `The later chapters use a different set, chosen after that study. The hybrid search's nearest-neighbor objective`
New: `The later chapters use a different set, chosen after that study: the hybrid search's nearest-neighbor objective`

**Fix 2.** Row: dependent clauses per sentence; fragment sentence. A one-line definition sentence follows the sentence that uses the term.
Paragraph: "Two molecules are tied when a representation gives" (line 1230).
Old: `for each dataset's base representations and augmentation features. Augmentation features are the small feature sets added to the hybrid after it is built (Section~\ref{sec:method:augmentation}).`
New: `for each dataset's base representations and for its augmentation features, the small feature sets added to the hybrid after it is built (Section~\ref{sec:method:augmentation}).`

**Fix 3.** Row: link sentence that only announces. The first sentence announces the subsection; the provenance it carries ("from earlier in this project") moves into the sentence that states the question.
Paragraph: "This section reuses a small sweep of published" (line 1411).
Old: `This section reuses a small sweep of published noise-handling methods from earlier in this project. In Chapter~\ref{chap:robustness}, model family explained the largest share of variance in noise robustness, and representation at most 8.1\% on QM9 (Table~\ref{tab:variance}). We therefore asked whether training methods built for noisy labels improve robustness, and whether representation still matters under them.`
New: `In Chapter~\ref{chap:robustness}, model family explained the largest share of variance in noise robustness, and representation at most 8.1\% on QM9 (Table~\ref{tab:variance}). Earlier in this project we therefore ran a small sweep of published noise-handling methods, to ask whether training methods built for noisy labels improve robustness and whether representation still matters under them.`
The "% TODO: Ch3 update" above this paragraph still applies to the 8.1%.

**Fix 4.** Row: fragment sentence; sentence length SD. "In all there were 221 configurations." is 6 words after an 11-word sentence.
Paragraph: "The study tested 12 published noise-handling methods on" (line 1417).
Old: `Others added an alternative GP kernel or an alternative loss function. In all there were 221 configurations.`
New: `Others added an alternative GP kernel or loss function, for 221 configurations in all.`

**Fix 5.** Row: three-item lists. A four-item list of method names sits inside the sentence's own three-step list, separated only by commas, so the reader cannot tell where the names end.
Paragraph: "The study tested 12 published noise-handling methods on" (line 1417).
Old: `Four methods, small-loss selection, confident learning, distance-based selection and multistage cleaning, detect suspected molecules, remove them and retrain (noise mitigation); the other eight change the training (noise-robust training).`
New: `Four methods detect suspected molecules, remove them and retrain (noise mitigation): small-loss selection, confident learning, distance-based selection and multistage cleaning; the other eight change the training (noise-robust training).`

**Fix 6.** Row: sentence length SD; dependent clauses. Two 15-word sentences in a row; the second qualifies the first.
Paragraph: "We measured noise robustness by retention, $R^2$ under" (line 1421).
Old: `Retention correlates more closely with clean $R^2$, so it separates robustness from accuracy less well. It is still a good enough proxy for comparing methods with and without a distance.`
New: `Retention correlates more closely with clean $R^2$, so it separates robustness from accuracy less well, but it is a good enough proxy for comparing methods with and without a distance.`

**Fix 7.** Row: sentence length SD. Two sentences of 10 and 16 words state one finding.
Paragraph: "We measured noise robustness by retention, $R^2$ under" (line 1421).
Old: `All 20 configurations with the lowest retention used a distance. None with a distance reached both a clean $R^2$ above 0.6 and a retention above 80\%.`
New: `All 20 configurations with the lowest retention used a distance, and none with a distance reached both a clean $R^2$ above 0.6 and a retention above 80\%.`
The TODO above this paragraph (whether "none" means the 20 worst or every configuration with a distance) still applies.

**Fix 8.** Row: three-item lists (rhetorical triplet "one X, one Y and Z"). Also, "an alternative explanation" did not say of what.
Paragraph: "The study used one dataset, one property and" (line 1427).
Old: `The study used one dataset, one property and Gaussian noise on every training label. It also added a distance to methods designed without one, so poor adaptation is an alternative explanation.`
New: `The study used Gaussian noise on every training label of a single property, the QM9 HOMO--LUMO gap. It also added a distance to methods designed without one, so poor adaptation is an alternative explanation for their low retention.`

**Fix 9.** Row: one consistent form (numbers). The same subsection writes "12 published noise-handling methods" (1417) and "the same twelve methods" (1427).
Paragraph: "The study used one dataset, one property and" (line 1427).
Old: `the same twelve methods`
New: `the same 12 methods`

**Fix 10.** Row: fragment sentence (9 words); dependent clauses.
Paragraph: "Neighbor-based noise detection treats a label far from" (line 1434).
Old: `concerns classification \citep{Brodley1999, frenay2014classification}. \citet{Martin2021} adapt`
New: `concerns classification \citep{Brodley1999, frenay2014classification}, but \citet{Martin2021} adapt`

**Fix 11.** No benchmark row; a reading fault. "which is no longer part of the method" can refer to the early hybrid or to greedy selection.
Paragraph: "The study also found neighbors in an early" (line 1470).
Old: `built by greedy forward selection, which is no longer part of the method.`
New: `built by greedy forward selection, a selection step no longer part of the method.`

**Fix 12.** Row: link/transition in conclusions (AI 25% against references 1.4–4.8%). The last sentence only points to Chapter 7, which 1427 already does. With it gone, the two remaining short sentences join.
Paragraph: "We found that methods built on a distance" (line 1482).
Old: `all used a distance (Section~\ref{sec:lnl_motivation}). Judging a label by its neighbors' labels usually made the model worse (Section~\ref{sec:flaws:neighbors}). Chapter~\ref{chap:noise} tests neighbor-based detection under other noise conditions.`
New: `all used a distance (Section~\ref{sec:lnl_motivation}), and judging a label by its neighbors' labels usually made the model worse (Section~\ref{sec:flaws:neighbors}).`

**Fix 13.** Row: limitation in conclusions (Chapter 4 now 25% of conclusion sentences; references 0–5.5%). Paragraph 1489 says three things already said in the chapter. "One dataset, Gaussian noise on every label" is in 1427. "Neighbors were noisy too" is in your sentence at 1465. "A distance added to methods designed without one" is in 1427. Cut the paragraph and its marker.
Paragraph: "We tested the published methods and the neighbor-based" (line 1489).
Old: `We tested the published methods and the neighbor-based cleaning on one dataset, the QM9 HOMO--LUMO gap, with Gaussian noise on every training label. Each molecule's neighbors were therefore noisy too. The sweep also added a distance to methods designed without one, so poor adaptation may explain part of their result.`
New: (paragraph removed)
An earlier pass made 1489 "the chapter's one limitations paragraph" but left 1427 in place, so the limits are stated twice. If you would rather keep the limits in Conclusions, cut the first two sentences of 1427 instead and keep its pointer to Chapter 7.

**Fix 14.** Row: figure/table references, one consistent form; also a pointer to a section that does not show the flaw. Lines 1136–1295 count constant features and tied molecules; no sentence there is about reading features chemically. The other four rows of the table name a quoted subsection.
Table row: "Features that are hard to read chemically" (line 1507, `tab:flaws:requirements`).
Old: `Features that are hard to read chemically (Section~\ref{sec:flaws:representations})`
New: `Features that are hard to read chemically (not measured in this chapter)`

## Structure fixes

**S1. Methods has one subsection and no text under its title.** Row: text between section title and first subsection. The only subsection is "Distance metrics for continuous molecular descriptors", moved there unchanged.
Option A: delete the line `\subsection{Distance metrics for continuous molecular descriptors}`, so the text sits under Methods directly. No lead-in is then needed. No text changes.
Option B: keep the subsection and add one sentence under `\section{Methods}`: `This section gives the distances used in this chapter and in the exploratory study of Section~\ref{sec:lnl_motivation}.`
Recommendation: A. A section with a single subsection is the reason a lead-in is missing.

**S2. Results has no text under its title.** Row: text between section title and first subsection. Your last Introduction paragraph ("The purpose of this chapter is to establish...") already gives the three parts, a few lines above, so a roadmap here would repeat it.
Option A: leave as is.
Option B: add under `\section{Results}`: `The results come in the order the chapter's questions were set: first the representations on their own (Section~\ref{sec:flaws:representations}), then the similarity computed over them (Section~\ref{sec:flaws:similarity}), then methods that take that similarity as input (Section~\ref{sec:flaws:consequences}).`
Recommendation: A, given the repetition. The benchmark row counts this as not done either way.

## Options for you to decide (not counted as fixes)

**O1. Method description inside Results.** In the AI Results text, 53% of coded sentences restate method (references 8.6–33.7%; your Chapter 4 Results 22%). The sentences are in 1417 (whole paragraph), 1421 (the first four sentences, on how retention is defined) and 1315 (datasets and distances for the four measurements). They describe how the exploratory study was run, which is what a Methods section holds. Moving 1417 and the first four sentences of 1421 under Methods would also give Methods more than one topic (see S1). Cost: the subsection "A small sweep of published noise-handling methods" would then open with your paragraph "The three groups of methods for noisy labels" and the findings. Nothing is reworded.

**O2. Literature inside Results.** Paragraph 1434 holds all three literature sentences in the AI Results text (9.4%; references 0–0.83%). It opens the subsection that tests neighbor-based detection, and Chapter 4 has no Discussion section, so Results is the only place near the test. Option: leave it. Its Wilson1972 citation repeats your Introduction paragraph "Any method that incorporates similarity measures" and Chapter 2; cutting the clause "an idea that goes back to editing nearest-neighbor training sets \citep{Wilson1972}" would remove the repeat.

**O3. Numbers in 1470.** "Its mean recovery with neighbor-based scores was $-0.9\%$ at $\sigma = 0.3$ and $-3.0\%$ at $\sigma = 0.6$, with 44.4\% and 27.8\% of configurations positive." This is two pairs in one sentence; your rule allows one. The table cannot take them as a row, because your sentence at 1442 says the table holds only single representations. Either keep both pairs, or drop the positive shares, which removes two numbers.

**O4. Two AI paragraphs state one claim at different strengths.** 1427 says poor adaptation "is an alternative explanation". 1489 says it "may explain part of their result". Fix 13 removes 1489. If you keep 1489 instead, choose one wording for both.

## Count

14 fixes to AI-written text (13 paragraph edits, 1 table row), 2 structure fixes, 4 options for you.
