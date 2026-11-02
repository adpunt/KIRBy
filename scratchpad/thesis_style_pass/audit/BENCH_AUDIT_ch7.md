# Chapter 7 against the benchmark table (audit, 2026-10-10)

Read-only audit. Nothing in `thesis.tex` was changed. Source: a copy of `thesis.tex` taken at 08:35 today, Chapter 7 at lines 4566–5994. Another workflow is building the Chapter 5–7 tables now, so table contents, table pointers and numbers in tables are not judged here.

Line numbers below are those of the 08:35 copy. Search for the quoted old text if lines have moved.

## Which text in Chapter 7 is AI-written

Almost all of it. 98 prose paragraphs carry a `% TODO: manual review` line directly above them with no `-- fixed:` or `-- moved:`. So do 28 of the 33 floats in `thesis.tex`, and the three tables brought in with `\input`. The five floats with no marker (tables at lines 4605, 4632, 4659, 4828 and 5915) count as the author's and are not judged.

The segmenting script (`segment_full_author.py`, the same rule as the benchmark table) calls 15 more pieces the author's, about 1,180 words. They are AI-written too:

- 6 are the second half of an AI paragraph after a display equation (lines 4693, 4721, 4791, 4796, 4806, 4892). The script starts a new paragraph after `\end{equation}`.
- 5 sit under a marker placed above their `\subsubsection` heading instead of above the paragraph (lines 4822, 4826, 4858, 4862, 4867).
- 4 are the list items under the AI lead-in at line 4872 (lines 4874–4877).

The measures below use the script's split, so they leave these 1,180 words out. One fix (Fix 40) is in one of them. The same misclassification puts them in the "Yours" column of the benchmark table, so that column carries some AI text.

No author prose was found in Chapter 7.

## Language measures

Computed with `compute_metrics.py` (`sent_metrics`, `normalise`) and the `extra_measures.py` functions, on the AI paragraphs of Chapter 7 alone. Surprisal is GPT-2 small on every AI sentence. Scripts and outputs: session scratchpad `ch7/` (`seg.py`, `metrics.py`, `apply.py`, `fixes.py`).

"After fixes" is the same measure after all 58 fixes below were applied to a scratch copy and the copy was re-segmented.

Reference range = lowest to highest of Buttenschoen, Dablander and Vost. "Yours" = the author's text of that section type across the whole thesis, from the benchmark table.

### Introduction (AI: 15 sentences, 259 words, 3 paragraphs)

Not a target for any row. Fifteen sentences give no stable mean, spread or share. Read instead: the three paragraphs open the chapter, name Chapter 6, state the two questions and the three groups of methods.

| Measure | Reference range | Yours | AI now | Verdict |
|---|---|---|---|---|
| Words per sentence | 25.3–30.8 | 20.8 | 17.3 | Not a target (15 sentences; short sentences are her preference). |
| Sentence length SD (words) | 8.89–15.0 | 9.31 | 6.46 | Not a target at 15 sentences. Lengths run 8 to 28 words. |
| Connectives per 100 words | 1.99–2.39 | 1.65 | 3.19 | Not a target. 8 connectives, 7 different ones; none repeated more than twice. |
| Pronouns and demonstratives per 100 words | 5.81–6.36 | 5.40 | 8.77 | Not a target. "this chapter", "we" and "it" in a chapter opening. |
| Lexical diversity, MTLD | 82.9–124.7 | 86.6 | 89.6 | Done (inside). |

I considered merging the two 9-word sentences of line 4572 and dropped it: the merge lowers the length SD from 6.46 to 5.90. No Introduction fix is listed.

### Methods (AI: 192 sentences, 3,276 words, 34 paragraphs)

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 22.3–28.0 | 20.1 | 17.1 | 18.0 | Not a target. Short sentences are her preference. |
| % sentences over 30 words | 21.1–32.7 | 12.2 | 3.6 | 3.8 | Not a target. |
| Dependent clauses per sentence | 0.46–0.95 | 0.50 | 0.54 | 0.59 | Done (inside). |
| Words per paragraph | 76.3–142.4 | 81.9 | 96.4 | 96.6 | Done (inside). |
| Lexical density % | 58.8–59.7 | 56.6 | 53.7 | 53.6 | Not a target. Plain verbs and function words; the author is at 56.6. |
| Nominalizations per 100 words | 3.6–5.0 | 3.8 | 2.6 | 2.6 | Not a target. Steps are named with verbs, the plain register she asked for. |
| % passive sentences | 40.8–45.6 | 28.2 | 20.3 | 20.8 | Not a target. "One experiment compares..." names the actor, which is accurate. |
| we/our/I per 100 words | 0.83–1.6 | 1.1 | 0.55 | 0.55 | Not a target, same reason. |
| Hedges per 100 words | 0.36–0.85 | 0.39 | 0.55 | 0.52 | Done (inside). |
| Boosters per 100 words | 0.14–0.26 | 0.10 | 0.09 | 0.09 | Not a target. Same as the author. |
| AI-guide phrases per 100 words | 0.0–0.10 | 0.01 | 0.0 | 0.0 | Done. |
| Numbers per 100 words | 1.8–2.1 | 2.5 | 1.5 | 1.5 | Not a target. Settings and counts are in the Methods tables (`tab:noise:kinds`, `tab:noise:levels`, `tab:noise:detectors`, `tab:noise:mitigation`, `tab:noise:robust`). |
| % sentences with 3+ numbers | 3.7–4.7 | 6.0 | 2.6 | 2.7 | Not a target, same reason. |
| MTLD | 56.9–81.9 | 69.6 | 73.0 | 74.3 | Done (inside). |
| MATTR-50 % | 75.2–79.8 | 78.5 | 77.8 | 77.9 | Done (inside). |
| Sentence length SD (words) | 12.1–18.1 | 9.19 | 6.82 | 6.77 | Not done. Merging fragments removes the shortest sentences, so the SD does not rise. It rises only if some sentences run past 35 words, which the author does not do either (her 9.19). The fixes target the fragments and flat runs she named, not this number. |
| Sentence length coefficient of variation | 0.56–0.67 | 0.47 | 0.42 | 0.39 | Not done, same reason. |
| Mean change in length between consecutive sentences | 12.5–17.1 | 9.13 | 8.60 | 8.45 | Not done, same reason. |
| GPT-2 surprisal, bits per token | 7.04–7.42 | 7.67 | 8.32 | 8.22 | Not a target. Higher means less predictable to GPT-2; dataset names and symbols (hERG $K_i$, IC$_{50}$, $\tilde s_i$) raise it. The author is above the range too. |
| Surprisal SD across sentences | 1.11–1.37 | 1.22 | 1.20 | 1.13 | Done (inside). |
| Connectives per 100 words | 1.16–1.76 | 1.80 | 2.19 | 2.09 | Partly. "so" fell from 18 to 16. 22 of the counted connectives are "first", "second" and "third" in "The first... The second..." enumerations. Fixes 37, 39 and 40 join three of these enumerations into fewer sentences; the words stay, so the count does not fall. |
| Distinct connectives / connectives used | 0.15–0.28 | 0.17 | 0.20 | 0.24 | Done (inside). |
| % sentences opening with a formulaic connector | 1.0–3.5 | 0.85 | 0.0 | 0.0 | Not a target. The author avoids them too. |
| Three-item lists per 100 sentences | 7.0–8.4 | 12.8 | 10.4 | 12.0 | Not a target. Read all 20 hits. Five are false hits ("blocker or non-blocker", "not on the level or the fold", "at three seeds, and reruns" and two more). The rest are content lists: k of 5, 10 and 20; the three detector kinds; the three selectors and three actions; the three spread-only detectors. It rises after fixes only because merges cut the sentence count. |
| Pronouns and demonstratives per 100 words | 3.6–5.5 | 5.04 | 5.97 | 5.95 | Not a target. Mostly "it"/"its" for an experiment or a method in the sentence before. |

### Results (AI: 287 sentences, 5,244 words, 58 paragraphs)

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 24.7–31.1 | 21.4 | 18.3 | 19.8 | Not a target. |
| % sentences over 30 words | 24.4–47.5 | 13.1 | 3.5 | 8.4 | Partly. Moves toward the author's 13.1. |
| Dependent clauses per sentence | 0.58–1.0 | 0.60 | 0.38 | 0.43 | Partly. |
| Words per paragraph | 99.2–177.7 | 107.5 | 90.4 | 90.2 | Partly. Three short paragraphs are merged or moved (Fixes 7, 8, 29). The rest of the gap is the author's own (107.5). |
| Lexical density % | 59.3–62.9 | 54.8 | 55.7 | 55.5 | Not a target. Same as the author. |
| Nominalizations per 100 words | 3.9–4.4 | 3.8 | 3.1 | 3.0 | Not a target. |
| % passive sentences | 26.2–30.6 | 14.9 | 7.7 | 8.4 | Not a target. Results name the model or the method as the actor. |
| we/our/I per 100 words | 0.26–1.6 | 0.67 | 0.25 | 0.20 | Not a target (at the lower edge). |
| Hedges per 100 words | 0.87–2.3 | 0.81 | 0.65 | 0.65 | Not a target. Weak evidence is stated as a fact ("No control has run", "picked out of many") rather than hedged, which follows her rule. |
| Boosters per 100 words | 0.24–0.48 | 0.23 | 0.08 | 0.06 | Not a target. |
| AI-guide phrases per 100 words | 0.09–0.58 | 0.05 | 0.0 | 0.0 | Done. |
| Numbers per 100 words | 1.2–5.9 | 2.9 | 2.65 | 2.76 | Done (inside). |
| % sentences with 3+ numbers | 3.1–20.3 | 9.5 | 4.5 | 5.6 | Done (inside). |
| MTLD | 55.9–83.8 | 69.9 | 67.4 | 66.4 | Done (inside). |
| MATTR-50 % | 76.0–80.4 | 78.1 | 77.1 | 76.9 | Done (inside). |
| Sentence length SD (words) | 9.52–12.8 | 8.46 | 6.52 | 6.57 | Not done. Same reason as Methods. |
| Sentence length coefficient of variation | 0.42–0.44 | 0.41 | 0.37 | 0.35 | Not done. |
| Mean change in length between consecutive sentences | 9.96–14.2 | 9.72 | 8.43 | 8.36 | Not done. |
| GPT-2 surprisal, bits per token | 6.80–7.14 | 7.77 | 8.65 | 8.63 | Not a target, same reason as Methods. |
| Surprisal SD across sentences | 1.01–1.04 | 1.22 | 1.06 | 1.02 | Done (inside after fixes). |
| Connectives per 100 words | 1.59–3.12 | 2.32 | 2.78 | 2.56 | Done (inside). |
| Distinct connectives / connectives used | 0.21–0.32 | 0.14 | 0.13 | 0.15 | Partly. 58 of the counted connectives are "first" and "second" in "first outer fold", "second hybrid", "first single representation": names, not connectives. They swamp the ratio. The real repeat is ", so": 24 uses before, 15 after (Fixes 13–28). |
| % sentences opening with a formulaic connector | 5.2–8.8 | 1.8 | 0.0 | 0.0 | Not a target. |
| Three-item lists per 100 sentences | 4.7–10.0 | 13.3 | 8.7 | 9.2 | Done (inside). |
| Pronouns and demonstratives per 100 words | 2.7–5.8 | 5.10 | 4.26 | 4.08 | Done (inside). |

Rhetorical moves, Results. The move codes exist for 247 of 287 sentences now and 183 of 251 after fixes, so the after column is a partial check. Shares are of coded sentences.

| Move | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Method/background restated | 8.6–33.7 | 18.7 | 7.7 | 7.7 | Not a target (slightly below). Fixes 11 and 12 remove two restatements that repeat earlier sections. |
| Pointer to figure/table | 3.1–18.2 | 1.00 | 2.8 | 3.3 | Done. The 40 uncoded sentences are mostly pointer sentences. Counted directly, 39 of 287 Results sentences (13.6%) name a table or figure, inside the range. 34 of 58 paragraphs carry a pointer. |
| Result as numbers | 7.9–24.8 | 16.9 | 17.0 | 18.6 | Done (inside). |
| Result in words | 8.3–12.4 | 19.9 | 22.7 | 23.5 | Not a target. Her rule is pattern in words with a pointer to the table. Fix 9 cuts three sentences that read table rows out. |
| Comparison of results | 18.1–23.6 | 15.2 | 23.5 | 27.9 | Not a target. Every Results paragraph compares a method with no cleaning or with a control; that is the chapter's design. |
| Interpretation | 7.4–27.1 | 16.2 | 8.9 | 6.0 | Done (inside before; the after value is on the partial code set). |
| Comparison with literature | 0.0–0.83 | 2.5 | 0.4 | 0.5 | Done. |
| Claim of strength | 0.83–3.6 | 1.7 | 2.4 | 2.7 | Done. |
| Limitation | 0.83–5.0 | 3.7 | 12.1 | 8.7 | Partly. Most are needed caveats (fold 0 only, no control, network spread, small antibody folds) and follow her rule to state weak evidence. Fixes 8, 44, 46, 54 and 56 merge or cut five; the line-4909 paragraph (Fix 5) states the three chapter-wide caveats once. |
| Link/transition | 1.7–6.4 | 4.2 | 1.2 | 1.1 | Not a target. Section lead-ins added by Fixes 5–7 are coded as results, not links. |
| Recommendation/implication | 0.0–4.3 | 0.0 | 1.2 | 0.0 | Done. |

Score pairs per paragraph (her rule: at most one). Read all 58 Results paragraphs: 56 have one or none. Two have two:

- Line 5394: a precision against a break-even (0.26 against 0.32) and an RMSE pair (0.587 to 0.557). Not a target; the first pair is a precision, not a score.
- Line 5945: Chapter 4's retention of 79.0% and a Spearman pair (0.341 to 0.448). Not a target; the first is a recall of Chapter 4.

### Conclusion (AI: 18 sentences, 278 words, 3 paragraphs)

Moves for the after column are hand-coded on the 12 sentences after fixes. The language measures are from the scripts.

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 25.0–32.5 | 21.0 | 15.4 | 23.8 | Done (just below the range, above the author). |
| % sentences over 30 words | 25.8–42.9 | 12.8 | 0.0 | 25.0 | Done. |
| Dependent clauses per sentence | 0.71–0.96 | 0.76 | 0.33 | 0.58 | Partly. |
| Words per paragraph | 95.3–138.9 | 123.1 | 92.7 | 143.0 | Done (just above the range). Two paragraphs; the 11-word end paragraph joins the second (Fix 2). |
| we/our/I per 100 words | 1.3–1.9 | 0.41 | 0.72 | 0.35 | Not a target. "we found" opens the chapter's findings; the author uses "we" less. |
| Hedges per 100 words | 0.95–2.2 | 1.4 | 0.36 | 1.40 | Done. |
| Sentence length SD (words) | 8.01–12.6 | 9.24 | 6.12 | 8.90 | Done. |
| Mean change in length between consecutive sentences | 8.53–12.3 | 9.40 | 9.27 | 9.30 | Done. |
| Three-item lists per 100 sentences | 9.4–14.3 | 12.8 | 27.8 | 33.3 | Not done. 4 hits in 12 sentences: the datasets that passed a control, and the order in which results should be confirmed (three lists). Each item is a named result. Splitting them would add sentences, not remove a triplet habit. |
| Limitation (move) | 0.0–5.5 | 15.3 | 33.3 | 16.7 | Partly. Fix 1 cuts two repeats and moves one status sentence to Methods. The two left (fold 0 only; no QM9 result for three methods) are specific. Near the author's 15.3. |
| Claim of strength (move) | 15.1–29.9 | 0.0 | 5.6 | 8.3 | Partly. "we found noisy labels... and, in several settings, corrected them" is the claim. Fix 4 adds the answer to the chapter's second question. |
| Recommendation/implication (move) | 16.4–17.9 | 18.8 | 11.1 | 8.3 | Not done. The one recommendation is the order in which to confirm results. The method-choice advice is in Chapter 8 (Fix 2 points there). |
| Link/transition (move) | 1.4–4.8 | 1.2 | 5.6 | 8.3 | Not a target. The one link is the forward pointer the structure table asks for. |
| GPT-2 surprisal, bits per token | 6.57–6.84 | 7.57 | 8.65 | 8.32 | Not a target. |
| Lexical diversity, MTLD | 104.2–167.3 | 80.8 | 142.5 | 142.3 | Done. |

## Structure rows that apply to Chapter 7

| Benchmark row | Status | Fix |
|---|---|---|
| Chapter opening links back | Done. Line 4572 names Chapter 6 (`chap:similarity`) and its finding. | None. |
| Chapter ends point forward | Not done. Line 5993 is an 11-word paragraph that only says Chapter 8 "brings these results together". | Fix 2. |
| Text between section title and first subsection | Not done in 3 places. `\section{Results}` (4897), `\subsection{Effect by noise type}` (5417) and `\subsection{Effect by dataset}` (5651) run straight into a subsection. Methods, Noise mitigation, Noise-robust training and the Results subsection "Changing the model instead of the labels" have lead-ins. | Fixes 5, 6, 7. All three move existing AI text; none adds new text. |
| Title case | Done. Every heading is sentence case. | None. |
| Section titles | Done, one word. "A shared offset per scaffold group" says "group" where the chapter says "scaffold family" throughout. Two titles repeat between Methods and Results ("Changing the model instead of the labels", "The expanded published sweep"); that pairing is deliberate and fine. | Fix 58. |
| Numbers in results text per 100 words | Done. 2.65, inside the references (1.2–5.9). Two paragraphs carry two pairs; both explained above. | None. |
| Results prose words per figure/table | Not a target. 5,244 AI words over 30 floats in Results, 175 words per float, below every reference (268–2,270). More floats than the references, which is what her rule asks for. | None. |
| Results sentences pointing to figure/table | Done. 13.6% counted directly (references 3.1–18.2). | Fix 30 removes "signed-off" from one pointer sentence. |
| Limitations | Partly. Conclusions: 6 of 18 sentences are limitations; the paragraph sits under an orphan `\label{sec:noise:limitations}` with no heading. Results: 12.1% of coded sentences. | Fixes 1, 8, 46, 54, 56. |
| Research questions | Partly. The Introduction (4572, 4575) asks whether noisy labels can be found and acted on, and whether a method helps the hybrid more than the best single representation. The Conclusions answer the first only. | Fix 4. |
| Hedges | Not a target. See the measures; weak evidence is stated, not hedged. | None. |
| Statistics | Done. No significance tests. Per-fold scores, draws and seeds are given; controls are named; "best of five random deletions" states that the control is optimistic (4990). | None. |
| Negative/positive results | Partly. Plain throughout except one sentence in the Conclusions: "through every treatment" is stronger than the Results' "almost every condition". | Fix 4. |
| Originality | Done. Methods states where no earlier method was found: cliff put-back (4776), the cleaned/recorded search-and-fit cross (4779), the buddy check on molecular labels (4685), the cliff-aware gap ("New" in the detectors table, line 4671). | None. |
| Caption length | Done. 28 AI captions in `thesis.tex`, mean 49.5 words, range 26–78, inside the reference means (49.8–73.5) at the lower edge. They follow her rule: what is shown, rows, metric, direction, fold. Two carry a status note ("Second run unfinished", line 5344; "The run is separate from that of Figure...", line 5457). Not judged here: the tables are being built now. | None. |
| Figure/table references | Done. "Table~\ref", "Figure~\ref", "Figures~\ref{} and~\ref{}" throughout. | None. |
| Decimals | Partly. AI prose has one four-decimal score pair (0.5072 and 0.3980, line 5595). Tables carry 18 four-decimal cells (0.8155, 0.8253, 0.4212 and others), each beside a near-tie; those belong to the table workflow. 1.4826 (three times) is a constant, not a score. | Fix 57. |
| Chemistry wording | Checked by reading. pIC$_{50}$ as the negative base-10 log of IC$_{50}$ in molar (4584), HOMO–LUMO gaps computed, not measured (4884), activity cliffs at ECFP4 Tanimoto ≥ 0.55 with a far label (4776), one ChEMBL record per measurement (4809): stated correctly. | None. |
| "you" in body | Done. No "you" in Chapter 7 prose. | None. |
| Abstract, Introduction length, final chapter length, front matter, acknowledgements, AI-use statement | Not a target. Not in this chapter. | None. |

## Content points found while reading (not language, no fix proposed)

- Line 4881 says "Five controls ran" and names four: random deletion, random put-back, shuffled detector scores, shuffled lifts. The fifth may be tuning on clean labels (line 4822) or QM9 (line 4884). Unverified which.
- `\label{sec:noise:discussion}` (5975) and `\label{sec:noise:limitations}` (5985) have no heading and nothing in `thesis.tex` refers to them.

## Fixes

58 fixes. Fixes 1–30 are ordered by importance. Fixes 31–58 are merges of short sentences, one decimal fix and one heading, in chapter order. Each quotes the first eight words of the paragraph as they appear in the `.tex` source, and the exact old text. No meaning or number changes. In the old text, ⏎ marks a line break.

### Fix 1. Line 5989, "Most results rest on the first outer fold."

Conclusions limitations paragraph: cut two repeats, move one status sentence to Methods. Row: Limitations (structure); limitation share in Conclusion; three-item lists.

Six of nine sentences are limitations (33% of the Conclusions sentences; references 0-5.5%, the author 15.3%). "Most methods have no control." repeats the Evaluation and controls paragraph. "QM9's HOMO--LUMO gap is our only clean control." repeats it a third time (Introduction, Evaluation and controls). The list of datasets with no noise results is a fact about the data, so it moves to the datasets paragraph of Methods (second edit below), where the TODO(missing result) line already sits. "Until those exist" becomes "yet" in a merged sentence. No result or caveat is lost.

- Old: `Most results rest on the first outer fold. The further folds that exist mostly agree with it, and the text gives the exceptions. The results to confirm first are, in order, the cleaning of SMAC colloidal stability's labels as delivered and relabeling and cliff put-back on hERG $K_i$. Next come cleaning under hERG's flips, the clipped-label corrections and the scaffold-family offsets. Most methods have no control. Changing the model, cleaning before the search and moving every label toward a Huber-fitted prediction gave large gains but have no QM9 result with no noise added. Until those exist, their gains cannot be split into noise removed and a better model. QM9's HOMO--LUMO gap is our only clean control. Rat microsomal clearance, anti-VEGF scan expression and binding, EGFR binding, angiopoietin-2 binding and HIC retention have no noise results yet.`
- New: `Most results rest on the first outer fold; the further folds that exist mostly agree with it, and the text gives the exceptions. The results to confirm first are, in order, the cleaning of SMAC colloidal stability's labels as delivered and relabeling and cliff put-back on hERG $K_i$, then cleaning under hERG's flips, the clipped-label corrections and the scaffold-family offsets. Changing the model, cleaning before the search and moving every label toward a Huber-fitted prediction gave large gains but have no QM9 result with no noise added, so their gains cannot yet be split into noise removed and a better model.`

Second edit, line 4584, "This chapter uses the datasets of Chapters~\ref{chap:method} and~\ref{chap:similarity}":

- Old: `Folds and metrics are those of Chapter~\ref{chap:method}.`
- New: `Folds and metrics are those of Chapter~\ref{chap:method}. Rat microsomal clearance, anti-VEGF scan expression and binding, EGFR binding, angiopoietin-2 binding and HIC retention have no noise results yet.`

### Fix 2. Line 5993, "Chapter~\ref{chap:discussion} brings these results together with the rest"

Chapter end only announces; it should name what comes next. Row: Chapter ends point forward; link sentence that only announces.

An 11-word paragraph. The new sentence names the two things Chapter 8 does (from its first paragraph, line 5998). Join it to the end of the limitations paragraph: delete the blank line and the % TODO: manual review line above it, and put the sentence before the % RESULT PENDING comment.

- Old: `Chapter~\ref{chap:discussion} brings these results together with the rest of the thesis.`
- New: `Chapter~\ref{chap:discussion} uses these results to answer the research questions of Chapter~\ref{chap:intro} and to set out how to choose a noise-handling method for a new dataset.`

### Fix 3. Line 5982, "On the first outer fold, we found noisy"

Conclusions: join the 6-word claim to its evidence. Row: Fragment; claim of strength in Conclusion.

The claim and the result that supports it are one statement.

- Old: `Corrections worked best with a flag. Raising labels clipped`
- New: `Corrections worked best with a flag: raising labels clipped`

### Fix 4. Line 5982, "On the first outer fold, we found noisy"

Conclusions: hybrid claim stronger than the Results; second question of the Introduction unanswered. Row: Negative/positive results (no inflation); research questions answered; claim of achievement.

Results (line 5963) say the hybrid stays ahead "in almost every small-molecule condition", not "through every treatment". The Introduction (line 4575) asks whether a method helps the hybrid more than the best single representation; the Conclusions do not answer it. The added sentence restates line 5968 and adds no new result. The second sentence is new text in the Conclusions; drop it if you prefer the Conclusions not to answer that question.

- Old: `The hybrid stayed ahead of the single representation on small molecules through every treatment.`
- New: `The hybrid stayed ahead of the single representation on small molecules in almost every condition, before and after treatment. Finding a detector's neighbors in the hybrid rather than in the model's own representation helped slightly more often, with large differences in both directions.`

### Fix 5. Line 4909, "Three things hold for every result in this"

Results section has no lead-in. Row: Text between section title and first subsection.

The paragraph states three caveats that hold for every result in the chapter, so it belongs to the whole section, not to its first subsection.

- Old (the paragraph): `Three things hold for every result in this chapter.`
- New: (no wording change) Move the whole paragraph, with its three comment lines (% SOURCE [N7-01], % TODO: NEW RESULTS [N7-01], % TODO: manual review), from below \subsection{Overall effect of noise handling} to directly below \section{Results} (line 4897).

### Fix 6. Line 5649, "Set side by side, the damage kinds sort"

"Effect by noise type" has no lead-in; its summary paragraph sits at the end. Row: Text between section title and first subsection; repeated content.

The paragraph sorts the damage kinds by whether a flag exists; that is the frame for the four subsubsections below it. The TODO above it says it overlaps the paragraph before; moving it removes the repetition at the end of the subsection. The TODO line stays.

- Old (the paragraph): `Set side by side, the damage kinds sort by whether a flag says which labels are wrong.`
- New: (no wording change apart from Fix 48) Move the whole paragraph, with its two % TODO: manual review lines and the % TODO: overlaps line, to directly below \subsection{Effect by noise type}\label{sec:noise:res:kind} (line 5417), so it leads the subsection instead of closing its last subsubsection.

### Fix 7. Line 5196, "On three antibody datasets, cleaning the labels as"

"Effect by dataset" has no lead-in; a one-sentence paragraph sits elsewhere. Row: Text between section title and first subsection; fragment paragraph.

The sentence is a one-sentence paragraph at the end of "The labels as delivered" whose only job is to point to the antibody section. As the lead-in of "Effect by dataset" it states the main pattern, and the pointer and "also" are no longer needed.

- Old (the paragraph): `On three antibody datasets, cleaning the labels as delivered also passes the shuffled-score control, and on those datasets it gave the largest gains (Section~\ref{sec:noise:res:antibodies}).`
- New: Move to directly below \subsection{Effect by dataset}\label{sec:noise:res:dataset} (line 5651), with its % TODO: manual review line, as: On three antibody datasets, cleaning the labels as delivered passes the shuffled-score control, and on those datasets it gave the largest gains.

### Fix 8. Line 4914, "Table~\ref{tab:noise:res:bygroup} shows how often each group of methods"

Group table paragraph re-describes the table caption; one-paragraph caveat folded in. Row: Re-describes the table -> cut, point to table; fragment paragraph.

The definition of a condition and the three outcomes are in the caption of tab_noise_bygroup.tex. The two-sentence caveat paragraph at line 4924 joins this paragraph as one sentence. Its "% TODO: decide" line stays where it is (second edit below), but its own "% TODO: manual review" line goes, since the paragraph is gone.

- Old: `A condition is one dataset, noise condition, model and method. ⏎ % TODO: manual review ⏎ A row sorts its conditions by outcome: the method improved both the best single representation and the best hybrid, only one of them, or neither. Where a method has a control, improving both also means beating it. Mitigation helps most often under injected damage, and least often on the labels as delivered.`
- New: `Mitigation helps most often under injected damage, and least often on the labels as delivered. The table counts fold 0 only: six of its experiments also have results on folds 1 and 2, and it leaves out MDR1--MDCK efflux, which has 24 conditions on fold 0.`

Second edit, line 4924, "Six of the table's experiments also have results" (delete the paragraph and its "% TODO: manual review" line, keep the "% TODO: decide" line):

- Old: `% TODO: manual review ⏎ % TODO: decide: the table counts n18's down-weight, label smoothing and out-of-fold move on labels as delivered (57 conditions outside QM9, 14 both improved); the signed-off list keeps only n18's deletion. ⏎ Six of the table's experiments also have results on folds 1 and 2, which it does not count. It also leaves out MDR1--MDCK efflux, which has 24 conditions on fold 0.`
- New: `% TODO: decide: the table counts n18's down-weight, label smoothing and out-of-fold move on labels as delivered (57 conditions outside QM9, 14 both improved); the signed-off list keeps only n18's deletion.`

### Fix 9. Line 5046, "Weighting keeps every molecule and every label. It"

Weighting paragraph reads the table out row by row. Row: Re-describes the table -> cut, point to table; result in words (above references).

Three sentences restate the detector, model and configuration of each table row. The one pattern the table does not show at a glance, that under the hERG flips only the hybrid moved, is kept.

- Old: `Weighting keeps every molecule and every label. It helped on hERG, LogD and Buchwald--Hartwig (Table~\ref{tab:noise:res:weighting}). Under the hERG flips, weighting by the gap over neighbor spread helped the hybrid and barely moved the single representation. Weighting by the neighbor gap in the best of four distances helped the network on LogD with outliers, on both the single representation and the hybrid. Down-weighting near-identical pairs whose labels disagree helped LightGBM on the Buchwald--Hartwig hybrid under Gaussian noise.`
- New: `Weighting keeps every molecule and every label, and it helped on hERG, LogD and Buchwald--Hartwig (Table~\ref{tab:noise:res:weighting}). Under the hERG flips it helped the hybrid and barely moved the single representation.`

### Fix 10. Line 5977, "On small molecules, a user who builds a"

Results close: repeated sentence and a "therefore" that points two paragraphs back. Row: Link sentence that only repeats; connective repeated.

The second sentence repeats the last sentence of line 5956. "therefore" refers to line 5963, two paragraphs earlier. The paragraph sits under an orphan \label{sec:noise:discussion} with no heading; nothing refers to that label.

- Old: `On small molecules, a user who builds a hybrid therefore loses nothing by cleaning afterwards. The ranking of noise-handling methods changed between the hybrid and the single representations in almost every condition. Whether`
- New: `On small molecules, a user who builds a hybrid loses nothing by cleaning afterwards. Whether`

### Fix 11. Line 5302, "A Gaussian process's own predicted spread plus its"

The three spread-only detectors are explained a third time; "the files mark them" is code talk. Row: Method/background restated; lexical diversity.

Methods line 4657 and Results line 5008 already say these three never read the label.

- Old: `The three that predict a spread from the features alone never see a label, and the files mark them as undefined under every damage kind that can be graded.`
- New: `The other three are undefined under every damage kind that can be graded (Section~\ref{sec:noise:detection}).`

### Fix 12. Line 4584, "This chapter uses the datasets of Chapters~\ref{chap:method} and~\ref{chap:similarity}"

"QM9 is the clean-label control" appears three times in Introduction and Methods. Row: Lexical diversity (MTLD); method restated.

Lines 4578, 4584 and 4884 each say it. Line 4884 is the paragraph that explains it; line 4578 introduces it. The middle one goes.

- Old: `QM9 is the clean-label control (Section~\ref{sec:noise:tests}), and hERG IC$_{50}$ is the one dataset`
- New: `hERG IC$_{50}$ is the one dataset`

### Fix 13. Line 4587, "Each dataset has two downstream models. In each"

"So" opener. Row: Connectives per 100 words; distinct connectives.

Ten sentences in the chapter open with "So". Here the consequence joins its cause.

- Old: `on the damaged labels. So the hybrid's features move with the damage.`
- New: `on the damaged labels, so the hybrid's features move with the damage.`

### Fix 14. Line 4630, "Most experiments first run a screen of two"

"So" opener. Row: Connectives; distinct connectives.

Same pattern; "therefore" varies the connective.

- Old: `So the levels nest:`
- New: `The levels therefore nest:`

### Fix 15. Line 5041, "A second test of whether the gain comes"

"So" opener. Row: Connectives.

Joined to its cause.

- Old: `makes the single representation worse. So the gain depends`
- New: `makes the single representation worse, so the gain depends`

### Fix 16. Line 5046, "Weighting keeps every molecule and every label. It"

"So" opener after a 3-word sentence. Row: Connectives; fragment.

"LightGBM barely moved." is three words and is the reason for the next sentence.

- Old: `LightGBM barely moved. So a network gain`
- New: `LightGBM barely moved, so a network gain`

### Fix 17. Line 5116, "On the first outer fold, deleting never helped"

"So" opener. Row: Connectives.

The conditional already carries the logic; "So" is dropped.

- Old: `So when deleting helps on a damaged dataset and beats random deletion, as it does under the hERG flips, the gain is noise removed.`
- New: `When deleting helps on a damaged dataset and beats random deletion, as it does under the hERG flips, the gain is noise removed.`

### Fix 18. Line 5207, "The cliff-aware gap does better than the residual"

"So" opener followed by two short parallel sentences. Row: Connectives; fragmentation; sentence length.

Three sentences of 6, 16 and 10 words state one finding and its two halves.

- Old: `So no detector is best everywhere. A neighbor detector is better at a shift that moves a few labels a long way. A model residual is as good or better at outliers.`
- New: `No detector is best everywhere: a neighbor detector is better at a shift that moves a few labels a long way, and a model residual is as good or better at outliers.`

### Fix 19. Line 5356, "The break-even precision explains when acting on a"

"So" opener; closing sentence restates the opening one. Row: Connectives; link sentence.

The paragraph opens with "The break-even precision explains when acting on a detector can work"; the close now gives the reason instead of a second summary.

- Old: `So a detector that ranks the damaged labels above chance can still lead to a worse model. Acting on its flags helps only above the break-even, which depends on the damage, the action and the model.`
- New: `A detector that ranks the damaged labels above chance can still lead to a worse model: acting on its flags helps only above the break-even, and the break-even depends on the damage, the action and the model.`

### Fix 20. Line 5557, "The per-molecule lift tests whether the size of"

"So" opener. Row: Connectives.

Rephrased without a connective; same claim.

- Old: `So the gain comes from knowing which labels are clipped and raising them by about the right average amount, which is what the constant lift does.`
- New: `What matters is knowing which labels are clipped and raising them by about the right average amount, which is what the constant lift does.`

### Fix 21. Line 5827, "Retuning on the damaged labels beats reusing settings"

"So" sentence that repeats the paragraph's first sentence. Row: Link sentence that only repeats -> cut.

The first sentence is "Retuning on the damaged labels beats reusing settings tuned on clean labels." This sentence says the same.

- Old: `So the gain comes from tuning on the damaged labels. `
- New: (delete)

### Fix 22. Line 5956, "The method that does best on the hybrid"

"So" opener. Row: Connectives.

The finding follows from the three exceptions just listed; no connective needed.

- Old: `So a ranking of noise-handling methods made on one representation does not carry to another.`
- New: `A ranking of noise-handling methods made on one representation does not carry to another.`

### Fix 23. Line 4603, "We damage a regression dataset in six ways"

", so" is the chapter's one causal connective (24 in Results, 18 in Methods). Row: Distinct connectives / connectives used (Results 0.13, references 0.21-0.32).

Vary: "because".

- Old: `Every other label still gets the ordinary error, so the 5\% carry`
- New: `Because every other label still gets the ordinary error, the 5\% carry`

### Fix 24. Line 4884, "QM9 is the clean-label control. Its HOMO--LUMO gaps"

", so". Row: Distinct connectives.

Plain "and"; the causal link is obvious.

- Old: `\citep{Ramakrishnan2014}, so they carry no assay error.`
- New: `\citep{Ramakrishnan2014} and carry no assay error.`

### Fix 25. Line 4881, "Five controls ran, each described with its experiment."

", so"; also "No other method ... its gains" mixes singular and plural. Row: Distinct connectives.

- Old: `No other method has a control, so its gains are weaker evidence.`
- New: `The other methods have no control, which makes their gains weaker evidence.`

### Fix 26. Line 5203, "This section uses injected damage alone (Section~\ref{sec:noise:detection}). The"

", so". Row: Distinct connectives.

- Old: `for the clipped molecules, so their residuals are small.`
- New: `for the clipped molecules, which leaves their residuals small.`

### Fix 27. Line 5629, "On the labels as delivered the same methods"

", so" (two in this paragraph). Row: Connective repeated; distinct connectives.

- Old: `Nothing had shifted a family, so subtracting`
- New: `Because nothing had shifted a family, subtracting`

### Fix 28. Line 4713, "A noise mitigation method has two halves (Table~\ref{tab:noise_terms})."

", so". Row: Distinct connectives.

A colon states the consequence.

- Old: `describes each method this way, so the methods reduce`
- New: `describes each method this way: the methods reduce`

### Fix 29. Line 5164, "In a second experiment, label smoothing on hERG"

Two-sentence paragraph on the same result as the paragraph before it. Row: Fragment paragraph -> merge.

Both paragraphs report label smoothing on hERG $K_i$'s labels as delivered.

- Old (the paragraph): `In a second experiment, label smoothing on hERG $K_i$ improved the network's hybrid on each of folds 0, 1 and 2 (Table~\ref{tab:noise:res:hergki_smoothing_folds}). The second single representation got worse on fold 0 and better on the other two.`
- New: Delete this paragraph and add, in the line-5123 paragraph directly after "label smoothing also improves the hybrid by more than any of three shuffles.": In a second experiment it improved the network's hybrid on each of folds 0, 1 and 2, while the second single representation got worse on fold 0 and better on the other two (Table~\ref{tab:noise:res:hergki_smoothing_folds}). Move the % SOURCE [N7-10] and % TODO: NEW RESULTS [N7-10] lines with it.

### Fix 30. Line 5080, "Figure~\ref{fig:noise:did_cleaning_work_detector} shows, on the first outer fold only,"

"signed-off" is project jargon the reader has not seen. Row: Pointer sentences (results open with the figure); readability.

The reader does not know which methods were signed off or by whom.

- Old: `how often each signed-off method improved`
- New: `how often each method improved`


### Fix 31. Line 4652, "Three settings involve no injection: the labels as"

Row: Link sentence that only announces -> cut. Each of the three is described in its own subsection.

- Old: `simulates several measurements per molecule. Each is described where it appears.`
- New: `simulates several measurements per molecule.`

### Fix 32. Line 4657, "A detector gives every training molecule a suspicion"

Row: Fragment; sentence length. 5-word sentence joined to the one it contrasts with.

- Old: `Most detectors only rank molecules. A few of the third kind`
- New: `Most detectors only rank molecules, but a few of the third kind`

### Fix 33. Line 4685, "The gap over neighbor spread is the buddy"

Row: Flat run of two 10-word sentences.

- Old: `\citep{Kordos2012}. We found no earlier use of either`
- New: `\citep{Kordos2012}, but we found no earlier use of either`

### Fix 34. Line 4702, "One experiment compares several further detectors with the"

Row: Sentence order. The cleanlab caveat sat after the sentence about the other detectors.

- Old: `Another is the cleanlab regression score \citep{zhou2024detecting}. The rest divide the residual by a spread for that molecule, from NGBoost, a quantile forest or a conformal interval. cleanlab draws its internal folds at random rather than by scaffold family.`
- New: `Another is the cleanlab regression score \citep{zhou2024detecting}, which draws its internal folds at random rather than by scaffold family. The rest divide the residual by a spread for that molecule, from NGBoost, a quantile forest or a conformal interval.`

### Fix 35. Line 4708, "A detector can be graded only where the"

Row: ", so" repeated (twice in this paragraph).

- Old: `of the training molecules, so that share is the chance level.`
- New: `of the training molecules; that share is the chance level.`

### Fix 36. Line 4779, "The hybrid's features are chosen against the labels,"

Row: Fragment (6 words).

- Old: `One experiment compares the two orders. It runs nine cleanups with the out-of-fold forest residual as the detector,`
- New: `One experiment compares the two orders, with nine cleanups that use the out-of-fold forest residual as the detector,`

### Fix 37. Line 4799, "A scaffold family is also a known flag."

Row: Ordinal scaffolding "The first... The second... The third"; two short sentences.

- Old: `from every family. The second subtracts it only from families`
- New: `from every family, and the second only from families`

### Fix 38. Line 4802, "A source is a known flag when each"

Row: Fragment (5 words).

- Old: `One experiment simulates eight sources. Each molecule is measured`
- New: `One experiment simulates eight sources, and each molecule is measured`

### Fix 39. Line 4809, "hERG IC$_{50}$ carries its own known flag: how"

Row: Fragments (5 and 8 words); ordinal scaffolding.

- Old: `Three methods use the flag. The first weights each molecule by its count.`
- New: `Of the three methods that use the flag, the first weights each molecule by its count.`

### Fix 40. Line 4867, "On hERG IC$_{50}$, four methods change what the"

Row: Ordinal scaffolding; flat run of short sentences.

- Old: `as an extra feature. The third gives NGBoost the same extra feature.`
- New: `as an extra feature, and the third gives NGBoost the same feature.`

### Fix 41. Line 4884, "QM9 is the clean-label control. Its HOMO--LUMO gaps"

Row: Fragmentation; one claim in two sentences.

- Old: `cannot come from removing noise. It comes from a change`
- New: `cannot come from removing noise; it comes from a change`

### Fix 42. Line 5116, "On the first outer fold, deleting never helped"

Row: Fragmentation.

- Old: `from 0.355 to 0.393 RMSE. LightGBM's hybrid`
- New: `from 0.355 to 0.393 RMSE, and LightGBM's hybrid`

### Fix 43. Line 5188, "Deleting does not pass its control on the"

Row: Fragment (4 words).

- Old: `Buchwald--Hartwig is different. Deleting the 20\%`
- New: `Buchwald--Hartwig is different: deleting the 20\%`

### Fix 44. Line 5193, "Putting activity-cliff molecules back after a cleanup can"

Row: Flat run (15, 16, 15, 13, 17 words).

- Old: `count as cliff molecules. That leaves too few`
- New: `count as cliff molecules, which leaves too few`

### Fix 45. Line 5297, "Scaling each feature by its importance before the"

Row: Fragment (6 words).

- Old: `SMAC colloidal stability. Under outliers scaling barely changed detection.`
- New: `SMAC colloidal stability, while under outliers scaling barely changed detection.`

### Fix 46. Line 5394, "Figure~\ref{fig:noise:detection_vs_cleaning} applies this test to the two learners."

Row: Fragment (6 words); "There" points back across a sentence.

- Old: `LogD has no break-even yet. There, under outliers, deleting at 5\% with a precision of 0.36`
- New: `On LogD, which has no break-even yet, deleting at 5\% under outliers with a precision of 0.36`

### Fix 47. Line 5629, "On the labels as delivered the same methods"

Row: ", so" repeated (twice in this paragraph).

- Old: `per family, so it should be used only`
- New: `per family and should be used only`

### Fix 48. Line 5649, "Set side by side, the damage kinds sort"

Row: Fragment (5 words).

- Old: `The gains there were smaller. They depended on whether`
- New: `The gains there were smaller and depended on whether`

### Fix 49. Line 5658, "SMAC colloidal stability gives the strongest controlled result"

Row: Fragment (8 words).

- Old: `to 0.403 Spearman. The same deletion also helps`
- New: `to 0.403 Spearman, and the same deletion also helps`

### Fix 50. Line 5686, "hERG IC$_{50}$ is the one dataset whose noise"

Row: Fragment (7 words). Line 5686 has ten sentences of 7 to 27 words.

- Old: `on the first draw. It helped on all three draws`
- New: `on the first draw, and helped on all three draws`

### Fix 51. Line 5686, "hERG IC$_{50}$ is the one dataset whose noise"

Row: Two short sentences reading out two table rows.

- Old: `from the same extra feature. The interval model`
- New: `from the same extra feature, and the interval model`

### Fix 52. Line 5736, "The largest single gain is on SMAC colloidal"

Row: Pronoun and demonstrative ("The same treatment"). The next sentence is about deletion, not the shrink just described; the shared thing is cleaning before the search.

- Old: `The same treatment can also go the other way.`
- New: `Cleaning before the search can also go the other way.`

### Fix 53. Line 5902, "The other noise-robust training methods are in Table~\ref{tab:noise:res:robust_broad}."

Row: Fragment (8 words).

- Old: `percentage points of yield. It beat random deletion at all three seeds.`
- New: `percentage points of yield and beat random deletion at all three seeds.`

### Fix 54. Line 5913, "Table~\ref{tab:noise:res:sweep} gives the expanded sweep by domain. Small-loss"

Row: Limitation sentence that says nothing ("we read its result as one dataset's").

- Old: `more often on Buchwald--Hartwig and the antibodies. Buchwald--Hartwig is one dataset, so we read its result as one dataset's.`
- New: `more often on Buchwald--Hartwig and the antibodies, though Buchwald--Hartwig is a single dataset.`

### Fix 55. Line 5968, "The closest test of the question this thesis"

Row: Fragment (9 words).

- Old: `with neighbors in the hybrid. It gets worse`
- New: `with neighbors in the hybrid and gets worse`

### Fix 56. Line 5968, "The closest test of the question this thesis"

Row: Two limitation sentences in a row.

- Old: `not typical ones. No control has run`
- New: `not typical ones, and no control has run`

### Fix 57. Line 5595, "On QM9 we also clipped a quarter of"

Row: Decimals: 3 for scores. The only four-decimal scores in the AI prose of this chapter.

- Old: `0.5072 before treatment and 0.3980 after`
- New: `0.507 before treatment and 0.398 after`

### Fix 58. Line 5597, "\subsubsection{A shared offset per scaffold group}\label{sec:noise:res:offsets}"

Row: Section titles; lexical consistency. The text says "scaffold family" everywhere else in the chapter.

- Old: `\subsubsection{A shared offset per scaffold group}`
- New: `\subsubsection{A shared offset per scaffold family}`
