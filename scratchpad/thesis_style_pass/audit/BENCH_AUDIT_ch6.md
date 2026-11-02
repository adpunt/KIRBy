# Chapter 6 against the benchmark table (audit, 2026-10-10)

Read-only audit. Nothing in `thesis.tex` was changed. Source: `thesis.tex` as saved at 08:35 today, Chapter 6 at lines 3661–4565. Another workflow was editing Chapter 6 tables and numbers while this ran. Every "old" text below was checked at 08:40 to occur exactly once in the live file. If that workflow edits a paragraph again, re-check the quote before applying.

## Which text in Chapter 6 is AI-written

Almost all of it. Every prose paragraph in the chapter has a plain `% TODO: manual review` above it, except two lists:

- Line 3671–3672: the two research questions (Introduction, `enumerate`). Author's.
- Lines 3691–3695 and 3724–3725: the `itemize` entries under "Feature spaces compared" and under the undefined similarity search, and lines 3764–3765 (the two weight formulas). The segmenter counts these as the author's, because the marker sits above the paragraph that ends in a colon, not above the list. They read as part of the AI paragraph above them. I did not change them either way.

AI prose by section type: Introduction 233 words (13 sentences, 3 paragraphs), Methods 2,508 words (144 sentences, 22 paragraphs), Results 1,699 words (89 sentences, 24 paragraphs), Conclusions 329 words (16 sentences, 5 paragraphs). Every caption and every table in the chapter is AI-written too.

## Language measures

Computed with `compute_metrics.py` (`nlp`, `sent_metrics`, `normalise`) and the `extra_measures.py` functions, using the same per-chapter script as the Chapter 3 audit. I re-segmented the 08:35 `thesis.tex` with a copy of `segment_full_author.py` written to the session scratchpad (`ch6bench/`). Surprisal is GPT-2 small on every AI sentence.

"After fixes" is the same measure with fixes 1–16 below applied to a scratch copy. Move rows were coded by hand for the 26 sentences that had no code in `moves_author_current.csv` or `moves_after_new_sentences.csv`.

Reference range = lowest to highest of Buttenschoen, Dablander and Vost. "Yours" = the author's own text in that section type across the thesis, from the benchmark table.

### Introduction (AI: 13 sentences, 233 words)

Not a target for any row. Thirteen sentences give no stable mean, spread or share. Read instead. Two things came out of reading. The first is a six-word fragment ("MDR1--MDCK efflux was added late.") followed by a limitation. The second is a list of four missing tables, which belongs in Methods (fix 4). The opening links back to Chapters 5 and 4 (structure row, done).

For the record: words per sentence 17.9 (references 25.3–30.8, yours 20.8), sentence length SD 9.09 (8.89–15.0), mean change between neighbors 12.6 (10.2–14.8), GPT-2 surprisal 7.53 (6.48–6.87).

### Methods (AI: 144 sentences, 2,508 words, 22 paragraphs)

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 22.3–28.0 | 20.1 | 17.4 | 19.3 | Not a target. Short is her preference. After the fixes it is close to her 20.1. |
| % sentences over 30 words | 21.1–32.7 | 12.2 | 0.7 | 5.3 | Partly. One sentence over 30 words in 144 is the uniformity problem. The merges (fixes 5b, 14d) make a few longer sentences. They do not aim for the reference share. |
| Dependent clauses per sentence | 0.46–0.95 | 0.50 | 0.40 | 0.46 | Done after fixes (lower edge). |
| Words per paragraph | 76.3–142.4 | 81.9 | 114.0 | 115.7 | Done (inside). |
| Lexical density % | 58.8–59.7 | 56.6 | 53.1 | 53.1 | Not a target. Below the author's own value too. It comes from function-word-heavy definitions ("one minus the weighted share of..."). Changing it would mean nominal style, which she rejected. |
| Nominalizations per 100 words | 3.6–5.0 | 3.8 | 3.59 | 3.57 | Done (at the lower edge, same as hers). |
| % passive sentences | 40.8–45.6 | 28.2 | 21.5 | 22.7 | Not a target. The text uses "we" and names the search as the actor, which is accurate. |
| we/our/I per 100 words | 0.83–1.6 | 1.1 | 1.24 | 1.18 | Done (inside). |
| Hedges per 100 words | 0.36–0.85 | 0.39 | 0.16 | 0.16 | Not a target. Methods state what was done. |
| Boosters per 100 words | 0.14–0.26 | 0.10 | 0.12 | 0.12 | Done (same as hers). |
| AI-guide phrases per 100 words | 0.0–0.10 | 0.01 | 0.0 | 0.0 | Done. |
| Numbers per 100 words | 1.8–2.1 | 2.5 | 2.15 | 2.12 | Done (just above the range, below hers). |
| % sentences with 3+ numbers | 3.7–4.7 | 6.0 | 4.17 | 4.55 | Done (inside). |
| MTLD | 56.9–81.9 | 69.6 | 65.0 | 69.0 | Done (inside). |
| MATTR-50 % | 75.2–79.8 | 78.5 | 76.8 | 76.9 | Done (inside). |
| Sentence length SD (words) | 12.1–18.1 | 9.19 | 6.48 | 6.56 | Not done. Most Methods paragraphs run at 12–27 words per sentence with no long sentence. The merges remove the shortest sentences (5–8 words), so the SD barely moves. A real change would need longer sentences that join definitions. I did not write those, because her rule is one fact per sentence. |
| Sentence length coefficient of variation | 0.56–0.67 | 0.47 | 0.39 | 0.36 | Not done, same reason. |
| Mean change in length between consecutive sentences | 12.5–17.1 | 9.13 | 8.38 | 8.49 | Not done, same reason. It is close to her 9.13. |
| GPT-2 surprisal, bits per token | 7.04–7.42 | 7.67 | 8.01 | 7.92 | Not a target. Higher means less predictable, and her own text is above the range as well. Dataset names and symbols raise it. |
| Surprisal SD across sentences | 1.11–1.37 | 1.22 | 1.15 | 1.04 | Done before fixes. After the fixes it is 1.04, just below the range. |
| Connectives per 100 words | 1.16–1.76 | 1.80 | 1.88 | 1.89 | Done (near hers). The count hides one repeat: "so" appears 13 times, 4 of them opening a sentence ("So the spread", "So a representation's weight", "So the weighted distances", "So the two sides"). Fixes 5a–5d bring it to 9 and keep one sentence-initial "So". |
| Distinct connectives / connectives used | 0.15–0.28 | 0.17 | 0.27 | 0.30 | Done. |
| % sentences opening with a formulaic connector | 1.0–3.5 | 0.85 | 0.0 | 0.0 | Not a target. She avoids them too. |
| Three-item lists per 100 sentences | 7.0–8.4 | 12.8 | 9.0 | 9.9 | Done (below hers). All 13 hits are content lists: settings, numbers of clusters, the six hybrids, the fields of the q-RASAR features. None is a rhythmic triplet. |
| Pronouns and demonstratives per 100 words | 3.6–5.5 | 5.04 | 5.85 | 5.63 | Done (near hers). |

### Results (AI: 89 sentences, 1,699 words, 24 paragraphs)

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 24.7–31.1 | 21.4 | 19.1 | 19.9 | Not a target (short is her preference). |
| % sentences over 30 words | 24.4–47.5 | 13.1 | 2.2 | 2.4 | Not done. See the SD row. |
| Dependent clauses per sentence | 0.58–1.0 | 0.60 | 0.25 | 0.26 | Not a target. Results sentences are "On X, the hybrid ...": a place, then a comparison. Adding clauses would add interpretation. |
| Words per paragraph | 99.2–177.7 | 107.5 | 70.8 | 73.4 | Partly. Fix 2 removes the 12-word one-sentence paragraph. The other short paragraphs each hold one comparison and one table pointer, which matches her rule. |
| Lexical density % | 59.3–62.9 | 54.8 | 57.0 | 57.0 | Not a target (above hers). |
| Nominalizations per 100 words | 3.9–4.4 | 3.8 | 3.53 | 3.55 | Not a target (same as hers). |
| % passive sentences | 26.2–30.6 | 14.9 | 1.1 | 1.2 | Not a target. Active comparisons are what she asked for. |
| we/our/I per 100 words | 0.26–1.6 | 0.67 | 0.35 | 0.36 | Done (inside). |
| Hedges per 100 words | 0.87–2.3 | 0.81 | 0.12 | 0.12 | Not a target. The results state what the tables show. Where evidence is weak, the text says so plainly ("we draw no conclusion from QM9", "we cannot separate size from domain"). Plain statements like these do not count as hedges. |
| Boosters per 100 words | 0.24–0.48 | 0.23 | 0.06 | 0.06 | Not a target. |
| AI-guide phrases per 100 words | 0.09–0.58 | 0.05 | 0.0 | 0.0 | Done. |
| Numbers per 100 words | 1.2–5.9 | 2.9 | 2.88 | 2.90 | Done. Every paragraph has at most one score pair, and each paragraph points to its table (her 10 Oct rule). |
| % sentences with 3+ numbers | 3.1–20.3 | 9.5 | 5.6 | 5.9 | Done (inside). |
| Move: method/background restated | 8.6–33.7 | 18.7 | 5.6 | 3.5 | Not a target. Fixes 7 and 16 cut two restatements that only repeated Methods. |
| Move: pointer to figure/table | 3.1–18.2 | 1.0 | 0.0 as a move; 33% of sentences carry a "(Table~...)" | same | Done in substance. No sentence is only a pointer, but 29 of 89 sentences name their table. That is above Buttenschoen's 18%. |
| Move: result as numbers | 7.9–24.8 | 16.9 | 14.6 | about the same | Done (inside). |
| Move: result in words | 8.3–12.4 | 19.9 | 31.5 | about the same | Not a target. Above the references by design: her rule is "pattern in words, point to the table". |
| Move: comparison of results | 18.1–23.6 | 15.2 | 38.2 | about the same | Not a target. Every result in this chapter is the hybrid against the single representation. |
| Move: interpretation | 7.4–27.1 | 16.2 | 1.1 | 1.2 | Not a target for this pass. Adding interpretation would mean writing new AI claims into her chapter. The one there ("Part of what the hybrid does for a distance is therefore what weighting does") is backed by the table. |
| Move: comparison with literature | 0–0.83 | 2.5 | 0.0 | 0.0 | Done (inside). |
| Move: claim of strength | 0.83–3.6 | 1.7 | 0.0 | 0.0 | Not a target. Results should not claim strength. |
| Move: limitation | 0.83–5.0 | 3.7 | 7.9 | 8.1 | Not a target. Read all 7. Each says which result is missing or too small to read: weighted cells missing on five datasets, the picked-again comparison favours the hybrid, no shuffled-weight control, size against domain, too few cliff antibodies, two QM9 folds, 48 cliff molecules. Each sits next to the number it qualifies, so I kept them all. The repeat in the Conclusions is cut instead (fix 3). |
| Move: link/transition | 1.7–6.4 | 4.2 | 0.0 | 1.2 | Done after fix 1 (the Results lead-in). |
| Move: recommendation/implication | 0–4.3 | 0.0 | 1.1 | 1.2 | Done (inside). |
| MTLD | 55.9–83.8 | 69.9 | 63.8 | 64.2 | Done (inside). |
| MATTR-50 % | 76.0–80.4 | 78.1 | 75.9 | 75.9 | Partly: just below the range. Dataset names repeat because results are given per dataset. Not worth changing. |
| Sentence length SD (words) | 9.52–12.8 | 8.46 | 6.36 | 6.18 | Not done. This is the main language problem in Results. Paragraph after paragraph runs 13–28 words per sentence, with almost nothing under 10 or over 30. The fixes merge 4 short sentences and make one 30-word sentence, which does not raise the SD. Fix 10 targets a related pattern instead: the sentence openings. |
| Sentence length coefficient of variation | 0.42–0.44 | 0.41 | 0.35 | 0.33 | Not done, same reason. |
| Mean change in length between consecutive sentences | 9.96–14.2 | 9.72 | 7.17 | 6.84 | Not done, same reason. |
| GPT-2 surprisal, bits per token | 6.80–7.14 | 7.77 | 8.75 | 8.72 | Not a target (names and symbols; hers is above the range too). |
| Surprisal SD across sentences | 1.01–1.04 | 1.22 | 1.02 | 1.04 | Done (inside). |
| Connectives per 100 words | 1.59–3.12 | 2.32 | 1.67 | 1.68 | Done (inside). |
| Distinct connectives / connectives used | 0.21–0.32 | 0.14 | 0.44 | 0.44 | Done. |
| % sentences opening with a formulaic connector | 5.2–8.8 | 1.81 | 0.0 | 0.0 | Not a target. A different opener repeats instead: 30 of 89 sentences (34%) open with "On". Fix 10 changes three of them. |
| Three-item lists per 100 sentences | 4.65–10.0 | 13.3 | 20.2 | 22.4 | Not a target. All 18 hits are lists of dataset names ("Caco-2, hERG $K_i$ and hERG blocking") or cluster counts ("3, 5 and 10"). They are content, not rhythm. Fix 2 adds one ("all six small-molecule datasets, all seven antibody datasets and Buchwald--Hartwig") to remove a one-sentence paragraph. |
| Pronouns and demonstratives per 100 words | 2.72–5.78 | 5.10 | 3.21 | 3.17 | Done (inside). |

### Conclusion (AI: 16 sentences, 329 words, 5 paragraphs)

Sixteen sentences is small. Treat the language rows as a reason to read, not as a measurement. The move rows are counts of 16 sentences.

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 25.0–32.5 | 21.0 | 20.6 | 21.6 | Not a target (same as hers). |
| % sentences over 30 words | 25.8–42.9 | 12.8 | 6.2 | 14.3 | Done after fixes (near hers). |
| Dependent clauses per sentence | 0.71–0.96 | 0.76 | 0.38 | 0.36 | Not a target at 16 sentences. |
| Words per paragraph | 95.3–138.9 | 123.1 | 65.8 | 60.6 | Not a target. Five short paragraphs, each with one job: small molecules, weighting, antibodies, missing results, next chapter. |
| Lexical density % | 61.7–64.3 | 55.7 | 61.1 | 61.4 | Done (near the lower edge). |
| Nominalizations per 100 words | 4.2–6.9 | 4.4 | 3.95 | 3.96 | Not a target. |
| % passive sentences | 14.6–30.1 | 25.5 | 0.0 | 0.0 | Not a target. |
| we/our/I per 100 words | 1.3–1.9 | 0.41 | 0.61 | 0.66 | Not a target. Above hers. "We found" opens the main finding. |
| Hedges per 100 words | 0.95–2.2 | 1.4 | 0.30 | 0.33 | Not a target. |
| Boosters per 100 words | 0.24–0.90 | 0.20 | 1.22 | 1.32 | Not a target. These are 4 hits in 329 words: "found" (twice), "must" and "central claim". None is an inflated claim. |
| Numbers per 100 words | 0.04–0.43 | 0.91 | 0.0 | 0.0 | Done. |
| Move: method/background restated | 10.4–15.9 | 14.1 | 6.2 | 7.1 | Done. The one restatement is the opening link back to Chapter 4. |
| Move: result in words | 6.8–14.3 | 24.7 | 18.8 | about 14 | Done. |
| Move: comparison of results | 3.0–17.8 | 7.1 | 37.5 | 35.7 | Not a target. The findings are comparisons. |
| Move: claim of strength | 15.1–29.9 | 0.0 | 0.0 | 0.0 | Not a target. Her own Conclusions have none, and she rejects inflation. The main finding is already stated as a finding with "we" ("we found that distance in a hybrid built for distance tracks label differences more closely"). |
| Move: limitation | 0–5.5 | 15.3 | 18.8 (3 of 16) | 14.3 (2 of 14) | Partly. Fix 3 cuts the sentence that repeats the Results caveat about the picked-again comparison. Two remain. The first is the list of missing QM9 results; it carries a `TODO(missing result)` and goes when the runs land. The second is the hERG-blocking and label-information sentence, which is a negative result, not a caveat. |
| Move: link/transition | 1.4–4.8 | 1.2 | 12.5 (2 of 16) | 14.3 | Not a target. Both sentences are the forward pointer to Chapter 7 (structure row). |
| Move: recommendation/implication | 16.4–17.9 | 18.8 | 6.2 | 7.1 | Not a target. The one recommendation ("we must measure a hybrid's benefit on each dataset") is the thesis's claim. Adding more would be an AI verdict. |
| MTLD | 104.2–167.3 | 80.8 | 77.5 | 93.7 | Partly (above hers, below the references). Fix 11 removes the third "on all six". |
| Sentence length SD (words) | 8.01–12.6 | 9.24 | 6.74 | 7.17 | Partly. |
| Mean change in length between consecutive sentences | 8.53–12.3 | 9.40 | 9.18 | 9.67 | Done (inside). |
| GPT-2 surprisal | 6.57–6.84 | 7.57 | 8.98 | 8.89 | Not a target. |
| Connectives per 100 words | 2.12–2.72 | 2.23 | 1.86 | 1.68 | Not a target at 16 sentences. |
| Three-item lists per 100 sentences | 9.4–14.3 | 12.8 | 18.8 (3 hits) | 21.4 (3 hits) | Not a target. All three are lists of datasets. |
| Pronouns and demonstratives per 100 words | 5.7–7.1 | 4.98 | 4.64 | 4.71 | Not a target (near hers). |

## Structure rows that apply to Chapter 6

| Benchmark row | Status | Fix |
|---|---|---|
| Chapter opening links back | Done. Line 3666 opens with Chapter 5 (`chap:method`), then Chapter 4 (`chap:flaws`). | None. |
| Chapter ends point forward | Done. Line 4564 names Chapter 7 and its question. | None. |
| Text between section title and first subsection | Not done for Results. `\section{Results}` (line 3830) runs straight into `\subsection{Distance against label difference}`, and its `% TODO: manual review` marker sits above an empty section. Methods has a lead-in (line 3681). The Experiments subsection has a lead-in (line 3773). Introduction and Conclusions have no subsections. | Fix 1. |
| Title case | Done. Every heading is sentence case. | None. |
| Section titles | Done. Topic noun phrases. "Three smaller checks" is a count, not a topic. It is acceptable, but "Feature count, choosing molecules to measure, label information" would name the topics. Optional, not listed as a fix. | None. |
| Numbers in results; results words per float; results pointers | Not a target now. Tables are being built by the other workflow. As of 08:35 the chapter has 17 floats against 1,699 words of Results prose, about 100 words per float. That is denser than any reference. Every Results paragraph points to a table. | None. |
| Limitations | Partly. See the Conclusion move row. | Fix 3. |
| Research questions | Done. Line 3669 states two, and the Conclusions answer the first in paragraph 1 and the second in paragraphs 1 and 3. | None. |
| Hedges | Not a target (see tables). | None. |
| Statistics | Done. Line 3684: means over five folds, per-fold values where the spread matters, no significance tests. Results give per-fold counts and spread ("higher in every fold on LogD", "worse in one of five folds"). | None. |
| Negative/positive results | Done. Losses are stated plainly (line 4557: "prediction from nearest neighbors is worse in the hybrid on..."; line 4554). | None. |
| Caption length | Not a target now (captions in flux). As of 08:35 the 17 captions average 63.5 words, inside the references (49.8–73.5). The two longest are `tab:similarity:retrieval_by_setting` (91 words) and `tab:similarity:retrieval` (87). Both repeat the classification definition of the label gap from line 3822. When the tables settle, that clause could point to Section `sec:similarity:experiments` instead. | None now. |
| Figure/table references | Done. "Table~\ref", "Section~\ref", "Equation~\ref" throughout. | None. |
| Decimals | Done in AI prose as of 08:35. Scores have three decimals. "0.0015 RMSE" (line 3759) is a difference under 0.010, so it keeps four (rule 4). The other workflow has just rounded 0.0155, 0.2315, 0.1535, 0.7115 and 0.0295 to three. | None. |
| Chemistry wording | Checked by reading line 3814. It keeps "activity cliff" for non-activity labels and says so. The ECFP4 Tanimoto threshold is 0.55, and the activity gap is set to flag 15% of training molecules. Both are stated as modifications of Hu 2015. | None. |
| "you" in body | Not a target. No "you". | None. |

## Fixes

There are 16 numbered fixes; 4, 5, 10, 12, 13 and 14 have parts (36 edits in all). They are ordered by importance. Each one gives the tex line, the paragraph's first eight words, the exact old text and the new text. No meaning or number changes.

### Fix 1. Line 3830, `\section{Results}` (row: text between section title and first subsection; link/transition)

Add a one-sentence lead-in. The existing `% TODO: manual review` at line 3829 covers it. It restates the two research questions in order and gives no result.

- Old: `\section{Results}` + blank line + `\subsection{Distance against label difference}`
- New: `\section{Results}` + blank line + `We first ask whether distance in a hybrid tracks label differences more closely than in a single representation, and then whether the methods built on that distance gain from it.` + blank line + `\subsection{Distance against label difference}`

### Fix 2. Lines 3882 and 3889, "In the distance hybrid, distance tracks label differences" (row: fragment paragraph; words per paragraph)

Line 3889 is a 12-word paragraph that only says "the same holds" for the other domains. It is cut off from its claim by the table `\input`. Fold it into line 3882, and delete line 3889 with its marker (line 3888).

- Old (3882): `more closely on all six small-molecule datasets (Table~\ref{tab:distance_tracks_activity}). It is higher in every fold on LogD`
- New: `more closely on all six small-molecule datasets, all seven antibody datasets and Buchwald--Hartwig (Table~\ref{tab:distance_tracks_activity}). It is higher in every fold on LogD`
- Old (3888–3889, delete with the blank line after): `% TODO: manual review` / `The same holds on all seven antibody datasets and Buchwald--Hartwig (Table~\ref{tab:distance_tracks_activity}).`

### Fix 3. Line 4561, "QM9 has no result for the best number" (row: limitation sentence in conclusions -> cut)

The second sentence repeats a caveat already given in Methods (line 3806, "So the two sides differ in two ways at once") and in Results (line 4031, "This comparison favors the hybrid"). Cut it from the Conclusions.

- Old: `...or choosing which molecules to measure next. The comparison with the setting picked again favors the hybrid, because only the hybrid side can use importance weighting and it chooses among more combinations.`
- New: `...or choosing which molecules to measure next.`

### Fix 4. Lines 3676 and 3681, "This chapter uses 14 datasets: six small-molecule, seven" (rows: fragment sentence; limitation in the Introduction)

The Introduction has a 6-word fragment and then a 30-word list of tables that lack one dataset. That is a Methods caveat. Line 3681 already holds the caveat on missing results, so move the two sentences there as one.

- 4a. Old (3676): `This chapter uses 14 datasets: six small-molecule, seven antibody and one reaction dataset. MDR1--MDCK efflux was added late. It has no row in the tables of distance in the predictive hybrids and in each feature space, the best number of features, importance weighting in the four models, and activity cliffs. Antibody datasets outnumber`
  New: `This chapter uses 14 datasets: six small-molecule, seven antibody and one reaction dataset. Antibody datasets outnumber`
- 4b. Old (3681): `did not finish in time (Section~\ref{sec:method:evaluation}), and Section~\ref{sec:similarity:conclusions} lists them.`
  New: `did not finish in time (Section~\ref{sec:method:evaluation}), and Section~\ref{sec:similarity:conclusions} lists them. MDR1--MDCK efflux was added late and has no row in the tables of distance in the predictive hybrids and in each feature space, the best number of features, importance weighting in the four models, and activity cliffs.`

Alternative if she wants the Introduction unchanged: only merge the fragment, `MDR1--MDCK efflux was added late and has no row in the tables of...`.

### Fix 5. Methods, sentence-initial "So" (row: connective repeated, "so" x13 in Methods, 4 opening a sentence)

Keep "So the two sides differ in two ways at once" (line 3806). That is the defect statement in the form she accepted. Change the other three, and one more "so" in a paragraph that has two.

- 5a. Line 3684, "The fold split, the sample of molecule pairs,". Old: `use fixed seeds. So the spread we report is the spread across` New: `use fixed seeds. The spread we report is therefore the spread across`
- 5b. Line 3710, "The score uses the Manhattan distance of Chapter~\ref{chap:flaws},". This also joins a 13-word and a 21-word sentence, and reverses the next sentence so that "so" does not follow "so".
  Old: `Each standardized feature adds a term of similar size to this distance. So a representation's weight in the distance, which the allocation method sets, is roughly its number of features divided by the budget. A feature that only takes the values 0 and 1 adds less, so a fingerprint weighs less than this per feature.`
  New: `Each standardized feature adds a term of similar size to this distance, so a representation's weight in the distance, which the allocation method sets, is roughly its number of features divided by the budget. A fingerprint weighs less than this per feature, because a feature that only takes the values 0 and 1 adds less.`
- 5c. Line 3768, "The exponent $a$ of the first formula is". Old: `So the weighted distances of Sections~\ref{sec:similarity:results_similarity} and~\ref{sec:similarity:results_knn} are the same distance only under cosine.` New: `The weighted distances of Sections~\ref{sec:similarity:results_similarity} and~\ref{sec:similarity:results_knn} are therefore the same distance only under cosine.`
- 5d. Line 3759, "We found no published source for the rank". Two "so" in four sentences. Old: `We found no published source for the rank scheme, so we count it as new.` New: `We found no published source for the rank scheme and count it as new.`

### Fix 6. Line 3942, "Importance weighting alone closes much of the gap" (row: sentence length variation; fragmentation)

The paragraph runs 9, 13, 8, 10 and 17 words. Join the two sentences that compare the same three values.

- Old: `On QM9 it stays below the unweighted hybrid. On both, the weighted hybrid is highest of the three.`
- New: `On QM9 it stays below the unweighted hybrid, and on both datasets the weighted hybrid is highest of the three.`

### Fix 7. Line 3981, "The single representation here is the one the" (row: method/background restated)

The first sentence repeats the definition from line 3699. Fold it into the contrast it sets up.

- Old: `The single representation here is the one the nearest-neighbor model ranked first for prediction. We can instead pick, with hindsight`
- New: `Instead of the representation the nearest-neighbor model ranked first for prediction, we can pick, with hindsight`

### Fix 8. Line 4255, "Retrieval (Section~\ref{sec:similarity:results_other}) shows the same pattern" (row: fragmentation; repeated "On" openers)

An 8-word sentence ("On the antibody datasets the distance does matter.") is followed by a 40-word sentence that holds two findings. Swap the weight: put the claim and its scope together, and give SMAC its own sentence.

- Old: `On the antibody datasets the distance does matter. On those other than EGFR binding and $\Delta\Delta G$, cosine distance gives the hybrid its smallest lead, and on SMAC colloidal stability it reverses the result:`
- New: `The distance does matter on the antibody datasets: on all of them but EGFR binding and $\Delta\Delta G$, cosine distance gives the hybrid its smallest lead. On SMAC colloidal stability, cosine reverses the result:`

### Fix 9. Line 4071, "Multiplying features by their importance mostly helps prediction" (row: sentence length variation)

The 9-word second sentence is the other half of the first sentence's contrast.

- Old: `(Table~\ref{tab:importance_weighting}, exceptions in bold). In the other models, weighting the single representation gains little or loses.`
- New: `(Table~\ref{tab:importance_weighting}, exceptions in bold), but in the other models, weighting the single representation gains little or loses.`

### Fix 10. Results, repeated opener "On" (30 of 89 sentences) (row: % sentences opening with a formulaic connector; lexical diversity)

Move the dataset to the end in three sentences where the claim matters more than the place.

- 10a. Line 4348, "On LogD, Caco-2, hERG $K_i$ and hERG blocking,". Old: `On LogD the gain on cliff molecules is slightly smaller, but` New: `The gain on cliff molecules is slightly smaller on LogD, but`
- 10b. Line 4387, "A molecule's distance to its nearest training molecules". Old: `On hERG blocking the hybrid tracks error worse for both models.` New: `The hybrid tracks error worse for both models on hERG blocking.`
- 10c. Line 4390, "On nanobody thermostability, $\Delta\Delta G$, expression titer, CHO". Old: `On Buchwald--Hartwig, $\Delta\Delta G$ and expression titer, distance in the single representation does not rise with error for either model.` New: `Distance in the single representation does not rise with error for either model on Buchwald--Hartwig, $\Delta\Delta G$ and expression titer.`

### Fix 11. Line 4551, "Chapter~\ref{chap:flaws} found that distance in standard representations barely" (row: lexical diversity; repetition)

"On all six" appears three times in four sentences. Join the last two of them.

- Old: `with the setting picked again. Neighbors retrieved in the hybrid have closer labels on all six.`
- New: `with the setting picked again, and the neighbors retrieved in the hybrid have closer labels.`

### Fix 12. Line 3822, "Retrieval measures how close a molecule's neighbors are" (row: fragment sentences, two 3-word sentences in parallel)

"Lower is better." and "Higher is better." are 3-word sentences that repeat the same pattern. Put each direction inside its definition.

- 12a. Old: `For each held-out molecule, the label gap is the weighted mean` New: `For each held-out molecule, the label gap (lower is better) is the weighted mean`
- 12b. Old: ` Lower is better. For clustering,` New: ` For clustering,`
- 12c. Old: `Label coherence is one minus` New: `Label coherence (higher is better) is one minus`
- 12d. Old: ` Higher is better. Clusters with` New: ` Clusters with`

### Fix 13. Line 3814, "An activity cliff is a pair of structurally" (row: fragmentation; 14 sentences, two of 5–6 words)

- 13a. The 5-word sentence states a change whose reason is the next sentence. Old: `We modify its activity side. A gap of two log units` New: `We modify its activity side, because a gap of two log units`
- 13b. Old: `whose labels span more than that gap. Its own label is never read.` New: `whose labels span more than that gap; its own label is never read.`

### Fix 14. Methods, short sentences that finish the previous one (row: sentence length variation; mean change between consecutive sentences)

Each of these short sentences completes the one before it. These merges are low priority, and any one can be skipped.

- 14a. Line 3689, "Each outer fold runs three searches on its". Old: `with Euclidean distance and five equally weighted neighbors. It uses fewer neighbors when a fold has under 25 training molecules.` New: `with Euclidean distance and five equally weighted neighbors, or fewer when a fold has under 25 training molecules.`
- 14b. Line 3704, "The search of Chapter~\ref{chap:method} can be written as". Old: `(Equation~\ref{eq:similarity_objective}). No downstream model is fitted.` New: `(Equation~\ref{eq:similarity_objective}), and no downstream model is fitted.`
- 14c. Line 3707, "\citet{Zheng2000} and \citet{Itskowitz2005} chose descriptors to maximize nearest-neighbor". Old: `within one descriptor set. We modify this by choosing` New: `within one descriptor set, and we modify this by choosing`
- 14d. Line 3710, "The score uses the Manhattan distance of Chapter~\ref{chap:flaws},". Old: `agreeing with shared structure. Cosine was better at predicting from neighbors.` New: `agreeing with shared structure, although cosine was better at predicting from neighbors.`
- 14e. Line 3719, "A neighbor at distance 0 receives all of". Old: `Both scores put chance at 0. The budget split needs this, because` New: `Both scores put chance at 0, which the budget split needs, because`
- 14f. Line 3762, "Importance weighting is feature-weighted nearest neighbors \citep{Wettschereck1997}, modified". Old: `$\mathbf{x}' = \mathbf{x} \odot \mathbf{w}$. Negative importances are set to 0.` New: `$\mathbf{x}' = \mathbf{x} \odot \mathbf{w}$, with negative importances set to 0.`
- 14g. Line 3806, "We predict each held-out molecule's label from its". Old: `at every setting of Table~\ref{tab:knn_settings}. That gives up to 216 combinations.` New: `at every setting of Table~\ref{tab:knn_settings}, up to 216 combinations.`

### Fix 15. Line 3773, "Table~\ref{tab:similarity_experiments} lists every similarity experiment. The paragraphs below" (row: link sentence that only announces)

Keep the lead-in, because it is the subsection's only text before the table. Make the two short sentences, 5 and 8 words, into one.

- Old: `lists every similarity experiment. The paragraphs below define what each one estimates.`
- New: `lists every similarity experiment, and the paragraphs below define what each one estimates.`

### Fix 16. Line 4334, "On hERG $K_i$ and Buchwald--Hartwig, we also simulated" (row: method/background restated)

"From three starting sets" repeats line 3827 and the caption of `tab:similarity_experiments`.

- Old: `we also simulated choosing which molecules to measure next, from three starting sets.`
- New: `we also simulated choosing which molecules to measure next.`

## Found while reading, not language: needs the author or the tables workflow

Each of these is unverified against the result files. I did not change any number.

- The dataset count disagrees within the chapter. Line 3676 says "14 datasets". Line 3681 says "Where a statement covers all 13 datasets". The `tab:similarity_experiments` rows and caption say "All 13" and "counted from the 13 of this chapter".
- The small-molecule count was moving during the audit. At 08:33 lines 4031 and 4551 said "five" small-molecule datasets. At 08:35 they say "six". Line 4339 still says "We report activity cliffs on the five small-molecule datasets only". That is correct, because MDR1 is left out of the cliff table, but the reason is not given there.
- Line 4071 says "support vector regressor" and then "support vector machine" in the same paragraph. Line 4387 also says "support vector machine's errors". If both mean the SVR downstream model, use one name. If the second is a classifier, say which dataset it runs on.
- Line 4334 (choosing molecules to measure, a form of Bayesian optimization) sits under the subsection "Prediction from nearest neighbors". It has no table and no subsection of its own.
- Line 4512 now says the median is "summed over components". The table caption says "summed over at most 50 components". These agree. The 08:33 text said "summed over single features", which did not.

## Not changed, on purpose

- "So the two sides differ in two ways at once" (line 3806) stays. It is the defect-statement form she accepted.
- The lists of dataset names in Results and Conclusions stay. They are what the three-item-list counter picks up, but they are content.
- The seven limitation sentences in Results stay. Each sits next to the number it qualifies.
- Sentence length uniformity in Results (SD 6.4 against her 8.5) is not fixed by this list. Raising it means either longer sentences that join two comparisons, which breaks her one-fact-per-sentence rule, or more very short ones. I left it for her to judge by reading. The paragraphs at lines 4066, 4348 and 4390 show the pattern most: four or five sentences of 19–27 words each.
