# Chapter 8 against the benchmark table (audit, 2026-10-10)

Read-only audit. Nothing in `thesis.tex` was changed. Source: `thesis.tex` as read at the time of this audit, lines 5995–6081 (`\chapter{Conclusions and Further Thoughts}` to the last SOURCE comment). The back matter after it (`Summary of Research Contributions`) is outside the benchmark's section types.

## Which text in Chapter 8 is AI-written

Every prose paragraph in Chapter 8 has a plain `% TODO: manual review` directly above it. None is marked `-- fixed:` or `-- moved:`. So the whole chapter is AI-written: 17 paragraphs, 83 sentences, 1,375 words.

| tex line | Section | First words | Section type in the benchmark |
|---|---|---|---|
| 5998 | (chapter opening) | "This thesis set out to handle label noise" | discussion |
| 6003 | Summary of findings | "The first half of the thesis asked four" | discussion |
| 6006 | Summary of findings | "Chapter 4 then asked why published methods for" | discussion |
| 6010 | Summary of findings | "The second half asked whether the representation can" | discussion |
| 6013 | Summary of findings | "Second, distance in a hybrid tracked the label" | discussion |
| 6016 | Summary of findings | "Third, noisy labels could be found and corrected" | discussion |
| 6019 | Summary of findings | "Fourth, importance helped in building the hybrid but" | discussion |
| 6024 | Choosing a noise-handling method | "No single method handled label noise on every" | discussion |
| 6027 | Choosing a noise-handling method | "We therefore recommend modeling the noise in a" | discussion |
| 6032–6040 | Contributions | "The contributions on label noise are the following." + items 1–6 | discussion |
| 6043–6048 | Contributions | "The contributions on the representation are the following." + items 7–9 | discussion |
| 6053 + 6057 | Limitations | "Most noise results rest on the first outer" | discussion (one LaTeX paragraph: only comment lines between them) |
| 6062 | Future work | "The hybrid search needs only several feature matrices" | discussion |
| 6065 | Future work | "The noise methods need more clean controls. QM9" | discussion |
| 6068 | Future work | "A hybrid can also be the input to" | discussion |
| 6073 | Concluding remarks | "Label noise in molecular data can be handled." | conclusion |
| 6076 | Concluding remarks | "The features also matter. When the noise methods" | conclusion |
| 6079 | Concluding remarks | "A new foundation model appears every few months" | conclusion |

The nine `\item`s of the two contribution lists have no marker of their own. The segmenter (`segment_full_author.py`) therefore tags them "hers". They sit under the AI-marked lead-in sentence and are part of the same block, so I counted them as AI-written. Without them, the discussion figures change little; the one that moves is words per paragraph (63.1 instead of 49.4), because each item counts as a paragraph.

The benchmark's own section-type rule puts every Chapter 8 section except "Concluding remarks" under **discussion**, and "Concluding remarks" under **conclusion**. I kept that rule.

## Language measures

Computed with `compute_metrics.py` (`sent_metrics`, `normalise`) and the `extra_measures.py` functions (MTLD, MATTR, sentence-length spread, GPT-2 surprisal on every sentence, connectives, three-item lists, pronouns). I re-segmented the current `thesis.tex` with `segment_full_author.py` first, into the session scratchpad (`ch8bench/`). Moves come from `moves_author_current.csv` and `moves_after_new_sentences.csv` by sentence text; 80 of 83 sentences matched. I coded the other three by reading ("This thesis set out..." link; "Second, distance in a hybrid..." comparison; "For finding noisy labels..." result in words).

"After fixes" is the same measure on a scratch copy with fixes 1 to 18 below applied. Fix 19 is optional and not included.

Reference range = lowest to highest of the reference theses that have the section (Discussion: Buttenschoen and Dablander only). "Yours" = the author's own text across the thesis, from the benchmark table.

### Discussion (AI: 70 sentences, 1,186 words, 24 paragraphs including 9 list items)

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 24.8–34.2 | 19.9 | 16.9 | 19.6 | Done after fixes, at the author's own mean. Short sentences are her preference; the merges remove fragments, not shortness. |
| % sentences over 30 words | 25.7–58.3 | 8.3 | 1.4 | 8.3 | Done after fixes, at the author's own share. |
| Dependent clauses per sentence | 0.75–0.84 | 0.25 | 0.49 | 0.60 | Not a target. Above the author's own 0.25. |
| Words per paragraph | 92.8–126.2 | 26.6 | 49.4 (63.1 without list items) | 56.1 | Partly. Two fragment paragraphs merged (fixes 2, 3). The nine list items keep the mean down; they are a list, not fragments. |
| Lexical density % | 60.4–61.1 | 60.9 | 60.4 | 60.2 | Done (at the lower edge). |
| Nominalizations per 100 words | 4.1–4.9 | 6.3 | 4.2 | 4.1 | Done (inside). |
| % passive sentences | 34.7–39.6 | 8.3 | 11.4 | 13.3 | Not a target. Active voice with named actors; the author's own Discussion is 8.3. |
| we/our/I per 100 words | 0.96–1.3 | 0.42 | 0.34 | 0.42 | Not a target. At the author's own rate. Fix 1 adds one "we". |
| Hedges per 100 words | 1.2–1.5 | 0.0 | 0.59 | 0.59 | Not a target. Hedge only weak evidence. One unhedged generalisation, fix 19 (optional). |
| Boosters per 100 words | 0.18–0.72 | 0.0 | 0.25 | 0.25 | Done (inside). |
| AI-guide phrases per 100 words | 0.18–0.24 | 0.0 | 0.0 | 0.0 | Done. None. |
| Numbers per 100 words | 0.24–1.2 | 0.0 | 0.25 | 0.25 | Done (inside). Three numbers in the chapter: 47.2%, 1,815 and 2,048. |
| % sentences with 3+ numbers | 2.0–2.1 | 0.0 | 0.0 | 0.0 | Not a target. |
| Move: method/background restated | 9.7–50.0 | – | 1.4 | 0.0 | Not a target. A summary chapter should not restate methods. |
| Move: pointer to figure/table | 0.0–2.4 | – | 0.0 | 0.0 | Done. |
| Move: result as numbers | 0.0–5.6 | – | 2.9 | 3.3 | Done. |
| Move: result in words | 2.4–16.7 | – | 24.3 | 20.0 | Not a target. "Summary of findings" is a research-question recap; results in words are its job. Fix 5 removes one result from the contributions list. |
| Move: comparison of results | 0.0–9.7 | – | 14.3 | 16.7 | Not a target, same reason. The answers are comparisons (hybrid against single representation). |
| Move: interpretation | 9.5–15.3 | – | 5.7 | 5.0 | Not done, not fixed. Adding interpretation means new content; author's call. |
| Move: comparison with literature | 4.2–9.5 | – | 0.0 | 0.0 | Not done, not fixed. No citation in the chapter. Adding one is new content; author's call. |
| Move: claim of strength | 4.8–9.7 | – | 20.0 | 21.7 | Not a target. 14 of the 15 are the contribution items, which are claims by design. |
| Move: limitation | 0.0–9.7 | – | 8.6 | 10.0 | Done. All six limitation sentences sit in the Limitations section. |
| Move: link/transition | 1.4–2.4 | – | 10.0 | 10.0 | Partly. Fix 1 removes one announcing sentence. The six left are the opening (2) and the three "asked" sentences that frame the research-question answers (structure row "Research questions"), plus the merged contributions lead-in. The share stays at 10% because the sentence count also drops. |
| Move: recommendation/implication | 18.1–19.0 | – | 12.9 | 13.3 | Not a target. Future work and the recommendation in 8.2 cover it. |
| MTLD | 81.1–93.8 | – | 92.0 | 93.9 | Done (inside). |
| MATTR-50 % | 79.8–82.2 | – | 80.9 | 81.3 | Done (inside). |
| Sentence length SD (words) | 9.21–14.1 | 9.24 (Conclusion) | 6.54 | 6.91 | Partly. Fragments merged; lengths now run 9 to 34 words. |
| Sentence length coefficient of variation | 0.38–0.43 | – | 0.39 | 0.36 | Done now; after fixes slightly below because the mean rises. Not worth chasing. |
| Mean change in length between consecutive sentences | 9.66–14.7 | – | 8.33 | 10.3 | Done after fixes (inside). |
| GPT-2 surprisal, bits per token | 6.39–7.04 | 7.57 (Conclusion) | 8.66 | 8.55 | Not a target. Dense named methods ("censored EM", "forest residual detector") raise it; the author's own text is also above the references. |
| Surprisal SD across sentences | 0.75–1.02 | – | 1.03 | 1.01 | Done (inside, edge). |
| Connectives per 100 words | 1.47–1.90 | – | 2.13 | 2.23 | Not a target. "First … Fourth" twice (research-question recap) supply 9 of the hits. Without the ordinals: 1.4. |
| Distinct connectives / connectives used | 0.48–0.65 | – | 0.48 | 0.50 | Done (inside). "so" x4 reduced to x3 (fix 17). |
| % sentences opening with a formulaic connector | 8.3–11.9 | – | 0.0 | 0.0 | Not a target. The author avoids them too. |
| Three-item lists per 100 sentences | 6.25–9.90 | – | 5.7 | 5.0 | Done. Fix 12 removes the nested list in the opening. The rest are content lists (controls, antibody features). |
| Pronouns and demonstratives per 100 words | 4.82–5.75 | – | 3.24 | 3.43 | Not a target. Low because nouns are repeated instead of "it"/"this"; that is clearer. |

### Conclusion: "Concluding remarks" (AI: 13 sentences, 189 words, 3 paragraphs)

Thirteen sentences is small. One sentence moves a share by 7.7 points, and MTLD is unstable under 200 words. Read the paragraphs rather than the numbers.

| Measure | Reference range | Yours | AI now | After fixes | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 25.0–32.5 | 21.0 | 14.5 | 19.2 | Done after fixes, near the author's 21.0. Three fragments merged (fixes 8, 9, 7). |
| % sentences over 30 words | 25.8–42.9 | 12.8 | 0.0 | 0.0 | Not a target (longest is now 30). |
| Dependent clauses per sentence | 0.71–0.96 | 0.76 | 0.38 | 0.70 | Done after fixes (at the edge). |
| Words per paragraph | 95.3–138.9 | 123.1 | 63 | 64 | Not done. Three short paragraphs, each with one point. Merging them is possible but loses the three-step close; author's call. |
| Lexical density % | 61.7–64.3 | 55.7 | 55.3 | 54.5 | Not a target. At the author's own 55.7. |
| Nominalizations per 100 words | 4.2–6.9 | 4.4 | 3.7 | 3.65 | Not a target. |
| % passive sentences | 14.6–30.1 | 25.5 | 23.1 | 30.0 | Done (inside). |
| we/our/I per 100 words | 1.3–1.9 | 0.41 | 1.06 | 1.04 | Not a target. Above the author's own rate. |
| Hedges per 100 words | 0.95–2.2 | 1.4 | 1.59 | 1.56 | Done (inside). One false hit: "appears" in "a new foundation model appears". |
| Boosters per 100 words | 0.24–0.90 | 0.20 | 0.0 | 0.0 | Not a target. |
| AI-guide phrases per 100 words | 0.43–0.97 | 0.0 | 0.0 | 0.0 | Done. |
| Numbers per 100 words | 0.04–0.43 | 0.91 | 0.0 | 0.0 | Done. No numbers in the closing, as the references. |
| % sentences with 3+ numbers | 0.0 | 3.2 | 0.0 | 0.0 | Done. |
| Move: method/background restated | 10.4–15.9 | – | 23.1 | 20.0 | Not a target at 13 sentences. The three are the "foundation model" premise and "the hybrid selects those features"; both carry the closing argument. |
| Move: pointer | 0.0–1.4 | – | 0.0 | 0.0 | Done. |
| Move: result as numbers | 0.0–3.2 | – | 0.0 | 0.0 | Done. |
| Move: result in words | 6.8–14.3 | – | 23.1 | 20.0 | Not a target at this size. |
| Move: comparison of results | 3.0–17.8 | – | 0.0 | 0.0 | Not a target. |
| Move: interpretation | 0.0–9.6 | – | 0.0 | 0.0 | Done. |
| Move: comparison with literature | 0.0–17.9 | – | 0.0 | 0.0 | Done. |
| Move: claim of strength | 15.1–29.9 | 0.0 | 23.1 | 30.0 | Done. Findings stated as claims, with "we" in two of them. |
| Move: limitation | 0.0–5.5 | 15.3 | 0.0 | 0.0 | Done. No limitation sentence in the closing; they are in 8.4. |
| Move: link/transition | 1.4–4.8 | 1.2 | 7.7 | 0.0 | Done after fix 7 ("Chapter 5 tested this..." folded into the sentence before). |
| Move: recommendation/implication | 16.4–17.9 | 18.8 | 23.1 | 30.0 | Not a target. Three sentences; the closing is a recommendation by design. |
| MTLD | 104.2–167.3 | 80.8 | 75.4 | 73.9 | Not a target. Under 200 words; near the author's 80.8. |
| MATTR-50 % | 83.1–87.9 | 79.4 | 78.5 | 77.5 | Not a target, same reason. |
| Sentence length SD (words) | 8.01–12.6 | 9.24 | 5.88 | 7.11 | Partly. Lengths now 11 to 30 words. |
| Sentence length coefficient of variation | 0.32–0.41 | 0.45 | 0.41 | 0.37 | Done (inside). |
| Mean change in length between consecutive sentences | 8.53–12.3 | 9.40 | 8.5 | 11.0 | Done (inside). |
| GPT-2 surprisal, bits per token | 6.57–6.84 | 7.57 | 8.20 | 7.78 | Not a target. Lower after fixes, near the author's 7.57. |
| Surprisal SD across sentences | 0.73–1.03 | 1.19 | 1.67 | 1.07 | Done after fixes (near the edge). The 4-word "The features also matter." was the outlier. |
| Connectives per 100 words | 2.12–2.72 | 2.23 | 1.07 | 1.05 | Not a target. Two connectives in 189 words; adding them for the count would be padding. |
| Distinct connectives / connectives used | 0.33–0.51 | 0.42 | 1.0 | 1.0 | Not a target (two connectives). |
| % sentences opening with a formulaic connector | 4.17–13.0 | 1.06 | 0.0 | 0.0 | Not a target. |
| Three-item lists per 100 sentences | 9.38–14.3 | 12.8 | 30.8 | 30.0 | Partly. Four hits in 13 sentences. Fix 8 removes the one that is rhythm ("predicted better…, gave better neighbors, and stayed ahead"). The three left are content: task/data/model/noise, model/representation/method, molecular/protein/antibody. |
| Pronouns and demonstratives per 100 words | 5.70–7.08 | 4.98 | 6.42 | 6.32 | Done (inside). |

## Structure rows that apply to Chapter 8

| Benchmark row | Status | Fix |
|---|---|---|
| Chapter opening links back | Done. Line 5998 names Chapter 1's research questions. Content note: its first sentence names only label noise, though the second half of the thesis is about the representation (see "Not changed"). | Fix 12 trims the nested list in the roadmap sentence. |
| Chapter ends point forward | Done. Ch8 is the last chapter; Future work (8.5) names the next tests, and the last paragraph (6079) says what a researcher does with the next foundation model. | None. |
| Text between section title and first subsection | Not applicable. Chapter 8 has no subsections. | None. |
| Title case | Not done. The chapter title is Title Case: "Conclusions and Further Thoughts". The back matter is too: `Summary of Research Contributions`, `Research Papers`, `Conference Presentations`. No AI marker sits above any of them, so they count as the author's. | Her call. Sentence case would be "Conclusions and further thoughts", "Summary of research contributions", "Research papers", "Conference presentations". |
| Section titles | Done. "Summary of findings", "Choosing a noise-handling method", "Contributions", "Limitations", "Future work", "Concluding remarks": topic noun phrases, no questions or claims. | None. |
| Numbers in results text / words per float / pointers | Not applicable. No Results section, no floats. | None. |
| Limitations | Done. Own section (8.4), six sentences, 100 words, each specific (first outer fold, one clean control, antibody dataset size, DNN clipping, untested link to dataset properties). None in Concluding remarks. | Fixes 16, 17: "Most" opens or carries four of the six sentences; vary two. |
| Research questions | Done. Answered in order: the four first-half questions (6003), the Chapter 4 question (6006), and the four second-half questions (6010–6019, "First" to "Fourth"), matching Ch1 lines 12 and 20–23. Two content gaps noted under "Not changed". | Fix 18 echoes Ch1's wording ("the method carries over"). |
| Final chapter length (~1,000–1,600) | Done. 1,375 words now, 1,371 after fixes. | None. |
| Hedges per 1,000 words | Partly. Discussion 5.9, Conclusion 15.9 (10.6 without the false hit "appears"). | Fix 19 (optional). |
| Statistics | Not applicable. No tests or intervals in Chapter 8. | None. |
| Negative/positive results | Done. Negatives stated plainly: "no model's uncertainty identified the corrupted labels", "added little", "worse on several datasets", "almost never on small molecules", "fell to chance". | None. |
| Originality | Done. Contributions section (8.3); item 5 says "for which we found no earlier use". Ordered by chapter, not ranked as Buttenschoen does. | None. |
| Caption length | Not applicable. No floats. | None. |
| Figure/table references | Done. Only `Chapter~\ref` and `Section~\ref`, in full. | None. |
| Decimals | Done. One decimal: 47.2% (percentages one decimal). | None. |
| Chemistry wording | Done for this chapter: hERG $K_i$, HOMO–LUMO gap, Buchwald–Hartwig, CDRs defined at first use in the chapter. | None. |
| "you" in body | Done. None in Chapter 8. | None. |

## Fixes

19 fixes, ordered by importance. Each quotes the paragraph's first eight words and the exact old text. No meaning or number changes. Fixes 1–18 were applied to a scratch copy to compute the "After fixes" columns; all old strings occur exactly once in `thesis.tex`.

### Fix 1. Lines 6032 and 6043, "The contributions on label noise are the following." (rows: link/transition; link sentence that only announces)

Two sentences only announce the list below them. Replace the first with a sentence that carries the split. Delete the second, and with it the break between the two lists, so items 1–9 form one list. Numbering is unchanged, since the second list already resumed at 7.

- Old (6032): `The contributions on label noise are the following.`
- New: `We make six contributions on label noise and three on the representation, in that order.`
- Old (6040–6044): `\end{enumerate}` + blank line + `% TODO: manual review` + `The contributions on the representation are the following.` + `\begin{enumerate}[resume]`
- New: nothing. The items of the second list follow item 6 directly.

### Fix 2. Line 6013, "Second, distance in a hybrid tracked the label" (rows: words per paragraph; fragment paragraph; comparison of results without its baseline)

A two-sentence paragraph. It is the second answer of four for the second half; the first sits in 6010. Merge it into 6010 (delete the blank line and the marker line between them). Also say what distance tracked the label better than. The baseline is from Ch6's conclusions (line 4554: "than in the single representation"). The scratch copy used for "After fixes" said "best single representation"; one word, no effect on the measures.

- Old: `...which reads only the regions or mutations that vary.` + blank line + `% TODO: manual review` + `Second, distance in a hybrid tracked the label better on all six small-molecule datasets,`
- New: `...which reads only the regions or mutations that vary. Second, distance in a hybrid tracked the label better than in the single representation on all six small-molecule datasets,`

### Fix 3. Line 6068, "A hybrid can also be the input to" (rows: words per paragraph; fragment paragraph -> merge)

A two-sentence paragraph on uses of the hybrid. The hybrid's other future work is in 6062; the noise future work (6065) sits between them. Append 6068 to the end of 6062 and delete 6068 with its marker. The Future work section then has one hybrid paragraph and one noise paragraph.

- Old (6062 end): `Runs on DNA promoter sequences have started but are not reported here.`
- New: `Runs on DNA promoter sequences have started but are not reported here. A hybrid can also be the input to the model that scores molecules for a generative model; the code exists and has not been run. Starting the search from the settings found on a similar dataset may save compute, and a deployment across real sites would test the separate-site results of Chapter~\ref{chap:method}.`
- Old (6067–6068): `% TODO: manual review` + the 6068 paragraph. New: deleted.

### Fix 4. Line 6019, "Fourth, importance helped in building the hybrid but" (rows: result in words; restatement; sentence length SD)

"but not in training" in the first sentence and "Using the importance scores again during training added little" in the fourth say the same thing. Keep the second wording, in the first sentence. Merge the 10-word mechanism sentence with the result it explains. "feature importance" names the term at its first use in the chapter.

- Old: `Fourth, importance helped in building the hybrid but not in training. The hybrid ranks features by importance within each representation. At the same number of features it beat recursive feature elimination and random features on all but one small-molecule dataset with a result. Using the importance scores again during training added little.`
- New: `Fourth, feature importance helped in building the hybrid, but using the importance scores again during training added little. The hybrid ranks features by importance within each representation, and at the same number of features it beat recursive feature elimination and random features on all but one small-molecule dataset with a result.`

The last sentence, "For finding noisy labels, the best feature space...", stays here. It answers the third part of Ch1's fourth question ("in finding noisy labels").

### Fix 5. Line 6039, item 6 "Tests of these methods on single representations" (rows: claim of strength; result in a contribution list; repetition)

The last sentence is a result, not a contribution. Line 6024 states it already ("The ranking of the methods also changed between the hybrid and the single representations").

- Old: `random deletion and shuffled detector scores (Chapter~\ref{chap:noise}). The ranking of the methods changed with the features.`
- New: `random deletion and shuffled detector scores (Chapter~\ref{chap:noise}).`

### Fix 6. Line 6003, "The first half of the thesis asked four" (rows: sentence length SD; fragment sentence)

The 7-word sentence finishes the comparison in the sentence before it. Merge them.

- Old: `On QM9 under Gaussian noise, model family explained 47.2\% of the variance in \aucnorm{}, the representation far less. The representation mattered more for predictive performance.`
- New: `On QM9 under Gaussian noise, model family explained 47.2\% of the variance in \aucnorm{} and the representation far less, whereas the representation mattered more for predictive performance.`

### Fix 7. Line 6079, "A new foundation model appears every few months" (rows: link/transition in conclusions; fragment sentence)

The 8-word sentence only points to where the claim was tested. Fold it into the claim.

- Old: `and the search decides whether the embedding contributes features and which. Chapter~\ref{chap:method} tested this on LogD and hERG blocking.`
- New: `and the search decides whether the embedding contributes features and which (tested on LogD and hERG blocking in Chapter~\ref{chap:method}).`

### Fix 8. Line 6076, "The features also matter. When the noise methods" (rows: fragment sentence; three-item lists in conclusions; surprisal SD)

A 4-word opener, and a closing tricolon of three claims. Join the opener to the sentence it introduces with a colon. Make the tricolon two parts: the two clean-label results, then the noise result.

- Old: `The features also matter. When the noise methods failed,`
- New: `The features also matter: when the noise methods failed,`
- Old: `On small molecules it predicted better than any single representation, gave better neighbors, and stayed ahead through every noise treatment.`
- New: `On small molecules it predicted better than any single representation and gave better neighbors, and it stayed ahead through every noise treatment.`

### Fix 9. Line 6073, "Label noise in molecular data can be handled." (rows: sentence length SD; words per sentence in conclusions)

An 8-word claim followed by the condition that qualifies it. Join them.

- Old: `Label noise in molecular data can be handled. Which method handles it depends on`
- New: `Label noise in molecular data can be handled, but which method handles it depends on`

### Fix 10. Line 6006, "Chapter 4 then asked why published methods for" (rows: sentence length SD; mean change in length)

The general statement and its example are two short sentences in a run of 10, 11, 15 and 11 words. Join them with a colon.

- Old: `most features of a standard representation barely vary. On hERG $K_i$, 1,815`
- New: `most features of a standard representation barely vary: on hERG $K_i$, 1,815`

### Fix 11. Line 6024, "No single method handled label noise on every" (rows: sentence length SD; parallel sentences)

Three examples in a row, each one sentence of 13 to 17 words with the same shape. Join the second and third.

- Old: `made the model worse on Caco-2's labels as delivered. Deleting the molecules the network fits worst`
- New: `made the model worse on Caco-2's labels as delivered, and deleting the molecules the network fits worst`

### Fix 12. Line 5998, "This thesis set out to handle label noise" (rows: three-item lists; chapter opening)

The roadmap is a three-verb list with a three-noun list inside it. The section titles already name limitations and future work.

- Old: `This chapter answers the research questions of Chapter~\ref{chap:intro}, sets out how to choose a noise-handling method, and lists the contributions, limitations and future work.`
- New: `This chapter answers the research questions of Chapter~\ref{chap:intro}, then sets out how to choose a noise-handling method and what the thesis contributes.`

### Fix 13. Line 6038, item 5 "Three methods for which we found no earlier" (rows: fragment sentence; link "These are")

The item names a count, then lists the three in a second sentence. Use a colon.

- Old: `Three methods for which we found no earlier use (Chapter~\ref{chap:noise}). These are the cliff-aware gap, cleaning the labels before the hybrid search, and returning activity-cliff molecules after a cleanup.`
- New: `Three methods for which we found no earlier use: the cliff-aware gap, cleaning the labels before the hybrid search, and returning activity-cliff molecules after a cleanup (Chapter~\ref{chap:noise}).`

### Fix 14. Line 6027, "We therefore recommend modeling the noise in a" (rows: sentence length SD; fragment sentence)

"Both come with a flag that a correction can use." (9 words) is about the two cases in the sentence before. Join with a semicolon.

- Old: `an assay with a reporting limit gives clipped labels. Both come with a flag`
- New: `an assay with a reporting limit gives clipped labels; both come with a flag`

### Fix 15. Line 6065, "The noise methods need more clean controls. QM9" (rows: sentence length SD; fragment sentence)

The 7-word opener and the QM9 sentence are claim and reason. Join with a colon.

- Old: `The noise methods need more clean controls. QM9 offers`
- New: `The noise methods need more clean controls: QM9 offers`

### Fix 16. Line 6057, "The hybrid needs data, and most antibody datasets" (rows: lexical diversity; repeated openings in limitations)

"Most" carries four of the six limitation sentences ("Most noise results", "most noise methods", "most antibody datasets", "Most DNN results"). Reorder the DNN sentence so it opens on its subject.

- Old: `Most DNN results were computed before the DNN began clipping each prediction to the range of its training labels.`
- New: `The DNN began clipping each prediction to the range of its training labels only after most of its results were computed.`

### Fix 17. Line 6053, "Most noise results rest on the first outer" (rows: distinct connectives; "so" x4 in the chapter)

"so" links cause and effect four times in the chapter (6003, 6006, 6053, 6065). Drop one here; the meaning is kept by the opening phrase.

- Old: `On the labels as delivered there is no ground truth, so a control can show`
- New: `With no ground truth for the labels as delivered, a control can show`

### Fix 18. Line 6010, "The second half asked whether the representation can" (rows: research questions answered in order)

"A method built for small molecules" reads as some other method. Ch1's first second-half question asks whether "the method carry over from small molecules to antibodies and reactions". Use the same words.

- Old: `A method built for small molecules also carried over to antibodies,`
- New: `The method also carried over from small molecules to antibodies,`

### Fix 19 (optional). Line 6027, "We therefore recommend modeling the noise in a" (row: hedges, hedge only weak evidence)

"A method that assumes noise that is not there does harm." is a general claim. The chapter's one example of it is the scaffold-family offsets on Caco-2 (6024). Not applied in the "After fixes" columns.

- Old: `A method that assumes noise that is not there does harm.`
- New: `A method that assumes noise that is not there can do harm.`

## Not changed, on purpose

These are content questions, not language. They are for the author; I did not change them.

- **Opening scope.** Line 5998 says "This thesis set out to handle label noise in molecular data". Half the research questions (Ch1 lines 17–23) are about adapting the representation. The Concluding remarks cover both halves.
- **"found and corrected" against the Limitations.** Line 6016 opens "Third, noisy labels could be found and corrected". Line 6053 says that on labels as delivered "a control can show that a gain beats chance but not that a label was corrected". The first follows Ch1's question 3 wording. The two sentences disagree unless "corrected" is read as "under injected noise only".
- **"most noise methods have no control" against contribution 6.** Line 6053 says most noise methods have no control. Item 6 lists three controls (QM9, random deletion, shuffled detector scores). These may refer to different method sets; not checked against Chapter 7.
- **Two research-question gaps, unverified against Chapters 3 and 5.** Ch1's second first-half question includes deterministic against probabilistic models; 6003's "Second" answers with random forests and boosted trees only. Ch1's first second-half question names antibodies and reactions; 6010 answers antibodies only.
- **Repetition across sections.** "random forests were the most consistent model" appears in 6003 and again in 6027. "A dataset's noise should be understood before a method is chosen" (6073) repeats 6027's recommendation; a closing restatement is normal, so it stays.
- **"the network"** (6016, 6024) is not named in the chapter; elsewhere the thesis says NN or DNN.
- **Title case** of the chapter title and back-matter headings: author's text, listed under the structure rows.
- Three remaining three-item lists in Concluding remarks are content lists and stay.
