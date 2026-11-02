# Chapter 2: fixes from BENCH_AUDIT_ch2.md, applied 2026-10-10

Chapter 2 holds no Chapter 5-7 tables, so no fix was blocked by tonight's table corrections. Every passage was re-found by its text. No number, table, \label or existing \ref was changed. Paragraphs marked "-- fixed:" were not touched.

| Fix | Passage | Outcome |
|---|---|---|
| 1 | Overview | Adapted. Applied as proposed, except "and then molecular similarity" became "along with molecular similarity" so that "then" is not used twice. "the methods for learning with noisy labels" is kept in full. |
| 2 | SVR, kernel sentences | Applied |
| 3 | SVR, three properties | Applied |
| 4 | Similar property principle | Adapted. The proposed merge made a sentence of about 37 words. Instead, "rely on it" is joined to the next sentence: "...rely on it, yet similarity is defined only relative to a representation and a coefficient". |
| 5 | Tanimoto/Euclidean sentences | Adapted. The two "counts" sentences are merged as proposed. The Chapter 4 pointer stays as its own sentence, because the dimensions paragraph was not moved (fix 6). |
| 6 | Dimensions paragraph | Adapted. The paragraph was not moved into the Tanimoto paragraph. Moving it would cross three of the author's paragraphs and delete a TODO marker. Only the in-place merge of the two short sentences was applied. |
| 7 | Activity cliff, both changes | Applied |
| 8 | Noise sources interact | Applied |
| 9 | Bemis-Murcko | Applied, the one-sentence option. The footnote option needs the author's agreement, so it was not used. |
| 10 | LNL paragraph | Adapted. No definition was cut. The kernel and noise-term items and the scaffold-family example are kept, because the table is not the only place a reader looks. The changes are merges only: detector + suspicion score joined with ", and"; "A known flag, in contrast, ..."; "Noise detection is a detector alone, whereas noise mitigation ..."; the figure pointer is now in parentheses. |
| 11 | Song2022 paragraph merged into LNL | Adapted. The Song paragraph is joined to the end of the LNL paragraph, before the sample-selection sentence, which now opens "The exception is ...". The proposed parenthetical form made a sentence of about 33 words, so the five groups are listed after a colon instead. The Song paragraph's duplicate "% TODO: manual review" line was removed in the join. The merged paragraph keeps its own marker. |
| 12 | Data-cleaning caption | Applied |
| 13 | Cortes2015 / Wilson1972 | Adapted. The Cortes merge was applied. The proposed Wilson sentence had two "which" clauses, so it was split after "the similar property principle" and the next sentence opens "However, activity cliffs break that principle". "the hybrid" became "the adaptive hybrid representation". |
| 14 | Implementation opener + "cluster run" | Applied |
| 15 | Regression noise + requested-noise paragraphs | Applied. The paragraphs were joined and the second marker removed. |
| 16 | Second function, degenerate scale | Adapted. The proposed text added the claim "a rank correlation with the scale is undefined", which the original does not state, so that claim was not added. The new text reads: "...gets the same scale, so the package flags these scales as degenerate instead of reporting a rank correlation of zero." The paragraph was joined to the next one and the second marker removed. |
| 17 | "In the KIRBy experiments" | Applied: "; in Chapter~\ref{chap:noise} this unit is". The residual-sized unit appears in Chapter 7 only: grep found it in the damage table and in the Chapter 7 detector and method text. It is not used in Chapters 5 or 6. |
| 18 | AUCnorm claim | Applied: "(Chapter~\ref{chap:robustness})". The backing text is at "we found \aucnorm{} to be less directly correlated..." and "unrelated to clean $R^2$" in Chapter 3. |
| 19 | "All of these use the clean test labels" | Applied |
| 20 | "It is reported within each noise level" | Applied |
| 21 | Three wrappers | Applied |
| 22 | Evaluation metrics sentences | Applied |
| 23 | Conclusions | Adapted. Both changes were applied, with two word changes. "fixes what counts as similar" became "shapes what counts as similar", because the coefficient also matters. "We implemented these conditions" became "We implemented the seven conditions", because "these" pointed two sentences back. |

## Extra pass over all AI paragraphs

Beyond the list above, only the two Overview and Conclusion word changes noted under fixes 1 and 23 were made. The remaining AI paragraphs were read against the seven targets and left unchanged:

- The censoring paragraph is a short fragment. It has no neighbor on the same topic to merge with.
- The classification, robustness and uncertainty specification lists must list their contents.
- The splits paragraph varies in sentence length already.

## Counts

- Applied: 14 (fixes 2, 3, 7, 8, 9, 12, 14, 15, 17, 18, 19, 20, 21, 22)
- Adapted: 9 (fixes 1, 4, 5, 6, 10, 11, 13, 16, 23)
- Skipped: 0
- Extra edits beyond the list: 0, apart from the word changes folded into fixes 1 and 23
