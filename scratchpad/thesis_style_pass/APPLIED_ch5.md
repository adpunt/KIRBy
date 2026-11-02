# Applied: PLAN_ch5.md (2026-10-10)

Many OLD blocks no longer matched. Since the plan was written, an earlier pass had already applied about thirty of its edits, and the 2026-10-10 number cut had rewritten most results paragraphs. Where an OLD block now matched only a `% CUT FROM TEXT` comment line, the live sentence was edited instead (or the edit skipped). Comment lines were never edited.

- E1 applied (Chapter 5 representation table, hERG K_i row: "inhibition" -> "binding").
- E2 skipped: already in the file.
- E3 applied.
- E4 skipped: "QM9 included" is already in the file. Chapter 2 now defines the Bemis--Murcko scaffold with \citep{Bemis1996}, so the definition is not repeated.
- E5 applied (COVID Moonshot checked against the Dablander thesis text, lines 2252 and 5025).
- E6 adapted: OLD is now a comment. The live results sentence "We used the three ChEMBL targets ..." became "the three targets ..., two from ChEMBL and one from the COVID Moonshot project".
- E7 skipped: already in the file (assays spelled out, in a slightly different wording).
- E8, E9, E10, E11 skipped: already in the file.
- E12 applied.
- E13 applied.
- E14 applied.
- E15 skipped: already in the file.
- E17 adapted: new lead-in paragraph added with a TODO line. Its last sentence was split, and "weighting by importance" became "using importance during training", to match the subsection title.
- E18 to E31 skipped: already in the file.
- E32 skipped: already in the file (both parts, with its TODO line).
- E33, E34, E35, E36, E38 skipped: already in the file.
- E39 adapted: the numbers cut had removed the LogD figure. Now "with a result" is added, the RMSE range is limited to the regression datasets, and the two-ways sentence is split. "not which of the two" became "but it does not say which of the two".
- E40 adapted: the long sentences were already split by the number cut. Only "rat clearance" -> "rat microsomal clearance" (twice) was applied.
- E41 skipped: commit 26ee52f reworked figure F2a after the plan was written, and it now plots held-out score against budget. The current caption fits the new figure better than the planned one, and the figures session owns this caption.
- E42 skipped: commit 9789969 reworked figure F3 (hatching, panels). The caption is left to the figures session.
- E43 applied (TODO line added inside the longtable, above \caption).
- E44 adapted: the numbers cut had already removed the Caco-2, QM9 and LogD pairs. Only the "The comparison is a fair one: ..." sentence was split into two plain sentences.
- E45 to E51 skipped: already in the file.
- E52, E53, E54, E55 applied.
- E56 adapted: the live text differs after the numbers cut. The first sentence is split, and the shuffling sentence is reworded as "We shuffled ... the drop in the hybrid's score is spread about equally over five to seven representations".
- E57 skipped: OLD is now a comment, and the live text is already split.
- E58 part 1 skipped: OLD is now a comment, and the live sentence is under 30 words. Part 2 applied. Part 3 skipped: OLD is now a comment, and the live text is already split.
- E59 part 1 skipped: OLD is now a comment, and the live text has no numbers and is split. Part 2 applied.
- E60 applied (both parts).
- E61 adapted: OLD is now a comment. The same split was made on the live sentence, which says "most datasets" in place of "15 of the 18".
- E62 applied (both parts).
- E63 applied.
- E64 applied.
- E65 adapted: only the first sentence (kernel diluted) was split. The antibody sentence is already split in the live text.
- E66 part 1 applied. Part 2 skipped: OLD is now a comment, and the live text is already split. Part 3 applied.
- E67 applied.
- E68 adapted: OLD is now a comment. The missing noun was fixed in the live sentence ("the hybrid's intervals cover close to 90\%").
- E69 part 1 skipped: already split in the live text. Parts 2 and 3 applied.
- E70 skipped (both parts): OLD is now a comment, and the live text is already split.
- E71 part 1 skipped: already split in the live text. Part 2 applied. Parts 3 and 4 skipped: already split in the live text.
- E72 applied.
- E73 part 1 applied. Part 2 adapted: the numbers cut changed this paragraph, so only the "No single setting suits ...:" sentence was split. Parts 3 and 4 applied.

Counts: applied 23 edits, adapted 9, skipped 39. Of the 39 skips, 37 were already made in the live text, and 2 (E41, E42) are figure captions left to the figures session. Edits with several parts count once, by their main outcome.

For the author:
- Chapter 5 still has a separate \section{Limitations} (sec:method:limitations). The rule is that limitations sit inside the Conclusions. This plan does not move it, so the structure or titles pass needs to.
- The Limitations and Conclusions still give numbers that the 2026-10-10 cut removed from the Results text. These are "up to 0.16 Spearman" and "no one holds more than 29\% of the signal". The Results text now says "a third" for the second.
