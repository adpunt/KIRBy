# Chapter 4 — benchmark edits (2026-10-10)

Backup: `thesis.tex.before_benchmark_edits_2026-10-10`. Only paragraphs with a "% TODO: manual review" marker (or "% TODO: review, needs manual review") were edited. Word counts are prose words with comment lines removed. No number was added or changed.

Chapter 4 prose word count: backup 7,549; now 7,499.

| First eight words | Words before | Words after | What changed |
|---|---|---|---|
| Two molecules are tied when a representation gives (Results) | 66 | 65 | "We counted ties" now opens with "Table~\ref{tab:flaws:ties} counts ties"; the trailing table pointer was dropped. |
| This section reuses a small sweep of published (Results) | 60 | 55 | "carried out earlier" to "from earlier"; "the largest share of the variance in robustness to label noise" to "the largest share of variance in noise robustness"; "the representation still matters" to "representation still matters". |
| The study tested 12 published noise-handling methods on (Results) | 118 | 115 | The separate sentence listing the five distances was folded into the sentence that introduces the distance, as a parenthesis. |
| We measured noise robustness by retention, $R^2$ under (Results) | 112 | 108 | "is more directly correlated with" to "correlates more closely with"; the "but" clause split into its own sentence; "whether a distance was used" to "using a distance"; the last sentence split in two. The meaning of the "none with a distance" claim is unchanged (the TODO above it still applies). |
| The study used one dataset, one property and (Results) | 59 | 55 | "one noise condition (Gaussian noise on every training label)" to "Gaussian noise on every training label"; "the distance step" to "a distance". |
| In the small sweep of published noise-handling methods, (Conclusions; now opens "We found that methods built on a distance") | 67 | 53 | Opens with the finding stated with "we". "did worst" now says the lowest-retention configurations all used a distance (the Results paragraph "We measured noise robustness" gives the 20). Cut the caveat that one neighbor-based configuration gave the best recovery at the higher noise level (it stays in Results, "Judging a label by its neighbors..."). Moved the "noise on every training label" caveat to the limitations paragraph. |
| Three limits apply to this chapter. The sweep (Conclusions; now opens "We tested the published methods and the") | 69 | 50 | Now the chapter's one limitations paragraph, stated with "we": one dataset, Gaussian noise on every label (so neighbors were noisy too), and the distance added to methods built without one. Cut the cliff-threshold caveat (0.55 and 15%); both values stay in Results, "Activity cliffs". Two numbers removed. |

Not edited:
- "The study also found neighbors in an early hybrid" (AI-marked, Results, rank 46). Its seven numbers fit the columns of Table~\ref{tab:code_input_neighbor_recovery}, but adding them as a row would contradict the author's sentence above the table ("Table ... holds only the configurations built on a single representation"). A wording edit made it longer, so it was undone.
- "Chapter~\ref{chap:method} builds the adaptive hybrid representation" (AI-marked, Conclusions): one sentence, already plain; not in the worklist.
- "The later chapters use a different set" (AI-marked, Methods, rank 422): outside the scope of this pass.
- The AI-marked row in Table~\ref{tab:flaws:requirements}: a table body.

Tables built: 0. Suggestions for the author's paragraphs: 17, in `SUGGESTIONS_ch4.md`.
