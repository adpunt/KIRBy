# Chapter 6: benchmark edits (2026-10-10)

Prose word count of Chapter 6, comment lines removed: backup 6017, now 5811.
Paragraphs edited: 29. Tables built: 0. Suggestions for the author's paragraphs: 2 (SUGGESTIONS_ch6.md).
No number was added or changed. Numbers were only removed (rank scheme paragraph; conclusions' limitations).
Every edited paragraph has a "% TODO: manual review" line above it, kept as it was. No comment, label, ref or table body changed.

| First eight words (before) | Words before | Words after | What changed |
|---|---|---|---|
| There are two ways to use a hybrid | 71 | 66 | Active voice ("we measure", "fits no downstream model"); shorter "predicted from its five nearest neighbors" chain. |
| A fold of the search is undefined when | 55 | 49 | Cut "rather than ranked on fewer folds" (a refused candidate is not ranked at all). |
| We found no published source for the rank | 66 | 66 | Cut the counts 27 and 20 of 48 and the fold spread 0.02 to 0.12; said in words (rank scheme won just over half the pairs, 1/d^2 most of the rest, equal weights none; margin below the fold spread). 0.0015 RMSE kept. |
| This follows \citet{Sheridan2004}, who showed that a molecule's | 108 | 106 | "showed" to "found"; merged the "This is done at..." sentence into the one before; "We do this in each feature space". |
| As in Chapter~\ref{chap:flaws}, distance in a standard representation | 49 | 47 | "the difference in labels" to "label difference"; split the 34-word sentence. |
| In the distance hybrid, distance tracks label differences | 34 | 31 | Fold sentence shortened ("higher in every fold on LogD and lower in two folds on Caco-2"). |
| The same holds on all seven antibody datasets | 14 | 11 | Cut "including SMAC colloidal stability" (it is one of the seven); added table pointer. |
| The predictive hybrids, built for each downstream model | 73 | 67 | Cut "rather than for distance" contrast; dataset list moved to the front; "all four predictive hybrids" to "all four". |
| Importance weighting on its own closes much of | 65 | 57 | "on its own" to "alone"; shorter LogD and QM9 sentences. Missing-result sentence kept here (removed from conclusions). |
| The single representation here is the one the | 75 | 59 | Cut "not the one whose distance best follows the label" contrast; the hindsight pick defined once, then one sentence listing the four datasets where it beats the distance hybrid, then the Caco-2 similarity-hybrid sentence. |
| At the setting fixed before the run, predicting | 49 | 46 | "predicting each molecule from" to "prediction from"; "one of five folds". |
| Picking the setting again for each fold does | 36 | 33 | Colon instead of "and"; cut "for example". |
| On the antibody datasets and Buchwald--Hartwig, the result (results) | 90 | 88 | Cut "also" and "however". |
| Multiplying features by their importance mostly helps prediction | 81 | 77 | Table pointer in parentheses with "exceptions in bold"; merged the setting sentences with a colon. |
| Weighting also improves prediction from all representations joined. | 81 | 74 | "At least one importance method helps on every antibody dataset"; shuffled-weights caveat shortened, kept next to its result. |
| Whether the hybrid wins depends on the dataset | 171 | 164 | "far more on the dataset"; "Weighting both sides"; "we cannot separate size from domain"; "for the hybrid" cut; "but" to "while". |
| Retrieval (Section~\ref{sec:similarity:results_other}) shows the same pattern. | 84 | 81 | Split the 40-word sentence in two; split the cosine sentence. |
| We also tried choosing which molecules to measure | 41 | 35 | Datasets first; "we also simulated"; cut repeated "on both". |
| We report activity cliffs on the five small-molecule | 88 | 79 | Shorter antibody sentence; "the one that makes 15% of the training molecules cliff molecules" to "the one for training molecules" (15% is in Methods); shorter share sentence. |
| On LogD, Caco-2, hERG $K_i$ and hERG blocking, | 100 | 90 | Cut "alike", "than on the rest" (twice), "cliff molecules" repeat. |
| QM9 is harder to read. The hybrid is | 75 | 73 | Cut the opener "QM9 is harder to read"; split the two-fold sentence; "Each model's hybrid is better in one of the two folds". |
| A molecule's distance to its nearest training molecules | 69 | 67 | "does rise ..., but weakly" to "rises weakly". |
| On nanobody thermostability, $\Delta\Delta G$, expression titer, CHO | 64 | 62 | Cut "at all". |
| The distance hybrid retrieves neighbors with closer labels | 82 | 76 | Clustering sentence rewritten without "agrees with retrieval ... where"; semicolon for the Caco-2 clause. |
| On the other domains it depends on the | 107 | 99 | Cut the link sentence "On the other domains it depends on the dataset"; Buchwald--Hartwig in its own sentence. |
| Chapter~\ref{chap:flaws} found that distance in standard representations barely | 92 | 91 | Finding stated with "we found"; table pointers dropped from the conclusions; "label difference". |
| Weighting features by importance does part of the | 54 | 54 | "it lets the single representation track label differences better"; "The hybrid's distance", "its kept features". |
| On the antibody datasets and Buchwald--Hartwig, the result (conclusions) | 87 | 86 | Opens with the finding (distance tracks label differences more closely; the methods built on it do not always gain); "we must measure". |
| Four limitations apply. QM9 has no result for | 127 | 48 | One short limitations paragraph: missing QM9 results and the comparison that favors the hybrid. Cut from here, still stated in Results next to their result: the missing weighted comparison (Importance weighting... paragraph), the looser held-out cliff rule (We report activity cliffs... paragraph), the 48 QM9 cliff molecules (QM9 paragraph). Cut "Four limitations apply" and the sentence that the cliff comparison is unaffected. |

Not edited (AI-marked, in Results, already short or no gain found): "On the small-molecule datasets and Buchwald--Hartwig, no representation", "A hybrid's prediction score peaks at more features", "The features a hybrid keeps carry no more", and the closing link paragraph "Methods that find noisy labels".

Checked but not changed (outside this pass): the two tables read from figures_ch5_7/ (tab_distance_tracks_activity, tab_nearest_neighbour_prediction) have 14 rows, including MDR1-MDCK efflux, while the chapter says 13 datasets and five small-molecule datasets. The comment above each \input says 13 rows.
