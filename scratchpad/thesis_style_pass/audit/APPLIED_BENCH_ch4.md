# Chapter 4: fixes from BENCH_AUDIT_ch4.md (2026-10-10)

No table audit (TABLES_ch5/6/7.md) touches Chapter 4, so every passage still had its old text.
Snapshot to revert from: thesis.tex.before_bench_fixes_2026-10-10.

| Fix | Paragraph (first words) | Result |
|---|---|---|
| 1 | "The later chapters use a different set" | Applied. |
| 2 | "Two molecules are tied when" | Applied. |
| 3 | "In Chapter 3, model family explained" (was "This section reuses a small sweep") | Applied. The "% TODO: Ch3 update" line stays above it. |
| 4 | "The study tested 12 published noise-handling methods" | Applied. Extra edit in the same paragraph: its first two sentences were joined ("..., adding Gaussian noise with sigma from 0 to 1.0 ..."). |
| 5 | same paragraph | Adapted. The method names follow a colon, as proposed, but "the other eight change the training" is now its own sentence instead of following a semicolon, so the list's end is clear. |
| 6 | "We measured noise robustness by retention" | Applied. |
| 7 | same paragraph | Applied. The TODO above it about what "none" covers still applies. |
| 8 | "The study added Gaussian noise to every training label" | Adapted. New text: "The study added Gaussian noise to every training label of a single dataset and property, the QM9 HOMO--LUMO gap." Keeps "one dataset" from the old text, which the audit's version dropped. Second sentence applied as proposed ("... for their low retention"). |
| 9 | same paragraph | Applied ("the same 12 methods"). |
| 10 | "Neighbor-based noise detection treats a label" | Applied. |
| 11 | "The study also found neighbors in an early hybrid" | Applied. |
| 12 | "We found that methods built on a distance" | Applied. The Chapter 7 pointer it removes is still in "The study added Gaussian noise..." (Results). |
| 13 | "We tested the published methods and the neighbor-based cleaning" (Conclusions) | Applied: paragraph and its "% TODO: manual review" line removed. Its three points remain in "The study added Gaussian noise..." (Results) and in your sentence "Because noise was added to every training label, the neighbors' labels are noisy too". Your paragraph "Feature selection cannot fix every flaw" is now the one limitations paragraph in Conclusions. |
| 14 | Table tab:flaws:requirements, last row | Skipped. It replaces a \ref, which tonight's rules forbid. The fault stands: Section "Flaws in standard representations" never shows features that are hard to read chemically. |
| S1 | Methods has one subsection | Skipped. Option A deletes a heading marked "-- moved:", which is not AI text to edit; option B adds text, which the rules forbid. |
| S2 | No text under Results | Skipped. The audit recommends leaving it; adding text is forbidden. |
| O1-O4 | Options for the author | Not applied; they are her decisions. O4 is settled by fix 13 (only "is an alternative explanation" remains). O3 (two score pairs in "The study also found neighbors...") still needs her choice. |

Re-read of the other marked paragraphs ("The four measurements below use", "Chapter 5 builds the adaptive hybrid representation"): no change needed. The marker under \section{Conclusions} with a blank line before "On a single dataset" was treated as hers, as the audit asks her to confirm; not edited.

No number, table, caption, \ref or \label changed.
