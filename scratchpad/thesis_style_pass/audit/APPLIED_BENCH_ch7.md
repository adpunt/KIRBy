# Chapter 7: fixes from BENCH_AUDIT_ch7.md, applied 2026-10-10

Written by Claude. File: `thesis.tex`, Chapter 7 (`chap:noise`). Snapshot before: `thesis.tex.before_bench_fixes_2026-10-10`.
Only paragraphs under `% TODO: manual review` were edited. No number, table, caption, `\ref` or `\label` changed.

## Fixes from the list

| Fix | Status | Note |
|---|---|---|
| 1 | Applied | Conclusions limitations paragraph: two repeats cut ("Most methods have no control", "QM9's HOMO--LUMO gap is our only clean control"); dataset list moved to the end of the Methods datasets paragraph. |
| 2 | Applied | Forward sentence joined to the end of the limitations paragraph, before `% RESULT PENDING`. The 11-word paragraph and its `% TODO: manual review` line removed (paragraph no longer exists). |
| 3 | Applied | |
| 4 | Applied | "single representation" kept in the added sentence ("the model's own single representation"). |
| 5 | Applied | Paragraph moved with its three comment lines under `\section{Results}`. Also joined "...held-out score; because ... optimistic" and "...fitted it, whereas LightGBM..." (sentence length). |
| 6 | Adapted | Moved with its three comment lines under `\subsection{Effect by noise type}`. "the damage kinds sort by whether a flag" now "the types of damage sort by whether a known flag" (coined phrase). |
| 7 | Applied | Moved under `\subsection{Effect by dataset}` without "also" and the pointer. Its TODO line moved with it. |
| 8 | Adapted | Applied. The `% TODO: decide` and `% TODO: manual review` lines of the deleted caveat paragraph kept where they were (after the table input), as comments only. |
| 9 | Applied | |
| 10 | Applied | |
| 11 | Applied | Old text still present after the table pass. |
| 12 | Applied | |
| 13 | Applied | |
| 14 | Applied | |
| 15 | Adapted | Text now says "makes both worse" (table correction); joined as "makes both worse, so the gain depends". |
| 16 | Applied | |
| 17 | Applied | Together with Fix 42. |
| 18 | Applied | |
| 19 | Applied | |
| 20 | Applied | |
| 21 | Applied | |
| 22 | Applied | |
| 23 | Applied | |
| 24 | Applied | |
| 25 | Applied | |
| 26 | Applied | |
| 27 | Applied | |
| 28 | Applied | |
| 29 | Applied | N7-10 sentence merged into the N7-09 paragraph after the "three shuffles" sentence; its SOURCE [N7-10] and NEW RESULTS [N7-10] lines moved above that paragraph. Its `% TODO: manual review` line removed with the paragraph. "On hERG," after it became "On hERG blocking," so it is not read as hERG $K_i$. |
| 30 | Applied | |
| 31 | Applied | |
| 32 | Applied | |
| 33 | Applied | |
| 34 | Applied | |
| 35 | Applied | |
| 36 | Applied | |
| 37 | Applied | |
| 38 | Applied | |
| 39 | Applied | |
| 40 | Applied | |
| 41 | Applied | |
| 42 | Applied | |
| 43 | Applied | |
| 44 | Applied | |
| 45 | Applied | |
| 46 | Applied | |
| 47 | Applied | |
| 48 | Applied | Inside the moved paragraph (Fix 6). |
| 49 | Applied | |
| 50 | Applied | |
| 51 | Applied | |
| 52 | Applied | |
| 53 | Applied | |
| 54 | Applied | |
| 55 | Applied | |
| 56 | Applied | |
| 57 | Skipped | Rounds 0.5072/0.3980 to three decimals. Tonight's rule: never change a number. |
| 58 | Skipped | A `\subsubsection` heading, not a paragraph under a `% TODO: manual review` line. |

## Further edits from the second read (seven style targets)

1. Methods, "Noise detection" close-pair paragraph: "It builds flag sets of the same size ... It then finds the share" now "It flags sets of molecules of the same size ..., and finds the share" (coined "flag sets"; two sentences joined).
2. Results, N7-06: "uses flag sets whose content we control" now "uses sets of flagged molecules whose content we control".
3. Methods, "Each dataset has two downstream models": the two exception sentences joined with "and".
4. Methods, expanded sweep: "the same network as the plain model. It has three hidden layers" joined with "with"; "wider in four ways. It runs" joined with a colon.
5. Results, N7-21: "The gain holds on the further folds." joined to the next sentence with a colon.
6. Results, N7-26: "took it from 0.321 to 0.280 RMSE. The random-intercept model was best" joined ("and it was best").
7. Results, N7-35: "Cleaning the search alone does worse than no cleaning." joined with "while".
8. Results, N7-44: "The labels are the same in both runs. The difference is" joined with a semicolon.

Not changed: "damage kind" elsewhere in the chapter. It is the column heading of `tab:noise:kinds` and the caption name; changing it in prose alone would leave the table and text using different words.
