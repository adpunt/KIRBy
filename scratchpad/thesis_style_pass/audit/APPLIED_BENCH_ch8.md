# Chapter 8: fixes from BENCH_AUDIT_ch8.md, applied 2026-10-10

Chapter 8 text had not changed between `thesis.tex.before_bench_fixes_2026-10-10` and the live file before these edits. The table audits (TABLES_ch5/6/7.md) changed nothing in Chapter 8. So every OLD string was found as written. No number, \ref, \label, table or caption was changed.

| Fix | Paragraph | Status | Note |
|---|---|---|---|
| 1 | Contributions lead-ins | Applied | First lead-in is now "We make six contributions on label noise and three on the representation, in that order." Second lead-in, its TODO line and the list break deleted; items 1-9 are one list, numbering unchanged. |
| 2 | "Second, distance in a hybrid tracked..." | Applied | Merged into the "second half asked" paragraph; its TODO line went with the paragraph break. Baseline wording "than in the single representation" taken from Ch6's conclusion (thesis.tex, Ch6 closing paragraph), not "best single representation". |
| 3 | "A hybrid can also be the input to..." | Applied | Appended to the first Future work paragraph; separate paragraph and its TODO line deleted. |
| 4 | "Fourth, importance helped..." | Applied | As proposed. |
| 5 | Contribution item 6 | Applied | Dropped "The ranking of the methods changed with the features." (already stated in 8.2). |
| 6 | "The first half of the thesis asked four..." | Applied | As proposed. |
| 7 | "A new foundation model appears..." | Applied | As proposed. |
| 8 | "The features also matter..." | Applied | Both parts (colon; three-part list made two parts). |
| 9 | "Label noise in molecular data can be handled." | Applied | As proposed. |
| 10 | "Chapter 4 then asked why..." | Applied | As proposed. |
| 11 | "No single method handled label noise..." | Applied | As proposed. |
| 12 | Chapter opening | Applied | As proposed. |
| 13 | Contribution item 5 | Applied | As proposed. |
| 14 | "We therefore recommend modeling..." | Applied | As proposed. |
| 15 | "The noise methods need more clean controls." | Applied | As proposed. |
| 16 | Limitations, DNN sentence | Applied | As proposed. |
| 17 | Limitations, "so" | Applied | As proposed. |
| 18 | "The second half asked..." | Applied | As proposed. |
| 19 (optional) | "A method that assumes noise that is not there does harm." | Applied | Now "can do harm": the chapter gives one example (Caco-2 scaffold offsets), so the general claim is hedged. |

## Extra fix from the re-read against the seven style targets

| Where | Change | Why |
|---|---|---|
| "Third, noisy labels could be found and corrected" | "added to the network" -> "added to the neural network" | Target 5: "the network" is never introduced in Chapter 8. Chapter 7 uses "the network" for the neural-network downstream model; the second use in 8.2 ("the network fits worst") now follows a first use that names it. |

Nothing else changed. The remaining three-item lists are content lists (controls, antibody features, task/data/model/noise). Content questions listed under "Not changed, on purpose" in BENCH_AUDIT_ch8.md are still open for the author.
