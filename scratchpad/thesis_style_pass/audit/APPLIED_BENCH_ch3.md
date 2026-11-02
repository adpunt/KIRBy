# Chapter 3: benchmark fixes applied (2026-10-10)

Source list: `BENCH_AUDIT_ch3.md`. File: `thesis.tex` (Edit tool only). `main.tex` not touched; the abstract had no Chapter 3 fix.
No number, \ref or \label changed. Every passage was re-found; none had changed since the audit.

| Fix | Where | Outcome |
|---|---|---|
| 1 | Introduction, one-sentence paragraph | Skipped. Joining it to the next paragraph would put the `% TODO: manual review` marker above the author's paragraph, so her text would later read as AI-marked. Left for her. |
| 2 | Log D / Caco-2 / hERG paragraph | Applied. "measured in both directions, and the efflux ratio is ..." |
| 3 | Six variant models sentence | Applied. Semicolon list, short name first, description after. |
| 4 | "The seven noise conditions ..." | Adapted. Dropped "used in this chapter". Did not merge into the next paragraph (author's text; same marker reason as fix 1). |
| 5 | mmap paragraph, Python/Rust hand-off | Applied. |
| 6 | Noise-drawing paragraph | Applied. "Validation labels get all three, with the amount in label units, ..." |
| 7 | "For every molecule, the Rust program records" | Applied. "the program records". |
| 8 | Log D sentence, repeated "so" | Applied. "and a basic amine ... therefore has a log $D$ below its log $P$". |
| 9 | Lead-in under `\section{Results}` | Skipped. It adds new AI prose with no marker of its own above it, inside the author's Results section, and the thesis is over its page limit. |
| 10 | Table `tab:condition_choice`, two rows | Applied. Grouped-shifted and censoring "Avoid" statements moved to "Extent of the evidence"; censoring "Avoid" is now a dash, which the caption already explains. Wording of the claims unchanged. |

## Second pass over the AI paragraphs

All seven marked paragraphs and the table were re-read against the seven style targets. No further edits:
- The record-field lists and the 13-model list are content lists, not filler triplets.
- The hERG sentences (8, 13 and 19 words) were left; merging them would chain two "so"/"and" clauses into one long sentence.
- No coined phrasing, conclusion-type sentences or table re-description in the AI text.

Counts: 7 applied, 1 adapted, 2 skipped, 0 extra.
