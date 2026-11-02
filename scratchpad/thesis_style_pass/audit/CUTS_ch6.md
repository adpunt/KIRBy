# Chapter 6 cuts, items 19 and 25 (2026-10-10)

Table numbers come from scratchpad/thesis_code_todo/build/main.aux (built 10 Oct 10:08):
- Table 6.13 = tab:similarity:knn_by_setting ("Prediction from nearest neighbors by setting and by fold")
- Table 6.14 = tab:similarity:retrieval_by_setting ("Label gap to the nearest neighbors by setting and by fold")
- Table 6.19 = tab:similarity:label_information ("Label information in the features a hybrid keeps")

All three were checked by caption and content against the decision descriptions.

New Additional files items go under the existing section `app:add:similarity`
("Hybrid representations for similarity and label noise"). They sit before the LNL settings item.
Each is one `\addfile{...}` with its own label, as the other items in that file are.

## Item 25: Tables 6.13 and 6.14 to the Additional files

Moved: both landscape longtables, cut from thesis.tex and pasted unchanged.
Each kept its caption, its label, and its SOURCE comment block.
I diffed the moved blocks against the thesis.tex originals before cutting: they are byte-identical.

New Additional files items:
- `app:add:similarity_knn_by_setting`, holding `tab:similarity:knn_by_setting`
- `app:add:similarity_retrieval_by_setting`, holding `tab:similarity:retrieval_by_setting`

Each new item has one intro line, marked "% TODO: review, needs manual review":
- "One row is one dataset and feature space. It supports Section~\ref{sec:similarity:results_knn}." (used for both)

References repointed in thesis.tex. Only the word "Table" became "Additional file"; the text is otherwise unchanged:
- sec:similarity:results_knn, "...worse in one of five folds on hERG K_i and Caco-2 (Additional file~\ref{tab:similarity:knn_by_setting})."
- "Whether the hybrid wins depends far more on the dataset ... (Additional file~\ref{tab:similarity:knn_by_setting})."
- "Retrieval ... shows the same pattern (Additional file~\ref{tab:similarity:retrieval_by_setting})."

No replacement sentence was added where the tables stood.
The two paragraphs just above them already say what the tables show, and they now carry the pointers.
Adding a sentence would only repeat them.
In place of each table, thesis.tex has a "% [MOVE 2026-10-10]" comment.

Comment edits in thesis.tex:
- The "% Check 2026-10-10" comment said "the table below's source". It now names tab:similarity:knn_by_setting as being in the Additional files.
- Removed the obsolete comment "% [2026-10-10] moved up from after Table tab:similarity:knn_by_setting ...". It referred to the table's old position.

No other file references these labels: THESIS_OUTLINE.md, the stage1 folder, tests/analysis and figures_ch5_7 were checked.

## Item 19: Chapter 6 label information, Table 6.19

Paragraph: AI-written ("% TODO: manual review" directly above it). It was two sentences and is now one.
- Kept: "Except on QM9 and Buchwald--Hartwig, the features a hybrid keeps carry no more label information than principal components of the same number (Additional file~\ref{tab:similarity:label_information})."
- Cut: "On LogD the median, summed over components, is 1.63 nats for the kept features against 1.68 for principal components, and lower for a random subset."
- The kept sentence has "% TODO: manual review" above it, and a "% [CUT 2026-10-10]" comment above that.

Moved: the table, with its caption, its label and its SOURCE comment, unchanged (byte-identical by diff).
It now sits in Additional files item `app:add:similarity_label_information`.
That item's intro line: "One row is one dataset. It supports Section~\ref{sec:similarity:results_other}."

References repointed: the only \ref was the sentence above; it now reads "Additional file~\ref{...}".
The Conclusions restate the same claim without a \ref, so nothing there changed.

No numbers, claims or table contents were changed.
Not built: neither main.tex nor additional_files.tex was compiled after the edits.
