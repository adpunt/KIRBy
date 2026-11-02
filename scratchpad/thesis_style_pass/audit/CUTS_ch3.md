# Cuts, Chapter 3 (items 5 and 11), 2026-10-10

Label-to-number check: scratchpad/thesis_code_todo/build/main.aux gives tab:condition_choice = Table 3.6 ("Which model to prefer under each noise condition") and fig:fig_mmap = Figure 3.1 ("In-memory and memory-mapped data loading"). Both confirmed by content in thesis.tex.

## Item 5: Table 3.6, cut
- Cut from thesis.tex (Ch3 Discussion): the whole table float `tab:condition_choice` with its `% TODO: manual review` line. It had no SOURCE comment.
- Label removed: `tab:condition_choice`. It had one reference, in the author's paragraph "Given that, our recommendation is ...".
- Reference repointed: the content is already in that same paragraph's following sentences, so the pointer was removed. "The pattern of the noise should also guide that choice (Table~\ref{tab:condition_choice})." became "The pattern of the noise should also guide that choice." Nothing else in her paragraph changed.
- No other file referenced the label (thesis.tex, additional_files_body.tex, THESIS_OUTLINE.md, thesis_appendix/ checked).
- Replacement sentence: none (cut, not moved).

## Item 11: memory-mapped pipeline and Figure 3.1, one sentence; figure to AF
- Moved unchanged to additional_files_body.tex, new item at the end of the Chapter 3 section (`app:add:robustness`), just before `\section{The flaws of the similarity assumption}`:
  `\addfile{Memory-mapped data files of the QM9 pipeline}` with `\label{app:add:mmap}`.
  It holds the AI-written paragraph "Rather than parse a CSV or Parquet file ..." (with its two TODO comments, including the 32/16 bytes vs 256/2,048 bytes figure check) and the figure float, caption and `\label{fig:fig_mmap}` unchanged.
- Labels moved: `fig:fig_mmap` (now in the Additional files). New label: `app:add:mmap`.
- References repointed: the only reference to `fig:fig_mmap` was inside the moved paragraph, so it now resolves inside the Additional files. No reference left in thesis.tex.
- Left in place in thesis.tex: the two following AI paragraphs (how the Rust program draws noise per split; what each record and run writes, and the halting checks). They are about noise injection, not the memory-mapped files.
- Replacement sentence in thesis.tex (QM9 data pipeline subsection), with `% TODO: manual review` above it:
  "Python and the Rust program exchange the data through memory-mapped binary files on disk, one record per molecule (Additional file~\ref{app:add:mmap})."
- The new reference resolves only after additional_files.tex is rebuilt (its .aux feeds the thesis through xr-hyper).
