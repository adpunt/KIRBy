# Thesis appendix items

This folder holds the tables and figures that go in the thesis appendices, one file per table or
figure. It was set up on 2026-09-27. Only the noise-robustness appendix (Appendix B, for
Chapter 3) uses it so far. The KIRBy appendices can get their own subfolder with the same rules
(for example `kirby/`) when those chapters are written.

## How it relates to the paper

The NoiseInject paper keeps its Additional files in `qsar_qm_models/supplementary/` (rules in
that folder's README). The thesis appendix starts from the same material, but it is a separate
copy:

- An item is copied from the paper **once**, by `seed_from_paper.py`. After that, editing the
  thesis copy never changes the paper, and editing the paper never changes the thesis.
- The thesis never reads the paper folder when it compiles. Everything it needs is in here.
- The name stays the same on both sides. So `pdv_descriptors` in the thesis is the copy of
  `pdv_descriptors` in the paper, and a search for the name finds both.

If the paper version changes and you want that change in the thesis, make it in the thesis copy
by hand, or delete the thesis copy and its index row and seed it again. The script never
overwrites a thesis copy, so thesis edits are only lost on purpose.

## What is here

| Path | What it is |
|---|---|
| `seed_from_paper.py` | Copies a paper item into `noise/`, renames its label and adds an index row |
| `noise/INDEX.md` | One row per item: name, title, where thesis.tex places it, status, date seeded |
| `noise/items/<name>.tex` | One table or figure, as a LaTeX fragment. Edit freely |
| `noise/figures/` | The PNGs those items include, copied at seeding |

## The rules

1. **An item's name is permanent**, lower case with underscores, and says what the item shows.
   It is the file name and gives two labels: `app:noise:<name>` on the table or figure itself,
   and `app:noise:sec:<name>` on the appendix section that holds it. Cite the section as
   `Appendix~\ref{app:noise:sec:<name>}` or the float as `Table~\ref{app:noise:<name>}`, never
   by a typed number.
2. **No numbers inside an item.** LaTeX numbers the appendix tables and figures. A caption never
   says "Additional file 5" or "Table S3".
3. **thesis.tex places each item** in Appendix B with
   `\section{<title>}\label{app:noise:sec:<name>}` then `\input{thesis_appendix/noise/items/<name>}`.
   Record the section in the index's "Thesis section" column.
4. **Keep the status column true.** It starts as the paper's status on the day of seeding
   (`current` or `stale`). `stale` means the content comes from the earlier version of the study
   (NDS, six noise strategies, mol2vec). Change it when you replace the content.
5. **An item the thesis needs that the paper does not have** goes straight into `noise/items/`
   with a row in `noise/INDEX.md`. Write "thesis only" in the Seeded column.

## How to

```
python3 thesis_appendix/seed_from_paper.py --list        # the paper's items, and which the thesis has
python3 thesis_appendix/seed_from_paper.py <name> ...    # copy one or more
```
The script looks for the paper repository at `../qsar_qm_models`. Set `QSAR_QM_MODELS` to use
another path.

`main.tex` loads `float` (the items use `[H]`) and puts `thesis_appendix/noise/figures/` on the
`\graphicspath`. If the thesis is compiled in Overleaf, upload the `thesis_appendix/` folder
with the same layout.
