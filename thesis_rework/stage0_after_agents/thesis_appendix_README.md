# Thesis appendix items

This folder holds the tables and figures that go in the thesis's "Additional files" appendix
(`\label{app:additional}`), one file per table or figure. The appendix has one section per chapter,
and each folder here serves one of them:

| Folder | Chapter | Section label | Item labels |
|---|---|---|---|
| `noise/` | Chapter 3, noise robustness | `app:add:robustness` | `app:noise:sec:<name>` |
| `flaws/` | Chapter 4, the flaws of the similarity assumption | `app:add:flaws` | `app:flaws:sec:<name>` |
| `method/` | Chapter 5, building adaptive hybrid representations | `app:add:method` (also `app:searchspace`) | `app:method:sec:<name>` |
| `similarity_noise/` | Chapter 6, similarity and label noise | `app:add:similarity` | `app:similarity_noise:sec:<name>` |

`noise/` keeps its name so its labels hold. The other three folders are created when their first
item is written; until then thesis.tex lists each planned item as a comment in its section.

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
| `<group>/INDEX.md`, `<group>/items/`, `<group>/figures/` | The same layout for `flaws/`, `method/` and `similarity_noise/`. These items are thesis only: nothing seeds them |

## The rules

1. **An item's name is permanent**, lower case with underscores, and says what the item shows.
   It is the file name and gives two labels: `app:<group>:<name>` on the table or figure itself,
   and `app:<group>:sec:<name>` on the Additional file entry that holds it. Cite the entry as
   `Additional file~\ref{app:<group>:sec:<name>}` or the float as `Table~\ref{app:<group>:<name>}`,
   never by a typed number.
2. **No numbers inside an item.** LaTeX numbers the Additional files and the tables and figures.
   A caption never says "Additional file 5" or "Table S3".
3. **thesis.tex places each item** as one numbered entry in its chapter's section, not as a
   `\section`: `\addfile{<title>}`, then `\label{app:<group>:sec:<name>}`, one sentence saying what
   the item shows, then `\input{thesis_appendix/<group>/items/<name>}`. Several items may share one
   entry (for example the three tables of the paper's Additional file 1); each keeps its own label
   on that entry. `\addfile` and its counter are defined at the top of the appendix in thesis.tex.
   Record the entry in the index's "Thesis section" column.
4. **Keep the status column true.** It starts as the paper's status on the day of seeding
   (`current` or `stale`). `stale` means the content comes from the earlier version of the study
   (NDS, six noise strategies, mol2vec). Change it when you replace the content.
5. **An item the thesis needs that the paper does not have** goes straight into its group's
   `items/` folder with a row in that group's `INDEX.md`. Write "thesis only" in the Seeded column.
   Every item for Chapters 4 to 6 is thesis only.

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
