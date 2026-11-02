# Handoff: fill every cell of Tables 4.1 and 4.2 (constant and near-constant features)

## Goal

The thesis has two tables of constant and near-constant feature counts:
- `tab:constant_features` (Table 4.1)
- `tab:constant_features_b` (Table 4.2)

Both are in Chapter 4, `thesis_rework/stage1_takeaways_bullets/thesis.tex`. Each has 17 representations as rows, and one dataset per column. Many cells show "--". Every cell must hold a real count before the visible TODO above each table can be removed.

The rules:
- Do not remove rows.
- Do not remove datasets.
- Do not write a number you did not compute in this session.

## Why the cells are empty (checked on 2026-09-30)

Most of the empty cells were already counted. The table script hid them.

- `scratchpad/thesis_code_todo/constant/count_constant.py` counted every saved `.npy` file in `/Volumes/seagate/kirby_cache/rep_cache/<dataset>/`. It wrote the counts to `constant/constant_counts.json`.
- `scratchpad/thesis_code_todo/constant/make_outputs.py` then printed "--" for any representation that is not in that dataset's `general_pool` (from `DATASET_REPS` in `tests/experiments/complexity/complexity_helpers.py`). It did this even when a count existed in the JSON.

### Counted in the JSON but hidden by the script (36 cells)

| Dataset | Hidden representations |
|---|---|
| herg_ki | top_torsion, graphmvp, molformer, chemberta, selformer, mol2vec |
| logd | atom_pair, graph_kernel, pdv, chemberta, selformer, mol2vec |
| qm9 | top_torsion |
| adme_mdr1 | top_torsion, graphmvp, molformer, chembert, chemberta, selformer, mol2vec |
| adme_hlm | ecfp4, top_torsion, graphmvp, chemberta, selformer, mol2vec |
| adme_rlm | top_torsion, graphmvp, chembert, chemberta, selformer, mol2vec |
| herg (Lhasa hERG blockers, 7,772 molecules) | pdv, graphmvp, smited, chembert, chemberta, selformer |

### No saved feature file, so never counted (12 cells)

| Dataset | Missing representations |
|---|---|
| caco2 | mordred, graphmvp, chembert, chemberta, selformer, mol2vec |
| qm9 | atom_pair, maccs, graph_kernel, pdv, grover, mhg_gnn |

## Steps

1. **Check that each hidden file is current before using it.**
   - These files are outside the dataset's base representations, so the full runs never read them. They may be stale.
   - Known risks:
     - herg_ki's `chemberta` was flagged stale (see commit 6107f77 and `tests/tools/pdv_cache_audit.py --rep`).
     - `chembert` (256 features) is not `chemberta` (384 features). Check the width of each file against the representation.
     - A second cache, `reps/<sha of smiles>`, is read before `rep_cache/`.
   - Use `tests/tools/pdv_cache_audit.py` with `--rep` to compare each file against a fresh featurization of a few molecules. Record the result for each cell.
2. **Compute the 12 missing feature files for caco2 and qm9.**
   - Use the repository's own featurisers, the same code path the full runs use.
   - This is not local work. The user runs it on the server. Write the exact commands for her, in the one-line `srun ... --pty /bin/bash` format. Never ssh or scp yourself.
   - MACCS keys for QM9 are also missing. The current caption says "n/a means no saved MACCS features exist for QM9". That sentence goes once the file exists.
3. **Recount and remake the tables.**
   - Rerun `count_constant.py` on the new files. It reads only saved files.
   - Change `make_outputs.py` so a cell prints the count whenever the JSON has one, not only for representations in the dataset's base list.
   - Regenerate both tables and the figure (`figures_code_input/constant_features.pdf`), which uses the same filter.
4. **Update the text around the tables.**
   - The paragraphs before Table 4.1 quote counts: ECFP4 1,815 to 1,997 near-constant features across "the seven datasets where ECFP4 is a base representation", and the learned embeddings with "0 on every dataset where they are base representations".
   - Once every cell is filled, reword these sentences to say "on every dataset". Check each number against the new JSON.
   - Take out "-- means the representation is not in that dataset's base representations" from the caption.
5. **Remove the visible TODO** (`\todo[inline]` above each table) only when no cell shows "--" or "n/a".

## Constraints

- Do not load datasets or run featurisers locally. Counting saved `.npy` files locally with `mmap_mode='r'` is fine.
- Never ssh or scp to the server.
- Do not write verdicts into the thesis. Numbers only.
- American spelling. Say "features", never "columns".
