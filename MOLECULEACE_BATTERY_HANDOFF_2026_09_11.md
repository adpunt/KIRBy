# Run one MoleculeACE target through the similarity battery

Written 2026-09-11. Everything below was read from code or measured on this machine
on that date. Nothing here is carried from a chat. Treat it as the whole brief.

Branch: `similarity-metrics-study`. Repository: `/Users/apunt/repos/KIRBy`.

---

## 1. What this is

`tests/experiments/similarity_metrics/run_battery_moleculeace.py` scores similarity
metrics on MoleculeACE targets. A target is one ChEMBL assay, shipped as one CSV inside
the installed `MoleculeACE` package. `tests/experiments/similarity_metrics/build_moleculeace.py`
turns each CSV into one `sim_<TARGET>.npz` file. The battery reads those npz files.

Per target the battery keeps whole Bemis-Murcko scaffold groups until it reaches 2000
molecules, then cuts 5 folds so a scaffold group stays inside one fold. On each held-out
fold it builds 47 distance matrices: 7 bit fingerprints times 5 coefficients is 35,
2 count fingerprints times 5 coefficients is 10, RDKit descriptors times 2 coefficients
is 2. It adds 2 more per cached ChemBERTa or MoLFormer file, and 2 more from triplet
encoders it trains on the training fold. Each distance matrix is scored on 9 evaluations,
and on 9 noise settings at 3 seeds each.

The headline number is `knn_rho`: predicting each molecule's activity from its five
nearest neighbours, scored by Spearman rho against measured activity.

## 2. What is on disk right now

**No MoleculeACE data exists on this machine.** Verified this session:

- `import MoleculeACE` raises `ModuleNotFoundError`. It is not in `pip list`.
- No file named `sim_CHEMBL*.npz` anywhere under the repository, `/Volumes/seagate`,
  or `~/.cache`.
- `/Volumes/seagate/kirby_cache` has no `similarity_metrics` directory at all.
- No `moleculeace_rows.json` and no `moleculeace_summary.json` anywhere.

Two empty directories do exist: `/Users/apunt/repos/KIRBy/similarity_metrics/sim_data`
dated 19 July, and `/Users/apunt/repos/KIRBy/tests/experiments/similarity_metrics/similarity_metrics/sim_data`
dated 16 July. The builder creates its output directory at import time, before it reads
anything. So those two directories say the builder was imported twice with the cache
variable unset, and wrote no file either time.

The battery script was last touched on 2026-09-11, the builder on 2026-07-11. Neither has
written a result.

## 3. Two faults to expect, both verified this session

**The battery cannot be run as a script from the repository root.** Inside `run_target`,
for each of `chemberta` and `molformer`, the script imports `emb_cache_name` under the
absolute name `experiments.similarity_metrics.learned_embeddings`. That name only
resolves when the `tests` directory is on the import path. Running
`python tests/experiments/similarity_metrics/run_battery_moleculeace.py` puts the
script's own directory on the path, not `tests`, so the import raises
`ModuleNotFoundError: No module named 'experiments'`. The import sits outside any
`try`, and it runs for every target. `main` catches the exception per target, prints
`FAIL`, and moves on — so every target fails and nothing is scored.

Setting `PYTHONPATH=/Users/apunt/repos/KIRBy/tests` fixes it without touching the file.
Confirmed both ways this session: without it the import fails, with it the import
succeeds. `setup.sh` sets no `PYTHONPATH`, and
`tests/slurm_scripts/submit_similarity_metrics.sh` launches the battery from the
repository root, so the cluster job hits the same fault.

**A result file exists even when nothing was scored.** `main` writes
`moleculeace_rows.json` and `moleculeace_summary.json` before it prints anything. Run
against an empty data directory it wrote a 2-byte `moleculeace_rows.json` holding `[]`
and a summary holding `n_targets: 0`, then raised `KeyError: 'ecfp4:tanimoto'` on the
last print. So do not report success because the two files appeared. Open them.

## 4. What done means

Done is all five of these:

1. One MoleculeACE target built into a `sim_<TARGET>.npz` from the installed package.
2. The battery run over that one target, to completion, on that real data.
3. `moleculeace_summary.json` on disk with `n_targets` equal to 1 and `n_metrics` equal
   to 49: 47 classical metrics plus the 2 triplet encoders.
4. A measured wall time in seconds for that one target, read off a clock, not estimated.
5. The arithmetic for 30 targets written out: seconds per target times 30, converted to
   hours, with the target's molecule count named beside it.

Done is not "the changes are applied" and not "it imports".

## 5. Before running: three checks

**Data.** `python -c "import MoleculeACE, os; print(os.path.dirname(MoleculeACE.__file__))"`.
If it raises, install it: `pip install MoleculeACE`. The builder finds the CSVs by
globbing `benchmark_data/*.csv` under the package directory, and falls back to
`CHEMBL*.csv` anywhere under it. If neither glob hits, the builder prints
`found 0 MoleculeACE CSVs` and exits having written nothing — that is the signal that
the package layout is not what the builder expects, and it is a real obstacle, not a
thing to work around silently.

**Loader.** Print the number of CSVs found before building anything:

```
cd /Users/apunt/repos/KIRBy
KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache KIRBY_LGBM_THREADS=1 python - <<'PY'
import sys; sys.path.insert(0, 'tests/experiments/similarity_metrics')
import build_moleculeace as B
csvs = B.find_csvs()
print(len(csvs), 'CSVs')
for p in csvs[:3]: print(' ', p)
PY
```

The builder drops any target with fewer than 100 molecules, so the number of npz files
it writes can be below the number of CSVs. Report both numbers.

**Output directory.** The builder writes to
`$KIRBY_CACHE_DIR/similarity_metrics/sim_data` and creates it. The battery reads from
the same path. With `KIRBY_CACHE_DIR` unset both become `./similarity_metrics/sim_data`
relative to wherever you are standing — that is how the two empty directories in
section 2 got made. Always set it.

## 6. The commands

The battery has no per-target flag. `main` globs every `sim_*.npz` in its data
directory and runs all of them. One target therefore means a data directory holding
exactly one npz file. Build into a scratch cache directory of its own:

```
cd /Users/apunt/repos/KIRBy
export ONE=/Volumes/seagate/kirby_cache_one_target
mkdir -p $ONE/similarity_metrics/sim_data

# build exactly one target
KIRBY_CACHE_DIR=$ONE KIRBY_LGBM_THREADS=1 python - <<'PY'
import sys; sys.path.insert(0, 'tests/experiments/similarity_metrics')
import build_moleculeace as B
csvs = B.find_csvs()
print(len(csvs), 'CSVs found')
B.build(csvs[0])
PY

ls -la $ONE/similarity_metrics/sim_data
```

`build` prints the molecule count and the scaffold count for the target it wrote. Record
both. Then run the battery over that one file, with the import path fixed and a clock on
it:

```
cd /Users/apunt/repos/KIRBy
KIRBY_CACHE_DIR=$ONE \
KIRBY_LGBM_THREADS=1 \
PYTHONPATH=/Users/apunt/repos/KIRBy/tests \
  /usr/bin/time -p python -u tests/experiments/similarity_metrics/run_battery_moleculeace.py \
  2>&1 | tee /tmp/moleculeace_one_target.log
```

`python -u` matters. The per-fold progress lines flush, the opening line does not, and a
buffered run that is killed leaves an empty log.

No GPU is needed. The ChemBERTa and MoLFormer arms are skipped when no cached `.npy`
vectors are present, and none are. The triplet encoders are pure torch on CPU, and torch
2.2.2 is installed here.

## 7. The battery does run. One dataset was taken end to end on 2026-09-11

No MoleculeACE target has ever been run. But the battery was run this session on a
stand-in with the same file schema, and it finished and wrote both result files. So the
code works once the import path in section 3 is fixed. That is the one thing you do not
have to find out again.

The stand-in was `tests/experiments/similarity_metrics/sim_herg_ki.npz`: 1415 molecules,
751 scaffold groups. It carries the identical set of arrays the MoleculeACE builder
writes — `build_local_datasets.py` imports `featurize`, `scaffolds` and `tanimoto`
straight from `build_moleculeace.py`, so the two builders cannot drift apart.

Measured, single process, no GPU, on this laptop:

| quantity | measured |
|---|---|
| molecules in the dataset | 1415 |
| held-out fold size | about 283 molecules |
| metrics scored per fold | 49 (47 classical, 2 triplet, 0 transformer) |
| one fold, start to finish | 450 to 500 seconds |
| the whole dataset, 5 folds | 2580 seconds, about 43 minutes |
| `moleculeace_rows.json` written | 226 kB |
| `moleculeace_summary.json` written | 6.5 kB |

At 43 minutes each, 30 targets in one process is about 21.5 hours. That is the arithmetic
to replace with your own: seconds for one real MoleculeACE target, times 30, in hours.
The 47 classical metrics do not change with the target, so the number of metrics is fixed
at 49 unless cached transformer vectors are present. A MoleculeACE target can be larger
than 1415 molecules, and the battery caps it at 2000, so expect a larger number than 43
minutes rather than a smaller one.

The stand-in's own ranking is a stand-in's, not a result. Do not carry any of its numbers
into a document. It is quoted here only so you know what a finished run looks like.

## 8. What to report back

- The number of MoleculeACE CSVs found, and the molecule count and scaffold count of
  the one target you built.
- The wall time in seconds for that one target, from `/usr/bin/time`.
- The contents of `moleculeace_summary.json`: `n_targets`, `n_metrics`, and the top
  five entries of `meanrank_knn` with their rank positions. A rank position of 1 is the
  best of the metrics scored.
- With one target the paired Wilcoxon is skipped — it needs at least 6 targets — so
  `wilcoxon_vs_ecfp4_tanimoto_on_knn` will be empty. Say so rather than reporting it as
  a failure.
- Seconds per target times 30, in hours.
- Anything you had to change to make it run, file by file, with what it was and what it
  became. If you had to edit a file, say which, and say it plainly.

## 9. Not in scope

- Do not run all 30 targets.
- Do not submit anything to the cluster. `tests/slurm_scripts/submit_similarity_metrics.sh`
  exists; leave it alone.
- Do not drop a target, a coefficient, a fingerprint or a noise setting to save time.
  Compute is free and a cut needs a reason that is degenerate, broken, or undefined.
- Do not edit `EXPERIMENTS_FINALIZED.md`. This experiment stays an open issue until the
  one-target numbers are in.
- If the battery cannot be made to run on one real target, stop and report the obstacle.
  A wrong edit is worse than a reported obstacle.
