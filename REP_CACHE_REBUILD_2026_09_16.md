# Rebuilding `pdv` and `chemberta` on the server

Measured 2026-09-16 by rebuilding cached matrices with the live builder and comparing them
row by row. Every number here came from that comparison. Anything not measured says so.

## What is wrong

Two builders were changed on 2026-08-30. Every affected cached matrix was written
2026-05-08. Nothing invalidated them, because `_identity.json` hashes the input strings and
carries nothing about the code.

**`create_pdv` stopped standardising.** Commit `15ef03e`, "PDV returns raw descriptors;
scaling is the caller's job, on training rows only". The cached `pdv` matrices are still
whole-dataset z-scores: every non-constant feature has mean 0 and standard deviation 1 across
all rows, and standardising a fresh build reproduces the cached file to 0.00078 on herg_ki,
which is float32 rounding. So a held-out molecule's own value set the scale of the training
features. Affected: `rep_cache/herg_ki/pdv.npy`, `rep_cache/caco2/pdv.npy`,
`rep_cache/logd/pdv.npy`. `pdv` is in the pruned pool of herg_ki and caco2; on logd it is
cached but not in the pool.

**`create_chemberta` changed which reader it loads.** Commit `e7a23cb`, "Read SMILES the way
ChemBERTa was actually pretrained". The reader now drops `@`, `[`, `]`, `H` and the `l` of
`Cl` — checked directly, `tok('@')` returns an empty list of ids. Two stereoisomers therefore
get one token sequence and one vector. The cached matrices give them different vectors.

| | the cache says | the builder says today | rows that differ |
|---|---|---|---|
| herg_ki, 1,412 molecules | 0 share a vector | **130 share a vector, 27 groups** | 853 |
| logd, 5,028 molecules | 0 share a vector | **709 share a vector, 336 groups** | 2,730 |
| qm9, 9,978 molecules | 0 | 0 | 0 — already correct |

QM9 is unaffected because its SMILES carry no stereochemistry to drop.

## What is NOT wrong

**The second feature store is already current.** `$KIRBY_CACHE_DIR/reps/<hash of the molecule
list>/` holds a separate copy, read by `find_best_hybrid_molecular.get_representations` rather
than by `load_data`. Checked: its logd `chemberta` has 709 tied rows and its logd `pdv` has
raw column means. None of its 52 matrices is a z-score. Nothing to do there.

So the two stores currently disagree with each other on the same molecules.

**Every other cheap builder matches.** `opig_shared_vectors_census.py --check-freshness 150`
across caco2, herg_ki and logd rebuilt `ecfp4`, `maccs`, `avalon`, `atom_pair`, `top_torsion`
and `rdkfp` and found no difference. Only `pdv` failed. The GPU-built embeddings were not
checked — that needs the server.

## The rebuild

Run on the machine where `$KIRBY_CACHE_DIR` lives. `. ./setup.sh` first, or every path
resolves to a different cache than the jobs use.

**1. Look at what would go.** Dry-run is the default.

```
python tests/purge_rep_cache.py herg_ki caco2 logd --reps pdv,chemberta
```

`--reps` is new (2026-09-16). Without it the script removes a dataset's whole pool, which
would throw away GPU-built embeddings that are still correct. It keeps `_identity.json`,
because the molecules have not changed.

Expect five files, 16.0 MB: `pdv` on all three, `chemberta` on herg_ki and logd. caco2 has no
cached `chemberta`.

**2. Delete the features AND the scores computed from them.**

```
python tests/purge_rep_cache.py herg_ki caco2 logd --reps pdv,chemberta --delete
python tests/purge_rep_cache.py herg_ki caco2 logd --all-caches --delete
```

The second line is not optional. `purge_rep_cache.py` says why in its own comments: when only
the code that builds a feature changes, the activity values stay byte-identical, so a stored
score is filed under the same key and is read straight back and reported. `result_db` and
`fold_scores` are the live ones. ⛔ The second line purges those datasets' scores for **every**
representation, not only the two — there is no per-representation key in them.

**3. Rebuild.** `precompute_reps.py` builds only what is absent from disk, so step 2 has to
come first.

Both builders run on CPU, and neither needs a GPU. They are nothing like each other in cost.
Measured on a laptop, one core:

| | molecules | seconds | seconds per molecule |
|---|---|---|---|
| `chemberta` herg_ki | 1,412 | 12 | 0.008 |
| `chemberta` logd | 5,028 | 22 | 0.004 |
| `pdv` caco2 | 2,158 | 357 | 0.17 |
| `pdv` herg_ki | 1,412 | **817** | **0.58** |

`pdv` is 70 times slower per molecule than `chemberta` on herg_ki, and herg_ki is 3.5 times
slower per molecule than caco2 — the rate tracks how large the molecules are, not how many
there are. logd's 5,028 molecules were not timed. At caco2's rate that is about 14 minutes and
at herg_ki's about 49 minutes; neither is measured.

So `chemberta` on all three is under a minute and can run anywhere. `pdv` on all three is
somewhere between 30 and 80 minutes on one core, which is a short SLURM job rather than a
login-node command.

`run_precompute_reps.sh` asks for `--gres=gpu:1`, 64G and 47:59:00 in its own header, and that
header says those are not measured. Nothing in this rebuild uses a GPU.

Directly, if you are on a node you can hold for an hour:

```
python tests/precompute_reps.py herg_ki caco2 logd --reps pdv,chemberta
```

Through SLURM, one dataset per job, which is what the `pdv` half deserves:

```
bash tests/slurm_scripts/submit_precompute_reps.sh one herg_ki --reps pdv,chemberta
```

`--reps` forwarding through the `one` mode is also new (2026-09-16). Before that the wrapper
dropped it and the job rebuilt nothing, because everything it wanted was already on disk.

**4. Check it took.**

```
KIRBY_CACHE_DIR=... python tests/experiments/opig_shared_vectors_census.py \
    --datasets herg_ki caco2 logd --no-pairwise --check-freshness 150
```

Nothing should print `CACHED MATRIX NO LONGER MATCHES THE CODE`. herg_ki `chemberta` should
come back 130 of 1,412 molecules sharing a row, and logd `chemberta` 709 of 5,028.

## Still open, and not fixed here

**`create_rppa` and `create_mirna_top200` still standardise on the whole sample set.**
`src/kirby/representations/genomic.py:4018` and `:4039` both call `StandardScaler().fit_transform`
on the full matrix. Their cached matrices for tcga_brca_pam50 are z-scores because the code
puts them there, not because the cache is old. Both are in that dataset's pruned pool.
Rebuilding changes nothing. This needs the same change `create_pdv` got in `15ef03e`: return
raw and let the caller scale on its own training rows.

**The GPU-built embeddings were never checked against their builders.** Nine commits touched
`src/kirby/representations/` since 2026-05-08. Two of them changed feature values outright and
are handled above. A third, `9665eab` on 2026-08-23, changed `graph_kernel` to handle
bond-less molecules instead of raising, which changes values for inorganic salts — unchecked.
`--check-freshness` only covers the eight builders that run on a laptop.
