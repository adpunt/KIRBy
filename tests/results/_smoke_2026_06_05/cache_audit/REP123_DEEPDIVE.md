# REP-1 / REP-2 / REP-3 Deep Dive — MACCS / PDV / mol2vec invalid-molecule row misalignment

READ-ONLY audit, 2026-06-05. Every claim cites `file:line` or `commit`. No date-only reasoning.

Scope: MASTER_ISSUES_2026_06_04.md:45 ("REP-1/2/3 — invalid-molecule row misalignment
in MACCS/PDV/mol2vec silently corrupts every result using those reps"), backed by
issues_preprocessing.md C1/C2/M1 and ISSUES.md §7.

---

## 1. What the bug is, precisely

The three molecular representation builders looped over `smiles_list`, parsed each
SMILES with `_safe_mol` (= `Chem.MolFromSmiles`, `molecular.py:59-64`), and on a
parse failure did `continue` **without appending any placeholder row**. Result: the
returned matrix had `N - k` rows (k = number of unparseable SMILES) while labels,
groups, and every other rep kept all `N` rows. When indexed together inside the CV
loop, row `i` of the rep no longer corresponds to label `i` — silent misalignment of
features to targets for every fold. ECFP4 (`create_ecfp4`) did it correctly all along:
it appends `np.zeros(n_bits)` for a `None` mol and never drops a row.

Pre-fix bodies (verbatim, from `git show 3236092^:src/kirby/representations/molecular.py`):

- `create_maccs` (pre-fix lines 206-228):
  ```python
  for i, smiles in enumerate(smiles_list):
      mol = _safe_mol(smiles)
      if mol is None:
          continue                      # <-- drops the row
      fp = MACCSkeys.GenMACCSKeys(mol)
      fingerprints.append(np.array(fp))
  return np.array(fingerprints, dtype=np.float32)   # N-k rows
  ```
- `create_pdv` (pre-fix ~lines 248-256): identical pattern — `if mol is None: continue`
  then `descriptors.append(...)`, `return np.array(descriptors)` with N-k rows. Because
  `create_pdv` feeds `create_pdv_binary_mean`, `create_pdv_rank`,
  `make_pdv_binary_mean_factory`, `make_pdv_rank_factory` (`molecular.py:971,991,1161,1183`),
  the misalignment propagated to those too.
- `create_mol2vec` (pre-fix ~lines 500-512): `if mol is None: continue`, embeddings
  list ends up N-k long.

This is the canonical "malformed molecular cache entry" candidate: the on-disk rep
matrix has the wrong number of rows.

---

## 2. Is it fixed in current code? Yes — commit `3236092`

Fix commit: **`3236092` "Phase 5 SLURM scripts + all Phase 0-4 accumulated changes"
(2026-03-11 13:54 +0000)**. `git log -S 'np.zeros(167, dtype=np.float32)' -- src/kirby/representations/molecular.py`
returns only this commit; same for the PDV/mol2vec zero-fills.

Current code (HEAD = `ea07762`):
- `create_maccs` (`molecular.py:825-829`): `if mol is None: fingerprints.append(np.zeros(167, dtype=np.float32)); continue`.
- `create_pdv` (`molecular.py:856-860`): `if mol is None: descriptors.append(np.zeros(n_desc, dtype=np.float32)); continue`.
- `create_mol2vec` (`molecular.py:1516-1519`): `if mol is None: embeddings.append(np.zeros(_MOL2VEC_MODEL.wv.vector_size, dtype=np.float32)); continue`.

All three now emit a zero row per invalid molecule, so the matrix is always exactly
`len(smiles_list)` rows. The fix matches the ECFP4 pattern exactly. NOTE the two
remaining `if mol is None: continue` at `molecular.py:1065` and `:1128` are inside
`create_sns` / `_transform_sns` (SNS, not mol2vec); `_transform_sns` pre-allocates a
full `np.zeros((len(smiles_list), vec_dimension))` and writes by index `i`
(`:1124,1138`), so an invalid mol just leaves a zero row — no misalignment there
either.

---

## 3. Shape-preserving or shape-changing? — SHAPE-CHANGING (self-healing) for the molecular pipeline

The fix changes the matrix from `N-k` rows (pre-fix) to `N` rows (post-fix) whenever
any invalid SMILES is present. So **relative to a pre-fix cached `.npy`, the row count
differs** — this is shape-CHANGING, and it is caught automatically:

- Experiment path (`tests/experiments/_shared.py:2447-2452`): reps flow into
  `_finalize_load_data`, whose row-count guard (`_shared.py:2062-2075`) raises a
  `ValueError` naming the dataset and the purge command if `arr.shape[0] != len(labels)`.
  A stale N-k-row cache therefore cannot be silently reused — it errors loudly and
  forces a `tests/purge_rep_cache.py <dataset> --delete`.
- Molecular survey path (`tests/find_best_hybrid_molecular.py:608-628`): the rep `.npy`
  cache (`_rep_cache_path`, keyed only by `sha256(smiles)` + rep name, NO header,
  NO row-count guard — `:515-533`) would reload an N-k matrix, but the survey would
  then crash downstream (rep N-k vs labels N) rather than produce a wrong-but-valid
  number. So a *completed, used* survey result implies the cached matrix already had
  N rows.

Contrast with the genuinely dangerous class (CACHE-1 silent-staleness, see
`CACHE_INVALIDATION_CONTRACT.md` §4.B): a fix that keeps the SAME shape but changes
VALUES (e.g. the Graphinity mutant-sequence parser). REP-1/2/3 is NOT that class — it
moves the row count, which every guard and the validated meta (`rep_shapes`/`rep_shape`,
`n_samples`; `find_best_hybrid.py:351,407,349,413`) detect.

---

## 4. Which datasets use maccs / pdv / mol2vec

From `DATASET_REPS` (`tests/experiments/complexity/complexity_helpers.py`) and the
molecular presets (`tests/find_best_hybrid_molecular.py:146-152`):

Molecular datasets (SMILES → the three builders directly):
- **herg** (`:96`), **herg_ki** (`:118`), **caco2** (`:131`), **logd** (`:143`),
  **qm9** (`:160`), **adme_hlm** (`:171`), **adme_rlm** (`:184`), **adme_mdr1** (`:197`)
  — these reference `pdv`, `mol2vec`, and/or `maccs` (maccs usually in the
  `augmentation` slot).
- the generic **`molecular`** preset (`:991`, lists `pdv`/`maccs`/`mol2vec`).
- molecular survey presets `fast`/`fast+hf`/`full` all include `maccs` and `pdv`;
  `full` includes `mol2vec` (`find_best_hybrid_molecular.py:146-152`).

Reaction datasets (the builders run on the drug/reactant/product SMILES via reaction
wrappers `create_*_maccs`/`create_*_pdv`/`*_mol2vec`, `_shared.py:2316-2323`):
- **buchwald_hartwig** (`:619`), **bh_doyle_plates** (`:626`) — `diff_maccs`,
  `reactant_pdv`, `product_maccs`, `reactant_maccs`, `diff_mol2vec`, `reactant_mol2vec`.

DTA datasets (drug-side SMILES): **kiba** (`:656`), **bindingdb_kd** (`:675`) —
`drug_pdv`, `drug_maccs`, `drug_mol2vec`.

(The reaction/DTA wrappers call into the same fixed `create_maccs/create_pdv/create_mol2vec`
internals; they inherit the fix.)

---

## 5. Do those datasets have pre-fix cache on disk? — Cannot confirm from here; and the bug almost certainly never produced a corrupt cache

Two independent reasons a manual purge for REP-1/2/3 is very likely unnecessary:

(a) **Shape-changing ⇒ self-healing (§3).** Even if a pre-fix N-k `.npy` exists, the
`_shared.py:2062-2075` row-count guard raises and points at `purge_rep_cache.py` on the
next experiment run; the survey path crashes. There is no silent-wrong-answer path for
a row-count change. (The proof the prompt asks for — "fix landed after caches were built
AND is shape-preserving" — fails its second clause: the fix is shape-CHANGING.)

(b) **The bug almost certainly never fired**, because every molecular loader feeding
these reps pre-filters SMILES to RDKit-parseable before rep build:
   - OpenADMET (logd, caco2) and ChEMBL (herg_ki) go through `_load_openadmet_endpoint`
     → `_standardise_smiles` → `standardize_smiles` (`molecular.py:106-132`), which
     returns `Chem.MolToSmiles(canonical=True)` of an already-parsed mol and drops rows
     where parse/standardize returns None (`find_best_hybrid_molecular.py:231-235`).
   - Custom CSVs filter on `Chem.MolFromSmiles(...) is not None` at load
     (`find_best_hybrid_molecular.py:472-479`).
   - QM9 filters to RDKit-valid indices (`datasets/qm9.py:105-113`) and uses PyG
     `data.smiles`.
   - herg (FLuID) / ChEMBL use canonical SMILES (`datasets/herg.py:92,226`).
   By the time SMILES reach `create_maccs/create_pdv/create_mol2vec`, `_safe_mol`
   returns non-None for every entry → k = 0 → cached matrices have full N rows even
   under the pre-fix code. (A residual edge case: a SMILES that survives standardisation
   but fails a later re-parse is theoretically possible but not demonstrated; if it ever
   happened pre-fix the run would have crashed at row-count, not silently corrupted.)

Date context (NOT used as proof, per protocol): fix committed 2026-03-11; the first
complexity/similarity result batch synced 2026-06-02 (project memory). Any cache built
to produce those results post-dates the fix, so it was built with the zero-fill code.

I did not (and cannot) inspect the server cache directory from this read-only repo; the
above stands on code semantics, not on disk listings.

---

## VERDICT

**A manual molecular cache purge is NOT required for REP-1/2/3.**

The fix (commit `3236092`, 2026-03-11) is **shape-CHANGING** (N-k rows → N rows when
invalid SMILES are present), which is the auto-invalidating / self-healing class:
- the experiment loader's row-count guard (`_shared.py:2062-2075`) RAISES on a stale
  N-k cache and names `purge_rep_cache.py`, so corruption cannot pass silently;
- the validated search/baseline meta also catch it via `rep_shapes`/`rep_shape` +
  `n_samples` (`find_best_hybrid.py:351,407`).

Moreover the bug almost certainly never created a corrupt cache at all, because every
molecular/reaction/DTA loader pre-filters SMILES to RDKit-parseable before the rep is
built (§5b), so k = 0 in practice and the matrices were full-length even pre-fix.

This is the OPPOSITE of the CACHE-1 silent-staleness trap (same-shape value change,
e.g. Graphinity) that genuinely DOES require a hand purge. REP-1/2/3 does not.

Recommended action: none. If you want belt-and-suspenders certainty on the server,
a DRY-RUN `python tests/purge_rep_cache.py <dataset>` per molecular/reaction/DTA dataset
will list any `.npy` whose row count would mismatch — but the loader guard already does
this automatically at run time, so no pre-emptive deletion is warranted. Do NOT
manufacture a purge here.
