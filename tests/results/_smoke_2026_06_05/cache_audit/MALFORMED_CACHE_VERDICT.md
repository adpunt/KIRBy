# Malformed-Cache Verdict — which on-disk caches MUST be purged before the rerun

Date: 2026-06-05. READ-ONLY audit. Every verdict grounded in code / commit / file:line below.

## How the cache decides "stale" (the mechanism that lets corruption hide)

`src/kirby/find_best_hybrid.py:182-251` (`_load_cache_with_meta`) validates a stored
`_cache_meta` against the caller's expected meta, key-by-key. The decisive fact:

- `ignore_keys = ('code_timestamp', 'rep_hashes', 'aug_hashes', 'rep_hash')`
  (find_best_hybrid.py:198-199). The **representation value-hashes are NOT checked.**
  (Deliberate: the 2026-06-04 CACHE-1 anti-churn fix — neural reps aren't bit-reproducible.)
- What IS checked (find_best_hybrid.py:343-422, `_make_search_meta` / `_make_baseline_meta`):
  `rep_names`, `rep_shapes`/`rep_shape`, `n_samples`, `labels_hash`, `groups_hash`,
  `budgets`, `tier`, `metric_name`, `task_type`, `n_folds`, `seed`, strategy/model config.

Therefore the purge rule is exact:

| Fix kind | Detected by cache? | Action |
|---|---|---|
| Changes a rep's VALUES, keeps the same shape/row-count | NO (value-hash ignored) | **MUST PURGE** |
| Changes rep dimensionality (shape) | YES — `rep_shapes` | auto-invalidates |
| Changes label values / row count / n_samples | YES — `labels_hash` + `n_samples` | auto-invalidates |
| Changes metric / folds / seed / tier / strategy | YES — those keys | auto-invalidates |
| Experiment-code-only fix (doesn't touch cached rep/label values) | n/a | no action |

**Cache write date (the cut-line):** the complexity/similarity result+cache batch was
synced from ARC on **2026-06-02** (memory project_complexity_similarity_results.md:3,11).
A value-changing rep/label fix only corrupts the cache if it landed **after** the cache was
written. Anything that landed **before** 2026-06-02 was already in force when the cache was
computed → the cache is correct → do NOT purge on that basis.

---

## Verdict table

| Fix | Commit (date) | Shape-preserving? | Affected datasets & reps | Verdict | Evidence |
|---|---|---|---|---|---|
| **Graphinity mutation-position regex** (chain group `[A-Za-z0-9]+` → `[A-Za-z]+`; greedy digit-steal corrupted every mutation position, e.g. `YH102A` parsed chain=`H10`/pos=2 instead of chain=`H`/pos=102) | **3cadb95** (2026-06-04, AFTER cache) | **YES** — fix changes the parsed mutation POSITION used to build mutation-aware reps; same row count, same labels, same rep dimensionality | graphinity_foldx, graphinity_rosetta, graphinity_augmented — all mutation-aware/DDG reps | **PURGE** | graphinity.py `_parse_complex_column` regex (3cadb95 diff lines ~86-97); landed after 2026-06-02; commit msg: "Graphinity mutation-position corruption (would poison the DDG re-run)"; dedicated purge tool exists: `tests/purge_graphinity_cache.py` |
| **REP-1/2/3 row misalignment** in MACCS / PDV / mol2vec (invalid SMILES did `continue` → dropped a row → all later rows shifted, matrix had fewer rows than `smiles_list`) | bug introduced 6ec6e81 (2026-03-10, `raise`→`continue`); **FIXED 3236092** (2026-03-11) by appending a zero-vector then `continue` (row alignment restored) | n/a — **fix predates the cache** | herg, herg_ki, caco2, logd, qm9, adme_hlm/rlm/mdr1 (maccs/pdv/mol2vec); reaction (diff_maccs/diff_mol2vec/reactant_mol2vec); DTA kiba/bindingdb_kd (drug_pdv) | **no action** | molecular.py:826-833 (maccs), :856-872 (pdv), :1516-1535 (mol2vec) all now append `np.zeros(...)` before `continue`; zero-fill introduced 2026-03-11 (3236092), ~3 months BEFORE the 2026-06-02 cache → cache was written WITH the fix. Note: the buggy intermediate (drop-row) was ALSO shape-CHANGING in row count anyway, so even a stale entry would mismatch on `n_samples`/`rep_shapes`. |
| **PDV z-score normalisation** (raw RDKit descriptors → `StandardScaler().fit_transform`) | **61e5a87** (2026-04-01, BEFORE cache) | YES (200D in, 200D out) | every dataset using `pdv`/`drug_pdv`/`reactant_pdv` | **no action** | molecular.py:867-870; landed 2026-04-01, two months before the 2026-06-02 cache → cache reflects the normalised values. (Side note: `fit_transform` is fit on all rows incl. test folds = a leakage concern, but NOT a post-cache value change, so no purge.) |
| **ginkgo_dev fold-column row filter** (added `& df[fold_col].notna()` — drops rows with null fold) | 3cadb95 (2026-06-04, AFTER cache) | **NO** — drops rows → changes row count, n_samples, label set | ginkgo_acsins/colloidal/heparin/hic/pr_cho/titer | **auto-invalidates** (but harmless to purge) | ginkgo_dev.py `load_ginkgo_dataset` (3cadb95 diff lines ~103-118); fewer rows ⇒ `n_samples` + `labels_hash` mismatch ⇒ cache recomputes on its own |
| **humanness heavy-chain FASTA selection** | 3cadb95 (2026-06-04) | n/a — adds warnings only; the returned file (`candidates[0]`) is unchanged | humanness_species, humanness_therapeutic | **no action** | humanness.py `_find_heavy_fasta` (3cadb95 diff lines ~94-129): only `print(WARNING ...)` added; return value identical |
| **S2ALM antibody embedding** | ea07762 (2026-06-04) | n/a — still `raise NotImplementedError` (stub, never produced cached data) | none | **no action** | antibody.py `embed_s2alm` (ea07762 diff) |
| **molecular.py additions** (create_padel, new SNS-factory helper, Boltz-2 molecular scaffold) | 3cadb95 / ea07762 (AFTER cache) | n/a — NEW rep names, not edits to existing reps | n/a | **no action** | molecular.py:915-954 (padel, raises if no Java), :4006-4054 (boltz2 scaffold, raises NotImplementedError); new `rep_name` ⇒ cache miss ⇒ fresh compute, never collides with existing entries |
| **antibody.py additions** (LoRA factory, new ANTIBODY_REPS) / **genomic.py additions** (autoencoder, UCE) / **dataset additions** (metz, CPTAC graceful-skip loader, ISS-78 docstring caveats) | ea07762 (AFTER cache) | n/a — additive / docstring-only | n/a | **no action** | ea07762 diffs: genomic.py:2379,4456 (new factories); dta.py `_DATASETS['metz']`; tcga.py `_download_cptac_proteomics` (raises FileNotFoundError, "does NOT cache junk"); flab/abagym/ddg_combined = ISS-78 write-up caveats only, no value change |

---

## MUST-PURGE datasets (proven post-cache, value-changing, shape-preserving)

1. **graphinity_foldx**
2. **graphinity_rosetta**
3. **graphinity_augmented**

Reason: the mutation-position regex fix (commit 3cadb95, 2026-06-04 — AFTER the 2026-06-02
cache) silently corrected the parsed mutation position while keeping row count, labels, and
rep dimensionality identical. The value-hash is in `ignore_keys`, so a stale entry would load
as a hit and silently reuse corrupt mutation-aware reps. Use the existing dry-run-first tool:
`tests/purge_graphinity_cache.py --delete` (run on the server; it exists precisely because
"CACHE-1 ignores rep-hash, so the Graphinity mutation-fix needs a manual purge").

Caveat / unknown: I could not confirm from local code whether any Graphinity job had
actually been run and cached as of 2026-06-02 (results sync was complexity/similarity JSONs;
no local server cache is visible). If Graphinity was never run, the purge is a no-op safety
pre-clean — still correct to run, never harmful.

## MUST-NOT-TOUCH (no evidence of corruption — purging these destroys valid results)

- All **molecular** caches using maccs / pdv / mol2vec: **herg, herg_ki, caco2, logd, qm9,
  adme_hlm, adme_rlm, adme_mdr1** — the REP-1/2/3 row-misalignment fix (2026-03-11) AND the
  PDV z-score fix (2026-04-01) both predate the 2026-06-02 cache; the cache was computed with
  them already in place.
- All **reaction** caches (buchwald_hartwig, bh_doyle_plates) using diff_maccs / diff_mol2vec
  / reactant_mol2vec / reactant_pdv — same reasoning.
- All **DTA** caches (kiba, bindingdb_kd) using drug_pdv / drug_mordred — same reasoning.
- **ginkgo_*** caches — the only post-cache change (fold-column filter) auto-invalidates via
  `n_samples` + `labels_hash`; no manual purge required (deleting is harmless but unnecessary).
- **humanness_species / humanness_therapeutic** — post-cache change is warning-only, return
  value unchanged.
- Everything touched only by ea07762/3cadb95 as ADDITIONS (metz, CPTAC, padel, boltz2 scaffold,
  S2ALM stub, new antibody/genomic factories) — new names ⇒ cache miss ⇒ fresh compute; they
  cannot serve a stale value for any existing dataset.

## Scope note / residual uncertainty

This audit covers `src/kirby/representations/` and `src/kirby/datasets/` value-changing fixes
relative to the 2026-06-02 cache cut-line. It does NOT re-litigate experiment-code fixes
(c0x/s/n/fed) — those change which search/metric is computed and are caught by the search
meta keys (tier/budgets/metric/folds/strategy), so they auto-invalidate rather than silently
reuse. The single genuine silent-corruption case post-cache is Graphinity.
