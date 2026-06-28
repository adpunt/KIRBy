# Cache "meta drift" root-cause report — 2026-06-28

All claims below were re-verified against HEAD code (not memory). HEAD has
`501f295` (SQLite cache-key redesign) as an ancestor (`git merge-base
--is-ancestor 501f295 HEAD` = true); only `4fa5d9a` touched
`find_best_hybrid.py` since. No server access — log-provenance claims are
flagged UNVERIFIABLE and listed as server checks.

---

## 1. Primary root cause (3 sentences)

The months-long stall was driven by SHA1 **value-hashes of non-bit-reproducible
float representation matrices** (`rep_hash` 4077 + `rep_hashes` 137 = ~85% of all
drift) being COMPARED inside the cache meta during the window 2026-04-17
(`7a17cf8`, which added the `"meta drift on"` print with `ignore_keys=('code_timestamp',)`)
through 2026-06-04 (`3cadb95`, which added `rep_hashes/aug_hashes/rep_hash/subset_hashes`
to `ignore_keys`): neural/transformer reps and threaded-BLAS/unseeded-PCA float
reductions are not bit-identical run-to-run, so every neural-rep cache rejected on
load and cold-rebuilt (days/job). **This is already fixed in the primary load
path** — `find_best_hybrid.py:198-200` now ignores those four value-hashes, none of
the four live `_load_cache_with_meta` call sites override that default, and the SQLite
redesign moved model/budget/evaluator/rep_names/pruning to per-row config lookups, so
in HEAD only `groups_hash`, `labels_hash`, and `aug_names` can still emit a drift line.
The genuinely-live remaining exposure is narrow: (a) `groups_hash`/`labels_hash` sit in
SQLite `IDENTITY_COLS` with NO row-rescue, so they now rest entirely on UPSTREAM
group/label-assignment determinism in `load_data` (the hash itself is deterministic);
(b) a missed-propagation **regression** in the checkpoint validator
(`find_best_hybrid.py:3408`) still rejects resume checkpoints on the very value-hashes the
load layer was fixed to ignore; (c) the `a274cf4` thread un-pin reintroduces rep-VALUE
non-determinism that misses the separate filename-keyed rep-VALUE cache (the 800× ginkgo
load-time swings) — distinct from the meta layer.

---

## 2. Per-field table

Drift string is emitted at exactly ONE place: `find_best_hybrid.py:247`, inside
`_load_cache_with_meta`, only for keys NOT in `ignore_keys` (loop `:241-244`).
"Live in HEAD?" = whether the field is a key in one of the four loaded metas
(folds `:447`, ft_embedding `:648`, hybrid_eval `:585`, baseline `:425`) AND not
in `ignore_keys`.

| field | count | category | live in HEAD? | mechanism | minimal fix | risk | verified |
|---|---|---|---|---|---|---|---|
| `rep_hash` | 4077 | regression (now neutralized) | NO (in ignore_keys `:199-200`; the only carrier, baseline meta `:430`, ignores it) | SHA1 of non-bit-reproducible float matrices (neural inference, BLAS reduction order, unseeded randomized PCA). 4077 = historical, pre-`3cadb95`. | Load layer: none (already ignored). Operational: confirm server HEAD ≥ `3cadb95`. Optional: stop computing the ignored hash. | near-zero (non-load-bearing) | YES (code) |
| `rep_hashes` | 137 | regression (LIVE in checkpoint path only) | NO in load path (ignore_keys); YES in checkpoint validator `:3408` | Plural value-hash `_hash_rep_dict` `:375`. Load path fixed, but `_load_checkpoint` is fed `expected_search_meta` (`:4676`) and skips ONLY `code_timestamp`, so it still rejects+`unlink()`s resume checkpoints on value drift. Emits a DIFFERENT string (`"Checkpoint … stale"` `:3412`) so it contributes 0 to the grep counts. | 1-line: widen `:3408` skip set to `('code_timestamp','rep_hashes','aug_hashes','rep_hash','subset_hashes')`. | very low (`rep_shapes` `:373-374` still guards dimensionality) | YES (code) |
| `groups_hash` | 886 | nondeterministic-bug (hash fixed; input not proven) | **YES** (folds `:458`, ft `:660`, baseline `:437`) + SQLite IDENTITY_COLS (`result_cache.py:47`) | Hash IS deterministic: `_canonical_bytes` `:279-282` factorizes object/string arrays to sorted-canonical int codes (the "factorize object groups" fix is PRESENT); mandatory Levenshtein (`_shared.py:828`, no difflib fallback) closes the antibody-cluster fork. Historical 886 = pre-fix object-pointer `.tobytes()` + difflib/Levenshtein server divergence. **Live residual = upstream group ASSIGNMENT not byte-stable** (row reorder / dtype / appearance-order factorize bypassing the object branch). Dual-layer with NO row-rescue → drift recomputes the entire grid + baselines + folds. | Do NOT touch the hash. Stabilize the INPUT in `load_data`/`precompute_hybrid_searches` (`_shared.py:2755`): emit groups in canonical sorted/stable order + stable dtype, permuting (X,y,groups) jointly; add 2-run equality assertion. Defensive: in `_canonical_bytes` add `elif a.dtype.kind in ('i','u'): a=a.astype(np.int64)` (int64 no-op, preserves existing caches). | low in find_best_hybrid; upstream reorder MUST permute X/y/groups together. Do NOT promote float-clustering splits to a default. | hash YES (code); input determinism NOT verified (needs server 2-run) |
| `labels_hash` | 120 | legit-invalidation (+ latent dtype trap) | **YES** (ft `:660`, hybrid_eval `:600`, baseline `:436`) + SQLite IDENTITY_COLS (`:47`) | Hash deterministic. Most of 120 = genuine label-DEFINITION code changes. Latent trap: numeric arrays pass through raw `tobytes` (`:278-282`), so float32 reaction labels (`_shared.py:2393`) vs float64 elsewhere of identical values hash differently → one-time drift on any dtype change. Same dual-layer no-row-rescue blast radius as groups_hash. | Pin label dtype at the loader: change `_shared.py:2393` reaction labels `float32→float64` to match every other domain. Do NOT upcast inside `_canonical_bytes` (float32→float64 of 0.1 ≠ float64 0.1, busts caches without unifying). + 2-run check. | medium: float64 change busts reaction caches once; downstream safe (sklearn/xgboost upcast internally). | YES (code) |
| `rep_names` | 168 | nondeterministic-bug (Y1) | NO (search-only `:372`; out of IDENTITY/live metas) | `tuple(sorted(base_reps.keys()))` — deterministic given a fixed loaded set (NOT hash-salting). Drift = a DIFFERENT set actually loaded: a rep silently fails (HF offline / download race / OOM) and is dropped, shrinking the set on a flaky node. Can't print in HEAD (168 historical from dead search bundle), but the rep-VALUE cache + SQLite key can still FORK between flaky and clean nodes. | Key the pool on REQUESTED names, not loaded `base_reps.keys()`; offline-pin HF reps (`HF_HUB_OFFLINE`/`local_files_only`). Keep loaders WARN per directive but RECORD dropped set. **LOCKED: rep_names stays as DATA identity; do NOT add pool/shapes to the key.** | low-med; one-time identity change; must NOT re-add pool/shapes. | YES (code) |
| `internal_evaluators` | 118 | legit-invalidation | NO (search-only `:388`; demoted to CONFIG_COLS) | `tuple(sorted(...))` — deterministic. Drift = evaluator set genuinely changed: ridge REMOVED (`3cadb95`) then RE-ADDED (`2792b94`). Current `_shared.py:272` has ridge KEPT (`['knn','ridge','rf','mlp','lightgbm','svm']`, user 2026-06-08), reclaiming 34 distance caches. Now per-row CONFIG → a future change is a new row, not a rebuild. | No hash fix. Ensure precompute + experiment resolve `_norm_eval` identically (both fall back to tier default). Freeze list across the wave. | none | YES (code) |
| `model_names` | 88 | legit-invalidation | NO (search-only `:376`; demoted to CONFIG_COLS) | `sorted()` of DEFAULT_MODELS filtered by per-dataset skip policy + n-caps — order-stable, names-only. Drift = list genuinely changed (skip-policy edits, DEFAULT_MODELS expansion, SVM uncapped n>5000 gate). BY DESIGN. Hidden hazard: `_applied_skips` (`:1397-1398`) and `_apply_budget_policy` (`:1424-1425`) SWALLOW exceptions → a policy that throws in one env silently re-keys. | No correctness fix (self-heals under frozen policy). Harden the two bare-excepts to warn-then-raise. Batch all skip/model edits before the wipe. | removing bare-except could turn a tolerated bug into a hard failure mid-sweep; gate behind warn-then-raise. | YES (code) |
| `pruning_checkpoints` | 58 | legit-invalidation | NO (search-only, conditional `:345`; dead in HEAD) | Added to meta ONLY when a non-default schedule is present. Drift = mixing `--pruning-mode aggressive_pruning` ([25,50,100]) with `default_pruning` (key omitted). Loader iterates expected keys only (`:241`) → a default run reading an aggressive cache is NOT flagged (asymmetric correctness hole). Config-change, not nondeterminism. | Operational: pin `--pruning-mode` to ONE value for the wave. Optional: compare pruning_*-scoped key SETS in `_load_cache_with_meta`. | scoped set-compare invalidates legitimately-extra-key caches once; scope to pruning_*. | YES (code) |
| `aug_names` | 29 | nondeterministic-bug | **YES** (hybrid_eval `:602`, NOT in ignore_keys) | `sorted(augmentations.keys())` `:385` — order-stable. Mostly legit one-time aug-list evolution. Recurring nondeterministic risk parallel to Y1: precompute keys on LOADED not REQUESTED augs via silent drop (`precompute_hybrids.py:352` `{k: all_reps[k] for k in aug_names if k in all_reps}`) → a flaky aug-rep load changes the tuple run-to-run. ONE of the few fields that CAN still print drift in HEAD. | At `precompute_hybrids.py:352` detect `missing=[k for k in aug_names if k not in all_reps]` and warn+record instead of silently dropping; key on REQUESTED aug_names. | one-time aug-cache rebuild; prefer warn over raise. | YES (code) |
| `budgets` | 20 | legit-invalidation (+ latent canonicalization gap) | NO (search-only `:381`; demoted to CONFIG_COLS) | `tuple(budgets)` `:381` — NOT re-sorted, unlike every other list field. Main path pre-sorts (`sorted(set(filtered))` `:4326`) so stable in practice. Drift = 2026-06-21 per-dataset budget cap (`4fa5d9a`, `<=500`) rejecting pre-cap 8-budget caches. Latent: a direct `_make_search_meta` caller passing an unsorted/dup list spuriously re-keys. | Self-canonical: `:381` → `tuple(sorted(set(int(b) for b in budgets)))`. Else self-heals on rebuild. | re-keys the few caches with unsorted/dup budgets once; land with the wipe. | YES (code) |

---

## 3. Which prior fixes regressed

1. **CHECKPOINT-VALIDATOR REGRESSION (live, code-confirmed).** The 2026-06-04
   `ignore_keys` fix (`3cadb95`, `find_best_hybrid.py:198-200`) was NEVER propagated
   to `_load_checkpoint`. That function (`:3382-3419`) is fed
   `checkpoint_meta=expected_search_meta` (`:4676`), which contains `rep_hashes`
   (`:375`) + `aug_hashes` (`:386`), yet its skip loop (`:3408`) skips only
   `code_timestamp`. So it STILL rejects + `unlink()`s the resume checkpoint on the
   exact value-hashes the load layer was fixed to ignore. It emits a DIFFERENT log
   string (`"Checkpoint … stale (<k> drifted)"` `:3412`), so it contributes 0 to the
   grep counts — but it restarts fold-0 screening from scratch on every crash/resume
   of a multi-day neural-rep search. Impact: PARTIAL in-search progress only (not the
   search/baseline/precompute reuse), so secondary in scope but a genuine 1-line fix.

2. **THREAD-PIN DETERMINISM REGRESSION (`a274cf4`, 2026-06-17, setup.sh).** Un-pinned
   OMP/OPENBLAS/MKL/NUMBA/VECLIB threads (`setup.sh:16-21`, now
   `SLURM_CPUS_PER_TASK`), reversing the ~2026-06-10 pin-to-1 determinism measure.
   This reintroduces float reduction-order non-determinism in rep VALUES → the
   separate filename-keyed rep-VALUE cache misses (the 800× ginkgo_aggregation
   3.8s-vs-3096s load swings). It does NOT cause `"meta drift on rep_hash"` (that field
   is ignored at load), so the prompt's stated mechanism is half-right: the un-pin hits
   the rep-VALUE cache, not the meta layer.

3. **NOT a regression but a DEPLOYMENT GAP (unverifiable here).** If recent
   POST-deploy server jobs still print `"meta drift on rep_hash / rep_names /
   model_names / internal_evaluators / budgets / pruning_checkpoints"`, the deployed
   checkout is OLDER than `501f295`/`3cadb95` (stale `git pull` / wrong branch) running
   the pre-fix monolithic search-bundle comparison. (A pre-`501f295` copy that loads the
   full search meta into the drift printer physically exists in the tree at
   `build/lib/kirby/find_best_hybrid.py:3979`, git-ignored — a concrete demonstration of
   what stale code would print.)

CONFIRMED-STILL-PRESENT prior fixes (did NOT regress): factorize-object-groups in
`_canonical_bytes:279-282`; mandatory Levenshtein at `_shared.py:828` (no difflib
fallback); ridge KEPT in `_shared.py:272`.

---

## 4. Recommended fix sequence (exact file:line)

1. **(Operational, BLOCKING)** Confirm server checkout at HEAD:
   `git -C /data/stat-ecr/scat9264/KIRBy rev-parse HEAD` must be ≥ `501f295`/`3cadb95`.
   If behind: `git pull` + re-`. setup.sh`. This alone may resolve all 7 config/value
   fields, whose drift is impossible on HEAD.
2. **(Operational)** Date-filter logs: re-grep `meta drift on` on ONLY
   post-`501f295`-deploy job logs. Prediction: only `groups_hash` / `labels_hash` /
   `aug_names` can fire in HEAD; the other 7 are pre-deploy historical.
3. **(Operational)** One-time wipe of stale `searches/*.pkl`, `hybrid_evals/*.pkl`, and
   the rep-VALUE cache (+ `tests/purge_graphinity_cache.py --delete`) before rerun.
   Freeze DISTANCE_EVALUATORS (ridge KEPT), DEFAULT_MODELS, dataset_policy skip tables,
   and `--pruning-mode` for the whole wave first.
4. **(Code, 1-line, highest ROI)** `find_best_hybrid.py:3408` — widen the checkpoint
   skip from `('code_timestamp',)` to also include
   `'rep_hashes','aug_hashes','rep_hash','subset_hashes'` (reuse the load-layer tuple).
   Stops resume-from-scratch on neural-rep value drift; `rep_shapes` `:373-374` still
   guards dimensionality.
5. **(Investigation, the live linchpin)** Audit upstream group/label ASSIGNMENT
   determinism in `load_data` / `precompute_hybrid_searches` (`_shared.py:2755`): add a
   2-run dry-load equality check on `groups_hash` + `labels_hash` (the only IDENTITY_COLS
   fields that can still cause the catastrophic dual-layer recompute). Do NOT touch
   `_canonical_bytes`.
6. **(Code, low-risk hardening, land with the wipe)**
   (a) `_canonical_bytes` (`find_best_hybrid.py:278`) add
   `elif a.dtype.kind in ('i','u'): a=a.astype(np.int64)`;
   (b) reaction labels `_shared.py:2393` `float32→float64`;
   (c) budgets self-canonical `find_best_hybrid.py:381` →
   `tuple(sorted(set(int(b) for b in budgets)))`.
7. **(Code, Y1 + aug parallel)** Key rep pool on REQUESTED names (not
   `base_reps.keys()`); at `precompute_hybrids.py:352` warn+record missing aug names
   instead of silent drop; offline-pin HF reps. Keep loaders WARN per directive.
   **LOCKED: rep_names stays; do NOT add pool/shapes to the key.**
8. **(Decision, NOT solo — user sign-off)** Whether to re-pin rep-GENERATION
   BLAS/OMP threads in `setup.sh` (scoped to rep-gen, not the xgboost/lightgbm model-fit
   step) to kill rep-VALUE-cache misses without the perf regression `a274cf4` backed out.
   Measure perf before/after.
9. **(Code, optional cleanup)** Stop COMPUTING the ignored value-hashes at `:375`/
   `:2758` (CPU waste on n×1280 ESM matrices); harden `_applied_skips` /
   `_apply_budget_policy` (`:1397-1398`/`:1424-1425`) to warn-then-raise instead of
   swallowing.

Items 1-3 and 5 (operational + investigation) precede any cache-key change. Items 6-9
are PROPOSALS for user sign-off — no solo cache-key redesign.

---

## 5. setup.sh thread-unpin verdict

`a274cf4` is a REAL determinism regression for the rep-VALUE cache, but it is NOT the
cause of any `"meta drift on"` line. `setup.sh:16-21` sets
`OMP/OPENBLAS/MKL/NUMBA/VECLIB_NUM_THREADS = SLURM_CPUS_PER_TASK` (8 on precompute),
reversing the prior pin-to-1. Multi-threaded BLAS changes float reduction ORDER →
rep matrices are no longer bit-identical run-to-run. Because the rep-VALUE cache is a
separate filename-keyed pickle (NOT the meta layer), non-identical bytes miss it and
reps recompute — the 800× ginkgo load-time swings. The meta `rep_hash` field is ignored
at load (`:199-200`), so this does NOT show up as drift. Verdict: the perf win is real
(xgboost/lightgbm are the dominant model-fit cost and parallelise via OpenMP), so do NOT
blanket-revert; the correct move is to SCOPE a thread pin to the rep-GENERATION step only
(leave model-fit multi-threaded) — a user decision, measure both. Note `a274cf4` itself
warns "Revert to 1 only if an outer joblib pool over configs is ever added," confirming
the un-pin was a deliberate perf tradeoff, not an accident.

---

## 6. Server commands the user must run to confirm

```
# (a) Is HEAD actually deployed? Settles historical-vs-live for 7 of 10 fields.
git -C /data/stat-ecr/scat9264/KIRBy rev-parse HEAD          # must be >= 501f295 / 3cadb95

# (b) Which fields STILL fire post-deploy? Date-filter to logs after the 2026-06-11 rerun.
grep -h "meta drift on" <post-deploy-job-logs>               # predict only groups/labels/aug_names

# (c) Is the mandatory Levenshtein import satisfied (groups_hash difflib-fork closed)?
python -c 'import Levenshtein; print(Levenshtein.__version__)'

# (d) Does the thread un-pin actually perturb rep VALUES on the server BLAS?
python -c 'import numpy; numpy.show_config()'
#   then _hash_array of one fixed rep under OMP_NUM_THREADS=1 vs 8 — compare.

# (e) THE LINCHPIN: are groups_hash + labels_hash byte-stable across two dry loads?
#   load each dataset twice, compare groups_hash + labels_hash (the IDENTITY_COLS fields).

# (f) Pruning-mode mixing + n-stability:
#   confirm which SLURM scripts pass --pruning-mode aggressive_pruning vs default_pruning,
#   and that n_samples is identical at precompute-time vs experiment-time per dataset.
```

---

## 7. What resume/HEAD already fixed vs what remains

**ALREADY FIXED in HEAD (code-confirmed):**
- `rep_hash` / `rep_hashes` / `aug_hashes` / `subset_hashes` ignored in the load path
  (`:198-200`); none of the 4 live call sites (`:697`/`:2096`/`:2506`/`:2786`) override.
  → the 4077+137 counts are historical, cannot recur in HEAD.
- SQLite redesign (`501f295`) moved `model_names`/`budgets`/`internal_evaluators`/
  `rep_names`/`pruning_checkpoints` out of any loaded meta into per-row CONFIG_COLS
  (`result_cache.py:56-61`) → a change there is a new ROW lookup, never a rebuild. Their
  88/20/118/168/58 counts are pre-`501f295` historical and CANNOT print drift in HEAD.
- `groups_hash`/`labels_hash` HASH is deterministic: `_canonical_bytes` factorize fix
  (`:279-282`) + mandatory Levenshtein (`_shared.py:828`) present.
- ridge KEPT in DISTANCE_EVALUATORS (`_shared.py:272`), reclaiming 34 distance caches.

**WHAT REMAINS (genuinely live or unverified):**
- **Upstream group/label ASSIGNMENT determinism** (the linchpin) — NOT proven byte-stable;
  the only mechanism that can still trigger the dual-layer IDENTITY_COLS recompute. Needs
  the 2-run dry-load check (server cmd e).
- **Checkpoint-validator regression** (`:3408`) — live, 1-line fix (sequence step 4).
- **`aug_names` silent-drop** in precompute (`precompute_hybrids.py:352`) — live, can still
  print drift in HEAD.
- **`rep_names` Y1 fork** — rep-VALUE cache / SQLite key can fork between flaky and clean
  nodes even though the field can't print drift.
- **Thread un-pin rep-VALUE non-determinism** (`a274cf4`) — live operational cost (800×
  load swings); decision pending (step 8).
- **Deployment verification** — whether the server runs HEAD at all (step 1). If not, the
  entire "mostly historical" conclusion collapses and the pre-`501f295` monolithic
  comparison is still active.
- Latent one-time traps to land with the wipe: reaction-label float32/64 (`_shared.py:2393`),
  int-group dtype, budgets non-canonical (`:381`).

**Locked constraints respected:** rep_names stays as DATA identity; no rep pool/shapes added
to keys; the direction is drop non-deterministic VALUE-hashes + stabilize groups/labels
INPUT; all cache-key changes are PROPOSALS for user sign-off, not solo redesign.
