# CACHE FIX PLAN

**Status:** draft, awaiting 3 background audit agents + user approval on scope.
**Owner:** Claude (this session).
**Date started:** 2026-04-15.
**Do not compact this file.** It is the single operational doc for this recovery.

---

## 0. What I'm taking responsibility for

Three sessions worth of "is the cache working" checks reported green because I was grepping for `Loaded cached result` in SLURM logs — which only proves a file was opened, not that it contained usable data. The real check is `pickle.load(f); assert n_results > 0`. I never ran that. Every "yes the cache is working" confirmation I gave before today was built on a check that can't distinguish a healthy cache from an empty one.

Additionally, prior Claude sessions bumped `_CACHE_VERSION` in commits 6b632af and 24ef60e without flagging that it would orphan the then-active April 4 precompute run. Those commits should have been gated on an explicit "do we accept invalidating precompute" user decision. They weren't.

I'm not going to spend more of this plan apologizing. What follows is the fix.

---

## 1. Three compounding defects (not one)

Symptom: experiments load cache files that contain `n_results=0` and error on `"No fitted result for model=knn"`.

### Defect A — `_CACHE_VERSION` bumps orphaned everything

- `_CACHE_VERSION` is prefixed to the SHA256 hash that names every cache file.
- April 4: 240 SLURM precompute jobs ran under `_CACHE_VERSION=12`.
- April 11 commit `6b632af` bumped to 13.
- April 11 commit `24ef60e` bumped to 14.
- Today's code hashes every lookup with `v14`. The April 4 files on disk were written with `v12`-prefixed hashes. These hashes can **never** collide — it's not a config mismatch, it's filename math.

### Defect B — model tier promotion changed what `full` tier computes

- `24ef60e` promoted `knn` and `gp` from `tier='extra'` to `tier='full'`.
- April 4 precompute ran `full` with 6 models: `lightgbm, mlp, rf, ridge, svm, xgboost`.
- Today's code expects `full` to have 8 models (adds `knn, gp`).
- Even if Defect A were fixed, loading an April 4 SearchResult and calling `get_fitted_hybrid(downstream_model='knn')` fails with `Available models: []` — because knn was never searched.

### Defect C — `6b632af` changed the algorithm, not just the cache key

- `6b632af` title: "Group-aware internal split in create_hybrid + cache version bump to v13".
- This commit changed how `create_hybrid` constructs its internal train/val split during feature selection — not a cache-key change, an **algorithmic** change.
- This means even a magical rename of pre-April-11 cache files would still give you search results from a stale algorithm. Those scores are not directly comparable to anything the current `create_hybrid` produces.

**Conclusion:** Every cache file written before April 11 is fully invalid. No rename, no migration, no patch can rescue them. The only path is delete + re-run.

---

## 2. What's salvageable

**Total JSON result files on disk: ~468** across `tests/results/`.

### 2.1 SAFE (no cache dependency — keep)

- `tests/results/universal_survey/` — all files Mar 5
- `tests/results/antibody_survey/` — all files Mar 5
- `tests/results/comparisons/` — mostly Mar 5–25
- `tests/results/phase5_baselines/` — Mar 5–28 entries only (Apr 7+ entries SUSPECT)
- Any file with mtime **before 2026-04-04** is pre-precompute-run and never touched the broken cache.

### 2.2 SUSPECT (probably need re-running)

- **all of `tests/results/similarity/`** — 12 experiments (s01–s12), mtime ~April 1, but ran in the post-SLURM batch window; depend on `precompute_hybrid_searches` in-session
- **all of `tests/results/noise/`** — 8 experiments (n01–n08), same story
- **`tests/results/framework/f15_adaptive_pruning/`** — Apr 6–8, ran during the window
- **any OPIG result** that came from `opig_blueprint_transfer.py` — explicitly calls `get_precomputed_search`
- **anything modified Apr 11–15** — after both version-bump commits landed
- **`tests/results/precompute_registry.json`** — Apr 10, contains stale registry entries

### 2.3 Authoritative truth source

`PIPELINE_READINESS_TODO.md` is the user's single-source-of-truth for "what results are locked." Anything marked "reviewed and locked" in that doc should be re-checked manually but is most likely safe. I (Claude) will **not** autonomously re-categorize anything as safe/suspect without user confirmation — this list is a starting point, not the verdict.

**Action for user (not me):** audit `PIPELINE_READINESS_TODO.md` against the SUSPECT list above. Anything locked there that falls in the SUSPECT list is a direct conflict that needs your judgment.

---

## 3. New cache architecture

### 3.1 Design principles

1. **Filename is the cache key.** No hashing. Looking at the cache directory answers "what's cached."
2. **No version numbers.** If semantics change, content validation on load catches it, cache entry is rebuilt. No mass-orphaning.
3. **Multi-level dedup.** Things that don't depend on augmentation strategy or allocation method (per-rep baselines, per-rep feature importance, fold splits) live in their own cache layers and are shared across all conditions of the same dataset.
4. **Validation on load, not trust.** Every load verifies that stored metadata matches the caller's request. Validation failure → recompute.
5. **Atomic writes.** Already present. Keep.
6. **Fail-safe.** Validation failure is never an error — it's a recompute.

### 3.2 Directory layout

```
~/.cache/kirby/
├── searches/
│   └── {dataset}__{tier}__{aug_strategy}__{speed}.pkl
│       # e.g. abagym_Ang2__general__uncapped__full.pkl
│       # Stores: full SearchResult. One per precompute condition.
│
├── baselines/
│   └── {dataset}__{rep}__{model}__{metric}__{folds}f_{seed}.pkl
│       # e.g. abagym_Ang2__esm2__rf__spearman__5f_42.pkl
│       # Stores: per-fold baseline scores. Shared across conditions
│       # (same dataset+rep+model → same baseline regardless of aug).
│
├── importances/
│   └── {dataset}__{rep}__{method}__{folds}f_{seed}.pkl
│       # e.g. abagym_Ang2__esm2__lightgbm_gain__5f_42.pkl
│       # Stores: per-fold feature importance rankings.
│
├── folds/
│   └── {dataset}__{folds}f_{seed}.pkl
│       # Stores: (train_idx, val_idx) tuples per fold. Trivially cheap,
│       # but prevents nondeterminism if groups-array drifts.
│
├── alloc_scores/  (optional, phase 2)
│   └── {dataset}__{rep}__{method}__{evaluator}__{metric}__{folds}f_{seed}.pkl
│       # Stores: per-rep allocation curve for one (method, evaluator).
│
└── hybrid_registry.json
    # Existing — keep. Index of dataset → condition → model → budget → blueprint.
```

### 3.3 Cache-slot validation schema

Every cached `.pkl` stores a `_cache_meta` header dict alongside its payload:

```python
{
    '_cache_meta': {
        'dataset_name': str,
        'n_samples': int,
        'rep_names': sorted tuple of str,          # for searches
        'rep_name': str,                            # for baselines/importances
        'rep_shape': tuple,                         # for baselines/importances
        'model_names': sorted tuple of str,         # for searches
        'model_name': str,                          # for baselines
        'metric_name': str,
        'task_type': str,
        'aug_strategy': str,                        # for searches
        'aug_names': sorted tuple,                  # for searches
        'speed': str,
        'n_folds': int,
        'seed': int,
        'labels_hash': int,                         # hash(labels.tobytes())
        'groups_hash': Optional[int],
        'code_timestamp': float,                    # time.time() at write
    },
    'payload': <SearchResult | baseline dict | ...>
}
```

**On load:** every field in `_cache_meta` must match what the caller expects. First mismatch → invalidate and recompute.

The `code_timestamp` is informational — not used for validation, but lets the user see "this cache slot was last rebuilt on XYZ" by listing filenames.

### 3.4 Invariants that make this safe

- Filename contains enough to identify the cache slot uniquely for a given dataset.
- Metadata contains enough to reject stale content (labels changed, rep array reshape, model list drift).
- Because filenames are human-readable, we can answer "do I have a cached search for abagym_Ang2 top tier uncapped?" by running `ls ~/.cache/kirby/searches/abagym_Ang2__top__*`.
- Because no hashing is used, changes to parameters that **should** be ignored (verbose, cache_dir, etc.) can never accidentally get mixed in.

### 3.5 What the new cache does NOT cover

- Rep feature extraction (already cached elsewhere by rep factories — unchanged)
- Raw data loading (already cached by `load_data` — unchanged)
- Models themselves (stateless by construction)

---

## 4. Exact files to edit

### 4.1 CRITICAL — blocks the entire fix

#### `src/kirby/find_best_hybrid.py`

- **Line 60**: delete `_CACHE_VERSION = 14`
- **Lines 2320–2383**: delete `_hybrid_cache_key()` entirely
- **Lines 2601–2619** (inside `find_best_hybrid`, cache load): replace SHA-based lookup with filename construction via `_cache_path(dataset_name, tier, aug_strategy, speed)`
- **Lines 2874–2891** (cache save): same filename construction + write `_cache_meta` header
- **Line 2468**: `dataset_name: Optional[str] = None` already exists but is unused. Make it required-when-`cache=True`, raise at line 2599 if missing.
- **New param**: `tier: Optional[str] = None`, same requirement
- **New function**: `_cache_path(dataset_name, tier, aug_strategy, speed)` → `Path(base / 'searches' / f'{dataset_name}__{tier}__{aug_strategy}__{speed}.pkl')`
- **New function**: `_validate_cache_meta(stored_meta, expected_meta)` → bool
- **New function**: `_load_or_compute_baseline(dataset_name, rep_name, X, labels, groups, model_name, model, n_folds, seed, metric_fn, metric_name)` — reads/writes `{cache_base}/baselines/{dataset_name}__{rep_name}__{model_name}__{metric_name}__{n_folds}f_{seed}.pkl`
- **Refactor `_evaluate_baselines`** (line 1432) to call `_load_or_compute_baseline` per (rep, model)
- **New fold cache**: `_load_or_compute_folds(dataset_name, groups, n_folds, seed)` — reads/writes `{cache_base}/folds/{dataset_name}__{n_folds}f_{seed}.pkl`

#### `tests/experiments/_shared.py` — three functions missing `dataset_name`

- **Line 1691** `precompute_hybrid_searches(reps, labels, groups, config, ...)`: add required `dataset_name: str` param
- **Line 1747** pass `dataset_name=dataset_name, tier=tier_name` to `find_best_hybrid`
- **Line 1821** `get_precomputed_search(reps, labels, groups, ...)`: add required `dataset_name: str`, `tier: str` params
- **Line 1879** pass through to `find_best_hybrid`
- **Line 1895** `get_fitted_hybrid(...)`: add `dataset_name: str`, `tier: str`
- **Line 1938** forward to `get_precomputed_search`

#### `tests/precompute_hybrids.py`

- **Line 309** `find_best_hybrid` call already has `dataset_name=dataset_name`. Add `tier=tier` (where `tier` comes from the condition loop at line 296).
- Condition string normalization: `(tier, strat)` tuple → canonical filename part. Map:
  - `('general', 'none')` → `general__none`
  - `('general', 'uncapped')` → `general__uncapped`
  - `('general', 'capped')` → `general__capped`
  - `('top', 'none')` → `top__none`, etc.
  - `('general_distance', 'none')` → `general_distance__none`
  - `('top+borderline', 'uncapped')` → `top_borderline__uncapped` (replace `+` with `_` for filesystem safety)

#### `tests/check_precompute_status.py`

- Current "ready" check only reads `hybrid_registry.json` — rewrite.
- **New ready definition**: for each expected `(dataset, tier, strategy)` condition, the file `{cache_base}/searches/{dataset}__{tier}__{strategy}__full.pkl` must exist AND `pickle.load(f)['_cache_meta']` validates AND `payload.best_by_model` is non-empty AND `payload.all_results` is non-empty.
- Retain the `expected_conditions()` helper but make it return `(dataset, tier, strat)` tuples, then check filesystem directly.

### 4.2 HIGH — needed for full coverage (~30 experiment files)

All ~30 files under `tests/experiments/{complexity,similarity,noise,federated,opig*}/` call either `get_precomputed_search(...)` or `get_fitted_hybrid(...)`. Every one already has `dataset_name` in scope (from `run(data, config, output_dir, dataset_name, **kwargs)`). The edit is mechanical:

```diff
- search = get_precomputed_search(reps, labels, groups, aug_condition=..., ...)
+ search = get_precomputed_search(reps, labels, groups, dataset_name=dataset_name,
+                                  tier='general', aug_condition=..., ...)
```

The `tier` argument will usually be `'general'` because most experiments operate on the general rep set. Experiments that use top-tier reps need `tier='top'`. I will audit each of these files individually during implementation.

### 4.3 MEDIUM — standalone wrappers

- `tests/find_best_hybrid_molecular.py` **line 976**: add `dataset_name=args.dataset, tier='general'`
- `tests/find_best_hybrid_dta.py` **line 976**: same
- `tests/find_best_hybrid_tcga.py` **line 1065**: same
- `tests/find_best_hybrid_nucleotide.py` **line 399**: currently passes no cache param — leave alone for Phase 1, reconsider in Phase 5

### 4.4 NO CHANGE needed

- `tests/experiments/{complexity,similarity,noise,federated}_helpers.py` — only import `SPEED_TIERS`, no cache calls
- `tests/dta_baseline_survey.py`, `tests/antibody_baseline_survey.py` — use a **separate** rep cache at `~/.cache/kirby/reps/...`, not the hybrid cache. Unaffected.
- `tests/precompute_ft_reps.py` — only imports splitter logic, no cache calls

---

## 5. Implementation plan (phased)

### Phase 0 — Emergency stop (user action)

**Blocks everything.** Nothing in phases 1-5 runs until phase 0 is done.

1. User cancels all currently-running SLURM jobs (Batch 2 + zombies). I will not generate `scancel` commands — standing rule. User lists squeue and decides.
2. User backs up `~/.cache/kirby/` (just in case — a `mv` to `~/.cache/kirby.old` is fine).
3. User confirms: "proceed with cache nuke + redesign."

### Phase 1 — Cache redesign implementation (Claude, local only)

1. Remove `_CACHE_VERSION` + `_hybrid_cache_key`
2. Add `_cache_path`, `_cache_meta_matches`, atomic read/write helpers
3. Add `dataset_name` + `tier` parameters to `find_best_hybrid` (required when `cache=True`)
4. Add per-rep baseline cache layer
5. Add fold split cache layer
6. Add content validation on every load
7. Update every caller (20+ files) to thread `dataset_name` + `tier`
8. Rewrite `check_precompute_status.py` to validate content, not filename presence
9. Update `precompute_hybrids.py` to pass tier through

### Phase 2 — Local verification (Claude, synthetic only)

**No heavy compute.** Tiny synthetic dataset:
- 50 samples, 3 reps (40D/20D/10D), regression
- `find_best_hybrid(cache=True, dataset_name='synth', tier='test', speed='full')`
- Call 1: verify compute + cache write
- Call 2: verify cache hit with verified metadata
- Call 3: mutate rep array, verify validation catches mismatch, recomputes
- Call 4: run "general" + "top" conditions, verify per-rep baselines are shared (baseline file created only once per rep)
- Call 5: delete one baseline cache file, verify search recomputes only the missing rep's baseline

This is a correctness test, not a performance test. Should run in <60 seconds locally.

### Phase 3 — User-driven cache deletion (user action, I provide exact command)

Per the "never autonomous cache clearing" standing rule, I will provide the user with:
- An exact `ls` command to audit current contents
- An exact `mv` command to move the old cache aside (reversible)
- An `rm -rf` command they can run **after** they've verified Phase 2 works

User runs these themselves. I will not run them.

### Phase 4 — Fresh precompute, tiered submission (user action, I provide commands)

**Cluster split.** The hybrid search loop uses lightgbm, mlp (sklearn), rf, ridge, svm, xgboost, knn, gp — **all CPU-bound**. No PyTorch in the search. **All precompute jobs go to ARC (CPU).**

**Scope exclusions (user-confirmed 2026-04-15):**
- **Rep materialization is out of scope for this plan.** User handles rep caches separately; all rep `.npy` files are complete by the time precompute runs. I do not touch `~/.cache/kirby/reps/`.
- **Nucleotide datasets are out of scope for precompute.** nt_H2AFZ, nt_H3K27ac, nt_H3K27me3, nt_H3K36me3, nt_H3K4me1 are handled via the separate nucleotide survey scripts, not via `tests/precompute_hybrids.py`. They do not appear in the Tier 3 submission below.

**Time estimates below are guesses, not commitments.** I've been wrong about runtimes before — treat them as a rough ordering, not a schedule. The tiering is driven by dataset *size* and *downstream impact*, not wall-clock accuracy.

**Rule: do the smoke test first. If it validates, submit everything else in parallel.** No multi-tier gating.

#### Step 1 — smoke test

Three tiny datasets, submit in parallel. If all three produce a valid `~/.cache/kirby/searches/{dataset}__general__none__full.pkl` with non-empty `best_by_model`, the code works.

```bash
sbatch tests/slurm_scripts/submit_precompute.sh flab_pk_jain_fcrnrt
sbatch tests/slurm_scripts/submit_precompute.sh abagym_HER2
sbatch tests/slurm_scripts/submit_precompute.sh humanness_therapeutic
```

#### Step 2 — everything else (once smoke test passes)

Fire the full precompute in one go. TCGA, large DMS, and large molecular go to larger partitions; everything else uses the standard submit script. Nucleotides excluded (handled by survey scripts).

```bash
# Standard (everything small/medium)
for ds in \
  logd caco2 flab_thermo_tresanco buchwald_hartwig herg_ki \
  flab_pk_jain_heprt flab_thermo_jain flab_aggregation_jain_hicrt flab_polyreact_jain_cic \
  ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer \
  abagym_Ang2 abagym_VEGF abagym_EGFR abagym_lysozyme \
  tz_her2_binary flab_binding_adams flab_expression_adams flab_immunogenicity_marks \
  bh_doyle_plates ddg_combined abcov_kd \
  flab_aggregation_jain flab_polyreact_jain_bvp flab_binding_phillips_h3 \
  adme_mdr1 graphinity_foldx graphinity_rosetta graphinity_augmented; do
  sbatch tests/slurm_scripts/submit_precompute.sh $ds
done

# Heavy — larger partition / memory
sbatch --partition=medium --mem=256G --time=8:00:00 tests/slurm_scripts/submit_precompute.sh tcga_luad_vs_lusc
sbatch --partition=medium --mem=256G --time=8:00:00 tests/slurm_scripts/submit_precompute.sh tcga_brca_pam50
sbatch --partition=long --time=48:00:00 tests/slurm_scripts/submit_precompute.sh flab_binding_koenig
sbatch --partition=long --time=48:00:00 tests/slurm_scripts/submit_precompute.sh flab_expression_koenig
for ds in qm9 adme_hlm adme_rlm; do
  sbatch --partition=medium --time=8:00:00 tests/slurm_scripts/submit_precompute.sh $ds
done
```

**This list is driven by `DATASET_REPS` minus nucleotides.** Cross-check with the full dataset inventory and adjust if anything is missing. I'll generate a definitive list from `DATASET_REPS` during Phase 1 and update this section.

### Phase 5 — Experiment resubmission

Only after Phase 4 tiers are done for the datasets those experiments need.

---

## 6. User approval gates

I will **not** proceed past any of these without explicit user "go":

1. **Design approval** — you read sections 1-3 and say "yes, that's the architecture I want"
2. **Scope approval** — you read section 4 (edit list, after agent returns) and confirm
3. **Phase 2 verification** — you see the synthetic test output and confirm it looks correct
4. **Phase 3 deletion** — you run the `mv` command yourself, confirm old cache is aside
5. **Phase 4 submission** — you approve each tier separately; I do not autogenerate bulk commands

---

## 7. Known tradeoffs

**Filename length.** `abagym_Ang2__general_distance__uncapped__full.pkl` is ~55 chars. Still well under filesystem limits. No problem.

**Collision risk.** If two datasets share a name (e.g., `abagym_Ang2` loaded from two different sources), the cache would collide. Mitigation: `dataset_name` is supplied by the experiment registry, is guaranteed unique in `DATASET_REPS`. Content validation catches mismatches anyway.

**Losing "different search, same data" dedup.** The old hash-based scheme would match any two calls with identical inputs regardless of what they were called from. The new scheme requires the caller to know `dataset_name` + `tier`. This is strictly more caller responsibility. Mitigation: every existing caller of `find_best_hybrid(cache=True)` already has access to a `dataset_name` — it just wasn't being passed.

**Stale metadata on model tier changes.** If we re-promote/demote models in a future update, old baseline caches will still match by filename but won't contain new models. Mitigation: content validation on load checks `model_names` tuple. On mismatch → recompute. No silent staleness.

**Multi-level cache means more files.** 41 datasets × 24 reps × 8 models = ~7900 baseline files. Fine — they're tiny. Better than re-computing them 6x per dataset per session.

---

## 8. What I will NOT do

- I will not run `scancel`, `rm -rf`, `mv`, or any destructive command for you. You do those yourself.
- I will not re-bump `_CACHE_VERSION`. It's gone.
- I will not skip content validation to "speed things up."
- I will not declare Phase 1 done until the Phase 2 synthetic test passes end-to-end.
- I will not declare Phase 4 done until you've verified the first tier of jobs produced non-empty cache files with valid metadata.

---

## 9. Status tracker

- [ ] Phase 0.1 — User cancels running jobs
- [ ] Phase 0.2 — User backs up old cache dir
- [ ] Phase 0.3 — User approves design (sections 1-3)
- [x] Phase 1.1 — Remove `_CACHE_VERSION` + `_hybrid_cache_key`, add filename helpers, content validation (**done**, `src/kirby/find_best_hybrid.py`)
- [x] Phase 1.2 — Thread `dataset_name` + `tier` through `_shared.py` (**done**, 3 helper functions)
- [x] Phase 1.3 — Update `precompute_hybrids.py` to pass `tier` (**done**)
- [x] Phase 1.4 — Rewrite `check_precompute_status.py` to validate content on disk (**done**)
- [x] Phase 1.5 — Add per-rep baseline cache layer (**done**, dedup across conditions)
- [~] Phase 1.6 — Thread `dataset_name` + `tier` through ~30 experiment files (**in progress**, agent running)
- [x] Phase 2 — Synthetic local test passes (**done**, 6 tests incl. mutation detection + cross-condition baseline dedup)
- [ ] Phase 3 — User deletes old cache
- [ ] Phase 4.0 — Smoke test on server
- [ ] Phase 4.1 — Full precompute re-run
- [ ] Phase 5 — Experiments resubmitted

---

## 10. Audit agent results — integrated into sections 2, 4, 5 above.

The three background agents completed:
- **Cache call-graph** → findings drove section 4 (exact edit list)
- **Salvageable results** → findings drove section 2
- **Precompute scope** → findings drove section 5 Phase 4 (tiered submission)

Full agent transcripts are in the task output files; kept out of this doc to keep it readable.

---

## 11. What the user needs to do RIGHT NOW

Phase 1 is complete locally. Synthetic test passed. Phase 1.6 (threading `dataset_name` through the 30+ experiment files) is in progress via a background agent. The remaining steps are on you:

1. **Cancel all running jobs.** Run `squeue -u $USER` on the server, then `scancel` the ones you want dead. I will not generate a bulk scancel — do it yourself, eyes open.

2. **Back up the old cache.** On the server:
   ```bash
   mv ~/.cache/kirby ~/.cache/kirby.old_2026-04-15
   ```
   Reversible. Takes one second. This is the cache deletion — I promise I am not autonomously running this. You type it.

3. **Pull the code changes and read this file.** I will list them in a summary message. Read sections 1, 3, 4, and the status tracker.

4. **Launch the smoke test** (Section 5, Phase 4, Step 1):
   ```bash
   sbatch tests/slurm_scripts/submit_precompute.sh flab_pk_jain_fcrnrt
   sbatch tests/slurm_scripts/submit_precompute.sh abagym_HER2
   sbatch tests/slurm_scripts/submit_precompute.sh humanness_therapeutic
   ```

5. **After each smoke-test job completes**, confirm the cache file is valid:
   ```bash
   python tests/check_precompute_status.py --dataset abagym_HER2 --verbose
   ```
   Should show `DONE` with all 3 conditions `ok`. If any show `missing`, `corrupt`, `empty`, or `stale` — stop and ping me.

6. **Once smoke test passes**, launch the full precompute in Section 5 Phase 4 Step 2.

7. **Wait for that to finish**, then resubmit experiments (Phase 5). I'll give you the exact commands once Phase 1.6 is complete.

What I am NOT going to do unasked:
- Run anything on your server
- Delete anything anywhere
- Bump `_CACHE_VERSION` again — it is gone
