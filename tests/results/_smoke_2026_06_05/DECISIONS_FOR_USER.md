# Smoke-test decisions for you — 2026-06-05

> **RESOLUTION LOG (2026-06-05, after discussion) — applied this session, uncommitted:**
> - **3.1 (pool hybrid rebuild) — INVESTIGATED, NOT changed (reverted).** Reusing the search's
>   already-fitted hybrid is NOT a free speedup: the fitted path allocates feature budget from the
>   blueprint's stored `rep_scores` (best-across-models baseline) while `get_fitted_pool_hybrid`'s
>   rebuild recomputes allocation via the internal evaluator — they produce **different hybrids**
>   (verified: identical inputs, different selected features; the cheap path was ~6× faster but gave
>   different values). So it's a design fork, not an optimization. Left for your decision (see 3.1 below).
> - **3.2 (silent pool failure) — DONE.** Failed pool conditions and "no usable hybrid" now print a
>   `[pool] WARNING:` always (not just verbose). `_shared.py`.
> - **3.3 (stale-rep guard) — DONE.** `_finalize_load_data` now raises on any rep whose row count ≠
>   label count (non-destructive, names the rep + points to the purge script). New
>   `tests/purge_rep_cache.py` (dry-run-first) does the deletion/recompute. `_shared.py`.
> - **3.5 / 3.6 — declined** (no more local-test investment needed).
> - **3.7 — rely on run-ordering** (you'll remember; no code change).
> - **4.1 (s11 on classification) — DONE.** s11 now skips `task_type=='classification'` cleanly.
> - **4.2 — no fix** (server-only is fine; it's heavy, not broken).
> - **4.4 — confirmed wired** (`imrie_pretrain.sh` runs `pretrain_chembl_dnn`; `imrie_eval.sh` after it).
> - **4.5 — confirmed** (`--budgets` flag stays live, default unchanged).
> - **4.6 (fed03 Part A reps) — DONE.** Threads the ChEMBL silo's rep list instead of `'all'`.
> - **4.7 — flagged** (c17 + graphinity need a generous wall-clock budget; no code change).
> - **4.3 — RESOLVED (no action): the GP gate handles it.** My original "cubic GP blowup at full
>   rosetta ~21K" was WRONG. GP carries `_max_samples=1000` (find_best_hybrid.py:1209) and is
>   auto-skipped when n>1000 in BOTH the hybrid search (`_resolve_models`) and graphinity's own
>   baseline loop (`get_model_factories(MODELS_LIST, 'regression', n_samples=n)`,
>   opig_graphinity_ddg.py:459). GP isn't even in `DOWNSTREAM_MODELS`
>   (`['rf','xgboost','ridge','svm','mlp','lightgbm']`, _shared.py:281), and SVM is also gated
>   (`_max_samples=10000`). So at 21K only rf/xgboost/lightgbm/ridge/mlp run — all sub-quadratic,
>   no cubic term. Graphinity is heavy-but-tractable → give it a generous wall-clock budget (4.7);
>   `--max-rosetta` capping is optional (linear wall savings only), not required.
> All edits syntax-compiled; 3.2/3.3/4.1 unit-tested; c22 clf re-run clean end-to-end.



One document, everything that needs your judgment, in plain English, every claim backed by the
actual code. This is the consolidated output of a 6-agent local smoke-test fleet that ran the
**complexity, similarity, noise, federated, forward-compatibility, and OPIG** experiment families
end-to-end on small slices of REAL data (no baseline survey, no Phase 5, per your scope).

Per-family detail lives alongside this file: `COMPLEXITY_REPORT.md`, `SIMILARITY_REPORT.md`,
`NOISE_REPORT.md`, `FEDERATED_REPORT.md`, `FORWARD_COMPAT_REPORT.md`, `OPIG_REPORT.md`.

---

## 0. Bottom line

- **Every experiment family runs end-to-end locally on real data.** Classification and regression
  paths both work. Nothing in the experiment code is broken in a way that would crash the server run.
- **5 genuine bugs were found and fixed** (Section 2). Two of them (c15, c16) would have silently
  corrupted results on the server; the rest were latent or smoke-only.
- **Nothing has been committed.** You are the sign-off gate. Review Section 2 (already-applied fixes)
  and Section 3 (changes I deliberately did NOT make because they touch the shared core / change
  behaviour — your call per your "no autonomous design" rule).
- **The one scary-sounding finding turned out to be a false alarm.** A complexity agent reported
  "classification hybrid search returns no scored config" (would break every classification
  experiment). I verified directly: on a clean cache, c22 and c24 both produce real classification
  hybrids (c22 AUC 0.68 with full blueprint output; c24 a real transfer matrix). The failure was a
  stale local `herg` cache, not a code bug — see item 3.4. It will not reproduce on a fresh server cache.

---

## 1. How it was tested (so you can trust the "it runs" claim)

- Real datasets, small subsets: regression = `buchwald_hartwig` / `logd` / `ginkgo_heparin`;
  classification = `humanness_therapeutic` / `herg`. Local-only representations (aac, blosum, onehot,
  ecfp4, maccs, diff_maccs, …) — no foundation models, no server.
- Subset via the existing `KIRBY_SMOKE_N` row cap + tiny search config + (where the experiment is
  pure repetition) a wall-clock timeout that proves several iterations then stops.
- The timeout decisions were made by **reading each experiment's actual loop structure**, not from
  memory/estimate docs — see each family report's "Timeout audit" section, which cites the loop at
  file:line for every experiment.

---

## 2. Genuine bugs found and FIXED in-lane (review before you commit)

All verified present in the files. None are committed.

| # | Where | The bug | Server impact | Fix |
|---|-------|---------|---------------|-----|
| 2.1 | `complexity/c15_ml_theory_depth.py:516` | Passed `speed='fast'` to the hybrid search. `'fast'` was deleted from SPEED_TIERS and now raises `KeyError`, which was caught and silently swallowed → the double-descent analysis ran **without its hybrid blueprints**. | **YES — silent result corruption.** | `speed='fast'` → `speed='full'`. Re-ran: 0 errors. |
| 2.2 | `complexity/c16_feature_selection_as_regularization.py:266` | Same `speed='fast'` bug on the full-dimension baseline → that baseline raised, was swallowed, and **every full-dim cell came back NaN**. | **YES — silent NaN results.** | `speed='fast'` → `speed='full'`. Re-ran: 0 errors. |
| 2.3 | `federated/fed01:105`, `fed04:141`, `fed03:122` | When an already-binary dataset (labels 0/1) is loaded through the plain `load_data` path (no `is_binary` flag), the `label >= pKi 6.0` binarization mapped **every** sample to class 0 — a degenerate single-class target. | Partial — the active `run_all` path uses the Lhasa loader which sets `is_binary`, so live runs weren't degenerate; the fix is the correct defensive behaviour for the other code path and any future already-binary molecular dataset. | Also detect already-binary by value (unique ⊆ {0,1}) and pass through; gated so continuous-pKi data still thresholds. |
| 2.4 | `opig_blueprint_transfer.py:160` | The two searches hardcoded `budgets=FULL_BUDGETS`, so the script's own `--budgets` flag was silently ignored (latent, not smoke-only). | Low (default unchanged) but the flag was dead. | `budgets = config.get('budgets', FULL_BUDGETS)` — default identical to before, flag now live. **Confirm you're happy the flag is now respected (item 4.5).** |
| 2.5 | `opig_humanness/cdr_flexibility/ginkgo_developability/graphinity_ddg/trastuzumab` + `similarity/s11` | Their **smoke-mode** helper asked for `speed='fast'` (same dead key). Under `KIRBY_SMOKE_N` the search failed silently and the **hybrid was quietly dropped** — a smoke run looked fine but had no hybrid row. | None on the server (only fires under `KIRBY_SMOKE_N`). | Changed to `'full'`. |

Also fixed (environment-only, macOS): `imrie_transfer_learning.py:44,48` got the OpenMP pre-import
guard (it segfaulted on macOS without it); `c13:845` now parses a comma `--reps` list correctly
(server uses `--reps all`, so no server impact).

---

## 3. DECISIONS NEEDED — shared core (I did NOT change these; they need your sign-off)

These touch `tests/experiments/_shared.py`, `src/kirby/find_best_hybrid.py`, or shared run behaviour.
Per your "no autonomous design / no gate or search-space changes without sign-off" rule, I left them
for you. Ordered by how much they matter for the server re-run.

### 3.1 — Wasted compute: the hybrid is rebuilt from scratch after the cached search (affects EVERY experiment, server-wide)
- **What it does:** before its real work, every experiment fetches the cached "best hybrid recipe",
  then builds the hybrid feature matrix.
- **The problem:** `get_fitted_pool_hybrid` (`_shared.py` ~line 3345) calls `create_hybrid(...)`
  **again** after fetching the cached search, re-running the feature-importance calculation. When the
  winning recipe uses a TensorFlow/SHAP importance method (`deeplift`/`treeshap`) or a neural/kernel
  internal scorer, that rebuild costs ~4–5 minutes of dead setup **per experiment**, before its first
  real iteration. Across ~26 experiment runs that is real wasted cluster time. This is the same
  "cache-churn" family as the CACHE-1 issue in your memory.
- **Recommended:** have `get_fitted_pool_hybrid` reuse the already-fitted hybrid stored in the cached
  search instead of rebuilding (or return the cached `X_hybrid` directly when the search was stored
  with `return_fitted=True`).
- **Evidence:** `_shared.py:~3345`; `SPEED_TIERS['full']` importance = `['deeplift','lightgbm_gain','treeshap']` in `find_best_hybrid.py`.
- **Decision:** fix now (broad speedup), or accept the cost for this run?

### 3.2 — A failed hybrid search is swallowed silently (this is what hid bugs 2.1/2.2/2.5)
- **What it does:** the pool helper tries each hybrid "condition" and collects the ones that work.
- **The problem:** `_shared.py:3236-3239` catches a per-condition exception and continues with **no
  message**. So a config error (like the dead `'fast'` key) silently disables the hybrid and the
  experiment reports results with no hybrid — looking fine. This is why three of the bugs in Section 2
  were invisible until an agent set `KIRBY_POOL_VERBOSE=1`.
- **Recommended:** emit a one-line warning when a pool condition fails, and a louder one when **all**
  conditions fail (the hybrid is then entirely absent).
- **Evidence:** `_shared.py:3236-3239`.
- **Decision:** add the warning (cheap insurance against silent corruption), or leave as-is?

### 3.3 — Stale cached representation reused with the WRONG number of rows (correctness guard)
- **What it does:** representation arrays are cached to disk and reused across runs.
- **The problem:** `_cache_rep` (`_shared.py:~173`) loads a cached `<rep>.npy` whenever the file
  exists, with **no check that its row count matches the current labels**. The on-disk `herg` cache
  was 1412 rows while the loader now returns 7772, so `load_data('herg')` returned mismatched
  labels/reps — a hard misalignment that corrupts any downstream experiment. (This is also the root
  cause of the "classification hybrid returns nothing" false alarm — see 3.4.)
- **Recommended:** in `_cache_rep`, reject a cached array whose first dimension ≠ expected row count
  and recompute. Plus, before the server re-run, **purge the stale `herg` caches** (a `--purge` like
  the existing `purge_graphinity_cache.py`).
- **Evidence:** `_shared.py:168-176`; observed herg 1412 vs 7772.
- **Decision:** add the row-count guard? And/or just purge herg caches on the server before rerun?

### 3.4 — (Resolved — informational) "Classification hybrid search returns no scored config"
- A complexity agent flagged this as possibly breaking every classification experiment (c08/c22/c24…).
  **I verified it is NOT a real bug:** on a clean cache, c22 and c24 produce real classification
  hybrids (c22 AUC 0.6835 with full blueprint/SHAP output; c24 a real fold-transfer matrix). The other
  families also ran classification hybrids fine (opig humanness/trastuzumab, similarity herg). The
  original failure was the stale `herg` cache from 3.3. **No action beyond 3.3** (purge + row guard).

### 3.5 — No way to shrink the hybrid search grid for a fast local smoke (quality-of-life)
- **What it does:** the cached hybrid search.
- **The problem:** `precompute_hybrid_searches` hardcodes `speed='full'`, `models=None`, and reads the
  module-level `FULL_BUDGETS` — it **ignores `config['budgets']/models/importance/evaluators`** (this
  is intentional, to keep the cache key identical to the server). The consequence is that the "tiny
  config" approach alone does not shrink the grid (648 configs at one budget), so heavy experiments
  (c13's two condition searches, c17's mutual-information phase, graphinity's 200-group search) can't
  fully complete in a quick local smoke — agents had to monkeypatch module globals or use a timeout.
- **Recommended (optional):** add ONE `KIRBY_SMOKE_N`-gated knob in `precompute_hybrid_searches` that,
  under smoke only, restricts to e.g. `models=['rf','ridge']`, one importance method, one evaluator.
  Gated so server runs are byte-identical. This would let every hybrid-heavy experiment complete a
  full local smoke instead of a timeout-clean partial.
- **Evidence:** `_shared.py:2685-2688`.
- **Decision:** add the smoke knob (improves future local testing), or rely on timeouts as we did here?

### 3.6 — `tests/find_best_hybrid_molecular.py::load_dataset` ignores `KIRBY_SMOKE_N`
- Minor: imrie and blueprint pay the full OpenADMET download + dedup of ~5–7k molecules before any row
  cap. A gated row cap there would speed every molecular smoke. Local-smoke only; no server effect.

### 3.7 — Federated skips precompute for `herg` → it pays the search uncached (run-ordering)
- **What it does:** `federated_helpers.run_all` skips the precompute step for `herg`/`lhasa_herg`
  (lines ~2188-2191), so federated runs a fresh `find_best_hybrid` for both tiers inside the experiment.
- **The problem:** on the server, if **federated runs before** complexity/similarity (which write the
  same `herg__*.pkl` search caches), it pays the full uncached search build inside the experiment
  (~tens of minutes here under contention). If it runs after, it hits cache.
- **Recommended:** either accept the ordering (run complexity/similarity herg precompute first), OR add
  a `precompute_hybrid_searches` call to the herg branch of federated's `run_all` so it is
  self-sufficient. (`run_all` is editable in-lane, but the precompute-skip is an intentional design
  choice tied to the Lhasa loader, so I'm flagging not changing.)
- **Decision:** rely on run-ordering, or make federated self-sufficient?

---

## 4. DECISIONS NEEDED — experiment design / scope

### 4.1 — `s11` (Bayesian optimization) is meaningless on classification
- **What it does:** active-learning / "pick the next molecule to test to find binders fastest" in the
  compact hybrid space.
- **The problem:** on binary labels the "best value found so far" jumps to 1.0 on the first positive,
  so every method ties at a saturated score — it runs without error but carries no signal. s11's own
  docstring calls it molecular library-screening; it already gates on domain and n≥200 but **not on
  task type**.
- **Recommended:** skip s11 for `task_type=='classification'` (matches its existing skip-with-flag
  pattern at `s11_bayesian_optimization.py:786-792`), or keep it and document that clf results are
  uninformative. Not changed — design call.

### 4.2 — `opig_cgr_comparison` has no way to limit which representations it builds
- **What it does:** compares CGR vs SMILES reaction representations for yield prediction.
- **The problem:** it generates the **full** reaction-rep matrix unconditionally (no `--reps` flag),
  including foundation-model reaction/molecular reps (ReactionT5, Chytorch, CGR-SMILES-LM, GROVER,
  mhg_gnn, graph_kernel, mordred variants). Too heavy for a local CPU smoke; the fingerprint core works.
- **Recommended:** (i) add a `--reps` selector (best — useful for smoke AND targeted server runs), or
  (ii) accept it as server-only. Not changed — your call on whether to alter the rep set.
- **Evidence:** `opig_cgr_comparison.py:158-189`.

### 4.3 — `opig_graphinity_ddg` heaviness — RESOLVED (see resolution log; GP is gated, no cubic blowup)

> CORRECTION: the analysis below over-stated the cost. GP (and SVM) are auto-skipped at the n-gate,
> and GP isn't in graphinity's baseline model list at all, so at full rosetta (~21K) only sub-quadratic
> models run. The experiment is heavy-but-tractable, needs only a generous wall budget, not a forced
> reduction. Kept for the record:

- **What it does:** sequence reps + simple ML for antibody–antigen ΔΔG, grouped by PDB complex.
- **The problem:** the subsample still yields ~200 **unique** groups, so the internal cross-validation
  trains every downstream model (including the cubic-cost Gaussian process, plus DNN/CNN) across 5
  folds — >13 min even solo, even with a single tiny rep. The cost is the group count × model set, not
  rep width. Same code path as the 4 antibody OPIG experiments that fully passed, so it's a
  tractability limit, not a bug — it will finish on the server.
- **Recommended:** run on server as-is, OR (for a faster smoke) cap distinct groups or drop gp/dnn/cnn
  from the internal search models under smoke. Not changed — search-design choice.
- **Evidence:** `opig_graphinity_ddg.py:367-370`.

### 4.4 — `imrie_transfer_learning` transfer-learning conditions need server-side pretrained weights
- The transfer-learning conditions (dnn_tl_full / dnn_tl_frozen) skip locally because
  `/Volumes/seagate/kirby/chembl_pretrain` doesn't exist. The scratch/ridge/hybrid path is verified;
  to exercise the actual transfer-learning path you must first run `kirby.models.pretrain_chembl_dnn`
  on the server. No code change — operational.

### 4.5 — Confirm the `opig_blueprint_transfer --budgets` flag should be live (from fix 2.4)
- The flag was dead and now works (default behaviour unchanged). Just confirm you want it respected.

### 4.6 — `fed03` Part A hardcodes `'all'` representations
- **What it does:** the source-based hERG federated comparison (three real data-source silos).
- **The problem:** to embed the Lhasa molecules it calls `get_molecular_representations(..., 'all')`
  (hardcoded `'all'`, `fed03_synthetic_private.py:218`), which forces foundation models even when a rep
  subset is wanted (and makes Part A un-smoke-able locally). Parts B and C take the rep list from the
  data and are fine.
- **Recommended:** thread the experiment's resolved rep list into that call instead of `'all'`. Not
  changed — it alters what reps Part A computes on the server, so it needs your sign-off.

### 4.7 — `c17` mutual-information phase is the slow part (confirm server wall budget)
- **What it does:** compares feature-selection strategies using a k-nearest-neighbour mutual-information
  estimator.
- **The problem:** the greedy mRMR step calls the estimator O(features × budget) times; even at 420
  features / 190 rows it didn't finish in 5 min locally. It's correct, just compute-heavy and it grows
  with feature count and n. The file already row-caps at 2000 and prefilters to 2000 features.
- **Recommended:** make sure c17's per-experiment server wall budget accounts for this phase (it can be
  hours at high feature counts / large n). No code change for correctness.
- **Evidence:** `c17_information_theoretic_decomposition.py:52`, `_shared.py:1582`.

---

## 5. Local-only artifacts — NO action needed (won't happen on the Linux server)

- **`c23` segfaults locally** on the `lightgbm` internal evaluator — the documented macOS OpenMP
  "double libomp" crash, not a code bug (condition A with the rf evaluator ran clean). Won't occur on Linux.
- **`deeplift`/TensorFlow segfault** in the precompute path on macOS — same OpenMP class. Linux unaffected.
  (If you want clean local smokes later, gate `deeplift` out under `KIRBY_SMOKE_N`.)
- **"meta drift on n_samples (stored=200, expected=400) — will recompute"** every run — a stale local
  cache header from earlier sessions; a fresh server precompute won't have it (~40s/run wasted locally only).
- **Fleet contention** (up to ~25 concurrent Python processes during this smoke) inflated local wall
  times 3–10×; not a defect. The server runs one job per (category, dataset).
- **`c21` is a weak smoke** on datasets without augmentation reps (buchwald/herg) — it correctly skips
  the augmentation tests there. Real signal only on datasets that have augmentation reps.

---

## 6. Suggested pre-server-rerun checklist

1. Review and (if happy) commit the Section 2 fixes — especially **2.1 / 2.2** (these were corrupting
   complexity results silently). Nothing is committed yet.
2. Decide the Section 3 shared-core items — at minimum **3.3** (purge stale `herg` caches before the
   rerun; ideally add the row-count guard) and **3.2** (the silent-swallow warning, which is what hid
   the 2.x bugs).
3. Decide the Section 4 scope items (s11-on-classification, cgr `--reps`, graphinity/imrie server-only,
   fed03 Part A reps).
4. The smoke-only knobs the agents added (gated on `KIRBY_SMOKE_N` in `_smoke_harness.py`,
   `similarity_helpers.py`, the opig scripts, `_run_*_locally.py`, `_smoke_driver.py`) are no-ops on the
   server — keep or discard as you like; they don't affect full runs.

— end —
