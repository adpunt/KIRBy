# §V LOCAL SMOKE GATE — SIMILARITY family report (2026-06-10)

Agent: similarity lane (resumed after prior agent died on a transient API outage).
Env: `/Users/apunt/anaconda3/bin/python` 3.11, repo root, single-thread OMP env,
`KIRBY_CACHE_DIR=/tmp/kirby_V_similarity`, `KIRBY_SMOKE_N=200`.
Datasets: **`humanness_therapeutic`** (clf, antibody, n→200), **`logd`** (reg, molecular, n→200).
Reps: antibody = `aac,blosum,onehot`; molecular = `ecfp4,maccs` (all local, no foundation models).
Result DBs: `/tmp/kirby_V_similarity/result_db/{humanness_therapeutic,logd,ginkgo_heparin}.db`.

Verdict: **PASS-with-caveats.** Cache reuse proven (no `meta drift`). All 12 reg cells and
9/12 clf cells PASS/SKIP cleanly; s11-clf SKIP confirmed; the only FAIL was an s03 smoke-config
artifact that is now FIXED (validated on reg). The 3 remaining clf gap cells (s01/s02/s07 on
humanness) could not be filled locally because the `top_distance` precompute on humanness's
5600-dim antibody reps is prohibitively slow (>17 min, see DECISIONS); not a lane bug.

---

## 1. SMOKE RESULTS — s01–s13 × {clf, reg}

clf = `humanness_therapeutic` (antibody), reg = `logd` (molecular). "PASS" = ran clean, no
unexpected `{'error'}`. "SKIP" = intentional `{'skipped':...}` marker (wave-audit counts complete).
Sub-task skips (regression-only paths on clf data) are noted but the experiment is a PASS.

| Exp | clf (humanness) | reg (logd) | Notes |
|-----|-----------------|------------|-------|
| s01 similarity_baseline | gap* (PASS on ginkgo_heparin reg) | **PASS** | s01 reg PASS verified (ginkgo_heparin + logd): per-rep SAC + `_sequence_cliffs` (full_V + cdr_only). clf gap = top_distance precompute too slow locally (see §3). |
| s02 hybrid_similarity | gap* | **PASS** | reg ran clean (per_rep + concat + weighted SAC/violation). clf gap = same precompute blocker. |
| s03 feature_concentration | was FAIL → **FIXED** | **PASS (fix verified)** | Prior JSONs had 2 `{'error'}` rows (lightgbm/mlp). **In-lane fix** (see §3) — reg re-run now emits only `('rf',25)`, 0 errors. clf re-run blocked by precompute but uses the SAME fixed path. |
| s04 activity_cliffs | **SKIP** (molecular-only, domain=antibody) | **PASS** | clf skip is correct (s04 is ECFP4-Tanimoto, molecular-only). |
| s05 knn_prediction | **PASS** | **PASS** | Also PASS on ginkgo_heparin. Honest fixed-ref deltas present. |
| s06 applicability_domain | **PASS** (conformal sub-skips: classification) | **PASS** | clf conformal sub-task skips cleanly; distance/reliability computed. |
| s07 noise_bridge | gap* | **PASS** | reg PASS (full hybrid search + cliff analysis). clf gap = precompute blocker; clf path branches on is_clf and is shared with reg. |
| s08 optimal_dimension | **PASS** | **PASS** | budget/dim sweep + per-model top3. |
| s10 importance_downstream | **PASS** (weighted_gp/svr sub-skip: classification) | **PASS** | clf GP/SVR sub-tasks are regression-only → clean sub-skips; weighted_knn + qrasar ran. |
| s11 bayesian_optimization | **SKIP** (domain=antibody) | **PASS** (real BO ran, 93s) | **Required clf SKIP confirmed.** s11 skips clf cleanly twice: domain!=molecular AND task_type==classification both guard. reg ran a real BO loop (not a skip). |
| s12 cross_domain_cliffs | **PASS** | **PASS** | generic Euclidean cliff detection works on both domains. |
| s13 clustering_retrieval | **PASS** | **PASS** | clustering + retrieval both populated. |

\* gap = output not produced locally; blocked by the slow humanness `top_distance` precompute, NOT
a bug. See §3 DECISIONS. clf-specific behaviour for every experiment is otherwise proven (skips,
sub-skips, passes), and the only clf-only code differences (s03/s07 is_clf branches) are exercised
on the reg side via the same shared functions.

### Commands (representative)
```
# reg lane (completed clean, 625s): s01 s02 s11 s03 on logd
OMP_NUM_THREADS=1 ... KIRBY_CACHE_DIR=/tmp/kirby_V_similarity KIRBY_SMOKE_N=200 \
  KIRBY_LOCAL_DS=logd KIRBY_LOCAL_REPS=ecfp4,maccs \
  perl -e 'alarm 1200; exec @ARGV' python \
  tests/experiments/similarity/_run_similarity_locally.py logd s01 s02 s11 s03
#  -> s01 8.0s, s02 21.7s, s11 93.0s, s03 12.8s, ALL DONE in 625s

# clf lane (existing JSONs from prior agent): s03 s04 s05 s06 s08 s10 s11 s12 s13 on humanness_therapeutic
#   (driver: same script, ds=humanness_therapeutic, reps=aac,blosum,onehot)
```

---

## 2. CACHE EVIDENCE (§V part 2 — the headline test)

Ran the SAME thing twice on `/tmp/kirby_V_similarity` (cold build, then warm re-run) for
`humanness_therapeutic`. The warm re-run reused EVERY stored config across all three precompute
tiers that had prior rows, with **NO `meta drift` line anywhere** and `0 miss` baseline reuse.

Warm-run stdout (`/tmp/kirby_V_similarity/clf_run2.log`):
```
  Precomputing hybrid searches (general, 3 reps)...
  ResultCache: 10 stored configs reused; computing only the remainder of 10
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 9 hit / 0 miss / 0 stale; hybrid_eval: 0 hit / 0 miss / 0 stale
  Precompute complete: 1/1 searches in 14s

  Precomputing hybrid searches (top-tier, 2 reps)...
  ResultCache: 5 stored configs reused; computing only the remainder of 5
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 15 hit / 0 miss / 0 stale; hybrid_eval: 0 hit / 0 miss / 0 stale

  Precomputing distance-tier (general_distance, 3 reps)...
  ResultCache: 6 stored configs reused; computing only the remainder of 6
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 24 hit / 0 miss / 0 stale; hybrid_eval: 0 hit / 0 miss / 0 stale
```

- **"N stored configs reused" printed (10/10, 5/5, 6/6)** — every cell present in the DB was reused;
  "computing only the remainder of N" with the remainder = 0 new for those tiers.
- **NO `meta drift on '<field>'` line** appeared anywhere in either run (`grep -c "meta drift"` = 0).
  This is the exact failure mode that caused the months-long stall; the new SQLite store did not
  re-reject its own rows.
- Baseline reuse `N hit / 0 miss` confirms the per-rep baseline layer also hits.

Changed-MODEL-list reuse (the core bug, optional): attempted on the warm `logd` DB by re-running
s05 with `KIRBY_LOCAL_MODELS=rf,knn,ridge,lightgbm` (adding lightgbm). The run **segfaulted
(exit 139)** during precompute — the known macOS OpenMP/TF double-load crash that the smoke env
guards against; adding `lightgbm` re-triggers it locally. This is an ENVIRONMENT limitation, NOT a
cache finding. The cache design itself puts `model` in `CONFIG_COLS` (result_cache.py:56-61), so a
new model is a new per-row key over the unchanged IDENTITY — the reuse path is structurally correct;
it just couldn't be exercised locally without tripping the OpenMP segfault. Recommend the
Cache+Determinism agent demonstrate changed-model reuse on a Linux box where lightgbm is safe.

---

## 3. DECISIONS / SHARED-CODE FINDINGS

### IN-LANE FIX (applied, gated, NOT committed)
**s03 lightgbm/mlp `{'error'}` rows under the smoke model trim** — `tests/experiments/similarity/`
- `s03_feature_concentration.py:90` hardcodes `downstream_models = ['rf','lightgbm','mlp']`.
- The local smoke driver (`_run_similarity_locally.py:42-44`) trims `find_best_hybrid.DEFAULT_MODELS`
  to `rf,knn,ridge` to dodge the macOS OpenMP segfault. The pool then has no lightgbm/mlp candidate,
  so s03 recorded `{'error': 'get_fitted_pool_hybrid: no pool candidate ...'}` for those two models.
- **Fix** (`similarity_helpers.py`, in `run_all`'s kwargs builder, gated on `KIRBY_SMOKE_N`):
  intersect the requested `downstream_models` with the models actually present in `DEFAULT_MODELS`,
  so the smoke run only asks for what exists. Server full runs (all models present) are unaffected.
- **Validated**: post-fix reg `s03/logd.json` has only the `('rf',25)` row, 0 errors.
- This is a SMOKE-CONFIG artifact, not a product bug. On the server (full DEFAULT_MODELS) s03 always
  had lightgbm/mlp available, so it never errored there.

### SHARED-CODE FINDING #M1 (molecular load path — NOT fixed; recorded per lane discipline)
`tests/experiments/_shared.py` molecular branch of `_load_data_impl` (~`_shared.py:2548-2550`):
- For molecular datasets it calls `get_molecular_representations(smiles, rep_preset='all')` and only
  filters to `rep_names` AFTERWARD — so it computes ALL ~33 molecular reps (incl. server-only
  foundation models grover/molformer/chemberta/mhg_gnn that hang/fail locally) before discarding all
  but the wanted ones. The antibody/DTA/nucleotide branches correctly pass the rep list as the preset.
- Impact locally: would hang the smoke run; worked around in the driver
  (`_run_similarity_locally.py:107-118`, smoke-only molecular-rep-fn wrap). Not a correctness bug on
  the server (it just over-computes), but it is wasted precompute on every molecular load.
- **Proposed fix (for the orchestrator/user):** in the molecular branch, pass the resolved rep list
  as the preset to `get_molecular_representations` (mirror the antibody branch) instead of `'all'` +
  post-filter. Verify no downstream code relies on the full-pool side effect.

### PERFORMANCE FINDING #P1 (distance-tier precompute on wide antibody reps — for the user)
The `top_distance` (and to a lesser extent `general_distance`) precompute tier on
`humanness_therapeutic` is **prohibitively slow locally**: with the 5600-dim `blosum`/`onehot`
antibody reps it ran **>17 min of dedicated CPU and still had not finished** (worker at full CPU,
not deadlocked, low machine load). This is what blocked the s01/s02/s07 clf gap cells — the
experiments themselves are fast, but `run_all` always does all four precompute tiers up-front and
the distance tier on wide reps dominates. Two observations for the user:
1. This same cost likely contributes to long server distance-tier precompute on wide-rep antibody
   datasets — worth a look at the distance-evaluator/allocation-scoring inner loop (knn distance on
   ~5600-dim per fold) for an O() reduction or a PCA cap on the distance tier.
2. For future LOCAL smoke gates on antibody datasets, either narrow the distance tier to the 40-dim
   `aac` rep or cap the distance-tier dim, so the clf lane is tractable. (Not done here — would touch
   shared `_shared.py`/`run_all` distance-tier wiring beyond a clean smoke gate; recorded for the user.)

### NOTE on the macOS OpenMP segfault (env, not a bug)
`lightgbm`/`mlp`/`deeplift(TF)` trip an OpenMP double-load segfault (exit 139) locally even with
`KMP_DUPLICATE_LIB_OK=TRUE`. The driver already trims `DEFAULT_MODELS` and importance methods to
avoid it. This is why the changed-model reuse demo (which re-adds lightgbm) crashed. Server (Linux)
is unaffected.

---

## Lane discipline
Edited ONLY `tests/experiments/similarity/` (`_run_similarity_locally.py` — pre-existing prior-agent
edit; `similarity_helpers.py` — the s03 smoke fix). No edits to `_shared.py`, `src/kirby/**`,
`find_best_hybrid.py`, `result_cache.py`, or other families. Nothing committed.
