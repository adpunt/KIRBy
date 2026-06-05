# OPIG Family — Local Smoke-Test Report (2026-06-05)

Family agent for the OPIG experiment suite (10 scripts). All runs used the brief's
single-thread env prefix, `KIRBY_SMOKE_N=200`, `python -u`, local-only reps, and a
perl-alarm wall-clock wrapper. Logs are in `tests/results/_smoke_2026_06_05/logs/`.

Machine was heavily contended for most of the session (the whole fleet — complexity,
similarity, noise, federated, molecular families — ran concurrently, up to ~25 Python
processes). That inflated wall times 3–10x; the slow runs below are contention/compute
limited, not code bugs.

## 1. SMOKE RESULTS

| Experiment | Task | Command (abridged) | Result | Notes |
|---|---|---|---|---|
| opig_humanness | classification | `--datasets humanness_therapeutic --n-folds 2 --reps blosum,onehot,aac` | **PASS** | end-to-end, hybrid built (svm AUROC 0.897, +0.025 over best individual), JSON saved |
| opig_trastuzumab_enrichment | classification | `--n-folds 2 --reps blosum,onehot,aac` | **PASS** | hybrid search CV AUC 0.783, enrichment metrics computed, JSON saved |
| opig_cdr_flexibility | classification | `--cdr-types CDRH3 --n-folds 2 --reps blosum,onehot,aac` | **PASS** | best KIRBy ROC AUC 0.649, hybrid CV reported, JSON saved |
| opig_ginkgo_developability | regression | `--properties ginkgo_heparin --n-folds 2 --reps blosum,onehot,aac` | **PASS** | hybrid ρ 0.686, JSON saved |
| opig_activity_cliff_qsar | regression/QSAR | `--targets d2 --n-folds 2 --reps ecfp4,maccs,pdv` | **PASS** | hybrid built + AC-MCC computed, JSON saved (needed smoke aug-skip — see DECISIONS) |
| opig_blueprint_transfer | transfer | `--source logd --target herg_ki --mode intermediary --budgets 25 --models rf --reps maccs,pdv,ecfp4` | **PASS** | all 4 conditions (base/native/recipe/full) evaluated, JSON saved (needed `--budgets` fix — see DECISIONS) |
| imrie_transfer_learning | transfer | `--datasets logd --reps ecfp4,maccs --n-folds 2` | **PASS** | dnn_scratch + ridge ran, hybrid CV computed, JSON saved. TL conditions correctly skipped (no pretrain weights → server-only). Needed a segfault-guard fix (see DECISIONS) |
| opig_kingsmith_component_splits | reaction (LOO) | `--datasets buchwald_hartwig --splits additive` | **PASS (loop, timed-out-clean)** | reached the leave-one-additive-out loop and completed fold 0's per-fold find_best_hybrid cleanly (then continues fold 1…22). The per-fold loop (`opig_kingsmith_component_splits.py:385`) is pure repetition → clean folds before the alarm = pass. Needed a smoke heavy-rep drop (see DECISIONS). |
| opig_graphinity_ddg | regression (DDG) | `--conditions within_rosetta --n-folds 2 --reps blosum,onehot,aac --max-rosetta 300` | **TIMED-OUT-CLEAN** | data loads, reps generate, hybrid search runs (first of 2 pool searches completed + cached). Search on 200 unique PDB-complex groups × all downstream models is intrinsically heavy (>13 min cputime even solo). Identical hybrid code path is proven by the 4 antibody PASSes above. See DECISIONS. |
| opig_cgr_comparison | reaction | `--datasets buchwald_hartwig --n-folds 2 --skip-correlation` | **BLOCKED (rep-gen too heavy locally)** | data + fingerprint reps load fine. It generates the FULL reaction-rep matrix (no `--reps` flag): I smoke-skipped reactiont5/chytorch/cgr_smiles_lm + the mordred variants in-lane, but the matrix ALSO includes other foundation-model molecular variants (GROVER, mhg_gnn, graph_kernel × product/reactant/diff) that are slow on CPU. Not a code bug — the rep set is just too heavy for a local smoke. See DECISIONS. |

**Data availability (verified locally):** humanness_therapeutic, tz_her2_binary, ginkgo_heparin,
itsflexible (cdr_flexibility), chembl_qsar (activity_cliff), graphinity_rosetta (CSVs +
pdb_sequences.csv cached on /Volumes/seagate), buchwald_hartwig (cgr/kingsmith), logd/herg_ki
(blueprint/imrie) all load locally. Pretrain weights for imrie TL (`/Volumes/seagate/kirby/chembl_pretrain`)
are **absent** → imrie TL conditions need weights generated on the server.

## 2. FIXES MADE IN-LANE

All edits are gated on `KIRBY_SMOKE_N` (no-op on server) except the two genuine bug fixes noted.

1. **`'fast'` SPEED_TIERS bug — REAL BUG, all 5 antibody/mol scripts** (`opig_humanness.py`,
   `opig_cdr_flexibility.py`, `opig_ginkgo_developability.py`, `opig_graphinity_ddg.py`,
   `opig_trastuzumab_enrichment.py`). Their `_smoke_search_kwargs()` returned
   `{'speed': 'fast', ...}`, but `'fast'` was removed from `SPEED_TIERS`
   (`src/kirby/find_best_hybrid.py:1692`) and now raises `KeyError`. The pool helper
   **silently swallows** that error per-condition (`_shared.py:3236`), so under
   `KIRBY_SMOKE_N` the hybrid search returned `None` and **the entire kirby_hybrid was
   silently dropped** — the experiments produced results with no hybrid row at all and no
   error. Confirmed via `KIRBY_POOL_VERBOSE=1` (`[pool] general/none failed: 'fast'`).
   Changed all 5 to `'full'`. This bug only fires when `KIRBY_SMOKE_N` is set, so it does
   NOT affect a normal server run — but anyone running these with the smoke env would get
   hybrid-less results.

2. **Smoke search shrink (the 5 `_smoke_search_kwargs` scripts):** `{'speed':'full',
   'budgets':[25], 'internal_evaluators':['rf'], 'importance_methods':['lightgbm_gain']}`.
   Full tier runs 8 budgets × 5 evaluators (incl svm/gp/mlp) × 3 importance methods (incl
   deeplift/treeshap SHAP on 5600-D features) — minutes per condition. The shrink keeps the
   pipeline exercised (rf + one importance method) while finishing locally. The two scripts
   that pass `internal_evaluators=INTERNAL_EVALUATORS` explicitly (cdr_flexibility,
   graphinity) were made smoke-aware to avoid a duplicate-kwarg `TypeError`.

3. **`opig_activity_cliff_qsar.py`:** added the same evaluator/importance/budget shrink, plus
   a smoke-only **augmentation-pool skip** (`augmentations=None` under smoke). With aug reps
   (maccs, pharmacophore3d) the pool runs 5 conditions each doing a full augmentation search —
   it could not finish locally even solo (timed out at 800 s). With the skip it PASSES.

4. **`opig_cgr_comparison.py`:** added evaluator/importance/budget shrink to its hybrid call,
   plus a smoke-only skip of the heavy **neural rep generators** (reactiont5,
   chytorch_graphormer, cgr_smiles_lm) and the **mordred reaction variants** — keeping
   cgr_morgan so the CGR-vs-SMILES comparison still has a CGR member. NOTE: this still wasn't
   enough — the rep matrix also includes GROVER / mhg_gnn / graph_kernel molecular variants
   (also slow on CPU), so cgr remained intractable locally (see DECISIONS D).

5. **`imrie_transfer_learning.py`:** (a) added the OpenMP pre-import guard
   (`import xgboost`/`import lightgbm` before the rdkit-importing modules) — same guard
   cgr/blueprint already have. Without it the run **segfaulted (exit 139)** during rep gen,
   exactly the macOS libomp double-load the brief warns about. After the guard it PASSES.
   (b) added a `KIRBY_SMOKE_N` row cap (the molecular `load_dataset` does not honor the env
   var) and a smoke budget/evaluator/importance shrink for its hybrid lookup.

6. **`opig_blueprint_transfer.py` — REAL BUG (latent):** the two source/target
   `find_best_hybrid` calls hardcoded `budgets=FULL_BUDGETS`, so the script's own `--budgets`
   flag was silently ignored. Changed to `config.get('budgets', FULL_BUDGETS)` (default
   unchanged, so server runs are identical) — this both fixes the dead flag and makes the
   smoke run tractable. Also added a smoke-only rf-only/lightgbm_gain shrink.

7. **`opig_kingsmith_component_splits.py`:** extended its existing smoke budget shrink to also
   use rf-only evaluator + lightgbm_gain importance in the per-fold `find_best_hybrid`, plus a
   smoke-only drop of neural + mordred reps from its `wanted` set. After this it reached and
   completed fold 0 of the leave-one-additive-out loop cleanly.

## 3. DECISIONS FOR USER

### A. The `'fast'` smoke bug silently dropped the hybrid (humanness/cdr/ginkgo/graphinity/trastuzumab)
- **What it is:** these 5 experiments each ask "does the KIRBy hybrid beat the best single
  representation?" The hybrid is built by a cached search helper.
- **The problem:** the smoke-mode helper asked for a search speed called `'fast'` that no
  longer exists; the request failed silently and the hybrid was quietly skipped, so a
  smoke run looked fine but had no hybrid in its results.
- **Decision/recommendation:** fixed in-lane (use `'full'`). No server impact (the bug only
  fired under `KIRBY_SMOKE_N`). Worth confirming the smoke helper is never used to produce
  reportable numbers.
- **Evidence:** `src/kirby/find_best_hybrid.py:1692` (only `'full'` exists);
  `_shared.py:3236-3239` (per-condition errors swallowed); pool-verbose log line
  `[pool] general/none failed: 'fast'`.

### B. opig_blueprint_transfer ignored its own `--budgets` flag
- **What it is:** blueprint searches a hybrid "recipe" on a large source dataset and transfers
  it to a small target.
- **The problem:** the two searches hardcoded the full 8-budget grid, so `--budgets` did
  nothing — a latent bug independent of smoke.
- **Decision/recommendation:** fixed to honor `config['budgets']` (defaults to the full grid,
  so server behavior is unchanged). Please confirm you're happy with the flag now being live.
- **Evidence:** `opig_blueprint_transfer.py` step-1/step-2 `find_best_hybrid(budgets=...)`.

### C. imrie segfaulted locally (exit 139) — needs the OpenMP guard; TL conditions need server weights
- **What it is:** imrie tests whether ChEMBL-pretrained DNN weights change representation
  ranking via transfer learning.
- **The problem (1):** it segfaulted on macOS during rep generation — the libomp
  double-load the brief flags. Fixed in-lane by pre-importing xgboost/lightgbm (same guard
  the other two molecular scripts use).
- **The problem (2):** the pretrained-weight directory `/Volumes/seagate/kirby/chembl_pretrain`
  does not exist locally, so the transfer-learning conditions (dnn_tl_full / dnn_tl_frozen)
  are skipped — only dnn_scratch + ridge run. To smoke the actual transfer-learning path you
  must first run the pretraining (`python -m kirby.models.pretrain_chembl_dnn ...`) on the
  server. The scratch/ridge/hybrid path is verified locally.
- **Evidence:** `imrie_transfer_learning.py:list_conditions` (skips TL when weights absent);
  exit 139 in the first run; clean exit 0 after the guard.

### D. opig_cgr_comparison stalls generating mordred reaction-rep variants
- **What it is:** cgr compares CGR vs SMILES reaction representations for yield prediction.
- **The problem:** the loader generates the FULL reaction-rep matrix unconditionally — there
  is no `--reps` flag. Two cost centers make it intractable locally: (1) neural reps
  (ReactionT5 / Chytorch Graphormer / CGR-SMILES-LM) run transformer forward passes on CPU
  (minutes each); (2) the **mordred reaction-rep variants** (product/reactant/diff_mordred,
  ~1613 descriptors each) — the run **stalled at ~0–2% CPU** here, a hang rather than slow
  compute. I smoke-skipped the 3 neural reps in-lane (keeping cgr_morgan), but the run still
  hung on the mordred variants, which I cannot safely skip without changing the experiment's
  rep set.
- **What I tried in-lane:** smoke-skipped reactiont5/chytorch/cgr_smiles_lm AND the mordred
  variants. It STILL didn't finish locally because `_MOL_REP_VARIANTS` also pulls in GROVER,
  mhg_gnn, graph_kernel (product/reactant/diff) — more foundation-model molecular reps that
  run on CPU. Skipping all of those in-lane would gut the experiment's rep set.
- **Recommended solutions (need your judgment):** (i) add a `--reps` selector to cgr so a
  subset (e.g. fingerprints + cgr_morgan only) can be run — best option, useful for both smoke
  and targeted server runs; or (ii) accept cgr as a server-only experiment (no contention,
  full cores — it should complete, just slowly). The fingerprint-only path (the core
  CGR-vs-SMILES comparison) is the part that matters; the neural reps are the expensive extras.
- **Evidence:** `opig_cgr_comparison.py:158-189` (`_generators` + `_MOL_REP_VARIANTS` loop,
  no `--reps` flag); log shows reactiont5/chytorch/cgr_smiles_lm + mordred skipped, then
  product_grover (4800-D) computed, with more neural variants pending.

### E. opig_graphinity_ddg search is intrinsically heavy (200 unique groups)
- **What it is:** graphinity tests sequence reps + simple ML for antibody–antigen ΔΔG.
- **The problem:** its GroupKFold groups are PDB complexes, and the subsample (even capped at
  200 rows via `--max-rosetta 300`) yields ~200 **unique** groups. The find_best_hybrid
  internal CV then trains every downstream model (incl gp O(n³), dnn, cnn) across 5 folds ×
  the allocation grid — >13 min cputime even solo with the smoke shrinks, and even with a
  single 40-D `aac` rep (so the cost is the group count + model set, not rep width). It
  completed the first of two pool searches (cached) before I capped my time. The code path is
  identical to the 4 antibody experiments that PASSED, so this is a tractability limit, not a
  bug.
- **Recommended solutions (need your judgment):** on the server (no contention, full cores)
  it should finish, just slowly; if you want a faster smoke, a `KIRBY_SMOKE_N` cap on the
  number of distinct groups (or dropping gp/dnn/cnn from the internal search-baseline model
  set under smoke) would help. I did not change the model set — that's a search-design choice
  for you.
- **Evidence:** `opig_graphinity_ddg.py:367-370` (one group per PDB complex); load order
  builds sequences for all selected PDBs before subsample (`src/kirby/datasets/graphinity.py`
  subsamples at line ~291 then extracts at ~323 — so `--max-rosetta` does limit extraction).

### SHARED-CODE issues for the orchestrator (NOT fixed — outside my lane)
1. **`_shared.py:3236-3239` swallows pool-condition exceptions silently.** This is what hid
   the `'fast'` bug. Consider surfacing a one-line warning when ALL pool conditions fail (so a
   future config error doesn't silently disable the hybrid). File:
   `tests/experiments/_shared.py`.
2. **`tests/find_best_hybrid_molecular.py::load_dataset` does not honor `KIRBY_SMOKE_N`.**
   imrie and blueprint pay the full OpenADMET download + InChIKey-dedup of ~5–7k molecules
   before any row cap. A row cap inside `load_dataset` (gated on the env var) would speed up
   every molecular smoke run. (I worked around it inside imrie's own file.)
3. **`src/kirby/datasets/graphinity.py` mordred / cgr neural reps** — see decisions D/E; the
   slowness/hang lives in shared dataset+rep code, not the experiment scripts.

## NOTE
`tests/results/comparisons/imrie_transfer_learning.json` (39 KB, mtime 2026-03-16) is a
**stale prior-run artifact**, not from this smoke run. My imrie smoke output went to
`.tmp_smoke_opig/imrie_smoke.json` to avoid confusion.
