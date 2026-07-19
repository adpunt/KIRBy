# find_best_hybrid — Rebuild Roadmap (2026-07-15)

Working plan. Nothing here is run yet. This document is the plan to *investigate*, *cut*,
then *rebuild*. Every verification listed here is a NEW local test — we do not trust any
existing result in `tests/results/`, `universal_survey`, `c01`, or any prior `.json`.

---

## 0. The goal, in your words

For one dataset:

1. Find the **best representation × best model** pair. That single score is **the target**.
2. Build the **best hybrid** for **that same model**.
3. Compare **best-rep × model** vs **best-hybrid × model**.

No medians. No averages. No suboptimal reps, no suboptimal hybrids, no suboptimal models
in the comparison. The comparison is deliberately "unfair" (best vs best) — that is the
point of the study, not a bug to be corrected.

Test datasets while building (confirmed 2026-07-16, chosen from survey n + signal, small/weak
ones ruled out). **Local is the default — server is a slow last resort** (major outages, low fair
usage, a "must-run" could take days), so the primary set is deliberately light enough to run
locally:

- **herg_ki** — n=1,412, RMSE, reps competitive (best hybrid stress test). Molecular.
- **logd** — n=5,028, RMSE, large + clean. Molecular.
- **flab_thermo_tresanco** — n=672, ρ=0.805, strongest-signal antibody, light enough for local.
- **buchwald_hartwig** — reaction yield, cgr_morgan reps (RDKit-cheap, local-friendly). **This is
  the proof-of-concept dataset** (see below) — a known, large hybrid win worth reproducing.

**The buchwald_hartwig proof of concept (what we want to see MORE of):**

| dataset | best single rep | best hybrid | win |
|---|---|---|---|
| buchwald_hartwig | cgr_morgan (ρ 0.833, rmse 15.08) | ρ 0.898, rmse 11.75 | **−3.3 rmse = −22%, ρ +0.065** |

This win came from **top-5** — concatenate the best features roughly equally across the top 5 base
reps, which on this dataset each carry *different types of information* (true complementarity,
Scenario 1). It is the clearest existing evidence the method works. Caveat: the *current* top_5
allocation mechanism is not how we want to do it (we have the top-N knob now) — the result is the
target, not the implementation.

**Optional scale-up (server only, run only if something MUST be tested at large n):**
flab_expression_koenig — n=4,275, ρ=0.715; 4,275 sequences × protein embeddings is server-heavy,
so it is not part of the routine local loop.

The real system runs on ~33 datasets; these are the local proving ground.

---

## 1. What the code does today (verified by reading, 2026-07-15)

| Piece | Current state | File:line |
|---|---|---|
| Outer CV | 5-fold GroupKFold / StratifiedGroupKFold | `find_best_hybrid.py:5930, 5992` |
| Inner search | fresh **5-fold** on the ~80% outer-train slice (NOT "4 of 5 outer folds") | `find_best_hybrid.py:6074`, `SPEED_TIERS['full']['n_folds']=5` :2010 |
| Structure | nested **5 × 5** | — |
| Budget cap | `filtered = [b for b in budgets if b <= n_samples]` — hybrid capped at n; baseline gets full native width | `find_best_hybrid.py:4816` |
| Allocation | 9 searched, **8 of 9 are performance_weighted variants** (~35 named total) | `hybrid.py:53-119`, default set `find_best_hybrid.py:2000-2010` |
| Importance | 8 methods exist; default 3 = `deeplift, lightgbm_gain, treeshap`; `mutual_info` was removed; **none are redundancy-aware** | `hybrid.py:2382`, defaults `find_best_hybrid.py:2000` |
| Rep pools | per-dataset dict `DATASET_REPS` with `general` / `top_tier` / `augmentation` | `complexity_helpers.py:94` |
| Ridge | present in `DOWNSTREAM_MODELS`, `DISTANCE_EVALUATORS`, model registry, internal-evaluator default | `_shared.py:326,297`; `hybrid.py:555,1216,3456`; `find_best_hybrid.py:1487` |
| Metric direction | centralised: `_resolve_metric` → `(fn, lower_is_better)`; `_score_to_comparable` = `-raw if lower_is_better else raw`; paired per-fold delta direction-corrected | `find_best_hybrid.py:1207, 1232, 6225` |
| Reported number | `delta_mean_of_folds` = mean of paired per-fold (hybrid − best-single), direction-corrected | `find_best_hybrid.py:6252` |

The metric-direction handling **reads correct** at every point checked. It is not currently
suspected — but it is fragile (one wrong `lower_is_better` breaks everything), so it gets an
explicit end-to-end gate below.

---

## 2. The core diagnosis (why the automation broke what worked by hand)

**The two things holding this whole experiment back are (1) limited hybrid budgets and (2) poor
rep-pool quality — too many reps all showing the same signal.** The reps may well contain the
right signals; the problem is there are 30+ of them and they duplicate each other, so no allocation
strategy can give the good rep enough room. These two compound and are the headline of everything
below.

The version you remember working = commit `c4afb0e`: **2 hand-picked distinct reps × ~100
features each**. The automation replaced "N features per rep" with "one global budget, split
across the whole pool by performance." How they bite:

**(a) The pool is too big and near-tied.** On a 29-rep pool scoring 0.17–0.58 (a 3.4× range),
performance_weighted allocation of a 100-feature budget gives the **best rep ~5 features** and
everything else 2–5. That is dust. This is the dominant lever.

**(b) The budget is capped at n_samples** while the baseline gets its full native width
(`find_best_hybrid.py:4816`). On small datasets the hybrid can never reach the width where it
would win; on large datasets the cap throws away the most valuable compute.

**The two together, made concrete (your measured example):** the best rep is **1,280 features
wide**. With a 30+ rep pool, a budget of **50 gives the hybrid only 7 of the best rep's features;
a budget of 250 gives only 16.** The good signal is 1,280-dimensional and the hybrid sees 7–16 of
it. **That will not work, by any allocation strategy — full stop.** The fix has to attack both
sides: far fewer, non-duplicate reps (so the budget isn't shattered) AND a budget large enough to
carry a real slice of a 1,280-wide rep. How exactly it's done is open; that it must happen is not.

Ordering of the levers (biggest first), to be re-measured locally, not assumed:

1. **How many reps enter the hybrid** — the single biggest driver of how many features the best
   rep gets. This is controlled by the **top-N knob (N = 2, 3, 5, more)**, a replicate parameter
   of the allocation categories — NOT by hard-pruning the pool to a fixed size. Decorrelation
   (below) removes only genuinely duplicate-signal reps; top-N then decides how many of the
   remaining, rank-ordered by base performance, actually feed the hybrid.
2. **Concentration** — low-temperature / rank-based allocation concentrates budget on the
   winner within whatever N reps are in play.
3. **Budget** — uncapping and going to ~500–1000 (or past the widest rep's width) matters, but
   mostly once N is small enough that the budget isn't shattered.
4. **Normalisation** — rescaling scores before allocation helps *modestly*. It is a real fix
   for the "low absolute metric starves the best rep" problem, but it is **not** the main
   lever; it cannot rescue a large near-tied pool on its own.

Everything below is built to *measure* this ordering per dataset and cut accordingly.

---

## 3. Method: TEST what exists → CUT to the best → only then rebuild

The governing rule, in your words: **do not keep adding things.** Test the components that
already exist, find the best settings and the things actually worth searching, and delete
everything else. The end state is a *smaller* search, not a bigger one. **At every step the
default action is to remove a component, not add one.** There are exactly **two sanctioned
exceptions** (your call):

1. **Increased budget** — the one axis we deliberately grow (uncapped, past the widest rep).
2. **Feature-importance scoring** — a genuine investigation into *adding or replacing* the scorers,
   because the existing ones are near-interchangeable and none is redundancy-aware. Candidates come
   from the research in Appendix B and enter as *test-then-keep-or-cut*, not permanent extras.

Everything else is subtract-only.

We do NOT touch `find_best_hybrid.py` / `hybrid.py` until the tests are done and the structure
is agreed.

How the tests run:
- All tests are **new** and **local**. No reuse of any prior result file.
- **Compute is not a constraint for these tests.** This roadmap is written to be handed to
  several separate chats, each taking one workstream — the testing is not meant to happen in one
  sitting. Be thorough; don't cut corners to save time.
- Breadth over repetitive depth: cover many settings/datasets rather than over-polishing one.
- **Ridge removed everywhere** before any test runs.

---

## 4. Phases

**Ordering rule you set:** cut the DATASETS first, then cut the REP POOLS — and do both
**per-dataset, together, without generalising.** I tend to generalise across datasets; this is
the one place not to. One dataset carries ~29 reps and several legitimate allocation strategies
break *only* in that high-rep regime — so **the rep spread across datasets is a prime suspect
alongside the allocation logic itself**, not a detail. Phases 1 and 3 are a back-and-forth
between us looking at each dataset individually, not an automated sweep.

### Phase 0 — Foundations (blocking gates)

**0.1 — Remove ridge entirely.** Model registry, `DOWNSTREAM_MODELS`, `DISTANCE_EVALUATORS`,
internal-evaluator defaults, and every fallback that defaults to `'ridge'`
(`hybrid.py:3437-3456`, `1216`, `555`, `262`; `find_best_hybrid.py:1487,1916,3540,3762`;
`screen.py:37,42`). Internal-evaluator default → `lightgbm` (decided).

**0.2 — Metric-direction end-to-end audit.** One explicit test that a lower-is-better dataset
(herg_ki or logd, RMSE) and a higher-is-better dataset (flab_thermo_tresanco, ρ) both produce a
positive `delta_mean_of_folds` when the hybrid genuinely wins, and negative when it genuinely
loses. Confirms no sign flip survives end to end.

### Phase 1 — Cut the datasets FIRST (collaborative, per-dataset, no generalising)

Before touching reps, settle which datasets are in. Prior scoping exists in
`FINAL_DATASET_LIST_2026_07_09.md` (33-set, 25 proposed cuts) but it is **not auto-accepted** — we
review it together, dataset by dataset. This gates everything downstream: rep pools, models, and
the whole search are decided per surviving dataset. No decision is generalised from one dataset to
another.

### Phase 2 — Base reps: do they work, and what is the target (per dataset)

**2.1 — Base-rep validity gate (the SIGNAL gate).** For each surviving dataset, score every base
rep with its real model under our light local CV. If the best base rep does not reproduce a sane
standalone score, the hybrid has nothing to pick up and every downstream number is meaningless —
this is the most important gate. (Prior "validity scare" was untuned ridge + dead antibody reps;
removing ridge is part of why this is worth re-checking cleanly.) This step also produces each
rep's per-fold **prediction vectors**, which Phase 3 needs for prediction-correlation
decorrelation.

**2.2 — Best rep × best model + top-2 models (fresh, local).** Rank all reps × all models (ridge
removed) per dataset. Output: the best rep × best model = **the target**, plus the **top-2 models**
for that dataset. Only those top-2 models are carried forward — no dataset tests every model.

Licenses the cut: **model axis → 2 per dataset.**

### Phase 3 — Cut the rep pools, per dataset (collaborative, no generalising)

The second cut, and the co-prime-suspect with allocation. Kill the `general` vs `top_tier`
distinction — there is one pool per dataset. Within it: (a) rank reps by base performance, (b)
remove only reps that are *duplicate signal* by **prediction correlation** (from 2.1's prediction
vectors) — we are deleting duplicates, NOT trimming to a fixed count. How many reps feed the hybrid
is then set by the **top-N knob (N = 2, 3, 5, more)** carried by the allocation categories in
Phase 4. N is a searched/replicate parameter, not a hard cap.

Done **per dataset, with you.** The high-rep dataset (~29) is exactly where allocation strategies
break, so its pool needs individual attention, not a rule copied from caco2. This is the
back-and-forth, not a sweep.

Licenses the cut: **general/top concept deleted; one decorrelated pool per dataset; rep count
chosen by top-N rather than a fixed prune.**

### Phase 4 — Correlation at the remaining search layers + allocation reality-check + augmented

Correlation analysis is not one study — it runs at **every layer of the search**, because
duplication anywhere wastes budget and inflates the search. The layers, and where each is handled:

1. **Reps** — drop duplicate-signal base reps (done in Phase 3).
2. **Features** — inside the selected set, don't hold the same feature-signal many times
   (redundancy-aware selection, 4.1).
3. **Importance methods** — do the methods rank features the same way? Cut the ones that agree
   (4.1).
4. **Allocation strategies** — do the strategies produce the same allocations/scores? Cut
   duplicates (4.2).
5. **Downstream models** — are the top-2 models' predictions so correlated that the second adds
   nothing? (checks whether "top-2" is even worth carrying; ties back to Phase 2.2).
6. **Augmented features vs the best rep** — is an augmented/borrowed feature already captured by
   the dominant rep? (4.4).

The measure of "duplicate" is **prediction correlation** everywhere — two things are redundant
when they make the same predictions — **not raw feature correlation**.

**4.1 — Feature-importance methods: test what exists, then investigate replacing/adding.** First,
score the existing methods against each other by prediction correlation — the repo's own note says
they give near-identical rankings (`EXPERIMENT_OVERVIEW.md:402`), so most collapse to one. Then —
this is an explicit EXCEPTION to "subtract, don't add" — evaluate the researched candidates
(Appendix B): a redundancy-aware selector (mRMR), an all-relevant wrapper (Boruta), a
significance-based SHAP selector (shap-select), and a neural selector (LassoNet) that matches our
DNN model. Keep the best 1–2 overall; one of them must be redundancy-aware.

Licenses the cut: **importance axis → 1–2 methods, at least one redundancy-aware.**

**4.2 — Allocation strategies: test what exists, cut duplicates.** Measure which of the existing
strategies actually produce *different* allocations / scores. Collapse the ~9 into a few genuinely
distinct categories. **Top-N (N = 2, 3, 5, more) becomes a replicate parameter of each category,
not a standalone strategy.** Include the greedy/complementarity-aware allocators already written but
never registered (`allocate_greedy_forward` etc.) as candidate categories — existing code, not new
additions.

Licenses the cut: **allocation axis → few categories × top-N replicate knob.**

**4.3 — Allocation reality-check: what does the strategy ACTUALLY do?** The central question you
raised — whatever a strategy claims in theory, look at what it produces in practice. For a dataset
+ best model, run each surviving strategy and report the *actual* output: which reps contributed
features, how many each, and — the key one — **are the best rep's genuinely informative features
present in the final hybrid, or did the per-rep quota truncate them out?** (Today feature selection
is a per-rep quota, never a global ranking — `find_best_hybrid` picks top-N *within each rep* by
that rep's own scores, so the best rep is always truncated to its slice and diluted by weaker
reps.) This tells us whether normalisation, top-N, or a global ranking is what actually restores
the best rep's signal — instead of guessing.

Licenses the cut: keeps only allocation strategies that demonstrably put the best rep's real signal
into the hybrid.

**4.4 — Augmented features: first-class, and SOLIDIFIED in the pipeline (not an afterthought).**
Augmented features already exist per dataset (`DATASET_REPS['augmentation']`) but are secondary and
loosely wired today. Two jobs: (i) make the augmentation-level hybrid search a **proper, integrated
stage of the pipeline** — clearly defined, cached, and searched upfront, not bolted on; (ii) cover
the two use cases:

- **(A) Individual-style augmented features** — small feature sets that don't deserve to be a
  standalone base rep, tried directly in the hybrid.
- **(B) Best-rep-plus** — take the single best rep and add either (a) augmented features, or (b) the
  **top-5 individual features from each of the other reps**.

Together with base-rep allocation, the hybrid then covers **both** scenarios explicitly:

- **Scenario 1 — signal spread across several reps** (multi-rep complementarity): the classic case.
- **Scenario 2 — one dominant rep, slightly improved** by a few augmented or borrowed features: the
  case the current global-budget split destroys, and the one 4.4(B) targets directly.

### Phase 5 — Budget & normalisation

**5.1 — Uncap the budget.** Remove `find_best_hybrid.py:4816`. Sweep budget independently of
n_samples, up to/past the widest rep (uncapped — decided).

**5.2 — Normalise per-rep performance before allocation.** So a low absolute metric doesn't starve
the best rep. Secondary lever. **Test BOTH schemes (min-max across reps AND rank-based) in the same
sweep, then cut to one** — same test-then-cut pattern as everywhere else. Test 4.3 may show
normalisation isn't the fix at all, in which case both are dropped.

**5.3 — Global-concat scoring for top-N only.** For the top-N reps only: concatenate their features
into one matrix, compute ONE global importance ranking, take the top D (D = budget). Concatenation
is a scoring device only — **not** registered as its own representation.

**5.4 — SIDE EXPERIMENT (test only, not a commitment): PCA-compress the residual.** Idea: spend
part of the budget on the best individually-selected features, then spend the *remaining* budget on
**PCA components of the leftover features of the best reps** — i.e. instead of discarding the
weaker signal, compress it into a few components so it isn't lost. Concretely: budget D split into
D1 selected top features + D2 PCA components of the not-selected features from the top reps. **Just
test whether this retains weaker signal usefully — do not properly implement it yet.** Note this
runs against the standing "importance-filtering over PCA" preference, so it is explicitly a probe,
not a default; keep it if and only if it clearly beats plain selection at the same total budget.

### Phase 6 — CV structure (falls out of the design; not a separate study)

Nested cross-validation is the standard method for one situation only: when you pick settings by
searching, the winning score is optimistically biased and nested CV removes that bias. So the
choice is simple and follows from how small Phases 2–3 make the search:

- **If a real search remains** (top-N × a few allocation categories × budget sweep) → keep nested
  CV, because we are selecting. On small/mid datasets, reduce the inner folds (e.g. 5→3) — a
  normal practical tweak, not a research question.
- **If the search collapses to essentially one fixed setting** → nothing is being selected, so
  there is no bias to remove and **plain k-fold is already honest**. The 5× nested overhead
  disappears.

No bespoke stability diagnostic, no special experiment. The fold decision is a consequence of the
design, made once we can see how small the search got.

### Phase 7 — Rebuild find_best_hybrid (discussion-gated, not started)

Only after 0–6 AND an explicit go-ahead. This is a destination, not a next action — no code is
modified before we have discussed and agreed the structure. The rebuilt module would consume: the
cut dataset list, per-dataset top-2 models, one decorrelated pool per dataset with rep count set by
the top-N knob, a few allocation categories, 1–2 importance methods (one redundancy-aware), a
solidified augmentation stage, an uncapped budget sweep, and the agreed CV structure. Target: a
search space small enough that pruning is unnecessary and honesty is cheap.

---

## 5. Decisions

### Decided (2026-07-15)

- **Budget ceiling:** uncapped — sweep budget independently of n_samples and of baseline width,
  up to and past the widest rep. Remove `find_best_hybrid.py:4816`. (Phase 5.1)
- **Internal-evaluator default (replacing ridge):** `lightgbm`. (Phase 0.1)
- **Ridge:** removed everywhere — no internal evaluator, no downstream model, no distance
  evaluator, no fallback. (Phase 0.1)
- **No hard-pruning the pool to a fixed size.** One decorrelated pool per dataset; the number of
  reps that feed the hybrid is set by the **top-N knob (N = 2, 3, 5, more)**, not a fixed count.
  Decorrelation only removes duplicate-signal reps. (Phases 3, 4.2)
- **Redundancy measure = prediction correlation** — two reps/features/methods are "duplicate"
  when they make the same predictions. NOT raw feature correlation. (Phases 3 & 4, all layers)
- **CV structure follows from the design, not a separate study.** Nested CV only if a real search
  remains; plain k-fold if the search collapses to a fixed setting; reduce inner folds on small n
  as a normal tweak. (Phase 6)
- **Compute is not a constraint for the tests; the roadmap spans multiple chats.** Each workstream
  can be handed to a separate session and run thoroughly.
- **Subtract, don't add.** Default action at every layer is to remove a component. Only genuine
  capability gaps may add a candidate, and only to test-and-keep-or-cut.
- **No code changes yet.** The rebuild (Phase 7) is a destination. Nothing in `find_best_hybrid.py`
  / `hybrid.py` is touched until the design is discussed and agreed.

### Also decided (2026-07-16)

- **Normalisation: test BOTH schemes** (min-max across reps AND rank-based) in one sweep, cut to
  one. Test 4.3 may drop both. (Phase 5.2)
- **Feature-importance candidate set (Phase 4.1):** test `lightgbm_gain` (marginal baseline) + mRMR
  + shap-select, with Boruta / LassoNet as backups; cut to 1–2, ≥1 redundancy-aware. See Appendix B.
- **Test datasets (confirmed):** primary (local) = **herg_ki** (1,412, RMSE), **logd** (5,028,
  RMSE), **flab_thermo_tresanco** (672, ρ=0.805), **buchwald_hartwig** (reaction yield; the
  proof-of-concept win, cgr_morgan reps are RDKit-cheap). Optional server-only scale-up =
  flab_expression_koenig (4,275, ρ=0.715). caco2/adme_hlm/abagym_EGFR dropped as testbeds.
- **Server is a slow last resort** (outages + low fair usage → days). Design tests to run LOCAL;
  only genuine must-run-at-scale items go to the server.
- **Rep decisions are dataset/domain-specific — do NOT generalise.** When running correlation
  studies, it is fine (and useful) to pull in extra datasets purely to get more data points for
  the correlation estimate; but the resulting rep/allocation choices are made per dataset/domain.

### Session log — 2026-07-16 (Chat 1: foundations + target)

- **Ridge removal DONE** on branch `ridge-removal-and-hybrid-foundations`. Removed as a downstream
  MODEL (model registry, `DOWNSTREAM_MODELS`, the two `'models'` config lists in `_shared.py`,
  `screen._default_models`), as an internal EVALUATOR (`hybrid._evaluate_allocation` ridge branch
  deleted; defaults on `_evaluate_allocation` / `allocate_greedy_feature` /
  `allocate_performance_weighted` → `lightgbm`; the `'match'` fallbacks → `lightgbm`), as a DISTANCE
  evaluator (`DISTANCE_EVALUATORS`), and all string fallbacks (xgboost ImportError fallback → `rf`
  to break a lightgbm↔xgboost cycle). Verified: `get_model('ridge')` now raises; `DEFAULT_MODELS`
  ridge-free; both allocator eval-defaults = lightgbm; all 5 files import clean.
- **Two Ridge uses LEFT PENDING a decision** — both are measurement/study roles, NOT hybrid models:
  (1) `_shared.py:~2013` Ridge computes the **L1 linearity complexity metric** (a dataset
  descriptor); (2) `_shared.py:~1648` `get_model_family_variants` uses Ridge as the **linear model
  family being studied** in c15 (ML-theory depth). Removing either changes what those experiments
  MEASURE (and their cached values). Fork: swap to a non-Ridge linear probe (LinearRegression/Lasso)
  or leave as-is. NOT touched.
- **buchwald_hartwig pool expanded** (user decision): `cgr_morgan` added to `general` + `top_tier`
  (it is the proof-of-concept best single rep, ρ0.833, previously only in bh_doyle_plates);
  `product_maccs` + `reactant_maccs` added to `augmentation` (bh_doyle reps that ARE wired in
  `_shared._rxn_generators` for the buchwald load path). **OPEN FORK:** the bh_doyle reactant_/diff_
  MOLECULAR-rep variants (reactant_avalon, diff_mhg_gnn, reactant_grover, reactant_mordred,
  diff_mol2vec, reactant_mol2vec, diff_avalon, diff_grover, reactant_mhg_gnn, diff_mordred) are the
  likely SOURCE of the top-5 complementarity that produced the ρ0.833→0.898 win, but they are NOT
  wired into `_rxn_generators` — computing them needs `create_reaction_rep_variant`
  (`representations/reaction.py:1346`) registered. Decision needed before the buchwald proof-of-
  concept can be fully reproduced.
- **Internal-evaluator low-impact claim → RE-VERIFY before cutting (user, 2026-07-16).** Prior
  measurement (`project_knob_sensitivity_2026_07_16`, spread ~0.003 = noise) was on the now-DROPPED
  caco2/adme_hlm testbeds. User: happy to default to the lowest-compute evaluator and cut the
  evaluator axis from the search — **but ONLY if the ~0-impact result reproduces on the new
  testbeds** (herg_ki/logd/tresanco/buchwald). Verification queued for this session's Phase-2 runs
  (measure honest-delta spread across the evaluator options, one knob varied). Fixed default is
  lightgbm regardless; the question is whether the axis can be dropped from any remaining search.
- **Feature-importance (user, 2026-07-16):** user independently reproduced the "scorers give
  near-identical rankings" result — confirms the Appendix-B premise. User is now investigating NEW
  algorithms beyond the current set; hold the Phase-4.1 bake-off candidate list open pending what
  they bring.
- **Local data access SETTLED:** rep matrices for the 4 test datasets reach the machine by a
  one-time **file pull from the server** (no server jobs) — the user can pull data/reps but is
  currently blocked on running jobs. buchwald_hartwig is already ~complete locally; herg_ki / logd /
  flab_thermo_tresanco need their heavy reps pulled (only ecfp4/maccs/pdv and aac/blosum/onehot are
  local). Land in `~/.cache/kirby/rep_cache/<ds>/` (default load path) or seagate + `KIRBY_CACHE_DIR`.

### Session log — 2026-07-18 (Chat 1 cont.: consolidation → dataset cut → validity batch)

**Phase status now:** 0 DONE · 1 (dataset cut) ~DONE, validity batch running to finalize · 2.1
(validity) IN FLIGHT on server · 3–7 not started. **Dataset-side decisions live in
`REP_PRUNING_LOG.md`** (the running log — read it, don't summarise from memory).

- **ONE BRANCH.** The `ridge-removal-and-hybrid-foundations` branch was a mistake (the user's server
  binds jobs to a single branch — see [[feedback_one_branch_only]]). Everything consolidated onto
  **`similarity-metrics-study`** (cherry-picked the ridge commit, deleted the stray branch, nothing
  lost). ALL work stays on this one branch from now on.
- **Ridge measurement uses RESOLVED (user):** L1 linearity metric → `LinearRegression`; c15
  linear-family → `Lasso` (LinearRegression has no capacity knob). **Nucleotide exception:** ridge is
  kept for NUCLEOTIDE datasets ONLY (user has a real history with ridge being force-defaulted; it is
  gone everywhere except nucleotide). Registry re-add for the hybrid stage still PENDING; the
  validity harness already applies ridge for nucleotide directly.
- **buchwald variants WIRED:** `create_reaction_rep_variant` registered in `_shared._rxn_generators`
  (generic reactant_/product_/diff_<molrep> handler); pool mirrors bh_doyle's full applicable set.
  The open fork is closed.
- **Local data staged + VERIFIED:** all 4 test datasets on seagate
  (`export KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache`), every rep's row-count matches labels.
- **Phase 1 dataset cut: 60 → 43.** Cut 15 nucleotide (curated to 6 keepers — weak signal + coverage
  inflation), `flab_immunogenicity_marks`, `flab_binding_phillips_h3`. **Key finding:** single-WT
  combinatorial libraries (flab phillips/adams/expression, tz_her2) have **no clean grouped split** —
  identity-clustering leaks (all seqs >90% identical) → the doc was right. Kept on `edit_distance`
  with a stated caveat (user). phillips_h3 also SATURATED (aac AUROC 0.98 while leaking) → cut. DTA
  is KEPT (kiba/bindingdb real signal; davis pending). `cdr_flexibility` is a diverse panel →
  identity-cluster fix valid (~0.67 honest), but needs a `load_data` path (PENDING).
- **CUT criterion (user):** cut only on genuinely BAD numbers OR no headroom (saturation). A broken
  split is NOT grounds to cut — fix it (literature standard) and re-judge on the honest number.
- **Phase 2.1 validity gate LAUNCHED:** `tests/validity_gate.py` (per-dataset base-rep validity,
  lightgbm + ridge-for-nucleotide, metric spearman/MCC, reads KIRBY_CACHE_DIR) + array script
  `tests/slurm_scripts/run_validity_gate.sh`. Submitted to server: **job 12587101, medium, 42 tasks
  (roster minus cdr_flexibility)**. Output → `$KIRBY_CACHE_DIR/validity/<ds>.json`. FAILs = cut
  candidates for the user → **final dataset list**.
- **Phase 3 (rep pool cut) METHOD, agreed with user:** (a) redundancy = **Spearman correlation of
  PREDICTION vectors**; (b) confirm with a **PAIRWISE concat test (never >2 reps at once** — user
  constraint, growing greedy = compute blowup); (c) **NO decision thresholds baked in** — the harness
  emits the numbers, the user decides; (d) target **≤10 reps for `general`, `top_tier` concept
  removed**. Pairwise concat is too slow locally → runs as a server batch on the survivors. Harnesses:
  `scratchpad/base_rep_validity_vectors.py`, `scratchpad/concat_pairwise.py`,
  `scratchpad/test_split_fixes.py`.
- **DONE 2026-07-19 (commit `2f44968` + follow-up):** (1) cdr_flexibility `load_data` path wired
  (CDR3-loop identity split @0.80 = option A, in `_shared.py`; now in the validity gate, roster 43);
  (2) ridge-for-nucleotide registry change (tier `extra`, injected only for `nt_*/gb_*`, unblocked in
  `dataset_policy.py`); (3) SLURM roster consolidated — `submit_all_datasets.sh` is now THE single
  submitter (de-cut + ginkgo/nucleotide fixed); 21 stale submitters deleted, plus dead docs
  `SERVER_RERUN_COMMANDS_2026_06_05.md` / `FINAL_DATASET_LIST_2026_07_09.md`. See
  [[project_single_submit_source_of_truth]]. **HELD:** `submit_missing_surveys.sh`,
  `submit_rerun_wave.sh` (kept per user).
- **Step-1 processing:** `tests/process_validity_results.py` aggregates `$KIRBY_CACHE_DIR/validity/*.json`
  into the dataset keep/cut table + per-rep rankings. **Still to build:** the step-2/3 rep-pruning
  batch (prediction-correlation + pairwise-concat → ≤10 reps) + selection-method sign-off.

---

## 6. Cache safety — MANDATORY for every change

Almost every change in this roadmap alters *what gets computed*, and the cache **does not key on
all of it**. Getting this wrong means stale results served as "DONE" — silently, with no error.
Recomputation is fine and expected; **silent staleness is the failure to prevent.**

**The rule (locked):** representations are **NOT** in the cache key, and there is **NO in-cache
versioning**. So a rep-pool change, a code change, a fold change, a metric-direction change, or a
model-set change does **not** necessarily change the cache key — the DB can hand back an entry
computed under the old setup. Therefore: **any such change ships with explicit, scoped
delete-instructions for the affected cache entries.** No change lands without them.

**What each change invalidates (delete before re-running the affected datasets):**

| Change in this roadmap | What goes stale → must be deleted |
|---|---|
| Remove ridge (Phase 0.1) | every search / baseline / fold cache that included ridge, + `result_db/<ds>.db` + result JSONs, for affected datasets |
| Cut / change a rep pool (Phase 3) | reps aren't in the key → **full wipe + rerun** for that dataset's search + baseline caches |
| Change folds / n_folds (Phase 6) | fold cache + `result_db/<ds>.db` + result JSONs — the **groups_hash blindspot**: a fold change may leave the hash unchanged, so the status tool falsely reports DONE |
| Uncap / change budgets (Phase 5) | search caches for the affected budgets |
| Change allocation / importance set (Phase 4) | search caches (the blueprint changes) |
| Metric-direction touch (Phase 0.2) | anything that stored a directional score |

**How the instructions must be written (your rules):**
- **Scoped, never blanket.** Name the exact datasets and the exact paths — `result_db/<ds>.db`,
  the dataset's search/baseline/fold cache entries, the dataset's result JSONs. Never a blanket
  `rm -rf` of the cache root. Blanket deletion is forbidden.
- **You run them, on the server.** Claude has no server access; deletions are pasted commands for
  you, triple-checked, one dataset at a time — never a bulk sweep (a reckless bulk delete has cost
  a week of queue priority before).
- **Delete → confirm gone → recompute.** After any fold/label/rep change, deleting
  `result_db/<ds>.db` + result JSONs is required, or the status tool will report a false DONE and
  the whole comparison runs on stale numbers.

Every phase in this document that changes computation will carry its own "delete this first" block
when it's specced out. This section is the standing reminder that none may skip it.

---

## 7. What I need to know about local feasibility (not a design decision — a fact)

Every new local test needs the rep matrices + labels for herg_ki / logd / flab_thermo_tresanco.
The cache is server-side (`KIRBY_CACHE_DIR=/data/stat-ecr/...`); there are no rep matrices on this
machine. For the two molecular sets, cheap reps (ecfp4, avalon, maccs, mordred, atom_pair) can be
generated locally, but the strong ones (molformer, graph_kernel, mhg_gnn, grover) are heavier. For
flab_thermo_tresanco (n=672), the protein-embedding reps (ESM2, antiberty, saprot, prot_t5) over
672 sequences are far more tractable locally than the 4,275-sequence flab_expression_koenig — which
is exactly why tresanco is the primary. Before the Phase 2 base-rep tests can run we must settle how
the matrices get here (compute a cheap subset locally, or push the cached matrices from the server —
a one-time push is cheaper than re-running searches on the strained server). Flagged, not acted on.

---

## Appendix A — Allocation degeneracy demo (local, no ML)

Faithful reproduction of the `hybrid.py` allocation math only (linear / softmax / rank / top_k
weighting + the proportional budget-exact allocator). **No sklearn/torch/shap, no data** — pure
arithmetic on a score vector, so it proves the *mechanism*, not a dataset result. This is the
concrete version of the section-2 hand-argument.

Reading: `best_rep_feats` = how many of the budget's features the single best rep actually gets.

**Big near-tied pool (29 reps, scores 0.17–0.58), budget 100 — today's situation:**

| strategy | best-rep feats | reps used | median feats/rep |
|---|---|---|---|
| performance_weighted | **5** | 29 | 3 |
| minmax_norm + linear | 7 | 28 | 3 |
| rank_linear | 7 | 27 | 3 |
| softmax T=0.1 | 14 | 23 | 2 |
| top_3 | **34** | 3 | 0 |

The best rep gets **5 of 100 features**. Normalisation lifts it to 7 — barely. Only concentration
(top_3) rescues it, and top_3 on a big pool ≈ pruning the pool.

**Same pool, budget raised to 500:** performance_weighted best-rep → 27, top_3 → 171. Budget
helps, but it's still split across 29 reps.

**Small N (4 reps, 0.44–0.58), budget 100:** performance_weighted best-rep → **28** (median 25).
A small top-N does more than any weighting change or budget bump. (4 is just an illustration
here, not a target — N is a knob.)

**Original c4afb0e (N=2), budget 200:** performance_weighted best-rep → **107**, median 100 —
i.e. the "~100 features per rep from 2 distinct reps" that actually worked.

**Takeaway (drives the phase ordering):** how many reps enter the hybrid (the top-N knob) is the
dominant lever, concentration works within that, budget is third, normalisation is a
real-but-modest fourth. performance_weighted is not "broken" — raw-score-proportional allocation
across many near-tied, compressed-range reps simply degenerates to uniform dust.

Script: `scratchpad/alloc_demo.py`.

---

## Appendix B — Feature-selection method research (2026-07-15)

Web research session, multiple keyword angles. This is the material for the sanctioned
feature-importance exception (Phase 4.1). Sources at the end.

### Why look beyond the current 3

The existing scorers are `deeplift`, `lightgbm_gain`, `treeshap` (3 defaults of 8 available). The
repo's own evidence says they carry almost no information *as a choice*: `EXPERIMENT_OVERVIEW.md:402`
records the importance axis as "~ZERO (all give identical rankings)", and `HYBRID_DEFECTS_2026_07_14.md`
measured `treeshap` best on 12/21 datasets, `lightgbm_gain` 7/21, `deeplift` 2/20 — weak and
inconsistent. Critically, **all three are *marginal*** — they score each feature on its own, so over
a pool of redundant reps they pick the same signal many times. **None is redundancy-aware.** That is
the capability gap, and it is why "add/replace importance scoring" is a sanctioned exception.

A cross-cutting finding reinforces this: under correlated features, both SHAP and permutation
importance mislead — permutation splits importance across correlated copies (all look weak), and SHAP
can even assign opposite-sign values to features correlated at 0.995. A marginal scorer over a
redundant pool is the wrong tool by construction.

### Candidate families (to test, then cut to 1–2 — one must be redundancy-aware)

1. **Redundancy-aware information-theoretic filters — the direct fix.**
   - **mRMR** (max-relevance / min-redundancy; Peng & Ding 2005): picks features that are relevant to
     the target *and* not redundant with those already picked. Canonical, cheap, model-free.
   - **JMI / CMIM / DISR**: refinements. DISR (normalised joint mutual information) is reported among
     the most effective MI criteria; CMIM/JMI trade off relevance vs redundancy via conditional/joint
     mutual information. Available in the `praznik` family of implementations.
   - *Why for us:* directly attacks "the hybrid holds the same signal many times." mRMR already tested
     positive in a prior probe (`HYBRID_DEFECTS` §Defect 8: mRMR beat the marginal selector on 7/9
     model×budget cells on herg_ki). Strongest first pick.

2. **All-relevant wrappers.**
   - **Boruta** (shadow/permuted features as a null): benchmarks repeatedly rank it the strongest on
     tabular data and robust to noise. Downside: wrapper cost (≈4× shap-select in one timing).
   - **Boruta-SHAP**: Boruta with SHAP as the importance signal.

3. **Significance-based SHAP.**
   - **shap-select** (2024): train once, compute SHAP on a *validation* set, regress the target on the
     SHAP values, drop features with insignificant or negative coefficients; recursive elimination +
     tiny L1 handle collinearity. Fast (~22s vs Boruta ~96s in their test), selects compact sets, and
     scores on validation not train (less overfit than tree-gain). Good cheap redundancy-aware-ish
     candidate.

4. **Neural / differentiable selection (matches our DNN downstream model).**
   - **LassoNet**: an L1 skip-connection that selects features inside a neural net. In the 2024 Nature
     benchmark, LassoNet was in the top set alongside RF, TreeSHAP and mRMR.
   - **Concrete Autoencoder**: differentiable top-k feature selection via a concrete selector layer —
     naturally fits our "pick top-D for a budget" framing.

5. **FDR-controlled (research direction, heavier).**
   - **Model-X knockoffs / DeepPINK / GRIP2 (2026)**: control the false-discovery rate of selected
     features. Principled but expensive; note as a later option, not a first candidate.

6. **Stability selection / RFE** — resampling-based robustness (stability selection) and a simple
   wrapper ranking (RFE). Useful as robustness baselines, not as the primary lever.

### The 2024 benchmark's practical verdict

The Nature *Scientific Reports* 2024 benchmark of feature-selection methods for nonlinear signals
found **Random Forests, TreeSHAP, mRMR, and LassoNet** the best performers across synthetic datasets
of varying complexity — i.e. one strong marginal tree scorer + one redundancy-aware filter + one
neural selector is a well-supported trio to test.

### Proposed candidate set for Phase 4.1 (test → cut to 1–2)

- Keep ONE marginal tree scorer as the baseline (`treeshap` or `lightgbm_gain` — they're
  interchangeable, so keep the cheaper: `lightgbm_gain`).
- Add **mRMR** (redundancy-aware filter; cheap; already tested positive here).
- Add **shap-select** (cheap, significance-based, validation-scored).
- Optionally **Boruta** (strongest in benchmarks, slower) and **LassoNet** (neural, matches the DNN
  downstream model) if the first three don't settle it.
- Cut to the best 1–2 by prediction-correlation + downstream score. At least one survivor must be
  redundancy-aware.

### Sources

- [Nature Scientific Reports 2024 — NN feature-selection benchmark](https://www.nature.com/articles/s41598-024-82583-5)
- [shap-select (arXiv 2410.06815)](https://arxiv.org/html/2410.06815v1)
- [mRMR overview](https://www.emergentmind.com/topics/maximum-relevance-minimum-redundancy-mrmr) · [Minimum-redundancy FS (Wikipedia)](https://en.wikipedia.org/wiki/Minimum_redundancy_feature_selection) · [praznik MRMR](https://rdrr.io/cran/praznik/man/MRMR.html)
- [Conditional MI feature selection (Applied Intelligence)](https://dl.acm.org/doi/abs/10.1007/s10489-021-02412-4)
- [LassoNet](https://github.com/lasso-net/lassonet) · [Concrete Autoencoders (Indirectly Parameterized, arXiv 2403.00563)](https://arxiv.org/pdf/2403.00563)
- [Composite Feature Selection via Deep Ensembles (NeurIPS 2022)](https://arxiv.org/pdf/2211.00631) · [GRIP2 deep knockoffs (arXiv 2602.00218)](https://arxiv.org/pdf/2602.00218)
- [Boruta / RFE / stability comparison (omics evaluation, PMC6433899)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6433899/)
- [SHAP vs permutation under correlation (MetricGate)](https://metricgate.com/blogs/permutation-importance-vs-shap/)
