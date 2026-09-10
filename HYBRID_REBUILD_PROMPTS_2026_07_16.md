# find_best_hybrid Rebuild — Chat Kickoff Prompts (2026-07-16)

The rebuild is designed to run across **several separate chats, one workstream each** (compute is
not the constraint; keeping each chat focused is). This file holds a ready-to-paste prompt for
every workstream.

**How to use:** for each chat, paste the **COMMON PREAMBLE** first, then the **body** for that
chat, into a fresh session. Run them roughly in order — dependencies are noted at the top of each
body. Ask me (the roadmap owner) to hand you the next one when a workstream lands.

---

## COMMON PREAMBLE (paste at the top of every chat)

```
We are rebuilding KIRBy's find_best_hybrid from scratch, following a written roadmap. Do NOT start
coding or run anything until you have read the plan and proposed your approach to me.

READ IN FULL first (not skim):
  1. HYBRID_REBUILD_ROADMAP_2026_07_15.md   — the whole plan (phases 0–7, decisions, cache §6,
     Appendix A allocation demo, Appendix B feature-selection research)
  2. Memory: project_hybrid_rebuild_2026_07_15.md    — locked decisions + goal
  3. Memory: project_knob_sensitivity_2026_07_16.md  — local knob-sensitivity results
  4. HYBRID_DEFECTS_2026_07_14.md            — the prior investigation the roadmap is built on

THE GOAL (never lose this): per dataset, find best rep × best model = THE target; build the best
hybrid for THAT SAME model; compare best-rep×model vs best-hybrid×model. Best-vs-best on purpose.
NO medians, NO averages, NO suboptimal arms.

THE TWO HEADLINE BLOCKERS this whole effort exists to fix:
  (1) hybrid budgets are too small, and (2) the rep pool is too big and duplicative (30+ reps all
  showing the same signal). Measured: a 1,280-wide best rep, budget 50 → hybrid sees 7 features;
  budget 250 → 16. That cannot work by any allocation strategy. Both sides must be fixed.

HARD CONSTRAINTS (violating these is the failure mode):
  - SUBTRACT, don't add. Only two sanctioned additions: budget, and feature-importance scoring.
  - NO code changes to find_best_hybrid.py / hybrid.py yet — this is analysis; discuss first.
  - Ridge is removed everywhere; internal evaluator default = lightgbm.
  - All tests NEW and LOCAL — do NOT trust tests/results/, universal_survey, c01, or any prior JSON.
  - Redundancy is measured by PREDICTION correlation, never raw feature correlation.
  - Rep/allocation decisions are DATASET/DOMAIN-SPECIFIC — do not generalise across datasets. (When
    estimating correlations you MAY pull in extra datasets just for more data points, but the choice
    is still made per dataset/domain.)
  - Metric direction matters — one wrong lower_is_better flips everything; verify it end to end.
  - Server is a slow last resort (outages, low fair usage → days). Design every test to run LOCAL.
  - CACHE SAFETY (roadmap §6): reps are NOT in the cache key. Any change that alters computation
    ships with explicit, SCOPED delete instructions (name the datasets + result_db/<ds>.db + that
    dataset's search/baseline/fold caches + result JSONs). NEVER a blanket wipe. I run deletions on
    the server, one dataset at a time. Recompute is fine; silent staleness is the enemy.

TEST DATASETS (local): herg_ki (n=1412, RMSE), logd (n=5028, RMSE), flab_thermo_tresanco
(n=672, ρ=0.805), buchwald_hartwig (reaction yield, cgr_morgan reps — the proof-of-concept win:
best rep cgr_morgan ρ0.833/rmse15.08 → hybrid ρ0.898/rmse11.75, a −22% rmse win from top-5
complementary reps). flab_expression_koenig (n=4275) is server-only scale-up — don't use locally.

I'm an engineer and I decide. Propose options with plain-English tradeoffs; do NOT design
autonomously. Come back with your proposed approach BEFORE running anything.
```

---

## CHAT 1 — Foundations + the target (roadmap Phases 0.1, 0.2, 2.1, 2.2)

*Dependencies: none. This is first. Produces the prediction vectors and target that later chats need.*

```
YOUR WORKSTREAM — foundations + target, LOCAL only:

1. Ridge removal (Phase 0.1): remove ridge from the model registry, DOWNSTREAM_MODELS,
   DISTANCE_EVALUATORS, and every fallback that defaults to 'ridge'; internal-evaluator default →
   lightgbm. This changes computation → propose the SCOPED cache-delete instructions for me to run
   before anything is re-scored. Do not run them yourself.

2. Local data access: load the four datasets' labels and produce the base-rep matrices locally.
   Cheap reps compute here; for heavier reps (molformer, graph_kernel, protein embeddings) tell me
   EXACTLY what to push from the server rather than assuming. Breadth over depth — no full nested CV
   needed for analysis.

3. Base-rep validity gate (Phase 2.1 — the SIGNAL gate): score every base rep with its real model
   under a light local CV. If the best base rep doesn't reproduce a sane standalone score, the
   hybrid has no signal to find. ALSO emit each rep's per-fold PREDICTION VECTORS — Chat 2 needs
   them for prediction-correlation decorrelation. Save them somewhere durable.

4. Metric-direction audit (Phase 0.2): confirm a lower-is-better set (herg_ki/logd, RMSE) and a
   higher-is-better set (flab_thermo_tresanco, ρ) both yield POSITIVE delta_mean_of_folds when the
   hybrid genuinely wins, negative when it loses.

5. The target (Phase 2.2): rank rep × model per dataset (ridge gone); report best rep × best model
   = the target, plus the top-2 downstream models per dataset (only those are carried forward).

Deliverables: the target per dataset, the top-2 models per dataset, the saved prediction vectors,
and the scoped cache-delete instructions for the ridge removal. Propose your approach to step 2
(local data access) FIRST.
```

---

## CHAT 2 — Cut the rep pools, per dataset (roadmap Phase 3)

*Dependencies: Chat 1's prediction vectors + target. Collaborative — done per dataset WITH me.*

```
YOUR WORKSTREAM — cut the rep pools, one of the two headline blockers ("too many duplicative reps").

Using Chat 1's per-rep prediction vectors:
1. Kill the general vs top_tier distinction — there is ONE pool per dataset.
2. Rank reps by base performance; compute the pairwise PREDICTION correlation between reps.
3. Identify duplicate-signal reps (reps whose predictions are near-identical to a stronger rep) and
   propose removing them. We are deleting DUPLICATES only — NOT trimming to a fixed count. How many
   reps ultimately feed the hybrid is set later by the top-N knob (N = 2, 3, 5, more).
4. Do this PER DATASET and bring each to me individually — do NOT generalise a rule from one dataset
   to another. The high-rep dataset(s) are exactly where this matters most. You may pull in extra
   datasets purely to get more correlation data points, but the cut is per dataset/domain.

Reference: buchwald_hartwig's proof-of-concept win came from 5 reps that each carry DIFFERENT
information — that is the kind of complementary, decorrelated pool we're trying to surface.

Deliverable: a proposed decorrelated pool per dataset, with the prediction-correlation evidence for
each removal, brought to me one dataset at a time. No code changes to find_best_hybrid/hybrid.py.
```

---

## CHAT 3 — Allocation strategies + reality-check (roadmap Phases 4.2, 4.3)

*Dependencies: Chat 1 (target, models); ideally Chat 2 (pruned pools). Uses the top-N knob.*

```
YOUR WORKSTREAM — test the existing allocation strategies, cut duplicates, and REALITY-CHECK them.

4.2 — Correlate the existing allocation strategies: which of the ~9 actually produce DIFFERENT
   allocations/scores (by prediction correlation of the resulting hybrids)? Collapse to a few
   genuinely distinct categories. Top-N (N = 2, 3, 5, more) is a REPLICATE parameter of each
   category, not a standalone strategy. Include the written-but-unregistered greedy/complementarity
   allocators (allocate_greedy_forward etc.) as candidate categories — existing code, not additions.

4.3 — Allocation reality-check (the central question): for a dataset + best model, run each
   surviving strategy and report what it ACTUALLY produces — which reps contributed features, how
   many each, and CRUCIALLY whether the best rep's genuinely informative features made it into the
   hybrid or got truncated by the per-rep quota. (Today selection is a per-rep quota, never a global
   ranking, so the best rep is always truncated + diluted.) This decides whether top-N, global
   ranking, or normalisation is what restores the best rep's signal.

Target behaviour to reproduce: buchwald_hartwig top-5 (equal-ish concat of 5 complementary reps →
−22% rmse). The WIN is the target; the current top_5 mechanism is NOT how we want to do it (we have
the top-N knob) — find the clean way to get that behaviour.

Deliverable: the distinct allocation categories to keep, and evidence of which put the best rep's
real signal into the hybrid. No code changes to find_best_hybrid/hybrid.py.
```

---

## CHAT 4 — Feature-importance candidate bake-off (roadmap Phase 4.1)

*Dependencies: Chat 1 (target, models, pool). This is the sanctioned add/replace exception.*

```
YOUR WORKSTREAM — the feature-importance scoring investigation (a SANCTIONED exception to
subtract-don't-add; see roadmap Appendix B).

1. Confirm locally that the existing scorers (lightgbm_gain, treeshap, deeplift) give near-identical
   feature rankings — the repo says the importance axis is "~ZERO" (EXPERIMENT_OVERVIEW.md:402) and
   the knob-sensitivity memory measured it as noise. If so, they collapse to one.

2. Test the researched candidates at a FIXED budget on the local datasets:
   - lightgbm_gain (marginal baseline, keep the cheap one)
   - mRMR (redundancy-aware filter — already tested positive on herg_ki)
   - shap-select (significance-based, validation-scored)
   - backups if unresolved: Boruta (all-relevant, slower), LassoNet (neural, matches the DNN model)
   Compare by prediction correlation between the selections AND by downstream score at equal budget.

3. Cut to the best 2-4 methods; at least one must be redundancy-aware.

Deliverable: the 2-4 importance methods to keep, with evidence. No code changes to
find_best_hybrid/hybrid.py.
```

---

## CHAT 5 — Budget, normalisation, global-concat, and the PCA-residual probe (roadmap Phase 5)

*Dependencies: Chat 1 (target); ideally Chats 2–4 (pool, allocation, importance settled). Attacks the OTHER headline blocker: budget.*

```
YOUR WORKSTREAM — budget (the second headline blocker) + the allocation refinements.

5.1 — Uncap the budget: remove the n_samples cap (find_best_hybrid.py:4816) and sweep budget past
   the widest rep. Show the honest hybrid-vs-target curve as budget grows, on the local datasets.
   This must let a 1,280-wide best rep actually contribute a real slice, not 7–16 features.

5.2 — Normalisation: test BOTH schemes (min-max across reps AND rank-based) in one sweep; cut to
   one. Note it may not help at all (Chat 3's reality-check may show the fix is top-N / global
   ranking, not normalisation) — drop both if so.

5.3 — Global-concat scoring for top-N: concatenate the top-N reps' features, compute ONE global
   importance ranking, take the top D. Concat is a scoring device only, NOT its own rep.

5.4 — SIDE EXPERIMENT (test only, do NOT properly implement): spend part of the budget on the best
   selected features, then spend the remaining budget on PCA COMPONENTS of the leftover features of
   the best reps — compress the weaker signal instead of discarding it. Keep only if it clearly
   beats plain selection at the same total budget. This runs against the standing
   importance-over-PCA preference, so it's a probe, not a default.

Deliverable: the budget policy (curve + recommended range), the normalisation verdict, whether
global-concat helps, and the PCA-residual probe result. No code changes to find_best_hybrid/hybrid.py.
```

---

## CHAT 6 — Augmentation as a first-class pipeline stage (roadmap Phase 4.4)

*Dependencies: Chat 1 (target). Can run in parallel with 2–5.*

```
YOUR WORKSTREAM — make augmented features first-class and SOLIDIFIED in the pipeline (not bolted on).

1. Make the augmentation-level hybrid search a proper, integrated, cached, upfront pipeline stage —
   clearly defined, not an afterthought. Map how it currently plugs in (DATASET_REPS['augmentation'])
   and specify how it SHOULD plug in.

2. Cover the two use cases:
   (A) Individual-style augmented features — small feature sets that don't deserve to be a standalone
       base rep, tried directly in the hybrid.
   (B) Best-rep-plus — the single best rep plus either (a) augmented features, or (b) the top-5
       individual features from each of the other reps.

3. Show this covers BOTH hybrid scenarios: signal spread across reps (complementarity), and one
   dominant rep slightly improved by a few borrowed/augmented features.

Deliverable: a concrete design for the augmentation stage + local evidence it helps on at least one
dataset. No code changes to find_best_hybrid/hybrid.py yet — design + evidence, then discuss.
```

---

## CHAT 7 — CV structure decision (roadmap Phase 6)

*Dependencies: Chats 2–5 (need to know how small the search got). Not a separate study — a decision.*

```
YOUR WORKSTREAM — settle the CV structure, which FALLS OUT of how small the search became.

Given the pool/allocation/importance/budget cuts from the prior chats, determine: does a real
search still remain (top-N × a few allocation categories × budget sweep), or has it collapsed to
essentially one fixed setting?
  - Real search remains → keep nested CV (standard, because we select); reduce inner folds on small
    n (e.g. 5→3) as a normal tweak.
  - Fixed setting → no selection → plain k-fold is already honest; drop the nested layer.

No bespoke stability diagnostic. Bring me the count of remaining searched configurations and the
resulting recommendation. No code changes yet.
```

---

## CHAT 8 — Rebuild find_best_hybrid (roadmap Phase 7)

*Dependencies: ALL of the above + my explicit go-ahead. Gated — do not start without it.*

```
YOUR WORKSTREAM — the rebuild itself. Do NOT begin until I have explicitly approved the design from
Chats 1–7. Consume: the cut dataset list, per-dataset top-2 models, one decorrelated pool per
dataset with rep count set by the top-N knob, the surviving allocation categories, 1–2 importance
methods (one redundancy-aware), the solidified augmentation stage, the uncapped budget sweep, and
the agreed CV structure. Target: a search space small enough that pruning is unnecessary and honesty
is cheap. Every step that changes computation ships with its scoped cache-delete instructions.
```
