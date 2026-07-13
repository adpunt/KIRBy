# Blueprint Generalization — Roadmap

**Goal.** The hybrid search is expensive (~7,300 configs × 5 folds per dataset).
Find one, or a handful of, *generalised blueprints* per domain that can be applied
to a new dataset with no search at all — and measure honestly what that costs you.

**Status:** Phase 0 (offline feasibility) DONE 2026-07-13. Phases 1–5 awaiting the
decisions in §7.

---

## 1. What a blueprint actually is

Read from the code, not from memory. `Blueprint` is a dataclass at
`src/kirby/find_best_hybrid.py:830-899`. The fields the search actually chooses are:

| field | candidates | where |
|---|---|---|
| `reps` | the dataset's rep pool | `DATASET_REPS`, `complexity_helpers.py:94+` |
| `allocation_method` | 9 (top_2/3/5, performance_weighted, entropy_1.0/2.0, power_2, rank_linear, rank_exp_d0.5) | `find_best_hybrid.py:1926` |
| `importance_method` | 3 (deeplift, lightgbm_gain, treeshap) | `find_best_hybrid.py:1927` |
| `internal_evaluator` | 5 (match, rf, svm, lightgbm, mlp) | `find_best_hybrid.py:1937` |
| `budget` | 8 (10…2000), capped at n and by model | `_shared.py:286` |
| *(model)* | 8, a separate axis, not a blueprint field | `DEFAULT_MODELS` |

`step_size` is hardcoded to 5. Everything else on the dataclass is provenance.

**There are no coefficients.** The per-rep feature allocation is a *deterministic
function* of the allocation rule applied to per-rep baseline scores
(`hybrid.py:437-550`). So "generalising a blueprint" means generalising a
**five-field recipe**, not a learned weight vector. That is a far easier target
than it sounds, and it is why this project is tractable.

**Cost asymmetry.** A search is 9 × 3 × 271 = **7,317 configs × 5 folds ≈ 44,000
model fits**. Applying a fixed blueprint is `n_folds` hybrid builds — and if the
blueprint carries `rep_scores`, `apply_blueprint()` skips per-rep fitting entirely
(`find_best_hybrid.py:5250-5252`). Three orders of magnitude.

---

## 2. The reframe: most of this study is already paid for

The search **scores every cell** of (model × allocation × importance × evaluator ×
budget) on every dataset and stores it — SQLite `result_db/<ds>.db`
(`src/kirby/result_cache.py:100`), and dumped to JSON by c02
(`c02_complexity_grid.py:255-263`).

**That grid IS the cross-dataset transfer matrix.** "What would dataset A's
blueprint score on dataset B?" is a *table lookup in B's grid*, not a new run.
The expensive part has already been bought. The core of this study is a table
join, not a compute campaign.

Two things the lookup cannot give you, which is what the server phase is for:
1. **The rep list.** Each dataset searched its *own* pool, so the grid can't tell
   you what a shared, canonical domain rep pool would do.
2. **Pruned allocations.** Allocation strategies that never place top-3 are dropped
   before the full-fold pass (`find_best_hybrid.py:3893-3897`), and *which* ones get
   dropped is dataset-specific. Those cells are holes.

---

## 3. Phase 0 results (offline, done, zero compute)

### 3a. A canonical per-domain rep pool exists — the idea is viable

Computed from `DATASET_REPS` × `dataset_policy.DATASET_REGISTRY_DATA`:

| domain | datasets | union of `general` reps | reps present in ≥80% of them |
|---|---|---|---|
| nucleotide | 21 | **12** | 11 |
| molecular | 8 | 18 | 14 |
| antibody | 21 | 32 | 21 |
| tcga | 2 | 22 | 19 |
| dta | 3 | 16 | 16 (full intersection) |
| reaction | 2 | 19 | 7 |

Nucleotide's *entire union* across 21 datasets is 12 reps. Molecular's is 18. The
rep vocabulary is already near-shared within a domain — this was the single most
likely thing to kill the project, and it doesn't.

Only **antibody (21), nucleotide (21), molecular (8)** have enough datasets to hold
some out. TCGA/reaction/DTA have 2–3 and can be reported but not validated.

### 3b. The recipe is strikingly stable across datasets

Leave-one-dataset-out over 25 antibody datasets, per model. The consensus recipe is
picked in **13–25 of 25** folds, and it is essentially the same one every time:

> **entropy_1.0 / entropy_2.0 allocation + treeshap / lightgbm_gain importance + budget 50**

Budget **50** wins for *every* downstream model. That is a real, quotable finding,
and conveniently 50 is small enough to be legal on every dataset regardless of n.

### 3c. The honest ceiling collapses — read this twice

Re-scoring the *existing* c02 grids with leave-one-fold-out selection (choose the
best cell on folds≠k, score it on fold k):

| how the search is scored | beats best single rep |
|---|---|
| **naive** (pick and score on the same folds) — what we report today | **132/155 (85%)** |
| **honest** (leave-one-fold-out selection) — the redesign protocol | **45/155 (29%)** |

**66% of the hybrid's apparent wins are selection optimism.** This independently
confirms the honest-comparison redesign from the outside, using data already on disk.

Per model, wins lost to optimism: ridge 17, mlp 18, rf 16, xgboost 15, lightgbm 11, svm 10.

Seven antibody datasets never honestly beat a single rep under *any* model:
`abagym_HER2, abagym_VEGF, abcov_kd, flab_aggregation_jain,
flab_aggregation_jain_hicrt, flab_immunogenicity_marks, ginkgo_heparin`.

**Caveat, and it matters:** c02 does not dump per-rep fold scores, so the *baseline*
here is still the best-of-26-reps picked on the same folds — itself optimistic. An
honest baseline would be *lower*, so 29% is a **lower bound**. Fixing this needs
`SearchResult.baseline_fold_scores`, which exists but is not serialised. See Phase 1.

### 3d. What this does to the blueprint question

A blind fixed recipe wins on ~2–4/25 datasets per model (~10%). Against the *naive*
85% that looks hopeless. Against the *honest* 29% it is a completely different
conversation — and 29% is a lower bound that will rise once the baseline is scored
honestly too.

So the real question is no longer "can a blueprint match the search." It is:

> **How much of the search's genuine, honestly-measured 29% edge survives when you
> freeze the recipe — and which single axis do you have to keep searching to get it back?**

That is a better paper than the one we set out to write.

---

## 4. Phase 1 — capture the right fields in the re-run **(URGENT, ~15 lines)**

Every experiment is being re-run under the honest-comparison redesign. **This is the
one window to make the re-run emit what the blueprint study needs.** Miss it and we
re-run a third time.

`c02_complexity_grid.py:255-263` currently dumps each config as
`{model, allocation, importance, budget, score_mean, score_std, fold_scores, n_features}`.

Missing, and needed:
- **`internal_evaluator`** — without it each (model, alloc, imp, budget) cell appears
  1–5 times with no label. The evaluator axis is currently unrecoverable from JSON.
- **`reps` / `rep_pool`** — which pool this grid was searched over.
- **condition tag** (tier + aug_strategy) — currently only in a separate summary dict.
- **`baseline_fold_scores`** — already on `SearchResult`; serialise it so the *baseline*
  arm can be scored honestly. Without this every win rate we quote is a lower bound.

Nothing else changes. No new compute.

---

## 5. Phase 2 — export the grids (server, cheap, no compute)

Write a read-only dumper: `result_db/<ds>.db` → one JSON per dataset with the identity
cols, all config cols (incl. `internal_evaluator`, `rep_pool`), `score_mean`,
`fold_scores_json`, and the unpickled `blueprint` (which carries `rep_scores`).
User runs it on ARC, rsyncs the JSONs back (JSON-only, per house rules).

`tests/analysis/general_hybrid_vs_base.py:99-291` (`export_dataset()`) already does
most of this and returns nearly the schema we want — extend rather than rewrite.

After this, the entire mining analysis is offline and free, for **every dataset ever
searched**, at full fidelity.

---

## 6. Phase 3 — offline mining (laptop, free) → Phase 4 — server validation

**Phase 3 (offline).** With the full grids:

1. **Variance decomposition — the key deliverable.** For each axis (allocation,
   importance, evaluator, budget, model), how much do you lose by *freezing* it at its
   domain-consensus value versus searching it? This tells us which parts of the search
   are dead weight. The likely outcome, given §3b: freeze allocation + importance +
   evaluator, and keep a *tiny* search over budget — 8 configs instead of 7,317.
   **That is a 900× cost cut and it is probably the real product.**
2. **Blueprint selection**, leave-one-dataset-out, honest on both arms.
3. **K = 1 vs a menu of ~3** with a cheap rule for picking among them.
4. Report win/tie/loss per dataset. No averaging across datasets (house rule; and §3
   already showed mean-of-ratios explodes here — the first cut of this pilot produced
   "−76" before I replaced it with rank-percentile and win-counts).

**Phase 4 (server).** Only what a lookup cannot answer. One new standalone experiment,
arms per (dataset, model), all honestly scored:

- **A.** best single rep (the currency)
- **B.** domain blueprint applied blind, canonical domain rep pool, no search
- **C.** domain blueprint + the one retained mini-search (probably budget)
- **D.** full native search (the oracle / upper bound)

Run on held-out datasets *never used to pick the blueprint*. Cost is
`K × datasets × folds` — negligible next to one search.

**Reuse, do not rebuild.** Three pieces of prior art already exist:
- `tests/experiments/opig_blueprint_transfer.py` — pairwise source→target with exactly
  the conditions A/B/C/D above (`base_best`, `native`, `recipe_transfer`, `full_transfer`).
  The new experiment is this, generalised from *one* source dataset to a *domain consensus*.
- `tests/experiments/complexity/c03_cross_dataset_transfer.py` — cross-dataset comparison
  of optimal configs; explicitly *not* transfer.
- `tests/experiments/complexity/c24_blueprint_transfer.py` — cross-*fold* transfer within
  one dataset, plus a Nogueira stability index worth reusing.

**Two cost traps to avoid in the "cheap" arm** (or it quietly stays expensive):
- Use `apply_blueprint()` with `rep_scores`, **not** `build_fold_hybrid()` — the latter
  has no `precomputed_alloc_scores` passthrough and refits the internal evaluator on every
  rep, every fold.
- Force `augmentation_strategy='none'`. `uncapped` A/B-tests up to 50 candidates
  sequentially *per fold* (`hybrid.py:2339-2350, 3375`) and will dominate the runtime.

---

## 7. Decisions I need from you

**1. What the blueprint has to beat.**
The blueprint will not match a full search. We need to say what "good enough" means.
Options: (a) beat the best single representation — the paper's existing currency;
(b) land within a stated margin of the full search; (c) maximise how many datasets it
wins on. This sets the selection objective, so it has to be picked before mining.

**2. Budget: freeze it, or keep searching it.**
Budget 50 won for every model in the pilot, and 50 is legal on every dataset. But budget
is also the axis most likely to be genuinely dataset-dependent. Options: (a) freeze at 50
— cleanest story; (b) make it a rule (a fraction of n); (c) freeze the other three axes
and keep an 8-config budget search — my recommendation, and the likely product.

**3. One blueprint per domain, or a menu.**
One is a much cleaner claim. A menu of ~3 will score better but needs a rule for choosing,
and that rule is itself a thing that can overfit. Changes the size of Phase 4.

**4. The rep pool.**
Strict intersection (only 6 reps for antibody — probably too thin) versus the ≥80%-coverage
pool (21 for antibody, 11 for nucleotide, 14 for molecular) versus letting each dataset keep
its own pool (in which case the blueprint is a *recipe only*, and the reps are still chosen
per dataset). Note the ≥80% pool means a few datasets are missing a rep the blueprint names.

**5. Which domains.**
Only antibody, nucleotide, and molecular can support held-out validation. TCGA, reaction and
DTA have 2–3 datasets — we can report a blueprint for them but cannot honestly validate it.
Attempt all six and be explicit about which are unvalidated, or restrict to the three?

**6. Phase 1 is time-critical.** The field additions in §4 need to land before the honest
re-runs go out, or the grids come back unmineable and we re-run again. This is the only
item with a deadline.
