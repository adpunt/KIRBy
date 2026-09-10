# Allocation-strategy work — hand-off (2026-08-21)

Read this before touching allocation strategies. It records what was built, **what was done wrong**
(so you don't repeat it), and what still has to happen to actually pick the best allocation methods —
plus the open questions on how to do that *smartly*.

---

## 0. One-paragraph state

The allocation search space is **{rep-selection incl. top-N} × {weighting} × {quota, global}**, evaluated
over **{model × budget × dataset}**. It is **NOT** "quota vs global" (that framing was the central
error). A global-pooling mechanism was built and an experiment (c31, job 12830012) was run, but that
experiment tested a lopsided sliver of the grid and cannot answer which strategies are best. SPEED_TIERS
was removed from the code. **RESOLVED (2026-09-01):** the proper sweep (c32) was run and five allocations
were chosen and wired into `DEFAULT_ALLOCATION_METHODS`. See **Section 8** for the outcome.

---

## 1. The allocation space (the correct model)

Three independent axes. A "strategy" is one point across all three.

- **Axis A — rep selection** (how many / which reps get any budget):
  - the **top-N knob**: 2, 3, 5, all
  - other selectors currently in code: `thresholded` (α), `score_gap` (gap), `min_floor` — *decide keep-or-cut*
- **Axis B — weighting** (how the budget splits among the selected reps):
  - `performance_weighted` (linear), `power_2`/`power_3`, `softmax`(T), `rank_linear`, `rank_exp`(decay),
    `entropy`(penalty), `fixed` (uniform)
- **Axis C — mechanism**:
  - `quota` (per-rep budget) — base `performance_weighted`
  - `global` (pool candidates, one global ranking) — base `weighted_global`; `equal`/`naive` are degenerate points

Full strategy = **(A) × (B) × (C)**. Already **cut** (tested & rejected, f01/f12/f03): the whole greedy
family (`greedy*`, `dominant_greedy*`).

---

## 2. What was done

- **Global-pooling mechanism built** — `allocate_weighted_global` in `src/kirby/hybrid.py` (additive,
  opt-in via `find_best_hybrid(..., allocation_methods=['weighted_global'])`), plus the **internal-ranker
  fix** (item 5: `create_hybrid`'s inner split `GroupShuffleSplit`→`GroupKFold`, applied unconditionally —
  it shifts grouped-data quota numbers too, so `find_best_hybrid` caches are stale by design). Committed
  `5a866f4`. **The engine function is only smoke-tested** — c31 validates the *algorithm* (its own impl),
  not `allocate_weighted_global` at scale. A c31-vs-engine parity check is still TODO before any experiment
  relies on it.
- **c31 model sweep** (job 12830012, done): weighted-global `{linear, power_2, top_2}` × POOL_TOTAL `{2,4}`
  × importance `{lightgbm_gain, treeshap}` × models `{lightgbm, rf, catboost}` × budgets `{500,1000}` × 10
  high-signal datasets, vs a single quota arm (`fixed`) + `equal` + `naive`. Results:
  `/Volumes/seagate/kirby_cache/c31_weighted_global/<ds>__<model>.json` + `_strategy_matrix.json`.
- **SPEED_TIERS removed entirely** — replaced by module constants in `find_best_hybrid.py`:
  `DEFAULT_IMPORTANCE_METHODS`, `DEFAULT_ALLOCATION_METHODS`, `DEFAULT_INTERNAL_EVALUATORS`,
  `DEFAULT_N_FOLDS`. The `speed` parameter is now an accepted-but-ignored vestige. 23 files updated,
  behavior-preserving, uncommitted. `DEFAULT_ALLOCATION_METHODS` is where the new grid config will live.
- **Corrected the model of the allocation space** (Section 1).

---

## 3. What was done INCORRECTLY — read this so it isn't repeated

1. **Framed everything as "quota vs global."** False binary. `quota` is a *mechanism* that hosts every
   weighting strategy; it is not one strategy. This mistake was made repeatedly.
2. **c31 tested only ONE quota weighting (`fixed`).** So the experiment structurally cannot compare the
   quota strategies against the global ones — it's a lopsided sliver of the grid, not a fair sweep.
3. **The top-N × weighting cross was never tested — and *can't* be, as the code stands.** `top_2` in the
   registry means "top-2 reps **fused with linear weighting**" (`performance_weighted` + `top_k(k=2)`).
   There is no way to express `top_3 × power_2`, `top_2 × rank`, etc. This code-level fusion of Axis A into
   Axis B is the thing that must be fixed before the grid is even runnable.
4. **Used statistical significance / bootstrap-CI-overlap as the decision rule.** With 5 folds the per-cell
   CIs are wide (~0.03–0.15 R²), so "is any strategy significantly better than quota" trivially returns
   "no," and the analysis collapsed to "keep quota, it's all within noise." **That erased the per-(dataset ×
   model × budget) variation, which is the entire point of KIRBy.** The resulting `MIN_ALLOCATION_SET`
   writeup was deleted and the memory conclusion retracted.

**Consequence:** do NOT trust any of these earlier outputs — "global beats quota 43/60", "linear ≈ coin
flip 28/60", "minimal set = {quota}", the density-bias framing as a decision driver. They answered the
wrong question with data that can't answer the right one.

---

## 4. Coverage of existing data (what is / isn't tested)

| Source | mechanism | selection (top-N) | weightings tested | models | crosses A×B? |
|---|---|---|---|---|---|
| Round-1 / allocation-revisit (`ALLOCATION_CATEGORIES_2026_07_19.md`) | quota only | all, + top_2/3/5 *fused linear* | perf_weighted, power_2, rank_linear, rank_exp, entropy, fixed | lightgbm (rf = 1-budget mini-sweep) | **no** |
| c31 sweep (12830012) | global (+ quota=`fixed` only, +equal +naive) | all, + top_2 *fused linear* | linear, power_2 | lightgbm, rf, catboost | **no** |

**Three holes:** (1) the **top-N × weighting cross** was never run in either mechanism; (2) **quota was
never swept over its weightings with more than one model** (c31 gave quota only `fixed`; round-1 gave the
weightings but only lightgbm); (3) the two mechanisms were tested with **non-overlapping weighting sets**,
so no `strategy × quota` vs `strategy × global` comparison exists for the same strategy. **A new sweep is
required.**

---

## 5. What needs to happen going forward

1. **CODE PREREQUISITE — make the grid expressible.** Decouple the top-N rep-count knob (Axis A) from the
   weighting scheme (Axis B) so *any* weighting can be applied to *any* top-N, in **both** mechanisms.
   Concretely: `top_k` must become a *selection stage that runs before* weighting, not a weighting itself;
   then a strategy is `(top-N) × (weighting) × (mechanism)`. Decide the fate of `thresholded`/`score_gap`/
   `min_floor` (keep as selectors or cut). This is where the new `DEFAULT_ALLOCATION_METHODS` gets its
   definition. **Nothing downstream is valid until this is done** — the current registry cannot represent
   the space.
2. **DESIGN THE SWEEP.** Cross `{top-N} × {weighting} × {quota, global}` over `{model × budget × dataset}`.
   The grid is large — **sample, don't exhaust** (Section 6).
3. **SELECT REPRESENTATIVES.** Pick a small set of strategies that *consistently* produce top hybrids per
   (dataset × model × budget). The goal is search-space reduction: a few robust representatives, not one
   winner and not the whole grid.
4. **Then** the earlier open items fold in: the c31-vs-engine parity check; the stale-cache wipe from the
   item-5 ranker change (`invalidate_allocation_caches.sh`).

---

## 6. OPEN QUESTIONS — how to find the best strategies *smartly*

These are genuinely unresolved; they need deciding before (or as part of) designing the sweep.

- **Sampling a large grid.** What is the minimum sweep that can *discriminate* the strategies? Full
  factorial is too big. Options: fractional factorial; Latin-hypercube over the axes; a two-stage design
  (coarse screen → refine the survivors); or an adaptive/bandit allocation that spends more runs on cells
  where strategies actually diverge. Which, and how small can it be?
- **A consistency metric that is NOT a significance test.** "Consistently produces the best" must respect
  that the winner varies per cell. Candidate measures: per-cell win counts; **near-best coverage** (how
  often a strategy is within ε of the cell's best); mean/median **rank** across cells; rank-aggregation
  (Borda). A strategy that is top-3 in 50/60 cells is real signal even if no single cell is CI-significant.
  Pick ε and the aggregation rule deliberately — and never collapse the per-cell variation with CIs again.
- **Model-conditioning.** In c31, rf behaved differently from lightgbm/catboost. Should the kept set be
  **per-model**, or one set the search chooses from per cell? Does budget also condition it?
- **Redundancy collapse.** Many weightings make near-identical predictions (round-1: >0.97 correlated).
  Cluster strategies by prediction correlation and keep one representative per cluster? At what granularity
  — weighting alone, or the full `(top-N × weighting × mechanism)` strategy?
- **Power vs consistency.** 5 folds ⇒ CIs too wide to call per-cell winners. Either raise power (more folds
  / repeated CV) to shrink CIs, or lean on **rank-consistency across many cells** instead of per-cell
  significance. Which — and can we afford repeated CV over this grid?
- **Objective: "best" vs "robust representative."** With a space this large, chasing THE best per cell
  invites winner's-curse/overfitting. The likely real objective is a **small set that is reliably near-best
  everywhere**, not the argmax per cell. Define this precisely up front — it changes the whole method.
- **How small is "a few"?** 3? 5? Must the set span both mechanisms, or can it live in one?

---

## 7. Pointers

- **Engine:** `src/kirby/hybrid.py` — `_compute_allocation_weights` (Axis-B weightings), `top_k` branch
  (the fused Axis-A knob to decouple), `allocate_weighted_global` (global mechanism), `NAMED_ALLOCATIONS`.
  `src/kirby/find_best_hybrid.py` — `DEFAULT_ALLOCATION_METHODS` (new home of the grid config).
- **Data:** c31 → `/Volumes/seagate/kirby_cache/c31_weighted_global/` (+ `_strategy_matrix.json`) —
  **wrong-question data, keep only as reference.** Round-1 → `ALLOCATION_CATEGORIES_2026_07_19.md`.
- **Memory:** `project_c31_model_sweep_results_2026_08_21` (carries the RETRACTION),
  `project_greedy_allocators_already_cut`, `project_hybrid_round3_results_2026_07_27`.
- **Cut, do not re-propose:** the greedy allocator family.

---

## 8. Outcome — the five chosen (2026-09-01)

The c32 sweep crossed `{weighting} × {quota, global}` over `{model × budget × dataset}`, with the
number-of-reps knob (Axis A) held as a **separate** axis rather than baked into the weighting. Each
(weighting × mechanism) pair was scored by how often it landed in the top three of its cell (9 datasets ×
3 models × 2 budgets; qm9 excluded for a corrupt rep cache). Five stood clear; the rest were consistently
lower. These are now the whole of `DEFAULT_ALLOCATION_METHODS` in `src/kirby/find_best_hybrid.py`:

| allocation (weighting \| mechanism) | code name |
|---|---|
| linear \| quota | `performance_weighted` |
| power_law² \| quota | `power_2` |
| softmax T=1 \| quota | `softmax_T1` |
| rank_exp d=0.3 \| quota | `rank_exp_d0.3` |
| power_law² \| global | `wglobal_power2` |

**Rep-count is unchanged** — all rep-counts are still swept; it is a separate axis, not encoded in this
list. Dropped from the default set: `top_2/3/5` (they welded a rep-count into the weighting slot),
`entropy_1.0/2.0`, `rank_linear`, `rank_exp_d0.5`. None are deleted from the engine; all stay reachable by
name for old configs. `wglobal_power2` was promoted from `EXTRA_ALLOCATION_METHODS` into the default set —
it bypasses the per-fold persistent cache by design, so its global arm recomputes each fold.

Coverage of the five over the 54 cells: a winning hybrid (beats the best single representation) in 54/54;
the cell's outright best strategy is inside the five in 41/54; in the 13 misses the median shortfall is
0.0023 R² (max 0.0096), and every miss was a dropped global arm. Evidence:
`c32_allocation_cross/*.json` and memory `project_c32_allocation_cross_2026_08_23`.
