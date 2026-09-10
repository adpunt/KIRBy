# c32 allocation-strategy analysis — handoff prompt

Paste this into a fresh session. It is self-contained. Read it in full before doing anything.

---

## The task in one sentence

The c32 sweep is finished; analyse it by **counting, per full strategy, how many cells it produced the best hybrid in** — and hand the user a small set of strategies that reliably win, for the user to choose from.

---

## ⛔ ERRORS FROM THE LAST SESSION — DO NOT REPEAT THESE

These are the specific mistakes made building this. Each one wasted the user's time. Read them before touching the data.

1. **NO averaging. NO mean rank. NO significance tests.** The winning strategy is different in every cell — that per-cell variation IS the result. The previous session computed a "mean rank across datasets" and the user (correctly) called it useless: it erased the variation that is the entire point. **Aggregate ONLY by counting** (how many cells a strategy is 1st in, 2nd in). Never collapse cells with a mean, median, or confidence-interval overlap test.

2. **A "strategy" is the FULL triple: (weighting × mechanism × N).** For example `power_2 | global | N=3`. Report every strategy at this full granularity. **Never** collapse to "families" or report a half-strategy like "top_2 wins" — that says nothing about whether the weighting, the mechanism, or the rep-count did the work. The user's exact words: "top_2 being a real winner is absolutely meaningless... I don't give a fuck about families."

3. **Check feature counts are equal before comparing.** In an earlier experiment (f01) one strategy (`fixed`) secretly received 10–20× more features than its rivals in the same cell, so its "wins" were a bigger-budget artifact, not a real result. c32 was built to give every strategy the same budget B — but **verify it**: read `nfeat` for every strategy in a cell and confirm they're all ≈ B before trusting any win. Don't assume.

4. **Present it in the format the user asked for**, not prose: a table listing **every strategy**, with columns = weighting, mechanism, N, #cells-won (top-1), #cells-top-2, and the **margin** (when it wins, how far ahead of 2nd place). The user called the earlier tables "dogshit" because they showed mean ranks instead of these counts.

5. **Plain English. Define every term. Answer first.** The user spent most of this session asking for plain language. Do not use jargon without spelling it out. Do not cite line numbers as "proof." When the user says they're confused, stop and restart from zero.

6. **Do not write conclusions into repo docs, and do not pick the final set yourself.** Present the counts in chat. The user is an engineer and makes the selection decision. Propose a recommended small set with the evidence; let them decide. No autonomous design.

7. **Verify paths and field names by reading the files** — don't assert structure from memory.

---

## 0. The data

- Location: `/Volumes/seagate/kirby_cache/c32_allocation_cross/*.json`
- **27 files** = 9 datasets × 3 models, named `<dataset>__<model>.json`.
- **qm9 is excluded by the user** (its rep cache was corrupt; not re-run, not needed). Do not include qm9.
- The 9 datasets: `adme_hlm, adme_mdr1, adme_rlm, bh_doyle_plates, buchwald_hartwig, caco2, flab_expression_koenig, herg_ki, logd`.
- Models: `lightgbm, rf, catboost`. Budgets: `500, 1000`.

## 1. What a "cell" is, and the strategies in it

A **cell** = one (dataset × model × budget) combination. 9 × 3 × 2 = **54 cells**.

Inside each cell there are **32 strategies** = 4 N-values × 4 weightings × 2 mechanisms:
- **N** (how many of the best reps are used): `2, 3, 5, 10` (10 = all reps).
- **weighting** (how budget splits among those reps): `linear, power_2, softmax_T1, rank_exp_d0.3`.
- **mechanism**: `quota` (each rep gets its own slice of budget) or `global` (all reps' candidate columns pooled, one ranking picks the final set).

Every strategy in a cell gets the same total feature budget B, so they are directly comparable.

## 2. How to read the JSON (verify this by opening one file first)

Each `<ds>__<model>.json` has this shape:
```
d["datasets"][ds]["by_model"][model]["strategies"]["B500"]  and ["B1000"]
```
Each of those is a dict keyed by strategy id `"<weighting>|<mechanism>|N<N>"`, e.g. `"power_2|global|N3"`, whose value has:
- `r2_delta_mean` — **the number to rank by.** It is this strategy's R² minus the target's R², averaged over 5 folds. **Higher is always better** (already sign-corrected for metric direction). Positive = beats the target; negative = loses to it.
- `r2_folds_won` — of 5 folds, how many this strategy beat the target on.
- `r2_delta_ci95` — a wide 5-fold bootstrap interval. **Information only — do NOT gate on it.**
- `native_delta_mean` — same idea in the dataset's own metric.
- `nfeat` — feature count (check it ≈ B).
- `scheme`, `mech`, `N`, `B` — the three axes, already split out for you.

The **target** each strategy is compared against is `["by_model"][model]["target"]` = the best single representation for that dataset×model, at full width. So `r2_delta_mean > 0` means "this strategy beats just using the single best representation."

## 3. The method — counting, not averaging

1. Load all 27 files. Build the 54 cells. In each cell you have 32 strategies with `r2_delta_mean`.
2. **Sanity check:** in a few cells, confirm all 32 `nfeat` ≈ B. If any strategy has a very different feature count, stop and report it (see error #3).
3. In each cell, rank the 32 strategies by `r2_delta_mean`, highest first.
4. For **every** strategy, across the 54 cells, count:
   - **top-1**: how many cells it ranked 1st in.
   - **top-2**: how many cells it was 1st or 2nd in.
   - **beats-target**: how many cells its `r2_delta_mean` was positive (i.e. beat the best single rep at all).
   - **margin when it wins**: in the cells it ranked 1st, the gap `r2_delta_mean(1st) − r2_delta_mean(2nd)`. Report the median and range. (This is "how far ahead of second place" — a big margin = a decisive win, ~0 = a coin-flip.)
5. Present **one table**: one row per strategy (all 32), columns = weighting, mechanism, N, top-1, top-2, beats-target, median win-margin. Sort by top-1. (All strategies appear in all 54 cells, so raw counts are directly comparable — no rate needed, but count/54 is fine to show.)
6. Then **three marginal count tables** to see which axis actually matters — again counts, never means:
   - by **mechanism**: quota won how many of 54 cells vs global.
   - by **N**: N=2 won how many, N=3, N=5, N=10.
   - by **weighting**: linear won how many, power_2, softmax_T1, rank_exp_d0.3.
   (For each axis, "won" = the cell's 1st-place strategy had that axis value.)
7. Note honestly: winners will vary a lot by dataset and model (that is expected and is the point). At **N=2** the four weightings are nearly identical (only 2 reps to split between), so expect ties there — flag it, don't treat it as signal.

## 4. The deliverable

The endpoint (from the allocation handoff, §5.3): **a small set of strategies that are reliably at or near the top across cells** — not the single argmax per cell (that overfits), and not the whole grid. Use top-1 + top-2 counts and win-margin to identify a few robust strategies. **Present the counts and a recommended small set; let the user pick.** Do not edit any source or docs to record a conclusion.

## 5. Pointers

- Sweep script (how the numbers were made): `tests/experiments/complexity/c32_allocation_strategy_cross.py`.
- Full build context + the `fixed` artifact + why prior data couldn't answer this: memory `project_c32_allocation_cross_2026_08_23.md`.
- Original allocation plan and its own list of past errors: `ALLOCATION_STRATEGY_HANDOFF_2026_08_21.md`.
- The engine is not to be edited or called; this is read-only analysis of JSON files.
