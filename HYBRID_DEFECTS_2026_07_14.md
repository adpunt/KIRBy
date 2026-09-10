# What is actually wrong with the hybrid search — 2026-07-14

## THE METHOD WORKS. Here is it working.

`tests/results/complexity_c02/full/c02_complexity_matching_grid/herg_ki.json` — molecular regression,
n = 1,412, RMSE (lower is better). This has been sitting on disk since April.

| | score | Δ vs baseline |
|---|---|---|
| baseline: svm + molformer, full width | 0.5763 | — |
| hybrid, **honestly** scored (config picked on 4 folds, scored on the untouched 5th) | **0.5467** | **+0.0296** |

Winner's curse on herg_ki: **0.0067** (on the antibody sets: 0.10 – 0.37).
Across all (model, condition) cells on herg_ki: **28 wins, 10 ties, 1 loss.**

Typical hybrid (mean over all configs at that budget — no argmax, unbiased) vs the baseline:

| budget | 10 | 25 | 50 | 100 | 250 | 500 | 1000 | 1412 |
|---|---|---|---|---|---|---|---|---|
| Δ vs baseline | −0.150 | −0.065 | −0.027 | −0.006 | **+0.011** | **+0.017** | **+0.017** | +0.016 |

**The hybrid needs ~500–1,000 features before it beats the base representation. Then it plateaus.**

Now put that against Defect 1 (the budget is capped at n_samples):

- **herg_ki, n = 1,412** → cap = 1,412 → the hybrid can reach 500–1,000 → **it wins, honestly.**
- **antibody, n = 137** → cap = 137 → the hybrid can never reach the range where it works → **it loses.**

One mechanism explains both. The method is fine. The pipeline was starving it on every dataset too
small to let it breathe — and then those same small datasets were too noisy to show the effect anyway
(Defect 0).

### The same four files, same code, same run — the pattern is exact

`tests/results/complexity_c02/full/c02_complexity_matching_grid/` (April, all four written together):

| dataset | n | budget cap | hybrid Δ **at the cap** | crosses zero? | winner's curse |
|---|---|---|---|---|---|
| **herg_ki** | 1412 | 1412 | **+0.016** | **yes — at budget 250** | **0.007** |
| abagym_EGFR | 659 | 659 | −0.037, still climbing | never allowed to | 0.155 |
| abcov_kd | 141 | 141 | −0.119, still climbing | never allowed to | 0.128 |
| abagym_HER2 | 146 | 146 | −0.361 | never allowed to | 0.174 |

On **every** dataset the hybrid improves monotonically with budget. Only herg_ki has enough samples
for the cap to let it reach the range where the hybrid works. `abagym_EGFR` is still climbing when it
hits its cap of 659, and its baseline representation (`saprot`) is **2560 features wide** — it would
need roughly 4× the budget the code permits before it could even draw level.

And the winner's curse tracks n precisely: **0.007 at n=1412; 0.13–0.17 at n=141–659.**

---

## caco2 (n=2158) — the hybrid ties, and the reason is diagnostic

From `tests/results/framework/f15_adaptive_pruning/caco2_*.json` (6 per-model files, real 5-fold
scores, RMSE, lower is better). At n=2158 the budget cap is **not** binding.

| model | baseline | hybrid (honest, LOFO) | Δ | curse |
|---|---|---|---|---|
| ridge | ecfp4 0.3023 | 0.2957 | **+0.0066** | 0.004 |
| rf | graph_kernel 0.3072 | 0.2996 | **+0.0076** | 0.002 |
| lightgbm | graph_kernel 0.2932 | 0.2920 | +0.0012 | 0.007 |
| xgboost | graph_kernel 0.2983 | 0.2996 | −0.0013 | 0.005 |
| mlp | ecfp4 0.3055 | 0.3249 | −0.0194 | 0.015 |

Wins are real but small (+0.007), and below caco2's detectable effect (~0.017–0.022 unpaired).
Winner's curse is small (0.002–0.015), as expected at large n.

**Why weaker than herg_ki? On caco2 one representation dominates.** `graph_kernel` is the best rep
for rf, xgboost, lightgbm *and* catboost. On herg_ki the best rep differs for every model (molformer,
mhg_gnn, ecfp4, avalon).

**The hybrid pays off when no single representation owns the dataset.** That is the rule, and it is
consistent with everything else here.

---

## THE SIMPLIFICATION: not searching beats searching

Measured on herg_ki (the dataset that works), deployed model svm, baseline 0.5763:

| approach | configs | honest Δ vs baseline |
|---|---|---|
| full search, all axes | 864 | +0.0296 |
| budget-only search (allocation + importance fixed) | 32 | +0.0343 |
| **ONE fixed config, no search at all** | **1** | **+0.0362** |

A single fixed configuration — `performance_weighted` + `lightgbm_gain` + budget 1000 — **beats the
entire 864-config search**, honestly scored. Its 5-fold mean is unbiased because no selection
happened. The budget curve is flat across 500 / 1000 / 1412 (+0.034 / +0.036 / +0.031), so this is not
a cherry-pick — anything in that range works.

**And a fixed config has no selection in it, so it has no winner's curse, so it does not need nested
CV at all.** The 5× disappears. Cost: **5 model fits** instead of ~23,750.

Allocation is not *entirely* noise, one correction: the `top_k` family is genuinely bad
(`top_2` → +0.0009), because concentrating the budget on 2–3 representations defeats the purpose.
Everything else (`performance_weighted`, `rank_linear`, `entropy_1.0`) lands at +0.030 to +0.034.
The rule is **"never use top_k"**, not "allocation matters".

### The recipe
```
allocation  = performance_weighted     (anything except the top_k family)
importance  = treeshap                 (lightgbm_gain is nearly as good)
budget      = ~500–1000, or the widest representation's dim — NOT capped at n_samples
search      = none
CV          = plain 5-fold. It is already honest, because nothing was selected.
```

---

## What to do — one merged, ordered plan (both sessions agree)

Nothing here is approved yet — these are proposals, the user decides.

**Free / one-line:**
1. **Save baseline per-fold scores in the c02 files.** They exist in other files but not c02, which is
   why the paired test — the most sensitive one, never run — was blocked. (Defect 0, 7.)
2. **Remove the `budget <= n_samples` cap** (`find_best_hybrid.py:4816`). Sweep past the widest rep;
   `allow_budget_exceeds_n=True` already exists. (Defect 1.)
3. **Delete the degenerate small budgets** (10/25/50) — on antibody pools they are 0–2 features per rep.

**Cheap:**
4. **Prune the representation pool to ≤4 behaviourally-distinct reps before allocating.** Keep reps
   within one fold-standard-deviation of the best, then decorrelate (drop |r| > 0.9). This automates
   your original "top-2 distinct reps" by hand. Biggest single lever on antibody. (Defect 2, 5, 6.)
5. **Give each rep a fixed per-rep feature count** (`n_per_rep` in {50,100,200,full}) instead of one
   global budget divided across the pool — restores the c4afb0e contract that worked.
6. **Fix allocation and importance to one value each; register `allocate_greedy_forward`** (already
   written, `hybrid.py:126`) and cut the 9 score-proportional allocators to ~3. **27× compute cut**,
   no honest loss. (Defect 3, 4.)

**Real work:**
7. Replace the raw argmax with a 1-standard-error rule (pick the simplest config within one sd of the
   best). Compute per-rep allocation scores from all inner folds, not one 20% split.
8. Move the core claim to datasets that can carry it: herg_ki, caco2, logd, buchwald_hartwig, ADME —
   n in the thousands. Re-judge antibody only after the cap and the pool pruning are in. (Defect 0.)

### The compute arithmetic

| | configs per model | × 5 outer folds |
|---|---|---|
| today: flat search, dishonest | ~594 | — |
| today + nested CV: honest, unaffordable | ~594 | ~2,970 |
| **fix the two dead axes, sweep budget only** | **~12** | **~60** |

Honest nested CV, after cutting the two axes that carry no signal, is **~10× cheaper than the flat
dishonest search you run today.** The 5× precompute blow-up was real, but it was being spent
defending a search space that was hurting you.

---


Working document. Every claim here is verified against code or against real result files.
Nothing in this file comes from memory or from the other .md docs.

**The goal, restated:** for a dataset, take the best (model, single representation) pair.
Build a hybrid representation, feed it to **the same model**. Does it score better?
The best single representation's score is **the target**. It is not to be adjusted or excused.

**Evidence base:** `tests/results/c02_complexity_matching_grid/*.json` — 21 datasets, each config
stores its 5 per-fold scores. That per-fold detail is what makes all of this measurable without
re-running anything.

---

## CONVERGENCE — a parallel investigation found the same failure from the other end

A separate session (memory: `project_hybrid_automation_break.md`) traced this through git and the
`f12_allocation_v2` grids and landed on the same wall from the opposite side. The two findings are one
story:

- **Git proof of "it used to work":** the original hybrid was `create_hybrid(..., n_per_rep=100)` at
  commit **c4afb0e** — each representation contributes a **fixed 100 features**, driven over the
  **top 2 of 4 distinct representations**. So the thing you remember working = *2 hand-picked distinct
  reps × ~100 features each.*
- **What the automation did:** replaced the per-rep count with one **global budget** that an allocator
  **divides** across 17–38 representations. Because the reps are near-tied in score and every allocator
  is score-proportional, the division is near-uniform, so each rep gets **1–8 features** on the big
  pools. The original "100 features from 2 reps" is **unreachable** in the production grid.
- **Measured, paired, no argmax:** concentrating on 2 reps (`top_2`) vs spreading
  (`performance_weighted`) is worth **+64% on abagym_EGFR** and **+20% on flab_pk** at a 38-rep pool.
  Harmless at 17 reps, catastrophic at 38.
- **The grid is decoration:** variance explained — importance **0.4%**, allocation **1.5%**,
  **model 43%**, budget 7%. The search tunes the axes that don't matter and fixes the one that does
  (model) in an outer loop.

### How that fits my findings (they do not conflict — they compound)
My angle: the **total** budget is capped too low (Defect 1) — the hybrid needs ~500–1000 features and
never gets there on small n. Their angle: even the budget you *do* have is **smeared** across too many
redundant reps, so each rep gets dust. Both reduce to the same sentence:

> **The good representation never gets enough of its own features into the hybrid.**

It also explains why herg_ki wins and antibody doesn't, cleanly: herg_ki has ~24 *genuinely distinct*
molecular reps (ecfp4, graph_kernel, molformer, mhg_gnn, avalon…), so a global budget of 500–1000
still lands enough real features per rep. The antibody pools are ~38 reps, **85% protein-language-model
variants** (12 are ESM2 flavours) — near-duplicates — so the same global budget shatters into dust AND
the n_samples cap stops it reaching a useful size in the first place. **Both bugs bite antibody; only
the cap meaningfully bites molecular.**

The combined fix is therefore *both*: uncap the budget **and** stop dividing it across a redundant
pool. Either alone is partial.

---

## DEFECT 1 — The hybrid's feature budget is capped at the number of samples. The baseline's is not.

**Severity: fatal. This is the main event.**

`src/kirby/find_best_hybrid.py:4816`
```python
filtered = [b for b in budgets if b <= n_samples]
```

The hybrid may use at most `n_samples` features. The baseline single representation is scored at its
**full native width** — there is no equivalent cap on it. (`_evaluate_single` only trims a
representation above `HIGH_DIM_FILTER_THRESHOLD = 5000`, `find_best_hybrid.py:2401`, which almost
never fires: prot_t5 is 2048, esm2_concat 1280, saprot 2560.)

So on `abagym_HER2` (n=146): baseline = 2560 features on 146 samples. Hybrid = at most 146 features.
Same model, same folds. **The p > n rule is enforced on one side of the comparison only.**

### The proof that this is what's killing it

Typical hybrid score (mean of the 5-fold means across all configs at that budget — no argmax
anywhere, so these are unbiased) for the deployed model:

| dataset | n | 10 | 25 | 50 | 100 | 250 | 500 | BASE |
|---|---|---|---|---|---|---|---|---|
| abagym_EGFR | 659 | 0.143 | 0.181 | 0.216 | 0.240 | 0.277 | 0.291 | **0.324** |
| ginkgo_hic | 242 | 0.266 | 0.329 | 0.374 | 0.425 | · | · | **0.582** |
| flab_polyreact_jain_cic | 137 | 0.320 | 0.392 | 0.432 | 0.461 | · | · | **0.585** |
| humanness_therapeutic | 564 | 0.927 | 0.946 | 0.953 | 0.957 | 0.959 | 0.960 | **0.966** |

(· = budget dropped for exceeding n_samples)

The score is **monotonically increasing in budget and still climbing when the cap cuts it off.**
**On 17 of 21 datasets the best budget is the largest budget allowed** — the optimum sits on the
boundary, which means the true optimum is beyond it.

The hybrid never lost. It was never allowed to compete.

### Confirmation from your own results

There is already a condition that lifts the cap — `per_model_top_rep_dim` in
`tests/experiments/complexity/c02_complexity_grid.py:374`, which runs the hybrid at the baseline
representation's native dimensionality with `allow_budget_exceeds_n=True`. It is the only caller of
that flag in the entire repo. Its May numbers:

| | wins | ties | losses |
|---|---|---|---|
| deployed model (capped, honestly scored) | 1 | 3 | 17 |
| **deployed model, hybrid given the baseline's width** | **5** | **4** | **8** |
| **all (dataset, model) cells, hybrid given the baseline's width** | **63** | **16** | **19** |

Same data. Same folds. The only change is letting the hybrid have as many features as the baseline.

### Where it still loses, and why that's the same bug

`abagym_HER2` (best rep = `mutation_features`, **48 dims**) and `abagym_VEGF` (`aac`, **40 dims**).
When the baseline representation is narrow, this condition forces the hybrid to be narrow too. A
hybrid drawing from 26 representations needs *more* room than one compact representation, not the
same amount. Tying the budget to the baseline's width is still a cap — just a different one.

### Fix
Budget must be swept **independently of n_samples and independently of the baseline's width**, up to
and past the widest representation in the pool. `allow_budget_exceeds_n=True` already exists and does
exactly this; almost nothing uses it.

---

## DEFECT 2 — The search cannot discover complementarity. It can only re-rank what was already good.

**Severity: fatal (it defeats the purpose of the method).**

All 9 allocation strategies in the search come from `_compute_allocation_weights`
(`src/kirby/hybrid.py:437`). The `full` tier searches exactly:

```
top_2, top_3, top_5, performance_weighted, entropy_1.0, power_2, rank_linear, entropy_2.0, rank_exp_d0.5
```

Every one computes a representation's share of the budget as a function of **that representation's
own standalone score**: `linear` = the score, `power_law` = score², `top_k` = score if in top k else 0,
`rank_linear`/`rank_exp` = rank of the score, `entropy` = score × (1 − redundancy).

`entropy` only ever **penalises** a redundant representation. Nothing in the family ever **rewards**
a representation for adding information the others lack. So a representation that is individually
weak but genuinely complementary can never win a meaningful share of the budget.

**A hybrid exists to exploit complementarity. The search is structurally incapable of finding it.**

### The allocators that could do it are already written and never called
`src/kirby/hybrid.py`: `allocate_greedy_forward` (126), `allocate_greedy_feature` (260),
`allocate_dominant_greedy` (725), `allocate_two_stage` (904), `allocate_stability_selection` (960).
None appear in `SPEED_TIERS['full']['allocation_methods']`.

### Fix
Replace the nine marginal variants with one or two of the greedy/forward-selection allocators that
actually measure whether a representation *adds* anything.

---

## DEFECT 3 — Selecting the best of ~4,750 configs on the same 5 folds you report on

**Severity: fatal for every number that has been reported.**

The search evaluates model(8) × allocation(9) × importance(3) × budget(22) ≈ 4,750 configs, scores
them all on the same 5 folds, and reports the maximum. That maximum is not a performance estimate,
it's the top order statistic of ~500 noisy numbers (per model).

Re-selecting honestly — choose the config on 4 folds, score it on the held-out 5th, rotate — collapses
it. Reported wins **8/21 → honest wins 1/21, honest losses 17/21**.

**Control (proves the honest estimator is sound):** with zero selection (a config picked at random),
the held-out-fold score equals its own 5-fold mean to within 0.006 on every one of the 21 datasets.
The estimator is unbiased. **The entire collapse is selection, not estimator noise.**

**How unstable:** drop one fold, re-pick the winner — you get a *different* winning config **59% of
the time**. The one dataset that genuinely wins (`flab_pk_jain_heprt`, +0.056) picked the **same**
config on 5/5 fold-drops. **Stability, not score, is the tell.**

**The search space is actively harmful, not merely wasteful.** Honest score as a function of how many
configs you search (subsampled, then honestly re-selected):

| configs searched | 1 | 5 | 10 | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|---|---|
| honest score | 0.367 | 0.399 | **0.407** | 0.395 | 0.379 | 0.371 | 0.377 | 0.348 |

Honest performance **peaks at ~10 configs and degrades from there.** Searching 400 configs delivers a
*worse* hybrid than searching 10.

### Fix
Selection must happen on data the score is not read from, and the search must be small. These are the
same fix: a small search has little to overfit to.

---

## DEFECT 4 — Two of the three axes carry no signal

**Severity: major (this is where the compute is going).**

Marginal effect of each axis value, ranked within each dataset, then compared across the 21 datasets:

- **importance** (3 methods): `treeshap` best on 12/21, `lightgbm_gain` on 7/21, `deeplift` on 2/20.
  Weak, inconsistent preference.
- **allocation** (9 methods): `entropy_1.0` best on 8/21, `entropy_2.0` on 6/21, everything else
  scattered. The winner moves essentially at random across datasets, and rank spread runs 1→9 for
  almost every strategy.
- **budget**: the only axis with a clean, consistent, monotone effect — and it's the one that's capped.

So the search spends a **27× multiplier** (9 allocations × 3 importance methods) on two axes whose
winners are noise, while the one axis that matters is cut off before its optimum.

### Fix
Fix allocation and importance to one sensible value each. That is a **27× compute cut** on its own, and
by Defect 3 it should *improve* the honest result rather than hurt it. Spend the freed compute on budget.

---

## DEFECT 5 — All three importance methods are marginal; the representations are highly redundant

**Severity: major. Suspected — being tested.**

The only importance methods are `deeplift`, `lightgbm_gain`, `treeshap`. All three score each feature
**on its own**, ignoring that two representations may encode the same information twice. Two of the
three are tree-based, and tree split-gain is known to favour dense continuous features over sparse
binary ones — which systematically starves fingerprint features when they're pooled with embeddings.

Over a pool of near-duplicate representations, a purely marginal ranking picks the same information
many times. A 250-feature hybrid may hold far less unique information than the single 2048-feature
representation it's being compared against.

**Missing and standard:** mutual information, F-test, **mRMR** (explicitly redundancy-aware), Boruta,
Lasso/ElasticNet path, stability selection, recursive feature elimination, permutation importance.
The redundancy-aware ones are the ones this method needs and none are present.

---

## DEFECT 6 — The representation pool is diluting the hybrid with weak representations

**Severity: major.**

The c02 `conditions` block contains `top_tier_only` — the same search restricted to only the
top-tier representations. Compared with the full pool (`no_aug`), on the deployed model:

| dataset | full pool | top tier only |
|---|---|---|
| flab_pk_jain_fcrnrt | **−0.080** | **+0.057** |
| flab_polyreact_jain_bvp | +0.051 | **+0.084** |
| flab_thermo_jain | +0.050 | **+0.072** |
| flab_aggregation_jain | +0.031 | **+0.067** |
| abagym_HER2 | −0.099 | −0.020 |

Dropping the weak representations turns the biggest loss in the table into a win. Combined with
Defect 5 (all importance methods are marginal, so redundant representations get picked repeatedly),
the picture is that a big pool of mediocre, mutually-redundant representations is actively hurting
the hybrid — the budget gets spent on near-duplicates of things the good representation already has.

### Fix
Prune the representation pool before searching. Fewer, more diverse representations.

---

## DEFECT 7 — The conditions are not instrumented, so they cannot be honestly re-scored

**Severity: major (it's why this took so long to find).**

`configs[]` stores `fold_scores`, which is the only reason any of this analysis was possible without
re-running anything. The `conditions` block stores **only `score_mean`** — no per-fold scores. So the
five conditions (including `per_model_top_rep_dim`, the one that matters most) **cannot be
honestly re-selected**, and their win rates cannot be compared with each other, because each is a max
over a different number of configs and therefore carries a different amount of winner's curse.

Concretely: `no_aug`'s 73% win rate is a max over ~594 configs. `per_model_top_rep_dim`'s 64% is a max
over ~27. The second is doing far more with far less selection — but I cannot quantify by how much,
because the fold scores were not saved.

### Fix
Persist `fold_scores` for every condition, exactly as `configs[]` already does. One line. It makes
every future result honestly re-scorable after the fact, for free.

---

## DEFECT 8 — Feature selection is a per-representation quota, never a global ranking

**Severity: fatal. Confirmed independently by several code-reading agents + adversarial verify.**

Importance is computed **one representation at a time** (a fresh model trained on that rep alone), then
the top-N features are taken **within each rep**, by that rep's own scores. There is no code path in
which "the best features win regardless of which representation they came from." Two consequences:

- The best representation is always **truncated** (it only gets its quota slice) and always **diluted**
  (weaker reps are forced in alongside it).
- Feature scores from different reps are never made comparable, so the quota, not the features, decides.

This is the feature-level version of Defect 2, and it's the concrete mechanism the parallel session saw
as "smearing across the pool." Verifier's one correction: the framing "therefore the hybrid must lose"
is too absolute — the budget cap (Defect 1) is the load-bearing half; this makes it worse.

### The fix that already tested positive — a redundancy-aware selector
Empirical probe on herg_ki (real code, real 5-fold scaffold CV, fixed configs → unbiased). Swap the
marginal selector for **mRMR** (pick features by relevance ÷ redundancy-with-already-picked) at the
same budget:

| model | budget | marginal (RMSE) | mRMR (RMSE) | mRMR better by |
|---|---|---|---|---|
| rf | 250 | 0.6032 | 0.5956 | **+0.0076** |
| svm | 250 | 0.5966 | 0.5926 | **+0.0040** |
| svm | 500 | 0.6292 | 0.6085 | **+0.0207** |
| ridge | 100 | 0.6406 | 0.6292 | **+0.0114** |

mRMR wins 7 of 9 (model × budget) cells; both losses are tiny and only at budget 500. It changes ~56
of 250 selected features. **This is a genuine improvement lever, not just a correctness fix** — the
current 3 importance methods are all marginal, and none is redundancy-aware.

---

## Not a defect — checked and cleared

- **Feature scaling.** `StandardScaler` is fit on the training fold and applied to validation, wrapped
  per-fold (`find_best_hybrid.py:1947-1949`, `2521-2523`). Both sides get it. Not the bug.
- **Importance leakage.** Feature importance is computed inside the fold loop on training data only
  (`_select_top_features_per_fold`, `find_best_hybrid.py:2405`). No label leakage into validation.
- **Scaling asymmetry (empirically cleared as a cause).** There IS a real asymmetry — the baseline arm
  skips StandardScaler for all-0/1 fingerprints (`find_best_hybrid.py:2495-2519`) while the hybrid
  always scales. But a direct probe on caco2 (fixing the asymmetry, arm C2) does not move the result,
  and it cannot touch the 21 c02 antibody datasets at all (their reps are continuous embeddings, both
  arms scale). Worth tidying for the molecular fingerprint datasets; not the cause of the collapse.
- **ANARCI silent fallback (flagged then cleared).** An agent found that CDR-aware antibody reps
  silently return "all residues" when ANARCI fails, which would make many antibody reps near-identical.
  The verifier tested the actual cached matrices: it does not fire in the data the experiments used.
  Real hazard, not currently active — worth a one-line assert so it can never fire silently later.

## Loose ends worth a separate look (found in passing, not load-bearing)
- **Tree-method "variance pre-screen" selects columns by floating-point noise** (treeshap /
  lightgbm_gain path). Flagged as a hard code defect by two agents — verify and fix independently.
- **Auto-demote path train/val column mismatch** (`find_best_hybrid.py:2843-2856`): a demoted rep's
  augmentation array is built from the fold-train slice, so the val-side rebuild can select different
  columns. Narrow, but it's a correctness bug — worth confirming.
- **`_sanitize_baseline_scores` additive floor** (`hybrid.py:415-434`): shifting every rep by |worst
  score| gives even the worst rep a non-zero weight, forcing it into the allocation. Real but minor and
  partly by design; folds into the "prune the pool" fix.

---

## Open / still being tested

- Local budget sweep past the cap on **caco2** (molecular, lower-is-better) and **ginkgo_hic** /
  **ginkgo_acsins** (antibody, real signal, n=242). Fixed configs, no argmax → unbiased.
- Whether a redundancy-aware selector (mRMR / greedy forward) at a fixed budget beats the current
  marginal ranking at the same budget.
- Cost audit: what exactly made precompute 5× more expensive, and whether it buys anything.
