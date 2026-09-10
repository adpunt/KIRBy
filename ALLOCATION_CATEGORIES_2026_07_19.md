# Allocation categories — which to keep, which to cut

Workstream 4.2 (which allocation categories are actually distinct) and 4.3 (is the fix
top-N, global ranking, or normalisation) of the `find_best_hybrid` rebuild.

Date: 2026-07-19. Branch: `similarity-metrics-study`.

Datasets measured: `herg_ki` (molecular), `logd` (molecular), `flab_thermo_tresanco`
(antibody). `buchwald_hartwig` (reaction) **did not finish** — see section 4 and the
caveats. Nothing in this document is averaged or medianed across datasets. Every number
belongs to one dataset.

---

## 1. Headline

**Answer first: the per-representation quota is not the defect. None of the three
proposed fixes — cutting to fewer representations ("top-N"), ranking all features
globally regardless of which representation they came from, or rescaling the
representation scores before dividing the budget — produces a win on any dataset that
finished. Two of the three are measurably worse than what is already in the code.**

The single clearest finding, and it is new:

> **Each representation's usable signal saturates at roughly 100–200 of its features,
> regardless of how wide the representation is.** On `herg_ki`, the ECFP4 fingerprint's
> full 2048 columns are worth no more than its best ~130 columns. On `logd`, Mordred's
> full 1613 columns are worth no more than its best ~175. On `flab_thermo_tresanco`,
> SaProt-CDR's full 1280 columns are worth no more than its best ~100. Above that point
> the truncated slice scores the same as the full representation, sometimes very slightly
> better (the tail is noise). Below roughly 50 columns it falls off a cliff.

That single fact reframes the whole phase. The question "does the quota cut the best
representation's informative features out?" has a precise answer: **only when the quota
lands below that saturation point, which happens exactly when
`total_budget / number_of_representations < ~150`.** At a total budget of 100 across ten
representations each representation gets 10 features and the damage is large and
statistically significant at the representation level. At a total budget of 1000 across
ten representations each gets 100 and the damage disappears.

And the second clearest finding, which is uncomfortable: **on `herg_ki` and
`flab_thermo_tresanco` that damage does not propagate to the hybrid at all.** The
configurations whose best representation is starved to 0.5% of its width score *better*
as full hybrids than the configurations that keep it nearly intact. The other
representations refill whatever the best one loses, because in these pools the
representations are near-redundant. On `logd` the damage does propagate weakly, and the
budget is the lever that fixes it.

### Categories that survive

Ten allocation settings were tested. They collapse into **four genuinely distinct
behaviours**, and only three are worth keeping.

| Keep | What it does | Why keep it |
|---|---|---|
| `fixed` | Every representation gets the same number of features (`budget / n_reps`). | Consistently top-2 or top-3 on every dataset; the nine weighted schemes converge onto it anyway. Simplest possible thing that works. |
| `performance_weighted` | Budget split in proportion to each representation's internally measured score. | The reference spread scheme. Best single cell on `herg_ki` (+0.0115). Keep one of `performance_weighted` / `power_2` / `rank_linear`, not all three. |
| `top_3` | Hard cut to the three best-scoring representations, budget split among them. | The only concentration behaviour that is ever competitive (`flab` +0.0424 at pool10 B1000). Keep as the *concentration* representative — but see the warning in section 3. |

| Cut | Why |
|---|---|
| `power_2`, `rank_linear`, `rank_exp_d0.5` | Monotone rescalings of `performance_weighted`. On every dataset they land inside the same ±0.005 band. Not distinct behaviours. |
| `top_2` | Actively dangerous. Drops the genuinely best representation entirely in 2–5 of 5 folds on every dataset, because the internal ranker disagrees with the cross-validation ranking. |
| `top_5` | Degenerate whenever the pool has five representations (identical to `performance_weighted`, `find_best_hybrid.py:3521`). Only meaningful on a ten-representation pool, where it is mid-pack. |
| `entropy_1.0`, `entropy_2.0` | Measure the wrong thing (see section 6) and collapse to 1–3 representations. `entropy_2.0` is not a hybrid at all. |

---

## 2. Per dataset

### 2.1 `herg_ki` — molecular, 1412 rows, 769 scaffold groups

Metric is root-mean-square error (RMSE); **lower is better**. Sign convention throughout:
a positive delta means the candidate beat the target.

**Target (best representation × best model): ECFP4 fingerprint × random forest,
RMSE 0.5993871**, per fold `[0.5723, 0.6504, 0.6253, 0.5804, 0.5685]`. Re-scored inside
this stage and reproduced the baseline stage exactly, so every paired comparison is
fold-aligned.

What the categories did:

- The spread family (`performance_weighted`, `power_2`, `rank_linear`, `fixed`) keeps
  every representation in every fold and allocates almost uniformly — e.g. at the
  ten-representation pool with budget 100, `performance_weighted` gave ECFP4
  `[10,10,9,10,10]` features and `fixed` gave `[10,10,10,10,10]`. Nine weighting schemes
  arriving at the same answer.
- The hard-cut family excluded the best representation outright. On the
  ten-representation pool, `top_2` and `entropy_2.0` gave ECFP4 **zero** features in
  5 of 5 folds at every budget; `top_3` in 4 of 5; `top_5` in 3 of 5. Across all 57
  configurations, 8 gave the best representation zero features in all five folds.
- Best result anywhere on this dataset: **+0.0115** (ten-representation pool, budget
  1000, `performance_weighted`), or **+0.0121** with the score-rescaling arm. Nothing
  reached statistical significance; the smallest p-value at hybrid level was 0.124.

**Beat the target:** the spread family, at every budget, by roughly +0.008 to +0.012.
**Lost to the target:** every hard-cut arm (`top_2` −0.0021 to −0.0134, degrading as
budget rises; `top_3` −0.0062 to −0.0109), and the global-ranking arm.

More representations helped, not fewer: at budget 100 the ten-representation pool gave
`performance_weighted` +0.0115 versus +0.0020 on the five-representation pool.

### 2.2 `logd` — molecular, 5028 rows, 2413 scaffold groups

Metric RMSE, lower is better. **Target: Mordred descriptors × LightGBM, RMSE 0.5051906**,
per fold `[0.4894, 0.5515, 0.4920, 0.5216, 0.4715]`, Spearman 0.9015.

This is the one dataset with a real, unanimous win.

- **Best cell: ten-representation pool, budget 1000, `power_2` — RMSE 0.4725, delta
  +0.0327, winning in 5 of 5 folds.** A 6.5% relative reduction in error, using 1000
  columns against the baseline's 1613.
- `fixed` came second at +0.0301, within 0.0026 of the best, while giving Mordred only
  6.2% of its native width.
- Budget is the lever, not the scheme. Moving from budget 100 to budget 1000 is worth
  about +0.03 and flips every arm from losing to unanimously winning. Changing the
  allocation scheme at a fixed budget is worth at most about +0.010 and usually +0.005.
- The best representation was never zeroed out on this dataset — every category gave
  Mordred a non-zero quota in every fold.
- Concentration (`top_2`, `top_3`) was the only non-negative behaviour at budget 100
  (+0.0006 / +0.0008) but was mid-pack to worst at budgets 500 and 1000. It helps only in
  the starved regime and costs you the win in the regime that actually works.

A useful ceiling measurement: throwing in **every** feature with no selection at all
(15139 columns) scores +0.0233. The best quota hybrid at +0.0327 beats that with fifteen
times fewer columns. So the total complementarity available in this pool is worth roughly
+0.02 to +0.03, and the hybrid is already at that ceiling.

### 2.3 `flab_thermo_tresanco` — antibody, 672 rows, 309 groups

Metric Spearman rank correlation; **higher is better** (opposite direction to the two
molecular sets — this is deliberate, it is how the metric-direction gate proves itself).

**Target: SaProt-CDR × random forest, Spearman 0.5959**, per fold
`[0.7836, 0.7195, 0.4613, 0.5245, 0.4905]`.

- 54 of 57 configurations beat the target. Best cell +0.0448 (five-representation pool,
  budget 100, `rank_exp_d0.5`); `top_3` at +0.0360 and +0.0424 in other cells.
- **The most striking single fact in the whole study is here:** both entropy settings gave
  the best representation **zero** features in all five folds at all three budgets on both
  pools — the best representation was completely removed — and `entropy_1.0` at the
  five-representation pool, budget 100 was still the **fourth-best cell in the study**
  (+0.0404). If quota-truncation of the best representation were the binding constraint,
  a configuration that truncates it to zero could not sit near the top.
- **But nothing here is statistically meaningful.** The target's own fold scores span
  0.4613 to 0.7836 — a 0.32 range around a 0.596 mean, roughly seven times larger than the
  biggest effect measured. The standard error on any cell's mean delta is about 0.024, and
  the entire spread between allocation categories is −0.006 to +0.045, about two standard
  errors wide in total. The only claim that survives is the direction (54 of 57 positive).

### 2.4 `buchwald_hartwig` — reaction: DID NOT RUN

The script was written, launched, and executing correctly, but had only completed dataset
load, the metric-direction gate, the target re-score, and the feature precomputation when
the time budget ran out. **Zero of the five measurements completed.** See section 4.

---

## 3. The 4.3 verdict, with evidence

**The question was: is the fix (a) cutting the pool to fewer representations, (b) ranking
all features globally regardless of origin, or (c) rescaling the representation scores
before splitting the budget? The measured answer is: none of them.**

### 3.1 Signal survival — does the quota actually destroy the best representation's signal?

The test: take **only** the columns of the best representation that the hybrid actually
kept (chosen by that representation's own LightGBM gain ranking, inside the training
fold), train the downstream model on just those, and score on the untouched validation
fold. Compare, paired fold by fold, against the same representation at its full native
width.

First, the reference curve — how much of each representation you actually need:

| Dataset / best representation | Native width | Score at 10 features | at 50 | at 100 | at 200 | Saturation point |
|---|---|---|---|---|---|---|
| `herg_ki` / ECFP4 (RMSE delta vs full) | 2048 | −0.1069 | −0.0328 | −0.0032 | +0.0056 | ~100–130 |
| `logd` / Mordred (RMSE delta vs full) | 1613 | −0.1325 | −0.0243 | −0.0105 | +0.0010 | ~150–200 |
| `flab` / SaProt-CDR (Spearman delta vs full) | 1280 | −0.0351 | −0.0175 | −0.0009 | +0.0014 | ~100 |

So each representation's whole contribution fits in roughly 100–200 of its features, and
the remaining 1000–1900 columns are noise.

Now the realised quotas against that curve:

**Where the quota lands below saturation, the damage is real and significant.** On
`herg_ki`, ten-representation pool, budget 100, ECFP4 gets 10 of 2048 features (0.49% of
its width) and its slice loses 0.10 to 0.20 RMSE against full width, at p = 0.0002 to
0.006, losing in 0 of 5 folds won. That is a 17–33% relative degradation and it is the
only statistically significant effect anywhere in this phase. Same picture on `logd`
(slice cost −0.117 to −0.133) and on `flab` (−0.019 to −0.060).

**Where the quota clears saturation, the damage vanishes.** On `herg_ki` at budget 1000
the quota reaches ~100–200 features and the slice delta is −0.0032 to +0.0116 with
p between 0.16 and 0.71. On `logd` at budgets 500 and 1000 the slice cost is −0.021 to
+0.003. On `flab` at budgets 500 and 1000 the slice is within ±0.018 of the full 1280
columns.

**And the damage does not propagate to the hybrid.** This is the decisive negative:

| `herg_ki`, ten reps, budget 100 | Best-rep slice alone | Same config as a full hybrid |
|---|---|---|
| `performance_weighted` | −0.1004 (p = 0.0003) | **+0.0115** (best cell in the study) |
| `fixed` | −0.1069 (p = 0.001) | **+0.0104** |
| `top_5` at budget 1000 | +0.0067 | **−0.0033** |

Slice performance and hybrid performance are **anti**-correlated across all 44
`herg_ki` configurations where a slice exists. On `logd` the other representations
recover roughly 0.10–0.12 RMSE of the truncation damage. On `flab` the best
representation can be removed entirely and the hybrid still wins.

**Conclusion: "the quota truncates the best representation's informative features" is
CONFIRMED as a fact about the features and REFUTED as an explanation of hybrid
performance.**

### 3.2 Global ranking counterfactual — REJECTED, on measurement

Built as a measurement only, with `hybrid.py` untouched: concatenate every representation
in the pool into one matrix, compute one LightGBM gain importance over the whole thing,
take the global top-B columns regardless of origin, score on the same folds.

| Dataset / pool / budget | Global ranking delta | Best quota hybrid at same budget |
|---|---|---|
| `herg_ki` pool10 B100 | −0.0060 | +0.0115 |
| `herg_ki` pool10 B1000 | −0.0041 | +0.0102 |
| `herg_ki` — all 6 cells | 5 of 6 negative, best +0.0004 | quota wins every matched cell |
| `logd` pool10 B1000 | −0.0033 | +0.0327 |
| `logd` pool5 B1000 | +0.0116 | +0.0261 |
| `logd` — all 6 cells | quota wins 5, ties 1 | — |
| `flab` pool5 B1000 | +0.0196 | +0.0326 |
| `flab` pool10 B1000 | **−0.0390** | 27 of 30 pool10 quota cells positive |

**And the mechanism is the point: global ranking gives the best representation FEWER
features than the quota does, not more.** On `logd` at budget 1000 the global top-1000
contained 546 columns from Grover and only **29 from Mordred, the best representation** —
and gave two representations zero. On `flab` at budget 1000 it dumped 540 of 1000 slots
into a single representation. On `herg_ki` the origin counts at budget 100 were
`{mhg_gnn 64, atom_pair 17, ecfp4 9, graph_kernel 7, top_torsion 3}`.

The reason is that LightGBM gain is not comparable across representations of different
density. Dense continuous embeddings accumulate gain across many tree splits; sparse
binary fingerprint bits are each used once. A single global importance therefore collapses
onto whichever representation the model happened to split on first. **The per-representation
quota's structural guarantee — every representation gets some columns — turns out to be
acting as a regulariser, and it is doing real work.**

*Honest limit:* this rejects **gain-based** global ranking. A scale-invariant importance
(per-representation-normalised gain, permutation importance, rank-normalised scores) was
not tested and might behave differently.

### 3.3 Normalisation — not rejected, but not a lever

The pathology being tested: `_sanitize_baseline_scores` (`hybrid.py:415`) shifts the
internal scores by their **minimum**, so the worst representation always gets weight
exactly zero and the weights encode only the differences between near-tied
representations. Two rescalings were fed in through the precomputed-scores path, so
`hybrid.py` was never modified.

- `herg_ki`: "skill over a null model" rescaling was **positive in 6 of 6 cells**, best
  +0.0121 at ten representations / budget 1000 — the single best number produced anywhere
  in the `herg_ki` study, edging the unrescaled +0.0094 in the same cell. But p = 0.215,
  and it sits inside the same +0.008 to +0.012 band every spread arm occupies.
  Tellingly, it **barely changes the allocation at all** — the best representation's quota
  was `[10,10,10,10,10]`, identical to unrescaled.
- `logd`: min-max rescaling gave +0.0271 unanimous at ten representations / budget 1000 —
  good, but the unmodified quota's `power_2` reached +0.0327 and `fixed` +0.0301 in the
  same cell. Mid-pack.
- `flab`: six cells, **no consistent direction**, every difference far below the ~0.024
  standard error.

**Why rescaling can't help much:** the internal representation scores are genuinely
near-tied, so any monotone rescaling lands on the same near-uniform allocation. It is
cheap, principled, and never clearly hurt. It is not a fix.

### 3.4 Top-N (fewer representations) — REJECTED, and it is the worst of the three

- Every hard-cut arm on `herg_ki`'s ten-representation pool is negative or barely
  positive at every budget, while every spread arm is positive at every budget. `top_2`
  degrades *monotonically as budget rises* (−0.0021 → −0.0056 → −0.0134).
- The mechanism is now identified: **`create_hybrid` re-scores each representation
  in-fold using a single `GroupShuffleSplit` (`hybrid.py:3476`, test_size 0.2, seed 42),
  and that one-shot ranking disagrees with the 5-fold cross-validated ranking.** It puts
  `mhg_gnn` above ECFP4 on `herg_ki`; on `flab` it prefers plain SaProt over SaProt-CDR.
  Any hard-cut arm then drops the genuinely best representation on the floor —
  reproducibly, in all five folds. Prescriptions of the form "prune the pool to four or
  fewer representations" would hard-code that mistake.
- On `logd`, concentration helps only at budget 100 (the starved regime) and costs you
  the win at budgets 500 and 1000.
- Direction of the evidence is the opposite of the proposal: **more representations
  helped**. On `herg_ki`, ten representations beat five for the spread family at every
  budget.

`flab` is the one place concentration looks good (`top_3` wins 5 of 5 folds in three
cells), but the entire between-category spread there is about two standard errors wide,
so it cannot carry the decision.

### 3.5 What the evidence actually points at instead

Two things, neither of which was on the menu:

1. **Total budget relative to the best representation's saturation point.** This is the
   only variable with a large, statistically significant, mechanistically explained
   effect. The implied rule is not a new allocator, it is a constraint:
   `budget >= n_representations × ~150`. That is roughly 1500 for a ten-representation
   pool. This constraint is invisible to every knob currently in the grid, and it is
   exactly what `find_best_hybrid.py:4816`'s budget-cap-at-`n_samples` silently violates
   on every small dataset.
2. **The internal evaluator's representation ranking.** One `GroupShuffleSplit` is not
   enough to rank near-tied representations, and every hard-cut arm inherits the error
   wholesale. Fixing the ranker would matter more than adding allocators — though note it
   would improve the hard-cut arms, which still would not beat the spread arms.

---

## 4. The `buchwald_hartwig` proof of concept

**Not reproduced. The run did not finish, and I am not going to dress that up.**

The proof of concept to beat is: best single representation Spearman 0.833 / RMSE 15.08
versus best hybrid Spearman 0.898 / RMSE 11.75, a 22% reduction in error, via "top-5
complementarity".

What is actually verified, from observed output only:

- The dataset loaded: 3955 rows, 22 groups. **Grouping is by ADDITIVE, not by scaffold** —
  this is the repository's own splitter on the repository's own groups, matching the
  earlier stages, but it differs from the brief's instruction that reaction data be
  scaffold-grouped. Flagging it because it would carry through every number.
- The metric-direction gate passed in-process.
- **The target, CGR-Morgan × random forest, scored Spearman 0.8336 — matching the proof
  of concept's quoted 0.833 to three decimals — but RMSE 14.366, not 15.08.** Same rank
  correlation, different error value. That means the proof of concept's RMSE figures were
  produced under a different scoring setup (different folds, or a different error
  convention), **and the 11.75 target may not be on the same scale as our 14.366.** That
  discrepancy should be resolved before 11.75 is used as a goal at all.

Everything else — allocation instrumentation, signal survival, truncation depth, the
global-ranking counterfactual — **did not run**. There is no verdict for this dataset.

On the mechanism question: the proof of concept names "top-5 complementarity", and the
greedy/complementarity allocator family (`greedy`, `greedy_feature`, `dominant_greedy`,
`two_stage`, `stability_selection`) is **already tested and cut** by the earlier `f01`,
`f12`, and `f03` experiments. It was therefore neither run nor proposed here. Whether the
win is reachable cleanly through a top-N knob is **unanswered**.

For context only, and explicitly *not* as this phase's evidence: the earlier hybrid stage
on these same folds found the best of 57 cells at RMSE 13.9415 (delta +0.4249, standard
deviation 0.714, p = 0.254 — not significant), nowhere near 11.75. It also found that all
allocation settings collapse into 2–3 groups at prediction correlation above 0.99, and no
pair falls below 0.97. That is a strong prior that the quota is not the binding constraint
here either — but it is a prior, and the test that would settle it never ran.

**To resume:** rerun
`/private/tmp/claude-501/-Users-apunt-repos-KIRBy/45a3af00-2ccf-4e35-a9c6-538fb157f521/scratchpad/stage43_bh.py`
with `KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache` and substantially more wall time. It
writes three intermediate checkpoints so a later pass can tell how far it got.

---

## 5. `fixed` — the per-representation budget, versus the divided global budget

`fixed` gives every representation the same number of features. It is the original
`c4afb0e` contract (`create_hybrid(n_per_rep=100)`), it was the best mean performer in the
`f01` experiment, and **it is currently unreachable through `find_best_hybrid` — it can
only be invoked by calling `create_hybrid` directly, which is what this study did.**

What it did:

| Dataset | Best cell | `fixed`'s rank | Gap to the best |
|---|---|---|---|
| `logd`, ten reps, budget 1000 | `power_2` +0.0327 | **2nd of 10** (+0.0301) | 0.0026 |
| `logd`, five reps, budget 1000 | +0.0261 | **2nd** (+0.0247) | 0.0014 |
| `herg_ki`, ten reps, budget 100 | `performance_weighted` +0.0115 | **2nd** (+0.0104) | 0.0011 |
| `flab` | spread of ±0.024 standard error | mid-pack, indistinguishable | — |

**Nine score-proportional weighting schemes buy essentially nothing over giving every
representation the same number of features.** On `logd` at ten representations, `fixed`
achieves second place while giving the best representation only 6.2% of its native width.
On `herg_ki` the weighted schemes literally converge onto it — `performance_weighted`
allocated `[10,10,9,10,10]` where `fixed` allocated `[10,10,10,10,10]`.

This is a **simplification argument, not a performance argument.** `fixed` is not better;
it is indistinguishably good and vastly simpler. If the goal is to subtract, `fixed` plus
one weighted scheme covers the whole observed behaviour space.

One known limitation, found by the harness: `fixed` cannot always reach the target total
width. If any representation in the pool is narrower than `budget / n_reps`, the hybrid
comes up short (on `flab` with a 40-column representation and budget 100, you get 90
columns, not 100). `build_hybrid` returns `total_width` and `realised_allocation` so this
can be detected rather than assumed.

---

## 6. What to cut — the subtract list

The governing rule is subtract, don't add. Here is the explicit list.

**Cut from the allocation grid (7 of 10 settings):**

1. **`power_2`** — a monotone rescaling of `performance_weighted`. Lands inside ±0.005 of
   it on every dataset. (Note: it produced `logd`'s single best cell, +0.0327 versus
   `performance_weighted`'s +0.0261 — but that gap is well inside fold noise and does not
   reproduce on the other two datasets.)
2. **`rank_linear`** — same family, same band.
3. **`rank_exp_d0.5`** — same family, same band. Produced `flab`'s best cell, again inside
   noise.
4. **`top_2`** — actively harmful. Dropped the best representation entirely in 5 of 5
   folds (`herg_ki` ten-rep pool), 3 of 5 (`flab` ten-rep), 2 of 5 (`flab` five-rep). Its
   error worsens as budget rises.
5. **`top_5`** — degenerate whenever the pool has five representations; identical to
   `performance_weighted` by the code's own rule at `find_best_hybrid.py:3521`. Keep
   `top_3` as the concentration representative instead.
6. **`entropy_1.0`** — see below.
7. **`entropy_2.0`** — see below. It collapses to a single representation on `herg_ki`'s
   five-representation pool. It is not a hybrid.

**Cut the entropy family specifically, and here is why it matters:** its redundancy
penalty is the correlation of **sample-wise L2 norms of raw features**
(`hybrid.py` roughly lines 512–548) — that is raw geometry, not the
prediction-correlation definition of redundancy this whole study uses. It measures the
wrong thing. Its actual observed behaviour is "evict the best representation entirely",
which happened to be harmless on `flab` and destructive on `herg_ki`. **Its occasional
good scores must not be credited as evidence that redundancy-aware allocation works.**

**Cut from the proposed fixes:**

8. **Global feature ranking (gain-based)** — loses to the quota in 16 of 18 measured cells
   across three datasets, and is actively negative on `flab`'s ten-representation pool
   (−0.039). Reject as instantiated; do not build it.
9. **"Prune the pool to four or fewer representations"** — the evidence points the
   opposite way. More representations helped on `herg_ki` and `logd`.

**Keep, provisionally:** score rescaling, but only if it is free. It never hurt and was
positive in 6 of 6 cells on `herg_ki`. It is not a lever and should not be presented as
one.

**Do not cut, and consider promoting:** the total-budget knob. It is the only variable
with a large, significant, mechanistically explained effect.

---

## 7. Caveats and limits

Consolidated from every stage. Read all of these before quoting any number.

### Metric direction gate: PASSED

This is load-bearing — one wrong `lower_is_better` flag inverts every conclusion.

- The gate passed on both directions and both validation datasets, at the raw-scoring
  layer **and** end to end through `hybrid.create_hybrid`.
- It uses label-derived signal matrices (z-scored labels plus Gaussian noise at three
  increasing levels), which are monotone by construction on any dataset.
- It asserts the two datasets actually span both directions, so it cannot pass by
  accidentally testing one direction twice.
- Each of the three finished datasets re-verified direction independently, in-process,
  before doing any work: `herg_ki` (RMSE, lower better) four separate ways; `logd`
  (RMSE, lower better) via label-derived matrices through the identical scoring path;
  `flab_thermo_tresanco` (Spearman, higher better) with a raw-units cross-check
  confirming the reported wins have genuinely higher Spearman.
- **The first version of this gate failed and was rebuilt rather than loosened.** Version
  one corrupted a real representation to build the better/worse ladder; it passed on
  `herg_ki` but failed on `flab` because that representation's own signal was too weak for
  the ladder to be monotone. That was a defect in the test construction, not in the metric
  direction. Reporting it because a less careful pass would have "fixed" it by weakening
  the assertion.

### What did not run

- **`buchwald_hartwig`: zero of five measurements completed.** Wall-clock misjudgement,
  not a code failure. See section 4. Do not read any field for that dataset as a partial
  result; read it as not-run.
- **A mid-run progress claim on `buchwald_hartwig` was overstated and is corrected here.**
  Two blocking gates were reported as passed when they had not printed to the log — the
  claim was read from the code, not from the output. On that dataset the
  per-representation quota mechanism is **asserted by code reading, not verified by
  execution.** (On `herg_ki`, `logd`, and `flab` the gate genuinely passed at runtime.)
- No repeated cross-validation anywhere. Single seed (42), single fold partition.
- Score rescaling was tested only through `performance_weighted`, not crossed with the
  other nine categories.
- No counterfactual was built that would measure the quota's *direct* cost inside the
  hybrid (raising the best representation's quota while holding the others fixed — that
  changes total width and breaks the width match).
- Neither arm is tuned. Target and hybrids both use default models.

### Statistical weakness — the most important caveat

- **On `herg_ki`, zero of 57 quota configurations, zero of 6 global-ranking arms, and zero
  of 12 rescaling arms beat the target at p < 0.05 on a paired five-fold t-test.** The
  smallest p-value at hybrid level anywhere is 0.124. Every ranking of allocation
  categories in this document is a description of noise-level differences at the hybrid
  level and must not be quoted as a result.
- **On `logd`, no significance testing was performed at all** — paired per-fold deltas and
  win/loss counts only. A 5-of-5 sweep at five folds is suggestive, not a p-value.
  Differences under about 0.005 RMSE between categories should be treated as
  indistinguishable; that threshold is a judgement call, not an empirical result.
- **On `flab`, the standard error on any cell's mean delta is about 0.024 and the entire
  between-category spread is −0.006 to +0.045** — roughly two standard errors wide in
  total. No claim of the form "category X beats category Y" is supported on this dataset,
  including the preference for `top_3`.
- Five folds cannot resolve a 0.01 effect. Per-fold delta standard deviations are
  0.010–0.031 against mean hybrid deltas of about 0.01. The correct phrase for every
  hybrid comparison is **"not detected"**, never "absent".
- The one place significance genuinely exists is the representation-level slice test
  (p as low as 0.0002 on `herg_ki`), and only because the effect there is ten times
  larger.

### Measurement construction caveats

- **The slice test measures the features, not the hybrid's use of them.** Scoring the best
  representation's slice *alone* tells you what that slice can do by itself. It does not
  tell you what those columns contribute inside the hybrid, where other representations
  are present and the model can substitute. The anti-correlation between slice delta and
  hybrid delta *is* that substitution. The slice test is a lower bound on the quota's
  cost, not a direct measure.
- **Truncation-depth recall is true by construction, not a measurement, on `herg_ki` and
  `logd`.** `create_hybrid` takes an `argsort` prefix of the same per-representation
  importance vector used to define the top-K, so recall at K equals `min(q,K)/K` exactly,
  always. There is no "the quota picked the wrong features" failure mode available. The
  informative quantity is the realised quota against the saturation point. (On `flab`
  recall was computed empirically as a set intersection, which is stronger.)
- **Folds where the best representation got zero features are excluded from slice means
  and reported separately.** On `flab` that is 16 of 57 cells. Do not compare a one-fold
  slice delta against a five-fold one. Averaging over surviving folds would have flattered
  those cells.
- The slice-versus-hybrid join on `logd` uses nearest-quota matching, because the realised
  quota varies slightly across folds. One case is genuinely bimodal (`top_2` at budget
  1000: one fold at 800, four near 503, mean 565 — a value no fold used). This affects
  precision, not direction; the curve is flat in that region.
- Global ranking was tested with **one** importance definition (LightGBM gain over the
  concatenation), chosen to match the quota path's own method for fairness. Gain is known
  to favour dense, high-cardinality features and the origin counts confirm it did here.
  A scale-invariant global criterion was not tested. **Read the rejection as "gain-based
  global ranking loses", not "global ranking cannot work".**
- The global-ranking arm on `flab` has an extra confound: fitting one LightGBM to a
  20352-column matrix at 538 training rows may be unstable for reasons unrelated to global
  ranking as an idea.
- On `flab`, the "skill over a null model" rescaling was deliberately **not** run: the
  internal evaluator returns R² for both metric directions and null R² is zero, so that
  rescaling is the identity there. Stated rather than silently omitted.
- Redundancy is always measured by **prediction correlation** (Spearman of out-of-fold
  prediction vectors), never raw feature correlation. `flab`'s 0.981 correlation between
  SaProt and SaProt-CDR, and the 0.86–0.92 hybrid-versus-target figures, are inherited
  from the previous stage, not recomputed here.
- On `herg_ki`, hybrid deltas are **reused from the earlier hybrid stage, not recomputed**
  in this one. This stage rebuilt every hybrid to instrument allocation across all five
  folds but did not re-score them. The target was independently re-scored and reproduced
  to seven decimal places, and the rebuilt allocations match, so the join is sound — but
  two numbers came from two processes.

### Pool and dataset caveats

- **`herg_ki`'s pool membership rests on a provisional table.** Five representations sit
  within 0.015 RMSE of ECFP4 while ECFP4's own fold-to-fold spread is 0.078, and the
  baseline stage's LightGBM sweep was only 6 of 24 complete. "ECFP4 is the best
  representation" is the *premise* of the whole signal-survival test on that dataset. If
  the true best is `mhg_gnn` — which the internal evaluator believes — then the hard-cut
  arms are not making the error attributed to them. Five folds cannot resolve this.
- **`logd`'s "best representation" is a near coin flip**: Mordred 0.50519 versus `mhg_gnn`
  0.50672, a 0.0015 gap inside fold variance. No conclusion flips, but the exact numbers
  would move.
- The signal-survival test isolates the **best** representation only. It does not test
  whether the quota truncated some *other* representation's uniquely complementary
  features. A representation whose contribution is narrow but essential could still be
  being cut, and that was not measured.
- `buchwald_hartwig` is split by ADDITIVE (22 groups), not scaffold. Twenty-two groups
  over five folds is coarse, and the target's per-fold RMSE spans 9.91 to 17.12.
- **Three datasets, three domains, no transfer claimed.** The `herg_ki` and `flab` central
  negative ("the quota truncates the signal and it doesn't matter") is specifically a
  consequence of those pools' redundancy. `logd` behaves differently. Nothing here
  transfers to `buchwald_hartwig` or to any other antibody set.
- `top_5` was not run on any five-representation pool (degenerate by the code's own rule).
  Reported, not silently dropped.

### Environment and compliance

- **macOS OpenMP guard is mandatory.** On this machine, importing `torch` and then calling
  `LGBMRegressor.fit()` segfaults (exit 139) inside `libomp`, with no traceback and no
  output. The `kirby` package imports `torch` transitively. Fixed inside the harness by
  making `import lightgbm` the first statement in `alloc_harness.py`. Any future script
  must import `alloc_harness` before `torch` or `kirby`, or redo the guard.
- **The working tree changed during these runs, and not by any of these stages.** Between
  the session-start snapshot and the end of the harness run,
  `src/kirby/representations/mutation.py` gained about 107 lines and a pre-existing
  modification to `src/kirby/find_best_hybrid.py` disappeared. This looks like a
  concurrent session on the same checkout. It means code facts verified early could have
  shifted later. `hybrid.py:53` (`NAMED_ALLOCATIONS`), `hybrid.py` around line 3627 (the
  `fixed` branch), and `find_best_hybrid.py:3521` should be re-verified before being
  depended on. The datasets that finished mitigated this by verifying the quota mechanism
  through **runtime assertion** rather than code reading alone.
- Timings on this machine are noisy — three back-to-back identical passes varied up to
  threefold. Counter-intuitively the smaller antibody dataset is 5–10× slower per pass
  than the larger molecular one, because its representations are dense floating-point
  embeddings rather than sparse binary fingerprints. Do not budget by row count.

**Constraint compliance, which does hold across every stage:**
zero files under `src/` modified; `find_best_hybrid()` never called, so `result_db` was
never read or written and the budget-cap-at-`n_samples` never applied;
`create_hybrid` called directly throughout; no ridge regression anywhere; no
greedy/complementarity allocator run or proposed; no random splitting — the repository's
own `cv_split` (GroupKFold on scaffolds for molecular, the loader's own groups for
antibody) with assertions that no group spans train and validation and that the
validation folds partition every row; `KIRBY_CACHE_DIR` treated as read-only; no
averaging or medians across datasets.

---

## 8. Open questions — for you to decide

These are decisions, not findings. Options and plain-English tradeoffs only.

**Q1. Do we rerun `buchwald_hartwig`, and at what cost?**
It is the only reaction dataset, the only one with a claimed large win, and the only pool
where the representations might be genuinely complementary rather than near-redundant.
Both molecular datasets and the antibody dataset concluded that the quota is not the
problem *because* their representations are redundant — that conclusion is exactly what
`buchwald_hartwig` was supposed to challenge.
*Option A:* rerun it in full (several hours of local compute) before deciding anything.
*Option B:* accept the three-dataset verdict and treat `buchwald_hartwig` as future work.
*Option C:* run only the signal-survival test on it — the single decisive measurement —
and skip the global-ranking and rescaling arms.

**Q2. The proof of concept's RMSE scale doesn't match ours. Resolve it or drop it?**
Our target reproduces the proof of concept's Spearman of 0.833 to three decimals but
gives RMSE 14.37, not 15.08. Same object, different error number, which means the 11.75
goal may be on a different scale entirely. *Option A:* find and re-run the original proof
of concept script to establish which scoring setup produced 15.08. *Option B:* abandon the
RMSE numbers and restate the target in Spearman only (0.833 → 0.898), which is the number
that does match.

**Q3. Should `fixed` become reachable through `find_best_hybrid`?**
It is second place on two datasets, within 0.003 of the best, and is currently only
reachable by calling `create_hybrid` directly. Making it reachable is a small change but
it is a change to the search space, which needs your sign-off.
*Option A:* expose it as an allocation method and cut several weighted schemes, so the
grid gets smaller not larger. *Option B:* leave the grid alone; `fixed` is not actually
better, only simpler. *Option C:* replace the whole weighted family with `fixed` and treat
uniform allocation as the default, keeping one weighted scheme as a comparison arm.

**Q4. Should the budget become saturation-aware?**
The one large, significant, explained effect in this phase is that each representation
needs roughly 100–200 features and the quota often gives it 10. The implied rule is
`budget >= n_representations × 150`. That is not a new allocator; it is a constraint on an
existing knob. *Option A:* add it as a hard floor. *Option B:* add it as a warning that
logs when the grid is searching a starved region. *Option C:* leave it, and simply stop
searching budgets below 500. Related and separate: `find_best_hybrid.py:4816` caps the
hybrid's budget at the row count, which silently violates this on every small dataset —
that is the previously identified defect and it now has a mechanism behind it.

**Q5. Do we fix the internal representation ranker?**
`create_hybrid` ranks representations on one `GroupShuffleSplit` and gets it wrong,
reproducibly, on both `herg_ki` and `flab`. Every hard-cut arm inherits that error.
*Option A:* replace with a small internal k-fold — more reliable, roughly k times the
internal cost. *Option B:* leave it and simply cut all the hard-cut arms, since they lose
to the spread arms anyway even when correctly ranked. *Option C:* leave it entirely; the
spread arms are insensitive to ranking errors because they keep every representation.
Option B is cheapest and is consistent with the subtract-don't-add rule.

**Q6. Is a scale-invariant global ranking worth one more test?**
Gain-based global ranking was rejected on measurement, and the origin counts explain why
(dense representations monopolise gain). A per-representation-normalised or
rank-normalised global importance might not have that failure mode. *Option A:* one more
counterfactual arm on `logd`, where the effects are largest and cleanest, roughly an hour.
*Option B:* treat global ranking as closed — the per-representation quota's diversity
guarantee is doing real work and we have three datasets saying so.

**Q7. How much weight do we put on `flab_thermo_tresanco` at all?**
Its fold scores span 0.46 to 0.78 around a mean of 0.60. Every effect measured is under
two standard errors. *Option A:* run repeated cross-validation (five repeats) to get a
usable error bar — expensive on this dataset, which is slow per pass. *Option B:* report
it as direction-only (54 of 57 configurations positive) and stop treating its cell
rankings as informative. *Option C:* replace it with a larger antibody dataset for the
allocation question.
