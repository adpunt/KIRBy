# Similarity-Optimized Hybrid — Forensic Analysis

**Date:** 2026-07-10
**Experiment:** `tests/experiments/similarity_optimized_hybrid.py`
**Results analysed:** `tests/results/sim_optimized_hybrid/{adme_hlm,adme_mdr1,herg_ki,qm9}.json` (4 datasets × 3 seeds, all COMPLETE; all four are regression tasks)
**Method:** 129-agent adversarial audit — every claim independently verified against code and result JSON by three hostile reviewers (code-reality / statistical-validity / alternative-explanation lenses). Claims refuted by ≥2 of 3 were dropped and are listed in §7. All numbers below were then re-verified by hand against the JSON.

**Citation caveat:** `HEAD` moved during this session (`2587df2` → `ed87967`; two `c21` commits added ~113 lines to `find_best_hybrid.py`, 5478 → 5591 lines). All file:line citations here are against the **current working tree at `ed87967`**. Earlier line numbers in the session transcript were correct at the time and are now stale.

---

## 1. Bottom line

**Payoff 1 is real.** Searching for a hybrid that maximises SAC — the similarity–activity correlation, meaning the Spearman rank correlation between how far apart two molecules sit in feature space and how different their measured properties are — genuinely produces a better similarity geometry than the best single base representation. The similarity hybrid beats the raw representation on held-out SAC in 4 of 4 datasets, and has a lower neighbour-lie rate (the fraction of a molecule's nearest neighbours whose property is very different) in 4 of 4. This survived every adversarial attempt to explain it away as a dimensionality or budget artifact.

**Payoff 2 is not a clean test.** The reported "no benefit on noise mitigation" is contaminated by one hard bug, one degenerate metric, and one instrumentation flaw. Separately, and more deeply, the design cannot express the geometry that local noise detection actually needs. So the honest statement is not "the similarity hybrid doesn't help noise mitigation" — it is "Payoff 2 as instrumented could not have detected it, and the objective being optimised is the wrong shape for the job."

**The single most diagnostic fact:** the search selects `pred_budget = 10` — the smallest value in the grid `[10, 25, 50, 100, 200]` — in **12 of 12** runs, while `sim_budget` varies across `{10, 25, 200}`. That is the fingerprint of a sign-inversion bug in how the predictive comparator is chosen.

---

## 2. Correction: what I got wrong earlier

I previously wrote: *"the objective can only choose how many features, never which ones."* **That is false as stated, and the pushback was correct.** The accurate statement has two parts:

1. **Across representations, the objective does change which specific features are selected.** The objective enters at `src/kirby/hybrid.py:1182` (`if objective == OBJECTIVE_SIMILARITY_SAC:`) inside `_evaluate_allocation`, where it scores trial *allocation vectors* — how many features each representation contributes. Different objectives select different allocation vectors. Because different representations' feature columns are disjoint, changing the per-representation counts changes exactly which columns enter the concatenated hybrid. At the level of the final feature set, the objective absolutely changes which features appear. The `picks` data confirms this empirically: sim and pred choose different allocation methods and different budgets on every dataset and every seed.

2. **Within any single representation, the objective is powerless.** `compute_feature_importance` (`src/kirby/hybrid.py:2382`) takes no `objective` parameter. The within-representation choice is a deterministic top-N prefix of that ranking (`np.argsort(importances)[-n_select:][::-1]`, `src/kirby/hybrid.py:1106`). Here the ranking is `lightgbm_gain` (`tests/experiments/similarity_optimized_hybrid.py:203`) — a label-supervised *predictive* importance, identical whether the objective is predictive or similarity.

The real limitation is **expressivity**, and it is a design limit rather than a bug. The reachable feature sets are confined to per-representation *prefixes* of a predictive ranking. The similarity objective can slide each representation's cut-point and choose which representations contribute, but it can never promote, skip, or isolate a feature *within* a representation. Concretely: a fingerprint bit that encodes a scaffold and carries strong pairwise-distance signal but low LightGBM gain — say it ranks 300th — can only be included by first dragging in the 299 higher-gain predictive features ahead of it. **The similarity objective can never select a structurally informative feature without taking the predictively-chosen ones first.** So the hybrid is "similarity-optimized" at the coarse allocation level only; its individual features were still chosen to predict.

---

## 3. Bugs — these invalidate reported numbers

### 3a. The predictive comparator selects the *worst* hybrid, not the best (most severe)

`_search` picks the winning hybrid across budgets with a direction-blind maximum:

```python
# tests/experiments/similarity_optimized_hybrid.py:214-216
by_budget = res.best_by_model_budget.get('rf', {})
best = max(by_budget.values(), key=lambda r: r['score_mean'])
```

`score_mean` is the **raw, un-negated** metric value (`src/kirby/find_best_hybrid.py:2847`, `'score_mean': float(np.mean(fold_scores))`). The engine applies direction elsewhere, via `_score_to_comparable(raw, lower_is_better)`, which negates for lower-is-better metrics (`src/kirby/find_best_hybrid.py:1164-1166`). The per-budget picks *inside* `best_by_model_budget` already honour it (`_key` at `4160-4161`, used at `4196`). The experiment's outer selection at line 215 does not.

Consequences by arm:

- **Predictive arm.** The predictive path forces `metric='rmse'` (`similarity_optimized_hybrid.py:212`), and RMSE (root-mean-square error) is lower-is-better (`find_best_hybrid.py:1010`). So `max(score_mean)` picks the budget with the *highest* error — the worst hybrid across budgets. Because smaller feature budgets generally predict worse, this lands on budget 10 in all 12 runs. The "predictive-optimized comparator" is in fact predictive-**pessimized**.
- **Similarity arm.** The objective forces `metric='sac'`, which is higher-is-better (`find_best_hybrid.py:1020`). Here `max(score_mean)` is **correct**. `sim_budget` therefore varies genuinely.
- **Raw-representation arm.** Selected independently of this line; **not corrupted**.

This bug hits regression only. `f1_weighted` is higher-is-better (`find_best_hybrid.py:1017`), so the classification datasets would be unaffected — but all four completed datasets are regression, so all four are affected.

**What is and is not invalidated.** Every sim-versus-pred number in both payoffs is void: the predictive comparator is systematically the worst hybrid per tier, so the sim-beats-pred gaps are inflated by the bug rather than earned. The **sim-versus-rep contrasts are untouched**, and those are what the Payoff 1 conclusion actually rests on.

One lucky consequence: on `adme_hlm`, `sim_budget` and `pred_budget` are *both* 10, so that dataset is accidentally dimension-matched — and there, sim SAC = **0.1915** versus pred SAC = **0.0940**. Same feature count, roughly double the SAC. That isolates the objective as the cause, even under the bug.

**Fix (one line).** `SearchResult` already carries the direction (`find_best_hybrid.py:928`):

```python
best = (min if res.lower_is_better else max)(by_budget.values(), key=lambda r: r['score_mean'])
```

This leaves the SAC side unchanged and corrects the predictive side.

### 3b. "Recovery" is a mean of ratios with a near-zero, sign-flipping denominator

Recovery is scored as `(score_after − noisy) / (clean − noisy)`, guarded only against literal division by zero (`similarity_optimized_hybrid.py:485`, `492`, `512`). The underlying metric is `r2_score` (`similarity_optimized_hybrid.py:179`), whose range is (−∞, 1]. So the denominator can be tiny on folds where injected noise barely hurt, and **negative** on folds where noise accidentally helped — silently flipping the sign of the result.

The `1e-9` guard stops only exact zeros. A denominator of 0.01 yields a "recovery" of ~20. Verified directly from the JSON:

| Cell | mean | std | n |
|---|---|---|---|
| `adme_mdr1` hetero `_global_smoothing.sim` | −25.311 | 78.665 | 9 |
| `adme_hlm` gaussian `_global_smoothing.rep` | −17.841 | 65.20 | 9 |
| `adme_hlm` hetero `activity_cliffs.rep.recovery_remove` | +51.357 | 115.25 | 9 |

Across all **504** recovery-mean entries: **29.2%** have `|mean| > 1` (impossible for a genuine recovery *fraction*), and **70.0%** have `std > |mean|` — the fold-to-fold spread exceeds the reported signal. With only n = 9 replicates (3 folds × 3 seeds, `N_FOLDS=3` at line 101) and a heavy-tailed ratio whose denominator concentrates near zero, the sample mean does not converge; it is dominated by the single smallest-denominator fold.

**These numbers are not recovery fractions and carry no recovery signal.** Do not read them, do not present them.

**Fix.** Report the paired raw difference `r2_after − r2_noisy` per fold — bounded, additive, well-behaved — and/or a rank-based test (Wilcoxon / Friedman) across folds. If a normalised fraction is still wanted, compute it once on the aggregate, after gating folds whose denominator is below a real floor. Never as a mean of per-fold ratios.

### 3c. Label leakage in the noise evaluation — real, but it does *not* favour the hybrids

In `_noise_eval`, both hybrid spaces are built from the **clean** labels of the exact rows that are then corrupted and handed to the detectors. Lines `465-466` set `y_tr` and derive `y_noisy`/`mask`; lines `470-471` and `473-474` build the sim/pred spaces from clean `y_tr`; line `509` runs `identify_noise(X_tr, y_noisy)` on those same rows. Per-fold importance is fit on clean `y_tr` (line `253`). The raw-representation space is capped label-free by top variance (lines `477-481`), so it gets no such help.

**Honesty note:** I initially expected this to inflate the hybrids relative to the raw representation. That claim was **refuted unanimously**. In the actual results the label-free raw representation is competitive with, and often beats, both hybrids on detection F1. So this is a genuine methodological flaw to fix for cleanliness, but it is **not** the explanation for any sim-over-rep gap, and it does not touch Payoff 1 (which uses the full uncapped representation and legitimately uses clean labels). One reviewer argued it is a within-fold relative comparison and not target leakage at all; the majority still recommends the fix as the honest deployment setting.

**Minimal fix.** Pass `y_noisy` instead of `y_tr` into the two `_build_hybrid_fold` calls at lines 470 and 473. Leave `_heldout_similarity` alone.

---

## 4. Design limits — working as intended, structurally unable to show the effect

### 4a. Within-representation features are still chosen to predict

See §2. This is the expressivity limit, and it is the reason a similarity objective cannot reach a similarity-shaped feature set.

### 4b. SAC is global and cliff-blind; noise detection needs local fidelity

SAC is `spearmanr(pairwise standardized-Euclidean distance, pairwise |Δy|)` over uniformly sampled pairs (`src/kirby/utils/noise_mitigation_methods.py:166-172`). Two structural facts follow.

**It is maximised by collapsing onto the label.** Setting the embedding equal to `y` gives distance = |Δy| exactly, so rho = 1 — the global maximum, in both the regression and the classification branch. At that maximum, small distance implies small |Δy| by construction, so the set of "close in space, far in label" pairs is **empty**.

**Activity cliffs are exactly those pairs.** The cliff detector flags `(distances < similarity_threshold) & (delta_y > activity_threshold)` (`noise_mitigation_methods.py:508`, thresholds at the 10th/90th percentiles, lines `490`/`501`). So maximising SAC is **provably antagonistic** to activity-cliff detection at the limit.

The falsifiable prediction holds. Verified from the JSON — the higher-SAC sim space has **lower cliff-detection recall than the raw representation in 8 of 8** dataset × noise cells:

| dataset | noise | recall sim | recall rep |
|---|---|---|---|
| adme_hlm | gaussian | 0.8573 | 0.9004 |
| adme_hlm | hetero | 0.7791 | 0.8467 |
| adme_mdr1 | gaussian | 0.6773 | 0.9156 |
| adme_mdr1 | hetero | **0.3613** | 0.7302 |
| herg_ki | gaussian | 0.8520 | 0.8643 |
| herg_ki | hetero | 0.8662 | 0.8794 |
| qm9 | gaussian | 0.8396 | 0.8947 |
| qm9 | hetero | 0.8240 | 0.9022 |

Cliff-detector **precision** is pinned near the injected base rate of 0.25 (`NOISE_FRAC=0.25`, line 106; observed 0.2587–0.2736) because its percentile thresholds flag a roughly fixed fraction. So F1 masks the effect entirely — **lead with recall, not F1**, for this detector.

*Caveats.* The 8/8 recall drop is directional (sign test p ≈ 0.004) but individually significant in only 2 of 8 cells, both on `adme_mdr1`. The recall loss does **not** track the size of the SAC gap: `adme_mdr1` has the *smallest* sim−rep SAC gap (+0.085) yet the *largest* recall collapse. The mechanism is supported at the limit and directionally, but is not a proven smooth monotone relationship.

### 4c. Half the detectors are at chance in every space, and F1 is confounded by flag fraction

Detectors run on default parameters (`get_mitigation_method(method)`, no kwargs, line 508), so their flag fractions are fixed by code defaults regardless of geometry.

Pooled detection precision in the sim space, across all 8 dataset × noise cells (base rate = 0.25):

| class | method | precision | verdict |
|---|---|---|---|
| similarity | `neighborhood_consensus` | 0.8500 | real signal |
| similarity | `distance_weighted` | 0.8439 | real signal |
| similarity | `knn_k5` | 0.2591 | **at chance** |
| similarity | `knn_k10` | 0.2467 | **at chance** |
| similarity | `knn_k20` | 0.2459 | **at chance** |
| similarity | `activity_cliffs` | 0.2655 | **at chance** |
| geometry | `lof` | 0.2627 | **at chance** |
| geometry | `mahalanobis` | 0.2562 | **at chance** |
| control | `zscore` | 0.9841 | real signal (label-only) |
| control | `prediction_error` | 0.7202 | real signal (model residual) |

Six of the ten detectors sit at the 0.25 base rate. **For those six, no geometry change in any space can help, because they carry no signal anywhere.**

**F1 is confounded by how many points you flag.** A random detector's F1 is 0.1429 at a 10% flag rate, 0.25 at 25%, and 0.40 if it flags everything. The cliff detector flags roughly 76% of samples, so its headline ~0.40 F1 is the flag-everything ceiling, not detection skill. The scale-invariant above-chance test is **precision against 0.25**, not F1.

Note that the two detectors with real signal in the *control* class (`zscore`, `prediction_error`) derive it from label residuals and label-only z-scores — **not** from representation-space geometry. So Payoff 2's detection test is largely being run on detectors that either cannot respond to geometry, or are scored with a flag-fraction-confounded metric.

*Do not assert* that `prediction_error` beats every similarity detector on precision — it leads on F1 and recall only. Two genuine similarity-class detectors are more precise at low recall.

---

## 5. The central puzzle: why does a real Payoff 1 win produce no Payoff 2 gain?

The Payoff 1 win is confirmed. Sim beats the raw representation on SAC by +0.089 / +0.085 / +0.148 / +0.267 on `adme_hlm` / `adme_mdr1` / `herg_ki` / `qm9`, and the neighbour-lie rate is lower for sim in 4 of 4. Yet noise-detection AUROC shows sim ≈ pred ≈ rep with differences under 0.03. Ranked explanations:

**1. [Confirmed — primary] SAC quality is the wrong axis for local noise detection.** SAC is a *global* rank correlation (§4b), and the within-representation features are still chosen *predictively* (§4a). So a better-SAC geometry improves the coarse, global similarity-property ranking — which is exactly what Payoff 1 measures — while, at the limit, *emptying* the local structure-close/label-far neighbourhoods that cliff and consensus detectors depend on. The geometry that wins Payoff 1 is close to the geometry that is worst for local detection. This is the honest headline.

**2. [Confirmed] Much of Payoff 2 is not a null — it is unmeasurable or mis-measured.** The recovery sub-payoff is degenerate noise (§3b); the predictive comparator arm is sign-inverted (§3a); detection F1 is flag-fraction-confounded, and six of ten detectors are at chance in every space (§4c). "Payoff 2 shows no effect" partly means "Payoff 2 as instrumented cannot show an effect."

**3. [Confirmed reversal — geometry *does* move one thing]** My stronger claim that "Payoff 2 has zero headroom" **did not survive**. `neighborhood_consensus` detection precision is higher in the sim space than in the raw representation in **8 of 8** cells — large on the two ADME datasets (+0.0594 to +0.1139) and small elsewhere (+0.0039 to +0.0108). `distance_weighted` is positive in 7 of 8. This is geometry-specific: the improvement tracks the space, and the geometry-agnostic controls do not move the same way. But it is **precision-only** (recall trades down, so F1 stays flat), which is precisely why the F1-led reporting hides it, and why the cross-dataset average dilutes it to zero.

So the accurate statement is: **Payoff 2 is null in aggregate, with a small, real, precision-only geometry effect on the consensus-style similarity detectors that the current metric and averaging conceal.**

**4. [Speculative] Leakage could be masking a gap in either direction.** Because the sim/pred spaces are built on clean labels of the corrupted rows (§3c), the honest `y_noisy` setting might change the picture. Empirically the leak does not currently favour the hybrids, so this is a "re-run to be sure," not a load-bearing explanation.

---

## 6. What is *not* wrong — this defends Payoff 1

Several confounds I proposed were investigated and **refuted**. Recording them matters, because each would otherwise look like a reason to distrust Payoff 1.

- **Dimensionality is not a confound.** The raw representation is full-dimensional (2048-d `atom_pair` for the two ADME sets; 768-d embeddings for `herg_ki`/`qm9`) while hybrids are ≤200-d. But the predictive hybrid is a *dimension-matched control*: equally narrow (budget 10), equally importance-filtered — and it does **not** beat the raw representation on SAC (`adme_hlm`: pred 0.0940 < rep 0.1027). On `adme_hlm`, sim and pred are *both* 10-dimensional, yet sim = 0.1915 versus pred = 0.0940. Same dimension, ~2× SAC. That isolates the **objective** as the cause. Further, SAC is a rank correlation and is invariant to the monotone distance rescaling that the distance-concentration argument invokes, so concentration cannot mechanically depress it.
- **Budget is not a confound.** Both arms search the identical grid through the same machinery, differing only in objective. Each objective selects its own best budget, so budget is a *mediator* of the objective's effect, not an external nuisance. (This holds *only after* fixing §3a — right now `pred_budget=10` is a bug artifact, not a genuine choice.)
- **The label-free cap on the raw representation does not rig Payoff 2 toward the hybrids** (§3c), and does not touch Payoff 1 at all.

**The one caveat that partially survives: the metric on binary fingerprints.** SAC uses standardized-Euclidean distance uniformly (`noise_mitigation_methods.py:166-167`), but the field-standard metric for binary molecular fingerprints is Tanimoto/Jaccard. Standardization divides each bit by `sqrt(p(1-p))`, up-weighting a rare bit (prevalence 0.01) by roughly 25× versus a common bit (prevalence 0.5) in squared distance. This applies to the raw-representation baseline on `adme_hlm` and `adme_mdr1`, whose best single representation is the binary `atom_pair` fingerprint. It does **not** apply to `herg_ki`/`qm9`, whose best representations are continuous embeddings. The comparison is internally consistent (sim and rep are scored on the same metric) and the gaps are large relative to the n=9 noise, so this is a **sensitivity check to run**, not a demonstrated artifact. A reviewer will ask whether the hybrid beats `atom_pair` scored in native Tanimoto space.

---

## 7. Claims that were tested and REFUTED — do not assert these

- "Payoff 2 has no headroom; geometry can never help." → Refuted; see §5.3.
- "The control detector `prediction_error` beats every similarity method on precision." → Refuted; it leads on F1/recall only.
- "The label-free raw-rep cap makes the sim/pred comparison unfair *in the hybrids' favour*." → Refuted unanimously.
- "Dimensionality confounds Payoff 1." → Refuted; `adme_hlm` is dimension-matched.
- "Payoff 1 is not budget-matched, so the headline is void." → Refuted (budget is a mediator, post-fix).
- "rho = 1 is unreachable because a structure-derived embedding cannot equal the label." → Refuted as a category error: SAC is a rank correlation between *distance* and *|Δy|*, not between the embedding and the label.
- "The observed ~0.53 max SAC on qm9 is a proven ceiling." → Refuted; it is the best value the search *found*.

---

## 8. Remediation plan

### Fix bugs first — these block any re-run

1. **Sign bug (§3a).** Apply the one-line direction fix at `similarity_optimized_hybrid.py:215` using `res.lower_is_better`. Until this is done, **every sim-versus-pred number is void.**
2. **Recovery metric (§3b).** Replace the mean-of-ratios with a per-fold paired raw difference (`r2_after − r2_noisy`) and/or a rank-based test. Stop presenting recovery as a fraction.
3. **Noise-eval leakage (§3c).** Change the two `y_tr` arguments to `y_noisy` at lines 470 and 473.

### Re-run and add the missing controls

4. Re-run the four completed datasets (and, budget permitting, the remaining four in `DEFAULT_DATASETS`, lines 93-94) after fixes 1–3. Report **sim-versus-rep as the primary contrast**; report sim-versus-pred only after the sign fix.
5. **Draw the chance line on every detection panel.** Plot precision against the 0.25 base rate, and the random-F1 curve as a function of flag fraction. Report **precision** as the above-chance test. Lead the cliff detector with **recall**.
6. **Add a within-representation random-feature control.** Replace the LightGBM-gain prefix with a random subset of the same size, to test whether predictive within-rep selection helps or hurts Payoff 2. This directly probes §4a and is cheap.
7. **Tanimoto sensitivity check (§6).** Re-score the `atom_pair` raw-rep baseline SAC in native Tanimoto space for `adme_hlm` and `adme_mdr1`.
8. **Report the consensus-detector effect explicitly** (§5.3) rather than burying it in a cross-dataset F1 average.

### Design changes — these need your signoff, per the no-autonomous-design rule

9. **Make the within-representation ranking objective-aware.** This is the change that would let a similarity objective actually reach a similarity-shaped feature set. Cheapest clean option: **per-feature SAC** — score each feature's own `|Δf|` against `|Δy|` on the same sampled pairs the objective already uses, returning a per-feature array that drops straight into the existing `argsort` prefix selection at `hybrid.py:1106` with no contract change. A supervised **Laplacian score** (He, Cai & Niyogi, NIPS 2005) is a same-cost second option. Greedy-forward-SAC is the most objective-aligned but roughly 100× costlier and needs surgery inside `_create_hybrid_with_allocation`. **mRMR is the wrong fit** — it is O(d²) and the reps run to 2048-d; reject it.

10. **Reconsider the objective itself.** No single objective is both collapse-proof and a direct measure of read-across utility: any objective that uses `y` to shape distances is, in its unconstrained limit, maximised by projecting onto the label axis. The research recommendation is a **paired objective**: a local, scaffold-grouped k-nearest-neighbour read-across error as the task loss, plus a trustworthiness/continuity term (Venna & Kaski) against an explicit reference chemical space to forbid the label-collapse solution. Report neighbour violation rate and local intrinsic dimension as diagnostics rather than optimising them. **Avoid SAC as a selection criterion** — it is global and cliff-blind. Adding a second geometry objective to `find_best_hybrid` is localised (a new constant in `_VALID_OBJECTIVES` at `find_best_hybrid.py:1029`, a **unique metric name** so the objective stays cache-isolated — the on-disk cache keys on metric name, not objective — a scorer, and generalising the ~4 SAC-specific branches).

11. **Do NOT build the representation-weighted distance hybrid without an explicit scoping decision.** I floated this earlier as the "structurally strongest idea." On inspection it is not a drop-in: it is a whole new distance-space modality (~700–1000 lines across three core files); it serves only distance-consuming models (k-NN, SVM, Gaussian process) and abandons the tree/linear/neural downstream models, which **narrows the project's forward-compatibility thesis**; and a per-representation weight vector structurally collides with the locked "representations never in the cache key" rule. It remains attractive in principle — it makes ECFP+Tanimoto a special case, so the hybrid could never lose to the field-standard baseline — but it is a scoping question, not a fix.

---

## 9. Where the audit disagreed

- **Unanimous (0 of 3 refuting votes):** the sign bug, the recovery degeneracy, the SAC/cliff antagonism, the at-chance detectors, the flag-fraction F1 confound, the feature-selection reachability limit.
- **Split (1 of 3 refuted):** the leakage framing. One reviewer holds it is a legitimate within-fold relative comparison and not leakage at all. The `y_noisy` fix is recommended regardless, as the honest deployment setting.
- **Refuted (≥2 of 3):** everything in §7.

Every file:line citation was re-verified against the working tree at `ed87967` on 2026-07-10. Every number was re-verified by hand against `tests/results/sim_optimized_hybrid/*.json`.
