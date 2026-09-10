# Hybrid allocation — findings, decisions, and next steps (single source)

Started 2026-07-19, updated 2026-07-23. **This is the one file to read.** Everything else
(`ALLOCATION_CATEGORIES_2026_07_19.md`, workflow transcripts) is raw material behind this.

---

## Vocabulary (defined once)

- **Representation ("rep")** — one way of encoding a molecule/sequence as a number matrix.
  One row per molecule, one column per feature. `ecfp4`, `mordred`, `saprot_cdr`.
- **Pool** — the reps a hybrid may draw from. The **knob** sets how many (N). We used N=5 and N=10.
- **Budget** — total feature columns in the finished hybrid. We used 100, 500, 1000.
- **Allocation strategy** — the rule for how many of the budget's columns each rep contributes.
  Ten were tested (see below).
- **Importance / feature-selection method** — how columns are *ranked* so the best are kept.
  A **separate axis** from allocation. Held **fixed at `lightgbm_gain`** in everything below.
- **Per-rep selection (today's behaviour)** — each rep ranks its *own* columns and hands over
  its best. A column from `ecfp4` is never compared against one from `mordred`.
- **Global selection** — glue reps together, score all columns on one scale, keep the best
  overall regardless of origin.
- **The target** — best single rep × best model. The number a hybrid must beat. No averages.

## The two axes, kept separate (this was the Finding-4 confusion)

| Axis | What it is | In this study |
|---|---|---|
| **Allocation strategy** | how the budget is split across reps | **10 tested, varied** |
| **Importance method** | how columns are ranked within the budget | **1 used, fixed (`lightgbm_gain`)** |

The feature-selectors you previously narrowed to 3 live on the **importance axis**. This study
**did not touch them** — it varied allocation only. So nothing here says anything about the
selectors. That is still open (roadmap Phase 4.1) and separate.

---

## What was run

Four datasets, **full size**, 5-fold grouped CV (`cv_split` → GroupKFold, deterministic).
**57 hybrids per dataset** = 10 allocation strategies × 3 budgets × 2 pool sizes = 60, minus
`top_5`-at-N=5 (meaningless, dropped at all 3 budgets) = 57.

The 10 allocation strategies: `top_2`, `top_3`, `top_5`, `performance_weighted`, `power_2`,
`rank_linear`, `rank_exp_d0.5`, `entropy_1.0`, `entropy_2.0`, `fixed`.

| Dataset | n | Target | 57 built? | Final (4.3) analysis? |
|---|---|---|---|---|
| herg_ki | 1412 | ecfp4 × rf, RMSE 0.5994 | yes | yes |
| logd | 5028 | mordred × lightgbm, RMSE 0.5052 | yes | yes |
| flab_thermo_tresanco | 672 | saprot_cdr × rf, Spearman 0.5959 | yes | yes |
| buchwald_hartwig | 3955 | cgr_morgan × rf, RMSE 14.37 / ρ 0.834 | yes | **NO — did not run** |

**adme_hlm was missing** and should have been included (below).

---

## FINDINGS

Framing per your instruction: the goal is **not** a single "best" strategy. It is a **small
candidate set for the search grid** (ideally smaller than today's 9), where the search still
picks the winner per dataset. Statistical significance is **not** a decision gate here — the
sample sizes cannot reach p<0.05, so it is dropped as a criterion.

### Finding 1 — feature selection is per-rep, never global (VERIFIED, real problem)
Confirmed by reading the actual columns selected: each rep keeps its own top-N by its own
importance. No cross-rep comparison exists in the code. This is the mechanism you have been
pointing at. **→ drives the global-selection rework (Round 2).**

### Finding 2 — each rep's useful signal saturates early (VERIFIED, worth formalising)
A rep stops improving after ~100–200 of its columns, regardless of width: `ecfp4` ~130/2048,
`mordred` ~175/1613, `saprot_cdr` ~100/1280. So a small per-rep quota only hurts below ~150.
It does happen (herg_ki, N=10, budget 100 → `ecfp4` gets 10/2048), but the redundant pool
refills the gap, so the hybrid does not visibly suffer. **→ new complexity experiment (Round 2).**

### Finding 3 — global selection was tested one way and lost for a FIXABLE reason
Glue pool, score all columns by `lightgbm_gain`, take best 1000 → lost 16/18. **Why:**
`lightgbm_gain` is not comparable across reps of different density. On logd/budget 1000 the
global top-1000 took 546 `grover` columns and 29 `mordred` (the *best* rep); two reps got zero.
So global handed the best rep *fewer* columns than the quota. **Not a fair test.** **→ rework
(Round 2).**

### Finding 4 — the 10 allocation strategies are barely distinct (ACCEPTED, with the caveat above)
Every strategy's predictions correlate >0.97 with every other's, and 0.971–0.990 with *just
the single best rep's* predictions. The allocation knob barely changes the answer.
**Important:** this is the allocation axis only, importance fixed. It does **not** contradict
your earlier feature-selector work — that is the other axis, untouched here.
`fixed` (equal columns per rep, the c4afb0e contract) sits mid-pack-to-2nd and is the only
strategy that is not a reweighting of one formula.

### Finding 5 — an internal ranker disagrees with cross-validation (VERIFIED, keep this thread)
**The clear version:** `create_hybrid` has to decide how good each rep is before it can split
the budget. It does this with **one single 80/20 split** — `GroupShuffleSplit(test_size,
random_state=42)` at `hybrid.py:3478` — scores each rep once on that split, and allocates the
budget from those one-shot scores. But the **honest** ranking (which we use to name the target)
comes from the **full 5-fold cross-validation**. These two rankings disagree.

Consequence: the strategies that keep only the top-2 or top-3 reps trust the one-shot ranking,
so they can **throw away the rep that 5-fold says is actually best** — and because the internal
split is seeded (`random_state=42`), it makes the *same* wrong call in all five outer folds.
So `top_2`/`top_3` looking bad is partly this bug, not the strategy itself. Any fix that keeps
only a few reps must first make the internal rep-ranking agree with the CV ranking (e.g. a
small internal k-fold instead of one split), or it will keep dropping the best rep.

### Finding 6 (correction) — herg_ki's "ecfp4 is best" premise: NOT load-bearing
I flagged the baseline table being 6/24 complete. You are right that it does not matter for the
mechanism findings (1, 3, 5) — those are about *how selection works*, not *which rep won*.
Dropped as a concern.

---

## buchwald_hartwig — RESOLVED by Exp C (this section corrected 2026-07-27)

**Correction.** An earlier version of this section (and Claude in chat) called buchwald's base
"catastrophic / failure territory" and its deltas "meaningless." **That was wrong.** Exp C
(Round 2, below) measured the scale properly and refuted it:

- **The base is real, not catastrophic.** Yield is on a **0–100 scale** (std 27.3). Target
  cgr_morgan × rf = RMSE 14.37, **Spearman 0.834, R² 0.71** — the model roughly halves the
  predict-the-mean error (27.3). 14.37 only "looked" like failure when misread as a normalised
  small number. It is not.
- **The PoC's 11.75 IS on our scale.** 15.08 → R² 0.70 (≈ our 14.37 → R² 0.71); 11.75 → R² 0.82,
  which sits right at the physics-descriptor literature ceiling (Ahneman/Doyle ~0.83 on the same
  leave-additive-out split). The 15.08-vs-14.37 gap is fold/convention noise, not a scale mismatch.
- **Still open:** the three 4.3 delta measurements (signal-survival, truncation depth,
  global-ranking counterfactual) were launched but **did not finish** — the background job died
  with the workflow. Only the signal verdict and cgr_morgan's saturation curve (~100 columns)
  completed. These deltas must still be run. See Exp C below.

---

## YOUR DECISIONS / DIRECTIVES (this session)

1. **`fixed` belongs in the candidate set** — since "top" became the pool-size knob, `fixed` is
   the one genuinely different (per-rep-budget) behaviour and should be a search-grid candidate.
2. **Stop using p<0.05 as a decision gate.** Sample sizes cannot reach it. Judge on effect
   direction/consistency and mechanism, not significance.
3. **Goal = a small candidate set for the grid, not one winner.** The search finds the best
   per dataset. Aim smaller than today's 9 allocation methods.
4. **Local tests must use FULL dataset sizes** (within CV). No signal → nothing works. Assert
   row counts every run.
5. **adme_hlm should be included.** (It was dropped as a testbed in the roadmap; revisit — it is
   one of only two datasets the earlier knob-sensitivity work used, so it checks transfer.)
6. **Global selection goes WITH the knob** — the concatenation is over whatever N reps the knob
   selects, not a fixed 5.
7. **Correlation filtering must actually be tested** (it never was, in any form). Both orders:
   filter-before-scoring and filter-after-selection.
8. **Buchwald's missing 4.3 analysis must be run.**

---

## ROUND 2 — experiments to build (your specs)

### Exp A — Per-rep signal saturation (NEW complexity experiment, no hybrids)
**Question:** where does each base rep's useful signal run out? For each rep, rank its columns
by importance, score the rep alone at increasing feature counts (e.g. 10, 25, 50, 100, 200,
300, 500, 1000, full) under full-size CV, and find the plateau. **No hybrids anywhere.**
Formalise as a new file in the complexity suite, matching existing complexity-experiment
conventions. Smoke-test on all four local datasets at full size.

### Exp B — Global selection done properly, tied to the knob
**Question:** does global selection beat the per-rep quota once the density-bias is removed?
Take the **top-N features per rep by importance** (N swept over 50, 100, 200, 300),
concatenate across **the knob's N reps**, then select the budget globally from that balanced
pool. Compare against: (i) the per-rep quota, (ii) the naive glue-everything global from
Finding 3. Full-size CV, all four datasets.

### Exp C — Finish buchwald 4.3
Run the signal-survival, truncation-depth, and global-ranking-counterfactual measurements that
did not run — but **first** settle the RMSE-scale question and whether buchwald has real signal
locally. If the base is failure-territory, say so and stop; do not report deltas on it.

### Later (not Round 2, but recorded): correlation filtering
Test dropping near-duplicate columns, both orders (before scoring / after selection). Needs a
definition of "duplicate" for columns (open question below).

---

## OPEN QUESTIONS FOR YOU (not decided)

- **Column-level "redundancy" definition.** For reps we correlate prediction vectors. A single
  column has no model, so options are: (a) correlate raw column values [you rejected this for
  reps]; (b) one-feature model per column, correlate predictions [honest, ~15k fits on logd];
  (c) mRMR-style — score each column against the residual left by already-picked columns
  [cheap]. This choice changes what "correlated" means throughout.
- **adme_hlm:** add it back to the local testbed set for Round 2? (recommend yes)
- **buchwald:** resolve the RMSE scale, then decide keep-or-drop based on absolute signal.

---

## VERIFIED / SAFE (so you can trust the mechanism findings)

- Metric direction checked at every stage with a deliberately-good and deliberately-bad matrix;
  a positive number always means the hybrid won. Both directions, both metric types.
- Split logic is **your** `cv_split` (GroupKFold, deterministic) — not a new invention.
- `find_best_hybrid` was never called → results DB never read/written → no cache poisoned.
- Nothing under `src/` was modified.

---

## ROUND 2 — RESULTS (2026-07-23)

These are the measured outcomes of the three Round 2 experiments specified above. Everything
was run at **full dataset size** (every row loaded, row-count checked against the label count),
on the four staged datasets (herg_ki, logd, flab_thermo_tresanco, buchwald_hartwig), using your
own `cv_split` (GroupKFold, deterministic). `adme_hlm` is not staged on the drive and was not
attempted. Nothing under `src/` was touched; the real hybrid evaluator (`find_best_hybrid`) was
never called, so no cache was read or written.

### Metric-direction safety check (done first, blocks everything else)

Before any experiment produced a number, a built-in check confirmed that a "better" prediction
and a "worse" prediction come out with the right sign. We built a deliberately-good predictor
and a deliberately-bad one from the true labels plus increasing amounts of random noise, so they
are good-to-bad *by construction*, never by damaging a real representation. On herg_ki (error
metric, where **lower is better**) the error climbed steadily from 0.109 up to 0.941 as we added
noise — so the good predictor scored 0.109 and the bad one 0.941, which is the correct
direction (worse predictor = larger error). On flab_thermo_tresanco (rank-correlation metric,
where **higher is better**) the correlation fell steadily from 0.992 down to 0.204 — so the good
predictor scored 0.992 and the bad one 0.204, again correct. This check is wired to stop the run
with an error if it ever fails, and it passed. Reminder of the direction convention used
throughout: herg_ki, logd and buchwald_hartwig use root-mean-square error (RMSE), lower is
better; flab_thermo_tresanco uses Spearman rank correlation, higher is better.

### Exp A — Where each representation's signal saturates (the plateau)

**What we did.** For each representation on its own (no hybrids anywhere), we ranked that
representation's own columns by `lightgbm_gain` importance using **only the training rows of
each fold** (so there is no leakage), then measured its score keeping only the top 10, 25, 50,
100, 200, 300, 500, 1000, or all of its columns. The "plateau" is the smallest number of columns
that already gets within a small tolerance of the full-width score. We ran two downstream models
(LightGBM and Random Forest) so we could see whether the plateau is a property of the data or of
the model.

**No new file was added, and it did not duplicate c05 or c16.** The experiment already existed
as an untracked file, `tests/experiments/complexity/c20_signal_saturation.py`, which implements
exactly this task; per the "subtract, don't add" directive it was verified and run rather than
duplicated. It is genuinely distinct from the two neighbours we were asked to check:
`c05_learning_curves.py` sweeps the **number of rows** (sample size) and builds a hybrid at each
size — a different axis and a hybrid experiment; `c16_feature_selection_as_regularization.py`
sweeps the **hybrid's** budget over a multi-representation blueprint and uses Ridge — a hybrid +
regularisation experiment. c20 is single-representation, no hybrid, no concatenation, and does
not use Ridge or `find_best_hybrid`.

**Plateau table** (full-size grouped 5-fold CV; `plateau@k` = smallest column count within
tolerance of full width; "FULL" = keeps improving to native width; smoke pool of 8
representations per dataset — disclosed below, not every staged representation):

*herg_ki (RMSE, n=1412):* ecfp4 LightGBM@200 / RF@100; rdkfp @500 / @100; maccs @100 / @50;
avalon @500 / @100; atom_pair @200 / @100; top_torsion @200 / @200; mordred @200 / @25; mol2vec
@100 / @25.

*logd (RMSE, n=5028 — the largest dataset):* ecfp4 LightGBM@500 / RF@100; rdkfp @500 / @100;
maccs @100 / @100; avalon @1000 / @100; atom_pair @500 / @100; top_torsion @1000 / @100; mordred
@100 / @25; mol2vec @100 / @25.

*flab_thermo_tresanco (Spearman, n=672):* aac @10 / @10; onehot @500 / @100; blosum @FULL / @100;
esm2 @300 / @300; antiberty @10 (weak signal) / @500; prot_t5 @50 / @FULL; structural @10 (weak
signal) / @10; mutation_features @FULL / @25.

*buchwald_hartwig (RMSE, n=3955):* drfp_2048 @50 / @50; rxnfp @50 / @25; reactant_ecfp4 @100 /
@50; diff_ecfp4 @25 / @25; cgr_morgan @25 / @50. Three representations — product_ecfp4,
product_mordred, ligand_descriptors — gave a **completely flat, no-signal curve** (error stuck
near the label spread of ~25–26 at every column count); their "plateau@10" is a meaningless
artifact of a flat line, not real saturation, and should not be read as "10 columns suffice".

**Did the earlier ~100–200 claim hold at full size? Partly.** It is a fair rough middle but not
a universal law, and it bends in two systematic ways. (1) **The model matters:** Random Forest
almost always plateaus early (25–100 columns) because its averaging already regularises, while
LightGBM often keeps using more (100–500+). (2) **Dataset size matters:** on the largest dataset,
logd, LightGBM's plateau moved *later* — several fingerprints only saturate at 500–1000 columns
(avalon and top_torsion at 1000), because more training rows let the model exploit more columns
before it overfits. A few wide, high-signal representations never plateau and keep improving to
full width (blosum and mutation_features under LightGBM; prot_t5 under Random Forest on flab).
**So the honest statement is: report the plateau per (representation, dataset, model), not as one
"~100–200" number.**

**Metric direction:** the blocking check described above ran first and passed on both directions,
so the plateau scores have the correct sign.

### Exp B — Global selection done properly, tied to the knob

**What we did.** Three ways of filling the budget were compared, per dataset, never averaged
across datasets. **Arm 1 (per-representation quota):** today's behaviour — each representation
keeps its own top columns. **Arm 2 (balanced global, tied to the knob):** first cap each of the
pool's representations at its own top-N (N swept over 50/100/200/300), concatenate across the
knob's N representations, then pick the budget from that balanced pool. **Arm 3 (naive global):**
glue everything together and take the global best columns with no per-representation cap — the
version that lost in Finding 3. The target each arm must beat is the best single representation ×
best model. Pools of 5 and 10 representations and budgets of 500 and 1000 were used. Positive
numbers mean the arm beat the target; the "folds-won" count is out of 5.

**Did balanced global (Arm 2) beat the per-representation quota (Arm 1)? Mostly yes, modestly,
wherever hybridising helps at all — and it never helped on buchwald.** Per dataset:

- **flab_thermo_tresanco (n=672, Spearman):** Arm 2 clearly best. At pool 10 / budget 500 it beat
  the target by +0.0505 (the single best cell in the whole study) versus the quota's +0.0248; at
  pool 5 / budget 500, +0.0354 versus +0.0248. This protein dataset showed the biggest margin.
- **logd (n=5028, RMSE):** Arm 2 wins at pool 10 (+0.0306 vs quota +0.0210 at budget 500; +0.0331
  vs +0.0302 at budget 1000) and roughly ties the quota at pool 5.
- **herg_ki (n=1412, RMSE):** Arm 2 wins at pool 5 / budget 500 (+0.0131 vs +0.0053) and roughly
  ties at pool 10 (+0.0216 vs +0.0183).
- **buchwald_hartwig (n=3955, RMSE):** none of the three arms beat the target — every hybrid was
  about 1.0 to 1.3 RMSE units *worse* than the single best representation (concat_ecfp4). Here the
  single representation dominates and any hybrid hurts; balanced is no help and marginally worse
  than the quota.

The wins are small (0.003–0.026) and folds-won is usually 2–4 of 5, not unanimous. The three arms'
predictions correlate 0.90–0.99 with each other, so they are largely reselecting the same signal.
The fair summary: **balanced global is a safe, usually-slightly-better replacement for the quota
on the three datasets where hybridising helps, biggest at the larger pool and on the protein
data — not a large win.**

**Did balanced global fix the density bias? Yes, and the origin counts show the mechanism.** The
naive arm (Arm 3) concentrates the budget onto one wide, dense representation: at pool 10 / budget
1000 the top representation grabbed 71% of the columns on herg (grover), 63% on logd (grover),
63% on flab (esm2_per_region) and 76% on buchwald (reactant_grover). This is exactly when naive
turns **negative** while balanced stays positive — herg pool 10 / budget 1000: naive −0.0244 vs
balanced +0.0216; logd: −0.0039 vs +0.0331; flab: −0.0506 vs +0.0345 — reproducing Finding 3 and
showing the per-representation cap removes it. The balanced arm's origins are equal by
construction (each representation is capped before the global ranking), so no single one can
swamp the pool. **Caveat on when naive is fine:** at the smaller pool of 5 with lower budget,
naive is competitive and occasionally wins (herg pool 5 / budget 1000: naive +0.0139 vs balanced
+0.0066), because pool 5 excludes the biggest dense fingerprints so the bias is mild. Naive fails
specifically when the pool contains a large dense fingerprint — which is exactly when the
top-N-per-representation cap earns its keep.

**Metric direction:** the blocking check passed (both directions) before any Exp B number.

### Exp C — Buchwald: is the base real, or is it failure territory? (verdict first)

**The scale question is resolved and the base is NOT catastrophic.** buchwald_hartwig predicts
**reaction yield on a 0–100 percentage scale** (measured: minimum 0, maximum 100, mean 33,
standard deviation 27.3). The target (cgr_morgan × Random Forest, measured this session on all
3955 rows with the leave-one-additive-out grouped split) is RMSE **14.37**, Spearman **0.834**,
and R-squared **0.71**. The "14.37 looks like failure" impression came entirely from reading 14.37
as if it were on a small normalised scale. It is not: it is 14 yield-percentage-points of error
on a 0–100 scale where simply **predicting the average yield already costs 27.3** — so the model
roughly **halves** the trivial error. For context, the published Ahneman/Doyle work on this exact
3955-reaction set reaches about R-squared 0.83 on the same hard leave-additive-out split using
tuned physics-based descriptors; our generic fingerprint at 0.71 sits below that ceiling but far
above noise. **Verdict: real, mediocre-to-good signal — continue, do not stop.**

**Is the proof-of-concept's 11.75 on our scale? Yes.** Converting through the label spread, the
proof-of-concept's 15.08 corresponds to R-squared 0.695 — essentially identical to our own
measured 14.37 (R-squared 0.706); the small 15.08-vs-14.37 gap is fold/scoring-convention noise,
**not** a scale mismatch. The proof-of-concept's improved 11.75 corresponds to R-squared 0.82,
which lands right at the physics-descriptor literature ceiling. So the earlier worry that "11.75
may not be on our scale at all" is **refuted** — both numbers are on the same 0–100 yield scale,
and the claimed 15.08→11.75 improvement is R-squared 0.70→0.82.

**The three measurements were therefore warranted and started, but did not finish.** Because the
signal is real, the signal-survival, truncation-depth, and global-ranking measurements were
launched (full 5-fold CV, `lightgbm_gain` importance, LightGBM downstream, calling the hybrid
building blocks directly, never `find_best_hybrid`). At the forced cutoff only the *reference
saturation curve* for the hero representation cgr_morgan had partly completed (4 of 5 folds): its
usable signal lives in roughly the top **100–200** of its 2048 columns (keeping 100 or 200 is
about as good as keeping all; keeping only 10 or 50 is clearly worse), the same shape seen on
herg and logd. The truncation-depth arm produced only one calibration fold (a top-5 / budget-100
quota gave cgr_morgan just 20 of its 2048 columns, i.e. below its saturation point, so the
representation-alone slice is expected to be starved — consistent with herg/logd, where that
starvation did **not** propagate to the full hybrid). **The global-ranking counterfactual did not
run at all.** None of these three delta measurements should be quoted as a result yet; the
background job checkpoints to `scratchpad/c43_step3_ckpt.json` and continues on its own.

### ROUND 2 Exp C is now COMPLETE (finished 2026-07-27) — see ROUND 3 below
The three buchwald delta measurements that died mid-run were re-run to completion. Results are
in the ROUND 3 section. Short version: on buchwald **no hybrid beats the best rep alone** — the
quota is a real but non-decisive defect there.

### Consolidated CAVEATS (all three experiments)

1. **Only herg_ki and flab_thermo_tresanco had their metric direction proven end-to-end** (RMSE
   and Spearman respectively). logd and buchwald_hartwig are taken as RMSE/lower-is-better from
   the registry; that direction was **not** independently gate-tested for them this session.
2. **Exp A ran a disclosed 8-representation smoke pool per dataset, not every staged
   representation** (herg/logd: ecfp4, rdkfp, maccs, avalon, atom_pair, top_torsion, mordred,
   mol2vec; flab: aac, onehot, blosum, esm2, antiberty, prot_t5, structural, mutation_features;
   buchwald: drfp_2048, rxnfp, product_ecfp4, reactant_ecfp4, diff_ecfp4, product_mordred,
   cgr_morgan, ligand_descriptors). No rows were subsampled — full n throughout. A full-pool run
   is just dropping the smoke flag and will be slower.
3. **Exp A: three buchwald representations are degenerate no-signal** (product_ecfp4,
   product_mordred, ligand_descriptors) — flat curves at the label spread; their plateau values
   are artifacts, not saturation. Two flab representations (antiberty, structural) are weak-signal,
   so their plateaus sit on very low scores and carry little information.
4. **Exp B's full grid was NOT recomputed from scratch this session.** A complete prior run of
   exactly this experiment (same harness, arms, pools, budgets, N-caps, passing direction gate,
   target reproduced) was already staged; those numbers are reported. This session
   *independently validated* the artifact by re-running flab from scratch (target matched exactly;
   two arm scores matched to within 0.002; the naive density-bias origin pattern reproduced). The
   herg/logd/buchwald grids were validated only transitively (same method, flab reproduced, all
   four report the target reproducing). Small non-exact matches are LightGBM's multi-thread
   run-to-run noise, not a method difference.
5. **Exp B origin counts wobble in the middle ranks** run-to-run (the global gain ranking over
   many near-tied fingerprint columns depends on threaded randomness). The **top** concentration —
   which dense representation dominates and its ≥60% share at large pool / high budget — is stable
   and is what the density-bias conclusion rests on; the exact per-representation counts are not.
6. **Exp B predictions are highly redundant across arms** (0.90–0.99 correlated). The arms mostly
   reselect the same signal, so a +0.02 delta should not be over-read as a large practical win.
7. **Exp C is incomplete.** Only the scale resolution and the "signal is real" verdict are
   finished deliverables. Of the three measurements, only a partial (4/5-fold) saturation curve
   and one calibration fold exist; the global-ranking counterfactual never ran. No Exp C delta is
   a result yet. No missing number was fabricated.
8. **Exp C's downstream model is LightGBM**, while the buchwald target anchor (14.37 RMSE, 0.834)
   was measured with Random Forest; the two are internally consistent within their own model but
   the LightGBM per-fold reference numbers are not a second estimate of the 14.37.
9. **Exp C's pool for the measurements is a 10-representation domain-reasoned set, not an
   exhaustively score-ranked pool and not the exact proof-of-concept top-5** (the true
   proof-of-concept top-5 likely used molecular-representation variants not wired into the reaction
   load path). "Best representation" is fixed to cgr_morgan by the anchor, not re-selected.
10. **buchwald's split is leave-one-additive-out** (22 additive groups over 5 folds) — the
    correct hard reaction split, but coarse, so the target's per-fold error legitimately ranges
    from 9.9 to 17.1.
11. **This ran machine-learning locally**, against the general no-heavy-local-compute preference,
    because the task explicitly required a full-size smoke; it is only LightGBM/Random-Forest
    cross-validation, never `find_best_hybrid`. Read-only cache on `/Volumes/seagate` was honoured
    — nothing was written there.
12. **No p<0.05 gate was used anywhere** (per your directive). Verdicts rest on effect direction,
    consistency across folds, magnitude relative to the scale, and mechanism.

### NEW open question for you

- **Exp C is still running and will finish on its own.** The buchwald signal is confirmed real,
  so the three delta measurements (signal-survival, truncation-depth, global-ranking
  counterfactual) are warranted — but they had not completed at cutoff. **Do you want the finished
  Exp C numbers folded into this file as a short addendum once the background job lands, or
  reported separately?** (No result is being claimed from them yet.)

---

## ROUND 3 — RESULTS (2026-07-27, all in R²)

Three pieces: (C) buchwald's finished delta measurements; (D) the new performance-weighted
global selection experiment; (E) the allocation-strategy removal case with the model axis added.
All full-size, `cv_split` GroupKFold, `lightgbm_gain` importance fixed, `create_hybrid` direct,
`find_best_hybrid` never called, nothing under `src/` touched. Reported in **R²** (coefficient of
determination on out-of-fold predictions; higher is better, scale-free). Native metric (RMSE /
Spearman) kept as a secondary check — it agreed in direction on every row. **No averaging across
datasets or models anywhere.**

### C — buchwald_hartwig, the three measurements (now finished)

Downstream model = lightgbm throughout (so target = cgr_morgan × lightgbm, R² **0.666**; the rf
anchor was 0.71 — different model, kept as reference only).

1. **Signal survival** (best rep's contributed slice vs cgr_morgan at full 2048 width): at budget
   100 the quota **starves** the best rep — it gets ~14–20 columns, the slice loses 0.05–0.08 R²,
   full-width wins 4–5/5 folds. At budget 1000 the starvation is **gone** — cgr gets ~140–250
   columns (above its ~100-column saturation), and the slice recovers to parity.
2. **Truncation depth** (recall of cgr_morgan's top-100 into the hybrid): budget 100 → **14–21%**;
   budget 1000 → **100%**. The binding constraint at small budget is the quota *count*, not ordering.
3. **Global-ranking counterfactual** (one global ranking vs per-rep quota, same budget): global R²
   0.632 (B100) / 0.649 (B1000); it **beats the quota** in most cells. But — the decisive point —
   **no hybrid, global or quota, beats cgr_morgan alone (0.666).** Best global 0.649, best quota
   0.646, both below the single rep.

**Verdict:** the per-rep quota is a real defect (starves the best rep at low budget, loses to a
global ranking) but **non-decisive on buchwald** — even fully fed, every hybrid lands below the
best single rep. On this dataset one rep is the ceiling and mixing dilutes it. **The proof-of-concept
hybrid win is NOT reproduced in an honest full-size setup.** (Origin note: a free global ranking
barely picks cgr — ~10/100 features — it favours reactiont5 and drfp instead.)

### D — Performance-weighted global selection (your new idea) — file `c31_perf_weighted_global_selection.py`

Idea: size each rep's slice of the global candidate pool by its **performance weight** (the same
`performance_weighted`=linear and `power_2`=score² families the allocators use), then rank the
pooled features globally and take the top B. Downstream = lightgbm for all arms and the target
(deliberate — model swing ~0.27 would drown the ~0.01–0.08 selection deltas; see caveat). R²Δ =
arm minus target (best rep × lightgbm); folds-won /5. Compared against Arm1 per-rep quota, Arm2
equal-cap balanced global, Arm3 naive global.

**herg_ki** (target mhg_gnn, R² 0.581): weighted **power_2 B1000 = +0.0341 (5/5)** > quota +0.0260
> equal-global +0.0131 > naive −0.0176 (loses at B500).
**logd** (target mordred, R² 0.810): weighted **power_2 B500 = +0.0287 (5/5)** > equal-global
+0.0242 > quota +0.0214 > naive **loses** (−0.011 to −0.012).
**flab_thermo_tresanco** (target saprot_cdr, R² 0.299): quota +0.0803 ≈ equal-global +0.0755 ≈
weighted +0.0730 — all comparable, weighted does **not** beat the simpler arms here; naive weakest.
(buchwald not run as a control — kept to the three hybridising datasets for time.)

**Density-bias check (the key one) — weighting KILLS it, does not re-introduce it.** Share of the
final budget taken by the *widest* rep:

| Dataset | Widest rep (width, standalone R²) | Naive | perf_weighted | power_2 |
|---|---|---|---|---|
| herg_ki | grover (4800, R² 0.531) | **52.2%** | 3.4% | **0.3%** |
| logd | grover (4800, R² 0.759) | **76.2%** | 13.0% | **6.6%** |
| flab | esm2_per_region (8960, R² 0.188) | 10.6% | 8.4% | 7.4% |

Naive lets a wide, mediocre rep (grover) grab 52–76% by sheer column count; performance-weighting
drops it to 0.3–7% and the pool is instead led by the *top* reps (herg: mhg_gnn 74%; logd:
mordred+mhg_gnn 84%). On flab there was no width bias to begin with, so weighting has little to fix.

**Verdict:** performance-weighted global is the **best combiner where combining helps** (herg, logd
— beats equal-global and quota) and **ties the simpler arms where one rep-type dominates** (flab).
It beats naive everywhere. `power_2` weighting suppresses the weak wide rep harder than linear, as
intended. It does **not** re-introduce the density bias — the opposite.

### E — Allocation-strategy removal case (with the model axis)

**Preferences are opposed across datasets, so averaging is invalid** (your rule, now proven):
spreaders (`power_2`, `performance_weighted`, `fixed`, `rank_linear`) win on herg_ki + logd;
concentrators (`top_2`, `top_3`) win on buchwald + flab. `top_2` is best on two datasets and worst
on two.

**The model axis is real:** a mini-sweep (10 strategies × {lightgbm, rf} × pool10 × budget 500)
found **13 preference flips** — a strategy competitive under one downstream model, dominated under
the other in the *same* dataset (e.g. herg `entropy_2.0`: +0.024 under lightgbm, worst under rf).
Under rf the strategies compress together (rf is robust to apportionment); under lightgbm the
choice matters more. **You cannot cut a strategy on one model's evidence.**

**Nothing is dominated in every cell and nothing is significant** (Round 1: 0/57 beat target at
p<0.05). So the cut basis is **behavioral redundancy**, not "X beats Y."

Cross-cell record over the 6 completed (dataset, model) cells:

| strategy | rank-1 | in top tie-band | dominated |
|---|---|---|---|
| power_2 | 2 | 6 | 0 (never dominated) |
| performance_weighted | 1 | 4 | 2 |
| top_3 | 1 | 4 | 2 |
| fixed | 0 | 4 | 2 |
| top_2 | 0 | 3 | 3 |
| top_5 | 1 | 3 | 3 |
| rank_linear | 1 | 3 | 3 |
| rank_exp_d0.5 | 0 | 3 | 3 (never rank-1) |
| entropy_1.0 | 0 | 3 | 3 (never rank-1) |
| entropy_2.0 | 0 | 2 | 4 (never rank-1; degenerate) |

**Proposed cuts (redundancy-based, each keeps a distinct behaviour covered by a survivor):**
- **`rank_exp_d0.5`** — a concentrator never rank-1 anywhere; its niche is covered by `top_3`.
- **`entropy_1.0`** — spread-cluster (≈0.98 prediction-corr with `power_2`); its redundancy penalty
  is raw L2-norm geometry, i.e. the wrong measure, so "the redundancy-aware arm" isn't a reason to keep it.
- **`entropy_2.0`** — degenerate (hands ~100% of budget to 1 rep; not a hybrid). Keep only as a control.

That is the **conservative 3-cut → 7 candidates**. An **aggressive 5-cut → 5** also drops `top_5`
(algebraic duplicate of `performance_weighted` at k≥n_reps) and `rank_linear` (spread-cluster
near-duplicate of `power_2`), leaving `{power_2, performance_weighted, top_2, top_3, fixed}` + the
pooled slot. Both keep ≥2 spreaders, ≥2 concentrators, and `fixed`.

### PROPOSED SEARCH GRID (for your decision — not adopted)

Fill the pooled slot with **performance-weighted global (`power_2`)** — it is the best combiner on
the two datasets where combining helps and fixes the density bias. Candidate grid:

`{ power_2, performance_weighted, top_2, top_3, fixed, perf_weighted_global }`  (6, from the 3-cut)
or the tighter `{ power_2, top_3, fixed, perf_weighted_global }` (4, if you take the 5-cut and treat
`power_2`/`top_3` as the spread/concentrate representatives).

### Cross-cutting conclusions (Round 3)

1. **Budget is the dominant lever, not the strategy** — B1000 ≫ B100 on every dataset; confirms the
   roadmap's budget-uncap (Phase 5.1). Strategy choice is second-order (~0.01–0.03 R²).
2. **The internal-ranker bug (Finding 5) must be fixed before trusting concentrators** — `top_2`/
   `top_3` drop the true-best rep because a single seeded 80/20 split ranks reps against the 5-fold CV.
3. **buchwald does not reproduce a hybrid win** — one rep is the ceiling. The PoC target needs to be
   re-examined or retired.

### ROUND 3 CAVEATS
1. **All Round-3 comparisons fix the downstream model to lightgbm** (except the allocation mini-sweep
   which used both lightgbm and rf). This is deliberate — the model swing (~0.27) dwarfs selection
   deltas (~0.01–0.08) and would confound the paired comparison — but it means "target" = best-rep ×
   lightgbm, not best-rep × best-model. A best-model sweep for weighted-global is not yet done.
2. **Nothing is statistically significant** at these budgets/pools; deltas are ~0.01–0.08 R² with
   noisy folds (flab target folds span R² 0.46–0.78). Conclusions rest on direction, consistency, and
   mechanism, not significance — per your instruction.
3. **Single seed, 5 folds, no repeated CV.** flab (n=672) is the noisiest.
4. **Weighted-global's pool-sizing weight uses the full-CV standalone score** (mild optimism); the
   final top-B selection and global ranking are strictly train-fold-only and leak-free.
5. **buchwald was not run in the weighted-global experiment** (control, dropped for time); its global
   behaviour is covered by Exp C above (global beats quota, but nothing beats the best rep).
6. **File cleanup pending your OK:** `exp_b_balanced_global.py` (23 Jul) is superseded by
   `c30_balanced_global_selection.py` (27 Jul) — safe to delete. The two `c43_*` files are NOT
   duplicates (signal verdict vs step-3 measurements) — keep both. New tracked-worthy experiment
   files: `c20_signal_saturation.py`, `c30_balanced_global_selection.py`, `c31_perf_weighted_global_selection.py`.
