# Similarity hybrids — design and testing guide (2026-08-24)

**Status: PROPOSAL, not signed off. Nothing here has been run. No number in this file
is a measured result.** Open decisions are at the end (§11).

Master record for the similarity thread: `SIMILARITY_METRICS_ROADMAP.md`.
Suites this plugs into: `tests/experiments/similarity/` (s01–s13) and
`tests/experiments/noise/` (n01–n15).

Every code fact in §2 and §3 was read out of the file on 2026-08-24. Doc claims in
this repo were not trusted — several are AI-written and were never measured.

---

## 1. The idea in one line

Pick a small set of features that makes two molecules *compare* well, rather than a
large set that makes a model *predict* well — and prove the small set is better at
the jobs that involve comparing molecules.

---

## 2. What the code actually does today

**2.1 The similarity suite asks for a prediction hybrid and then measures similarity
on it.** Every s-experiment that needs a combined representation pulls the best
already-computed one out of the shared store. That store was built to maximise model
accuracy.

**2.2 The engine has two similarity targets and almost nothing uses them.**
`objective=` takes `predictive`, `similarity_sac` (distance vs activity difference)
and `similarity` (activity read-across). Under either similarity setting **no
downstream model is fit anywhere** — not in the allocator, not in the per-fold
scoring, not in the per-representation baseline. Grep across
`tests/experiments/similarity/`: zero hits. The only caller in the repo is
`tests/experiments/similarity_optimized_hybrid.py`, and it only ever uses
`similarity_sac`. **`objective='similarity'` has no caller at all.**

**2.3 The global-pool allocator is not in the default set.** `DEFAULT_ALLOCATION_METHODS`
is nine names and **all nine resolve to the same underlying allocator**
(`performance_weighted`). `EXTRA_ALLOCATION_METHODS = ['weighted_global']` is a
comment — no code reads it. The registry has 43 names that collapse to 5 real
behaviours: 29 → performance-weighted, 7 → greedy, 3 → dominant-greedy,
3 → weighted-global, 1 → fixed.

**2.4 Every distance in both suites is Euclidean or cosine, never a set coefficient.**
41 call sites of the shared distance function; none passes a metric name. In the noise
suite `n11` already sweeps `['euclidean','cosine']`; `n12` hardcodes euclidean; `n02`
and `n07` hardcode cosine for detection and euclidean for smoothing *in the same
experiment*. Tanimoto appears in the similarity suite exactly once, to define what
counts as an activity cliff — never as a coefficient under test.

**2.5 The structural scorer is not an optimisation target. UPDATED 2026-08-26 — it is
no longer unreachable.** There is still no structural objective constant, so it cannot
be selected on. But the reason given here for why it could not be computed inside the
search is now fixed. The MCS pairs store row numbers into the whole dataset; the search
renumbered each fold's rows from 0; nothing carried the original numbers in. Measured on
a 100-row fold of a 200-row set: fidelity 0.9978 with the original row numbers, -0.0671
without. `find_best_hybrid`, `_run_search` and `_evaluate_hybrid_config` now take
`row_index` and `structural_pairs`, `nested_cv_evaluate` fills `row_index` with the
outer-training row numbers, and every configuration's metrics carry `struct_fidelity`
per fold. Recording it does not change which configuration wins (identical to 1e-12).
Whether to add a fourth objective is open — see `STRUCTURAL_OBJECTIVE_PROMPT.md`.
The similarity suite still does not use `nested_cv_evaluate` at all.

**2.6 Inside a representation, columns are still chosen by a label-fitted model.**
Under a geometry objective the *cross-representation split* is geometry-driven, but the
feature-importance step is not skipped — it still ranks columns with lightgbm gain /
treeshap / deeplift against the labels. So "the metric was built by geometry" is only
half true today.

**2.7 The allocator ranks representations by distance-vs-activity-difference even when
the objective is read-across.** Read-across is applied only when scoring the assembled
hybrid on the held-out fold. So the thing being optimised during selection and the
thing being reported are different statistics.

**2.8 Two axes of the search are provably degenerate under a geometry objective.**
The allocator returns its score before reading `model`, `metric_fn` or
`internal_evaluator`. So every downstream model and every internal evaluator produce
byte-identical allocations — but the search still builds one configuration per model
per evaluator, and recomputes the per-representation baseline once per model name.
Meanwhile budgets are *still* filtered by a per-model cap for models that are never
instantiated (svm ≤ 500, gp/mlp ≤ 1000, knn ≤ 2000, tree models uncapped).

**2.9 Two silent-zero traps.** `objective='similarity'` with `groups=None` returns
exactly 0.0 for every configuration and every baseline — a search that "completes"
with all scores identical. And the scope gates already skip s02/s07/s08 above 15,000
rows and skip several noise experiments on non-molecular data.

**2.10 The cache-key fix is written but uncommitted.** The change that namespaces the
objective into the tier, the baseline filename and the hybrid-eval filename exists only
as unstaged lines in the working tree. The same unstaged diff also **deletes `speed`
from the result-database identity**, which changes the identity of every stored row.

---

## 3. Prerequisites — code that must be fixed before any run

These are not optional. Each one either crashes, silently drops folds, or produces a
number that is not what it claims to be.

| # | Problem | Consequence if ignored |
|---|---|---|
| P1 | The per-fold factory-representation baseline loop has no geometry branch — it always fits a model and calls the metric function, which under a geometry objective is a deliberate raising guard. | Family B + a similarity target **crashes**. That is exactly the configuration the whole study turns on. |
| P2 | The augmentation scoring helpers inside `create_hybrid` (9 call sites) never forward `objective`. Same for the greedy budget-fill path. | The raised error is swallowed per fold; folds silently vanish until "all folds failed", or a config survives on fewer folds than its competitors and is compared against them anyway. **Force augmentations off until fixed.** |
| P3 | The word `objective` does not appear anywhere in the experiment-layer plumbing. Six functions would each need it threaded: the precompute, the dict populator, the precomputed-search getter, the pool-best getter, the best-for-downstream picker, and the per-fold rebuilder. | Step 1 is not "one switch". |
| P4 | The in-memory precomputed-search fast path is keyed on the augmentation condition alone. | A similarity request against a store populated with prediction searches is **handed the prediction search with no error**. This is the most dangerous defect in the list: it does not crash, it lies. |
| P5 | The per-fold rebuilder never forwards the blueprint's objective, even though the blueprint carries one. All eight rebuild sites in the similarity suite also omit groups, model and metric. | The search picks a geometry-optimal recipe, then every fold rebuild re-runs the allocator **predictively**, and the internal accept/reject holdout is a seeded **random 20% split scored by R²** — on scaffold-split data. That violates the no-random-splits rule inside eight experiments. |
| P6 | The persistent hybrid-evaluation cache records representation names but no width, and ignores subset hashes. | A column-dropped run and a full-width run collide, and the dropped run can be served the full-width score. That is precisely Family B's shape. |
| P7 | The objective cache-key change is uncommitted, and the same diff removes `speed` from the row identity. | A geometry run and a prediction run share a filename until it lands; and landing it invalidates every existing stored row. **Commit and decide the wipe before anything else.** |
| P8 | `src/kirby/similarity/` is untracked and unstaged in a tree with ~110 modified files and nothing staged. | It is the only held-out-validated version of the column-dropping procedure. Any checkout or reset loses it. **Commit it first.** |
| P9 | Family B has no landing site: the experiment loader filters factory representations out of the base list, only builds factories when fine-tuning is on, and its factory builder knows four types, none of them a similarity recipe. The suites take their list from the `general` entry, where no `sim_*` name is registered. | Step "add Family B" is blocked by three separate filters. |
| P10 | Every noise experiment picks its base representation from a hardcoded three-name list, otherwise falls back to the first name alphabetically. | Adding a `sim_*` name could silently make it the *base* arm — i.e. the control becomes the treatment. |

**P0 (do first): commit the untracked similarity package and the uncommitted engine
diff, and decide the cache wipe.** Everything else is unsafe until then.

---

## 4. The grid

Six axes. Everything reported per dataset — never averaged across datasets.

### ⚠ Scope rule — read before trimming anything

**The default is FULL coverage of every axis.** This runs on the server, overnight
compute is free, and an under-explored grid costs far more than a long one — the whole
reason this study exists is that earlier work cut corners and produced results nobody
could interpret.

Anything omitted must be omitted for **one of three stated reasons**, never for cost:

1. **Degenerate** — the code provably produces an identical result (must be verified by
   assertion, not assumed).
2. **Broken** — it raises or silently drops folds. The response is to **fix it**, not to
   drop it. Only drop if the fix is out of scope, and say so.
3. **Undefined** — the quantity does not exist on that object (e.g. a set-overlap
   coefficient on standardised continuous columns).

A theory about why something *should* lose is **not** a reason to exclude it. Predictions
of that kind belong in the write-up as a stated expectation, tested.

**No silent caps.** If any run is bounded (top-N, sampled pairs, a dropped cell), log
what was dropped in the output. A truncated run that does not say so reads as complete.

### Axis A — the feature-picking method (two families, head to head, one winner)

| Family | What it does | State |
|---|---|---|
| **A. Allocation strategies** | Score each representation, split the budget between them, take each one's best columns. | Built; never run with a similarity target |
| **B. Direct structural picking** | Ignore representations as units — score columns directly against structural truth. Drop the ones that hurt (the column-dropping procedure), rank the fine-grained ones by how well they track shared substructure. | Built in `src/kirby/similarity/selection.py`, held-out validated, untracked, never scored |

### Axis B — block normalisation (**new, and probably the most important axis**)

In a distance, **the number of columns a representation gets *is* its weight in the
metric**. The concatenation step just glues raw columns together with only a
per-column standardiser after it, so squared Euclidean distance is the sum over every
kept column — a representation holding 400 of a 500 budget contributes ~80% of every
comparison.

Your own July code already breaks this coupling: it divides each standardised block by
the square root of its width, so a block's influence stops depending on how many
columns it won. **No allocator in the registry does anything equivalent.**

Two levels: **raw** (current behaviour) and **equal-block**. Applied identically to
every arm. It separates "these columns" from "this much weight" — without it, every
allocator result is confounded.

⚠ **CORRECTED 2026-08-26: √width is the right divisor only for the Euclidean family.**
After z-scoring every feature has unit variance, so a block's expected squared-distance
contribution is proportional to its feature count. Measured on a 2048-wide block glued to
a 128-wide one, both z-scored: the wide block's mean pairwise distance is 4.007x the
narrow one under Euclidean and 15.999x under Manhattan. Dividing by √width brings
Euclidean to 1.002 and leaves Manhattan at 4. **Manhattan, Canberra, Bray-Curtis and
every set-overlap coefficient are sums of per-feature terms and need division by the
width itself** (measured 1.000). Chebyshev takes a maximum rather than a sum and grows
only logarithmically with width, so both divisors over-correct it; leave it unscaled.
Using one divisor throughout makes "equal-block" mean two different things depending on
which coefficient occupies the cell. `block_divisor` in
`src/kirby/similarity/preprocessing.py` returns the right one per coefficient.

### Axis C — what counts as good while picking (the optimisation target)

**Correction to what I told you earlier: only two of these are selectable today.**

| Target | Selectable now? | Notes |
|---|---|---|
| distance vs activity difference | **yes** (`similarity_sac`) | this is what the allocator uses under *both* similarity settings |
| activity read-across | **partly** (`similarity`) | scores the assembled hybrid on the held-out fold, but the allocator still ranks representations by the other one |
| model accuracy (control) | yes | |
| structural overlap | **no** | no objective constant; computed only on the outer fold; the inner-fold index-alignment problem is why |
| blends | **no** | need a scoring hook that does not exist |

So the blend table I gave you is not deliverable as stated. The honest version:

**Phase 1 (runnable after P1–P3):** three targets — distance-vs-activity-difference,
read-across, model accuracy. That already answers "does selecting for comparison beat
selecting for prediction", which is the study's core question.

**Phase 2 (needs new code):** make the allocator score a *weighted sum* of targets, and
make structural overlap available as a target. Two pieces of work: (i) solve the
pair-index alignment so structural fidelity can be scored on an inner fold, (ii) let the
allocator take a blend rather than a single statistic. Then the blends become:

| # | Blend | Question |
|---|---|---|
| 1 | read-across + structure | the two yardsticks, balanced |
| 2 | read-across + accuracy | does keeping predictive power cost similarity |
| 3 | structure + accuracy | same question from the structure side |
| 4 | read-across + distance-vs-activity-difference | global vs local activity coherence |
| 5 | structure + distance-vs-activity-difference | structure vs global activity |
| 6 | all three (equal thirds) | the everything arm |

**Weightings: 0.25 / 0.5 / 0.75 on every blend, all four/five pairs.** The weighting is a
scalar in the score — it costs one multiply, and the whole point of the blend axis is to
find where the trade-off sits. Fixing it at half-and-half assumes the answer.

### Axis D — budget

**10, 25, 50, 100, 250, 500, 1000, 2000** — the full budget list.

Two reasons not to stop at 250, despite the theory pointing small:

- **A capped similarity arm cannot be fairly compared to the prediction control.** The
  prediction hybrids in this framework run to 2000. If the similarity arm stops at 250
  and loses at the top end, that is an artefact of the cap, not a result.
- The argument for small budgets is a *prediction* to be tested, not an assumption to
  build in. (The prediction: a distance sums the difference over every kept column, so
  a useless column adds noise to every comparison, while a tree ignores it for free —
  signal-to-noise degrading roughly as √(useful ÷ total), which has no analogue on the
  prediction side.) If the optimum really is at 25, the sweep will show it, and the
  large-budget cells are the evidence that it is a real turnover rather than a boundary.

Note the existing dimension experiment already sweeps 10–500 asking a related question.
Reuse its machinery rather than writing a second sweep, but **do not inherit its 500
ceiling** — extend it.

**Watch out:** budgets are still filtered by a per-model cap even though no model is fit
(see 2.8). Pin to an uncapped model name or the top of this range silently disappears
for some arms and the comparison stops being like-for-like.

### Axis E — the comparison function

This is one of the axes the user named explicitly. **Run it properly — the only real
constraint is which coefficient is *defined* on which object.**

| Space | Coefficients that are defined there |
|---|---|
| standardised stacked blocks (the hybrids) | Euclidean, Manhattan, cosine, correlation, Canberra, Chebyshev, fractional Minkowski. ⚠ **CORRECTED 2026-08-26: Bray-Curtis and Mahalanobis do NOT belong here.** Bray-Curtis needs features >= 0; on z-scored features its median is exactly 1.00, over half of pairs score above 1, and it stops being monotone in how far apart two rows are. `compute_distance_matrix` now refuses that cell. Mahalanobis is fine unweighted but has no per-feature weighting that survives its own covariance re-estimation (weighted and unweighted differ by 3.6e-15), so it cannot be an arm of the weighting axis. `f18`'s own `needs_nonneg=True` flag already contradicted this row. |
| raw count / on-off fingerprints (baselines + the count-space arm) | Tanimoto (Jaccard), Dice, Cosine, Soergel, plus the weighted forms |

The shared distance function already accepts a metric name and forwards it, so the
standardised-space list is available today at no build cost. **The weighted set
coefficients do not exist and need building** (§4 Axis F).

Two real constraints to respect, both correctness not cost:

- Set coefficients on standardised continuous columns are **undefined**, not merely
  weak. Do not run that cell; log it as not-applicable rather than leaving a gap.
- Selection is hardcoded Euclidean in three engine places. A scoring-time-only swap
  means features chosen under one distance and graded under another — which is a
  legitimate arm, but it must be **labelled** as such, and the matched arm (chosen and
  graded under the same coefficient) needs those three sites parameterised. Build that;
  it is the version that answers the question.

Expectation to state and test, not to assume: the literature reports the coefficient is
low-stakes on standard fingerprints (Bajusz 2015; Todeschini 2012). Nobody has tested it
on a *selected multi-block* space, which is exactly what this study builds.

### Axis F — importance weighting (user addition, 2026-08-24)

Weight each column by how important it is, so an important column counts for more in
every comparison. **Three places this can enter, not two** — and they cost wildly
different amounts.

| Place | Exists? | Cost |
|---|---|---|
| **F1. During selection** — weight columns while deciding which to keep | no | expensive: changes what gets selected, so the search re-runs |
| **F2. Inside the comparison function** — weighted Tanimoto / Dice / cosine | **weighted set coefficients: still nothing.** UPDATED 2026-08-26: the weighting machinery is built (`fold_weights` in `src/kirby/similarity/preprocessing.py`) and the fault it was hiding is fixed — one `sqrt(w)` was applied before every metric, right for Euclidean and cosine, wrong for Manhattan/Chebyshev/Bray-Curtis (they need w), and an exact no-op for Canberra and Mahalanobis. The ten set coefficients themselves are unbuilt — see `COEFFICIENT_BUILD_PROMPT.md`. | ⚠ NOT always "rescale an already-built matrix": Canberra and correlation cannot be weighted by any rescale at all and need the weight inside the formula |
| **F3. At final scoring** — weight the columns before the downstream job | **already exists** in s02, s08 and the cliff/neighbour/domain/noise-bridge experiments; n12 is a dedicated ablation of exactly this | cheap |

**What already exists.** The shared distance function takes per-column weights and
applies the square root of each, after standardising. The similarity suite already runs
weighted arms alongside unweighted ones, and the noise suite has a whole experiment
(n12) asking whether importance-scaling before nearest-neighbour search improves noise
detection and whether it is redundant with hybridising. **So F3 is not new work — it is
existing arms that need to be carried through to the new representations.** F1 and F2
are the genuinely new pieces.

**What does not exist: weighted set coefficients.** There is no weighted Tanimoto or
weighted Dice anywhere in the repo — only weighted Euclidean. Both are well-defined
(weight each term of the intersection and union sums) but they need raw count or binary
columns, so **F2 is coupled to Axis E**: weighted set coefficients only apply to the
raw-count side experiment, not to the standardised stacked blocks.

**Sub-axis: where the weights come from.** Today they always come from a label-fitted
model. For this study there is a second source worth testing, and it matters:

| Source | Uses labels? | Note |
|---|---|---|
| model importance (gain / tree-shap / deeplift) | **yes** | what every existing weighted arm uses |
| geometry-derived — score each column by how well its differences track shared substructure | **no** | already implemented as the fine-grained column ranker in the July code |

This is not a detail. The study's anti-circularity argument is that the metric is not
built from labels. Every weighted arm today quietly breaks that. A label-free weight
source is the version that keeps the argument intact — and it is already written.

**Two landmines, both verified in the code.**

1. **Order of operations.** Weighting *before* standardising is a mathematical no-op —
   a per-column standardiser exactly cancels a per-column multiply. This has already bitten
   this repo once: the n12 docstring records that the entire importance-weighting ablation
   silently produced *identical geometry* for weighted and unweighted arms until it was
   fixed. Weights must be applied **after** standardising. Assert it in the smoke gate.
2. **Two incompatible normalisation conventions are live right now.** The shared
   nearest-neighbour helper and n02 normalise the weights to sum to 1; n12 normalises
   them to average 1. These differ by a factor of the column count. Rank-based scores
   (Spearman, neighbour order) are unaffected by a uniform rescale, but anything with an
   absolute distance threshold is not — and the two conventions make numbers
   non-comparable between experiments. **Pick one, apply it everywhere, and state it.**
   Average-1 is the better choice: it keeps columns near their standardised scale instead
   of shrinking every distance as the budget grows.

**Full sweep.** Weight source {model importance × its three methods, geometry-derived,
none} × strength {none, sqrt, linear, squared, top-k} at both placements F2 and F3, over
**every arm**, not just the winner — F2 and F3 rescale an already-built matrix, so they
add multiplications, not searches, and restricting them to the winner would mean the
weighting result is conditional on a choice made without weighting.

F1 (weighting during selection) is the one that genuinely re-runs the search. Run it
across the **full budget list on both families**; it is the arm that tests whether
weighting has to be inside the selection to pay off, and answering it at one budget would
not settle it.

**Importance method is its own axis.** The framework has three (gain, tree-shap,
deeplift). Locally only one runs — the default path segfaults on this laptop — so **the
local arm is silently pinned to one method**. On the server, sweep all three. Note the
earlier measurement that the importance axis is near-noise for *prediction* (0.0001–0.012);
whether that holds for a distance is exactly what is untested here.

---

## 5. Which allocation strategies — the recommendation

You asked what I think. Here is the reasoning, then the set.

**Why the prediction answer should not transfer.** Your prediction study measured the
allocation axis as a weak knob (0.009–0.018 swing) and found complementarity selectors
lose 8/8, because the tree absorbs redundancy downstream. **For a distance there is no
downstream absorber.** Two near-duplicate representations each given 200 columns make
the shared direction count twice, silently tilting the metric. And a useless column is
free for a tree but corrupts every pairwise comparison. So: concentration should matter
*more* for similarity than for prediction, and redundancy penalties are the family that
should flip sign. If the axis turns out to be weak here too, that is itself the headline
— it would mean the metric is dominated by one block regardless of the allocator.

**A scale problem to MEASURE, not to prune around.** Under a geometry objective the
per-representation score is a Spearman correlation — near zero, often negative. The
sanitiser shifts every score by the minimum, so scores {0.30, 0.05, −0.10} become
{0.40, 0.15, 0.00}: the worst representation is forced to *exactly zero budget*, and the
best one's share moves from 86% to 73%. Every scheme that treats the score as a
proportion (linear, the power laws, the softmaxes, thresholded, score-gap) inherits that.

My first draft cut those schemes on this argument. **That was wrong** — it is a
prediction, not a measurement, and it would have removed most of the weighting families
from a study whose subject is weighting. **Run them all.** Alongside, record a cheap
diagnostic per fold: how much each representation's share moves when the worst
representation is dropped from the pool. If the ratio-scale schemes really are measuring
the sanitiser, that diagnostic shows it *and* gives a publishable methods finding. If
they are not, nothing was lost.

⛔ **GREEDY IS OUT — this section previously argued for reviving it. That argument is
WITHDRAWN, 2026-08-26, by the owner's decision.** The greedy family (`greedy`,
`greedy_forced_s*`, `greedy_fill_s*`, `dominant_greedy`, `dom_greedy_f*`) was closed on
2026-04-09 and re-confirmed on 2026-08-25 and 2026-08-26. It is not an arm of this study
at any budget, under any objective. Do not re-open it, do not repair its discarded
step_size/patience settings, and do not cite the geometry-objective cost argument as a
reason to retry it — that argument was made here and rejected. The scope rule's
"fix it rather than drop it" clause does **not** apply to an arm the owner has cut.
See the ⛔ block above `NAMED_ALLOCATIONS` in `src/kirby/hybrid.py`.

**The neutral floor — BUILT 2026-08-26.** `fixed` was the equal split all along, but it
gave every representation a flat 100 features and ignored `budget`, which is why its
apparent f01 win was really a bigger-budget artefact. It now divides `budget` evenly
across the representations (`allocate_equal_split` in `src/kirby/hybrid.py`), so it is
the neutral reference point at the same total budget. The old budget-blind behaviour is
still reachable by passing `n_per_rep` explicitly. The text below is superseded:
no reference point. Run `fixed` too, labelled budget-blind, since it is a real historical
arm — but it cannot serve as the floor.

### The set — everything the registry can do, organised

**Mechanism (the branches that are in scope):**
per-representation quota · global pool · equal split at budget (`fixed`, fixed
2026-08-26 — it now divides `budget` across the representations instead of spending a
flat 100 each). Greedy and dominant-greedy are CUT and are not arms.

**Concentration / weighting (the full family, not a subset):**
top-N {2, 3, 5, all} · min-floor {the floor values in the registry} · rank-based
{linear, exponential decay} · ratio-scale {linear, power 2, power 3, the softmax
temperatures, thresholded/score-gap} · the redundancy-penalised {entropy} family.

Two things to state rather than silently collapse: thresholded and score-gap are the
**same knob** under two names (alpha = 1 − gap), so run one and note the alias; and
min-floor never drops a representation, so it is the spread arm, not a selector.

**One new weighting scheme worth building** — a redundancy penalty defined on the
*distance*, not on the raw matrix. The existing entropy scheme penalises a representation
by its correlation with an already-accepted one, but measures that between the per-sample
lengths of the **raw, unscaled** matrices — which for molecular data is close to a
molecule-size proxy, so the penalty fires almost uniformly. The right quantity is the
correlation between two representations' **pairwise-distance vectors** over the sampled
pairs, which is exactly what gets double-counted in the concatenated metric. Cheap: a
handful of distance vectors and their correlations on 5,000 points. Run it beside the
existing entropy scheme so the study can say whether the redundancy *idea* or that
specific *implementation* was the problem.

**Crossed with:** block normalisation {raw, equal-block} × the full budget list × both
column-ranking sources (label-supervised importance, geometry-derived) × all three
importance methods on the server.

**Pinning that is legitimate:** the downstream model and internal evaluator are
**provably degenerate** under a geometry objective — the allocator returns before reading
either. Pin them to one uncapped name. **But verify it rather than trusting it:** assert
in the smoke gate that two different model names give byte-identical allocations. If that
assertion fails, the degeneracy claim is wrong and the axis reopens.

**Three headline numbers:** the concentration gap (top-2 minus min-floor), the block-weight
gap (raw minus equal-block), and the mechanism gap (global pool minus quota). If the
block-weight gap swamps the others, the honest conclusion is that for similarity the
weighting matters more than the allocator — a stronger result than any allocator ranking.

**One thing to fix before global-pool can answer its own question.** The global-pool
allocator sizes each representation's slice by geometry, but then picks the final columns
with lightgbm gain against the labels. So if it wins you cannot tell whether pooling
helped or whether a supervised ranker beat the quotas. The fix is to register the July
study's label-free column ranker (score each column by how well its differences track
shared substructure) as a selectable importance method, and run the contrast both ways.

**Nothing is excluded on cost.** Three things are handled rather than dropped, per the
scope rule: thresholded and score-gap are run **once** with the alias
noted, because they are arithmetically the same knob and running both would be a
duplicate cell, not extra coverage; and `fixed` runs, labelled budget-blind, since a
budget-blind arm cannot be read as a point on the budget axis.

The only genuine exclusions are cells where the quantity is **undefined** — a set-overlap
coefficient on standardised continuous columns (§4 Axis E) — and those get logged as
not-applicable rather than silently omitted.

**Three allocators exist in the code but are unreachable** (two-stage, stability
selection, greedy-feature): no registry entry, no dispatch branch. Decision needed
(§12): wire them in — they are written and this is the study that would test them — or
record them as dead code. Do not leave them undecided a third time.

---

## 6. Baselines and controls

Thin baselines make the whole grid unreadable. Required:

- **Count fingerprint** — the July study's own strong baseline, what every July number
  was measured against. Without it this study cannot reproduce or contradict July.
- **ECFP4 + Tanimoto** — the field standard.
- **Each individual representation**, reported separately. Never an average.
- **Random columns at each budget** — without this the budget axis cannot separate
  "these columns" from "this many columns".
- **Each single selected block on its own** (column-dropped descriptors; fine-grained
  ranked columns) — already registered as controls in the molecular runner.
- **The prediction-tuned hybrid used as a similarity space** — this was locked on
  21 Aug as one of three side-by-side arms. Dropping it would reverse a signed-off
  decision, so it stays.

**Size confound control — a gate, not an extra.** MCS distance falls as molecules differ
in atom count, and Euclidean distance on stacked count and descriptor blocks rises with
molecule size. So both axes of the July structural result could be driven by size alone.
The roadmap flags this as open decision 5 and it was never run. **Run a matched
heavy-atom-count pair control before the head-to-head**, or the whole grid may be
optimising for molecular size.

---

## 7. What it gets judged on

Two grading yardsticks, always both, never merged: **activity read-across**
(scaffold-grouped nearest-neighbour Spearman) and **structural fidelity** (distance vs
shared substructure).

Then the downstream jobs. Score every arm on all four deliverable metrics at the
held-out fold, and report the rank correlation between the allocator's own selection
score and each deliverable — because the allocator optimises a *global* statistic while
retrieval, cliffs and label cleaning are all *local*. If that correlation is weak, the
finding is "the selection objective is the wrong knob", and it outranks any allocator
ranking.

| Job | Where |
|---|---|
| per-representation similarity baseline | s01 |
| hybrid as a similarity space | s02 |
| activity cliffs | s04 |
| nearest-neighbour prediction | s05 |
| applicability domain | s06 |
| noise/cliff false-positive bridge | s07 |
| best dimension for similarity vs prediction | s08 |
| clustering + retrieval | s13 |
| **synthetic-noise remediation** | n07, n11 |
| **importance-weighted distance ablation** (Axis F3) | n12 — already asks exactly this question; carry the new representations through it rather than rebuilding |
| **real-noise ground truth (replicate disagreement)** | **n06 — the study's stated novelty, currently unmentioned anywhere** |
| feature concentration | s03 — carries a full-data descriptive-only caveat in its own docstring; run it, report it as descriptive |
| importance downstream | s10 |
| Bayesian optimisation in the space | s11 |
| cross-domain cliffs | s12 |
| cliff-stratified error, sample weighting, remediation, noisy-label survey, noise distributions, cliff landscape, clean-vs-noisy blueprint | n01, n02, n07, n09, n13, n14, n15 |

**The default is the whole of both suites.** s01–s13 and n01–n15 minus the ones the
scope gates already skip for size or domain (§2.9) — and those skips must be **listed in
the result table**, not left as blanks, so nobody reads a gap as a null result.

**On the existing noise-cleaning results:** `similarity_noise_cleaning.py` re-runs the
column selection on the full pair set and then reports structural fidelity on that same
pair set. It carries the in-sample circularity that the honest re-run was written to
fix — and even that fix split the *pair list* 50/50, so the same molecule still appears
on both sides. **Mark those numbers as in-sample. They need a molecule-level held-out
split before they can be cited.**

---

## 8. Datasets

**Default: the full molecular roster**, not a hand-picked three. This is a server study
and the whole point is whether the answer is stable across datasets — a two-dataset
result cannot show that, and the repo's own history is full of preferences that flipped
between datasets.

| Dataset | Why | State |
|---|---|---|
| herg_ki | real measured pKi, the July anchor | ready |
| logd | real measured LogD, the second July anchor | ready |
| caco2 | in the roster | **will not load locally — open since the handoff** |
| adme_hlm, adme_rlm, adme_mdr1 | in the roster, real assay data | adme_hlm won't load locally; the other two untested for this |
| adme_solubility | already has cached structural pairs | ready |
| **qm9 (4k local / 10k server)** | computed property, **no measurement noise** — isolates structure | **needs work — §8.1** |
| an antibody set (e.g. flab_thermo_tresanco) | tests whether the method is molecule-specific | optional; the column-dropping method already transferred to antibodies in July |
| buchwald_hartwig | reaction data, already has a rep cache on the drive | optional |

Local runs are limited to what loads and to one importance method (§11 item 2); the
server run should take the full roster. Trim only if a dataset fails a stated gate
(too few scaffold groups, no structural pairs buildable, above the size skip), and
**record which gate**.

### 8.1 QM9 — what it buys and what it costs

**What it buys.** The target is the computed HOMO–LUMO gap. That is a DFT number, not an
assay reading, so **there is no measurement noise in it**. That makes QM9 a clean
structure-property landscape: it isolates the structural axis with the activity-noise
axis switched off, and it is a genuine methodology-first contrast against two noisy
bioactivity sets. It also means QM9 **cannot** support the real-noise arm — only
synthetic injection. Say that explicitly rather than quietly running it.

**What it costs.** Six concrete obstacles, all verified:

1. **No representation cache exists for QM9** — not on this laptop, not on the seagate
   cache root. It has 10 representations registered. This is the caching job you asked
   for and it is a real build, not a copy.
2. **The sample count is hardcoded at 10,000** at the call site, and there is no size
   flag anywhere in the path from the command line down to the loader. The loader itself
   supports a sample count; nothing exposes it. **Needs a decision (§11): add a size
   argument, or register a 4k subset as its own dataset name.**
3. Useful property: with the fixed seed the 4,000-molecule draw is a **strict subset** of
   the 10,000-molecule draw, so a 4k local run and a 10k server run are nested rather
   than disjoint.
4. **The QM9 loader is the one molecular loader that does not check the external drive.**
   It looks only under the cache root, which has no QM9 directory — so with the cache
   pointed at the drive it would try to download rather than use the staged copy
   (`/Volumes/seagate/QM9`, 235MB of raw structures). One symlink fixes it.
5. **A real bug in the fallback path.** If the processed file lacks per-molecule
   structures, the loader parses the raw structure file instead — and indexes it by raw
   position while indexing labels by filtered position. Two filters have already removed
   ~4,500 molecules by then, so **every structure would be paired with the wrong
   molecule's label**. Whether it fires depends on the processed file, which cannot be
   checked locally (see 6). This must be verified before any QM9 number is trusted.
6. **The graph library will not import on this laptop** — a compiled-extension mismatch
   the setup script already warns about. So the processed file cannot be opened locally
   and item 5 cannot be settled here.
7. Scaffold grouping works: ~1,390 scaffold groups in a 4,000-molecule sample, but the
   largest single group is the *empty* scaffold (fully acyclic molecules) at ~11%. Fine
   for grouped folds; worth stating.

**Caching to the drive:** the mechanism is the cache-root environment variable pointed at
`/Volumes/seagate/kirby_cache`, which already holds representation caches for four
datasets. Building QM9's 10 representations there makes them reusable and is worth doing
regardless of this study. Items 4–6 are prerequisites.

### 8.2 MCS pairs cannot be reused from the July study

The July loader caps logd to a ~1,400-molecule scaffold subsample, and the cached pair
file for logd tops out at row index 1,426 — while the framework's logd is 5,039 rows. All
three cached pair files are indexed in *study* space. **Reusing them against
framework-loaded data pairs each structural distance with the wrong molecule.** Every
pair set must be rebuilt through the framework's content-hashed pair cache, which is
keyed on the exact molecule list.

---

## 9. Order of work

0. **Commit** the untracked similarity package and the uncommitted engine diff. Decide
   the cache wipe (the `speed` removal invalidates existing stored rows).
1. **Prerequisites P1–P6.** Geometry branch in the factory-representation loop; forward
   the objective into every allocator call site; thread the objective through the six
   plumbing functions and key the in-memory store by it; forward the blueprint objective
   and the real groups/model/metric into the per-fold rebuilder; give column-dropped runs
   their own cache namespace.
2. **Smoke gate.** Two folds, assert the number of scored folds equals the number
   requested for every configuration. Silent fold loss is the failure mode most likely to
   corrupt this study. Also assert groups are non-null (see 2.9).
3. **Size confound control** (§6). Gate.
4. **Rebuild MCS pairs** through the framework cache for every dataset used.
5. **Register Family B** so the suite can request it (P9, P10).
6. **Build the missing pieces** so the grid is complete rather than trimmed. State as of
   2026-08-26:
   - ~~equal-quota allocator~~ **DONE.** `fixed` is the equal split at budget now
     (`allocate_equal_split`, `src/kirby/hybrid.py`).
   - ~~distance-based redundancy penalty~~ **BUILT, UNRUN.** `pairdist_redundancy`, six
     registered strengths, f21 submitted.
   - ~~greedy step/patience fix, greedy budget-fill objective fix~~ **WONTFIX.** Greedy
     is cut. Repairing either would resurrect cut arms.
   - **weighted set coefficients — HALF DONE.** The weighting machinery and the
     preprocessing each family needs are built and tested. The ten formulas are not
     written: `COEFFICIENT_BUILD_PROMPT.md`.
   - **still open:** the label-free column ranker as a selectable importance method, and
     the coefficient parameterised at the three selection sites.
7. **Head-to-head — the full grid: Axis A × B × C-phase-1 × D × E × F, full roster.**
   **Decision point: pick the winning family.** Report per-cell win counts on every
   deliverable metric, with a stated paired test at the fold level. Not a mean rank, not
   an unqualified point difference. Log every skipped cell and why.
8. **Axis C phase 2** — structural target + the blends with their weightings. New code
   (§4 Axis C). Smoke-assert the weighting order of operations first (§4 Axis F).
9. Downstream jobs across both suites with the winner, including the real-noise arm.
10. Cleanup (§10).

**Only step 7 is large.** If it needs splitting, split it by **dataset**, so every
dataset gets the complete grid — never by axis, which would leave holes that look like
results. Log any split in the output.

---

## 10. Cleanup — ordered so nothing breaks

Each deletion follows its replacement, not before.

| Item | Note |
|---|---|
| `src/kirby/similarity/` | **Commit first.** It is the source of Family B and the only held-out-validated column-dropping code. Delete only once its parts live in the main suite. |
| `tests/experiments/noise/_similarity_hybrid_space.py` | **Has two live callers** (n11, n13) and holds one thing that exists nowhere else: canonical-structure alignment of the study's cached rows onto an experiment's molecules with an all-or-nothing coverage gate. That gate must be **reimplemented, not just deleted**. |
| `tests/experiments/similarity_optimized_hybrid.py` | Standalone; its objective switch is the thing being moved into the suite. |
| Duplicate implementations | Corrected counts: structural fidelity **13 copies** under tests plus one in the engine; distance-vs-activity-difference 4; shared-substructure distance 2; read-across 1 shared plus 3 independent re-implementations. Roughly three times what I previously stated. |

**Keep:** the s01–s13 suite; the July study folder as the record of what was measured.

---

## 11. Inherited open items — carried, not dropped

From the handoff, the 21-Aug plan and the roadmap. Each needs a mark: prerequisite,
deferred, or cut.

1. **caco2 and adme_hlm will not load locally.** Two of your four stated local datasets.
   The dataset question cannot be fully answered until this is fixed. → prerequisite?
2. **The engine's similarity change has never been verified on the server.** Locally it
   runs only with lightgbm importance/evaluator/model because the default path segfaults
   — so any local run is **pinned to one importance method, which is a hidden axis**.
   → prerequisite.
3. **The column-blacklist parameter exists and has zero callers**; the saboteur finder
   has produced no result file and its demo notebook has never been executed.
4. **"Characterisation experiment home = noise"** was locked on 21 Aug. This plan spreads
   the work across both suites. → **that is a decision reversal and needs your sign-off.**
5. **Roadmap avenues 1, 2 and 4** (learned/triplet metrics, n-ary similarity, the
   alternative-fingerprint grid) and handoff item D (learned representations tested
   fairly) — all untried. You have said learned representations wait for the server.
   → deferred.
6. **Significance testing.** The roadmap asks for a paired test / Bradley-Terry; nothing
   in the current plan states one. → must be stated before step 6.
7. **The nine block combinations from this chat were never scored** — both runs were
   stopped. They are Family B's candidate list.

---

## 12. Open decisions — need your answer before sign-off

1. **QM9 route:** add a size argument to the loader path, or register a 4k subset as its
   own dataset name? (Second is cleaner for caching and for the result database.)
2. **QM9 verification:** the structure/label misalignment (§8.1 item 5) cannot be checked
   locally. Verify on the server before use, or run QM9 in a later pass?
3. **Dataset roster:** the full molecular set in §8, or a stated subset? If a subset,
   which gate excludes the rest.
4. **Budget ceiling:** the full list to 2000, or a stated cap? A cap makes the
   comparison against the prediction control unfair (§4 Axis D) — if you want one, it
   should apply to both arms.
5. **Axis E build:** parameterise the coefficient at the three selection sites (so
   features are chosen and graded under the same one), or run scoring-time-only swaps and
   label them? Recommend building it — otherwise the axis answers a different question.
6. **Unreachable allocators:** two-stage, stability-selection and greedy-feature are
   written but not wired to the dispatch. Wire them in for this study, or record them as
   dead code? They have been sitting undecided through two plans.
7. **Blends:** accept the phase-1 / phase-2 split, given that structure is not a
   selectable target today?
8. **Item 4 in §11** — the "home = noise" reversal. Confirm or restore.
9. **Cache wipe** triggered by the `speed` removal — wipe the result database, or pin
   the change?
10. **Axis F scope:** confirm the full weighting sweep on every arm (not just the
    winner), and the weight-normalisation convention (average-1 recommended) so the two
    live conventions stop disagreeing.
11. **Compute:** confirm the grid runs unsplit, or split by dataset (never by axis). This
    plan assumes overnight server compute is free and that under-exploring costs more
    than a long run.
---

## Changelog

- 2026-08-26 — corrected in four places against measured code, and the greedy revival
  argument in §5 was WITHDRAWN by the owner's decision. Corrections: §2.5 (the reason
  structural fidelity could not be scored inside the search is fixed), Axis B (√width is
  the right divisor only for the Euclidean family; the Manhattan family needs the width
  itself), Axis E (Bray-Curtis and Mahalanobis do not belong on standardised blocks),
  Axis F2 (Canberra and correlation cannot be weighted by any rescale). §5's "the neutral
  floor does not exist" is superseded — `fixed` is the equal split at budget now. The
  build order in §9 item 6 now records which pieces are done. Still a proposal; the grid
  itself has not been run.
- 2026-08-24 — created; rewritten after a full read of the noise suite, the allocation
  registry, the engine's geometry paths, QM9, and the duplicate inventory; then **scope
  restored** after the user challenged excessive pruning. The first draft cut the
  ratio-scale weighting families, most of the allocation registry, the budget ceiling,
  the blend weightings, the comparison-function axis, and reduced several sweeps to
  "the winner only" — all on cost or on untested theory. Those cuts are reversed and the
  scope rule at the head of §4 now governs. Proposal only; nothing run.
