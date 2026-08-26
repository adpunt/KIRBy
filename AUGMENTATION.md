# Augmentation — the single source

**This is the only augmentation document.** It replaces and supersedes four files, all deleted
on 2026-08-25 after being read in full and folded in here:
`AUGMENTATION_STAGE_DESIGN_2026_08_19.md`, `AUGMENTATION_STAGE_DESIGN_2026_08_21.md`,
`AUGMENTATION_FINDINGS_AND_HANDOFF_2026_08_24.md`, `AUGMENTATION_COMPLAINT_LEDGER_2026_08_25.md`.
The brief itself stays where it is, in `HYBRID_REBUILD_ROADMAP_2026_07_15.md` Phase 4.4 and
`HYBRID_REBUILD_PROMPTS_2026_07_16.md` Chat 6, because both cover the whole rebuild and not just
this workstream.

Every code claim in §3 and §4 was re-read in the source on 2026-08-25 (§4.1 and §4.1b re-verified after the fix) and is marked
**verified** or **not verified**. Every number in §5 is either recomputed from a result file
still on disk or marked as unbacked. Nothing here is an opinion presented as a measurement.

Do not add a second augmentation document. Update this one.

---

## 1. The question

From the roadmap, Phase 4.4, in your words:

> Make the augmentation-level hybrid search a **proper, integrated stage of the pipeline** —
> clearly defined, cached, and searched upfront, not bolted on. Cover two use cases:
> **(A)** small feature sets that don't deserve to be a standalone base representation, tried
> directly in the hybrid; **(B)** the single best representation plus either (a) augmented
> features or (b) the **top-5 individual features from each of the other representations**.
> Together with base-representation allocation the hybrid then covers **Scenario 1**, signal
> spread across several representations, and **Scenario 2**, one dominant representation
> slightly improved.

**The question is not "does augmenting work".** Assume it does. The question is
**which specific way of doing it works**, across the dataset roster. If exactly one way works,
that is a complete answer. A method losing on 20 of 30 datasets is not a reason to drop it if it
is the best method on the other 10.

The four stages of this workstream, in order:

1. Work out what is broken and fix it. **← we are here.**
2. Define the experimental grid: which augmented representations, which selection method, which
   allocation method, and whether extras go onto a base representation or onto a hybrid.
3. Run that grid on 10 datasets on the server.
4. Pick the single best method (or the smallest set of methods that covers the roster) and wire
   it into the hybrid pipeline.

---

## 2. Decisions already taken — do not re-litigate

| date | decision | status today |
|---|---|---|
| 08-20 | Supplement size is a **percentage of the budget**, the percentage found by sweeping | **built in a scratch script only** (§5.1). The engine has no percentage sizing; the shipping path uses a hard-coded ladder of absolute counts. |
| 08-20 | Keep rule = **rank once, cut at the cap.** No accept/reject loop, no inner holdout | partly. The loop is gone, but the cut is a fixed ladder (0/5/10/20/40) and each rung is still scored on one inner holdout. **Conflicts with the 08-21 post-mortem — see §7.1.** |
| 08-20 | Fix both cache defects; the cache is being rewritten anyway | the per-run collision is fixed. The recipe-rebuild gap is not, and cannot be without new blueprint fields. |
| 08-21 | Include the multi-representation hybrid as a base, not only the best single representation | done, arm exists. |
| 08-21 | Editing `c21` is inside the boundary; measure the defects before fixing them | done. |
| 08-21 | Top-5 per donor, not 64 | done, via the per-source cap. |
| 08-23 | Drop the three full-width representations from logD's augmentation list | done — logD's list is `maccs`, `pharmacophore3d`, `usrcat` (276 features, was 3,604). |
| 08-24 | Fix the random splitting in the honest re-score | done in the honest evaluator, **inert everywhere else** (§4.1). |
| 08-24 | Remove the dormant routine | **not done.** Both candidates turned out to be reachable or referenced (§7.4). |
| 08-24 | Keep the base-aware ranker as the default | done. |
| 08-25 | Nothing runs locally for more than 3 hours; route it to the server instead of cutting axes | standing. |
| 08-25 | Never cut axes, arms, budgets or datasets for compute. Cuts need a stated reason: degenerate, broken, or undefined | standing. |

Two standing assumptions, stated so they can be overturned deliberately: the augmentation pool
is **exempt from the solo-score representation cut** (judged only by marginal gain on top of a
base, never by how well it predicts alone), and lightgbm is the default downstream model for
augmentation runs with a second model as a robustness check.

---

## 3. What the augmentation stage is today — verified 2026-08-25

### 3.1 Where the pool comes from

The per-dataset pool is `DATASET_REPS[<dataset>]['augmentation']` in
`tests/experiments/complexity/complexity_helpers.py`. **`src/kirby/` never imports it.** Nothing
about a representation makes it "an augmentation" — only which list its name sits in; `maccs` is
a base representation on one dataset and an augmentation representation on another.

`load_data` filters to the base pool when no names are given and hard-sets `augmentations: None`
on every return branch, so augmentation representations are present **only if requested by
name**. Seven places in the tree each roll their own load-and-pass block. This is how attempt 1
silently ran with an empty pool. **Verified.**

### 3.2 Augmentation is not searched

The search grid is a six-way product over representation set × allocation × model × budget ×
importance method × internal evaluator. **Every augmentation parameter — the strategy, the
ranker, the per-source cap, the threshold, the budget — sits outside the grid** and is spliced
identically into every configuration. Varying augmentation means issuing separate outer calls,
which is what the precompute driver does, emitting one condition per (pool, strategy) pair. So
"searched upfront" is false today. **Verified.**

### 3.3 The eleven paths that decide which extra features get added

Three run in the shipping pipeline. Exactly one has been exercised since the 24 August rebuild,
and only in scratch scripts. **No stored pipeline result exists under the code as it stands.**

| # | path | setting | live? | state |
|---|---|---|---|---|
| 1 | Rank the pooled candidates once, then choose a block size from a fixed ladder (0, 5, 10, 20, 40); ties keep nothing. Extras are appended **outside** the budget | `uncapped` — **the default** | yes | rebuilt 08-24. §4.5 |
| 2 | Same ladder and same tie rule, but the base is cut to (budget − k) columns so extras must **displace** base features and the total stays inside the budget | `capped` | yes, for n ≤ 250 | **rebuilt 08-25.** §4.2 |
| 3 | Keep or drop each donor's top-20 block whole | `greedy_ablation` | no — framework experiments only | order-dependent, base-blind, ratchets |
| 4 | Concatenate every donor's top 20, no test at all | `all` | no — one archived script | **ranks against the full label vector.** §4.3 |
| 5 | Off | `none` | yes | the control arm; no defects of its own |
| 6 | The old add-one-at-a-time loop that ratcheted | (former `uncapped`) | **removed 08-24** | **produced every augmentation result previously on disk.** Those numbers came from code that no longer exists. |
| 7 | ~~A second fill-then-replace routine, ~290 lines~~ | — | **deleted 08-25** | It was never a lost method: on 24 March its branch was replaced by `capped` and `uncapped`, which are the two halves it was split into. It was the superseded ancestor, left in the file. |
| 8 | The silent fall-through | any unrecognised string | latent | §4.4 |
| 9 | Auto-demotion: when a top-k allocation gives a base representation zero budget, its top 20 features are injected into the extras pool | automatic | yes, on every default augmented search | §4.6 |
| 10 | Budget-floor top-up before augmentation | — | no | reachable only from three unlisted allocation names |
| 11 | A fourth implementation living inside a framework experiment, with its own private copy of the code | — | out of tree | **the only augmentation mechanism with real swept results on disk** (1,152 configurations, 6 datasets). Those results are the origin of the "these knobs don't matter" claim in the project record — and they were measured with an in-sample residual ranker that the engine's own code now describes as useless, so they do not transfer. |

There is also a wrapper written for exactly use case B — best representation plus features from
the others. **Zero callers, ever.** **Verified.**

### 3.4 The two ranking rules, and why they fight each other

- **`residual_correlation`** — default since 08-24. Fits a cheap forest on the base, takes
  out-of-fold residuals, ranks each candidate by absolute correlation with those residuals. It
  asks "does this add what the base cannot explain".
- **`rf_importance`** — base-blind. Fits a forest on the candidate block alone. It asks "does
  this predict the property by itself". **No caller anywhere sets it**, so this axis has never
  been varied in the shipping engine.

**They pull in opposite directions, and the per-source cap always wins.** The cap trims each
donor to its top-N by solo predictive importance **whatever the ranker setting says**. So on the
default path the cap first discards exactly the features the residual ranker exists to find, then
hands that ranker a shortlist pre-selected for the property it is trying not to reward.
**Verified** — this is the mechanical reason a near-copy of a base feature survives selection.

The residual ranker also has **three silent fallbacks**, each degrading toward the base-blind
rule it replaces: out-of-fold residuals → in-sample residuals → centred labels (which *is* the
base-blind criterion) → all zeros (which leaves the pool in dictionary order). **None of the four
states is recorded anywhere.** A run that fell back is indistinguishable from one that did not.
It also scores each candidate in isolation, so two identical features both rank top and both get
admitted. **Verified.**

### 3.5 The internal evaluator setting has no effect during a search. **Verified.**

When a downstream model is supplied, the allocation scorer fits that model and calls the real
metric; the `internal_evaluator` setting is only read when no model is supplied. The search always
supplies one. So `internal_evaluator` — a five-value search axis (`match`, `rf`, `svm`,
`lightgbm`, `mlp`) that is also part of the stored result's identity — cannot change any searched
result. Five configurations are evaluated where one would do.

This is outside the augmentation workstream and is recorded here only because it was found while
tracing the same code path. It should be checked properly before anyone reads meaning into an
evaluator comparison.

### 3.6 The knobs

Per-source cap (reaches only `uncapped`, never `capped`, but is in the cache identity for both);
per-donor feature count; keep threshold; candidate cut-off (frozen at 50, **no production caller
can change it, and it is not in the cache identity at all**); patience (hard-coded at 10, inert
on the default path — the code says so); which model judges the admission (never reaches it);
the search objective (never forwarded); the internal holdout; the hard-coded ladder.

The default path has **no budget handling at all** — it appends outside the budget. So an
augmentation-off arm and an augmentation-on arm at the same nominal budget are **not the same
width**, and never were.

---

## 4. The defects, ordered by whether they block stage 2

### 4.1 The 24 August fix was inert. **Fixed 2026-08-25 (`8512055`).**

The per-fold rebuild helper gained four optional settings on 24 August — the scaffold grouping,
the real downstream model, the real metric, and the metric direction — and **zero of its 36 call
sites passed any of them**, including the augmentation experiment itself. With the grouping
absent the admission holdout fell back to a fixed-seed 20% random split of a scaffold-grouped
dataset; with the model and metric absent every keep-or-drop decision was judged by an internal
R² score instead of the reported metric.

What was wrong is worth stating precisely, because it is larger than augmentation. **The search
passes all four.** So the search selected features under the scaffold grouping with the real
model and metric, and every rebuild of that same recipe selected them under a random split with a
stand-in score. The rebuild was not a less precise copy of the search — it was a different recipe.

Now: 36 of 36 call sites pass the fold's grouping slice, and 29 of them also pass the matched
downstream model and the real metric. Seven pass the grouping only, each for a stated reason — two
where one hybrid is deliberately shared across several downstream models, four where the arm has
no scikit-learn estimator at all (a neural trainer, an unsupervised arm, a capacity probe, and one
where the estimator class is bound after the call), and one scratch script. The honest evaluator's
rebuild also now receives the model, the metric and the blueprint's objective.

**This changes which features get selected, so every stored augmented result and every stored
greedy-allocated result is superseded.**

### 4.1b A crash on the default allocation. **Found and fixed 2026-08-25, pre-existing.**

Found by testing the change above. When a top-k allocation gives a base representation zero
budget, that representation's best features are automatically demoted into the extras pool —
inside the hybrid builder. The rebuild helper's caller holds its own validation-side extras
dictionary, which has no entry for a representation demoted after it was handed over, so the
rebuild raised. **top-3 is the default allocation and a pool is typically ten representations**,
so on a searched blueprint this fired on most folds, and surfaced as a failed fold wherever the
caller wrapped the call in a try/except — which most do. The builder now records which columns it
demoted and the rebuild takes the same columns from its own validation representations.

### 4.2 `capped` had every original defect. **Rebuilt 2026-08-25 on your instruction.**

It is emitted by the precompute driver as a standard condition for every dataset of 250 rows or
fewer, and thirteen rows of the per-dataset policy table name it. What was wrong:

- The acceptance bar was a running maximum, so early accepts raised it permanently — the exact
  ratchet the default path was rebuilt to remove.
- **The per-source cap was never forwarded to it**, so a 2,560-wide donor competed against a
  40-wide one on feature count alone — while the cap still changed the stored result's identity,
  so two runs that behaved identically were filed as different results.
- Its eviction bookkeeping was wrong after the first swap: the replacement was written into the
  base slot but the base count was decremented, so an already-admitted extra could be evicted
  again as if it were a base feature, and a genuine base feature at the tail became unreachable.
- Under shipping allocations the budget is already spent when it starts, so its fill phase never
  ran — in practice `capped` meant "evict only".
- Every candidate triggered a model refit plus a permutation-importance pass to pick the base
  feature to evict, which is why its own precompute note records that it "has only ever completed
  for n ≤ 242".

It now uses the same shape as the default path, with displacement as the mechanism rather than a
fallback: rank the candidates once, rank the base hybrid's own columns once, then choose k from
the same ladder, keeping the base's best (budget − k) columns plus the top k candidates. Keeping
the base intact is k = 0 and wins ties. The per-source cap is forwarded. Nothing is swapped in
place, so the eviction bookkeeping is gone rather than repaired. Cost drops from roughly fifty
sequential fits plus a permutation pass per candidate to one ranking fit plus five scoring fits.

**The residual weakness is shared with the default path and is not fixed:** each rung is still
scored once on one internal slice, so a near-tie is resolved by luck. That matters more here than
for the default path, because here losing a coin flip throws away real base features. See §7.5.

### 4.3 `all` ranked against the full label vector. **Fixed 2026-08-26.**

Every other path ranks on the inner-training rows. `all` ranked each donor's top 20 on the
**complete** label vector, including the rows it was then scored on. It mattered because `all` is
the do-nothing arm that real selection has to beat, and on logD `all` won (+0.0131 against
+0.0058). A comparison against a dishonest null is worth nothing. `all` now ranks on the
inner-training rows like everything else, so **the logD comparison in §5.2 must be re-run.**

Note also that the name is wrong: it is not "all", it is top-20 per donor.

### 4.4 The dispatcher had no fallback branch. **Fixed 2026-08-26.**

The chain over the strategy names had no final `else` and raised nothing. An unrecognised string
yielded a base-only hybrid while the strategy name was still written into the stored recipe as
though it had been honoured. The allocation dispatcher a few hundred lines earlier does raise and
list its valid names; the augmentation one now matches it.

Checked before fixing: every strategy string set anywhere in the tree was among the names the
dispatcher knows, so no stored result was ever wrong because of it. The hazard was a typo or a
renamed strategy failing in silence.

Related and milder: when the augmentation pool is empty the strategy is normalised to `none`
before the cache key is built. That normalisation is correct and documented. The hazard is
upstream — a run that **forgot to load the pool** is recorded as a perfectly valid
no-augmentation result rather than erroring.

### 4.5 The default path still chooses on one slice. **Verified.**

Ranking is fit on the inner-training rows and each ladder rung is scored on the inner validation
rows, so the ranking itself is not contaminated. But five block sizes are compared on one draw,
and the best of five roughly-tied numbers is taken. That is why noise still gets admitted on some
folds. Ties go to keeping nothing, which is the one thing protecting it.

### 4.6 Auto-demotion. **Partly verified.**

Two real problems: the extras dictionary is **mutated in place**, so injected donors persist
across every model evaluated afterwards in that fold; and the trigger requires augmentation to be
switched on, so an augmentation-off control arm is **not the same recipe minus the extras** — it
also silently loses the demotion.

A previous session claimed this path trains on one set of feature positions and validates on a
different one, producing wrong numbers. **I read that code and could not confirm it** — the
training, validation and full-data slices all use the same positions. Treat that claim as
withdrawn unless someone reproduces it.

### 4.7 Nothing recorded what was kept. **Fixed 2026-08-26.**

The stored recipe carried the strategy name and the per-donor cap. The recipe did not carry the
identity of the surviving features. A winning run could not be explained, and a rebuild could not
be checked against what the search chose.

The saved recipe now carries, per fold, which donor gave which feature indices. Recording it was
free: a stored result's key is a fixed set of settings in a SQLite table, and the recipe itself
rides in that row as a pickled blob. Adding data inside the blob changes no key and forces no
rebuild. Old recipes saved before the field existed still load, with the field left empty.

Worth keeping straight, because the two are easy to confuse. Adding a **setting** that a search
can vary means widening the table's key. The table is created with `CREATE TABLE IF NOT EXISTS`
and has no version marker and no upgrade path — the code's own comment says a stale layout must be
wiped and rebuilt. So a new **setting** means deleting every dataset's results database. A new
recorded **fact** costs nothing.

---

## 5. What has actually been measured

### 5.1 The budget-split run, 21 August — the strongest evidence that exists

Results on disk at `<cache>/aug_budget_split/<dataset>__lightgbm.json`, four datasets, five
group-aware folds, two model seeds. **Recomputed from those files on 2026-08-25**, not copied
from a previous summary.

The design is your 20 August decision: hold the finished feature count fixed at B, and split B
between the base representation's own top features and the supplement at 0/5/10/25/50 percent,
so the supplement must **displace** base features to earn its slots. Three donor sources. Every
cell has a matched-shape random control, and a base-only arm at the same width.

| dataset | cells where extras beat spending the whole budget on the base | and where selection also beat matched random | best cell |
|---|---|---|---|
| logD | **48 of 48** | 47 of 48 | +0.0293 RMSE, 5/5 folds |
| hERG | 42 of 47 | 31 of 47 | +0.0140 RMSE, 4/5 folds |
| buchwald | 14 of 47 | 13 of 47 | +0.66 RMSE on a 0–100 yield scale, 3/5 folds |
| antibody thermostability | 10 of 33 | 4 of 33 | +0.0257 Spearman, 5/5 folds |

I verified the honesty of this run myself: every ranking call takes training-row slices only, and
scoring is on the held-out rows. No test fold enters the ranking.

**Three caveats that must travel with these numbers.** Two model seeds, not three. Donors are
pre-trimmed to 64 features each rather than the top 5 the brief specifies, and the random control
is drawn from that same pre-trimmed pool, so it tests the final ranking rather than the whole
chain. And — most importantly — **the ranker used here is a third one that exists only in that
script**: a single global lightgbm-gain ranking computed over the base features and the pool
together. It is not either of the engine's two rankers. So this run shows that extras can earn
budget; it does not show that any path the pipeline can currently run will find them.

### 5.2 What the shipping default does, measured through the real engine

Run on 25 August on logD and hERG with two planted decoys in the candidate pool: a block of pure
random numbers, and a lightly-jittered copy of features already inside the base representation.
A working selection rule must take zero of both.

Features taken from the planted donors, per fold:

| | pure noise | copies of base features |
|---|---|---|
| logD, 5 folds | 0, 3, 0, 5, 0 | 1, 5, 0, 5, 1 |
| hERG, 5 folds | 1, 0, 5, 5, 0 | 0, 1, 5, 5, 1 |

On logD the rule claimed +0.0080 and delivered +0.0058 on the untouched fold, while adding every
real extra feature with **no selection at all** delivered **+0.0131**. On hERG it claimed +0.0193
and delivered −0.0013.

**Read this narrowly.** It exercised **one** of the eleven paths, with extras appended on top of
a base already at its full native width — past saturation, where nothing can help. It is not
evidence about augmentation. It is evidence that the current default path admits duplicates and
noise, which §3.4 explains mechanically.

### 5.3 The before-and-after of the 24 August rebuild

| dataset | n | claimed | delivered | folds improved | noise features admitted |
|---|---|---|---|---|---|
| logD | 5,028 | +0.0087 | +0.0083 | 4 of 5 | 2.0 |
| hERG | 1,412 | +0.0183 | +0.0032 | 2 of 3 | 0.0 |
| antibody thermostability | 672 | +0.0056 | −0.0045 | 0 of 2 acted | 1.0 |

hERG before the rebuild: claimed +0.0214, delivered −0.0041, admitted 1.7 noise features per
fold. So the rebuild stopped it actively hurting.

**These numbers have no result files behind them.** The scripts wrote to a session scratch
directory that no longer exists. They are text only, and cannot be re-checked without re-running.
They also came from purpose-built scripts, not the pipeline.

### 5.4 The two framework experiments that were believed to validate the accept-reject loop

Both were cited as the evidence base for keeping the one-at-a-time loop. **Neither supports it.**
Recomputed from their stored result files on 2026-08-25.

**The March augmentation-strategy experiment (four datasets, 66 conditions each) has no
baseline.** Both of its no-augmentation control arms — all representations as base, and core
representations only — **failed on every row of all four datasets**, six rows each, zero scored,
with the recorded error "augmentation budget must be positive, got 0". Only the three
augmentation arms produced numbers, so there is nothing to compare them against and no claim of
the form "augmentation helps" can be read off it. Among the three arms that did score, the winner
differs by dataset: adding every donor's top features with **no test at all** wins outright on the
antibody set, whole-block keep-or-drop wins on the reaction set, and the fill-then-replace arm
wins on both gene-expression sets.

Separately, that experiment never tested the one-at-a-time append loop at all. Its three arms are
whole-block keep-or-drop, take-everything, and fill-then-replace.

**The March design sweep (1,152 configurations × 6 datasets) never touched the shipping code and
scores itself on its own selection slice.** It carries a private copy of a fill-then-replace
routine, and it hands the *same* validation slice to the selection loop and to the before/after
scoring. So every delta it reports is the claimed gain on the slice the selection optimised —
exactly the quantity that was later measured at +0.021 claimed against −0.004 delivered. Coverage
was also far thinner than the configuration count suggests:

| dataset | rows | rows with no candidates offered | rows where nothing was added |
|---|---|---|---|
| hERG | 1152 | **1152** | 1152 |
| antibody (jain) | 1152 | 768 | 775 |
| antibody thermostability | 1152 | 768 | 846 |
| buchwald | 1152 | 384 | 670 |
| gene expression (brca) | 1152 | 0 | 992 |
| gene expression (luad) | 1152 | 0 | 664 |

On hERG it never ran at all. And the two ranking rules it compared are indistinguishable in it —
mean deltas agree to the fourth decimal on every dataset — which is where "the ranker choice does
not matter" came from. That comparison used an in-sample residual ranker, which the engine's own
code now describes as useless, so it does not transfer to the current ranker.

### 5.5 Smoke test of the rebuilt `capped`, on real data — 2026-08-25

**This is a smoke test, not a result**: one dataset, three folds, one budget, one model. It exists
to show the rebuilt path runs correctly and behaves sensibly, not to rank anything. hERG, budget
500, five extra blocks capped at five features per donor, lightgbm, three scaffold-group folds,
through the real per-fold rebuild.

| arm | finished width | outer RMSE | extras kept | base features displaced | noise features taken |
|---|---|---|---|---|---|
| off (control) | 500 | 0.6113 | 0 | 0 | — |
| `uncapped` (appends outside budget) | 512 | 0.6003 | 11.7 | 0 | — |
| **`capped` (displaces, rebuilt)** | **500** | **0.5873** | 18.3 | 18.3 | — |
| `capped`, with a 40-feature pure-noise donor planted | 500 | 0.5917 | 18.3 | 18.3 | **8** |
| `uncapped`, same noise donor planted | 503 | 0.6139 | 3.3 | 0 | 0 |

What it establishes. The rebuilt path runs, stays inside the budget, produces matching train and
validation widths, and is calibrated — it claims +0.0257 on its internal slice and delivers
+0.0240 against the control. It is the best of the three arms **at the same width as the
control**, so on this dataset the gain is not extra model capacity; it is a better use of the same
budget. That is the same direction as §5.1.

What it also shows, and this is the open weakness. Offered a donor of pure random numbers,
`capped` took eight of them across three folds and its outer score got worse. `uncapped` took none
— but its score got worse too, and it admitted a third as many real extras, because the noise
donor crowded the pooled candidate list. So both paths degrade when junk is offered, in different
ways. That is §7.5, and it is undecided.

### 5.6 Measurements deliberately dropped

- **The base-width factorial on hERG.** Reported mid-run as a clean gradient, then retracted:
  it was a single fold and it vanished across five. Do not cite a base-width effect.
- **Everything from attempt 1 (19 August).** Its augmentation pool never loaded, so use case A
  never ran; and at one model seed, 128 features of pure noise scored above the target. The run
  was discarded.
- **The claim that the antibody dataset is arithmetically unwinnable.** It was argued from
  scaffold-family counts, never tested. The only measured fact is that this dataset's
  fold-to-fold spread is 0.1546, which is large relative to every effect being looked for. Keep
  the fact; the conclusion is not evidence.

---

## 6. Closed — do not raise these again

Fixed and verified: the per-source cap no longer collides two runs on one stored row; speed tiers
are gone; logD's augmentation list no longer contains three full-width representations; top-5 per
donor is honoured; the base representation now enters at full native width so "budgets too small"
and "pool too big" cannot apply to the single-representation arm; the ratcheting bar is out of
the default path; the residual ranker is out-of-fold and is the default; the honest re-score
passes the scaffold grouping in five places.

Settled by decision: representations are never in the cache key and there is no in-cache
versioning; the augmentation pool is exempt from the solo-score cut; grids are split by dataset,
never by axis.

Delivered: the literature search on rejecting useless features, with five costed options,
citations, and an excluded-with-reasons section. It is summarised in §7.5 and needs a decision,
but the search itself does not need repeating.

---

## 7. Open

### 7.1 The 20-versus-21 August contradiction — resolved by evidence, needs your sign-off

On 20 August you decided: rank once, cut at the cap, no accept/reject loop, no inner holdout.
On 21 August your own post-mortem listed *"the existing loop was switched off and replaced with a
home-made rank-once"* as failure mode number one.

**They are not opposites.** The post-mortem's complaint was that a mechanism was replaced
*without ever being run*; it was not a claim that the mechanism was known to work. §5.4 checked
that: the two framework experiments cited as its evidence base do not support it. One has no
working control arm on any dataset. The other never ran the shipping code, never ran at all on
hERG, and reads its own selection slice as the result.

So **no experiment in this repository has ever validated the accept-reject loop**, and the
20 August decision stands unopposed by evidence. What it lacks is a head-to-head.

One thing still deserves recording: on 24 August the substitution was made **in the shipping
engine**, inside a commit whose subject line is about removing tiers, so it is invisible in the
history.

**Recommendation: carry both as arms of the stage-2 grid.** It costs one axis, it is the
head-to-head that has never been run, and it satisfies the post-mortem's actual complaint.

### 7.2 ~~The 24 August fix has to reach the experiments~~ — done 2026-08-25

See §4.1. All 36 rebuild call sites now pass the grouping; 29 also pass the matched model and the
real metric. A pre-existing crash on the default allocation was found and fixed in the same pass
(§4.1b). Stage 2 is no longer blocked on this.

### 7.3 Percentage-of-budget sizing exists only in a scratch script

Your 20 August decision. The engine has a hard-coded ladder of absolute counts with no parameter
and no override, and the only implementation of the percentage design is the 21 August script.
For the grid in stage 2, the sizing axis has to be expressible by the engine or the grid cannot
include it.

### 7.4 ~~What happens to `capped` and to the unreachable routine~~ — decided 2026-08-25

**Decided by you: drop the fill-then-replace routine, repair `capped`.** Both done. The routine
was deleted (it was the superseded ancestor of `capped`, not a distinct method — see §3.3), and
`capped` was rebuilt (§4.2). Net effect on the engine: −242 lines.

**Also decided: greedy accept/reject is not coming back.** It was removed on 24 August, nothing in
the repository ever validated it (§5.4), it was measured claiming roughly ten times what it
delivered while admitting pure noise, and it costs about fifty model fits per fold against five
for the rebuilt paths. Recorded as dropped with that reason so it is not re-proposed.

So the admission methods that exist are five: off, take everything with no test, keep or drop each
donor's block whole, `capped`, `uncapped`. The original inventory counted eleven *code paths*; the
rest were a defect, two mechanisms that decide what gets **offered** rather than what gets kept,
one duplicate implementation out of tree, and the ancestor now deleted.

The original text of this decision is kept below for the reasoning.

<details>
<summary>Original framing (superseded)</summary>

You said on 24 August to remove the dormant code. Neither candidate was removed. Researched
properly on 2026-08-25:

**`capped` is not dormant, but its live footprint is much smaller than "14 datasets".** The
precompute driver emits it **only when a dataset has 250 rows or fewer** — its own note says it
"has only ever completed for n ≤ 242". Thirteen rows of the per-dataset policy table name it, but
the ones I can check are all far above that gate (logD 5,028; hERG 1,412; buchwald 3,955), so
those rows never emit it. It is also read by the precompute status checker, by one
gene-expression experiment, and by a federated experiment that reads it off saved recipes.

**Its design is the one the best evidence supports; its implementation is the broken one.**
`capped` is the only shipping path where extras must **displace** base features to earn their
place, rather than being appended outside the budget. That is exactly the design of the 21 August
budget-split run in §5.1 — the strongest result in this whole workstream, 48 of 48 cells positive
on logD. So the mechanism is not dead weight. But as implemented it carries the running-maximum
bar that the default path was rebuilt to remove, never receives the per-donor cap (so the widest
donor wins on feature count alone), has broken eviction bookkeeping after the first swap, and
under every shipping allocation the budget is already spent when it starts — so its fill phase
never runs and "capped" means "evict only".

Four options, and this is a real decision rather than a cleanup:

1. **Leave it.** It keeps shipping for small datasets and keeps producing numbers selected by the
   mechanism measured as unsound. Cheapest, and the least defensible.
2. **Fix it** — same treatment the default path got, plus forwarding the per-donor cap and
   repairing the eviction bookkeeping. Then it is a genuine second arm in the stage-2 grid, and it
   is the arm that implements the displacement design.
3. **Replace it.** Drop the routine and put displacement into the grid as a budget-split arm
   instead, which is what the 21 August script already does and what your 20 August percentage
   decision describes.
4. **Remove it outright** — which also means removing it from the precompute driver, the status
   checker, the policy table and the federated experiment.

**Recommendation: 2 or 3, not 1 or 4.** The displacement idea has the best evidence behind it of
anything here; what is broken is the code, not the concept. Between them, 3 is closer to your
20 August decision and 2 preserves the existing cache condition.

**The unreachable ~290-line routine** genuinely has no dispatch branch. The framework experiment
that shares its name keeps its own private copy, so deleting the library version breaks nothing.
That one is a clean delete whenever you want it.

</details>

### 7.5 Which rule keeps useless features out — five options, cheapest first

| | approach | cost | catches a duplicate of the base? |
|---|---|---|---|
| 1 | **Add everything, select nothing.** Let the trees ignore what is useless | **less than now** | n/a |
| 2 | **Read the numbers you already have differently** — pair errors per molecule within scaffold families, require a margin over adding nothing, require 4 folds of 5 | **free** | no |
| 3 | **Screen whole blocks conditionally** — remove what the base explains from both sides before comparing, threshold from the block and family counts | ~1.2× | **yes** |
| 4 | **Prevent the harm instead of detecting it** — fit the base first, let a small second model correct only its errors | ~2× | no |
| 5 | **Drop each block, refit, measure the damage** | 5–7× | yes |

Boruta and knockoffs were excluded with stated reasons: Boruta is built to keep everything
related to the property including duplicates of what you have, and a 2025 benchmark measured its
false-discovery rate above 0.75 against a 0.1 target; knockoffs at a 10% target return either ten
or more features or exactly zero, so "three features help a little" outputs zero forever.

Option 1 is the null the others have to beat, and §5.2 already shows it beating the current
default on logD. My reading, offered as a reading and not a result: options 1, 2 and 3 are cheap
enough to be **arms in the stage-2 grid** rather than a decision made before it.

**This got sharper on 25 August.** Both admission paths now choose a block size by taking the best
of five scores read off one internal slice, with no margin — so a near-tie is settled by luck. For
the appending path the worst case is a few useless extra features. For the displacing path the
worst case is throwing away real base features to make room for them, which is strictly worse. A
synthetic check showed exactly that: offered nothing but random numbers against a base that was
mostly uninformative, it preferred twenty random features over twenty base features, because the
two scored within noise of each other.

The margin from option 2 is what fixes this, and the value should be swept in stage 3, not picked
by me. The value has to be stored as a **setting**, so it can differ between runs, which means it
enters the key of the results table. Two ways:

- Reuse `greedy_tolerance`. Already a search parameter, defaults to 0.0, already in the key. One
  number would then control two unrelated things: when the greedy allocators stop, and how much
  better a block of extras must be. Costs nothing.
- Add a new setting. Cleaner, and free **today** because the cache is empty. Widening the key
  later means deleting every dataset's results database, because the table has no upgrade path
  (§4.7).

Not built. Awaiting the choice between the two.

### 7.6 Augmentation is not a search axis

Comparing one admission rule against another currently requires separate outer calls. That is
workable for the stage-3 grid but it is the opposite of the brief's "searched upfront", so it has
to be decided whether stage 4 makes it a real axis.

### 7.7 Nothing on the server reflects any of this

No stored result exists under the current code for any of the eleven paths. The cache was
cleared. Stage 3 is therefore a full regeneration, and it must not be submitted until §7.2 is
fixed, otherwise it measures the defect.

### 7.8 The dataset roster for stage 3

Only four datasets have representations cached locally: hERG, logD, buchwald, and one antibody
set. All three measured so far are **flat-topped** — no representation clearly dominates; logD's
top two are 0.0004 apart against fold spreads around 0.03. So **Scenario 2 has no local testbed**,
and the ten datasets for the server run have not been chosen. That choice is part of stage 2.

---

## 8. Where things live

**Engine.** The admission paths, the two rankers and the auto-demotion are all in
`src/kirby/hybrid.py`. The search, the cache identity and the honest re-score are in
`src/kirby/find_best_hybrid.py`. Neither imports the per-dataset pool.

**Pools.** `DATASET_REPS` in `tests/experiments/complexity/complexity_helpers.py`.

**Experiment.** `tests/experiments/complexity/c21_single_rep_augmentation.py` implements every
arm in the brief — base alone, base plus curated blocks, base plus top-5 from each other
representation, both, the multi-representation hybrid, and that hybrid plus the blocks. It has
matched-width random controls and can be pinned to one base representation and one model. It is
committed. It does **not** pass the grouping, the model or the metric (§4.2 of the blocker list).

**Results still on disk.** `<cache>/aug_budget_split/` — the 21 August run, four datasets.
`<cache>/aug_stage/` — attempt 1, discarded.

**Scratch scripts.** `scratchpad/aug_budget_split.py` and `aug_budget_split_margin.py` produced
and analyse §5.1. `scratchpad/aug_decoy_check.py` produced §5.2. `scratchpad/aug_pass1_target.py`
establishes the best representation per dataset under the real metric.
`scratchpad/aug_mechanism_sweep.py` is an **unreviewed draft** of the stage-2 grid, written
ahead of the decisions above and never run to completion — treat it as a proposal, not as agreed
work.

**Local machine only.** The first model fit after data is loaded kills the interpreter with no
traceback, because two copies of a threading library get loaded. Every scratch script works
around it by fitting a two-tree model before anything else is imported. This is not a server
problem.
