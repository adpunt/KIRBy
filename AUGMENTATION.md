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

1. Work out what is broken and fix it. **Done 25-26 August.**
2. Define the experimental grid: which augmented representations, which selection method, which
   allocation method, and whether extras go onto a base representation or onto a hybrid.
   **← we are here.** The grid is §9. The code changes are §4.8, §8 and §9.13. The cost is
   §9.13, **not §9.7 or §9.12** — both of those counted only augmentation model fits and missed
   the largest cost entirely.
3. Run that grid on the server, on the **six** datasets in §9.10 — cut from the fourteen
   in §7.8 by you on 26 August, for time.
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
| 08-26 | The 250-row gate on `capped` is stale but is **not changed until the new cost is measured** | pending, §9.8. |
| 08-25 | Never cut axes, arms, budgets or datasets for compute. Cuts need a stated reason: degenerate, broken, or undefined | standing. |
| 08-26 | The roster is the framework ten plus the four widest solo-score gaps — fourteen, one job each | done, §7.8. |
| 08-26 | The margin sweeps **multiples of that dataset's own fold spread**, not raw metric units | done, §9.4. |
| 08-26 | The candidate cut-off gets **its own experiment** sweeping 50, 100, 250. Not 500, not uncut | done, `f25`. |
| 08-26 | The work goes into **both** `c21` and a new f-numbered framework experiment | done, §8. |
| 08-26 | **Cut for time, not for a scientific reason:** 6 datasets not 14, 3 folds not 5, 2 downstream models not 3 | done, §9.10. What it costs in coverage is written there. |

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

### 3.3 The paths that decide which extra features get added

Eleven code paths were inventoried. **Five were actual ways of deciding what to keep**: `none`,
`all`, `greedy_ablation`, `capped`, `uncapped`. `greedy_ablation` was cut on 27 August (§9.11),
so **four** are live. Of the other six, one was a defect (row 8), two
decide what gets *offered* rather than what gets kept (rows 9 and 10), one is a duplicate living
inside a framework experiment (row 11), one was deleted as a superseded ancestor (row 7), and one
was removed in August and is staying removed (row 6).

**No stored pipeline result exists under the code as it stands.**

| # | path | setting | live? | state |
|---|---|---|---|---|
| 1 | Rank the pooled candidates once, then choose a block size from a fixed ladder (0, 5, 10, 20, 40); ties keep nothing. Extras are appended **outside** the budget | `uncapped` — **the default** | yes | rebuilt 08-24. §4.5 |
| 2 | Same ladder and same tie rule, but the base is cut to (budget − k) columns so extras must **displace** base features and the total stays inside the budget | `capped` | yes, for n ≤ 250 | **rebuilt 08-25.** §4.2 |
| 3 | Keep or drop each donor's top-20 block whole | `greedy_ablation` | **CUT 2026-08-27** (§9.11) | order-dependent, base-blind, ratchets. Still dispatchable so old recipes do not crash; nothing selects it |
| 4 | Concatenate every donor's top 20, no test at all — the do-nothing arm | `all` | no — one archived script | ranking **fixed 08-26**; §5.2's logD comparison must be re-run. §4.3 |
| 5 | Off | `none` | yes | the control arm; no defects of its own |
| 6 | The old add-one-at-a-time loop that ratcheted | (former `uncapped`) | **removed 08-24** | **produced every augmentation result previously on disk.** Those numbers came from code that no longer exists. |
| 7 | ~~A second fill-then-replace routine, ~290 lines~~ | — | **deleted 08-25** | It was never a lost method: on 24 March its branch was replaced by `capped` and `uncapped`, which are the two halves it was split into. It was the superseded ancestor, left in the file. |
| 8 | ~~The silent fall-through~~ | any unrecognised string | **fixed 08-26** | now raises and lists the five valid names. §4.4 |
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

### 3.6 The settings, and which of them a caller can reach

Re-read 2026-08-26. Four were reachable in name only: the stored result filed two values as two
separate rows, but the recipe carried neither, so every per-fold rebuild took the engine's own
default whatever the run was configured with. **Fixed 2026-08-26 — see §4.8.**

| setting | what it does | in the stored result's key? | on the recipe? |
|---|---|---|---|
| `aug_per_source_budget` | features taken from each donor before pooling | yes, inside `aug_names` | yes |
| `candidate_ranking` | which of the two ranking rules runs | yes | **added 08-26** |
| `augmentation_threshold` | how much a whole block must improve for `greedy_ablation`, which is CUT (§9.11) | yes | **added 08-26** |
| `augmentation_budget` | features per donor for `all` (and for `greedy_ablation`, CUT) | yes | **added 08-26** |
| `augmentation_margin` | how much better than keeping none a block must score | yes | yes |
| `augmentation_fraction` | extras sized as a fraction instead of the ladder | yes | yes |
| `aug_max_candidates` | how long the ranked candidate list is | **no** | **added 08-26** |

The per-source cap now reaches `capped` as well as `uncapped`; it did not until 2026-08-25.

`patience` is hard-coded at 10 and inert on both rebuilt paths — there is no sequential accept loop
left to be patient about, and the code says so. The default path still has no budget handling: it
appends outside the budget, so an augmentation-off arm and an augmentation-on arm at the same
nominal budget are **not the same width**, and never were.

**The candidate cut-off stays out of the key on purpose.** Adding a key column widens the results
table's primary key, and that table has no upgrade path, so it would force every
`result_db/*.db` to be deleted. It is swept instead by `f25_candidate_cutoff`, which constructs
recipes directly and never writes to `result_db`. The search itself always runs at 50 and records
that on the recipe, so two searches can never differ in it and collide on one row.

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

### 4.8 Three settings were in the stored result's key but not on the recipe. **Fixed 2026-08-26.**

This is §4.1 again, three settings later, and it was found the same way — by reading the rebuild
helpers rather than trusting that a setting named in a signature arrives anywhere.

`candidate_ranking`, `augmentation_threshold` and `augmentation_budget` are all columns of
`CONFIG_COLS` in `src/kirby/result_cache.py`, so the search correctly files two values as two
separate rows. None of them was a field on `Blueprint`. Neither rebuild helper passed them —
not `_shared.build_fold_hybrid`, and not the engine's own rebuild in `find_best_hybrid.py`. So a
search that ran under `rf_importance` and filed its row under `rf_importance` was rebuilt, on every
fold, under `residual_correlation`. Nothing raised.

The row said one thing and the rebuild did another, exactly as in §4.1. The scale is smaller only
because no caller had ever set any of the three: **every production run in the repository's history
used `residual_correlation`, threshold `None` and 20 features per donor**, so no stored result is
retroactively wrong. What was broken was the ability to vary them at all, which is what the
stage-3 grid needs.

All three are now recorded on the recipe and forwarded by both rebuild helpers, along with
`aug_max_candidates`. Adding recipe fields **changes no key**: a recipe rides inside a stored row
as a pickled blob, so nothing is invalidated and nothing has to be deleted. Recipes pickled before
the fields existed still load and report the old defaults, which is what they ran under.

Verified in-session by a round trip — a recipe built with `rf_importance`, put through
`to_dict`/`from_dict` and through `pickle`, then followed into `create_hybrid` and asserted on
arrival. Also verified by measurement: on synthetic data the two ranking rules now give different
answers (RMSE 2.1510 against 2.3011, and 1 noise feature admitted against 2), which they could not
have done from any caller before this change.

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
### 5.7 First real f24 run — flab_thermo_tresanco, 2026-08-26. A CHECK, NOT A RESULT.

**One dataset, three folds, one downstream model, one model seed, the curated donor only.** It was
run to prove the path works on real data, and it measures nothing that should be quoted. Conditions:
base representation `saprot` at its full 1,280 features, chosen from
`rep_prune/flab_thermo_tresanco/scores.json`; random forest; `single` placement; 5 features per
donor. The only curated donor is `mutation_features`, 48 features wide.

**Fold spread: 0.0407 Spearman**, over these three folds. §5.6 records 0.1546 for this dataset from
an earlier run under different conditions. Both are measured; neither supersedes the other, and the
conditions decide which applies. Every difference below is set against 0.0407.

| arm | Spearman | features kept per fold | pure-noise features taken | near-copy features taken |
|---|--:|--:|--:|--:|
| no extras | 0.5534 | 0 | — | — |
| `all`, top-5 per donor | 0.5768 | 15 | 15 | 15 |
| `all`, top-5, **matched random donors** | 0.5758 | 15 | 15 | 15 |
| `all`, top-20 | 0.5656 | 60 | 60 | 60 |
| `all`, top-20, **matched random donors** | 0.5860 | 60 | 60 | 60 |
| `all`, top-64 | 0.5761 | 160 | 144 | 192 |
| `all`, top-64, **matched random donors** | 0.5916 | 160 | 144 | 192 |
| `capped`, margin 0 | 0.6009 | 8.3 | 9 | 14 |
| `capped`, stricter margin | 0.5715 | 1.7 | 2 | 2 |
| `capped`, strictest margin | 0.5534 | 0 | 0 | 0 |

Counts of decoy features are totals over the three folds.

Three things this establishes about the machinery, none of them about which method is better.

**The decoys work, and they are not a side check.** `capped` at margin 0 is the only arm whose
difference from no-extras (+0.0475) exceeds the fold spread. It kept about 25 features across the
three folds and 23 of them were planted junk. Reading its score without its decoy counts would have
recorded a win.

**The margin axis works and it is doing something.** Three margin multiples produced three distinct
outcomes from one ladder scoring — 8.3 features kept, then 1.7, then 0 — and the strictest lands
exactly on the no-extras score of 0.5534, which is what keeping nothing should give. Decoy admission
fell from 23 features to 4 to 0.

**The matched random donor is not optional.** `all` beat no-extras at every donor width, and the
matched random donors beat the real curated donor at top-20 (0.5860 against 0.5656) and top-64
(0.5916 against 0.5761). On this dataset `mutation_features` contributes nothing that random
features of the same shape do not.

`uncapped` and the matched random control for `capped` had not completed when this was written.


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

### 7.3 ~~Percentage-of-budget sizing exists only in a scratch script~~ — built 2026-08-26

**Built as `augmentation_fraction`, default None.** None keeps the old fixed ladder, so nothing
changes unless the value is set. Set means: take exactly `round(fraction x width)` top-ranked
extras, with **no internal comparison at all** — the rest of your 20 August decision, "rank once,
cut at the cap, no inner holdout". The grid sweeps the value; the engine never picks it. Setting a
fraction and setting a margin are alternatives, not a stack: with a fraction there is nothing left
to choose, so the margin is not consulted.

The reference width differs by strategy, because the two mean different things. `capped` keeps the
total inside the budget, so the fraction is of the finished width. `uncapped` appends outside the
budget, so the fraction is of the base hybrid's own width.

Measured, synthetic, three budgets:

| budget | fixed ladder | 5% | 25% |
|---|---|---|---|
| 60 | 10 extras | 3 | 15 |
| 120 | 40 extras | 6 | 30 |
| 240 | 5 extras | 12 | 50 |

The fixed ladder's counts are unrelated to the budget. The fractions scale.

**Whether a fraction beats the ladder on real data is still not measured.** The comparison was
started on hERG on 26 August and abandoned: the machine was running four of the user's own
experiments plus a Rust build, load average 55, and the run was getting 3.7% of one core. It was
still at load 43 on 8 cores later the same day.

**It is not being re-run as a separate local job. It is the size axis of the stage-3 grid** —
`ladder` against `frac0.05`, `frac0.10` and `frac0.25`, on the six datasets rather than on
hERG alone, with the decoy counts alongside. Recorded here so it does not read as skipped.

**One thing that measurement would have got wrong.** The fraction axis is degenerate at the
candidate cut-off of 50. The number of extras is
`min(round(fraction × width), pool width, cut-off)` — three ceilings, smallest wins
(`hybrid.py:2533-2546`, `:2759-2764`). At a budget of 1000 the three fractions ask for 50, 100 and
250 and every one of them clamps to 50, so all three would have produced the same answer. On
herg_ki with 5 features per donor, `f24 --count-only` found 60 of 686 configurations degenerate
at the time (before `greedy_ablation` was cut)
for exactly this reason and skips them with the arithmetic printed. That is why the cut-off needed
its own experiment before the size axis could mean anything.

### 7.3b (superseded framing)

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

**Built 2026-08-26 as `augmentation_margin`, on your instruction.** A block of extras must now
beat keeping none by more than the margin, and must also be the best rung that clears it.
Default 0.0 reproduces the old rule exactly: any improvement wins, however small.

Chosen over reusing `greedy_tolerance` for two reasons. Adding a setting widens the key of the
results table, which has no upgrade path (§4.7), so every `result_db/*.db` has to be deleted —
free on 26 August because the cache was already empty, ruinous after stage 3 fills it. And the two
numbers mean different things: `greedy_tolerance` decides when the greedy allocators stop, the
margin decides how much better a block of extras must be. Sharing one number would make a stored
result ambiguous later, and your own 25 August note says `dominant_greedy` was never actually cut.

Measured effect, synthetic, a donor of 40 pure random features against a 30-feature budget:

| margin | `capped` keeps | base features thrown away | `uncapped` keeps |
|---|---|---|---|
| 0.0 | 5 | 5 | 20 |
| 0.005 | 5 | 5 | 20 |
| 0.01 | 5 | 5 | 20 |
| 0.02 | **0** | **0** | 20 |

`capped` stops taking junk at 0.02. `uncapped` still takes 20 junk features at 0.02, so it needs a
larger margin on this data. Which value to use is exactly what stage 3 should sweep, and the two
paths may not want the same value.

**The margin is now an axis, not a decision.** No value is chosen.

### 7.6 Augmentation is not a search axis

Comparing one admission rule against another currently requires separate outer calls. That is
workable for the stage-3 grid but it is the opposite of the brief's "searched upfront", so it has
to be decided whether stage 4 makes it a real axis.

### 7.7 Nothing on the server reflects any of this

No stored result exists under the current code for any of the eleven paths. The cache was
cleared. Stage 3 is therefore a full regeneration, and it must not be submitted until §7.2 is
fixed, otherwise it measures the defect.

### 7.8 ~~The dataset roster for stage 3~~ — chosen 2026-08-26

**Fourteen, one job each.** The framework ten — the roster
`tests/slurm_scripts/run_f26_redundancy_penalty.sh` already submits — plus the four datasets with
the widest gap between the best and second-best solo representation.

The gap for all 42 datasets was recomputed on 2026-08-26 from
`<cache>/rep_prune/<ds>/scores.json`, the same file `DATASET_STATUS.md`'s "best solo" column is
built from. The framework ten are all flat-topped, gaps 0.0005 to 0.0279, so none of them can
answer whether extras help a **dominant** representation. These four can:

| dataset | n | best solo | runner-up | gap |
|---|--:|---|---|--:|
| abagym_VEGF | 418 | mutation_features 0.4907 | saprot 0.3942 | **0.0965** |
| abcov_kd | 141 | igt5 0.3849 | antiberty_cdr 0.3217 | **0.0632** |
| ginkgo_acsins | 242 | igt5 0.5319 | ablang2 0.4789 | **0.0530** |
| ddg_combined | 429 | mutation_features 0.3492 | onehot 0.2973 | **0.0519** |

All four have an empty curated `augmentation` list in `DATASET_REPS`, so only the
borrow-top-5-from-each-other-representation donor runs on them. That is not a defect; it is what
those datasets have.

The four with every representation on the seagate drive are herg_ki, logD, buchwald_hartwig and
flab_thermo_tresanco. Their gaps are 0.0118, 0.0018, 0.0202 and 0.0167 — none of them dominant.
Two earlier figures do not survive re-checking: this document said logD's top two are 0.0004
apart, which matches neither on-disk source (`rep_prune` gives 0.0018, `validity` gives 0.0009).
Use 0.0018 and say where it came from.

---

## 8. Where things live

**Engine.** The admission paths, the two rankers and the auto-demotion are all in
`src/kirby/hybrid.py`. The search, the cache identity and the honest re-score are in
`src/kirby/find_best_hybrid.py`. Neither imports the per-dataset pool.

**Pools.** `DATASET_REPS` in `tests/experiments/complexity/complexity_helpers.py`.

**Experiments.** Three files, one question each.

`tests/experiments/complexity/c21_single_rep_augmentation.py` — **where extras go.** Base alone,
base plus curated blocks, base plus top-5 from each other representation, both, the
multi-representation hybrid, and that hybrid plus the blocks. Matched-width random controls, and
both planted decoys. It **does** pass the scaffold grouping, the real model and the real metric
into the admission decision, at lines 453-461 and 233-239 — this document said otherwise until
2026-08-26 and was wrong; that was fixed in stage 1 along with the other 35 rebuild call sites.

`tests/experiments/framework/f24_augmentation_admission.py` — **which admission method.** The five
methods crossed with the donor source, the per-donor cap, the candidate ranker, the size rule and
the margin. New 2026-08-26.

`tests/experiments/framework/f25_candidate_cutoff.py` — **how long the ranked candidate list should
be.** 50, 100, 250, everything else pinned. New 2026-08-26.

`tests/analysis/read_f24_augmentation.py` — reads the results. Per-cell win counts **per dataset**,
never a mean across them. It leads with the decoy counts, before any score: a configuration that
took planted noise or near-copies has not won anything, whatever it scored.

**Submitters.** `run_f24_augmentation_cost.sh` — the cost measurement, one array task per dataset
over the **six** (§9.10). **Run this first.** `run_f25_candidate_cutoff.sh` — 6 tasks; run this second,
and set the grid's `--max-candidates` from what it says. `run_f24_augmentation_grid.sh` — 56 tasks,
**168** tasks: six datasets, each one single-representation task plus 3 budgets by 9 allocation
methods (§9.13). Keep the dataset list
identical across all three or the cost numbers stop matching the run.

Two things `c21` could not do before 2026-08-26. Its admission method was the hardcoded string
`'uncapped'` in two places, so it answered the placement question for one rule only; it is now
`--strategies`. And `--models` was registered twice, once in `create_experiment_parser` and again
in `c21` itself, so argparse raised before `parse_args` and **every command-line start of the
script died**. It reached the cluster only through `run_dataset.py complexity <ds>`, which calls
`run()` in process, which is why that went unnoticed.

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

---

## 9. The grid — written 2026-08-26, stage 2

Nothing below has been run. This is the design and what it costs.

### 9.1 The two placements

| placement | base | budget |
|---|---|---|
| single representation | the dataset's strongest solo one, at its full native width | its own width, not an axis |
| multi-representation hybrid | `weighted_global` over the ten-representation pool | 500, 1000, 2000 |

The strongest solo representation is read from `<cache>/rep_prune/<ds>/scores.json`, not chosen. On
herg_ki that is `mhg_gnn`, beating `graph_kernel` by 0.0118. When that file is missing the widest
representation is used and the output says **NOT MEASURED** rather than presenting a guess.

### 9.2 The axes

| axis | values | count |
|---|---|---|
| donor source | curated `augmentation` list · top-5 of each other base representation · both | 3 |
| admission method | `none` · `all` · `capped` · `uncapped` | 4 |
| features per donor | 5 · 64 · off | 3 |
| candidate ranking | `residual_correlation` · `rf_importance` | 2 |
| size rule | the ladder of 0/5/10/20/40, or a fraction of 0.05 / 0.10 / 0.25 | 4 |
| margin, ladder only | 0 · 0.1 · 0.25 · 0.5 · 1.0 × that dataset's own fold spread | 5 |
| features per donor for `all` | 5 · 20 · 64 | 3 |
| ~~block threshold~~ | `greedy_ablation` only, and it is cut — §9.11 | — |
| downstream model | lightgbm · rf (xgboost cut for time, §9.10) | 2 |
| allocation method, hybrid placement only | the nine the search sweeps, §9.13 | 9 |
| internal evaluator | **not an axis** — dead code when a model is supplied, §9.13 | 1 |

Fractions stop at 0.25 because the cut-off ceiling is 250, and a quarter of a 1000 budget is
exactly 250 extras.

A margin and a fraction are alternatives, never a stack: with a fraction set the size is already
fixed, so the engine does not consult the margin (`hybrid.py:2605-2611`, `:2795-2797`). Pairing
them would file configurations differing only in a number nothing reads. So the size axis is five
ladder-with-margin settings plus three fraction settings, eight in all.

### 9.3 Which method reads which axis

An axis a method does not read is collapsed to one value and the reason is printed, never dropped
in silence.

| method | donor source | per-donor cap | ranking | size rule | margin | configurations |
|---|---|---|---|---|---|--:|
| `none` | — | — | — | — | — | 1 |
| `all` | yes | no | no | `augmentation_budget` | no | 9 |
| ~~`greedy_ablation`~~ | cut 2026-08-27, §9.11. Dispatchable, never selected | | | | | — |
| `capped` | yes | yes | yes | yes | yes | 144 |
| `uncapped` | yes | yes | yes | yes | yes | 144 |

**These counts are superseded.** They predate `greedy_ablation` being cut (§9.11) and the margin
becoming a readout rather than an enumerated axis (§9.13). `--count-only` prints the live number
per dataset; do not quote a count from this document.

### 9.4 The margin is in fold-spread units, not metric units

`augmentation_margin` reaches the engine as a raw number in the units of the dataset's metric. 0.02
is two thirds of logD's fold-to-fold RMSE spread and about an eighth of tresanco's Spearman spread
of 0.1546 — measured under different conditions from the 0.0407 in §5.7; both are real and the
conditions decide which applies. Either way one list of absolute numbers means very different
strictness on different datasets.

The experiment measures the base-only arm's fold-to-fold standard deviation **first**, multiplies,
and passes the product in. Nothing in `src/kirby/` changed; the engine still takes an absolute
number. Every result row carries the multiplier, the measured spread and the product, so a stored
number can be read back in either unit.

### 9.5 The three controls

- **Matched-shape random donor** — the real donors replaced by randomly drawn features of the same
  widths. Its own arm, so the configuration count doubles. Adding features changes model capacity
  by itself; without this a win cannot be told apart from "the model got more features".
- **Planted pure-noise donor** — Gaussian noise, as wide as the real donors combined.
- **Planted near-copy donor** — features copied out of the base representation and roughed up by 5%
  of each feature's own standard deviation. The only decoy that catches a rule rewarding what the
  base already carries.

Both planted donors go into the same pool as the real ones, so they add no configurations. What
they buy is a count per fold. The builders live in `tests/experiments/_shared.py` as
`plant_decoys` / `decoy_counts` — one implementation, shared by `c21`, `f24` and `f25`.

### 9.6 Degeneracy is asserted, never assumed

A configuration is skipped only when it can be shown to give the same answer as another one, and
the arithmetic is printed. Measured on herg_ki before the 27 August cuts: **60 of 686 were
degenerate**, all for one reason —
at 5 features per donor the pooled candidate list is 25 features wide for the curated donors and 45
for the borrowed ones, so a fraction asking for 102 or 256 extras clamps to the same count that
0.05 already produced.

`all` at 64 features per donor is **not** degenerate on herg_ki: its widest curated donor, `maccs`,
is 167 features, so 64 and 20 genuinely differ there. On a dataset whose donors are all narrower
than 64 it would be, and the check would catch it.

### 9.7 What it costs — SUPERSEDED by §9.13, kept for the counting method

**The totals below are wrong and the reason is worth keeping.** They count augmentation model
fits. Building the base hybrid is not an augmentation fit, so it counted as zero — and on the
multi-representation placement that build is the dominant cost. See §9.13.



Counted from the engine, not guessed. Per fold a `capped` configuration on the ladder costs one
forest fit per donor wider than the cap, plus three forest fits for the out-of-fold residual
ranking (one for `rf_importance`), plus one forest fit to rank the base features, plus five fits of
the real downstream model. A fraction replaces the five scoring fits with two. `uncapped` is one
fit fewer than `capped`. `all` costs no scoring fits at all — it runs no test.

`f24 --count-only` on herg_ki, measured 2026-08-26 with nothing fitted:

| method | configurations | fits per fold | fits total |
|---|--:|--:|--:|
| `none` | 1 | 0 | 0 |
| `all` | 18 | 84 | 2,520 |
| `greedy_ablation` | 90 | 885 | 26,550 |
| `capped` | 228 | 1,932 | 57,960 |
| `uncapped` | 228 | 1,818 | 54,540 |

111,690 fits per placement-and-budget; four such combinations, so **2,260 configurations and
446,760 model fits for herg_ki.** flab_thermo_tresanco is 2,164 configurations and 374,040 fits.
`f25` is far smaller: 144 configurations, 22,440 fits.

**Seconds per fit — measured, with a caveat that must travel with it.** 10.82 s on
flab_thermo_tresanco: 194.8 s of fitting for 18 fits of the single most expensive configuration
(`capped`, both donor sources, no per-donor cap, the out-of-fold residual ranking, the full ladder),
representation loading excluded. Measured on the local machine at load average 43 on 8 cores, with
`rf` and one model seed. **That is an upper bound, not the server's number** — the machine was five
times oversubscribed and the grid also runs lightgbm and xgboost.

At that rate one dataset is 374,040 × 10.82 s ≈ **1,124 hours, or 47 days.** Even split four ways
by placement and budget it is 281 hours a piece, still past the 7-day partition. A quiet 16-core
node should be several times faster, but by how much is not known until
`tests/slurm_scripts/run_f24_augmentation_cost.sh` runs there — one fold of one configuration per
dataset, over all six. It has to run on the server anyway: two of the six have no
representations on the local drive at all.

**So the grid does not fit as one job per dataset, and that is a splitting problem, not a reason to
cut an axis.** `run_f24_augmentation_grid.sh` submits **168** array tasks — six datasets × the
placement-and-budget combinations. If the server's measured rate still overruns, split further
within a dataset by admission method (five more sub-jobs) before anything is dropped.

### 9.8 The 250-row gate on `capped` — REMOVED 2026-08-27

Gone, on your instruction. `CAPPED_MAX_N` no longer exists anywhere in the tree.

It only ever gated the **precompute** and the experiments that read it. It never touched the
augmentation study: f24, f25 and c21 build recipes directly and had zero references to it.

Why it existed: the old `capped` refitted a model and ran a permutation-importance pass for every
candidate feature, which is what "has only ever completed for n <= 242" measured. The rebuilt
`capped` (25 August) costs one ranking fit, one base-ranking fit and five scoring fits per fold.
At 250 rows the gate was excluding `capped` from every molecular dataset — logD 5,028 rows,
hERG-Ki 1,412, buchwald 3,955 — and `capped` is the only method where an extra feature has to be
worth more than a base one.

Removed from six files: the constant and two gates in `tests/experiments/_shared.py`, the
condition list in `tests/precompute_hybrids.py`, the expected-condition list in
`tests/check_precompute_status.py`, and the capped-or-uncapped switch in `c02`, `c05` and `c07`,
which now always ask for `capped`.

**Two of these have to stay in step.** `precompute_hybrids.py` emits the conditions and
`_shared.get_precomputed_search` asks for them. If one has `capped` and the other does not, the
lookup misses and the experiment cold-builds the search inside the job. Both are unconditional now.

**What this costs is not measured.** Precompute jobs now carry a third condition on datasets that
previously had two. If one overruns, shorten that job — do not put the gate back.

### 9.9 What the grid does NOT need

The precompute driver does not have to emit the grid. Every augmentation setting sits outside the
search grid and is spliced identically into every configuration, so varying one has always needed a
separate outer call — but `c21`, `f24` and `f25` all sidestep that entirely: they construct the
recipe and call `build_fold_hybrid`, never `find_best_hybrid`. Nothing they run touches
`result_db`, so nothing has to be deleted before or after and no stored result can be evicted.

The driver's only outstanding change is the row-count gate in §9.8. Whether augmentation becomes a
real search axis is still §7.6, and still a stage-4 question.

### 9.10 The cuts of 26 August, and what each one costs

Two different things happened on 26 August and they must not be read as one.

**Removing repeated work. Cuts nothing, and needs no sign-off.**

The margin is not a compute axis. `augment_capped` scores five block sizes and gets five numbers.
The margin only decides which of the five wins. Measured: six margin values returned the identical
no-extras score of −2.4595016587 and produced **two** distinct outcomes. Four of those six runs
re-fit five models to reach an answer already computed. The ladder now runs once and every margin
is read off its rung scores; a margin that picks a different block size still gets its own run, so
the axis is fully covered.

The candidate ranking does not depend on the downstream model. Measured: the order it offered
features in was identical under a random forest, gradient boosting, ridge, and no model at all.
`model` appears in that function's signature and its docstring and nowhere in its body. So the
grid was re-fitting it once per model for a number that cannot change.

`create_hybrid` now takes a candidate ranking a caller has already computed, in the same way it
already takes `precomputed_alloc_scores` for allocation. **Verified bit-identical** against the
recomputing path on 24 combinations of method, ranking rule, per-donor cap and size rule — the
features kept, how many, and every rung score. The check lives in
`tests/smoke/smoke_c21_augmentation_axes.py` and must stay green.

Effect on herg_ki: 446,760 model fits down to 90,000, with every value of every axis still run.

**Cutting coverage. This is the user's call and is recorded as such.**

Six datasets rather than fourteen; three folds rather than five; two downstream models rather than
three. The reason is time. It is not a scientific reason and must not be written up as one.

| what | kept | dropped | what is lost |
|---|---|---|---|
| datasets | herg_ki, logD, buchwald_hartwig, flab_thermo_tresanco, abagym_VEGF, abcov_kd | caco2, both flab_pk_jain sets, ginkgo_hic, ginkgo_acsins, ddg_combined, tz_her2_binary, tcga_luad_vs_lusc | TCGA and nucleotide are now unrepresented. The run cannot say whether a method that wins on molecules also wins on gene expression or on DNA. |
| folds | 3 | 5 | The fold-to-fold spread is measured on three numbers, and it is the unit every margin and threshold is expressed in. tresanco's spread has been measured at 0.1546 and at 0.0407 under different conditions (§5.7); both are large next to the effects being looked for. |
| models | lightgbm, rf | xgboost | Two learners rather than three. lightgbm and rf are a boosted and a bagged learner, so they are not near-duplicates; xgboost would have been a third boosted one. |

The six were picked to keep both metrics (RMSE and Spearman), both flat-topped and dominant
datasets, and all four whose representations are on the local drive, so counts and timings can be
checked without the cluster.

**Re-run the winning method per dataset at five folds before quoting any number.** Three folds is
enough to rank methods against each other. It is not enough to publish a delta.

### 9.11 `greedy_ablation` is cut. There are four admission methods, not five.

Decided by you on 2026-08-26: *"No greedy at all."* It sits inside the standing greedy decision,
which already covers greedy accept/reject as a feature-admission rule. The one-feature-at-a-time
sibling of it was deleted on 24 August and that deletion stands; this was the whole-block version.

**A very small budget would not have rescued it, and the reason is in the function rather than in
any argument about greedy.** Its cost is one forest fit per donor wider than `augmentation_budget`,
plus one scoring fit per donor, plus one baseline. That count is set by the **number of donors** and
does not fall when the budget falls. A smaller budget makes each fit cheaper, not fewer of them.
And running one method at a budget the other four do not use would make it incomparable with them,
which is the one thing this experiment exists to do.

It was also the largest single cost — 10,020 of herg_ki's 22,500 fits per placement and budget,
from 45 configurations (3 donor sources × 3 per-donor feature counts × 5 threshold values). Unlike
the margin, its threshold cannot be read off one run: a kept block changes what the next block is
scored against.

`greedy_augmentation_ablation` stays in `src/kirby/hybrid.py` and `create_hybrid` still dispatches
the name, so old recipes and saved results do not crash. Nothing selects it. Naming it explicitly
still runs it, and `c21` prints a warning saying it is cut and must not be reported as part of the
study.

### 9.12 Where the cost ended up — SUPERSEDED by §9.13

Same counting error as §9.7: augmentation fits only, base hybrid build counted as zero.



| | fits, herg_ki | node-days per dataset | days per array task |
|---|--:|--:|--:|
| as built: 14 datasets, 5 folds, 3 models | 446,760 | 55.9 | 14.0 |
| repeated work removed, and your cuts | 90,000 | 11.3 | 2.8 |
| `greedy_ablation` removed | **49,920** | **6.3** | **1.6** |

Roughly **38 node-days for all six datasets**, against 783 for the fourteen-dataset full factorial.
At 10.82 seconds per fit, which is the local upper bound; the server will be faster.

### 9.13 Allocation is an axis, and what the grid actually costs — 2026-08-27

**Allocation was missing and is now an axis.** f24 built every hybrid with one allocation method,
`weighted_global`, so it could not say whether the winning admission method survives the allocation
methods the search uses. It now sweeps the nine in
`find_best_hybrid.DEFAULT_ALLOCATION_METHODS`: `top_2`, `top_3`, `top_5`, `performance_weighted`,
`entropy_1.0`, `power_2`, `rank_linear`, `entropy_2.0`, `rank_exp_d0.5`. Not the 39 in
`NAMED_ALLOCATIONS`, and not the cut greedy family.

It applies to the multi-representation placement only. With one representation, 100% of the budget
goes to it whatever the allocation says, so the axis is collapsed there with that reason printed.

**The internal evaluator is NOT an axis, and that is a code fact.** `_evaluate_allocation` returns
from its model branch before it ever reaches the evaluator branch, and f24 always supplies a real
model. Two evaluators would give two identical numbers. Same finding as §3.5, re-checked.

**Importance is not an axis either.** Not for a reason — it simply was not asked for.

### What building a hybrid actually costs

Measured 2026-08-27 on flab_thermo_tresanco: 672 rows, 10 representations, 16,736 features, budget
500, one fold, random forest, local at load 7.8 on 8 cores.

| what | seconds |
|---|--:|
| feature importance over the pool | 22.5 |
| scoring the ten representations, one model fit each | ~240 |
| everything else, including the augmentation | ~74 |
| **one call to `build_fold_hybrid`** | **~336** |

`create_hybrid` redid all of it on **every call**. The grid calls it once per configuration, model
and fold — about 8,800 times for one dataset at one budget. None of that work depends on the
augmentation settings, and the representation scores do not depend on the allocation method
either: the allocation only turns them into a budget split.

### The fix, and the wrong version of it

**The engine now returns the scores it computed.** `allocate_performance_weighted` takes a
`score_sink` and fills it with the **raw** scores, before the non-negative shift — raw because
feeding shifted values back through `precomputed_scores` would shift them twice. `create_hybrid`
surfaces them on `feature_info` as `allocation_baseline_scores`. That is data, not a cache key, so
no stored result is invalidated. f24 harvests them from the first build of each fold and model and
hands them to every configuration after it.

| configuration | before | after | | features |
|---|--:|--:|--:|---|
| `performance_weighted` none | 214.7s | 17.8s | 12.1x | identical |
| `performance_weighted` capped | 341.7s | 72.5s | 4.7x | identical |
| `top_3` none | 473.0s | 20.5s | 23.1x | identical |
| `top_3` capped | 394.8s | 105.3s | 3.8x | identical |

**The first attempt at this was wrong and is recorded so it is not repeated.** It computed the
scores inside the experiment, on the outer fold's validation rows. `create_hybrid` scores them on
its own internal carve of the *training* rows. A `top_k` or `performance_weighted` allocation turns
those scores into which representations get budget, so the two disagreed and the selected features
changed — verified different on all four configurations, and reverted. Do not re-add a
recomputation; take the engine's own numbers.

**`build_fold_hybrid` was also ignoring `blueprint.rep_scores`**, which the engine's own
`apply_blueprint` forwards. Those scores came from the search and are on the right rows. A rebuild
without them could select different representations than the search did. Fixed.

### What this leaves the cost at

Unknown, and deliberately so. The counts in §9.7 and §9.12 were augmentation fits only. The real
figure is a per-configuration wall time on the multi-representation placement, and the only
measurement is local and under load.

`run_f24_augmentation_cost.sh` now times the multi-representation placement. It used to time the
single-representation one, where the allocation does nothing — it would have reported a healthy
number and said nothing about the placement that sets the wall. **Run it before the grid, and set
the grid's wall from it.**

`run_f24_augmentation_grid.sh` is currently 28 tasks per dataset: one single-representation task,
plus 3 budgets by 9 allocation methods. 168 tasks for six datasets. Split by dataset and by
allocation — every allocation still runs on every dataset.


### 9.14 Seconds per fit, measured on the server — job 12913493, read 2026-09-02

All six array tasks completed, 8 to 26 minutes each.

`run_f24_augmentation_cost.sh` times one fold of `capped__both__residual_correlation__ladder` on
the multi-representation placement, budget 500, `weighted_global`, `lightgbm`: 18 model fits per
dataset, representation loading excluded.

| dataset | seconds per fit | base-only fold spread |
|---|--:|--:|
| `abcov_kd` | 1.78 | 0.0215 spearman |
| `flab_thermo_tresanco` | 4.91 | 0.0274 spearman |
| `abagym_VEGF` | 7.45 | 0.0188 spearman |
| `herg_ki` | 15.28 | 0.0120 rmse |
| `buchwald_hartwig` | 42.74 | 1.6198 rmse |
| `logd` | 49.46 | 0.0220 rmse |

**Two things travel with the number.** It was measured with `lightgbm` alone; the grid runs
`lightgbm` and `rf`. And the first `build_fold_hybrid` of a fold and model pays for scoring the ten
representations, after which every configuration reuses those scores — the timing run divides that
one-off over 18 fits, a grid task divides it over one to two thousand. So each wall below is an
upper bound, and the size of the gap is not measured.

**What one array task costs.** A grid task is one dataset, one placement-and-budget, one
allocation, over 3 folds and 2 models. Fit counts are `--count-only` output from the same job.

| dataset | configurations | fits per task | task wall | `capped` alone | `uncapped` alone | `all` alone |
|---|--:|--:|--:|--:|--:|--:|
| `abcov_kd` | 159 | 9,288 | 4.6 h | 2.2 h | 2.0 h | 0.3 h |
| `flab_thermo_tresanco` | 163 | 9,528 | 13.0 h | 6.3 h | 5.7 h | 0.9 h |
| `abagym_VEGF` | 255 | 14,928 | 30.9 h | 15.5 h | 14.1 h | 1.3 h |
| `herg_ki` | 187 | 12,480 | 53.0 h | 25.7 h | 23.5 h | 3.8 h |
| `buchwald_hartwig` | 163 | 10,464 | 124.2 h | 59.6 h | 54.4 h | 10.3 h |
| `logd` | 163 | 10,224 | 140.5 h | 67.6 h | 61.7 h | 11.2 h |

The script's wall is `47:59:00`. Three of the six are over it as currently split, `logd` by a
factor of 2.9.

At 28 tasks a dataset the six come to **427 node-days**, against the 38 node-days §9.12 quoted
before allocation became an axis.

**The count is a floor.** `--count-only` prints the ladder run once per configuration and reads
every margin multiple off its rung scores; a multiple that picks a different block size gets its
own run. On synthetic data six multiples gave two outcomes, so the `capped` and `uncapped` rows can
be up to twice the fits above. Nothing else changes.

### 9.15 The nine allocation methods in the grid script are a stale copy — 2026-09-02

`run_f24_augmentation_grid.sh` sweeps `top_2`, `top_3`, `top_5`, `performance_weighted`,
`entropy_1.0`, `power_2`, `rank_linear`, `entropy_2.0`, `rank_exp_d0.5`, and `f24`'s
`ALLOCATIONS_SEARCHED` says those are `find_best_hybrid.DEFAULT_ALLOCATION_METHODS`. They are not,
as of `1ce281e` on 2026-09-01. That set is now five: `performance_weighted`, `power_2`,
`softmax_T1`, `rank_exp_d0.3`, `wglobal_power2`.

Two names are in both lists. All fourteen dispatch — every one is a key of
`hybrid.NAMED_ALLOCATIONS` or a base name `create_hybrid` branches on — so the grid runs either
way. At five it is 16 array tasks a dataset instead of 28, 96 for the six, and **244 node-days**
instead of 427.

Undecided. Which list the grid sweeps is yours.

### 9.16 The roster and settings of 2 September, chosen by you

Replaces the six of 26 August and the nine allocation methods of 27 August.

**Datasets — four.**

| dataset | domain | metric | rows | seconds per fit |
|---|---|---|--:|--:|
| `buchwald_hartwig` | reaction | rmse | 3,955 | 42.74, measured |
| `caco2` | molecular | rmse | 2,158 | not measured |
| `flab_expression_koenig` | antibody | spearman | 4,275 | not measured |
| `tcga_brca_pam50` | TCGA | f1_weighted | 900 | not measured |

`tcga_brca_pam50` is the first classification dataset this study has run. `caco2` and
`tcga_brca_pam50` were both among the eight dropped on 26 August. Dropped in turn are `herg_ki`,
`logd`, `flab_thermo_tresanco`, `abagym_VEGF` and `abcov_kd`, so antibody DMS and the n=141
antibody set are no longer covered.

`humanness_therapeutic` was proposed and dropped the same day. Two things about it, both found on
2026-09-02: its curated augmentation list in `DATASET_REPS` is empty, so only the borrow donor
source would run; and its metric is AUC while f24 scores every fit from `m.predict`, hard class
labels rather than probabilities. `_safe_roc_auc` accepts those without raising, so it would have
returned a number that is not AUROC. **This is an open defect in f24 for any AUC dataset.**
`f1_weighted` is correct off hard predictions, so `tcga_brca_pam50` is unaffected.

**Allocation methods — three, all quota.** Linear (`performance_weighted`), power-law squared
(`power_2`), and rank-exponential. All three resolve to `allocate_performance_weighted` with a
different weighting scheme. **Which rank-exponential decay is not yet chosen**: `rank_exp_d0.1`,
`rank_exp_d0.3` and `rank_exp_d0.5` all exist, the current default set uses 0.3, and the stale list
in the f24 script used 0.5.

**Budget — 250 only**, down from 500, 1000 and 2000. The grid can no longer say whether the winning
admission method depends on the budget.

**Downstream models — `lightgbm` and `rf`.** No change; that is already `DEFAULT_MODELS`.

**Internal evaluator — no effect, and the setting does not exist in f24.** Re-read 2026-09-02:
`hybrid._evaluate_allocation` fits whatever `model` it is handed and returns before the
`internal_evaluator` branch, and f24 always passes a real model. Same finding as §3.5 and §9.13.

**What this leaves.** 4 array tasks a dataset — one single-representation, plus one budget by three
allocation methods — so 16 tasks for the four datasets. The per-task wall is unchanged by any of
this except the budget, and `buchwald_hartwig` was 124 hours at budget 500 against a 48-hour limit.
How much budget 250 takes off is not measured. `run_f24_augmentation_grid.sh` is therefore NOT yet
edited: how far to split inside a dataset is still open, and the three missing seconds-per-fit
numbers are what should settle it.

### 9.17 The four-dataset roster, timed at budget 250 — job 8707403, read 2026-09-04

All four tasks completed on 2026-09-02, 9 to 23 minutes each, on `short` at 16 cores and 64G.
Same measurement as §9.14: one fold of `capped__both__residual_correlation__ladder` on the
multi-representation placement, `lightgbm`, 18 model fits — but at budget 250, not 500.

| dataset | seconds per fit | configurations | fits per task | fold spread |
|---|--:|--:|--:|--:|
| `caco2` | 13.00 | 283 | 18,240 | 0.0037 rmse |
| `tcga_brca_pam50` | 22.15 | 187 | 11,352 | 0.0305 f1_weighted |
| `buchwald_hartwig` | 31.31 | 163 | 10,464 | 1.6198 rmse |
| `flab_expression_koenig` | 66.41 | 163 | 9,768 | 0.0073 spearman |

**Budget 250 against budget 500, measured.** `buchwald_hartwig` is the only dataset timed at both:
42.74 seconds a fit at 500, 31.31 at 250. **27% cheaper**, not half.

**Row count does not predict the cost.** `flab_expression_koenig` has 4,275 rows and is the most
expensive of any dataset yet measured at 66.41 seconds a fit — worse than `logd` at 5,028 rows and
49.46. `tcga_brca_pam50` has 900 rows and costs 22.15, more than `herg_ki` at 1,412 rows and 15.28;
its representations are omics and far wider than a fingerprint. Any estimate made from row count
alone, including the one written on 2026-09-02 for this roster, was wrong.

**`caco2` has the widest configuration count yet: 283**, against 163 for the two datasets with a
smaller curated donor list. Its curated augmentation list holds five entries, so more per-donor cap
and donor-source combinations survive the degeneracy check.

**What one array task costs**, three folds and two models, against the 48-hour wall:

| dataset | whole task | `capped` alone | `capped` and one model |
|---|--:|--:|--:|
| `caco2` | 65.9 h | 32.8 h | 16.4 h |
| `tcga_brca_pam50` | 69.8 h | 34.3 h | 17.1 h |
| `buchwald_hartwig` | 91.0 h | 43.6 h | 21.8 h |
| `flab_expression_koenig` | 180.2 h | 87.2 h | 43.6 h |

All four are over the wall as currently split. Splitting by admission method fits three of the
four; `flab_expression_koenig` at 87.2 hours does not. Splitting by admission method and by model
fits all four, `flab_expression_koenig` at 43.6 hours with 10% to spare — and not at the doubled
ceiling, where it returns to 87.2.

At 4 tasks a dataset the roster totals **68 node-days**. That holds the whole task at the hybrid
placement's cost, so it over-counts the one single-representation task per dataset, which builds no
hybrid.

---

## 10. Stage 3 — what the grid measured. Jobs 8724201 and 8724202, read 2026-09-07

All 64 array tasks completed and all 64 result files are on disk. Four datasets, four
admission methods, three quota allocation methods, budget 250, 3 folds, 3 seeds, two downstream
models. **3 folds ranks methods against each other and cannot support a quoted delta** — every
number below is a count or a multiple of that dataset's own fold spread.

A configuration is counted as surviving only if all three hold: it took neither planted decoy, it
beat the no-extras control, and it beat its own matched random donor.

### 10.1 The counts

| dataset | configurations | took no decoys | beat no-extras | beat random | all three | clearing one fold spread |
|---|--:|--:|--:|--:|--:|--:|
| `buchwald_hartwig` | 1,120 | 274 | 492 | 218 | **0** | 0 |
| `caco2` | 1,443 | 491 | 506 | 547 | **122** | 33 |
| `flab_expression_koenig` | 874 | 429 | 573 | 440 | **107** | 40 |
| `tcga_brca_pam50` | 1,221 | 185 | 831 | 428 | **4** | 0 |

Per dataset, never pooled across them. Read with `tests/analysis/read_f24_stage3_summary.py`;
the per-file detail is `tests/analysis/read_f24_augmentation.py`.

### 10.2 `all` took a planted decoy in every configuration it ran

288 configurations across the four datasets, 0 of them clean. `all` adds the whole donor pool
without a test, and both planted donors — the Gaussian-noise block and the 5%-jittered near-copy of
a base feature — are in that pool. Option 1 of §7.5, "add everything, select nothing", is the null
the others had to beat; on the decoy test it does not pass.

### 10.3 `capped` and `uncapped` are not separable on this evidence

They survive in near-identical numbers on every dataset, and the top survivors appear in pairs.
`flab_expression_koenig`, `both__64__residual_correlation__frac0.05`, one fold, single placement:

| | features in | extras kept | lightgbm | rf |
|---|--:|--:|--:|--:|
| `capped` | 2,048 | 102 | 0.73265 | 0.72884 |
| `uncapped` | 2,150 | 102 | 0.73269 | 0.72879 |

`capped` is displacing: the width stays at the base representation's 2,048 while 102 extras go in,
so 102 base features were thrown out. `uncapped` keeps all 2,048 and widens to 2,150. The two
score 0.00005 apart against a fold spread of 0.0073 — 150 times smaller than the spread. `caco2`
behaves the same way at 317 features against 333.

This is the head-to-head §7.1 asked for, at 3 folds.

### 10.4 Where the survivors are

Every survivor clearing one fold spread is on the **single-representation** placement. On the
multi-representation placement at budget 250 the extras are a small fraction of the budget —
`tcga_brca_pam50`'s four survivors kept 1.7 features on average — and none of them clears its
dataset's spread.

Among the survivors that do clear a spread, `residual_correlation` is the candidate ranking in
every top-ranked one, `both` is the donor source in most, `frac0.05` is the commonest size rule,
and the margin multiple is 0. Best margins over no-extras: `flab_expression_koenig` 7.8x the fold
spread, `caco2` 3.0x.

### 10.5 What this run cannot say

`buchwald_hartwig` and `tcga_brca_pam50` produced nothing that clears their own fold spread, so on
those two the grid has no winner to report rather than a winner that lost.

The budget was cut to 250 alone on 2026-09-02, so nothing here says whether the picture changes at
500, 1000 or 2000 — and §10.4 is a statement about extras competing against a 250-feature budget.

Three folds. A winner per dataset needs a five-fold re-run before any of these numbers is quoted.

---

## 11. The candidate ranking is now four rules, not two — 2026-09-07

Stage 3 swept two ways of ranking augmentation candidates, and they were the only two the engine
implemented. `rf_importance` collapsed on `caco2` — 4 runs of 234 beat the base, against 42 of 257
for `residual_correlation` — so two options was not a comparison.

Four more were built on 2026-09-07 and two of them were dropped the same day. What is swept now:

| rule | ranks a candidate by | family |
|---|---|---|
| `residual_correlation` | correlation with what the base representation cannot explain | base-aware |
| `rf_importance` | random forest split importance against the labels | base-blind, tree |
| `lightgbm_gain` | LightGBM gain, the scorer the hybrid budget already uses | base-blind, tree |
| `mutual_info` | mutual information with the labels | base-blind, not a tree |

`rf_importance` and `lightgbm_gain` both read where a tree chose to split, so they cannot disagree
about a feature no tree splits on. `mutual_info` can.

### 11.1 The two deeplift rules were dropped

A block of pure random numbers is planted in every candidate pool, and a working rule never takes
it. Measured on `herg_ki`, curated donors, per-donor cap 64, rf, 3 folds, every rule ranking the
identical pool:

| rule | features kept per fold | of those, from the random block |
|---|--:|--:|
| `residual_correlation` | 16.8 | 1.1 |
| `rf_importance` | 14.7 | 0.8 |
| `lightgbm_gain` | 16.2 | 0.9 |
| `mutual_info` | 10.9 | 0.0 |
| `deeplift`, as first built | 14.0 | 9.9 |
| `deeplift`, after early stopping | 10.9 | 4.4 |
| `residual_deeplift`, after early stopping | 10.9 | 5.8 |

The random block is matched in width to the real donors combined and capped like them, so it is
roughly 14% of that pool. A rule choosing at random would take 14%. `residual_correlation` takes
6.5%; `deeplift` was still at 40% after the fix.

**What the fix was.** The network trained for a fixed 100 rounds of full-batch Adam with nothing
checking whether it was still improving, on about 940 rows against 340 candidate columns. It now
holds out 20% of rows, trains in mini-batches with weight decay, and stops when the held-out loss
stops improving. That halved the uptake and no more.

**Two explanations ruled out.** The network does learn — on a pool of that shape it reaches R²
0.62 on held-out rows. And the attribution is not broken — on synthetic data whose label is built
from 8 known columns, DeepLIFT ranks exactly those 8 first, the same as plain correlation does.

**Not tried.** The attribution is measured over every row, including the ones the network trained
on. Measuring it only on held-out rows would take memorisation out of the statistic.

The engine keeps both rules: `hybrid.CANDIDATE_RANKINGS` still lists all six and `--rankings
deeplift` still runs. They are not swept.

### 11.2 What the local test costs

1.46 seconds per model fit, measured on `herg_ki` here, 8 cores, single-representation placement.
That is the placement's own number — the 13 to 66 seconds in §9.17 were the hybrid placement,
which builds a hybrid on every call.

`uncapped`, per-donor caps 64 and unlimited, 3 folds, 2 models, real and random arms:

| scope | configurations | model fits | hours |
|---|--:|--:|--:|
| all three donor sources | 96 | 10,656 | 4.3 |
| `curated` donors only | 32 | 2,208 | 0.9 |
| `borrow` donors only | 32 | 3,744 | 1.5 |
| `both` donors only | 32 | 4,704 | 1.9 |

Per dataset. Split by donor source, every piece fits inside three hours locally.

**Warming one dataset's representation cache costs about an hour, once.** `herg_ki` was missing
four 3D representations and generated them in 3,962 seconds: `pharmacophore3d` 2,029,
`usrcat` 731, `coulomb_matrix` 687, `usr` 516. `caco2` and `flab_expression_koenig` will each pay
their own before any timing on them means anything.
