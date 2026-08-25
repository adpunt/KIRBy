# Augmentation — the single source

**This is the only augmentation document.** It replaces and supersedes four files, all deleted
on 2026-08-25 after being read in full and folded in here:
`AUGMENTATION_STAGE_DESIGN_2026_08_19.md`, `AUGMENTATION_STAGE_DESIGN_2026_08_21.md`,
`AUGMENTATION_FINDINGS_AND_HANDOFF_2026_08_24.md`, `AUGMENTATION_COMPLAINT_LEDGER_2026_08_25.md`.
The brief itself stays where it is, in `HYBRID_REBUILD_ROADMAP_2026_07_15.md` Phase 4.4 and
`HYBRID_REBUILD_PROMPTS_2026_07_16.md` Chat 6, because both cover the whole rebuild and not just
this workstream.

Every code claim in §3 and §4 was re-read in the source on 2026-08-25 and is marked
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
| 1 | Rank the pooled candidates once, then choose a block size from a fixed ladder (0, 5, 10, 20, 40); ties keep nothing | `uncapped` — **the default** | yes | rebuilt 08-24. §4.5 |
| 2 | Fill spare budget one feature at a time, then evict base features to make room | `capped` | yes, for n ≤ the capped threshold | **every original defect intact.** §4.2 |
| 3 | Keep or drop each donor's top-20 block whole | `greedy_ablation` | no — framework experiments only | order-dependent, base-blind, ratchets |
| 4 | Concatenate every donor's top 20, no test at all | `all` | no — one archived script | **ranks against the full label vector.** §4.3 |
| 5 | Off | `none` | yes | the control arm; no defects of its own |
| 6 | The old add-one-at-a-time loop that ratcheted | (former `uncapped`) | **removed 08-24** | **produced every augmentation result previously on disk.** Those numbers came from code that no longer exists. |
| 7 | A second fill-then-replace routine, ~290 lines, in the shipping module | — | **unreachable**, no dispatch branch | §7.4 |
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

### 3.5 The knobs

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

### 4.1 The 24 August fix is inert. **Verified. This is the blocker.**

The per-fold rebuild helper in the experiment layer gained four optional settings on 24 August —
the scaffold grouping, the real downstream model, the real metric, and the metric direction. It
forwards all four correctly.

**Of its 36 call sites across the whole tree, zero pass any of them** — including the
single-representation augmentation experiment itself. With the grouping absent, the admission
holdout falls back to a fixed-seed 20% **random** split of a scaffold-grouped dataset. With the
model and metric absent, every keep-or-drop decision is judged by an internal lightgbm R² score
instead of the metric the result is reported on.

So the two defects named across three documents as the most likely cause of the whole problem are
**completely live in every experiment**. Nothing measured through the experiment layer since
24 August reflects the fix.

### 4.2 `capped` still has every original defect, and it still ships. **Verified.**

It is emitted by the precompute driver as a standard condition for every dataset under the size
threshold, and about 14 rows of the per-dataset policy table name it.

- The acceptance bar is a running maximum, so early accepts raise it permanently — the exact
  ratchet the default path was rebuilt to remove.
- **The per-source cap is never forwarded to it**, so a 2,560-wide donor competes against a
  40-wide one on feature count alone — while the cap still changes the stored result's identity,
  so two runs that behaved identically are filed as different results.
- Its eviction bookkeeping is wrong after the first swap: the replacement is written into the
  base slot but the base count is decremented, so an already-admitted extra can be evicted again
  as if it were a base feature, and a genuine base feature at the tail becomes unreachable.
- Under shipping allocations the budget is already spent when it starts, so its fill phase never
  runs — in practice `capped` means "evict only".

### 4.3 `all` ranks against the full label vector. **Verified.**

Every other path ranks on the inner-training rows. This one ranks each donor's top 20 on the
**complete** label vector, including the rows it is then scored on. It matters because "add
everything, no selection" is the null that real selection has to beat, and as implemented that
null is not honest.

Note also that the name is wrong: it is not "all", it is top-20 per donor.

### 4.4 The dispatcher has no fallback branch. **Verified as latent, not live.**

The chain over the four strategy names has no final `else` and raises nothing. An unrecognised
string yields a base-only hybrid while the strategy name is still written into the stored recipe
as though it had been honoured. The allocation dispatcher a few lines earlier does raise and list
its valid names.

I checked every strategy string set anywhere in the tree: all of them are among the four the
dispatcher knows. **So this is a trap, not a live wrong result** — a typo or a renamed strategy
would fail silently.

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

### 4.7 Nothing records what was kept. **Verified.**

The stored recipe carries the strategy and the per-source cap but cannot reconstruct the pool or
which features survived. **No artifact anywhere records which augmented features were admitted.**
A win cannot be explained or honestly re-scored later.

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

### 5.4 Measurements deliberately dropped

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

### 7.1 Your 20 August decision contradicts your 21 August post-mortem. Only you can settle it.

On 20 August you decided: rank once, cut at the cap, no accept/reject loop, no inner holdout.
On 21 August your own post-mortem of attempt 1 listed *"the existing loop was switched off and
replaced with a home-made rank-once"* as failure mode number one. Those are opposites, a day
apart, and nothing reconciles them. On 24 August the rank-once substitution was made **in the
shipping engine**, inside a commit whose subject line is about removing tiers, so it is invisible
in the history.

**Proposed resolution: settle it by inclusion, not by argument.** Both the ladder and the
one-at-a-time paths become arms of the stage-2 grid and the data decides. That costs one axis and
removes the need for either of us to be right.

### 7.2 The 24 August fix has to actually reach the experiments

Zero of 36 call sites pass the grouping, the model or the metric. Until they do, every
augmentation result the experiment layer can produce is selected on a random split and judged by
the wrong score. This blocks stage 2 — running the grid before this lands measures the defect,
not the method.

### 7.3 Percentage-of-budget sizing exists only in a scratch script

Your 20 August decision. The engine has a hard-coded ladder of absolute counts with no parameter
and no override, and the only implementation of the percentage design is the 21 August script.
For the grid in stage 2, the sizing axis has to be expressible by the engine or the grid cannot
include it.

### 7.4 What happens to `capped` and to the unreachable routine

You said on 24 August to remove the dormant code. Neither candidate was removed, and the reason
matters:

- **`capped`** is not dormant. The precompute driver emits it, the per-dataset policy table names
  it for about 14 datasets, and a federated experiment reads it off saved recipes. Removing it
  means removing it from those places too. It has never been compared against the default path,
  and a comparison today would be unfair to it because it still contains the broken loop.
- **The unreachable ~290-line routine** genuinely has no dispatch branch. The framework
  experiment that shares its name keeps its own private copy, so deleting the library version
  breaks nothing. This one is a clean delete whenever you want it.

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
