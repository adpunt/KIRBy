# Stage 2 kickoff prompt — define the augmentation grid

Paste everything below the line into a fresh chat.

---

We are on stage 2 of 4 of the augmentation workstream.

**READ FIRST, IN FULL, BEFORE DOING ANYTHING.** Do not grep it, do not skim it, do not summarise
it back to me.

1. `AUGMENTATION.md` in the repo root. It is the ONLY augmentation document. Four earlier ones
   were deleted on 25 August because they contradicted each other. Do not write a second one.
   Update `AUGMENTATION.md` in place.
2. `CLAUDE.md` in the repo root. Those are the rules for how you write to me. They were measured,
   not invented.

**THE QUESTION IS NOT "does augmenting work".** Assume it does. The question is **which specific
way of doing it works**, across my dataset roster. If exactly one way works, that is a complete
answer. A method losing on 20 of 30 datasets is not a reason to drop it if it is the best method
on the other 10.

**STAGE 2 IS A DESIGN AND PLUMBING JOB. DO NOT RUN THE GRID.** Stage 3 runs it on the server.
Deliverable is three things: a written grid in `AUGMENTATION.md`, the code changes that make every
axis actually reachable, and a measured cost estimate per dataset. Nothing else.

## What stage 1 already did — do not redo it

Six commits on 25–26 August, all pushed. The per-fold rebuild now selects features under the same
conditions the search did: scaffold grouping, real model, real metric, at all 36 call sites. A
crash on the default `top_3` allocation is fixed. The superseded fill-then-replace routine is
deleted. `capped` was rebuilt to displace base features using the same size ladder as `uncapped`.
Saved recipes now record which extra features each fold kept. The no-selection arm ranks on
training rows only. Unknown strategy names raise instead of silently producing a plain hybrid.
Two settings were added: `augmentation_margin` (how much better than keeping none a block must
score) and `augmentation_fraction` (size the extras as a fraction of the budget instead of the
fixed 0/5/10/20/40 ladder). Both default to the old behaviour.

## The axes that exist and are reachable today

- **Where extras go**: the best single representation at full width, or the multi-representation
  hybrid at a budget. These are use case B and use case A from the roadmap. Both are needed —
  Scenario 2 and Scenario 1 respectively.
- **Donor source**: the curated `augmentation` representations; the top-5 features of each other
  base representation; or both.
- **Admission method**: `none`, `all`, `greedy_ablation`, `capped`, `uncapped`. Five. That is the
  real count — the earlier inventory of eleven counted a defect, two mechanisms that decide what
  gets *offered* rather than kept, a duplicate out of tree, and a deleted ancestor.
- **Size rule**: the fixed ladder, or `augmentation_fraction`.
- **Margin**: `augmentation_margin`. Only meaningful with the ladder; a fraction fixes the size so
  there is nothing to accept or reject.
- **Candidate ranker**: `rf_importance` (judges a feature by how well it predicts alone) or
  `residual_correlation` (judges it by what the base cannot already explain). **This axis has
  never been varied in the shipping engine.** No caller has ever set the first one.
- **Per-donor cap** (`aug_per_source_budget`): 5, 64, or uncapped.
- Budget, allocation method, downstream model, dataset.

## Three things that are NOT reachable, and are stage 2's plumbing work

1. **The candidate cut-off is frozen at 50.** No caller anywhere can change it, and it is not in
   the stored result's identity at all. If it is to be an axis it needs exposing and adding to the
   key. Adding a key setting means deleting every `result_db/*.db`; the cache is empty right now,
   so do it now or never.
2. **`capped` is gated to datasets of 250 rows or fewer** by the precompute driver, which prints
   "SKIP capped ... projects past 7-day partition". **That gate is now stale.** It was set because
   the old `capped` refitted a model and ran permutation importance for every candidate. The
   rebuilt `capped` does one ranking fit plus five scoring fits per fold. Re-measure the cost and
   change the gate, or `capped` cannot be tested on a single molecular dataset — and `capped` is
   the strategy with the best evidence behind it (§5.1 of `AUGMENTATION.md`).
3. **Augmentation is not a search axis.** Every augmentation setting sits outside the search grid
   and is spliced identically into every configuration, so varying one means a separate outer call
   from the driver. The driver currently emits one condition per (pool, strategy) pair. It will
   need to emit the grid you design.

## Controls that must be in the grid, not optional

- A **matched-shape random donor** on every cell. Adding features changes model capacity by
  itself. Without this control a win is uninterpretable.
- A **planted pure-noise donor**. Anything that keeps those features is broken.
- A **planted near-copy of a base feature**, lightly jittered. Only that decoy catches a rule that
  rewards duplicating what the base already has. No amount of pure noise reveals it.

## Rules I will hold you to

- **Do not prune the search space for compute.** Cutting an axis, arm, budget or dataset needs a
  stated reason: degenerate, broken, or undefined. A theory about why something will lose is not a
  reason. If a grid must be split, split it by dataset, never by axis. Never cap one arm below its
  comparator's range.
- **Do not write a substitute harness.** Work goes into the existing experiment suites. Seven
  throwaway scripts already exist from earlier attempts and none of them produced a result I could
  use.
- **Never average across datasets.** Report per-cell win counts. A mean rank across datasets hides
  exactly the thing I am looking for.
- **Nothing runs locally for more than 3 hours.** Estimate wall time before starting and tell me.
  Over 3 hours means build a server job, not cut an axis.
- **I am an engineer and I decide.** Offer options with the numbers attached. Do not design alone
  and present it as settled.
- **No claim without having just read the thing.** Not memory, not a document from an earlier
  session. Open the file in this session.

## Decisions I owe you, which you should put to me as a short list with numbers attached

- Which 10 datasets. Only 4 have representations cached locally: hERG, logD, buchwald, and one
  antibody set. All three measured so far are flat-topped, meaning no single representation
  dominates, so **Scenario 2 currently has no testbed anywhere.** That needs solving in the
  dataset choice.
- Which axes are real axes and which are fixed at one value.
- Whether the candidate cut-off becomes an axis, given that adding it means wiping the cache now.
- Which values `augmentation_fraction` and `augmentation_margin` sweep.

## One measurement stage 1 left unfinished

Whether sizing extras as a fraction beats the fixed ladder, on real data. The run was started on
hERG on 26 August and abandoned: the machine was running four of my own experiments plus a Rust
build, load average 55, and it was getting 3.7% of one core. Either re-run it when the machine is
free, or fold it into the stage 3 grid as an axis. Do not skip it silently.
