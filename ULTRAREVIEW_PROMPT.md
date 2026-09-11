# Ultra-review: KIRBy framework and experiment suite

Paste everything below the line into a fresh Claude Code session in this repo.

---

Audit this repository for **logic bugs** — measurements that run to completion and produce a
number that does not mean what the file says it means. Implementation bugs (crashes, wrong
types, unhandled exceptions) are in scope but secondary; a crash announces itself, a wrong
comparison does not.

Read `CLAUDE.md` at the repo root first. Every finding you write must follow it.

## Scope

**The engine** — `src/kirby/find_best_hybrid.py` (10,237 lines), `src/kirby/hybrid.py` (6,230),
`src/kirby/similarity/` (7 modules), and the suite's shared layer
`tests/experiments/_shared.py` (6,275).

**The framework** — `tests/experiments/framework/`, 40 files, f-numbered. These are build
decisions: which allocation method, which importance method, which weighting, which distance.

**The suite** — 174 files that are the publishable work:

| directory | files |
|---|---|
| `tests/experiments/complexity/` | 35 |
| `tests/experiments/noise/` | 25 |
| `tests/experiments/similarity/` | 17 |
| `tests/experiments/similarity_metrics/` | 34 |
| `tests/experiments/federated/` | 12 |
| `tests/experiments/opig_*.py` (repo root of `experiments/`) | 11 |

Out of scope: `tests/experiments/annitia/`, `tests/archive/`, anything under `tests/analysis/`
unless a suite file's number is read wrong there.

## What a logic bug is here

These are the shapes that have actually been found in this repository. Hunt for more of each,
and for shapes not on this list.

**Two things compared that differ in more than one way.** The finding is the *count* of
differences, not the first one. An arm built by hand ranked by `lightgbm_gain` against an arm
whose ranking came from a search that may have chosen `deeplift` differs in two ways at once,
so the difference it reports cannot be read.

**A subtraction across two models.** One side fitted with random forest, the other with
lightgbm, difference reported as an effect.

**A fraction whose top and bottom live in different feature spaces.** A recovery percentage
where the ceiling was measured in features chosen from clean labels and the damaged arm in
features chosen from noisy ones.

**A comparator that is a maximum.** A baseline picked as "whichever representation and
neighbour count scored best" on the same numbers it is then compared against. The other arm has
to beat a maximum over many choices, not one alternative.

**Selection that saw the rows it is scored on.** A recipe, a representation subset, a top-3, a
threshold, or a feature ranking chosen on all rows and then scored on a held-out fold of those
same rows. Check both directions: the arm may be honest and its *comparator* not.

**Fold construction that differs between the search and the scoring.** Both must build folds the
same way — same splitter, same `n_splits`, same seed, same `task_type`. A regression path using
`GroupKFold` and a classification path using `StratifiedGroupKFold` diverge silently when
`task_type` is not threaded through.

**Rows in one matrix that came from different spaces.** Per-fold feature selection glued into
one matrix, checked only for having the same *number* of features. Column 7 then holds a
different feature depending on which fold the row came from.

**A row that restates its own input.** A score of quantity A against quantity B where both are
computed from the same distance matrix. It can only come out near 1.

**A degenerate arm reported as a result.** A threshold that puts every molecule in one class. A
fold whose subset is 2 rows, on which Spearman is ±1. A control that is constant.

**A hardcoded list that predates a change.** Representation lists, model lists and dataset
rosters typed into a file rather than read from the one place they live. Check every such list
against its source and against the pruning log — three names in one file were cut in August for
correlating 0.94–0.98 with a sibling that was kept.

**Metric direction.** `lower_is_better` stamped next to a metric name the registry fills in the
other way. Ranking by a difference whose two sides have different baselines. Averaging a
quantity that is only comparable within a group.

**A quantity that stops growing while its denominator keeps growing.** Information per feature
where the estimator caps at 50 principal components — comparable between arms at one budget,
meaningless across budgets, and an average across budgets mixes the two.

**A silent substitution.** A lookup asked for a recipe that does not exist returning a different
one and writing it into the result file under the requested name.

**A cache key missing something that changes the answer.** A resume fingerprint that records
rows, labels, groups and folds but not the setting the run was varied on, so the second run is
skipped and the numbers stay on the first setting.

## Cross-reference the finalization documents

Three documents describe what these files are supposed to do:

- `EXPERIMENTS_FINALIZED.md` — every one of the 116 in-scope suite files. Per entry: **What it
  asks**, **What it computes**, **What comes out**, **Hybrids**, **Still open — work, not a
  decision**, **Decision — mine**, and either `@@` (an open question awaiting the owner) or
  `**DECIDED — NOT YET IMPLEMENTED**`.
- `FRAMEWORK_FINALIZATION.md` — the framework, §3 open decisions and §4 what was implemented.
- `EXPERIMENTS_CLOSED.md` — 104 findings whose fix is claimed to exist on disk. Its own header
  says the 104 are **not independently verified** and that one such claim was wrong.

Use them three ways.

**Do not re-report what is already recorded.** A finding already written under *Still open* in
that file's entry is known. Only report it if what the entry says is wrong or incomplete — say
which part.

**A document claim contradicted by the code is itself a finding.** Report it as a finding about
the document, naming both sides. Two examples that were real: an entry said a file falls back to
ridge when the code already fell back to lightgbm; a module docstring quoted three numbers that
the underlying measurement does not support.

**A `Fixed` label in `EXPERIMENTS_CLOSED.md` is a claim, not a fact.** Spot-check it. If the
change is not on disk, that is a finding.

**What it asks vs what it computes.** For each file, read its entry's **What it asks**, then
read the code, then answer: does what it computes answer that question? A file that computes
something adjacent and reports it as the answer is the highest-value finding in this audit — one
asks whether a filter finds label errors and reports what fraction of flagged molecules are
activity cliffs, which is real chemistry, not a mistake.

## Do not re-derive these

Settled. Re-raising any of them wastes the pass.

- **Greedy allocators are cut.** All of them, including `dominant_greedy`. Not a bug, not a gap.
- **SAC is cut** — as an objective and as the budget split.
- **Ridge is banned as a downstream evaluator** except where the owner has approved it: c16
  (approved), c15 and c13 (pending approval), c08 (deliberate mismatched-model control). A ridge
  used as a *measuring instrument* — fitting one principal component to measure reconstruction —
  is not a downstream model and is not in scope.
- **Search-space cuts need a stated reason**: degenerate, broken, or undefined. Never propose
  cutting an axis, arm, budget or dataset for compute. Server compute is free.
- **`replicate` is QM9 only.** The other three datasets have folds.

## How to report

Report through the `ReportFindings` tool, most severe first. For each finding:

- **Name both sides and count the disagreements.** "The first four are written out by hand and
  rank features by lightgbm_gain. The fifth uses whatever the search picked, which may be
  deeplift. So the fifth differs from the other four in two ways at once, not one."
- **No fix in the finding.** State what is broken. Nothing about what to do.
- **A concrete failure scenario**: what input or state, what wrong number comes out. Not "could
  lead to incorrect results".
- **Every number carries its unit** — features, molecules, folds, configurations, model fits —
  and every fraction says what the top and bottom count before the number appears.
- **No line numbers or function names in the summary sentence.** Put the location in the `file`
  and `line` fields where the tool expects it.
- **Plain language.** No invented terms. No acronym the owner has not used. Never write
  "read-across" — say "predicting each molecule's activity from its five nearest neighbours".

Severity order: a number that is currently wrong and quotable beats a number that would be wrong
under a setting nobody uses, which beats a file that cannot start.

## Method

1. Read `CLAUDE.md`, then the three finalization documents' structure — not all 600 KB, but
   enough to know what each entry claims.
2. Read the engine and `_shared.py` first. Most suite-level logic bugs are one caller passing
   the wrong thing to a shared function, and you cannot see that without knowing what the shared
   function does with it.
3. Fan out over the six directories. One agent per directory is too coarse — split by the bug
   shapes above, so each agent sweeps every file for one class of defect and can compare files
   against each other. A leak in one file is usually a leak in four.
4. **Verify adversarially before reporting.** For each candidate finding, a separate agent tries
   to refute it by reading the code again. Anything it cannot confirm from the code as it is on
   disk right now is dropped, not softened. No finding survives on "the documentation says".
5. Say what you did not cover. If a directory was sampled rather than swept, or a claim was left
   unverified, name it. A silent gap reads as coverage.

Every claim must come from code read in this session. Not from memory, not from a summary, not
from a document another session wrote.
