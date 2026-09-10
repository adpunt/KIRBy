# Handoff — similarity hybrid representations

Paste this whole file into a fresh Claude Code session in the KIRBy repo.

---

## 0. How to talk to this user — read this first, it matters more than the work

In the session that produced this plan, **80% of the user's messages were them asking
for an explanation to be made comprehensible.** Not asking for work — asking what a word
meant. 22 separate times. They said they were considering cancelling their subscription
over it.

There is a memory file with a mechanical pre-send checklist:
`~/.claude/projects/-Users-apunt-repos-KIRBy/memory/feedback_explain_plainly_hard_rule.md`.
**Open it and run the checklist before every message.** The short version:

- Before sending, check every specific noun. Did the user use this word first, or is it
  in their code? If not, define it in the same sentence or delete it. **Never invent a
  name and then explain on top of it** — that is what caused the collapse.
- First sentence answers the question. No preamble.
- Cut to 10%. They measured it.
- No file paths or line numbers in chat as evidence. Describe the process step.
- "What is X?" gets the literal arithmetic, not the purpose.
- If they say they are lost: **stop, do not re-explain in different words, restart from
  zero.** Their confusion means the framing is wrong.

Second rule, equally load-bearing: **build what was asked, do not pad the turn with
analysis nobody requested, and never write an AI-authored verdict into their docs.** A
conclusion written into a results file is indistinguishable from a measured result six
months later. Record what was run and what came out. Not what it means.

---

## 1. What this work is, in plain terms

A similarity metric is a representation plus a comparison function — the field standard
is an ECFP4 fingerprint compared with Tanimoto.

The user's idea: a **small** set of features may compare molecules better than a large
set chosen to predict well. They want that built the way the rest of the framework
builds combined representations — with the budget-splitting strategies — but aimed at
comparison instead of accuracy, then tested against different comparison functions, then
tested on the jobs that actually involve comparing molecules, including **cleaning wrong
activity numbers out of the data**, which is the study's stated novel contribution.

Two grading yardsticks, always both, never merged into one number:
1. **activity read-across** — do the molecules the metric calls close actually have
   similar measured activity (scaffold-grouped nearest-neighbour Spearman)
2. **structural fidelity** — does distance track shared substructure (maximum common
   substructure overlap)

The point of two is anti-circularity: grade against measured activity or against
structure, never against another fingerprint.

---

## 2. The spec

**`SIMILARITY_HYBRID_DESIGN_2026_08_24.md` — read it in full before doing anything.**
It is the build spec. Do not rewrite it, do not summarise it from memory, do not
re-derive its findings. It contains:

- §2 what the code actually does today (10 verified facts)
- §3 ten prerequisites that must be fixed before any run
- §4 the six axes of the grid
- §5 which budget-splitting strategies to use and why
- §6 baselines and controls
- §7 what it gets judged on, and which existing experiments do the judging
- §8 datasets, including QM9's six obstacles
- §9 order of work
- §10 cleanup, ordered so nothing breaks
- §11 inherited open items from earlier plans
- §12 **ten open decisions that need the user's answer**

Other documents worth reading, in this order: `SIMILARITY_METRICS_ROADMAP.md` (the master
record of the whole thread), `SIMILARITY_STUDY_HANDOFF.md` (the earlier handoff and the
literature summary), `SIMILARITY_FRAMEWORK_INTEGRATION_PLAN_2026_08_21.md` (decisions
locked on 21 August — one of which this plan reverses, see §11 item 4).

**Warning: several claims in these documents were written by an AI and never measured.**
The plan's §2 and §3 were verified by reading code on 2026-08-24. Anything else, check
before repeating it.

---

## 3. First thing to do in the new session

**Do not start building.** Ask the user the ten decisions in §12 of the plan. They said
explicitly there is more back-and-forth before sign-off.

Then, in this order:

1. **Step 0 — commit.** `src/kirby/similarity/` is untracked and unstaged in a working
   tree with ~110 modified files and nothing staged. It is the only held-out-validated
   copy of the column-dropping procedure and any checkout would lose it. The engine
   change that separates similarity results from prediction results in the cache is also
   sitting uncommitted, and the same diff deletes a field from the stored-result identity
   — which invalidates every row already stored. That is decision 9.
2. **Prerequisites P1–P6** in §3 of the plan. Two of them matter most: the user's own
   selection method combined with a similarity target **crashes** today, and the
   in-memory result store is keyed such that a similarity request can be handed the
   **prediction** result with no error. The second does not crash — it returns a wrong
   number that looks right.
3. **Smoke gate.** Two folds. Assert the number of scored folds equals the number
   requested for every configuration — silent fold loss is the most likely way this study
   gets quietly corrupted. Assert scaffold groups are non-null: without them one of the
   two similarity targets scores exactly 0.0 for every arm and the run looks complete.
4. Everything else follows §9 of the plan.

---

## 3b. ⚠ Do not prune the search space

The user challenged the first draft of this plan for exactly this, and they were right.
**The default is full coverage of every axis.** This runs on the server; overnight
compute is free; an under-explored grid costs far more than a long one. The repo's whole
problem history is corner-cutting that produced results nobody could interpret.

Something may be dropped for **three reasons only**, and each must be stated in the
output: it is **degenerate** (verified by assertion, never assumed), it is **broken**
(fix it — only drop if the fix is genuinely out of scope), or the quantity is
**undefined** on that object. A theory about why something should lose is not a reason
to exclude it; it is an expectation to write down and test.

**No silent caps.** Any bound — top-N, sampled pairs, a skipped cell, a dataset that
failed a gate — gets logged in the result. A truncated run that does not say so reads as
complete.

If the grid must be split, split it **by dataset** so every dataset gets the whole grid.
Never split by axis; that leaves holes that look like results.

---

## 4. Hard constraints — non-negotiable, from the user's standing memory

- **No server access.** The user runs every server operation and pastes the output.
  Never ssh, never scp, never sbatch. JSON-only rsync.
- **Never bulk delete, cancel or reset.** The user lost a week of queue priority to a
  reckless job cancellation. Triple-check anything destructive and ask first.
- **No random splits, ever.** Scaffold-group-aware cross-validation only. (Note: the
  per-fold rebuilder currently violates this inside eight experiments — that is
  prerequisite P5.)
- **Baselines are individual representations, never averaged.** No mean baselines.
- **Representations are never in the cache key.** If the representation list changes,
  wipe and re-run. No versioning inside the cache.
- **The user is an engineer — no solo design.** Propose options, they decide. Never
  change the search space, the gates, or the engine without sign-off.
- **No heavy local compute** except the sanctioned similarity work. Import and syntax
  checks otherwise. The engine segfaults locally on the default importance path — local
  runs must use lightgbm for importance, evaluator and model, which means **any local
  result is pinned to one importance method and that is a hidden axis**.
- **Keep the documentation in sync**, and never claim something was done that was not.

---

## 5. State of the repo right now

- Branch `similarity-metrics-study`. ~110 modified tracked files, **nothing staged**.
- `src/kirby/similarity/` — untracked, four modules, never produced a result.
- The objective cache-key fix — uncommitted, unstaged.
- No result file on disk for the two budget-splitting studies (c31, c32) in this
  checkout. They may exist on the server; the user has results coming in.
- No representation cache exists for QM9 anywhere locally.
- The graph library will not import on this laptop (compiled-extension mismatch), which
  blocks checking one real QM9 bug — see §8.1 of the plan.

---

## 6. What was decided in the session that produced this plan

- The work goes into the **existing** suites (`tests/experiments/similarity/` s01–s13 and
  `tests/experiments/noise/`), not another standalone script. The user was explicit and
  had to correct the assistant on this.
- Budget sweep: the full list (10 up to 2000). Capping the similarity arm below the
  prediction arm's range would make the head-to-head unfair — that is decision 4.
- QM9 at a 4,000-molecule subset, with its representations cached to the external drive.
- Learned representations wait for the server.
- Two families of feature-picking get compared head to head and **one wins** — the
  framework's budget-splitting strategies versus the July study's direct structural
  column-picking.
- Importance weighting is an axis, in three places: during selection, inside the
  comparison function, and at final scoring (§4 Axis F).

## 7. What is still open

All ten decisions in §12 of the plan, plus the seven inherited items in §11. Nothing is
signed off yet.
