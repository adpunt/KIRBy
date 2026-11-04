# Handoff: build the Paper 3 noise grid decisions, and review the noise mitigation grid

Written 2026-11-04. Paste everything below the line into a fresh Claude Code chat in
`~/repos/KIRBy`.

---

Read `~/repos/KIRBy/CLAUDE.md` first and follow it in every reply. Then read `MEMORY.md` in the
memory folder.

**`PAPER_3_NOISE_EXPERIMENTS.md` is the plan.** Read all of it before anything else. Where this
file disagrees with it, it is right. This file only says where the work touches the code, from a
reading on 2026-11-04. Re-read the code; line numbers move.

Adelaide is the engineer and decides. Nothing beyond that document's section 1 is decided.

This chat has two jobs. Part A builds what she has decided. Part B reviews everything inside the
noise mitigation grid with her, which is where she expects the largest cuts: "those should be
massive cuts. Right now I'm throwing everything at the kitchen sink for noise."

# Part A: build the decisions

## Before writing code

Bring these to her, one per line, each with what changes either way:

1. Do the cuts and `base_1` and `hybrid` only apply to n43, n45, n46 and n47, or n44 only?
2. Does `hetero_seen` keep levels 0, 0.1, 0.3, 1.0, or run level 1 only?
3. How does she name the results for the second run?
4. What time limit should the flags report against?

Then give her your plan for both parts, before you build anything. Say which order you recommend.
Time flags give Part B its timing data, and Part B's cuts change what the rest of Part A has to touch.

## What to build

**Model inputs.** `n44.MODEL_INPUTS` is `('base_1', 'base_2', 'hybrid')`. `n45` reads
`tuple(n44.MODEL_INPUTS)`, so a change to n44 changes n45 too, whatever she answers on scope. `n46`
has its own `MODEL_INPUTS`. `n43` and `n47` name these inputs in their own code.

**Noise kinds.** Two places list them, so one cut has two homes:
- `DECIDED_REGRESSION_CONDITIONS` in `noise_helpers.py`, with the assert right below it and
  `SCREEN_DROPPED_CONDITIONS`, holds `sparse_dprime_up_p10`. Many experiments read it.
- `MODEL_ERROR_CONDITIONS` is in both n44 and n43, and holds `dense_oofmse`, `hetero_seen`,
  `hetero_hidden`.

Keep `sparse_dprime_up_p10` usable by anything out of the run list that still needs it. Check who
reads each list before changing it.

**Results first, then checks.** Today `run_unit`'s `emit` passes `exempt=u.cliff_train is not None`, so
`n42._Unit.emit` writes both put-backs on every row. `n07.dprime_rows` writes twins whenever
`controls_enabled()`. The first run needs neither. The second run needs a way to run them only for
the results she names. Check two things:
- `controls` is in the `.units` checkpoint fingerprint, so switching it may throw away finished pieces.
- Rows of the first run must stay readable beside rows of the second.

**The same thing must not run twice.** Section 6 lists level 0 and `reliability_weight`, and she wants
others looked for. Compare what each piece asks for by training set, not by name, and report what you
find.
- Level 0: every kind's comparison reads level-0 rows, and rows go to one file per kind. So they need one
  shared place, and every reader needs to know it. `inject_condition` returns the delivered labels at
  level 0.
- `reliability_weight`: reached in `run_unit`'s neighbour loop, once per scored detector, and it ignores
  the scores. It reads `reads: 'none'` in `CLEANUPS`.

**Time flags.** Section 8. Time each detector's scoring, cleanup and model fit. Report only; never drop
anything automatically. Remember that cached fits and refused results take almost no time, and say which
is which.

## What not to touch

- 3 draws, 5 folds and the dataset's two decided models.
- Detectors, cleanups, their settings and noise levels, until she decides them in Part B.
- The cliff definition. That is `HANDOFF_AC_STUDY.md`, and she is taking the cliff grid next.
- The results in the server folder `paper3_grid_size/n44_logd`. It is a measurement on the old grid.

# Part B: review the noise mitigation grid

Her words: "I am not sure that I want to use all the detectors, this needs to be looked over based on
preliminary results. Everything in here needs to be reviewed more thoroughly." And 2026-11-04: it should
look "at more than just detectors, but everything in that noise mitigation grid".

**The grid is section 5 of `PAPER_3_NOISE_EXPERIMENTS.md`, and every axis in it is up for review:**

| Axis | Today, n44 | Where |
|---|---|---|
| Detectors that give a score | 32 | `REGISTRY_DETECTORS`, `RESIDUAL_DETECTORS`, `N41_DETECTORS`, `CONTROL_DETECTORS` |
| Neighbour detectors | 4 | `NEIGHBOUR_DETECTORS` |
| Flag detectors | `conformal_bh` 6 settings, `two_learner_small_loss` 3 | `CONFORMAL_FLAG_SETTINGS`, `n32.ITLM_ALPHAS` |
| Oracle flags | 3 shares x 5 precisions | `FLAG_SHARES`, `PRECISIONS` |
| Cleanups | 26 | `CLEANUPS` |
| Cleanup settings | Flag shares 0.05, 0.1, 0.2; strengths per cleanup; 3 winsorise values | `cleanup_settings`, `STRENGTHS`, `WINSOR_C` |
| Detector feature space | Own, plus hybrid on a base input | `run_unit` |
| Neighbour settings | 3 kept per piece from the screen; each with k and a counting scheme | `screen_top`, `n42._Similarity` |
| Which detectors pair with which cleanups | Pairing rules | `_pairs_with` |
| `gp_smooth` | Refused above 1,000 training molecules | Section 6 of the plan |

**What to read:**
- `~/Documents/ch7_noise_methods_audit.html`, section `#s6`: her two lists of methods that raised the top
  score. `HANDOFF_WHAT_WORKED.md` holds its outcome.
- Thesis Chapter 7, "Detecting and handling label noise" in `thesis.tex`, Results section.
- `/Volumes/seagate/kirby_results_2026-10-10b/noise`: the result files behind it.
- `thesis_provenance/METHOD_PROVENANCE.md` and `NOISE_CODE_FIX_DECISIONS_2026-10-26.md`, for where each
  method came from and which code changed since.
- `tests/experiments/noise/ROBUSTNESS_CHECKS.md`.

**How to do it:**
- Map every axis against the results first. Do not dive into one and leave the rest.
- "Worked" means raised the top score. The random comparison and QM9 support it and do not sort methods.
  Do not change that definition without telling her.
- Give score pairs in the dataset's own metric, not tallies, and no significance tests.
- Those results come from older code, mostly on fold 0, with settings cut by hand for the thesis. Say so where it
  matters. A method that never ran is not a method that failed.
- Say which entries have no results, and which are the same training set under another name.
- Offer cuts with what each saves in results and what is lost, from the plan's section 4 arithmetic. She decides.
  Bring at most three decisions at once, one per line.
- Her cuts go into `PAPER_3_NOISE_EXPERIMENTS.md` as soon as she makes them, with her words. Mark anything of
  yours as unreviewed, and never write a verdict in as a finding.

**Connected:** `HANDOFF_AC_STUDY.md`'s `cliff_aware_cleanup` waits on her best detector and best cleanup, so
tell her if the review changes those. The cliff grid itself comes after this, and is not this chat's.

# Rules for this work

- Never ssh to the server. She runs every server command: one `srun ... --pty /bin/bash` line, then plain
  commands inside it. No sbatch. Interactive sessions last at most 4 hours. No `rm` in anything she pastes.
- No patient data.
- Update `_smoke_n44.py` and the other smoke tests for what changes, and run them. `PAPER_3_NOISE_EXPERIMENTS.md`
  section 4's counts should match the new code; re-run `scratchpad/paper3_grid_counts/` and say so if they don't.
- When something is done, update `PAPER_3_NOISE_EXPERIMENTS.md`, so it stays the record.
