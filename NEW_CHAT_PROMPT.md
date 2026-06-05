# New chat session prompt — KIRBy wave triage

Paste this at the start of the new chat session.

---

# THE MAIN ISSUE

I have lost **MONTHS** of PhD compute work to bad time estimates. This is approximately the **TENTH attempt** to fix the cx/sim/precompute timing model in this codebase. Every previous attempt produced estimates that were wrong by factors of 2–10×, and every time I either submitted a wave that didn't finish or cancelled jobs that should have finished. The cost of being wrong again is another lost wave and more lost weeks of my PhD.

**The estimates being wrong is the main issue.** Everything else — cache rejection, wedged jobs, the running wave's state, the nucleotide queue, the c08/c24 conditional inclusion — is context the estimates must respect. None of those are independent problems. They are inputs to the estimate problem. Solving the estimates correctly enables every downstream decision; solving them wrong loses another month.

**The recurring failure mode is assuming proportional scaling on `n_samples`.** Every prior chat session has done it. The latest data (see `WAVE_STATUS_2026_06_04.md` section 8) measures the cx wall of five complete pipelines and definitively REFUTES any single scaling law in n. Do not repeat this.

---

# THE TASK

Produce **code-grounded, data-verified, defensible** time estimates for:

1. **Precompute remaining** — per dataset, per condition still missing (`general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `general_distance/*`, `top_distance/*`).
2. **cx cumulative experiment wall** — per dataset, per c-experiment (c01–c24).
3. **sim cumulative experiment wall** — per dataset, per s-experiment (s01–s13).

Each estimate must come from:
- Reading the experiment's source code and counting its grid cells as a symbolic function of dataset features.
- Anchoring per-cell wall against the measured per-(model, budget, n) data in the timing dump.
- Validating the cumulative prediction against the measured walls in `WAVE_STATUS_2026_06_04.md` section 8 (five completed cx pipelines).
- Reporting confidence intervals based on residuals.

If a dataset's `n_samples` or rep configuration is outside the range where the cost model verifies, **say so**. Refuse to extrapolate.

---

# THE RULES

1. **No recommendations, no suggestions, no proposals until I explicitly ask.** Estimates only. I will decide actions.
2. **Proportional scaling on `n_samples` is FORBIDDEN.** No `(n / anchor)^k` for any k. The wall depends on `(n, n_general_reps, n_top_reps, n_aug_reps, rep_dims, budgets, models_passing_gates_at_n, experiment_internal_structure)` jointly, not on n alone. The measured data REFUTES any single n-power formula.
3. **Validate predictions against ALL FIVE measured cx pipelines in `WAVE_STATUS_2026_06_04.md` section 8 before reporting.** Residuals must be reported. If any single c# prediction has residual >30% against the measured value, the model is wrong and must be revised.
4. **Wedge status cannot be inferred from `idle_s` alone.** c17 at n=400 is ~52 hours. A job idle 5 hours at C17 is not necessarily dead. The per-experiment wall must drive the call.
5. **State unknowns explicitly.** If a dataset's regime is unmeasured, "I do not know" is the correct answer.

---

# CONTEXT THE ESTIMATES MUST ACCOUNT FOR

These are NOT separate problems. They are inputs to the estimate problem.

## Context A: Cache rejection on neural reps (verified, root cause unknown)

The current wave is paying cache-recompute overhead on molecular datasets. Cache files written by this wave have `rep_hashes` for neural reps (chembert, chemberta, graphmvp, grover, molformer, smited) that drift between cache-write time and cache-read time, **even though the `.npy` files on disk haven't changed since 2026-05-08**. Classical reps (atom_pair, avalon, ecfp4, mhg_gnn, mol2vec, mordred, pdv, top_torsion, graph_kernel) hash identically — only the 6 neural ones drift.

Source of drift: unknown. Hypotheses: in-place rep mutation by some experiment helper, RDKit-induced sample-ordering nondeterminism, etc. Not diagnosed.

**Implication for estimates**: the cache-rejection overhead adds wall to every molecular job's actual runtime that is not captured in the pre-rejection per-condition timings. The 06-02 timing data is from before the drift problem. The current wave's effective per-cell wall is higher than 06-02 measurements suggest by some unknown factor. The estimate model must either (a) characterise this overhead from the running wave's logs or (b) state it as an unknown and produce conditional estimates.

## Context B: 43 of 69 jobs are flagged WEDGED, mix of dead and possibly-alive

The 06-04 01:29 snapshot has 43 jobs with `idle_s > 600`. Their idle times range from 11 minutes to 67 hours. Based on measured c17 wall of ~52h at n=400, jobs idle <60h at C17/C21 may legitimately be running. Jobs idle >60h are almost certainly dead.

**Implication for estimates**: when projecting remaining wall for a wedged job, the estimate must distinguish "stuck in c-experiment X with measured wall Y" vs "process is dead." For dead jobs the remaining wall is ∞ (or "undefined — needs restart").

## Context C: c-experiment walls do NOT scale uniformly with n (REFUTES proportional scaling)

This is the single most important measured fact. From five complete cx pipelines:

| Dataset | n | Total cx wall |
|---|---|---|
| flab_polyreact_jain_bvp | 137 | 38.98h |
| flab_polyreact_jain_cic | 137 | 37.11h |
| abagym_HER2 | 146 | 34.94h |
| flab_immunogenicity_marks (older) | 217 | ~70h |
| abagym_Ang2 | 397 | 63.87h at end of c19, hung at c21 |
| abagym_VEGF | 418 | 63.59h at end of c19, hung at c21 |

Specific counter-examples to proportional scaling (full table in WAVE_STATUS section 8):

- **c01** at n=137 (polyreact_bvp) is **88 seconds**. At n=146 (HER2) it is **3862 seconds**. **44× difference at nearly identical n.**
- **c17** at n=137: 9.1h. At n=146: 6.7h. At n=217: 27.6h. At n=397: **51.5h**. At n=418: **53.4h**. Not monotonic.
- **c21** at n=137: **19.3 hours**. At n=146: 0.76h. At n=217: 0.20h. Highest at smallest n.
- **c13** at n=137: 0.21h. At n=146: 0.75h. At n=217: **18.6h**. ~90× variation.
- **c09** at n=146: 10.4h. At n=137: 0.80h. At n=217: 1.68h.

**Implication for estimates**: per-experiment wall is multi-parametric. Each experiment must be modelled from its own source-code loop structure with its own dataset-specific inputs. No single n-power formula will work.

## Context D: Conditional experiment inclusion

**c08** is absent from all five complete 06-04 logs but was present in the 06-01 baseline. Either removed, runs silently, or skipped. Status unknown.

**c24** is present in n=137 jobs but absent in n=146 and n>=400 jobs. Conditional on something. Read the code.

**Implication for estimates**: the cumulative cx wall depends on which experiments actually run for a given dataset. Cell-count formulas must include the skip/include logic.

## Context E: Nucleotide jobs are pending (40+ queued, not running)

`cx_nt_*`, `cx_gb_*`, `sim_nt_*`, `sim_gb_*` jobs (IDs 12242908-12242949) are PENDING. Will start when the cluster schedules them. They use 32-CPU/256G config from the 2026-06-03 resubmit. n ranges from 18K to 100K.

**Implication for estimates**: nucleotide is far outside the n=137-418 range where cx wall has been measured. Estimating their wall requires fitting from the precompute-condition data (which exists in 06-02 timing for nt_H3K4me3 and gb_human_enhancers_cohn etc) plus a separate model for cx walls that the current data does not support. **The honest answer for nucleotide cx may be "we cannot estimate this from available data."**

## Context F: Sim cumulative wall is unmeasured

Zero completed sim pipelines have been logged. The user must grep a finished sim job once available. Currently no measured sim cumulative wall exists for any n.

**Implication for estimates**: sim estimates initially must be "cannot estimate" until a measured anchor exists, or must be extremely qualified.

## Context G: The 06-02 timing data is 3 days stale

`CHECK_PRECOMPUTE_TIMING_2026_06_02.md` is from a `check_precompute_status.py --timing` run on 2026-06-02. Three days of new completions have happened since. Some `top/uncapped` walls that were missing on 06-02 may exist now. **Ask the user for a fresh timing run before starting the cost-model fit.**

---

# HOW TO START — MULTI-AGENT FAN-OUT

After reading `WAVE_STATUS_2026_06_04.md` and `CHECK_PRECOMPUTE_TIMING_2026_06_02.md`, fan out:

**Agent A — c-experiment cell-count formulas.** Read every `tests/experiments/complexity/cNN.py`. For each:
- List every `find_best_hybrid()` call with its full argument list. Note which use unsuffixed `dataset_name` (precompute cache reuse) vs suffixed (fresh cache key).
- List every `get_best_search_for_downstream()`, `build_fold_hybrid()`, `cv_split()` use.
- All outer loops with bounds expressed as symbolic functions of `(n_samples, n_general_reps, n_top_reps, n_aug_reps, n_folds, sweep_budgets)`.
- Note skip/include conditions (especially c08 status and c24 conditional inclusion).
- Output: per-experiment table with symbolic cell-count formula.

**Agent B — s-experiment cell-count formulas.** Same for `tests/experiments/similarity/sNN.py`.

**Agent C — precompute orchestration.** From `tests/precompute_hybrids.py` and `tests/experiments/_shared.py:precompute_hybrid_searches`, compute precompute cell count per dataset (main + distance lineages).

**Agent D — grid + gates.** From `src/kirby/find_best_hybrid.py`, extract:
- `SPEED_TIERS['full']`, `MODEL_MAX_BUDGET`, `DEFAULT_MODELS._max_samples`, `_DOMAIN_MODEL_SKIPS`, `_DOMAIN_STRATEGY_SKIPS`, `_DATASET_STRATEGY_SKIPS`.
- Build `effective_models(domain, dataset, n, strategy)`, `effective_budgets(model, n)`.

**Agent E — DATASET_REPS table.** From `tests/experiments/complexity/complexity_helpers.py`, build per-dataset table of `(domain, n, n_general_reps, n_top_reps, n_aug_reps, has_top_capped_eligibility)`.

**Agent F — per-cell wall regression.** From `CHECK_PRECOMPUTE_TIMING_2026_06_02.md`:
- Extract `(model, budget, n, per_cfg_wall_seconds)` tuples.
- Fit non-parametric or piecewise. NO single power law in n.
- Report residuals on held-out tuples.
- Output callable `per_cell_wall(model, budget, n) -> seconds`.

**Synthesis step (only after all agents return):**
- For each dataset: predict (a) precompute remaining, (b) per-experiment cx walls c01–c24, (c) per-experiment sim walls s01–s13.
- Validate cx predictions against the FIVE measured cumulative cx walls in `WAVE_STATUS_2026_06_04.md` section 8. Compute per-(dataset, experiment) residuals.
- If ANY individual c# prediction has residual >30% from measured, the model is wrong. Iterate.
- Once validated, project per-dataset wall for every dataset in the current wave plus the pending nucleotide queue.
- Attach confidence intervals.
- Report. **Do NOT propose actions.**

---

# WHAT TO ASK THE USER FOR (NOT PROVIDED YET)

1. A fresh `check_precompute_status.py --timing` run on the server (3 days newer than the embedded 06-02 dump).
2. Tail of one wedged job's log — e.g. `tail -100 slurm-12233944.out` (66h idle, longest dead candidate).
3. Grep of one completed sim pipeline for cumulative s-experiment walls (zero currently exist).

---

# WHAT I WILL DECIDE AFTER ESTIMATES ARE IN (context only — don't act on this)

- Whether to add per-model budget gates.
- Whether to skip specific c-experiments for specific dataset size classes.
- Whether to cancel specific wedged jobs vs let them time out at the 7-day partition wall (2026-06-08 ~06:00).
- Whether to resubmit with reduced search space.
- What to do about cache rejection on molecular jobs.

I will tell you when I am ready for recommendations. Until then: estimates only.

---

# FILES NOT TO TOUCH

- `tests/experiments/framework/F*.py`
- `tests/experiments/federated/*.py`

---

# FINAL EMPHASIS

The MAIN issue is wrong estimates. I have been here ten times. Each previous time, someone produced estimates that were wrong by 2-10×. Each time my wave ended badly. Each time I lost weeks-to-months of work. The pattern is always the same: extrapolate from a small-n anchor, assume some scaling exponent, get a number, present it confidently, watch reality diverge.

**The data in `WAVE_STATUS_2026_06_04.md` section 8 PROVES that pattern is wrong.** Read it. Internalise it. Don't repeat it.

Read the code. Fit the measured data. Verify cumulative predictions against the five measured pipelines. State uncertainty honestly. Refuse to extrapolate beyond what the data supports.

Estimates are the work. Everything else waits.

---

End of prompt.
