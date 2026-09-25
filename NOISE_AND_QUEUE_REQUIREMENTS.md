# Your instructions, each one checked — and how to run it

Written 2026-09-25. Every status was checked by reading the code in this session, not taken from an agent's report.

**Nothing below has run anywhere.** On the laptop, the only checks were that every file compiles: 132 Python files, 0 failures, plus `bash -n` on every shell script. The server checks in Part 2 are the first real run.

**None of it is committed yet.** The server pulls from GitHub, so it sees none of this until you say "commit".

What each status means:
- **BUILT**: the code does it and compiles, but it has not run.
- **PARTLY**: part of it is built; the missing part is named.
- **NOT DONE**: not built, and the reason is given.
- **YOURS**: waiting on a decision from you.

---

## Part 1 — Prompt 1, instruction by instruction

**1. "try all of these with different types of noise (noise conditions), and for those which rely on similarity, different mechanisms of measuring similarity"**
BUILT.
- Every noise experiment that finds neighbours now takes a similarity setting. It loops over `similarity_space.settings_from_env()` and k = 5, 9, 10 and 20.
- The noise conditions come from each experiment's existing condition sweep.
- By grep, 23 files call the similarity sweep. The files with no neighbours at all are n15, n16, n19, n26, n27, n29–n31, n33, n34, n38, n39 and c10. They get rows and caching only.

**2. "Trial sets - full run, sweeping over noise conditions and different forms of molecular similarity … full sample size using things like fast tier"**
BUILT as stage 1 (`tests/slurm_scripts/runbook/stage1_prelim_noise.py`).
- Full sample size.
- Fast tier where `dataset_policy` sets it.
- `--pruning-mode calibrated`.
- `KIRBY_NO_COLD_SEARCH=1`.

**3. "the main task in mind was similarity with an application for noise"**
BUILT.
- n42 (`tests/experiments/noise/n42_similarity_noise_sweep.py`) crosses every data-changing mitigation method with every similarity setting.
- The individual experiments carry the same axis.

**4. "these shared memories need to be fixed"** (the shared machinery list)
PARTLY.
- **Built:**
  - The four neighbour estimators in `src/kirby/utils/noise_estimation.py` take `precomputed_neighbors=`.
  - `n07.cleaned_training_set` takes `similarity_neighbours=`/`similarity_k=`.
  - The neighbour calls in `lnl_methods.py` were changed.
- **Not changed:**
  - The six filter classes in `noise_mitigation_methods.py`. No experiment in the suite calls them.
  - `GaussianProcessEstimator`.
  - n11's `HybridNoiseEstimator`, which cannot take a neighbour list. It is marked TODO at n11:1094 and n22:658.

**5. "Similarity calculated using hybrids … the top base rep as a comparison point … augmented … base rep + some augmented features"**
BUILT.
- The similarity spaces are:
  - ECFP4 Tanimoto;
  - each model's top 2 single representations;
  - each of those two plus augmented features, both capped and uncapped;
  - the prediction-built hybrid;
  - the similarity-built hybrid;
  - forest proximity.
- These are defined in `similarity_space.py`, where `FEATURE_SPACES` starts at line 105.
- The precompute does not store the augmented versions, so they are built per fold with c21's recipe and cached.

**6. "Importance weighting - try with and without WHEREVER POSSIBLE … different types of weighting"**
BUILT.
- The weightings are none, lightgbm_gain, treeshap and deeplift (`similarity_space.py:112`).
- The weights are learned from the noisy training labels (your decision).
- Not weighted: Tanimoto and forest proximity, because they have no features to weight.
- n12 and n21 keep their own older weighting next to the new one. Replacing it would change their existing numbers; see the TODOs at n12:1037 and n21:298.

**7. "two types of molecular representations … These need to be swept separately"**
BUILT.
- The model input is unchanged: each model's top 2 single representations, and the prediction-built hybrid.
- The similarity space is a separate loop.

**8. "focus on for this suite: n02, n07, n09, n11, n12, n13, n17, n18, n19, n20, n21, n23, n24, n25 … and n26-n41"**
BUILT: all of them are wired. n24 and n25 were never registered with the runner; they are now (`noise_helpers.py:105-106`).

**9. "Train RF and SVM on D … on D' … D' should be better than D"**
BUILT.
- Every experiment that changes training data writes one row per fold: score on D, score on D', score on clean labels, and the gain. The gain uses each dataset's own direction.
- Rows go to `<output>/<experiment>/<dataset>[__condition].rows.jsonl`.
- **Not covered:**
  - n09's eleven methods other than distance_selection, and n13's equivalents. They never produce a cleaned training set; they change the training itself.
  - n33's yeo_johnson, c_mixup and target_noise_bagging get rows but no control twin.

**10. "All the other experiments … add TODO comments"**
BUILT. There are 30 `TODO(similarity-sweep 2026-09-25)` comments across 22 files, each at the line concerned.

**11. "DO NOT touch [my hacky runner] … you can create sample slurm scripts"**
PARTLY.
- `_local_grid.py` and everything under `complexity/` were not touched. They still have uncommitted edits that are not mine.
- `run_dataset.py` and `submit_dataset.sh` were changed, because Prompt 2 told me to handle the runner.

**12. "cache some useful information (saved similarity values?)"**
BUILT.
- `noise_cache.py` stores neighbour lists, distances, hybrid builds, fits, out-of-fold predictions, detector scores, standardised matrices and forest leaves.
- They live under `$KIRBY_CACHE_DIR/noise_cache/<dataset>/`, the one cache on stat-ecr.
- Keys use a hash of the label vector, not the noise condition name. So at level 0 every condition shares one entry.

**13. "Modify the experimental code itself to accommodate different methodologies of similarity"**
BUILT. See 1 and 4.

**14. "Construct run scripts for the initial sweep"**
BUILT: `stage1_prelim_noise.py`, as one job array.

**15. "Compute optimisation - jobs are timing out and the results aren't being saved"**
BUILT.
- Every noise experiment saves per fold or per condition and resumes.
- Similarity, federated and OPIG save partial results (`tests/experiments/_checkpoint.py`).
- Each experiment runs in its own process, so one crash cannot take the others down.

**16. "Pay attention to metric direction"**
BUILT.
- `METRIC_DIRECTION_AUDIT.md` found one error, in n06. It is fixed at n06:395-412.
- n34's recovery share now refuses a level where the noise improved the score.
- Every new row carries `lower_is_better`.

**17. "a single experiment which sweeps all of these conditions"**
BUILT: n42. **See warning A below.**

**18. "make WAY better use of caching … dedicate agents to figuring out where the compute bottlenecks are"**
BUILT. The cached items are the ones the bottleneck audit found (plan section 3).

**19. "the two controls … implemented for the full set of experiments but not the trial"**
BUILT.
- `controls.py` is off unless `KIRBY_NOISE_CONTROLS=1`.
- Stage 4 sets it; stage 1 does not.
- No control twin exists where a method changes every label or adds rows: n29's three per-record methods, and n33 and n39's methods listed in 9.

**20. "clear documentation … trial … what gets set from the trial … full set"**
PARTLY.
- Built:
  - this file;
  - `tests/slurm_scripts/runbook/README.md`;
  - `tests/experiments/noise/NOISE_SEARCH_SPACE.md`, which lists every axis to cut from.
- The list of trial datasets is still yours.

**21. "Start with documentation + a plan, then a dynamic workflow"**
Done. The plan is `~/.claude/plans/pasted-content-id-0428-noise-experiment-fizzy-ladybug.md`.

**22. "quick analysis plan where I can see the results"**
BUILT, not tried on any rows: `tests/analysis/analyse_noise_prelim.py`.
- It gives per dataset and per model, never averaged across datasets:
  - clean vs D;
  - D vs D' as folds won;
  - D' per similarity setting;
  - model input × similarity space.

**23. "use the models/reps/conditions that are set from the precompute, don't force any unnecessary large searches"**
BUILT: `KIRBY_NO_COLD_SEARCH=1` everywhere. **See warning B below.**

**24. "Let the first chat handle all of the precompute … tell me the current state"**
The state was given in the plan, from repo records only. The live check is the `precompute_status.py` command in Part 2.

**25. "I would prefer this to run on GPU"**
Replaced by your later answer, "keep it to CPU".
- Answer given: no model moves to a GPU by itself.
- The time goes to SVM, RF, tuned xgboost and treeshap/deeplift. None of these can move to a GPU without changing every number relative to the precomputed search cache.
- Details are in the runbook README under "GPU".

**26. "confirm with me what gets swept before implementation"**
Done. The plan was approved with your corrections.

**27. "one real local run per experiment … SUPER SMALL … caco2 and a smaller version of qm9"**
Replaced by your correction: skip it, and run checks on the server. See Part 2, step 3.

**28. "label smoothing … I want all label smoothing methods tested in this trial sweep"**
NOT DONE; YOURS.
- You said this needs a discussion first. No new smoothing method was built.
- The seven existing smoothers in n07 are all in the sweep.

**29. "server commands to determine the trial datasets"**
BUILT: Part 2, step 2.

**Your corrections to the plan:**
- "no hybrids + augmented features": BUILT.
- CPU: BUILT.
- Light local smoke, then short HTC/srun checks: BUILT as `server_checks.sh`. The local smoke was skipped once you said nothing should run locally.
- "Noisy": BUILT.
- Fast tier + adaptive pruning: BUILT (`calibrated`).
- Document the search space: BUILT (`NOISE_SEARCH_SPACE.md`).
- Discuss label smoothing and n20 first: YOURS.
- "Everything enters the trial": BUILT. Every registered noise experiment is in stage 1.
- Caching, not cutting: BUILT.
- No decision rules: none were written.
- Tune for HTC: BUILT.
- Metric direction audit: BUILT.

---

## Part 1 — Prompt 2, instruction by instruction

**"1. All of these should be swept"** (neighbour counts)
BUILT: k = 5, 9, 10, 20 (`similarity_space.py:120`).

**"2. Both?"** (capped and uncapped augmented)
BUILT: both are similarity spaces.

**"3. Yes let the limit go"** (`prox_smooth`)
BUILT and pushed (`ec5b3f3`).

**"4. Yes fix run_all"**
BUILT.
- A failed experiment now writes `<output>/_errors/<dataset>.json` with its traceback.
- `run_dataset.py` exits 1 after all experiments have run.
- An import failure and a `SystemExit` are now recorded as failures instead of skipped.

**"5. Fix it"** (account)
BUILT.
- The hardcoded `--account=stat-ecr` is gone from `submit_dataset.sh`.
- `where_to_submit.sh --best-cluster-account` reads fair share on both clusters and returns the best cluster and account.

**"things MUST SAVE INTERMEDIATE VALUES and have failsafes if one experiment fails"**
BUILT.
- One process per experiment.
- Per-fold saves and resume.
- A per-job summary of exit codes in `_job_summaries/`.

**"trials for the rest … two datasets, a classification and a regression … srun … just make sure the experiment is going along without error"**
BUILT: `stage2_trials.sh` prints the srun lines. Complexity files are not changed (see 11). They get process isolation from the job script, but no saves inside an experiment.

**"I don't know if all the noise experiments are properly hooked up to the runner"**
BUILT.
- n24, n25 and n42 are now registered.
- `--force-rerun` now reaches the per-condition files; before, it silently skipped them.

**"ordering 1–4"**
BUILT: stage 1 to stage 4 scripts, in `tests/slurm_scripts/runbook/`.

**"make use of my fair share … HTC"**
BUILT. Jobs default to `--clusters=htc` with the best account, which was stat-cadd at 0.0491 when you measured it. That runs CPU jobs on htc's CPU nodes.

**"The cache lives in one place … needs to be called correctly"**
BUILT.
- Every job sources `setup.sh` with `KIRBY_SKIP_INSTALLS=1`, which sets `KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby`.
- Skip mode also stops each job re-pointing the shared stat-cadd venv at its own checkout.
- It also stops jobs building HMMER into stat-cadd, which is 99.9% full.

---

## Warnings you need before running

**A. n42's full cross is very large.**
- `NOISE_SEARCH_SPACE.md` counts about 2.4–2.7 × 10⁹ downstream fits per regression dataset for n42 alone.
- That count was written by an agent and I have not re-derived it.
- The similarity axis is 67 settings × 4 values of k per model per fold.
- This is the search space the prelim run exists to cut, but the prelim itself runs it.
- The individual experiments carry the same axis. `KIRBY_SIMILARITY_SETTINGS` chooses how much of it a job runs:
  - unset: everything;
  - `model_input`: today's neighbours only, reproduces today's numbers;
  - a comma list of setting names.
- Which one stage 1 uses is your decision.

**B. Five experiments run their own feature search, and `KIRBY_NO_COLD_SEARCH=1` refuses any search the precompute did not store.**
- The five are n01, n09, n15, n20 and c10.
- They will skip or refuse those folds unless the searches exist.
- The TODOs are at n01:486, n09:607, n15:161, n20:1151 and c10:443.

**C. n18's numbers change.**
- Its neighbour detector used to count molecules from the same scaffold family as neighbours.
- It now excludes them, as n07, n24 and n25 do, so every neighbor_residual number in n18 moves.
- n23 got the same fix.

**D. Still broken: n07 crashes on `grouped_shifted` and `grouped_wider`.**
- It never passes scaffold groups to the noise injection.
- TODO at n07:3223.
- This was on my list and was not fixed.

**E. Not changed, waiting on you:**
- The seeding differences between experiments: n07's outlier share, the shift seeds in n31/n32/n41, and the relative seeds in n38/n40/n41.
- n20's search.
- Label smoothing.
- The complexity files.

**F. Server checks run for 2 hours each by default.**
- With the full similarity axis, most noise experiments will not finish their first run in 2 hours.
- The second run, which checks the cache, then never starts.
- `server_checks.sh --similarity-settings model_input` checks crashes and caching quickly. The full axis then runs in stage 1.

---

## Part 2 — What to type on arc-login, in order

**1. Get the code and the one cache.**
```bash
cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only origin similarity-metrics-study
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
echo $KIRBY_CACHE_DIR          # must print /data/stat-ecr/scat9264/.cache/kirby
bash tests/slurm_scripts/where_to_submit.sh --best-cluster-account     # e.g. "htc stat-cadd"
```

**2. Which datasets have their precompute.** Use the list from your first prompt:
```bash
python tests/tools/precompute_status.py tcga_brca_pam50 caco2 buchwald_hartwig logd flab_expression_koenig herg_ki humanness_therapeutic nt_H3K4me3 flab_thermo_tresanco tcga_luad_vs_lusc qm9 bh_doyle_plates nt_promoter_tata adme_mdr1 adme_rlm tz_her2_binary herg
python tests/precompute_coverage.py herg_ki herg caco2 --incomplete-only
```
Pick the trial datasets from what prints DONE.

**3. Server checks: every experiment briefly, many sruns at once.** Replace the dataset names with yours:
```bash
bash tests/slurm_scripts/runbook/server_checks.sh --datasets "herg_ki herg" --dry-run
bash tests/slurm_scripts/runbook/server_checks.sh --datasets "herg_ki herg"
bash tests/slurm_scripts/runbook/server_checks_status.sh
bash tests/slurm_scripts/runbook/server_checks_status.sh --not-passed
bash tests/slurm_scripts/runbook/server_checks_status.sh --tail noise_herg_ki_n07
```
- For noise, "passed" means four things:
  - no crash;
  - rows on disk;
  - the second run is all cache hits;
  - the recipes and representations read match the precompute.
- For the other categories it means exit 0 with no `_errors/` file.
- A run cut off by the time limit, with no error in its log, counts as progressing.

**4. Stage 1: prelim noise on the trial datasets.**
```bash
python tests/slurm_scripts/runbook/stage1_prelim_noise.py --datasets "herg_ki herg" --dry-run
python tests/slurm_scripts/runbook/stage1_prelim_noise.py --datasets "herg_ki herg" --max-concurrent 40
bash   tests/slurm_scripts/runbook/progress.sh tests/results/noise_prelim_sweep logs/noise_prelim
```
Then read the results:
```bash
python tests/analysis/analyse_noise_prelim.py tests/results/noise_prelim_sweep
```

**5. Stage 2: two-dataset trials for complexity, similarity, federated and OPIG.** These are blocked on the precompute.
```bash
bash tests/slurm_scripts/runbook/stage2_trials.sh --datasets "herg_ki herg" --fed-datasets "caco2"     # prints srun lines
bash tests/slurm_scripts/runbook/stage2_trials.sh --status
```

**6. Stage 3: the full suite except noise.** This is blocked on stage 2.
```bash
bash tests/slurm_scripts/runbook/stage3_full.sh --dry-run
bash tests/slurm_scripts/runbook/stage3_full.sh
```

**7. Stage 4: full noise, with controls on.** This is blocked on stage 1 and your analysis.
```bash
bash tests/slurm_scripts/runbook/stage4_noise.sh --dry-run
bash tests/slurm_scripts/runbook/stage4_noise.sh --max-concurrent 200
```

**Failures, any time:**
```bash
bash tests/slurm_scripts/runbook/progress.sh <results dir> <log dir>
squeue -M arc,htc -u $USER -o '%.18i %.24j %.9P %.2t %.10M %.10L %R'
```
Each experiment's exit code is in `<results dir>/_job_summaries/`:
- 1 = Python error;
- 124 = timeout;
- 137 = out of memory or wall time;
- 139 = segfault.

Tracebacks are in `<results dir>/_errors/`.
