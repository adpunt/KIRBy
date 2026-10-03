# Handoff: bring every experiment in line with DECISIONS.md, with per-fold searches inside the experiments

Written 2026-10-03 by the session that worked on this branch on 2026-10-01 to 03. Nothing in this file is a measured result. Every claim about code was true when written; re-read the code before acting on any of it.

## 0. Read first, in this order

1. `/Users/apunt/repos/KIRBy/CLAUDE.md`: how to write to Adelaide. It loads every turn. Follow it in every message.
2. The memory index, `~/.claude/projects/-Users-apunt-repos-KIRBy/memory/MEMORY.md`. Read every ⛔ line. Then read these two topic files in full:
   - `project_similarity_hybrid_set_2026_10_02.md`: what was built on 2026-10-02 and 03, with commit hashes.
   - `project_server_triage_2026_09_27.md`: the server thread, updated in place.
3. **The decisions record: `/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/DECISIONS.md`. Read it in full.** It is about 78 KB, and it is not in git. Another chat was still editing it on 2026-10-03 at 22:18. Re-read it before every decision, and never edit it unless she asks. It has these sections:
   - Step 1: the two approved downstream models per dataset. knn is not allowed as a downstream model. gp is allowed except on very large datasets. ridge was still unanswered.
   - Step 2: budgets, several versions; v3 is "current proposal".
   - Step 3: importance scoring.
   - Fast-tier datasets.
   - Step 4: allocation methods.
   - Step 5: internal evaluator, v3.
   - Step 6: representation count.
   - Step 7: the distance search uses the same rules, with downstream = the decided pair.
   - Step 8: augmentation.
   - Step 9: knn, and who reads the distance search.

   Its own helper is `scratchpad/precompute_cuts/cuts.py`.
4. `tests/slurm_scripts/runbook/SUBMIT_TONIGHT.md`: how the experiment jobs are submitted today. It will need rewriting once precompute moves into the experiments.
5. The full transcript of the previous session, if a detail is missing here: `~/.claude/projects/-Users-apunt-repos-KIRBy/ca64421e-dc29-46df-9b65-750ff7f769ad.jsonl`. It is very long, so search it; do not read it whole.

## 1. What Adelaide asked for, verbatim (2026-10-03)

> Okay so now I need to make sure the following is happening:
> No extra precompute scripts, precompute gets added to experiments - if the precomputed per-fold searches don't exist, do them. Slight concern - if similarity and complexity experiments are running at the same time, how do I ensure the searches don't overlap? Is there some temporary flag I can use to indicate whether a search is in progress or not? This feels dangerous if a script times out
>
> Across all experiments, things need to be uniform
> Make sure these per-dataset settings are used, both for general and distance
>
> So there should be per-fold precompute searches for every experiment that falls in line with the agreed upon cuts in DECISIONS.md
> There have been SO many issues with experiments do extra searches, aren't using per-fold hybrids, all of that. Every single experiment needs to be examined.
> The general standard is when you're selecting hybrids, per each of the two models (assuming everything is running on a per-fold basis):
> - 2 best base reps
> - 2 best hybrids
> Obviously some things will differ
> Some things will need to call additional searches - make sure those additional searches use those cuts
> The noise experiments are another beast - those need to be cut down in other ways but you just need to make sure they're using the 2 best base reps/hybrids per model, so 8 configurations total
>
> So first I'll need to make sure that every single experiment is in line with DECISIONS.md. I know for a fact they are currently all over the place.
> KNN searches will be run for the appropriate experiments when they come up.
> Basically, every experiment script needs to fall in line. And any extra find_best_hybrid searches or augmentation steps or anything like that also need to fall in line. Unless its KNN be VERY careful to only use each datasets approved two models. PLEASE
> Then you'll need to make sure that the searches get run without formal precompute. I don't have time to queue up separate scripts for that. They should be included in the experiments, with the understanding that there may be overlap with start times within a single dataset. Unless you use dependencies and force complexity experiments to come first and take care of all the searching. Again that searching must per per-fold. And each experiment must function on this per-fold basis and take the appropriate search results per fold. I cannot stress enough how this needs to be handled carefully.

Plus, from the same message: "All of this should go in the handoff prompt … I want everything squared off."

## 2. ⛔ The new standard conflicts with what was built on 2026-10-02. Resolve this with her first.

On 2026-10-02 she asked for "top 2 general hybrid + top distance hybrid". The previous session built that for the five downstream models in `DOWNSTREAM_MODELS` (rf, xgboost, svm, dnn, lightgbm):
- `tests/experiments/_shared.py`: `get_rule_hybrids`, `build_hybrid_from_entry`, `RULE_N_GENERAL=2`, `RULE_N_DISTANCE=1`, `RULE_N_BASE=2`.
- Commits `c7f2385` (shared reader), `0ca00d4` (complexity), `1e4dd00` (similarity), `8bfa372` and `4960a02` (federated and OPIG), `a904c36` and `5912b86` (noise).

Her 2026-10-03 standard says something different:
- the two **approved** models per dataset (DECISIONS.md Step 1), not five;
- 2 best single representations + 2 best hybrids per model;
- per fold;
- noise: 2 × (2 + 2) = 8 configurations.

Questions to put to her in plain words before building:
- **"2 best hybrids":** is it 2 from the general search, or 1 general + 1 distance? Or is it per search, so 2 general + 2 distance? DECISIONS.md Step 7 and Step 9 say who reads the distance search; read them first. Do not count your own reading as her answer.
- **Experiments that read both search families:** "for prediction" and "for similarity", `objective='similarity'`. Is the 2 + 2 per family?
- **knn:** it is not a downstream model, but similarity and noise experiments use it. Step 9 covers this. She said "KNN searches will be run for the appropriate experiments when they come up."

Whatever she answers, change `RULE_N_*` and the model list in **one place** (`get_rule_hybrids` and its callers), not 90 files.

## 3. The structural change she wants: per-fold searches inside the experiments, no separate precompute

**Today:**
- Precompute = `tests/precompute_hybrids.py` driven by `tests/slurm_scripts/run_precompute_dataset.sh`, chained by `extend_precompute_chain.sh`.
- It runs whole-dataset searches.
- Experiments rank those searches' configurations on the other folds' mean (`get_fold_pool_recipes`, `get_rule_hybrids(fold_i=k)`). That holds out a fold's scores but not its rows. DECISIONS.md calls the whole-dataset search "leaky, accepted".

She now wants **per-fold** searches: one search per outer fold, on that fold's training rows. Each experiment takes each fold's result. Pieces that already exist:
- `_shared.get_fold_best_search(reps, labels, groups, *, train_idx, fold_i, …)` (`_shared.py` around line 1497). It accepts `distance=`.
- Per-fold tiers in the result database: local `herg_ki.db` and `logd.db` hold `general__cond-heldout_fold0..4`.
- c11 moved its recipe search inside each fold (memory: `project_c11_per_fold_recipe_2026_09_04.md`).
- `nested_cv_evaluate` / `run_nested_cv` in `src/kirby/find_best_hybrid.py`.
- Search results are stored in SQLite, `result_db/<dataset>.db`. Rows are flushed every 25 configurations, and a rerun skips configurations already stored. So a search that times out resumes.
- `KIRBY_NO_COLD_SEARCH` refuses a search the precompute did not store. Since `30ad832`, a search with at most `KIRBY_COLD_SEARCH_MAX_CONFIGS` (default 500) configurations left runs inside the experiment instead. That guard has to be rethought once experiments are meant to search.

**Her concurrency question:** two experiments may start the same per-fold search on one dataset at the same time. She asked about a "temporary flag" and worries about timeouts. Options to put to her, with what each costs:
- **(a) A lock file per (dataset, tier, fold, objective, search settings), with a heartbeat.** The owner rewrites a timestamp every minute or so. Another job that finds a fresh lock waits, polling. A lock whose heartbeat is older than N minutes is stale and may be taken over. A timeout therefore never blocks forever, and the stored rows let the new owner resume. On the shared filesystem (NFS) use atomic create (`os.open` with `O_CREAT|O_EXCL`) plus the heartbeat, not `flock`.
- **(b) Dependencies.** Complexity jobs for a dataset run first and do all the searching; similarity, federated, OPIG and noise for that dataset wait (`afterany`). This is simpler, but it serialises the start, and a complexity failure delays the rest; `afterany` still lets them go.
- **(c) Both.** Dependencies for order, a lock as the safety net.

Also check SQLite: more than one writer to one `result_db/<dataset>.db` on NFS is unsafe. The precompute chain was deliberately one writer per dataset.

**Running precompute** must stop or be reconciled once experiments search. Its stored whole-dataset searches may still be useful as a cache, or may be ignored. Ask her. Do not cancel jobs without her say-so.

## 4. The audit she wants: every experiment, every search, every augmentation step

Scope: every file in:
- `tests/experiments/complexity/c*.py`
- `tests/experiments/similarity/s*.py`
- `tests/experiments/noise/n*.py`, plus `c10_noise_robustness.py`
- `tests/experiments/federated/fed*.py`
- `tests/experiments/opig_*.py`

For each file:
- every `find_best_hybrid` / `get_fold_best_search` / `run_nested_cv` / `nested_cv_evaluate` call;
- every pool read (`get_best_search_for_downstream`, `get_fold_pool_recipes`, `get_rule_hybrids`, `pool_recipes_by_family`, `get_fitted_pool_hybrid`, `top_hybrids`, `select_model_rep_pairs`);
- every `create_hybrid` / augmentation step.

Check each one against DECISIONS.md: which models, budgets, importance, allocation, evaluator, representation count, augmentation, general and distance. Check that it is per fold.

A 2026-10-02 audit of the hybrid set (read-only) produced a per-file table: tier, how many hybrids, families, base side, which helper. It is in the previous transcript; search "Hybrid-selection rule audit". Most of it was then changed by the commits in section 2, so re-audit; do not reuse it.

Known facts from that work, to re-verify:
- Experiments that run their own searches: c02, c05, c07, c11, c13, c14, c15 part c, c33, n01, n09, n15, n20, s11, fed05, opig_activity_cliff_qsar, opig_blueprint_transfer, opig_ginkgo, opig_kingsmith, opig_trastuzumab_enrichment, opig_generator_scoring_swap, opig_scaffold_hybrid, opig_shared_vectors_remedy. Several of these added a distance search on 2026-10-02. Each must now use the DECISIONS.md cuts and the two approved models.
- Search-as-object-of-study, kept on purpose:
  - n09 and n20 search on noisy or cleaned labels;
  - c10 searches under noisy labels;
  - c24 measures search stability.
  Ask her whether the cuts apply to these too.
- No-hybrid files, left alone: s01, s09 (summary), c01, c04, c06, c12, c20, c21, c23, c31, c32, c43_*, n14, n16, opig_scaffold_similarity.
- Model lists today come from `DOWNSTREAM_MODELS` (5), plus gp on datasets under 1,000 molecules (`dataset_policy.small_dataset_extra_models`). DISTANCE_MODELS = the 5 + knn. **None of this reads the two approved models from DECISIONS.md yet.** That is the core change. The approved pair must come from one place in code, generated from or checked against DECISIONS.md, and be read everywhere.
- `dataset_policy.FAST_SPEED_DATASETS`: 14 datasets search the fast grid, including bindingdb_kd and kiba since `18da121`. DECISIONS.md has a "Fast-tier datasets" section. Reconcile them.

## 5. Everything else the previous session recorded, by topic

### Fixed and pushed, 2026-10-01 to 03 (branch `similarity-metrics-study`)

- **`ca31882`**: `noise_cache._load` treats an unreadable file (NFS stale handle) as a miss. n07 died on it before.
- **`5b01654`**: fed02 `_recipe_for` restored. `01e309a` had truncated it.
- **`18da121`**: bindingdb_kd and kiba moved to the fast grid. Changed in `dataset_policy.py` and in `run_precompute_dataset.sh`'s LARGE list; the two must agree or the job aborts.
- **`16ed9c0`**: `tests/tools/pdv_cache_audit.py --fresh` rebuilds 200 molecules and compares them with the cache. chembert on herg_ki, logd, qm9 and adme_hlm: all match. herg_ki chemberta was rebuilt in the server cache.
- **`000a856`**: distance searches run under both objectives (predictive and similarity).
- **`30ad832`**: the small-search rule above.
- **`7e17900`**: rebuilds honour `blueprint.rep_count` (new `find_best_hybrid.restrict_to_rep_count`, used by `apply_blueprint`, `get_fitted_pool_hybrid` and `build_hybrid_from_entry`). Before it, a recipe searched on 2 of 10 representations was rebuilt over all 10.
- **`beec9c0`**: fed01–03 build each hybrid from its full recipe at each swept budget. fed02 C6 was stuck at 100 features. fed03 Part B C3 now sweeps the budget.
- **`94dacdf`**: noise neighbour spaces cut to 7, 58 settings. ⛔ She has since said another chat is changing the neighbour-based methods. **Do not touch the noise neighbour/similarity sweep.**
- **`31715d4`, `93cf729`, `9237ea9`**: submit pipeline.
  - One error file per experiment.
  - Quickest experiments first in each job (`RUN_ORDER` in `submit_dataset.sh`).
  - A resubmit skips experiments whose `_job_summaries` row says ok.
  - `--output-root` is required.
  - It waits on each dataset's `pc_` jobs and has a duplicate guard.
  - s09, c03 and c14 are submitted as cross-dataset jobs via `runbook/cross_one.sh`: s09 short and afterany on the similarity jobs, c03 long and afterany on pc_ jobs, c14 long with no dependency.
  - This all assumes the separate precompute and must be revised for section 3.

### Open, needs her decision

1. **The section 2 conflict.** It comes first.
2. **Section 3:** the concurrency design, and what happens to the running precompute.
3. **abagym_Ang2, abagym_EGFR, abagym_VEGF, and probably flab_binding_koenig and flab_expression_koenig.**
   - The fault: their folds group by mutated position modulo 5, so each held-out fold is one group. The similarity score excludes same-group neighbours, so it cannot be computed. Every similarity-objective search fails: 22,050 errors per abagym log, 0 rows stored.
   - Recommended: leave the similarity objective out on them, as for tz_her2_binary (add them to `SIMILARITY_UNDEFINED_OTHER` in `dataset_policy.py`).
   - First confirm koenig: `python tests/precompute_coverage.py flab_binding_koenig flab_expression_koenig | grep -E "^[a-z]|similarity"`.
   - She has not answered.
4. **c14** is marked "delete" in `EXPERIMENTS_FINALIZED.md:317`. It is submitted anyway because she asked for cross-dataset jobs. It has no checkpoint and saves only at the end. Its 5 tasks come from its docstring.
5. **Wilcoxon tests** in s09 (lines ~708, ~718) and c14 (~918, ~957) break her no-significance-tests rule. Ask whether to remove them.
6. **c03** loads datasets without augmentation, so it reads only the 'none' search. Ask.
7. **n20** has no distance hybrid, deliberately: compute.
8. **Federated per-silo helpers** (`federated_feature_selection`, `fluid_within_kirby_across` in `federated_helpers.py`) take only 3 recipe settings. Two slots that agree on those build the same features; this is recorded per run.
9. **opig_shared_vectors_remedy** has no tie counts for general2. `c14.rule_general2_outer` returns no recipe.
10. **`tests/experiments/opig_generator_scoring_swap.py`** is untracked. It was restored from `dirty-local-2026-09-28` commit `1829cd0` after an agent edited an older copy by mistake. The edited copy is in the previous session's scratchpad. Its owner chat is unknown.
11. **`_smoke_n25.py`** still has a stale `per_space['base_1']` lookup around line 326. **`NOISE_SEARCH_SPACE.md` §3.1** still says n42 has 3 model inputs; it is 5.
12. **Handled elsewhere, do not touch:** knn settings (s02), label smoothing (f36), the noise neighbour-based methods, and s02's distance search under the similarity objective.

### Server state, as of 2026-10-03 evening

- **Precompute jobs (`pc_*`)** are running or queued on htc for about 20 datasets.
  - Extra links were added, and 9 excess links were cancelled.
  - bindingdb_kd and kiba: their new fast-grid chain is 8960153–8960160; their old full-grid jobs were cancelled.
  - DONE datasets got one extra distance job each (8960162–8960180) to build the similarity-chosen distance searches.
  - Coverage figures under-count datasets run with calibrated pruning: pruned configurations are never stored.
  - Queued jobs run the job script as it was when submitted, but read Python at start. So `precompute_hybrids.py` changes reach them; `run_precompute_dataset.sh` changes do not.
- **Experiment jobs (`x_*`, `opig_*`):**
  - The 11 running on arc were cancelled.
  - All pending ones were held. She was told to cancel the held ones before submitting fresh.
  - Nothing from tonight's `SUBMIT_TONIGHT.md` had been confirmed submitted when this was written. Ask her.
- **Fair share** (higher is better): stat-ecr arc 0.025, htc 0.023; stat-cadd arc 0.015, htc 0.013. Use stat-ecr.
- **Server repo:** `/data/stat-ecr/scat9264/KIRBy`. Setup: `export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh`. Cache: `/data/stat-ecr/scat9264/.cache/kirby`. Never ssh. Give her commands to paste.

## 6. How to work with her (from memory and from this session)

- **Plain language, per CLAUDE.md.**
  - Her words: representations, folds, budgets, allocation methods, objectives, datasets, features.
  - No invented labels.
  - State a fault as two things that disagree, before any fix.
  - One decision per line, with numbers and a recommendation.
  - Never the multiple-choice tool.
- **She decides.** Do not cut axes, datasets or experiments for compute. A cut needs a stated reason: degenerate, broken or undefined.
- **Server commands:**
  - Always push before giving them.
  - Single lines.
  - For interactive work, one `srun … --pty /bin/bash` line, then plain commands to type inside.
  - Never `bash -c` wrappers or shell variables holding commands.
  - Heavy work never on the login node (chemberta failed there: libcufft).
  - Read-only checks first. Never bulk scancel without listing the jobs first.
- **Verify by reading code in the session.** Memory and documents are dated claims.
  - Every subagent report from the previous session was checked before commit.
  - Toy checks are not real-data checks. Say which was done.
- **Commit and push normally** (`git push` without asking). Do not commit files other chats own:
  - `scratchpad/`
  - `thesis_appendix/*`
  - the untracked f34/f36/f37 files and `opig_generator_scoring_swap.py`

## 7. Suggested order

1. Read section 0's documents.
2. Put the section 2 questions and the section 3 concurrency options to her, as decisions in plain words. Wait.
3. Implement the approved-pair source and the per-fold search-plus-lock machinery in shared code.
4. Audit and convert the experiments, category by category, with subagents on disjoint files. The shared code is edited only by you.
5. Rewrite `SUBMIT_TONIGHT.md` and the stage-3 submit for "experiments do their own per-fold searches".
6. Short server checks: herg plus one antibody dataset, one experiment per category, time-boxed. Then the full submission.
