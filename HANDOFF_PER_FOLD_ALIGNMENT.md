# Handoff: bring every experiment in line with DECISIONS.md, per fold, no separate precompute

Written 2026-10-03 at the end of a long session. Read this first, then `CLAUDE.md` (how to write to Adelaide) and `MEMORY.md`. Adelaide has asked for a fresh chat so nothing is lost.

## 0. Rules that cost real work when broken (from CLAUDE.md and memory, restated because they apply here)

- Plain words, no invented jargon. Say what is broken before the fix. One decision per line. Never a bare fraction.
- Verify in code this session before claiming anything. Memory files and repo docs are dated claims.
- She decides. Offer options with numbers; do not design alone.
- Never use the multiple-choice question tool. Never ssh/scp (she runs commands). Never sbatch for runs she asks for (local or srun only).
- Do not edit framework experiments f00-f15. Noise experiments are fair game.
- Appendices go in `additional_files_body.tex`, never the thesis PDF.

## 1. What Adelaide asked for (her words, tidied)

The cuts per dataset are now decided and recorded in `scratchpad/precompute_cuts/DECISIONS.md`. Now make the code match.

1. **No extra precompute scripts.** The per-fold searches get run by the experiments themselves: if a needed per-fold search does not exist, the experiment runs it. She has no time to queue separate precompute scripts.
2. **Per-fold, everywhere.** The current whole-dataset precompute leaks (see section 5). Every experiment must take the search result for each fold, built on that fold's training rows. Each experiment must work on that per-fold basis.
3. **Uniform.** The per-dataset settings in DECISIONS.md apply to both the general and the distance search, in every experiment. Any extra `find_best_hybrid` call, augmentation step or similar must use those same cuts.
4. **The standard way to select hybrids,** per each of the dataset's two models, per fold: the 2 best base representations and the 2 best hybrids. Some experiments will differ.
5. **Noise experiments** get cut down in other ways, but they must use the 2 best base representations and 2 best hybrids per model, which is 8 configurations in total.
6. **Models.** Use only each dataset's approved two models. Be VERY careful about this. knn is the one exception: knn searches get run for the experiments that need them, when they come up (list in DECISIONS.md Step 9).
7. **Every single experiment has to be examined.** She says they are currently all over the place: extra searches, not using per-fold hybrids.
8. **Concurrency worry (open design question, her words):** if the similarity and complexity experiments run at the same time on one dataset, how do we make sure the same search is not built twice? Is there a flag marking a search as in progress? This feels dangerous if a script times out (a stale flag would block everyone). Options she mentioned: accept overlapping start times, or use dependencies so the complexity experiments run first and do all the searching. Nothing has been decided. Do not decide it for her.
9. **A bug to fix too** (section 6): svm is fitted on unscaled labels.

## 2. Where the record is

- **`scratchpad/precompute_cuts/DECISIONS.md`** is the single record. It stays as a loose file in the repo folder; Adelaide does not want it committed.
- Not every section is authoritative. The file still contains superseded proposal tables I never deleted. The authoritative ones are those with a bold, dated DECISION cell:
  - Step 1: two models per dataset (41 rows).
  - Step 2 (v3): budget range per dataset (41 rows). Seven rows were revised on 2026-10-03 after the scorer limit was removed; each shows "was ...".
  - Step 3 (v2, cautious): importance cuts. Six rows revised, same note.
  - "Fast-tier datasets: the stored grid is the cut": ten datasets that only ran lightgbm_gain, three allocation methods and rep counts all/3. Do not extend them.
  - Step 4: allocation cuts. Three rows revised. suzuki is HELD, see section 7.
  - Step 5 (v3): internal evaluator, with match as its own evaluator.
  - Step 6: representation count, general search.
  - Step 7: distance search (`general_distance`), all five axes.
  - Step 8: augmentation.
  - Step 9: knn, and who reads the distance search.
- Superseded (ignore, delete when she agrees): Step 2 (first, spread rule), Step 2 (v2), Step 3 (first), Step 5 (first), Step 5 (v2).
- **Method files:** `tests/experiments/framework/F33_PRUNING_PROCEDURE.md` (older pruning procedure, not the method used here), `tests/compare_nested_vs_reselection.py` header (about 19 experiments use reselection, 5 use nested CV).
- **Memory file for the plan:** `~/.claude/projects/-Users-apunt-repos-KIRBy/memory/project_precompute_leaky_per_fold_plan_2026_10_03.md`, plus a new one for this record, `project_precompute_cuts_decisions_2026_10_03.md`.
- **Data used:** slim copies of the stored results at `/Volumes/seagate/slim_precompute/<dataset>.slim.db` (41 datasets; copied from the server with `tests/tools/export_precompute_slim.py`, commit 4cbb9bd). They keep tiers `general` and `general_distance`, objective predictive, augmentation none/capped/uncapped, without the model blob. chembl_qsar has nothing stored.
- **Analysis scripts:** `scratchpad/precompute_cuts/scripts/` (`axes.py` runs every axis; `aug.py` the augmentation comparison; `budgets.py`, `cuts.py` earlier versions; `dist_all.json`, `rep_general.json`, `aug_res.json` outputs). They read slim files from a `slim/` folder next to them; point that at `/Volumes/seagate/slim_precompute`. `axes.py` reads decisions out of DECISIONS.md.

## 3. The rules the cuts came from

- Scores come from the whole-dataset precompute (leaky; Adelaide accepted it for choosing the grid; the thesis must say so). Only the decisions came from it. The per-fold searches must be fresh.
- Downstream models: each dataset has two. knn not allowed as one of the two. ridge is excluded (removed everywhere on 2026-07-16 per `_shared.py:495-502`). gp is allowed except on massive datasets (none of the gp picks is large, 197 to 672 rows per `BILL_OF_MATERIALS.md`, a dated source).
- Ranking: average score of the 10 best configurations, in the dataset's own metric (rmse lower is better, every other metric higher).
- Band: 3% of the best value for datasets under about 1,000 rows, 1% otherwise. Small list: abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer, humanness_therapeutic, tcga_brca_pam50, tcga_luad_vs_lusc. The three reaction datasets and cdr_flexibility, herg_ki, lhasa_herg were treated as 1% (row counts unknown or unverified).
- Budget: range from the lowest to the highest budget inside the band, widened to at least 3 consecutive budgets.
- Importance, allocation, internal evaluator, representation count: cut a value only if its average is worse than the best by more than the band AND it holds 0 or 1 of the 30 best configurations overall. Evaluators are rf, lightgbm, dnn and match (the downstream model's own type).
- The axes were applied in order: budget, importance, allocation, evaluator, rep count, each on the configurations surviving the earlier cuts.
- The distance search has its own budgets (10 to 500 only) and evaluators (knn, rf, dnn, lightgbm, svm).
- Augmentation: only ginkgo_hic (uncapped only) and ginkgo_colloidal (uncapped) keep augmented searches; none for every other dataset. Whether that applies to the general search too, not just the distance search where it helped, is not settled.
- Similarity-objective searches: follow the same cuts and the same two models (inherited, not measured; almost none are stored).
- A second pass re-derived every distance axis and the general rep count independently: zero disagreements with `axes.py`.

## 4. What I found about who reads what (two subagent reports, NOT re-verified by me line by line)

### Distance search (`general_distance`)
- `get_rule_hybrids` reads it by default: `RULE_N_DISTANCE = 1` (`tests/experiments/_shared.py:7064-7066`), loop at `:7130-7137`. So every caller that does not pass `n_distance=0` reads it. This is how s04 and s13 get it.
- All 41 datasets are read by at least one distance-reading experiment (the complexity experiments c05, c07, c08, c09, c11, c16, c18, c19, c22 read stored distance searches with no dataset filter; c02, c13, c33 run their own). `run_all` in similarity and complexity also warms the distance search on every dataset (`similarity_helpers.py:1415-1445`, `complexity_helpers.py:1087-1126`).
- The nucleotide datasets (nt_* and gb_human_enhancers_cohn) are skipped by every similarity experiment (`_shared.py:3672-3676`) and by n01, n02, n11, n12, n14; they reach the distance search only through complexity and some noise experiments (n07, n09, n13, n15, n18, n21, n23, n24, n25).
- knn as a downstream model is needed by s02, s05, s06, s08, s10, s13 and the OPIG shared-vectors census, so on the 35 non-nucleotide datasets. s02 and s08 also skip datasets over 15,000 rows (kiba, bindingdb_kd). No stored-read experiment filters on a knn internal evaluator; only framework f34 asks for one and is in no submit script (`f34_neighbour_weighting_cross.py:239`).
- The evaluator list is not part of the stored-search identity (`result_cache.py` IDENTITY_COLS vs CONFIG_COLS), so dropping evaluators does not change which stored search is read. The per-fold `.pkl` caches built from `_make_search_meta` were not checked.
- Datasets with no stored distance search yet but read by experiments: bindingdb_kd, kiba, lhasa_herg.
- DISCREPANCY: one report says c03 runs on all 41 datasets (`R/submit_all_stages.sh:387-422`), the other says its default is logd, caco2, herg_ki (`c03:872`) and it sees no augmentations. Unresolved.

### Augmentation
- 32 of 41 datasets have augmentation features defined (`complexity_helpers.py:136+`). The nine without: abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, cdr_flexibility, humanness_therapeutic, lhasa_herg, tz_her2_binary, tz_her2_multiclass. bindingdb_kd and kiba never get uncapped (`UNCAPPED_SKIP_DATASETS`, `precompute_hybrids.py:615-624`, `_shared.py:5989`). The distance search gets capped only when n < 10000.
- Pool readers (`get_best_search_for_downstream`, `get_rule_hybrids`, `get_fold_pool_recipes`) treat augmented searches as optional candidates; ties go none, then uncapped, then capped (`_shared.py:6331-6340`). A missing member is skipped with a warning (`_shared.py:6107-6118`). `KIRBY_NO_COLD_SEARCH=1` is on in every stage-3 job (`submit_everything.sh:94, :364`) so a large missing search is skipped, not built (`find_best_hybrid.py:636-750`, threshold `KIRBY_COLD_SEARCH_MAX_CONFIGS` default 500).
- `run_all` in all four categories asks for none, uncapped and capped on every dataset at start-up (`_shared.py:5544-5548`), even on the distance tier. A failure is stored as None (`_shared.py:5605-5625`).
- Experiments that build their OWN augmented searches (not governed by a precompute cut): c02 (own none/uncapped/capped arms), c05 and c07 (subsample searches, always capped), c06 (uncapped only), c11, c14, c15c, c16, c23, c24, c33, s11 (`search_on_fold`, `_shared.py:1388-1480`), OPIG ginkgo (uncapped on five ginkgo datasets), OPIG shared_vectors_remedy (sweeps none/uncapped/capped/all on herg_ki). c21 needs the augmentation features, not stored searches. Noise `base_k_aug_uncapped` (n23, n32, `similarity_space.py`) also need only the features, uncapped only since 2026-10-03.
- Framework f24 and f25 and f36 are run by hand and in no submit script.
- Code places that would change to actually save compute: `run_precompute_dataset.sh:401-408` (capped/uncapped passes for every dataset), `_shared.py:5544` (unconditional ask), and each own-search experiment.
- Open from Adelaide: whether the experiments that compare augmented against plain by design (c02, c06, c21, shared_vectors_remedy, noise `base_k_aug`) keep running augmentation on all datasets or only on the two ginkgo datasets. Not answered.

### How datasets reach experiments
- Roster is `sorted(DATASET_REPS) - {chembl_qsar, davis}` = the 41 (`R/stage1_prelim_noise.py:85, 192-194`).
- Stage 3 (tonight's submit, `runbook/SUBMIT_TONIGHT.md:50-72`): 99 jobs. Noise is NOT in stage 3: it is stage 1 (trial datasets) or stage 4 (full roster).
- `should_skip_experiment` (`_shared.py:3672-3716`) applies in similarity and noise `run_all` only (not complexity or federated). Regression-only noise experiments skip the 14 classification datasets.
- 14 classification datasets: lhasa_herg, herg, flab_binding_adams, tz_her2_binary, tz_her2_multiclass, humanness_therapeutic, cdr_flexibility, tcga_luad_vs_lusc, tcga_brca_pam50, the five nt_*, gb_human_enhancers_cohn.
- Experiment lists with own distance or own-search use (federated fed05, OPIG jobs, noise n01, complexity c02/c13/c33/c14/c15c) are in the two reports; ask the next chat to regenerate them with the examination below rather than trust this summary.

## 5. The fault this all comes from
The precompute runs one search over every row of a dataset. Experiments then re-pick the winner per fold from stored per-fold scores (`fold_reselection`, `_shared.py:1814`). Those scores come from models trained on rows that include the held-out fold, so every stored hybrid score is selection-leaked. Single-representation per-fold scores and the representation matrices are not affected. The clean path is `run_nested_cv` / `nested_cv_evaluate`. Importance is computed per fold on that fold's training rows and keyed by a hash of those rows, so it is not reusable across folds. Cost reference: the full per-fold grid is about 204,000 model fits per dataset (script header, not re-measured); herg_ki at budgets 100 and 200, 3 outer folds, took 622.7 s.

## 6. svm fitted on unscaled labels: FIXED 2026-10-05 (commit fb1df54). Background below.

Resolution: the search's svm and the distance-search svm scorer now standardise the labels (`SVRScaledTarget` in `src/kirby/hybrid.py`). Local tests: only the reaction datasets change (buchwald_hartwig, bh_doyle_plates; suzuki's test produced no result). Every other dataset tested moved by 0.01 or less. Adelaide kept the existing two-model picks. Not fixed: the plain `SVR` variants at `_shared.py:3186-3191` (c06). The commit also carried another session's uncommitted concurrency edits to `find_best_hybrid.py` (KIRBY_SEARCH_REVERSE, re-reading the store every 25 configurations, per-fold searches never refused); Adelaide has not said whether to keep them.
- Adelaide's report: svm is `SVR()` at its defaults. On buchwald_hartwig yields (standard deviation 27.3) it scores 17.86 rmse on fold 0, against 11.94 once the labels are standardised. I did NOT reproduce those two numbers.
- What I verified in code this session:
  - The search's svm regressor is `SVR()` with sklearn defaults: `find_best_hybrid.py:2791`.
  - The distance-search scorer is `SVR(kernel='rbf', C=1.0)` fitted on scaled features but raw labels: `hybrid.py:2945-2960` (`X_train_scaled`, `y_train`).
  - The tuning pipeline scales features only, not labels: `find_best_hybrid.py:3590`.
  - A grep of `src/kirby/*.py` found no label scaling (`TransformedTargetRegressor`, mean/std of `y_train`).
- In the stored results, svm is clearly behind the other models on the three reaction datasets, whose yields have a large spread: best score per model (rmse): buchwald_hartwig rf 13.59, lightgbm 14.06, dnn 14.26, xgboost 14.71, svm 15.67; bh_doyle_plates rf 14.99, dnn 15.33, lightgbm 15.33, xgboost 15.70, svm 16.17; suzuki rf 18.97, lightgbm 19.03, xgboost 19.44, dnn 19.51, svm 20.75.
- Consequences to work out (not done): svm may have been unfairly demoted on every dataset whose labels have a large spread (the three reaction sets at least), so the two-model choice for those datasets could change; every svm number everywhere (as downstream model AND as internal scorer in the distance search) is affected where the label scale is large; label spread per dataset has not been checked. The two models for datasets whose pair includes svm are also affected if the label scale is not small. Tell Adelaide this before re-running anything; she decides what to re-run.

## 7. Open items and things I got wrong

Open, her decisions or checks pending:
- Concurrency: how to stop two experiments building the same per-fold search at once, and what to do if a job times out (section 1, item 8).
- Noise-experiment fold rule: how many outer folds (5 or 1) and how the noise experiments formalise their own per-experiment search. She postponed this. Do not decide it for her.
- Exemptions from the two-model rule: f29 and c23 compare models; whether knn stays in similarity experiments beyond the Step 9 list.
- suzuki allocation is HELD: power_2 and performance_weighted both score exactly 21.006 rmse, wglobal_power2 20.99, against 19.09 for softmax_T1. Looks like an untrained fallback value; cause not checked. It may relate to the svm label-scale defect or not; unchecked.
- Augmentation scope (section 4).
- Datasets with no stored distance search: bindingdb_kd, kiba, lhasa_herg.
- Whether any stored augmented searches exist on the cluster (code read only).
- Diverged fits are present in the data (rmse 61,766 on adme_hlm, 134,277 on bh_doyle_plates); the top-10 averages ignore them.
- The band is a relative percentage, so it cuts nothing on saturated metrics (tcga_luad_vs_lusc auc about 0.999, humanness_therapeutic auc about 0.97, flab_binding_adams auc about 0.63). Ties at rank 30 exist on suzuki and lhasa_herg; unequal row counts per category bias the "30 best" count.

Things I got wrong in this session that you should not repeat:
- Early budget, importance and allocation tables limited the scorer to the two chosen models. Corrected on 2026-10-03; sixteen decisions across fifteen datasets were revised and re-recorded.
- One bullet in my budget summary miscounted how many datasets keep the widest budget (2000). Do not rely on it; recount from the Step 2 (v3) table.
- Tables of 41 rows must be shown in the reply, not only in tool output; she could not see tool output.

## 8. Suggested order for the next chat
1. Read CLAUDE.md, MEMORY.md, this file, DECISIONS.md (authoritative sections only).
2. Settle with Adelaide: the concurrency design, and what to do about the svm label scaling (it changes numbers everywhere svm is used).
3. Examine every experiment against DECISIONS.md: which models it uses, whether it reads per-fold or whole-dataset searches, every extra `find_best_hybrid` call, every augmentation step. One experiment at a time, with file:line, recorded in a new section of DECISIONS.md or a new file she can read. Use her words: representations, folds, budgets, allocation methods, objectives, datasets.
4. Only then change code, and run the "does it run" check per `feedback_crash_check_means_timebox_not_finish.md` (normal settings, time-boxed, no shrinking).

## 9. Small leftovers from the svm work (2026-10-05), none blocking
- suzuki's local svm test produced no result (the process ended without output; cause unknown).
- bh_doyle_plates: the second representation (reactant_grover) failed in the local test with `IndexError: list index out of range`; cause not looked at.
- Target spread was measured only for the nine datasets screened (buchwald_hartwig 27.3 and bh_doyle_plates 27.5 are large; abagym_EGFR, abagym_VEGF, adme_hlm, adme_rlm, caco2, ddg_combined, flab_expression_koenig all under about 2). Not measured: suzuki, qm9, logd, herg_ki, the four other ginkgo sets, flab_thermo_tresanco, abagym_Ang2, abcov_kd, adme_mdr1, flab_binding_koenig, flab_expression_adams, kiba, bindingdb_kd. Assumed small.
- dnn cannot be fitted in the local environment (broken TensorFlow import), so the local svm-vs-picked-models comparison covered rf, lightgbm, xgboost only.
- Local test scripts: `scratchpad/precompute_cuts/scripts/svm_screen.py` and `svm_reaction.py`; results in `scratchpad/precompute_cuts/svm_screen/`.
- Two things read from code and never tested: that the per-fold searches get fresh svm numbers rather than old cached ones, and that no experiment still reads the old whole-dataset results for the reaction datasets.
