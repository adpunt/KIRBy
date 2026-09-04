# Finishing the experimental suite

116 experiment files. Complexity, noise, similarity, similarity_metrics, federated, OPIG.
Each read in full, twice. Framework and annitia were out of scope.

**107 of the 116 are decided.** The decision is written into each entry. They go into the
fix prompts as they stand. You do not need to read them unless you disagree.

What is left is below. Nothing else needs you.

---

## Everything still open

### 1. Compute — 18 experiments will not finish

This is the one I will not decide. Cutting an axis is your call. For each of the 18 the
section further down gives the size, what is spending it, and what can come off for free.
The free savings are already taken in the 107 decisions above — duplicate builds, cache keys,
baseline loops that recompute what the search already holds. What remains is axis choices.

`c05` `c07` `c10` `c13` `c15` `c16` `c17` `c23` `c24` `fed01` `fed03` `fed05` `n09` `n11` `n13` `opig_kingsmith_component_splits` `opig_scaffold_hybrid` `s07`

### 2. The similarity objective — on hold for f28

No similarity experiment passes `objective='similarity'`. Every hybrid the suite scores as a
similarity space was picked by a fitted model's accuracy. Whether to re-run under the geometry
objective depends on whether that objective works, which f28 answers.

`s02` `s03` `s04` `s05` `s06` `s07` `s08` `s10` `s12` `s13`

### 3. Nine files only you can call

**`c03`** — Delete, or rewrite to run over all 26 datasets. It is orphaned with no result files. c14 and c24 cover the transfer half but nothing else computes complexity-descriptor against winning budget. My lean: rewrite, it costs no model fits.

**`c14`** — Routed through `nested_cv_evaluate` on 2026-09-04 under the split audit, at 15,000 hybrid builds per task; it reports `delta_mean_of_folds`. Whether c14 runs at all is still yours: it is orphaned and has never written a file.

**`c18`** — Whether the diversity-versus-depth arms are rebuilt through create_hybrid, and where that lives. It is an allocation bake-off, so by your placement rule it belongs in framework/ as a new f-number, not as an edit here. That call is yours.

**`c30`** — Delete and read the same three arms off c31, or take the three-line repair. Deleting loses budget 100 and the 50/300 caps. c31 already ran those arms on 10 datasets and 3 models.

**`s07`** — Delete, or rewrite the held-out arm. 14 of 21 stored files carry no number and 4 more sit at exactly 1.0 for every arm. My lean: delete unless this question is going in the paper.

**`ranker_experiment`** — Delete and read framework/f22, or rewrite it per fold. f22 asks the same headline question leak-free and has a stored result on one of these two datasets. Keeping it means the other ten selectors matter to you. 2026-09-04: the per-fold rewrite was built under the split audit — every ranker re-runs inside each of the five folds on training-fold labels and training-fold pairs, and structural fidelity is scored on a held-out scaffold half — so the choice is now delete or run, not delete or write.

**`similarity_noise_cleaning`** — Quote only the optimism gap now and delete once framework/f23 has run, or keep it. f23 answers the ranking question and scores every space instead of a top-3. Depends on f23 landing.

**`sm_battery_moleculeace`** — Whether the 30-target battery runs at all. Nothing has ever been produced, so there is no re-run cost — only new compute. Run one target before sizing the job.

**`sm_data_loader`** — Whether the three local datasets are capped to the same size. logd is subsampled to 1,427 and herg_ki is whole at 1,415, so cross-dataset comparisons are not on equal footing.

### 4. Thirty-eight open questions inside decided files

The file is decided. One thing inside it is not.

| file | still open |
|---|---|
| `c06` | Whether the matched internal evaluator arm is worth making reachable. Today 'match' never resolves for a family-prefixed variant, so that arm is silently absent. |
| `c10` | Whether the hybrid recipe is re-searched under noise. Symmetric costs 56 noise-aware searches per dataset. |
| `c12` | Whether the modality-combination arm stays a raw concatenation or becomes a real hybrid. |
| `c13` | Whether molecular datasets get an explicit --new-reps list. Without one c13 returns 'no_new_reps' on all eight. |
| `c15` | 15c: widen the budget band so the far bucket is reachable, or drop the sub-analysis. As written the instability ratio can never be computed. |
| `c16` | Whether all three model families share one hybrid build. It changes what c16 measures. |
| `c17` | ~~The per-fold selection rewrite.~~ Done 2026-09-04 under the split audit. All five strategies now select on each fold's training rows and are scored on its held-out rows; the mRMR ranking is built once per fold rather than once per dataset, and recursive elimination runs per fold. Cost: one greedy mRMR pass per fold instead of one per dataset, and n_folds recursive-elimination fits per budget instead of one. |
| `c21` | Whether the multi-representation reference arm gains its own control. |
| `c23` | ~~Rewrite c23's headline through nested CV, or delete it.~~ Rewritten 2026-09-04 under the split audit: every (model, evaluator) pair goes through `nested_cv_evaluate` and each matched-minus-mismatched delta is paired per outer fold. Cost: five searches per pair instead of one. Whether f13 makes c23 redundant is still open. |
| `n01` | Whether the hybrid recipe is searched per fold. (Separate, and done 2026-09-04 under the split audit: reaction and drug-target affinity now get a cliff reference built outside the representations being graded, and TCGA records `cliff_reference_independent: False` instead of claiming one.) |
| `n02` | Whether the importance scorer behind the weighted distance is swept rather than fixed to lightgbm_gain. |
| `n06` | ~~Whether the replicate rows stay. Their feature vectors are identical, so a neighbour estimator recovers the group spread close to by construction.~~ Settled 2026-09-04 under the split audit: the rows stay and the score changes. The four nearest-neighbour estimators take `exclude_groups`, so a row never draws a neighbour from its own (drug, target) pair, and `leave_one_out` holds the whole pair out instead of relying on out-of-bag. The hybrid space is rebuilt protein-fold by protein-fold. On a fixture of 12 pairs x 4 replicates, distance_weighted_residual's correlation with replicate spread fell from +0.68 to +0.04. |
| `n07` | ~~Whether the hybrid is built on the noisy labels rather than the clean ones it then corrupts.~~ Done 2026-09-04 under the split audit: each fold builds its features from that fold and noise draw's noisy training labels, one build per (fold, noise seed) instead of one per fold, with the clean-label space kept as `per_model_per_strategy_oracle_features`. Still open: whether the RECIPE moves to a noise-searched pool, which needs searches the stored 0.05/0.10/0.20 gaussian lineage does not cover. |
| `s02` | TBD with the objective question. Also whether budget is swept rather than fixed at 100. |
| `s05` | Whether fixed_ref keeps its in-sample baseline. |
| `s08` | Which model scores accept-or-reject before the 16 datasets re-run. |
| `s10` | Whether budget becomes a reported axis, which is a separate server job. |
| `s12` | TBD with the objective question. |
| `scaffold_ddg` | Whether every complex still contributes 400 pairs regardless of its size. |
| `sm_base_rep_comparison` | What a group is on the adams libraries, which decides the fold grouping. |
| `nary_similarity` | How the roughness row is handled — it is a restatement of its own target. |
| `triplet_similarity` | Whether triplets are resampled each epoch. That changes the method. |
| `fed02` | Settled 2026-09-04. C5 stays; its three representations are now ranked on a 3-fold group-aware split inside the training rows. |
| `fed03` | Whether Part B's silos are rebuilt from DATASET_REPS, and whether Part C stays or moves into fed05. |
| `fed04` | Settled 2026-09-04. CLASSIFICATION_CUTOFFS gives each dataset its own cutoff in its own label units; a dataset with no row there is refused by name. |
| `fed05` | The dms silo. abagym_HER2 is not a registry key, so condition 5 always reports a skip. |
| `opig_cdr_flexibility` | Whether the combined CDRH3-plus-CDRL3 run stays. Its features cannot express which loop the label describes. |
| `opig_scaffold_hybrid` | Option 2, storing the per-representation feature count, is the only way to answer whether the hybrid used the scaffold features. 301,000 cold fits. |
| `structural_geometry_survey` | Whether the hybrid arm is repaired or removed. Both live htc jobs record 'no fitted rf result' until this is settled. |

---

## What I decided, in one list

The 107. Each entry below carries the same line plus the evidence.

| file | decision |
|---|---|
| `c01` | Swap in _predict_for_metric at _shared.py:1836 and record the width actually fitted. Keep the baseline arm. |
| `c02` | Do all four repairs. Only the PCA-control fix costs fits: 800 across four datasets. |
| `c04` | Memo the per-fold importance (8x on the dominant cost, no number changes) and add a single-representation comparator. The comparator is the standing baseline rule, not a choice. |
| `c05` | Take the search.baselines swap. It fixes the KeyError crash, the PCA confound and the hard-label AUROC in one edit and removes 4,550 fits per dataset. |
| `c06` | Store the per-fold scores and guard the borrowed importance on the fold count. Both free. |
| `c07` | Add groups, model, metric_fn and lower_is_better to build_fold_hybrid at c07:338. |
| `c08` | Remove the greedy dispatch at c08:76 and c08:179 — greedy is closed, this is not a choice. Add the four arguments at c08:220. |
| `c09` | Do the five local repairs and recompute the ANOVA over configurations rather than fold scores. |
| `c10` | Fix the local faults and switch to build_fold_hybrid. |
| `c11` | Five one-line repairs, the _emit fix, and take the PCA rotation off the baselines rather than adding it to the hybrid. Plus, 2026-09-04: search the recipe inside each outer training fold, so the hybrid arm pays the tax the ensemble arm already pays. |
| `c12` | Crash fixes, dead-name cleanup, print the honest_rescore number in the summary. |
| `c13` | Tie the blueprint to the model that scores it. Delete the nine stale result files so run_all stops skipping it. |
| `c15` | Take the free dedupe: 160 identical rebuilds per dataset inside a loop where the hybrid does not depend on what the loop varies. |
| `c16` | Fix the LogReg default index and the capped_full NaN column. |
| `c17` | Take the tie-break fix and the budget check. |
| `c19` | Move the build per fold, fix the recording, take the two one-liners. The eight stored files re-run either way. |
| `c21` | Cap the decoys to the donor widths conditions C and D actually use, and fix the scoring line. |
| `c22` | Do it. The re-run is 5 builds and 10 fits per dataset and the current per-representation SHAP cannot be defended. |
| `c23` | Fix the ridge evaluator names and the cache namespace. |
| `c24` | Apply the feature-set transfer and build_fold_hybrid together, plus the two one-liners. Same re-run. |
| `c20` | Run-repair, hoist the ranking out of the budget loop, rename the plateau field. |
| `exp_b` | Delete. Move the metric-direction gate into _alloc_harness first — four other scripts want it. c30 is the later rewrite of the same three arms, c31 the one after that. |
| `c43_signal` | Repair it. Ten model fits, and it is the anchor c43_step3 currently hardcodes as a literal. |
| `c43_step3` | Take option 1. It is the only file in this group that builds through the engine. |
| `c31` | Restore the full 15-scheme weighting family and re-run. Cutting 12 of 15 removed the axis the experiment exists to measure. |
| `c32` | Add the exact-budget top-up and trim, fifteen lines, then re-run. Median task is 23 minutes. |
| `n01` | Take the local fixes. Fix the empty-dict test so a failed hybrid pool stops wiping the per-representation baselines. |
| `n02` | Merge condition 5 into the conditions 3-4 loop and pair the fold sets. Saves 15 rebuilds per dataset. |
| `n06` | Fix the output shape and add per-representation spaces. The concatenation-only comparator is against the baseline rule. |
| `n07` | Add the caveat field and the six one-line fixes. |
| `n09` | Pass model and metric_fn to build_fold_hybrid, and store fold_scores. Neither costs a fit. |
| `n11` | Guard the two summary loops. Without it the grid runs for hours on a classification dataset and writes nothing at all. |
| `n12` | Route through build_fold_hybrid and add the zero-importance marker. |
| `n13` | Fix the summary tally so hybrid_sim stops counting as a baseline, then the bare create_hybrid. The tally recomputes from a stored file with no re-run. |
| `n14` | Set n_jobs from KIRBY_LGBM_THREADS in make_model, then re-run the smoke on caco2. |
| `n15` | Paired per-fold delta and NaN exclusion together. |
| `sim_hybrid_space` | ~~Scaler fix, then persist sim_info so a refused arm is distinguishable from a null result.~~ Done 2026-09-04 under the split audit: n11 and n13 take hybrid_sim as built and both persist a `hybrid_sim` block carrying the availability reason and per-fold info. `build_similarity_hybrid` also takes `selection_rows`, so each fold's blocks are chosen from pairs with both molecules in its training rows. |
| `handtrace_noise` | Delete. It tests local copies of two formulas, so it can never fail when the real ones change. |
| `smoke_harness` | Save the returned dict. |
| `smoke_noisy_search` | Add cache=True, then run it once so checks 6 to 9 are reachable at all. |
| `s01` | Hoist the pair sample, record the violation-rate reference, fix the dense-bound comment. |
| `s02` | Three one-line fixes: the weighted-versus-weighted delta, the recorded blueprint, the lineage. |
| `s03` | Fix the density denominator and the JSON keys, then re-run. The 22 files on disk answer none of its questions. |
| `s04` | Build ECFP4 from the SMILES so the cliff rule stops depending on the pool, and keep per-fold cliff scores. |
| `s05` | budget_override, hoisted build, search-only helper, delta_mean_of_folds and task_type in one pass, then re-run 22 datasets. |
| `s06` | Find the NaN in the weighted arms, then add per-representation arms to both blocks. |
| `s08` | budget_override and a kNN model factory together. Without both the file answers neither of its questions. |
| `s09` | Three one-line repairs plus an individual-representation comparator to replace the concatenation. Then point the submit script at this module. |
| `s10` | Pass budget 100 explicitly and store it, then the four repairs. |
| `s11` | Route both builds through build_fold_hybrid with groups, the model and the metric. Refuse the hybrid conditions when there is no blueprint and record that. Then the four repairs. |
| `s12` | Fix the scanned-set contamination, write the recipe into the result, do the housekeeping pass. |
| `s13` | Pad the fold lists first — it can already be producing a wrong delta. Then the singleton-cluster fix, the dimension match, and drop the discarded build. |
| `_summary_keys` | Leave it. |
| `_run_similarity_locally` | Take the normalising version so the smoke gate actually runs the five experiments, then fix the allocation set. |
| `simeval` | Make the four returns NaN, set cliff_auroc False in the activity-grounded table, add the noise_recovery_defined flag. |
| `rankers` | Delete the five unused scorers plus RANKERS and LABEL_FREE. |
| `blacklist` | Delete. src/kirby/similarity/selection.py already does the structural half per fold and is wired in. Nothing imports this and nothing stores its output. |
| `scaffold_experiment` | Use the registry grouping, delete the position_gated row (it is the Grantham grader itself), fix the filename so a smoke run stops overwriting the full run, re-run all five. 2026-09-04: the five-nearest-neighbour score is now refused below 99% single-mutation variants, the same threshold position_gated was already refused on, and frac_single, the position-group count and the variant count go into every output row. |
| `scaffold_ddg` | Chain and complex keys, and the crash message. |
| `scaffold_reps` | Fix blosum_matrix — one line, and it sets a number in ten result files. Take the wildtype as an argument before anyone quotes an adams or phillips_h3 result. |
| `scaffold_feature_isolation` | Fix the damage share, add --cap and --output. Hold the re-run until blosum_matrix lands, since that sets every number here. |
| `sm_feature_type_study` | Three one-line fixes, then delete run() and main() and keep the module. Nothing reads its JSON. |
| `sm_feature_ablation` | Blacklist swap and the full-size 3D block, plus the pair split for the fine-2D selection. 2026-09-04: molecules are split by whole scaffold group instead of pairs; the fine-2D ranking, the z-score mean/sd and the drop-one sequence are fitted on one half and every number is measured on the other, with the old fit-equals-score numbers kept as a second arm. |
| `sm_base_rep_comparison` | Paired bootstrap and the classification dispatch. 2026-09-04: frac_single, the position-group count and the variant count are written into every row, the score is refused below 99% single-mutation variants, and two flags record whether a rho is arithmetic — every train-test distance identical, or every variant drawing the same five neighbours. |
| `sm_aggregate` | Recording fixes, label-free ranking, and run the noise sweep. One injection at one dose is what makes the two noise columns unquotable. 2026-09-04: A6_dca's keep order is taken against MCS distance, not the activity gap it is scored on, and the keep order, what was dropped and the highest correlation between two metrics go into the row. |
| `creative_hybrids` | Move MACCS and the feature-count sweep to the honest builder, then delete this file. |
| `creative_hybrids_v2` | Fix the output path collision first, then fold into the honest arm next door. |
| `winning_recipe_honest` | Delete. similarity_noise_cleaning already splits at molecule level, which is what this file set out to do. Correct the two documents quoting 0.410 and 0.687. 2026-09-04: under the split audit this file was moved to the same molecule split — scaffold halves, pairs kept only when both endpoints are on one side, z-score mean/sd from the select half — so if it is deleted, nothing measured is lost. |
| `winning_recipe_vs_standard` | Fix the output path, then add the field-default comparator the noise-cleaning script lacks. 2026-09-04: every row, comparators included, is now scored on a held-out scaffold half, and struct_fidelity_insample carries the old number beside it. |
| `structural_embeddings` | Fix create_map4 (one line in src/kirby/representations/molecular.py), add the molecule-size control, run all three datasets before quoting anything. |
| `structural_yardstick` | Take the rewrite. It removes 9 minutes of repeated MCS work per dataset, puts logd in the suite row space, and reports the size control that changes the answer. |
| `shape_similarity` | Both fixes in one pass and re-run all three. Seconds, because the conformers are cached. |
| `aug3d_similarity` | Add the molecule-size control and re-run. The e3fp coefficient conclusion reverses on all three datasets without it. |
| `learned_embeddings` | All three, before any run of the MoleculeACE battery. The structural arm currently trains on about 140 triplets. |
| `extra_fingerprints` | Build MAP4, add adme_solubility, store per-fold values, re-run all three. 120 kNN evaluations, zero model fits. |
| `augmented_descriptor_reps` | Two one-line fixes: consult the cache before the package probe, and z-score per fold. |
| `nary_similarity` | NaN, symmetry and provenance fixes. The whole file re-runs in under 5 seconds. |
| `triplet_similarity` | Write the JSON and return the surviving triplet count before any full run. |
| `sm_mcs_cache` | Add an output flag and honour SIM_DATA_DIR, and record the SMILES-ordering hash. Skip the timeout audit — 87 minutes per dataset and it changed nothing measurable. |
| `sm_nanrank` | Leave it. Twelve scripts import it and all three functions do what their names say. |
| `sm_build_local` | Deduplicate by InChIKey to match the framework loader, and record the source CSV and the SMILES-ordering hash. |
| `sm_build_moleculeace` | Take the activity-column fallback fix. One line, no rebuild. |
| `sm_build_adme` | Merge into build_local_datasets and delete. Same question, and the stricter filter changes no stored row. |
| `fed01` | Pass the metric, and give the three noise seeds three different draws. Today all nine runs flip the same labels. 2026-09-04: one recipe search per noise level, trained on labels corrupted at that level (6 searches per dataset, not 1); condition 6 rebuilt as condition 5 on cleaned rows. |
| `fed02` | Route C4 and C5 through build_fold_hybrid, pass C6 the metric, and fix the duplicate silo assignment on caco2. 2026-09-04: C5's three representations ranked train-only, 5 representations x 3 inner folds of extra fits per outer fold. |
| `fed03` | Rename the dataset key so the submit path stops exiting, pass the metric each part reports, and load herg_ki for Part A's first silo. |
| `fed04` | Pass groups, the resolved evaluator and the searched recipe. Three small edits, no stored result to lose. 2026-09-04: per-dataset classification cutoff; the recipe search no longer reads data['precomputed_searches'] and is ranked on auc, not rmse. |
| `fed05` | cache=True and a per-silo recipe. Both one-line, nothing to lose. 2026-09-04: every silo's 80/20 split is drawn once, up front, and the recipe search runs on that silo's training rows only. Same number of searches, on 80% of the rows. |
| `lhasa_herg_loader` | Fix the PKL branch so it derives the label column the SDF branch already derives. |
| `caco2_runner` | Add --reps to run_dataset.py and delete all three runners together. |
| `herg_runner` | Settle on 'herg' as the spelling, then delete with the other two. |
| `logd_runner` | Delete with caco2.py and herg.py once run_dataset.py takes --reps. |
| `smoke_driver` | Set the thread variables and fix the default representation list. |
| `opig_activity_cliff_qsar` | Build the hybrid per fold and recompute the cliff numbers from those predictions. Fix the sign convention and the paired delta in the same pass. |
| `opig_blueprint_transfer` | Add groups= at line 250 and make the delta paired. The stored file is a smoke run, so nothing real is invalidated. |
| `opig_cdr_flexibility` | Add the honest block. No fits, and it is the only thing between this file and an answer to its third question. |
| `opig_cgr_comparison` | Take the honest-block arithmetic and the per-model recipe. Neither needs a re-run. |
| `opig_ginkgo_developability` | Set metric='spearman' and match the representation preprocessing across the two arms in one edit, print the honest block, re-run once. |
| `opig_graphinity_ddg` | Fix the submit flag and the pdb_ids filter. Until both are done it either does not start or reports an inflated transfer number. |
| `opig_humanness` | Delete lines 316-321, print delta_mean_of_folds, take the four one-line fixes. The held-out table already answers question 2. |
| `opig_kingsmith_component_splits` | Restore metric='rmse' and keep the per-fold scores before this is submitted. Both cheap, both change which recipe wins. |
| `opig_trastuzumab_enrichment` | Take the per-fold enrichment rewrite. It is the file's own question 3. |
| `opig_scaffold_hybrid` | Options 1, 3 and 4 now. They cost nothing and stop the file printing a one-fold delta as a result. |
| `opig_scaffold_similarity` | Fold into scaffold_experiment and feature_type_study. That fixes the 0.0 zeros, the uneven grader count and the missing whole-sequence comparison at once. |
| `sim_optimized_hybrid` | Pass budgets explicitly and remove greedy and dominant_greedy from SIM_ALLOCATIONS. Both are closed and this file can still select one as its winner. |
| `structural_geometry_survey` | Add the base-representation-alone arm. Without it no augmentation delta can be computed from the file. |
| `read_sim_optimized_results` | Fix the three print blocks alongside the experiment, not before it. |

---

## Fix prompts

`FIX_1_DEAD_IMPORTS.md` — the five files that cannot run.  
`FIX_2_FEDERATED_METRIC.md` — nine calls passing a model with no metric.  
`FIX_3_HONEST_RESCORE.md` — 23 files reporting a searched maximum against an unsearched baseline.  
`FIX_4_SMALL_DEFECTS.md` — AUROC on hard labels, undefined correlations written as 0.0, dead flags, mangled cache names.  
`HANDOFF_CREATE_HYBRID_AUDIT.md` — sweep every `create_hybrid` call, including `framework/`.

The 107 decisions above are not yet in those prompts. Say the word and I will write the
per-file work into them.

---

## What the split audit changed here, 2026-09-04

Eight findings from `SPLIT_AUDIT_2026_09_03.md`, the data-and-plumbing package. Each entry
in that document carries the detail; this is what moved under the files in this one.

- **Resume no longer trusts a file's existence.** Every result file written by the
  complexity, similarity, noise and federated helpers now carries an `input_fingerprint`:
  row count, label hash, group hash, group count, split count, and the sorted
  representation names with their widths. The three helpers with a resume path compare it
  and recompute when any field moved, naming which one. A file written before the stamp has
  none and always recomputes. A code change is still invisible to it — `--force-rerun` is
  still the way to clear one.
- **Fold 0's term is dropped from both arms of every leave-one-fold-out delta.** The search
  filters its own recipe list on fold 0 — mid-flight under `aggressive_pruning`, which is
  what the cluster submitters default to, and again in the end-of-fold-0 strategy prune,
  which fires under `default_pruning` too because five allocation methods beat a
  `strategy_top_n` of 3. `SearchResult.fold0_pruned` records it, the persisted block carries
  `exclude_folds`, and 17 readers pass it. Deltas now cover four folds, not five, wherever
  the candidates came from a search.
- **A recipe picked over every row now says so.** `get_best_search_for_downstream` returns
  `selection_scope: 'whole_dataset'`; s02 and s08 pass it into `honest_rescore` and s09
  prints it beside the number. Rebuilding per fold fixes the feature weights, not the choice
  of allocation method, budget, importance method or evaluator.
- **f27's `recipes` and `both` pools build their representations.** The eleven sim_ recipes
  have no whole-dataset matrix, so `load_data` never built them and the pools came back
  holding the general representations under the recipes' name. They are now built as
  per-fold factories and passed to `nested_cv_evaluate` as `outer_rep_builders`, so each is
  refit once per outer fold and scored on the outer test fold in the hybrid and as a
  baseline. A requested one that fails to build now raises.
- **The four gb_ genomics datasets group by k-mer content**, not by an md5 of the sequence.
- **The seven Jain endpoints keep their split**, and the loader now records that it produced
  one antibody per group, so the claim reads as unseen-antibody and not unseen-germline.
- **TCGA batch correction is gone**, for every TCGA task. The folds hold out whole tissue
  source sites and ComBat has no batch parameter for a site it never saw.
- **s07's held-out noise score is built per outer fold**, recipe and features both, so the
  hybrid no longer picks its features using the labels of the fold it is scored on.
- **s04, s05, s12 and s13 search the hybrid recipe inside each fold**, on that fold's
  training rows, through a new shared `get_fold_best_search`. It runs the same search
  `precompute_hybrid_searches` runs, on the fold's training slice, filed under the fold's
  nested-CV cache namespace, and returns the same `(search, metadata)` pair the
  whole-dataset pool helper returns — so a fold loop swaps one call for the other. The
  folds now come from `nested_cv_outer_folds`, the split those cache entries are named
  for; it is the same group-aware split as before. Each file records which recipe each
  fold picked, because they can differ. Cost: one search per fold per dataset instead of
  one per dataset, and one per model per fold in s12.
- **s02 refits the hybrid's features per fold.** Its held-out SAC used to slice the
  whole-dataset matrix, whose features were chosen with every held-out label in hand,
  while the single representations it is compared against got no selection at all. The
  recipe there is still the whole-dataset pool's, and the file now says which of the two
  is refit.
- **s08 fits every model at the budget being swept.** The shortcut that copied
  `search.best_by_model[model]['fold_scores']` in whenever the budget matched is gone, so
  the six budgets give six scores instead of five identical ones and the search's own.
  s08 also scores kNN now — its model list is `DOWNSTREAM_MODELS` plus `knn`, local to
  that file — which is what s09's `optimal_dim_ratio` needs.
- **s11 searches the hybrid's five settings on each seed's seed rows** (your call, the
  alternative being the fixed defaults already in the file). One search per seed, shared
  across the hybrid conditions, cached under a namespace naming the seed. The pool labels
  the Bayesian loop reveals one at a time no longer reach the hybrid before the loop does.
- **`firstpass_notready.py`'s s05 table reads the held-out delta**,
  `honest_rescore_result.knn.delta_mean_of_folds`, not `fixed_ref`, whose baseline is the
  best representation at its best k on the same folds. `fixed_ref` is still written and
  still says so in its own note.

---

## Two things about which code is running

Not decisions. They change what "broken" means everywhere below.

**Your laptop and the server run different engine code.** `src/kirby/` has 642 uncommitted
lines. The server pulls from git. It does not have them. About twenty defects in this
document are already fixed here and still live there.

**Four of the five similarity submit modes run a different file.**
`run_similarity_experiments.sh` modes `quick`, `full_a`, `full_b` and `full_cross` run
`tests/similarity_experiments.py` — 2,335 lines with its own copies of s01 through s09. Only
`full_mol` runs the modules in this document. `submit_all_datasets.sh similarity` also runs
the modules.
## The 18 that will not finish

For each one: the size, what is spending it, and what can be taken off without removing an arm.
I have cut nothing. The axis choices are yours.

### `c05` — complexity/c05_learning_curves.py

**Size.** Per dataset under run_all: 5 sample sizes x 3 seeds, collapsed to 1 seed at the full-data point = 13 searches. Each search is 5 allocation methods x 3 importance methods x 5 internal evaluators x 8 budgets x 5 downstream models = 3000 configurations x 5 folds = 15,000 model fits. 13 x 15,000 = 195,000 fits.

**What can come off, no axis removed:**

- Delete the separate baseline loop at c05:253-270 and read the per-representation numbers out of search.baselines instead. The search already computes them, on the same folds, at the same seed, at full width for anything under 5,000 features, and using probabilities where the metric is AUROC (find_best_hybrid.py:4149-4291, _evaluate_single at 3280-3360).
  *Saves:* Removes about 4,550 duplicated model fits per dataset: 10 representations x 7 models x 5 folds x 13 subsample searches. Adds none.
- If you want the separate loop kept, the alternative is one line at c05:254: skip a model that was not resolved at the full row count.
  *Saves:* Saves the wasted gp baselines: 10 representations x 5 folds x 3 seeds x the number of sample sizes under 1,000. On buchwald_hartwig that is 300 gp fits.
- At c05:226 keep the true dataset_name and pass cache_namespace=f"n{n_sub}_s{seed}" instead. That is the supported field (find_best_hybrid.py:5945-5948) and c06:179-180 already does the equivalent.
  *Saves:* Any subsample search already cached under the old invented names becomes unreachable and re-runs. How many such files sit on the cluster is not established here.

### `c07` — complexity/c07_data_complexity_partitions.py

**Size.** 3 partitions x 2 halves = 6 searches per dataset, each on up to 2000 rows (the whole dataset is first capped to 2000 rows by whole scaffold groups at c07:91). Each search runs the default 7-model set: 5 allocation methods x 3 importance methods x sum over models of (evaluators x budgets under MODEL_MAX_BUDGET) = about 3,465 configurations x 5 folds = 17,325 fits. 6 x 17,325 = about 104,000 model fits per dataset, plus 4 models x 5 folds x 6 halves = 120 fits for the full-data-recipe arm.

**What can come off, no axis removed:**

- Extend the full-data recipe list at c07:59 to the engine's default seven models, so every model the per-half search can pick has a full-data counterpart.
  *Saves:* 3 more get_best_search_for_downstream lookups per dataset. Each is a pool read when the precompute cache is warm and a full find_best_hybrid when it is not (_shared.py:4218-4240).
- Give the two arms the same fold count: pass n_folds=config['n_splits'] to find_best_hybrid at c07:230, or drop config['n_splits'] and use 5 in the second arm.
  *Saves:* First choice: the six searches drop from 5 folds to 1 on the command-line path, and every cached row built at 5 folds is unreachable. Second choice: the second arm goes from 1 fold to 5, so 4 x 5 x 6 = 120 fits instead of 24.
- At c07:240 keep the true dataset_name and pass cache_namespace=f"{partition_name}_{subpop_name}" instead (find_best_hybrid.py:5945-5948).
  *Saves:* Any partition search cached under the old invented names re-runs. How many exist on the cluster is not established here.

### `c10` — complexity/c10_noise_robustness.py

**Size.** Eleven regression conditions: ten dose-matched at 6 levels plus censoring at 7 = 67 (condition, level) points, times 5 folds. Phase 1 = 10 reps x 67 x 5 = 3,350 random-forest fits. Phase 2 = 8 budgets x 67 x 5 = 2,680 create_hybrid builds plus 2,680 forest fits; each build re-scores the 10 representations with the internal evaluator, roughly 29,500 further fits.

**No free saving found.** Shortening this one means changing an axis, which is your call.

### `c13` — complexity/c13_forward_compatibility.py

**Size.** One find_best_hybrid = 5 allocations x 3 importance methods x 5 internal evaluators x 8 budgets (FULL_BUDGETS, which c13 uses regardless of config['budgets'] — c13:344-350) = 600 configurations x 5 inner folds = 3,000 create_hybrid builds, each scored against the downstream model set, so roughly 15,000 model fits before adaptive pruning. With augmentations present the pool asks for three aug conditions (none/uncapped/capped), which are three separate cache files, so each condition costs 3 searches. Per new representation that is 3 conditions x 3 aug conditions = 9 searches.

**What can come off, no axis removed:**

- Tie the blueprint to the model that scores it: replace the five-candidate loop at c13:209 with a single get_best_search_for_downstream call for best_b_model.
  *Saves:* Fewer pool calls, no change in searches — the underlying searches are cached on disk, so the five candidate calls already reload rather than recompute after the first. The nine stored files are already unusable for comparison, so nothing extra goes stale.

### `c15` — complexity/c15_ml_theory_depth.py

**Size.** Per dataset, 10 base representations, 5 folds, 8 budgets. 15a: 8 full-data create_hybrid calls. 15b: 9 more (8 budgets plus a third rebuild of the b=50 hybrid at line 321).

**What can come off, no axis removed:**

- Take the repeated builds out of 15a, 15b and 15d. Build one hybrid per budget in run() and hand the dict to all three; in 15d hoist build_fold_hybrid above the trial loop, since the hybrid depends only on the real training labels.
  *Saves:* Per dataset with 5 folds and the 8 default budgets: 15a 8 builds, 15b 9, 15d 240, total 307 including 15c's 50. This removes 8 duplicates between 15a and 15b, the third budget-50 build, and 160 of 15d's 240, so 169 of 307 go.

### `c16` — complexity/c16_feature_selection_as_regularization.py

**Size.** Regression families give 6 Ridge alphas + 4 RF depths + 4 SVM C values = 14 variants. Budgets = FULL_BUDGETS filtered to b <= n and b <= total_dim (up to 8) plus capped_full = 9. 5 folds.

**What can come off, no axis removed:**

- Build the hybrid once per budget and reuse it across all 14 regularization settings, instead of passing model=model_factory() into build_fold_hybrid at line 154.
  *Saves:* 630 build_fold_hybrid calls per dataset (14 x 9 budgets x 5 folds) drop to 45. At 336 s per call, measured on flab_thermo_tresanco, about 54 hours per dataset down to about 4.

### `c17` — complexity/c17_information_theoretic_decomposition.py

**Size.** The cost is mRMR, not model fits. With n_samples >= 2000 the largest budget is 2000, and _greedy_mrmr's inner loop makes one MI estimate per remaining candidate per selected feature: about 2000^2/2 = 2,000,000 calls to estimate_mutual_information_knn. Each of those runs mutual_info_regression four times (once for the estimate and three times for the permutation null, _shared.py:2270), so about 8,000,000 sklearn kNN MI fits on 2000 rows.

**No free saving found.** Shortening this one means changing an axis, which is your call.

### `c23` — complexity/c23_evaluator_model_matching.py

**Size.** Per dataset, roughly ten constructible models (gp drops above 1000 rows, svm above 10000). Matched/near/far = 30 find_best_hybrid calls, each 1 model x 1 evaluator x 8 budgets x 5 allocations x 3 importance methods = 120 configurations x 5 folds = 600 downstream fits, so 18,000. The neural-size arm adds 600.

**What can come off, no axis removed:**

- Pass cache_namespace=f"{model_name}_{evaluator}" and leave dataset_name intact. The parameter exists at find_best_hybrid.py:5948 and is applied to cache paths only, at line 6265-6266, below the line where the policies resolve.
  *Saves:* One line at c23:100 and one at c23:249. On the capped datasets it removes two of eight budgets from every one of the 42 calls, roughly a quarter of the configurations.

### `c24` — complexity/c24_blueprint_transfer.py

**Size.** One full default find_best_hybrid per outer fold: 5 allocation methods x 3 importance methods x 8 budgets x 5 internal evaluators x 7 default models = 4,200 configurations x 5 inner folds = 21,000 downstream fits. Times 5 outer folds = 105,000 downstream fits per dataset, before create_hybrid's own step-wise allocation evaluation inside every one of those 21,000 configurations. The transfer step adds only 20 builds and 20 fits.

**What can come off, no axis removed:**

- Replace both bare create_hybrid calls with build_fold_hybrid (_shared.py:3559). It forwards groups, the internal evaluator, the augmentation strategy and margin and fraction, the candidate ranking, the augmentation budgets and the blueprint's own representation scores, and it refuses model without metric_fn.
  *Saves:* No change in the number of builds or fits. The builds get slower, because passing a model means the block-size choice fits that model (hybrid.py:3302-3311) rather than the cheap internal evaluator.
- Pass cache_namespace=f"fold{fold_i}" and leave dataset_name intact at line 191.
  *Saves:* One line. Removes roughly a quarter of the configurations on capped datasets.

### `fed01` — federated/fed01_noise_heterogeneity.py

**Size.** create_hybrid calls per (fed_model, seed) run: 5 folds x [6 sigma levels x (C0 1 + C5 5 silos + C6 1) = 42, plus 5 for the federated-hybrid detector setting, plus 5 smooth factors x 5 silos = 25] = 5 x 72 = 360. The sweep is 3 fed models x 3 seeds = 9 runs, so 3,240 create_hybrid calls per dataset, each scoring all 10 representations on an internal split and running treeshap importance on all 10. Separate plain model fits: per sigma, 6 teachers + 25 teacher cross-validation fits + C1 + the C4 student + 4 downstream = 37; x 6 sigma x 5 folds x 9 runs = 9,990, plus roughly 1,600 more from Phase 5's re-clustering and Phase 3's eight detectors over six settings.

**What can come off, no axis removed:**

- For the identical noise draws: pass seed=fed_seed to inject_noise_per_silo at fed01:249, 608 and 686.
  *Saves:* Three lines. No change to the number of fits.
- For the missing precompute: run precompute_hybrid_searches for herg too, by lifting the block at federated_helpers.py:2338-2365 out of the non-herg branch.
  *Saves:* No number changes. The same search runs, once, earlier.
- Decide which representation list herg is meant to use. Today fed01 gets the 'all' preset, up to 33 base representations; DATASET_REPS['herg']['general'] names ten and is the source of truth every other suite reads.
  *Saves:* One line either way (pass DATASET_REPS['herg']['general'] as rep_names at federated_helpers.py:2248). The compute difference is not established; representation generation is cached per SMILES set.

### `fed03` — federated/fed03_synthetic_private.py

**Size.** Part C dominates. find_best_hybrid at fed03:1000-1007 runs with defaults: 8 budgets (FULL_BUDGETS) x 5 allocation methods x 3 importance methods x 5 internal evaluators = 600 configs, x 5 folds x 5 downstream models (DOWNSTREAM_MODELS) = 15,000 model fits per silo, x 4 silos = 60,000 fits, on top of 3 importance computations per representation per fold over the rep_preset='all' set (~25 reps, fed03:983). run_all wraps every experiment in a 3 fed_model x 3 seed sweep (federated_helpers.py:252-254, 2395-2410), so this is entered 9 times and only the on-disk find_best_hybrid cache stops it repeating.

**What can come off, no axis removed:**

- Change one word in tests/slurm_scripts/submit_all_datasets.sh:55, from lhasa_herg to herg, so Part A can be submitted.
  *Saves:* Nothing goes stale. tests/results/federated does not exist, so no stored fed03 file changes.
- Load the ChEMBL hERG Ki set for Part A's first silo instead of reusing the Lhasa training set that run_all hands in.
  *Saves:* About 15 lines. Pooled rows go from about 12,155 to about 13,567 (1,412 + 7,772 + 4,383 minus overlap).
- Record, per fold, what C1's per-silo scores were and pick the baseline silo by its training score, not its test score.
  *Saves:* About 10 lines in each aggregation helper. One extra model fit per silo per fold to score on training rows, so 3 per fold in Part A: 135 across 5 folds and 9 runs.

### `fed05` — federated/fed05_antibody_federated.py

**Size.** run_all sweeps 3 fed_models x 3 fed_seeds = 9 full runs. Each run calls find_best_hybrid once per loaded silo with budgets=FULL_BUDGETS (8) x 5 allocations x 3 importance methods x 5 internal evaluators = 600 configurations, x 12 default models x 5 folds = 36,000 model fits per silo. Two silos load, so 72,000 per run.

**What can come off, no axis removed:**

- Search: pass cache=True at fed05:199. As of 2026-09-04 that search runs on the silo's training rows only, so a cached row from a whole-silo search is a miss.
  *Saves:* One line. Turns 18 searches into 2 with the two silos that load today.
- Naming: file fed05's output under a fixed name, 'antibody', and drop fed05 from SINGLE_DATASET_EXPERIMENTS so it is not repeated once per molecular runner.
  *Saves:* Two lines, at federated_helpers.py:226 and the save call at 2426. Removes two of the three fed05 invocations — about two thirds of fed05's total cost across the suite.

### `n09` — noise/n09_lnl_survey.py

**Size.** Per dataset at --mode full: 6 sigmas x 5 folds = 30 inner find_best_hybrid calls. Each searches DEFAULT_ALLOCATION_METHODS (5) x DEFAULT_IMPORTANCE_METHODS (3) x DEFAULT_INTERNAL_EVALUATORS (5) x NOISY_SEARCH_BUDGETS (3) = 225 configs x 3 inner folds = 675 create_hybrid constructions plus fits, plus 10 representation baselines x 3 folds = 30. That is 30 x 705 = about 21,150 fits, of which 20,250 are full create_hybrid builds (each re-scores 10 representations plus importance).

**What can come off, no axis removed:**

- Do NOT swap the inline re-search for the cached noisy precompute. Recorded here so it is not proposed later.
  *Saves:* Not applicable; this is a decision to leave the design alone.
- The compute, stated so you can decide where it runs. 6 levels x 5 folds = 30 inner searches per dataset.
  *Saves:* Cost not established. I did not verify whether the file-level metadata check alone stops a noisy run reading a clean baseline file.

### `n11` — noise/n11_estimation_remediation_grid.py

**Size.** Per dataset at --mode full: 6 sigmas x 5 folds x 3 noise seeds = 90 cells. Spaces: base + predictive hybrid + hybrid_sim + 9 other base representations = 12. Estimator keys: 10 BASE + 3 TORCH + ngboost = 14 estimators, of which 5 are neighbour-based and run at 2 distance metrics, giving 5x2 + 9 = 19 keys.

**What can come off, no axis removed:**

- Fit the clean baseline once per (fold, seed) rather than once per fold, so the gap subtracts two means over the same 90 numbers.
  *Saves:* 60 extra random forest fits per dataset, against 574,560 already run. Every recovery number moves by however much the two means differ.

### `n13` — noise/n13_noise_distributions.py

**Size.** 10 conditions x 6 levels x 5 folds x 3 noise seeds = 900 inner cells per dataset. Per cell: 5 noisy-baseline fits + 5 honest-hybrid fits + 3 spaces x 12 LNL methods = 36 DNN trainings at 50 epochs + 3 spaces x 3 estimators x 4 models = 36 weighted fits, i.e. 82 model fits.

**What can come off, no axis removed:**

- Alternative to the previous option: build one hybrid space per downstream model instead of one shared space, reusing the `_hyb_space_cache` the honest arm already fills in the same cell.
  *Saves:* Track B's per-cell fits stay at 36 but read from spaces already built, so zero extra hybrid builds. Track A gains nothing.
- For the honest re-score arm, either pass every model's distinct blueprint as a candidate for every model, or stop describing the arm as a leave-one-fold-out re-selection and report it as the paired per-fold delta it is.
  *Saves:* First: with 5 models and up to 5 distinct blueprints, per-cell honest fits go from 5 to at most 25, i.e. up to 18,000 extra model fits per dataset over 900 cells, with no extra hybrid builds since the spaces are already cached.
- Compute the level-0.0 cell once per (fold, seed) and copy the result into the other nine condition names.
  *Saves:* Saves 135 of the 900 inner cells: 11,070 model fits per dataset, of which 4,860 are 50-epoch neural-net trainings, and up to 810 hybrid builds. At the 261.9 s per build recorded in _shared.py:3739-3742 that is up to 59 h of hybrid construction per dataset.

### `opig_kingsmith_component_splits` — opig_kingsmith_component_splits.py

**Size.** Per dataset the outer folds are 23 + 15 + 4 + 3 = 45. Each outer fold runs a full cold search: n_train ~3,790 so gp is skipped and svm survives the uncapped cap, leaving 6 models; 8 budgets (all of FULL_BUDGETS fit under n); 5 allocation methods; 3 importance methods; 5 internal evaluators = 3,600 configs x 5 inner folds = 18,000 model fits, plus 10 x 6 x 5 = 300 search-internal baselines. 45 x 18,300 = ~824,000 fits per dataset, ~1.65 MILLION across buchwald_hartwig and bh_doyle_plates.

**What can come off, no axis removed:**

- Put `metric='rmse'` back into the search call at line 424, where 9bd48ac removed it.
  *Saves:* One line. It invalidates no stored result, because there is none.
- Submit it as eight tasks instead of one, using the flags that already exist: `--datasets` and `--splits`, one task per (dataset, component axis).
  *Saves:* Eight tasks. The largest, additive on either dataset, is 23 folds x about 18,300 fits = about 420,000; the smallest, base, is 3 folds = about 55,000.

### `opig_scaffold_hybrid` — opig_scaffold_hybrid.py

**Size.** Per inner search: 5 allocation methods x sum over models of (budgets x 3 importance methods x evaluators). Models resolved from models=None are rf, xgboost, knn, lightgbm, mlp, svm (gp is dropped, _max_samples=1000 < n~4000). Budgets are FULL_BUDGETS=[10,25,50,100,250,500,1000,2000] capped per model: 8 for rf/xgboost/lightgbm/knn, 7 for mlp (<=1000), 6 for svm (<=500).

**What can come off, no axis removed:**

- Make nested_cv_evaluate record, per outer fold, how many features each representation contributed — the numbers are in `finfo` at find_best_hybrid.py:8311 and are currently read only for a total width — then re-run all four datasets.
  *Saves:* Per outer fold: 5 allocations x 3 importance methods x (rf 8 budgets x 4 evaluators + xgboost 8 x 5 + lightgbm 8 x 4 + knn 8 x 5 + mlp 7 x 4 + svm 6 x 4) = 2,940 configurations, each over 5 inner folds = 14,700 fits, plus 12 representations x 6 models x 5 inner folds = 360 baseline fits. About 15,060 per outer fold, x 5 outer folds x 4 datasets = about 301,000 model fits, before adaptive pruning trims some after inner fold 0.
- Either wire the three flags through — --models and --budgets into run_nested_cv's kwargs, --max-folds into n_outer_folds — or delete them from the parser.
  *Saves:* Zero model fits either way. No stored result invalidated.

### `s07` — similarity/s07_noise_bridge.py

**Size.** No downstream model is fitted at all — the cost is distance matrices. _contamination_blocks scores 10 representations plus 2 hybrid candidates = 12 arms, and each arm loops over every fold recomputing cdist(X_train, X_train) plus a full argsort. Under the per-group scheme the fold count is the number of distinct scaffold groups G, so the work is 12 x G x O(n^2).

**No free saving found.** Shortening this one means changing an axis, which is your call.
## Delete or keep — 30 files

Each one: what it asks, what already answers that, and what you lose if it goes.
Some of these delete only a block, not the file. Nothing here deletes itself.

### `blacklist` — similarity_metrics/blacklist.py

**Asks.** Which individual columns inside a feature block make the block's similarity geometry worse, stably across scaffold folds?

**Proposal.** Delete it. src/kirby/similarity/selection.py:57 already answers the structural half per fold, corrects exactly the fitted-noise problem this file has, and is wired into src/kirby/similarity/recipes.py:125 and two scripts under tests/.

**What you lose.** Nothing imports this file and nothing stores its output, so no experiment and no result file changes. What is lost is the half that has no equivalent: the yardstick that scores by predicting each molecule's activity from its five nearest neighbours, and the count of how many scaffold folds independently dropped a feature. selection.py refits per fold instead of producing one fixed list, so it also cannot feed find_best_hybrid's feature_blacklist.

### `c03` — complexity/c03_cross_dataset_transfer.py

**Asks.** Does a dataset's intrinsic complexity (effective rank, participation ratio, C1/C2/L1/S3) predict the winning feature budget?

**Proposal.** Delete c03.

**What you lose.** Nothing goes stale — there are no result files. c14_cross_domain_summary answers hybrid-versus-best-single across datasets, and c24_blueprint_transfer answers source-to-target transfer. But neither computes the complexity-descriptor-against-winning-budget correlation or the model/dataset/interaction split of the winning budget.

### `c09` — complexity/c09_formal_analysis.py

**Asks.** How much of the variation in c02's grid scores is explained by the downstream model, by the feature budget, and by their interaction?

**Proposal.** Delete 9a and keep the other three sub-analyses.

**What you lose.** No measurement is lost. Every number 9a reads is already in c02's stored configs, 3425 rows per dataset. Only the eta-squared summary goes.

### `c13` — complexity/c13_forward_compatibility.py

**Asks.** Does adding a new representation to the hybrid's search space improve the hybrid beyond what a same-width random-Gaussian matrix earns from selection alone?

**Proposal.** Delete the nine JSON files in tests/results/complexity/c13_forward_compatibility/ (and the older copies in tests/results/c13_forward_compatibility/), or pass force_rerun=True.

**What you lose.** run_all stops skipping c13 on those nine datasets. Until this is done, no c13 fix takes effect through the normal path.

### `c14` — complexity/c14_cross_domain_summary.py

**Asks.** Across molecular, antibody and TCGA tasks, does the KIRBy hybrid beat the best single base representation?

**Proposal.** Delete c14.

**What you lose.** The cross-domain roll-up disappears. No other complexity experiment answers it: nothing under tests/experiments/complexity/ calls nested_cv_evaluate or run_nested_cv, and c02, the nearest sibling, has the same searched-maximum-versus-unsearched-mean defect (audit 3.3, c02:236-278).

### `c15` — complexity/c15_ml_theory_depth.py

**Asks.** Does a hybrid at budget b have less distance concentration, less hubness and a better-conditioned RBF kernel than each individual base representation of larger width, and than PCA at the same number of components?

**Proposal.** Delete 15c and keep 15a, 15b and 15d. State that the interpolation question is unanswered.

**What you lose.** The double_descent key disappears from the result file. 15a, 15b and 15d are untouched.

### `c18` — complexity/c18_diversity_vs_depth.py

**Asks.** At a fixed budget, does spreading features across all representations beat concentrating them in the top 1, 2 or 3 representations?

**Proposal.** Delete c18. If the arms are meant to be the engine's own allocation methods, the question is an allocation bake-off and c32 already sweeps allocation methods across datasets.

**What you lose.** The diversity-versus-depth question moves to whichever allocation sweep she is already running. Nothing is lost from the result files, since all eight are from a 13-to-27 representation pool at two budgets.

### `c23` — complexity/c23_evaluator_model_matching.py

**Asks.** Does pinning the internal evaluator to the downstream model's own family produce a better-scoring hybrid than pinning it to a different family?

**Proposal.** Delete c23 and let tests/experiments/framework/f13_evaluator_model_matching.py carry the question. f13 asks the same thing, lists only dispatchable evaluators (rf, xgboost, lightgbm, mlp, svm at f13:56), and its number is a fold mean over one fixed recipe (f13:74-119) rather than a maximum over a grid. It is also where an evaluator bake-off belongs by the placement rule.

**What you lose.** The seven c23 result files stop being maintained. The evaluator-versus-model question keeps a home. Nothing in the complexity paper that reads c23's JSON would still find it.

### `c30` — complexity/c30_balanced_global_selection.py

**Asks.** Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat giving each representation an equal fixed quota of the budget?

**Proposal.** Delete c30 and read the same three arms off c31's stored files at /Volumes/seagate/kirby_cache/c31_weighted_global.

**What you lose.** You lose budget 100 and the caps of 50 and 300. You keep the equal-quota, equal-cap-then-global-rank and no-cap-then-global-rank comparison on 10 datasets and 3 downstream models instead of 4 datasets and one. Note c31's stored representation pools are pre-pruning — see the c31 record.

### `caco2_runner` — federated/caco2.py

**Asks.** Can every federated experiment be run against caco2 from one command, with an optional cheap representation list for local smoke runs?

**Proposal.** Delete caco2.py and use `python tests/experiments/run_dataset.py federated caco2`.

**What you lose.** One entry point instead of two. The --reps override is lost, but _smoke_driver.py already provides it for local runs. No SLURM script calls this file.

### `creative_hybrids` — similarity_metrics/creative_hybrids.py

**Asks.** Does stacking descriptors, MACCS and a structurally-selected atompair/torsion block onto the count fingerprint raise the 5-nearest-neighbour activity Spearman above ecfp4_count with Tanimoto?

**Proposal.** Delete the file and do nothing else.

**What you lose.** No stored result is lost, because there is none. The MACCS block and the question of how many atompair/torsion features to keep stop being asked anywhere in the folder.

### `exp_b` — complexity/exp_b_balanced_global.py

**Asks.** Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat the equal per-representation quota?

**Proposal.** Delete exp_b.

**What you lose.** c30 is the later rewrite of the same three arms and adds the engine check. c31 has already run the same three arms on 10 datasets and 3 models with its results on disk. Nothing in exp_b is unique except its direction gate.

### `fed02` — federated/fed02_rep_diversity_silos.py

**Asks.** Does a feature-level hybrid built across five representations (C4) beat the best of those five representations used alone (C1)?

**Proposal.** For C5's test-fold selection, the third choice: delete C5. It asks whether restricting the hybrid to the three best representations beats using all five, which is what the top_3 allocation already is, and the search sweeps that allocation.

**What you lose.** One fewer condition, and no need to solve the selection problem. C4's allocation still reports which representations got budget.

### `fed03` — federated/fed03_synthetic_private.py

**Asks.** Part A: on hERG molecules pooled from three sources, does allocating a feature budget across silos by their own validation scores beat the best single silo, the two FLuID prediction-averaging ensembles, and the privacy-violating pooled model, measured by MCC?

**Proposal.** Delete Part C (fed03:949-1457) and its dispatch branch, and let fed05 answer the antibody federated question.

**What you lose.** fed05_antibody_federated.py:14-30 already runs independent silo baselines, cross-silo importance transfer and federated feature selection over ddg_combined, abagym_HER2 and flab_thermo_tresanco. Part C uses those three plus graphinity_foldx. fed05 is in SELF_LOADING_EXPERIMENTS (federated_helpers.py:231), so it runs on the submit path and Part C does not.

### `handtrace_noise` — noise/_handtrace_noise.py

**Asks.** Does the retention percentage give a number below 100 for a degraded model under both RMSE and R2?

**Proposal.** Delete the file.

**What you lose.** One less file. Nothing breaks: the only references are two audit documents and NOISE_AUDIT_STATUS_2026_06_04.md:103, no code imports it.

### `herg_runner` — federated/herg.py

**Asks.** Can every federated experiment be run against Lhasa's curated hERG classification set from one command?

**Proposal.** Delete herg.py and use `python tests/experiments/run_dataset.py federated herg`.

**What you lose.** One entry point. The --reps override is lost, but _smoke_driver.py already provides it locally. Nothing on the server calls this file.

### `logd_runner` — federated/logd.py

**Asks.** Can every federated experiment be run against logd from one command, with an optional cheap representation list for local smoke runs?

**Proposal.** Delete logd.py and use `python tests/experiments/run_dataset.py federated logd`.

**What you lose.** One entry point instead of two. _smoke_driver.py already covers the local representation override. Nothing on the server calls this file.

### `opig_activity_cliff_qsar` — opig_activity_cliff_qsar.py

**Asks.** Which single molecular representation gives the lowest MAE on activity-cliff-rich ChEMBL targets?

**Proposal.** Delete lines 370-381 and drop kirby_hybrid from the table, leaving the honest_rescore block as the only hybrid number.

**What you lose.** The main table has no hybrid row. Question 3 loses its hybrid answer entirely: honest_rescore reports mae only, never cliff MCC. Nothing stored goes stale.

### `opig_cdr_flexibility` — opig_cdr_flexibility.py

**Asks.** Which sequence-based antibody representation best separates flexible from rigid CDR3 loops (ROC AUC, PR AUC, MCC, balanced accuracy)?

**Proposal.** Drop the combined run at main() lines 482-489. The reason is that the features cannot express the label, not compute: an antibody present in both CDR files becomes two rows with identical heavy and light chain features and possibly opposite flexibility labels.

**What you lose.** The results file loses its 'combined' block. The two single-loop runs are unaffected and are the interpretable ones.

### `opig_cdr_flexibility` — opig_cdr_flexibility.py

**Asks.** Which sequence-based antibody representation best separates flexible from rigid CDR3 loops (ROC AUC, PR AUC, MCC, balanced accuracy)?

**Proposal.** Delete the file and run cdr_flexibility through the dataset roster instead. load_data('cdr_flexibility') exists at _shared.py:3179-3198, so the comment at submit_all_datasets.sh:43 saying it is 'pending a load_data path' is stale.

**What you lose.** You lose the comparison to the paper's 0.82 GNN number, which only this file makes, and you lose the per-loop breakdown. You get the framework's own hybrid-versus-single comparison instead. The roster path groups by 80% CDR3-loop identity, not V-gene family, so it is a different split and not the same experiment.

### `opig_humanness` — opig_humanness.py

**Asks.** Which of the ten antibody representations separates human from non-human therapeutic VH best, per downstream model, under 5-fold identity-grouped CV?

**Proposal.** Delete lines 316-321 so no `kirby_hybrid` row is written at all. The held-out comparison already exists: `build_honest_rescore_input(search)` at line 315 stores it, and lines 456-477 print it per downstream model.

**What you lose.** `results` loses the `kirby_hybrid` entry and the Summary Delta at lines 433-439 goes with it. Question 2 is then answered only by the held-out table. This is what opig_trastuzumab_enrichment.py already did on 2026-05-25 for the identical row; its note is at lines 326-333 of that file.

### `opig_scaffold_similarity` — opig_scaffold_similarity.py

**Asks.** Which mutation-difference representation and which distance formula give pair distances that track a substitution-matrix distance, once the differing-position count is partialled out?

**Proposal.** Delete tests/experiments/opig_scaffold_similarity.py and run scaffold_experiment.py per dataset plus feature_type_study.py instead.

**What you lose.** Loses graphinity_foldx, which is the only multi-scaffold set and exists nowhere else; loses the single results file covering all six datasets; loses the controls block, in particular the recorded count contamination (0.12 on koenig, 0.80 on flab_binding_adams, 0.92 on phillips_h3) and the C3 finding that position_gated is bit-for-bit the Grantham truth. scaffold_experiment covers C1 by marking circular graders and refusing position_gated on a multi-mutation library, but does not measure C3.

### `ranker_experiment` — similarity_metrics/ranker_experiment.py

**Asks.** Does selecting features FOR similarity beat selecting them predictively, when the selected space is judged by kNN prediction from five nearest neighbours and by agreement with shared substructure?

**Proposal.** Delete it and read f22_similarity_feature_sources.py instead. f22 asks the same headline question -- does ranking features by their agreement with shared substructure beat ranking them by their agreement with the activity gap -- and asks it per fold: it restricts the MCS pairs to training endpoints at lines 366-372, ranks on training rows only at line 386, and reports both yardsticks on the held-out fold.

**What you lose.** The activity-versus-structural comparison gets a leakage-free answer that already exists: tests/results/f22_similarity_feature_sources/herg_ki.json, 234 KB, written 26 August, on one of this file's two datasets. f22 also sweeps four budgets, two budget splits and four ways of gluing the kept features, none of which this file has. What is lost is ten of the thirteen selectors: f22 carries activity, structure and a RandomForest importance, and does not carry distance correlation, mutual information, Lasso, RandomForest on pair differences, RFE, RReliefF, the Laplacian score, greedy-forward, or the two rank fusions.

### `s07` — similarity/s07_noise_bridge.py

**Asks.** When a k-nearest-neighbour label-consistency filter flags molecules as noisy, does the hybrid space put fewer genuine activity cliffs among the flagged set than any single representation does?

**Proposal.** Delete the file. The sqrt(importance)-weighted distance question is asked better by n12_importance_weighted_distance.py, which injects noise so it knows the ground truth. The cliff question is covered by n01_activity_cliff_framework.py and n14_activity_cliff_landscape.py.

**What you lose.** The suite loses the question 'does a neighbour label-consistency filter flag fewer real cliffs in the hybrid space'. No other file asks exactly that. The 21 stored files become orphaned, and s09 loses two series.

### `similarity_noise_cleaning` — similarity_metrics/similarity_noise_cleaning.py

**Asks.** Does a similarity space selected without labels, for maximum-common-substructure fidelity, find deliberately corrupted activity values better than count-ECFP4 Tanimoto?

**Proposal.** Delete similarity_noise_cleaning.py and keep similarity_noise_cleaning.json.

**What you lose.** f23_noise_cleaning_metric.py answers the same two questions, scores every space, reports how many beat the baseline instead of a maximum, and takes its representations from f22 rather than four hand-written combinations. What is lost is the in-sample arm. The stored JSON is 41 KB, covers herg_ki, logd and adme_solubility, holds both arms in full, and is untracked — it exists on this laptop only, so it must be moved before the file goes.

### `sm_battery_moleculeace` — similarity_metrics/run_battery_moleculeace.py

**Asks.** Which representation x coefficient makes a similarity space whose neighbourhoods best agree with measured activity, judged on held-out scaffold folds?

**Proposal.** Delete it and do not run the 30-target battery.

**What you lose.** No file in the repo then answers the metric-ranking question across 30 targets. feature_ablation.py, ranker_experiment.py and creative_hybrids_v2.py answer it on herg_ki, logd and adme_solubility only, and they have stored results.

### `sm_build_adme` — similarity_metrics/build_adme_solubility.py

**Asks.** Does a third local dataset, log solubility, exist in the same schema as the herg_ki and logd pilot so the metric grid can be run on it?

**Proposal.** Add adme_solubility to build_local_datasets.SPECS, move the stricter blank filter into that builder, and delete this file. build_local_datasets.py supersedes it.

**What you lose.** One builder for all three local datasets. The stricter filter then applies to herg_ki and logd too, and I measured that it keeps the same 1415 and 5039 rows, so nothing they hold changes.

### `sm_feature_type_study` — similarity_metrics/feature_type_study.py

**Asks.** Which amino-acid property scales earn a place in a Euclidean distance space that reproduces BLOSUM sequence distance once mutation count is partialled out?

**Proposal.** Delete run() and main(). Keep the module: the 14 scales, property_table, score_subset and greedy_forward.

**What you lose.** feature_type_study.json loses its producer. Nothing reads that file; only feature_type_study.py:261 names it. The three hypotheses stay answered by opig_scaffold_similarity.py --mode features, on flab_expression_koenig, flab_binding_koenig, flab_expression_adams, flab_binding_adams, flab_binding_phillips_h3 and graphinity_foldx.

### `structural_geometry_survey` — structural_geometry_survey.py

**Asks.** Do structure-derived geometry features (VH-VL angles, CDR shape, paratope charge and the rest) carry standalone signal on antibody property prediction?

**Proposal.** Remove the hybrid arm: lines 326-357, and 'kirby_hybrid' from standalone_names at line 371.

**What you lose.** The survey answers its first two questions only, and answers them without the discarded search. The third question, whether a hybrid over base plus geometry beats the geometry groups, is then not asked here at all.

### `structural_yardstick` — similarity_metrics/structural_yardstick.py

**Asks.** When the judge is shared substructure (MCS) rather than activity, which representation-and-coefficient pair ranks first?

**Proposal.** Delete the file and rebuild the 47-metric rank table inside the framework, repointing mcs_cache.py:44 at kirby.similarity.mcs.mcs_distance.

**What you lose.** src/kirby/similarity/mcs.py:40 already holds the same mcs_distance definition, and its pair files carry smiles_sha and n_atoms, which these do not, so the row space and the size control both come for free. Nothing in the framework currently produces the 47-metric table, so deleting removes an answer that has no replacement yet.

### `winning_recipe_honest` — similarity_metrics/winning_recipe_honest.py

**Asks.** Once features are chosen on one half of the MCS pairs and structure is scored on the other half, does the stacked space still beat ecfp4_count with Tanimoto on distance-versus-MCS agreement?

**Proposal.** Delete the file, and replace the numbers in the two documents with the ones already on disk.

**What you lose.** similarity_noise_cleaning.py:29-33 does exactly what this file set out to do, and does it at molecule level: whole Bemis-Murcko scaffold groups split 50/50 at line 89, the descriptor drop and the feature ranking fitted on desc[fit_idx] at line 175, the z-score means from the fit rows only at line 122. similarity_noise_cleaning.json holds both arms for all three datasets. On herg_ki the stacked space scores 0.4462 structure held-out against baseline 0.1107, and 0.6471 activity against 0.5802, over 2005 held-out pairs and 706 held-out molecules.

## Reference

### Counts

| | yes | partial | no |
|---|---|---|---|
| complexity (30) | 1 | 20 | 9 |
| noise (14) | 1 | 12 | 1 |
| similarity, s01–s13 (15) | 2 | 11 | 2 |
| similarity, standalone (3) | 0 | 3 | 0 |
| similarity_metrics (33) | 5 | 27 | 1 |
| federated (10) | 3 | 5 | 2 |
| OPIG (11) | 0 | 10 | 1 |

Hybrid: 56 through the engine, 42 need none, 15 hand-roll a concatenation, 3 should use the
engine and do not.

Not reachable from any submit path: `c03 c12 c14 c20 c30 c32 exp_b c43_signal c43_step3 n06 s09`
plus most of similarity_metrics. OPIG has no registry entry.

### Four experiments cannot compute their own headline number

- `c15` sub-analysis 15c. Budgets are `n*[0.8, 0.9, 1.0, 1.1, 1.2]`. The near-p=n window is
  `[0.7, 1.3]`. Every budget falls inside it, so the far bucket is always empty and the
  instability ratio is never written.
- `s08` budget sweep. The fold loop reads the across-all-budgets blueprint, so all six budget
  rows rebuild at one width.
- `s08` `knn_score`. `DOWNSTREAM_MODELS` has no `knn`, so it is NaN at every budget and
  `best_knn_budget` is always None.
- `structural_geometry_survey`. `find_best_hybrid` is called without `return_fitted=True`, so
  the hybrid arm records "search returned no fitted rf result" every run. A 21,000-fit search
  runs and is thrown away.

### Two patterns behind many of the per-file problems

**No `groups` into `create_hybrid`** — 16 files: `c03 c07 c08 c09 c10 c24 n11 n12 n13 s11 fed01
fed02 fed03 fed04 fed05 opig_blueprint_transfer`. At `hybrid.py:4440`, with groups the internal
accept-or-reject holdout is a `GroupKFold` fold. Without groups it is
`np.random.RandomState(42).permutation`, a random 20% carve. That carve decides which features
get in. `build_fold_hybrid` passes groups through. `HANDOFF_CREATE_HYBRID_AUDIT.md` is a prompt
for a fresh chat to sweep all of them, including `framework/`.

**AUROC on hard 0/1 labels** — `_shared.py:1836` scores with `model.predict(X_val)`. Eight
datasets are registered with metric `auc`: `dataset_policy.py` lines 226, 248, 250, 258, 260,
261, 273, 315. `_predict_for_metric` at `find_best_hybrid.py:2427` returns probabilities only
when the metric is AUC, so it is safe everywhere else. Affects `c01 c05 c19 c21 n11` and
`framework/f06`.

### Smoke tests

107 runs here. 76 ran, 5 failed, 26 not reached.

```
KIRBY_SMOKE_N=60 python -u <script> --dataset logd --mode intermediary \
    --reps ecfp4,avalon --models rf,knn --output <scratch>
```

The default ten-model list segfaults on this machine. Pin the model list. `c02` is the exception
— it passes `models=None` into the search so its cache key matches the SLURM precompute, so it
cannot be pinned at all. Only representations already in `~/.cache/kirby/rep_cache` can be used.

The noise suite has its own answer. `_smoke_harness.py` sets `OMP`, `OPENBLAS`, `MKL`, `VECLIB`
and `NUMEXPR` to one thread before importing numpy, and forces the internal evaluator to `rf`.
That kept `n07`, `n13` and `n15` alive where the direct command line segfaulted.

Failed: `c02`, `n01`, `n14` segfault. `_smoke_noisy_search` dies at check 6 of 6 — the searches
above it run with caching off and never create the `result_db` directory that check opens.
`n13`'s suggested command exits 0 without running the experiment.

NaN reached a headline without crashing in `n15` (the lightgbm cell is `nan / nan / +nan`, next
to four real cells), `s04`, `s06` (24 NaN, 3 null, every weighted arm), `s07`, `s09`
(`AD_corr_delta = +nan`, and it feeds the Wilcoxon) and `triplet_similarity`.

Two smoke runs overwrote full-run result files with 60-row ones and had to be restored by hand:
`scaffold_experiment.json` and `scaffold_ddg_foldx.json`. Neither script tags its output with
the row cap.

### Against `AUDIT_HYBRID_USAGE_2026_09_03.md`

All 309 items checked against the code. 282 confirmed. 27 already fixed since it was written.
28 refuted. 2 unreachable. Its 26 confirmed blockers that this audit missed are in the entries,
marked *Also caught by the hybrid-usage audit*. The similarity objective and the federated
metric are both its findings.

Where it is wrong:

- §1.6 says the harness "no longer exists" and no copy was sought. It was never committed. The
  copy a later reader found was a symlink one of my own audit agents wrote at 13:16 today, with
  a compiled cache beside it at 13:17. I deleted that directory. Left there it would make `c20`,
  `c30` and the two `c43` files appear to run here and still fail on the server.
- §1.3 names `opig_cgr_comparison`, `opig_ginkgo_developability`, `c11` and `similarity_helpers`
  as reporting the search's own score. None does. There is no `score_mean` in
  `similarity_helpers.py`. `c11`'s number comes from its own fold loop at line 699.
  `opig_ginkgo` runs a per-fold search on training rows only.
- §2.2 lists `s06:489-498` and `federated_helpers:1533-1539` as missing groups. `s06` passes
  groups on both hybrid builds. There is no `create_hybrid` at `federated_helpers:1533`.
- §2.6 says `MODEL_MAX_BUDGET` and the budget policy drop budgets silently. Both print every
  dropped budget. A model left with none raises.
- Nine of its §3.1 engine items describe code since repaired, mostly in the uncommitted diff. So
  they are true on the server and false here.

What it has that this does not: §1.4 on ranking configurations across unequal fold counts, §1.5
and §2.8 on the result database ignoring the metric, and all of §3.1 on the engine internals.
This document does not cover `src/kirby/`.

### Standard KNN tuning — settled 2026-09-04

Every KNN in the experimental suite tunes over the same grid. It lives in one place,
`_HP_SEARCH_SPACES['knn']` in `src/kirby/find_best_hybrid.py`, and is applied per fold by
`_tune_on_training_fold` on the feature matrix that fold is about to be fitted on.

| axis | values |
|---|---|
| `n_neighbors` | 3, 5, 7, 11, 15 |
| `weights` | `distance` |
| metric | euclidean, manhattan, minkowski p=0.5, cosine |

20 combinations. It is stored as TWO blocks, not one dict, because euclidean is minkowski
p=2 and manhattan is minkowski p=1, and sklearn silently ignores `p` unless
`metric='minkowski'`. Crossing metric with p in a single dict would give three identical
euclidean cells wearing different labels.

**Why `distance` only.** Measured over 100 paired cells, 4 datasets, 3 representations each,
k at 3, 5 and 15. Distance beat uniform in 96 of 100, median +0.0129 Spearman. All four
exceptions were flab_thermo_tresanco at k=3, largest margin 0.0039 against a fold-to-fold
spread of 0.026 on those cells. The gap widens with k: median +0.0059 at k=3, +0.0151 at
k=5, +0.0284 at k=15.

**Why these four metrics.** f18 measured the coefficient axis on herg_ki and logd, and the
`readacross` yardstick there is itself a k=5 nearest-neighbour prediction. Fractional
Minkowski beat euclidean on read-across on both datasets (herg_ki 0.4619 vs 0.4453, logd
0.7102 vs 0.6896). A separate run over 60 comparisons on 4 datasets counted which metric
supplies the best answer: cosine 24, manhattan 22, minkowski p=0.5 12, euclidean 2.
Chebyshev and Mahalanobis are excluded — both collapse in f18 (chebyshev 0.2062 on herg_ki,
mahalanobis 0.0497). Correlation is excluded as a duplicate of cosine, which agree at
Spearman 0.9995 on wide standardised blocks. Minkowski p=0.3 is excluded as a duplicate of
p=0.5, Spearman 0.990.

**Preprocessing gates the metric list.** Only the continuous coefficients are defined on
z-scored features. Soergel, Bray-Curtis, Canberra, Tanimoto and the other set-overlap
formulas need non-negative or binary input and return plausible wrong numbers rather than
errors on standardised data — see `src/kirby/similarity/preprocessing.py` and
`SIMILARITY_COEFFICIENT_RESEARCH_2026_08_26.md` §3. f18 skipped canberra and braycurtis on
continuous input for this reason.

**Cost.** p=0.5 is not a true metric, the triangle inequality fails, so sklearn falls back
to brute force and computes every pair. Measured 7 to 50 seconds per representation for 5
folds, against 0.1 to 2 seconds for the tree-backed metrics. That one cell dominates the
grid's wall time on wide representations.

**The shape-reasoning exception.** `‖x−y‖_p^p`, the same quantity without the outer root, IS
a metric for p <= 1 — 0 triangle violations in 5000 random triples where the rooted form had
30. It gives an identical neighbour set (Spearman 1.000 with the rooted form, verified), so
it is indistinguishable under uniform weights. Under distance weights it scores 0.003 to
0.007 lower on herg_ki, 4 representations, rooted better in 4 of 4. It is NOT in the standard
grid. Use it where something reasons about the shape of a neighbourhood rather than
retrieving one — density ratios, LOF, consistency filters comparing distances across
different neighbourhoods.

**What is not settled.** Whether the best metric is a property of the representation rather
than the dataset. Cosine won almost everywhere on the protein language-model representations
(saprot, esm3, esmc) and was near-last on descriptor and fingerprint blocks. Two runs and an
eyeball; not tested.

**Where KNN is skipped entirely.** `tests/experiments/dataset_policy.py` removes knn from the
whole `nucleotide` domain under every strategy, and from `capped` on adme_hlm, adme_mdr1,
adme_rlm, caco2, logd, bh_doyle_plates, herg, qm9, bindingdb_kd, kiba and davis, and from
`uncapped` on herg, qm9 and logd. A knn search on nt_promoter_tata builds zero configurations
and returns nothing. The recorded reasons are wall time and "weakest contributor in prior
surveys".

### How to read an entry

The line under the heading says whether the file needs you at all. Then the verdict on whether
the code answers its own question, whether the hybrid goes through the engine, the compute
class, and whether `run_dataset.py` dispatches it. *On re-read* is where the second pass
corrected the first. That happened on 69 of 116 files. **Options** are cheapest first. *My read*
is one line and is not a decision.

---

# Complexity — 30 files

## c01 — `complexity/c01_rep_characterization.py`

**DECIDED.** Swap in _predict_for_metric at _shared.py:1836 and record the width actually fitted. Keep the baseline arm.

PARTIAL · hybrid: none needed · compute: fine · dispatched

For each base representation it computes intrinsic dimensionality (PCA 90/95/99 thresholds + Levina-Bickel MLE), mutual information with the labels, effective rank, and a per-model cross-validated score for that representation on its own. No hybrid is built.

**Q.** How intrinsically low-dimensional is each base representation on this dataset?
**Q.** How much information does each base representation carry about the label?
**Q.** How well does each base representation predict the label on its own, per downstream model?

*On re-read:* The two scoring problems are real and I confirmed both in the code today. Two other things in the audit are off.

Why:
- On a dataset scored by area under the ROC curve, the baseline fits the model and then hands hard 0/1 predictions to the AUROC function (_shared.py:1836, `model.predict(X_val)`). Three stored c01 files are on such datasets: humanness_therapeutic, tcga_luad_vs_lusc, tz_her2_binary.
- The engine does this correctly and c01 does not. find_best_hybrid.py:4405 calls _predict_for_metric before scoring a baseline, so c02's per-representation numbers use probabilities and c01's do not.
- Any representation wider than a fold's training set is replaced by a PCA of it before the baseline is fitted (_shared.py:1832, cap_dimensions with max_dim defaulting to n_train). On ginkgo_acsins, n=242 and 5 folds means n_train is about 194, so 26 of the 27 representations in the stored file were compressed.
- c02's baselines are not compressed. The engine keeps the full width and only trims to the top 5000 features by LightGBM importance above that (find_best_hybrid.py:3180, HIGH_DIM_FILTER_THRESHOLD).

*Smoke:* ran clean. exit 0 in ~90s.

<details><summary>Options considered</summary>

1. In _shared.py:1836, score with `_predict_for_metric(model, X_val, metric_fn)` instead of `model.predict(X_val)`, importing it from kirby.find_best_hybrid. I read that function: its signature is `_predict_for_metric(model, X, metric_fn)` at find_best_hybrid.py:2427 and it returns probabilities only when metric_fn is the built-in AUC function, so every regression and every f1_weighted dataset is untouched. *Cost:* c01 baselines are 10 representations x 5 models x 5 folds = 250 model fits per dataset.
2. Delete c01's baseline loop (c01:98-107) and take the per-representation scores from c02's file instead. c02's search already scores every representation on its own, per model, per fold, at native width and with probabilities for AUROC. *Cost:* Saves 250 model fits per dataset, 6,500 across the 26 datasets that have c01 files.
3. Keep both paths and record what happened: store the width actually fitted alongside each baseline, so a reader can see when the number came from a PCA of the representation rather than the representation. *Cost:* Zero model fits to make the code right.

</details>

## c02 — `complexity/c02_complexity_grid.py`

**DECIDED.** Do all four repairs. Only the PCA-control fix costs fits: 800 across four datasets.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Runs find_best_hybrid over the full grid (5 allocation methods x 3 importance methods x 8 budgets x 5 internal evaluators x the engine's 7-model default) under up to four conditions: no augmentation, uncapped augmentation, capped augmentation, and frozen-plus-fine-tuned representations. Then adds a per-model pass at the native width of that model's best single representation, PCA-on-the-concatenation baselines at every budget, and a model x budget score table.

**Q.** Does each downstream model have a different optimal feature budget?
**Q.** Does the hybrid beat the best single base representation, per model?
**Q.** Which augmentation condition (none / uncapped / capped / frozen+fine-tuned) gives the best hybrid for each model?
**Q.** Does importance-based selection beat PCA of the concatenation at the same feature count?

Why:
- The delta printed at c02:423 and implied by the stored best_by_model minus baselines compares a winner picked from 3,720 hybrid configurations against a winner picked from 10 representations, on the same 5 folds. It is biased toward the hybrid.
- The honest paired number already exists. honest_rescore at _shared.py:881 is pure arithmetic on per-fold scores, no fitting and no I/O, and c02 already writes its input block at line 549.
- That block is built from the uncapped search only (c02:549, `build_honest_rescore_input(search)` where `search = search_uncapped` at line 209). No-augmentation, capped, frozen-plus-fine-tuned and the per-model top-representation-width pass contribute nothing to it.
- `conditions` (c02:465) stores only score_mean and score_std per model per condition. So the question 'which augmentation condition wins for this model' is an argmax over four means with no way to apply the held-out-fold tax later.

*Smoke:* FAILED. exit 139 (segfault) at 'Evaluating full-dimension baselines...'.

<details><summary>Options considered</summary>

1. Add `'fold_scores': r.get('fold_scores', [])` to the dict built in _extract_best (c02:454-462), and call build_honest_rescore_input once per condition instead of once on the uncapped search, storing the four blocks under a new key. *Cost:* Zero extra model fits.
2. Print the honest number beside the optimistic one at c02:423, by calling honest_rescore on the block c02 already builds and reporting delta_mean_of_folds. *Cost:* Zero model fits.
3. At c02:368 pass `task_type=data['task_type']` and drop `mode=mode`. *Cost:* PCA baselines are 8 budgets x 5 models x 5 folds = 200 model fits per dataset.

</details>

## c03 — `complexity/c03_cross_dataset_transfer.py`

**YOURS.** Delete, or rewrite to run over all 26 datasets. It is orphaned with no result files. c14 and c24 cover the transfer half but nothing else computes complexity-descriptor against winning budget. My lean: rewrite, it costs no model fits.

PARTIAL · hybrid: engine, per fold · compute: heavy · orphaned

For each of several datasets it computes a Lorena-style complexity profile, pulls the best precomputed distance-tier hybrid search keyed to the SVM downstream model (falling back to RF), then Spearman-correlates each complexity descriptor against that dataset's winning budget. It also fits SVR and ElasticNet per fold on a hybrid rebuilt from the winning blueprint, and decomposes the per-model best-budget matrix into model, dataset and interaction sums of squares.

**Q.** Does a dataset's intrinsic complexity (effective rank, participation ratio, C1/C2/L1/S3) predict the winning feature budget?
**Q.** How much of the variation in winning budget is attributable to the model, to the dataset, and to their interaction?
**Q.** Do two regressors that were never in the search (SVR, ElasticNet) work on the winning hybrid?

*On re-read:* Both audits treat c03 as live. It is not run.

Why:
- c03 has never produced a result file and nothing launches it.
- Every full complexity run pays for the distance-tier search on every dataset for c03's benefit. That is 5 allocation methods x 3 importance methods x 6 budgets (DISTANCE_BUDGETS) x 5 evaluators (DISTANCE_EVALUATORS) x 7 models = 3,150 configurations, x 5 folds x 2 augmentation conditions = 31,500 model fits per dataset.
- create_hybrid at c03:160 gets no groups. hybrid.py:4293 accepts a groups argument and with none it carves a RandomState(42) 20 percent block of the training rows.
- create_hybrid sits inside the `for novel_name, novel_model` loop at c03:141 but none of its arguments depends on the model, so the identical hybrid is built twice per fold.

*Smoke:* not run. exceeded 6 min at 'Generating smited...

<details><summary>Options considered</summary>

1. Delete c03. *Cost:* Frees the distance-tier precompute inside complexity's run_all, 31,500 model fits per dataset, on datasets the noise suite does not also cover.
2. Keep the two arms that need no search. Compute the complexity profile per dataset and correlate it against the winning budgets already stored in c02's best_by_model, which carries budget, score and dim for every model on every dataset. *Cost:* compute_dataset_complexity (_shared.py:2330) does one truncated SVD, one linear fit and one nearest-neighbour build per dataset.
3. Repair c03 where it stands and add it to SINGLE_DATASET_EXPERIMENTS. Replace the create_hybrid call at c03:160-168 with build_fold_hybrid (_shared.py:3559). *Cost:* The distance-tier pool is already precomputed by run_all, so the search itself is cache-served.

</details>

## c04 — `complexity/c04_importance_as_metric.py`

**DECIDED.** Memo the per-fold importance (8x on the dominant cost, no number changes) and add a single-representation comparator. The comparator is the standing baseline rule, not a choice.

PARTIAL · hybrid: hand-rolled · compute: heavy · dispatched

Concatenates every base representation, computes per-fold feature importance on the training rows with each importance method, turns those scores into per-feature weights (sqrt then normalise to sum 1), and compares weighted against unweighted k-nearest-neighbour regression and inverse-distance retrieval. It also cross-tests selecting features by one importance method while weighting by another, and repeats the retrieval comparison on a 25-feature subset.

**Q.** Does weighting the concatenated feature space by importance scores improve k-nearest-neighbour prediction over the unweighted space?
**Q.** Does it improve inverse-distance-weighted retrieval of the label from the ten nearest training molecules?
**Q.** Does the importance method that best selects features differ from the one that best weights them?

Why:
- The same importance scores are computed 120 times when 15 would do. _evaluate_knn_and_sim runs 3 methods x 5 folds = 15.
- The unweighted arm is recomputed once per importance method and gives the same number each time. In ginkgo_hic.json the unweighted_r2 at k=5 is -0.09556394893439427 under deeplift, lightgbm_gain and treeshap alike.
- Every score is r2_score whatever the dataset's metric is. ginkgo_hic is registered as spearman and the file carries no metric name anywhere.
- The feature selection at c04:162 and c04:207 is one ranking by np.argsort over the importance scores concatenated across all representations. That is what the engine's weighted_global allocator computes.

*Smoke:* not run. exceeded 6 min at 'Cross-test matrix (select x weight)...'.

<details><summary>Options considered</summary>

1. Wrap compute_feature_importance in a small memo keyed by (method, fold number) and call that from all three helpers. The three already slice train_reps the same way off the same cv_split, so the key is well defined. *Cost:* 120 importance computations drop to 15, a factor of 8 on the dominant expense.
2. Restructure so the fold loop is outermost: split once, compute the three importance results for that fold, then run the k-nearest-neighbour, cross-test and low-dimensional arms off them. Hoist the unweighted arm out of the per-method loop too. *Cost:* Same 120-to-15 reduction, plus roughly 30 redundant nearest-neighbour builds and k-nearest-neighbour fits removed.
3. Give the weighted space a comparator: run the same weighted-versus-unweighted retrieval on each representation on its own, using the same per-fold importance the memo already holds. *Cost:* No model fits.

</details>

## c05 — `complexity/c05_learning_curves.py`

**DECIDED.** Take the search.baselines swap. It fixes the KeyError crash, the PCA confound and the hard-label AUROC in one edit and removes 4,550 fits per dataset.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

At each of ~5 subsample sizes (drawn as whole scaffold groups, 3 seeds) it runs a full find_best_hybrid search on the subsample and records the winning score per downstream model, against the best individual base representation scored by an independent group-CV. It also re-fits the full-data winning recipe on each subsample as a second arm.

**Q.** Does the hybrid's advantage over the best individual base representation grow, shrink or hold as the number of training molecules rises?
**Q.** Does the winning recipe (budget, importance method, allocation method) change with sample size?
**Q.** Does the recipe chosen on the full data still work when re-fitted on a small subsample, compared with searching afresh at that size?

*On re-read:* The crash is real but it starts far earlier than the audit says. The audit blames svm and datasets over 10,000 rows.

Why:
- Any dataset with more than 1,000 rows dies with a KeyError before c05 writes anything.
- The baseline arm PCA-compresses every representation down to the training row count; at the 250-row point that is 200 components, while the hybrid arm keeps its raw selected features.
- On a classification dataset the baseline arm computes AUROC from hard 0/1 predictions, because evaluate_single_rep_cv calls model.predict; the search scores the same metric on probabilities.
- The hybrid number is the best of about 3,000 searched recipes and the baseline is the best of 10 representations, both scored on the folds that chose them, so the plotted gap is inflated by an unknown amount.
- **Blocking** `complexity/c05_learning_curves.py:270` — KeyError kills the whole experiment on any dataset over 10,000 rows.

*Smoke:* not run. exceeded 6 min inside the first hybrid search (last line 'Search' then the tensorflow banner).

<details><summary>Options considered</summary>

1. Delete the separate baseline loop at c05:253-270 and read the per-representation numbers out of search.baselines instead. The search already computes them, on the same folds, at the same seed, at full width for anything under 5,000 features, and using probabilities where the metric is AUROC (find_best_hybrid.py:4149-4291, _evaluate_single at 3280-3360). *Cost:* Removes about 4,550 duplicated model fits per dataset: 10 representations x 7 models x 5 folds x 13 subsample searches.
2. If you want the separate loop kept, the alternative is one line at c05:254: skip a model that was not resolved at the full row count. *Cost:* Saves the wasted gp baselines: 10 representations x 5 folds x 3 seeds x the number of sample sizes under 1,000.
3. Or keep the loop and widen the reported model set instead: build results['models'] from the union of model_names and every subsample's model list, and pass that union as models= to the search at c05:216. *Cost:* Adds gp to the hybrid search at every sample size at or under 1,000: 5 allocation methods x 3 importance methods x 5 evaluators x the surviving budgets, times 5 folds, per subsample search.

</details>

## c06 — `complexity/c06_within_family_gradient.py`

**DECIDED.** Store the per-fold scores and guard the borrowed importance on the fold count. Both free.

**Still open:** Whether the matched internal evaluator arm is worth making reachable. Today 'match' never resolves for a family-prefixed variant, so that arm is silently absent.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

For each of ~28 capacity variants across 7 model families it runs a separate find_best_hybrid with that variant as the only downstream model, records the best score at every budget, takes the budget where the score peaks, and Spearman-correlates capacity index against that peak budget within each family.

**Q.** Within a model family, does the feature budget at which the hybrid scores best rise with the model's capacity?
**Q.** Which importance method and allocation method wins at each (capacity variant, budget) cell?

Why:
- The internal evaluator is never the variant itself. The downstream model is registered as 'RF:depth=3' (c06:158), and the matched-evaluator lookup at find_best_hybrid.py:4663-4670 is a dict of canonical names, so it returns nothing.
- The linear family runs ungated. get_model_family_variants returns 'Lasso' for regression (_shared.py:2068), _FAMILY_TO_CANONICAL maps 'Ridge' (c06:39), so the lookup returns None and c06:103 skips the gate.
- Run from the command line the reused importance is built on the wrong folds. get_precomputed_search is called without n_folds (c06:126-134) so it builds at 5 folds; the variant searches use config['n_splits'], which the default --mode intermediary sets to 1 (_shared.py:5316).
- The comment at c06:120-122 says a missing precompute falls back to building importance from scratch. It does not.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Drop 'match' from the evaluator list passed at c06:184 and say in the payload that four evaluators were searched. *Cost:* Zero fits.
2. Or register the downstream model under its canonical family name ('rf', 'knn', 'mlp', 'svm', 'gp', 'xgboost') and keep the variants apart with condition_suffix, which c06:180 already sets per variant. *Cost:* Adds a fifth evaluator to 3 of the 7 families: 5 x 3 x 8 x 1 = 120 extra configurations per variant, times 4 variants, times 5 folds = 2,400 extra fits per dataset for those three families.
3. The evaluator that IS the capacity variant cannot be searched today. find_best_hybrid sorts the evaluator list into the cache key as strings (find_best_hybrid.py:545), so an estimator object cannot be passed as an evaluator name. *Cost:* Cost not established.

</details>

## c07 — `complexity/c07_data_complexity_partitions.py`

**DECIDED.** Add groups, model, metric_fn and lower_is_better to build_fold_hybrid at c07:338.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Splits each dataset in half three ways (frequent vs rare scaffolds, small vs large molecules by heavy-atom count, low vs high within-scaffold label variance), runs a full find_best_hybrid on each half, and records whether the winning recipe differs between the halves. It also re-fits the full-data winning recipe on each half as a second arm, and computes descriptive complexity numbers (mutual information, effective rank, intrinsic dimension) per half.

**Q.** Does the winning hybrid recipe — model, budget, importance method, allocation method — differ between an easy and a hard subpopulation of the same dataset?
**Q.** How far does the budget move between the two halves?
**Q.** Does the full-data winning recipe still score well when re-fitted on a subpopulation, or does the subpopulation need its own search?

Why:
- The full-data-recipe arm judges its features on a random holdout. build_fold_hybrid at c07:338-342 is called with no groups, no model and no metric_fn.
- The target_variance partition is not a label-variance contrast. scaffold_var.get(g, 0) at c07:168 gives every scaffold with one member a variance of 0, and 0 is always at or below the median, so every single-member scaffold lands in 'easy'.
- The two arms of the same row use different fold counts. find_best_hybrid at c07:230 is called with no n_folds so it uses 5; the full-data-recipe arm at c07:302-304 uses config['n_splits'], which the command-line default sets to 1.
- The two arms also cover different model sets. The full-data recipes are drawn for ['rf','svm','lightgbm','mlp'] at c07:59, while the search at c07:230 passes no models and gets the engine's default seven: rf, xgboost, knn, lightgbm, mlp, svm, gp.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Pass the four arguments build_fold_hybrid exists to forward, at c07:338: groups=sub_groups[tr_i], model=factory(), metric_fn=metric_fn_part, lower_is_better=lower_is_better. All four are already in scope — factory at c07:315, metric_fn_part at c07:349, lower_is_better at c07:45. *Cost:* The outer fits are unchanged: at most 4 models x 5 folds x 6 halves = 120 per dataset.
2. For target_variance, drop rows whose scaffold has one member from both halves before the median split at c07:167-169. *Cost:* Zero fits added.
3. Or keep the split as it is and rename it. 'Every single-member scaffold plus the low-variance ones' is a real subpopulation, just not the one the name promises. *Cost:* Zero.

</details>

## c08 — `complexity/c08_calibration.py`

**DECIDED.** Remove the greedy dispatch at c08:76 and c08:179 — greedy is closed, this is not a choice. Add the four arguments at c08:220.

NO · hybrid: hand-rolled · compute: fine · dispatched

Builds four hybrids per dataset — the recipe the pool search picked, the same recipe at a tiny budget, the same recipe at the full concatenated width, and a low-capacity Ridge or LogisticRegression at the full width — then runs split-conformal prediction per fold and reports coverage, interval width or prediction-set size, and calibration error for each.

**Q.** Does the hybrid recipe that scores best also give the best-calibrated uncertainty, compared with the same recipe run at too small or too large a feature budget?
**Q.** Does the correlation between the model's own uncertainty and its actual error change with the feature budget?

*On re-read:* The code findings all hold — I re-read c08:72-73, c08:179-180, c08:220-228, c08:244-245 and c08:251/360, and hybrid.py:4482-4501 confirms that internal_evaluator='match' with model=None collapses to 'lightgbm'. What is wrong is the claim that a quarter of the reported conditions was built by greedy.

Why:
- The condition labelled 'optimal' is not the recipe the search chose. create_hybrid at c08:220-228 gets the internal evaluator from the recipe but no model and no metric_fn, so 'match' resolves to lightgbm and features are admitted on r2_score or accuracy rather than the dataset's metric.
- Every condition's feature selection judges keep-or-drop on a random 20 percent carve of the training rows, because create_hybrid gets no groups. The outer split is scaffold-aware; the selection inside it is not.
- The conformal calibration rows are drawn by np.random.RandomState(42+fold_i).permutation (c08:251 and c08:360), so they share scaffolds with the rows the model was fitted on. The residual quantile is then too small and coverage is biased below 0.9.
- 'mismatched' hardcodes the cut greedy allocator at c08:179. It is the default set of conditions, so any re-run runs it.
- **Blocking** `complexity/c08_calibration.py:179` — The 'mismatched' condition hardcodes allocation_method='greedy' and importance_method='treeshap'.
- **Blocking** `complexity/c08_calibration.py:76` — The module-level fallback also sets optimal_alloc='greedy' and optimal_imp='treeshap'.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. At c08:179 build 'mismatched' with optimal_alloc and optimal_imp instead of 'greedy' and 'treeshap'. *Cost:* Zero fits added; one create_hybrid rebuild per fold becomes reusable if you cache it.
2. For the module-level fallback at c08:72-73, either raise when neither the c02 file nor the pool gives a recipe, so a run without a recipe cannot produce a result. *Cost:* Zero fits.
3. Or keep a fallback but make it a searched default — performance_weighted and lightgbm_gain (find_best_hybrid.py:2704, 2713) — and write a flag into the result saying the fallback fired. *Cost:* Zero fits.

</details>

## c09 — `complexity/c09_formal_analysis.py`

**DECIDED.** Do the five local repairs and recompute the ANOVA over configurations rather than fold scores.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Four separate analyses in one file: an ANOVA of c02's grid scores by model and budget; mutual information between the selected features and the property plus a variance-retention measure against the full concatenated features, per budget; effective rank of the hybrid against PCA and a random subset at the same width; and a train-minus-held-out gap per model per budget. Only the last one fits downstream models; the first reads c02's JSON.

**Q.** How much of the variation in c02's grid scores is explained by the downstream model, by the feature budget, and by their interaction?
**Q.** As the budget rises, how much information about the property do the selected features carry, how much of the full concatenated feature set do they retain, and do they span more independent directions than PCA or a random subset of the same width?
**Q.** For each model and budget, is the train-to-held-out gap widening (overfitting) or is the train score itself poor (underfitting)?

*On re-read:* Two claims are not what the code does. (1) The findings say 9d only fits models "when c06 results are absent".

Why:
- All nine stored files were written between 27 May and 2 June. c09 has changed in five commits since, and the mutual-information estimator was replaced on 3 June in commit 017a8c0.
- mi_xy is exactly 0 in 35 of the 36 stored budget entries, because the estimator in force at the time returned 0 for everything.
- pca_rank is NaN in 27 of the 36 stored budget entries. Line 462 caps the component count by the full row count, and line 522 fits the PCA on the training rows only.
- The ANOVA treats every fold score of every configuration as an independent observation. On abcov_kd that is 3425 configurations and 17124 observations over 40 (model, budget) cells holding 74 to 101 configurations each.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Fix the five local faults and re-run: cap the PCA component count by the training rows; drop the greedy/ridge fallback and skip the cell with a printed reason instead, as c10 does at its lines 341-350; pass data=data; pass groups; route --reps through parse_rep_args. *Cost:* 9b plus 9c is about 80 hybrid builds per dataset.
2. Recompute the ANOVA with one observation per configuration instead of one per fold score, and with allocation and importance as their own factors. *Cost:* Pure arithmetic on the 26 stored c02 files, which carry model, budget, allocation, importance and per-fold scores.
3. Delete 9a and keep the other three sub-analyses. *Cost:* No re-run of anything.

</details>

## c10 — `complexity/c10_noise_robustness.py`

**DECIDED.** Fix the local faults and switch to build_fold_hybrid.

**Still open:** Whether the hybrid recipe is re-searched under noise. Symmetric costs 56 noise-aware searches per dataset.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Injects label noise into the training labels only, at a grid of levels under eleven noise conditions, and scores a random forest on clean held-out labels. Does it once per individual representation and once per budget for a hybrid built from the cached blueprint, then summarises each degradation curve with auc_norm, and measures how much the selected feature set changes as noise rises.

**Q.** Does a hybrid lose accuracy more slowly than an individual representation as label noise rises?
**Q.** Does a larger feature budget buy more noise robustness?
**Q.** Do the same features keep being selected as the noise level rises, and how similar are the selections across folds at zero noise?

*On re-read:* The demoted-representation failure at line 387 is narrower than stated. create_hybrid only moves zero-budget representations into the augmentation pool when the augmentation dictionary is non-empty and the strategy is not 'none' (hybrid.py:4941-4943).

Why:
- Nothing has ever been written. tests/results/complexity/c10_noise_robustness/ is empty for every dataset.
- The Nogueira stability index is wrong by a factor of the feature count. The shortfall from 1 comes out at 1/d of the real one, so phi sits near 1.0 for any realistic count (lines 84-90).
- d is taken as the number of features ever selected across folds, not the number available. total_features is passed in at line 535 and used only for the ==0 test at line 57.
- The recipe is chosen by a search on the clean labels (lines 155-162). label_noise= is never passed, though get_best_search_for_downstream resolves the noise-searched lineage for exactly this case (_shared.py:4334-4338).

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Fix the local faults before it ever runs: the Nogueira formula and its d, the guard from three levels to four, per-fold scores into level_scores alongside the mean, and --reps through parse_rep_args. *Cost:* No stored result is invalidated, because none exists.
2. Replace the two bare create_hybrid calls (lines 378 and 501) with build_fold_hybrid (_shared.py:3559), keeping budget_override=budget so the sweep axis stays pinned as line 380 does now. *Cost:* About fifteen lines in each of the two places.
3. Give the hybrid arm a recipe searched on noisy labels: pass label_noise= the spec for that condition and level. *Cost:* 56 non-zero (condition, level) points per dataset: ten conditions at five non-zero levels, plus censoring at six.

</details>

## c11 — `complexity/c11_ensemble_vs_hybrid.py`

**DECIDED.** Five one-line repairs, the _emit fix, and take the PCA rotation off the baselines rather than adding it to the hybrid.

**Also decided 2026-09-04, from the split audit:** search the hybrid's recipe inside each outer training fold. The ensemble arm already picked its members, weights and aggregator inside the fold; the hybrid arm took its recipe from a search over every row, so the two arms were not comparable and the gap ran the hybrid's way. `--recipe-search per_fold` is now the default, and sweeps the same augmentation conditions the whole-dataset pool compares. It costs one pool of searches per outer fold per dataset: five without augmentation representations, fifteen with. `--c11-recipe-search whole_data` on `run_dataset.py` keeps the old free path for a dataset whose number is not quoted, and renames every head-to-head field it writes so none reads as held out.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Trains one model per representation on every outer fold and on a three-fold inner CV inside each outer training split, then builds ensemble processes that pick their members, weights and aggregator from the inner scores alone and aggregate the stored outer predictions. Compares those against the best single representation, a PCA of the concatenated representations, and a per-fold-refitted hybrid, and reports an inference cost row for every arm.

**Q.** Does a single model on the fused feature vector match an ensemble of one model per representation, when the ensemble picks its members and aggregator per fold from that fold's training split alone?
**Q.** What does each arm cost at inference - how many featurisers, how many deployed models, how many stored bytes per sample - and how many representations does the blueprint actually draw a column from?
**Q.** Does the hybrid beat the best single representation under leave-one-fold-out reselection?

Why:
- The nine stored files are from 27 May to 2 June and predate the July rewrite. They carry none of head_to_head, honest_rescore, break_even or general_reps, which run() returns now (lines 1111-1138).
- build_fold_hybrid is called with no budget_override at line 650, so it builds at blueprint.budget. get_best_for_downstream falls back to another budget whenever the requested one is missing or scored NaN (_shared.py:4785-4818).
- Every single-representation baseline and every ensemble member is PCA-rotated. _prep calls cap_dimensions with no max_dim (line 168), which defaults to the training row count (_shared.py:1799-1800).
- head_to_head['hybrid_minus_single'] at line 982 is the difference of two leave-one-fold-out means. honest_rescore computes delta_mean_of_folds for exactly this and documents that the two diverge whenever either arm drops a fold (_shared.py:912-917).

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Five one-line repairs: budget_override=budget at line 650; metric_name=metric_name on both honest_rescore calls; read delta_mean_of_folds instead of _delta(h, b) at line 982; use config['budgets'] at line 383; route --reps through parse_rep_args. *Cost:* Only budget_override changes a fit.
2. In _emit, take the fold's rep set from auto_choice instead of substituting 'all'. *Cost:* About four lines.
3. Take the PCA rotation off the baselines: in _prep, scale and return, and let each representation be scored at its own width, as the engine's baseline path does. *Cost:* 5 models x 10 representations x (5 outer + 15 inner) = 1000 fits per dataset, all re-run, on wider inputs.

</details>

## c12 — `complexity/c12_modality_ablation.py`

**DECIDED.** Crash fixes, dead-name cleanup, print the honest_rescore number in the summary.

**Still open:** Whether the modality-combination arm stays a raw concatenation or becomes a real hybrid.

PARTIAL · hybrid: hand-rolled · compute: fine · live

On a TCGA dataset it scores every loaded representation on its own with 5 classification models over TSS-grouped folds, picks the best representation inside each omics modality, then np.hstack's the best representation of each modality pair and of all modalities at once and scores those the same way. It also persists a leave-one-fold-out re-score block.

**Q.** For a TCGA task, which omics modality gives the best prediction on its own?
**Q.** Does column-concatenating the best representation of two modalities beat the better of the two modalities alone?
**Q.** Does concatenating one best representation per modality beat the best single modality?

*On re-read:* Every substantive finding holds. Two derived counts are off.

Why:
- The multi-modality arm is a plain np.hstack of two full-width matrices (c12:334), so what is measured is whether raw concatenation helps, not whether KIRBy's budgeted selection picks a modality.
- The printed and stored delta_vs_best_single is a maximum over models on the combination side and a maximum over models and over representations on the single side (c12:497, 548, 567-570), so both arms are inflated.
- The unbiased number already exists in the file — honest_rescore runs at c12:747 and prints at c12:753-759 — but the summary table at c12:650-691 does not use it.
- Nineteen expression names in FACTORY_EXPRESSION_REPS (c12:137-158) and promoter_meth (c12:89) can never be loaded, because find_best_hybrid_tcga.py:354 drops every name in FACTORY_REPS before computing anything.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Keep the concatenation and say so. Rename delta_vs_best_single and the HELPS/HURTS flags in the printout and the JSON so they read as a concatenation result, and drop the claim that this is about the hybrid. *Cost:* No model fits change.
2. Score each modality pair through the real engine instead of np.hstack: one find_best_hybrid per pair, then build_fold_hybrid per fold with groups, the model instance and the metric. *Cost:* On tcga_luad_vs_lusc: 16 searches (15 pairs plus all-combined), each 5 allocations x 3 importance methods x 5 internal evaluators x 8 budgets = 600 configurations x 5 inner folds = 3,000 hybrid builds.
3. Print the honest_rescore numbers in the summary table instead of the double maximum, and print delta_mean_of_folds rather than delta. *Cost:* No model fits change — honest_rescore is arithmetic on scores already computed.

</details>

## c13 — `complexity/c13_forward_compatibility.py`

**DECIDED.** Tie the blueprint to the model that scores it. Delete the nine stale result files so run_all stops skipping it.

**Still open:** Whether molecular datasets get an explicit --new-reps list. Without one c13 returns 'no_new_reps' on all eight.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Splits the loaded representations into 'new' and 'legacy' by a hand-maintained name list, then for each new representation r runs five arms: r with one model, r with the best model, a hybrid over the legacy representations only, a hybrid over legacy plus r, and a hybrid over legacy plus a random-Gaussian matrix of r's width. It reports the debiased gain D minus D_random and the gain over r's own best single model.

**Q.** Does adding a new representation to the hybrid's search space improve the hybrid beyond what a same-width random-Gaussian matrix earns from selection alone?
**Q.** Does the hybrid built with the new representation beat that representation's own best single-model score?
**Q.** Does the new representation carry signal above a random-Gaussian matrix and above a per-column-shuffled copy of itself?

*On re-read:* Three corrections. (1) The audit's section 1.5 says c13's with-representation and without-representation arms share result-database rows.

Why:
- On every molecular dataset c13 produces nothing: logd's ten general representations (complexity_helpers.py:181-183) are all in _LEGACY_MOLECULAR (c13:84-92), so _detect_new_reps returns an empty list and run() returns {'error':'no_new_reps'} at c13:396. Same for caco2, herg_ki, qm9 and the adme sets.
- On flab_thermo_tresanco the split runs the other way: 8 of the 10 general representations are new and only onehot and esm2_early are legacy, so condition C, described as the current KIRBy ceiling, is a hybrid over two representations.
- The recipe and the model that consumes it are chosen independently: the blueprint is whichever of rf, svm, lightgbm, mlp, knn scored best in the search (c13:209-234), while the per-fold rebuild and scoring use best_b_model, the winner on the new representation alone (c13:275-277, 487, 529, 620).
- knn is searched as a blueprint candidate (c13:209) but is not a downstream model anywhere else in the suite, and xgboost, which is one, is never searched here.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. For molecular datasets, pass --new-reps explicitly per dataset instead of relying on auto-detection. *Cost:* No code change.
2. Tie the blueprint to the model that scores it: replace the five-candidate loop at c13:209 with a single get_best_search_for_downstream call for best_b_model. *Cost:* Fewer pool calls, no change in searches — the underlying searches are cached on disk, so the five candidate calls already reload rather than recompute after the first.
3. Route conditions D and D_random through run_nested_cv (_shared.py:535) so the blueprint is searched on outer-train only and scored on an outer fold it never saw, and read delta_mean_of_folds off the result. *Cost:* run_nested_cv runs one full search per outer fold, so each condition's search cost is multiplied by the outer fold count: 5 x 3,000 = 15,000 hybrid builds per condition per augmentation condition.

</details>

## c14 — `complexity/c14_cross_domain_summary.py`

**YOURS.** Delete, or route the headline through run_nested_cv at 15,000 hybrid builds per task. It is orphaned, has never written a file, and nothing else in complexity/ calls nested_cv_evaluate. My lean: delete.

NO · hybrid: engine, per fold · compute: heavy · orphaned

For each named task it scores every base representation with each of the five downstream models over group-aware folds and keeps the best, then asks the pool helper for the best precomputed hybrid search and takes that search's own winning score. It reports the difference, a win rate, a per-domain breakdown and a Wilcoxon signed-rank test across tasks.

**Q.** Across molecular, antibody and TCGA tasks, does the KIRBy hybrid beat the best single base representation?
**Q.** What is the win rate and the mean improvement, per domain?
**Q.** Is the per-task improvement consistent enough to reject zero under a signed-rank test?

Why:
- The two arms are not comparable. hybrid_score at c14:279 is the search's own maximum over its grid — 5 allocations x 3 importance methods x 5 internal evaluators x 8 budgets — scored on the folds that selected it.
- Every headline is that difference: delta, hybrid_wins, win_rate, mean_delta, pct_improvement and the Wilcoxon (c14:287-293, 421-434, 483).
- In intermediary mode config['n_splits'] is 1, so the single-representation arm is one GroupShuffleSplit while the hybrid arm's score comes from the search's own 5-fold internal cross-validation; best_single_fold_scores and hybrid_fold_scores are then stored side by side as if paired (c14:333, 349).
- The Wilcoxon at c14:483 pools deltas measured in RMSE, Spearman, MCC and F1 across tasks, and the signed-rank statistic ranks them by absolute size, so the domain whose metric swings most numerically dominates.
- **Blocking** `complexity/c14_cross_domain_summary.py:279` — hybrid_score is the search's own winner-of-the-grid score on the rows that selected it; best_single_score is an unsearched CV mean.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Replace the pool lookup at c14:247-279 with run_nested_cv (_shared.py:535) per task and read delta_mean_of_folds and best_single_fold_scores off the result. Pass n_outer_folds=5 explicitly, because run_nested_cv defaults it to config['n_splits'], which is 1 in intermediary mode. *Cost:* run_nested_cv runs one full search per outer fold: 5 x 600 configurations x 5 inner folds = 15,000 hybrid builds per task per augmentation condition, against 3 cache reads now.
2. Leave the arms as they are, rename delta and its dependants to say they compare a searched maximum against an unsearched mean, and remove the Wilcoxon. *Cost:* No model fits change.
3. Delete c14. *Cost:* Nothing is lost from disk — no c14 result has ever been written.

</details>

## c15 — `complexity/c15_ml_theory_depth.py`

**DECIDED.** Take the free dedupe: 160 identical rebuilds per dataset inside a loop where the hybrid does not depend on what the loop varies.

**Still open:** 15c: widen the budget band so the far bucket is reachable, or drop the sub-analysis. As written the instability ratio can never be computed.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Four sub-analyses on one dataset: pairwise-distance concentration and hubness per representation and per hybrid budget (15a); RBF kernel condition number, effective rank and kernel-target alignment plus a gamma sweep (15b); a budget sweep through p = n comparing hybrid, PCA and a random feature subset under RandomForest and Ridge (15c); and a random-label memorisation score (Rademacher proxy) for hybrid, PCA and the full concatenation (15d). Regression-only; it returns {'skipped': True} on classification.

**Q.** Does a hybrid at budget b have less distance concentration, less hubness and a better-conditioned RBF kernel than each individual base representation of larger width, and than PCA at the same number of components?
**Q.** As the number of features passes the number of training rows, does the hybrid show the test-error spike that PCA and a random feature subset show?
**Q.** Does compressing to a budget reduce how much of a random label vector a Ridge model can memorise, relative to PCA at the same width and to the full concatenation?

*On re-read:* Two things.

FIRST. The audit says the hybrid keeps the full budget at 1.1n and 1.2n while PCA is held at n-1.

Why:
- 15c cannot compute the instability ratio it exists for: line 482 puts every budget between 0.8n and 1.2n, line 597 defines the near-p=n window as 0.7 to 1.3, so the far bucket is always empty and far_ok at line 604 is always False.
- All nine stored files carry instability_ratio_pca and instability_ratio_hybrid anyway, written between 15 May and 2 June, before that guard existed; the values include 13,744,391 on abagym_VEGF and 604,514 on abagym_HER2, which is a near-zero variance divided by the 1e-10 floor at line 614.
- 15d's reference arm is the concatenation of all ten representations, PCA-rotated to min(total_dim, n-1) and fed to Ridge (lines 850 and 855-857); concatenation baselines are not allowed here.
- 15d asks the pool for a blueprint chosen for ridge, the pool has no ridge, and lines 975-982 silently take rf's instead; the result file records nothing about which model the blueprint was chosen for.
- **Blocking** `complexity/c15_ml_theory_depth.py:482` — The budget list is n*[0.8, 0.9, 1.0, 1.1, 1.2] and the near-p=n window at line 597 is [0.7, 1.3], so every budget falls inside the window.
- **Blocking** `complexity/c15_ml_theory_depth.py:540` — PCA and the random subset are capped at n_samples - 1 features while the hybrid at line 543 uses the full budget.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Take the repeated builds out of 15a, 15b and 15d. Build one hybrid per budget in run() and hand the dict to all three; in 15d hoist build_fold_hybrid above the trial loop, since the hybrid depends only on the real training labels. *Cost:* Per dataset with 5 folds and the 8 default budgets: 15a 8 builds, 15b 9, 15d 240, total 307 including 15c's 50.
2. Make 15c's sweep span both sides of p=n: pass allow_budget_exceeds_n=True to the per-budget search (find_best_hybrid.py:5978), pass budget_override=budget to build_fold_hybrid at line 715, lift the n-1 cap on the random-subset arm at line 540, and add budgets outside the 0.7-1.3 band so the far bucket has points. Record that PCA stays capped, because PCA cannot return more components than rows. *Cost:* The band goes from 5 budgets to whatever she picks.
3. Delete 15c and keep 15a, 15b and 15d. State that the interpolation question is unanswered. *Cost:* Removes 5 find_best_hybrid searches and 50 build_fold_hybrid calls per dataset.

</details>

## c16 — `complexity/c16_feature_selection_as_regularization.py`

**DECIDED.** Fix the LogReg default index and the capped_full NaN column.

**Still open:** Whether all three model families share one hybrid build. It changes what c16 measures.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Builds a score grid of regularization strength against hybrid budget for three model families (Ridge or LogisticRegression, RandomForest max_depth, SVM C), rebuilding the hybrid per fold from a precomputed blueprint at each budget. Reports the spread of scores across regularization settings at each budget, compares the best budget at default regularization against the best regularization at the largest budget, and matches the train-minus-validation gap of a small budget to the gap of a regularization setting at the largest budget.

**Q.** Does the spread of validation scores across regularization settings shrink as the hybrid budget shrinks, for each of Ridge/LogReg, RandomForest and SVM?
**Q.** Does the best budget at default regularization beat the best regularization setting at the largest budget?
**Q.** Which regularization strength at the largest budget produces the same train-minus-validation gap as a given small budget at default regularization?

*On re-read:* The audit says the largest-budget column blanks when total_dim is at or below min(2n, 2048). In the nine stored files total_dim runs from 11,704 to 82,632 and capped_full runs from 274 to 2048, so that condition holds in none of them.

Why:
- Seven of the nine stored files have an all-NaN largest-budget column, so hybrid_wins is None and the L2-equivalence block is empty on seven of nine datasets.
- capped_full is min(total_dim, 2n, 2048) at line 199, and 2n is above n_samples by construction; the engine drops such a budget and searches n_samples, so the column is labelled with a budget that was never searched.
- Line 147 skips every fold when the budget reaches total_dim, and on that path capped_full_error stays None while capped_full_blueprint_built can still read true, so the file gives no reason for the blank.
- Every model family is scored on a feature set chosen for random forest (lines 244-250 and 266-269), and the claim the experiment makes is about Ridge and SVM.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Name the default regularization per family instead of taking the middle of the list. _get_regularization_families returns the variants; return the default value beside them, or key it in a small dict, and read it at lines 370, 415 and 445. *Cost:* One function.
2. Set capped_full to a budget the engine will actually search: min(total_dim, n_samples). Alternatively pass allow_budget_exceeds_n=True (find_best_hybrid.py:5978) to the fast search at line 270 and keep 2n. *Cost:* Neither changes the number of fits: 14 regularization settings x the number of budgets x 5 folds either way.
3. Build the hybrid once per budget and reuse it across all 14 regularization settings, instead of passing model=model_factory() into build_fold_hybrid at line 154. *Cost:* 630 build_fold_hybrid calls per dataset (14 x 9 budgets x 5 folds) drop to 45.

</details>

## c17 — `complexity/c17_information_theoretic_decomposition.py`

**DECIDED.** Take the tie-break fix and the budget check.

**Still open:** The per-fold selection rewrite. It is the only thing that makes the five arms comparable, and it is expensive.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

At each budget it builds five feature sets of that size from the concatenated pool (hybrid via the engine, PCA components, greedy mRMR, RFE with a RandomForest, and a random subset) and reports for each the estimated mutual information with the target, that information divided by the feature count, the effective rank, and a cross-validated RandomForest score. Regression-only.

**Q.** At a fixed budget, which of hybrid, PCA, mRMR, RFE and a random subset produces the feature block with the most mutual information with the target per feature?
**Q.** Which of the five gives the best cross-validated RandomForest score at each budget?

*On re-read:* The audit blames the width mismatch on augmentation features being appended above the budget. That is not what the stored files show.

Why:
- best_strategy names the hybrid the winner when every density is 0.0, purely because 'hybrid' is first in the list at line 295; three of eight stored files are in that state.
- The mutual information estimate is 0.0 for every strategy at every budget in five of the eight stored files, so the quantity the experiment exists to compare is dead on those datasets.
- The hybrid arm's feature count does not follow the budget while the other four arms' do, so the per-budget rows compare blocks of different sizes on three of eight datasets.
- The hybrid, mRMR and RFE arms all select using every row's label and are then cross-validated on those same rows; PCA and the random subset use no labels at all, and that is the comparison the file reports.
- **Blocking** `complexity/c17_information_theoretic_decomposition.py:323` — The hybrid, mRMR (line 289) and RFE (line 347) all select features using every row's label, and the downstream score is then cross-validated on those pre-selected features.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Write best_strategy only when the densities actually differ. Return None, plus a reason, when every density is 0.0 or all are equal. *Cost:* Zero model fits, zero mutual-information estimates.
2. After get_fitted_pool_hybrid at line 323, check that the returned blueprint's budget equals the requested budget, and record an error for that cell when it does not. *Cost:* One line.
3. Build the hybrid at the requested budget rather than accepting the pool's nearest. That needs a direct create_hybrid call with budget=budget, because get_fitted_pool_hybrid has no budget_override (_shared.py:4571-4611). *Cost:* One create_hybrid per budget per dataset, so 4 more builds on these files.

</details>

## c18 — `complexity/c18_diversity_vs_depth.py`

**YOURS.** Whether the diversity-versus-depth arms are rebuilt through create_hybrid, and where that lives. It is an allocation bake-off, so by your placement rule it belongs in framework/ as a new f-number, not as an edit here. That call is yours.

PARTIAL · hybrid: hand-rolled · compute: heavy · dispatched

At each budget and within each fold it ranks the representations by summed LightGBM gain on the training rows, then builds five feature sets of that budget and scores each with a RandomForest on the held-out fold: the top-1, top-2 and top-3 representations pooled and truncated to the budget, an equal share taken from every representation, and the engine's own hybrid rebuilt from the precomputed blueprint.

**Q.** At a fixed budget, does spreading features across all representations beat concentrating them in the top 1, 2 or 3 representations?
**Q.** Does the engine's own allocation beat both the concentrated and the equal-share alternatives at the same budget?

Why:
- natural is the only arm the engine builds; the other four are hand-rolled and do not reproduce what the engine does under the same names.
- _select_top_features at lines 49-65 pools the top-k representations and takes the global top-budget features by importance, while the engine's top_k splits the budget across the k representations; depth_2 collapses to depth_1 whenever one representation's gain scores dominate, and the docstring at line 12 says the opposite of what the code does.
- The natural arm can be at a different width from the four it is compared with, and lines 264-275 store only the requested budget, so the file cannot show it.
- The depth and uniform arms always rank with lightgbm_gain (lines 44, 57, 79) while natural uses the blueprint's own importance method, so any natural-versus-depth gap mixes allocation with importance method.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Run all five arms through create_hybrid. 'top_2', 'top_3' and 'fixed' are already registered in hybrid.py:374-400, and create_hybrid resolves named allocations at hybrid.py:4616; depth_1 is performance_weighted with weighting_scheme='top_k' and weighting_params={'k': 1}, which passes through the same kwargs path. *Cost:* 4 arms x 5 folds x 8 budgets = 160 create_hybrid calls per dataset, on top of the natural arm's 40, replacing 1040 LightGBM fits.
2. Keep the arms hand-rolled and fix the three things that change a number: split the budget across the top-k representations instead of pooling them, spend the whole budget in uniform_all, and record the realised feature count beside the requested budget. *Cost:* No change in the number of fits.
3. Delete c18. If the arms are meant to be the engine's own allocation methods, the question is an allocation bake-off and c32 already sweeps allocation methods across datasets. *Cost:* Nothing to re-run.

</details>

## c19 — `complexity/c19_importance_weighted_training.py`

**DECIDED.** Move the build per fold, fix the recording, take the two one-liners. The eight stored files re-run either way.

PARTIAL · hybrid: engine, per fold · compute: fine · dispatched

Builds ONE hybrid feature matrix through the pool helper, then scores five importance-weighting treatments of that matrix (unweighted, features scaled by sqrt-importance, sample weights from neighbour label agreement, both, and drop features below median importance) with rf/lightgbm/xgboost under grouped CV. Reports each treatment's score, its difference from unweighted, and which treatment won the most model cells.

**Q.** On the hybrid feature matrix, does rescaling features by sqrt of their importance change the downstream score against leaving them alone?
**Q.** Does weighting each training row by how well its label agrees with its ten nearest neighbours change the downstream score?
**Q.** Does deleting every hybrid feature whose importance is below the median change the downstream score?

*On re-read:* One item is wrong. The audit says `sample_X = X_full` at line 176 is assigned and never read.

Why:
- Which features the hybrid contains was decided on every row, including each fold's validation rows, and the five weighting treatments are then compared inside folds of those same rows.
- The stored budget is the number requested, not the number the recipe was built at, and the two can differ by a factor of twenty.
- On humanness_therapeutic the AUROC is computed from hard class labels instead of class-1 probabilities.
- The JSON records nothing about which hybrid was tested: not the width, not the recipe, not the tier or augmentation condition that won the pool lookup.
- **Blocking** `complexity/c19_importance_weighted_training.py:143` — get_fitted_pool_hybrid builds X_hybrid over every row, then line 178 splits that matrix into train/val folds.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Move the hybrid build inside the fold loop. Ask get_best_search_for_downstream for the recipe once, then call build_fold_hybrid per fold with groups[train_idx]. *Cost:* Downstream fits stay at 75 per dataset.
2. Leave the build where it is and record what it was. Write bp.budget, X_hybrid.shape[1], meta['tier'] and meta['aug_condition'] into the results dict, and store fold_scores beside each mean. *Cost:* No extra model fits.
3. Score with _predict_for_metric(m, X_va_s, metric_fn) from kirby.find_best_hybrid instead of model.predict at line 226. Its signature is (model, X, metric_fn) at find_best_hybrid.py:2427 and it returns probabilities only when metric_fn is the built-in AUC. *Cost:* One import and one line.

</details>

## c20 — `complexity/c20_signal_saturation.py`

**DECIDED.** Run-repair, hoist the ranking out of the budget loop, rename the plateau field.

NO · hybrid: none needed · compute: heavy · orphaned

For one base representation at a time, ranks that representation's own columns by lightgbm_gain on each fold's training rows, scores the top-k columns on the held-out rows for k in 10..1000 plus the native width, and reports the smallest k that lands within tolerance of the full-width score. No hybrid anywhere, by design.

**Q.** For each base representation on its own, what is the smallest number of its own columns that scores within tolerance of using all of them?
**Q.** Does that saturation point differ between lightgbm and rf as the downstream model?

*On re-read:* Three of the four findings hold. The import at c20:50-54 points at a scratchpad directory that is gone, and nothing in the repo defines H.STAGED_DATASETS (c20:288) — I grepped, the only occurrence is that one use.

Why:
- The file dies at import, so it has never produced a number.
- With --dataset omitted it reads H.STAGED_DATASETS, which nothing defines, so the all-datasets path fails even after the import is repaired.
- Without --out nothing is written; the whole result is printed to stdout and lost when the process exits.
- The field called plateau_k is the first feature count in the ladder that lands within tolerance, so a lucky 10 is reported even when 25 and 50 are much worse.
- **Blocking** `complexity/c20_signal_saturation.py:50` — _HARNESS_DIR points at /private/tmp/claude-501/.../45a3af00-.../scratchpad, which does not exist.
- **Blocking** `complexity/c20_signal_saturation.py:288` — H.STAGED_DATASETS is read when --dataset is omitted, but _alloc_harness (the surviving harness) defines no such name.

*Smoke:* ran clean. As shipped it dies at import: exit 1, ModuleNotFoundError: No module named 'round2_harness' (c20:54).

<details><summary>Options considered</summary>

1. Make it run: replace lines 50-54 with the two-line repo-root bootstrap c31 uses at c31:79-87, replace H.STAGED_DATASETS at line 288 with list(H.EXPECTED_DIRECTION), and default --out to tests/results/complexity/c20 so a plain run writes a file. *Cost:* Three edits.
2. Rename plateau_k to first_k_within_tol and leave the rule alone. *Cost:* One line, plus the matching string in plateau_definition at line 201-205.
3. Redefine it: report the smallest feature count where that rung AND every larger rung in the ladder sit within tolerance. *Cost:* About four lines inside _find_plateau.

</details>

## c21 — `complexity/c21_single_rep_augmentation.py`

**DECIDED.** Cap the decoys to the donor widths conditions C and D actually use, and fix the scoring line.

**Still open:** Whether the multi-representation reference arm gains its own control.

YES · hybrid: engine, per fold · compute: heavy · dispatched

Takes one base representation at its full native width and asks whether adding a few extra columns helps: from the dataset's curated augmentation representations, from the top-5 importance columns of every other pool representation, from both, and from two matched-width random-column controls. Runs the same extras against a fixed multi-representation hybrid at budgets 500/1000/2000 as the reference, with a pure-noise donor and a jittered copy of the base planted in every donor pool and their admitted-column counts recorded per fold.

**Q.** Does adding a handful of extra columns to a single strong base representation beat that representation alone?
**Q.** Do the same extras do more on one representation at full width, or on the multi-representation hybrid at a budget?
**Q.** Does the admission rule take any columns from a pure-noise donor or from a jittered near-copy of the base representation?

Why:
- Under conditions C and D the pure-noise donor is uncapped while every real donor is capped to five features, so the decoy count on those two conditions is not comparable to the decoy count on condition B.
- _fit_score_seeds at line 149 passes m.predict straight to metric_fn, so on an AUROC dataset the metric receives hard class labels.
- The base-representation score is recomputed inside the arm loop although it depends only on the base representation, the model and the fold.
- The two multi-representation reference arms have no matched-width random control, so an H1 gain over H0 cannot be separated from the hybrid simply carrying more features.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Cap the two planted donors the same way the real ones are capped: after plant_decoys, extend per_source with the noise and echo donor names at top_n whenever per_source is non-empty. *Cost:* No change in the number of model fits — the ranker's pool shrinks, the fits per fold are the same.
2. Alternatively, leave the cap alone and size the noise donor to the post-cap total width instead, so the noise block is as wide as what the real donors actually offer. *Cost:* Same fit count.
3. Score with _predict_for_metric(m, Xva, metric_fn) inside _fit_score_seeds. *Cost:* One import and one line.

</details>

## c22 — `complexity/c22_explainability.py`

**DECIDED.** Do it. The re-run is 5 builds and 10 fits per dataset and the current per-representation SHAP cannot be defended.

PARTIAL · hybrid: engine, per fold · compute: fine · dispatched

Picks the winning blueprint from the precomputed search pool, rebuilds that hybrid per fold on training rows, fits rf and xgboost, runs SHAP on whichever scored better, averages the per-feature SHAP across folds, and maps those values back to source representations to report how much each representation contributed.

**Q.** For the winning hybrid, how many features does each base representation contribute?
**Q.** How much of the fitted model's SHAP importance comes from each source representation?
**Q.** Which individual hybrid features carry the most SHAP importance, and which representation and column did each come from?

Why:
- The per-representation SHAP breakdown mixes folds. avg_shap is the mean over all folds (line 241) but feature_info is taken from the last fold (line 231), and aggregate_importance_by_rep walks feature_info in order taking importances[current_idx : current_idx + n_features] (importance/hybrid.py:452-460).
- The comment at line 228 asserting the structure is deterministic given the blueprint is wrong; create_hybrid allocates the budget from each fold's own training rows, so which representation gets how many features moves between folds.
- The width guard at lines 239-246 does not catch this. It compares total widths only, and all seven stored files have identical widths on every fold (50, 10, 50, 141, 217, 25, 100), so shap_by_rep was written in all seven.
- Only the last fold's allocation reaches the JSON under the key 'allocation', with no per-fold record, so fold-to-fold movement is invisible afterwards.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Aggregate per fold, then average the shares. Call aggregate_importance_by_rep(all_feature_infos[i], all_shap_values[i]) inside the fold loop, and average those per-representation shares across folds instead of averaging the raw SHAP vector first. *Cost:* No extra model fits and no extra SHAP passes — the per-fold vectors already exist inside the loop and are currently thrown away.

</details>

## c23 — `complexity/c23_evaluator_model_matching.py`

**DECIDED.** Fix the ridge evaluator names and the cache namespace.

**Still open:** Rewrite c23's headline through nested CV, or delete it and let framework/f13 answer the evaluator question. f13 already asks it with only dispatchable evaluators.

NO · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

For each of eleven downstream models, runs a separate full find_best_hybrid search with the internal evaluator pinned to the same model family, to a near family, and to a far family, plus one neural-size mismatch and one run over the three tree evaluators together. Reports each arm's search score and its difference from the matched arm.

**Q.** Does pinning the internal evaluator to the downstream model's own family produce a better-scoring hybrid than pinning it to a different family?
**Q.** Does the size of the mismatch matter — is a near family better than a far one?
**Q.** Does letting the search choose among rf, lightgbm and xgboost beat pinning the evaluator to the downstream model?

*On re-read:* The ridge finding is right about the cause and wrong about the consequence, and the consequence is worse. The audit says six of the thirty-three matched/near/far cells produce no result.

Why:
- Six of the thirty-three matched/near/far cells measure an equal budget split rather than a ridge-evaluated one, and nothing in the JSON marks them.
- Every reported number is search.best_by_model[m]['score_mean'], the maximum over 120 configurations scored on the folds they were selected on (find_best_hybrid.py:5641-5645). Each delta is a difference of two such maxima, and nothing here goes through nested_cv_evaluate.
- Condition E maximises over three evaluators and is subtracted from condition A, which maximises over one. For rf, lightgbm and xgboost, A's evaluator is inside E's set, so E can never lose.
- dataset_name is mangled to f"{dataset_name}_{model_name}_{evaluator}" at line 100. The registered budget cap and model-skip policy resolve on that mangled string and find nothing, so on logd, herg_ki, caco2, qm9, adme_hlm, adme_rlm, adme_mdr1, herg, davis and the five koenig/adams/phillips sets c23 searches budgets 1000 and 2000 that every other experiment drops.
- **Blocking** `complexity/c23_evaluator_model_matching.py:54` — 'ridge' is used as an internal evaluator in five MATCH_TABLE entries and as ridge's own matched evaluator.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Replace 'ridge' in MATCH_TABLE with an evaluator create_hybrid dispatches. The dispatchable list is named in the raise itself at hybrid.py:2415-2418: rf, xgboost, lightgbm, mlp, svm, gp, knn, catboost, dnn, cnn, ngboost, bnn. *Cost:* Six find_best_hybrid calls re-run per dataset.
2. Pass cache_namespace=f"{model_name}_{evaluator}" and leave dataset_name intact. The parameter exists at find_best_hybrid.py:5948 and is applied to cache paths only, at line 6265-6266, below the line where the policies resolve. *Cost:* One line at c23:100 and one at c23:249.
3. Give condition E the same number of options as condition A: report E's winning evaluator and E's own score, and drop delta_vs_matched from the E rows. *Cost:* No re-run; the field is computed at line 270 from numbers already stored.

</details>

## c24 — `complexity/c24_blueprint_transfer.py`

**DECIDED.** Apply the feature-set transfer and build_fold_hybrid together, plus the two one-liners. Same re-run.

NO · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Runs a full find_best_hybrid on each fold's training rows to get that fold's winning recipe, scores it on that fold's held-out rows, then applies each fold's recipe to every other fold — rebuilding the hybrid on the target fold's own training rows — and reports own-minus-cross differences plus a Nogueira stability index over the selected feature sets.

**Q.** Does the recipe discovered on one fold score as well on another fold as that fold's own recipe?
**Q.** How stable is the actual set of selected features across folds?

Why:
- The cross-fold arm transfers only the four recipe fields and then re-selects every feature on the target fold's own training rows, so the feature selection the experiment exists to test never crosses a fold. Two folds with the same recipe give delta exactly 0, which line 395 reads as 'STABLE (small delta)'.
- create_hybrid is called bare at lines 219 and 302 with no groups, so its internal admission holdout falls back to a RandomState(42) 20% random carve of the training rows (hybrid.py:4552-4556). On a scaffold-grouped dataset that holdout shares scaffolds with its own training rows.
- The same two calls drop model, metric_fn, lower_is_better, step_size, candidate_ranking, augmentation_budget, aug_per_source_budget and augmentation_margin, so accept and reject inside create_hybrid is scored with r2 or accuracy rather than the dataset's metric, and the rebuild silently uses create_hybrid's defaults instead of the recipe the search chose.
- The Nogueira denominator uses d = the number of features selected on at least one fold (line 77), not the total feature count. total_features is computed at line 135 and passed in at line 401 but reaches the formula only through the M<2 guard, so phi is biased and the stored values near 0.99 are not the quantity the reference defines.
- **Blocking** `complexity/c24_blueprint_transfer.py:302` — The cross-fold arm re-runs create_hybrid on the TARGET fold's training rows, so only the four recipe fields transfer and the features are re-selected.
- **Blocking** `complexity/c24_blueprint_transfer.py:219` — create_hybrid is called without groups, so its internal admission holdout falls back to a RandomState(42) 20% random split of the training rows.

*Smoke:* ran clean. module imports clean, exit 0.

<details><summary>Options considered</summary>

1. Make the transfer transfer. Keep each source fold's feature_info from line 219, and in the cross-fold loop apply it to the target fold's representations with apply_feature_selection instead of rebuilding. *Cost:* Cheaper than today.
2. Replace both bare create_hybrid calls with build_fold_hybrid (_shared.py:3559). It forwards groups, the internal evaluator, the augmentation strategy and margin and fraction, the candidate ranking, the augmentation budgets and the blueprint's own representation scores, and it refuses model without metric_fn. *Cost:* No change in the number of builds or fits.
3. Use total_features as Nogueira's d instead of the union size, or state in the JSON that the stored phi uses the union. *Cost:* One line.

</details>

## c30 — `complexity/c30_balanced_global_selection.py`

**YOURS.** Delete and read the same three arms off c31, or take the three-line repair. Deleting loses budget 100 and the 50/300 caps. c31 already ran those arms on 10 datasets and 3 models.

*Changed 2026-09-04 (SPLIT_AUDIT c30:121, exp_b:142):* option 3 was taken. The scratchpad import is gone — c30 uses `tests/experiments/_alloc_harness` and writes `tests/results/complexity/c30_balanced_global.json`, so it no longer collides with exp_b's filename. The target and pool5/pool10 are ranked inside each outer fold's training rows by a 3-fold inner group-aware CV over all ten representations; the target's score for a fold is that fold's held-out score for the representation its own training rows chose. The out-of-fold arrays are now actually stripped before writing. Roughly 150 extra fits per dataset on top of the 235 counted below. It has still never produced a result file.

NO · hybrid: hand-rolled · compute: fine · orphaned

Three feature-selection arms per dataset at budgets 100/500/1000 over the top-5 and top-10 reps: equal per-rep quota, a per-rep top-N cap then one global lightgbm_gain rank, and no cap then one global rank. Everything is hand-rolled on a 5-fold GroupKFold; lightgbm is the fixed downstream model.

**Q.** Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat giving each representation an equal fixed quota of the budget?
**Q.** Does that per-representation cap remove the bias toward the widest representation that the no-cap global ranking shows?
**Q.** Does the manual equal-quota selection reproduce create_hybrid(allocation_method='fixed') on one fold?

*On re-read:* The five findings hold. c30:28-30 imports from the missing scratchpad.

Why:
- The file dies at import, so it has never produced a number.
- Its output path is the same missing directory, and its filename collides with exp_b_balanced_global.py, so repairing only the import would have the two overwrite each other.
- The best single representation, which every arm is measured against, is picked using scores from the same five held-out folds.
- A comment claims bulky arrays are stripped before writing; nothing is stripped, and summarize never put them in the output.
- **Blocking** `complexity/c30_balanced_global_selection.py:28-30` — Imports round2_harness from a hardcoded /private/tmp scratchpad directory that no longer exists.
- **Blocking** `complexity/c30_balanced_global_selection.py:277-279` — Writes expB_{ds}.json into the same missing scratchpad directory.

*Smoke:* ran clean. Imports clean only because I created the missing harness symlink first.

<details><summary>Options considered</summary>

1. Delete c30 and read the same three arms off c31's stored files at /Volumes/seagate/kirby_cache/c31_weighted_global. *Cost:* Nothing to re-run.
2. Repair three lines and run it: the import, the output directory to tests/results/complexity, and the filename to c30_balanced_global_{ds}.json so it stops colliding with exp_b. *Cost:* Per dataset: 235 downstream lightgbm fits and about 100 importance rankings, counted off the loops (50 standalone, 5 reproducibility, 30 quota, 120 equal-cap, 30 no-cap).
3. Repair it and also pick the best single representation inside each fold's training rows, so the target is not chosen on the folds it is judged on. *Cost:* Rewrites the target block, roughly lines 101-119, plus an inner group-aware split per fold.

</details>

## c31 — `complexity/c31_perf_weighted_global_selection.py`

**DECIDED.** Restore the full 15-scheme weighting family and re-run. Cutting 12 of 15 removed the axis the experiment exists to measure.

*Changed 2026-09-04 (SPLIT_AUDIT c31:428, c31:711):* the 15-scheme restoration is not done and is not mine. Two reporting changes are. A second baseline `target_trainpick` sits beside the whole-data target: it picks the representation per fold from that fold's inner group-aware CV scores and reads its held-out score for the fold, and every arm carries `r2_delta_mean_vs_trainpick` and `native_delta_mean_vs_trainpick`. In `best_scheme_by_cell`, `r2_delta_mean` is renamed `r2_delta_max_over_schemes` and its confidence interval is removed — it was built for one fixed weighting and does not cover choosing the best of three; `scheme_fold_wins_vs_target` and the per-scheme deltas are written instead. Costs no fits.

PARTIAL · hybrid: hand-rolled · compute: heavy · live

For each downstream model, sizes each representation's slice of a shared candidate pool by a performance weight, glues the candidate columns, runs one global importance ranking and keeps the top B. Compares that against the best single representation and against three controls (equal quota, equal cap then global, no cap then global) at budgets 500/1000, with a group-block bootstrap on the pooled out-of-fold predictions.

**Q.** Does sizing each representation's candidate slice by a performance weight beat giving every representation an equal cap before the same global ranking?
**Q.** Which weighting shape (linear, power 2, top-2) sizes the candidate pool best, and does the answer change with the downstream model or with the importance method used for the global ranking?
**Q.** Does any of these beat the best single representation at a matched budget, by more than the group-block bootstrap noise floor?

*On re-read:* Some of it holds, some does not, and the biggest thing is in neither audit.

HOLDS. The scheme cut at c31:128-138 is real and its stated reason is wall time — 3 weightings where the earlier commit 84cb140 had 14.

Why:
- The stored results were measured on representation pools that no longer exist — 8 of 10 datasets picked a top-10 pool containing representations DATASET_REPS has since cut.
- The weighting family is the question c31 asks, and 11 of its 14 weightings were cut for wall time, which is a compute reason rather than degenerate, broken or undefined.
- The docstring says the pool-sizing weighting is guaranteed identical to the allocation strategies; the shared part is the weighting function, not the score fed into it.
- The target and the pool membership are ranked on the same five folds the paired delta is measured on. This is disclosed in CAVEATS, unlike c32.

*Smoke:* not run. exceeded 6 min at '[auto] caco2: filtering to 10 general reps from DATASET_REPS' — it had not yet printed 'Loaded 2158 molecules'.

<details><summary>Options considered</summary>

1. Re-run c31 unchanged, on the current rosters. *Cost:* 30 tasks.
2. Restore the 14-scheme SCHEMES dict from commit 84cb140 (keeping top_2 alongside top_3, so 15) and re-run. *Cost:* Downstream fits per task go from about 390 to about 870.
3. Change the docstring at 13-15 and the CAVEATS to say what is shared: the weighting function, driven by a per-representation score from an inner group-aware split under the downstream model, not from create_hybrid's internal carve under the internal evaluator. *Cost:* Two comment blocks.

</details>

## c32 — `complexity/c32_allocation_strategy_cross.py`

**DECIDED.** Add the exact-budget top-up and trim, fifteen lines, then re-run. Median task is 23 minutes.

*Changed 2026-09-04 (SPLIT_AUDIT c32:204, c32:309):* the exact-budget top-up and trim is not done and is not mine. The ranking is. The target and the N=2/3/5 subsets now come from each outer fold's own training rows, via the inner group-aware CV that already sized the slots, so the caveat asserting no whole-data leak is now true of all three. `target.best_rep_per_fold`, `folds_agreeing_with_whole_data` and `subset_by_n_per_fold` record what each fold picked, and the target's per-fold R-squared and native arrays are now written, so absolute per-fold scores can be rebuilt from the JSON. Costs no extra fits. The 30 files under `/Volumes/seagate/kirby_cache/c32_allocation_cross/` predate this and are not comparable to anything written after it.

PARTIAL · hybrid: should use engine, does not · compute: heavy · live

Crosses three axes the earlier runs only moved one at a time: how many top representations get budget (2/3/5/10), the weighting shape (linear, power 2, softmax, rank-exp), and the mechanism (per-representation weighted quota, or weighted candidate slices then one global ranking). 32 strategies per (dataset, model, budget) cell, each paired per fold against the best single representation.

**Q.** Inside the grid interior nobody has run, does a given weighting shape beat the others once the number of representations receiving budget is also varied?
**Q.** Does the per-representation weighted quota or the weighted-slice-then-global-rank mechanism win, and does that flip with the number of representations, the weighting, the budget or the downstream model?
**Q.** Does any full strategy beat the best single representation at a matched budget?

Why:
- The quota mechanism does not spend the budget it claims. In 279 of 864 matched pairs it is compared against a global arm that spent the full budget while it spent less — the same shape as f01's fixed-arm bigger-budget artifact, running the other way.
- A caveat in the file asserts the quota mechanism equals allocate_performance_weighted. It does not; the engine's top-up and trim are absent.
- A caveat asserts no whole-data leak in the sizing. The pool, the N-subsets and the target are all ranked on the folds the deltas are measured on, and N is the axis the experiment exists to move.
- The target's per-fold R-squared and native arrays are never written, so no absolute per-fold score can be rebuilt from the JSON — only the deltas. c31:436-438 writes them.

*Smoke:* not run. exceeded 6 min at '----- MODEL: rf -----'.

<details><summary>Options considered</summary>

1. Add the engine's top-up and trim to the quota arm — after line 331, walk the representations best-first filling to the budget and trim overshoot from the worst, matching hybrid.py:1400-1424 — then re-run. Also write the target's per-fold arrays into mout['target'] before you resubmit. *Cost:* 27 non-qm9 task files go stale and are rewritten.
2. Route both mechanisms through the engine: create_hybrid(allocation_method='performance_weighted', weighting_scheme=..., weighting_params=..., precomputed_alloc_scores=fold_hib[i]) for quota, and allocation_method='weighted_global' with pool_total_mult=2 for global. *Cost:* Same re-run as the first option, median 1,397 seconds per task.
3. Change nothing and handle it in the analysis: report win counts only over cells where every strategy spent at least 98 percent of the budget. *Cost:* No re-run.

</details>

## exp_b — `complexity/exp_b_balanced_global.py`

**DECIDED.** Delete. Move the metric-direction gate into _alloc_harness first — four other scripts want it. c30 is the later rewrite of the same three arms, c31 the one after that.

*Changed 2026-09-04 (SPLIT_AUDIT exp_b:142):* the delete has not happened — the finding was filed against this file, so it was fixed here as well as in c30. The three blocking items are gone: it imports `_alloc_harness`, its direction gate is `H.assert_metric_direction` (the missing `_label_signal_ladder` is not needed), and it writes into `tests/results/complexity/exp_b/`, no longer sharing c30's filename. best representation, pool5, pool10 and the comparator model are all picked per fold from a 3-fold inner group-aware CV on that fold's training rows; every arm in a fold uses that fold's model. Deleting it costs nothing that is not now also in c30.

NO · hybrid: hand-rolled · compute: fine · superseded

The same three arms as c30 — equal quota, per-representation cap then one global ranking, no cap then one global ranking — over the top-5 and top-10 representations at budgets 100/500/1000, run across four datasets in one process, with a blocking metric-direction check and a downstream model chosen from lightgbm and rf.

**Q.** Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat the equal per-representation quota?
**Q.** Does the cap remove the width bias the no-cap global ranking shows?
**Q.** Does either global arm beat the best single representation at a matched budget?

Why:
- The file dies at import, so it has never produced a number.
- H._label_signal_ladder, called at line 104 to build the direction gate, is defined nowhere in the repository — I grepped, that call is the only occurrence.
- Its output directory is the missing scratchpad, and its filename expB_{ds}.json is the same one c30 writes.
- The best single representation is picked under lightgbm, and then the downstream model is swept over lightgbm and rf on that one representation, so the target is not the best representation-and-model pair the docstring at line 11 claims.
- **Blocking** `complexity/exp_b_balanced_global.py:34-36` — Imports round2_harness from a hardcoded /private/tmp scratchpad path that no longer exists.
- **Blocking** `complexity/exp_b_balanced_global.py:104` — H._label_signal_ladder is not defined in tests/experiments/_alloc_harness.py, the only surviving copy of that harness.
- **Blocking** `complexity/exp_b_balanced_global.py:63-64` — OUTDIR is the deleted scratchpad directory, and the filename expB_{ds}.json (line 321) is the same one c30_balanced_global_selection.py:277 writes.

*Smoke:* ran clean. Imports clean only with the harness symlink in place.

<details><summary>Options considered</summary>

1. Delete exp_b. *Cost:* Nothing to re-run.
2. Lift the direction gate into _alloc_harness before deleting the rest: write the missing ladder builder there and expose exp_b's direction_gate under a real name. *Cost:* The ladder builder is about six lines — a single feature made of the labels plus Gaussian noise at each of five standard deviations, which is what the call at exp_b:102-106 wants.

</details>

## c43_signal — `complexity/c43_buchwald_signal.py`

**DECIDED.** Repair it. Ten model fits, and it is the anchor c43_step3 currently hardcodes as a literal.

NO · hybrid: none needed · compute: fine · orphaned

A one-representation diagnostic on buchwald_hartwig: scores cgr_morgan with random forest over 5 group-aware folds and compares the RMSE against the label spread and against predicting the training-fold mean.

**Q.** Does buchwald_hartwig carry real predictive signal under cgr_morgan with a random forest, or is the RMSE close to predicting the training mean?
**Q.** What is the RMSE scale relative to the label standard deviation, so a later hybrid delta can be read as large or small?

Why:
- The file dies at import, so it has never produced a number.
- The anchor RMSE it exists to establish is printed to stdout and never written, and another script hardcodes that number as a literal.
- A group-size line built with np.bincount prints and is then superseded by the correct np.unique version two lines below it.
- **Blocking** `complexity/c43_buchwald_signal.py:14-18` — Imports round2_harness from a hardcoded /private/tmp scratchpad path that no longer exists.

*Smoke:* ran clean. Imports clean only with the harness symlink in place (c43_buchwald_signal.py:15-18).

<details><summary>Options considered</summary>

1. Repair the import, write the result dict to tests/results/complexity/c43_buchwald_signal.json, and delete the dead bincount line. *Cost:* Three edits.

</details>

## c43_step3 — `complexity/c43_buchwald_step3.py`

**DECIDED.** Take option 1. It is the only file in this group that builds through the engine.

NO · hybrid: engine, per fold · compute: fine · orphaned

Three measurements on buchwald_hartwig over 10 hand-picked representations: how cgr_morgan's RMSE degrades as it is truncated to K top features, how many features cgr_morgan actually gets inside a real create_hybrid call and how that slice scores alone, and how a single global lightgbm_gain ranking over the whole concatenated pool compares at the same budget.

**Q.** When cgr_morgan is truncated to its top K features, at what K does its accuracy stop matching its full width?
**Q.** How many features does cgr_morgan actually receive under create_hybrid at budget 100 and 1000, and does that slice alone still carry the representation's signal?
**Q.** Does one global lightgbm_gain ranking over the concatenated pool beat the quota hybrid at a matched budget, and which representations does it draw from?

*On re-read:* The four findings hold. The import at 19-21 and the checkpoint path at 25 both point at the missing scratchpad.

Why:
- The file dies at import.
- Even with the import repaired it raises KeyError, because 6 of its 10 hand-picked representations are no longer in the buchwald_hartwig roster.
- The headline delta at line 131 subtracts a hardcoded random-forest RMSE from a lightgbm hybrid RMSE, so it crosses the model axis.
- n_per_rep=B at line 96 reads as if it sets the per-representation budget under top_5 and performance_weighted; the engine ignores it outside the 'fixed' path.
- **Blocking** `complexity/c43_buchwald_step3.py:19-25` — Imports round2_harness from a deleted /private/tmp scratchpad path, and CKPT points into the same missing directory, so the checkpoint write at line 82 fails as well.

*Smoke:* ran clean. Imports clean only with the harness symlink in place (c43_buchwald_step3.py:20-21).

<details><summary>Options considered</summary>

1. Repair the import and the checkpoint path, replace POOL with the current DATASET_REPS['buchwald_hartwig']['general'] list, drop n_per_rep from line 96, and compute the headline delta from full_rmse_per_fold at line 122 instead of the literal 14.3664. *Cost:* About six edits.
2. Repair it as above but leave POOL hand-picked, and record in the output that the pool is not the roster. *Cost:* Cost not established — six representation matrices would have to be built or found in cache first, and I have not checked whether they are cached.
3. Delete measurements (a) and (c) and keep only (b), the realised feature count for cgr_morgan inside create_hybrid and how that slice scores alone. *Cost:* Removes roughly 45 of about 300 downstream fits for (a) and 20 for (c), leaving about 240.

</details>


---

# Noise — 14 files

## n01 — `noise/n01_activity_cliff_framework.py`

**DECIDED.** Take the local fixes. Fix the empty-dict test so a failed hybrid pool stops wiping the per-representation baselines.

**Still open:** Whether the hybrid recipe is searched per fold.

**Done 2026-09-04 under the split audit.** The parenthesis below was not true of reaction, TCGA or drug-target affinity: all three took the Euclidean-on-concatenated branch, so the cliffs a representation was graded on were built from that representation plus nine others, and whichever contributed the most features defined most of them. Reaction now gets a reference from reactant and product Morgan bits off `rxn_smiles`, which `load_data` carries through; drug-target affinity gets Tanimoto on the drug, zeroed between different proteins. TCGA has no source outside its own representations and now records `cliff_reference_independent: False` with each representation's feature count, instead of the docstring claiming one. Also confirmed, no change needed: each fold's budget split no longer comes from whole-dataset representation scores — `build_fold_hybrid` stopped forwarding the blueprint's `rep_scores`, so `create_hybrid` re-scores on the fold's own training rows.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Marks each molecule as a cliff or non-cliff compound from a reference that does not use any of the representations (ECFP4 Tanimoto for molecular, sequence edit distance for antibody, reactant-and-product Morgan Tanimoto for reaction, drug Tanimoto within one protein for drug-target affinity, Euclidean on the concatenated representations only where none of those exist), then runs 5-fold scaffold-grouped CV per model per representation and reports the metric on the cliff rows, the non-cliff rows, and the difference. The hybrid is rebuilt per fold from a searched blueprint and gets the same treatment.

**Q.** Is the prediction error on activity-cliff compounds worse than on non-cliff compounds, and by how much?
**Q.** Is that cliff/non-cliff gap smaller for the hybrid than for the individual base representations?
**Q.** Does the answer to the second question hold across rf, xgboost, svm, mlp and lightgbm, or only for rf?

*On re-read:* Three things in the audit do not survive re-reading.

(1) The stored results. tests/results/noise/n01_activity_cliff_framework/ holds five files, dated 1 April and 26 May 2026.

Why:
- The recipe the hybrid is built from was searched over the whole dataset, then scored on folds of that same dataset, so each fold's held-out rows helped choose it; the individual representations get no equivalent tuning, and the cliff gap comparison between them is the experiment's whole claim.
- --reps on the command line is discarded: n01:620 hands the comma list to load_data's rep_preset and leaves rep_names at None, so _shared.py:2809 auto-filters to the ten DATASET_REPS general representations instead. parse_rep_args at _shared.py:5280 was written for exactly this and n01 never calls it.
- n01:202 tests `if bp_by_model is None`, and an empty dict is not None. If every blueprint fetch at n01:520-542 fails, models_to_run becomes empty, get_model_factories returns nothing, and n01:213 returns error='no_model_factories' — so the per-representation baselines, which need no blueprint at all, are silently never computed.
- n01:598 stamps lower_is_better=True, which describes the cliff/non-cliff gap, and save_experiment_results then fills metric_name from the dataset registry. On ginkgo_hic that writes metric_name='spearman' beside lower_is_better=True.

*Smoke:* FAILED. exit 139 (segfault) after 160s, right after the 'EXPERIMENT N01: CLIFF-STRATIFIED PREDICTION ERROR' header; PYTHONFAULTHANDLER printed only 'Fatal Python error:' before dying.

<details><summary>Options considered</summary>

1. Fix the eight local defects and leave the whole-dataset recipe alone. Call parse_rep_args at n01:620 and pass rep_names through. *Cost:* About twenty lines across eight places.
2. Move the search inside the fold loop. Take the folds from find_best_hybrid.nested_cv_outer_folds (line 7585) instead of make_cv_splitter, run the search per fold on that fold's training rows under nested_cv_cache_namespace (line 7575), and build each fold's hybrid from that fold's own blueprint. *Cost:* Five searches per dataset instead of one, per outer fold, cached under their own namespaces.
3. Leave the recipe as it is and record in the write-up that the hybrid's recipe was chosen with every row visible while the representations were not tuned at all. *Cost:* Nothing.

</details>

## n02 — `noise/n02_sample_weighting.py`

**DECIDED.** Merge condition 5 into the conditions 3-4 loop and pair the fold sets. Saves 15 rebuilds per dataset.

**Still open:** Whether the importance scorer behind the weighted distance is swept rather than fixed to lightgbm_gain.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Scores each training row by how well its label agrees with its 10 nearest neighbours in a distance where every feature is scaled by the square root of its lightgbm_gain importance, then refits with those scores as sample weights. Runs five conditions — each base representation weighted and unweighted, the hybrid weighted and unweighted, and the hybrid with ActivityCliffAwareEstimator weights — for rf, xgboost and lightgbm.

**Q.** Do importance-weighted sample weights improve held-out prediction relative to training unweighted?
**Q.** Does that improvement differ between an individual base representation and the hybrid?
**Q.** Do cliff-aware weights beat plain neighbour-agreement weights on the hybrid?

*On re-read:* The audit says the base-versus-hybrid comparison the docstring names is never computed. Half of that is wrong.

Why:
- Condition 5 rebuilds hybrids that conditions 3-4 already built. n02:452 calls build_fold_hybrid with the same model_bp, the same model and the same folds as n02:352, because both loops call cv_split with identical arguments and cv_split is deterministic (_shared.py:1140).
- If the rf blueprint fetch returns nothing, n02:286 returns a bare {'error': 'no_blueprint'} dict and discards results['per_rep'], which by then cost 10 representations x 3 models x 5 folds x 2 fits = 300 model fits. It also skips make_structured_output, so the saved file gets no meta block.
- A failed weighted fit appends nan at n02:243 while the matching unweighted score stays in the list. n02:248 takes a plain mean and n02:250 a nanmean, so `delta` at n02:252 subtracts two means taken over different sets of folds.
- results['summary'] is initialised empty at n02:177 and never filled, so every file carries an empty summary block.

*Smoke:* not run. not attempted, time

<details><summary>Options considered</summary>

1. Move condition 5 into the conditions 3-4 fold loop. The cliff-aware weights only need X_train_h, y_train and X_val_h, which that loop already has in scope at n02:352-365. *Cost:* Half the hybrid builds per dataset removed: 30 to 15.
2. Make the weighted and unweighted means paired. Keep a fold only when both fits succeeded, take a plain mean of each, and write n_folds_paired alongside. *Cost:* About fifteen lines in three places (n02:243-252, n02:283-286, n02:405-414).
3. Add svm to WEIGHTED_MODELS at n02:54, and either fill results['summary'] again with the block commit 1474bd1 removed, or delete the empty key and let per_model_top3 be the answer. *Cost:* svm adds one model to every loop: conditions 1-2 go from 300 to 400 downstream fits per dataset, conditions 3-5 from 15 to 20 hybrid builds after the duplicate above is removed.

</details>

## n06 — `noise/n06_bindingdb_ground_truth.py`

**DECIDED.** Fix the output shape and add per-representation spaces. The concatenation-only comparator is against the baseline rule.

**Settled 2026-09-04 under the split audit.** The replicate rows stay; what changed is what each estimator may look at. The four nearest-neighbour estimators take `exclude_groups`, so a row never draws a neighbour from its own (drug, target) pair, and `leave_one_out` holds the whole pair out through group-disjoint folds instead of relying on out-of-bag, which never leaves a row's repeats out. The hybrid space is rebuilt protein-fold by protein-fold, so a row's features are chosen from rows outside its own protein group; `hybrid_space_provenance` records that, or why it fell back. On a fixture of 12 pairs x 4 replicates, distance_weighted_residual's correlation with replicate spread fell from +0.68 to +0.04. Two of the bullets below are now stale: `cross_val_residual` does receive the protein grouping (checked by constructing all ten estimators), and the hybrid space is no longer fitted on all rows.

PARTIAL · hybrid: engine, per fold · compute: heavy · orphaned

Groups BindingDB rows by (drug_id, target_id) from the loader metadata, takes the standard deviation of each group's pKd values as a real assay-noise magnitude, then runs 10 noise estimators over two feature spaces and reports the Spearman correlation between each estimator's per-row noise score and that group standard deviation, over the rows in disagreeing groups only.

**Q.** Which noise estimator's per-row score agrees best with real repeat-measurement disagreement in BindingDB?
**Q.** Does the estimator agree better when it is run over the hybrid feature space than over the concatenation of all representations?

*On re-read:* The audit calls n06:249 a blocker on the grounds that the Gaussian process would be handed several thousand rows and never finish. That is wrong.

Why:
- The only space the hybrid is compared against is the hstack of every representation, built at n06:208 and named 'concat' at n06:211. Individual representations are never scored as spaces, so nothing here says whether a single representation beats both.
- The hybrid space is fitted on all rows with all labels at n06:218, and those labels are what the replicate standard deviation is computed from. The concatenation had no label access.
- Rows in the same replicate group are the same drug and the same target, so their feature vectors are identical in every representation. neighbor_residual, local_variance_ratio and distance_weighted_residual find each row's own replicate as its nearest neighbour at distance zero, so they recover the group standard deviation close to by construction.
- cross_val_residual splits with cross_val_predict(model, X, y, cv=5) at noise_estimation.py:451 — plain KFold, no groups, and it never reaches _oof_apply. The same drug-target pair sits on both sides of a fold.
- **Blocking** `noise/n06_bindingdb_ground_truth.py:249` — When there are more replicate rows than GP_MAX the branch sets sub_idx = rep_idx, i.e.

*Smoke:* ran clean. exit 0, ~2 min, 'ALL N06 SMOKE TESTS PASSED'.

<details><summary>Options considered</summary>

1. Score every base representation as its own space, alongside the hybrid and the concatenation. n06:211 builds `spaces` as a dict; add one entry per representation from `reps` before the loop at n06:269. *Cost:* Estimator runs go from 2 spaces x 10 estimators = 20 to 12 x 10 = 120 on a ten-representation dataset.
2. Score the estimators on one row per replicate group instead of on all of them. Keep a single representative row from each group, correlate that row's noise score against its group's standard deviation, and leave the rest of the rows in place as ordinary training data for the estimators that fit a model. *Cost:* Fewer rows scored, so cheaper, not dearer.
3. Search the hybrid on the rows that have no replicate, then apply its feature selection to the replicate rows. That means calling find_best_hybrid on the non-replicate subset rather than get_fitted_pool_hybrid at n06:218, because the pool's precomputed searches were built over all rows. *Cost:* One search per dataset that the precompute cache does not hold, so it runs cold.

</details>

## n07 — `noise/n07_remediation.py`

**DECIDED.** Add the caveat field and the six one-line fixes.

**Still open:** Whether the RECIPE (allocation method, importance method, internal evaluator, budget) moves to a noise-searched pool. The stored noisy lineage is 0.05/0.10/0.20 gaussian and n07 runs the uniform condition at p=0.1, so it needs searches that do not exist.

**Done 2026-09-04 under the split audit — option 2 below.** The build_fold_hybrid call moved inside the noise-seed loop and takes `y_tr_noisy`; one clean-label build per fold stays, and every noisy condition is ALSO measured in it under `per_model_per_strategy_oracle_features`, so the head start is readable. Hybrid builds per dataset go from 25 to 100, as costed there. Separately, the recovery denominator: a percentage is now NaN when the injected noise did not degrade its reference, with the reason under `recovery_undefined`, and which sign counts as degradation comes from the metric's direction — so the smoke run's -0.0269 gap yields NaN instead of 434%, -448% and 264%.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Injects outlier-condition noise into each fold's training labels, then compares four cleanup strategies (remove / downweight / label_smooth / knn_smooth) against a clean and a do-nothing baseline, in a per-model hybrid feature space and again in each individual base representation. Reports recovery as (strategy - noisy) / (clean - noisy).

**Q.** Which of remove, downweight, label_smooth or knn_smooth recovers the most of the clean-to-noisy performance gap, separately for each of the five downstream models, in that model's hybrid feature space?
**Q.** Which strategy recovers the most in each individual base representation (random forest only)?
**Q.** Side-car written to the same file: does the hybrid beat the best single representation under leave-one-fold-out re-selection, on the CLEAN-label search?

*On re-read:* Two things in the earlier findings do not survive re-reading. First: it says a hybrid-versus-single-representation read across the two blocks compares five models against one.

Why:
- The hybrid feature space is chosen using the clean labels the run then corrupts: line 503 hands build_fold_hybrid y_train_clean, and injection happens 29 lines later at 532.
- downweight cannot run for mlp at all, because sklearn's MLPRegressor.fit accepts no sample_weight, so that cell is always empty.
- Line 324 writes augmentation_strategy onto the Blueprint object the pool holds; n09, n11, n12 and n13 read that same object later in the same run_all process.
- Recovery divides by the clean-to-noisy gap with only a 1e-10 floor at line 683; the smoke run had a gap of -0.0269 and reported 434%, -448% and 264%.

*Smoke:* ran, NaNs present. exit 0, 'n07 RAN OK in 16.9s'.

<details><summary>Options considered</summary>

1. Say in the saved file that the hybrid feature mix was selected AND built on clean labels, the way n11's docstring already does. Add one field beside 'claim_scope' at line 421. *Cost:* Zero model fits.
2. Build the hybrid on the noisy labels: move the build_fold_hybrid call at line 502 inside the noise-seed loop and pass y_tr_noisy instead of y_train_clean. Keep one clean-label build for the no_noise arm. *Cost:* Hybrid builds per dataset go from 25 (5 models x 5 folds) to 100 (25 clean plus 5 models x 5 folds x 3 seeds).
3. Also search the recipe on noisy labels. get_best_search_for_downstream takes label_noise= (_shared.py:4322) and resolves data['precomputed_searches_noisy'] (line 4457). *Cost:* It cannot be done as written.

</details>

## n09 — `noise/n09_lnl_survey.py`

**DECIDED.** Pass model and metric_fn to build_fold_hybrid, and store fold_scores. Neither costs a fit.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Sweeps six injected noise levels; at each level and fold it re-runs find_best_hybrid on the NOISY training labels to build a hybrid, then trains a plain DNN and each of twelve learning-with-noisy-labels methods on three feature spaces (one fixed base representation, the hybrid, and an importance-weighted hybrid). Reports which methods rank top-3 in each space and whether that ranking differs.

**Q.** Does swapping one fixed base representation for a per-fold, noise-searched hybrid change which learning-with-noisy-labels methods come out top-3 at the highest injected noise level?
**Q.** Does importance-weighting the hybrid's columns change that ranking again, relative to the unweighted hybrid?
**Q.** How much of the clean-label score does each (feature space, LNL method) pair retain as the injected level rises from 0 to 1.0 of the label spread?

Why:
- build_fold_hybrid at line 330 is given neither model= nor metric_fn=, so accept and reject are scored by lightgbm with r2 or accuracy, while the inner search that chose the recipe at line 296 ran under random forest and the dataset's own metric.
- Line 328 silently falls back to the clean-label pool blueprint when the inner noisy search returns nothing scorable, and no field records which folds fell back. The fairness claim in the docstring at lines 40-52 rests entirely on that re-search.
- Only mean, standard deviation and fold count are written (lines 448-475). hybrid_improves, hybrid_harms and ranking_changed subtract two unpaired means, and the file carries no per-fold numbers to redo them paired.
- Condition A is recorded even when the hybrid build failed at line 338, so base_no_lnl and hybrid_no_lnl can be means over different fold sets and are still subtracted from each other.

*Smoke:* not run. not attempted, time

<details><summary>Options considered</summary>

1. Pass the random forest instance and the dataset metric into build_fold_hybrid at line 330: model=_rf_factory(), metric_fn=metric_fn, lower_is_better=lower_is_better. _rf_factory already exists at line 293 and build_fold_hybrid accepts all three (_shared.py:3562-3563). *Cost:* No change to the number of searches or trainings.
2. Record the fallback at line 328: count, per level, how many folds used the clean-label pool blueprint because the inner search returned nothing. *Cost:* Zero fits.
3. Or refuse the fold instead of falling back: when the inner search returns nothing scorable, skip the hybrid arms for that fold rather than substituting a clean-label recipe. *Cost:* Zero extra fits; it removes the neural-net trainings for the skipped folds.

</details>

## n11 — `noise/n11_estimation_remediation_grid.py`

**DECIDED.** Guard the two summary loops. Without it the grid runs for hours on a classification dataset and writes nothing at all.

**Done 2026-09-04 under the split audit.** `best_config`, `top_10` and `robust_configs` are renamed `*_search_max` and say they are a search result. Beside them, `heldout_recovery` re-picks the winner from the other folds and reads it on the held-out one — arithmetic over per-fold scores, no extra fitting — and reports whether one config won every fold. `read_n11` prints both. hybrid_sim is also rebuilt per fold from that fold's training rows, so the third question below is no longer answered in a space the held-out molecules helped choose.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

For each injected noise level, fold and noise seed, it builds ~12 candidate feature spaces (one base representation, a hybrid, a label-blind similarity hybrid, and the other nine base representations), fits every available noise estimator in each space, then applies four alphas x three weighting and four smoothing methods and re-trains a random forest on the base representation. Reports recovery of the clean-to-noisy gap for every cell.

**Q.** Which combination of noise estimator, distance metric, remediation method and alpha recovers the most of the clean-to-noisy performance gap, and which combinations stay positive at more than one noise level?
**Q.** Does computing the noise scores in a hybrid feature space, rather than in one base representation, raise that recovery? (the code labels this comparison oracle-biased and tells the reader not to headline it)
**Q.** Does the label-blind similarity hybrid (hybrid_sim), whose selection never saw activity labels, recover more than the predictive hybrid or the base representation?

Why:
- _compute_summary kills the run after the whole grid has been computed, and nothing is written. Line 678 calls .get('recovery') on every value under an estimator key.
- On a classification dataset the resolved metric is auc (_shared.py:1538) and lines 318, 334, 499 and 527 feed model.predict, which returns hard class labels, to metric_fn. herg is the one binary molecular dataset in the submit list (submit_all_datasets.sh:35).
- Line 199 calls create_hybrid directly. No groups is passed, so what accepts or rejects each feature is a random 20% carve of the training rows.
- Line 221 PCA-caps the nine other representation spaces to the training-row count. The base space at line 190, the hybrid space at line 209 and hybrid_sim at line 522 are not capped.
- **Blocking** `noise/n11_estimation_remediation_grid.py:678` — `rv.get('recovery', float('nan'))` is called on every value in est_data, but est_data also carries diagnostic markers forwarded verbatim at line 607: '_errors' is a list and '_smoothing_skipped_clf' is a bool.

*Smoke:* not run. not attempted, time

<details><summary>Options considered</summary>

1. Guard the two summary loops the way the print loop at line 637 already does: skip a key starting with an underscore, and skip anything that is not a dict. Two lines, at 678 and 770. *Cost:* Zero fits.
2. Use kirby.find_best_hybrid._predict_for_metric(model, X, metric_fn) at lines 318, 334, 499 and 527. It is already imported this way by c02 and f15, and returns model.predict unless metric_fn is the AUC one (find_best_hybrid.py:2427-2443). *Cost:* Zero extra fits.
3. Replace the bare create_hybrid at line 199 with build_fold_hybrid (_shared.py:3559), passing groups[train_idx], a random forest instance and metric_fn. _build_estimation_spaces needs the fold's validation representations threaded in as well; the returned validation half is discarded. *Cost:* No change to the 574,560 random forest fits or the 20,520 estimator fits.

</details>

## n12 — `noise/n12_importance_weighted_distance.py`

**DECIDED.** Route through build_fold_hybrid and add the zero-importance marker.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Injects synthetic training-label noise at 6 levels x 5 folds, builds 5 feature spaces (base rep, base rep scaled by sqrt(importance), hybrid, hybrid scaled two ways), runs 3 kNN noise estimators x 3 k values in each space, converts the noise scores to sample weights and refits a RandomForest on the BASE rep with those weights. Reports the fraction of the clean-vs-noisy gap recovered per space.

**Q.** Does scaling z-scored features by sqrt(feature importance) before computing kNN distances improve neighbour-based detection of corrupted training labels?
**Q.** Is that weighting redundant once the features come from a hybrid instead of a single base representation?
**Q.** Which neighbour count k and which importance method (lightgbm_gain vs treeshap) give the best recovery?

*On re-read:* Two corrections. First, the spread reported beside each condition mean is not a fold spread.

Why:
- The hybrid is built by a bare create_hybrid at n12:146 with no groups, so the 20 percent of training rows that accept or reject each feature is a random carve sharing scaffolds with the rows it judges.
- That same call omits eight recipe fields the blueprint carries, so the rebuilt hybrid is not the recipe the search chose.
- When an importance method returns all zeros, `_weight_features` returns the input unchanged (n12:104-106), so the weighted condition is byte-identical to the unweighted one and is still reported as a separate condition; the smoke saw exactly this under lightgbm_gain.
- Per-fold scores are discarded at n12:456, so no paired per-fold delta can be reconstructed from the saved file later.
- **Blocking** `noise/n12_importance_weighted_distance.py:146` — create_hybrid is called with groups=None.

*Smoke:* not run. exceeded 6 min at '=== Sigma=1.0 ==='.

<details><summary>Options considered</summary>

1. Replace the create_hybrid at n12:146 with `_shared.build_fold_hybrid`, passing groups[train_idx], an rf instance, metric_fn and lower_is_better. It needs the fold's validation slice, which `_build_fold_shared` does not currently receive, so its signature and its one caller at n12:415 change too. *Cost:* About 12 lines.
2. In `_weight_features` (n12:96), return a marker when the importance sums below 1e-10 instead of silently returning the input, and have `_build_conditions` drop that condition and record the reason. *Cost:* About 6 lines.
3. Keep the per-fold score lists in the saved payload alongside the means, and take the std over folds rather than over the pooled (estimator, neighbour count, importance method) grid. *Cost:* About 15 lines in the aggregation at n12:447-470 and in `_compute_summary`.

</details>

## n13 — `noise/n13_noise_distributions.py`

**DECIDED.** Fix the summary tally so hybrid_sim stops counting as a baseline, then the bare create_hybrid. The tally recomputes from a stored file with no re-run.

**Done 2026-09-04 under the split audit.** The `honest_rescore` key is renamed `per_fold_scores_single_recipe`: it held one hybrid recipe per model and one base representation, so the leave-one-fold-out re-pick it promised had nothing to pick from. It now carries `reselection: 'none_single_recipe_per_model'` and the recipe count. Nothing read the old key. hybrid_sim is also rebuilt per fold from that fold's training rows.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Sweeps 10 dose-matched regression noise conditions (or 6 classification flip models) x 6 levels x 5 folds x 3 noise seeds. In each cell it builds three feature spaces (base rep, predictive hybrid, label-blind similarity hybrid), runs all 12 label-noise-learning methods on each space (Track A) and 3 noise estimators x 4 models with sample weights (Track B), plus a per-model matched-blueprint hybrid trained plainly and fed to the honest_rescore helper.

**Q.** Does the ranking of label-noise-learning methods measured under Gaussian noise transfer to the other injected noise shapes?
**Q.** Does the hybrid feature space keep its advantage over the base representation across every injected noise shape?
**Q.** Does noise detection performed in a hybrid space improve base-representation training more than detection in the base space (Track B)?

Why:
- The hybrid-advantage tally at n13:948-951 sends every space not literally named 'hybrid' into the base pool, so the label-blind similarity hybrid is counted as a baseline; `hybrid_advantage_persists` is hybrid against base-plus-another-hybrid.
- The Track A and Track B space is built by a bare create_hybrid at n13:185 with no groups, while the honest arm 400 lines later at n13:596-604 builds it correctly through build_fold_hybrid; the two arms of one file disagree.
- `_bp_sig` (n13:141-155) leaves out the blueprint's representation set and its per-representation scores, and it is the key for `_hyb_space_cache` at n13:584, so two models whose blueprints differ only in those two things share one hybrid space.
- The honest re-score arm is given one candidate per model (n13:818-825) and one baseline representation per model (n13:817), so the leave-one-fold-out selection returns the same item on every held-out fold and neither arm pays a selection tax; the comment at n13:514-521 says otherwise.
- **Blocking** `noise/n13_noise_distributions.py:951` — In _compute_summary the space loop sends every space that is not literally named 'hybrid' into base_scores, so the hybrid_sim arm is counted as a baseline.
- **Blocking** `noise/n13_noise_distributions.py:185` — create_hybrid is called with groups=None, so the internal admission holdout falls back to hybrid.py:4450's np.random.RandomState(42) random 20% split instead of a group-aware fold.
- **Also caught by the hybrid-usage audit** — _seed_combined was used and never defined, so every Track A method raised NameError into a bare except and Track A produced nothing.
- **Also caught by the hybrid-usage audit** — The summary iterated the module-level NOISE_TYPES, which is None, raising TypeError and killing the summary; it now iterates the noise types the run produced.

*Smoke:* FAILED. The suggested command exits 0 without running the experiment.

<details><summary>Options considered</summary>

1. In `_compute_summary`, split the Track A spaces into three arms: base representation, predictive hybrid, similarity hybrid. Report each pairwise win rate separately instead of pooling. *Cost:* About 20 lines at n13:939-1010.
2. Alternative to the previous option: exclude the similarity hybrid from the tally and report it nowhere. *Cost:* One line at n13:948.
3. Replace the create_hybrid at n13:185 with build_fold_hybrid, passing groups[train_idx] and leaving model and metric_fn unset. `_build_spaces` gets the fold's validation representations, and the returned validation matrix replaces the manual rebuild at n13:620-631. *Cost:* About 20 lines added, about 12 deleted.

</details>

## n14 — `noise/n14_activity_cliff_landscape.py`

**DECIDED.** Set n_jobs from KIRBY_LGBM_THREADS in make_model, then re-run the smoke on caco2.

**Done 2026-09-04 under the split audit.** `from_linear` divides labels by their own standard deviation, so every residual the stage-0 diagnostics return is in standard deviations of the label while the Kramer floor of 0.54 is in pKi. `_rescale_stage0` multiplies the residual spread, the five tier spreads, the five tier biases and the overall bias by that standard deviation and writes them under `*_label_units`, beside the floor and a label-units bias flag. On a herg-like label column with standard deviation 1.198, a reported 0.45 is 0.539 pKi — at the floor, not under it.

YES · hybrid: none needed · compute: fine · dispatched

A survey that trains no downstream comparison. It builds a high-confidence activity-cliff adjacency from matched molecular pairs when the loader supplies them, else from ECFP4-Tanimoto similarity above 0.55, crossed with an absolute 2-log-unit activity gap; then reports cliff prevalence, cliff-graph degree and connected-component structure, three feasibility flags, and the housing stage-0 residual diagnostics (kurtosis, U-shape heteroscedasticity, bias) from a group-aware out-of-fold LightGBM.

**Q.** How many activity cliffs does this dataset actually contain, and what fraction of its compounds participate in one?
**Q.** Are those cliffs clustered into congeneric series or isolated pairs, i.e. do neighbour-based or per-point methods apply?
**Q.** Does this dataset support the deferred method families at all: enough cliffs to calibrate a positive-unlabelled classifier, replicate measurements, more than one assay?

Why:
- The smoke run died with a segmentation fault right after the 'Feasibility:' line, which is the point where `stage0_diagnostics` starts. The only compute there is LightGBM: `oof_predict` fits one `make_model` per fold (core.py:123), and `DEFAULT_LGB_PARAMS` (core.py:31-42) sets no `n_jobs`, so it runs on every thread.
- A segmentation fault is not an exception, so the try/except at n14:337-353 does not catch it and the process dies. The cliff survey — which is what the experiment is for — is computed before that point and is lost with it.
- The stage-0 feature matrix is every loaded representation side by side (n14:339). There is no width cap.
- `_cliff_mask_from_similarity` builds a second n x n float64 array for the label gaps (n14:123) on top of the n x n similarity matrix. At 15,000 rows those are two 1.8 GB arrays live at once.

*Smoke:* FAILED. exit 139 (segfault) inside the experiment, after the 'Feasibility:' line, despite the harness pinning OMP/MKL/OPENBLAS to 1 thread and restricting internal evaluators to rf — so the guardrail that saved n07/n13/n15 did not hold here.

<details><summary>Options considered</summary>

1. Read `KIRBY_LGBM_THREADS` in `make_model` (src/kirby/utils/noisy_labels/core.py:45-49) and set `n_jobs` from it, defaulting to -1 as now. *Cost:* Two lines in a shared module used by n14 and by the noisy-labels correction family.
2. Alternative to the previous option: give `stage0_diagnostics` a `params` argument that it forwards to `oof_log_predict`, and have n14 pass `n_jobs`. *Cost:* About 5 lines across diagnostics.py:22-38 and n14:337.
3. Compute the label-gap comparison inside the similarity mask without materialising the second n x n array, or state the real memory ceiling in the comment at n14:274 as two n x n arrays plus a boolean. *Cost:* The chunked form is about 15 lines and changes no number.

</details>

## n15 — `noise/n15_clean_vs_noisy_hybrid.py`

**DECIDED.** Paired per-fold delta and NaN exclusion together.

**Done 2026-09-04 under the split audit, user decision this session.** The last bullet below is fixed: `find_best_hybrid` takes `score_on_noisy_labels`, and n15's noisy lineage passes it, so its inner validation folds are scored on the noisy labels and the winning recipe is picked with no clean label anywhere. `tier` and the noise id both carry a `__selnoisy` marker, so the clean-scored and noisy-scored lineages cannot serve each other from cache. The three stored noisy searches per model per dataset rebuild. n07 and n11 deliberately stay on the clean-searched recipe until n15's number is read.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

For each of 3 noise levels and each downstream model, fetches two hybrid blueprints - one searched on clean labels, one searched on noisy training labels and scored on clean validation labels - then rebuilds both per fold on the noisy training labels, trains the model on each, scores both on clean validation labels, and reports the difference as oracle_advantage. Also reports how far apart the two blueprints are.

**Q.** How much of the hybrid's apparent advantage under label noise comes from the blueprint having been searched on clean labels?
**Q.** Do a clean-label search and a noisy-label search pick the same blueprint - same representation set, allocation, importance method and budget?

*On re-read:* The audit said the missing labels-hash assertion means a cached noisy search built under a different spec could be used silently. That cannot happen.

Why:
- One dead model voids the only number the experiment produces. `mean_oracle_advantage` at n15:275 is a plain mean over cells, and a cell whose two arms both scored NaN is not excluded — the check at n15:240 only tests that the score lists are non-empty.
- `c_mean` and `n_mean` (n15:243-244) are means over whatever folds each arm completed. Scores are appended only on success at n15:235.
- The docstring at n15:47-49 promises a labels-hash assertion that does not exist; `noisy_labels_hash` is written into the results at n15:175 and read nowhere.
- ~~The noisy blueprint is still selected by scoring on clean validation labels across all rows, including each outer held-out fold. Both arms carry that equally, so the difference stands, but the docstring at n15:22-23 calls the noisy arm what an honest practitioner could have picked, and it is not.~~ **FIXED 2026-09-04 (split audit, P8-engine).** find_best_hybrid takes `score_on_noisy_labels`, which scores the inner validation folds on the noisy labels too; n15 passes it for the noisy lineage. oracle_advantage now covers the head start from choosing as well as the head start from fitting, so it is a bigger number than any n15 run before this date produced. The noisy lineage is a different search: it rides `__selnoisy` on tier and on the noise id, so it shares no cached baseline, hybrid-eval fold or stored row with the old one.

*Smoke:* ran, NaNs present. exit 0, 'n15 RAN OK in 5.9s', full summary block printed.

<details><summary>Options considered</summary>

1. Compute the summary over cells whose advantage is finite, and record how many cells were dropped and which models they were. *Cost:* About 8 lines at n15:266-280.
2. Replace the difference of means with a paired per-fold delta: pair the two arms fold by fold, average over folds where both scored, and record that fold count. *Cost:* About 10 lines at n15:230-250.
3. Delete the assertion sentence at n15:47-49, or write the assertion it describes by comparing the recorded `noisy_labels_hash` against a hash the search stores. *Cost:* Deleting is one line.

</details>

## sim_hybrid_space — `noise/_similarity_hybrid_space.py`

**DECIDED.** Scaler fix, then persist sim_info so a refused arm is distinguishable from a null result.

PARTIAL · hybrid: hand-rolled · compute: fine · live

Builds one fixed feature matrix per molecular dataset — z-scored, equal-block-scaled [ecfp4_count x0.5 | descriptors minus the MCS-selected drops | 49-D 3D pharmacophore | top-100 fine 2D] — from the similarity study's cached arrays, then re-indexes it onto the experiment's molecules by canonical SMILES. Returns (None, reason) unless every experiment molecule is in the cache.

**Q.** Can the similarity study's winning structural space be handed to a noise experiment as an extra representation, aligned row-for-row to that experiment's molecules?
**Q.** Is every one of the experiment's molecules present in the study cache, so the arm is all-or-nothing rather than partial?
**Q.** Are the MCS pairs that drive the block selection numbered against the molecule list actually in hand?

*On re-read:* Everything the findings file states is true in the code. It omits the most serious problem, which the other audit does name (AUDIT_HYBRID_USAGE_2026_09_03.md:385).

Why:
- n11 and n13 re-standardise the returned matrix, which removes the equal-block scaling and the x0.5 count down-weight, so the arm is not the similarity study's distance space.
- The refusal reason is printed and dropped. n11:444-449 and n13:308-313 print sim_info and never put it in the results dict, so a run with no hybrid_sim rows cannot be told from a run where the arm was refused.
- The 49-feature 3D block is taken by row position with no proof of ordering. load_pharm3d_full returns pharm3d[:n] (creative_hybrids_v2.py:111); sim_logd_spatial.npz holds 1427 rows and only n_built, usr, usrcat, pharm3d, no molecule hash.
- The descriptor drop and the fine-2D pick run again, uncached, on every call. n11 and n13 each pay it separately.

*Smoke:* not run. not attempted, time.

<details><summary>Options considered</summary>

1. Stop re-standardising hybrid_sim. In n11:522 use X_sim_full[train_idx] directly; in n13:567-570 use X_sim_full[fd['train_idx']] and X_sim_full[fd['val_idx']] directly. *Cost:* Two lines.
2. Return the block boundaries from build_similarity_hybrid alongside the matrix, then in n11 and n13 fit a StandardScaler on the training rows and re-apply /sqrt(width) and the [0.5,1,1,1] multiplier per block afterwards. *Cost:* The return contract of build_similarity_hybrid changes, so both callers change.
3. Put sim_info into the saved results. n11 already builds its results dict at :451; n13 has the same shape. *Cost:* One line in each of n11 and n13.

</details>

## handtrace_noise — `noise/_handtrace_noise.py`

**DECIDED.** Delete. It tests local copies of two formulas, so it can never fail when the real ones change.

NO · hybrid: none needed · compute: fine · orphaned

Prints seven arithmetic checks on two percentage formulas — clean/noisy for a lower-is-better metric, noisy/clean for higher-is-better, and (denoised - noisy)/(clean - noisy) — comparing each against a hand-computed number. Both formulas are re-typed inside this file; no noise experiment is imported.

**Q.** Does the retention percentage give a number below 100 for a degraded model under both RMSE and R2?
**Q.** Does the recovery percentage come out positive for an improvement under both metric directions, so the gap's sign carries the metric direction?

Why:
- The file checks its own copies of two formulas, not the experiments'. retention at :32-35 and recovery at :51-52 are local functions.
- The docstring at :4-5 says the opposite: 'feeds them to the REAL functions, and asserts the code returns the hand-computed value'.
- The copies drop every condition the real code refuses on. n09 returns NaN when either score is at or below 1e-10 (n09_lnl_survey.py:530-536).
- The three cited locations are all stale. n09's formula is _score_ratio_pct at n09_lnl_survey.py:519-537, not :518-526.
- **Blocking** `noise/_handtrace_noise.py:32-35` — retention() is a local re-implementation, not the experiment's function.
- **Blocking** `noise/_handtrace_noise.py:51-52` — recovery() is likewise a local copy.

*Smoke:* ran clean. exit 0, seconds, '8/8 hand-trace checks passed'.

<details><summary>Options considered</summary>

1. Delete the file. *Cost:* Zero model fits.
2. Make the three formulas importable, then call them. Lift n09's _score_ratio_pct to module level with lower_is_better as an argument (it is nested inside run() at n09:134 and closes over it today). *Cost:* Three experiment files edited plus this one.
3. Keep the copies and fix only the docstring at :4-5 and the three stale citations at :30 and :50, saying plainly that this walks the sign conventions by hand and does not touch the experiments. *Cost:* Three comment lines.

</details>

## smoke_harness — `noise/_smoke_harness.py`

**DECIDED.** Save the returned dict.

PARTIAL · hybrid: engine, per fold · compute: fine · live

Loads one small real dataset with two or three cheap reps, shrinks the in-process search constants, runs the clean and one noisy hybrid precompute, then calls a single noise experiment's run() end to end and reports whether it crashed. A breakage check, not a measurement.

**Q.** Does experiment nXX's run() complete end to end on a small real dataset without raising?
**Q.** Do the clean and the noisy-label hybrid precomputes both produce a blueprint the experiment can read out of data['precomputed_searches'] and data['precomputed_searches_noisy']?

*On re-read:* Two things in the findings are wrong. First, they say results go to /tmp/noise_smoke 'via the experiment's own save path (:223-224)'.

Why:
- The harness never saves what the experiment returns, so no smoke run can check the results JSON.
- The documented entry point is dead. :9 says 'python _smoke_harness.py n05'; n05 was cut (noise_helpers.py:57-58) and is absent from the map at noise_helpers.py:181-192, so _import_experiment raises KeyError at noise_helpers.py:194.
- tiny_config() at :80-91 reads as the search grid and is not one. precompute_hybrid_searches never touches its config argument at all — the name appears only in its signature and docstring across _shared.py:3835-3995.
- tiny_config() names allocations=['greedy'] at :86. greedy is cut, and the key is never read.

*Smoke:* ran, NaNs present. Works, and it is the only way I found to run the noise experiments locally without the segfault: it sets OMP/OPENBLAS/MKL/VECLIB/NUMEXPR to 1 and KMP_DUPLICATE_LIB_OK before importing numpy, shrinks FULL_BUDGETS to [25] and forces DEFAULT_INTERNAL_EVALUATORS=['rf'].

<details><summary>Options considered</summary>

1. Save the returned dict. After mod.run(...) at :242, copy the two lines noise_helpers.py uses at :474-475: exp_full_name = _experiment_full_name(args.exp), then save_experiment_results(result, out_dir, exp_full_name, ds). *Cost:* Two lines plus one import.
2. Cut tiny_config() down to {'n_splits': 2}, which is the only key anything reads, and say in a comment that budget and evaluator shrinking happens at :41 and :54-59 instead. *Cost:* Six lines removed.
3. Make the noisy precompute and n15 agree. Either build every spec in noise_helpers.noisy_search_specs(data['task_type']) at :146 instead of the single 0.20 spec, or pass run_kwargs['noise_specs'] = smoke_noise_specs(...) unconditionally rather than only under KIRBY_SMOKE_N at :234-239. *Cost:* First form triples the noisy precompute — three find_best_hybrid calls instead of one, each over 1 dataset x 2 search folds x budget [25] x 2 allocation methods x 1 importance method x 1 internal evaluator, plus baselines across the full default model set.

</details>

## smoke_noisy_search — `noise/_smoke_noisy_search.py`

**DECIDED.** Add cache=True, then run it once so checks 6 to 9 are reachable at all.

PARTIAL · hybrid: engine, per fold · compute: fine · orphaned

Nine checks on synthetic 400-row data with two random reps: that a NoiseSpec draw is reproducible, that every noise condition delivers the same root-mean-square amount at one level, that a noisy find_best_hybrid run does not return the clean numbers, that its cache entries are namespaced by the noise id, that more noise gives a worse baseline, and that label_noise plus rep_factories is refused.

**Q.** Does one NoiseSpec produce a byte-identical noise draw every time, and do different conditions at one level deliver the same amount of noise?
**Q.** Does a find_best_hybrid run with label_noise return different baselines and a different cache identity from the clean run, rather than silently serving the clean results?
**Q.** Does the engine refuse label_noise together with rep_factories, which would fine-tune the representations on clean labels?

Why:
- The search helper at :63-70 does not pass cache=True, and find_best_hybrid.py:5919 defaults it to False. So no result database and no search pickle is written, and the silent failure this file exists to catch cannot happen in this run.
- Check 6 at :133 opens $KIRBY_CACHE_DIR/result_db/noisysmoke.db, which those runs never create. The smoke run failed there with sqlite3.OperationalError, exit 1, and checks 7, 8 and 9 never ran.
- Checks 4 and 5 do test something real, and it is worth knowing which. The per-representation baseline store is switched on by dataset_name and metric alone, independent of cache (find_best_hybrid.py:4143-4147), and its filename carries the noise id (find_best_hybrid.py:186-187).
- Check 8 at :151-155 asserts a strict ordering of three random-forest scores at noise 0.1, 0.6 and 1.0, with no tolerance and one seed. It has never run, so whether it is stable is unknown.
- **Blocking** `noise/_smoke_noisy_search.py:63-70` — search() omits cache=True, so find_best_hybrid runs with cache=False.
- **Blocking** `noise/_smoke_noisy_search.py:133` — sqlite3.connect on $KIRBY_CACHE_DIR/result_db/noisysmoke.db, a file that the cache=False runs above never create, then SELECT DISTINCT tier FROM results.

*Smoke:* FAILED. exit 1 at check 6 of 6.

<details><summary>Options considered</summary>

1. Add cache=True to the search helper at :64-70. Everything find_best_hybrid demands under cache=True is already there: dataset_name is DS at :67, tier is 'general' at :63, metric is the string 'r2' at :66, and there is no custom splitter and no rep_factories (find_best_hybrid.py:6442-6473). *Cost:* One line.
2. Wrap check 6 so a missing database is reported as a FAIL rather than raised, and move the rmtree at :173 into a finally block. *Cost:* About six lines.
3. Loosen check 8 at :151-155 to require only that the ends separate, scores[0] > scores[2], leaving the middle level free to tie. *Cost:* One line.

</details>


---

# Similarity, s01–s13 — 15 files

## s01 — `similarity/s01_similarity_baseline.py`

**DECIDED.** Hoist the pair sample, record the violation-rate reference, fix the dense-bound comment.

YES · hybrid: none needed · compute: fine · dispatched

For each base representation it builds a standardized-Euclidean distance matrix over the whole dataset and reports three numbers: the Spearman correlation between pairwise distance and pairwise |activity difference| (called SAC in the code), the fraction of the 5 nearest neighbours whose activity difference exceeds the median pair difference, and SALI. A 200-draw label-permutation test gives each SAC a p-value.

**Q.** For each base representation, how strongly does distance between two molecules track the difference in their activity?
**Q.** Is that correlation distinguishable from what a random label assignment gives?
**Q.** How often do the five nearest neighbours of a molecule have a very different activity?

Why:
- The sampled pairs are re-drawn 201 times per representation. similarity_helpers.py:81 calls np.triu_indices(n, k=1) on every invocation, and s01_similarity_baseline.py:122-125 calls it 200 more times for the label-permutation test.
- compute_sali holds five n-by-n float64 matrices at once (similarity_helpers.py:98-108: sim, delta_y, denom, sali_matrix, plus the caller's D). At the switch point n=15000 that is 9 GB.
- violation_rate compares each neighbour pair against the median activity difference over all pairs (similarity_helpers.py:145-151), so a distance carrying no information scores about 0.5 for a regression label. That reference is computed inside the helper and thrown away, so a reader of violation_rate alone cannot tell a good representation from a useless one.

*Smoke:* ran clean. exit 0, 31s total / s01 2.3s.

<details><summary>Options considered</summary>

1. Draw the 5000 pairs once in _run_per_rep and pass them into similarity_activity_correlation, instead of letting the helper re-draw them on each of the 201 calls. *Cost:* Zero model fits change.
2. Write the no-information violation rate into the JSON next to violation_rate. For a regression label it is 0.5 by construction; for a classification label it is the fraction of all pairs in different classes. *Cost:* Zero model fits.
3. Leave S01_MAX_DENSE_N at 15000 and correct the comment to say 9 GB peak inside compute_sali, not 7 GB per matrix. *Cost:* Zero.

</details>

## s02 — `similarity/s02_hybrid_similarity.py`

**DECIDED.** Three one-line fixes: the weighted-versus-weighted delta, the recorded blueprint, the lineage.

**Still open:** TBD with the objective question. Also whether budget is swept rather than fixed at 100.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Builds one hybrid through the real engine at budget 100 for downstream model knn, then compares the Spearman correlation between distance and activity difference for the hybrid against the same quantity for every single representation. It does this twice: once on the whole dataset (the headline keys best_sac and sac_delta_from_best_rep), and once with a 5-fold group-aware split where the importance weights are refit on train rows and the correlation is measured on the held-out rows.

**Q.** Does an importance-weighted distance over the hybrid's selected features correlate better with activity difference than the best single representation's distance?
**Q.** Which importance method makes the best weighting for that distance?

Why:
- The headline delta subtracts an unweighted single representation from an importance-weighted hybrid. Line 411 is sac_weighted - best_rep_sac.
- The recipe written to the JSON is not the recipe that made the numbers. Line 413 reads search.best_by_model['knn']['blueprint'], which _shared.py:4756-4790 documents as the best across all budgets.
- s02 asks for the knn hybrid from the general lineage; s04 asks for the same name from the distance lineage. s02:324-328 omits distance=True, so the pool searches FULL_BUDGETS with INTERNAL_EVALUATORS.
- s02:324 omits data=data. _shared.py:4459-4463 only reads data['precomputed_searches'] when data is passed, so the pool that run_all just filled in memory (similarity_helpers.py:945-958) is bypassed and the search is loaded from disk, or rebuilt if the disk key misses.
- **Also caught by the hybrid-usage audit** — s02 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran clean. exit 0, s02 done in 7.0s.

<details><summary>Options considered</summary>

1. At line 411 subtract best_rep_sac_weighted instead of best_rep_sac, and write both deltas so the weighted-vs-weighted and weighted-vs-unweighted numbers are separable. Point s09_cross_dataset_summary.py:236-240 at the new key. *Cost:* Zero model fits, no re-run.
2. At line 413 write _meta['best_config']['blueprint'] instead of search.best_by_model['knn']['blueprint']. _meta is already bound at line 324. *Cost:* Zero model fits.
3. At line 324 pass data=data. *Cost:* Zero model fits.

</details>

## s03 — `similarity/s03_feature_concentration.py`

**DECIDED.** Fix the density denominator and the JSON keys, then re-run. The 22 files on disk answer none of its questions.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

For each of 3 downstream models x 8 budgets it builds a hybrid on full data, then for each base representation compares the mutual information with the label of four feature sets of matched size: the full representation, the features the hybrid selected, a random subset of the same size, and a PCA projection to the same dimension. It also reports effective rank of each set and the Gini coefficient of the importance scores.

**Q.** Do the features the hybrid selects from a representation carry more mutual information with the label than a random subset of the same size?
**Q.** Do they carry more than a PCA projection to the same number of dimensions?
**Q.** How unevenly is importance spread across a representation's features?

*On re-read:* Two things. First, the audit's section 4 item 2 is applied: lines 155-163 now read selected_indices off the feature_info the rebuild returned.

Why:
- All 22 stored s03 files contain mi_full and effective_rank_full and nothing else. The selected-versus-random-versus-PCA comparison, which is the whole experiment, is missing from every representation of every cell.
- mi_density_ratio divides two numbers measured over different numbers of features. Line 212 divides mi_full by the representation's full width, e.g.
- The JSON keys are Python tuples stringified. Line 231 keys results['per_rep'] by (downstream_model, budget), and convert_numpy at _shared.py:4891 writes str(k), so the key on disk is the literal string "('rf', 50)".
- downstream_models is hardcoded to ['rf','lightgbm','mlp'] at line 117, dropping svm and xgboost from DOWNSTREAM_MODELS. The reason given at line 114-115 is that the full set would explode result size with little gain, which is a theory about the outcome.
- **Blocking** `similarity/s03_feature_concentration.py:212` — mi_density_full divides mi_full by n_features (e.g.
- **Also caught by the hybrid-usage audit** — s03 asks the pool for a hybrid without passing objective='similarity'.
- **Also caught by the hybrid-usage audit** — s03 now reads the selected features off the feature_info the full-data rebuild returns instead of search.fitted_results, which was empty, so the selected-versus-random-versus-PCA comparison had been skipped everywhere.

*Smoke:* ran clean. exit 0, s03 done in 9.1s.

<details><summary>Options considered</summary>

1. Re-run s03. This is not optional if you want the experiment's answer at all. *Cost:* 24 calls to get_fitted_pool_hybrid per dataset (3 models x 8 budgets), each one a full-data create_hybrid rebuild that re-scores all 10 representations and recomputes importance.
2. Before that re-run, divide mi_full by the number of components actually summed, min(50, rows//5, n_features), and store that count next to it. *Cost:* Zero extra model fits.
3. Key results['per_rep'] with f"{downstream_model}|{budget}" instead of the tuple. *Cost:* Zero model fits.

</details>

## s04 — `similarity/s04_activity_cliffs.py`

**DECIDED.** Build ECFP4 from the SMILES so the cliff rule stops depending on the pool, and keep per-fold cliff scores.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Labels every molecule as a cliff compound or not, using ECFP4 Tanimoto >= 0.55 paired with |activity difference| > 2 x MAD(y). Then runs group-aware CV and, on each fold, scores a 5-nearest-neighbour prediction from every single representation (unweighted and importance-weighted), from a per-fold hybrid (unweighted and importance-weighted), and a random forest on the best single representation — reporting the dataset metric and RMSE separately on the cliff rows and the non-cliff rows of the held-out fold.

**Q.** On molecules that sit on activity cliffs, does a 5-nearest-neighbour prediction from the hybrid's feature space beat one from any single base representation?
**Q.** Does weighting the distance by feature importance help on cliffs specifically?
**Q.** How much worse is every method on cliff compounds than on non-cliff compounds?

*On re-read:* The findings describe a saved payload with cliff_definition, per_rep score_cliff, hybrid.unweighted and honest_rescore. Nothing like that exists on disk.

Why:
- s04 has no measured output at all. Every one of the 23 stored files is the molecular-only skip stub.
- The only held-out comparison in the file is not about cliffs. Line 219 pushes _score_all, the metric over every compound in the fold, into fold_scores_pos.
- s04 aborts with {'error': 'ecfp4 not available'} at line 87 whenever ecfp4 is not among the modelling representations. Of the eight molecular datasets in DATASET_REPS, adme_hlm is the only one whose 10-representation list lacks it (complexity_helpers.py:216-221, cut 2026-08-20).
- score_cliff averages over whichever folds happened to hold more than one cliff compound (line 227). The number of contributing folds is not written to the JSON, so a mean over two folds is indistinguishable from a mean over five.
- **Also caught by the hybrid-usage audit** — s04 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran, NaNs present. exit 0, s04 done in 1.7s, file written.

<details><summary>Options considered</summary>

1. Compute the ECFP4 fingerprint from data['smiles'] when ecfp4 is not in the representation pool, instead of returning the error at line 87. *Cost:* Zero model fits.
2. Record per-fold cliff scores alongside the overall ones, and build the honest block from the cliff list rather than fold_scores_pos. *Cost:* Zero extra model fits — the cliff scores are already computed inside _evaluate_condition at lines 227-236 and then averaged away.
3. Leave the honest block as the overall metric and state that the cliff comparison is same-fold only. *Cost:* Zero.

</details>

## s05 — `similarity/s05_knn_prediction.py`

**DECIDED.** budget_override, hoisted build, search-only helper, delta_mean_of_folds and task_type in one pass, then re-run 22 datasets.

**Still open:** Whether fixed_ref keeps its in-sample baseline.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Runs its own 5-fold group-aware CV of k-nearest-neighbour regression/classification on a per-fold hybrid rebuilt from a distance-lineage blueprint, at 6 budgets x 2 weighting conditions x k in {5,10}, against per-representation kNN baselines at the same k values. Writes a post-hoc max, a pre-registered fixed cell, and a leave-one-fold-out re-score block.

**Q.** Does kNN in the hybrid feature space beat kNN in each single representation's space, on the same folds?
**Q.** Does scaling the hybrid's standardized features by sqrt(importance) improve kNN over the unweighted hybrid?
**Q.** How does the kNN score move with the feature budget (10 to 500)?

Why:
- A budget row can hold a hybrid built at a different budget. get_best_for_downstream returns the best config across all budgets when the asked-for budget has no finite score (_shared.py:4785-4800), build_fold_hybrid builds at blueprint.budget (_shared.py:3689), and the row is filed under the budget that was asked for (s05:244).
- That is not hypothetical. Of the 22 stored files in tests/results/similarity/s05_knn_prediction, 18 hold at least one budget row whose evaluator, allocation method and score match another row to 16 digits.
- The delta labelled HONEST is unfair to the hybrid, in the hybrid's disfavour. fixed_ref is one pre-chosen cell (s05:433-440).
- Sixty of the 120 per-fold hybrid builds per dataset repeat a build already done. Weighting is applied after the build at s05:204-212, so hybrid_weighted and hybrid_unweighted reach build_fold_hybrid with identical arguments (s05:176-186).
- **Also caught by the hybrid-usage audit** — s05 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran clean. exit 0, s05 done in 3.5s.

<details><summary>Options considered</summary>

1. Add budget_override=budget to the build_fold_hybrid call at s05:176-186. The parameter already exists and does exactly this (_shared.py:3565, 3688-3689); c15 uses it for the same purpose. *Cost:* Build count per dataset is unchanged at 120 create_hybrid calls.
2. Instead, leave the build alone and store blueprint.budget in the row beside the requested budget at s05:238-247. *Cost:* Zero change to any computed number, but the field only appears in files written after the change, so all 22 still re-run to carry it.
3. Hoist the build out of the weighted/unweighted loop: build once per (budget, k, fold), then score both conditions off the same X_tr_h. *Cost:* 120 create_hybrid calls per dataset become 60.

</details>

## s06 — `similarity/s06_applicability_domain.py`

**DECIDED.** Find the NaN in the weighted arms, then add per-representation arms to both blocks.

PARTIAL · hybrid: engine, per fold · compute: fine · dispatched

Per fold, builds a hybrid on the training rows, fits a random forest on it and on the concatenation of all representations, then correlates each validation molecule's mean distance to its 5 nearest training molecules with that molecule's absolute residual, in four spaces (concat, weighted concat, hybrid, weighted hybrid) plus every single representation. Also writes a distance-decile error table and a split-conformal interval comparison of concat against hybrid.

**Q.** Does distance to the training set in the hybrid space rank prediction error better than distance in the concatenation of all representations?
**Q.** Does the same distance rank error better than distance in each single representation's space?
**Q.** Do conformal intervals built on hybrid features cover at 90% with a smaller mean width than intervals built on the concatenation?

Why:
- Every importance-weighted arm came back NaN in today's smoke run, in both folds: concat_weighted, hybrid_weighted, and the same two in the neutral_model block. s09 then reports AD_corr_delta as nan.
- The comparator for the headline is the concatenation of every representation (s06:132), and s09:487-497 forms ad_corr_delta as hybrid_weighted minus concat. Concatenation baselines are not admissible.
- Per-representation correlations are scored against errors from the random forest fitted on the hybrid features (s06:408-414 read all_errors). The errors from the model that never sees the hybrid space are computed in the same fold loop as all_errors_neutral, but they are never paired with the per-representation distances.
- The conformal comparison holds two arms, concat and hybrid (s06:500-503). No individual representation enters it, and only budget 100 is ever asked for (s06:63, s06:136).
- **Also caught by the hybrid-usage audit** — s06 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran, NaNs present. exit 0, s06 done in 6.2s, file written (result_status=ok), 24 NaN + 3 null.

<details><summary>Options considered</summary>

1. Before anything else, record how many entries of each weight vector are above zero — one integer per fold for w and for w_concat — and print it. Then re-run one dataset. *Cost:* One dataset, 10 create_hybrid calls and 20 random forest fits.
2. Add every representation as its own arm to the conformal loop at s06:500-503, and pair the per-representation distances with all_errors_neutral as well as all_errors at s06:408-414. *Cost:* Random forest fits per dataset go from 20 to 70: the conformal loop goes from 2 arms to 12 across 5 folds.
3. Pass budget=budget to get_best_for_downstream at s06:144, so the rebuilt width is the one the pool was asked for. *Cost:* 10 hybrid builds and 20 random forest fits per dataset, unchanged in count.

</details>

## s07 — `similarity/s07_noise_bridge.py`

**YOURS.** Delete, or rewrite the held-out arm. 14 of 21 stored files carry no number and 4 more sit at exactly 1.0 for every arm. My lean: delete unless this question is going in the paper.

NO · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Defines ground-truth activity cliffs from ECFP4 Tanimoto (molecular) or heavy-chain edit distance (antibody), then flags the 10% of molecules whose 5 nearest neighbours disagree most in label, and reports what share of those flagged molecules are cliffs, in the hybrid space, the importance-weighted hybrid space, and each single representation's space. Repeats the same statistic under two fold schemes and hands both to the leave-one-fold-out re-scorer.

**Q.** When a k-nearest-neighbour label-consistency filter flags molecules as noisy, does the hybrid space put fewer genuine activity cliffs among the flagged set than any single representation does?
**Q.** Does scaling the hybrid's features by sqrt(importance) reduce that cliff share further?
**Q.** Does the answer survive holding out folds, under one-fold-per-scaffold and under the suite's 5-fold group splitter?

Why:
- The arm called held-out is not held out. hybrid_embeddings holds X_hybrid and X_hybrid_w, both built from every row's labels at s07:346-353 and s07:410.
- The statistic does not discriminate on any dataset that has run. 21 files sit in tests/results/similarity/s07_noise_bridge.
- No molecular dataset has ever produced an s07 result. All 21 stored files are antibody or reaction sets.
- Ground-truth cliffs on molecular data are computed from reps['ecfp4'] (s07:300-301), which is also one of the scored representations and can be selected into the hybrid. The docstring at s07:8-11 calls the definition representation-independent.
- **Blocking** `similarity/s07_noise_bridge.py:388` — hybrid_embeddings holds X_hybrid (and X_hybrid_w), both produced by get_fitted_pool_hybrid at line 348 with all labels.
- **Blocking** `similarity/s07_noise_bridge.py:221` — _leave_one_group_out_splits yields one fold per distinct group value, and _knn_noise_contamination_cv rebuilds a full train x train distance matrix and argsort inside every fold, for all 12 arms.
- **Also caught by the hybrid-usage audit** — s07 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran, NaNs present. exit 0, s07 done in 0.6s.

<details><summary>Options considered</summary>

1. Delete the file. The sqrt(importance)-weighted distance question is asked better by n12_importance_weighted_distance.py, which injects noise so it knows the ground truth. *Cost:* Zero fits.
2. Rebuild the hybrid inside each fold with build_fold_hybrid on the training rows only, and keep the shared 5-fold scheme, dropping the one-fold-per-group block. The reason for dropping it is stated in the code itself at s07:88-99: a fold holding one or two molecules can only return 0.0, 1.0 or NaN. *Cost:* 5 create_hybrid calls per dataset for the hybrid arms, plus 10 train-by-train distance matrices and argsorts per fold for the representations.
3. Change nothing in the measurement and instead record, in both honest_rescore blocks, that the candidate matrices were fitted on every row. Then stop s09 reading their delta_mean_of_folds as an honest number. *Cost:* No fits.

</details>

## s08 — `similarity/s08_optimal_dimension.py`

**DECIDED.** budget_override and a kNN model factory together. Without both the file answers neither of its questions.

**Still open:** Which model scores accept-or-reject before the 16 datasets re-run.

NO · hybrid: engine, per fold · compute: heavy · dispatched

For each of six feature budgets it builds a full-data hybrid and records its intrinsic dimension, its distance-activity Spearman, its 5-nearest-neighbour label violation rate and its rank correlation with the all-representation concatenation, then runs a 5-fold CV to get per-model prediction scores and a held-out distance-activity Spearman. Compares the budget that maximises the correlation against the budget that optimises prediction, and re-scores both against per-representation baselines.

**Q.** Does the feature budget that maximises the distance-activity Spearman differ from the budget that optimises the downstream prediction score?
**Q.** At a matched budget, does the hybrid's distance-activity Spearman beat every single representation's, on held-out folds?
**Q.** Does scaling the held-out distance by the train-fold importance weights raise that correlation above the unweighted version?

Why:
- The dimension sweep runs at one dimension in its cross-validated half. bp at s08:200 is search.best_by_model['knn']['blueprint'], which find_best_hybrid builds as the maximum over every config for kNN across all budgets (find_best_hybrid.py:5640-5645).
- The stored files show it. abcov_kd carries rf 0.36892170360844895 at both budget rows although actual_dim is 50 and 100.
- kNN is never fitted, and kNN is the model the experiment exists to compare against random forest. DOWNSTREAM_MODELS is ['rf','xgboost','svm','mlp','lightgbm'] (_shared.py:336) and s08:116 passes it unchanged.
- Prediction scores from two different fold definitions are averaged into one per-budget curve. s08:215-220 takes cached fold_scores off the search when its budget matches the loop's; every other model is fitted on this experiment's folds at s08:288-296.
- **Blocking** `similarity/s08_optimal_dimension.py:200` — bp = search.best_by_model['knn']['blueprint'] is the best config across all budgets, not the config at the loop's budget.
- **Blocking** `similarity/s08_optimal_dimension.py:116` — model_factories = get_model_factories(DOWNSTREAM_MODELS, ...) and DOWNSTREAM_MODELS (_shared.py:336) contains no 'knn'.
- **Also caught by the hybrid-usage audit** — s08 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran clean. exit 0, s08 done in 3.9s, file written.

<details><summary>Options considered</summary>

1. Add budget_override=budget to the build_fold_hybrid call at s08:250-253. The parameter exists for exactly this (_shared.py:3565, 3688-3689). *Cost:* Build count per dataset unchanged: 6 full-data plus 30 per-fold create_hybrid calls.
2. Pass DOWNSTREAM_MODELS + ['knn'] at s08:116. 'knn' is a real entry in DEFAULT_MODELS (find_best_hybrid.py:2167-2174), so get_model_factories will build it. *Cost:* 6 budgets by 5 folds = 30 extra kNN fits per dataset.
3. Delete the cached-fold-score reuse at s08:215-220 and fit every model fresh in every fold. *Cost:* At most one budget per model can be served from cache, so at most 5 model-and-budget pairs by 5 folds = 25 extra fits per dataset.

</details>

## s09 — `similarity/s09_cross_dataset_summary.py`

**DECIDED.** Three one-line repairs plus an individual-representation comparator to replace the concatenation. Then point the submit script at this module.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Reads the saved JSON of s02, s04, s05, s06, s07 and s08 for each dataset and pulls out one delta per comparison, each labelled with the arm it came from and the key it was read from. Fits nothing; it prints the per-dataset values, refuses to average them, and runs a Wilcoxon signed-rank test over the per-dataset values when at least 3 datasets contributed.

**Q.** Does the hybrid beat the best single representation on the same sign, in the same direction, on every dataset, for each of the six similarity comparisons s02-s08 measured?
**Q.** For each comparison, does the held-out arm (winner re-picked on the other folds) agree with the selection-optimistic arm (winner picked on the scores that are reported)?
**Q.** Is the pattern of per-dataset signs unlikely under a Wilcoxon signed-rank test?

Why:
- The applicability-domain number is the hybrid minus a concatenation of all ten representations (s09:481), and concatenation baselines are banned as comparators.
- The first difference read from s02 goes into the aggregated series with no check that it is a number (s09:246), so one dataset missing best_individual_rep or best_sac turns that whole series' Wilcoxon p into NaN.
- honest_rescore is handed bool(block.get('lower_is_better')) at s09:185; hr_direction exists at _shared.py:843 precisely because a missing key reads as higher-is-better and inverts every RMSE difference. s04 does store the flag today (s04:506), so this is latent.
- The output name is fixed at results.json (s09:778 passes no dataset), so a second run over a different --datasets list overwrites the first.

*Smoke:* ran, NaNs present. exit 0.

<details><summary>Options considered</summary>

1. Three one-line repairs, in place: wrap the s02 difference at s09:246 in the file's own _is_number check (every other series already has one); swap bool(block.get('lower_is_better')) at s09:185 for _shared.hr_direction(block), which is already inside a try that turns a raise into a stored reason; pass the dataset list into the output name at s09:778 so a second run does not overwrite the first. *Cost:* Zero model fits.
2. Read the best individual-representation correlation out of s06's own stored numbers and make that the aggregated applicability-domain difference. In the block at s09:475-492, corr_result already carries one entry per representation; take the best of those instead of corrs.get('concat'). *Cost:* Zero model fits and no re-run of s06 — the numbers are already in its result files.
3. Keep the concatenation difference as the aggregated number and add the individual-representation one beside it as a second series with its own label and source string. *Cost:* Same as above: zero fits, seconds to re-run, nothing stale.

</details>

## s10 — `similarity/s10_importance_downstream.py`

**DECIDED.** Pass budget 100 explicitly and store it, then the four repairs.

**Still open:** Whether budget becomes a reported axis, which is a separate server job.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

For each base representation and for a per-fold hybrid, multiplies every feature column by the square root of its normalised LightGBM-gain importance and scores four downstream estimators weighted against unweighted: kNN at k in {5,10,20}, a Gaussian process with an RBF kernel, an SVR with an RBF kernel, and a random forest fed four extra similarity descriptors (neighbour label mean and spread, nearest and mean distance). Splits are group-aware, importance is computed on training rows only, and every delta is sign-oriented so positive means improvement.

**Q.** Does scaling each feature by the square root of its importance improve a kNN, a Gaussian process, or an SVR, relative to the same estimator on the same unscaled features?
**Q.** Does appending four similarity descriptors computed from training-row neighbours improve a random forest, and does computing those descriptors in the importance-scaled space improve it further?
**Q.** Do the answers differ between an individual base representation and the hybrid?

Why:
- Under a full similarity run the hybrid arm of all four tasks is 10 features wide, not 100 (s10:956). The file's own command-line default is 100 (s10:1213).
- The budget the blueprint resolved to is printed (s10:979) and never stored, so a result file cannot say how wide its hybrid was.
- A weighted arm that fails on every fold is stored as NaN with no reason attached (s10:288, 457, 607, 789). The hybrid-build failures beside them do record the exception type and message (s10:107).
- cv_split is called with mode=mode at s10:1024. That argument is ignored and raises a DeprecationWarning (_shared.py:1174-1179), once per run.
- **Also caught by the hybrid-usage audit** — s10 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran clean. exit 0, s10 done in 13.8s (task 1 1.5s, task 2 GP 6.0s, task 3 SVR 0.2s, task 4 q-RASAR 6.1s), file written.

<details><summary>Options considered</summary>

1. Pass the file's own default explicitly: budget = kwargs.get('budget', 100) at s10:956, and store it next to hybrid_status at s10:1002. *Cost:* Zero extra fits.
2. Pass budget=None at s10:956 and store the budget the blueprint came back with. *Cost:* Zero extra fits, same 30 builds per dataset.
3. Loop the four tasks over every budget in DISTANCE_BUDGETS = [10, 25, 50, 100, 250, 500] and store one hybrid block per budget. *Cost:* Six times the hybrid work per dataset: 180 create_hybrid builds instead of 30, and 390 downstream fits instead of 65.

</details>

## s11 — `similarity/s11_bayesian_optimization.py`

**DECIDED.** Route both builds through build_fold_hybrid with groups, the model and the metric. Refuse the hybrid conditions when there is no blueprint and record that. Then the four repairs.

PARTIAL · hybrid: should use engine, does not · compute: heavy · dispatched

Simulates library screening: takes a group-aware 5% seed set and an 80% pool, builds a feature space on the seed rows only, then runs 100 Expected-Improvement iterations picking one pool molecule at a time and revealing its label. Eleven feature spaces compete, including a 50-feature KIRBy hybrid, PCA to 50, a random Gaussian projection to 50, raw ECFP4 with an RBF kernel, raw ECFP4 with a Tanimoto kernel, the best single representation, and random acquisition.

**Q.** Does Expected-Improvement Bayesian optimisation find good molecules in fewer revealed labels when the surrogate lives in a 50-feature KIRBy hybrid space than in raw 2048-dimensional ECFP4, a 50-component PCA, a 50-component random projection, the best single representation, or random acquisition?
**Q.** Does rebuilding the hybrid every 25 iterations on the labels revealed so far beat building it once on the seed set (hybrid_50d vs hybrid_50d_refit)?
**Q.** In hybrid space, does an NGBoost, MC-dropout BNN or Gauche Tanimoto-GP surrogate beat the sklearn RBF GP?

*On re-read:* The shape mismatch at s11:1457 and s11:1495 is real code but cannot fire today, so it is not a blocker. load_data returns 'augmentations': None at every one of its seven return points (_shared.py:2860, 2940, 2987, 3105, 3174, 3242, 3276), and the augment flag is parsed at _shared.py:2791 and never read inside _load_data_impl.

Why:
- create_hybrid is called with no groups at s11:696 and s11:1447, so the seed rows are split by np.random.RandomState(42).permutation (hybrid.py:4554). That split accepts or rejects every feature.
- Neither call passes a model or a metric, so internal_evaluator='match' resolves to lightgbm (hybrid.py:4501) and features are accepted on r2 rather than on the dataset's metric. The estimator every hybrid condition actually uses is a Gaussian process.
- When the blueprint search raises, bp becomes None and nothing is recorded (s11:893). Every hybrid condition then builds under a hand-picked entropy_1.0 / lightgbm_gain / match recipe (s11:682-685, s11:1429-1431), and no stored field distinguishes that from a searched recipe.
- auc_mean uses np.mean at s11:803. One failed seed appends [nan] at s11:1216, its area is float(c[0]) = NaN at s11:782, and the headline for that condition is NaN even when the other two seeds ran.
- **Blocking** `similarity/s11_bayesian_optimization.py:1457` — _run_bo_with_rebuild transforms the pool with apply_feature_selection alone, at :1457 and again after each rebuild at :1495, but create_hybrid hstacks the selected augmentation columns onto the returned matrix (hybrid.py:4942, 4993, 5031, 5069).

*Smoke:* ran clean. exit 0, 55s total / s11 25.2s.

<details><summary>Options considered</summary>

1. Smallest change that closes the random split and the wrong accept metric: at s11:696 and s11:1447, add groups=groups[seed_idx] (and groups[lab_idx_arr] at the rebuild), model=GaussianProcessRegressor(...), metric_fn and lower_is_better. Both are already in scope in run() at s11:895. *Cost:* No change in the number of builds: about 24 create_hybrid builds and 2250 surrogate fits per dataset either way.
2. Route both hybrid builds through _shared.build_fold_hybrid instead of bare create_hybrid. It takes (train_reps, val_reps, y_train, blueprint) plus groups, model, metric_fn, lower_is_better and budget_override (_shared.py:3559-3566). *Cost:* Same build count, about 24 per dataset.
3. Decide what happens when the blueprint search fails. Either store a hybrid_status string the way s10:1002 does and keep running under the hand-picked recipe, or skip every hybrid condition with a recorded reason, the way s11 already skips a missing surrogate package at s11:1096-1099. *Cost:* Zero fits either way, or the whole hybrid arm's fits removed on a failed search.

</details>

## s12 — `similarity/s12_cross_domain_cliffs.py`

**DECIDED.** Fix the scanned-set contamination, write the recipe into the result, do the housekeeping pass.

**Still open:** TBD with the objective question.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Marks each compound as an activity cliff or not (structurally close pair, large activity gap) using standardised Euclidean distance on every representation concatenated, then scores each single representation and the hybrid separately on the cliff rows and the non-cliff rows of each held-out fold, for five downstream models. Reports the hybrid-minus-best-single-representation difference on the cliff rows, both as a same-fold argmax and as a leave-one-fold-out re-pick.

**Q.** Does the hybrid predict the activity of compounds that sit on an activity cliff better than the best single base representation does, per downstream model?
**Q.** Does that advantage survive re-picking the winning base representation on the other folds instead of on the fold being scored?
**Q.** What fraction of the dataset are cliff compounds once 'structurally similar' is defined as the top few per cent most-similar pairs rather than an absolute similarity cutoff?

*On re-read:* The subsample cap is real but far smaller than the audit says. It claims herg_ki has 7772 rows.

Why:
- Rows past the 5000th are never tested for cliff membership, and s12:326 and s12:391 then count every one of them as a non-cliff row.
- The recipe that won is thrown away: s12:270-292 resolves the allocation method, importance method, internal evaluator and budget per model, and the saved result at s12:408-417 records none of them.
- --sim-threshold is parsed at s12:527 and never sets a cutoff, because the default at s12:213 is a list of percentiles and that sends the code down the sweep path at s12:134-147.
- When zero cliffs are found the widening loop at s12:231-241 can finish at the 30% percentile and record that as if it were the planned setting, because the flag is only set on the 5% break.
- **Also caught by the hybrid-usage audit** — s12 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran clean. exit 0, s12 done in 1.8s, file written.

<details><summary>Options considered</summary>

1. Keep the 5000-row cap, but stop treating unscanned rows as non-cliff rows. Return the scanned row numbers alongside the cliff row numbers, and require membership of that scanned set at s12:326 and s12:391. *Cost:* Two lines in identify_cliff_pairs_euclidean plus two in run.
2. Scan every row: pass max_compounds=len(labels) from s12:221. *Cost:* One line.
3. Write the recipe into the result. Add the four blueprint fields from bp_by_model, per model, to the results dict at s12:408. *Cost:* About five lines.

</details>

## s13 — `similarity/s13_clustering_retrieval.py`

**DECIDED.** Pad the fold lists first — it can already be producing a wrong delta. Then the singleton-cluster fix, the dimension match, and drop the discarded build.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Under group-aware CV, builds three kinds of feature space per fold on the train rows only (each single representation, a PCA of everything concatenated, and the hybrid), fits k-means and a nearest-neighbour index on train, then measures on the held-out rows how tightly each cluster's activities agree and how close the five retrieved train neighbours' activities are. Averages per fold and reports hybrid versus best single representation on both axes, with a leave-one-fold-out re-pick.

**Q.** Do k-means clusters built in the hybrid space group held-out compounds with more similar activities than clusters built in any single base representation's space?
**Q.** Are the five nearest training neighbours of a held-out compound closer in activity when the distance is measured in the hybrid space than in any single base representation's space?
**Q.** Does weighting the hybrid's features by their lightgbm gain improve that neighbour retrieval further?

Why:
- A fold that produces nothing for one space silently shifts every later fold of that space by one position. s13:288-295 appends only on success and s13:283 skips the whole hybrid space when its build raises.
- The dimension control is matched to nothing. pca_dim at s13:259 uses the kwarg budget, which is 100 and is never set from the blueprint, while the hybrid's width is blueprint.budget drawn from [10,25,50,100,250,500].
- The dimension control never enters a comparison anyway. top3_retrieval and top3_clustering filter on names starting 'rep_' (s13:374, s13:379), and so do both honest blocks (s13:431, s13:450), so 'pca_matched' is computed, scored and stored but compared with nothing.
- On a classification dataset a cluster holding a single held-out point scores purity 1.0, because s13:114 accepts mask.sum() >= 1 where the regression path at s13:125 requires 2.
- **Blocking** `similarity/s13_clustering_retrieval.py:283-286, 288-295, 237-242` — Per-fold scores are appended positionally with no placeholder for a fold that produced nothing.
- **Also caught by the hybrid-usage audit** — s13 asks the pool for a hybrid without passing objective='similarity'.

*Smoke:* ran clean. exit 0, s13 done in 10.6s, file written.

<details><summary>Options considered</summary>

1. Pad the per-fold lists. Decide each fold's space list up front (the ten representations, pca_matched, hybrid), and append float('nan') for any space that produced nothing on that fold. *Cost:* About six lines in run.
2. Match the dimension control to the hybrid's requested width: use blueprint.budget in place of budget at s13:259, falling back to 100 when the blueprint is missing. *Cost:* One line.
3. Match it to the width the hybrid actually got: move the PCA block after the hybrid block and use Xh_tr.shape[1], keeping blueprint.budget as the fallback when the hybrid build fails. *Cost:* About ten lines of reordering inside the fold loop.

</details>

## _summary_keys — `similarity/_summary_keys.py`

**DECIDED.** Leave it.

YES · hybrid: none needed · compute: fine · live

Stdlib-only lookup table mapping each of the six per-dataset inputs the cross-dataset summary reads to every key spelling that names it, plus the check that refuses to write an aggregate when not one dataset contributed a metric.

**Q.** Under which key names can each of the six per-dataset similarity results be found, so either aggregator can read the other's output?
**Q.** Did any dataset contribute a single metric to the cross-dataset aggregate, and if not, which input was each dataset missing?

Why:
- One small thing, and it only affects the wording of a refused run. get_input tests `if value` at line 41, so an input that is present but empty reads as absent, and missing_metric_report then says "absent" at line 71 when it should say "present but produced no field".

*Smoke:* ran clean. exit 0.

<details><summary>Options considered</summary>

1. Leave it as it is. *Cost:* Zero.
2. Distinguish present-but-empty from absent: test `key in ds_results` at line 40 rather than the truth of the value. *Cost:* Two lines.

</details>

## _run_similarity_locally — `similarity/_run_similarity_locally.py`

**DECIDED.** Take the normalising version so the smoke gate actually runs the five experiments, then fix the allocation set.

PARTIAL · hybrid: none needed · compute: fine · live

Throwaway driver that shrinks the engine in memory (FULL_BUDGETS, DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DEFAULT_MODELS, DEFAULT_IMPORTANCE_METHODS, DEFAULT_ALLOCATION_METHODS, DEFAULT_INTERNAL_EVALUATORS), blanks the dataset's augmentation representations, wraps the molecular representation builder so a comma list is not expanded to all 33, then calls similarity_helpers.run_all on a small real dataset.

**Q.** Does the similarity suite run end to end on a small slice of a real dataset, on a laptop, without foundation models or the server?
**Q.** Does shrinking the engine's module-level defaults actually shrink the precompute, given that precompute_hybrid_searches reads those constants and not the config dict?

Why:
- Both similarity steps of the smoke gate run no experiment at all. run_smoke_gate.sh:156-157 passes full module names, line 75 forwards them untouched, and similarity_helpers.py:716-733 keys its map by short id.
- The allocation set the gate exercises no longer matches production. Line 54 sets performance_weighted and top_2.
- force_rerun is not passed at line 124. Since 2026-09-04 similarity_helpers only resumes off a file whose stored input_fingerprint matches the data just loaded, so a changed fold grouping, label vector or representation list recomputes on its own. A CODE change still does not: the fingerprint cannot see it, so a second hand-run into the same KIRBY_LOCAL_OUT after one still prints RESUME and reports the old numbers.
- The usage block at lines 8-13 presents KIRBY_SMOKE_N as this driver's own setting. The driver never reads it.
- **Blocking** `similarity/_run_similarity_locally.py:75` — argv[1:] is forwarded verbatim to run_all(experiments=...), which resolves them through similarity_helpers._import_experiment's module_map keyed by short ids ('s13').

*Smoke:* ran clean. exit 0 on herg_ki in six separate invocations; it is the only working local entry point for this suite and it honours KIRBY_LOCAL_OUT, so nothing was written into tests/results.

<details><summary>Options considered</summary>

1. Accept both spellings: normalise each argument to its short id before passing it on, by matching against similarity_helpers module_map values as well as its keys. *Cost:* About five lines in main.
2. Refuse an id that is not in SINGLE_DATASET_EXPERIMENTS, and change run_smoke_gate.sh:156-157 to the short ids. *Cost:* Three lines in the driver, one line in the shell script.
3. Bring the gate's allocation set back onto production: replace ['performance_weighted', 'top_2'] at line 54 with names from the production five. *Cost:* One line.

</details>


---

# Similarity, standalone — 3 files

## sim_optimized_hybrid — `similarity_optimized_hybrid.py`

**DECIDED.** Pass budgets explicitly and remove greedy and dominant_greedy from SIM_ALLOCATIONS. Both are closed and this file can still select one as its winner.

PARTIAL · hybrid: engine, per fold · compute: heavy · live

Runs two find_best_hybrid searches on the same molecular dataset that differ only in the objective ('similarity' vs 'predictive'), rebuilds each winner per fold with create_hybrid, and compares the two spaces plus the best single representation on held-out distance-property correlation, on label-noise detection and correction under three noise conditions, and on importance-weighted distances.

**Q.** Does a hybrid selected by held-out nearest-neighbour activity preserve similarity on held-out rows (distance-property Spearman, neighbour-lie rate) better than a predictive-selected hybrid and than the best single representation?
**Q.** Under injected label noise, do noise-detection and label-correction methods work better in that geometry than in the predictive hybrid or the raw representation?
**Q.** Does weighting distances by feature importance (none / sqrt / squared / top-50) sharpen the similarity-based detectors?

*On re-read:* Two items are wrong on re-reading. (1) The audit says this experiment "can and does select a cut allocator as its winning blueprint".

Why:
- Two of the five allocation methods the file asks for cannot run under the geometry objective at all, and both are cut anyway.
- The budget grid the file declares never reaches the search: SIM_BUDGETS at line 108 is dead, and find_best_hybrid warns and substitutes [50, 150, 300] (find_best_hybrid.py:6288-6295).
- The fold count never reaches the search either, so it runs five folds (DEFAULT_N_FOLDS, find_best_hybrid.py:2742) while both payoffs score three.
- Both losses came from one deleted line: commit 9bd48ac removed 'speed=\'full\', budgets=SIM_BUDGETS, n_folds=N_FOLDS,' while removing the speed tiers, and took the two wanted arguments with it.

*Smoke:* not run. exceeded 6 min, killed at 360.3s, still inside the first hybrid search of seed 0.

<details><summary>Options considered</summary>

1. Put the deleted arguments back and drop the two cut allocation methods. Add 'budgets=SIM_BUDGETS, n_folds=N_FOLDS,' to the keyword dict at line 290 (the 9bd48ac line minus speed=), and remove 'greedy' and 'dominant_greedy' from SIM_ALLOCATIONS at line 110. *Cost:* Nothing on disk is invalidated: tests/results/sim_optimized_hybrid is empty, and no file under tests/results or tests/results_server contains the key payoff1_heldout_similarity.
2. Write down what each payoff actually measured. In _build_result: record the feature width of each of the three spaces per payoff, mark 'gaussian' as a condition whose detection cells are undefined, and store the per-fold values alongside the mean. *Cost:* The width and the gaussian marker are free at write time.
3. Choose both blueprints inside an outer fold instead of on every row, by routing the two searches through nested_cv_evaluate (find_best_hybrid.py:7932) and running the payoffs on each outer test fold. *Cost:* One search becomes five, one per outer fold, per arm per dataset per seed.

</details>

## structural_geometry_survey — `structural_geometry_survey.py`

**DECIDED.** Add the base-representation-alone arm. Without it no augmentation delta can be computed from the file.

**Still open:** Whether the hybrid arm is repaired or removed. Both live htc jobs record 'no fitted rf result' until this is settled.

PARTIAL · hybrid: should use engine, does not · compute: heavy · live

Predicts an antibody structure per sequence, derives 7 geometry feature groups plus their concatenation, scores each group standalone under group-aware 5-fold CV, and scores every base representation with each geometry group hstacked onto it. It also launches a find_best_hybrid search over base plus geometry, but never gets a hybrid out of it.

**Q.** Do structure-derived geometry features (VH-VL angles, CDR shape, paratope charge and the rest) carry standalone signal on antibody property prediction?
**Q.** Does concatenating a geometry feature group onto a base antibody representation change the score?
**Q.** Does a KIRBy hybrid over base representations plus geometry beat the geometry groups scored on their own?

Why:
- The hybrid arm has never produced a number and cannot. find_best_hybrid is called at line 335 without return_fitted=True, and fitted_results is only populated when that flag is set (find_best_hybrid.py:7056-7057), so line 342 always gets None and line 350 records "search returned no fitted 'rf' result".
- The search that feeds that arm runs in full and is discarded. Per dataset it is 8 budgets (FULL_BUDGETS) x 5 allocation methods x 3 importance methods x 5 internal evaluators x 7 models = 4,200 configurations on fold 0, then about 2,520 on the remaining four folds after the fold-0 prune at 0.25 keeps three allocation methods: roughly 14,000 model fits, plus 560 baseline fits and 400 representation-scoring fits.
- What reaches the file is small by comparison: 8 geometry groups x 2 models x 5 folds = 80 standalone fits, plus 9 base representations x 8 geometry groups x 2 models x 5 folds = 720.
- Nothing is cached. dataset_name and tier are passed at lines 337-338 but cache defaults to False (find_best_hybrid.py:6081), so the discarded search is recomputed from scratch on every run and writes no reusable row.
- **Blocking** `structural_geometry_survey.py:336` — find_best_hybrid is called without return_fitted=True.

*Smoke:* not run. exceeded 6 min at 'Computing geometry features (672 structures)...'.

<details><summary>Options considered</summary>

1. Score each base representation on its own, in the loop at line 409, with the same evaluate_rep call the other two arms use. *Cost:* 9 base representations x 2 models x 5 folds = 90 fits per dataset, against the 800 that already reach the file: about 12 percent more useful work.
2. Add return_fitted=True and cache=True to the find_best_hybrid call at line 335. *Cost:* Two keywords.
3. Score the hybrid arm through nested_cv_evaluate (find_best_hybrid.py:7932) instead: search inside each outer fold, score on the untouched outer test rows. *Cost:* One search becomes five, one per outer fold: roughly 70,000 model fits per dataset instead of 14,000.

</details>

## read_sim_optimized_results — `read_sim_optimized_results.py`

**DECIDED.** Fix the three print blocks alongside the experiment, not before it.

PARTIAL · hybrid: none needed · compute: fine · live

Prints a compact per-dataset summary of the JSON that similarity_optimized_hybrid writes: whether the two searches picked different blueprints, the distance-property correlation and neighbour-lie rate, per-method detection precision against a chance line, estimator AUROC and precision@k, and the importance-weighting cells. Reads only; computes nothing.

**Q.** Did the similarity search and the predictive search pick different blueprints, and on how many seeds?
**Q.** For each noise condition, how do the three spaces compare on damage, global label smoothing, per-method detection precision and removal delta?
**Q.** Which noise estimators rank the corrupted rows best in each space?

Why:
- The warning at lines 53-56 is wrong. It says that an identical allocation method and budget on both arms make every payoff a comparison of a space against itself.
- The per-method precision table at lines 98-107 is printed for every condition against the single noise_frac value. Under 'gaussian' every molecule is perturbed, so precision is 1.0 for any method and the 'AT CHANCE' marker can never fire.
- The payoff 3 block at lines 126-134 prints 'sim precision' without naming the condition the sweep ran under, although the JSON records it as payoff3['noise_type'].
- Only means are printed. The file already carries lo, hi, std and n from _ci, and none of it is shown, so a reader cannot see how wide any number is.

*Smoke:* ran clean. exit 0 both ways, in under a second.

<details><summary>Options considered</summary>

1. Rewrite the three print blocks: replace the 'IDENTICAL means degenerate' warning with a line saying the two arms picked the same allocation method and budget but not necessarily the same features, skip or mark the detection table under 'gaussian', print payoff3['noise_type'] above the weighting cells, and add pred to the precision line. *Cost:* No model fits and no re-run.

</details>


---

# Similarity metrics — 33 files

## simeval — `similarity_metrics/simeval.py`

**DECIDED.** Make the four returns NaN, set cliff_auroc False in the activity-grounded table, add the noise_recovery_defined flag.

PARTIAL · hybrid: none needed · compute: fine · live

A helper module, not an experiment. It turns a feature matrix into an (n,n) distance matrix under six coefficients (tanimoto, dice, cosine, soergel, tversky, euclidean), and scores one distance matrix on nine activity-grounded statistics: scaffold-grouped kNN prediction from five nearest neighbours (rho and -RMSE), MODI, Patterson neighbourhood bound, cliff AUROC, neighbour-lie rate, BEDROC enrichment, and injected-noise detection AUROC plus recovery correlation.

**Q.** Given a distance matrix over molecules, how activity-coherent are its neighbourhoods?
**Q.** How does each similarity coefficient's geometry depart from ECFP4 Tanimoto (diagnostic only, never ranked)?
**Q.** Do a metric's neighbourhoods expose injected label noise, across a dose x contaminated-fraction grid?

*On re-read:* The five defects are all really there, but three line numbers point one line early: eval_patterson returns 0.0 at line 204, eval_modi_binary at 183, eval_noise_recovery at 385. There is a fourth of the same kind the audit missed: diag_ecfp_agreement returns 0.0 at line 428 when its Spearman is undefined.

Why:
- Four scorers return 0.0 for a case they cannot compute, and 0.0 is a legal value on every one of those four scales.
- eval_patterson's 0.0 is the best score the statistic can produce, because it otherwise returns a negated percentile of absolute activity difference, so a distance matrix too degenerate to form ten close pairs wins the comparison.
- eval_modi_binary's 0.0 is the worst legal MODI, so an undefined result and a measured floor are stored as the same number.
- eval_noise_recovery's 0.0 is a real correlation value, and f23_noise_cleaning_metric.py:186 keeps it because it filters on np.isfinite, which a folded zero passes.

*Smoke:* ran clean. exit 0, ~5s.

<details><summary>Options considered</summary>

1. Change the four returns from 0.0 to float('nan') -- lines 183, 204, 385, 428 -- and set the cliff_auroc entry of EVAL_IS_ACTIVITY_GROUNDED to False. Then add a noise_recovery_defined flag next to noise_recovery in aggregate.py:139, matching the readacross_defined and struct_fidelity_defined flags already written two lines above it. *Cost:* Five one-line edits in simeval.py plus two lines in aggregate.py.

</details>

## rankers — `similarity_metrics/rankers.py`

**DECIDED.** Delete the five unused scorers plus RANKERS and LABEL_FREE.

YES · hybrid: none needed · compute: fine · live

A helper module of nine feature-importance scorers meant as similarity-aware alternatives to the engine's compute_feature_importance. Seven score every feature by the dependence between that feature's absolute pair difference and a target -- either the absolute activity gap or a structural distance.

**Q.** Can a feature be scored by how well its pair-difference tracks the activity gap, or a structural distance, rather than by a fitted model's importance?
**Q.** Does fusing an activity-target scorer with a structural-target scorer beat either alone (that fusion is what combine_ranks supports)?

Why:
- Five of the seven scorers -- r1_activity_spearman, r2_structural_spearman, r3_distance_corr, r4_mutual_info, r7_kta -- are called by nothing in the repository, and the RANKERS registry that names them is imported by nothing.
- Those five accept a 1-D structural target and align it by slicing struct[:len(i)] at lines 49, 67, 76 and 136; the pair indices were already filtered by i != j, so the slice is misaligned whenever any sampled pair drew the same molecule twice, and nothing raises.
- r7_kta at line 142 returns the signed centred cosine while the copy that actually runs, ranker_experiment.sc_kta at line 98, returns its absolute value, and both are labelled R7.
- RANKERS maps r3, r4 and r7 to one name each, but each of those takes target='activity' or 'structural', so anything iterating the registry gets only the activity mode -- the very comparison the module docstring is built around.

*Smoke:* ran clean. exit 0, ~10s.

<details><summary>Options considered</summary>

1. Delete r1_activity_spearman, r2_structural_spearman, r3_distance_corr, r4_mutual_info, r7_kta, RANKERS and LABEL_FREE. Keep _spearman_cols, _dcor, r5_rrelieff, r6_laplacian and combine_ranks, which are the five things other files import. *Cost:* About 45 lines removed from one file.
2. Keep all seven and make the 1-D target path raise instead of slice: at lines 49, 67, 76 and 136 require struct.ndim == 2, or require len(struct) == len(i) after the i != j filter. *Cost:* Four one-line edits.

</details>

## ranker_experiment — `similarity_metrics/ranker_experiment.py`

**YOURS.** Delete and read framework/f22, or rewrite it per fold. f22 asks the same headline question leak-free and has a stored result on one of these two datasets. Keeping it means the other ten selectors matter to you.

NO · hybrid: hand-rolled · compute: fine · superseded

Pools four representations (ecfp4_count, atompair_count, desc, maccs -- about 2423 features), z-scores them, and runs thirteen feature selectors over 8000 cached molecule pairs against two targets: the absolute activity gap and the MCS distance. Each selection keeps 100 features per representation from the top-N representations, and the resulting space is scored two ways -- scaffold-grouped kNN prediction from five nearest neighbours, and Spearman between the space's pairwise distance and MCS distance -- under cosine and euclidean.

**Q.** Does selecting features FOR similarity beat selecting them predictively, when the selected space is judged by kNN prediction from five nearest neighbours and by agreement with shared substructure?
**Q.** Does a label-free structural selection target (MCS distance) beat a supervised activity target (the absolute activity gap)?
**Q.** Does any selected 400-feature space beat the standard count fingerprint under its native Tanimoto?

Why:
- Every selector is fitted on all 8000 cached pairs drawn from every molecule and, for the activity target, on the whole label column at line 252; the selected features are then scored by a scaffold GroupKFold at line 205 over those same molecules, so the activity arms saw the labels of every held-out fold and the structural arms did not.
- The three baselines at line 294 keep their full 1024 features against 400 selected, so any gap mixes a width change with a selection change and the stored rows record nothing that separates the two.
- r9_greedy returns exactly the pre-filtered set at the default settings, so the one selector meant to reward features that complement each other measures only marginal Spearman.
- r7_kta and sk_fscore are the same ranker, so two of the thirteen selectors are one.
- **Blocking** `similarity_metrics/ranker_experiment.py:252` — Feature selection sees the labels of every molecule, then the selected space is scored by a GroupKFold kNN over those same molecules (line 205).

*Smoke:* ran clean. exit 0, 33s.

<details><summary>Options considered</summary>

1. Delete it and read f22_similarity_feature_sources.py instead. f22 asks the same headline question -- does ranking features by their agreement with shared substructure beat ranking them by their agreement with the activity gap -- and asks it per fold: it restricts the MCS pairs to training endpoints at lines 366-372, ranks on training rows only at line 386, and reports both yardsticks on the held-out fold. *Cost:* Deletes one 348-line file plus its two untracked result files, ranker_experiment.json and ranker_experiment_quick.json.
2. Move the selection inside the fold. Rewrite run_dataset so each of the five scaffold folds restricts the cached pairs to training endpoints, z-scores on training rows, scores and allocates on those rows only, then builds the distance and predicts each held-out molecule from its five nearest training neighbours. *Cost:* Rewrites run_dataset (72 lines) and score_space (12 lines).
3. Keep the file as it is and record what it measured. Change the docstring at line 22 and add a field to every emitted row saying that the selection saw the labels of all molecules. *Cost:* Two lines.

</details>

## blacklist — `similarity_metrics/blacklist.py`

**DECIDED.** Delete. src/kirby/similarity/selection.py already does the structural half per fold and is wired in. Nothing imports this and nothing stores its output.

PARTIAL · hybrid: none needed · compute: heavy · orphaned

A standalone library that finds 'saboteur' columns inside one representation block by greedy backward elimination inside each scaffold GroupKFold fold, under one of two yardsticks: Spearman between the block's pairwise squared distance and MCS distance over cached pairs, or a train-reference-to-test kNN prediction Spearman. A column reaches the blacklist only if it is dropped in a majority of folds and its mean held-out gain clears a threshold.

**Q.** Which individual columns inside a feature block make the block's similarity geometry worse, stably across scaffold folds?
**Q.** How much does removing those columns raise the held-out yardstick (the returned delta)?

*On re-read:* The eight defects are all really there and the line numbers are right. Two corrections.

Why:
- With yardstick='readacross', the feature dropped at each greedy step is the one that most raises the score on the test fold (line 190), while the module docstring at lines 18-22 says every drop is decided on training folds and re-measured on held-out folds.
- The returned delta for that yardstick is therefore an optimisation gain measured on the rows it optimised, guaranteed non-negative by construction, and the inner docstring at line 161 admits this while the module header does not.
- In that same path train_gain and heldout_gain are both set to best_gain (lines 195-196), so mean_train_gain and mean_heldout_gain in the returned ranking are identical for every feature and read as perfect train/held-out agreement.
- Even under the structural yardstick, features enter the blacklist by their mean held-out gain (line 313) and the before/after is then re-measured on those same held-out pairs (lines 324-333), so the reported delta is selected and scored on the same rows.
- **Blocking** `similarity_metrics/blacklist.py:190` — For yardstick='readacross' the column dropped at each greedy step is the one that most raises the score on the TEST fold.

*Smoke:* ran clean. exit 0, ~8s.

<details><summary>Options considered</summary>

1. Delete it. src/kirby/similarity/selection.py:57 already answers the structural half per fold, corrects exactly the fitted-noise problem this file has, and is wired into src/kirby/similarity/recipes.py:125 and two scripts under tests/. *Cost:* Deletes one 380-line file.
2. Fix the two blockers and keep it as the producer of a fixed, stability-counted blacklist. Rewrite _readacross_fold_greedy (lines 159-202) so each drop is picked by an inner GroupKFold over the training rows, then re-measured once on the test fold; record the two numbers separately at lines 195-196; delete the no-op at lines 184-187 and either implement cand_cap or remove the parameter and its documentation at line 231. *Cost:* One function of 44 lines rewritten, plus about 6 lines elsewhere.
3. Leave the structural path alone and make the other one refuse: raise NotImplementedError where yardstick == 'readacross' is dispatched at line 276, citing line 190. *Cost:* Two lines.

</details>

## scaffold_experiment — `similarity_metrics/scaffold_experiment.py`

**DECIDED.** Use the registry grouping, delete the position_gated row (it is the Grantham grader itself), fix the filename so a smoke run stops overwriting the full run, re-run all five.

PARTIAL · hybrid: none needed · compute: fine · live

Builds five scaffold-mutation representations plus two whole-sequence base representations (aac, blosum_mean) on one FLAb variant library, turns each into a pairwise distance under cosine and euclidean, and scores every space two ways: Spearman of its pair distances against three substitution-matrix graders, and a five-nearest-neighbour prediction of the fitness label under GroupKFold. Writes one row per (representation, coefficient) to JSON.

**Q.** Does describing a variant by what changed relative to the wildtype order variant pairs more like an independent substitution matrix than describing it by its whole sequence (aac, blosum_mean) does?
**Q.** Which distance coefficient, cosine or euclidean, gives the better-ordered space?
**Q.** Do the five nearest neighbours in each space predict the measured fitness under a position-grouped split?

Why:
- On flab_binding_adams, flab_expression_adams and flab_binding_phillips_h3 the folds are built from each variant's first mutated position (line 78), and a variant in those libraries carries several mutations, so a held-out variant shares mutations with its own training variants.
- The stored numbers show it: in scaffold_experiment_flab_binding_phillips_h3.json even `aac`, a whole-sequence amino-acid composition, reaches readacross_rho 0.815, and MutID reaches 0.892.
- run() guards the position_gated row on the single-mutation fraction (line 256) but never guards the grouping, so the three combinatorial files were written with 14 rows and a leaky readacross_rho.
- The position_gated row (line 256) is the Grantham grader itself on a single-mutation library, and scaffold_ddg_experiment.py:27-29 already states this and refuses to score it there.

*Smoke:* ran, NaNs present. exit 0, ~7s.

<details><summary>Options considered</summary>

1. Choose the fold groups per dataset, the way tests/antibody_baseline_survey.py:66-74 already does for these exact three datasets: for adams, expression_adams and phillips_h3 set groups from d['edit_distances'], the number of mutations from the wildtype; keep the first-mutated-position groups where every variant carries one mutation. *Cost:* 14 spaces x 5 folds = 70 five-nearest-neighbour fits per dataset, 210 over the three.
2. Write readacross_rho as NaN wherever the single-mutation fraction is below 0.99, with the same printed refusal run() already gives position_gated at lines 262-265. *Cost:* Zero nearest-neighbour fits on those three datasets.
3. Delete the position_gated block (lines 256-265) and say in the docstring why, copying the sentence scaffold_ddg_experiment.py:27-29 already carries. *Cost:* Zero fits.

</details>

## scaffold_ddg — `similarity_metrics/scaffold_ddg_experiment.py`

**DECIDED.** Chain and complex keys, and the crash message.

**Still open:** Whether every complex still contributes 400 pairs regardless of its size.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Runs the same five scaffold-mutation representations on graphinity single-point protein ddG data, sampling 400 variant pairs inside each PDB complex with at least 25 mutations, pooling those pairs across complexes and correlating each representation's pair distance against three substitution-matrix graders. Structural axis only, no labels used.

**Q.** Do the five scaffold-mutation representations order pairs of single-point mutants like an independent substitution matrix, on a library where every variant carries exactly one mutation so the mutation-count confound cannot act?
**Q.** Which distance coefficient, cosine or euclidean, orders those pairs better?

Why:
- A mutation's position is its residue number with the chain letter dropped (line 111), so a heavy-chain residue 100 and a light-chain residue 100 are treated as the same site by grader_pairs and by the same-position branch.
- Complexes are keyed by PDB id alone (lines 109, 127) while the loader keys sequences by (pdb, ab_chain), so one PDB with two antibody chain pairings is pooled as one complex and its within-complex pairs cross two different wildtypes.
- The BLOSUM grader is more distorted here than anywhere else in this suite: grader_pairs (lines 96-101) gives a same-position pair one substitution term and a different-position pair two, and each term carries an offset of about 11 under the max-minus-score matrix, so the two groups sit about 11 apart before any chemistry enters.
- The headline clean-best in the smoke was dPhysChem_pos euclidean at BLOSUM 0.382, and a euclidean distance on a position-indexed representation separates same-position from different-position pairs, which is what that 11-unit step also does.

*Smoke:* ran, NaNs present. exit 0, ~25s.

<details><summary>Options considered</summary>

1. Build the position from (chain, residue number) instead of the residue number alone at line 111, using the m[0]['chain'] the loader already supplies at graphinity.py:353. *Cost:* Zero model fits.
2. Return ab_chain from load_graphinity_dataset and key the complex loop on (pdb, ab_chain) at lines 127-128. *Cost:* Zero model fits.
3. Raise a message naming --min-complex when no complex qualifies, instead of letting np.concatenate fail at line 156. *Cost:* Zero fits, nothing stale.

</details>

## scaffold_feature_isolation — `similarity_metrics/scaffold_feature_isolation.py`

**DECIDED.** Fix the damage share, add --cap and --output. Hold the re-run until blosum_matrix lands, since that sets every number here.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Takes dPhysChem and dPhysChem_pos on three FLAb variant libraries and scores feature subsets instead of whole representations: each of the five property scales alone, all five minus one, and dPhysChem_pos with each mutated position removed in turn. Every number is a Spearman of pair distance against the BLOSUM grader, raw and with the Hamming component partialled out.

**Q.** Which of the five physicochemical scales inside dPhysChem and dPhysChem_pos carry the agreement with the BLOSUM grader, alone and when dropped?
**Q.** Is that agreement carried by a few mutated positions or spread across all of them?

Why:
- Every number here is graded by BLOSUM alone (line 80), so whatever is decided about blosum_matrix in scaffold_reps sets every row in this file.
- top5_damage_share divides by the sum of signed damages (lines 133-134), which can be near zero or negative when dropping a position improves the score; the stored phillips_h3 value is 1.015.
- phillips_h3 has 15 mutated positions and adams has 20, so "the top five" is a third and a quarter of all positions there; koenig has 225.
- An undefined correlation is written as 0.0 (lines 63-68), while the other three files in this directory now write NaN plus a *_defined flag.

*Smoke:* ran clean. exit 0, ~5 min (it exceeded the Bash tool's 2-min default on the first attempt; re-run with a longer wall).

<details><summary>Options considered</summary>

1. Report the top-5 damage sum and the total damage as two numbers instead of dividing them (lines 133-134). *Cost:* Zero model fits.
2. Return NaN plus a *_defined flag from score() instead of 0.0, matching scaffold_experiment, scaffold_ddg_experiment and _nanrank. *Cost:* Zero model fits.
3. Delete the docstring claim at lines 23-26 about the registered pipeline representations. *Cost:* Zero fits, nothing stale.

</details>

## scaffold_reps — `similarity_metrics/scaffold_reps.py`

**DECIDED.** Fix blosum_matrix — one line, and it sets a number in ten result files. Take the wildtype as an argument before anyone quotes an adams or phillips_h3 result.

PARTIAL · hybrid: none needed · compute: fine · live

Pure library. Holds the amino-acid property scales, the Grantham, BLOSUM and identity substitution matrices, sequence encoding and wildtype inference, the five scaffold-mutation representations, the grader-weighted pair distance used as ground truth, and the position-gated distance.

**Q.** Does it build the five scaffold-mutation representations, the three graders and the two scaffold-relative distances correctly for the scripts that import it?

*On re-read:* Two corrections. First, the -1 sentinel item at line 227 is wrong.

Why:
- blosum_matrix returns S.max() - S (line 79), so two identical residues sit a nonzero distance apart, 0 to 7 depending on which residue it is.
- That puts same-position pairs and different-position pairs on baselines about one wildtype self-score apart, and MutID, BaseDelta and dPhysChem_pos separate exactly those two groups, so the term sits inside their BLOSUM number.
- Two variants carrying the same mutation at the same position get a nonzero contribution there, which does not cancel on the combinatorial libraries where that happens often.
- The wildtype is the per-column most common residue (lines 105-106), so on a site split near evenly between two residues the choice is a near-tie and every mutation record for that site inverts.

*Smoke:* ran clean. exit 0, ~4s, no results file.

<details><summary>Options considered</summary>

1. Make blosum_matrix return 0.5*(S[i,i] + S[j,j]) - S[i,j] instead of S.max() - S, so two identical residues are at distance zero. *Cost:* Ten result files go stale: the five scaffold_experiment JSONs plus scaffold_experiment_quick.json, scaffold_ddg_foldx.json, scaffold_feature_isolation.json, feature_type_study.json, and tests/experiments/results/opig_scaffold_similarity.json.
2. Keep the per-column mode as the wildtype but record, per site, the mode's share of the variants, and print the lowest one. *Cost:* Zero fits, nothing stale.
3. Give encode_sequences a wildtype argument, defaulting to the current per-column mode, and pass the published parent sequence for the combinatorial libraries. *Cost:* Zero fits by itself.

</details>

## sm_feature_type_study — `similarity_metrics/feature_type_study.py`

**DECIDED.** Three one-line fixes, then delete run() and main() and keep the module. Nothing reads its JSON.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Builds 14 amino-acid property scales into a positional delta-from-wildtype representation of FLAb variants, then scores subsets of those scales by the count-controlled Spearman between their Euclidean pair distance and a BLOSUM62 sequence distance over 8000 randomly sampled variant pairs. No model is trained and no measured label is ever read.

**Q.** Which amino-acid property scales earn a place in a Euclidean distance space that reproduces BLOSUM sequence distance once mutation count is partialled out?
**Q.** Does forward selection on that objective beat taking the top-k scales by their solo score, i.e. does redundancy between scales cost more than weak solo signal?
**Q.** Is the score monotone in the number of scales included, or is there an optimum below all 14?

*On re-read:* Three corrections. First, the greedy start at 0.0 (:143) never bit: on all three stored datasets the best single scale scores 0.390, 0.544 and 0.583, all above zero, so no run has ever returned an empty set.

Why:
- Scales are chosen and the gain is reported on the same 8000 pairs; there is no second pair set anywhere in the file.
- opig_scaffold_similarity.py --mode features is the same computation on six datasets and writes a results document, so this file's run() and main() are a smaller copy of it.
- opig_scaffold_similarity's copy uses plain np.argsort and max, so it does not carry the non-finite handling this file got in commit 49375c1.
- The raw-units arm at :202 calls build(dataset, zscore=False), which reloads FLAb and re-encodes every sequence; opig_scaffold_similarity does the same comparison off the already-built context.

*Smoke:* ran clean. exit 0, ~20s, no results file (I called build/greedy_forward directly, so feature_type_study.json was not touched).

<details><summary>Options considered</summary>

1. Delete run() and main(). Keep the module: the 14 scales, property_table, score_subset and greedy_forward. *Cost:* No re-run of anything stored.
2. Split the 8000 pairs in half. Choose scales on one half, report every gain on the other. *Cost:* Each rank correlation runs over 4000 pairs instead of 8000; the number of calls is unchanged at roughly 130 per dataset.
3. Carve the variants in two. Draw the selection pairs inside one set and the reporting pairs inside the other. *Cost:* About 15 lines in build().

</details>

## sm_feature_ablation — `similarity_metrics/feature_ablation.py`

**DECIDED.** Blacklist swap and the full-size 3D block, plus the pair split for the fine-2D selection.

PARTIAL · hybrid: hand-rolled · compute: fine · orphaned

Exhaustively scores every subset of four z-scored feature blocks (ECFP4 counts, 208 RDKit descriptors, top-100 2D columns, MACCS keys) concatenated into one Euclidean distance, on two yardsticks: scaffold-grouped 5-nearest-neighbour Spearman against the measured label, and Spearman against cached MCS distance. Then runs a drop-one greedy over the 208 descriptors to find which ones lower structural fidelity.

**Q.** Which representation blocks help and which hurt when they are concatenated into one similarity space, judged against predicting each molecule's activity from its five nearest neighbours and against MCS structural fidelity?
**Q.** Do individual RDKit descriptors inside the physicochemical block actively lower structural fidelity, and which ones?
**Q.** Does a 3D pharmacophore block add anything over the 2D blocks?

Why:
- The DATA REALITY paragraph at :35-43 is false. sim_herg_ki_spatial.npz holds n_built=1415 of 1415 and sim_logd_spatial.npz holds 1427 of 1427, both complete.
- desc_drop_one_greedy at :186 decides each drop and reports the result on the same 8000 pairs. On herg_ki it drops 143 of 208 descriptors and moves structural fidelity 0.250 to 0.495 with nothing held out.
- blacklist.py already does that job honestly and is not called from here: every drop decided on training folds, re-measured on held-out pairs, and a feature reaching the list only if a majority of folds drop it.
- select_fine2D at :96 picks 100 of 2048 features by their agreement with MCS on the same 8000 pairs the structural score then reads, so every subset containing fine2D is optimistic on that yardstick.

*Smoke:* ran clean. exit 0, ~15s, no results file.

<details><summary>Options considered</summary>

1. Run the 3D block at full size as a supplement. run_shape_supp already caps with n_sub = min(n_sub, have) at :259, so main() needs a large n_sub and a second call for logd. *Cost:* 31 subsets by 5 folds = 155 neighbour fits per dataset.
2. Instead of a supplement, make shape a fifth block of run_full. *Cost:* 155 neighbour fits per dataset instead of 75, and one exhaustive table per dataset instead of two.
3. Replace desc_drop_one_greedy with blacklist.identify_saboteurs(p['desc'], DESC_NAMES, p['y'], p['groups'], m, yardstick='structural'). The signature is at blacklist.py:208 and the module's own smoke block at the end of that file makes exactly this call on herg_ki. *Cost:* The current single pass is about 19,600 rank correlations on herg_ki and 19,400 on logd, over 8000 pairs each.

</details>

## sm_base_rep_comparison — `similarity_metrics/base_rep_comparison.py`

**DECIDED.** Paired bootstrap and the classification dispatch.

**Still open:** What a group is on the adams libraries, which decides the fold grouping.

PARTIAL · hybrid: none needed · compute: fine · orphaned

On four FLAb variant libraries, builds six distances from two edit-describing representations and three whole-sequence representations, guesses each held-out variant's label from its five nearest training neighbours under GroupKFold, and reports the Spearman of guessed against true labels with a 1000-draw bootstrap interval. Two null controls (random distance, constant distance) are included.

**Q.** Does describing what a variant changed relative to the wildtype beat describing its whole sequence, when a variant's measured label is guessed from its five nearest neighbours?
**Q.** Do the reported correlations separate from a random distance and from a constant distance?

*On re-read:* Two corrections. The degenerate one-hot arm is on both koenig sets, not one: on flab_expression_koenig it scores -0.0443 and the constant control scores -0.0443; on flab_binding_koenig both score -0.0478.

Why:
- flab_binding_adams arrives as 0/1 labels. The FLAb loader binarises it at pKd 5.0 (src/kirby/datasets/flab.py:86, applied at :310-316) and sets d['task_type'] to 'classification' (:396).
- The constant control is far from zero on both adams sets, so the null the file compares against is not calibrated there.
- The verdict at :200-205 calls a difference a tie whenever two marginal intervals overlap. On flab_expression_koenig it ties aac:cosine 0.482 [0.457, 0.507] against dphyschem 0.467 [0.442, 0.493].
- The paired bootstrap that would settle those comparisons is already computed and thrown away. boot_ci rebuilds RandomState(seed) at :97 and is called with seed=0 for every arm at :136, so every arm resamples the same row indices; boots is local and only the percentiles come back (:107).

*Smoke:* ran clean. exit 0, ~20s, no results file.

<details><summary>Options considered</summary>

1. Return the bootstrap draws from boot_ci and compute the difference between the best edit representation and the best whole-sequence representation on those same draws. *Cost:* No new correlations: the 1000 resamples per arm are already computed at :100-104 and discarded.
2. Dispatch on d['task_type'] and score the binarised set as a classification. scaffold_experiment.readacross_classification at :142 is the same folds and the same distance-weighted neighbours, scored with AUROC recentred so 0 is random. *Cost:* 8 distances by 5 folds = 40 neighbour fits, plus 8 times 1000 bootstrap AUROCs.
3. Decide what a fold group is on the two adams libraries, then change :116. *Cost:* 16 of 32 rows re-run: 16 distances by 5 folds = 80 neighbour fits plus 16,000 bootstrap correlations.

</details>

## sm_aggregate — `similarity_metrics/aggregate.py`

**DECIDED.** Recording fixes, label-free ranking, and run the noise sweep. One injection at one dose is what makes the two noise columns unquotable.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Builds six single-metric distance matrices on herg_ki and logd, fuses them six ways (min-max score average, mean rank, reciprocal rank fusion, Borda, consensus vote, and a redundancy-pruned score fusion), and scores all twelve on four yardsticks: neighbour-based label prediction under GroupKFold, MCS structural fidelity, injected-noise detection AUROC, and noise recovery correlation.

**Q.** Does fusing several complementary similarity metrics beat the best single metric at predicting each molecule's activity from its five nearest neighbours?
**Q.** Does fusion beat the best single metric at MCS structural fidelity, at detecting injected label noise, and at correcting it?
**Q.** Which of the six fusion rules wins, and does pruning redundant metrics before fusing help?

*On re-read:* Two corrections, both about arms that are not what they look like. A6_dca kept all six metrics on both datasets, so it is A1_score exactly: the stored rows agree to four decimals on herg_ki (0.7210, -0.0614, 0.654, 0.786) and on logd (0.6360, 0.4194, 0.629, 0.724).

Why:
- The two noise columns come from one injection: seed 0, dose 0.6, contaminated fraction 0.15 (:152, simeval.py:299). simeval.noise_sweep_scores at simeval.py:393 supplies the dose-by-fraction grid and is not called.
- A4_borda is A2_meanrank rescaled, so one of the six aggregators is not a distinct rule.
- A6_dca is A1_score in this run, because thr=0.9 pruned nothing on either dataset.
- a6_dca at :113 orders the metrics by their agreement with the activity gap over pairs drawn from every molecule, then :133 scores the fused distance with scaffold folds on those same labels. It changed nothing here, but it is live for any dataset where the pruning bites.

*Smoke:* ran clean. exit 0, ~20s, no results file (aggregate_experiment.json untouched).

<details><summary>Options considered</summary>

1. Call simeval.noise_sweep_scores(D, p['y']) once per matrix and store the 9 cells on the row. *Cost:* 27 injections by 12 matrices by 2 scores = 648 evaluations per dataset, 1296 across both, up from 24.
2. Rank the metrics in a6_dca by their agreement with MCS distance, which the file already loads at :150-151, instead of by the activity gap. *Cost:* One line.
3. Record what this run already implies: a3_rrf's k=60 and a5_consensus's k=20, the noise seed, dose and fraction, a defined flag on the two noise fields, allow_nan=False at :196, and a line in the output saying that A2, A3, A4 and A5 are asymmetric and that struct_fidelity reads one side only. *Cost:* About six lines.

</details>

## creative_hybrids — `similarity_metrics/creative_hybrids.py`

**DECIDED.** Move MACCS and the feature-count sweep to the honest builder, then delete this file.

PARTIAL · hybrid: hand-rolled · compute: fine · superseded

Builds 11 similarity spaces on one dataset by z-scoring and concatenating ecfp4_count, RDKit descriptors, MACCS, and the top-K atompair/torsion features picked by their correlation with MCS distance, then scores each space two ways: Spearman of a 5-nearest-neighbour prediction under scaffold GroupKFold, and Spearman between the space's pairwise distances and MCS distance. A second sub-table adds the 49-feature 3D pharmacophore block on whatever rows the conformer cache covers.

**Q.** Does stacking descriptors, MACCS and a structurally-selected atompair/torsion block onto the count fingerprint raise the 5-nearest-neighbour activity Spearman above ecfp4_count with Tanimoto?
**Q.** Does that stacking raise agreement between the space's pairwise distances and MCS distance?
**Q.** How much of any gain is the coefficient change (Tanimoto to z-scored Euclidean) rather than the added blocks, and how many atompair/torsion features does the fine block need (50, 100, 250)?

Why:
- Nothing is written to disk: main() at line 256 calls run() and drops the returned dict, and no creative_hybrids.json exists in the folder.
- _zscore at lines 53-58 takes each feature's mean and standard deviation from all molecules, and eval_knn_readacross at line 113 then holds scaffold folds out of a distance matrix built from statistics that saw them.
- select_fine2d at line 154 ranks features against MCS distance over pairs I,J, and struct_fidelity at line 109 scores against MCS distance over those same pairs I,J.
- The activity number is one pooled Spearman over all out-of-fold predictions (simeval.py:163-166), so there is no fold-to-fold spread on any row.

<details><summary>Options considered</summary>

1. Add a json.dump of run()'s return value in main(), and an --out flag. *Cost:* About 5 lines.
2. Move the MACCS block and the 50/100/250 feature-count arms into similarity_noise_cleaning.py's combo list at lines 181-187, then delete creative_hybrids.py. *Cost:* The combos list grows from 4 entries to 7.
3. Delete the file and do nothing else. *Cost:* Zero re-runs.

</details>

## creative_hybrids_v2 — `similarity_metrics/creative_hybrids_v2.py`

**DECIDED.** Fix the output path collision first, then fold into the honest arm next door.

PARTIAL · hybrid: hand-rolled · compute: fine · live

Re-runs the same stacking idea with the 3D pharmacophore block at full row count and with a per-dataset greedy descriptor drop: repeatedly remove the single RDKit descriptor whose removal most raises the correlation between Euclidean distance and MCS distance. Scores 10 spaces per dataset on the 5-nearest-neighbour activity Spearman and on MCS agreement, across herg_ki, logd and adme_solubility, and writes creative_hybrids_v2.json.

**Q.** Does count fingerprint + descriptors-minus-saboteurs + 3D pharmacophore + fine 2D beat ecfp4_count with Tanimoto on the 5-nearest-neighbour activity Spearman, on all three datasets?
**Q.** Does it beat it on the correlation between pairwise distance and MCS distance?
**Q.** Does halving the weight on the count block change which stacked space wins?

*On re-read:* The bugs are all confirmed. Two facts about the stored file are missing.

Why:
- make_D at lines 76-94 takes every feature's mean and standard deviation from all molecules, so each activity number is measured on a geometry fitted on its own held-out rows.
- desc_blacklist_greedy at line 196 and select_fine2d at line 201 are fitted on the 8000 cached MCS pairs, and struct_fidelity at line 166 is scored on those same 8000 pairs.
- The descriptor drop removes 143 of 208 descriptors on herg_ki, 140 on logd and 147 on adme_solubility, with no held-out check anywhere in this file.
- Line 296 uses the same default output path for a single dataset and for --all, so a one-dataset re-run silently replaces the three-dataset file.

<details><summary>Options considered</summary>

1. Add the two spaces this file scores that similarity_noise_cleaning.py does not — count+physchem_full, and the count block alone as a z-scored Euclidean space — to that file's combos at lines 181-187, and to the pool _topk selects from at lines 236-240. *Cost:* Two entries added to a list, and two names removed from the exclusion at lines 238-240.
2. Rewrite run() to fit the descriptor drop, the feature ranking and the z-score means on one scaffold half and report on the other, reusing _scaffold_halves, _pairs_within, _zscore_fit and _make_D_fit from similarity_noise_cleaning.py. *Cost:* About 40 lines inside run().
3. Key the default output path to the dataset, so a single-dataset run writes creative_hybrids_v2_<dataset>.json and only --all writes the three-dataset file. *Cost:* One line at 296.

</details>

## winning_recipe_honest — `similarity_metrics/winning_recipe_honest.py`

**DECIDED.** Delete. similarity_noise_cleaning already splits at molecule level, which is what this file set out to do. Correct the two documents quoting 0.410 and 0.687.

PARTIAL · hybrid: hand-rolled · compute: fine · superseded

Takes creative_hybrids_v2's winning stacked space and re-grades its structure number after splitting the 8000 cached MCS pairs 50/50: the descriptor drop and the atompair/torsion selection are fitted on one half, the distance-versus-MCS correlation is reported on the disjoint half. Compares that space against ecfp4_count with Tanimoto and ecfp4 with Tanimoto on two datasets, and writes both the in-sample and the held-out structure number side by side.

**Q.** Once features are chosen on one half of the MCS pairs and structure is scored on the other half, does the stacked space still beat ecfp4_count with Tanimoto on distance-versus-MCS agreement?
**Q.** How much of the in-sample structure gain was inflation from choosing and scoring on the same pairs?
**Q.** Does the stacked space also beat the two single-representation Tanimoto baselines on the 5-nearest-neighbour activity Spearman?

*On re-read:* winning_recipe_honest.json does not exist. It is not in tests/experiments/similarity_metrics/, and git ls-files lists only the .py file.

Why:
- The 50/50 split at lines 36-40 permutes the 8000 MCS pairs, not the molecules, so every molecule sits in both halves and the feature choices have already seen every molecule they are graded on.
- The docstring at lines 8-9 calls the resulting number the REAL answer, and two documents repeat it as a held-out result.
- Line 71 scores weights=[0.5, 1, 1, 1] and labels the row WINNING, while winning_recipe_vs_standard.py:98 labels the equal-weight space WINNING. On herg_ki and adme_solubility the equal-weight space is the one that tops creative_hybrids_v2.json.
- The returned payload at lines 83-84 holds dataset, n, rows, desc_kept and desc_total. It records neither the weighting, nor whether the 3D block was present, nor the split seed, nor how many pairs were in each half.

<details><summary>Options considered</summary>

1. Delete the file, and replace the numbers in the two documents with the ones already on disk. *Cost:* No stored result is lost, because none exists.
2. Keep the file and add the one comparator it has that similarity_noise_cleaning lacks: ecfp4 with Tanimoto, the field default, at line 68. *Cost:* One entry in the combos handling at similarity_noise_cleaning.py:216, plus 10 nearest-neighbour runs per dataset.

</details>

## winning_recipe_vs_standard — `similarity_metrics/winning_recipe_vs_standard.py`

**DECIDED.** Fix the output path, then add the field-default comparator the noise-cleaning script lacks.

PARTIAL · hybrid: hand-rolled · compute: fine · live

The headline table: builds the equal-weight stacked space (count fingerprint + descriptors-minus-saboteurs + top-100 atompair/torsion features + 49-feature 3D pharmacophore) and scores it against three single-representation comparators, ecfp4 with Tanimoto, ecfp4_count with Tanimoto, and ecfp4_count as a z-scored Euclidean space, on the 5-nearest-neighbour activity Spearman and on distance-versus-MCS agreement. Writes winning_recipe_vs_standard.json.

**Q.** Does the stacked space beat the two field-standard fingerprint similarity metrics on the 5-nearest-neighbour activity Spearman?
**Q.** Does it beat them on the correlation between pairwise distance and MCS distance?
**Q.** Is any gain the added blocks rather than the switch from Tanimoto to z-scored Euclidean?

*On re-read:* The three bugs are confirmed. Two corrections.

Why:
- desc_blacklist_greedy at line 73 and select_fine2d at line 75 are fitted on the full pair set I,J,mcs, and every score_space call at lines 89-101 grades structure on that same pair set. On herg_ki the drop keeps 65 of 208 descriptors.
- The docstring at lines 29-32 presents both yardsticks as honest.
- make_D is imported unchanged at line 55, so the z-score means behind the winning row come from all molecules including the scaffold folds the activity number holds out.
- Line 174 uses the same default output path for one dataset and for --all. The stored file holds herg_ki only, while creative_hybrids_v2.json holds all three for the same spaces.

<details><summary>Options considered</summary>

1. Build the winning row through the fit-half machinery in similarity_noise_cleaning.py — _scaffold_halves at line 89, _pairs_within at line 108, _make_D_fit at line 131 — and pass desc[fit_idx] to desc_blacklist_greedy. Report the in-sample and held-out structure numbers side by side, keeping all three comparators. *Cost:* About 25 lines inside run().
2. Key the default output path to the dataset, so a single-dataset run writes winning_recipe_vs_standard_<dataset>.json and only --all writes the combined file. Also catch any exception per dataset, not only FileNotFoundError. *Cost:* Two lines, 171 and 174.
3. Fold the three comparators into similarity_noise_cleaning.py's combos, make its _score_arm score every space rather than the baseline plus the top-3 union, and delete this file. *Cost:* About 30 lines across two functions.

</details>

## structural_embeddings — `similarity_metrics/structural_embeddings.py`

**DECIDED.** Fix create_map4 (one line in src/kirby/representations/molecular.py), add the molecule-size control, run all three datasets before quoting anything.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Builds MAP4, TDA and mol2vec matrices for one dataset, turns each into 2 distance matrices, and scores every distance matrix on two numbers: the Spearman of activity predicted from the 5 nearest neighbours under scaffold-grouped GroupKFold, and the Spearman between its pairwise distance and the cached MCS distance. The comparison arm is ecfp4_count with Tanimoto, reported on its own, never averaged.

**Q.** Does a MAP4, TDA or mol2vec distance space predict activity from a molecule's five nearest neighbours better than ecfp4_count with Tanimoto?
**Q.** Does any of those three track shared-substructure (MCS) distance better than ecfp4_count with Tanimoto?
**Q.** Do the three KIRBy builders create_map4 / create_tda / create_mol2vec run at all in this environment?

*On re-read:* Two corrections. (1) The audit says create_map4 fails because the installed map4 package exposes only map4.map4.MAP4 and no MAP4Calculator.

Why:
- create_map4 raises ImportError for every caller in the repo, and 'map4' is a dispatchable representation name in tests/find_best_hybrid_molecular.py:602 and 'drug_map4' in tests/find_best_hybrid_dta.py:113, so a whole representation is unreachable.
- The MAP4 arm's structural number is molecule size and nothing else: measured this session at full herg_ki n=1415 over the 8000 cached pairs, map4:tanimoto scores -0.406 against MCS distance raw and +0.034 once the pair's combined heavy-atom count is partialled out of both sides.
- There is no full-n payload for this script at all; structural_embeddings_quick.json is 400 molecules of herg_ki, and its MAP4 headline is the artifact above.
- The output path at line 280 is fixed with no --out, so any --quick run overwrites the only payload on disk.

*Smoke:* ran clean. exit 0, ~260s.

<details><summary>Options considered</summary>

1. Change src/kirby/representations/molecular.py:3845 from `from map4 import MAP4Calculator` to `from map4 import MAP4 as MAP4Calculator`. *Cost:* One line.
2. Add the size-controlled structural number to score_metric, add --out, and refuse a Spearman under 10 pairs. Then run all three datasets at full n for the first time. *Cost:* About six lines.
3. Cache the TDA and mol2vec matrices per dataset the way shape_similarity.py:57-91 caches its conformers. *Cost:* About twenty lines, copied from build_spatial_reps.

</details>

## structural_yardstick — `similarity_metrics/structural_yardstick.py`

**DECIDED.** Take the rewrite. It removes 9 minutes of repeated MCS work per dataset, puts logd in the suite row space, and reports the size control that changes the answer.

PARTIAL · hybrid: none needed · compute: fine · superseded

Samples 6000 random molecule pairs per dataset, computes an RDKit MCS distance for each, and for 47 representation-and-coefficient combinations correlates that combination's pairwise distance against the MCS distance and against the activity gap. Ranks the 47 by the MCS correlation.

**Q.** When the judge is shared substructure (MCS) rather than activity, which representation-and-coefficient pair ranks first?
**Q.** Where do ecfp4:tanimoto and ecfp4_count:tanimoto land in that ranking?
**Q.** Does a distance that tracks shared substructure also track the activity gap, or do the two judges disagree?

Why:
- The rank table's headline does not survive a molecule-size control. Raw, euclidean holds ranks 1 to 8.
- The half of that headline about count fingerprints does hold: ecfp4_count:tanimoto is rank 30 raw and rank 36 controlled.
- Line 78 loads sim_<name>.npz directly instead of _data.load_prepared, so logd is 5039 molecules here and 1427 everywhere else in this suite. The I and J at line 83-85 are a third row numbering, and mcs_pairs_logd.npz tops out at index 1426.
- It recomputes 6000 MCS distances per dataset per run, while mcs_pairs_<ds>.npz already holds 8000 under the same definition on load_prepared numbering, with the absolute activity gap stored as 'dy' as well.

*Smoke:* ran clean. exit 0, ~8s.

<details><summary>Options considered</summary>

1. Rewrite run(): load through _data.load_prepared, take the pairs from mcs_pairs_<ds>.npz instead of drawing and computing new ones, add the size-controlled correlation beside the raw one, and add --out. *Cost:* MCS computations drop from 6000 per dataset to zero.
2. Leave the row space alone. Add only the size-controlled correlation, --out, and a fallback when ecfp4 is missing. *Cost:* About ten lines.
3. Delete the file and rebuild the 47-metric rank table inside the framework, repointing mcs_cache.py:44 at kirby.similarity.mcs.mcs_distance. *Cost:* Cost not established — the framework has no equivalent rank-table script to point at.

</details>

## shape_similarity — `similarity_metrics/shape_similarity.py`

**DECIDED.** Both fixes in one pass and re-run all three. Seconds, because the conformers are cached.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Builds USR (12 features), USRCAT (60) and a 3D pharmacophore descriptor (49) on ETKDG conformers, makes a cosine and a z-scored euclidean distance matrix from each, and scores all six plus the ecfp4_count Tanimoto baseline on activity predicted from the 5 nearest neighbours under scaffold-grouped GroupKFold and on Spearman against the cached MCS distance. Caches the three matrices to sim_<ds>_spatial.npz.

**Q.** Does a 3D shape or pharmacophore distance predict activity from a molecule's five nearest neighbours better than ecfp4_count with Tanimoto?
**Q.** Does one of them track shared-substructure (MCS) distance better than that baseline?
**Q.** Does the choice of coefficient, cosine on raw features versus euclidean on z-scored features, change which 3D descriptor wins?

*On re-read:* Two of the audit's four points are real as code but have never happened. (1) Failed-conformer rows kept as all zeros: I counted them in all three cached files this session.

Why:
- The structural claim for pharm3d is mostly molecule size. Measured this session over the cached MCS pairs: on herg_ki pharm3d:euclidean is 0.152 raw and 0.111 size-controlled against a baseline of 0.045 raw and 0.089 controlled, so its advantage falls from +0.106 to +0.022.
- The control does not always shrink a number, so this is not a blanket discount: on adme_solubility the baseline rises from 0.301 to 0.381 while pharm3d:euclidean falls from 0.193 to 0.143.
- shape_similarity.json holds herg_ki and logd only, even though sim_adme_solubility_spatial.npz already has all 2173 conformers built.
- One pooled Spearman per distance space, so nothing in the payload says whether a gap of +0.02 or +0.10 is larger than the fold-to-fold spread.

*Smoke:* ran clean. exit 0, 0.4s of compute.

<details><summary>Options considered</summary>

1. Add the size-controlled MCS correlation and struct_fidelity_defined to score_metric, then re-run herg_ki, logd and adme_solubility. *Cost:* About six lines.
2. Also report the per-fold spread by scoring each GroupKFold fold separately inside this file, instead of reading one pooled number back from simeval.eval_knn_readacross. *Cost:* About fifteen lines: a GroupKFold loop and a KNeighborsRegressor with metric='precomputed', which is what simeval.py:155-166 already does.

</details>

## aug3d_similarity — `similarity_metrics/aug3d_similarity.py`

**DECIDED.** Add the molecule-size control and re-run. The e3fp coefficient conclusion reverses on all three datasets without it.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Builds E3FP (1024 bits) and the Coulomb-matrix eigenvalue spectrum (50 features) on ETKDG conformers, makes tanimoto/cosine/euclidean spaces from E3FP and cosine/euclidean from Coulomb, and scores all five plus the ecfp4_count Tanimoto baseline on activity predicted from the 5 nearest neighbours under scaffold-grouped GroupKFold, on the negative RMSE of that same prediction, and on Spearman against the cached MCS distance. Caches to sim_<ds>_3daug.npz.

**Q.** Does an E3FP or Coulomb-eigenvalue distance predict activity from a molecule's five nearest neighbours better than ecfp4_count with Tanimoto?
**Q.** Does either track shared-substructure (MCS) distance better than that baseline?
**Q.** Which coefficient, tanimoto, cosine or z-scored euclidean, makes the best distance out of the same 3D representation?

*On re-read:* Same two corrections as shape_similarity, checked separately. (1) All-zero rows from failed conformers: sim_herg_ki_3daug.npz, sim_logd_3daug.npz and sim_adme_solubility_3daug.npz have 0 all-zero rows out of 1415, 1427 and 2173, for both e3fp and coulomb.

Why:
- The file's largest structural number is a molecule-size effect, and controlling for it inverts which coefficient wins. Measured this session over the cached MCS pairs on herg_ki: e3fp:euclidean is 0.512 raw and 0.110 size-controlled, while e3fp:tanimoto is 0.104 raw and 0.202 controlled and e3fp:cosine is 0.102 raw and 0.199 controlled.
- The same inversion holds on the other two datasets. On logd e3fp:euclidean goes 0.485 to 0.205 while e3fp:tanimoto goes 0.386 to 0.432.
- One pooled Spearman and one pooled RMSE per distance space, so no fold-to-fold spread exists in the payload.
- The reuse test at line 95 is have_n >= build_n, and build_n is max(need_n, have_n) two lines above, so it can only be true when the early return at line 86 has already fired, unless a representation is missing from the cache file.

*Smoke:* ran clean. exit 0, 0.5s of compute.

<details><summary>Options considered</summary>

1. Add the size-controlled MCS correlation to score_metric, then re-run all three datasets. *Cost:* About four lines, calling _partial_spearman at find_best_hybrid.py:1656 with the pair distance, the MCS distance and atoms_i + atoms_j.
2. Also score each GroupKFold fold separately inside this file rather than taking one pooled pair of numbers from simeval.eval_knn_readacross. *Cost:* About fifteen lines, mirroring the GroupKFold and KNeighborsRegressor loop at simeval.py:155-166.
3. Leave the numbers alone and delete the three dead lines: the reuse test at line 95, the no-op re-index at line 195, and the module-scope logging.disable at line 46. *Cost:* Three lines.

</details>

## learned_embeddings — `similarity_metrics/learned_embeddings.py`

**DECIDED.** All three, before any run of the MoleculeACE battery. The structural arm currently trains on about 140 triplets.

PARTIAL · hybrid: none needed · compute: fine · live

Two things, neither of which scores anything. main() walks every sim_*.npz under the cache directory and writes a mean-pooled ChemBERTa and MoLFormer embedding per dataset to emb_<ds>_<which>.npy.

**Q.** What mean-pooled ChemBERTa and MoLFormer embedding does each molecule in each prepared dataset get, cached for the battery to score later?
**Q.** Can a distance be learned from triplets, with positives defined either by shared scaffold or by smaller activity gap, fitted on training scaffolds and applied to held-out ones?

*On re-read:* The starvation is confirmed and I measured it directly rather than from a probability. On the real GroupKFold training folds of sim_herg_ki.npz, structural mode keeps 124, 142, 140, 147 and 156 triplets out of about 19,700 valid draws; activity mode keeps about 19,700 every fold.

Why:
- A 1024 to 512 to 256 to 128 network is fitted on about 140 triplets whenever mode='structural'. If the filter at line 86 ever keeps nothing, torch.randperm(0) at line 99 makes the loop at line 100 empty, and the function returns a randomly initialised encoder with no message.
- sample_triplets is called once at line 94, before the epoch loop, so all 40 epochs reuse the same triplet set instead of drawing new ones.
- The two callers scale their input differently. run_battery_moleculeace.py:100 fits on z(base[tr]) and line 101 applies the encoder to z(base[te]), each standardised by its own fold's mean and standard deviation, so the held-out fingerprints arrive on a different scale from the one the encoder was fitted on.
- Nothing records how many triplets survived, so a run on 140 and a run on 20,000 leave the same trace.
- **Blocking** `similarity_metrics/learned_embeddings.py:79` — In structural mode a triplet is kept only when the anchor shares a scaffold group with exactly one of two uniformly random other molecules.

*Smoke:* ran clean. exit 0, ~25s, no results file.

<details><summary>Options considered</summary>

1. Rewrite sample_triplets for mode='structural': pick an anchor, pick its positive from that anchor's own scaffold group, pick its negative from any other group. *Cost:* About ten lines: build a group-to-members index once, then draw within and outside it.
2. Raise when the surviving triplet count is below a floor, and return the count alongside the transform. *Cost:* Three lines here, one line in each of the two callers.
3. Move the standardisation inside train_triplet: fit the mean and standard deviation on the training rows, and apply the same ones inside transform. Then drop the z() wrappers at run_battery_moleculeace.py:100 and 101. *Cost:* About six lines here and two lines in run_battery_moleculeace.py.

</details>

## extra_fingerprints — `similarity_metrics/extra_fingerprints.py`

**DECIDED.** Build MAP4, add adme_solubility, store per-fold values, re-run all three. 120 kNN evaluations, zero model fits.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Builds Avalon and SECFP6 bit fingerprints plus the native MHFP6 MinHash vector for herg_ki, turns each into a distance matrix, and scores each against count-ECFP4 Tanimoto on two yardsticks: scaffold-grouped five-nearest-neighbour activity prediction, and Spearman against cached maximum-common-substructure distance. No model is fitted.

**Q.** Does an Avalon, SECFP6 or native MHFP6 MinHash distance predict activity from a molecule's five nearest neighbours better than count-ECFP4 Tanimoto?
**Q.** Do those distances track maximum-common-substructure distance better than count-ECFP4 Tanimoto?
**Q.** Which of Avalon, MHFP6 and MAP4 can be built in this environment at all?

*On re-read:* Two audit claims do not match what is on disk. First, extra_fingerprints.json says "map4_feasible": false, not true.

Why:
- `map4_feasible` at line 289 records whether an import worked, not whether a fingerprint was built. Nothing in this file builds a MAP4 fingerprint.
- MAP4 is now buildable here with no extra package, so the question "which of Avalon, MHFP6 and MAP4 can be built" has a third yes and no scored row.
- One Spearman per metric is stored, pooled over all five folds. The stored gaps are 0.02 to 0.05 and there is no fold-to-fold spread to place them against.
- adme_solubility has a maximum-common-substructure pair file here and has never been run.

*Smoke:* ran clean. exit 0, ~25s.

<details><summary>Options considered</summary>

1. Build MAP4 and score it with tanimoto and cosine, mirroring the Avalon block at lines 204-212. *Cost:* About 8 lines, shaped like build_avalon at 93-97.
2. Delete the map4 probe at lines 70-75 and the map4_feasible key at line 289. *Cost:* Two lines.
3. Add adme_solubility to the dataset default at line 273. *Cost:* One line.

</details>

## augmented_descriptor_reps — `similarity_metrics/augmented_descriptor_reps.py`

**DECIDED.** Two one-line fixes: consult the cache before the package probe, and z-score per fold.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Builds PDV (200 RDKit descriptors), Mordred (1613) and PaDEL descriptor blocks, z-scores each, and scores cosine and Euclidean distance on them against count-ECFP4 Tanimoto on the same two yardsticks as extra_fingerprints. Built matrices are cached to sim_<ds>_augmented.npz.

**Q.** Does a z-scored physicochemical descriptor block under cosine or Euclidean distance predict activity from five nearest neighbours better than count-ECFP4 Tanimoto?
**Q.** Does it track maximum-common-substructure distance better?
**Q.** Which of PDV, Mordred and PaDEL can be built here?

*On re-read:* The audit is right that line 188 skips a representation before the cache is read at line 194. It is inert on this machine: mordred 1.2.0 is in site-packages, so the cache is reached.

Why:
- The per-feature mean and standard deviation at line 210 come from all n molecules. Scoring at line 110 then judges held-out folds inside that same standardised geometry.
- A cached descriptor block cannot be reached when its package import fails, which is the one situation the cache exists for.
- `--quick` bypasses the cache, so the smoke rebuilds 1613 Mordred descriptors and does not finish in six minutes.
- One Spearman per metric, pooled over five folds, with no per-fold value stored.

*Smoke:* not run. exceeded 6 min, killed at '84%|########3 | 251/300' of the Mordred descriptor build (exit 142, SIGALRM at 330s).

<details><summary>Options considered</summary>

1. Split the molecules into a fit half and a held-out half by whole scaffold group. Take the mean and standard deviation from the fit half. *Cost:* Rewrites run().
2. Change nothing and record in the write-up that the descriptor distances are standardised over every molecule while the Tanimoto baseline fits nothing. *Cost:* Nothing re-runs.
3. Read the cache before the package check: let `ok` at line 188 decide only whether to BUILD, not whether to reach the representation. *Cost:* One line.

</details>

## nary_similarity — `similarity_metrics/nary_similarity.py`

**DECIDED.** NaN, symmetry and provenance fixes. The whole file re-runs in under 5 seconds.

**Still open:** How the roughness row is handled — it is a restatement of its own target.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Implements iSIM set cohesion, per-molecule complementary similarity and an iCliff-style local roughness on ecfp4_count, then tests whether complementary similarity or roughness ranks the molecules whose activity disagrees with their neighbours, and whether inflating distances to high-outlierness neighbours raises five-nearest-neighbour activity prediction.

**Q.** Do complementary similarity or iCliff roughness rank the molecules whose activity disagrees with their structural neighbours better than activity extremeness or mean distance?
**Q.** Does inflating the count-fingerprint distance toward high-outlierness molecules raise the Spearman of five-nearest-neighbour activity prediction without lowering maximum-common-substructure fidelity?

Why:
- roughness_iCliff at line 189 and the ranking target at line 190 are the same quantity in the same distance matrix — the closeness-weighted mean activity gap to the 8 nearest neighbours, against the gap to the median of the 5 nearest. Its +0.78 on herg_ki and +0.67 on logd are not evidence about finding activity cliffs.
- complementary_sim is NOT circular. It comes from the fingerprint feature sums at lines 96-108, never from the distance matrix or the labels.
- The re-weighted distance at line 209 is built from an outlierness min-max scaled across every molecule at line 110. It is then scored by GroupKFold against a Tanimoto baseline that fits nothing.
- Two Spearman results are still folded to 0.0 when undefined, at lines 146 and 198. Commit 49375c1 of 2026-08-26 changed that to NaN everywhere else here.
- **Blocking** `similarity_metrics/nary_similarity.py:189` — `local_roughness(D, y, k=8)` and the target `neighbor_disagreement(D, y, k=5)` at line 190 are the same quantity with a different neighbour count and aggregator, both computed in the same D.

*Smoke:* ran, NaNs present. exit 0, 0.4s.

<details><summary>Options considered</summary>

1. Build the ranking target from a different fingerprint's distance — simeval.dist_tanimoto(p['maccs']) or p['atompair_count'], both in load_prepared's ALL_REPS at _data.py:19-21. *Cost:* Three lines.
2. Keep the code and record that roughness_iCliff and the ranking target are the same quantity in the same space, so only complementary_sim and the two naive rankings are evidence. *Cost:* Nothing runs.
3. Make the two undefined-Spearman substitutions NaN with a defined flag beside each, matching the eight other files here. *Cost:* Lines 146 and 198, plus the printing at 252 and 261-262.

</details>

## triplet_similarity — `similarity_metrics/triplet_similarity.py`

**DECIDED.** Write the JSON and return the surviving triplet count before any full run.

**Still open:** Whether triplets are resampled each epoch. That changes the method.

PARTIAL · hybrid: none needed · compute: fine · orphaned

Trains a torch MLP over ecfp4_count with a triplet margin loss on train scaffolds only, in two modes (positive = same Bemis-Murcko scaffold, or positive = smaller activity gap), embeds every molecule, and scores cosine distance in embedding space against count-fingerprint Tanimoto on held-out scaffolds for both five-nearest-neighbour activity prediction and maximum-common-substructure fidelity.

**Q.** Does a triplet-trained embedding whose positives are same-scaffold pairs beat count-ECFP4 Tanimoto on held-out-scaffold five-nearest-neighbour activity prediction and on maximum-common-substructure fidelity?
**Q.** Does a triplet-trained embedding whose positives are smaller-activity-gap pairs beat the same baseline on the same two yardsticks?

*On re-read:* One audit item is wrong. It says there is no minimum-pairs check on the held-out maximum-common-substructure pairs, unlike similarity_noise_cleaning.py:79.

Why:
- main() at line 265 returns `results` and writes no file. Ten torch encoder trainings produce numbers that live only in terminal output.
- In structural mode the sampled triplets are filtered to anchor-positive pairs sharing a scaffold, at learned_embeddings.py:86. The surviving count is neither printed nor returned, so a null result for triplet_structural cannot be told apart from an encoder that barely trained.
- sample_triplets() is called once at learned_embeddings.py:94, before the epoch loop at line 98. All 40 epochs replay one fixed triplet set.
- one_split at line 81 takes a seed it never uses; GroupKFold does not shuffle. evaluate_fold at line 92 takes N and never uses it.
- **Blocking** `similarity_metrics/triplet_similarity.py:265` — main() returns `results` and writes no file.

*Smoke:* ran, NaNs present. exit 0, ~40s, no results file.

<details><summary>Options considered</summary>

1. Write `results` to JSON, and add an --out, mirroring extra_fingerprints.py:287-293. *Cost:* About 5 lines.
2. Return the surviving triplet count from train_triplet and store it per fold and per mode. *Cost:* One line at learned_embeddings.py:92 and 109, and one at triplet_similarity.py:122-124.
3. Resample triplets each epoch — move the call at learned_embeddings.py:94 inside the loop at line 98. *Cost:* One line.

</details>

## similarity_noise_cleaning — `similarity_metrics/similarity_noise_cleaning.py`

**YOURS.** Quote only the optimism gap now and delete once framework/f23 has run, or keep it. f23 answers the ranking question and scores every space instead of a top-3. Depends on f23 landing.

PARTIAL · hybrid: hand-rolled · compute: fine · superseded

Splits the molecules into a fit half and a held-out half by whole scaffold group, fits the descriptor drop-one blacklist, the fine-2D column selection and the z-score statistics on the fit half only, then scores injected-label-noise detection AUROC and recovery correlation on the held-out half for the top-3 spaces of a structure ranking and the top-3 of an activity ranking, against count-fingerprint Tanimoto. Runs the same thing again with fit == score as an in-sample arm so the optimism is visible.

**Q.** Does a similarity space selected without labels, for maximum-common-substructure fidelity, find deliberately corrupted activity values better than count-ECFP4 Tanimoto?
**Q.** Does it do so better than a space selected for activity coherence, which would be the circular choice?
**Q.** How much of the earlier fit-equals-score number was optimism?

Why:
- Half the candidate spaces never reach the noise battery. _topk at line 237 keeps 3 of the 6 non-baseline spaces per ranking, with no degenerate, broken or undefined reason.
- The verdict quotes a maximum. best_s at line 287 is the top-3 space with the highest held-out recovery, and line 289 reports that maximum minus one baseline.
- The combos at lines 182-187 leave out count+physchem_full, which creative_hybrids_v2.py:224 has, while the docstring at lines 7-9 says these are the exact combos from that file.
- json.dump runs only under --out, at lines 402-403. The two commands in the docstring at lines 58-59 print tables and write nothing.

*Smoke:* ran clean. exit 0, 303.7s — just inside the 6-minute cap.

<details><summary>Options considered</summary>

1. Delete similarity_noise_cleaning.py and keep similarity_noise_cleaning.json. *Cost:* Nothing re-runs.
2. Keep it as the optimism companion to f23 and fix the three items that move a number: score every space instead of three (the constant at line 77), add count+physchem_full to the combos at lines 182-187, and replace the max-of-3 verdict with the count of spaces beating the baseline, the shape f23 uses at its lines 314-321. *Cost:* 9 spaces x 9 sigma-by-rate cells x 3 seeds x 2 scorers = 486 neighbour evaluations per arm, up from 270 to 324.
3. Change nothing and quote only the in-sample-versus-held-out gap from the stored file, leaving the space ranking to f23. *Cost:* Nothing runs.

</details>

## sm_battery_moleculeace — `similarity_metrics/run_battery_moleculeace.py`

**YOURS.** Whether the 30-target battery runs at all. Nothing has ever been produced, so there is no re-run cost — only new compute. Run one target before sizing the job.

PARTIAL · hybrid: none needed · compute: heavy · live

For each MoleculeACE target it subsamples to 2000 molecules by whole scaffold groups, splits 5-fold GroupKFold, and on each held-out fold builds ~49 distance matrices (10 representations x their coefficients, plus transformer embeddings and two triplet encoders) and scores every one on the nine-evaluation activity battery in simeval.py plus a 3x3 noise dose/rate sweep. It then ranks metrics across targets and runs a paired Wilcoxon of each metric against ECFP4:Tanimoto.

**Q.** Which representation x coefficient makes a similarity space whose neighbourhoods best agree with measured activity, judged on held-out scaffold folds?
**Q.** Does a learned distance (ChemBERTa, MoLFormer, or a triplet encoder trained on train-fold scaffolds) beat the best classical coefficient on that same battery?
**Q.** Does each metric's neighbourhood expose injected label error, and does the ranking of metrics invert between low and high noise doses?

Why:
- It has never run. MoleculeACE is not importable here, and tests/experiments/similarity_metrics/similarity_metrics/sim_data is empty.
- The triplet encoder is standardised twice with different constants: z(base[tr]) at :100 and z(base[te]) at :101, where z at :50-52 uses each input's own mean and standard deviation.
- One held-out fold that raises discards every already-computed fold of that target, because run_target sits inside a single try at :198-202.
- Inside each held-out fold, eval_knn_readacross opens a 5-fold scaffold split (simeval.py:155-157), so a fold holding fewer than 5 scaffold groups raises.

*Smoke:* not run. not attempted, time

<details><summary>Options considered</summary>

1. Fix the four defects that change a number, then run one target on the server to get a wall time before sizing the job. Standardise on the train fold and apply those constants to the held-out fold. *Cost:* About 40 lines across run_target, aggregate and main.
2. Fix only the standardisation and the per-fold failure, and leave the statistics as written. *Cost:* About 15 lines.
3. Delete it and do not run the 30-target battery. *Cost:* 222 lines removed.

</details>

## sm_mcs_cache — `similarity_metrics/mcs_cache.py`

**DECIDED.** Add an output flag and honour SIM_DATA_DIR, and record the SMILES-ordering hash. Skip the timeout audit — 87 minutes per dataset and it changed nothing measurable.

PARTIAL · hybrid: none needed · compute: fine · live

Draws 8000 random molecule pairs per dataset on the _data.load_prepared row order and computes the maximum-common-substructure distance for each, caching I, J, mcs and the activity gap dy to mcs_pairs_<name>.npz. It is a precompute step, not an experiment.

**Q.** What is the maximum-common-substructure distance between each of a fixed set of 8000 molecule pairs, so that the ranker experiments and the structural yardstick score against identical pairs?

*On re-read:* The timeout finding is a true statement about the code and its measured effect is nil. I recomputed 300 of the 8000 cached herg_ki pairs at timeout=2: every distance matched the stored value exactly.

Why:
- build writes mcs_pairs_<name>.npz into the module's own directory at :72, with no output flag, and 14 scripts in this directory read those files.
- Those files are not in git, so a run with a smaller --pairs destroys about 12 minutes of computation per dataset with no copy to fall back on.
- mcs_distance at structural_yardstick.py:69 passes timeout=2 to rdFMCS.FindMCS and never reads the returned canceled flag.
- Pairs are drawn uniformly at random at :59-62, so all three cached sets are dominated by dissimilar pairs — mean distance 0.859 for herg_ki, 0.816 for logd, 0.848 for adme_solubility.

*Smoke:* ran clean. exit 0, 7s.

<details><summary>Options considered</summary>

1. Refuse to overwrite an existing mcs_pairs_<name>.npz unless a new --force flag is given, and honour SIM_DATA_DIR for the output the way the two dataset builders already do. *Cost:* About 3 lines.
2. On the timeout: write the measured number into the docstring and leave the value alone. *Cost:* Two lines of comment.
3. On the timeout: raise it to 30 seconds and rebuild all three pair files. *Cost:* Two of the six timed-out pairs took the full 30 seconds.

</details>

## sm_data_loader — `similarity_metrics/_data.py`

**YOURS.** Whether the three local datasets are capped to the same size. logd is subsampled to 1,427 and herg_ki is whole at 1,415, so cross-dataset comparisons are not on equal footing.

YES · hybrid: none needed · compute: fine · live

One loader that reads sim_<name>.npz and returns y, scaffold groups, SMILES, the ECFP4-Tanimoto reference matrix and ten representation matrices, with logd cut to a deterministic whole-scaffold-group subsample so the n x n distance matrices stay tractable.

**Q.** Which molecules, in which row order, does every local similarity script index over?
**Q.** Does the logd subsample keep whole Bemis-Murcko scaffold groups together, so no scaffold straddles the subsample boundary?

*On re-read:* The silent skip at :73 cannot bite today. I read the keys of all three npz files and every one holds all ten representation names in ALL_REPS, so no caller can hit a missing key.

Why:
- The three datasets are compared at three different molecule counts — 1415, 1427 and 2173 — and every neighbourhood-based score in this directory depends on that count.
- The self-test at :83 loops over herg_ki and logd only, so running the file does not mention adme_solubility even though sim_adme_solubility.npz sits beside it.
- The docstring at :4-7 describes a two-dataset world.

*Smoke:* ran clean. exit 0, ~15s.

<details><summary>Options considered</summary>

1. Change the test at :60 from name == 'logd' to a size test, so any dataset above CAP_N gets the same scaffold subsample. *Cost:* One line.
2. Leave the counts as they are, add adme_solubility to the self-test at :83, and say the three sizes in the docstring. *Cost:* Three lines.

</details>

## sm_nanrank — `similarity_metrics/_nanrank.py`

**DECIDED.** Leave it. Twelve scripts import it and all three functions do what their names say.

YES · hybrid: none needed · compute: fine · live

Three functions that keep an uncomputable Spearman from winning a ranking or poisoning an average: is_defined, rank_desc (a sort key sending non-finite values last) and nanmean (mean over the finite entries, NaN when none are finite), plus n_defined for the sample size behind a nanmean.

**Q.** Does a correlation that could not be computed stay distinguishable from a measured zero correlation?
**Q.** Does it lose every ranking and stay out of every average, rather than sitting at the top because max() never rejects a NaN?

Why:
- The docstring at :50-51 describes np.nanmean warning on an all-NaN input. The early return at :53 makes that unreachable, and by the time :55-57 runs, a[np.isfinite(a)] is never empty.

*Smoke:* ran, NaNs present. exit 0, instant, no results file.

<details><summary>Options considered</summary>

1. Reword those two docstring lines, or leave them. *Cost:* Two lines.

</details>

## sm_build_local — `similarity_metrics/build_local_datasets.py`

**DECIDED.** Deduplicate by InChIKey to match the framework loader, and record the source CSV and the SMILES-ordering hash.

YES · hybrid: none needed · compute: fine · live

Rebuilds sim_herg_ki.npz and sim_logd.npz from two cached CSVs, importing featurize, scaffolds and tanimoto from build_moleculeace so the local pilot and the MoleculeACE battery cannot drift apart.

**Q.** Do the two local pilot datasets reproduce from ~/.cache/kirby CSVs with the same ten representations, scaffold groups and Tanimoto reference the MoleculeACE builder writes?

*On re-read:* The duplicate-molecule finding is real, small, and confined to one dataset. Measured: chembl_herg_ki.csv holds 1415 rows and 1412 distinct molecules by InChIKey, so three molecules appear twice, with activity gaps of 0.16, 0.58 and 0.18 pKi.

Why:
- Three molecules appear twice in sim_herg_ki.npz. Each sits at distance zero from its twin in every space, is its own nearest neighbour, and gets its twin's label as its neighbour-predicted activity.
- scaffolds(mols) runs at :49 for the npz and again at :51 for the printed line, so Bemis-Murcko perception over 5039 logd molecules runs twice.
- SPECS[name] at :29 raises a bare KeyError on a mistyped dataset name, with no list of valid names.
- The npz records no source path, no build date and no hash of the SMILES order, which is the same gap mcs_cache.py's own docstring warns about for its pair files.

*Smoke:* not run. not attempted, time

<details><summary>Options considered</summary>

1. Deduplicate by InChIKey, matching what tests/find_best_hybrid_molecular.py:498-504 already does for the framework. *Cost:* About 5 lines, plus a 12-minute rebuild of the herg_ki pair file, plus re-running the 14 scripts that read it — 16 JSON files in this directory.
2. Keep all 1415 rows and store the duplicate count in the npz so it is visible to anything reading it. *Cost:* Two lines.
3. Fix the double scaffold call at :49/:51 and give the KeyError at :29 a message listing the valid names. *Cost:* Two lines.

</details>

## sm_build_adme — `similarity_metrics/build_adme_solubility.py`

**DECIDED.** Merge into build_local_datasets and delete. Same question, and the stricter filter changes no stored row.

YES · hybrid: none needed · compute: fine · live

Builds sim_adme_solubility.npz from polaris_adme_fang.csv, dropping rows with a missing or non-finite LOG_SOLUBILITY and rows RDKit cannot parse, using the same imported featurisation as the other two builders.

**Q.** Does a third local dataset, log solubility, exist in the same schema as the herg_ki and logd pilot so the metric grid can be run on it?

*On re-read:* sim_adme_solubility.npz holds one duplicated molecule, not zero. Its two rows differ by 0.099 in log solubility.

Why:
- This file and build_local_datasets.py are the same 25 lines with one difference. This one drops 'NaN' and 'nan' strings and non-finite floats at :32 and :38-39; the other checks only None, empty and 'NA' at :33-34.
- One molecule appears twice in sim_adme_solubility.npz, so it is its own nearest neighbour in every space.
- The npz records no source path, no build date and no hash of the SMILES order.

*Smoke:* ran clean. exit 0, ~110s.

<details><summary>Options considered</summary>

1. Add adme_solubility to build_local_datasets.SPECS, move the stricter blank filter into that builder, and delete this file. build_local_datasets.py supersedes it. *Cost:* About 10 lines moved, 57 lines deleted.

</details>

## sm_build_moleculeace — `similarity_metrics/build_moleculeace.py`

**DECIDED.** Take the activity-column fallback fix. One line, no rebuild.

PARTIAL · hybrid: none needed · compute: fine · live

Discovers the benchmark CSVs inside an installed MoleculeACE package, reads SMILES and the pActivity column, computes its own Bemis-Murcko scaffold groups instead of using MoleculeACE's split column, and writes one npz per target in the local pilot's schema. It also holds featurize, scaffolds and tanimoto, which the two local builders import.

**Q.** Does each MoleculeACE target become an npz with our own scaffold groups and the same ten representations, so the battery can be run on 30 targets instead of two?
**Q.** How many targets actually survive the parse and the 100-molecule floor?

*On re-read:* Two corrections. First, the activity-column fallback at :97 is less arbitrary than described: both 'y' and 'y [pEC50/pKi]' match its test, it takes the last matching field, and on the header documented at :10 that is 'y [pEC50/pKi]' — the intended one.

Why:
- find_csvs, build and main have never run. MoleculeACE is not importable here and the output directory is empty, so the 30 targets at :3-4 is a claim about the package, not a count.
- featurize, scaffolds and tanimoto are the source of every representation in all three stored datasets, so changing them invalidates those npz files and everything computed from them.
- The descriptor block at :61-69 writes 208 features positionally and stores no names. feature_ablation.py:59-61 regenerates the same list from Descriptors._descList at read time, so an RDKit upgrade that reorders or extends it mislabels every descriptor finding with no error.
- Ipc reaches 8.5e16 in sim_herg_ki.npz against 3.3e3 for the next-largest descriptor. Nothing clips or log-transforms it.

*Smoke:* not run. not attempted, time

<details><summary>Options considered</summary>

1. Store the 208 descriptor names in the npz, and have feature_ablation read them from the file when present, falling back to the live list when absent. *Cost:* One line in featurize, a few in feature_ablation.
2. Carry cliff_mol into the npz and write the skipped-target list to a file. *Cost:* About 4 lines.
3. On Ipc: leave it and note the 2 outlier molecules, or log-transform that one feature. *Cost:* Leaving it costs nothing.

</details>


---

# Federated — 10 files

## fed01 — `federated/fed01_noise_heterogeneity.py`

**DECIDED.** Pass the metric, and give the three noise seeds three different draws. Today all nine runs flip the same labels.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

On hERG binary activity it flips a rising fraction of training labels across five K-Means clusters of Morgan fingerprints, then compares seven conditions on clean test labels over 5 scaffold-group folds: a centralized hybrid, a pooled Morgan-fingerprint model, three FLuID prediction-level ensembles, KIRBy federated feature selection, and the same with per-silo noise detection. It also scores eight noise detectors against the true flip mask and correlates each silo's noise level with the budget it received.

**Q.** As the fraction of flipped training labels rises unevenly across five clusters, does KIRBy federated feature selection (C5) hold MCC on clean test labels better than FLuID's three prediction-level ensembles (C2, C3, C4)?
**Q.** Does running noise detectors inside each silo and rebuilding the hybrid on the surviving rows (C6) recover MCC over C5?
**Q.** Do the eight detectors find the flipped labels, and does a silo with more flipped labels receive a smaller share of the 100-feature budget?

*On re-read:* The audit's section 1.2 says that passing a model with no metric makes every representation score minus infinity and the budget fall back to an equal split. That is not what happens today.

Why:
- C5 asks for a budget split under the swept model but never says which metric to score it with, so all five of its per-silo hybrid builds fail on every noise level and every fold.
- 2,475 of fed01's 3,240 hybrid builds per dataset are on that path: 5 silos x 6 noise levels plus 5 smooth factors x 5 silos, per fold, over 5 folds and the 3-model x 3-seed sweep.
- The recipe that configures C0, C5 and C6 is fetched once at fed01:144-153 from a search on the clean labels of every row, so the KIRBy conditions were configured with access to both the labels the experiment then flips and every fold's test rows.
- inject_noise_per_silo is called at fed01:249, 608 and 686 with no seed, so all nine runs of the sweep flip exactly the same labels and the reported std is a model-and-fold spread, not a noise spread.

*Smoke:* not run. exceeded 6 min (killed at 340s) at last log line 'herg: 200 samples (pre-binarized), 97 active, 103 inactive' followed by a TensorFlow cpu_feature_guard line and then silence.

<details><summary>Options considered</summary>

1. For the C5 failure: give federated_feature_selection metric_fn and lower_is_better parameters and forward them to its create_hybrid call, then pass fed01's metric_fn from line 411 and line 738. Both values already exist unused at fed01:142. *Cost:* About 6 lines across fed01 and federated_helpers.py.
2. For the C5 failure, the cheaper alternative: drop model=downstream_model_instance from federated_helpers.py:1106 and pass internal_evaluator=resolved_evaluator instead. *Cost:* One line.
3. For C0 and C6: replace the bare create_hybrid at fed01:313 and 530 with _shared.build_fold_hybrid, passing the blueprint, groups[train_idx], and the model and metric together. *Cost:* About 15 lines replaced by about 8, twice.

</details>

## fed02 — `federated/fed02_rep_diversity_silos.py`

**DECIDED.** Route C4 and C5 through build_fold_hybrid, pass C6 the metric, and fix the duplicate silo assignment on caco2.

**Still open:** Whether C5 stays. It picks its three representations using the test fold's own scores.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Assigns five molecular representations to five silos that all see the same molecules, then over 5 scaffold-group folds compares each silo alone, two FLuID prediction-level ensembles over those five models, a hybrid over all five representations, a hybrid over the three that scored best, and a pipeline that K-Means-clusters within each representation before running the federated selection across the resulting virtual silos.

**Q.** Does a feature-level hybrid built across five representations (C4) beat the best of those five representations used alone (C1)?
**Q.** Does the hybrid beat combining the same five representations at the prediction level, by 8-nearest-neighbour similarity weighting (C2) or by inverse cross-validated error weighting (C3)?
**Q.** Does restricting the hybrid to the three best-scoring representations (C5), or clustering within each representation first (C6), beat the hybrid over all five?

*On re-read:* Same correction as fed01 to the audit's section 1.2: src/kirby/hybrid.py:2172-2186 now raises on a model passed with no metric, rather than scoring every representation minus infinity and splitting equally. That edit is uncommitted, so the committed code still does the silent equal split.

Why:
- C4, C5 and C6 all ask for a budget split under the swept model without saying which metric scores it, so every KIRBy condition fails and the experiment reduces to C1, C2 and C3.
- That is 765 hybrid builds per dataset: C4 1 + C5 1 + C6 up to 15 per fold, over 5 folds and the 3-model x 3-seed sweep.
- C5 picks its three representations by sorting single_silo at fed02:407-413, whose MCC and RMSE were computed on the test fold at fed02:261-267, so C5 chooses its inputs by looking at the answer and cannot be compared with C4.
- On caco2 silos B and E both hold avalon: the direct assignment at fed02:80-81 has no duplicate check while the fallback loop at 85 does. mordred is not in caco2's representation list, so B falls back to avalon and E is assigned avalon outright.
- **Blocking** `federated/fed02_rep_diversity_silos.py:80` — _get_silo_reps guards against a duplicate only in the fallback branch (line 85 checks 'fallback not in resolved.values()'); the direct-assignment branch at 80-81 does not check.
- **Blocking** `federated/fed02_rep_diversity_silos.py:408` — single_silo holds metrics computed on the test fold (fed02:265-267).
- **Also caught by the hybrid-usage audit** — fed02 C4 passes model= to create_hybrid with no metric_fn.
- **Also caught by the hybrid-usage audit** — fed02 C5 passes model= to create_hybrid with no metric_fn.

*Smoke:* not run. exceeded 6 min (killed at 355s) at last log line 'Cache caco2__avalon__xgboost__rmse__5f_42.pkl: meta drift on rep_shape (stored=(60, 2048), expected=(200, 2048)) - will recompute.' Caco2 loading alone took 201.6s of the 355s (2158 molecules, all 5 reps cache hits, subsampled to 200), leaving the run inside precompute_hybrid_searches when killed.

<details><summary>Options considered</summary>

1. For C4 and C5: replace the create_hybrid calls at fed02:370-379 and 421-430 with _shared.build_fold_hybrid, passing bp, groups[train_idx], _match_model, metric_fn and lower_is_better. metric_fn and lower_is_better already exist unused at fed02:115. *Cost:* About 20 lines replaced by about 12.
2. For C6: give fluid_within_kirby_across metric_fn and lower_is_better parameters and forward them to create_hybrid at federated_helpers.py:1875-1881. *Cost:* About 5 lines.
3. For C6, the alternative: drop model=downstream_model_instance from federated_helpers.py:1881 and pass internal_evaluator=resolved_evaluator instead. *Cost:* One line.

</details>

## fed03 — `federated/fed03_synthetic_private.py`

**DECIDED.** Rename the dataset key so the submit path stops exiting, pass the metric each part reports, and load herg_ki for Part A's first silo.

**Still open:** Whether Part B's silos are rebuilt from DATASET_REPS, and whether Part C stays or moves into fed05.

NO · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

One file holding three unrelated experiments dispatched by dataset name. Part A pools hERG molecules from three sources as silos and compares KIRBy per-silo feature selection against single-silo models, two FLuID prediction-averaging ensembles and a pooled model, by MCC over 5 scaffold folds.

**Q.** Part A: on hERG molecules pooled from three sources, does allocating a feature budget across silos by their own validation scores beat the best single silo, the two FLuID prediction-averaging ensembles, and the privacy-violating pooled model, measured by MCC?
**Q.** Part B: when each org holds a different family of representations, does a hybrid built across all three families beat the best single family, and does building it silo-by-silo and concatenating the selections cost anything?
**Q.** Part C: does the feature-importance profile learned on one antibody silo, applied by raw column index to another silo's representations, beat that silo's own hybrid?

*On re-read:* Five corrections. (a) Part A's C4 does not run on the 10-representation general list.

Why:
- The server has never run Part A: submit_all_datasets.sh:55 sends the name lhasa_herg, and run_dataset.py:97 exits because lhasa_herg is not a key in DATASET_REPS (only herg is).
- Part A's first silo, labelled chembl, is the Lhasa training set, so cross-silo deduplication then deletes those same molecules from the silo labelled lhasa_train and three sources become two.
- Every hybrid in this file raises today: hybrid.py:2173 refuses a model with no metric_fn, and all five create_hybrid calls (462, 750, 813, 1091, 1237) plus federated_helpers.py:1100 pass a model and no metric.
- Part A compares three different feature spaces as though they were one: C1/C2/C3/C5 use 2048-bit feature-invariant Morgan fingerprints (line 273, useFeatures=True), C4 and C7 use all 33 representations, C6 uses plain ecfp4.
- **Blocking** `federated/fed03_synthetic_private.py:1477-1481 (with run_dataset.py:97 and tests/slurm_scripts/submit_all_datasets.sh:55)` — The federated submit path submits the dataset name 'lhasa_herg' (FEDERATED_DATASETS="lhasa_herg caco2 logd").
- **Blocking** `federated/fed03_synthetic_private.py:90-93` — Part A treats data['reps']/['labels']/['smiles'] as the ChEMBL silo.
- **Also caught by the hybrid-usage audit** — fed03 Part A C6 passes model= to create_hybrid with no metric_fn.
- **Also caught by the hybrid-usage audit** — fed03 Part B C2 passes model= to create_hybrid with no metric_fn.

*Smoke:* not run. not attempted, time.

<details><summary>Options considered</summary>

1. Change one word in tests/slurm_scripts/submit_all_datasets.sh:55, from lhasa_herg to herg, so Part A can be submitted. *Cost:* Nothing goes stale.
2. Register lhasa_herg in DATASET_REPS instead, keeping the submit script as it is. *Cost:* Same as the rename: no stored result changes.
3. Pass the dataset's registered metric alongside the model at all five create_hybrid calls, and add a metric_fn parameter to federated_feature_selection. *Cost:* Two lines per call plus one parameter on federated_helpers.py:1013.

</details>

## fed04 — `federated/fed04_cluster_aware_importance.py`

**DECIDED.** Pass groups, the resolved evaluator and the searched recipe. Three small edits, no stored result to lose.

**Still open:** The hardcoded pKi 6.0 threshold. On caco2 and logd it collapses the target to one class.

PARTIAL · hybrid: engine, per fold · compute: heavy · dispatched

Splits a dataset into 5 scaffold-group folds; inside each training fold it K-Means-clusters the molecules on Morgan fingerprints, runs create_hybrid once per cluster and once on the whole fold, and compares the per-cluster feature allocations to each other and a cluster-conditional classifier to a single global one. Repeats the clustering at five smooth_factor values.

**Q.** Does the per-representation share of a fixed 100-feature budget change from one K-Means cluster of the training molecules to another, measured as mean pairwise Spearman between per-cluster importance profiles?
**Q.** Does a classifier trained per cluster on that cluster's own hybrid beat one classifier trained on a single fold-wide hybrid, in MCC/BAC/F1/AUROC on held-out molecules?
**Q.** Does the clustering smooth_factor change how stable those per-cluster profiles are?

*On re-read:* Three things in the earlier notes are wrong. create_hybrid is called twice, at fed04:89 and fed04:255; fed04:334 is cluster_training_only, not a hybrid build.

Why:
- On caco2 and logd, fed04 cuts the label into two classes at 6.0 (fed04:44, applied at fed04:150). LogD and log efflux ratio almost never reach 6.0, so every molecule lands in the same class.
- The recipe every hybrid is built from is asked for with the binary label, but the store it is served from is keyed on the augmentation setting alone and never looks at the label (_shared.py:4198). On caco2 and logd it returns a recipe searched on the original continuous label under RMSE.
- fed04:179 takes metric_name from get_domain_metric, which is 'rmse' for caco2 and logd, and passes it alongside task_type='classification'. On herg there is no in-memory store, so that pair goes into a real search (_shared.py:4212, explicit metric wins).
- Neither create_hybrid call passes groups. The fifth of the training rows that decides which representation gets features is then a random block (hybrid.py:4561) sharing scaffolds with the rows it is judging, instead of the GroupKFold fold at hybrid.py:4541.
- **Blocking** `federated/fed04_cluster_aware_importance.py:44` — PKI_THRESHOLD = 6.0 is hardcoded and applied to any non-binary label set (line 150).
- **Also caught by the hybrid-usage audit** — The logd and caco2 wrappers threshold their labels at pKi 6.0.

*Smoke:* not run. exceeded 6 min (killed at 360s) at last log line 'herg: 60 samples, 28 active, 32 inactive' plus a TensorFlow cpu_feature_guard line, then silence for ~5 minutes.

<details><summary>Options considered</summary>

1. Threshold: raise instead of thresholding when the label is not already 0/1. caco2 and logd then refuse fed04 with a named message; herg keeps running. *Cost:* About five lines at fed04:146-150.
2. Threshold: take the cut from each dataset's own label distribution — the median, or a quantile you name — and write the value into the results config block. *Cost:* About ten lines at fed04:44-150 plus one key in the config block written at fed04:405.
3. Threshold: keep the continuous label and give fed04 a regression path — fit_fed_regressor and compute_reg_metrics instead of fit_fed_classifier and compute_clf_metrics. *Cost:* The four blocks at fed04:285-326 plus the metric list at fed04:387.

</details>

## fed05 — `federated/fed05_antibody_federated.py`

**DECIDED.** cache=True and a per-silo recipe. Both one-line, nothing to lose.

**Still open:** The dms silo. abagym_HER2 is not a registry key, so condition 5 always reports a skip.

NO · hybrid: engine, per fold · compute: WILL NOT FINISH · dispatched

Loads three antibody datasets as silos, searches a hybrid blueprint per silo with find_best_hybrid, then runs six conditions on one 80/20 group-aware split per silo: a per-silo hybrid, cross-silo transfer of selected feature indices, within-silo ESM2 sub-clustering, a pooled z-scored noise-budget check, a pooled binding hybrid, and a hybrid over nine representations against an inverse-RMSE prediction ensemble of three-representation models.

**Q.** On each antibody silo separately, what does a hybrid built with that silo's own blueprint score on its held-out 20% (RMSE and Spearman)?
**Q.** Does re-using another silo's selected feature indices beat the target silo's own hybrid on that target silo's held-out rows?
**Q.** When three silos' z-scored labels are pooled and Gaussian noise is added to one, does that silo receive a smaller share of the feature budget than the clean silos?

*On re-read:* The dms silo fails for a simpler reason than the notes give. antibody_baseline_survey.py:150 already maps the short name 'abagym_HER2' onto the full loader name 'abagym_HER2_2021_trastuzumab', so the loader is reachable.

Why:
- The dms silo never loads, so condition 5 always writes 'need >= 2 binding silos' and condition 4 pools two silos instead of three.
- Each silo's recipe search at fed05:199 runs with caching off, so all nine runs of the 3-model 3-seed sweep repeat an identical search.
- The recipe is searched with augmentation_strategy='none' at fed05:206 but the hybrid is built with 'uncapped' whenever the silo has augmentations (fed05:138). The allocation reported as best was chosen over a different feature set than the one used.
- The searched budget is thrown away: create_hybrid at fed05:144 passes no budget and runs at its default 100, while the search swept all eight of FULL_BUDGETS.
- **Blocking** `federated/fed05_antibody_federated.py:69` — 'abagym_HER2' is not a key of DATASET_REGISTRY (tests/antibody_baseline_survey.py:53-106 — it was cut 2026-07-09 and only the loader was retained).

*Smoke:* not run. exceeded 6 min (killed at 300s) at last log line 'Adaptive pruning: evaluating fold 0 for 1860 configs...'.

<details><summary>Options considered</summary>

1. dms silo: add 'abagym_HER2': ('abagym', 'position_modulo') to DATASET_REGISTRY in tests/antibody_baseline_survey.py. *Cost:* One line.
2. dms silo: point SILO_DATASETS['dms'] at a binding set that is still registered — flab_binding_koenig, flab_binding_phillips_h3, flab_binding_adams, or abagym_Ang2/EGFR/VEGF. *Cost:* One line.
3. dms silo: keep two silos and say so — drop 'dms' from SILO_DATASETS and delete condition 5. *Cost:* About 95 lines removed (fed05:509-603).

</details>

## lhasa_herg_loader — `federated/lhasa_herg_loader.py`

**DECIDED.** Fix the PKL branch so it derives the label column the SDF branch already derives.

PARTIAL · hybrid: none needed · compute: fine · live

Reads Lhasa's curated hERG training (7,772 molecules) and test (4,383) sets from PKL or SDF, extracts SMILES from the stored RDKit Mol objects, and returns SMILES, binary labels, InChIKeys and the DataFrame. Also provides dedup_across_silos, which drops a molecule from every silo after the first by InChIKey.

**Q.** Which molecules, binary hERG activity labels and InChIKeys make up Lhasa's curated training and test sets?
**Q.** Which molecules appear in more than one silo, so a downstream federated split does not train and test on the same compound?

*On re-read:* The earlier note says load_lhasa_herg_test is never called and the external test set is unused. Both are called: fed03_synthetic_private.py:146 calls load_lhasa_herg_test and fed03:166 calls dedup_across_silos.

Why:
- The PKL branch returns the DataFrame untouched (lines 36-39). The rename to ACTIVITY (line 48) and the CLASS derivation (lines 70-71) exist only in the SDF branch, and line 113 then reads df_valid['CLASS'], so a PKL without that column raises KeyError.
- activity_col is threaded down to _load_pkl_or_sdf and ignored on the PKL path, so the deliberate difference between the training set's 'Lhasa Chembl Call' and the test set's 'Overall Conservative Call' has no effect whenever PKLs are present.
- Line 71 labels anything that is not the exact string 'Inactive' as active, including NaN, an empty string, and any equivocal call.
- dedup_across_silos keeps a molecule whose InChIKey is 'UNKNOWN' in every silo it appears in (lines 185-188) and only warns. That is the duplication the function exists to remove.

*Smoke:* ran clean. exit 0.

<details><summary>Options considered</summary>

1. Move the ACTIVITY rename and the CLASS derivation out of the SDF branch so both formats run them, and raise a named error when neither CLASS nor the activity column is present. *Cost:* About twelve lines in _load_pkl_or_sdf.
2. Label rule at line 71: map only the exact strings you name to active and to inactive, and raise on anything else instead of defaulting to active. *Cost:* About five lines.
3. dedup: drop UNKNOWN-keyed molecules from every silo after the first, or fall back to canonical SMILES as the key. *Cost:* About six lines at lines 184-193.

</details>

## caco2_runner — `federated/caco2.py`

**DECIDED.** Add --reps to run_dataset.py and delete all three runners together.

YES · hybrid: none needed · compute: fine · superseded

Forty-five-line argparse wrapper that calls federated_helpers.run_all with dataset_name='caco2', passing through --experiments, --output and an optional --reps override.

**Q.** Can every federated experiment be run against caco2 from one command, with an optional cheap representation list for local smoke runs?

*On re-read:* The earlier note says the runner offers no way to exclude fed04. It does: --experiments takes a list, so `--experiments fed01 fed02 fed03 fed05` runs everything but fed04.

Why:
- Running the default dispatches fed04, which on caco2 cuts the label at 6.0 and gets one class.
- It also dispatches fed05, which ignores caco2 entirely, loads three antibody silos, and files them as caco2.json.
- It is a 45-line copy of logd.py with one string changed, and run_dataset.py's own docstring says it replaces these per-dataset files.

<details><summary>Options considered</summary>

1. Delete caco2.py and use `python tests/experiments/run_dataset.py federated caco2`. *Cost:* 45 lines deleted.
2. Add --reps to run_dataset.py, then delete caco2.py, logd.py and herg.py together. *Cost:* About six lines added to run_dataset.py, 135 lines deleted across the three runners.

</details>

## herg_runner — `federated/herg.py`

**DECIDED.** Settle on 'herg' as the spelling, then delete with the other two.

YES · hybrid: none needed · compute: fine · superseded

Forty-five-line argparse wrapper that calls federated_helpers.run_all with dataset_name='lhasa_herg', which routes run_all to load_lhasa_herg_data instead of the normal load_data path.

**Q.** Can every federated experiment be run against Lhasa's curated hERG classification set from one command?

*On re-read:* The earlier note says `run_dataset.py federated lhasa_herg` exits 1 and concludes there is no equivalent. It exits 1, but the equivalent works: run_dataset.py validates against DATASET_REPS, which has 'herg', and run_all routes both 'herg' and 'lhasa_herg' to the Lhasa loader (federated_helpers.py:2237 and 2265).

Why:
- DATASET = 'lhasa_herg' (herg.py:21) is not a DATASET_REPS key, so the same run through run_dataset.py has to be spelled 'herg'. One dataset, two live spellings.
- In the federated category alone, 'herg' means Lhasa's curated classification set, not the TDC herg every other category loads under that name.
- On this path run_all never fills data['precomputed_searches'], so fed04 cold-builds its recipe search on the first of the nine sweep runs. That one search is cached to disk (_shared.py:4231) and the other eight read it back.
- It also dispatches fed05's three antibody silos and files them as lhasa_herg.json.

<details><summary>Options considered</summary>

1. Delete herg.py and use `python tests/experiments/run_dataset.py federated herg`. *Cost:* 45 lines deleted.
2. Settle on one spelling: either add 'lhasa_herg' as an alias in DATASET_REPS, or drop it from run_all's two checks at federated_helpers.py:2237 and 2265 so only 'herg' routes to Lhasa. *Cost:* One or two lines.

</details>

## logd_runner — `federated/logd.py`

**DECIDED.** Delete with caco2.py and herg.py once run_dataset.py takes --reps.

YES · hybrid: none needed · compute: fine · superseded

Forty-five-line argparse wrapper that calls federated_helpers.run_all with dataset_name='logd', passing through --experiments, --output and an optional --reps override. Byte-for-byte the same as caco2.py apart from the dataset name.

**Q.** Can every federated experiment be run against logd from one command, with an optional cheap representation list for local smoke runs?

*On re-read:* Same correction as caco2.py: --experiments does let you exclude fed04, it just does not warn. The file is byte-identical to caco2.py apart from DATASET = 'logd' at line 21, and nothing on the server calls it.

Why:
- Running the default dispatches fed04, which cuts logD at 6.0; only a small tail of logD exceeds that, so the target is near-constant and the MCC comparison has nothing in it.
- It also dispatches fed05, which ignores logd, loads three antibody silos, and files them as logd.json.
- It is a duplicate of caco2.py with one string changed, and run_dataset.py's docstring says it replaces these files.

<details><summary>Options considered</summary>

1. Delete logd.py and use `python tests/experiments/run_dataset.py federated logd`. *Cost:* 45 lines deleted.

</details>

## smoke_driver — `federated/_smoke_driver.py`

**DECIDED.** Set the thread variables and fix the default representation list.

PARTIAL · hybrid: none needed · compute: fine · live

Local-only launcher that refuses to run unless KIRBY_SMOKE_N is set, rebinds _shared.FULL_BUDGETS to [25], and calls federated_helpers.run_all directly with an explicit small representation list so run_dataset.py's foundation-model defaults are bypassed.

**Q.** Can the federated suite be exercised end to end on this laptop, on a few hundred rows and a handful of cheap representations, without pulling in server-only foundation models or the eight-budget precompute grid?

Why:
- It does not set KIRBY_LGBM_THREADS=1. hybrid.py:56-64 records deeplift followed by LightGBM at n_jobs=-1 as the local crash, and this file exists only for local runs.
- Nothing shrinks the recipe search fed04 triggers: it runs the engine's default model list over 5 folds. That is why fed04 cannot be smoked from here today.
- The default representation list at line 31 includes maccs and pdv, both cut from DATASET_REPS['herg']['general'] on 2026-08-24. The default smoke run exercises representations herg no longer uses.
- At 60 rows fed04 computes nothing: under smoke K drops to 2 and folds to 2, so each cluster is about 15 rows, below the 20-row floor at fed04:76. The usage example says 200; nothing in the file states the floor.

<details><summary>Options considered</summary>

1. Set KIRBY_LGBM_THREADS=1 and OMP_NUM_THREADS=1 inside the driver, before the imports, unless already set. *Cost:* Three lines after line 37.
2. Shrink the recipe search under smoke: either give get_best_search_for_downstream a models and n_folds override that the smoke config fills, or have fed04 skip that call under KIRBY_SMOKE_N and use its own defaults. *Cost:* About five lines in fed04 around its line 184, or a smoke branch at _shared.py:4220.
3. Refuse a KIRBY_SMOKE_N below the row count fed04 needs, and put the number in the message. *Cost:* Four lines at line 36.

</details>


---

# OPIG — 11 files

## opig_activity_cliff_qsar — `opig_activity_cliff_qsar.py`

**FIXED 2026-09-04 (split audit, option 3).** The kirby_hybrid representation is no longer built or scored, so the leaked row and the Hybrid MAE and Delta columns are gone. The honest_rescore table is the only hybrid number the file prints. The activity-cliff question still has no held-out hybrid answer — cliff MCC and sensitivity are computed from pooled predictions, and there are no hybrid predictions any more to pool. The sign-convention mismatch and delta_mean_of_folds are untouched.

**DECIDED.** Build the hybrid per fold and recompute the cliff numbers from those predictions. Fix the sign convention and the paired delta in the same pass.

PARTIAL · hybrid: engine, per fold · compute: heavy · live

On 3 ChEMBL targets (d2, fxa, mpro) it evaluates 7 molecular representations plus one hybrid under 5 downstream models with scaffold GroupKFold, reporting MAE/RMSE/R2/Spearman, and turns the out-of-fold predictions into a matched-molecular-pair activity-cliff MCC. The hybrid is materialised from a search result fitted on all rows.

**Q.** Which single molecular representation gives the lowest MAE on activity-cliff-rich ChEMBL targets?
**Q.** Does the hybrid representation beat the best single representation on MAE?
**Q.** Does the representation choice change how often a model misses an activity cliff (AC-MCC, AC-sensitivity)?

*On re-read:* The leak is real and I confirmed it: line 370 reads search.fitted_results['rf'], line 375 applies those feature indices to all rows, line 402 scores them with GroupKFold on the same rows. Three things in the earlier write-up are wrong.

Why:
- The kirby_hybrid row in the main table is built from feature indices fitted on all rows, then scored on folds of those same rows, so every MAE, RMSE, R2, Spearman and cliff number for it saw its own validation labels.
- The activity-cliff question has no held-out answer at all, because cliff MCC and sensitivity are computed only from those contaminated pooled predictions.
- The cross-target summary at line 529 prints hybrid minus best, so a negative number means the hybrid won; the honest table 11 lines below prints the opposite sign for the same comparison.
- The honest table prints r['delta'], a difference of two means, when honest_rescore also returns delta_mean_of_folds, the paired number its own docstring says to report.
- **Blocking** `opig_activity_cliff_qsar.py:370-381` — The `kirby_hybrid` representation is built from `search.fitted_results['rf'].selected_indices`, which find_best_hybrid fits on the full dataset with the full labels (find_best_hybrid.py:6455-6461).

<details><summary>Options considered</summary>

1. Build the hybrid inside the fold loop instead of once on all rows. Use build_fold_hybrid (_shared.py:3559) with the blueprint from get_best_for_downstream(search, model), passing groups[train_idx], the model instance and the mae metric. *Cost:* 5 models x 5 folds x 3 targets = 75 create_hybrid calls, plus 75 downstream fits.
2. Keep one rf recipe but build it per fold, so the hybrid is one extra representation in the existing loop rather than a full per-model rebuild. *Cost:* 5 folds x 3 targets = 15 create_hybrid calls, roughly 1.2 h at the 262-336 s measured figure.
3. Delete lines 370-381 and drop kirby_hybrid from the table, leaving the honest_rescore block as the only hybrid number. *Cost:* Removes 75 downstream fits and the full-data refit.

</details>

## opig_blueprint_transfer — `opig_blueprint_transfer.py`

**FIXED 2026-09-04 (split audit).** base_best, native, and both source-side arms now take their single representation and their recipe from a search run inside each held-out fold's training rows — n_folds searches per dataset instead of one, on both source and target. Each fold's picks are stored as per_fold_base_rep and per_fold_native_blueprint; the whole-dataset picks stay beside them under *_whole_data as context. recipe_transfer and full_transfer are unchanged: their recipe and feature indices come from the source dataset. The missing groups= at line 250, precomputed_alloc_scores, and the unpaired delta_vs_base are untouched.

**DECIDED.** Add groups= at line 250 and make the delta paired. The stored file is a smoke run, so nothing real is invalidated.

PARTIAL · hybrid: engine, per fold · compute: heavy · live

Searches a hybrid blueprint on a large source dataset and a small target dataset, then on the target's folds compares four arms: best single representation, the target's own blueprint rebuilt per fold, the source's recipe rebuilt on target training rows, and the source's exact feature indices applied unchanged. Also runs best-single vs hybrid on the source as context.

**Q.** Do the source dataset's exact selected feature indices, applied unchanged to the target, beat the target's best single representation?
**Q.** Does the source's recipe (allocation method, budget, importance method, evaluator), re-run on target training rows, match the target's own searched blueprint?
**Q.** Does the hybrid beat the best single representation on the source dataset itself?

*On re-read:* Every defect I checked is really there. Line 250 calls create_hybrid with no groups= and no precomputed_alloc_scores=; with no groups it carves a RandomState(42) 20% split (hybrid.py:4561).

Why:
- The source hybrid whose exact feature indices are published as the transferable result is built with no groups, so its keep-or-drop decisions are judged on a holdout that shares scaffolds with its own training rows.
- That same build does not receive source_bp.rep_scores, while native and recipe_transfer do get them through build_fold_hybrid (_shared.py:3746-3756), so full_transfer differs from the other two in how the budget was split as well as in what it is testing. **Superseded 2026-09-04 (split audit):** a blueprint's rep_scores are each representation's score over every row the search ran on, so handing them to a fold rebuild sets that fold's budget split from a number that saw its own held-out rows. build_fold_hybrid stopped falling back to them (`_shared.py:4192-4199`) and apply_blueprint stopped passing them into any labelled rebuild (P8-engine). All three arms now score their representations on their own fold's training rows, so they are already level and option 2 below is withdrawn.
- The second question the docstring asks, whether the source recipe matches the target's own, is never computed: lines 456-461 only ever subtract base_best.
- delta_vs_base subtracts two nanmeans, so a condition that lost a fold to an exception is compared over 4 folds against a 5-fold mean, with nothing recorded to say so.

<details><summary>Options considered</summary>

1. Add groups=source_groups to the create_hybrid at line 250. *Cost:* One line.
2. ~~Add precomputed_alloc_scores=source_bp.rep_scores to the same call, so the budget split comes from the same scores the other two hybrid conditions use.~~ **WITHDRAWN 2026-09-04** — see the bullet above. The other two arms no longer receive rep_scores either, so all three already score their representations on their own fold's training rows. Nothing to do here. *Cost:* one internal-evaluator fit per representation per rebuild, already paid.
3. Replace _agg and the delta block (lines 427-461) with a paired mean: per fold, subtract base_best from each condition where both scored, then average and record how many folds paired. Add recipe_transfer minus native and full_transfer minus native as their own paired numbers. *Cost:* About 25 lines.

</details>

## opig_cdr_flexibility — `opig_cdr_flexibility.py`

**FIXED 2026-09-04 (split audit, option 1).** build_honest_rescore_input then honest_rescore; the per-model held-out hybrid-versus-single table is printed and saved under honest_comparison, with the per-fold block beside it under honest_rescore. kirby_hybrid_search_score_mean is renamed search_best_score_mean and demoted to a diagnostic, so an old reader fails loudly rather than plotting a search maximum as a result. The combined CDRH3-plus-CDRL3 run is untouched and still open.

**DECIDED.** Add the honest block. No fits, and it is the only thing between this file and an answer to its third question.

**Still open:** Whether the combined CDRH3-plus-CDRL3 run stays. Its features cannot express which loop the label describes.

PARTIAL · hybrid: engine, per fold · compute: fine · orphaned

Classifies CDR loops as flexible or rigid from whole-Fv sequence representations, on CDRH3 alone, CDRL3 alone and the two combined, with 8 antibody representations x 5 models under StratifiedGroupKFold grouped by IGHV germline family. The hybrid is not materialised; only the hybrid search's own internal CV score is printed and stored.

**Q.** Which sequence-based antibody representation best separates flexible from rigid CDR3 loops (ROC AUC, PR AUC, MCC, balanced accuracy)?
**Q.** How far below the paper's structure-based ITsFlexible GNN (ROC AUC 0.82) does the best sequence-based representation land?
**Q.** Does the hybrid representation beat the best single representation on this task?

*On re-read:* The main finding stands but its stated reason is wrong. The earlier write-up says the hybrid number and the per-representation table come from different splitters.

Why:
- The hybrid's reported number is the best configuration's score under the search's own cross-validation, chosen by maximum over 5 allocation methods, 3 importance methods, 5 evaluators and 4 budgets on the very folds that produce it, printed beside single representations that each got one draw.
- That hybrid number is for rf alone, while the table beside it covers five models.
- The combined run concatenates the CDRH3 and CDRL3 row sets, but no feature says which loop the label refers to, so an antibody in both files contributes two rows with identical features and possibly opposite labels.
- Neither per-fold scores nor the winning blueprint reach the JSON, so nothing in the file can be recomputed later.

<details><summary>Options considered</summary>

1. Call build_honest_rescore_input(search) and honest_rescore on the result, store the block under 'honest_rescore', and print the per-model held-out hybrid-versus-single delta. opig_activity_cliff_qsar.py:365 and 552 already do exactly this and are the pattern to copy. *Cost:* No model fits at all.
2. Drop the combined run at main() lines 482-489. The reason is that the features cannot express the label, not compute: an antibody present in both CDR files becomes two rows with identical heavy and light chain features and possibly opposite flexibility labels. *Cost:* One search removed, which is 5 x 3 x 5 x 4 x 7 x 5 = 10,500 configuration-fold evaluations, plus 8 representations x 5 models x 5 folds = 200 fits.
3. Keep the combined run and add the loop identity as a feature. The loader already returns per-row cdr_types (itsflexible.py:180) and this file discards it. *Cost:* The combined search re-runs cold, 10,500 configuration-fold evaluations, plus 200 table fits.

</details>

## opig_cgr_comparison — `opig_cgr_comparison.py`

**FIXED 2026-09-04 (split audit, option 1).** The CGR delta is the difference of the two pools' leave-one-fold-out hybrid scores, taken from the honest blocks the file already stored, not of the two full-data hybrid_eval RMSEs. Both pool scores print above the Delta row so the subtraction is readable. The rf-only recipe scored under five models, and the any-one-model verdict at the end, are untouched.

**DECIDED.** Take the honest-block arithmetic and the per-model recipe. Neither needs a re-run.

PARTIAL · hybrid: engine, per fold · compute: heavy · live

For each of 3 reaction-yield datasets it generates ~44 reaction representations, scores every one with 5 models under GroupKFold, then runs two hybrid searches (pool without the 3 CGR reps, pool with them) and compares the two resulting hybrids. It also computes pairwise Linear CKA between the 7 family representations.

**Q.** Does the best CGR representation beat the best SMILES representation within the fingerprint family and within the neural family, by RF RMSE?
**Q.** Does adding the 3 CGR representations to the hybrid pool lower the hybrid's RMSE, per downstream model?
**Q.** Do the CGR representations get any features allocated in the SMILES+CGR hybrid, and how many?
**Q.** How similar are the CGR and SMILES representations by Linear CKA?

*On re-read:* The audit says a representation with the wrong number of rows is skipped from the individual table at :605 but still handed to find_best_hybrid at :634. It cannot be.

Why:
- The CGR-versus-no-CGR number at :660 subtracts two hybrids that each picked their features while every scoring fold was visible, so both sides are optimistic.
- The two pools are not the same size, 41 representations without the three CGR ones against 44 with them, so that optimism does not cancel in the subtraction.
- The search runs for random forest only (:357), and the single recipe random forest chose is then scored under xgboost, svm, mlp and lightgbm at :392-399.
- The printed verdict at :802 says 'Hybrid pool improves with CGR' when any one of the five models has a negative delta.

<details><summary>Options considered</summary>

1. Compute the CGR-versus-no-CGR number from the two honest blocks the file already stores, instead of from hybrid_eval. honest_rescore returns hybrid_fold_scores per model, fold-aligned (tests/experiments/_shared.py:988). *Cost:* Zero model fits.
2. Rebuild the pool comparison so each pool's recipe is chosen on training rows only, per outer fold, through nested_cv_evaluate. *Cost:* Each search becomes five searches, one per outer fold.
3. Score each downstream model on its own hybrid. search.fitted_results is a dict keyed by model name, one entry per model in best_by_model (src/kirby/find_best_hybrid.py:7056-7062). *Cost:* The full-data refit for all six models is already paid inside find_best_hybrid.

</details>

## opig_ginkgo_developability — `opig_ginkgo_developability.py`

**DECIDED.** Set metric='spearman' and match the representation preprocessing across the two arms in one edit, print the honest block, re-run once.

PARTIAL · hybrid: engine, per fold · compute: heavy · live

For each Ginkgo developability property it builds a hybrid per fold on the training rows only and scores the held-out fold, and scores every base representation on the identical folds. It uses the competition's pre-computed hierarchical cluster fold IDs as the splits.

**Q.** Which antibody representation best predicts each developability property, by Spearman rho on the competition's cluster folds?
**Q.** Does the per-fold hybrid beat the best single representation on each property?
**Q.** How do KIRBy's numbers sit against the competition's best team CV and test Spearman?

*On re-read:* Three corrections. First, the audit has the preprocessing asymmetry backwards.

Why:
- The per-fold search at :300-316 passes no metric, so find_best_hybrid uses its default r2 (src/kirby/find_best_hybrid.py:6043), while every number the file prints and ranks is Spearman rho. The registry says every ginkgo property is spearman (tests/experiments/dataset_policy.py:265-270).
- The 'Hybrid wins X/N' line at :601 takes the best model on the hybrid side and the best representation-and-model pair on the baseline side, on the same folds.
- The honest block built at :529-533 is written to JSON and never evaluated or printed anywhere in this file.
- The representation list at :76-84 is hand-picked and is not the pruned ten the user signed off per property. For ginkgo_hic the roster is esm2_concat, aac, esm3, prot_t5, esm2_early, igt5, esm2_entropy_top20, esmc, ablang2, esm2_midlayers (complexity_helpers.py:428-434); this file drops esm3, esmc, esm2_early, esm2_entropy_top20 and esm2_midlayers, and adds blosum, onehot and antiberty.

<details><summary>Options considered</summary>

1. Pass metric='spearman' to find_best_hybrid at :300. *Cost:* metric_name is part of the search cache identity (src/kirby/find_best_hybrid.py:534), so every stored per-fold search misses and rebuilds.
2. Read the representation list per property from DATASET_REPS['general'] and DATASET_REPS['augmentation'] in tests/experiments/complexity/complexity_helpers.py:412-452, keeping --reps as an override. Only five Ginkgo properties have an entry there. *Cost:* rep_names and rep_hashes are both in the search cache identity (find_best_hybrid.py:531-533), so this triggers the same cold rebuild as the metric change.
3. Call honest_rescore on the block built at :529 and print delta_mean_of_folds beside the same-fold Delta. A working template is opig_graphinity_ddg.py:605-648. *Cost:* No model fits.

</details>

## opig_graphinity_ddg — `opig_graphinity_ddg.py`

**FIXED 2026-09-04 (split audit).** User chose to delete rather than re-register: within_foldx and within_rosetta are gone from ALL_CONDITIONS and CONDITION_DATASETS, with the reason in the module docstring, and --max-rosetta and the direct graphinity load go with them. Question 1 above is dropped. Both submit lines now pass --max-foldx 5000 instead of the --max-samples 5000 that argparse rejected. The pdb_ids filter and the representation roster are separate entries and untouched.

**DECIDED.** Fix the submit flag and the pdb_ids filter. Until both are done it either does not start or reports an inflated transfer number.

NO · hybrid: engine, per fold · compute: fine · live

Intends four conditions on antibody-antigen binding free energy change: within-distribution FoldX, within-distribution Rosetta, experimental SKEMPI, and train-on-synthetic-test-on-experimental. Only the last two actually execute; the two synthetic conditions are dropped at a registry check before any data is loaded.

**Q.** Can sequence-only representations predict experimental SKEMPI binding free energy change under leave-one-complex-out splits, and how far is that from the structure-based EGNN Pearson r?
**Q.** Does a model trained on FoldX synthetic data transfer to experimental SKEMPI?
**Q.** Does the hybrid beat the best single representation on the experimental condition?

*On re-read:* Two of the three blockers are one thing, and it is not a defect. graphinity_foldx and graphinity_rosetta were cut on 2026-07-09 because their labels are FoldX and Rosetta output rather than measurements.

Why:
- Both submit lines pass --max-samples 5000, which this file's parser does not define, so argparse exits 2 and the job produces nothing (tests/slurm_scripts/submit_opig_priority.sh:31 and tests/slurm_scripts/submit_opig_experiments.sh:63).
- The synthetic-to-experimental leakage filter at :231-234 never fires. It looks for 'pdb_ids' or 'pdb' on the dict returned by antibody_baseline_survey.load_dataset, and that dict has neither key (tests/antibody_baseline_survey.py:224-236).
- The representation list at :73-77 is not the pruned eight the user signed off for ddg_combined. The roster is mutation_features, onehot, antiberty_cdr, esm2_layer24, prot_t5_cdr, esm2_concat, esm2_wt_delta, antiberty with aac as augmentation (complexity_helpers.py:284-292).
- On the experimental condition the hybrid is chosen once over every row (:433-449) and then scored on folds drawn from those same rows (:484), so the Delta at :574 is the optimistic one.
- **Blocking** `opig_graphinity_ddg.py:714` — main() skips a condition when CONDITION_DATASETS[cond] is not in antibody_baseline_survey.DATASET_REGISTRY.
- **Blocking** `opig_graphinity_ddg.py:232` — The PDB leakage filter looks for 'pdb_ids' or 'pdb' on test_data, which comes from antibody_baseline_survey.load_dataset.
- **Blocking** `tests/slurm_scripts/submit_opig_priority.sh:31` — The submit line passes --max-samples 5000.

<details><summary>Options considered</summary>

1. Change --max-samples 5000 to --max-foldx 5000 in both submit scripts, or delete the flag; 5000 is already the default at :670. *Cost:* One line in each of two files.
2. Add 'pdb_ids': data.get('pdb_ids') to the dict load_dataset returns at tests/antibody_baseline_survey.py:224-236. The loader already supplies it; _assign_groups reads data['pdb_ids'] two lines below to build the complex groups (:268-270), and load_ddg_combined_dataset returns it (src/kirby/datasets/ddg_combined.py:974). *Cost:* One line.
3. Read the representation list from DATASET_REPS['ddg_combined'] in tests/experiments/complexity/complexity_helpers.py:284-292, keeping --reps as an override. The file already passes data.get('mutations') into compute_or_load_reps at :393, which is what mutation_features and esm2_wt_delta need. *Cost:* rep_names and rep_hashes are in the search cache identity (find_best_hybrid.py:531-533), so the experimental search rebuilds.

</details>

## opig_humanness — `opig_humanness.py`

**FIXED 2026-09-04 (split audit, option 1).** No kirby_hybrid row is written, so the scoring loop drops from 11 representations to 10 and the Summary Delta that read off it is gone. The computed held-out comparison is saved to the result file under honest_comparison, beside the raw per-fold block. delta_mean_of_folds and the adaptive_pruning meta collision are untouched.

**DECIDED.** Delete lines 316-321, print delta_mean_of_folds, take the four one-line fixes. The held-out table already answers question 2.

PARTIAL · hybrid: engine, per fold · compute: heavy · live

Loads a humanness classification dataset, clusters sequences at 95% identity for groups, builds 10 antibody representations, runs one find_best_hybrid search (metric auc, downstream rf) and materialises its full-data winner as an extra representation called kirby_hybrid, then scores every representation under 5 downstream models with 5-fold StratifiedGroupKFold. It also stores the search's per-fold candidate and baseline scores as an honest_rescore input block.

**Q.** Which of the ten antibody representations separates human from non-human therapeutic VH best, per downstream model, under 5-fold identity-grouped CV?
**Q.** Does a hybrid built by find_best_hybrid beat the best single representation on AUC, per model, when the winner is re-picked on held-out folds?
**Q.** How do the classical encodings (blosum, onehot, aac) compare with the sequence language-model representations on this task?

*On re-read:* Every code claim holds. Two things change the priority.

First, the `humanness_species` fault never bites.

Why:
- `reps['kirby_hybrid']` at line 320 is built from features chosen using all 564 labels, then scored under the same folds at line 335, and written into `results` with the same keys as the honestly-scored single representations.
- The printed Summary Delta at line 437 subtracts the best single representation from that full-data row, so the headline number is not the held-out one.
- Line 471 prints `r['delta']`, the difference of two means. `honest_rescore` also returns `delta_mean_of_folds`, the mean of the per-fold differences, and its own note at _shared.py:963-970 says the two diverge when either arm misses a fold.
- `adaptive_pruning=0.25` at line 304 changes nothing about what runs, but it enters the search meta, so this search and the shared precompute write the same file with different meta and invalidate each other. tests/precompute_hybrids.py:593 leaves it at the True default, which becomes 0.15.

<details><summary>Options considered</summary>

1. Delete lines 316-321 so no `kirby_hybrid` row is written at all. The held-out comparison already exists: `build_honest_rescore_input(search)` at line 315 stores it, and lines 456-477 print it per downstream model. *Cost:* The scoring loop drops from 11 representations to 10, so 11 x 5 models x 5 folds = 275 fits becomes 250.
2. The other side of the same choice: keep the row and label it. Add a key such as `'selection': 'full_data'` to the entry written at line 347, and skip `kirby_hybrid` in the best-hybrid loop at lines 420-427. *Cost:* No change in model fits, and nothing stored goes stale, same as above.
3. Print `r['delta_mean_of_folds']` at line 471 rather than `r['delta']`, and add `r['n_paired_folds']` beside the fold count. *Cost:* One line, no fits, no re-run.

</details>

## opig_kingsmith_component_splits — `opig_kingsmith_component_splits.py`

**FIXED 2026-09-04 (split audit, option 2).** evaluate_rep and the per-fold hybrid loop keep fold-index-aligned scores — a fold whose fit raises now records a NaN in place rather than shortening the list — and both write r2_per_fold and rmse_per_fold into the JSON. The Delta is computed per downstream model: on each held-out component the base representation is re-picked from the other components and subtracted from the hybrid's score on that same component. Stored as paired_hybrid_delta. The missing metric='rmse' at the search call is a separate entry and still open.

**DECIDED.** Restore metric='rmse' and keep the per-fold scores before this is submitted. Both cheap, both change which recipe wins.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · orphaned

Loads a Buchwald-Hartwig yield dataset, builds the 10 base plus 4 augmentation reaction representations named in DATASET_REPS, and for each of four component axes runs true leave-one-group-out GroupKFold (n_folds = n_groups). Per outer fold it runs a fresh find_best_hybrid on training rows only, applies the fold's rf-selected features to train and validation, and scores 5 downstream models; it separately scores every raw representation on the identical folds.

**Q.** How far do R2, RMSE and Spearman fall for each reaction representation as the component held out gets scarcer (additive 23 groups, aryl_halide 15, ligand 4, base 3)?
**Q.** Does a hybrid built per fold on training rows only beat the best single representation on each component axis?
**Q.** Which component axis is hardest to extrapolate across?

Why:
- Line 424 runs each per-fold search with no `metric=`, so it takes find_best_hybrid's default 'r2' (find_best_hybrid.py:6043). The registered metric for both datasets is rmse (tests/experiments/dataset_policy.py:277-278).
- That default is worse than a mismatch here. The scoring loop at line 328 treats a fold with `ss_tot == 0` as undefined and drops it, because holding out one component leaves near-constant yields.
- The reported delta at line 674 is the largest r2 over 14 representations x 5 models minus the largest over the hybrid's 5 models, both maximised on the same fold scores they are read from. The two maxima come from pools of different size, so the subtraction is not like-for-like.
- A fold whose search raises at line 441, or returns no fitted 'rf' at line 451, is dropped from the hybrid arm only. Every representation still averages over all folds, and line 674 subtracts the two means without comparing `n_folds_completed`.

<details><summary>Options considered</summary>

1. Put `metric='rmse'` back into the search call at line 424, where 9bd48ac removed it. *Cost:* One line.
2. Return the per-fold score lists from `_summarize_scores` at line 342 and write them into the JSON, then pair the two arms in `print_analysis`. On each held-out component, pick the best representation from the other folds' means, score it on that fold, and subtract the hybrid's score on the same fold. *Cost:* One function rewritten plus the reading half of `print_analysis`.
3. Submit it as eight tasks instead of one, using the flags that already exist: `--datasets` and `--splits`, one task per (dataset, component axis). *Cost:* Eight tasks.

</details>

## opig_trastuzumab_enrichment — `opig_trastuzumab_enrichment.py`

**FIXED 2026-09-04 (split audit, option 1).** build_honest_rescore_input then honest_rescore; the per-model held-out AUROC table is printed and saved under honest_comparison. hybrid_search.rf_cv_auc is renamed search_best_cv_auc and its note says it is a search maximum, not a held-out score. The hybrid still has no enrichment number — that is option 2 and it is still open — and the mismatched mutation lists under KIRBY_SMOKE_N and the two enrichment denominators are untouched.

**DECIDED.** Take the per-fold enrichment rewrite. It is the file's own question 3.

PARTIAL · hybrid: engine, per fold · compute: heavy · live

Loads 5,000 trastuzumab CDRH3 variants with binary HER2-binding labels, grouped by edit-distance shell from wildtype. Ranks variants zero-shot by the mean-LLR column of the two LLR representations, then trains 5 downstream models on each of 11 representations under 5-fold StratifiedGroupKFold and measures binder enrichment in the top 1/5/10% of each held-out fold.

**Q.** Which of the eleven antibody representations, with which downstream model, puts the most binders in the top 1/5/10% of a held-out edit-distance-shell fold?
**Q.** Do the two mean-LLR scores rank binders above non-binders with no training at all?
**Q.** Does a hybrid beat the best single representation at binder enrichment?

*On re-read:* The two central gaps are confirmed. Three details are wrong.

The smoke fault has the wrong mechanism.

Why:
- The file asks whether a hybrid beats the best single representation at binder enrichment, and produces no enrichment number for the hybrid at all. `hybrid_search.rf_cv_auc` at line 361 is the winning recipe's own mean over the folds it was chosen on, on AUC rather than enrichment.
- The search already carries what a held-out comparison needs. `search.all_results` holds per-fold scores for every candidate recipe and `search.baseline_fold_scores` holds them for every single representation (find_best_hybrid.py:1247).
- Under KIRBY_SMOKE_N the two mean-LLR representations are built from mismatched mutation lists, so every local smoke run of the zero-shot arm reports numbers computed from the wrong rows.
- The zero-shot enrichment at line 395 ranks all 5,000 rows against the whole-dataset binder rate; the supervised enrichment at line 241 uses each fold's own rate over about 1,000 rows. Both land under `enrich_*` and `ef_*` in the same file with nothing distinguishing the denominators.

<details><summary>Options considered</summary>

1. Store the held-out block. Add `honest_block = build_honest_rescore_input(search)` after line 368, put it in the returned dict at line 436, and print it the way opig_humanness.py:456-477 does. *Cost:* Zero extra model fits.
2. The other side of the same choice: score the hybrid on enrichment. Move the search inside the fold loop, build it on training rows only, apply `apply_feature_selection` to the held-out rows, and put the result through `evaluate_rep_enrichment`. *Cost:* One whole-data search of 6 models x 8 budgets x 5 allocation methods x 3 importance methods x 5 internal evaluators = 3,600 configurations x 5 inner folds = about 18,000 fits becomes five per-fold searches on 4,000 rows each, about 90,000 fits.
3. Add 'mutations' to the per-row key list at tests/experiments/_shared.py:2693. *Cost:* One word in a shared helper.

</details>

## opig_scaffold_hybrid — `opig_scaffold_hybrid.py`

**DECIDED.** Options 1, 3 and 4 now. They cost nothing and stop the file printing a one-fold delta as a result.

**Still open:** Option 2, storing the per-representation feature count, is the only way to answer whether the hybrid used the scaffold features. 301,000 cold fits.

PARTIAL · hybrid: engine, per fold · compute: WILL NOT FINISH · live

Computes two mutation-difference representations (dphyschem, 5 features; dphyschem_pos, sequence-length x 5 features) from the same sequences load_data returned, injects them into the 10-representation pool for four single-wildtype antibody DMS sets, and runs run_nested_cv under cache tier 'scaffold_cmp'. Then it reads per-representation honest fold means and counts scaffold names inside the winning blueprints.

**Q.** On the honest outer folds, does each scaffold-difference representation score higher than every base representation, per model?
**Q.** Does the hybrid actually put scaffold-difference features into the winning recipe?
**Q.** Does the hybrid beat the best single representation on these four datasets (delta_mean_of_folds)?

*On re-read:* Three corrections after re-reading.

First, the audit's section 4 item 5 is already applied in the working tree. `git diff tests/experiments/opig_scaffold_hybrid.py` shows summarise_one now reading `delta_mean_of_folds` and `blueprints_per_fold` out of `by_model`, and it also collects `n_paired_folds`.

Why:
- Counting a scaffold representation's name inside a winning recipe cannot tell you whether the hybrid gave it any features, because that name list is the whole input pool: the stored flab_expression_adams recipes list all 12 names at allocation top_2 with budget 10, and top_2 (hybrid.py:400) gives features to at most 2 representations.
- Nothing in the stored JSON records how many features each representation received, so question 2 cannot be answered from the four committed files either; the per-representation counts exist in the `finfo` that apply_blueprint returns at find_best_hybrid.py:8311 and are used only for a total width.
- The stored run does not answer whether the hybrid beats the best single representation: flab_binding_adams has 0 paired folds and delta nan on all six models, flab_binding_koenig has 2 of 5, flab_expression_adams has 1 of 5, and flab_expression_koenig's file is 146 bytes with no models in it.
- print_summary prints the delta with no paired-fold count beside it, so flab_expression_adams prints rf delta = +1.843 from one outer fold of five rows while that hybrid's own fold scores run about 0.29.
- **Blocking** `opig_scaffold_hybrid.py:135` — Counts a scaffold representation as 'picked' when its name appears in bp['reps'].
- **Also caught by the hybrid-usage audit** — summarise_one read delta_mean_of_folds and blueprints_per_fold at the top level of NestedCVResult.to_dict(), where they do not exist, so both of the file's answers printed empty.

<details><summary>Options considered</summary>

1. Report only what the four stored files actually support. In summarise_one and print_summary: print each delta beside its paired-fold count and the fold scores it was averaged over; relabel the blueprint count as the representations offered to the allocator, or drop it; and read block['lower_is_better'] before choosing best_base_rep and before counting scaffold_rank. *Cost:* Zero model fits.
2. Make nested_cv_evaluate record, per outer fold, how many features each representation contributed — the numbers are in `finfo` at find_best_hybrid.py:8311 and are currently read only for a total width — then re-run all four datasets. *Cost:* Per outer fold: 5 allocations x 3 importance methods x (rf 8 budgets x 4 evaluators + xgboost 8 x 5 + lightgbm 8 x 4 + knn 8 x 5 + mlp 7 x 4 + svm 6 x 4) = 2,940 configurations, each over 5 inner folds = 14,700 fits, plus 12 representations x 6 models x 5 inner folds = 360 baseline fits.
3. Either wire the three flags through — --models and --budgets into run_nested_cv's kwargs, --max-folds into n_outer_folds — or delete them from the parser. *Cost:* Zero model fits either way.

</details>

## opig_scaffold_similarity — `opig_scaffold_similarity.py`

**FIXED 2026-09-04 (split audit).** --mode features splits the variants of each dataset in two halves, keeps only pairs whose two variants share a half, chooses greedy's subset and the top-k ordering and the truncation peak on one half, and reports every score from the other. --n-pairs default raised from 8000 to 32000 so each half carries more pairs than the whole sample used to: measured on flab_expression_koenig, 15254 in the choosing half and 16548 in the reporting half, 25 s to load. greedy scored 0.609 on the reporting half against 0.614 on its own, still above top-k 0.562 and all-14 0.506. The 0.0-for-undefined writes and the uneven grader count in struct_ctrl_best are separate entries and untouched.

**DECIDED.** Fold into scaffold_experiment and feature_type_study. That fixes the 0.0 zeros, the uneven grader count and the missing whole-sequence comparison at once.

PARTIAL · hybrid: none needed · compute: fine · live

Samples pairs of variants from six mutation libraries, computes a distance between each pair under five mutation-difference representations and two distance formulas, and Spearman-correlates that distance against a substitution-matrix distance between the same two sequences, with the differing-position count partialled out. No model is trained.

**Q.** Which mutation-difference representation and which distance formula give pair distances that track a substitution-matrix distance, once the differing-position count is partialled out?
**Q.** Does a property feature earn a place in the distance because it correlates with the grader on its own, or because it is not redundant with the features already chosen?
**Q.** Are the numbers honest: is the grader built on the representation's own basis, is the grader mostly counting differing positions, and is any proposed distance algebraically the grader itself?

Why:
- An undefined Spearman is written to the results file as 0.0 at lines 262 and 267, and on both koenig sets every MutID and BaseDelta entry is 0.0 for that reason: on a one-mutation library the MutID euclidean distance between two distinct variants is the constant sqrt(2), so the correlation does not exist.
- Those 0.0 values then compete in the max at line 270 and again in the per-dataset winner at line 355, so an undefined correlation can outrank a measured negative one.
- The sibling helper does the opposite and says so: feature_type_study.score_subset (line 126) returns NaN and its docstring reads 'Never 0.0, which would read as a measured zero correlation'.
- struct_ctrl_best at line 270 takes the max over a different number of graders per representation — 2 for MutID and BaseDelta, 1 for dPhysChem and dPhysChem_pos because grantham shares their basis, and 0 for SubScore, which is why SubScore is nan on every dataset and never appears in the ranking.

<details><summary>Options considered</summary>

1. Two edits and re-run: at lines 262 and 267 keep a non-finite correlation as NaN instead of writing 0.0, and record beside struct_ctrl_best how many graders it was maxed over. *Cost:* Zero model fits — this file trains nothing.
2. Make this file a runner over the two studies that already answer its first two questions: call scaffold_experiment.run(dataset) for the representation ranking and feature_type_study.run(dataset) for the feature subset, and keep only the graphinity adapter and the controls block here. *Cost:* Zero model fits.
3. Delete tests/experiments/opig_scaffold_similarity.py and run scaffold_experiment.py per dataset plus feature_type_study.py instead. *Cost:* Zero model fits.

</details>
