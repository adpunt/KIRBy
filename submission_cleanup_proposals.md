# Proposed removals from the KIRBy repository

Nothing here has been done. Each group waits for the author to approve it.
Generated 2026-10-23 from `git ls-files`, by path and size only.

Tracked files now: 1846. Files in no group, kept as they are: 757.

"git rm --cached" removes a file from the repository but leaves it on this disk.
"git rm" removes it from both; it stays recoverable from git history.

## Summary

One row per group. Files is the number of tracked files in the group; size is their total on disk.

| Group | What | Files | Size | Proposed action |
|---|---|---|---|---|
| P | Files named after the excluded datasets (patient or participant data) | 298 | 2.9 MB | git rm --cached (stop tracking, keep on disk) |
| T | Thesis files | 161 | 14.6 MB | git rm --cached and add to .gitignore (keep on disk), or move to a private repository. Author to choose. |
| R | Result files | 84 | 10.6 MB | git rm --cached (keep on disk) |
| L | Cluster job logs (slurm-*.out, *.out, *.log) | 126 | 22.7 MB | git rm |
| S | scratchpad/ | 130 | 2.7 MB | git rm |
| A | tests/archive/ | 23 | 519 KB | git rm |
| N | Notes, audits, handoffs, roadmaps and job trackers (*.md) | 254 | 7.0 MB | git rm, except the ones the author keeps |
| O | Stray files at the root and unreferenced scripts | 10 | 308 KB | git rm |
| C | catboost_info/ | 3 | 39 KB | git rm |

## Before any of this: the public history

github.com/adpunt/KIRBy is public now (an anonymous request returned HTTP 200 on 2026-10-23).
Every file below is therefore already in its public git history, whatever is removed from the current tree.
Zenodo archives only the tagged snapshot, not the history.
So removing files makes the archived release clean, but leaves them readable on GitHub.
Two ways to change that, both the author's call:

- publish the release from a new repository that starts from the cleaned tree, with no history, and make the current one private
- rewrite the history of this repository (for example with git filter-repo), which needs a force-push

## Things to add, not remove

### The hERG data file

`tests/data_cache/chembl_herg_ki.csv` (88 KB, 1,415 compounds) is ignored by `.gitignore` line 55 (`*.csv`), so it is not in the repository.
The runner refuses to fetch hERG from ChEMBL unless `KIRBY_ALLOW_CHEMBL_FETCH=1` is set.
So on a fresh clone, every hERG command in the README stops with an error, the quick example included.
Proposal: track it and its stamp `chembl_herg_ki.provenance.json` (887 B) with `git add -f`.
The README already describes the file as being in `tests/data_cache/`.

The logD and Caco-2 file `openadmet_train.csv` is not on this laptop; the runner downloads it from Hugging Face.
Nothing records whether the Hugging Face file has changed since the paper's runs.
Unverified: whether the copy on ARC still matches the Hugging Face file.
Options: ship the copy from ARC (size not checked from here), or record its MD5 in the README.

### .gitignore lines

So the removed groups do not come back:

```
slurm-*.out
*.log
scratchpad/
catboost_info/
```

## Personal and cluster paths in the files that stay

The personal home path (`/Users/` followed by the account name) was in 4 files outside the groups above.
All four now use `~` instead, which points at the same folder on this laptop:
`src/kirby/datasets/herg.py`, `tests/experiments/federated/federated_helpers.py`,
`tests/experiments/federated/lhasa_herg_loader.py`, `tests/experiments/similarity_metrics/structural_embeddings.py`.
None of them is used by the assay runner.

Cluster paths (`/data/stat-cadd/scat9264`, `/data/stat-ecr/scat9264`, the group name `stat-cadd`) stay in the files below.
Nothing has been changed in them.

- `tests/alternative_data_noise_robustness.py` has them as the last two places it looks for the qsar_qm_models checkout, at lines 109-110, 150-151, 288-289, 333-334 and 380-381.
  The handoff forbids changing this file's behaviour, so they stay.
  Removing them would not change any number. It would only stop the runner finding qsar_qm_models on ARC when neither `QSAR_QM_MODELS_ROOT` nor a sibling checkout is there. Author to decide.
- `setup.sh` is the ARC environment for KIRBy's own jobs: venv, Hugging Face, torch and pip caches.
  The paper's jobs use `qsar_qm_models/setup.sh` instead.
  Options: leave it as a documented ARC file, or read each path from an environment variable with today's path as the default.
- `tests/slurm_scripts/` (135 files) are ARC job scripts for KIRBy's other experiments.
  Options: leave them as a record of how those were run, or remove the folder.

One row per file. The number is how many lines in that file name a cluster path.

| Path | Lines |
|---|---|
| `fix_pyg.sh` | 1 |
| `notebooks/kirby_quickstart.ipynb` | 1 |
| `setup.sh` | 11 |
| `src/kirby/datasets/cross_species_ness.py` | 1 |
| `src/kirby/datasets/herg.py` | 2 |
| `src/kirby/datasets/pdbbind.py` | 2 |
| `src/kirby/datasets/uspto.py` | 1 |
| `src/kirby/representations/reaction.py` | 4 |
| `tests/alternative_data_noise_robustness.py` | 10 |
| `tests/audit_full_wave.py` | 1 |
| `tests/check_rep_pools.py` | 1 |
| `tests/experiments/_shared.py` | 1 |
| `tests/experiments/federated/federated_helpers.py` | 2 |
| `tests/experiments/federated/lhasa_herg_loader.py` | 2 |
| `tests/experiments/structural_geometry_survey.py` | 2 |
| `tests/scripts/diagnose_dataset_cache.py` | 2 |
| `tests/scripts/plan_precompute.py` | 4 |
| `tests/slurm_scripts/archive/clean_and_test.sh` | 1 |
| `tests/slurm_scripts/archive/run_all.sh` | 1 |
| `tests/slurm_scripts/archive/run_all_resume.sh` | 1 |
| `tests/slurm_scripts/archive/run_allocation_experiment.sh` | 1 |
| `tests/slurm_scripts/archive/run_alternative_datasets.sh` | 1 |
| `tests/slurm_scripts/archive/run_antibody_complexity.sh` | 1 |
| `tests/slurm_scripts/archive/run_antibody_plm.sh` | 1 |
| `tests/slurm_scripts/archive/run_antibody_scenarios.sh` | 1 |
| `tests/slurm_scripts/archive/run_baseline_survey.sh` | 1 |
| `tests/slurm_scripts/archive/run_c11_c12.sh` | 1 |
| `tests/slurm_scripts/archive/run_finetune_antibody.sh` | 1 |
| `tests/slurm_scripts/archive/run_finetune_molecular.sh` | 1 |
| `tests/slurm_scripts/archive/run_herg_only.sh` | 1 |
| `tests/slurm_scripts/archive/run_molecular_scenarios.sh` | 1 |
| `tests/slurm_scripts/archive/run_precache_reps.sh` | 1 |
| `tests/slurm_scripts/archive/run_reduction_experiments.sh` | 1 |
| `tests/slurm_scripts/cancel_to_keepset.sh` | 1 |
| `tests/slurm_scripts/delete_av2_data.sh` | 3 |
| `tests/slurm_scripts/diagnostic.sh` | 1 |
| `tests/slurm_scripts/extend_precompute_chain.sh` | 4 |
| `tests/slurm_scripts/imrie_eval.sh` | 2 |
| `tests/slurm_scripts/imrie_pretrain.sh` | 3 |
| `tests/slurm_scripts/invalidate_allocation_caches.sh` | 1 |
| `tests/slurm_scripts/invalidate_refolded.sh` | 1 |
| `tests/slurm_scripts/measure_stat_cadd.sh` | 3 |
| `tests/slurm_scripts/migrate_cache_to_ecr.sh` | 4 |
| `tests/slurm_scripts/migrate_medium_to_long.sh` | 1 |
| `tests/slurm_scripts/migrate_models_to_ecr.sh` | 8 |
| `tests/slurm_scripts/migrate_short_to_medium.sh` | 1 |
| `tests/slurm_scripts/noise_prelim/check_one.sbatch` | 1 |
| `tests/slurm_scripts/precache_models.sh` | 1 |
| `tests/slurm_scripts/preflight_check.sh` | 1 |
| `tests/slurm_scripts/pretrain_rodent.sh` | 4 |
| `tests/slurm_scripts/pretrain_species_token.sh` | 5 |
| `tests/slurm_scripts/queue_noise_conditions.py` | 3 |
| `tests/slurm_scripts/run_blueprint_trace.sh` | 2 |
| `tests/slurm_scripts/run_c31_weighted_global.sh` | 1 |
| `tests/slurm_scripts/run_c32_allocation_cross.sh` | 1 |
| `tests/slurm_scripts/run_complexity_experiments.sh` | 1 |
| `tests/slurm_scripts/run_dta_scenarios.sh` | 1 |
| `tests/slurm_scripts/run_f22_similarity_feature_sources.sh` | 2 |
| `tests/slurm_scripts/run_f26_redundancy_penalty.sh` | 1 |
| `tests/slurm_scripts/run_f27_objective_and_budget.sh` | 1 |
| `tests/slurm_scripts/run_f28_objective_weight_space.sh` | 1 |
| `tests/slurm_scripts/run_f30_importance_methods.sh` | 3 |
| `tests/slurm_scripts/run_f32_budget_split_normalisation.sh` | 2 |
| `tests/slurm_scripts/run_forward_compat_gate.sh` | 2 |
| `tests/slurm_scripts/run_geometry_survey_srun.sh` | 1 |
| `tests/slurm_scripts/run_hybrid_arm_parallel.sh` | 2 |
| `tests/slurm_scripts/run_precache_molecular.sh` | 1 |
| `tests/slurm_scripts/run_precompute_dataset.sh` | 2 |
| `tests/slurm_scripts/run_similarity_experiments.sh` | 1 |
| `tests/slurm_scripts/run_smoke_gate.sh` | 1 |
| `tests/slurm_scripts/run_validity_gate.sh` | 1 |
| `tests/slurm_scripts/runbook/noise_smoke_chat5.sh` | 1 |
| `tests/slurm_scripts/runbook/server_checks.sh` | 5 |
| `tests/slurm_scripts/runbook/submit_all_stages.sh` | 1 |
| `tests/slurm_scripts/schedule_snapshots.sh` | 1 |
| `tests/slurm_scripts/snapshot.sh` | 1 |
| `tests/slurm_scripts/submit_antibody_ft_fill.sh` | 1 |
| `tests/slurm_scripts/submit_antibody_survey.sh` | 1 |
| `tests/slurm_scripts/submit_augmentation_showcase.sh` | 2 |
| `tests/slurm_scripts/submit_bovine_assembly.sh` | 2 |
| `tests/slurm_scripts/submit_c02_bugtest.sh` | 2 |
| `tests/slurm_scripts/submit_c02_full.sh` | 2 |
| `tests/slurm_scripts/submit_c02_resubmit.sh` | 2 |
| `tests/slurm_scripts/submit_canine_sra_assembly.sh` | 2 |
| `tests/slurm_scripts/submit_cgr_comparison.sh` | 1 |
| `tests/slurm_scripts/submit_cgr_comparison_resubmit.sh` | 2 |
| `tests/slurm_scripts/submit_cgr_smiles_pretrain.sh` | 3 |
| `tests/slurm_scripts/submit_comparisons.sh` | 1 |
| `tests/slurm_scripts/submit_dataset.sh` | 6 |
| `tests/slurm_scripts/submit_davis_investigation.sh` | 3 |
| `tests/slurm_scripts/submit_dta_baseline_resurvey.sh` | 2 |
| `tests/slurm_scripts/submit_dta_bindingdb.sh` | 2 |
| `tests/slurm_scripts/submit_everything.sh` | 1 |
| `tests/slurm_scripts/submit_f00_crelative.sh` | 1 |
| `tests/slurm_scripts/submit_f12.sh` | 1 |
| `tests/slurm_scripts/submit_f13.sh` | 1 |
| `tests/slurm_scripts/submit_fc_smoketest.sh` | 1 |
| `tests/slurm_scripts/submit_forward_compat.sh` | 1 |
| `tests/slurm_scripts/submit_foundation_hybrid_vet.sh` | 2 |
| `tests/slurm_scripts/submit_framework.sh` | 1 |
| `tests/slurm_scripts/submit_framework_arc.sh` | 1 |
| `tests/slurm_scripts/submit_framework_f08.sh` | 1 |
| `tests/slurm_scripts/submit_framework_f10.sh` | 1 |
| `tests/slurm_scripts/submit_framework_f11.sh` | 1 |
| `tests/slurm_scripts/submit_framework_f12.sh` | 1 |
| `tests/slurm_scripts/submit_framework_f13.sh` | 1 |
| `tests/slurm_scripts/submit_framework_f14.sh` | 1 |
| `tests/slurm_scripts/submit_framework_f15.sh` | 1 |
| `tests/slurm_scripts/submit_ft_precompute.sh` | 3 |
| `tests/slurm_scripts/submit_ft_precompute_adme.sh` | 3 |
| `tests/slurm_scripts/submit_geometry_survey.sh` | 1 |
| `tests/slurm_scripts/submit_herg_clf_survey.sh` | 1 |
| `tests/slurm_scripts/submit_missing_surveys.sh` | 1 |
| `tests/slurm_scripts/submit_mitra_hep.sh` | 4 |
| `tests/slurm_scripts/submit_molecular_aug_fill.sh` | 1 |
| `tests/slurm_scripts/submit_molecular_ft_fill.sh` | 1 |
| `tests/slurm_scripts/submit_molecular_graph_survey.sh` | 1 |
| `tests/slurm_scripts/submit_nucleotide_survey.sh` | 1 |
| `tests/slurm_scripts/submit_opig_experiments.sh` | 1 |
| `tests/slurm_scripts/submit_opig_priority.sh` | 3 |
| `tests/slurm_scripts/submit_opig_shared_vectors.sh` | 2 |
| `tests/slurm_scripts/submit_opig_surveys.sh` | 2 |
| `tests/slurm_scripts/submit_phase5_arc.sh` | 1 |
| `tests/slurm_scripts/submit_phase5_baselines.sh` | 1 |
| `tests/slurm_scripts/submit_phase5_htc.sh` | 1 |
| `tests/slurm_scripts/submit_phase5_hybrids.sh` | 1 |
| `tests/slurm_scripts/submit_precompute_extra_passes.sh` | 1 |
| `tests/slurm_scripts/submit_pretrain_rep.sh` | 1 |
| `tests/slurm_scripts/submit_pretrain_rep_model_transfer.sh` | 1 |
| `tests/slurm_scripts/submit_reaction_cgr_resurvey.sh` | 2 |
| `tests/slurm_scripts/submit_reaction_survey.sh` | 1 |
| `tests/slurm_scripts/submit_rep_prune.sh` | 1 |
| `tests/slurm_scripts/submit_rodent_to_vet.sh` | 4 |
| `tests/slurm_scripts/submit_same_target_transfer.sh` | 3 |
| `tests/slurm_scripts/submit_scaffold_hybrid.sh` | 2 |
| `tests/slurm_scripts/submit_schnet_e2e.sh` | 1 |
| `tests/slurm_scripts/submit_sim_optimized_hybrid.sh` | 5 |
| `tests/slurm_scripts/submit_similarity_metrics.sh` | 3 |
| `tests/slurm_scripts/submit_species_token_multispecies.sh` | 4 |
| `tests/slurm_scripts/submit_structural_protein_survey.sh` | 1 |
| `tests/slurm_scripts/submit_universal_survey.sh` | 1 |
| `tests/slurm_scripts/submit_validation_gp_rerun.sh` | 2 |
| `tests/slurm_scripts/submit_validation_pdv_rerun.sh` | 1 |
| `tests/slurm_scripts/submit_validation_svm_rerun.sh` | 1 |
| `tests/slurm_scripts/submit_x_ab_ness.sh` | 1 |
| `tests/slurm_scripts/submit_x_mol_rd_analysis.sh` | 1 |
| `tests/slurm_scripts/submit_x_mol_species_admet.sh` | 1 |
| `tests/slurm_scripts/test_alloc_determinism.sh` | 1 |
| `tests/slurm_scripts/verify_c02_after_fix.sh` | 1 |
| `tests/slurm_scripts/verify_c02_fresh_dataset.sh` | 1 |
| `tests/slurm_scripts/where_to_submit.sh` | 7 |
| `tests/slurm_scripts/worker.sh` | 1 |
| `tests/supervisor_presentation.py` | 1 |
| `tests/test_phase4b_models.py` | 1 |
| `tests/test_py311_reps.py` | 1 |
| `tests/tools/close_open_items.sh` | 4 |
| `tests/verify_run_status.py` | 1 |

## Notes on single groups

- Group P includes six files under `src/kirby/` (`datasets/annitia_maslg.py`, `datasets/nhanes_liver.py`, `datasets/tcga.py`, and three `finetuning/tcga_*.py`).
  `src/kirby/representations/genomic.py`, `src/kirby/finetuning/__init__.py` and `src/kirby/representations/forward_compat_omics.py` import them inside functions.
  Removing them breaks those genomic functions when called. It does not affect the assay runner or any test that passes now.
- Group T includes `tests/analysis/`, which holds the thesis figure and number scripts.
- `src/clamp/` and `src/noise_inject/` are on this disk but not tracked. pytest no longer collects them.
  `src/noise_inject/` is an older copy of NoiseInject; the installed one is `~/repos/NoiseInject`.

## Groups

### P. Files named after the excluded datasets (patient or participant data)

Why: Their path names a dataset that must not appear in the study. Classified by path only; none was opened. Removing them from the current tree does not remove them from the public git history.

Proposed action: git rm --cached (stop tracking, keep on disk).

298 files, 2.9 MB. One row per file, largest first.

| Path | Size |
|---|---|
| `tests/results/annitia/full/variant_correlation_hep.png` | 169 KB |
| `tests/results/annitia/full/variant_correlation_death.png` | 162 KB |
| `tests/results/annitia/full/patient_overlap.png` | 149 KB |
| `tests/results/annitia/full/patient_representation_train.csv` | 90 KB |
| `src/kirby/datasets/tcga.py` | 65 KB |
| `tests/results/annitia/full/valtest_contention_death.csv` | 60 KB |
| `tests/results/annitia/full/valtest_contention_hepatic.csv` | 59 KB |
| `tests/find_best_hybrid_tcga.py` | 56 KB |
| `tests/archive/test_tcga.py` | 37 KB |
| `tests/experiments/annitia/competition_methods_eval.py` | 37 KB |
| `tests/experiments/annitia/experiment_branches.py` | 34 KB |
| `tests/experiments/annitia/views.py` | 34 KB |
| `tests/experiments/annitia/deep_failure_audit.py` | 32 KB |
| `tests/results/annitia/full/patient_representation_val.csv` | 30 KB |
| `issues_tcga.md` | 30 KB |
| `tests/experiments/annitia/refuse_variants.py` | 29 KB |
| `tests/experiments/annitia/survival.py` | 27 KB |
| `tests/experiments/annitia/experiment_cd.py` | 27 KB |
| `ANNITIA_TRANSFER_ROADMAP.md` | 26 KB |
| `src/kirby/datasets/nhanes_liver.py` | 25 KB |
| `tests/experiments/annitia/failure_candidates.py` | 24 KB |
| `tests/experiments/annitia/distribution_and_cohorts.py` | 24 KB |
| `tests/experiments/annitia/dig_deeper.py` | 23 KB |
| `tests/experiments/annitia/transfer_endpoint.py` | 23 KB |
| `tests/experiments/annitia/sensitivity_audit.py` | 23 KB |
| `tests/experiments/annitia/new_high_ev_candidates.py` | 22 KB |
| `tests/experiments/annitia/experiment_research.py` | 22 KB |
| `tests/experiments/annitia/refuse_meta_aware.py` | 21 KB |
| `tests/experiments/annitia/build_v2_zips.py` | 21 KB |
| `ANNITIA_ROADMAP.md` | 21 KB |
| `tests/experiments/annitia/build_submission_zip.py` | 21 KB |
| `tests/experiments/annitia/method_local_tests.py` | 20 KB |
| `tests/experiments/annitia/noise_candidates.py` | 20 KB |
| `tests/experiments/annitia/calibration_metrics.py` | 20 KB |
| `tests/experiments/tcga_foundation_comparison.py` | 20 KB |
| `tests/experiments/annitia/refuse_hierarchical.py` | 20 KB |
| `tests/experiments/annitia/lb_reverse_engineer.py` | 19 KB |
| `tests/experiments/annitia/nhanes_transfer.py` | 19 KB |
| `tests/experiments/annitia/synth/masld_simulator.py` | 18 KB |
| `tests/experiments/annitia/uncertainty_candidates.py` | 18 KB |
| `tests/experiments/annitia/reverse_fusion.py` | 18 KB |
| `tests/experiments/annitia/dsm.py` | 17 KB |
| `tests/experiments/annitia/experiment_hep_boost.py` | 17 KB |
| `tests/experiments/annitia/analyse_residuals.py` | 17 KB |
| `tests/experiments/annitia/experiment_external.py` | 16 KB |
| `tests/experiments/annitia/patient_representation.py` | 16 KB |
| `tests/experiments/annitia/hep_aggressive.py` | 16 KB |
| `tests/experiments/annitia/synth/run_grid.py` | 16 KB |
| `tests/experiments/annitia/validate_top_candidates.py` | 15 KB |
| `tests/experiments/annitia/experiment_be.py` | 15 KB |
| `tests/experiments/annitia/deephit.py` | 15 KB |
| `tests/experiments/annitia/experiment_hparam.py` | 14 KB |
| `src/kirby/finetuning/tcga_geneformer.py` | 14 KB |
| `tests/experiments/annitia/experiment_full.py` | 14 KB |
| `tests/experiments/annitia/build_submission_with_notebook.py` | 14 KB |
| `tests/experiments/annitia/search.py` | 13 KB |
| `tests/experiments/annitia/run_deephit.py` | 13 KB |
| `tests/experiments/annitia/refuse_honest.py` | 13 KB |
| `tests/experiments/annitia/fusion.py` | 13 KB |
| `tests/experiments/annitia/build_aggressive_candidates.py` | 12 KB |
| `tests/experiments/annitia/run_tabpfn.py` | 12 KB |
| `tests/experiments/annitia/km_target_xgb.py` | 11 KB |
| `tests/experiments/annitia/pretrain_nhanes.py` | 11 KB |
| `tests/experiments/annitia/experiment_research_v2.py` | 11 KB |
| `tests/experiments/annitia/blend_search.py` | 11 KB |
| `tests/experiments/annitia/run_dsm.py` | 11 KB |
| `tests/experiments/annitia/synth/counterfactual.py` | 11 KB |
| `tests/experiments/annitia/run.py` | 11 KB |
| `tests/experiments/annitia/synth/outcome_conditioned.py` | 11 KB |
| `src/kirby/datasets/annitia_maslg.py` | 10 KB |
| `tests/experiments/annitia/pycox_dt.py` | 10 KB |
| `tests/experiments/annitia/per_patient_specialist.py` | 10 KB |
| `tests/experiments/annitia/residual_stacker.py` | 10 KB |
| `tests/experiments/annitia/refuse_holdout.py` | 10 KB |
| `tests/experiments/annitia/combine_branches.py` | 10 KB |
| `src/kirby/finetuning/tcga_scfoundation.py` | 9 KB |
| `tests/experiments/annitia/failure_analysis.py` | 9 KB |
| `tests/experiments/annitia/pretrained_mlp.py` | 9 KB |
| `tests/experiments/annitia/oof_to_leaderboard.py` | 9 KB |
| `tests/experiments/annitia/landmark.py` | 9 KB |
| `tests/experiments/annitia/sanity_audit.py` | 9 KB |
| `tests/experiments/annitia/build_journal.py` | 9 KB |
| `src/kirby/finetuning/tcga_bulkrnabert.py` | 9 KB |
| `tests/experiments/annitia/_fpca.py` | 9 KB |
| `tests/experiments/annitia/synthetic_aug.py` | 9 KB |
| `tests/experiments/annitia/_tsfm.py` | 8 KB |
| `tests/experiments/annitia/audit_local_validation.py` | 8 KB |
| `tests/experiments/annitia/oof_diagnostics.py` | 8 KB |
| `tests/experiments/annitia/pseudo_obs.py` | 8 KB |
| `tests/experiments/annitia/_schema.py` | 8 KB |
| `tests/experiments/annitia/cross_reference_failures.py` | 8 KB |
| `tests/experiments/annitia/synth/protocols.py` | 8 KB |
| `tests/experiments/annitia/MORNING_BRIEFING.md` | 8 KB |
| `tests/experiments/annitia/hep_max_specialist.py` | 8 KB |
| `tests/experiments/annitia/synth/MASLD_SIMULATOR_NOTES.md` | 8 KB |
| `tests/experiments/annitia/synth/mlp_hazard.py` | 8 KB |
| `tests/experiments/annitia/tabsurv.py` | 7 KB |
| `tests/archive/download_tcga.py` | 7 KB |
| `scripts/download_tcga_gdc.py` | 7 KB |
| `AUDIT_NHANES_GRID.md` | 7 KB |
| `tests/experiments/annitia/fine_gray.py` | 7 KB |
| `tests/experiments/annitia/importance.py` | 7 KB |
| `tests/experiments/annitia/_cache.py` | 7 KB |
| `tests/experiments/annitia/failure_focused_fusion.py` | 7 KB |
| `tests/experiments/annitia/feature_weighting.py` | 7 KB |
| `tests/results/annitia/validate_top.md` | 7 KB |
| `tests/experiments/annitia/README.md` | 6 KB |
| `tests/experiments/annitia/cross_pollinate.py` | 6 KB |
| `tests/experiments/annitia/LOCAL_TESTING.md` | 6 KB |
| `tests/experiments/annitia/tabpfn.py` | 6 KB |
| `tests/experiments/annitia/trend_features_hep.py` | 6 KB |
| `tests/experiments/annitia/self_train.py` | 6 KB |
| `tests/experiments/annitia/refuse_diverse.py` | 6 KB |
| `tests/experiments/annitia/SUBMISSION_PLAN_2026_05_08.md` | 6 KB |
| `tests/experiments/annitia/build_submission_from_eval.py` | 6 KB |
| `tests/experiments/annitia/_gp_smooth.py` | 6 KB |
| `tests/experiments/annitia/ranking_loss.py` | 6 KB |
| `tests/experiments/annitia/GENERALIZATION_IDEAS.md` | 6 KB |
| `tests/experiments/annitia/robust_cox.py` | 6 KB |
| `tests/experiments/annitia/valtest_contention.py` | 6 KB |
| `tests/experiments/annitia/refuse_isotonic.py` | 6 KB |
| `tests/experiments/annitia/_survival_discretiser.py` | 5 KB |
| `tests/experiments/annitia/refuse_nnls.py` | 5 KB |
| `tests/results/annitia/dig_deeper_report.md` | 5 KB |
| `tests/experiments/annitia/bootstrap_candidate_ci.py` | 5 KB |
| `tests/experiments/annitia/survtrace_branch.py` | 5 KB |
| `tests/results/annitia/full/failure_analysis.json` | 5 KB |
| `tests/experiments/annitia/permutation_null.py` | 5 KB |
| `tests/experiments/annitia/USER_SUGGESTIONS.md` | 5 KB |
| `tests/experiments/annitia/predict_lb_calibrated.py` | 5 KB |
| `tests/experiments/annitia/mitra_survival.py` | 5 KB |
| `tests/experiments/annitia/run_all_local.sh` | 5 KB |
| `tests/experiments/annitia/discretised_survival.py` | 5 KB |
| `tests/experiments/annitia/tabicl.py` | 5 KB |
| `tests/experiments/annitia/bagging.py` | 5 KB |
| `tests/experiments/annitia/_slurm_setup.sh` | 5 KB |
| `tests/results/annitia/full/oof_diagnostics.md` | 4 KB |
| `tests/experiments/annitia/post_round_sanity.py` | 4 KB |
| `tests/experiments/annitia/snapshot_ensemble.py` | 4 KB |
| `tests/experiments/annitia/trust_score.py` | 4 KB |
| `tests/experiments/annitia/pre_submit_check.py` | 4 KB |
| `tests/experiments/annitia/holdout_fusion_test.py` | 4 KB |
| `tests/experiments/annitia/submit.py` | 4 KB |
| `tests/experiments/annitia/compare_with_anchor.py` | 4 KB |
| `tests/results/annitia/full/cross_reference_death.csv` | 4 KB |
| `tests/results/annitia/full/cross_reference_hepatic.csv` | 4 KB |
| `tests/experiments/annitia/anchor_conformal_hep.py` | 4 KB |
| `tests/experiments/annitia/tabtune_survival.py` | 4 KB |
| `tests/experiments/annitia/mahal_meta_weight.py` | 4 KB |
| `tests/experiments/annitia/_oof.py` | 4 KB |
| `tests/experiments/annitia/classbal_hep.py` | 4 KB |
| `tests/experiments/annitia/refuse_bootstrap.py` | 3 KB |
| `tests/experiments/annitia/run_finegray_hep.py` | 3 KB |
| `tests/experiments/annitia/submit_deephit.py` | 3 KB |
| `tests/slurm_scripts/run_tcga_new_reps.sh` | 3 KB |
| `tests/slurm_scripts/submit_nhanes_t2_htc.sh` | 3 KB |
| `tests/experiments/annitia/leaderboard_history.json` | 3 KB |
| `tests/results/annitia/submissions/sub_master_cherry_pick_hep_anomBlend_death_wapp_journal.md` | 3 KB |
| `tests/experiments/annitia/tier1_fallback.sh` | 3 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_aware_weighted_v4_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_anomaly_blend_meta_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_aware_weighted_no_nnls_v3_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_aware_weighted_no_nnls_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_coxnet_stack_with_uncertainty_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_weighted_anomaly_per_patient_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_aggressive_specialist_only_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_aware_weighted_v2_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_aware_weighted_v3_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_new_variants_only_journal.md` | 3 KB |
| `tests/results/annitia/submissions/sub_master_disagreement_router_meta_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_aggressive_routed_blend_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_failure_hard_specialist_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_aware_weighted_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_soft_blend_disagreement_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_per_patient_specialist_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_deepfn_targeted_death_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_grand_all_v2_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_grand_all_v3_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_new_binwise_corrected_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_new_inference_dropout_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_aggressive_topk_only_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_anchor_conformal_hep_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_proven_meta_weighted_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_anomaly_router_meta_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_cluster_router_meta_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_failure_fn_recovery_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_failure_fp_suppress_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_aware_top5_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_repdisc_specialized_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_specialist_override_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_ultra_meta_weighted_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_aggressive_extreme_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_disagreement_route_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_failure_xref_aware_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_grand_all_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_method_pseudolabel_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_noise_classbalance_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_aggressive_xstack_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_best_per_endpoint_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_bootstrap_dropout_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_death_per_horizon_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_deepfn_specialist_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_hepmax_specialist_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_weighted_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_noise_censoraware_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_subgroup_hep_only_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_unc_bootstrap_var_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_unc_disagree_soft_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_uncertainty_route_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_best_combined_v2_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_deepfn_isolation_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_hepspec_v2_tuned_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_method_conformal_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_new_cohort_split_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_new_per_endpoint_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_residual_stacker_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_death_conformal_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_death_supremacy_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_deepfn_targeted_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_hepspec_v2_pure_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_noise_multiseed_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_residual_greedy_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_subgroup_routed_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_unc_mahal_blend_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_failure_robust_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_subgroup_blend_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_unc_specialist_journal.md` | 2 KB |
| `tests/results/annitia/eval/comparison.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_LEARNED_STACK_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_anomaly_blend_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_best_combined_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_cluster_route_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_top5_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_proven_median_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_repdisc_aware_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_repdisc_blend_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_unc_conformal_journal.md` | 2 KB |
| `tests/results/annitia/full/_build_grid.py` | 2 KB |
| `tests/results/annitia/submissions/sub_master_holdout_pick_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_isotonic_avg_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_all_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_method_mixup_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_rank_trimmed_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_hepmax_pure_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_noise_huber_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_proven_meta_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_rank_median_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_blend_top1_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_blend_top2_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_blend_top3_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_blend_top4_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_blend_top5_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_mahal_meta_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_nnls_stack_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_noise_ipcw_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_ultra_meta_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_rank_mean_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_BEST_BET_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_meta_avg_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_diverse_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_new_bma_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_round1b_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_honest_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_round3_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_FINAL_journal.md` | 2 KB |
| `tests/results/annitia/submissions/sub_master_HIER_journal.md` | 2 KB |
| `tests/slurm_scripts/submit_annitia_nhanes.sh` | 2 KB |
| `tests/slurm_scripts/submit_annitia_tsfm.sh` | 2 KB |
| `tests/archive/prepare_tcga_labels.py` | 2 KB |
| `tests/slurm_scripts/submit_annitia_stack_hparam.sh` | 2 KB |
| `tests/slurm_scripts/submit_annitia.sh` | 2 KB |
| `tests/results/annitia/full/oof_to_leaderboard.md` | 2 KB |
| `tests/results/annitia/eval/trust.md` | 2 KB |
| `tests/slurm_scripts/submit_annitia_deephit.sh` | 2 KB |
| `tests/slurm_scripts/submit_annitia_tabpfn.sh` | 2 KB |
| `tests/download_tcga_features.sh` | 2 KB |
| `tests/slurm_scripts/submit_annitia_dsm.sh` | 1 KB |
| `tests/slurm_scripts/run_tcga_expanded.sh` | 1 KB |
| `tests/slurm_scripts/run_finetune_tcga.sh` | 1 KB |
| `tests/results/annitia/full/failure_analysis.md` | 1 KB |
| `tests/slurm_scripts/submit_annitia_weighted.sh` | 1 KB |
| `tests/slurm_scripts/submit_annitia_stacking.sh` | 1 KB |
| `tests/results/annitia/eval/anchor_correlation.md` | 1 KB |
| `tests/slurm_scripts/submit_annitia_views.sh` | 1 KB |
| `tests/slurm_scripts/submit_annitia_xgbcox.sh` | 1 KB |
| `tests/results/annitia/eval/predict_lb.md` | 1 KB |
| `tests/slurm_scripts/run_tcga_scenarios.sh` | 1 KB |
| `tests/results/annitia/full/cross_reference.md` | 962 B |
| `tests/results/annitia/eval/permutation_null.md` | 871 B |
| `tests/slurm_scripts/submit_annitia_medium.sh` | 869 B |
| `tests/slurm_scripts/submit_annitia_short.sh` | 865 B |
| `tests/slurm_scripts/submit_annitia_long.sh` | 863 B |
| `tests/results/annitia/eval/verdict_fusion_clean.md` | 772 B |
| `tests/results/annitia/eval/verdict_fusion_tied.md` | 772 B |
| `tests/experiments/annitia/synth/__init__.py` | 699 B |
| `tests/results/annitia/full/valtest_contention.md` | 587 B |
| `tests/results/annitia/full/cross_reference.json` | 402 B |
| `tests/experiments/annitia/__init__.py` | 239 B |

### T. Thesis files

Why: The thesis source, its figures, its appendix, its analysis scripts and a built PDF. They are not code the paper needs, and they are already public on GitHub.

Proposed action: git rm --cached and add to .gitignore (keep on disk), or move to a private repository. Author to choose..

161 files, 14.6 MB. One row per file, largest first.

| Path | Size |
|---|---|
| `thesis_preview.pdf` | 3.1 MB |
| `thesis.tex` | 626 KB |
| `thesis_appendix/noise/figures/fig1_supp_ecfp4_overview.png` | 561 KB |
| `thesis_figures_to_upload/fig1_global_overview.png` | 553 KB |
| `figures_ch5_7/f02a_best_number_of_features.png` | 537 KB |
| `figures_ch5_7/f09_censoring_ladder.png` | 504 KB |
| `thesis_figures_to_upload/fig_validation_overview.png` | 453 KB |
| `thesis_figures_to_upload/fig_interaction.png` | 419 KB |
| `figures_ch5_7/f08_did_the_cleaning_work_detector.png` | 395 KB |
| `figures_ch5_7/f02a_best_number_of_features_antibodies.png` | 319 KB |
| `thesis_figures_to_upload/fig_uncertainty_combined.png` | 291 KB |
| `THESIS_OUTLINE.md` | 288 KB |
| `thesis_rework/stage1_takeaways_bullets/additional_files_body.tex` | 282 KB |
| `tests/analysis/ch7_numbers.py` | 261 KB |
| `figures_ch5_7/f03_blueprints.png` | 249 KB |
| `figures_ch5_7/f11_score_vs_damage_level.png` | 240 KB |
| `thesis_figures_to_upload/fig_validation_combined.png` | 226 KB |
| `figures_ch5_7/f02a_best_number_of_features_smallmol.png` | 215 KB |
| `figures_ch5_7/d7_noise_flow.png` | 211 KB |
| `tests/analysis/data/f3_logd_feature_names.csv` | 210 KB |
| `figures_ch5_7/f03b_blueprint_logd.png` | 208 KB |
| `figures_ch5_7/f08_did_the_cleaning_work_noise_robust_training.png` | 201 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_ddg_combined.png` | 183 KB |
| `thesis_figures_to_upload/fig_nn_family_comparison.png` | 178 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_herg_ki.png` | 177 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_combined.png` | 177 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_logd.png` | 176 KB |
| `thesis_figures_to_upload/fig2_anova_decomposition.png` | 172 KB |
| `refs.bib` | 170 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_ginkgo_colloidal.png` | 168 KB |
| `thesis_appendix/noise/figures/fig3_ranking_consistency.png` | 155 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_caco2.png` | 148 KB |
| `thesis_appendix/noise/figures/fig_validation_anova.png` | 147 KB |
| `figures_ch5_7/f08_did_the_cleaning_work_known_flag.png` | 137 KB |
| `thesis_figures_to_upload/fig_methods_noise_strategies.png` | 129 KB |
| `thesis_rework/stage1_takeaways_bullets/figures_code_input/adaptive_pruning_funnel.png` | 126 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_ginkgo_titer.png` | 120 KB |
| `figures_ch5_7/f10_detection_vs_cleaning_flab_thermo_tresanco.png` | 105 KB |
| `tools/build_thesis.py` | 98 KB |
| `tests/analysis/firstpass_notready.py` | 89 KB |
| `tests/analysis/ch7_sources/n32.out` | 80 KB |
| `tests/analysis/figures/s02_weighted_vs_baselines.png` | 79 KB |
| `tests/analysis/latest_report.txt` | 78 KB |
| `tests/analysis/thesis_figures.py` | 75 KB |
| `tests/analysis/figures/c02_hybrid_vs_baselines.png` | 75 KB |
| `tests/analysis/figures/opig/c04_importance.png` | 55 KB |
| `tests/analysis/ch7_numbers_expected.json` | 53 KB |
| `tests/analysis/figures/opig/ginkgo_dumbbell.png` | 52 KB |
| `tests/analysis/figures/opig/mol_c02_ci.png` | 44 KB |
| `thesis_supervisor_TH/lens_ch5.md` | 36 KB |
| `thesis_supervisor_TH/map_ch5.md` | 36 KB |
| `thesis_supervisor_TH/map_ch4.md` | 34 KB |
| `thesis_supervisor_TH/lens_ch1_2.md` | 34 KB |
| `thesis_supervisor_TH/lens_ch6.md` | 34 KB |
| `tests/analysis/thesis_loader_f9_11.py` | 33 KB |
| `thesis_supervisor_TH/lens_ch7_8.md` | 31 KB |
| `tests/analysis/ch7_sources/n20_out.txt` | 28 KB |
| `thesis_supervisor_TH/lens_ch3_4.md` | 27 KB |
| `tests/analysis/blueprint_trace.py` | 26 KB |
| `tests/analysis/thesis_loader_f2.py` | 24 KB |
| `tests/analysis/opig_slide_figures.py` | 24 KB |
| `tests/analysis/ch7_sources/group_table/GROUP_TABLE.md` | 23 KB |
| `tests/analysis/general_hybrid_vs_base.py` | 21 KB |
| `main.tex` | 21 KB |
| `tests/analysis/quick_hybrid_summary.py` | 21 KB |
| `tests/analysis/thesis_figure_style.py` | 20 KB |
| `tests/analysis/ch7_sources/n40.out` | 20 KB |
| `thesis_supervisor_TH/map_ch6_8.md` | 19 KB |
| `tests/analysis/read_f28_objective_weight_space.py` | 18 KB |
| `tests/analysis/thesis_loader_f8.py` | 18 KB |
| `tests/analysis/thesis_loader_f56.py` | 18 KB |
| `tests/analysis/analyse_noise_prelim.py` | 17 KB |
| `tests/analysis/f15_pruning_analysis.py` | 16 KB |
| `tests/analysis/ch7_sources/n30_all_levels_fold0.txt` | 15 KB |
| `thesis_appendix/noise/items/hyperparameters_default.tex` | 14 KB |
| `tests/analysis/ch7_sources/group_table/tables_tex.py` | 14 KB |
| `tests/analysis/test_internal_split_impact.py` | 14 KB |
| `tests/analysis/compare_distance_metrics.py` | 13 KB |
| `tests/analysis/inspect_results.py` | 12 KB |
| `tests/analysis/read_f24_augmentation.py` | 12 KB |
| `tests/analysis/thesis_loader_f3.py` | 12 KB |
| `tests/analysis/analyse_geometry_survey.py` | 12 KB |
| `thesis_supervisor_TH/comments_ch4.md` | 12 KB |
| `thesis_supervisor_TH/comments_ch5.md` | 12 KB |
| `tests/analysis/ch7_sources/group_table/build.py` | 9 KB |
| `tests/analysis/read_f27_objective_and_budget.py` | 9 KB |
| `tests/analysis/print_hybrid_vs_baseline.py` | 8 KB |
| `tests/analysis/thesis_loader_f1.py` | 8 KB |
| `tools/verify_thesis.py` | 8 KB |
| `MISSING_BIB_KEYS.txt` | 7 KB |
| `tests/analysis/make_supervisor_plots.py` | 6 KB |
| `tests/analysis/read_f29_model_and_tuning.py` | 6 KB |
| `thesis_appendix/seed_from_paper.py` | 6 KB |
| `THESIS_TERMINOLOGY.md` | 6 KB |
| `tests/analysis/read_geometry_survey.py` | 5 KB |
| `tests/analysis/read_f24_stage3_summary.py` | 5 KB |
| `tests/analysis/thesis_diagram_ch7_flow.py` | 5 KB |
| `thesis_appendix/noise/items/hyperparameters_tuned.tex` | 5 KB |
| `preamble.tex` | 5 KB |
| `thesis_appendix/noise/items/pdv_descriptors.tex` | 5 KB |
| `thesis_appendix/noise/items/model_redundancy.tex` | 4 KB |
| `thesis_supervisor_TH/TH_LENS.md` | 4 KB |
| `tests/analysis/f10_combined.py` | 4 KB |
| `figures_ch5_7/tab_noise_mitigation_broad.tex` | 3 KB |
| `figures_ch5_7/tab_noise_bygroup.tex` | 3 KB |
| `thesis_appendix/README.md` | 3 KB |
| `thesis_supervisor_TH/README.md` | 3 KB |
| `thesis_appendix/noise/INDEX.md` | 3 KB |
| `thesis_appendix/noise/items/representation_redundancy.tex` | 3 KB |
| `tests/analysis/ch7_sources/n32.py` | 3 KB |
| `tests/analysis/check_rep_cache.py` | 3 KB |
| `thesis_appendix/noise/items/model_icc.tex` | 3 KB |
| `tests/analysis/ch7_sources/n12cmp.py` | 3 KB |
| `thesis_supervisor_TH/comments_ch6.md` | 3 KB |
| `additional_files.tex` | 3 KB |
| `tests/analysis/ch7_sources/n20.py` | 2 KB |
| `tests/analysis/ch7_sources/n41.py` | 2 KB |
| `tests/analysis/ch7_sources/n40.py` | 2 KB |
| `ch5_conclusions_new.tex` | 2 KB |
| `tests/analysis/ch7_sources/n09_c.py` | 2 KB |
| `tests/analysis/ch7_sources/group_table/parse.py` | 2 KB |
| `figures_ch5_7/tab_hybrid_vs_best_single.tex` | 2 KB |
| `thesis_appendix/noise/items/uncertainty_metrics_ecfp4.tex` | 2 KB |
| `tests/analysis/ch7_sources/n12d.py` | 2 KB |
| `figures_ch5_7/tab_noise_robust_broad.tex` | 2 KB |
| `tests/analysis/ch7_sources/n12.py` | 2 KB |
| `tests/analysis/ch7_sources/n21v.py` | 2 KB |
| `tests/analysis/ch7_sources/n21.py` | 2 KB |
| `tests/analysis/ch7_sources/n12w.py` | 2 KB |
| `thesis_appendix/noise/items/bayesian_transformation_tests.tex` | 2 KB |
| `tests/analysis/ch7_sources/n07u.py` | 2 KB |
| `thesis_appendix/noise/items/excluded_configurations.tex` | 2 KB |
| `tests/analysis/ch7_sources/n07b.py` | 2 KB |
| `thesis_appendix/noise/items/rf_vs_qrf_assay.tex` | 2 KB |
| `tests/analysis/ch7_sources/n12b.py` | 2 KB |
| `tests/analysis/ch7_sources/n07.py` | 1 KB |
| `thesis_appendix/noise/items/ecfp4_overview.tex` | 1 KB |
| `thesis_appendix/noise/items/assay_variance_decomposition.tex` | 1 KB |
| `figures_ch5_7/tab_nearest_neighbour_prediction.tex` | 1 KB |
| `thesis_appendix/noise/items/clean_r2_vs_robustness.tex` | 1 KB |
| `tests/analysis/ch7_sources/n09_b.py` | 1 KB |
| `tests/analysis/ch7_sources/scan8.py` | 1 KB |
| `tests/analysis/ch7_sources/n33.py` | 1 KB |
| `figures_ch5_7/tab_distance_tracks_activity.tex` | 1 KB |
| `tests/analysis/ch7_sources/n30.py` | 1 KB |
| `tests/analysis/ch7_sources/n23.py` | 1 KB |
| `tests/analysis/ch7_sources/n25q.py` | 1 KB |
| `tests/analysis/ch7_sources/group_table/guard_run.py` | 977 B |
| `tests/analysis/ch7_sources/n23b.py` | 974 B |
| `tests/analysis/ch7_sources/n24f.py` | 973 B |
| `tests/analysis/ch7_sources/n38.py` | 906 B |
| `tests/analysis/ch7_sources/load2.py` | 851 B |
| `tests/analysis/ch7_sources/n18q.py` | 765 B |
| `tests/analysis/ch7_sources/load.py` | 749 B |
| `tests/analysis/ch7_sources/n07q.py` | 719 B |
| `tests/analysis/ch7_sources/n23q.py` | 713 B |
| `tests/analysis/ch7_sources/n09_a.py` | 712 B |
| `tests/analysis/ch7_sources/n24q.py` | 631 B |
| `tests/analysis/ch7_sources/n09_load.py` | 562 B |
| `thesis_supervisor_TH/comments_ch7.md` | 464 B |
| `tests/analysis/ch7_sources/n23j.py` | 379 B |

### R. Result files

Why: Outputs of earlier runs, smoke tests and server copies. The handoff says never delete a result file, so the proposal is to stop tracking them only.

Proposed action: git rm --cached (keep on disk).

84 files, 10.6 MB. One row per file, largest first.

| Path | Size |
|---|---|
| `tests/results_server/phase2_analysis/figure4_uncertainty_error.png` | 2.0 MB |
| `tests/results_server/phase2_analysis/figure5_calibration.png` | 1.7 MB |
| `tests/results_server/phase2_analysis/figure6_decomposition.png` | 1.3 MB |
| `tests/results_server/phase0_figures_improved/figure1_global_robustness_landscape.png` | 886 KB |
| `tests/results_server/phase0_figures_improved/figure2_representation_effects.png` | 857 KB |
| `tests/results_server/phase2_analysis/figure7_representation_comparison.png` | 567 KB |
| `tests/results_server/phase2_analysis/figure8_gnn_failure.png` | 412 KB |
| `tests/results_server/phase0_figures_improved/supplementary_s2_nsi_distributions.png` | 402 KB |
| `tests/results_server/phase0_figures_improved/supplementary_degradation_curves.png` | 384 KB |
| `tests/results/framework/f33_pruning/ANALYSIS_steps1to3_repcount_2026_09_16.txt` | 321 KB |
| `tests/results_server/phase0_figures_improved/supplementary_s1_baseline_vs_retention.png` | 320 KB |
| `tests/results_server/phase0_figures_improved/figure_gnn_analysis.png` | 316 KB |
| `tests/results/framework/f33_pruning/ANALYSIS_steps1to3_2026_09_12.txt` | 268 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed02_caco2_reg.log` | 212 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed02_herg_clf.log` | 56 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed01_herg_warm.log` | 56 KB |
| `tests/results/framework/f33_pruning/ANALYSIS_steps4to5_repcount_2026_09_16.txt` | 56 KB |
| `tests/results/framework/f33_pruning/ANALYSIS_steps4to5_voteoff_2026_09_12.txt` | 53 KB |
| `tests/results/framework/f33_pruning/ANALYSIS_steps4to5_2026_09_12.txt` | 51 KB |
| `tests/results/_smoke_2026_06_05/DECISIONS_FOR_USER.md` | 21 KB |
| `tests/results/_smoke_2026_06_05/COMPLEXITY_REPORT.md` | 18 KB |
| `tests/results/_smoke_2026_06_05/cache_audit/CACHE_INVALIDATION_CONTRACT.md` | 17 KB |
| `tests/results/_smoke_2026_06_05/OPIG_REPORT.md` | 15 KB |
| `tests/results/_smoke_2026_06_05/FEDERATED_REPORT.md` | 14 KB |
| `tests/results/_smoke_V_2026_06_10/COMPLEXITY_REPORT.md` | 14 KB |
| `tests/results/_smoke_2026_06_05/cache_audit/PRE_RERUN_CHECKLIST.md` | 14 KB |
| `tests/results/_smoke_2026_06_05/SIMILARITY_REPORT.md` | 14 KB |
| `tests/results/_smoke/opig_molrxn/cgr2.log` | 14 KB |
| `tests/results/_smoke_V_2026_06_10/NOISE_REPORT.md` | 13 KB |
| `tests/results/_smoke_2026_06_05/NOISE_REPORT.md` | 12 KB |
| `tests/results/_smoke/opig_molrxn/fbh4.log` | 12 KB |
| `tests/results/_smoke_V_2026_06_10/SIMILARITY_REPORT.md` | 11 KB |
| `tests/results/_smoke_V_2026_06_10/FEDERATED_REPORT.md` | 11 KB |
| `tests/results/_smoke_2026_06_05/FORWARD_COMPAT_REPORT.md` | 10 KB |
| `tests/results/_smoke_2026_06_05/cache_audit/REP123_DEEPDIVE.md` | 10 KB |
| `tests/results/_smoke_2026_06_05/cache_audit/MALFORMED_CACHE_VERDICT.md` | 9 KB |
| `tests/results/_smoke_V_2026_06_10/CACHE_DETERMINISM_REPORT.md` | 8 KB |
| `tests/results/_smoke_V_2026_06_10/PRECOMPUTE_REPORT.md` | 8 KB |
| `tests/results/_smoke_V_2026_06_10/OPIG_REPORT.md` | 8 KB |
| `tests/results/_smoke/opig_molrxn/ks3.log` | 8 KB |
| `tests/results/_smoke_2026_06_05/RERUN_PLAYBOOK_2026_06_05.md` | 7 KB |
| `tests/results/_smoke_2026_06_05/AGENT_BRIEF.md` | 6 KB |
| `tests/results/_smoke_V_2026_06_10/BRIEF.md` | 6 KB |
| `tests/results/_smoke/opig_molrxn/ks2.log` | 6 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed01_herg.log` | 6 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed01_herg_cold.log` | 5 KB |
| `tests/results/framework/f39_screen_choice/herg_ki_v2.jsonl` | 5 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed03_herg.log` | 4 KB |
| `tests/results/_smoke/opig_molrxn/bp.log` | 4 KB |
| `tests/results/framework/f39_screen_choice/herg_ki.jsonl` | 4 KB |
| `tests/results/framework/f39_screen_choice/caco2.jsonl` | 4 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed04_herg.log` | 3 KB |
| `tests/results/molecular_ft_fill/adme_mdr1_ft_blueprint.py` | 3 KB |
| `tests/results/molecular_ft_fill/adme_rlm_ft_blueprint.py` | 3 KB |
| `tests/results/molecular_ft_fill/adme_hlm_ft_blueprint.py` | 3 KB |
| `tests/results/molecular_ft_fill/qm9_ft_blueprint.py` | 3 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed05.log` | 3 KB |
| `tests/results_server/phase0_figures_improved/statistical_tests_phase0.txt` | 2 KB |
| `tests/results/_smoke/opig_molrxn/fbh3.log` | 2 KB |
| `tests/results_server/phase2_analysis/gauche_analysis.txt` | 2 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/cache_evidence.log` | 2 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/fed05_probe.log` | 2 KB |
| `tests/results/_smoke/opig_molrxn/cgr.log` | 2 KB |
| `tests/results/_smoke_V_2026_06_10/fed_logs/cache_evidence2.log` | 2 KB |
| `tests/results_server/phase0_figures_improved/table1_top20_robust_configs.tex` | 1 KB |
| `tests/results_server/phase0_figures_improved/anova_variance_decomposition.txt` | 1 KB |
| `tests/results/_smoke/opig_molrxn/fbh2.log` | 1 KB |
| `tests/results/_smoke/opig_molrxn/ac.log` | 954 B |
| `tests/results/_smoke/opig_molrxn/ac_clean.log` | 954 B |
| `tests/results/_smoke/opig_molrxn/ks.log` | 940 B |
| `tests/results/_smoke/check_json.py` | 776 B |
| `tests/results/_smoke/drive.py` | 605 B |
| `tests/results/_smoke/opig_molrxn/fbh.log` | 503 B |
| `tests/results/_smoke/run_cdr.sh` | 359 B |
| `tests/results/_smoke/opig_molrxn/order.log` | 204 B |
| `tests/results/_smoke/opig_molrxn/models.log` | 117 B |
| `tests/results/_smoke/opig_molrxn/models2.log` | 106 B |
| `tests/results/_smoke/opig_molrxn/models3.log` | 106 B |
| `tests/results/_smoke/opig_molrxn/knn.log` | 87 B |
| `tests/results/_smoke/opig_molrxn/seq.log` | 57 B |
| `tests/results/_smoke/opig_molrxn/seq2.log` | 48 B |
| `tests/results/_smoke/opig_molrxn/nolgb.log` | 33 B |
| `tests/results/_smoke/opig_molrxn/fault.txt` | 20 B |
| `tests/results/_smoke/opig_molrxn/trip.log` | 20 B |

### L. Cluster job logs (slurm-*.out, *.out, *.log)

Why: Scheduler output from past jobs. Not code, not results the paper reads.

Proposed action: git rm.

126 files, 22.7 MB. One row per file, largest first.

| Path | Size |
|---|---|
| `slurm-12039999.out` | 3.5 MB |
| `slurm-11784360.out` | 1.3 MB |
| `slurm-12044246.out` | 1.3 MB |
| `slurm-11784359.out` | 949 KB |
| `slurm-12013711.out` | 839 KB |
| `slurm-11831697.out` | 828 KB |
| `slurm-11848056.out` | 801 KB |
| `slurm-12013735.out` | 797 KB |
| `slurm-11831695.out` | 671 KB |
| `slurm-12013715.out` | 664 KB |
| `slurm-11831696.out` | 594 KB |
| `slurm-11814319.out` | 580 KB |
| `slurm-12013710.out` | 578 KB |
| `slurm-12014452.out` | 577 KB |
| `slurm-12039997.out` | 507 KB |
| `slurm-12039987.out` | 473 KB |
| `slurm-12014447.out` | 436 KB |
| `slurm-12039988.out` | 387 KB |
| `slurm-7568450.out` | 363 KB |
| `slurm-11825839.out` | 335 KB |
| `slurm-12013713.out` | 330 KB |
| `slurm-11831698.out` | 321 KB |
| `slurm-12014449.out` | 312 KB |
| `slurm-12013712.out` | 305 KB |
| `slurm-12039986.out` | 286 KB |
| `slurm-12040000.out` | 262 KB |
| `slurm-12040001.out` | 242 KB |
| `slurm-7568449.out` | 225 KB |
| `slurm-12013708.out` | 216 KB |
| `slurm-12039994.out` | 205 KB |
| `slurm-11443767.out` | 200 KB |
| `slurm-12039983.out` | 198 KB |
| `slurm-12044254.out` | 198 KB |
| `slurm-12040022.out` | 191 KB |
| `slurm-12039998.out` | 190 KB |
| `slurm-11825838.out` | 189 KB |
| `slurm-12039996.out` | 179 KB |
| `slurm-12039991.out` | 164 KB |
| `slurm-12040007.out` | 135 KB |
| `slurm-12040002.out` | 130 KB |
| `slurm-12013714.out` | 98 KB |
| `slurm-12013724.out` | 96 KB |
| `slurm-11667042.out` | 95 KB |
| `slurm-12039989.out` | 82 KB |
| `slurm-11668193.out` | 80 KB |
| `slurm-12013730.out` | 76 KB |
| `slurm-11668198.out` | 75 KB |
| `slurm-12013725.out` | 65 KB |
| `slurm-12013718.out` | 56 KB |
| `slurm-11595868.out` | 47 KB |
| `slurm-11595869.out` | 45 KB |
| `slurm-7440311.out` | 45 KB |
| `slurm-7440312.out` | 45 KB |
| `slurm-7440313.out` | 45 KB |
| `slurm-7440309.out` | 45 KB |
| `slurm-7440310.out` | 44 KB |
| `slurm-12044247.out` | 44 KB |
| `slurm-12013731.out` | 37 KB |
| `slurm-11292552.out` | 36 KB |
| `slurm-12013736.out` | 34 KB |
| `slurm-12014473.out` | 33 KB |
| `slurm-12014472.out` | 31 KB |
| `slurm-11667041.out` | 24 KB |
| `slurm-12014470.out` | 22 KB |
| `slurm-11848055.out` | 22 KB |
| `slurm-11692950.out` | 20 KB |
| `slurm-7483446.out` | 19 KB |
| `slurm-12039990.out` | 19 KB |
| `slurm-12014468.out` | 18 KB |
| `slurm-11666857.out` | 14 KB |
| `slurm-11848054.out` | 11 KB |
| `slurm-11848053.out` | 10 KB |
| `slurm-12014469.out` | 10 KB |
| `slurm-11825840.out` | 9 KB |
| `slurm-11816411.out` | 9 KB |
| `slurm-12040020.out` | 8 KB |
| `slurm-11825841.out` | 8 KB |
| `slurm-12040019.out` | 8 KB |
| `tests/experiments/feature_selectors/full_run.log` | 8 KB |
| `tests/experiments/feature_selectors/full_run_LEAKY_top2.log` | 8 KB |
| `slurm-12040017.out` | 8 KB |
| `slurm-12040016.out` | 8 KB |
| `slurm-12040026.out` | 7 KB |
| `slurm-12040025.out` | 7 KB |
| `slurm-7548427.out` | 7 KB |
| `tests/experiments/similarity_metrics/fa_run.log` | 6 KB |
| `slurm-12014471.out` | 6 KB |
| `slurm-12044279.out` | 6 KB |
| `slurm-12040012.out` | 6 KB |
| `slurm-12040013.out` | 6 KB |
| `slurm-12040009.out` | 6 KB |
| `slurm-12040015.out` | 6 KB |
| `slurm-12040011.out` | 5 KB |
| `slurm-12039985.out` | 5 KB |
| `slurm-12044257.out` | 5 KB |
| `slurm-12044270.out` | 5 KB |
| `slurm-12044271.out` | 5 KB |
| `slurm-12044274.out` | 5 KB |
| `slurm-12040024.out` | 5 KB |
| `slurm-12044262.out` | 5 KB |
| `slurm-12044275.out` | 5 KB |
| `slurm-12044263.out` | 5 KB |
| `slurm-12044266.out` | 5 KB |
| `slurm-12044267.out` | 5 KB |
| `slurm-12044264.out` | 5 KB |
| `slurm-12044268.out` | 5 KB |
| `slurm-12044273.out` | 5 KB |
| `slurm-12044276.out` | 5 KB |
| `slurm-12044269.out` | 4 KB |
| `slurm-12044272.out` | 4 KB |
| `slurm-11604969.out` | 4 KB |
| `slurm-12044265.out` | 4 KB |
| `slurm-12044278.out` | 4 KB |
| `slurm-12044261.out` | 4 KB |
| `slurm-12044245.out` | 4 KB |
| `slurm-12013727.out` | 4 KB |
| `slurm-11604966.out` | 4 KB |
| `slurm-12014464.out` | 3 KB |
| `tests/experiments/feature_selectors/smoke_herg.log` | 2 KB |
| `slurm-7459731.out` | 681 B |
| `slurm-7459732.out` | 681 B |
| `slurm-7459733.out` | 681 B |
| `slurm-7459734.out` | 681 B |
| `slurm-11605026.out` | 514 B |
| `slurm-11605033.out` | 514 B |
| `slurm-11605040.out` | 514 B |

### S. scratchpad/

Why: Temporary working files from analysis sessions.

Proposed action: git rm.

130 files, 2.7 MB. One row per file, largest first.

| Path | Size |
|---|---|
| `scratchpad/f24_stage3_read.txt` | 642 KB |
| `scratchpad/opig_verify/cgr.log` | 310 KB |
| `scratchpad/rank_caco2.log` | 214 KB |
| `scratchpad/opig_verify/ginkgo_hic_wide.log` | 184 KB |
| `scratchpad/opig_verify/ginkgo_hic.log` | 125 KB |
| `scratchpad/opig_verify/trastuzumab.log` | 123 KB |
| `scratchpad/precompute_cuts/DECISIONS.md` | 80 KB |
| `scratchpad/opig_verify/kingsmith.log` | 75 KB |
| `scratchpad/opig_verify/scaffold_hybrid.log` | 67 KB |
| `scratchpad/opig_verify/cdr_flexibility.log` | 63 KB |
| `scratchpad/mc_caco2.log` | 43 KB |
| `scratchpad/DISCARDED_wrong_base_rep_caco2.log` | 41 KB |
| `scratchpad/aug_budget.log` | 33 KB |
| `scratchpad/f24_ranking_timing_herg.log` | 31 KB |
| `scratchpad/rank_flab_expression_koenig.log` | 26 KB |
| `scratchpad/opig_verify/graphinity2.log` | 25 KB |
| `scratchpad/opig_verify/blueprint.log` | 21 KB |
| `scratchpad/aug_sweep_all.log` | 21 KB |
| `scratchpad/aug_stage_sweep.py` | 20 KB |
| `scratchpad/opig_verify/qsar_mpro.log` | 20 KB |
| `scratchpad/smoke_logd.log` | 20 KB |
| `scratchpad/f24_deeplift_fix.log` | 19 KB |
| `scratchpad/aug_mechanism_sweep.py` | 17 KB |
| `scratchpad/aug_budget_split.py` | 15 KB |
| `scratchpad/aug_mechanism_autopsy.py` | 14 KB |
| `scratchpad/DISCARDED_wrong_base_rep_koenig.log` | 14 KB |
| `scratchpad/opig_verify/census.log` | 14 KB |
| `scratchpad/opig_verify/humanness_b500.log` | 12 KB |
| `scratchpad/aug_decoy_check.py` | 12 KB |
| `scratchpad/slurm-12913493_2.out` | 12 KB |
| `scratchpad/slurm-12913493_3.out` | 11 KB |
| `scratchpad/slurm-12913493_0.out` | 11 KB |
| `scratchpad/slurm-12913493_1.out` | 11 KB |
| `scratchpad/cache_audit/findings/rep-pool-not-in-result-db-identity.md` | 10 KB |
| `scratchpad/aug_dilution_factorial.py` | 10 KB |
| `scratchpad/aug_tresanco_run1_DISCARDED.log` | 9 KB |
| `scratchpad/aug_fix_verify.py` | 8 KB |
| `scratchpad/mc_buchwald_greedy.log` | 8 KB |
| `scratchpad/pruning_run.log` | 8 KB |
| `scratchpad/aug_pass1_target.py` | 8 KB |
| `scratchpad/cache_audit/findings/server-wipe-procedure.md` | 7 KB |
| `scratchpad/f24_stage3_pooled.txt` | 7 KB |
| `scratchpad/opig_verify/humanness.log` | 7 KB |
| `scratchpad/run_c21_scoped.py` | 6 KB |
| `scratchpad/mc_buchwald.log` | 5 KB |
| `scratchpad/mc_buchwald_cap512.log` | 5 KB |
| `scratchpad/base_rep_validity_vectors.py` | 5 KB |
| `scratchpad/cache_audit/findings/distance-precompute-drops-models.md` | 5 KB |
| `scratchpad/test_split_fixes.py` | 5 KB |
| `scratchpad/f24_stage3_summary.py` | 5 KB |
| `scratchpad/aug_stage_report.py` | 5 KB |
| `scratchpad/concat_pairwise.py` | 5 KB |
| `scratchpad/cache_audit/findings/pool-not-in-hybrid-eval-key.md` | 5 KB |
| `scratchpad/opig_verify/graphinity.log` | 4 KB |
| `scratchpad/split_test2.log` | 4 KB |
| `scratchpad/rank_caco2_control.log` | 4 KB |
| `scratchpad/cache_audit/findings/hybrid-fold-key-ignores-kwargs.md` | 4 KB |
| `scratchpad/cache_audit/findings/fed05-federated-fs-passes-a-model-with-no-metric.md` | 4 KB |
| `scratchpad/aug_budget_split_margin.py` | 4 KB |
| `scratchpad/cache_audit/findings/cdr-cache-not-keyed-to-sequence-list.md` | 4 KB |
| `scratchpad/cache_audit/findings/rep-cache-name-only-key.md` | 3 KB |
| `scratchpad/opig_verify/scaffold_similarity.log` | 3 KB |
| `scratchpad/cache_audit/findings/mcs-pairs-carry-no-ordering-hash.md` | 3 KB |
| `scratchpad/cache_audit/findings/s09-pools-unchecked-result-files.md` | 3 KB |
| `scratchpad/herg_single_count.log` | 3 KB |
| `scratchpad/sweep_herg.log` | 3 KB |
| `scratchpad/decoy_herg.log` | 3 KB |
| `scratchpad/cache_audit/findings/dataset-name-is-the-only-key-for-five-caches.md` | 3 KB |
| `scratchpad/cache_audit/findings/c23-c24-mangle-dataset-name.md` | 3 KB |
| `scratchpad/split_test.log` | 3 KB |
| `scratchpad/cache_audit/findings/noisy-search-not-keyed-on-injector.md` | 3 KB |
| `scratchpad/aug_tresanco.log` | 3 KB |
| `scratchpad/cache_audit/findings/fitted-pool-rebuild-drops-recipe-fields.md` | 3 KB |
| `scratchpad/cache_audit/findings/invalidate-scripts-wipe-the-home-cache.md` | 3 KB |
| `scratchpad/cache_audit/findings/c30-c31-c32-resume-keeps-old-numbers-under-new-config.md` | 3 KB |
| `scratchpad/cache_audit/findings/alloc-failure-encoding-changed.md` | 3 KB |
| `scratchpad/rank_flab_expression_koenig_control.log` | 3 KB |
| `scratchpad/cache_audit/findings/baseline-missing-struct-pairs-id.md` | 3 KB |
| `scratchpad/cache_audit/findings/fold-search-memo-thin-key.md` | 3 KB |
| `scratchpad/cache_audit/findings/metric-column-overwritten-by-objective-name.md` | 3 KB |
| `scratchpad/cache_audit/findings/internal-evaluators-list-disagrees.md` | 3 KB |
| `scratchpad/cache_audit/findings/mangled-dataset-name-bypasses-the-budget-policy.md` | 3 KB |
| `scratchpad/decoy_logd.log` | 3 KB |
| `scratchpad/cache_audit/findings/fed01-c0-hybrid-gets-no-groups.md` | 3 KB |
| `scratchpad/cache_audit/findings/noisy-precompute-tier-never-read.md` | 3 KB |
| `scratchpad/cache_audit/findings/rep-factories-id-not-in-result-db-identity.md` | 3 KB |
| `scratchpad/cache_audit/findings/fold-scores-code-not-in-key.md` | 3 KB |
| `scratchpad/cache_audit/findings/c18-natural-arm-runs-at-a-different-budget.md` | 3 KB |
| `scratchpad/cache_audit/findings/resume-fingerprint-ignores-the-metric.md` | 3 KB |
| `scratchpad/cache_audit/findings/baseline-missing-tune-knn-k.md` | 3 KB |
| `scratchpad/cache_audit/findings/noise-grid-absent-from-resume-fingerprint.md` | 3 KB |
| `scratchpad/cache_audit/findings/c10-recipe-chosen-on-clean-labels.md` | 3 KB |
| `scratchpad/cache_audit/findings/tune-not-in-result-db-identity.md` | 3 KB |
| `scratchpad/thesis_ch5_8/prcho_tables.py` | 3 KB |
| `scratchpad/cache_audit/findings/baseline-missing-splitter-identity.md` | 3 KB |
| `scratchpad/cache_audit/findings/n09-silent-clean-recipe-fallback.md` | 2 KB |
| `scratchpad/cache_audit/findings/aug-margin-fraction-dead-key-params.md` | 2 KB |
| `scratchpad/show_ranking_results.py` | 2 KB |
| `scratchpad/cache_audit/findings/aug-logic-version-cannot-invalidate-a-result-db-row.md` | 2 KB |
| `scratchpad/cache_audit/findings/fold-scores-evaluator-fallback.md` | 2 KB |
| `scratchpad/cache_audit/findings/c14-nested-cv-stores-nothing.md` | 2 KB |
| `scratchpad/deeplift_diagnose.py` | 2 KB |
| `scratchpad/cache_audit/findings/folds-guard-only-under-cache-true.md` | 2 KB |
| `scratchpad/cache_audit/findings/cached-door-internal-evaluator-svm.md` | 2 KB |
| `scratchpad/cache_audit/findings/c30-c31-c32-gate-not-rerun-for-new-datasets.md` | 2 KB |
| `scratchpad/logd_run.log` | 2 KB |
| `scratchpad/validity_verify.log` | 2 KB |
| `scratchpad/cache_audit/findings/baseline-model-name-only.md` | 2 KB |
| `scratchpad/cache_audit/findings/embedding-file-name-omits-the-model-id.md` | 2 KB |
| `scratchpad/pairwise_run.log` | 1 KB |
| `scratchpad/opig_verify/herg_rank.log` | 1 KB |
| `scratchpad/smoke_sweep_single.log` | 1 KB |
| `scratchpad/step3_run.log` | 1 KB |
| `scratchpad/_lgbm_noise.py` | 1 KB |
| `scratchpad/read_ranking_run.py` | 1 KB |
| `scratchpad/run_ranking_test.sh` | 1 KB |
| `scratchpad/smoke_local.log` | 1 KB |
| `scratchpad/opig_verify/env.sh` | 1 KB |
| `scratchpad/run_smoke_with_warmup.py` | 957 B |
| `scratchpad/check_order_fix.py` | 872 B |
| `scratchpad/check_rankings.py` | 863 B |
| `scratchpad/check_deeplift_only.py` | 802 B |
| `scratchpad/mc_koenig.log` | 769 B |
| `scratchpad/check_torch_lgbm.py` | 759 B |
| `scratchpad/_time_uncapped.py` | 613 B |
| `scratchpad/_probe_load.py` | 456 B |
| `scratchpad/rank_koenig.driver` | 448 B |
| `scratchpad/rank_caco2.driver` | 380 B |
| `scratchpad/smoke_herg.log` | 86 B |
| `scratchpad/ranking_dispatch.log` | 0 B |

### A. tests/archive/

Why: Retired scripts. Five of them import functions that no longer exist, which is what broke pytest collection; pytest.ini now skips this folder.

Proposed action: git rm.

23 files, 519 KB. One row per file, largest first.

| Path | Size |
|---|---|
| `tests/archive/benchmark_importance_aggregation.py` | 65 KB |
| `tests/archive/test_datasets.py` | 41 KB |
| `tests/archive/analyse_alternative_results.py` | 38 KB |
| `tests/archive/label_denoising.py` | 32 KB |
| `tests/archive/fast_test.py` | 31 KB |
| `tests/archive/test_hybrid_user_scenarios_antibody.py` | 30 KB |
| `tests/archive/phase2_uncertainty.py` | 29 KB |
| `tests/archive/dimensionality_reduction_experiments.py` | 26 KB |
| `tests/archive/hybrid_master_fast.py` | 25 KB |
| `tests/archive/qm9_graphs_uncertainty.py` | 22 KB |
| `tests/archive/test_defaults.py` | 21 KB |
| `tests/archive/test_sabdab.py` | 21 KB |
| `tests/archive/graph_kernel_analysis.py` | 19 KB |
| `tests/archive/test_hybrid_user_scenarios.py` | 19 KB |
| `tests/archive/test_herg.py` | 15 KB |
| `tests/archive/test_hybrid_aggregation.py` | 15 KB |
| `tests/archive/test_hybrid_low_dim.py` | 15 KB |
| `tests/archive/test_esol.py` | 12 KB |
| `tests/archive/test_qm9.py` | 12 KB |
| `tests/archive/sanity_check_importance.py` | 11 KB |
| `tests/archive/test_pretrained_for_qsar.py` | 8 KB |
| `tests/archive/fast_validation.py` | 7 KB |
| `tests/archive/extract_correlation_matrix.py` | 7 KB |

### N. Notes, audits, handoffs, roadmaps and job trackers (*.md)

Why: Working notes. README.md is not in this group. Some may be worth keeping, for example SETUP.md or the design notes in docs/; the author picks.

Proposed action: git rm, except the ones the author keeps.

254 files, 7.0 MB. One row per file, largest first.

| Path | Size |
|---|---|
| `EXPERIMENTS_FINALIZED.md` | 678 KB |
| `EXPERIMENT_COMPUTE_AUDIT_2026_09_18.md` | 287 KB |
| `PRELIM_RESULTS_ANALYSIS.md` | 267 KB |
| `SPLIT_AUDIT_2026_09_03.md` | 202 KB |
| `SLIDE_FINDINGS.md` | 192 KB |
| `REPRODUCIBILITY_LEDGER.md` | 149 KB |
| `NOISE_AUDIT_2026_09_10.md` | 148 KB |
| `HANDOFF_EXPERIMENT_ALIGNMENT.md` | 147 KB |
| `SIMILARITY_AUTORESEARCH_STATUS.md` | 145 KB |
| `FRAMEWORK_FINALIZATION.md` | 133 KB |
| `BILL_OF_MATERIALS.md` | 116 KB |
| `ESTIMATES_INVESTIGATION_2026_06_04.md` | 113 KB |
| `SIMILARITY_METRICS_ROADMAP.md` | 110 KB |
| `PIPELINE_READINESS_TODO.md` | 103 KB |
| `AUGMENTATION.md` | 95 KB |
| `ISSUES.md` | 85 KB |
| `MASTER_ISSUES_2026_06_04.md` | 85 KB |
| `EXPERIMENT_DEBUGGING.md` | 85 KB |
| `WAVE_SNAPSHOT_2026_06_04_0207.md` | 77 KB |
| `HANDOFF_SPLIT_FIXES_2026_09_04.md` | 61 KB |
| `CROSS_SPECIES_ROADMAP.md` | 60 KB |
| `EXPERIMENT_REFACTOR_ROADMAP.md` | 58 KB |
| `PRECOMPUTE_JOB_TRACKER.md` | 57 KB |
| `PRECOMPUTE_RUN_PLAN_2026_09_14.md` | 55 KB |
| `WAVE_STATUS_2026_06_04.md` | 55 KB |
| `SIMILARITY_COEFFICIENT_RESEARCH_2026_08_26.md` | 54 KB |
| `PERTURBATION_SIMILARITY_SUITE_DESIGN.md` | 52 KB |
| `tests/experiments/framework/KNN_SEARCH_SPACE_RECORD.md` | 51 KB |
| `tests/experiments/noise/NOISE_COVERAGE.md` | 50 KB |
| `validity_summary.md` | 49 KB |
| `SIMILARITY_HYBRID_DESIGN_2026_08_24.md` | 49 KB |
| `DATASET_STATUS_2026_07_07.md` | 49 KB |
| `SETUP.md` | 46 KB |
| `HYBRID_REBUILD_ROADMAP_2026_07_15.md` | 45 KB |
| `tests/experiments/noise/OPEN_THREADS.md` | 43 KB |
| `METRIC_DIRECTION_AUDIT_2026_09_10.md` | 42 KB |
| `FEDERATED_EXPERIMENT_ROADMAP.md` | 42 KB |
| `ALLOCATION_CATEGORIES_2026_07_19.md` | 42 KB |
| `PAPER_DRAFT.md` | 41 KB |
| `SYNTHESIS_2026_06_04.md` | 41 KB |
| `EXPERIMENTS_CLOSED.md` | 40 KB |
| `issues_benchmarks.md` | 40 KB |
| `HYBRID_ALLOCATION_FINDINGS_PLAIN_2026_07_19.md` | 39 KB |
| `ROGII_ROADMAP.md` | 39 KB |
| `SELECTION_SHAPE_PLAN_2026_09_18.md` | 38 KB |
| `EXPERIMENT_DESIGN_DISCUSSION.md` | 37 KB |
| `DATASET_REPS_REFERENCE.md` | 36 KB |
| `CHECK_PRECOMPUTE_TIMING_2026_06_02.md` | 36 KB |
| `tests/experiments/noise/LABEL_SMOOTHING_EXPANSION.md` | 36 KB |
| `CACHE_AUDIT_2026_09_10.md` | 36 KB |
| `AUDIT_DOCUMENTATION.md` | 34 KB |
| `STATUS_ACCOUNTING_2026_06_04.md` | 34 KB |
| `AUDIT_HYBRID_USAGE_2026_09_03.md` | 33 KB |
| `PRECOMPUTE_EXPERIMENT_BUGS_2026_04_17.md` | 33 KB |
| `issues_forward_compat.md` | 32 KB |
| `DIGITAL_TWIN_REACTION_ROADMAP.md` | 31 KB |
| `issues_splitting.md` | 30 KB |
| `COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md` | 29 KB |
| `FRAMEWORK_EXPERIMENT_ROADMAP.md` | 29 KB |
| `OPIG_COLLAB_ROADMAP.md` | 28 KB |
| `PRETRAIN_REP_ROADMAP.md` | 27 KB |
| `SIM_OPTIMIZED_HYBRID_ANALYSIS_2026_07_10.md` | 26 KB |
| `HYBRID_DEFECTS_2026_07_14.md` | 26 KB |
| `ANALYSIS.md` | 26 KB |
| `UNCERTAINTY_AUDIT_2026_09_16.md` | 26 KB |
| `EXPERIMENT_STATUS_PER_DATASET.md` | 26 KB |
| `CACHE_FIX_PLAN.md` | 25 KB |
| `POOL_REFACTOR_ROADMAP.md` | 25 KB |
| `COMPLEXITY_SIMILARITY_RESULTS_2026_06_02.md` | 25 KB |
| `TIME_ESTIMATION_METHOD.md` | 25 KB |
| `REP_PRUNING_LOG.md` | 25 KB |
| `issues_preprocessing.md` | 25 KB |
| `tests/experiments/framework/F36.md` | 24 KB |
| `PAPER_OUTLINE.md` | 24 KB |
| `issues_antibody.md` | 24 KB |
| `EXPERIMENT_OVERVIEW.md` | 24 KB |
| `issues_dta_reaction.md` | 24 KB |
| `RESULTS_ANALYSIS_2026_05_13.md` | 23 KB |
| `tests/slurm_scripts/runbook/README.md` | 22 KB |
| `N06_HANDOFF_2026_09_11.md` | 22 KB |
| `AUDIT_NOISE.md` | 22 KB |
| `tests/slurm_scripts/runbook/SUBMIT_TONIGHT.md` | 21 KB |
| `EXPERIMENT_AUDIT_2026_04_02.md` | 21 KB |
| `PIPELINE_OVERHAUL_ROADMAP.md` | 21 KB |
| `tests/experiments/framework/F38_ANALYSIS_HANDOFF.md` | 21 KB |
| `CGR_EMBEDDING_RESEARCH.md` | 21 KB |
| `NOISE_EXPERIMENT_ROADMAP.md` | 20 KB |
| `CACHE_DRIFT_ROOTCAUSE_2026_06_28.md` | 20 KB |
| `DECISIONS_FOR_USER.md` | 20 KB |
| `HANDOFF_FIGURES.md` | 20 KB |
| `issues_molecular.md` | 19 KB |
| `tests/experiments/noise/METRIC_DIRECTION_AUDIT.md` | 19 KB |
| `NOISE_HANDOFF_2026_09_16.md` | 18 KB |
| `tests/experiments/noise/NOISE_SEARCH_SPACE.md` | 18 KB |
| `docs/cross_species_pretraining_design.md` | 18 KB |
| `docs/cross_species_ness_design.md` | 18 KB |
| `IMPORTANCE_SCORING_TRIAGE_2026_09_03.md` | 18 KB |
| `NOISE_AND_QUEUE_REQUIREMENTS.md` | 18 KB |
| `COMPLEXITY_RESULTS_GUIDE.md` | 17 KB |
| `AUDIT_SIMILARITY.md` | 16 KB |
| `scratchpad_hybrid_diagram_prompt.md` | 16 KB |
| `SIMILARITY_CODE_FAULTS_HANDOFF_2026_08_26.md` | 16 KB |
| `OPIG_MOLRXN_SMOKE_2026_06_03.md` | 16 KB |
| `SUPERVISOR_MEETING.md` | 16 KB |
| `SIMILARITY_HANDOFF_PROMPT.md` | 16 KB |
| `tests/EXPERIMENT_TRACKER.md` | 16 KB |
| `OPIG_EXPERIMENT_AUDIT.md` | 15 KB |
| `LOGIC_AUDIT_OUTSTANDING_2026_09_12.md` | 15 KB |
| `SECTION_DECISION_BRIEFS_2026_06_04.md` | 15 KB |
| `OPIG_ABDDG_SMOKE_2026_06_03.md` | 15 KB |
| `EXPERIMENT_HYBRID_ROADMAP.md` | 15 KB |
| `docs/cross_species_admet_design.md` | 15 KB |
| `HYBRID_REBUILD_PROMPTS_2026_07_16.md` | 15 KB |
| `DATASET_EXPANSION_2026_07_09.md` | 14 KB |
| `SUBMIT.md` | 14 KB |
| `SIMILARITY_OPEN_WORK_2026_08_27.md` | 14 KB |
| `AUDIT_FEDERATED.md` | 14 KB |
| `ESTIMATES_UPDATED_2026_06_04.md` | 14 KB |
| `SCAFFOLD_SIMILARITY_RESULTS.md` | 13 KB |
| `PRESENTATION_ANALYSIS_PROTOCOL.md` | 13 KB |
| `SIMILARITY_STUDY_HANDOFF.md` | 13 KB |
| `NEW_CHAT_PROMPT.md` | 13 KB |
| `SIMILARITY_HANDOFF_2026_08_27.md` | 13 KB |
| `AUDIT_DISTRIBUTION_AND_COHORTS.md` | 13 KB |
| `MULTISTAGE_PRUNING_PLAN.md` | 13 KB |
| `BLUEPRINT_GENERALIZATION_ROADMAP.md` | 12 KB |
| `AUDIT_OPIG.md` | 12 KB |
| `ALLOCATION_STRATEGY_HANDOFF_2026_08_21.md` | 12 KB |
| `AUDIT_NEW_CANDIDATES.md` | 12 KB |
| `COVERAGE_AUDIT_2026_06_04.md` | 12 KB |
| `SIMILARITY_OBJECTIVE_RESULTS_2026_07_10.md` | 12 KB |
| `AUDIT_DIAGNOSTICS_VARIANCE.md` | 12 KB |
| `SIMILARITY_METRICS_RESULTS_2026_07_11.md` | 11 KB |
| `AUDIT_GAP_ANALYSIS.md` | 11 KB |
| `AUDIT_FAILURE_CANDIDATES.md` | 11 KB |
| `SUPERVISOR_BRIEF_2026_05_13.md` | 11 KB |
| `tests/experiments/framework/F34_F35_HANDOFF.md` | 11 KB |
| `tests/experiments/framework/F38_STEPS.md` | 11 KB |
| `tests/slurm_scripts/noise_prelim/README.md` | 11 KB |
| `MOLECULEACE_BATTERY_HANDOFF_2026_09_11.md` | 11 KB |
| `SETUP_PRETRAINED_MODELS.md` | 11 KB |
| `SCAFFOLD_DIFF_HANDOFF_PROMPT.md` | 11 KB |
| `FEDERATED_SMOKE_2026_06_03.md` | 10 KB |
| `CLAUDE.md` | 10 KB |
| `RESULT_CAVEATS.md` | 10 KB |
| `EXPERIMENT_GROUPS_2026_09_18.md` | 10 KB |
| `NOISE_AUDIT_STATUS_2026_06_04.md` | 10 KB |
| `tests/slurm_scripts/runbook/WORKERS.md` | 10 KB |
| `BUDGET_SPLIT_NORMALIZATION_HANDOFF_2026_09_09.md` | 10 KB |
| `ULTRAREVIEW_PROMPT.md` | 10 KB |
| `AUDIT_REDUNDANCY.md` | 10 KB |
| `JOB_TRACKER_2026_05_22.md` | 10 KB |
| `SIMILARITY_HYBRID_HANDOFF_2026_08_24.md` | 10 KB |
| `AUDIT_SCORING.md` | 10 KB |
| `SIMILARITY_FRAMEWORK_INTEGRATION_PLAN_2026_08_21.md` | 10 KB |
| `SIMILARITY_FRAMEWORK_INTEGRATION_SPEC_2026_08_21.md` | 10 KB |
| `tests/experiments/noise/LOCAL_RUN_PLAN.md` | 9 KB |
| `EMPTY_RESULT_FILES_2026_08_27.md` | 9 KB |
| `HYBRID_IMPORTANCE_DEFECTS_2026_09_03.md` | 9 KB |
| `docs/cross_species_data_inventory.md` | 9 KB |
| `STRUCTURAL_OBJECTIVE_PROMPT.md` | 9 KB |
| `PAPER_FACTCHECK_2026_07_10.md` | 9 KB |
| `JOB_TRACKER_2026_05_14.md` | 9 KB |
| `tests/experiments/framework/F38_HANDOFF.md` | 9 KB |
| `PRUNING_RECALIBRATION_HANDOFF.md` | 9 KB |
| `SELECTOR_PREDICTION_RESULTS_2026_07_18.md` | 9 KB |
| `AUDIT_COMPLEXITY.md` | 9 KB |
| `paper.md` | 9 KB |
| `AUDIT_LOCAL_VALIDATION.md` | 9 KB |
| `AUDIT_REPRESENTATION_DISCOVERY.md` | 8 KB |
| `AUDIT_TRUE_OOF.md` | 8 KB |
| `SIMILARITY_HYBRID_NOT_BUILT_HANDOFF_2026_09_19.md` | 8 KB |
| `SCALING_MOVE_HANDOFF_2026_09_04.md` | 8 KB |
| `C32_ANALYSIS_HANDOFF_2026_08_29.md` | 8 KB |
| `SIMILARITY_FRAMEWORK_INTEGRATION_HANDOFF_2026_08_21.md` | 8 KB |
| `COEFFICIENT_BUILD_PROMPT.md` | 8 KB |
| `NEXT_SESSION_PROMPT.md` | 8 KB |
| `AUDIT_CALIBRATION.md` | 7 KB |
| `AUDIT_UNCERTAINTY.md` | 7 KB |
| `tests/experiments/noise/HANDOFF_CENSORING_DEEPDIVE.md` | 7 KB |
| `AUGMENTATION_STAGE2_PROMPT.md` | 7 KB |
| `AUDIT_AGGRESSIVE.md` | 7 KB |
| `SLURM_DEBUG_REPORT.md` | 7 KB |
| `DATASET_STATUS.md` | 7 KB |
| `AUDIT_METHOD_TESTS.md` | 7 KB |
| `REP_CACHE_REBUILD_2026_09_16.md` | 7 KB |
| `AUDIT_PATIENT_OVERLAP.md` | 7 KB |
| `AUDIT_LOWER_IS_BETTER_2026_06_29.md` | 7 KB |
| `JOB_TRACKER_2026_05_20.md` | 7 KB |
| `SCAFFOLD_DIFF_PUBLISHABILITY_HANDOFF.md` | 7 KB |
| `AUDIT_HPARAM.md` | 7 KB |
| `AUDIT_SENSITIVITY.md` | 6 KB |
| `OPIG_SLIDE_CONTENT.md` | 6 KB |
| `AUDIT_LB_PREDICTOR_ROBUSTNESS.md` | 6 KB |
| `SIMILARITY_LITERATURE_RESEARCH_PROMPT.md` | 6 KB |
| `EXPERIMENT_BUGS.md` | 6 KB |
| `AUDIT_FOUNDATION_RETRY.md` | 6 KB |
| `AUDIT_MULTISEED.md` | 6 KB |
| `tests/experiments/framework/F33_PRUNING_PROCEDURE.md` | 6 KB |
| `EXPERIMENT_RUN_STATUS_2026_05_11.md` | 6 KB |
| `NOISE_CLEANING_SIMILARITY_PROMPT.md` | 6 KB |
| `AUDIT_BLEND_SEARCH.md` | 6 KB |
| `AUDIT_DEATH_SUPREMACY.md` | 6 KB |
| `ANALYSIS_PLAN.md` | 6 KB |
| `AUDIT_BROKEN_SUBMISSION.md` | 5 KB |
| `tests/experiments/noise/PREDICTIONS.md` | 5 KB |
| `GEOMETRY_SURVEY_RESULT_2026_09_11.md` | 5 KB |
| `POST_PRECOMPUTE_COMMANDS.md` | 5 KB |
| `OPIG_RESULTS.md` | 5 KB |
| `AUDIT_HEPSPEC_V2.md` | 5 KB |
| `tests/experiments/framework/F33_PRUNING_PROTOCOL.md` | 5 KB |
| `PRECOMPUTE_REP_CACHE_HANDOFF_2026_09_06.md` | 5 KB |
| `AUDIT_DEEP_FAILURES.md` | 5 KB |
| `AUDIT_LEAKAGE_HUNT.md` | 5 KB |
| `CHANGES_BEFORE_RERUN.md` | 5 KB |
| `AUDIT_REVERSE_ENGINEER.md` | 5 KB |
| `competition_methods/fusion/F01_caruana_bagged_selection.md` | 5 KB |
| `competition_methods/README.md` | 4 KB |
| `CACHE_OPERATIONS_2026_06_04.md` | 4 KB |
| `AUDIT_TORCH27.md` | 4 KB |
| `tests/experiments/noise/RUNNING_WHILE_CLAUDE_IS_OFF.md` | 4 KB |
| `tests/experiments/similarity_metrics/README.md` | 4 KB |
| `HANDOFF_CREATE_HYBRID_AUDIT.md` | 4 KB |
| `F30_LOCAL_RUN_NOTES_2026_09_05.md` | 4 KB |
| `SVM_CAP_RESUBMIT_2026_05_27.md` | 3 KB |
| `FIX_4_SMALL_DEFECTS.md` | 3 KB |
| `competition_methods/validation/V09_nested_cross_validation.md` | 3 KB |
| `competition_methods/fusion/F06_diversity_penalized_selection.md` | 3 KB |
| `competition_methods/fusion/F05_stacking_nested_cv.md` | 3 KB |
| `competition_methods/fusion/F02_geomean_rank_pool.md` | 3 KB |
| `competition_methods/fusion/F10_snapshot_averaging.md` | 3 KB |
| `competition_methods/validation/V08_holdout_fusion_test_gate.md` | 3 KB |
| `tests/experiments/framework/F38_DECISIONS.md` | 3 KB |
| `competition_methods/fusion/F03_hodges_lehmann_pseudomedian.md` | 3 KB |
| `competition_methods/fusion/F09_trimmed_winsorized_mean.md` | 3 KB |
| `competition_methods/validation/V12_royston_d_schemper_v.md` | 3 KB |
| `AUDIT_CROSS_POLLINATE.md` | 3 KB |
| `competition_methods/validation/V11_brier_residual_diversity.md` | 2 KB |
| `competition_methods/validation/V10_permutation_null_delta_c.md` | 2 KB |
| `competition_methods/validation/V04_uno_ipcw_concordance.md` | 2 KB |
| `competition_methods/validation/V02_harrell_bootstrap_optimism.md` | 2 KB |
| `competition_methods/fusion/F07_bayesian_model_averaging.md` | 2 KB |
| `FIX_1_DEAD_IMPORTS.md` | 2 KB |
| `competition_methods/fusion/F04_bates_granger.md` | 2 KB |
| `competition_methods/validation/V07_adversarial_validation.md` | 2 KB |
| `FIX_3_FOLD_RESELECTION.md` | 2 KB |
| `competition_methods/validation/V05_integrated_brier_score.md` | 2 KB |
| `competition_methods/fusion/F08_frequentist_model_averaging.md` | 2 KB |
| `FIX_2_FEDERATED_METRIC.md` | 2 KB |
| `competition_methods/validation/V03_632plus_bootstrap.md` | 2 KB |
| `AUDIT_PER_PATIENT.md` | 2 KB |
| `competition_methods/validation/V01_repeated_stratified_kfold.md` | 2 KB |
| `competition_methods/validation/V06_cumulative_dynamic_auc.md` | 2 KB |
| `REDUNDANCY_PENALTY_PROMPT.md` | 2 KB |

### O. Stray files at the root and unreferenced scripts

Why: Empty files (240, 340), a saved command output (imports), a saved web page (rl.html), a chat export (serverChat.txt), an old patch, and four scripts in scripts/ that no tracked code or shell script names.

Proposed action: git rm.

10 files, 308 KB. One row per file, largest first.

| Path | Size |
|---|---|
| `rl.html` | 216 KB |
| `serverChat.txt` | 51 KB |
| `scripts/setup_server_models.sh` | 10 KB |
| `scripts/discover_vet_targets.py` | 8 KB |
| `scripts/verify_models.py` | 8 KB |
| `scripts/run_molecular_baselines.py` | 8 KB |
| `DOWNSTREAM_MODELS_ALIGNMENT.patch` | 7 KB |
| `imports` | 667 B |
| `240` | 0 B |
| `340` | 0 B |

### C. catboost_info/

Why: Training logs CatBoost writes by default into the working directory.

Proposed action: git rm.

3 files, 39 KB. One row per file, largest first.

| Path | Size |
|---|---|
| `catboost_info/learn/events.out.tfevents` | 24 KB |
| `catboost_info/learn_error.tsv` | 8 KB |
| `catboost_info/time_left.tsv` | 7 KB |

## Test results, 2026-10-23

Each test file in `tests/` was run on its own, with a 5-minute limit per file, in the anaconda base Python 3.11.7 on the laptop.
This was done because the full `python -m pytest` run stopped twice with no CPU use, after about 330 of the 360 tests.
26 of the 33 test files pass. The other 7 are below.
None of the 7 tests the assay runner, `tests/noise_column.py`, `src/kirby/noise_spec.py` or `src/kirby/noise_robustness.py`.
Nothing in them has been changed, and no test has been weakened.

One row per test file.

| Test file | Result | What it reports |
|---|---|---|
| `tests/test_phase4b_models.py` | 2 failed, then still running at 5 minutes | loads the ESM3, SaProt and Boltz-2 protein models from the external drive; one run sat at 0% CPU for over 30 minutes |
| `tests/test_py311_reps.py` | still running at 5 minutes, no output | not investigated |
| `tests/test_cross_species_ness.py` | 1 of 31 failed | the canine sequence loader's error message no longer contains the word the test looks for (`data_inventory`) |
| `tests/test_metric_direction_persisted.py` | does not import | `No module named 'experiments'`; it passes only when `tests/` is on the import path |
| `tests/test_noise_alignment.py` | 1 failed | `tests/experiments/noise/noise_helpers.py` lists one more noise condition (`sparse_dprime_up_p10`) than the test expects |
| `tests/test_x_mol_rd_analysis.py` | 1 of 18 failed | a cache file name gained `__full`, and the test expects the old name |
| `tests/scripts/test_alloc_determinism.py` | error | it is a script with a function named `test_one(dataset)`, not a pytest test, so pytest finds no `dataset` fixture |

Options, per file: fix the test or the code, or move the file out of `tests/` so pytest skips it. Author to choose.
