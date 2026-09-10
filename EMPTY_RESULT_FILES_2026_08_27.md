# Result files holding no measurement

`classify_result_payload()` in `tests/experiments/_shared.py` run recursively over
`tests/results` on 2026-08-27. 1541 JSON files: 1401 hold a measurement, 85 carry the
`skipped` marker, 55 hold neither. Those 55 are listed below.

Size is bytes. Modification date is the file mtime, local time. Reason is quoted from
the file — an `error`, `reason` or `note` string it stores. Sections are the top-level
keys outside the metadata set (`timestamp`, `experiment`, `dataset`, `config`, `meta`,
`metric_name`, `lower_is_better`, and the rest of `RESULT_META_KEYS`).

None of the 55 carries `result_status` or `has_computed_results`. No file anywhere under
`tests/results` carries either key: all 1541 predate the stamp `save_experiment_results`
now writes.

| # | Directory | Dataset | Bytes | Modified | Reason recorded in the file | Sections |
|---|---|---|---|---|---|---|
| 1 | `tests/results/annitia/external` | zeroshot | 976 | 2026-05-08 03:11 | — | fib4_high, fib4_high_2007, fib4_high_2017, t2dm_only, no_viral |
| 2 | `tests/results/c07_data_complexity_partitions` | abcov_kd | 93 | 2026-05-08 18:48 | — | partitions, search_speed |
| 3 | `tests/results/c07_data_complexity_partitions` | flab_immunogenicity_marks | 93 | 2026-05-08 22:57 | — | partitions, search_speed |
| 4 | `tests/results/c07_data_complexity_partitions` | flab_polyreact_jain_bvp | 93 | 2026-05-08 13:48 | — | partitions, search_speed |
| 5 | `tests/results/c07_data_complexity_partitions` | flab_polyreact_jain_cic | 93 | 2026-05-08 15:57 | — | partitions, search_speed |
| 6 | `tests/results/c07_data_complexity_partitions` | ginkgo_heparin | 93 | 2026-05-20 03:22 | — | partitions, search_speed |
| 7 | `tests/results/c07_data_complexity_partitions` | humanness_therapeutic | 93 | 2026-05-13 20:53 | — | partitions, search_speed |
| 8 | `tests/results/c10_noise_robustness` | abagym_Ang2 | 87 | 2026-05-13 20:30 | noiseInject not installed | error |
| 9 | `tests/results/c10_noise_robustness` | abagym_EGFR | 87 | 2026-05-14 04:36 | noiseInject not installed | error |
| 10 | `tests/results/c10_noise_robustness` | abagym_HER2 | 87 | 2026-05-08 11:36 | noiseInject not installed | error |
| 11 | `tests/results/c10_noise_robustness` | abagym_VEGF | 87 | 2026-05-13 20:36 | noiseInject not installed | error |
| 12 | `tests/results/c10_noise_robustness` | abcov_kd | 87 | 2026-05-08 19:59 | noiseInject not installed | error |
| 13 | `tests/results/c10_noise_robustness` | flab_immunogenicity_marks | 87 | 2026-05-09 00:46 | noiseInject not installed | error |
| 14 | `tests/results/c10_noise_robustness` | flab_polyreact_jain_bvp | 87 | 2026-05-08 14:42 | noiseInject not installed | error |
| 15 | `tests/results/c10_noise_robustness` | flab_polyreact_jain_cic | 87 | 2026-05-08 17:01 | noiseInject not installed | error |
| 16 | `tests/results/c10_noise_robustness` | humanness_therapeutic | 87 | 2026-05-14 01:06 | noiseInject not installed | error |
| 17 | `tests/results/c21_single_rep_augmentation` | humanness_therapeutic | 120 | 2026-05-19 20:15 | no_augmentation_reps | error |
| 18 | `tests/results/c24_blueprint_transfer` | abagym_Ang2 | 80 | 2026-05-18 05:31 | insufficient_folds | error |
| 19 | `tests/results/c24_blueprint_transfer` | abagym_VEGF | 80 | 2026-05-18 10:35 | insufficient_folds | error |
| 20 | `tests/results/comparisons` | antibody_region_entropy | 294 | 2026-03-10 14:10 | — | flab_binding_phillips_h3, flab_expression_koenig, flab_binding_koen… |
| 21 | `tests/results/comparisons/comparisons` | antibody_region_entropy | 294 | 2026-03-25 18:27 | — | flab_binding_phillips_h3, flab_expression_koenig, flab_binding_koen… |
| 22 | `tests/results/complexity/c06_within_family_gradient` | humanness_therapeutic | 110 | 2026-06-01 10:59 | regression_only | error |
| 23 | `tests/results/complexity/c10_noise_robustness` | abagym_Ang2 | 87 | 2026-06-01 13:33 | noiseInject not installed | error |
| 24 | `tests/results/complexity/c10_noise_robustness` | abagym_EGFR | 87 | 2026-05-27 20:09 | noiseInject not installed | error |
| 25 | `tests/results/complexity/c10_noise_robustness` | abagym_HER2 | 87 | 2026-06-02 05:33 | noiseInject not installed | error |
| 26 | `tests/results/complexity/c10_noise_robustness` | abagym_VEGF | 87 | 2026-06-01 14:05 | noiseInject not installed | error |
| 27 | `tests/results/complexity/c10_noise_robustness` | abcov_kd | 87 | 2026-06-01 08:42 | noiseInject not installed | error |
| 28 | `tests/results/complexity/c10_noise_robustness` | flab_immunogenicity_marks | 87 | 2026-06-01 10:47 | noiseInject not installed | error |
| 29 | `tests/results/complexity/c10_noise_robustness` | flab_polyreact_jain_bvp | 87 | 2026-06-01 08:57 | noiseInject not installed | error |
| 30 | `tests/results/complexity/c10_noise_robustness` | flab_polyreact_jain_cic | 87 | 2026-06-01 08:38 | noiseInject not installed | error |
| 31 | `tests/results/complexity/c10_noise_robustness` | humanness_therapeutic | 87 | 2026-06-01 13:41 | noiseInject not installed | error |
| 32 | `tests/results/complexity/c21_single_rep_augmentation` | humanness_therapeutic | 120 | 2026-05-19 20:15 | no_augmentation_reps | error |
| 33 | `tests/results/complexity/c24_blueprint_transfer` | abagym_Ang2 | 80 | 2026-05-18 05:31 | insufficient_folds | error |
| 34 | `tests/results/complexity/c24_blueprint_transfer` | abagym_VEGF | 80 | 2026-05-18 10:35 | insufficient_folds | error |
| 35 | `tests/results/pretrain_rep_experiment` | ginkgo_hic__pt_oas_species | 107 | 2026-05-15 03:35 | insufficient_shared_reps | error, shared_reps |
| 36 | `tests/results/s07_noise_bridge` | buchwald_hartwig | 73 | 2026-05-14 22:10 | — | cliff_method |
| 37 | `tests/results/similarity/similarity/s03_feature_concentration` | tcga_brca_pam50 | 118 | 2026-04-01 00:58 | y_true and y_pred have different number of output (1!=4) | error |
| 38 | `tests/results/similarity/similarity/s03_feature_concentration` | tcga_luad_vs_lusc | 95 | 2026-04-01 00:58 | No fitted result for aug=uncapped | error |
| 39 | `tests/results/similarity/similarity/s04_activity_cliffs` | abcov_kd | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 40 | `tests/results/similarity/similarity/s04_activity_cliffs` | ddg_combined | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 41 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_aggregation_jain | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 42 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_aggregation_jain_hicrt | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 43 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_immunogenicity_marks | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 44 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_pk_jain_fcrnrt | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 45 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_pk_jain_heprt | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 46 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_polyreact_jain_bvp | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 47 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_polyreact_jain_cic | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 48 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_thermo_jain | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 49 | `tests/results/similarity/similarity/s04_activity_cliffs` | flab_thermo_tresanco | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 50 | `tests/results/similarity/similarity/s04_activity_cliffs` | tcga_brca_pam50 | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 51 | `tests/results/similarity/similarity/s04_activity_cliffs` | tcga_luad_vs_lusc | 81 | 2026-04-01 00:58 | ecfp4 not available | error |
| 52 | `tests/results/similarity/similarity/s06_applicability_domain` | tcga_brca_pam50 | 118 | 2026-04-01 00:58 | y_true and y_pred have different number of output (1!=4) | error |
| 53 | `tests/results/similarity/similarity/s06_applicability_domain` | tcga_luad_vs_lusc | 95 | 2026-04-01 00:58 | No fitted result for aug=uncapped | error |
| 54 | `tests/results/similarity/similarity/s07_noise_bridge` | tcga_brca_pam50 | 118 | 2026-04-01 00:58 | y_true and y_pred have different number of output (1!=4) | error |
| 55 | `tests/results/similarity/similarity/s07_noise_bridge` | tcga_luad_vs_lusc | 95 | 2026-04-01 00:58 | No fitted result for aug=uncapped | error |

## The four with no reason string

`tests/results/c07_data_complexity_partitions/*.json` (6 files) — `partitions` is `{}`,
`search_speed` is the string `"full"`.

`tests/results/comparisons/antibody_region_entropy.json` and
`tests/results/comparisons/comparisons/antibody_region_entropy.json` — five dataset keys,
each holding `heavy` and/or `light` set to `{}`.

`tests/results/annitia/external/zeroshot.json` — eight cohort keys, each holding
`hepatic_err` and `death_err` set to the string
`"'NoneType' object is not callable"`, and `hep_passes_gate` / `death_passes_gate` set
to `false`.

`tests/results/s07_noise_bridge/buchwald_hartwig.json` — one key, `cliff_method`, set to
the string `"none"`.

