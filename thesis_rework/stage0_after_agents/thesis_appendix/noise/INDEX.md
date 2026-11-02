# Thesis appendix items: Chapter 3, noise robustness

One row is one table or figure in the Chapter 3 section of the thesis's "Additional files"
appendix (`\label{app:add:robustness}`, the first section of `\label{app:additional}`). The name is permanent:
it is the file name in `items/`, the label `app:noise:<name>`, and the same name the paper uses
for the item this one was copied from (`qsar_qm_models/supplementary/INDEX.md`).

**Thesis section** is the Additional file entry that `\input`s the item: the entry's
`app:noise:sec:<name>` label in the Chapter 3 section. One entry can hold several items; the three
tables of the paper's Additional file 1 (hyperparameters_default, hyperparameters_tuned and
pdv_descriptors) share one entry, which carries all three labels. **Status** starts as the paper's status
on the day of seeding and is then the thesis's own: change it when you edit the copy.
**Seeded** is the date it was copied from the paper; after that date the two are independent.

`thesis_appendix/seed_from_paper.py` appends a row here each time it copies an item. Fill in
the Thesis section column by hand when you place the item in thesis.tex.

| Name | Kind | Title | Thesis section | Status | Seeded | Figures |
|---|---|---|---|---|---|---|
| hyperparameters_default | table | Default hyperparameters for every model configuration | Additional files, Chapter 3 section, entry app:noise:sec:hyperparameters_default (one entry shared with the other two tables of the paper's Additional file 1) | current (at seeding) | 2026-09-27 | — |
| hyperparameters_tuned | table | Tuned settings that replace a default, per dataset | Additional files, Chapter 3 section, entry app:noise:sec:hyperparameters_tuned (one entry shared with the other two tables of the paper's Additional file 1) | current (at seeding) | 2026-09-27 | — |
| pdv_descriptors | table | The 200 descriptors that make up PDV | Additional files, Chapter 3 section, entry app:noise:sec:pdv_descriptors (one entry shared with the other two tables of the paper's Additional file 1) | current (at seeding) | 2026-09-27 | — |
| model_redundancy | table | Pairwise Spearman correlations between model NDS profiles | Additional files, Chapter 3 section, entry app:noise:sec:model_redundancy | stale (at seeding) | 2026-09-27 | — |
| representation_redundancy | table | Pairwise Spearman correlations between representation NDS profiles | Additional files, Chapter 3 section, entry app:noise:sec:representation_redundancy | stale (at seeding) | 2026-09-27 | — |
| model_icc | table | ICC(1,1) for model pairs | Additional files, Chapter 3 section, entry app:noise:sec:model_icc | stale (at seeding) | 2026-09-27 | — |
| excluded_configurations | table | Configurations excluded from the robustness analysis | Additional files, Chapter 3 section, entry app:noise:sec:excluded_configurations | stale (at seeding) | 2026-09-27 | — |
| ecfp4_overview | figure | Global overview of noise robustness on ECFP4 | Additional files, Chapter 3 section, entry app:noise:sec:ecfp4_overview | stale (at seeding) | 2026-09-27 | fig1_supp_ecfp4_overview.png |
| clean_r2_vs_robustness | figure | Clean R² against NDS, PDV, Gaussian | Additional files, Chapter 3 section, entry app:noise:sec:clean_r2_vs_robustness | stale (at seeding) | 2026-09-27 | fig3_ranking_consistency.png |
| bayesian_transformation_tests | table | Wilcoxon tests of the Bayesian transformations by representation | Additional files, Chapter 3 section, entry app:noise:sec:bayesian_transformation_tests | stale (at seeding) | 2026-09-27 | — |
| uncertainty_metrics_ecfp4 | table | Uncertainty metrics for probabilistic models on ECFP4 | Additional files, Chapter 3 section, entry app:noise:sec:uncertainty_metrics_ecfp4 | stale (at seeding) | 2026-09-27 | — |
| assay_variance_decomposition | figure | ANOVA variance decomposition on the three assay datasets | Additional files, Chapter 3 section, entry app:noise:sec:assay_variance_decomposition | stale (at seeding) | 2026-09-27 | fig_validation_anova.png |
| rf_vs_qrf_assay | table | RF against QRF on the three assay datasets | Additional files, Chapter 3 section, entry app:noise:sec:rf_vs_qrf_assay | stale (at seeding) | 2026-09-27 | — |
