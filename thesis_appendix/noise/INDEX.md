# Thesis appendix items: noise robustness (Appendix B)

One row is one table or figure in the thesis's noise-robustness appendix. The name is permanent:
it is the file name in `items/`, the label `app:noise:<name>`, and the same name the paper uses
for the item this one was copied from (`qsar_qm_models/supplementary/INDEX.md`).

**Thesis section** is where thesis.tex `\input`s the item. **Status** starts as the paper's status
on the day of seeding and is then the thesis's own: change it when you edit the copy.
**Seeded** is the date it was copied from the paper; after that date the two are independent.

`thesis_appendix/seed_from_paper.py` appends a row here each time it copies an item. Fill in
the Thesis section column by hand when you place the item in thesis.tex.

| Name | Kind | Title | Thesis section | Status | Seeded | Figures |
|---|---|---|---|---|---|---|
| hyperparameters_default | table | Default hyperparameters for every model configuration | Appendix B, app:noise:sec:hyperparameters_default | current (at seeding) | 2026-09-27 | — |
| hyperparameters_tuned | table | Tuned settings that replace a default, per dataset | Appendix B, app:noise:sec:hyperparameters_tuned | current (at seeding) | 2026-09-27 | — |
| pdv_descriptors | table | The 200 descriptors that make up PDV | Appendix B, app:noise:sec:pdv_descriptors | current (at seeding) | 2026-09-27 | — |
| model_redundancy | table | Pairwise Spearman correlations between model NDS profiles | Appendix B, app:noise:sec:model_redundancy | stale (at seeding) | 2026-09-27 | — |
| representation_redundancy | table | Pairwise Spearman correlations between representation NDS profiles | Appendix B, app:noise:sec:representation_redundancy | stale (at seeding) | 2026-09-27 | — |
| model_icc | table | ICC(1,1) for model pairs | Appendix B, app:noise:sec:model_icc | stale (at seeding) | 2026-09-27 | — |
| excluded_configurations | table | Configurations excluded from the robustness analysis | Appendix B, app:noise:sec:excluded_configurations | stale (at seeding) | 2026-09-27 | — |
| ecfp4_overview | figure | Global overview of noise robustness on ECFP4 | Appendix B, app:noise:sec:ecfp4_overview | stale (at seeding) | 2026-09-27 | fig1_supp_ecfp4_overview.png |
| clean_r2_vs_robustness | figure | Clean R² against NDS, PDV, Gaussian | Appendix B, app:noise:sec:clean_r2_vs_robustness | stale (at seeding) | 2026-09-27 | fig3_ranking_consistency.png |
| bayesian_transformation_tests | table | Wilcoxon tests of the Bayesian transformations by representation | Appendix B, app:noise:sec:bayesian_transformation_tests | stale (at seeding) | 2026-09-27 | — |
| uncertainty_metrics_ecfp4 | table | Uncertainty metrics for probabilistic models on ECFP4 | Appendix B, app:noise:sec:uncertainty_metrics_ecfp4 | stale (at seeding) | 2026-09-27 | — |
| assay_variance_decomposition | figure | ANOVA variance decomposition on the three assay datasets | Appendix B, app:noise:sec:assay_variance_decomposition | stale (at seeding) | 2026-09-27 | fig_validation_anova.png |
| rf_vs_qrf_assay | table | RF against QRF on the three assay datasets | Appendix B, app:noise:sec:rf_vs_qrf_assay | stale (at seeding) | 2026-09-27 | — |
