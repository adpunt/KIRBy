# Pipeline Readiness TODO — BOM Section 10 Dataset-by-Dataset Review

**Purpose:** Track which datasets have been assessed for the pipeline readiness table. Each dataset needs a user-confirmed row with base reps and augmentation signal. This document is the single source of truth for what's been done and what remains.

**Last updated:** 2026-03-24

---

## Rules for Each Dataset Review

1. Show ALL standalone results: every rep sorted by primary metric, with dimension, all metrics, best model, rep family
2. Show ALL augmentation test results: base + augment, scores, deltas
3. Show ALL supplementary data that exists (fine-tuning, forward compat, graph survey, comparisons, OPIG, DMS)
4. DISCUSS — explain what I think the cutoff should be and why, then wait for user decisions
5. User decides the cutoff and augmentation selections
6. Write the row ONLY after user confirms
7. Do NOT skip supplementary data — fine-tuning, OPIG, comparisons all inform the decision

**Primary metrics by domain:**
- Molecular: RMSE (lower = better), with R² and ρ as supporting
- Antibody (regression): Spearman ρ (higher = better)
- TCGA classification: AUROC (binary) or F1w (multi-class)
- Reaction: R² (higher = better)
- Nucleotide: MCC (higher = better)
- DTA: CI (higher = better)
- Antibody CLF: AUROC / F1w

**Viability thresholds (starting points, user may override):**
- Molecular: R² > 0.5
- Reaction: R² > 0.2
- TCGA clf: AUROC > 0.6
- TCGA multi-class: F1w > 0.4
- Antibody: ρ > 0.15
- DTA: CI > 0.55

**FT rep notation:**
- `*` = **FT-only.** The frozen version of this rep is dead (produces no signal). There is no fallback — if FT is turned off, this rep is excluded entirely.
- `**` = **FT improves, frozen fallback exists.** The FT version is better, but the frozen version is a viable base rep in its own right. When FT is disabled, the frozen version substitutes in.

---

## Completed

| # | Dataset | Domain | Samples | Status | BOM Location |
|---|---------|--------|---------|--------|-------------|
| 1 | caco2 | Molecular | 2,158 | DONE | Section 10 (new format) |
| 2 | herg | Molecular | 1,412 | DONE | Section 10 (new format) |
| 3 | logd | Molecular | 5,028 | DONE | Section 10 (new format) |
| 4 | qm9 | Molecular | 9,978 | DONE | Section 10 (new format) |
| 5 | adme_hlm | Molecular | 3,086 | DONE | Section 10 (new format) |
| 6 | adme_rlm | Molecular | 3,053 | DONE | Section 10 (new format) |
| 7 | adme_mdr1 | Molecular | 2,641 | DONE | Section 10 (new format) |
| 8 | adme_solubility | Molecular | 2,172 | DROPPED | R²=0.32 ceiling, not viable |
| 47 | tcga_luad_vs_lusc | TCGA | 1,017 | DONE | Section 10 (new format) |
| 48 | tcga_brca_pam50 | TCGA | 821 | DONE | Section 10 (new format) |
| 49 | tcga_brca_survival | TCGA | 151 | DROPPED | All R² negative |
| 9 | flab_thermo_tresanco | Antibody | 672 | DONE | Section 10 (new format) |
| 10 | flab_pk_jain_heprt | Antibody | 130 | DONE | Section 10 (new format) |
| 11 | flab_pk_jain_fcrnrt | Antibody | 132 | DONE | Section 10 (new format) |
| 12 | flab_polyreact_jain_cic | Antibody | 137 | DONE | Section 10 (new format) |
| 13 | flab_aggregation_jain_hicrt | Antibody | 137 | DONE | Section 10 (new format) |
| 14 | flab_thermo_jain | Antibody | 137 | DONE | Section 10 (new format) |
| 15 | abcov_kd | Antibody | 141 | DONE | Section 10 (new format) |
| 16 | flab_aggregation_jain | Antibody | 137 | DONE | Section 10 (new format) |
| 17 | flab_immunogenicity_marks | Antibody | 217 | DONE | Section 10 (new format) |
| 18 | flab_polyreact_jain_bvp | Antibody | 137 | DONE | Section 10 (new format) |
| 21 | ddg_combined | Antibody | 429 | DONE | Section 10 (new format) |
| 34 | tz_her2_binary | Ab CLF | 5,000 | DONE | Section 10 (new format) |
| 35 | humanness_therapeutic | Ab CLF | 564 | DONE | Section 10 (new format) |
| 38 | ginkgo_acsins | Ginkgo | 242 | DONE | Section 10 (new format) |
| 40 | ginkgo_colloidal | Ginkgo | 242 | DONE | Section 10 (new format) |
| 41 | ginkgo_heparin | Ginkgo | 94 | DONE | Section 10 (new format) |
| 42 | ginkgo_hic | Ginkgo | 242 | DONE | Section 10 (new format) |
| 43 | ginkgo_pr_cho | Ginkgo | 197 | DONE | Section 10 (new format) |
| 45 | ginkgo_titer | Ginkgo | 239 | DONE | Section 10 (new format) |
| 39 | ginkgo_aggregation | Ginkgo | 242 | DROPPED | best ρ=0.181, near random |
| 44 | ginkgo_purity | Ginkgo | 242 | DROPPED | best ρ=0.220, weak + high variance |
| 46 | ginkgo_tm2 | Ginkgo | 193 | DROPPED | best ρ=0.166, near random |
| 28 | abagym_all | AbAgym | 2,247 | DROPPED | best ρ=0.145, below viability |
| 29 | abagym_Ang2 | AbAgym | 397 | DONE | Section 10 (new format) |
| 30 | abagym_EGFR | AbAgym | 659 | DONE | Section 10 (new format) |
| 31 | abagym_HER2 | AbAgym | 146 | DONE | Section 10 (new format) |
| 32 | abagym_lysozyme | AbAgym | 627 | DONE | Section 10 (new format) |
| 33 | abagym_VEGF | AbAgym | 418 | DONE | Section 10 (new format) |

**caco2 (2,158 samples):**
| Tier | Reps |
|------|------|
| Top Tier (8) | graph_kernel, mol2vec, mhg_gnn, ecfp4, pdv, grover, avalon, mordred |
| General (12) | graph_kernel, mol2vec, mhg_gnn, ecfp4, pdv, grover, avalon, mordred, top_torsion, smited, atom_pair, molformer |
| Augmentation (2) | maccs, pharmacophore3d |

**herg (1,412 samples):**
| Tier | Reps |
|------|------|
| Top Tier (8) | mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, mol2vec, ecfp4 |
| General (15) | mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, mol2vec, ecfp4, molformer, chemberta, atom_pair, chembert, smited, top_torsion, graphmvp |
| Augmentation (5) | maccs, pharmacophore3d, coulomb_matrix, usr, usrcat |

**logd (5,028 samples):**
| Tier | Reps |
|------|------|
| Top Tier (8) | mhg_gnn, mordred, grover, mol2vec, pdv, ecfp4, smited, molformer |
| General (14) | mhg_gnn, mordred, grover, mol2vec, pdv, ecfp4, smited, molformer, avalon, atom_pair, chemberta, top_torsion, chembert, graph_kernel |
| Augmentation (7) | selformer, rdkfp, graphmvp, maccs, pharmacophore3d, molclr, usrcat |

**qm9 (9,978 samples):**
| Tier | Reps |
|------|------|
| Top Tier (6) | mordred, pdv, grover, mhg_gnn, avalon, e2e_schnet* |
| General (16) | mordred, pdv, grover, mhg_gnn, avalon, rdkfp, atom_pair, smited, ecfp4, selformer, maccs, chemberta, mol2vec, chembert, graphmvp, molformer |
| Augmentation (4) | top_torsion, pharmacophore3d, usrcat, coulomb_matrix |

*e2e_schnet: factory rep, requires per-fold 3D fine-tuning (128D)

**adme_hlm (3,086 samples) — hard task, R² ceiling 0.46, viability by ρ > 0.5:**
| Tier | Reps |
|------|------|
| Top Tier (6) | mordred, pdv, mhg_gnn, smited, molformer, grover |
| General (16) | mordred, pdv, mhg_gnn, smited, molformer, grover, mol2vec, chemberta, atom_pair, chembert, graph_kernel, avalon, top_torsion, ecfp4, selformer, graphmvp |
| Augmentation (5) | pharmacophore3d, maccs, usrcat, coulomb_matrix, usr |

**adme_rlm (3,053 samples) — hard task, R² ceiling 0.49, viability by ρ > 0.5:**
| Tier | Reps |
|------|------|
| Top Tier (6) | pdv, mhg_gnn, smited, mordred, molformer, grover |
| General (16) | pdv, mhg_gnn, smited, mordred, molformer, grover, mol2vec, chemberta, chembert, graph_kernel, atom_pair, selformer, avalon, ecfp4, top_torsion, graphmvp |
| Augmentation (5) | pharmacophore3d, maccs, usrcat, coulomb_matrix, usr |

**adme_mdr1 (2,641 samples):**
| Tier | Reps |
|------|------|
| Top Tier (5) | mordred, pdv, mhg_gnn, avalon, grover |
| General (16) | mordred, pdv, mhg_gnn, avalon, grover, smited, molformer, atom_pair, graph_kernel, maccs, chemberta, ecfp4, chembert, selformer, top_torsion, rdkfp |
| Augmentation (2) | pharmacophore3d, maccs |

### Molecular Domain-Level Decisions

**FT reps (apply to all molecular datasets):**
- lora_molformer** and contrastive_molformer** are added as general-tier base reps across all molecular datasets. They don't compete with the top tier but consistently beat maccs and provide FT-adapted LM signal. Frozen molformer is the fallback when FT is disabled.
- All other FT variants (chemberta FT, selformer FT, freeze_head anything) are excluded — they never beat ecfp4.

**Augmentation (apply to all molecular datasets):**
- pharmacophore3d and maccs are domain-wide augmentation reps for all molecular datasets. pharmacophore3d is the most consistent augment rep tested (13/15 improve in graph survey, positive on 5/6 phase5 datasets). maccs shows small but reliable signal everywhere (6/7 datasets).
- usrcat, coulomb_matrix, usr: inconsistent across datasets. Need further testing before domain-wide inclusion.

**ACTION items:**
- Run E2E SchNet on logd, herg, caco2 to determine if it generalizes beyond qm9 (currently confirmed top tier on qm9 only, R²=0.95).
- Run usrcat, coulomb_matrix, usr augmentation tests on remaining molecular datasets (adme_hlm, adme_rlm, adme_mdr1, adme_solubility) to resolve inconsistency. Small test — just these 3 reps as augments on the top base rep per dataset.

### TCGA Domain-Level Decisions

**tcga_luad_vs_lusc (1,017 samples, binary, AUROC):**
| Tier | Reps |
|------|------|
| Top Tier (8) | scfoundation_ft*, gene_geneformer_v2, gene_bulkrnabert, methyl_hvg, gene_geneformer, gene_chromosome_arm, wsi_titan, methyl_pca50, cnv_pca50 |
| General (28) | all 28 reps with AUROC > 0.6 + geneformer_ft** |
| Augmentation | none — no augment improved the base |

**tcga_brca_pam50 (821 samples, 4-class, F1w):**
| Tier | Reps |
|------|------|
| Top Tier (3) | scfoundation_ft*, gene_geneformer_v2, gene_geneformer, gene_bulkrnabert |
| General (27) | all 27 reps with F1w > 0.4 + geneformer_ft** |
| Augmentation (6) | hrd_score, mut_tmb, purity_consensus, gene_stats, gene_chromosome_arm, immune_xcell |

**FT reps (apply to both TCGA datasets):**
- scfoundation_ft* — FT-only, frozen scfoundation is dead (AUROC=0.500). Best single result: LUAD 0.988 AUROC, BRCA 0.877 F1w (hybrid). No fallback.
- geneformer_ft** — FT improves (LUAD 0.977, BRCA 0.967 AUROC), but frozen geneformer_v2 (0.946/0.861) is already top tier. Frozen V2 substitutes when FT disabled.

**Dead reps (both datasets):** clinical, gene_scfoundation (frozen), heterogeneity_math, msi_status, wsi_conch, wsi_hoptimus1, wsi_uni2h, wsi_virchow2 — all AUROC=0.500 exactly.

**Pending factory reps:** gene_pca100, gene_hallmark, gene_progeny showed strong results in FT survey (gene_pca100: 0.915 AUROC LUAD, 0.806 F1w BRCA). Not yet in phase5 baselines — will likely enter general tier when available.

### Antibody Inter-Family Domain-Level Decisions

**General tier cutoff: ~60% of best ρ per dataset.** Antibody-specific LMs (ablang2, igt5) and general protein LMs (prot_t5, esm2 variants) dominate. Rankings are highly dataset-specific.

**Domain-wide dead reps:** aido_protein, igbert, igbert_cls, mint, xtrimopglm — NaN or near-zero on all datasets.

**Size-dependent reps:** esm3 is spectacular on tresanco (n=672, ρ=0.805) but dead on n≈130 datasets. blosum/onehot strong on tresanco but weak on small datasets.

**flab_thermo_tresanco (n=672, best ρ=0.805, cutoff=0.48):**
| Tier | Reps |
|------|------|
| Top Tier (9) | esm3, saprot, blosum, esmc, prot_t5, igt5, onehot, prot_t5_cdr, esm2_early |
| General (21) | all reps with ρ > 0.48 |
| Augmentation (1) | mutation_features (+0.012 on esm3) |

**flab_pk_jain_heprt (n=130, best ρ=0.783, cutoff=0.47):**
| Tier | Reps |
|------|------|
| Top Tier (6) | igt5, mutation_features, esm2_concat, ablang2, prot_t5, prot_t5_cdr |
| General (7) | all reps with ρ > 0.47 |
| Augmentation (2) | paratope, structural (+0.019 on esm2_concat) |

**flab_pk_jain_fcrnrt (n=132, best ρ=0.595, cutoff=0.36):**
| Tier | Reps |
|------|------|
| Top Tier (2) | ablang2, igt5 |
| General (11) | all reps with ρ > 0.36 |
| Augmentation (1) | aac (+0.011 on ablang2) |

**flab_polyreact_jain_cic (n=137, best ρ=0.586, cutoff=0.35):**
| Tier | Reps |
|------|------|
| Top Tier (3) | prot_t5, prot_t5_cdr, esm2_early |
| General (12) | all reps with ρ > 0.35 |
| Augmentation | none |

**flab_aggregation_jain_hicrt (n=137, best ρ=0.506, cutoff=0.30):**
| Tier | Reps |
|------|------|
| Top Tier (3) | prot_t5, prot_t5_cdr, esm2_early |
| General (14) | all reps with ρ > 0.30 |
| Augmentation (2) | paratope/structural (+0.032), mutation_features (+0.025) |

**flab_thermo_jain (n=137, best ρ=0.457, cutoff=0.27):**
| Tier | Reps |
|------|------|
| Top Tier (4) | ablang2, abroberta, igt5, esm3 |
| General (16) | all reps with ρ > 0.27 |
| Augmentation (1) | aac (+0.032 on ablang2) |

**abcov_kd (n=141, best ρ=0.438, cutoff=0.26):**
| Tier | Reps |
|------|------|
| Top Tier (8) | esm2_entropy_top20, prot_t5, blosum, antiberty, esm2_midlayers, abroberta, esm2, ablang2 |
| General (28) | all reps with ρ > 0.26 |
| Augmentation | none |

**flab_aggregation_jain (n=137, best ρ=0.425, cutoff=0.26):**
| Tier | Reps |
|------|------|
| Top Tier (3) | esm2_concat, ablang2, igt5 |
| General (25) | all reps with ρ > 0.26 |
| Augmentation (1) | paratope/structural (+0.008 on igt5) |

**flab_immunogenicity_marks (n=217, best ρ=0.399, cutoff=0.24):**
| Tier | Reps |
|------|------|
| Top Tier (8) | antiberty_cdr, antiberty, esm2_entropy_top20, igt5, prot_t5, prot_t5_cdr, abroberta_cdr, abroberta |
| General (29) | all reps with ρ > 0.24 |
| Augmentation | none |

**flab_polyreact_jain_bvp (n=137, best ρ=0.374, cutoff=0.22):**
| Tier | Reps |
|------|------|
| Top Tier (4) | ablang2, igt5, antiberty, antiberty_cdr |
| General (23) | all reps with ρ > 0.22 |
| Augmentation | none |

**ddg_combined (n=429, best ρ=0.293, cutoff=0.18):**
| Tier | Reps |
|------|------|
| Top Tier (3) | esm2_wt_delta, prot_t5, esm2_midlayers |
| General (19) | all reps with ρ > 0.18 |
| Augmentation (1) | aac (+0.026 on esm2_wt_delta) |

### AbAgym Domain-Level Decisions

DMS datasets — mutation_features is the standout rep (top 1-2 on lysozyme, HER2, VEGF). Rankings are antigen-specific. abagym_all (mixed antigens) dropped — signal dilutes when antigens are combined.

**abagym_Ang2 (n=397, best ρ=0.472):**
| Tier | Reps |
|------|------|
| Top Tier (4) | saprot, esm3, esm2_midlayers, esm2_layer24 |
| General (24) | all reps with ρ > 0.28 (through esm2_concat) |
| Augmentation | none |

**abagym_EGFR (n=659, best ρ=0.308):**
| Tier | Reps |
|------|------|
| Top Tier (5) | antiberty, antiberty_cdr, esm2_layer24, esm2_early, esm2_midlayers |
| General (26) | all reps with ρ > 0.18 (through aac) |
| Augmentation | none |

**abagym_HER2 (n=146, best ρ=0.369):**
| Tier | Reps |
|------|------|
| Top Tier (3) | mutation_features, prot_t5_cdr, prot_t5 |
| General (13) | all reps with ρ > 0.24 (through ablang2) |
| Augmentation | none |

**abagym_lysozyme (n=627, best ρ=0.490):**
| Tier | Reps |
|------|------|
| Top Tier (1) | mutation_features |
| General (3) | mutation_features, aac, blosum |
| Augmentation (2) | structural (+0.007), antiberty_llr (+0.009) |

**abagym_VEGF (n=418, best ρ=0.613):**
| Tier | Reps |
|------|------|
| Top Tier (4) | mutation_features, aac, prot_t5_cdr, prot_t5 |
| General (24) | all reps with ρ > 0.39 (through esm2_entropy_top20) |
| Augmentation | none |

### Antibody CLF Domain-Level Decisions

Classification tasks (AUROC). Different character from regression — sequence encodings (blosum, onehot) do well.

**tz_her2_binary (n=5,000, AUROC):**
| Tier | Reps |
|------|------|
| Top Tier (2) | blosum, onehot |
| General (29) | all reps with AUROC > 0.80 (through aac) |
| Augmentation | none |

**humanness_therapeutic (n=564, AUROC):**
| Tier | Reps |
|------|------|
| Top Tier (5) | antiberty, igt5, abroberta, ablang2, blosum |
| General (10) | all reps with AUROC > 0.83 (through aac) |
| Augmentation | none |

### Ginkgo Domain-Level Decisions

Developability properties (regression, Spearman ρ, pre-computed hierarchical cluster splits). paratope/structural shows small but consistent augmentation signal across datasets.

**Domain-wide augmentation:** paratope/structural (+0.004 to +0.007 on multiple datasets)

**ginkgo_acsins (n=242, best ρ=0.560):**
| Tier | Reps |
|------|------|
| Top Tier (2) | igt5, ablang2 |
| General (26) | all reps with ρ > 0.24 (through esm2_entropy_top20) |
| Augmentation | paratope/structural |

**ginkgo_colloidal (n=242, best ρ=0.417):**
| Tier | Reps |
|------|------|
| Top Tier (2) | prot_t5, prot_t5_cdr |
| General (23) | all reps with ρ > 0.20 (through blosum) |
| Augmentation | paratope/structural |

**ginkgo_heparin (n=94, best ρ=0.739):**
| Tier | Reps |
|------|------|
| Top Tier (4) | esm2_concat, prot_t5, prot_t5_cdr, ablang2 |
| General (29) | all reps with ρ > 0.36 (through blosum) |
| Augmentation | paratope/structural |

**ginkgo_hic (n=242, best ρ=0.604):**
| Tier | Reps |
|------|------|
| Top Tier (3) | esm2_concat, prot_t5, prot_t5_cdr |
| General (23) | all reps with ρ > 0.35 (through esm2_layer24) |
| Augmentation | paratope/structural |

**ginkgo_pr_cho (n=197, best ρ=0.506):**
| Tier | Reps |
|------|------|
| Top Tier (2) | prot_t5, prot_t5_cdr |
| General (23) | all reps with ρ > 0.22 (through abroberta) |
| Augmentation | paratope/structural |

**ginkgo_titer (n=239, best ρ=0.399):**
| Tier | Reps |
|------|------|
| Top Tier (2) | saprot, saprot_cdr |
| General (23) | all reps with ρ > 0.19 (through structural) |
| Augmentation | paratope/structural |

---

## In Progress

---

## TODO — Molecular (8 datasets)

| # | Dataset | Samples | Reps | Augs | Supplementary Data | Status |
|---|---------|---------|------|------|--------------------|--------|
| 3 | logd | 5,028 | 24 | 6 | FT (8 files), graph survey, Imrie TL, FT comparison | DONE |
| 4 | qm9 | 9,978 | 22 | 14 | FT N/A, graph survey, qm9_3d_comparison (SchNet/DimeNet++) | DONE |
| 5 | adme_hlm | 3,086 | 24 | 7 | No FT/comparison data | DONE |
| 6 | adme_rlm | 3,053 | 24 | 6 | No FT/comparison data | DONE |
| 7 | adme_mdr1 | 2,641 | 22 | 14 | No FT/comparison data | DONE |
| 8 | adme_solubility | 2,172 | 22 | 14 | No FT/comparison data | DROPPED — R²=0.32 ceiling |

**Molecular supplementary data (applies across multiple datasets):**
- `finetune_survey/`: 8 FT strategies × {logd, caco2, herg} — LoRA, LoRA r2/r4, freeze, contrastive, GNN, GROVER
- `comparisons/molecular_finetune_comparison.json`: frozen vs FT for 25 conditions × {logd, caco2, herg}
- `comparisons/imrie_transfer_learning.json`: ChEMBL DNN pretraining × {logd, caco2, herg}
- `molecular_graph_survey/`: GNN augmentation for {herg, logd, qm9}
- `comparisons/qm9_3d_comparison/`: SchNet, DimeNet++ E2E vs frozen reps (qm9 only)

---

## TODO — Antibody Inter-Family (14 datasets)

Old Section 10G has a bulk summary with 11 datasets. Each needs individual cutoff + augmentation decisions.

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 9 | flab_thermo_tresanco | 672 | 34 | 4 | DONE |
| 10 | flab_pk_jain_heprt | 130 | 37 | 4 | DONE |
| 11 | flab_pk_jain_fcrnrt | 132 | 37 | 4 | DONE |
| 12 | flab_polyreact_jain_cic | 137 | 37 | 4 | DONE |
| 13 | flab_aggregation_jain_hicrt | 137 | 37 | 4 | DONE |
| 14 | flab_thermo_jain | 137 | 37 | 4 | DONE |
| 15 | abcov_kd | 141 | 33 | 3 | DONE |
| 16 | flab_aggregation_jain | 137 | 37 | 4 | DONE |
| 17 | flab_immunogenicity_marks | 217 | 37 | 4 | DONE |
| 18 | flab_polyreact_jain_bvp | 137 | 37 | 4 | DONE |
| 19 | flab_thermo_hutchinson | 192 | 38 | 5 | DROPPED — best ρ=0.289, weak ceiling |
| 20 | flab_aggregation_jain_sas | 137 | 37 | 4 | DROPPED — best ρ=0.230, only 8 viable reps |
| 21 | ddg_combined | 429 | 34 | 4 | DONE — also has OPIG DDG data (opig/ddg/, may differ from phase5) |
| 22 | ddg_combined_full | 1,536 | 30 | 8 | DROPPED — best ρ=0.231, only 1 viable rep |

**Antibody supplementary data:**
- `finetune_survey/ft_ab_*.json`: 4 FT strategies × {abagym_Ang2, flab_binding_koenig, flab_expression_koenig}
- `comparisons/antibody_finetune_comparison.json`: frozen vs FT, 6 datasets
- `comparisons/antibody_region_ablation.json`: region ablation, 5 datasets
- `comparisons/antibody_region_entropy.json`: entropy analysis, 5 datasets
- `forward_compat/exp1_mint/`: MINT on ddg_combined, flab_binding_koenig, flab_expression_koenig, flab_thermo_tresanco
- `forward_compat/exp2_saprot/`: SaProt on same 4 datasets
- `forward_compat/exp4_ablm/`: AbLM reps on 5 datasets
- `forward_compat/exp6_xtrimopglm/`: xTrimoPGLM on 3 datasets
- `forward_compat/exp9_aido_protein/`: AIDO-Protein on 3 datasets
- `antibody_survey/`: 3 ESM2 region reps × 24 datasets
- `dms_baselines/`: 9 DMS datasets × 20 specialized reps (varpos, LLR scores, etc.)
- `geometry_survey/`: 3D geometry reps on 15 antibody datasets (may be empty)

---

## TODO — Antibody DMS / Large DMS (datasets with ≥1000 samples or DMS-specific reps)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 23 | flab_binding_koenig | 4,275 | 23 | 0 | BLOCKED — 23/42 reps, jobs 7331342-46 pending |
| 24 | flab_expression_koenig | 4,275 | 23 | 0 | BLOCKED — 23/42 reps, jobs 7331342-46 pending |
| 25 | flab_binding_adams | 5,000 | 4 | 2 | BLOCKED — only 4 reps, 38 pending |
| 26 | flab_expression_adams | 5,000 | 4 | 2 | BLOCKED — only 4 reps, 38 pending |
| 27 | flab_binding_phillips_h3 | 5,000 | 4 | 2 | BLOCKED — only 4 reps, 38 pending |

**Note:** koenig datasets have 23 reps (enough to start). Adams/Phillips have only 4 — defer until server jobs complete.

---

## TODO — AbAgym (6 datasets)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 28 | abagym_all | 2,247 | 35 | 5 | DROPPED — best ρ=0.145, below viability |
| 29 | abagym_Ang2 | 397 | 36 | 5 | DONE |
| 30 | abagym_EGFR | 659 | 36 | 4 | DONE |
| 31 | abagym_HER2 | 146 | 36 | 4 | DONE |
| 32 | abagym_lysozyme | 627 | 36 | 3 | DONE |
| 33 | abagym_VEGF | 418 | 36 | 4 | DONE |

**Supplementary:** FT comparison (abagym_Ang2), DMS baselines (all 5 individual + all), `pp_survey/` (physicochemical hybrids, 5 individual + koenig×2), `tuned_survey/` (tuned hybrids, same 7 datasets)

---

## TODO — Antibody CLF (3 datasets)

| # | Dataset | Samples | Reps | Augs | Metric | Status |
|---|---------|---------|------|------|--------|--------|
| 34 | tz_her2_binary | 5,000 | 37 | 5 | AUROC/F1w | DONE |
| 35 | humanness_therapeutic | 564 | 16 | 4 | AUROC/F1w | DONE |
| 36 | tz_her2_multiclass | 5,000 | 1 | 0 | F1w | BLOCKED — only 1 rep |
| 37 | humanness_species | — | 0 | 0 | AUROC/F1w | BLOCKED — no file, just submitted |

**Note:** tz_her2_multiclass has only structural rep. humanness_species has no file yet (just submitted 2026-03-24).

---

## TODO — Ginkgo Developability (9 datasets)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 38 | ginkgo_acsins | 242 | 34 | 2-3 | DONE |
| 39 | ginkgo_aggregation | 242 | 34 | 2-3 | DROPPED — best ρ=0.181 |
| 40 | ginkgo_colloidal | 242 | 34 | 2-3 | DONE |
| 41 | ginkgo_heparin | 94 | 34 | 2-3 | DONE |
| 42 | ginkgo_hic | 242 | 34 | 2-3 | DONE |
| 43 | ginkgo_pr_cho | 197 | 34 | 2-3 | DONE |
| 44 | ginkgo_purity | 242 | 34 | 2-3 | DROPPED — best ρ=0.220 |
| 45 | ginkgo_titer | 239 | 34 | 2-3 | DONE |
| 46 | ginkgo_tm2 | 193 | 34 | 2-3 | DROPPED — best ρ=0.166 |

---

## TODO — TCGA (3 datasets)

Old Section 10D/10E have full writeups with tiers. Need to reconcile with new format.

| # | Dataset | Samples | Reps | Augs | Metric | Status |
|---|---------|---------|------|------|--------|--------|
| 47 | tcga_luad_vs_lusc | 1,017 | 36 | 20 | AUROC | DONE |
| 48 | tcga_brca_pam50 | 821 | 36 | 20 | F1w | DONE |
| 49 | tcga_brca_survival | 151 | 36 | 20 | R² | DROPPED (10F) — all R² negative |

**TCGA supplementary data:**
- `finetune_survey/ft_tcga_*.json`: 3 strategies (base, geneformer, scfoundation) × 2 datasets
- `comparisons/tcga_foundation_comparison/`: frozen vs FT summary + per-dataset
- `comparisons/wsi_abmil_comparison/`: 6 WSI models × 3 pooling strategies × 2 datasets
- `tcga_new_reps/`: 19 factory reps (earlier run, may be stale)

---

## TODO — Reaction (3 datasets)

Old Sections 10B/10C/12D/12E have writeups. Need new-format review.

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 50 | buchwald_hartwig | 3,955 | 38 | 6 | TODO — expanded reps wired (14 original + 24 molecular variants) |
| 51 | suzuki | 5,760 | 38 | 4 | TODO — expanded reps wired (14 original + 24 molecular variants) |
| 52 | bh_doyle_plates | 4,312 | 38 | 6 | TODO — expanded reps wired (14 original + 24 molecular variants) |

**Reaction supplementary data:**
- `comparisons/reaction_rep_comparison.json`: rep comparison across all 3 datasets
- `forward_compat/exp7_reactiont5/`: 14 reps + reactiont5 on all 3 datasets

---

## TODO — DTA (2-3 datasets)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 53 | davis_frozen | — | 20 | — | TODO — old dta_survey/ data only |
| 54 | kiba_frozen | — | 20 | — | TODO — old dta_survey/ data |
| 55 | kiba_ft | — | 20 | — | TODO — old dta_survey/ data |

**DTA supplementary data:**
- `dta_survey/`: 20 reps per dataset (older survey)
- `comparisons/dta_e2e_comparison/`: E2E DTA models (davis, kiba)
- `comparisons/dta_finetune_comparison/`: DTA FT comparison
- `structural_protein_survey/kiba.json`: cold-target with structural protein reps (6 reps)
- Phase 5 DTA: PENDING (jobs in queue, no results yet)

---

## TODO — Nucleotide (19 datasets)

| # | Dataset | Samples | Reps | Metric | Status |
|---|---------|---------|------|--------|--------|
| 56 | gb_human_nontata_promoters | 36,131 | 3 | MCC | TODO |
| 57 | nt_enhancers | — | 3-6 | MCC | TODO |
| 58 | nt_enhancers_types | — | 3-6 | MCC | TODO |
| 59 | nt_H2AFZ | — | 3-6 | MCC | TODO |
| 60 | nt_H3K27ac | — | 3-6 | MCC | TODO |
| 61 | nt_H3K27me3 | — | 3-6 | MCC | TODO |
| 62 | nt_H3K36me3 | — | 3-6 | MCC | TODO |
| 63 | nt_H3K4me1 | — | 3-6 | MCC | TODO |
| 64 | nt_H3K4me2 | — | 3-6 | MCC | TODO |
| 65 | nt_H3K4me3 | — | 3-6 | MCC | TODO |
| 66 | nt_H3K9ac | — | 3-6 | MCC | TODO |
| 67 | nt_H3K9me3 | — | 3-6 | MCC | TODO |
| 68 | nt_H4K20me1 | — | 3-6 | MCC | TODO |
| 69 | nt_promoter_all | — | 6 | MCC | TODO |
| 70 | nt_promoter_no_tata | — | 6 | MCC | TODO |
| 71 | nt_promoter_tata | — | 6 | MCC | TODO |
| 72 | nt_splice_sites_acceptors | — | 3 | MCC | TODO |
| 73 | nt_splice_sites_all | — | 3 | MCC | TODO |
| 74 | nt_splice_sites_donors | — | 3 | MCC | TODO |

**Note:** Only classical reps (kmer4, kmer6, tetranuc_freq, etc.). No LM reps. Very few reps per dataset.

**BLOCKED: DNA foundation model reps needed before pipeline readiness can be assessed.** Current reps are all nucleotide composition variants (kmer4 ≈ kmer6 ≈ tetranuc_freq) — no information diversity to hybridize. Need:
- DNABERT-2 (was requested but never implemented)
- Nucleotide Transformer (InstaDeep)
- Hyena-DNA, Caduceus, or other genomic FMs from the genomics domain plan

Without LM reps, this domain has no meaningful tier structure — the top 3 reps encode the same information at different resolutions. Pipeline readiness decisions deferred until FM reps are added.

**Preliminary decisions (apply when FM reps arrive):**
- DROP nt_splice_sites_all (MCC=0.245) and nt_H3K9me3 (MCC=0.239) — near random
- kmer4, kmer6, tetranuc_freq will be general-tier classical baselines alongside whatever FM reps are added
- dinuc_freq, gc_content are lower-tier classical; complexity is augmentation-only

---

## TODO — OPIG DDG (beyond what's in antibody inter-family)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| — | ddg_combined | 429 | 34 | 4 | Listed above (#21) in antibody inter-family |
| — | ddg_combined_full | 1,536 | 30 | 8 | Listed above (#22) |
| 75 | graphinity_foldx | — | 0 | 0 | BLOCKED — no results, jobs pending |
| 76 | graphinity_rosetta | — | 0 | 0 | BLOCKED — no results, jobs pending |
| 77 | graphinity_augmented | — | 0 | 0 | BLOCKED — no results, jobs pending |

---

## Summary Counts

| Domain | Total Datasets | DONE | IN PROGRESS | TODO | BLOCKED | DROPPED |
|--------|---------------|------|-------------|------|---------|---------|
| Molecular | 8 | 7 | 0 | 0 | 0 | 1 (adme_solubility) |
| Ab Inter-Family | 14 | 11 | 0 | 0 | 0 | 3 (hutchinson, sas, ddg_combined_full) |
| Ab DMS/Large | 5 | 0 | 0 | 0 | 5 (all waiting for jobs 7331342-46) | 0 |
| AbAgym | 6 | 5 | 0 | 0 | 0 | 1 (abagym_all) |
| Ab CLF | 4 | 2 | 0 | 0 | 2 (multiclass, species) | 0 |
| Ginkgo | 9 | 6 | 0 | 0 | 0 | 3 (aggregation, purity, tm2) |
| TCGA | 3 | 2 | 0 | 0 | 0 | 1 (survival) |
| Reaction | 3 | 0 | 0 | 3 (reps wired, need survey run) | 0 | 0 |
| DTA | 3 | 0 | 0 | 3 (partial data) | 0 | 0 |
| Nucleotide | 19 | 0 | 0 | 0 | 17 (need DNA FM reps) | 2 (splice_all, H3K9me3) |
| Graphinity DDG | 3 | 0 | 0 | 0 | 3 | 0 |
| **TOTAL** | **77** | **33** | **0** | **0** | **32** | **12** |

---

## Existing BOM Sections to Reconcile

These sections have older-format writeups that were done before this dataset-by-dataset review process. They need to be checked against new data and converted to the new format (base reps + augmentation signal rows):

- **10A. LogD** — has full tier table, needs new-format row
- **10B. Buchwald-Hartwig** — has tiers, needs new-format row
- **10C. Suzuki** — special augmentation experiment, needs new-format row
- **10D. LUAD vs LUSC** — has full table, needs new-format row
- **10E. BRCA PAM50** — has full table, needs new-format row
- **10G. Antibody Inter-Family** — bulk summary of 11 datasets, each needs individual row
- **12A-12F** — locked pipelines with tiers, these inform but don't replace the readiness rows

---

## Action Items (accumulated during pipeline readiness review)

### New experiments needed

1. **E2E SchNet on more molecular datasets** — currently qm9 only (R²=0.95). Run on logd, herg, caco2 to determine if it generalizes beyond quantum properties. If yes, domain-wide top tier.
2. **Augmentation per-dataset assignment (usrcat, coulomb_matrix, usr)** — these are dataset-specific, not domain-wide. Need to test which ones help on which datasets. Small test: run these 3 as augments on the top base rep for each molecular dataset that doesn't already have augmentation data for them.
3. **DNA foundation model reps** — DNABERT-2 was requested but never implemented. Blocks all 17 nucleotide datasets. Also consider: Nucleotide Transformer (InstaDeep), Hyena-DNA, Caduceus. Without these, nucleotide domain has no information diversity.

### Implementation gaps

4. **DNABERT-2 rep implementation** — add to `src/kirby/representations/` and wire into nucleotide survey dispatch. This fell through the cracks during vibe coding.
5. ~~**Expand reaction rep set**~~ **DONE** — mordred, mhg_gnn, grover, mol2vec, chemberta, molformer, avalon, pharmacophore3d × {reactant_, product_, diff_} = 24 new reps wired into `_load_reaction()` via generic `create_reaction_rep_variant()` factory. Binary reps (avalon) use OR-combine for multi-reactant aggregation; continuous reps use mean-pool. Total reaction reps: 14 original + 24 = 38.

### Waiting for server results (rsync when complete)

6. **DTA** (jobs 7331383-87) — davis, kiba, bindingdb_kd. 5 jobs pending. Full DTA rep set already wired.
7. **Graphinity DDG** (jobs 7331380-82) — foldx, rosetta, augmented. 3 jobs pending. Zero results currently.
8. **Ab DMS fill** (jobs 7331342-46) — adams (binding+expression), phillips_h3, koenig (binding+expression). Filling from 4-23 reps to 42.
9. **Ab CLF fill** (jobs 7343797-98) — tz_her2_multiclass (1→42 reps), humanness_species (just submitted).

### Decisions already locked

- **Molecular FT:** lora_molformer and contrastive_molformer → general base reps for all molecular datasets. All other FT variants excluded.
- **Molecular augmentation:** pharmacophore3d and maccs → domain-wide augmentation for all molecular datasets.
- **Molecular drops:** adme_solubility (R²=0.32 ceiling), unimol_v2 (dead everywhere).
- **Nucleotide drops:** nt_splice_sites_all (MCC=0.245), nt_H3K9me3 (MCC=0.239).
