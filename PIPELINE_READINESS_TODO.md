# Pipeline Readiness TODO

**Purpose:** Single operational document for the KIRBy experiment pipeline. Tracks dataset readiness, methodology status, blockers, and the path from preliminary runs to publication-quality results.

**Last updated:** 2026-03-28

---

## Workflow Vision

The pipeline follows a staged approach:

1. **Preliminary runs:** Once F12/F13 results land and methodology is finalized, immediately run complexity/similarity/noise experiments on the 33 ready datasets. These are screening-level — they use the finalized allocation/evaluator defaults but are not the final paper results.

2. **Incremental onboarding:** As new datasets get prepared (surveys completed, tier decisions made, DATASET_REPS populated), run them through the relevant experiments when they're ready. Complexity, similarity, and noise apply to all datasets. OPIG and federated experiments are specific to certain datasets only.

3. **Refinement:** Modify experiments based on preliminary results. This is expected — experiments start in a preliminary running state, not a final one. Iterate on what's interesting.

4. **Final runs:** When the picture is clear — which datasets tell the story, which experiments produce interesting results — trigger the selected experiments in full search mode on the selected datasets for publication-quality results.

**The goal is seamless flow:** F12/F13 finalize → run experiments on ready datasets → onboard new datasets as their surveys land → refine → select → publish.

---

## Current Blockers

### F12/F13 (framework experiments) — BLOCKS EVERYTHING

F12 tests 31 allocation strategies × 2 importance × 3 evaluators × 4 budgets × 4 models = 2,976 configs/dataset across 6 datasets. F13 tests evaluator × model matching (600 configs/dataset). Results finalize SPEED_TIERS and allocation defaults. Currently running on ARC (`f12-speedup` branch). See `FRAMEWORK_EXPERIMENT_ROADMAP.md`.

Until F12/F13 complete:
- 26 experiments hardcode `allocation_method='greedy'` (the worst per F01). Full list in `EXPERIMENT_BUGS.md`.
- SPEED_TIERS in `find_best_hybrid.py` use stale defaults.
- All current experiment results are screening-level, not final.

After F12/F13:
1. Update allocation defaults in all 26 experiments
2. Update internal evaluator
3. Clear all caches on server (`~/.cache/kirby/hybrid_search/`, `~/.cache/kirby/create_hybrid_cache/`)
4. Pull code on server (`git pull`)
5. Begin preliminary runs on 33 ready datasets

### FT experiments — stuck in HTC GPU queue (likely 1+ week)

- Molecular FT fill (4 datasets: qm9, adme_hlm, adme_rlm, adme_mdr1): `submit_molecular_ft_fill.sh`
- Antibody ESM-2 LoRA fill (29 datasets): `submit_antibody_ft_fill.sh`
- Per-dataset FT swap decisions blocked until these land.
- Don't wait on these for preliminary experiment runs — FT is an enhancement, not a requirement.

### New dataset surveys — some likely timed out

Need to check server status and re-submit failures. These block pipeline readiness for their respective datasets:
- Reaction: buchwald_hartwig, suzuki, bh_doyle_plates
- DTA: davis, kiba, bindingdb_kd (DTA has been "super buggy")
- Nucleotide: 19 datasets — survey results on server (nucleotide_survey/, Mar 26-28), need pull + review. Some files ~7KB (possibly partial FM reps)
- Graphinity DDG: foldx, rosetta, augmented
- Ab DMS fill: adams (binding+expression), phillips_h3, koenig (binding+expression)
- Ab CLF fill: tz_her2_multiclass, humanness_species

### Cache warning

Complexity and similarity experiments that ran previously used stale hybrid caches (pre-augmentation defaults: step_size=10, patience=3, augmentation_strategy='greedy_ablation'). Must clear caches after F12/F13 finalization before any new runs.

---

## Methodology Status

| Component | Status | Blocked on |
|-----------|--------|-----------|
| Allocation strategy | Screening defaults; 26 experiments use greedy (worst) | F12 |
| Internal evaluator | RF hardcoded; F07 showed Ridge wins on antibody | F12/F13 |
| SPEED_TIERS | Stale | F12/F13 |
| Augmentation defaults | **FINALIZED:** uncapped, step_size=5, patience=20 | — |
| FT swap rules | **PARTIAL:** molecular 3/7 done, TCGA done, antibody pending | HTC queue |
| Budget grid | **FINALIZED:** [10,25,50,100,200,500,1000,2000] screening | — |
| Metrics | **FINALIZED:** domain-aware (RMSE/CI/AUROC/F1w/ρ/MCC) | — |
| Classification support | **FINALIZED:** C02/C08/C09/C10/C11/N01/N07 fixed | — |
| Augmentation passing | **FINALIZED:** C11/C16-C19, all similarity, all noise fixed | — |

**FT decisions already locked (from `EXPERIMENT_DESIGN_DISCUSSION.md` §9):**
- Molecular: lora_molformer, contrastive_molformer, graphmvp_ft → general base reps. Per-dataset swap only where FT beats frozen. herg keeps frozen molformer. All chemberta/selformer/freeze_head FT excluded.
- TCGA: scfoundation_ft is FT-only (frozen dead). geneformer_ft general tier (frozen v2 fallback).
- Antibody/DTA FT: pending HTC results.
- Excluded everywhere: antiberty_ft (broken), all DTA protein LoRA (broken), chemberta/selformer FT (worse on all datasets).

---

## Experiment Categories

5 categories. Complexity, similarity, and noise apply to **all** ready datasets. OPIG and federated are dataset-specific.

| Category | Experiments | Datasets | Helpers | Roadmap |
|----------|------------|----------|---------|---------|
| Complexity | C01-C21 | All 33 ready | `complexity_helpers.py` | `EXPERIMENT_REFACTOR_ROADMAP.md` |
| Similarity | S01-S13 | All 33 ready | `similarity_helpers.py` | `EXPERIMENT_REFACTOR_ROADMAP.md` |
| Noise | N01-N13 all active | All 33 ready | `noise_helpers.py` | `NOISE_EXPERIMENT_ROADMAP.md` |
| OPIG | 6 standalone scripts | Specific OPIG datasets | — | `OPIG_COLLAB_ROADMAP.md` |
| Federated | fed01-fed03 active, fed04/fed06 planned | herg, caco2, logd | `federated_helpers.py` | `FEDERATED_EXPERIMENT_ROADMAP.md` |

**Noise (N09-N13):** All implemented. N09 (LNL survey, 12 methods), N10 (hybrid neighborhood LNL), N11 (estimation × remediation grid), N12 (importance-weighted distance), N13 (alt noise distributions). New utility: `src/kirby/utils/lnl_methods.py` (12 training-loop LNL methods). See `NOISE_EXPERIMENT_ROADMAP.md`.

**Federated (fed01-fed06):** Fed01 (noise heterogeneity), fed02 (rep diversity silos), fed03 (synthetic private datasets) are implemented. Fed04 (cluster-aware importance) and fed06 (antibody extension) still planned. Datasets: herg, caco2, logd. Also has `lhasa_herg_loader.py`. See `FEDERATED_EXPERIMENT_ROADMAP.md`.

**Impact on per-dataset files:** None — the unified runner handles dispatch. Adding a new experiment to any category just requires registering it in the helpers file. Adding a new category requires 1 helpers file + 1 entry in `run_dataset.py`'s `CATEGORIES` dict.

---

## Per-Dataset Experiment Runner — DONE (2026-03-27)

Replaced ~100 per-dataset boilerplate .py files with a single unified runner:

```bash
python tests/experiments/run_dataset.py complexity herg
python tests/experiments/run_dataset.py noise herg --experiments n01 n02
python tests/experiments/run_dataset.py similarity logd --output /custom/path
python tests/experiments/run_dataset.py federated herg
```

**SLURM:** Unified into 2 scripts (was 6):
```bash
sbatch tests/slurm_scripts/submit_dataset.sh complexity herg
sbatch tests/slurm_scripts/submit_dataset.sh noise herg "n01 n02"
bash tests/slurm_scripts/submit_all_datasets.sh complexity              # all 33
bash tests/slurm_scripts/submit_all_datasets.sh noise --domain molecular # 7
bash tests/slurm_scripts/submit_all_datasets.sh federated               # herg only
```

Override SLURM resources at sbatch time: `sbatch --time=96:00:00 --mem=128G submit_dataset.sh federated herg`

**To add a new dataset:** Add to `DATASET_REPS` in `complexity_helpers.py` + add to domain list in `submit_all_datasets.sh`. 0 new files.

**To add a new experiment:** Create the .py file + register in the category's helpers `module_map`/`name_map` + uncomment in `SINGLE_DATASET_EXPERIMENTS`. 0 other changes.

**To add a new category:** Create `{category}_helpers.py` with `run_all()` + `SINGLE_DATASET_EXPERIMENTS` + add 1 entry to `CATEGORIES` in `run_dataset.py` + add prefix in `submit_all_datasets.sh`.

---

## New Dataset Pipeline

When a dataset becomes ready (survey results land from server), follow this checklist:

1. **Pull results:** rsync survey JSON from server to `tests/results/`
2. **Review data:** Show ALL standalone results sorted by primary metric, all augmentation tests, all supplementary data (FT, forward compat, graph survey, comparisons)
3. **Decide tiers:** User decides top tier, general, augmentation rep cutoffs and viability
4. **Update DATASET_REPS:** Add entry in `complexity_helpers.py` with `general` + `augmentation` + `top_tier` rep lists
5. **Add to submit lists:** Add dataset name to the appropriate domain list in `submit_all_datasets.sh`
6. **Submit preliminary runs:** `sbatch submit_dataset.sh complexity {dataset_name}` (repeat for similarity, noise)
7. **Record here:** Update this document with tier tables and mark DONE

For OPIG-specific datasets: also check `OPIG_COLLAB_ROADMAP.md` for relevant standalone experiments.
For federated-eligible datasets: also add to federated experiment list per `FEDERATED_EXPERIMENT_ROADMAP.md`.

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
| General (15) | graph_kernel, mol2vec, mhg_gnn, ecfp4, pdv, grover, avalon, mordred, top_torsion, smited, atom_pair, molformer, lora_molformer**, contrastive_molformer**, graphmvp_ft** |
| Augmentation (5) | maccs, pharmacophore3d, usrcat, coulomb_matrix, usr |

**herg (1,412 samples):**
| Tier | Reps |
|------|------|
| Top Tier (8) | mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, mol2vec, ecfp4 |
| General (16) | mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, mol2vec, ecfp4, molformer, chemberta, atom_pair, chembert, smited, top_torsion, graphmvp, graphmvp_ft** |
| Augmentation (5) | maccs, pharmacophore3d, coulomb_matrix, usr, usrcat |

*Note: lora_molformer and contrastive_molformer excluded from herg — both worse than frozen molformer on this dataset (-0.051 and -0.005 R² respectively). Frozen molformer stays.*

**logd (5,028 samples):**
| Tier | Reps |
|------|------|
| Top Tier (8) | mhg_gnn, mordred, grover, mol2vec, pdv, ecfp4, smited, molformer |
| General (17) | mhg_gnn, mordred, grover, mol2vec, pdv, ecfp4, smited, molformer, avalon, atom_pair, chemberta, top_torsion, chembert, graph_kernel, lora_molformer**, contrastive_molformer**, graphmvp_ft** |
| Augmentation (6) | selformer, rdkfp, maccs, pharmacophore3d, molclr, usrcat |

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
| Augmentation (3) | pharmacophore3d, maccs, coulomb_matrix |

### Molecular Domain-Level Decisions

**FT reps (apply to all molecular datasets):**
- lora_molformer** and contrastive_molformer** are added as general-tier base reps across all molecular datasets. They don't compete with the top tier but consistently beat maccs and provide FT-adapted LM signal. Frozen molformer is the fallback when FT is disabled.
- graphmvp_ft** added as general-tier base rep across all molecular datasets. Beats frozen graphmvp on all 3 tested datasets (logd +0.096, caco2 +0.025, herg +0.054 R²). Frozen graphmvp is the fallback when FT is disabled.
- All other FT variants (chemberta FT, selformer FT, freeze_head anything) are excluded — they never beat ecfp4.

**Per-dataset FT swap rules (verified from molecular_finetune_comparison.json):**
- lora_molformer replaces molformer on: logd (+0.020 R²), caco2 (+0.052 R²). Frozen kept on: herg (-0.051 R²).
- contrastive_molformer replaces molformer on: logd (+0.014 R²), caco2 (+0.014 R²). Frozen kept on: herg (-0.005 R², negligible).
- graphmvp_ft replaces graphmvp on: logd (+0.096 R²), caco2 (+0.025 R²), herg (+0.054 R²).
- **qm9, adme_hlm, adme_rlm, adme_mdr1: FT results PENDING** (submit_molecular_ft_fill.sh).

**Augmentation (apply to all molecular datasets):**
- pharmacophore3d and maccs are domain-wide augmentation reps for all molecular datasets. pharmacophore3d is the most consistent augment rep tested (13/15 improve in graph survey, positive on 5/6 phase5 datasets). maccs shows small but reliable signal everywhere (6/7 datasets).
- usrcat, coulomb_matrix, usr: now added to all molecular datasets (caco2 and adme_mdr1 were missing, fixed 2026-03-28).

**ACTION items:**
- Run E2E SchNet on logd, herg, caco2 to determine if it generalizes beyond qm9. torch-cluster CUDA bug fixed (2026-03-28) — was producing NaN on all 3D methods due to CPU-only torch-cluster on GPU. Fallback radius graph now used automatically.

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
| 23 | flab_binding_koenig | 4,275 | 34 | 12 | DONE |
| 24 | flab_expression_koenig | 4,275 | 34 | 12 | DONE |
| 25 | flab_binding_adams | 5,000 | 35 | 10 | DONE |
| 26 | flab_expression_adams | 5,000 | 35 | 10 | DONE |
| 27 | flab_binding_phillips_h3 | 5,000 | 35 | 10 | DONE |

**flab_binding_koenig (n=4,275, best ρ=0.475, cutoff=0.28):**
| Tier | Reps |
|------|------|
| Top Tier (8) | ablang2, saprot, saprot_cdr, esm3, esm3_cdr, prot_t5, prot_t5_cdr, igt5 |
| General (26) | all reps with ρ > 0.28 (through antiberty_cdr) |
| Augmentation (4) | esm2_llr, mutation_features, antiberty_llr, aac |

**flab_expression_koenig (n=4,275, best ρ=0.715, cutoff=0.43):**
| Tier | Reps |
|------|------|
| Top Tier (7) | esm3, esm3_cdr, prot_t5_cdr, prot_t5, saprot, saprot_cdr, igt5 |
| General (27) | all reps with ρ > 0.43 (through aac) |
| Augmentation (3) | mutation_features, esm2_llr, aac |

**flab_binding_adams (n=5,000, best ρ=0.669, cutoff=0.50):**
| Tier | Reps |
|------|------|
| Top Tier (8) | blosum, onehot, esm3, esm3_cdr, saprot, saprot_cdr, esm2_cdr, esm2_cdr3 |
| General (26) | all reps with ρ > 0.50 (through esm2_mutsite) |
| Augmentation (2) | aac, paratope |

**flab_expression_adams (n=5,000, best ρ=0.511, cutoff=0.42):**
| Tier | Reps |
|------|------|
| Top Tier (7) | blosum, saprot, saprot_cdr, esm3, esm3_cdr, prot_t5, prot_t5_cdr |
| General (28) | all reps with ρ > 0.42 (through antiberty_cdr) |
| Augmentation (1) | esm2_llr |

**flab_binding_phillips_h3 (n=5,000, best ρ=0.722, cutoff=0.67):**
| Tier | Reps |
|------|------|
| Top Tier (4) | onehot, blosum, igt5, esm2_early |
| General (29) | all reps with ρ > 0.67 (through abroberta) |
| Augmentation (3) | paratope, aac, mutation_features |

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
| 50 | buchwald_hartwig | 3,955 | 14 | 6 | DONE |
| 51 | suzuki | 5,760 | 14 | 4 | DROPPED — R²=0.453 ceiling, only 1 viable rep |
| 52 | bh_doyle_plates | 4,312 | 35 | 24 | DONE |

**buchwald_hartwig (n=3,955, best R²=0.718, cutoff=0.50):**
| Tier | Reps |
|------|------|
| Top Tier (3) | reactant_ecfp4, diff_ecfp4, drfp_2048 |
| General (8) | all reps with R² > 0.50 (through rxnfp) |
| Augmentation | none |

**bh_doyle_plates (n=4,312, best R²=0.638, cutoff=0.50):**
| Tier | Reps |
|------|------|
| Top Tier (6) | reactant_avalon, diff_avalon, diff_mhg_gnn, diff_ecfp4, reactant_grover, reactant_ecfp4 |
| General (17) | all reps with R² > 0.50 (through diff_mordred) |
| Augmentation (6) | product_maccs, reactant_maccs, onehot_components, diff_maccs, product_pharmacophore3d, reactant_pharmacophore3d |

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
| 56 | gb_human_nontata_promoters | 36,131 | 6 | MCC | Classical only — needs resubmit with FM reps |
| 57 | nt_enhancers | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 58 | nt_enhancers_types | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 59 | nt_H2AFZ | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 60 | nt_H3K27ac | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 61 | nt_H3K27me3 | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 62 | nt_H3K36me3 | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 63 | nt_H3K4me1 | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 64 | nt_H3K4me2 | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 65 | nt_H3K4me3 | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 66 | nt_H3K9ac | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 67 | nt_H3K9me3 | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 68 | nt_H4K20me1 | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 69 | nt_promoter_all | 31,584 | 6 | MCC | Classical only — needs resubmit with FM reps |
| 70 | nt_promoter_no_tata | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 71 | nt_promoter_tata | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 72 | nt_splice_sites_acceptors | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 73 | nt_splice_sites_all | — | 6 | MCC | Classical only — needs resubmit with FM reps |
| 74 | nt_splice_sites_donors | — | 6 | MCC | Classical only — needs resubmit with FM reps |

**Root cause:** `DATASET_REGISTRY` in `run_baseline_survey.py` had default preset `'fast'` (6 classical reps only) for all nucleotide datasets. Survey was submitted without `--reps all` override, so FM reps never ran. **Fixed 2026-03-28** (commit a2a0d2d): default changed to `'all'` (26 reps: 14 FM + 11 classical + 1 factory).

**Reps implemented (26 total):** 14 foundation models + 11 classical + 1 factory. Code: `src/kirby/representations/nucleotide.py`. SLURM: `submit_nucleotide_survey.sh` (short partition, 11:59:00, GPU).

**Resubmission:** Pull on server, run `bash tests/slurm_scripts/submit_nucleotide_survey.sh`. Existing classical results will merge in automatically. Output: `tests/results/nucleotide_survey/`.

**Preliminary decisions (from classical-only results):**
- DROP nt_splice_sites_all (MCC=0.245) and nt_H3K9me3 (MCC=0.239) — near random (pending confirmation after FM reps land)
- kmer4, kmer6, tetranuc_freq will be general-tier classical baselines
- dinuc_freq, gc_content are lower-tier classical; complexity is augmentation-only

---

## TODO — OPIG DDG (beyond what's in antibody inter-family)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| — | ddg_combined | 429 | 34 | 4 | Listed above (#21) in antibody inter-family |
| — | ddg_combined_full | 1,536 | 30 | 8 | Listed above (#22) |
| 75 | graphinity_foldx | ~50K | 0 | 0 | BLOCKED — bus error (OOM), resubmitting with 370G |
| 76 | graphinity_rosetta | ~21K | 29 | 0 | DROPPED — all R² negative, best ρ=0.079 |
| 77 | graphinity_augmented | ~50K | 0 | 0 | BLOCKED — bus error (OOM) + NaN light_seq fix, resubmitting with 370G |

---

## Summary Counts

| Domain | Total Datasets | DONE | IN PROGRESS | TODO | BLOCKED | DROPPED |
|--------|---------------|------|-------------|------|---------|---------|
| Molecular | 8 | 7 | 0 | 0 | 0 | 1 (adme_solubility) |
| Ab Inter-Family | 14 | 11 | 0 | 0 | 0 | 3 (hutchinson, sas, ddg_combined_full) |
| Ab DMS/Large | 5 | 5 | 0 | 0 | 0 | 0 |
| AbAgym | 6 | 5 | 0 | 0 | 0 | 1 (abagym_all) |
| Ab CLF | 4 | 2 | 0 | 0 | 2 (multiclass, species) | 0 |
| Ginkgo | 9 | 6 | 0 | 0 | 0 | 3 (aggregation, purity, tm2) |
| TCGA | 3 | 2 | 0 | 0 | 0 | 1 (survival) |
| Reaction | 3 | 2 | 0 | 0 | 0 | 1 (suzuki) |
| DTA | 3 | 0 | 0 | 3 (partial data) | 0 | 0 |
| Nucleotide | 19 | 0 | 0 | 0 | 17 (survey on server, needs pull + review) | 2 (splice_all, H3K9me3) |
| Graphinity DDG | 3 | 0 | 0 | 0 | 2 (foldx, augmented — OOM, resubmitting) | 1 (rosetta) |
| **TOTAL** | **77** | **40** | **0** | **0** | **23** | **14** |

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

## Action Items

### IMMEDIATE — When F12/F13 results land

1. **Analyze F12/F13 results** — determine winning (allocation × evaluator × importance) triple
2. **Update 26 experiments** that hardcode `allocation_method='greedy'` (list in `EXPERIMENT_BUGS.md`: C08, C09, C10, C11, C16-C19, S11, S13, N01-N08)
3. **Rebuild SPEED_TIERS** in `find_best_hybrid.py` based on F12 results
4. **Clear hybrid caches on server** — `rm -rf ~/.cache/kirby/hybrid_search/ ~/.cache/kirby/create_hybrid_cache/`
5. **Pull code on server** — `git pull` to get all fixes (classification metrics, augmentation passing, mutation reps, etc.)
6. **Run preliminary experiments** on all 33 ready datasets — complexity, similarity, noise

### NEXT — New datasets (check server, re-submit failures)

7. **Check server for timed-out/completed jobs** — some of these have been sitting for days and may have timed out. Check status, pull what's done, re-submit what failed:
   - Reaction: buchwald_hartwig, suzuki, bh_doyle_plates
   - DTA: davis, kiba, bindingdb_kd (DTA has been buggy)
   - Nucleotide: 22 datasets (first-ever run)
   - Graphinity DDG: foldx, rosetta, augmented
   - Ab DMS fill: adams (binding+expression), phillips_h3, koenig (binding+expression)
   - Ab CLF fill: tz_her2_multiclass, humanness_species
8. **As results land** — follow the New Dataset Pipeline checklist (above) for each: review → decide tiers → update DATASET_REPS → add to submit lists → run experiments
9. **Track what's scattered** — new rep caches, survey results, and augmentation data are spread across multiple server result directories. Consolidate when pulling.

### BLOCKED — FT experiments (HTC queue, ~1 week)

10. **Molecular FT fill** — lora_molformer, contrastive_molformer, graphmvp_ft on qm9, adme_hlm, adme_rlm, adme_mdr1. Script: `submit_molecular_ft_fill.sh`.
11. **Antibody ESM-2 LoRA fill** — esm2_lora on 29 active antibody datasets. Script: `submit_antibody_ft_fill.sh`.
12. **When FT results land:** update per-dataset tier tables (swap frozen→FT only where FT wins), then implement compute-once .npy saving + flag toggle.

### IMPLEMENTATION — New experiment categories

13. ~~**Noise N13 (alt noise distributions)**~~ **DONE** (2026-03-28) — N09-N13 all implemented. 5 new experiments + `lnl_methods.py` (12 LNL methods). See `NOISE_EXPERIMENT_ROADMAP.md`.
14. **Federated fed04 + fed06** — fed01-fed03 are implemented. Fed04 (cluster-aware importance) and fed06 (antibody extension, deprioritized) still planned. See `FEDERATED_EXPERIMENT_ROADMAP.md`.

### IMPLEMENTATION — Experiment infrastructure

16. ~~**Implement per-dataset file redesign**~~ **DONE** (2026-03-27) — replaced ~100 boilerplate .py files + 6 SLURM scripts with `run_dataset.py` + `submit_dataset.sh` + `submit_all_datasets.sh`. N10-N13 and fed02-fed06 pre-registered in helpers.
17. **S01-S09 classification metrics** — lower priority, but S04/S05/S06 compute R² on predictions which is wrong for classification datasets. See `EXPERIMENT_BUGS.md` Tier 2 similarity.

### EXPERIMENTS — Supplementary runs needed

18. **E2E SchNet on more molecular datasets** — ft_schnet already wired in `make_ft_factories` (line 632-641 of run_baseline_survey.py). Previous HTC runs failed due to torch-cluster bug (now fixed). Resubmitting herg, caco2, logd on HTC with `--finetune`. Output: `tests/results/schnet_e2e/`.
19. ~~**Augmentation per-dataset assignment (usrcat, coulomb_matrix, usr)**~~ **DONE** (2026-03-28) — Results from `molecular_aug_fill/`: caco2 all 3 positive (usrcat +0.021, usr +0.020, coulomb +0.015), adme_mdr1 coulomb_matrix +0.092, logd/qm9 flat or negative. DATASET_REPS already had correct assignments.
20. ~~**DNA foundation model rep implementation**~~ **DONE** — 14 FM reps implemented in `src/kirby/representations/nucleotide.py`. Survey results on server at `tests/results/nucleotide_survey/`.
21. **Nucleotide tier review** — pull survey results, review all 22 datasets with FM reps, decide tiers. Implementation done, pipeline readiness pending review.

### Decisions already locked

- **Molecular FT:** lora_molformer, contrastive_molformer, graphmvp_ft → general base reps. Per-dataset: swap frozen→FT only where FT beats frozen. herg keeps frozen molformer. All other FT variants (chemberta, selformer, freeze_head) excluded.
- **Molecular augmentation (confirmed):** pharmacophore3d, maccs → domain-wide. usrcat, coulomb_matrix, usr → dataset-specific: caco2 (all 3), adme_mdr1 (coulomb_matrix only +0.092), herg (all 3 from original survey). logd/qm9 don't benefit.
- **Molecular drops:** adme_solubility (R²=0.32 ceiling), unimol_v2 (dead everywhere).
- **Nucleotide drops:** nt_splice_sites_all (MCC=0.245), nt_H3K9me3 (MCC=0.239).
- **Augmentation defaults:** uncapped, step_size=5, patience=20. Per-experiment overrides in `EXPERIMENT_BUGS.md`.
- **Budgets:** [10,25,50,100,200,500,1000,2000] screening; [10,25,50,100,200,300,500,750,1000,1500,2000] full.
- **Two-stage search:** ABANDONED (F04 showed worse on 9/10 datasets).
- **Stability selection:** ABANDONED (F01/F11).
- ~~**Expand reaction rep set**~~ **DONE** — 14 original + 24 molecular variants = 38 total reaction reps.

---

## Dependency Chain

```
F12/F13 (ARC, running)
  → Finalize SPEED_TIERS + allocation defaults
  → Update 26 experiments
  → Clear caches
  → PRELIMINARY RUNS on 33 ready datasets (complexity/similarity/noise)

FT fill (HTC, queued ~1 week)
  → Molecular FT swap decisions (4 datasets)
  → Antibody FT swap decisions (29 datasets)
  → (Does NOT block preliminary runs — FT is enhancement)

New dataset surveys (HTC, check status)
  → Pipeline readiness decisions per dataset
  → Add to DATASET_REPS + submit lists
  → Run experiments when ready

Noise extensions (N09-N13 ALL DONE)
  → N09-N13 deploy alongside N01-N08 on same 33 datasets
  → New utility: lnl_methods.py (12 training-loop LNL methods)

Federated experiments (fed01-fed04, fed06 written, NEED REWORK)
  → All 5 experiments bypass KIRBy's config system (hardcoded models, budgets, allocations)
  → All use single 80/20 ShuffleSplit (should use cv_split with scaffold groups)
  → Config dict passed to run() but never referenced
  → No augmentation support
  → See FEDERATED_EXPERIMENT_ROADMAP.md "Known Issues" section for full list
  → Must refactor to use cv_split, config dict, get_model_factories before running

OPIG standalone experiments (6 scripts, NEED RECONSTRUCTION)
  → ALL 6 experiments deviate from original papers' methodology
  → Audit completed 2026-03-28: see OPIG_EXPERIMENT_AUDIT.md
  → Key issues per experiment:
    - Trastuzumab: paper did zero-shot library design, KIRBy does supervised classification (different task)
    - Graphinity: paper uses CDR identity cutoff analysis, KIRBy only groups by PDB (weaker)
    - CDR flexibility: paper trains on 73K general loops, KIRBy only uses tiny CDR test sets
    - Ginkgo: missing Top-10% recall metric (benchmark's secondary metric)
    - Activity cliff: missing half-AC exclusion, MMP partition analysis, MLP/KNN/GIN models
    - Humanness: audit against original repo NOT YET DONE
  → Cross-cutting: all 6 hardcode Ridge+RF instead of using config; all reimplement eval loops
  → Data loaders are CLEAN (verified against original repos)
  → MUST reconstruct before running
```

---

## Reference: Other Documentation

These documents are separate and maintained independently. Referenced here for navigation:

| Document | What it covers | When to consult |
|----------|---------------|-----------------|
| `EXPERIMENT_DESIGN_DISCUSSION.md` | Closed design issues, F12/F13 design, FT results tables, server job history | Historical decisions and methodology rationale |
| `EXPERIMENT_REFACTOR_ROADMAP.md` | Experiment → thesis chapter mapping, 4-step workflow, full inventory | Understanding what each C/S/N experiment tests |
| `EXPERIMENT_BUGS.md` | Tier-based bug tracker, greedy allocation list, hybrid usage audit | Before updating experiments after F12/F13 |
| `FRAMEWORK_EXPERIMENT_ROADMAP.md` | F00-F13 design and results | F12/F13 analysis |
| `NOISE_EXPERIMENT_ROADMAP.md` | N09-N13 design, LNL survey, noise estimation work | Implementing noise extensions |
| `FEDERATED_EXPERIMENT_ROADMAP.md` | fed01-fed06 design, FLuID integration | Implementing federated experiments |
| `OPIG_COLLAB_ROADMAP.md` | 11 OPIG lab publication items | OPIG-specific dataset decisions |
| `OPIG_EXPERIMENT_AUDIT.md` | 6/6 OPIG experiments audited against original GitHub repos | **Before running or modifying any OPIG experiment** |
| `BILL_OF_MATERIALS.md` | Comprehensive rep/dataset inventory, baseline results | Looking up what reps exist and their dimensions |
| `EXPERIMENT_OVERVIEW.md` | Reference for datasets, prediction tasks, representations | Quick lookup |
