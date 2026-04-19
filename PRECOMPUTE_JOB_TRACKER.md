# Precompute Hybrid Search — Job ID Tracker

**Submitted:** 2026-04-05 (v6 — treeshap added to full tier)
**Script:** `tests/slurm_scripts/submit_precompute.sh` (all domains)
**Total:** 40 datasets × 4 conditions = 160 SLURM jobs (IDs 11729406–11729565)
**Partition:** medium (48h max) | **Memory:** 64G | **CPUs:** 4 | **Time:** 12h | **Speed:** full (5-fold, 13 strategies, 3 importance methods)
**Cache version:** v9 | **KIRBY_SKIP_INSTALLS:** 1 | **FT:** off

## Conditions (4 per dataset)

| # | Condition | Reps | Augmentation |
|---|-----------|------|-------------|
| 1 | `general_none` | general | none |
| 2 | `general_uncapped` | general | uncapped |
| 3 | `top_none` | top_tier | none |
| 4 | `top_uncapped` | top_tier | uncapped |

## Quick status check

```bash
sacct -j 11729406-11729565 --format=State --noheader | grep -v "\.ba\|\.ex" | sort | uniq -c
sacct -j 11729406-11729565 --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex" | grep -E "FAILED|TIMEOUT|OUT_OF_ME"
```

---

## MOLECULAR (7 × 4 = 28 jobs)

### 1. herg_ki
| Job ID | Condition |
|--------|-----------|
| 11729406 | general_none |
| 11729407 | general_uncapped |
| 11729408 | top_none |
| 11729409 | top_uncapped |

### 2. caco2
| Job ID | Condition |
|--------|-----------|
| 11729410 | general_none |
| 11729411 | general_uncapped |
| 11729412 | top_none |
| 11729413 | top_uncapped |

### 3. logd
| Job ID | Condition |
|--------|-----------|
| 11729414 | general_none |
| 11729415 | general_uncapped |
| 11729416 | top_none |
| 11729417 | top_uncapped |

### 4. qm9
| Job ID | Condition |
|--------|-----------|
| 11729418 | general_none |
| 11729419 | general_uncapped |
| 11729420 | top_none |
| 11729421 | top_uncapped |

### 5. adme_hlm
| Job ID | Condition |
|--------|-----------|
| 11729422 | general_none |
| 11729423 | general_uncapped |
| 11729424 | top_none |
| 11729425 | top_uncapped |

### 6. adme_rlm
| Job ID | Condition |
|--------|-----------|
| 11729426 | general_none |
| 11729427 | general_uncapped |
| 11729428 | top_none |
| 11729429 | top_uncapped |

### 7. adme_mdr1
| Job ID | Condition |
|--------|-----------|
| 11729430 | general_none |
| 11729431 | general_uncapped |
| 11729432 | top_none |
| 11729433 | top_uncapped |

---

## ANTIBODY INTER-FAMILY (11 × 4 = 44 jobs)

### 8. flab_thermo_tresanco
| Job ID | Condition |
|--------|-----------|
| 11729434 | general_none |
| 11729435 | general_uncapped |
| 11729436 | top_none |
| 11729437 | top_uncapped |

### 9. flab_pk_jain_heprt
| Job ID | Condition |
|--------|-----------|
| 11729438 | general_none |
| 11729439 | general_uncapped |
| 11729440 | top_none |
| 11729441 | top_uncapped |

### 10. flab_pk_jain_fcrnrt
| Job ID | Condition |
|--------|-----------|
| 11729442 | general_none |
| 11729443 | general_uncapped |
| 11729444 | top_none |
| 11729445 | top_uncapped |

### 11. flab_polyreact_jain_cic
| Job ID | Condition |
|--------|-----------|
| 11729446 | general_none |
| 11729447 | general_uncapped |
| 11729448 | top_none |
| 11729449 | top_uncapped |

### 12. flab_aggregation_jain_hicrt
| Job ID | Condition |
|--------|-----------|
| 11729450 | general_none |
| 11729451 | general_uncapped |
| 11729452 | top_none |
| 11729453 | top_uncapped |

### 13. flab_thermo_jain
| Job ID | Condition |
|--------|-----------|
| 11729454 | general_none |
| 11729455 | general_uncapped |
| 11729456 | top_none |
| 11729457 | top_uncapped |

### 14. abcov_kd
| Job ID | Condition |
|--------|-----------|
| 11729458 | general_none |
| 11729459 | general_uncapped |
| 11729460 | top_none |
| 11729461 | top_uncapped |

### 15. flab_aggregation_jain
| Job ID | Condition |
|--------|-----------|
| 11729462 | general_none |
| 11729463 | general_uncapped |
| 11729464 | top_none |
| 11729465 | top_uncapped |

### 16. flab_immunogenicity_marks
| Job ID | Condition |
|--------|-----------|
| 11729466 | general_none |
| 11729467 | general_uncapped |
| 11729468 | top_none |
| 11729469 | top_uncapped |

### 17. flab_polyreact_jain_bvp
| Job ID | Condition |
|--------|-----------|
| 11729470 | general_none |
| 11729471 | general_uncapped |
| 11729472 | top_none |
| 11729473 | top_uncapped |

### 18. ddg_combined
| Job ID | Condition |
|--------|-----------|
| 11729474 | general_none |
| 11729475 | general_uncapped |
| 11729476 | top_none |
| 11729477 | top_uncapped |

---

## ANTIBODY DMS (5 × 4 = 20 jobs)

### 19. flab_binding_koenig
| Job ID | Condition |
|--------|-----------|
| 11729478 | general_none |
| 11729479 | general_uncapped |
| 11729480 | top_none |
| 11729481 | top_uncapped |

### 20. flab_expression_koenig
| Job ID | Condition |
|--------|-----------|
| 11729482 | general_none |
| 11729483 | general_uncapped |
| 11729484 | top_none |
| 11729485 | top_uncapped |

### 21. flab_binding_adams
| Job ID | Condition |
|--------|-----------|
| 11729486 | general_none |
| 11729487 | general_uncapped |
| 11729488 | top_none |
| 11729489 | top_uncapped |

### 22. flab_expression_adams
| Job ID | Condition |
|--------|-----------|
| 11729490 | general_none |
| 11729491 | general_uncapped |
| 11729492 | top_none |
| 11729493 | top_uncapped |

### 23. flab_binding_phillips_h3
| Job ID | Condition |
|--------|-----------|
| 11729494 | general_none |
| 11729495 | general_uncapped |
| 11729496 | top_none |
| 11729497 | top_uncapped |

---

## ABAGYM (5 × 4 = 20 jobs)

### 24. abagym_Ang2
| Job ID | Condition |
|--------|-----------|
| 11729498 | general_none |
| 11729499 | general_uncapped |
| 11729500 | top_none |
| 11729501 | top_uncapped |

### 25. abagym_EGFR
| Job ID | Condition |
|--------|-----------|
| 11729502 | general_none |
| 11729503 | general_uncapped |
| 11729504 | top_none |
| 11729505 | top_uncapped |

### 26. abagym_HER2
| Job ID | Condition |
|--------|-----------|
| 11729506 | general_none |
| 11729507 | general_uncapped |
| 11729508 | top_none |
| 11729509 | top_uncapped |

### 27. abagym_lysozyme
| Job ID | Condition |
|--------|-----------|
| 11729510 | general_none |
| 11729511 | general_uncapped |
| 11729512 | top_none |
| 11729513 | top_uncapped |

### 28. abagym_VEGF
| Job ID | Condition |
|--------|-----------|
| 11729514 | general_none |
| 11729515 | general_uncapped |
| 11729516 | top_none |
| 11729517 | top_uncapped |

---

## ANTIBODY CLF (2 × 4 = 8 jobs)

### 29. tz_her2_binary
| Job ID | Condition |
|--------|-----------|
| 11729518 | general_none |
| 11729519 | general_uncapped |
| 11729520 | top_none |
| 11729521 | top_uncapped |

### 30. humanness_therapeutic
| Job ID | Condition |
|--------|-----------|
| 11729522 | general_none |
| 11729523 | general_uncapped |
| 11729524 | top_none |
| 11729525 | top_uncapped |

---

## GINKGO (6 × 4 = 24 jobs)

### 31. ginkgo_acsins
| Job ID | Condition |
|--------|-----------|
| 11729526 | general_none |
| 11729527 | general_uncapped |
| 11729528 | top_none |
| 11729529 | top_uncapped |

### 32. ginkgo_colloidal
| Job ID | Condition |
|--------|-----------|
| 11729530 | general_none |
| 11729531 | general_uncapped |
| 11729532 | top_none |
| 11729533 | top_uncapped |

### 33. ginkgo_heparin
| Job ID | Condition |
|--------|-----------|
| 11729534 | general_none |
| 11729535 | general_uncapped |
| 11729536 | top_none |
| 11729537 | top_uncapped |

### 34. ginkgo_hic
| Job ID | Condition |
|--------|-----------|
| 11729538 | general_none |
| 11729539 | general_uncapped |
| 11729540 | top_none |
| 11729541 | top_uncapped |

### 35. ginkgo_pr_cho
| Job ID | Condition |
|--------|-----------|
| 11729542 | general_none |
| 11729543 | general_uncapped |
| 11729544 | top_none |
| 11729545 | top_uncapped |

### 36. ginkgo_titer
| Job ID | Condition |
|--------|-----------|
| 11729546 | general_none |
| 11729547 | general_uncapped |
| 11729548 | top_none |
| 11729549 | top_uncapped |

---

## TCGA (2 × 4 = 8 jobs)

### 37. tcga_luad_vs_lusc
| Job ID | Condition |
|--------|-----------|
| 11729550 | general_none |
| 11729551 | general_uncapped |
| 11729552 | top_none |
| 11729553 | top_uncapped |

### 38. tcga_brca_pam50
| Job ID | Condition |
|--------|-----------|
| 11729554 | general_none |
| 11729555 | general_uncapped |
| 11729556 | top_none |
| 11729557 | top_uncapped |

---

## REACTION (2 × 4 = 8 jobs)

### 39. buchwald_hartwig
| Job ID | Condition |
|--------|-----------|
| 11729558 | general_none |
| 11729559 | general_uncapped |
| 11729560 | top_none |
| 11729561 | top_uncapped |

### 40. bh_doyle_plates
| Job ID | Condition |
|--------|-----------|
| 11729562 | general_none |
| 11729563 | general_uncapped |
| 11729564 | top_none |
| 11729565 | top_uncapped |

---

## SUBMISSION HISTORY

| Date | Version | Jobs | Details |
|------|---------|------|---------|
| 2026-04-04 | v1 | 240 | long, 128-256G, fast speed, 6 conditions. `list(set())` bug. Cancelled. |
| 2026-04-04 | v2 | 23 | Resubmit after group fix. Still fast speed. Cancelled. |
| 2026-04-04 | v3 | 160 | long, 128-256G, full speed, 4 conditions. 256G queue stuck + FT on. Cancelled. |
| 2026-04-05 | v4 | 160 | medium, 64G, full speed, no FT. setup.sh install checks hung. Cancelled. |
| 2026-04-05 | v5 | 160 | + KIRBY_SKIP_INSTALLS. Missing treeshap in full tier. Cancelled. |
| 2026-04-05 | **v6** | **160 (11729406–11729565)** | **medium, 64G, 4 CPUs, full speed (3 imp methods), 4 conditions, no FT, SKIP_INSTALLS, cache v9.** |

---

## HTC SURVEY JOBS (2026-04-06)

**Partition:** long | **Purpose:** Baseline surveys — if they pass, datasets get added to precompute.

> **⚠️ WARNING:** These were submitted from `/home/scat9264/` by mistake — SLURM logs at `/home/scat9264/slurm-<jobid>.out` instead of `/data/`. Always `cd /data/stat-cadd/scat9264/KIRBy` before running `sbatch`.

### DTA

| Job ID | Name | Status |
|--------|------|--------|
| 7440291 | dta_davis | TIMEOUT (48h wall limit) |
| 7440292 | dta_kiba | DONE |
| 7440293 | dta_bindingdb_kd | **TIMEOUT** (4d wall limit, 2026-04-11) — end-of-run save only, zero output on disk |
| 7483661 | dta_bindingdb_kd (pre-c18342f heredoc bug) | **CANCELLED** (2026-04-15, same heredoc bug as 7483641-46) |
| 7483718 | dta_bindingdb_kd (post-c18342f, per-rep checkpoint) | **TIMEOUT** (2026-04-18, 1d23h59m, exit 0:0). Died at contact_graph 50/1219 (~58 min/protein). 17 protein reps + 7 drug reps cached as `.npy`; protein_contact_graph + protein_usr_shape + protein_pocket + protein_pocket_composition + protein_esm2_pocket never computed. Superseded by 7515602 (post-d829df1 contact_graph speedup). |
| 7467954 | dta_davis (resubmit, --skip-existing) | **TIMEOUT** (2026-04-15, 48h wall) |
| 7483719 | dta_davis (post-c18342f + a81b0eb) | **RUNNING** (2026-04-19, 2d15h+ elapsed, long+GPU+256G, 96h wall) |
| 7457314 | dta_kiba_fill (--skip-existing) | FAILED (--wrap bug) |
| 7483635 | dta_kiba_fill (pre-c18342f) | CANCELLED |
| 7483717 | dta_kiba_fill (post-c18342f) | **COMPLETED** (23min, 2026-04-16) — **BUT destroyed kiba.json** (69KB→583B) due to merge bug in a81b0eb. Fixed in d1c45b9. |
| 7496790 | dta_kiba (full re-run, post-d1c45b9 merge fix) | **COMPLETED** (17h11m, 2026-04-18). Output `tests/results/phase5_baselines/dta/kiba.json`. Several ESM-family reps showed large negative R² (e.g. esm2 R²=-1.58, esmc R²=-2.37) but CIs in normal range — flag for analysis but not a job failure. |
| 7515602 | dta_bindingdb_kd (post-d829df1, hand-rolled --wrap) | **FAILED** (14s, ExitCode 127:0, 2026-04-20). Long inline `--wrap` line-split by SLURM, python invocation orphaned from args (same bug class as 7457285/86, 7467966-9). Plus server git was on divergent branches → silent `git pull 2>/dev/null` swallowed the failure. Reset server to origin/main + dedicated heredoc script (commit 42a22c4). Superseded by 7515675. |
| 7515675 | dta_bindingdb_kd (post-42a22c4, heredoc script) | **SUBMITTED** (2026-04-20, long+GPU+128G, 24h wall). Uses `tests/slurm_scripts/submit_dta_bindingdb.sh` (real `#SBATCH` directives, no `--wrap`). `git pull --ff-only` so any future divergence fails loudly. Estimated runtime 3-6h. |

### Antibody CLF

| Job ID | Name | Status |
|--------|------|--------|
| 7440294 | abclf_tz | FAILED (KeyError: 'models', --reps all bug) |
| 7440295 | abclf_humanness | FAILED (KeyError: 'models', --reps all bug) |
| 7457285 | abclf_tz (resubmit, bug fixed) | **FAILED** (ExitCode 126:0, 7min — sbatch --wrap line-splitting bug; `python` prefix orphaned from script path at slurm_script line 5) |
| 7457286 | abclf_humanness (resubmit, bug fixed) | **FAILED** (same — 126:0, 6min) |
| 7483631 | abclf_tz_her2_binary (pre-c18342f) | **CANCELLED** (2026-04-15, bad heredoc) |
| 7483632 | abclf_tz_her2_multiclass (pre-c18342f) | **CANCELLED** |
| 7483633 | abclf_humanness_species (pre-c18342f) | **CANCELLED** |
| 7483634 | abclf_humanness_therapeutic (pre-c18342f) | **CANCELLED** |
| 7483713 | abclf_tz_her2_binary (post-c18342f) | **COMPLETED** (3h51m, 2026-04-16) |
| 7483714 | abclf_tz_her2_multiclass (post-c18342f) | **COMPLETED** (8h22m, 2026-04-16) |
| 7483715 | abclf_humanness_species (post-c18342f) | **COMPLETED** (5h26m, 2026-04-16) — **output file missing from disk** despite "Saved" in slurm log. Resubmitted as 7496824. |
| 7483716 | abclf_humanness_therapeutic (post-c18342f) | **COMPLETED** (38min, 2026-04-16) |
| 7496824 | abclf_humanness_species (resubmit) | **FAILED** (ExitCode 127, 4min — long --reps argument split across slurm_script lines by SLURM. Fixed by using `REPS=` shell variable.) |
| 7498632 | abclf_humanness_species (resubmit 2, REPS= variable) | **COMPLETED** (4h04m, 2026-04-18). Output `tests/results/opig/clf/humanness_species.json`. **DATASET CUT (2026-04-19)** — most reps hit 0.99+ AUROC (germline V/J usage ≈ trivially separable per species), and igbert/igbert_cls/mint/xtrimopglm collapsed to majority-class 0.679/0.549 across all folds (paired-chain models incompatible with VH-only input). Not informative; remove from precompute config + tier table. |
| 7483636 | abclf_tz_her2_binary (DUPLICATE) | **CANCELLED** (2026-04-15) |
| 7483637 | abclf_tz_her2_multiclass (DUPLICATE) | **CANCELLED** |
| 7483638 | abclf_humanness_species (DUPLICATE) | **CANCELLED** |
| 7483639 | abclf_humanness_therapeutic (DUPLICATE) | **CANCELLED** |
| 7483640 | dta_kiba_fill (DUPLICATE) | **CANCELLED** |

### SchNet (molecular)

| Job ID | Name | Status |
|--------|------|--------|
| 7440325 | schnet_logd | **FAILED** (SLURM exit 0:0 but experiment broken, 4h17m, 2026-04-12) — baselines ran but all 3 FT reps (ft_schnet, lora_chemberta, lora_molformer) emitted `FT FACTORY ERROR: No data left in file` from `np.load(cache_path)` on every fold. Root cause: fc36c6c np.savez suffix bug (zero-byte cache entries). |
| 7440326 | schnet_herg_ki | **FAILED** (1h50m, 2026-04-12) — same FT cache corruption |
| 7440327 | schnet_caco2 | **FAILED** (1h13m, 2026-04-12) — same FT cache corruption |
| 7483721 | schnet_logd (post-fc36c6c) | **COMPLETED** (1h05m, 2026-04-16) — real lora_chemberta/lora_molformer scores, **ft_schnet=NaN** (torch_cluster not CUDA-compiled) |
| 7483722 | schnet_herg (post-fc36c6c) | **FAILED** (7min, 2026-04-16, ExitCode 1:0) — TDC loader crash |
| 7483723 | schnet_caco2 (post-fc36c6c) | **COMPLETED** (45min, 2026-04-16) — same: ft_schnet=NaN (CUDA issue) |
| 7498292 | schnet_logd (post-b59c11e CPU fallback) | **COMPLETED** (3h26m, 2026-04-18). Real ft_schnet finally landed: ft_schnet=0.609 (ridge), lora_chemberta=0.757, lora_molformer=0.565. CPU fallback works. |
| 7498293 | schnet_herg (post-b59c11e + 67803ab FLuID loader) | **FAILED** (7min, 2026-04-18, ExitCode 1:0). FLuID `hERG_lhasa_training.pkl` not on server — `data_dir='data/fluid'` was hardcoded and never existed. Fixed in d3e2cc1 (env-var resolution + multiple fallback paths) + user scp'd pickles to `/data/stat-cadd/scat9264/lhasa_data/`. Superseded by 7515601. |
| 7498294 | schnet_caco2 (post-b59c11e) | **COMPLETED** (1h06m, 2026-04-18). Real ft_schnet via CPU fallback. |
| 7515601 | schnet_herg (post-d3e2cc1 + scp'd FLuID) | **SUBMITTED** (2026-04-20, long+GPU+128G, 24h wall). |

### Nucleotide (genomics)

| Job ID | Name | Status |
|--------|------|--------|
| 7440301 | nuc_nt_promoter_all | DONE (partial, pre-fix) |
| 7440302 | nuc_nt_promoter_tata | DONE (partial, pre-fix) |
| 7440303 | nuc_nt_promoter_no_tata | DONE (partial, pre-fix) |
| 7440304 | nuc_nt_enhancers | CANCELLED (pre-fix, replaced by 7459731) |
| 7440305 | nuc_nt_enhancers_types | CANCELLED (pre-fix, replaced by 7459732) |
| 7440306 | nuc_nt_splice_sites_all | DONE (partial, pre-fix) |
| 7440307 | nuc_nt_splice_sites_acceptors | CANCELLED (pre-fix, replaced by 7459733) |
| 7440308 | nuc_nt_splice_sites_donors | CANCELLED (pre-fix, replaced by 7459734) |
| 7459731 | nuc_nt_enhancers (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7459732 | nuc_nt_enhancers_types (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7459733 | nuc_nt_splice_sites_acceptors (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7459734 | nuc_nt_splice_sites_donors (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7467966 | nuc_nt_enhancers (resubmit 2) | **FAILED** (ExitCode 126:0, 6min — sbatch --wrap line-splitting bug; `mkdir -p` orphaned from operand at slurm_script line 5) |
| 7467967 | nuc_nt_enhancers_types (resubmit 2) | **FAILED** (same — 126:0, 6min) |
| 7467968 | nuc_nt_splice_sites_acceptors (resubmit 2) | **FAILED** (same — 126:0, 6min) |
| 7467969 | nuc_nt_splice_sites_donors (resubmit 2) | **FAILED** (same — 126:0, 6min) |
| 7483641 | nuc_nt_enhancers (heredoc resubmit 3, pre-c18342f) | **FAILED** (2026-04-15, heredoc `\\<LF>` parsing bug — bash joined continuation lines, argparse got empty-string args) |
| 7483642 | nuc_nt_enhancers_types (pre-c18342f) | **FAILED** (same) |
| 7483643 | nuc_nt_splice_sites_acceptors (pre-c18342f) | **FAILED** (same — user confirmed) |
| 7483644 | nuc_nt_splice_sites_donors (pre-c18342f) | **FAILED** (same) |
| 7483707 | nuc_nt_enhancers (post-c18342f) | **COMPLETED** (3m39s, 2026-04-15) — gena_lm still 32000D (old cache loaded pre-3a5b5d5). Needs gena_refix. |
| 7483708 | nuc_nt_enhancers_types (post-c18342f) | **COMPLETED** (3m17s) — same, gena_lm 32000D |
| 7483709 | nuc_nt_splice_sites_acceptors (post-c18342f) | **COMPLETED** (3m29s) — same |
| 7483710 | nuc_nt_splice_sites_donors (post-c18342f) | **COMPLETED** (3m18s) — same |
| 7440309 | nuc_nt_H2AFZ | DONE (2026-04-12) — **gena_lm CONTAMINATED (32000D logits)**, superseded by 7485363 refix |
| 7440310 | nuc_nt_H3K27ac | DONE — **gena_lm CONTAMINATED**, superseded by 7485364 refix |
| 7440311 | nuc_nt_H3K27me3 | DONE — **gena_lm CONTAMINATED**, superseded by 7485365 refix |
| 7440312 | nuc_nt_H3K36me3 | DONE — **gena_lm CONTAMINATED**, superseded by 7485366 refix |
| 7440313 | nuc_nt_H3K4me1 | DONE — **gena_lm CONTAMINATED**, superseded by 7485367 refix |
| 7440314 | nuc_nt_H3K4me2 | DONE (1d14h, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485368 refix |
| 7440315 | nuc_nt_H3K4me3 | DONE (1d3h, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485369 refix |
| 7440316 | nuc_nt_H3K9ac | DONE (1d11h, 2026-04-13) — **gena_lm CONTAMINATED**, superseded by 7485370 refix |
| 7440317 | nuc_nt_H3K9me3 | DONE (1d20h, 2026-04-13) — **gena_lm CONTAMINATED**, superseded by 7485371 refix |
| 7440318 | nuc_nt_H4K20me1 | DONE (22h, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485372 refix |
| 7440319 | nuc_gb_human_nontata_promoters | DONE (10h38m, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485377 refix |
| 7440320 | nuc_gb_human_enhancers_cohn | DONE (16h35m, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485378 refix |
| 7440321 | nuc_gb_demo_coding_vs_intergenomic | **OUT_OF_MEMORY** (1h44m, 2026-04-12) — 128G insufficient, resubmit with 256G |
| 7440322 | nuc_gb_demo_human_or_worm | **OUT_OF_MEMORY** (1h46m, 2026-04-12) — 128G insufficient, resubmit with 256G |
| 7483645 | nuc_gb_demo_coding_vs_intergenomic (pre-c18342f) | **FAILED** (same heredoc bug) |
| 7483646 | nuc_gb_demo_human_or_worm (pre-c18342f) | **FAILED** (same heredoc bug) |
| 7483711 | nuc_gb_demo_coding (post-c18342f, 256G) | **CANCELLED** (12h42m in, 2026-04-16) |
| 7483712 | nuc_gb_demo_human_or_worm (post-c18342f, 256G) | **CANCELLED** (12h42m) |
| 7496807 | nuc_gb_demo_coding (resubmit, 256G) | **COMPLETED** (6h24m, 2026-04-18). Winner: nt_v2_250m+mlp=0.8392 MCC. |
| 7496808 | nuc_gb_demo_human_or_worm (resubmit, 256G) | **COMPLETED** (6h25m, 2026-04-18). |

### Nucleotide gena_lm refix (2026-04-15)

Target: fix broken gena_lm entries in 16 completed nucleotide JSONs.
Root cause: `_embed_gena_lm` was returning 32000D LM head logits
instead of 1024D hidden states (fixed in 3a5b5d5). Each job runs
`run_baseline_survey.py <ds> --reps gena_lm --n-folds 5 --output
tests/results/nucleotide_survey/`. No --skip-existing: the merge
logic in run_baseline_survey.py lines 1243-1264 overwrites ONLY the
gena_lm entry while preserving all other reps. Preflight check
inside each job aborts if any stale `nuc_*/gena_lm.npy` cache files
remain (they should have been deleted via rm before submission).

Resource: long+GPU+64G+6h per job.

| Job ID | Name | Status |
|--------|------|--------|
| 7485363 | gena_refix_nt_H2AFZ | **COMPLETED** (34min, 2026-04-15) — confirmed 1024D gena_lm |
| 7485364-78 | gena_refix (15 remaining) | **FAILED** (instant, ExitCode 2) — preflight false-positive on newly-written gena_lm.npy. Fixed in 0913452 (preflight removed). |

**Round 2 (post-0913452, no preflight):**

| Job ID | Name | Status |
|--------|------|--------|
| 7496791 | gena_refix_nt_H2AFZ | **COMPLETED** (31min, 2026-04-17) |
| 7496792 | gena_refix_nt_H3K27ac | **COMPLETED** (35min) |
| 7496793 | gena_refix_nt_H3K27me3 | **COMPLETED** (35min) |
| 7496794 | gena_refix_nt_H3K36me3 | **COMPLETED** (36min) |
| 7496795 | gena_refix_nt_H3K4me1 | **COMPLETED** (34min) |
| 7496796 | gena_refix_nt_H3K4me2 | **COMPLETED** (31min) |
| 7496797 | gena_refix_nt_H3K4me3 | **COMPLETED** (19min) |
| 7496798 | gena_refix_nt_H3K9ac | **COMPLETED** (22min) |
| 7496799 | gena_refix_nt_H3K9me3 | **COMPLETED** (26min) |
| 7496800 | gena_refix_nt_H4K20me1 | **COMPLETED** (36min) |
| 7496801 | gena_refix_nt_promoter_all | **COMPLETED** (25min) |
| 7496802 | gena_refix_nt_promoter_tata | **COMPLETED** (8min) |
| 7496803 | gena_refix_nt_promoter_no_tata | **COMPLETED** (26min) |
| 7496804 | gena_refix_nt_splice_sites_all | **COMPLETED** (40min) |
| 7496805 | gena_refix_gb_human_nontata_promoters | **COMPLETED** (30min) |
| 7496806 | gena_refix_gb_human_enhancers_cohn | **COMPLETED** (27min) |

**Round 3 (2026-04-17): 4 datasets missed from rounds 1-2 (the freshly-surveyed nt_enhancers/types + splice_acceptors/donors that ran pre-3a5b5d5 with old cache):**

| Job ID | Name | Status |
|--------|------|--------|
| 7499338 | gena_refix_nt_enhancers | **COMPLETED** (~30min, 2026-04-17) |
| 7499339 | gena_refix_nt_enhancers_types | **COMPLETED** (44min, 2026-04-18) |
| 7499340 | gena_refix_nt_splice_sites_acceptors | **COMPLETED** (31min, 2026-04-18). Winner: gena_lm+ridge=0.0317 MCC (weak — confirms classicals dominate this task). |
| 7499341 | gena_refix_nt_splice_sites_donors | **COMPLETED** (30min, 2026-04-18) |

All 20 nucleotide datasets (16 + 4) now have corrected 1024D gena_lm. Tier tables for the 6 newly-completed nucleotide datasets (4 round-3 + 2 gb_demo) pending PIPELINE_READINESS_TODO update — see analysis section below.

### CGR-SMILES pretraining (Phase 4)

Single job: corpus generation (~1M USPTO → CGR-SMILES) + continued MLM pretraining (50K steps).
Partition: long, 128G, 8h, 16 CPU, no GPU.
Output: `/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/`

| Job ID | Name | Stage | Status |
|--------|------|-------|--------|
| 11747734 | cgr_smiles | corpus + pretrain | FAILED — chython not installed, 0/392K conversions. Fixed in 9fa01f8. |
| 7467971 | cgr_smiles | corpus + pretrain | FAILED — chython installed but 0/392K conversions. Empty corpus guard caught it. Diagnostics added in 4b7d9de. |
| 11748808 | cgr_smiles | corpus + pretrain | FAILED — CGRContainer.__len__ missing, smoke test caught it. Fixed in f0b5e86. |
| 11749515 | cgr_smiles | corpus + pretrain | **FAILED** (2026-04-11) — corpus stage OK (392324/392533, 99.9%), pretrain stage crashed: `AttributeError: 'AdamW' object has no attribute 'train'` in accelerate/optimizer.py line 128. Root cause: transformers Trainer → accelerate AcceleratedOptimizer.train() → calls inner optimizer.train() which torch.optim.AdamW does not implement (accelerate/transformers version mismatch). Fixed in 2c367d6 (manual training loop). |
| 7483446 | cgr_pretrain | pretrain only | **COMPLETED** (10min56s, 2026-04-15) — checkpoint at `/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint/` |

### Reaction re-survey — UNBLOCKED

CGR pretrain complete. Re-survey with 41 reps (38 original + 3 CGR) not yet submitted.
Existing reaction JSONs are from Mar 19-26 (pre-CGR). Need new submission script (heredoc-style).

| Job ID | Name | Datasets | Status |
|--------|------|----------|--------|
| — | rxn_baselines | buchwald_hartwig, bh_doyle_plates, suzuki | **NOT YET SUBMITTED** — needs heredoc submit script with cgr_smiles_lm |

---

## 2026-04-17 — Consolidated precompute (v10, registry-backed) — 114 jobs

**Submitted:** 2026-04-17.
**Script:** `tests/scripts/plan_precompute.py --consolidated --cluster arc`.
**Infrastructure:** Phase A-H + D.2 hybrid_evals dedup (commits b28d092 → eccb6a3).
**Cache:** content-validated, no version number. 23-field meta per search cache.
**Mode:** consolidated — one sbatch per `(dataset, standard|distance)`, each
running all applicable tier × aug_strategy conditions sequentially inside a
single python process. 2 jobs per dataset instead of 10 = 5× fewer sbatch
calls vs v6-v9 per-condition submissions.

### Superseded (cancelled 2026-04-17)

| Job ID range | Notes |
|--------------|-------|
| 11781449-11781478 | 30 jobs, per-condition mode for flab_pk_jain_fcrnrt + abagym_HER2 + humanness_therapeutic. Cancelled + resubmitted as part of the new consolidated batch. |

### Summary by partition

| Cluster | Partition | Range | Count | Band(s) | Time | Mem |
|---------|-----------|-------|-------|---------|------|-----|
| arc | short | 11781997-11782074 | 74 | XS / S / M | 12:00:00 | 64-128G |
| arc | medium | 11782049-11782050, 11782075-11782082 | 10 | L, XL | 2-00:00:00 | 128-256G |
| arc | long | 11782083-11782113 (11782090 missing) | 30 | NT | 2-00:00:00 | 128G |

Total: **114 jobs, 57 datasets**.

### Datasets by partition

**arc short** (XS/S/M band, fits 12h):
- Molecular (8 datasets × 2 = 16): caco2, logd, qm9, adme_hlm, adme_rlm, adme_mdr1, herg_ki, herg
- Antibody inter-family + DMS (24 × 2 = 48): 5 abagym (Ang2, EGFR, VEGF, lysozyme, HER2), abcov_kd, 12 flab_* (all except thermo_tresanco), 6 ginkgo (acsins, colloidal, heparin, hic, pr_cho, titer), humanness_therapeutic, ddg_combined
- Reaction (2 × 2 = 4): buchwald_hartwig, bh_doyle_plates
- DTA (3 × 2 = 6): davis, kiba, bindingdb_kd

**arc medium** (L/XL band, needs ≥12h buffer):
- flab_thermo_tresanco (L; empirically 8h on general_uncapped)
- tz_her2_binary, tz_her2_multiclass (L; n=5000)
- tcga_luad_vs_lusc, tcga_brca_pam50 (XL; high-dim reps)

**arc long** (NT band, 15 nucleotide datasets × 2 = 30):
- nt_H2AFZ, nt_H3K27ac, nt_H3K27me3, nt_H3K36me3, nt_H3K4me1,
  nt_H3K4me2, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_H4K20me1,
  nt_promoter_all, nt_promoter_tata, nt_promoter_no_tata,
  gb_human_nontata_promoters, gb_human_enhancers_cohn

### Conditions per dataset (consolidated mode)

Each dataset gets 2 sbatch submissions:

| Job suffix | Conditions run inside the process |
|------------|----------------------------------|
| `_all_standard` | general_none, general_uncapped, general_capped, top_none, top_uncapped, top_capped |
| `_all_distance` | general_distance_none, general_distance_uncapped, top_distance_none, top_distance_uncapped (DISTANCE_EVALUATORS + DISTANCE_BUDGETS) |

### Status checks

```bash
# Live queue
squeue --clusters=all -u $USER | wc -l

# Summary after completion
sacct -j 11781997-11782113 --format=State --noheader -X | sort | uniq -c

# Failures
sacct -j 11781997-11782113 --format=JobID,JobName%40,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"

# [cache] summary lines from Phase E — confirms observability + D.2 dedup
grep -h "^\[cache\]" slurm-*.out | sort | uniq -c
```

### Expected outcomes

- XS band (abagym_HER2, ginkgo_heparin, ginkgo_pr_cho): 1-3h each
- Full short-partition drain: ~12h depending on priority
- Medium partition L/XL jobs: 8h–1d typical
- Long partition nucleotide: 6-12h each (reps pre-cached as .npy)
- `[cache] hybrid_eval: N hit` appears in SLURM logs when top_k allocation picks the same subset across general and top tier searches — first production test of Phase D.2 dedup
