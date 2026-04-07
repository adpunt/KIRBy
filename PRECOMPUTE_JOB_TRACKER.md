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
| 7440291 | dta_davis | R |
| 7440292 | dta_kiba | PD |
| 7440293 | dta_bindingdb | PD |

### Antibody CLF

| Job ID | Name | Status |
|--------|------|--------|
| 7440294 | abclf_tz | PD |
| 7440295 | abclf_humanness | PD |

### SchNet (molecular)

| Job ID | Name | Status |
|--------|------|--------|
| 7440325 | schnet_logd | PD |
| 7440326 | schnet_herg | PD |
| 7440327 | schnet_caco2 | PD |

### Nucleotide (genomics)

| Job ID | Name | Status |
|--------|------|--------|
| 7440301 | nuc_nt_promoter | PD |
| 7440302 | nuc_nt_promoter | PD |
| 7440303 | nuc_nt_promoter | PD |
| 7440304 | nuc_nt_enhancer | PD |
| 7440305 | nuc_nt_enhancer | PD |
| 7440306 | nuc_nt_splice | PD |
| 7440307 | nuc_nt_splice | PD |
| 7440308 | nuc_nt_splice | PD |
| 7440309 | nuc_nt_H3 | PD |
| 7440310 | nuc_nt_H3 | PD |
| 7440311 | nuc_nt_H3 | PD |
| 7440312 | nuc_nt_H3 | PD |
| 7440313 | nuc_nt_H3 | PD |
| 7440314 | nuc_nt_H3 | PD |
| 7440315 | nuc_nt_H3 | PD |
| 7440316 | nuc_nt_H3 | PD |
| 7440317 | nuc_nt_H3 | PD |
| 7440318 | nuc_nt_H3 | PD |
| 7440319 | nuc_gb_human | PD |
| 7440320 | nuc_gb_human | PD |
| 7440321 | nuc_gb_demo | PD |
| 7440322 | nuc_gb_demo | PD |
