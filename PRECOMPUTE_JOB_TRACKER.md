# Precompute Hybrid Search — Job ID Tracker

**Submitted:** 2026-04-05 (v4 — medium partition, 64G, no FT, deterministic groups, full speed)
**Script:** `tests/slurm_scripts/submit_precompute.sh` (all domains)
**Total:** 40 datasets × 4 conditions = 160 SLURM jobs (IDs 11728481–11728687)
**Partition:** medium (48h max) | **Memory:** 64G | **Time:** 12h | **Speed:** full (5-fold, 13 strategies)
**Cache version:** v9 (deterministic scaffold groups)

## Conditions (4 per dataset)

| # | Condition | Reps | Augmentation |
|---|-----------|------|-------------|
| 1 | `general_none` | general | none |
| 2 | `general_uncapped` | general | uncapped |
| 3 | `top_none` | top_tier | none |
| 4 | `top_uncapped` | top_tier | uncapped |

## Quick status check

```bash
sacct -j 11728481-11728687 --format=State --noheader | grep -v "\.ba\|\.ex" | sort | uniq -c
sacct -j 11728481-11728687 --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex" | grep -E "FAILED|TIMEOUT|OUT_OF_ME"
```

---

## MOLECULAR (7 × 4 = 28 jobs)

### 1. herg_ki — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728481 | general_none | PENDING |
| 11728482 | general_uncapped | PENDING |
| 11728483 | top_none | PENDING |
| 11728484 | top_uncapped | PENDING |

### 2. caco2 — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728485 | general_none | PENDING |
| 11728486 | general_uncapped | PENDING |
| 11728487 | top_none | PENDING |
| 11728488 | top_uncapped | PENDING |

### 3. logd — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728489 | general_none | PENDING |
| 11728490 | general_uncapped | PENDING |
| 11728491 | top_none | PENDING |
| 11728492 | top_uncapped | PENDING |

### 4. qm9 — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728493 | general_none | PENDING |
| 11728494 | general_uncapped | PENDING |
| 11728495 | top_none | PENDING |
| 11728496 | top_uncapped | PENDING |

### 5. adme_hlm — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728497 | general_none | PENDING |
| 11728498 | general_uncapped | PENDING |
| 11728499 | top_none | PENDING |
| 11728500 | top_uncapped | PENDING |

### 6. adme_rlm — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728501 | general_none | PENDING |
| 11728502 | general_uncapped | PENDING |
| 11728503 | top_none | PENDING |
| 11728504 | top_uncapped | PENDING |

### 7. adme_mdr1 — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728505 | general_none | PENDING |
| 11728506 | general_uncapped | PENDING |
| 11728507 | top_none | PENDING |
| 11728508 | top_uncapped | PENDING |

---

## ANTIBODY INTER-FAMILY (11 × 4 = 44 jobs)

### 8. flab_thermo_tresanco — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728509 | general_none | PENDING |
| 11728510 | general_uncapped | PENDING |
| 11728511 | top_none | PENDING |
| 11728512 | top_uncapped | PENDING |

### 9. flab_pk_jain_heprt — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728513 | general_none | PENDING |
| 11728514 | general_uncapped | PENDING |
| 11728515 | top_none | PENDING |
| 11728516 | top_uncapped | PENDING |

### 10. flab_pk_jain_fcrnrt — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728517 | general_none | PENDING |
| 11728518 | general_uncapped | PENDING |
| 11728519 | top_none | PENDING |
| 11728520 | top_uncapped | PENDING |

### 11. flab_polyreact_jain_cic — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728521 | general_none | PENDING |
| 11728522 | general_uncapped | PENDING |
| 11728523 | top_none | PENDING |
| 11728524 | top_uncapped | PENDING |

### 12. flab_aggregation_jain_hicrt — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728525 | general_none | PENDING |
| 11728526 | general_uncapped | PENDING |
| 11728527 | top_none | PENDING |
| 11728561 | top_uncapped | PENDING |

### 13. flab_thermo_jain — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728562 | general_none | PENDING |
| 11728563 | general_uncapped | PENDING |
| 11728564 | top_none | PENDING |
| 11728565 | top_uncapped | PENDING |

### 14. abcov_kd — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728566 | general_none | PENDING |
| 11728567 | general_uncapped | PENDING |
| 11728568 | top_none | PENDING |
| 11728569 | top_uncapped | PENDING |

### 15. flab_aggregation_jain — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728570 | general_none | PENDING |
| 11728571 | general_uncapped | PENDING |
| 11728572 | top_none | PENDING |
| 11728573 | top_uncapped | PENDING |

### 16. flab_immunogenicity_marks — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728574 | general_none | PENDING |
| 11728575 | general_uncapped | PENDING |
| 11728576 | top_none | PENDING |
| 11728577 | top_uncapped | PENDING |

### 17. flab_polyreact_jain_bvp — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728578 | general_none | PENDING |
| 11728579 | general_uncapped | PENDING |
| 11728580 | top_none | PENDING |
| 11728581 | top_uncapped | PENDING |

### 18. ddg_combined — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728582 | general_none | PENDING |
| 11728583 | general_uncapped | PENDING |
| 11728584 | top_none | PENDING |
| 11728585 | top_uncapped | PENDING |

---

## ANTIBODY DMS (5 × 4 = 20 jobs)

### 19. flab_binding_koenig — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728586 | general_none | PENDING |
| 11728587 | general_uncapped | PENDING |
| 11728588 | top_none | PENDING |
| 11728589 | top_uncapped | PENDING |

### 20. flab_expression_koenig — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728590 | general_none | PENDING |
| 11728591 | general_uncapped | PENDING |
| 11728592 | top_none | PENDING |
| 11728593 | top_uncapped | PENDING |

### 21. flab_binding_adams — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728594 | general_none | PENDING |
| 11728595 | general_uncapped | PENDING |
| 11728596 | top_none | PENDING |
| 11728597 | top_uncapped | PENDING |

### 22. flab_expression_adams — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728598 | general_none | PENDING |
| 11728599 | general_uncapped | PENDING |
| 11728600 | top_none | PENDING |
| 11728601 | top_uncapped | PENDING |

### 23. flab_binding_phillips_h3 — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728602 | general_none | PENDING |
| 11728603 | general_uncapped | PENDING |
| 11728604 | top_none | PENDING |
| 11728605 | top_uncapped | PENDING |

---

## ABAGYM (5 × 4 = 20 jobs)

### 24. abagym_Ang2 — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728606 | general_none | PENDING |
| 11728607 | general_uncapped | PENDING |
| 11728608 | top_none | PENDING |
| 11728609 | top_uncapped | PENDING |

### 25. abagym_EGFR — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728610 | general_none | PENDING |
| 11728611 | general_uncapped | PENDING |
| 11728612 | top_none | PENDING |
| 11728613 | top_uncapped | PENDING |

### 26. abagym_HER2 — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728614 | general_none | PENDING |
| 11728615 | general_uncapped | PENDING |
| 11728616 | top_none | PENDING |
| 11728617 | top_uncapped | PENDING |

### 27. abagym_lysozyme — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728618 | general_none | PENDING |
| 11728619 | general_uncapped | PENDING |
| 11728620 | top_none | PENDING |
| 11728621 | top_uncapped | PENDING |

### 28. abagym_VEGF — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728622 | general_none | PENDING |
| 11728624 | general_uncapped | PENDING |
| 11728626 | top_none | PENDING |
| 11728627 | top_uncapped | PENDING |

---

## ANTIBODY CLF (2 × 4 = 8 jobs)

### 29. tz_her2_binary — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728628 | general_none | PENDING |
| 11728629 | general_uncapped | PENDING |
| 11728630 | top_none | PENDING |
| 11728631 | top_uncapped | PENDING |

### 30. humanness_therapeutic — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728632 | general_none | PENDING |
| 11728633 | general_uncapped | PENDING |
| 11728634 | top_none | PENDING |
| 11728635 | top_uncapped | PENDING |

---

## GINKGO (6 × 4 = 24 jobs)

### 31. ginkgo_acsins — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728636 | general_none | PENDING |
| 11728645 | general_uncapped | PENDING |
| 11728646 | top_none | PENDING |
| 11728647 | top_uncapped | PENDING |

### 32. ginkgo_colloidal — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728648 | general_none | PENDING |
| 11728649 | general_uncapped | PENDING |
| 11728650 | top_none | PENDING |
| 11728651 | top_uncapped | PENDING |

### 33. ginkgo_heparin — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728652 | general_none | PENDING |
| 11728653 | general_uncapped | PENDING |
| 11728654 | top_none | PENDING |
| 11728655 | top_uncapped | PENDING |

### 34. ginkgo_hic — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728656 | general_none | PENDING |
| 11728657 | general_uncapped | PENDING |
| 11728658 | top_none | PENDING |
| 11728659 | top_uncapped | PENDING |

### 35. ginkgo_pr_cho — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728660 | general_none | PENDING |
| 11728661 | general_uncapped | PENDING |
| 11728662 | top_none | PENDING |
| 11728663 | top_uncapped | PENDING |

### 36. ginkgo_titer — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728664 | general_none | PENDING |
| 11728665 | general_uncapped | PENDING |
| 11728666 | top_none | PENDING |
| 11728667 | top_uncapped | PENDING |

---

## TCGA (2 × 4 = 8 jobs)

### 37. tcga_luad_vs_lusc — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728668 | general_none | PENDING |
| 11728669 | general_uncapped | PENDING |
| 11728670 | top_none | PENDING |
| 11728671 | top_uncapped | PENDING |

### 38. tcga_brca_pam50 — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728672 | general_none | PENDING |
| 11728673 | general_uncapped | PENDING |
| 11728674 | top_none | PENDING |
| 11728675 | top_uncapped | PENDING |

---

## REACTION (2 × 4 = 8 jobs)

### 39. buchwald_hartwig — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728676 | general_none | PENDING |
| 11728677 | general_uncapped | PENDING |
| 11728678 | top_none | PENDING |
| 11728679 | top_uncapped | PENDING |

### 40. bh_doyle_plates — 64G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11728680 | general_none | PENDING |
| 11728681 | general_uncapped | PENDING |
| 11728682 | top_none | PENDING |
| 11728687 | top_uncapped | PENDING |

---

## SUBMISSION HISTORY

| Date | Version | Jobs | Details |
|------|---------|------|---------|
| 2026-04-04 | v1 | 240 (11707673–11707913) | long, 128-256G, fast speed, 6 conditions. **BUG:** `list(set())` non-deterministic scaffold groups. Cancelled. |
| 2026-04-04 | v2 | 23 (11716574–11716596) | Resubmit after group fix. Still fast speed. Cancelled. |
| 2026-04-04 | v3 | 160 (11717057–11717216) | long, 128-256G, full speed, 4 conditions. **STUCK:** 256G too large for queue, 12h+ wait. FT flag on (shouldn't be). Cancelled. |
| 2026-04-05 | **v4** | **160 (11728481–11728687)** | **medium, 64G, full speed, 4 conditions, no FT, cache v9, deterministic groups.** |
