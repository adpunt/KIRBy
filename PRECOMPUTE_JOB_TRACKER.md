# Precompute Hybrid Search — Job ID Tracker

**Submitted:** 2026-04-04
**Script:** `tests/slurm_scripts/submit_precompute.sh` (all domains)
**Total:** 40 datasets × 6 conditions = 240 SLURM jobs (IDs 11707673–11707913)
**Partition:** long | **Speed tier:** fast

**Source of truth:** `sacct -u scat9264 -S 2026-04-04 --format=JobID,JobName%45,State,Elapsed,ExitCode`

## Conditions (submitted in this order per dataset)

| # | Condition | Reps | Augmentation |
|---|-----------|------|-------------|
| 1 | `general_none` | general | none |
| 2 | `general_uncapped` | general | uncapped |
| 3 | `general_capped` | general | capped |
| 4 | `top_none` | top_tier | none |
| 5 | `top_uncapped` | top_tier | uncapped |
| 6 | `top_capped` | top_tier | capped |

## Quick status check commands

```bash
# Check all precompute jobs
sacct -u scat9264 -S 2026-04-04 --name="pc_%" --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex\|extern\|batch"

# Check one dataset (replace DATASET)
sacct -u scat9264 -S 2026-04-04 --name="pc_DATASET%" --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex"

# Count by state
sacct -u scat9264 -S 2026-04-04 --name="pc_%" --format=State --noheader | grep -v "\.ba\|\.ex" | sort | uniq -c | sort -rn

# Find all FAILED or TIMEOUT jobs
sacct -u scat9264 -S 2026-04-04 --name="pc_%" --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex" | grep -E "FAILED|TIMEOUT|OUT_OF_ME"
```

---

## MOLECULAR (7 datasets × 6 = 42 jobs, IDs 11707673–11707714)

### 1. herg_ki (1,412 samples) — 256G / 12h — RESUBMITTED (v8→v9)

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| **11716574** | general_none | PENDING (resub) | — |
| **11716575** | general_uncapped | PENDING (resub) | — |
| **11716576** | general_capped | PENDING (resub) | — |
| **11716577** | top_none | PENDING (resub) | — |
| **11716578** | top_uncapped | PENDING (resub) | — |
| **11716579** | top_capped | PENDING (resub) | — |

### 2. caco2 (2,158 samples) — 256G / 12h — RESUBMITTED (v8→v9)

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| **11716580** | general_none | PENDING (resub) | — |
| **11716581** | general_uncapped | PENDING (resub) | — |
| **11716582** | general_capped | PENDING (resub) | — |
| **11716583** | top_none | PENDING (resub) | — |
| **11716584** | top_uncapped | PENDING (resub) | — |
| **11716585** | top_capped | PENDING (resub) | — |

### 3. logd (5,028 samples) — 256G / 12h — RESUBMITTED (v8→v9)

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| **11716586** | general_none | PENDING (resub) | — |
| **11716587** | general_uncapped | PENDING (resub) | — |
| **11716588** | general_capped | PENDING (resub) | — |
| **11716589** | top_none | PENDING (resub) | — |
| **11716590** | top_uncapped | PENDING (resub) | — |
| **11716591** | top_capped | PENDING (resub) | — |

### 4. qm9 (9,978 samples) — 256G / 12h — PARTIAL RESUB (v8→v9)

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| **11716592** | general_none | PENDING (resub) | — |
| **11716593** | general_uncapped | PENDING (resub) | — |
| **11716594** | general_capped | PENDING (resub) | — |
| 11707694 | top_none | PENDING (original) | — |
| 11707695 | top_uncapped | PENDING (original) | — |
| 11707696 | top_capped | PENDING (original) | — |

### 5. adme_hlm (3,086 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707697 | general_none | PENDING | — |
| 11707698 | general_uncapped | PENDING | — |
| 11707699 | general_capped | PENDING | — |
| 11707700 | top_none | PENDING | — |
| 11707701 | top_uncapped | PENDING | — |
| 11707702 | top_capped | PENDING | — |

### 6. adme_rlm (3,053 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707703 | general_none | PENDING | — |
| 11707704 | general_uncapped | PENDING | — |
| 11707705 | general_capped | PENDING | — |
| 11707706 | top_none | PENDING | — |
| 11707707 | top_uncapped | PENDING | — |
| 11707708 | top_capped | PENDING | — |

### 7. adme_mdr1 (2,641 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707709 | general_none | PENDING | — |
| 11707710 | general_uncapped | PENDING | — |
| 11707711 | general_capped | PENDING | — |
| 11707712 | top_none | PENDING | — |
| 11707713 | top_uncapped | PENDING | — |
| 11707714 | top_capped | PENDING | — |

---

## ANTIBODY INTER-FAMILY (11 datasets × 6 = 66 jobs, IDs 11707715–11707780)

### 8. flab_thermo_tresanco (672 samples) — 256G / 12h — PARTIAL RESUB (v8→v9)

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| **11716595** | general_none | PENDING (resub) | — |
| **11716596** | general_uncapped | PENDING (resub) | — |
| 11707717 | general_capped | PENDING (original) | — |
| 11707718 | top_none | PENDING (original) | — |
| 11707719 | top_uncapped | PENDING (original) | — |
| 11707720 | top_capped | PENDING (original) | — |

### 9. flab_pk_jain_heprt (130 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707721 | general_none | PENDING | — |
| 11707722 | general_uncapped | PENDING | — |
| 11707723 | general_capped | PENDING | — |
| 11707724 | top_none | PENDING | — |
| 11707725 | top_uncapped | PENDING | — |
| 11707726 | top_capped | PENDING | — |

### 10. flab_pk_jain_fcrnrt (132 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707727 | general_none | PENDING | — |
| 11707728 | general_uncapped | PENDING | — |
| 11707729 | general_capped | PENDING | — |
| 11707730 | top_none | PENDING | — |
| 11707731 | top_uncapped | PENDING | — |
| 11707732 | top_capped | PENDING | — |

### 11. flab_polyreact_jain_cic (137 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707733 | general_none | PENDING | — |
| 11707734 | general_uncapped | PENDING | — |
| 11707735 | general_capped | PENDING | — |
| 11707736 | top_none | PENDING | — |
| 11707737 | top_uncapped | PENDING | — |
| 11707738 | top_capped | PENDING | — |

### 12. flab_aggregation_jain_hicrt (137 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707739 | general_none | PENDING | — |
| 11707740 | general_uncapped | PENDING | — |
| 11707741 | general_capped | PENDING | — |
| 11707742 | top_none | PENDING | — |
| 11707743 | top_uncapped | PENDING | — |
| 11707744 | top_capped | PENDING | — |

### 13. flab_thermo_jain (137 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707745 | general_none | PENDING | — |
| 11707746 | general_uncapped | PENDING | — |
| 11707747 | general_capped | PENDING | — |
| 11707748 | top_none | PENDING | — |
| 11707749 | top_uncapped | PENDING | — |
| 11707750 | top_capped | PENDING | — |

### 14. abcov_kd (141 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707751 | general_none | PENDING | — |
| 11707752 | general_uncapped | PENDING | — |
| 11707753 | general_capped | PENDING | — |
| 11707754 | top_none | PENDING | — |
| 11707755 | top_uncapped | PENDING | — |
| 11707756 | top_capped | PENDING | — |

### 15. flab_aggregation_jain (137 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707757 | general_none | PENDING | — |
| 11707758 | general_uncapped | PENDING | — |
| 11707759 | general_capped | PENDING | — |
| 11707760 | top_none | PENDING | — |
| 11707761 | top_uncapped | PENDING | — |
| 11707762 | top_capped | PENDING | — |

### 16. flab_immunogenicity_marks (217 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707763 | general_none | PENDING | — |
| 11707764 | general_uncapped | PENDING | — |
| 11707765 | general_capped | PENDING | — |
| 11707766 | top_none | PENDING | — |
| 11707767 | top_uncapped | PENDING | — |
| 11707768 | top_capped | PENDING | — |

### 17. flab_polyreact_jain_bvp (137 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707769 | general_none | PENDING | — |
| 11707770 | general_uncapped | PENDING | — |
| 11707771 | general_capped | PENDING | — |
| 11707772 | top_none | PENDING | — |
| 11707773 | top_uncapped | PENDING | — |
| 11707774 | top_capped | PENDING | — |

### 18. ddg_combined (429 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707775 | general_none | PENDING | — |
| 11707776 | general_uncapped | PENDING | — |
| 11707777 | general_capped | PENDING | — |
| 11707778 | top_none | PENDING | — |
| 11707779 | top_uncapped | PENDING | — |
| 11707780 | top_capped | PENDING | — |

---

## ANTIBODY DMS (5 datasets × 6 = 30 jobs, IDs 11707781–11707810)

### 19. flab_binding_koenig (4,275 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707781 | general_none | PENDING | — |
| 11707782 | general_uncapped | PENDING | — |
| 11707783 | general_capped | PENDING | — |
| 11707784 | top_none | PENDING | — |
| 11707785 | top_uncapped | PENDING | — |
| 11707786 | top_capped | PENDING | — |

### 20. flab_expression_koenig (4,275 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707787 | general_none | PENDING | — |
| 11707788 | general_uncapped | PENDING | — |
| 11707789 | general_capped | PENDING | — |
| 11707790 | top_none | PENDING | — |
| 11707791 | top_uncapped | PENDING | — |
| 11707792 | top_capped | PENDING | — |

### 21. flab_binding_adams (5,000 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707793 | general_none | PENDING | — |
| 11707794 | general_uncapped | PENDING | — |
| 11707795 | general_capped | PENDING | — |
| 11707796 | top_none | PENDING | — |
| 11707797 | top_uncapped | PENDING | — |
| 11707798 | top_capped | PENDING | — |

### 22. flab_expression_adams (5,000 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707799 | general_none | PENDING | — |
| 11707800 | general_uncapped | PENDING | — |
| 11707801 | general_capped | PENDING | — |
| 11707802 | top_none | PENDING | — |
| 11707803 | top_uncapped | PENDING | — |
| 11707804 | top_capped | PENDING | — |

### 23. flab_binding_phillips_h3 (5,000 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707805 | general_none | PENDING | — |
| 11707806 | general_uncapped | PENDING | — |
| 11707807 | general_capped | PENDING | — |
| 11707808 | top_none | PENDING | — |
| 11707809 | top_uncapped | PENDING | — |
| 11707810 | top_capped | PENDING | — |

---

## ABAGYM (5 datasets × 6 = 30 jobs, IDs 11707811–11707840)

### 24. abagym_Ang2 (397 samples) — 256G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707811 | general_none | PENDING | — |
| 11707812 | general_uncapped | PENDING | — |
| 11707813 | general_capped | PENDING | — |
| 11707814 | top_none | PENDING | — |
| 11707815 | top_uncapped | PENDING | — |
| 11707816 | top_capped | PENDING | — |

### 25. abagym_EGFR (659 samples) — 256G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707817 | general_none | PENDING | — |
| 11707818 | general_uncapped | PENDING | — |
| 11707819 | general_capped | PENDING | — |
| 11707820 | top_none | PENDING | — |
| 11707821 | top_uncapped | PENDING | — |
| 11707822 | top_capped | PENDING | — |

### 26. abagym_HER2 (146 samples) — 256G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707823 | general_none | PENDING | — |
| 11707824 | general_uncapped | PENDING | — |
| 11707825 | general_capped | PENDING | — |
| 11707826 | top_none | PENDING | — |
| 11707827 | top_uncapped | PENDING | — |
| 11707828 | top_capped | PENDING | — |

### 27. abagym_lysozyme (627 samples) — 256G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707829 | general_none | PENDING | — |
| 11707830 | general_uncapped | PENDING | — |
| 11707831 | general_capped | PENDING | — |
| 11707832 | top_none | PENDING | — |
| 11707833 | top_uncapped | PENDING | — |
| 11707834 | top_capped | PENDING | — |

### 28. abagym_VEGF (418 samples) — 256G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707835 | general_none | PENDING | — |
| 11707836 | general_uncapped | PENDING | — |
| 11707837 | general_capped | PENDING | — |
| 11707838 | top_none | PENDING | — |
| 11707839 | top_uncapped | PENDING | — |
| 11707840 | top_capped | PENDING | — |

---

## ANTIBODY CLF + GINKGO (8 datasets × 6 = 48 jobs, IDs 11707841–11707888)

### 29. tz_her2_binary (5,000 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707841 | general_none | PENDING | — |
| 11707842 | general_uncapped | PENDING | — |
| 11707843 | general_capped | PENDING | — |
| 11707844 | top_none | PENDING | — |
| 11707845 | top_uncapped | PENDING | — |
| 11707846 | top_capped | PENDING | — |

### 30. humanness_therapeutic (564 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707847 | general_none | PENDING | — |
| 11707848 | general_uncapped | PENDING | — |
| 11707849 | general_capped | PENDING | — |
| 11707850 | top_none | PENDING | — |
| 11707851 | top_uncapped | PENDING | — |
| 11707852 | top_capped | PENDING | — |

### 31. ginkgo_acsins (242 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707853 | general_none | PENDING | — |
| 11707854 | general_uncapped | PENDING | — |
| 11707855 | general_capped | PENDING | — |
| 11707856 | top_none | PENDING | — |
| 11707857 | top_uncapped | PENDING | — |
| 11707858 | top_capped | PENDING | — |

### 32. ginkgo_colloidal (242 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707859 | general_none | PENDING | — |
| 11707860 | general_uncapped | PENDING | — |
| 11707861 | general_capped | PENDING | — |
| 11707862 | top_none | PENDING | — |
| 11707863 | top_uncapped | PENDING | — |
| 11707864 | top_capped | PENDING | — |

### 33. ginkgo_heparin (94 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707865 | general_none | PENDING | — |
| 11707866 | general_uncapped | PENDING | — |
| 11707867 | general_capped | PENDING | — |
| 11707868 | top_none | PENDING | — |
| 11707869 | top_uncapped | PENDING | — |
| 11707870 | top_capped | PENDING | — |

### 34. ginkgo_hic (242 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707871 | general_none | PENDING | — |
| 11707872 | general_uncapped | PENDING | — |
| 11707873 | general_capped | PENDING | — |
| 11707874 | top_none | PENDING | — |
| 11707875 | top_uncapped | PENDING | — |
| 11707876 | top_capped | PENDING | — |

### 35. ginkgo_pr_cho (197 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707877 | general_none | PENDING | — |
| 11707878 | general_uncapped | PENDING | — |
| 11707879 | general_capped | PENDING | — |
| 11707880 | top_none | PENDING | — |
| 11707881 | top_uncapped | PENDING | — |
| 11707882 | top_capped | PENDING | — |

### 36. ginkgo_titer (239 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707883 | general_none | PENDING | — |
| 11707884 | general_uncapped | PENDING | — |
| 11707885 | general_capped | PENDING | — |
| 11707886 | top_none | PENDING | — |
| 11707887 | top_uncapped | PENDING | — |
| 11707888 | top_capped | PENDING | — |

---

## TCGA (2 datasets × 6 = 12 jobs, IDs 11707889–11707900)

### 37. tcga_luad_vs_lusc (1,017 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707889 | general_none | PENDING | — |
| 11707890 | general_uncapped | PENDING | — |
| 11707891 | general_capped | PENDING | — |
| 11707892 | top_none | PENDING | — |
| 11707893 | top_uncapped | PENDING | — |
| 11707894 | top_capped | PENDING | — |

### 38. tcga_brca_pam50 (821 samples) — 256G / 12h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707895 | general_none | PENDING | — |
| 11707896 | general_uncapped | PENDING | — |
| 11707897 | general_capped | PENDING | — |
| 11707898 | top_none | PENDING | — |
| 11707899 | top_uncapped | PENDING | — |
| 11707900 | top_capped | PENDING | — |

---

## REACTION (2 datasets × 6 = 12 jobs, IDs 11707901–11707913)

### 39. buchwald_hartwig (3,955 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707901 | general_none | PENDING | — |
| 11707902 | general_uncapped | PENDING | — |
| 11707904 | general_capped | PENDING | — |
| 11707905 | top_none | PENDING | — |
| 11707906 | top_uncapped | PENDING | — |
| 11707907 | top_capped | PENDING | — |

*Note: ID 11707903 allocated to another user between submissions.*

### 40. bh_doyle_plates (4,312 samples) — 128G / 8h

| Job ID | Condition | Status | Elapsed |
|--------|-----------|--------|---------|
| 11707908 | general_none | PENDING | — |
| 11707909 | general_uncapped | PENDING | — |
| 11707910 | general_capped | PENDING | — |
| 11707911 | top_none | PENDING | — |
| 11707912 | top_uncapped | PENDING | — |
| 11707913 | top_capped | PENDING | — |

---

## RUNTIME PATTERNS (from completed jobs)

Early completions reveal a clear pattern — use this to estimate timeout risk:

| Condition type | Typical runtime (molecular) | Typical runtime (antibody) |
|---------------|---------------------------|---------------------------|
| `_none` | 10–42 min | ~7h (tresanco, 672 samples) |
| `_uncapped` | 20 min – 3.5h | ~8h (tresanco) |
| `_capped` | 4–6h | likely 8–12h |
| `top_none` | 10–38 min | — |
| `top_uncapped` | 20 min – 2.7h | — |
| `top_capped` | 2.5–4h | — |

**Key takeaway:** `*_capped` conditions are by far the slowest (6h for herg_ki with only 15 general reps). Datasets with many reps + large N + capped augmentation are the timeout risk.

## TIMEOUT RISK (datasets most likely to hit time limit)

| Risk | Dataset | N | General reps | Time limit | Why |
|------|---------|---|-------------|-----------|-----|
| HIGH | flab_binding_adams | 5,000 | 26 | 12h | Large N + many reps |
| HIGH | flab_expression_adams | 5,000 | 28 | 12h | Large N + many reps |
| HIGH | flab_binding_phillips_h3 | 5,000 | 29 | 12h | Large N + most reps |
| HIGH | tz_her2_binary | 5,000 | 29 | 8h | Large N + many reps + SHORT limit |
| HIGH | flab_binding_koenig | 4,275 | 26 | 12h | Large N + many reps |
| HIGH | flab_expression_koenig | 4,275 | 27 | 12h | Large N + many reps |
| MED | qm9 | 9,978 | 16 | 12h | Largest N, fewer reps |
| MED | bh_doyle_plates | 4,312 | ? | 8h | Large N + short limit |
| MED | buchwald_hartwig | 3,955 | ? | 8h | Large N + short limit |

---

## RESUBMISSION LOG

*When a job fails/times out, record the resubmission here.*

**Resubmit command template:**
```bash
sbatch --job-name="pc_<DATASET>_<CONDITION>" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --time=<NEW_TIME> --partition=long \
    --mem=<MEM> \
    --mail-user="adelaide.punt@stcatz.ox.ac.uk" --mail-type=FAIL \
    --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && python tests/precompute_hybrids.py <DATASET> --condition <CONDITION> --speed fast"
```

| Date | Dataset | Condition | Old Job ID | Old State | New Job ID | New Time Limit |
|------|---------|-----------|-----------|-----------|-----------|---------------|
| 2026-04-04 | herg_ki | general_none | 11707673 | COMPLETED (stale v8) | 11716574 | 12:00:00 |
| 2026-04-04 | herg_ki | general_uncapped | 11707674 | COMPLETED (stale v8) | 11716575 | 12:00:00 |
| 2026-04-04 | herg_ki | general_capped | 11707675 | COMPLETED (stale v8) | 11716576 | 12:00:00 |
| 2026-04-04 | herg_ki | top_none | 11707676 | COMPLETED (stale v8) | 11716577 | 12:00:00 |
| 2026-04-04 | herg_ki | top_uncapped | 11707677 | COMPLETED (stale v8) | 11716578 | 12:00:00 |
| 2026-04-04 | herg_ki | top_capped | 11707678 | COMPLETED (stale v8) | 11716579 | 12:00:00 |
| 2026-04-04 | caco2 | general_none | 11707679 | COMPLETED (stale v8) | 11716580 | 12:00:00 |
| 2026-04-04 | caco2 | general_uncapped | 11707680 | COMPLETED (stale v8) | 11716581 | 12:00:00 |
| 2026-04-04 | caco2 | general_capped | 11707681 | CANCELLED | 11716582 | 12:00:00 |
| 2026-04-04 | caco2 | top_none | 11707682 | COMPLETED (stale v8) | 11716583 | 12:00:00 |
| 2026-04-04 | caco2 | top_uncapped | 11707683 | COMPLETED (stale v8) | 11716584 | 12:00:00 |
| 2026-04-04 | caco2 | top_capped | 11707684 | CANCELLED | 11716585 | 12:00:00 |
| 2026-04-04 | logd | general_none | 11707685 | COMPLETED (stale v8) | 11716586 | 12:00:00 |
| 2026-04-04 | logd | general_uncapped | 11707686 | COMPLETED (stale v8) | 11716587 | 12:00:00 |
| 2026-04-04 | logd | general_capped | 11707687 | CANCELLED | 11716588 | 12:00:00 |
| 2026-04-04 | logd | top_none | 11707688 | COMPLETED (stale v8) | 11716589 | 12:00:00 |
| 2026-04-04 | logd | top_uncapped | 11707689 | COMPLETED (stale v8) | 11716590 | 12:00:00 |
| 2026-04-04 | logd | top_capped | 11707690 | CANCELLED | 11716591 | 12:00:00 |
| 2026-04-04 | qm9 | general_none | 11707691 | COMPLETED (stale v8) | 11716592 | 12:00:00 |
| 2026-04-04 | qm9 | general_uncapped | 11707692 | CANCELLED | 11716593 | 12:00:00 |
| 2026-04-04 | qm9 | general_capped | 11707693 | CANCELLED | 11716594 | 12:00:00 |
| 2026-04-04 | tresanco | general_none | 11707715 | COMPLETED (stale v8) | 11716595 | 12:00:00 |
| 2026-04-04 | tresanco | general_uncapped | 11707716 | COMPLETED (stale v8) | 11716596 | 12:00:00 |

**Reason:** All resubmitted due to non-deterministic scaffold group indices (PYTHONHASHSEED bug). Fix: `list(set())` → `sorted(set())` in `assign_scaffold_groups`. Cache version bumped v8→v9. Stale v8 caches cleared.

---

## FT PRECOMPUTE (12 jobs, HTC long partition, GPU)

**Submitted:** 2026-04-04
**Cluster:** HTC (GPU)
**Script:** `tests/slurm_scripts/submit_precompute_ft.sh`
**Total:** 12 jobs (7 molecular + 5 antibody) — IDs 7440351–7440362
**Resources:** 1 GPU, 128G, 12h, long partition
**Speeds:** both fast (3-fold) and full (5-fold) per job

**Status check:**
```bash
sacct -u scat9264 -S 2026-04-04 --name="ft_%" --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex\|extern\|batch"
```

### Molecular FT (7 jobs)

| Job ID | Name | Dataset | FT Rep | Status |
|--------|------|---------|--------|--------|
| 7440351 | ft_logd_lora_mf | logd | lora_molformer | PENDING |
| 7440352 | ft_logd_contr_mf | logd | contrastive_molformer | PENDING |
| 7440353 | ft_logd_gnn_gmvp | logd | gnn_graphmvp | PENDING |
| 7440354 | ft_caco2_lora_mf | caco2 | lora_molformer | PENDING |
| 7440355 | ft_qm9_lora_mf | qm9 | lora_molformer | PENDING |
| 7440356 | ft_herg_gnn_gmvp | herg_ki | gnn_graphmvp | PENDING |
| 7440357 | ft_mdr1_gnn_gmvp | adme_mdr1 | gnn_graphmvp | PENDING |

### Antibody FT (5 jobs)

| Job ID | Name | Dataset | FT Rep | Status |
|--------|------|---------|--------|--------|
| 7440358 | ft_phil_h3_lora | flab_binding_phillips_h3 | antibody_lora_esm2 | PENDING |
| 7440359 | ft_ang2_lora | abagym_Ang2 | antibody_lora_esm2 | PENDING |
| 7440360 | ft_egfr_lora | abagym_EGFR | antibody_lora_esm2 | PENDING |
| 7440361 | ft_her2_lora | abagym_HER2 | antibody_lora_esm2 | PENDING |
| 7440362 | ft_abcov_lora | abcov_kd | antibody_lora_esm2 | PENDING |
