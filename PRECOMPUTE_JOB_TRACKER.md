# Precompute Hybrid Search — Job ID Tracker

**Submitted:** 2026-04-04 (v3 — deterministic groups, full speed, no capped)
**Script:** `tests/slurm_scripts/submit_precompute.sh` (all domains)
**Total:** 40 datasets × 4 conditions = 160 SLURM jobs (IDs 11717057–11717216)
**Partition:** long | **Speed tier:** full (5-fold, 13 strategies)
**Cache version:** v9 (deterministic scaffold groups)

**Previous submissions cancelled:** v1 (IDs 11707673–11707913, fast speed, stale v8 cache) and v2 (IDs 11716574–11716596, fast speed resubmit). Both had non-deterministic scaffold group indices (`list(set())` bug) and used fast speed (3-fold) instead of full (5-fold).

## Conditions (4 per dataset)

| # | Condition | Reps | Augmentation |
|---|-----------|------|-------------|
| 1 | `general_none` | general | none |
| 2 | `general_uncapped` | general | uncapped |
| 3 | `top_none` | top_tier | none |
| 4 | `top_uncapped` | top_tier | uncapped |

Capped conditions removed — `_shared.py` already skips capped in precompute. Experiments needing capped (C02/C05/C06/C07/C09) call `create_hybrid` directly.

## Quick status check commands

```bash
# Check all precompute jobs
sacct -u scat9264 -S 2026-04-04 --name="pc_%" --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex"

# Count by state
sacct -u scat9264 -S 2026-04-04 -j 11717057-11717216 --format=State --noheader | grep -v "\.ba\|\.ex" | sort | uniq -c | sort -rn

# Find FAILED or TIMEOUT
sacct -u scat9264 -S 2026-04-04 -j 11717057-11717216 --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex" | grep -E "FAILED|TIMEOUT|OUT_OF_ME"
```

---

## MOLECULAR (7 datasets × 4 = 28 jobs, IDs 11717057–11717084)

### 1. herg_ki — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717057 | general_none | PENDING |
| 11717058 | general_uncapped | PENDING |
| 11717059 | top_none | PENDING |
| 11717060 | top_uncapped | PENDING |

### 2. caco2 — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717061 | general_none | PENDING |
| 11717062 | general_uncapped | PENDING |
| 11717063 | top_none | PENDING |
| 11717064 | top_uncapped | PENDING |

### 3. logd — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717065 | general_none | PENDING |
| 11717066 | general_uncapped | PENDING |
| 11717067 | top_none | PENDING |
| 11717068 | top_uncapped | PENDING |

### 4. qm9 — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717069 | general_none | PENDING |
| 11717070 | general_uncapped | PENDING |
| 11717071 | top_none | PENDING |
| 11717072 | top_uncapped | PENDING |

### 5. adme_hlm — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717073 | general_none | PENDING |
| 11717074 | general_uncapped | PENDING |
| 11717075 | top_none | PENDING |
| 11717076 | top_uncapped | PENDING |

### 6. adme_rlm — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717077 | general_none | PENDING |
| 11717078 | general_uncapped | PENDING |
| 11717079 | top_none | PENDING |
| 11717080 | top_uncapped | PENDING |

### 7. adme_mdr1 — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717081 | general_none | PENDING |
| 11717082 | general_uncapped | PENDING |
| 11717083 | top_none | PENDING |
| 11717084 | top_uncapped | PENDING |

---

## ANTIBODY INTER-FAMILY (11 datasets × 4 = 44 jobs, IDs 11717085–11717128)

### 8. flab_thermo_tresanco — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717085 | general_none | PENDING |
| 11717086 | general_uncapped | PENDING |
| 11717087 | top_none | PENDING |
| 11717088 | top_uncapped | PENDING |

### 9. flab_pk_jain_heprt — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717089 | general_none | PENDING |
| 11717090 | general_uncapped | PENDING |
| 11717091 | top_none | PENDING |
| 11717092 | top_uncapped | PENDING |

### 10. flab_pk_jain_fcrnrt — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717093 | general_none | PENDING |
| 11717094 | general_uncapped | PENDING |
| 11717095 | top_none | PENDING |
| 11717096 | top_uncapped | PENDING |

### 11. flab_polyreact_jain_cic — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717097 | general_none | PENDING |
| 11717098 | general_uncapped | PENDING |
| 11717099 | top_none | PENDING |
| 11717100 | top_uncapped | PENDING |

### 12. flab_aggregation_jain_hicrt — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717101 | general_none | PENDING |
| 11717102 | general_uncapped | PENDING |
| 11717103 | top_none | PENDING |
| 11717104 | top_uncapped | PENDING |

### 13. flab_thermo_jain — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717105 | general_none | PENDING |
| 11717106 | general_uncapped | PENDING |
| 11717107 | top_none | PENDING |
| 11717108 | top_uncapped | PENDING |

### 14. abcov_kd — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717109 | general_none | PENDING |
| 11717110 | general_uncapped | PENDING |
| 11717111 | top_none | PENDING |
| 11717112 | top_uncapped | PENDING |

### 15. flab_aggregation_jain — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717113 | general_none | PENDING |
| 11717114 | general_uncapped | PENDING |
| 11717115 | top_none | PENDING |
| 11717116 | top_uncapped | PENDING |

### 16. flab_immunogenicity_marks — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717117 | general_none | PENDING |
| 11717118 | general_uncapped | PENDING |
| 11717119 | top_none | PENDING |
| 11717120 | top_uncapped | PENDING |

### 17. flab_polyreact_jain_bvp — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717121 | general_none | PENDING |
| 11717122 | general_uncapped | PENDING |
| 11717123 | top_none | PENDING |
| 11717124 | top_uncapped | PENDING |

### 18. ddg_combined — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717125 | general_none | PENDING |
| 11717126 | general_uncapped | PENDING |
| 11717127 | top_none | PENDING |
| 11717128 | top_uncapped | PENDING |

---

## ANTIBODY DMS (5 datasets × 4 = 20 jobs, IDs 11717129–11717148)

### 19. flab_binding_koenig — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717129 | general_none | PENDING |
| 11717130 | general_uncapped | PENDING |
| 11717131 | top_none | PENDING |
| 11717132 | top_uncapped | PENDING |

### 20. flab_expression_koenig — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717133 | general_none | PENDING |
| 11717134 | general_uncapped | PENDING |
| 11717135 | top_none | PENDING |
| 11717136 | top_uncapped | PENDING |

### 21. flab_binding_adams — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717137 | general_none | PENDING |
| 11717138 | general_uncapped | PENDING |
| 11717139 | top_none | PENDING |
| 11717140 | top_uncapped | PENDING |

### 22. flab_expression_adams — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717141 | general_none | PENDING |
| 11717142 | general_uncapped | PENDING |
| 11717143 | top_none | PENDING |
| 11717144 | top_uncapped | PENDING |

### 23. flab_binding_phillips_h3 — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717145 | general_none | PENDING |
| 11717146 | general_uncapped | PENDING |
| 11717147 | top_none | PENDING |
| 11717148 | top_uncapped | PENDING |

---

## ABAGYM (5 datasets × 4 = 20 jobs, IDs 11717149–11717168)

### 24. abagym_Ang2 — 256G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717149 | general_none | PENDING |
| 11717150 | general_uncapped | PENDING |
| 11717151 | top_none | PENDING |
| 11717152 | top_uncapped | PENDING |

### 25. abagym_EGFR — 256G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717153 | general_none | PENDING |
| 11717154 | general_uncapped | PENDING |
| 11717155 | top_none | PENDING |
| 11717156 | top_uncapped | PENDING |

### 26. abagym_HER2 — 256G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717157 | general_none | PENDING |
| 11717158 | general_uncapped | PENDING |
| 11717159 | top_none | PENDING |
| 11717160 | top_uncapped | PENDING |

### 27. abagym_lysozyme — 256G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717161 | general_none | PENDING |
| 11717162 | general_uncapped | PENDING |
| 11717163 | top_none | PENDING |
| 11717164 | top_uncapped | PENDING |

### 28. abagym_VEGF — 256G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717165 | general_none | PENDING |
| 11717166 | general_uncapped | PENDING |
| 11717167 | top_none | PENDING |
| 11717168 | top_uncapped | PENDING |

---

## ANTIBODY CLF (2 datasets × 4 = 8 jobs, IDs 11717169–11717176)

### 29. tz_her2_binary — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717169 | general_none | PENDING |
| 11717170 | general_uncapped | PENDING |
| 11717171 | top_none | PENDING |
| 11717172 | top_uncapped | PENDING |

### 30. humanness_therapeutic — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717173 | general_none | PENDING |
| 11717174 | general_uncapped | PENDING |
| 11717175 | top_none | PENDING |
| 11717176 | top_uncapped | PENDING |

---

## GINKGO (6 datasets × 4 = 24 jobs, IDs 11717177–11717200)

### 31. ginkgo_acsins — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717177 | general_none | PENDING |
| 11717178 | general_uncapped | PENDING |
| 11717179 | top_none | PENDING |
| 11717180 | top_uncapped | PENDING |

### 32. ginkgo_colloidal — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717181 | general_none | PENDING |
| 11717182 | general_uncapped | PENDING |
| 11717183 | top_none | PENDING |
| 11717184 | top_uncapped | PENDING |

### 33. ginkgo_heparin — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717185 | general_none | PENDING |
| 11717186 | general_uncapped | PENDING |
| 11717187 | top_none | PENDING |
| 11717188 | top_uncapped | PENDING |

### 34. ginkgo_hic — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717189 | general_none | PENDING |
| 11717190 | general_uncapped | PENDING |
| 11717191 | top_none | PENDING |
| 11717192 | top_uncapped | PENDING |

### 35. ginkgo_pr_cho — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717193 | general_none | PENDING |
| 11717194 | general_uncapped | PENDING |
| 11717195 | top_none | PENDING |
| 11717196 | top_uncapped | PENDING |

### 36. ginkgo_titer — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717197 | general_none | PENDING |
| 11717198 | general_uncapped | PENDING |
| 11717199 | top_none | PENDING |
| 11717200 | top_uncapped | PENDING |

---

## TCGA (2 datasets × 4 = 8 jobs, IDs 11717201–11717208)

### 37. tcga_luad_vs_lusc — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717201 | general_none | PENDING |
| 11717202 | general_uncapped | PENDING |
| 11717203 | top_none | PENDING |
| 11717204 | top_uncapped | PENDING |

### 38. tcga_brca_pam50 — 256G / 12h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717205 | general_none | PENDING |
| 11717206 | general_uncapped | PENDING |
| 11717207 | top_none | PENDING |
| 11717208 | top_uncapped | PENDING |

---

## REACTION (2 datasets × 4 = 8 jobs, IDs 11717209–11717216)

### 39. buchwald_hartwig — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717209 | general_none | PENDING |
| 11717210 | general_uncapped | PENDING |
| 11717211 | top_none | PENDING |
| 11717212 | top_uncapped | PENDING |

### 40. bh_doyle_plates — 128G / 8h
| Job ID | Condition | Status |
|--------|-----------|--------|
| 11717213 | general_none | PENDING |
| 11717214 | general_uncapped | PENDING |
| 11717215 | top_none | PENDING |
| 11717216 | top_uncapped | PENDING |

---

## RESUBMISSION LOG

| Date | Event | Details |
|------|-------|---------|
| 2026-04-04 | v1 submitted | 240 jobs (11707673–11707913), fast speed, 6 conditions. **BUG:** `list(set())` in scaffold groups → non-deterministic cache keys |
| 2026-04-04 | v2 resubmit | 23 jobs (11716574–11716596) resubmitted after group fix. Still fast speed. |
| 2026-04-04 | v1+v2 cancelled | All cancelled. Caches cleared. |
| 2026-04-04 | **v3 submitted** | **160 jobs (11717057–11717216)**. Full speed (5-fold), 4 conditions (capped removed), cache v9, deterministic groups. |
