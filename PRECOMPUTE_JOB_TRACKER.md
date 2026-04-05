# Precompute Hybrid Search — Job ID Tracker

**Submitted:** 2026-04-05 (v5 — KIRBY_SKIP_INSTALLS, medium, 64G, no FT, full speed)
**Script:** `tests/slurm_scripts/submit_precompute.sh` (all domains)
**Total:** 40 datasets × 4 conditions = 160 SLURM jobs (IDs 11729100–11729261)
**Partition:** medium (48h max) | **Memory:** 64G | **CPUs:** 4 | **Time:** 12h | **Speed:** full (5-fold, 13 strategies)
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
sacct -j 11729100-11729261 --format=State --noheader | grep -v "\.ba\|\.ex" | sort | uniq -c
sacct -j 11729100-11729261 --format=JobID,JobName%45,State,Elapsed,ExitCode | grep -v "\.ba\|\.ex" | grep -E "FAILED|TIMEOUT|OUT_OF_ME"
```

---

## MOLECULAR (7 × 4 = 28 jobs)

### 1. herg_ki
| Job ID | Condition |
|--------|-----------|
| 11729100 | general_none |
| 11729101 | general_uncapped |
| 11729102 | top_none |
| 11729103 | top_uncapped |

### 2. caco2
| Job ID | Condition |
|--------|-----------|
| 11729104 | general_none |
| 11729105 | general_uncapped |
| 11729106 | top_none |
| 11729107 | top_uncapped |

### 3. logd
| Job ID | Condition |
|--------|-----------|
| 11729108 | general_none |
| 11729109 | general_uncapped |
| 11729110 | top_none |
| 11729111 | top_uncapped |

### 4. qm9
| Job ID | Condition |
|--------|-----------|
| 11729112 | general_none |
| 11729113 | general_uncapped |
| 11729114 | top_none |
| 11729115 | top_uncapped |

### 5. adme_hlm
| Job ID | Condition |
|--------|-----------|
| 11729116 | general_none |
| 11729117 | general_uncapped |
| 11729118 | top_none |
| 11729119 | top_uncapped |

### 6. adme_rlm
| Job ID | Condition |
|--------|-----------|
| 11729120 | general_none |
| 11729121 | general_uncapped |
| 11729122 | top_none |
| 11729123 | top_uncapped |

### 7. adme_mdr1
| Job ID | Condition |
|--------|-----------|
| 11729124 | general_none |
| 11729125 | general_uncapped |
| 11729126 | top_none |
| 11729127 | top_uncapped |

---

## ANTIBODY INTER-FAMILY (11 × 4 = 44 jobs)

### 8. flab_thermo_tresanco
| Job ID | Condition |
|--------|-----------|
| 11729128 | general_none |
| 11729129 | general_uncapped |
| 11729130 | top_none |
| 11729132 | top_uncapped |

### 9. flab_pk_jain_heprt
| Job ID | Condition |
|--------|-----------|
| 11729133 | general_none |
| 11729134 | general_uncapped |
| 11729135 | top_none |
| 11729136 | top_uncapped |

### 10. flab_pk_jain_fcrnrt
| Job ID | Condition |
|--------|-----------|
| 11729137 | general_none |
| 11729138 | general_uncapped |
| 11729139 | top_none |
| 11729140 | top_uncapped |

### 11. flab_polyreact_jain_cic
| Job ID | Condition |
|--------|-----------|
| 11729141 | general_none |
| 11729142 | general_uncapped |
| 11729143 | top_none |
| 11729144 | top_uncapped |

### 12. flab_aggregation_jain_hicrt
| Job ID | Condition |
|--------|-----------|
| 11729145 | general_none |
| 11729146 | general_uncapped |
| 11729147 | top_none |
| 11729148 | top_uncapped |

### 13. flab_thermo_jain
| Job ID | Condition |
|--------|-----------|
| 11729149 | general_none |
| 11729150 | general_uncapped |
| 11729151 | top_none |
| 11729152 | top_uncapped |

### 14. abcov_kd
| Job ID | Condition |
|--------|-----------|
| 11729153 | general_none |
| 11729154 | general_uncapped |
| 11729155 | top_none |
| 11729156 | top_uncapped |

### 15. flab_aggregation_jain
| Job ID | Condition |
|--------|-----------|
| 11729157 | general_none |
| 11729158 | general_uncapped |
| 11729159 | top_none |
| 11729160 | top_uncapped |

### 16. flab_immunogenicity_marks
| Job ID | Condition |
|--------|-----------|
| 11729161 | general_none |
| 11729162 | general_uncapped |
| 11729163 | top_none |
| 11729164 | top_uncapped |

### 17. flab_polyreact_jain_bvp
| Job ID | Condition |
|--------|-----------|
| 11729165 | general_none |
| 11729166 | general_uncapped |
| 11729167 | top_none |
| 11729168 | top_uncapped |

### 18. ddg_combined
| Job ID | Condition |
|--------|-----------|
| 11729169 | general_none |
| 11729170 | general_uncapped |
| 11729171 | top_none |
| 11729172 | top_uncapped |

---

## ANTIBODY DMS (5 × 4 = 20 jobs)

### 19. flab_binding_koenig
| Job ID | Condition |
|--------|-----------|
| 11729173 | general_none |
| 11729174 | general_uncapped |
| 11729175 | top_none |
| 11729176 | top_uncapped |

### 20. flab_expression_koenig
| Job ID | Condition |
|--------|-----------|
| 11729177 | general_none |
| 11729178 | general_uncapped |
| 11729179 | top_none |
| 11729180 | top_uncapped |

### 21. flab_binding_adams
| Job ID | Condition |
|--------|-----------|
| 11729181 | general_none |
| 11729182 | general_uncapped |
| 11729183 | top_none |
| 11729184 | top_uncapped |

### 22. flab_expression_adams
| Job ID | Condition |
|--------|-----------|
| 11729185 | general_none |
| 11729186 | general_uncapped |
| 11729187 | top_none |
| 11729188 | top_uncapped |

### 23. flab_binding_phillips_h3
| Job ID | Condition |
|--------|-----------|
| 11729189 | general_none |
| 11729190 | general_uncapped |
| 11729191 | top_none |
| 11729192 | top_uncapped |

---

## ABAGYM (5 × 4 = 20 jobs)

### 24. abagym_Ang2
| Job ID | Condition |
|--------|-----------|
| 11729193 | general_none |
| 11729194 | general_uncapped |
| 11729195 | top_none |
| 11729196 | top_uncapped |

### 25. abagym_EGFR
| Job ID | Condition |
|--------|-----------|
| 11729197 | general_none |
| 11729198 | general_uncapped |
| 11729199 | top_none |
| 11729200 | top_uncapped |

### 26. abagym_HER2
| Job ID | Condition |
|--------|-----------|
| 11729201 | general_none |
| 11729202 | general_uncapped |
| 11729203 | top_none |
| 11729204 | top_uncapped |

### 27. abagym_lysozyme
| Job ID | Condition |
|--------|-----------|
| 11729205 | general_none |
| 11729206 | general_uncapped |
| 11729207 | top_none |
| 11729208 | top_uncapped |

### 28. abagym_VEGF
| Job ID | Condition |
|--------|-----------|
| 11729209 | general_none |
| 11729210 | general_uncapped |
| 11729211 | top_none |
| 11729212 | top_uncapped |

---

## ANTIBODY CLF (2 × 4 = 8 jobs)

### 29. tz_her2_binary
| Job ID | Condition |
|--------|-----------|
| 11729213 | general_none |
| 11729214 | general_uncapped |
| 11729215 | top_none |
| 11729216 | top_uncapped |

### 30. humanness_therapeutic
| Job ID | Condition |
|--------|-----------|
| 11729217 | general_none |
| 11729218 | general_uncapped |
| 11729219 | top_none |
| 11729220 | top_uncapped |

---

## GINKGO (6 × 4 = 24 jobs)

### 31. ginkgo_acsins
| Job ID | Condition |
|--------|-----------|
| 11729221 | general_none |
| 11729222 | general_uncapped |
| 11729223 | top_none |
| 11729224 | top_uncapped |

### 32. ginkgo_colloidal
| Job ID | Condition |
|--------|-----------|
| 11729225 | general_none |
| 11729226 | general_uncapped |
| 11729227 | top_none |
| 11729228 | top_uncapped |

### 33. ginkgo_heparin
| Job ID | Condition |
|--------|-----------|
| 11729229 | general_none |
| 11729231 | general_uncapped |
| 11729232 | top_none |
| 11729233 | top_uncapped |

### 34. ginkgo_hic
| Job ID | Condition |
|--------|-----------|
| 11729234 | general_none |
| 11729235 | general_uncapped |
| 11729236 | top_none |
| 11729237 | top_uncapped |

### 35. ginkgo_pr_cho
| Job ID | Condition |
|--------|-----------|
| 11729238 | general_none |
| 11729239 | general_uncapped |
| 11729240 | top_none |
| 11729241 | top_uncapped |

### 36. ginkgo_titer
| Job ID | Condition |
|--------|-----------|
| 11729242 | general_none |
| 11729243 | general_uncapped |
| 11729244 | top_none |
| 11729245 | top_uncapped |

---

## TCGA (2 × 4 = 8 jobs)

### 37. tcga_luad_vs_lusc
| Job ID | Condition |
|--------|-----------|
| 11729246 | general_none |
| 11729247 | general_uncapped |
| 11729248 | top_none |
| 11729249 | top_uncapped |

### 38. tcga_brca_pam50
| Job ID | Condition |
|--------|-----------|
| 11729250 | general_none |
| 11729251 | general_uncapped |
| 11729252 | top_none |
| 11729253 | top_uncapped |

---

## REACTION (2 × 4 = 8 jobs)

### 39. buchwald_hartwig
| Job ID | Condition |
|--------|-----------|
| 11729254 | general_none |
| 11729255 | general_uncapped |
| 11729256 | top_none |
| 11729257 | top_uncapped |

### 40. bh_doyle_plates
| Job ID | Condition |
|--------|-----------|
| 11729258 | general_none |
| 11729259 | general_uncapped |
| 11729260 | top_none |
| 11729261 | top_uncapped |

---

## SUBMISSION HISTORY

| Date | Version | Jobs | Details |
|------|---------|------|---------|
| 2026-04-04 | v1 | 240 (11707673–11707913) | long, 128-256G, fast speed, 6 conditions. **BUG:** `list(set())` non-deterministic scaffold groups. Cancelled. |
| 2026-04-04 | v2 | 23 (11716574–11716596) | Resubmit after group fix. Still fast speed. Cancelled. |
| 2026-04-04 | v3 | 160 (11717057–11717216) | long, 128-256G, full speed, 4 conditions. **STUCK:** 256G too large for queue. FT flag on. Cancelled. |
| 2026-04-05 | v4 | 160 (11728481–11728687) | medium, 64G, full speed, 4 conditions, no FT. **STUCK:** setup.sh install checks (30+ import checks × 11 jobs on one node). Cancelled. |
| 2026-04-05 | **v5** | **160 (11729100–11729261)** | **medium, 64G, 4 CPUs, full speed, 4 conditions, no FT, KIRBY_SKIP_INSTALLS=1, cache v9.** |
