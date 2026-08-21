# KIRBy Dataset Status — canonical roster

**This is the living source of truth for which datasets are in the study, their status, and why.**
It is not a dated log (see `REP_PRUNING_LOG.md` for the chronological pruning record). When a dataset's
status changes, edit the row here.

Status legend:
- **ACTIVE-10** — in the roster, rep pool finalized to 10 complementary reps (`DATASET_REPS[ds]['general']`).
- **ACTIVE-8** — in the roster, pool finalized to <10 (too few complementary reps survived; noted).
- **PENDING-PRUNE** — in the roster, rep-correlation evidence staged, pool not yet cut to 10.
- **PENDING-RERUN** — in the roster, rep_prune evidence failed silently and was resubmitted (multiclass/loader fix, commit 5974a61).
- **PARKED** — kept in `DATASET_REPS` but deliberately not submitted (open investigation).
- **CUT** — removed from the roster; reason given.

Rep-screening metric = Spearman ρ (regression) / AUROC (classification), from the group-CV OOF
`rep_prune` evidence. "best solo" = the single strongest base rep on that yardstick.

---

## Active roster

### Molecular — regression (7)  ·  scaffold-group K-fold  ·  report Spearman ρ + RMSE
| dataset | n | best solo | status |
|---|--:|--:|---|
| caco2 | 2,158 | +0.694 | ACTIVE-10 |
| logd | 5,028 | +0.890 | ACTIVE-10 |
| herg_ki | 1,412 | +0.724 | ACTIVE-10 |
| adme_hlm | 3,086 | +0.669 | ACTIVE-10 |
| adme_rlm | 3,053 | +0.684 | ACTIVE-10 |
| adme_mdr1 | 2,641 | +0.731 | ACTIVE-10 |
| qm9 | 9,978 | +0.957 | ACTIVE-10 |

### Molecular — classification (1)  ·  report AUROC/MCC
| dataset | n | best solo | status |
|---|--:|--:|---|
| herg | 7,772 | — | PENDING-RERUN |

### DTA — drug-target affinity (2 active + 1 parked)  ·  cold-protein split  ·  report CI
| dataset | n | best solo | status |
|---|--:|--:|---|
| kiba | 30,000 | +0.484 | ACTIVE-10 (5 drug + 5 protein) |
| bindingdb_kd | 30,000 | +0.518 | ACTIVE-10 (5 drug + 5 protein) |
| davis | — | — | PARKED (per-drug-mean degeneracy, job 7568428) |

### Reaction (2)  ·  report Spearman ρ + R²
| dataset | n | best solo | status |
|---|--:|--:|---|
| buchwald_hartwig | 3,955 | +0.818 | ACTIVE-10 |
| bh_doyle_plates | 4,312 | +0.791 | ACTIVE-10 |

### Antibody — inter-family / DMS (7)  ·  report Spearman ρ + R² (adams = edit-distance split)
| dataset | n | best solo | status |
|---|--:|--:|---|
| flab_thermo_tresanco | 672 | +0.621 | ACTIVE-10 |
| abcov_kd | 141 | +0.385 | ACTIVE-10 (p≫n — see caveat) |
| ddg_combined | 429 | +0.349 | ACTIVE-8 (weak; 6 reps negative solo, cut all solo<0.1) |
| flab_binding_koenig | 4,275 | +0.438 | ACTIVE-10 |
| flab_expression_koenig | 4,275 | +0.689 | ACTIVE-10 |
| flab_binding_adams | 5,000 | +0.599 (AUROC) | ACTIVE-10 |
| flab_expression_adams | 5,000 | +0.323 | ACTIVE-10 (weak) |

### AbAgym — affinity-maturation DMS (3)  ·  report Spearman ρ
| dataset | n | best solo | status |
|---|--:|--:|---|
| abagym_Ang2 | 397 | +0.573 | ACTIVE-10 |
| abagym_EGFR | 659 | +0.279 | ACTIVE-10 (weak) |
| abagym_VEGF | 418 | +0.491 | ACTIVE-10 |

### Antibody — classification (3 active + 1 rerun)  ·  report AUROC / F1
| dataset | n | best solo | status |
|---|--:|--:|---|
| tz_her2_binary | 5,000 | +0.918 | ACTIVE-10 (saturated) |
| tz_her2_multiclass | 5,000 | — | PENDING-RERUN |
| humanness_therapeutic | 564 | +0.958 | ACTIVE-10 (saturated, p≫n) |
| cdr_flexibility | 328 | +0.894 | ACTIVE-10 (saturated, p≫n) |

### Ginkgo — developability (5)  ·  small-n (p≫n)  ·  report Spearman ρ
| dataset | n | best solo | status |
|---|--:|--:|---|
| ginkgo_acsins | 242 | +0.532 | ACTIVE-10 |
| ginkgo_colloidal | 242 | +0.231 | ACTIVE-10 (weak) |
| ginkgo_hic | 242 | +0.355 | ACTIVE-10 |
| ginkgo_pr_cho | 197 | +0.498 | ACTIVE-10 |
| ginkgo_titer | 239 | +0.345 | ACTIVE-10 (weak) |

### TCGA — expression subtyping (1 active + 1 rerun)  ·  report AUROC / F1
| dataset | n | best solo | status |
|---|--:|--:|---|
| tcga_luad_vs_lusc | 1,017 | +0.981 | ACTIVE-10 (multi-modal omics; kept 1-2 per modality) |
| tcga_brca_pam50 | ~1,000 | — | PENDING-RERUN (multiclass) |

### Nucleotide (6)  ·  chromosome/natural groups  ·  ridge exception ON  ·  report MCC
| dataset | n | best solo | status |
|---|--:|--:|---|
| nt_H3K27me3 | 33,000 | +0.845 | ACTIVE-10 |
| nt_H3K4me3 | 18,244 | +0.903 | ACTIVE-8 (small k-mer pool; greedy left 8) |
| nt_H3K9ac | 24,278 | +0.826 | ACTIVE-10 |
| nt_H3K9me3 | 28,288 | +0.721 | PENDING-PRUNE |
| nt_promoter_tata | 5,274 | +0.913 | PENDING-PRUNE |
| gb_human_enhancers_cohn | 27,791 | +0.821 | PENDING-PRUNE |

---

## Cut datasets (removed from the roster)

| dataset | domain | reason |
|---|---|---|
| ginkgo_tm2 | ginkgo | Near-dead target — best base-rep solo ρ +0.092, most reps negative. No signal under the honest split. (2026-08-20) |
| ginkgo_aggregation | ginkgo | Near-dead target — best solo +0.179, most reps negative. (2026-08-20) |
| ginkgo_purity | ginkgo | Near-dead target — best solo +0.151, most reps near zero. (2026-08-20) |
| flab_binding_phillips_h3 | antibody | Single-WT combinatorial library, no clean grouped split, saturated (aac AUROC 0.98 while leaking). (2026-07-18) |
| flab_immunogenicity_marks | antibody | User call (2026-07-18). |
| nucleotide (15 of 21) | nucleotide | Curated 21→6 on weak signal / coverage (2026-07-18). |

**Cut criterion (user):** cut ONLY when the numbers are genuinely bad — no meaningful signal even under
an honest/near-random split. A broken split is NOT grounds to cut (fix the split, re-judge).

---

## Standing caveats

- **p≫n (small-n antibody/ginkgo).** abcov_kd (n=141), abagym, ginkgo, ddg, flab_thermo use high-dim
  frozen protein embeddings (1024–5600 dim) on n=141–672 — up to 40× the sample count. Base-rep SOLO
  scores here are raw full-dim fits in the p≫n regime, so they are noisier than the molecular ones.
  Reps stay usable because the hybrid feature-selects (compresses) them; the fragile part is the raw
  base-rep *baseline*, which needs the same feature budget for an honest best-rep-vs-hybrid contest.
- **top_tier retired.** Every finalized pool sets `top_tier == general` (one 10-rep pool per dataset).
- **Cache.** Reps are not in the cache key, so finalizing a pool invalidates that dataset's
  search/baseline caches — wipe + re-score before trusting new hybrid numbers.

**Roster size:** 40 dataset keys in `DATASET_REPS` (incl. parked davis). Rep pools finalized to 10 on
26 datasets so far; remaining are PENDING-PRUNE (ab_clf, tcga_luad, nucleotide) or PENDING-RERUN.
