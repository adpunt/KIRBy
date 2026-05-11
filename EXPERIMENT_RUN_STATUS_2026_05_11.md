# Experiment Run Status — 2026-05-11

Snapshot of the complexity / similarity / OPIG / exp_%j wave submitted ~2026-05-08.
Source: `sacct -u $USER -S 2026-05-08` (server) + local JSONs from previous wave (2026-04-29).

---

## Complexity (`cx_*`) — 65 jobs

### ✅ COMPLETED (13)
| Job | Dataset | Elapsed |
|---|---|---|
| 12103583 | herg_ki | 15:17 |
| 12103596 | flab_polyreact_jain_cic | 2d 14:55 |
| 12103601 | flab_immunogenicity_marks | 2d 10:30 |
| 12103602 | flab_polyreact_jain_bvp | 2d 10:27 |
| 12103603 | ddg_combined | 07:26 |
| 12103604 | flab_binding_koenig | 06:55 |
| 12103605 | flab_expression_koenig | 13:35 |
| 12103606 | flab_binding_adams | 14:04 |
| 12103607 | flab_expression_adams | 09:28 |
| 12103608 | flab_binding_phillips_h3 | 09:28 |
| 12103611 | abagym_HER2 | 1d 12:22 |
| 12103612 | abagym_lysozyme | 1d 14:07 |
| 12103613 | abagym_VEGF | 06:45 |

### 🟡 RUNNING (26) — at ~2d 18h, wall=7d
caco2, logd, qm9, herg, kiba, bindingdb_kd, adme_hlm, adme_rlm, adme_mdr1,
flab_thermo_tresanco, flab_pk_jain_heprt, flab_pk_jain_fcrnrt, flab_aggregation_jain_hicrt,
flab_thermo_jain, flab_aggregation_jain, abcov_kd, abagym_Ang2, abagym_EGFR,
ginkgo_acsins, ginkgo_colloidal, ginkgo_heparin, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer,
buchwald_hartwig, bh_doyle_plates

### 🔴 CANCELLED (26) — all at ~05:32 elapsed, manual `scancel` pattern
**Antibody clf (3):** tz_her2_binary, tz_her2_multiclass, humanness_therapeutic
**TCGA (2):** tcga_luad_vs_lusc, tcga_brca_pam50
**Nucleotide (17):** nt_H2AFZ, nt_H3K27ac, nt_H3K27me3, nt_H3K36me3, nt_H3K4me1,
nt_H3K4me2, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_H4K20me1, nt_enhancers,
nt_enhancers_types, nt_promoter_all, nt_promoter_no_tata, nt_promoter_tata,
nt_splice_sites_acceptors, nt_splice_sites_donors
**Genomic Benchmarks (4):** gb_demo_coding_vs_intergenomic_seqs, gb_demo_human_or_worm,
gb_human_enhancers_cohn, gb_human_nontata_promoters

**OPEN QUESTION:** Were these scancelled intentionally? If not, the whole TCGA + nucleotide + 3 ab_clf complexity sweep is missing from this wave.

---

## Similarity (`sim_*`) — 65 jobs

### ✅ COMPLETED (51) — 78% done
All molecular (herg_ki, caco2, logd, qm9), all DTA (kiba, bindingdb_kd), all 3 ADME,
herg clf, all 17 nucleotide, all 4 gb_*, both TCGA (luad_vs_lusc, brca_pam50),
5/6 ginkgo (acsins, heparin, hic, titer + buchwald), both reactions (buchwald done,
bh_doyle still running), abagym Ang2/EGFR/HER2/VEGF, most flab_*,
flab_thermo_tresanco, flab_polyreact_jain_cic, flab_aggregation_jain_hicrt,
flab_thermo_jain, abcov_kd, flab_immunogenicity_marks, flab_polyreact_jain_bvp,
flab_binding_koenig, flab_binding_adams, flab_expression_adams.

### 🟡 RUNNING (11) — at ~2d 13-15h
flab_pk_jain_heprt, flab_pk_jain_fcrnrt, flab_aggregation_jain,
flab_expression_koenig, flab_binding_phillips_h3, ddg_combined,
abagym_lysozyme, ginkgo_colloidal, ginkgo_pr_cho, bh_doyle_plates,
nt_promoter_tata

### 🔴 CANCELLED (3) — at ~00:16-00:20 elapsed
tz_her2_binary, tz_her2_multiclass, humanness_therapeutic (same trio as complexity).

---

## OPIG (`opig_*`) — 6 jobs, ALL FAILED

| Job | Name | State | Elapsed | Notes |
|---|---|---|---|---|
| 12103714 | opig_humanness | FAILED 1:0 | 00:00:04 | instant crash |
| 12103715 | opig_trastuzumab | FAILED 1:0 | 00:00:04 | instant crash |
| 12103716 | opig_ginkgo | TIMEOUT | 23:59:36 | hit 24h wall |
| 12103717 | opig_graphinity | FAILED 1:0 | 00:00:04 | instant crash |
| 12103718 | opig_qsar | FAILED 0:0 | 00:00:05 | weird exit-0 fail |
| 12103719 | opig_blueprint_transfer | TIMEOUT | 23:59:21 | hit 24h wall |

**ZERO OPIG output from this wave.**

Action: inspect `slurm-1210371{4,5,7,8}.out` for the import/arg error; bump wall on 716/719.

---

## `exp_%j` jobs — UNIDENTIFIED CATEGORY (~50+ jobs)

Job IDs 12111839–12115452+. Name was never overridden via `submit_all_datasets.sh`,
so these were submitted directly via `sbatch submit_dataset.sh CATEGORY DATASET`
(which leaves `--job-name=exp_%j` as default).

- 11 COMPLETED so far (1.5h – 17h elapsed): 12111861, 12111862, 12111866, 12111867,
  12115429, 12115430, 12115432, 12115433, 12115446, 12115447, 12115448
- ~40+ still RUNNING at 2d 03-11h

**Action:** identify with
```
scontrol show job 12111839 | grep -E "Command|StdOut|WorkDir"
```

---

## Previous wave (local JSONs, 2026-04-29) — bugs still on disk

These were known issues in the run that fed the local cache. Whether the current
wave fixes them depends on which commit the server pulled before re-submission.

| Experiment | Issue | Affected |
|---|---|---|
| `c10_noise_robustness` | `noiseInject not installed` | ALL 9 datasets |
| `s04_activity_cliffs` | `ecfp4 not available` (sent to non-molecular) | 13 datasets |
| `s12_cross_domain_cliffs` | `no_cliffs_found` (threshold?) | 14 datasets |
| `s03/s06/s07` | `aug=uncapped` cache miss | tcga_luad_vs_lusc |
| `s03/s06/s07` | `y_true/y_pred shape 1!=4` (multiclass clf bug) | tcga_brca_pam50 |
| `opig/covariate_shift/` | empty — never produced output | — |

---

## Other (peripheral)
- `annitia_short` TIMEOUT; `annitia_medium`, `annitia_long` COMPLETED
- `kirby_av_rsync` TIMEOUT x2
- `uavm_visualise` FAILED
- `val_svm_rerun` FAILED x2

---

## Action items (priority order)

1. **Inspect OPIG instant-crashes:** `head -50 slurm-1210371{4,5,7,8}.out` on server. All
   4 died in <5s → likely a single shared import/path error.
2. **Confirm the 26 cancellations were intentional.** If not, this is a regression —
   resubmit TCGA + nucleotide + ab_clf complexity.
3. **Identify `exp_%j` jobs** via `scontrol show job 12111839`.
4. **Bump OPIG wall** for `opig_ginkgo` (716) and `opig_blueprint_transfer` (719) to
   ≥48h before re-submitting.
5. After running jobs land, **rsync JSONs down** and re-audit:
   ```
   rsync -avz --include='*/' --include='*.json' --exclude='*' \
     gateway:/data/stat-ecr/scat9264/KIRBy/tests/results/ \
     /Users/apunt/repos/KIRBy/tests/results_server/
   ```
