# Per-Dataset Experiment Status — 2026-05-11

Single source of truth across the 2026-05-08 submission and all subsequent
resubmits (Wave 1 = 12103xxx, Wave 2 = 12111xxx, Wave 3 = 12115xxx).

Data sources:
- `sacct -u $USER -S 2026-05-08`
- `python tests/audit_full_wave.py` (log + fresh-file audit; mtime ≥ job start)
- `squeue -u $USER` for in-flight state

## Verdict legend

- **OK** — every expected experiment has a fresh JSON without an `error` key.
- **EFFECTIVELY OK** — `clean_finish` log, 11/12 sim or 19/20 cx fresh, the
  remaining 1 is a self-reported methodological no-data (e.g. s12
  `no_cliffs_found` on antibody, c10 `noiseInject not installed`). Not a
  job failure. Don't resubmit unless you install the missing dep.
- **PARTIAL** — some fresh JSONs, not all; reason given.
- **EMPTY** — zero fresh JSONs from any wave.
- **RUNNING** — a job for this (category, dataset) is in flight; status
  unknown until it finishes.
- **WAVE-3 RUNNING (over W1 RUNNING)** — duplicate parallel jobs. One is
  redundant.

## Top-line stats

- COMPLETED jobs audited: 79
- Verdicts: OK = 4, PARTIAL = 19 (most are "effectively OK"), EMPTY = 56
- Datasets with zero OK across all waves and no resubmit running: **14**
- Datasets with parallel duplicate cx_ jobs running: **25** (Wave 2 + Wave 3)
- Wave 3 jobs that died in <5 minutes (python_error): **3** (gb_demo_human_or_worm, gb_human_enhancers_cohn, gb_human_nontata_promoters — similarity only)

---

## A. Molecular regression (n=6)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| caco2 | W1 12103649 EMPTY (bus); no resubmit | W1 12103584 RUNNING | sim: resubmit. cx: wait for W1. |
| logd | W1 12103650 EMPTY (bus); no resubmit | W1 12103585 RUNNING | sim: resubmit. cx: wait for W1. |
| qm9 | W1 12103651 EMPTY (bus); no resubmit | W1 12103586 RUNNING | sim: resubmit. cx: wait for W1. |
| adme_hlm | W1 12103652 EMPTY (bus); no resubmit | W1 12103587 RUNNING | sim: resubmit. cx: wait for W1. |
| adme_rlm | W1 12103653 EMPTY (bus); no resubmit | W1 12103588 RUNNING | sim: resubmit. cx: wait for W1. |
| adme_mdr1 | W1 12103654 EMPTY (bus); no resubmit | W1 12103589 RUNNING | sim: resubmit. cx: wait for W1. |

## B. Molecular classification (n=2)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| herg | W1 12103655 EMPTY (bus); no resubmit | W1 12103590 RUNNING | sim: resubmit. cx: wait for W1. |
| herg_ki | W1 12103648 EMPTY (bus); W3 12115463 RUNNING | W1 12103583 EMPTY (bus); W3 12115462 RUNNING | Wait for W3. |

## C. DTA (n=2)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| kiba | W1 12103656 EMPTY (bus); W3 12115466 RUNNING | W1 12103591 RUNNING | Wait. |
| bindingdb_kd | W1 12103657 EMPTY (bus); W3 12115434 RUNNING | W1 12103592 RUNNING | Wait. |

## D. Reaction (n=2)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| buchwald_hartwig | W1 12103688 EMPTY (bus); W3 12115435 RUNNING | W1 12103623 RUNNING | Wait. |
| bh_doyle_plates | W1 12103689 RUNNING | W1 12103624 RUNNING | Wait for W1. |

## E. DDG (n=1)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| ddg_combined | W1 12103668 RUNNING | W1 12103603 EMPTY (bus); W3 12115436 RUNNING | Wait. |

## F. Antibody — flab regressions (n=14)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| flab_thermo_tresanco | W1 12103658 EMPTY (bus); W3 12115450 RUNNING | W1 12103593 RUNNING | Wait. |
| flab_thermo_jain | W1 12103663 EMPTY (bus); W3 12115449 RUNNING | W1 12103598 RUNNING | Wait. |
| flab_pk_jain_heprt | W1 12103659 RUNNING | W1 12103594 RUNNING | Wait for W1. |
| flab_pk_jain_fcrnrt | W1 12103660 RUNNING | W1 12103595 RUNNING | Wait for W1. |
| flab_polyreact_jain_cic | W1 12103661 EFFECTIVELY OK (11/12, s12 no-cliffs); W3 12115448 redundant 11/12 | W1 12103596 PARTIAL 17/20 (bus); no resubmit | sim: ✅ done. cx: **resubmit** for missing c23/c24 (c10 needs noiseInject). |
| flab_polyreact_jain_bvp | W1 12103667 EFFECTIVELY OK (11/12); W3 12115447 redundant 11/12 | W1 12103602 PARTIAL 17/20 (bus); no resubmit | sim: ✅ done. cx: **resubmit** for c23/c24. |
| flab_aggregation_jain | W1 12103665 RUNNING | W1 12103600 RUNNING | Wait. |
| flab_aggregation_jain_hicrt | W1 12103662 EMPTY (bus); W3 12115437 RUNNING | W1 12103597 RUNNING | Wait. |
| flab_immunogenicity_marks | W1 12103666 **OK 12/12**; W3 12115446 OK redundant | W1 12103601 PARTIAL 12/20 (bus); no resubmit | sim: ✅ done. cx: **resubmit** for c13+c15-c24. |
| flab_binding_koenig | W1 12103669 EMPTY (bus); W3 12115441 RUNNING | W1 12103604 EMPTY (bus); W3 12115440 RUNNING | Wait. |
| flab_binding_adams | W1 12103671 EMPTY (bus); W3 12115439 RUNNING | W1 12103606 EMPTY (bus); W3 12115438 RUNNING | Wait. |
| flab_binding_phillips_h3 | W1 12103673 RUNNING | W1 12103608 EMPTY (bus); W3 12115442 RUNNING | Wait. |
| flab_expression_koenig | W1 12103670 RUNNING | W1 12103605 EMPTY (bus); W3 12115445 RUNNING | Wait. |
| flab_expression_adams | W1 12103672 EMPTY (bus); W3 12115444 RUNNING | W1 12103607 EMPTY (bus); W3 12115443 RUNNING | Wait. |

## G. Antibody — other (n=1)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| abcov_kd | W1 12103664 EFFECTIVELY OK (11/12); W3 12115433 redundant 11/12 | W1 12103599 RUNNING | sim: ✅ done. cx: wait for W1. |

## H. Abagym (n=5)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| abagym_Ang2 | W1 12103674 EFFECTIVELY OK (finished_then_bus, 11/12) | W1 12103609 RUNNING | sim: ✅ done. cx: wait. |
| abagym_EGFR | W1 12103675 PARTIAL 11/12 (bus_mid); W3 12115429 EFFECTIVELY OK (11/12) | W1 12103610 RUNNING | sim: ✅ done. cx: wait. |
| abagym_HER2 | W1 12103676 EFFECTIVELY OK (11/12); W3 12115430 redundant 11/12 | W1 12103611 PARTIAL 17/20 (bus); no resubmit | sim: ✅ done. cx: **resubmit** for c23/c24. |
| abagym_VEGF | W1 12103678 PARTIAL 11/12 (bus_mid); W3 12115432 EFFECTIVELY OK (11/12) | W1 12103613 PARTIAL 9/20 (bus, short); W3 12115431 RUNNING | sim: ✅ done. cx: wait for W3. |
| abagym_lysozyme | W1 12103677 RUNNING | W1 12103612 EMPTY (bus); **no resubmit** | sim: wait. cx: **resubmit**. |

## I. Antibody classification (n=3)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| tz_her2_binary | W1 cancelled; W2 12111865 RUNNING; W3 12115505 RUNNING (duplicate) | W1 cancelled; W2 12111860 RUNNING; W3 12115504 RUNNING (duplicate) | Wait. **Cancel one duplicate** per pair. |
| tz_her2_multiclass | W1 cancelled; W2 12111866 EMPTY (bus); W3 12115507 RUNNING | W1 cancelled; W2 12111861 EMPTY (bus); W3 12115506 RUNNING | Wait for W3. |
| humanness_therapeutic | W1 cancelled; W2 12111867 **OK 12/12**; W3 12115465 OK redundant | W1 cancelled; W2 12111862 PARTIAL 9/20 (bus); W3 12115464 RUNNING | sim: ✅ done. cx: wait for W3. |

## J. Ginkgo (n=6)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| ginkgo_acsins | W1 12103682 EMPTY (bus); W3 12115458 RUNNING | W1 12103617 RUNNING | Wait. |
| ginkgo_colloidal | W1 12103683 RUNNING | W1 12103618 RUNNING | Wait for W1. |
| ginkgo_heparin | W1 12103684 EMPTY (bus); W3 12115459 RUNNING | W1 12103619 RUNNING | Wait. |
| ginkgo_hic | W1 12103685 EMPTY (bus); W3 12115460 RUNNING | W1 12103620 RUNNING | Wait. |
| ginkgo_pr_cho | W1 12103686 RUNNING | W1 12103621 RUNNING | Wait for W1. |
| ginkgo_titer | W1 12103687 EMPTY (bus); W3 12115461 RUNNING | W1 12103622 RUNNING | Wait. |

## K. TCGA (n=2)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| tcga_luad_vs_lusc | W1 12103690 EMPTY (bus); W3 12115503 RUNNING | W1 cancelled; W2 12111863 RUNNING; W3 12115502 RUNNING (duplicate) | Wait. **Cancel one cx duplicate**. |
| tcga_brca_pam50 | W1 12103691 EMPTY (bus); W3 12115501 RUNNING | W1 cancelled; W2 12111864 RUNNING; W3 12115500 RUNNING (duplicate) | Wait. **Cancel one cx duplicate**. |

## L. Nucleotide (n=17)

Pattern is uniform: W1 sim EMPTY (bus), W1 cx cancelled, W2 cx RUNNING, W3 sim+cx RUNNING. All 17 cx datasets have **parallel duplicate** Wave-2 + Wave-3 jobs.

| Dataset | Similarity | Complexity (DUPLICATE jobs running) | Action |
|---|---|---|---|
| nt_H2AFZ | W1 12103692 EMPTY; W3 12115472 RUNNING | W2 12111850 + W3 12115471 | Wait. **Cancel cx W3 (12115471)**. |
| nt_H3K27ac | W1 12103693 EMPTY; W3 12115474 RUNNING | W2 12111851 + W3 12115473 | Cancel cx W3 (12115473). |
| nt_H3K27me3 | W1 12103694 EMPTY; W3 12115476 RUNNING | W2 12111852 + W3 12115475 | Cancel cx W3 (12115475). |
| nt_H3K36me3 | W1 12103695 EMPTY; W3 12115478 RUNNING | W2 12111853 + W3 12115477 | Cancel cx W3 (12115477). |
| nt_H3K4me1 | W1 12103696 EMPTY; W3 12115480 RUNNING | W2 12111854 + W3 12115479 | Cancel cx W3 (12115479). |
| nt_H3K4me2 | W1 12103697 EMPTY; W3 12115482 RUNNING | W2 12111855 + W3 12115481 | Cancel cx W3 (12115481). |
| nt_H3K4me3 | W1 12103698 EMPTY; W3 12115484 RUNNING | W2 12111856 + W3 12115483 | Cancel cx W3 (12115483). |
| nt_H3K9ac | W1 12103699 EMPTY; W3 12115486 RUNNING | W2 12111857 + W3 12115485 | Cancel cx W3 (12115485). |
| nt_H3K9me3 | W1 12103700 EMPTY; W3 12115488 RUNNING | W2 12111858 + W3 12115487 | Cancel cx W3 (12115487). |
| nt_H4K20me1 | W1 12103701 EMPTY; W3 12115490 RUNNING | W2 12111859 + W3 12115489 | Cancel cx W3 (12115489). |
| nt_enhancers | W1 12103702 EMPTY; W3 12115468 RUNNING | W2 12111848 + W3 12115467 | Cancel cx W3 (12115467). |
| nt_enhancers_types | W1 12103703 EMPTY; W3 12115470 RUNNING | W2 12111849 + W3 12115469 | Cancel cx W3 (12115469). |
| nt_promoter_all | W1 12103704 EMPTY; W3 12115492 RUNNING | W2 12111843 + W3 12115491 | Cancel cx W3 (12115491). |
| nt_promoter_no_tata | W1 12103705 EMPTY; W3 12115494 RUNNING | W2 12111844 + W3 12115493 | Cancel cx W3 (12115493). |
| nt_promoter_tata | W1 12103706 RUNNING (W1 sim still going); **no W3 sim** | W2 12111845 + W3 12115495 | sim: wait for W1. cx: cancel W3 (12115495). |
| nt_splice_sites_acceptors | W1 12103707 EMPTY; W3 12115497 RUNNING | W2 12111846 + W3 12115496 | Cancel cx W3 (12115496). |
| nt_splice_sites_donors | W1 12103708 EMPTY; W3 12115499 RUNNING | W2 12111847 + W3 12115498 | Cancel cx W3 (12115498). |

## M. Genomic Benchmarks (n=4)

| Dataset | Similarity | Complexity | Action |
|---|---|---|---|
| gb_demo_coding_vs_intergenomic_seqs | W1 12103709 EMPTY (bus); **no sim resubmit** | W1 cancelled; W2 12111841 RUNNING; W3 12115451 RUNNING (duplicate) | sim: **resubmit**. cx: cancel W3 duplicate (12115451). |
| gb_demo_human_or_worm | W1 12103710 EMPTY (bus, 13 min — quick fail); W3 12115453 **python_error** (4:38) | W1 cancelled; W2 12111842 RUNNING; W3 12115452 RUNNING (duplicate) | sim: **investigate slurm-12115453.out, then resubmit**. cx: cancel W3 (12115452). |
| gb_human_enhancers_cohn | W1 12103711 EMPTY (bus); W3 12115455 **python_error** (4:38) | W1 cancelled; W2 12111839 RUNNING; W3 12115454 RUNNING (duplicate) | sim: **investigate slurm-12115455.out**. cx: cancel W3 (12115454). |
| gb_human_nontata_promoters | W1 12103712 EMPTY (bus); W3 12115457 **python_error** (4:38) | W1 cancelled; W2 12111840 RUNNING; W3 12115456 RUNNING (duplicate) | sim: **investigate slurm-12115457.out**. cx: cancel W3 (12115456). |

---

## Consolidated action list

### Investigate first (3 fast-failing sim jobs, same 4:38 elapsed)
```
head -50 slurm-12115453.out slurm-12115455.out slurm-12115457.out
grep -E "Traceback|Error|raise" slurm-12115453.out
```

### Resubmit (no job currently in flight)

**Similarity — fresh full submits (7 datasets):**
```
sbatch --job-name=sim_caco2 tests/slurm_scripts/submit_dataset.sh similarity caco2
sbatch --job-name=sim_logd tests/slurm_scripts/submit_dataset.sh similarity logd
sbatch --job-name=sim_qm9 tests/slurm_scripts/submit_dataset.sh similarity qm9
sbatch --job-name=sim_adme_hlm tests/slurm_scripts/submit_dataset.sh similarity adme_hlm
sbatch --job-name=sim_adme_rlm tests/slurm_scripts/submit_dataset.sh similarity adme_rlm
sbatch --job-name=sim_adme_mdr1 tests/slurm_scripts/submit_dataset.sh similarity adme_mdr1
sbatch --job-name=sim_herg tests/slurm_scripts/submit_dataset.sh similarity herg
sbatch --job-name=sim_gb_demo_coding_vs_intergenomic_seqs tests/slurm_scripts/submit_dataset.sh similarity gb_demo_coding_vs_intergenomic_seqs
```

**Complexity — partial resubmits (4 datasets, need c23/c24 plus c10 if noiseInject is installed):**
```
sbatch --job-name=cx_flab_polyreact_jain_cic_p tests/slurm_scripts/submit_dataset.sh complexity flab_polyreact_jain_cic "c23 c24"
sbatch --job-name=cx_flab_polyreact_jain_bvp_p tests/slurm_scripts/submit_dataset.sh complexity flab_polyreact_jain_bvp "c23 c24"
sbatch --job-name=cx_abagym_HER2_p tests/slurm_scripts/submit_dataset.sh complexity abagym_HER2 "c23 c24"
sbatch --job-name=cx_flab_immunogenicity_marks_p tests/slurm_scripts/submit_dataset.sh complexity flab_immunogenicity_marks "c13 c15 c16 c17 c18 c19 c21 c22 c23 c24"
```

**Complexity — full resubmit (1 dataset):**
```
sbatch --job-name=cx_abagym_lysozyme tests/slurm_scripts/submit_dataset.sh complexity abagym_lysozyme
```

### Cancel duplicate complexity jobs (one per pair, 25 total — VERIFY EACH BEFORE CANCELLING)

Per the no-bulk-scancel rule, **do not** paste these as a block. Cancel one
at a time, confirming with `squeue -j <pair>` first. The Wave 3 ID is the
newer redundant one in each case.

cx duplicates to cancel (newer = W3): 12115471, 12115473, 12115475, 12115477,
12115479, 12115481, 12115483, 12115485, 12115487, 12115489, 12115467,
12115469, 12115491, 12115493, 12115495, 12115496, 12115498, 12115502,
12115500, 12115504, 12115506, 12115451, 12115452, 12115454, 12115456.

### Wait

Everything else is either in flight from Wave 1 (for orig RUNNING) or in
flight from Wave 3. Re-run `python tests/audit_full_wave.py
--include-running` later to track convergence.

---

## Open known bugs (separate from the wave)

- **noiseInject not installed** — `c10_noise_robustness` self-fails on every
  complexity job. Install: `pip install noiseInject`.
- **s12 `no_cliffs_found`** — methodological; expected on antibody / non-chem
  datasets. Counts as "errored" in the audit but is a real result, not a
  failure.
- **`submit_dataset.sh` masks failures** — bus errors/python crashes return
  exit 0 because the script doesn't propagate `$?`. Add `set -e` or check
  `${PIPESTATUS}` so sacct reflects the truth.
