# Per-Dataset Experiment Status — 2026-05-11 (evening, post-diagnosis)

Now grounded in (a) `audit_full_wave.py --include-running` output, plus
(b) verified slurm log tails for the suspicious cases. Three concrete
findings replaced earlier guesses.

## Verified findings (replacing earlier guesses)

### 1. The "no progress markers" jobs are NOT stuck

Audit said 14 Wave 1 jobs (e.g. 12103594, all 6 ginkgo cx, etc.) had no
progress markers in tail-200. They're actually all in a **second
precompute stage** that my classifier didn't recognize:

```
[2435/2980] mlp/entropy_2.0/treeshap/b=100/ev=mlp: score=0.2015+/-0.150
```

That's the per-config eval loop after fold 0 pruning. Range observed:
~50–95% through these loops. **All 14 are alive and progressing.** Log
mtimes are recent (today).

Updated `audit_full_wave.py` (next commit) recognizes this stage. Re-run
to see the actual progress on these jobs.

### 2. `sim/abagym_Ang2` missing s03 — real numerical bug

```
s03 FAILED: SVD did not converge
File "tests/experiments/../experiments/similarity/s03_feature_concentration.py", line 86
```

Dataset-specific `scipy.linalg.svd` non-convergence. Not a wave problem;
a real bug in `s03_feature_concentration.py:86`. Fix: switch to
`linalg.svd(..., lapack_driver='gesvd')` or wrap with NaN-handling.
**Don't resubmit Ang2 alone — fix s03 first.**

### 3. The 3 four-minute sim/gb_* failures are a CACHE RACE

```
FileNotFoundError: '16748.txt'
shutil.rmtree(str(dest_path))
```

The `genomic_benchmarks` library tries to `rmtree` the cache dir while
another concurrent job is reading from it. Classic concurrent-download
race. Affects:
- 12115453 sim/gb_demo_human_or_worm
- 12115455 sim/gb_human_enhancers_cohn
- 12115457 sim/gb_human_nontata_promoters

The cx_gb_* Wave 3 jobs (12115451/2/4/6) are still RUNNING — they got
luckier with timing, OR the cache was already built by the W2 cx_gb_*
jobs (12111839–42) which started earlier.

**Fix before any gb_* resubmit:** either (a) pre-build the
genomic_benchmarks cache once, then submit, OR (b) submit gb_* jobs
sequentially (one at a time) so they don't race.

---

## Top-line counts (from this audit)

```
Verdict counts: OK = 4, PARTIAL = 21, EMPTY = 55, RUNNING = 129
```

- **OK (4)**: sim/flab_immunogenicity_marks (W1 + W3), sim/humanness_therapeutic (W2 + W3)
- **Effectively OK (sim antibody, errored=s12 only)**: 8 rows — abagym_EGFR/HER2/VEGF, abcov_kd, flab_polyreact_jain_bvp/cic (plus W1 duplicates of some). All confirmed `errored=s12` only.
- **PARTIAL real losses (cx)**: 7 datasets need targeted resubmits.
- **EMPTY (55)**: Wave 1 bus errors, mostly in precompute fold 0.
- **RUNNING (129)**: All in some precompute stage. None have written a per-experiment JSON yet, **except**:
  - cx/abagym_Ang2 (12103609): last_done=c15, fresh=10
  - cx/abagym_EGFR (12103610): last_done=c11, fresh=9
  - cx/abagym_VEGF W3 (12115431): fresh=9 (precompute reused from cache)
  - cx/humanness_therapeutic W3 (12115464): last_done=c11, fresh=9
  - sim/ginkgo_heparin W3 (12115459): last_done=s04, fresh=4

So 5 RUNNING jobs are actually past precompute and running per-experiment.

---

## Per-dataset status

### A. Molecular regression — all SIM EMPTY, all CX RUNNING in precompute stage 2

| Dataset | Sim | Cx |
|---|---|---|
| caco2 | W1 12103649 **EMPTY** (bus); no resubmit running | W1 12103584 RUNNING (fold 0: 2000/7425) |
| logd | W1 12103650 **EMPTY** (bus); no resubmit running | W1 12103585 RUNNING (precompute) |
| qm9 | W1 12103651 **EMPTY** (bus); no resubmit running | W1 12103586 RUNNING (fold 0: 1800/6480) |
| adme_hlm | W1 12103652 **EMPTY** (bus); no resubmit running | W1 12103587 RUNNING (fold 0: 2000/7425) |
| adme_rlm | W1 12103653 **EMPTY** (bus); no resubmit running | W1 12103588 RUNNING (fold 0: 2000/7425) |
| adme_mdr1 | W1 12103654 **EMPTY** (bus); no resubmit running | W1 12103589 RUNNING (fold 0: 1800/7425) |

### B. Molecular classification

| Dataset | Sim | Cx |
|---|---|---|
| herg | W1 12103655 **EMPTY** (bus); no resubmit running | W1 12103590 RUNNING (fold 0: 3000/6480) |
| herg_ki | W1 12103648 **EMPTY**; W3 12115463 RUNNING (fold 0: 4800/7425) | W1 12103583 **EMPTY** (bus); W3 12115462 RUNNING (fold 0: 4800/7425) |

### C. DTA

| Dataset | Sim | Cx |
|---|---|---|
| kiba | W1 12103656 **EMPTY**; W3 12115466 RUNNING | W1 12103591 RUNNING (pruning) |
| bindingdb_kd | W1 12103657 **EMPTY**; W3 12115434 RUNNING | W1 12103592 RUNNING (pruning) |

### D. Reaction

| Dataset | Sim | Cx |
|---|---|---|
| buchwald_hartwig | W1 12103688 **EMPTY**; W3 12115435 RUNNING | W1 12103623 RUNNING (config 2133/5022) |
| bh_doyle_plates | W1 12103689 RUNNING (pruning) | W1 12103624 RUNNING (pruning) |

### E. DDG

| Dataset | Sim | Cx |
|---|---|---|
| ddg_combined | W1 12103668 RUNNING (config 810/3852) | W1 12103603 **EMPTY** (bus); W3 12115436 RUNNING |

### F. Antibody — flab regressions

| Dataset | Sim | Cx |
|---|---|---|
| flab_thermo_tresanco | W1 12103658 **EMPTY**; W3 12115450 RUNNING | W1 12103593 RUNNING (fold 0: 5000/6804) |
| flab_thermo_jain | W1 12103663 **EMPTY**; W3 12115449 RUNNING | W1 12103598 RUNNING (config 2433/3290) |
| flab_pk_jain_heprt | W1 12103659 RUNNING (config 2424/2980) | W1 12103594 RUNNING (config 2437/2980) |
| flab_pk_jain_fcrnrt | W1 12103660 RUNNING (fold 0: 2600/3240) | W1 12103595 RUNNING (fold 0: 2600/3240) |
| flab_polyreact_jain_cic | W1 12103661 **EFFECTIVELY OK** errored=s12; W3 12115448 same | W1 12103596 **PARTIAL 17/20**, errored=c10 missing=c23,c24 |
| flab_polyreact_jain_bvp | W1 12103667 **EFFECTIVELY OK** errored=s12; W3 12115447 same | W1 12103602 **PARTIAL 17/20**, errored=c10 missing=c23,c24 |
| flab_aggregation_jain | W1 12103665 RUNNING (fold 0: 3600/3780, ~95%) | W1 12103600 RUNNING (fold 0: 3600/3780) |
| flab_aggregation_jain_hicrt | W1 12103662 **EMPTY**; W3 12115437 RUNNING | W1 12103597 RUNNING (config 2784/3305) |
| flab_immunogenicity_marks | W1 12103666 **OK 12/12**; W3 12115446 OK redundant | W1 12103601 **PARTIAL 12/20**, errored=c10 missing=c17–c24 |
| flab_binding_koenig | W1 12103669 **EMPTY**; W3 12115441 RUNNING | W1 12103604 **EMPTY** (bus); W3 12115440 RUNNING |
| flab_binding_adams | W1 12103671 **EMPTY**; W3 12115439 RUNNING (fold 0: 5800/7425) | W1 12103606 **EMPTY**; W3 12115438 RUNNING (fold 0: 5800/7425) |
| flab_binding_phillips_h3 | W1 12103673 RUNNING (fold 0: 1000/7425) | W1 12103608 **EMPTY**; W3 12115442 RUNNING |
| flab_expression_koenig | W1 12103670 RUNNING (fold 0: 1000/7425) | W1 12103605 **EMPTY**; W3 12115445 RUNNING |
| flab_expression_adams | W1 12103672 **EMPTY**; W3 12115444 RUNNING | W1 12103607 **EMPTY**; W3 12115443 RUNNING (fold 0: 5600/7425) |

### G. Antibody — other

| Dataset | Sim | Cx |
|---|---|---|
| abcov_kd | W1 12103664 **EFFECTIVELY OK** errored=s12; W3 12115433 same | W1 12103599 **PARTIAL 17/20**, errored=c10 missing=c23,c24 |

### H. Abagym

| Dataset | Sim | Cx |
|---|---|---|
| abagym_Ang2 | W1 12103674 PARTIAL 11/12, **missing=s03** ← s03 SVD bug | W1 12103609 RUNNING — already past precompute, last_done=c15, fresh=10 |
| abagym_EGFR | W1 12103675 errored=s12; W3 12115429 **EFFECTIVELY OK** errored=s12 | W1 12103610 RUNNING — already past precompute, last_done=c11, fresh=9 |
| abagym_HER2 | W1 12103676 **EFFECTIVELY OK**; W3 12115430 same | W1 12103611 **PARTIAL 17/20**, errored=c10 missing=c23,c24 |
| abagym_VEGF | W1 12103678 errored=s12; W3 12115432 **EFFECTIVELY OK** | W1 12103613 PARTIAL 9/20 (bus early); W3 12115431 RUNNING fresh=9 missing=c13–c24 |
| abagym_lysozyme | W1 12103677 RUNNING (fold 0: 4200/5292) | W1 12103612 **EMPTY** (bus); **no resubmit running** |

### I. Antibody classification

| Dataset | Sim | Cx |
|---|---|---|
| tz_her2_binary | W1 cancelled; W2 12111865 RUNNING; W3 12115505 RUNNING (DUPLICATE) | W1 cancelled; W2 12111860 RUNNING; W3 12115504 RUNNING (DUPLICATE) |
| tz_her2_multiclass | W1 cancelled; W2 12111866 **EMPTY** (bus); W3 12115507 RUNNING (fold 0: 6800/7425) | W1 cancelled; W2 12111861 **EMPTY** (bus); W3 12115506 RUNNING (fold 0: 6800/7425) |
| humanness_therapeutic | W1 cancelled; W2 12111867 **OK 12/12**; W3 12115465 OK redundant | W1 cancelled; W2 12111862 **PARTIAL 9/20** missing=c13–c24; W3 12115464 RUNNING last_done=c11 fresh=9 |

### J. Ginkgo

| Dataset | Sim | Cx |
|---|---|---|
| ginkgo_acsins | W1 12103682 **EMPTY**; W3 12115458 RUNNING | W1 12103617 RUNNING (config 2264/2400, ~94%) |
| ginkgo_colloidal | W1 12103683 RUNNING (config 1640/3325) | W1 12103618 RUNNING (config 1660/3330) |
| ginkgo_heparin | W1 12103684 PARTIAL 4/12 missing=s05–s13 (bus); W3 12115459 RUNNING last_done=s04 fresh=4 (about to hit same point) | W1 12103619 RUNNING (config 2396/2608, ~92%) |
| ginkgo_hic | W1 12103685 **EMPTY** (bus); W3 12115460 RUNNING | W1 12103620 RUNNING (config 1703/3350) |
| ginkgo_pr_cho | W1 12103686 RUNNING (config 1736/3600) | W1 12103621 RUNNING (config 1733/3600) |
| ginkgo_titer | W1 12103687 **EMPTY** (bus); W3 12115461 RUNNING | W1 12103622 RUNNING (config 1509/3095) |

### K. TCGA

| Dataset | Sim | Cx |
|---|---|---|
| tcga_luad_vs_lusc | W1 12103690 **EMPTY** (bus); W3 12115503 RUNNING (fold 0: 4600/7425) | W1 cancelled; W2 12111863 RUNNING; W3 12115502 RUNNING (DUPLICATE) |
| tcga_brca_pam50 | W1 12103691 **EMPTY** (bus); W3 12115501 RUNNING | W1 cancelled; W2 12111864 RUNNING; W3 12115500 RUNNING (DUPLICATE) |

### L. Nucleotide (17 datasets — all DUPLICATE complexity jobs running)

All sim W1 jobs **EMPTY** (bus in precompute_importance — they died very
early). All sim W3 jobs RUNNING (fold 0 progress varies). All cx W2 +
W3 jobs RUNNING in parallel.

Sim progress sample (W3): nt_H3K4me3 fold 0: 5600/5724 (~98%);
nt_promoter_all fold 0: 5000/5724; nt_promoter_no_tata fold 0: 5000/5724.
Many in 2000–5000 range.

| Dataset | Sim status | Cx duplicates |
|---|---|---|
| nt_H2AFZ | W1 EMPTY; W3 12115472 RUNNING fold 0: 2400/5724 | W2 12111850 + W3 12115471 |
| nt_H3K27ac | W1 EMPTY; W3 12115474 RUNNING fold 0: 4200/5724 | W2 12111851 + W3 12115473 |
| nt_H3K27me3 | W1 EMPTY; W3 12115476 RUNNING fold 0: 2200/5724 | W2 12111852 + W3 12115475 |
| nt_H3K36me3 | W1 EMPTY; W3 12115478 RUNNING fold 0: 2400/5724 | W2 12111853 + W3 12115477 |
| nt_H3K4me1 | W1 EMPTY; W3 12115480 RUNNING precompute_importance | W2 12111854 + W3 12115479 |
| nt_H3K4me2 | W1 EMPTY; W3 12115482 RUNNING fold 0: 3000/5724 | W2 12111855 + W3 12115481 |
| nt_H3K4me3 | W1 EMPTY; W3 12115484 RUNNING fold 0: 5600/5724 (~98%) | W2 12111856 + W3 12115483 |
| nt_H3K9ac | W1 EMPTY; W3 12115486 RUNNING fold 0: 4600/5724 | W2 12111857 + W3 12115485 |
| nt_H3K9me3 | W1 EMPTY; W3 12115488 RUNNING fold 0: 4400/5724 | W2 12111858 + W3 12115487 |
| nt_H4K20me1 | W1 EMPTY; W3 12115490 RUNNING fold 0: 3200/5724 | W2 12111859 + W3 12115489 |
| nt_enhancers | W1 EMPTY; W3 12115468 RUNNING fold 0: 3671/5724 | W2 12111848 + W3 12115467 |
| nt_enhancers_types | W1 EMPTY; W3 12115470 RUNNING fold 0: 600/5724 | W2 12111849 + W3 12115469 |
| nt_promoter_all | W1 EMPTY; W3 12115492 RUNNING fold 0: 5000/5724 | W2 12111843 + W3 12115491 |
| nt_promoter_no_tata | W1 EMPTY; W3 12115494 RUNNING fold 0: 5000/5724 | W2 12111844 + W3 12115493 |
| nt_promoter_tata | W1 12103706 RUNNING (pruning) — **no sim W3 resubmit** | W2 12111845 + W3 12115495 |
| nt_splice_sites_acceptors | W1 EMPTY; W3 12115497 RUNNING fold 0: 1200/5724 | W2 12111846 + W3 12115496 |
| nt_splice_sites_donors | W1 EMPTY; W3 12115499 RUNNING fold 0: 2600/5724 | W2 12111847 + W3 12115498 |

### M. Genomic Benchmarks (cache race condition territory)

| Dataset | Sim | Cx |
|---|---|---|
| gb_demo_coding_vs_intergenomic_seqs | W1 12103709 **EMPTY** (bus); **no resubmit** | W2 12111841 RUNNING fold 0: 1000/5724; W3 12115451 RUNNING precompute_importance (DUPLICATE) |
| gb_demo_human_or_worm | W1 12103710 **EMPTY** (bus); W3 12115453 **python_error: cache race** (4:38) | W2 12111842 RUNNING fold 0: 4400/5724; W3 12115452 RUNNING (DUPLICATE) |
| gb_human_enhancers_cohn | W1 12103711 **EMPTY** (bus); W3 12115455 **python_error: cache race** | W2 12111839 RUNNING fold 0: 5200/5724; W3 12115454 RUNNING (DUPLICATE) |
| gb_human_nontata_promoters | W1 12103712 **EMPTY** (bus); W3 12115457 **python_error: cache race** | W2 12111840 RUNNING fold 0: 4600/5724; W3 12115456 RUNNING (DUPLICATE) |

---

## Action list (revised, in priority order)

### A. Investigations done — no further investigation needed

The 3 things I asked you to investigate now have root causes:
- "Stuck" jobs → not stuck, in stage 2 of precompute
- abagym_Ang2 s03 → SVD non-convergence bug in s03_feature_concentration.py:86
- 3 four-minute gb_* deaths → cache race in genomic_benchmarks library

### B. Fix the s03 bug, then resubmit Ang2

**Code fix needed in `tests/experiments/similarity/s03_feature_concentration.py` line 86.** The SVD call needs to fall back to `gesvd` driver or catch `LinAlgError`. Skip this code change and the resubmit is pointless.

After fix:
```
sbatch --job-name=sim_abagym_Ang2_p tests/slurm_scripts/submit_dataset.sh similarity abagym_Ang2 "s03"
```

### C. Fix the gb_* cache race before any gb_* resubmit

Either pre-warm the cache:
```
# Run once on a login node or in a small interactive job
python -c "
from genomic_benchmarks.loc2seq import download_dataset
for ds in ['demo_human_or_worm','human_enhancers_cohn','human_nontata_promoters','demo_coding_vs_intergenomic_seqs']:
    download_dataset(ds, version=0)
"
```

Or submit gb_* sim jobs serially (one at a time) so they don't fight over the cache:
```
sbatch --job-name=sim_gb_demo_coding_vs_intergenomic_seqs tests/slurm_scripts/submit_dataset.sh similarity gb_demo_coding_vs_intergenomic_seqs
# wait for it to complete before submitting the next one
```

After cache is built (or one job is confirmed past the download stage):
```
sbatch --job-name=sim_gb_demo_human_or_worm tests/slurm_scripts/submit_dataset.sh similarity gb_demo_human_or_worm
sbatch --job-name=sim_gb_human_enhancers_cohn tests/slurm_scripts/submit_dataset.sh similarity gb_human_enhancers_cohn
sbatch --job-name=sim_gb_human_nontata_promoters tests/slurm_scripts/submit_dataset.sh similarity gb_human_nontata_promoters
```

### D. Resubmit the cx PARTIALs (5 datasets, missing 2-10 experiments each)

These are safe — no in-flight job covers them.

```
sbatch --job-name=cx_abcov_kd_p tests/slurm_scripts/submit_dataset.sh complexity abcov_kd "c23 c24"
sbatch --job-name=cx_abagym_HER2_p tests/slurm_scripts/submit_dataset.sh complexity abagym_HER2 "c23 c24"
sbatch --job-name=cx_flab_polyreact_jain_cic_p tests/slurm_scripts/submit_dataset.sh complexity flab_polyreact_jain_cic "c23 c24"
sbatch --job-name=cx_flab_polyreact_jain_bvp_p tests/slurm_scripts/submit_dataset.sh complexity flab_polyreact_jain_bvp "c23 c24"
sbatch --job-name=cx_flab_immunogenicity_marks_p tests/slurm_scripts/submit_dataset.sh complexity flab_immunogenicity_marks "c17 c18 c19 c21 c22 c23 c24"
```

Skip: cx/abagym_VEGF and cx/humanness_therapeutic — W3 resubmits are
already in-flight past precompute and saving experiments.

### E. Resubmit the cx full EMPTY (1 dataset, no in-flight)

```
sbatch --job-name=cx_abagym_lysozyme tests/slurm_scripts/submit_dataset.sh complexity abagym_lysozyme
```

### F. Resubmit the sim full EMPTY without in-flight resubmits (8 datasets)

```
sbatch --job-name=sim_caco2 tests/slurm_scripts/submit_dataset.sh similarity caco2
sbatch --job-name=sim_logd tests/slurm_scripts/submit_dataset.sh similarity logd
sbatch --job-name=sim_qm9 tests/slurm_scripts/submit_dataset.sh similarity qm9
sbatch --job-name=sim_adme_hlm tests/slurm_scripts/submit_dataset.sh similarity adme_hlm
sbatch --job-name=sim_adme_rlm tests/slurm_scripts/submit_dataset.sh similarity adme_rlm
sbatch --job-name=sim_adme_mdr1 tests/slurm_scripts/submit_dataset.sh similarity adme_mdr1
sbatch --job-name=sim_herg tests/slurm_scripts/submit_dataset.sh similarity herg
# sim_gb_demo_coding_vs_intergenomic_seqs — see section C above (cache race)
```

### G. Duplicate complexity jobs (W2 + W3 both running) — optional cleanup

25 cx datasets have a Wave 2 RUNNING + Wave 3 RUNNING duplicate. They'll
both produce the same output; the second writer wins. If compute is
tight, cancel the W3 of each pair (newer, less progress) one-by-one. If
compute is fine, leave them — redundancy is harmless.

Newer Wave 3 cx IDs to consider cancelling (paste **one at a time**,
verify with `squeue -j <pair>`): 12115471, 12115473, 12115475, 12115477,
12115479, 12115481, 12115483, 12115485, 12115487, 12115489, 12115467,
12115469, 12115491, 12115493, 12115495, 12115496, 12115498, 12115500,
12115502, 12115504, 12115506, 12115451, 12115452, 12115454, 12115456.

---

## Known bugs (sticky)

| Bug | Action |
|---|---|
| Bus error in Wave 1 precompute | User says fixed; Wave 1 long-runners may still hit it |
| noiseInject not installed → c10 always errors | `pip install noiseInject` |
| s12 no_cliffs_found on antibody | Methodological, not a bug |
| submit_dataset.sh masks failures (exit 0 on bus) | Add `set -e` or `exit $?` |
| s03 SVD non-convergence on abagym_Ang2 | Fix `s03_feature_concentration.py:86` |
| genomic_benchmarks cache race | Pre-warm cache or serialize submits |

---

## Precompute timing (from `check_precompute_status.py --timing --incomplete`)

65 datasets checked → **13 DONE, 28 partial, 24 missing**. The flags below are filtered to the rule you gave me: ignore tiny-dataset GP dominance that finishes in hours; flag (a) one evaluator eating days on its own, (b) any single condition projected past ~4 days wall, (c) anything that would push a dataset to 10+ days across all 6 conditions.

### 🚩 FLAG 1 — Suspected timing-calculation bug in `capped` (RF 10–20× slower than uncapped)

Five datasets show the same pattern: RF in **capped** is 435–605 s/config, but RF in **uncapped** on the same dataset is 13–33 s/config. The other evaluators in capped are mildly elevated (2–3×), but RF is the only one that explodes. Either the capped wrapper is mis-attributing wall time to RF, or RF retraining inside the augmentation loop is genuinely running 15× more often than it should.

| Dataset | n | RF capped (s/cfg) | RF uncapped (s/cfg) | Ratio | Capped total |
|---|---:|---:|---:|---:|---:|
| flab_aggregation_jain_hicrt | 137 | **604.00** | 32.90 | 18.4× | **137.13 h (5.7 d)** |
| flab_pk_jain_fcrnrt | 132 | **435.77** | 13.28 | 32.8× | **97.59 h (4.1 d)** |
| ginkgo_colloidal | 242 | **474.91** | 12.02 | 39.5× | **113.17 h (4.7 d)** |
| ginkgo_hic | 242 | **437.40** | 12.20 | 35.9× | **103.71 h (4.3 d)** |
| ginkgo_pr_cho | 197 | **536.95** | 12.23 | 43.9× | **119.38 h (5.0 d)** |

These are tiny datasets (n=132–242). RF on a 132-row table should not take 7 minutes/config under any reasonable book-keeping. **Diagnose the capped wall-time accounting before any large-scale capped resubmission** — if the bug is in instrumentation only, results are fine; if RF is genuinely retraining 15× per augmentation pass, capped output is suspect.

### 🚩 FLAG 2 — Single evaluator burning ≥1 day in one condition

Ordered by single-evaluator hours. Everything below is one model in one condition — multiply by 6 to project full-dataset cost.

| Dataset | Condition | Evaluator | Wall (h) | Days | Total cond (h) | Notes |
|---|---|---|---:|---:|---:|---|
| caco2 | general/none | **GP** | 75.27 | 3.1 | 79.76 | n=2158, not tiny — flag |
| flab_thermo_tresanco | general/uncapped | **GP** | 66.44 | 2.8 | 128.49 | n=672, MLP also 40.33h |
| caco2 | top/none | **GP** | 62.01 | 2.6 | 64.91 | n=2158 |
| tcga_luad_vs_lusc | general/uncapped | **GP** | 57.70 | 2.4 | 90.35 | n=1017 |
| flab_thermo_tresanco | general/uncapped | **MLP** | 40.33 | 1.7 | (same row) | second-worst in same cond |
| nt_H2AFZ | general/none | **Ridge** | 37.39 | 1.6 | 76.39 | n=33000, nucleotide |
| nt_H3K4me1 | general/none | **Ridge** | 35.94 | 1.5 | 75.01 | n=33000 |
| herg_ki | general/none | **GP** | 31.34 | 1.3 | 39.62 | n=1412 |
| nt_H3K27me3 | general/none | **Ridge** | 30.63 | 1.3 | 71.57 | n=33000 |
| nt_H3K27ac | general/none | **Ridge** | 27.14 | 1.1 | 62.58 | n=31616 |
| ddg_combined | general/uncapped | **GP** | 26.18 | 1.1 | 41.37 | n=429 |
| nt_H3K9ac | general/none | **Ridge** | 22.58 | 0.94 | 50.81 | n=24278, near 1d |
| herg_ki | top/none | **GP** | 19.49 | 0.81 | 21.91 | borderline |
| qm9 | general/none | **SVM** | 18.46 | 0.77 | 45.06 | n=9978, borderline |

(All 5 capped RF cases from FLAG 1 also belong here — omitted to avoid double-counting.)

### 🚩 FLAG 3 — Conditions projected past 4 days wall (>96 h in one of six conditions)

| Dataset | Condition | Wall (h) | Days | Bottleneck |
|---|---|---:|---:|---|
| flab_aggregation_jain_hicrt | general/capped | 137.13 | **5.7** | RF (FLAG 1) |
| flab_thermo_tresanco | general/uncapped | 128.49 | **5.4** | GP + MLP |
| ginkgo_pr_cho | general/capped | 119.38 | **5.0** | RF (FLAG 1) |
| ginkgo_colloidal | general/capped | 113.17 | **4.7** | RF (FLAG 1) |
| ginkgo_hic | general/capped | 103.71 | **4.3** | RF (FLAG 1) |
| flab_pk_jain_fcrnrt | general/capped | 97.59 | 4.1 | RF (FLAG 1) |
| tcga_luad_vs_lusc | general/uncapped | 90.35 | 3.8 | GP |
| caco2 | general/none | 79.76 | 3.3 | GP |
| nt_H2AFZ | general/none | 76.39 | 3.2 | Ridge |
| nt_H3K4me1 | general/none | 75.01 | 3.1 | Ridge |
| nt_H3K27me3 | general/none | 71.57 | 3.0 | Ridge |
| caco2 | top/none | 64.91 | 2.7 | GP |
| nt_H3K27ac | general/none | 62.58 | 2.6 | Ridge |
| nt_H3K9ac | general/none | 50.81 | 2.1 | Ridge |

### 🚩 FLAG 4 — Projected to exceed 10 days across all 6 conditions

Datasets where the worst observed condition × the missing condition count would push total wall >240 h. The 7-day SLURM wall (`--time=7-00:00:00`) means **any one of these will not finish a single job to completion** — they need partitioning or per-condition submission.

| Dataset | Observed worst (h) | Missing conds | Risk |
|---|---:|---|---|
| **flab_aggregation_jain_hicrt** | 137 capped | top/uncapped, top/capped | top/capped likely repeats the RF bug → **≥7d single job** |
| **flab_pk_jain_fcrnrt** | 97 capped | top/capped only | top/capped alone is the risk |
| **ginkgo_colloidal** | 113 capped | top/capped only | RF bug recurs → likely ≥4d on its own |
| **ginkgo_hic** | 103 capped | top/capped only | same |
| **ginkgo_pr_cho** | 119 capped | top/capped only | same |
| **flab_thermo_tresanco** | 128 uncapped | general/capped + 3 top | general/capped will be **massive** if RF bug applies |
| **caco2** | 79 + 65 (both none) | uncapped + capped × top/general | GP at n=2158 in uncapped is unbounded — **definite >10d** |
| **tcga_luad_vs_lusc** | 90 uncapped | general/capped + 3 top | GP scaling — **likely >10d total** |
| **herg_ki** | 39 + 22 (both none) | uncapped + capped × top/general | GP at n=1412, scales — **possibly >10d** |
| **nt_H2AFZ / H3K4me1 / H3K27me3 / H3K27ac / H3K9ac / H3K4me3** | 50–76 none | uncapped + capped × top/general | Ridge at n=24k–33k — uncapped/capped will be far worse |

### 🚩 FLAG 5 — Datasets with zero precompute caches (still missing entirely)

```
flab_binding_adams        flab_binding_koenig        flab_binding_phillips_h3
flab_expression_adams     flab_expression_koenig     tz_her2_multiclass
bindingdb_kd              kiba                       herg
logd                      bh_doyle_plates            buchwald_hartwig
gb_demo_coding_vs_intergenomic_seqs                  gb_demo_human_or_worm
gb_human_nontata_promoters
nt_H3K36me3   nt_H3K4me2   nt_H4K20me1
nt_enhancers  nt_enhancers_types
nt_promoter_all   nt_promoter_no_tata   nt_promoter_tata
nt_splice_sites_acceptors
```

(24 datasets. The Wave-3 RUNNING jobs from Section A–M above will produce caches for many of these once they progress past stage 2; this list is current state, not lost work.)

### What's NOT flagged (tiny + GP-dominated, finishes in hours)

For context — these match the "GP eats 99% but whole thing in hours" pattern you said to ignore:

- ddg_combined general/none: GP 76% share but total 1.66h ✅
- abagym_lysozyme general/none: GP 66% but total 0.59h ✅
- ginkgo_acsins / ginkgo_titer general/none: total <1h ✅
- All `top/none` conditions on Ginkgo / flab_pk: total <0.2h ✅
- adme_hlm / adme_mdr1 / adme_rlm general/none: GP 94% but 8–13h, n=2.6–3k — borderline, *not* flagged

### Recommended next moves

1. **Diagnose the capped RF timing first.** Read one cache pickle's `_cache_meta` for `flab_aggregation_jain_hicrt__general__capped__full.pkl` and inspect the per-config time entries — if RF entries are individually 600s, the bug is in evaluation; if entries are 30s but the per-config count is inflated, the bug is in book-keeping/loop structure. Cheapest test before resubmitting any capped condition.
2. **Don't resubmit caco2 / tcga_luad_vs_lusc / flab_thermo_tresanco** with one job per dataset under the 7-day wall — they will time out. Split by `(condition × evaluator-subset)` or shorten `EVALUATORS`.
3. **Nucleotide Ridge wall**: at n=30k+, Ridge at 35–124 s/config on a 1080-config grid is a Ridge-implementation cost (likely the high-α grid + per-fold refit). Worth checking whether the Ridge grid was meant to be that wide on nucleotide.
4. Everything else (the per-dataset table in Section A–M) is already on the right track; resubmit list under "Action list" still stands.
