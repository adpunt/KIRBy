# Per-Dataset Experiment Status — 2026-05-11 (afternoon)

Single source of truth, built from a fresh `audit_full_wave.py
--include-running` (commit 280166e). Every row below is grounded in the
audit output you pasted; no projections from older data.

## How to read this

- **OK** = all expected JSONs fresh, no errors. (4 datasets total.)
- **EFFECTIVELY OK** = sim antibody with `errored=s12` (no_cliffs_found is
  a methodological no-data, not a job failure). 11/12 fresh, the 1 missing
  is s12. Confirmed by the audit `errored=s12` column.
- **PARTIAL (resubmittable)** = COMPLETED job with some fresh files; specific
  missing/errored experiment IDs listed.
- **EMPTY** = COMPLETED job with zero fresh files (Wave 1 bus errors).
- **RUNNING — IN_PROGRESS** = currently RUNNING with a known progress
  marker. Includes which fold/checkpoint it's at. **No verdict yet.**
- **RUNNING — UNKNOWN** = currently RUNNING but the log tail has no
  progress markers. Could be stuck. Worth a closer look.

## Audit headline

```
Verdict counts: OK = 4, PARTIAL = 21, EMPTY = 55, RUNNING = 129
```

Of the 21 PARTIAL: 8 are sim/antibody with the `errored=s12` pattern
(effectively OK); 13 are real losses needing targeted resubmits.

Of the 129 RUNNING: **every single one is still in `fold 0` of
`precompute_hybrid_searches`.** Not one has reached the per-experiment
loop yet. That means none of them have written any per-experiment JSON.
Some are very close to fold 0 completion (>90%), others just past 10%.

---

## Genuinely complete (OK)

| Dataset | Category | Job ID(s) |
|---|---|---|
| flab_immunogenicity_marks | similarity | W1 12103666 + W3 12115446 (redundant) |
| humanness_therapeutic | similarity | W2 12111867 + W3 12115465 (redundant) |

That's the entire "done" list.

---

## Effectively complete (sim antibody, errored=s12 only)

These finished cleanly, with `s12_cross_domain_cliffs` self-reporting
`no_cliffs_found` because antibody embeddings don't form chemistry-style
cliffs. The audit calls them PARTIAL 11/12 but they don't need resubmit
unless you change the s12 method.

| Dataset | Best Job |
|---|---|
| sim/abagym_EGFR | W3 12115429 |
| sim/abagym_HER2 | W3 12115430 |
| sim/abagym_VEGF | W3 12115432 |
| sim/abcov_kd | W3 12115433 |
| sim/flab_polyreact_jain_bvp | W3 12115447 |
| sim/flab_polyreact_jain_cic | W3 12115448 |

Note: there are also W1 copies of these same 6 with the same 11/12 pattern.
Redundant.

---

## PARTIAL — COMPLETED, real losses, ready to resubmit

All these are complexity (none of these jobs got far enough; sim partials
above are all s12-only).

| Dataset | Best Job | Fresh | Errored | Missing (resubmit these) |
|---|---|---|---|---|
| cx/abcov_kd | W1 12103599 | 17/20 | c10 (noiseInject) | c23, c24 |
| cx/abagym_HER2 | W1 12103611 | 17/20 | c10 | c23, c24 |
| cx/flab_polyreact_jain_cic | W1 12103596 | 17/20 | c10 | c23, c24 |
| cx/flab_polyreact_jain_bvp | W1 12103602 | 17/20 | c10 | c23, c24 |
| cx/flab_immunogenicity_marks | W1 12103601 | 12/20 | c10 | c17, c18, c19, c21, c22, c23, c24 |
| cx/abagym_VEGF | W1 12103613 | 9/20 | c10 | c13, c15, c16, c17, c18, c19, c21, c22, c23, c24 |
| cx/humanness_therapeutic | W2 12111862 | 9/20 | c10 | c13, c15, c16, c17, c18, c19, c21, c22, c23, c24 |

(Note: W3 12115431 cx/abagym_VEGF and W3 12115464 cx/humanness_therapeutic
are RUNNING and at 9 fresh / last_done=c11 — same progress as the
COMPLETED partials. If they finish without bus-erroring, they'll cover
these gaps. But based on Wave 1 evidence, the bus error tends to hit at
c22→c23. Wait and see.)

### sim/abagym_Ang2 — odd one out

| Dataset | Best Job | Fresh | Pattern |
|---|---|---|---|
| sim/abagym_Ang2 | W1 12103674 (finished_then_bus) | 11/12 | missing=**s03** (not s12!) |

This is different from the rest. The job finished all 12, bus-errored on
teardown, but `s03_feature_concentration` didn't write its JSON.
**Investigate before resubmitting** — see "Things to investigate" below.

---

## EMPTY — COMPLETED jobs with zero output (Wave 1 bus errors)

These are pure losses. All bus_mid_run in precompute / fold 0 pruning.

**Complexity (8 datasets):**
- cx/herg_ki (12103583)
- cx/ddg_combined (12103603)
- cx/flab_binding_koenig (12103604)
- cx/flab_expression_koenig (12103605)
- cx/flab_binding_adams (12103606)
- cx/flab_expression_adams (12103607)
- cx/flab_binding_phillips_h3 (12103608)
- cx/abagym_lysozyme (12103612)
- cx/tz_her2_multiclass (W2 12111861)

**Similarity (38 datasets):**
- All molecular regression: herg_ki, caco2, logd, qm9, adme_hlm, adme_rlm, adme_mdr1
- All molecular clf: herg
- All DTA: kiba, bindingdb_kd
- Reaction: buchwald_hartwig
- DDG: (W1 12103668 still RUNNING — not in this EMPTY list)
- Antibody flab regression: flab_thermo_tresanco, flab_thermo_jain,
  flab_aggregation_jain_hicrt, flab_binding_koenig, flab_binding_adams,
  flab_expression_adams
- Ginkgo: ginkgo_acsins, ginkgo_hic, ginkgo_titer; partial:
  ginkgo_heparin (4/12, see below)
- TCGA: tcga_luad_vs_lusc, tcga_brca_pam50
- All 17 nucleotide (nt_*)
- All 4 gb_* genomic benchmarks
- Ab clf: tz_her2_multiclass

**Also EMPTY but a different cause — 3 sim Wave 3 jobs `python_error` in 4:38:**
- sim/gb_demo_human_or_worm (12115453)
- sim/gb_human_enhancers_cohn (12115455)
- sim/gb_human_nontata_promoters (12115457)

These crashed immediately — investigate before resubmitting (see below).

### sim/ginkgo_heparin (partial EMPTY)

`12103684` ran 7:34, got fresh=4, missing=s05…s13. Bus-errored after
s04_activity_cliffs. Wave 3 resubmit 12115459 is RUNNING and at 4 fresh
again with last_done=s04 — about to enter the same danger zone. Watch
this one.

---

## RUNNING — every job's actual progress

All RUNNING jobs are in fold 0 of `precompute_hybrid_searches`. No
per-experiment JSONs written yet. Progress markers below are from the
audit `prog:` column.

### Wave 1 RUNNING (3+ days elapsed each, in fold 0)

Progress varies; some close to fold 0 done, some only ~30% in:

| Dataset (W1 cx) | Progress |
|---|---|
| caco2 (12103584) | fold 0: 2000/7425 |
| logd (12103585) | pruning, no fold marker |
| qm9 (12103586) | fold 0: 1800/6480 |
| adme_hlm (12103587) | fold 0: 2000/7425 |
| adme_rlm (12103588) | fold 0: 2000/7425 |
| adme_mdr1 (12103589) | fold 0: 1800/7425 |
| herg (12103590) | fold 0: 3000/6480 |
| kiba (12103591) | pruning |
| bindingdb_kd (12103592) | pruning |
| flab_thermo_tresanco (12103593) | fold 0: 5000/6804 (~73%) |
| flab_pk_jain_heprt (12103594) | **no progress markers — possibly stuck** |
| flab_pk_jain_fcrnrt (12103595) | fold 0: 2600/3240 (~80%) |
| flab_aggregation_jain_hicrt (12103597) | **no progress markers** |
| flab_thermo_jain (12103598) | **no progress markers** |
| flab_aggregation_jain (12103600) | fold 0: 3600/3780 (~95%) |
| abagym_Ang2 (12103609) | last_done=c15, fresh=10 ✓ already past precompute! |
| abagym_EGFR (12103610) | last_done=c11, fresh=9 ✓ already past precompute! |
| ginkgo_acsins/colloidal/heparin/hic/pr_cho/titer (12103617–22) | **no progress markers** |
| buchwald_hartwig (12103623) | **no progress markers** |
| bh_doyle_plates (12103624) | pruning |

| Dataset (W1 sim) | Progress |
|---|---|
| flab_pk_jain_heprt (12103659) | **no progress markers** |
| flab_pk_jain_fcrnrt (12103660) | fold 0: 2600/3240 (~80%) |
| flab_aggregation_jain (12103665) | fold 0: 3600/3780 (~95%) |
| ddg_combined (12103668) | **no progress markers** |
| flab_expression_koenig (12103670) | fold 0: 1000/7425 |
| flab_binding_phillips_h3 (12103673) | fold 0: 1000/7425 |
| abagym_lysozyme (12103677) | fold 0: 4200/5292 (~79%) |
| ginkgo_colloidal (12103683) | **no progress markers** |
| ginkgo_pr_cho (12103686) | **no progress markers** |
| bh_doyle_plates (12103689) | pruning |
| nt_promoter_tata (12103706) | pruning |

**The "no progress markers" rows are concerning.** The classifier looks at
the last ~200 lines of the log and finds nothing matching a known stage
marker. Could be stuck, could be slow IO, could be a log buffer issue.
Worth tailing one to check.

### Wave 2 RUNNING (~2.5–3 days elapsed; complexity-only batch)

Wave 2 is the manual resubmit of the W1-cancelled datasets. All in
fold 0: NNN/5724 (smaller search). Progress ranges from 600 to 5600 out
of 5724:

| W2 progress band | Count | Examples |
|---|---|---|
| fold 0 ≥ 5000/5724 (~88%+) | 4 | gb_human_enhancers_cohn 5200, nt_promoter_no_tata 5000, nt_H3K4me3 5600, nt_enhancers 5200 |
| fold 0 3000–5000 | 8 | various nt_*, gb_* |
| fold 0 < 3000 | 6 | nt_H3K4me1 1000, nt_enhancers_types 600, nt_splice_sites_acceptors 1200, nt_splice_sites_donors 2600, gb_demo_coding 1000, nt_H3K27me3 2200, nt_H3K36me3 2400 |
| unknown / no markers | 4 | nt_promoter_tata, tz_her2_binary, tz_her2_multiclass (W2 12111861 actually COMPLETED bus_mid), tcga_brca_pam50 |
| COMPLETED in W2 | 4 | tz_her2_multiclass (bus, empty), humanness_therapeutic (bus, partial 9/20), sim/tz_her2_multiclass (bus, empty), sim/humanness_therapeutic (clean_finish OK) |

### Wave 3 RUNNING (~2 days elapsed; mix of cx + sim resubmits)

Same fold 0: NNN/5724 (and a few different search sizes for non-nt
datasets). Notable rows:

| Dataset | Job | Progress | Notes |
|---|---|---|---|
| cx/abagym_VEGF | 12115431 | unknown markers, fresh=9 | already past precompute (uses prev cache?) |
| cx/humanness_therapeutic | 12115464 | last_done=c11, fresh=9 | already running experiments |
| sim/ginkgo_heparin | 12115459 | last_done=s04, fresh=4 | identical to W1 partial — about to hit same bus error region |
| sim/nt_H3K4me3 | 12115484 | fold 0: 5600/5724 (~98%) | very close to fold 0 done |
| sim/tz_her2_multiclass | 12115507 | fold 0: 6800/7425 (~92%) | very close |
| cx/tz_her2_multiclass | 12115506 | fold 0: 6800/7425 (~92%) | very close |
| sim/nt_promoter_all | 12115492 | fold 0: 5000/5724 | close |
| Several | fold 0 in 4000-5000 range | mid-late | watch |
| Several | fold 0 < 2000 | early | many days from finishing fold 0 |

---

## Things to investigate (before more resubmits)

### 1. The 3 four-minute Wave 3 python_error sim jobs

```
head -50 slurm-12115453.out slurm-12115455.out slurm-12115457.out | less
grep -E "Traceback|Error|raise|ImportError" slurm-12115453.out
```

All three died in identical 4:38 elapsed — same root cause almost
certainly. Fix once, resubmit all 3.

### 2. sim/abagym_Ang2 missing s03 (not s12)

Different from the s12 pattern. Job ID 12103674.

```
grep -E "s03|feature_concentration|FAILED" slurm-12103674.out | head -40
```

If s03 has a real bug for this dataset, fix it before any resubmits.

### 3. Wave 1 RUNNING jobs with no progress markers

These have been "RUNNING" for 3+ days but the audit's tail scan finds no
fold/stage signal. Could be stuck (deadlock, swapping), could be very
slow IO. Worth peeking at one:

```
JOB=12103594  # cx_flab_pk_jain_heprt
tail -100 slurm-${JOB}.out
ls -la slurm-${JOB}.out                 # check mtime; if recent → still writing
sstat -j ${JOB} --format=AveRSS,MaxRSS,CPUTimeRAW 2>/dev/null  # memory usage
```

If the log mtime is hours old, the job is hung — scancel + resubmit.

If you see the same on the ginkgo_*  W1 RUNNING jobs (12103617–22) too —
those have no markers either — that's 6 jobs probably stuck.

---

## What to resubmit RIGHT NOW

### Targeted complexity resubmits (PARTIAL → fill in missing)

Submit_dataset.sh accepts a 3rd arg = space-separated experiment IDs.
**Per your scancel rule, paste these one at a time and verify.**

```
sbatch --job-name=cx_abcov_kd_p tests/slurm_scripts/submit_dataset.sh complexity abcov_kd "c23 c24"
sbatch --job-name=cx_abagym_HER2_p tests/slurm_scripts/submit_dataset.sh complexity abagym_HER2 "c23 c24"
sbatch --job-name=cx_flab_polyreact_jain_cic_p tests/slurm_scripts/submit_dataset.sh complexity flab_polyreact_jain_cic "c23 c24"
sbatch --job-name=cx_flab_polyreact_jain_bvp_p tests/slurm_scripts/submit_dataset.sh complexity flab_polyreact_jain_bvp "c23 c24"
sbatch --job-name=cx_flab_immunogenicity_marks_p tests/slurm_scripts/submit_dataset.sh complexity flab_immunogenicity_marks "c17 c18 c19 c21 c22 c23 c24"
```

**Do not resubmit** cx/abagym_VEGF or cx/humanness_therapeutic (partials).
Their Wave 3 resubmits (12115431, 12115464) are already running and at
the same progress point — wait for those.

### Full complexity resubmits (EMPTY, no resubmit running)

```
sbatch --job-name=cx_abagym_lysozyme tests/slurm_scripts/submit_dataset.sh complexity abagym_lysozyme
```

cx/abagym_lysozyme is the only cx EMPTY with no resubmit in flight.

(All other cx EMPTY datasets — herg_ki, ddg_combined, flab_binding_*,
flab_expression_*, flab_binding_phillips_h3 — already have W3 jobs
running. Don't double up.)

### Full similarity resubmits (EMPTY, no resubmit running)

Check before submitting: every dataset below should have NO RUNNING sim
job for it. Verify with `squeue -u $USER --name=sim_<dataset>`.

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

(All other sim EMPTY datasets already have W3 resubmits running.)

### Don't resubmit yet

The 3 four-minute python_error sims (gb_demo_human_or_worm,
gb_human_enhancers_cohn, gb_human_nontata_promoters) — investigate root
cause first, then resubmit.

sim/abagym_Ang2 — investigate s03 first.

---

## How to check progress later

```bash
cd /data/stat-ecr/scat9264/KIRBy
git pull
python tests/audit_full_wave.py --include-running | tee /tmp/audit_$(date +%H%M).txt
diff /tmp/audit_HHMM_old.txt /tmp/audit_HHMM_new.txt | less   # spot what moved
```

**Things to watch for next:**
- Has any RUNNING job logged `[saved` for the first time? (= precompute
  finally finished, per-experiment loop started)
- Have any RUNNING jobs flipped to bus_mid_run? (= we lose them)
- Have the "no progress markers" W1 jobs gained any? (= they're alive)

---

## Open known bugs

| Bug | Where | Action |
|---|---|---|
| Bus error in precompute | Affects every Wave 1 cx + sim. ~50 jobs lost. | User says fixed for new submissions. Wave 1 jobs still vulnerable as they're long-lived. |
| `noiseInject not installed` | c10 self-errors on every complexity dataset (1 errored slot in every PARTIAL/OK cx row). | `pip install noiseInject` if you want c10 results. |
| `s12 no_cliffs_found` | Methodological — antibody data has no chem-style cliffs. Counts as errored, not failed. | None unless you change s12 method. |
| `submit_dataset.sh` masks failures | sacct shows COMPLETED 0:0 for bus errors because bash doesn't propagate `$?`. | Add `set -e` or `exit $?` after the python call. |
| python_error in ~4 min for 3 W3 sim gb_* jobs | New failure mode, root cause unknown | Investigate slurm-12115453/5/7.out before resubmitting |
| sim/abagym_Ang2 missing s03 | Single-experiment crash, not the s12 pattern | Investigate slurm-12103674.out |
