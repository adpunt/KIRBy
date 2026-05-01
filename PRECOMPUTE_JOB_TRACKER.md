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
| 7440291 | dta_davis | TIMEOUT (48h wall limit) |
| 7440292 | dta_kiba | DONE |
| 7440293 | dta_bindingdb_kd | **TIMEOUT** (4d wall limit, 2026-04-11) — end-of-run save only, zero output on disk |
| 7483661 | dta_bindingdb_kd (pre-c18342f heredoc bug) | **CANCELLED** (2026-04-15, same heredoc bug as 7483641-46) |
| 7483718 | dta_bindingdb_kd (post-c18342f, per-rep checkpoint) | **TIMEOUT** (2026-04-18, 1d23h59m, exit 0:0). Died at contact_graph 50/1219 (~58 min/protein). 17 protein reps + 7 drug reps cached as `.npy`; protein_contact_graph + protein_usr_shape + protein_pocket + protein_pocket_composition + protein_esm2_pocket never computed. Superseded by 7515602 (post-d829df1 contact_graph speedup). |
| 7467954 | dta_davis (resubmit, --skip-existing) | **TIMEOUT** (2026-04-15, 48h wall) |
| 7483719 | dta_davis (post-c18342f + a81b0eb) | **TIMEOUT** (4d wall, 2026-04-21, ExitCode 0:0). `davis.json` written at 12:02 (67KB): 24 baselines + 1 FT (`protein_esm2_lora`) populated. 4 tier-1 reps absent (`alphafold_db`, `descriptors`, `esmfold`, `pocket_composition`) — expected, not produced by standard pipeline. No fill run needed. |
| 7457314 | dta_kiba_fill (--skip-existing) | FAILED (--wrap bug) |
| 7483635 | dta_kiba_fill (pre-c18342f) | CANCELLED |
| 7483717 | dta_kiba_fill (post-c18342f) | **COMPLETED** (23min, 2026-04-16) — **BUT destroyed kiba.json** (69KB→583B) due to merge bug in a81b0eb. Fixed in d1c45b9. |
| 7496790 | dta_kiba (full re-run, post-d1c45b9 merge fix) | **COMPLETED** (17h11m, 2026-04-18). Output `tests/results/phase5_baselines/dta/kiba.json`. Several ESM-family reps showed large negative R² (e.g. esm2 R²=-1.58, esmc R²=-2.37) but CIs in normal range — flag for analysis but not a job failure. |
| 7515602 | dta_bindingdb_kd (post-d829df1, hand-rolled --wrap) | **FAILED** (14s, ExitCode 127:0, 2026-04-20). Long inline `--wrap` line-split by SLURM, python invocation orphaned from args (same bug class as 7457285/86, 7467966-9). Plus server git was on divergent branches → silent `git pull 2>/dev/null` swallowed the failure. Reset server to origin/main + dedicated heredoc script (commit 42a22c4). Superseded by 7515675. |
| 7515675 | dta_bindingdb_kd (post-42a22c4, heredoc script v1) | **FAILED** (argparse "unrecognized arguments" + blank-line args). Heredoc backslash line-continuation (`\\<LF>`) garbled between heredoc → sbatch → compute-node bash. Same bug class as 7483641-46. Fixed in db6d769 (collapse python invocation to single line). Superseded by 7516299. |
| 7516299 | dta_bindingdb_kd (post-db6d769, single-line python) | **FAILED** (1m23s, ExitCode 2:0, 2026-04-20). Stale job script: `bash submit_dta_bindingdb.sh` was run BEFORE the db6d769 fix landed on the server, so sbatch captured the OLD `\\` continuation heredoc. In-job `git pull` updated the python source but the SLURM job script itself was already locked at submission time. Superseded by 7520843 after `git pull` on login node. |
| 7520843 | dta_bindingdb_kd (post-login-pull, db6d769 in script) | **COMPLETED** (15h34m, 2026-04-21, ExitCode 0:0). `bindingdb_kd.json` written at 08:29 (70KB). Contact_graph speedup (d829df1) confirmed working — full run under 24h wall. |

### Antibody CLF

| Job ID | Name | Status |
|--------|------|--------|
| 7440294 | abclf_tz | FAILED (KeyError: 'models', --reps all bug) |
| 7440295 | abclf_humanness | FAILED (KeyError: 'models', --reps all bug) |
| 7457285 | abclf_tz (resubmit, bug fixed) | **FAILED** (ExitCode 126:0, 7min — sbatch --wrap line-splitting bug; `python` prefix orphaned from script path at slurm_script line 5) |
| 7457286 | abclf_humanness (resubmit, bug fixed) | **FAILED** (same — 126:0, 6min) |
| 7483631 | abclf_tz_her2_binary (pre-c18342f) | **CANCELLED** (2026-04-15, bad heredoc) |
| 7483632 | abclf_tz_her2_multiclass (pre-c18342f) | **CANCELLED** |
| 7483633 | abclf_humanness_species (pre-c18342f) | **CANCELLED** |
| 7483634 | abclf_humanness_therapeutic (pre-c18342f) | **CANCELLED** |
| 7483713 | abclf_tz_her2_binary (post-c18342f) | **COMPLETED** (3h51m, 2026-04-16) |
| 7483714 | abclf_tz_her2_multiclass (post-c18342f) | **COMPLETED** (8h22m, 2026-04-16) |
| 7483715 | abclf_humanness_species (post-c18342f) | **COMPLETED** (5h26m, 2026-04-16) — **output file missing from disk** despite "Saved" in slurm log. Resubmitted as 7496824. |
| 7483716 | abclf_humanness_therapeutic (post-c18342f) | **COMPLETED** (38min, 2026-04-16) |
| 7496824 | abclf_humanness_species (resubmit) | **FAILED** (ExitCode 127, 4min — long --reps argument split across slurm_script lines by SLURM. Fixed by using `REPS=` shell variable.) |
| 7498632 | abclf_humanness_species (resubmit 2, REPS= variable) | **COMPLETED** (4h04m, 2026-04-18). Output `tests/results/opig/clf/humanness_species.json`. **DATASET CUT (2026-04-19)** — most reps hit 0.99+ AUROC (germline V/J usage ≈ trivially separable per species), and igbert/igbert_cls/mint/xtrimopglm collapsed to majority-class 0.679/0.549 across all folds (paired-chain models incompatible with VH-only input). Not informative; remove from precompute config + tier table. |
| 7483636 | abclf_tz_her2_binary (DUPLICATE) | **CANCELLED** (2026-04-15) |
| 7483637 | abclf_tz_her2_multiclass (DUPLICATE) | **CANCELLED** |
| 7483638 | abclf_humanness_species (DUPLICATE) | **CANCELLED** |
| 7483639 | abclf_humanness_therapeutic (DUPLICATE) | **CANCELLED** |
| 7483640 | dta_kiba_fill (DUPLICATE) | **CANCELLED** |

### SchNet (molecular)

| Job ID | Name | Status |
|--------|------|--------|
| 7440325 | schnet_logd | **FAILED** (SLURM exit 0:0 but experiment broken, 4h17m, 2026-04-12) — baselines ran but all 3 FT reps (ft_schnet, lora_chemberta, lora_molformer) emitted `FT FACTORY ERROR: No data left in file` from `np.load(cache_path)` on every fold. Root cause: fc36c6c np.savez suffix bug (zero-byte cache entries). |
| 7440326 | schnet_herg_ki | **FAILED** (1h50m, 2026-04-12) — same FT cache corruption |
| 7440327 | schnet_caco2 | **FAILED** (1h13m, 2026-04-12) — same FT cache corruption |
| 7483721 | schnet_logd (post-fc36c6c) | **COMPLETED** (1h05m, 2026-04-16) — real lora_chemberta/lora_molformer scores, **ft_schnet=NaN** (torch_cluster not CUDA-compiled) |
| 7483722 | schnet_herg (post-fc36c6c) | **FAILED** (7min, 2026-04-16, ExitCode 1:0) — TDC loader crash |
| 7483723 | schnet_caco2 (post-fc36c6c) | **COMPLETED** (45min, 2026-04-16) — same: ft_schnet=NaN (CUDA issue) |
| 7498292 | schnet_logd (post-b59c11e CPU fallback) | **COMPLETED** (3h26m, 2026-04-18). Real ft_schnet finally landed: ft_schnet=0.609 (ridge), lora_chemberta=0.757, lora_molformer=0.565. CPU fallback works. |
| 7498293 | schnet_herg (post-b59c11e + 67803ab FLuID loader) | **FAILED** (7min, 2026-04-18, ExitCode 1:0). FLuID `hERG_lhasa_training.pkl` not on server — `data_dir='data/fluid'` was hardcoded and never existed. Fixed in d3e2cc1 (env-var resolution + multiple fallback paths) + user scp'd pickles to `/data/stat-cadd/scat9264/lhasa_data/`. Superseded by 7515601. |
| 7498294 | schnet_caco2 (post-b59c11e) | **COMPLETED** (1h06m, 2026-04-18). Real ft_schnet via CPU fallback. |
| 7515601 | schnet_herg (post-d3e2cc1 + scp'd FLuID) | **COMPLETED** (4h05m, 2026-04-20). 22 frozen baselines + 3 FT reps (3-fold screening). Winner mhg_gnn+xgb=0.7712 acc. ft_schnet=0.684 (works post-CPU-fallback but not competitive vs frozen). `unimol_v2` collapsed to majority class (0.533) — flag for investigation. Tier table NOT assigned (3-fold screening is not publication-grade). |

### Nucleotide (genomics)

| Job ID | Name | Status |
|--------|------|--------|
| 7440301 | nuc_nt_promoter_all | DONE (partial, pre-fix) |
| 7440302 | nuc_nt_promoter_tata | DONE (partial, pre-fix) |
| 7440303 | nuc_nt_promoter_no_tata | DONE (partial, pre-fix) |
| 7440304 | nuc_nt_enhancers | CANCELLED (pre-fix, replaced by 7459731) |
| 7440305 | nuc_nt_enhancers_types | CANCELLED (pre-fix, replaced by 7459732) |
| 7440306 | nuc_nt_splice_sites_all | DONE (partial, pre-fix) |
| 7440307 | nuc_nt_splice_sites_acceptors | CANCELLED (pre-fix, replaced by 7459733) |
| 7440308 | nuc_nt_splice_sites_donors | CANCELLED (pre-fix, replaced by 7459734) |
| 7459731 | nuc_nt_enhancers (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7459732 | nuc_nt_enhancers_types (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7459733 | nuc_nt_splice_sites_acceptors (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7459734 | nuc_nt_splice_sites_donors (resubmit, --skip-existing, post-fix) | FAILED (--models arg parse bug) |
| 7467966 | nuc_nt_enhancers (resubmit 2) | **FAILED** (ExitCode 126:0, 6min — sbatch --wrap line-splitting bug; `mkdir -p` orphaned from operand at slurm_script line 5) |
| 7467967 | nuc_nt_enhancers_types (resubmit 2) | **FAILED** (same — 126:0, 6min) |
| 7467968 | nuc_nt_splice_sites_acceptors (resubmit 2) | **FAILED** (same — 126:0, 6min) |
| 7467969 | nuc_nt_splice_sites_donors (resubmit 2) | **FAILED** (same — 126:0, 6min) |
| 7483641 | nuc_nt_enhancers (heredoc resubmit 3, pre-c18342f) | **FAILED** (2026-04-15, heredoc `\\<LF>` parsing bug — bash joined continuation lines, argparse got empty-string args) |
| 7483642 | nuc_nt_enhancers_types (pre-c18342f) | **FAILED** (same) |
| 7483643 | nuc_nt_splice_sites_acceptors (pre-c18342f) | **FAILED** (same — user confirmed) |
| 7483644 | nuc_nt_splice_sites_donors (pre-c18342f) | **FAILED** (same) |
| 7483707 | nuc_nt_enhancers (post-c18342f) | **COMPLETED** (3m39s, 2026-04-15) — gena_lm still 32000D (old cache loaded pre-3a5b5d5). Needs gena_refix. |
| 7483708 | nuc_nt_enhancers_types (post-c18342f) | **COMPLETED** (3m17s) — same, gena_lm 32000D |
| 7483709 | nuc_nt_splice_sites_acceptors (post-c18342f) | **COMPLETED** (3m29s) — same |
| 7483710 | nuc_nt_splice_sites_donors (post-c18342f) | **COMPLETED** (3m18s) — same |
| 7440309 | nuc_nt_H2AFZ | DONE (2026-04-12) — **gena_lm CONTAMINATED (32000D logits)**, superseded by 7485363 refix |
| 7440310 | nuc_nt_H3K27ac | DONE — **gena_lm CONTAMINATED**, superseded by 7485364 refix |
| 7440311 | nuc_nt_H3K27me3 | DONE — **gena_lm CONTAMINATED**, superseded by 7485365 refix |
| 7440312 | nuc_nt_H3K36me3 | DONE — **gena_lm CONTAMINATED**, superseded by 7485366 refix |
| 7440313 | nuc_nt_H3K4me1 | DONE — **gena_lm CONTAMINATED**, superseded by 7485367 refix |
| 7440314 | nuc_nt_H3K4me2 | DONE (1d14h, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485368 refix |
| 7440315 | nuc_nt_H3K4me3 | DONE (1d3h, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485369 refix |
| 7440316 | nuc_nt_H3K9ac | DONE (1d11h, 2026-04-13) — **gena_lm CONTAMINATED**, superseded by 7485370 refix |
| 7440317 | nuc_nt_H3K9me3 | DONE (1d20h, 2026-04-13) — **gena_lm CONTAMINATED**, superseded by 7485371 refix |
| 7440318 | nuc_nt_H4K20me1 | DONE (22h, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485372 refix |
| 7440319 | nuc_gb_human_nontata_promoters | DONE (10h38m, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485377 refix |
| 7440320 | nuc_gb_human_enhancers_cohn | DONE (16h35m, 2026-04-12) — **gena_lm CONTAMINATED**, superseded by 7485378 refix |
| 7440321 | nuc_gb_demo_coding_vs_intergenomic | **OUT_OF_MEMORY** (1h44m, 2026-04-12) — 128G insufficient, resubmit with 256G |
| 7440322 | nuc_gb_demo_human_or_worm | **OUT_OF_MEMORY** (1h46m, 2026-04-12) — 128G insufficient, resubmit with 256G |
| 7483645 | nuc_gb_demo_coding_vs_intergenomic (pre-c18342f) | **FAILED** (same heredoc bug) |
| 7483646 | nuc_gb_demo_human_or_worm (pre-c18342f) | **FAILED** (same heredoc bug) |
| 7483711 | nuc_gb_demo_coding (post-c18342f, 256G) | **CANCELLED** (12h42m in, 2026-04-16) |
| 7483712 | nuc_gb_demo_human_or_worm (post-c18342f, 256G) | **CANCELLED** (12h42m) |
| 7496807 | nuc_gb_demo_coding (resubmit, 256G) | **COMPLETED** (6h24m, 2026-04-18). Winner: nt_v2_250m+mlp=0.8392 MCC. |
| 7496808 | nuc_gb_demo_human_or_worm (resubmit, 256G) | **COMPLETED** (6h25m, 2026-04-18). |

### Nucleotide gena_lm refix (2026-04-15)

Target: fix broken gena_lm entries in 16 completed nucleotide JSONs.
Root cause: `_embed_gena_lm` was returning 32000D LM head logits
instead of 1024D hidden states (fixed in 3a5b5d5). Each job runs
`run_baseline_survey.py <ds> --reps gena_lm --n-folds 5 --output
tests/results/nucleotide_survey/`. No --skip-existing: the merge
logic in run_baseline_survey.py lines 1243-1264 overwrites ONLY the
gena_lm entry while preserving all other reps. Preflight check
inside each job aborts if any stale `nuc_*/gena_lm.npy` cache files
remain (they should have been deleted via rm before submission).

Resource: long+GPU+64G+6h per job.

| Job ID | Name | Status |
|--------|------|--------|
| 7485363 | gena_refix_nt_H2AFZ | **COMPLETED** (34min, 2026-04-15) — confirmed 1024D gena_lm |
| 7485364-78 | gena_refix (15 remaining) | **FAILED** (instant, ExitCode 2) — preflight false-positive on newly-written gena_lm.npy. Fixed in 0913452 (preflight removed). |

**Round 2 (post-0913452, no preflight):**

| Job ID | Name | Status |
|--------|------|--------|
| 7496791 | gena_refix_nt_H2AFZ | **COMPLETED** (31min, 2026-04-17) |
| 7496792 | gena_refix_nt_H3K27ac | **COMPLETED** (35min) |
| 7496793 | gena_refix_nt_H3K27me3 | **COMPLETED** (35min) |
| 7496794 | gena_refix_nt_H3K36me3 | **COMPLETED** (36min) |
| 7496795 | gena_refix_nt_H3K4me1 | **COMPLETED** (34min) |
| 7496796 | gena_refix_nt_H3K4me2 | **COMPLETED** (31min) |
| 7496797 | gena_refix_nt_H3K4me3 | **COMPLETED** (19min) |
| 7496798 | gena_refix_nt_H3K9ac | **COMPLETED** (22min) |
| 7496799 | gena_refix_nt_H3K9me3 | **COMPLETED** (26min) |
| 7496800 | gena_refix_nt_H4K20me1 | **COMPLETED** (36min) |
| 7496801 | gena_refix_nt_promoter_all | **COMPLETED** (25min) |
| 7496802 | gena_refix_nt_promoter_tata | **COMPLETED** (8min) |
| 7496803 | gena_refix_nt_promoter_no_tata | **COMPLETED** (26min) |
| 7496804 | gena_refix_nt_splice_sites_all | **COMPLETED** (40min) |
| 7496805 | gena_refix_gb_human_nontata_promoters | **COMPLETED** (30min) |
| 7496806 | gena_refix_gb_human_enhancers_cohn | **COMPLETED** (27min) |

**Round 3 (2026-04-17): 4 datasets missed from rounds 1-2 (the freshly-surveyed nt_enhancers/types + splice_acceptors/donors that ran pre-3a5b5d5 with old cache):**

| Job ID | Name | Status |
|--------|------|--------|
| 7499338 | gena_refix_nt_enhancers | **COMPLETED** (~30min, 2026-04-17) |
| 7499339 | gena_refix_nt_enhancers_types | **COMPLETED** (44min, 2026-04-18) |
| 7499340 | gena_refix_nt_splice_sites_acceptors | **COMPLETED** (31min, 2026-04-18). Winner: gena_lm+ridge=0.0317 MCC (weak — confirms classicals dominate this task). |
| 7499341 | gena_refix_nt_splice_sites_donors | **COMPLETED** (30min, 2026-04-18) |

All 20 nucleotide datasets (16 + 4) now have corrected 1024D gena_lm. Tier tables for the 6 newly-completed nucleotide datasets (4 round-3 + 2 gb_demo) updated in PIPELINE_READINESS_TODO 2026-04-20 (commits a-f).

### Precompute for the 6 new nucleotide datasets (2026-04-20)

Submitted via `tests/slurm_scripts/submit_precompute_new_nucleotide.sh` (commit b58b1d0). Registry entries added (`src/kirby/registry.py`).

**4 nt_\* datasets (n=33K, default 6 models, long+128G+2d):**

| Job ID | Name |
|--------|------|
| 11794061 | pc_nt_enhancers_all_standard |
| 11794062 | pc_nt_enhancers_all_distance |
| 11794063 | pc_nt_enhancers_types_all_standard |
| 11794064 | pc_nt_enhancers_types_all_distance |
| 11794065 | pc_nt_splice_sites_acceptors_all_standard |
| 11794066 | pc_nt_splice_sites_acceptors_all_distance |
| 11794067 | pc_nt_splice_sites_donors_all_standard |
| 11794068 | pc_nt_splice_sites_donors_all_distance |

**2 gb_demo_\* datasets (n=100K, drop SVM, long+256G+4d):**

| Job ID | Name |
|--------|------|
| 11794069 | pc_gb_demo_coding_vs_intergenomic_seqs_all_standard |
| 11794070 | pc_gb_demo_coding_vs_intergenomic_seqs_all_distance |
| 11794071 | pc_gb_demo_human_or_worm_all_standard |
| 11794072 | pc_gb_demo_human_or_worm_all_distance |

`--models ridge,rf,xgboost,lightgbm,mlp` (SVM dropped due to O(n²-n³) kernel cost at n=100K).

### Phase B FT precompute (HTC GPU, 2026-04-20)

Submitted via `tests/slurm_scripts/submit_ft_precompute.sh` (commit 2a8ec09). Goal: populate per-fold `.npz` cache at `$KIRBY_CACHE_DIR/finetuned_reps/` (`/data/stat-cadd/scat9264/.cache/kirby/finetuned_reps/`) so downstream experiments can re-use FT embeddings via `load_or_compute()` without re-running fine-tuning. TCGA skipped per user.

| Job ID | Name | Resource | Coverage |
|--------|------|----------|----------|
| 7516559 | ftpc_molecular | long, 128G, 24h, GPU | **COMPLETED** (14m57s, 2026-04-20). 5 of 8 datasets had FT reps configured in `DATASET_REPS` (logd: 3 reps, qm9: 1, caco2/herg_ki/adme_*: subset). Cached `lora_molformer + contrastive_molformer + gnn_graphmvp` per fold per dataset. The 3 datasets without ft_reps in `DATASET_REPS` were skipped silently (expected). **Does NOT cover** `lora_chemberta` or `ft_schnet` (those are in `make_ft_factories` only). |
| 7516560 | ftpc_antibody | long, 128G, 4d, GPU | **FAILED** (8s, ExitCode 1:0, 2026-04-20). Git ref-lock race: 5 jobs started concurrently, all ran `git pull --ff-only` at startup → `cannot lock ref 'refs/remotes/origin/main'`. Superseded by 7520844 (no `git pull` in wrap; user pulls on login first). |
| 7516561 | ftpc_dta_davis | long, 256G, 4d, GPU | **COMPLETED but FT NaN** (6h27m, 2026-04-20). Baselines populated; protein_esm2_lora returned NaN on every fold due to **CUDA OOM during holdout eval** (full 2400-sample holdout passed through ESM-2 650M in one forward → 44 GiB A40 OOM). Cache NOT poisoned (factory caught exception, no .npz written). Fixed in 7b128aa: bf16 autocast + gradient checkpointing + mini-batched holdout eval + dynamic per-batch tokenization + max_length 1024→512 + `_FACTORY_VERSION` v1→v2. Superseded by 7521193. |
| 7516562 | ftpc_dta_kiba | long, 256G, 4d, GPU | **CANCELLED** (would have hit same OOM). Superseded by 7521194. |
| 7516563 | ftpc_dta_bindingdb_kd | long, 256G, 4d, GPU | **CANCELLED** (same OOM risk + hybrid concurrency). Superseded by 7521196. |
| 7520844 | ftpc_antibody (post-race-fix, no git pull in wrap) | **COMPLETED** (2h08m, 2026-04-21, ExitCode 0:0). Antibody FT reps cached per fold to `$KIRBY_CACHE_DIR/finetuned_reps/`. |
| 7521193 | ftpc_dta_davis_v2 (post-7b128aa OOM fix) | **COMPLETED** (9h04m, 2026-04-21, ExitCode 0:0). 9× 153MB `.npz` files written (Apr 21 07:46–15:39) = davis protein_esm2_lora per fold. OOM fix confirmed — v2 factory (bf16 + grad ckpt + mini-batched holdout) works on 44GiB A40. Real FT scores present in `davis.json`. |
| 7521194 | ftpc_dta_kiba_v2 (post-7b128aa) | **COMPLETED** (13h54m, 2026-04-22, ExitCode 0:0). v2 OOM fix held. Pull updated `kiba.json`. |
| 7521196 | ftpc_dta_bindingdb_kd_v2 (post-7b128aa) | **COMPLETED** (16h55m, 2026-04-22, ExitCode 0:0). v2 OOM fix held. Pull updated `bindingdb_kd.json`. |

### FT precompute fill-in (2026-04-21)

Gap found while auditing FT coverage: `adme_hlm` + `adme_rlm` had no `ft_reps`
configured in `DATASET_REPS`, so 7516559 (ftpc_molecular) skipped them. User
confirmed D1: add the 3 successful FT reps (`lora_molformer`,
`contrastive_molformer`, `gnn_graphmvp`) to both. `adme_solubility` is DROPPED
(R²=0.32 ceiling per PIPELINE_READINESS_TODO:313), excluded.

Config change: commit `fac257e` (tests/experiments/complexity/complexity_helpers.py).
Submit script: `tests/slurm_scripts/submit_ft_precompute_adme.sh`.

| Job ID | Name | Status |
|--------|------|--------|
| 7528901 | ftpc_adme_hlm | **COMPLETED** (11m52s, 2026-04-22, ExitCode 0:0). 3 FT reps cached per fold. |
| 7528902 | ftpc_adme_rlm | **COMPLETED** (11m47s, 2026-04-22, ExitCode 0:0). 3 FT reps cached per fold. |
| 7529104 / 7529105 | ftpc_adme_hlm/rlm (dup) | **CANCELLED** (instant, accidental resubmit). |

### DTA baseline re-survey — importance-filter fix (2026-04-23)

Re-runs davis/kiba/bindingdb_kd under commit 506e4bb (per-fold importance
filter for reps with dim > 500, keep top 200 by LightGBM importance; also
drops MLP from DTA protein-rep pool). Fix motivated by R² blowups like
`drug_mordred+ridge R²=-40` and `drug_mordred+mlp R²=-2247` on bindingdb_kd.
FT cache from the v2 precompute jobs (7521193/4/6) is reused — no refit.
Submit script: `tests/slurm_scripts/submit_dta_baseline_resurvey.sh`.

| Job ID | Name | Partition | Status |
|--------|------|-----------|--------|
| 7551136 | dtaresurvey_davis | long, 256G, 4d, GPU | **SUBMITTED 2026-04-22 evening** |
| 7551137 | dtaresurvey_kiba | long, 256G, 4d, GPU | **SUBMITTED 2026-04-22 evening** |
| 7551138 | dtaresurvey_bindingdb_kd | long, 256G, 4d, GPU | **SUBMITTED 2026-04-22 evening** |

Expected runtime: 9–17h per dataset (similar to v2 timings). Outputs
overwrite `tests/results/phase5_baselines/dta/{davis,kiba,bindingdb_kd}.json`.
Pull all 3 JSONs when complete, then do DTA tier review.

**Re-run** of 7551136-38 happened automatically (jobs 7542406-08 first, then
7551136-38 second). All 3 latest runs **COMPLETED** 2026-04-25:
- 7551136 davis: 1h49m
- 7551137 kiba: 2h50m
- 7551138 bindingdb_kd: 2h40m

### Davis baseline investigation — drug-rep ceiling test (2026-04-28)

Tests whether 9 drug reps tied at CI=0.7567 on davis are degenerate
per-drug-mean fitting (every drug appears in train under cold-protein
splits) vs real learning. Trivial drug-mean predictor + per-fold ridge
predictions for {drug_avalon, drug_chemberta, drug_ecfp4, drug_maccs} +
pairwise correlation. Script: `tests/scripts/davis_baseline_investigation.py`.

| Job ID | Name | Partition | Status |
|--------|------|-----------|--------|
| 7568428 | davis_invest | short, 32G, 2h, no GPU | **SUBMITTED 2026-04-28** |

Output: `tests/results/phase5_baselines/dta/davis_investigation.json`.
Determines whether the 9 drug reps stay in the davis general tier or get
downgraded to augmentation/dropped.

### C02 frozen vs frozen+FT comparison — full run (2026-04-23)

Script: `tests/slurm_scripts/submit_c02_full.sh`. 12 jobs, all `--mode full
--reps all`, one GPU per dataset. Bug test skipped; FT cache confirmed warm
from the ftpc_* jobs above. Outputs
`tests/results/complexity_c02/full/<dataset>_c02_complexity_matching_grid.json`.

| Job ID | Dataset | Partition |
|--------|---------|-----------|
| 7548418 | c02_caco2 | long, 128G, 4d, GPU |
| 7548419 | c02_logd | long, 128G, 4d, GPU |
| 7548420 | c02_qm9 | long, 128G, 4d, GPU |
| 7548421 | c02_herg_ki | long, 128G, 4d, GPU |
| 7548422 | c02_adme_mdr1 | long, 128G, 4d, GPU |
| 7548423 | c02_adme_hlm | long, 128G, 4d, GPU |
| 7548424 | c02_adme_rlm | long, 128G, 4d, GPU |
| 7548425 | c02_abcov_kd | long, 128G, 4d, GPU |
| 7548426 | c02_flab_binding_phillips_h3 | long, 128G, 4d, GPU |
| 7548427 | c02_abagym_Ang2 | long, 128G, 4d, GPU |
| 7548428 | c02_abagym_EGFR | long, 128G, 4d, GPU |
| 7548429 | c02_abagym_HER2 | long, 128G, 4d, GPU |

### OPIG C7 — CGR vs SMILES comparison (2026-04-23)

Script: `tests/slurm_scripts/submit_cgr_comparison.sh all`. CGR-SMILES
checkpoint from job 7483446 (2026-04-15) is on disk; comparison tests
whether CGR reps add unique signal beyond SMILES-based reps for reaction
yield prediction. Outputs `tests/results/comparisons/opig_cgr_comparison_<ds>.json`.

| Job ID | Name | Dataset | Partition | Status |
|--------|------|---------|-----------|--------|
| 7548433 | cgr_bh | buchwald_hartwig | long, 256G, 24h | **TIMEOUT** (24h wall, 2026-04-24). Superseded by 7568447. |
| 7548434 | cgr_doyle | bh_doyle_plates | long, 256G, 24h | **TIMEOUT** (24h wall, 2026-04-24). Superseded by 7568448. |
| 7548435 | cgr_suzuki | suzuki | long, 256G, 24h | **COMPLETED** (9h18m, 2026-04-23). |
| 7568447 | cgr_bh | buchwald_hartwig | long, 256G, **4d** | **SUBMITTED 2026-04-28** (resubmit, 4d wall via `submit_cgr_comparison_resubmit.sh`). |
| 7568448 | cgr_doyle | bh_doyle_plates | long, 256G, **4d** | **SUBMITTED 2026-04-28** (resubmit, 4d wall). |

### CGR-SMILES pretraining (Phase 4)

Single job: corpus generation (~1M USPTO → CGR-SMILES) + continued MLM pretraining (50K steps).
Partition: long, 128G, 8h, 16 CPU, no GPU.
Output: `/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/`

| Job ID | Name | Stage | Status |
|--------|------|-------|--------|
| 11747734 | cgr_smiles | corpus + pretrain | FAILED — chython not installed, 0/392K conversions. Fixed in 9fa01f8. |
| 7467971 | cgr_smiles | corpus + pretrain | FAILED — chython installed but 0/392K conversions. Empty corpus guard caught it. Diagnostics added in 4b7d9de. |
| 11748808 | cgr_smiles | corpus + pretrain | FAILED — CGRContainer.__len__ missing, smoke test caught it. Fixed in f0b5e86. |
| 11749515 | cgr_smiles | corpus + pretrain | **FAILED** (2026-04-11) — corpus stage OK (392324/392533, 99.9%), pretrain stage crashed: `AttributeError: 'AdamW' object has no attribute 'train'` in accelerate/optimizer.py line 128. Root cause: transformers Trainer → accelerate AcceleratedOptimizer.train() → calls inner optimizer.train() which torch.optim.AdamW does not implement (accelerate/transformers version mismatch). Fixed in 2c367d6 (manual training loop). |
| 7483446 | cgr_pretrain | pretrain only | **COMPLETED** (10min56s, 2026-04-15) — checkpoint at `/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint/` |

### Reaction re-survey — UNBLOCKED

CGR pretrain complete. Re-survey with 41 reps (38 original + 3 CGR) not yet submitted.
Existing reaction JSONs are from Mar 19-26 (pre-CGR). Need new submission script (heredoc-style).

| Job ID | Name | Datasets | Status |
|--------|------|----------|--------|
| 7568449 | rxn_buchwald_hartwig | buchwald_hartwig | **COMPLETED 2026-04-28 BUT MISSED CGR REPS** — ran before chython/rxnmapper/chytorch were installed in venv; cgr_morgan/chytorch_graphormer/cgr_smiles_lm silently skipped. JSON has 35 reps, none CGR. |
| 7568450 | rxn_bh_doyle_plates | bh_doyle_plates | **COMPLETED 2026-05-01** — 84min, all 3 CGR reps landed. cgr_morgan promoted to Top Tier (R²=0.590, RF, RMSE=16.67). chytorch_graphormer (R²=−0.09) and cgr_smiles_lm (R²=0.37) dropped. Tier in PIPELINE_READINESS_TODO §10 updated. |
| 7589759 | rxn_buchwald_hartwig | buchwald_hartwig | **SUBMITTED 2026-05-02** via `submit_reaction_cgr_resurvey.sh bh` — re-run after CGR deps installed. `--skip-existing` so only the 3 CGR reps will compute. |
| — | rxn_suzuki | suzuki | NOT submitted (suzuki dropped per pipeline doc, R²=0.453 ceiling). |

---

## 2026-04-17 — Consolidated precompute (v10, registry-backed) — 114 jobs

**Submitted:** 2026-04-17.
**Script:** `tests/scripts/plan_precompute.py --consolidated --cluster arc`.
**Infrastructure:** Phase A-H + D.2 hybrid_evals dedup (commits b28d092 → eccb6a3).
**Cache:** content-validated, no version number. 23-field meta per search cache.
**Mode:** consolidated — one sbatch per `(dataset, standard|distance)`, each
running all applicable tier × aug_strategy conditions sequentially inside a
single python process. 2 jobs per dataset instead of 10 = 5× fewer sbatch
calls vs v6-v9 per-condition submissions.

### Superseded (cancelled 2026-04-17)

| Job ID range | Notes |
|--------------|-------|
| 11781449-11781478 | 30 jobs, per-condition mode for flab_pk_jain_fcrnrt + abagym_HER2 + humanness_therapeutic. Cancelled + resubmitted as part of the new consolidated batch. |

### Summary by partition

| Cluster | Partition | Range | Count | Band(s) | Time | Mem |
|---------|-----------|-------|-------|---------|------|-----|
| arc | short | 11781997-11782074 | 74 | XS / S / M | 12:00:00 | 64-128G |
| arc | medium | 11782049-11782050, 11782075-11782082 | 10 | L, XL | 2-00:00:00 | 128-256G |
| arc | long | 11782083-11782113 (11782090 missing) | 30 | NT | 2-00:00:00 | 128G |

Total: **114 jobs, 57 datasets**.

### Datasets by partition

**arc short** (XS/S/M band, fits 12h):
- Molecular (8 datasets × 2 = 16): caco2, logd, qm9, adme_hlm, adme_rlm, adme_mdr1, herg_ki, herg
- Antibody inter-family + DMS (24 × 2 = 48): 5 abagym (Ang2, EGFR, VEGF, lysozyme, HER2), abcov_kd, 12 flab_* (all except thermo_tresanco), 6 ginkgo (acsins, colloidal, heparin, hic, pr_cho, titer), humanness_therapeutic, ddg_combined
- Reaction (2 × 2 = 4): buchwald_hartwig, bh_doyle_plates
- DTA (3 × 2 = 6): davis, kiba, bindingdb_kd

**arc medium** (L/XL band, needs ≥12h buffer):
- flab_thermo_tresanco (L; empirically 8h on general_uncapped)
- tz_her2_binary, tz_her2_multiclass (L; n=5000)
- tcga_luad_vs_lusc, tcga_brca_pam50 (XL; high-dim reps)

**arc long** (NT band, 15 nucleotide datasets × 2 = 30):
- nt_H2AFZ, nt_H3K27ac, nt_H3K27me3, nt_H3K36me3, nt_H3K4me1,
  nt_H3K4me2, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_H4K20me1,
  nt_promoter_all, nt_promoter_tata, nt_promoter_no_tata,
  gb_human_nontata_promoters, gb_human_enhancers_cohn

### Conditions per dataset (consolidated mode)

Each dataset gets 2 sbatch submissions:

| Job suffix | Conditions run inside the process |
|------------|----------------------------------|
| `_all_standard` | general_none, general_uncapped, general_capped, top_none, top_uncapped, top_capped |
| `_all_distance` | general_distance_none, general_distance_uncapped, top_distance_none, top_distance_uncapped (DISTANCE_EVALUATORS + DISTANCE_BUDGETS) |

### Status checks

```bash
# Live queue
squeue --clusters=all -u $USER | wc -l

# Summary after completion
sacct -j 11781997-11782113 --format=State --noheader -X | sort | uniq -c

# Failures
sacct -j 11781997-11782113 --format=JobID,JobName%40,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"

# [cache] summary lines from Phase E — confirms observability + D.2 dedup
grep -h "^\[cache\]" slurm-*.out | sort | uniq -c
```

### Expected outcomes

- XS band (abagym_HER2, ginkgo_heparin, ginkgo_pr_cho): 1-3h each
- Full short-partition drain: ~12h depending on priority
- Medium partition L/XL jobs: 8h–1d typical
- Long partition nucleotide: 6-12h each (reps pre-cached as .npy)
- `[cache] hybrid_eval: N hit` appears in SLURM logs when top_k allocation picks the same subset across general and top tier searches — first production test of Phase D.2 dedup

---

## 2026-04-20 — M-band timeout resubmit (12 datasets × 2 = 24 jobs)

**Submitted:** 2026-04-20 | **Trigger:** 29 TIMEOUTs on short partition (12h). Root cause: many reps + GP enabled pushes consolidated runtime past 12h even for XS/S datasets. Fix: upshifted to M band in `_KNOWN_LONG_RUNNERS` (commit 507bc55).
**Cluster:** arc | **Partition:** medium (2d/128G) | **Mode:** consolidated (standard + distance bundles)
**ID range:** 11799082–11799105

| Dataset | Standard | Distance |
|---------|----------|----------|
| ginkgo_heparin | 11799082 | 11799083 |
| ginkgo_acsins | 11799084 | 11799085 |
| ginkgo_colloidal | 11799086 | 11799087 |
| ginkgo_hic | 11799088 | 11799089 |
| ginkgo_pr_cho | 11799090 | 11799091 |
| ginkgo_titer | 11799092 | 11799093 |
| abagym_Ang2 | 11799094 | 11799095 |
| abagym_EGFR | 11799096 | 11799097 |
| abagym_HER2 | 11799098 | 11799099 |
| abagym_VEGF | 11799100 | 11799101 |
| abagym_lysozyme | 11799102 | 11799103 |
| humanness_therapeutic | 11799104 | 11799105 |

### Status checks

```bash
sacct -j 11799082-11799105 --format=State --noheader -X | sort | uniq -c
sacct -j 11799082-11799105 --format=JobID,JobName%40,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"
```

---

## 2026-04-22 — Medium→Long migration (33 pending, script-driven)

**Trigger:** 33 of the 2026-04-20 M-band jobs (11799082–11799105) plus 8 older ones (11784367–11784374 tcga+tz) and 1 resubmit (11790814 humanness) were stuck PD on `medium` with `Priority` reason. Migrated to `long` (7d) via `tests/slurm_scripts/migrate_medium_to_long.sh`. Running jobs on medium (flab_*, herg_ki, logd, qm9) left alone.

32 of 33 migrated. Duplicate `pc_humanness_therapeutic_all_standard` (11799104) scancelled manually before migrate; 11790814 kept as the survivor.

**Old ID → New ID (on long, 7d, 64G):**

| Old | New | Dataset / kind |
|-----|-----|----------------|
| 11784367 | 11825809 | tcga_brca_pam50 / all_standard |
| 11784368 | 11825808 | tcga_brca_pam50 / all_distance |
| 11784369 | 11825807 | tcga_luad_vs_lusc / all_standard |
| 11784370 | 11825806 | tcga_luad_vs_lusc / all_distance |
| 11784371 | 11825805 | tz_her2_binary / all_standard |
| 11784372 | 11825804 | tz_her2_binary / all_distance |
| 11784373 | 11825802 | tz_her2_multiclass / all_standard |
| 11784374 | 11825801 | tz_her2_multiclass / all_distance |
| 11790814 | 11825810 | humanness_therapeutic / all_standard |
| 11799082 | 11825833 | ginkgo_heparin / all_standard |
| 11799083 | 11825832 | ginkgo_heparin / all_distance |
| 11799084 | 11825831 | ginkgo_acsins / all_standard |
| 11799085 | 11825830 | ginkgo_acsins / all_distance |
| 11799086 | 11825829 | ginkgo_colloidal / all_standard |
| 11799087 | 11825828 | ginkgo_colloidal / all_distance |
| 11799088 | 11825827 | ginkgo_hic / all_standard |
| 11799089 | 11825826 | ginkgo_hic / all_distance |
| 11799090 | 11825825 | ginkgo_pr_cho / all_standard |
| 11799091 | 11825824 | ginkgo_pr_cho / all_distance |
| 11799092 | 11825823 | ginkgo_titer / all_standard |
| 11799093 | 11825822 | ginkgo_titer / all_distance |
| 11799094 | 11825821 | abagym_Ang2 / all_standard |
| 11799095 | 11825820 | abagym_Ang2 / all_distance |
| 11799096 | 11825819 | abagym_EGFR / all_standard |
| 11799097 | 11825818 | abagym_EGFR / all_distance |
| 11799098 | 11825817 | abagym_HER2 / all_standard |
| 11799099 | 11825816 | abagym_HER2 / all_distance |
| 11799100 | 11825815 | abagym_VEGF / all_standard |
| 11799101 | 11825814 | abagym_VEGF / all_distance |
| 11799102 | 11825813 | abagym_lysozyme / all_standard |
| 11799103 | 11825812 | abagym_lysozyme / all_distance |
| 11799104 | (scancelled) | humanness_therapeutic / all_standard — duplicate of 11790814 |
| 11799105 | 11825811 | humanness_therapeutic / all_distance |

New ID range: **11825801–11825833** (11825803 unused).

### Status checks

```bash
sacct -j 11825801-11825833 --format=State --noheader -X | sort | uniq -c
sacct -j 11825801-11825833 --format=JobID,JobName%40,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"
```

---

## 2026-04-22 — Bundled fill-in submission (57 jobs, long)

**Script:** `tests/slurm_scripts/resubmit_missing_bundled.sh` (commit 8577241). After migrating the 33 stuck PDs, the status check still reported 164 missing per-condition entries across 30 datasets. Bundled into 30 `_all_standard` + 27 `_all_distance` sbatch submissions (3 datasets have distance already covered by earlier runs, so only 27 distance bundles).
**Partition:** long (7d). 64G default; 128G for `nt_*` / `gb_*` nucleotide loaders.
**ID range:** 11825934–11825990.

### `_all_standard` bundles (30 × long, 7d)

| Job ID | Dataset | Mem |
|--------|---------|-----|
| 11825934 | adme_hlm | 64G |
| 11825935 | adme_mdr1 | 64G |
| 11825936 | adme_rlm | 64G |
| 11825937 | bh_doyle_plates | 64G |
| 11825938 | buchwald_hartwig | 64G |
| 11825939 | caco2 | 64G |
| 11825940 | ddg_combined | 64G |
| 11825941 | flab_aggregation_jain | 64G |
| 11825942 | flab_aggregation_jain_hicrt | 64G |
| 11825943 | flab_binding_adams | 64G |
| 11825944 | flab_binding_koenig | 64G |
| 11825945 | flab_binding_phillips_h3 | 64G |
| 11825946 | flab_expression_adams | 64G |
| 11825947 | flab_expression_koenig | 64G |
| 11825948 | flab_pk_jain_fcrnrt | 64G |
| 11825949 | gb_human_enhancers_cohn | 128G |
| 11825950 | gb_human_nontata_promoters | 128G |
| 11825951 | nt_H2AFZ | 128G |
| 11825952 | nt_H3K27ac | 128G |
| 11825953 | nt_H3K27me3 | 128G |
| 11825954 | nt_H3K36me3 | 128G |
| 11825955 | nt_H3K4me1 | 128G |
| 11825956 | nt_H3K4me2 | 128G |
| 11825957 | nt_H3K4me3 | 128G |
| 11825958 | nt_H3K9ac | 128G |
| 11825959 | nt_H3K9me3 | 128G |
| 11825960 | nt_H4K20me1 | 128G |
| 11825961 | nt_promoter_all | 128G |
| 11825962 | nt_promoter_no_tata | 128G |
| 11825963 | nt_promoter_tata | 128G |

### `_all_distance` bundles (27 × long, 7d)

| Job ID | Dataset | Mem |
|--------|---------|-----|
| 11825964 | adme_hlm | 64G |
| 11825965 | adme_mdr1 | 64G |
| 11825966 | adme_rlm | 64G |
| 11825967 | bh_doyle_plates | 64G |
| 11825968 | buchwald_hartwig | 64G |
| 11825969 | caco2 | 64G |
| 11825970 | ddg_combined | 64G |
| 11825971 | flab_binding_adams | 64G |
| 11825972 | flab_binding_koenig | 64G |
| 11825973 | flab_binding_phillips_h3 | 64G |
| 11825974 | flab_expression_adams | 64G |
| 11825975 | flab_expression_koenig | 64G |
| 11825976 | gb_human_enhancers_cohn | 128G |
| 11825977 | gb_human_nontata_promoters | 128G |
| 11825978 | nt_H2AFZ | 128G |
| 11825979 | nt_H3K27ac | 128G |
| 11825980 | nt_H3K27me3 | 128G |
| 11825981 | nt_H3K36me3 | 128G |
| 11825982 | nt_H3K4me1 | 128G |
| 11825983 | nt_H3K4me2 | 128G |
| 11825984 | nt_H3K4me3 | 128G |
| 11825985 | nt_H3K9ac | 128G |
| 11825986 | nt_H3K9me3 | 128G |
| 11825987 | nt_H4K20me1 | 128G |
| 11825988 | nt_promoter_all | 128G |
| 11825989 | nt_promoter_no_tata | 128G |
| 11825990 | nt_promoter_tata | 128G |

3 datasets had distance already populated (flab_aggregation_jain, flab_aggregation_jain_hicrt, flab_pk_jain_fcrnrt) so they appear in `_all_standard` only.

### Status checks

```bash
sacct -j 11825934-11825990 --format=State --noheader -X | sort | uniq -c
sacct -j 11825934-11825990 --format=JobID,JobName%40,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"
```

---

## 2026-04-26 — Complexity experiment suite (40 jobs, aggressive_pruning)

**NOTE: these are EXPERIMENT jobs, not precompute jobs.** Tracker re-used
because the operational shape is the same. Pre-existing precompute pending
jobs were cancelled before this submission. C02 + later complexity
experiments will fill any precompute gaps on demand via the on-disk cache.

**Submitted:** 2026-04-26 via
`bash tests/slurm_scripts/submit_all_datasets.sh complexity`
(prefix `cx_`, default `PRUNING_MODE=aggressive_pruning`).

**Partition:** long | **Wall:** 7-00:00:00 | **CPUs:** 8 | **Mem:** 128G default,
128G override on listed datasets per `get_mem_override` (no change from
that table — ten datasets carry the override).

**Pruning:** `aggressive_pruning` — single-stage 0.15 strategy prune globally;
multi-stage `td=0.5 at ckpt=[50]` applied to gp/svm/mlp ONLY when
n_samples >= 1000. Smaller datasets fall back to the existing
single-stage default automatically.

**ID range:** 12013701–12013740.

| Job ID | Dataset | Mem |
|--------|---------|-----|
| 12013701 | herg_ki | 128G |
| 12013702 | caco2 | 128G |
| 12013703 | logd | 128G |
| 12013704 | qm9 | 128G |
| 12013705 | adme_hlm | 128G |
| 12013706 | adme_rlm | 128G |
| 12013707 | adme_mdr1 | 128G |
| 12013708 | flab_thermo_tresanco | 128G |
| 12013709 | flab_pk_jain_heprt | 128G |
| 12013710 | flab_pk_jain_fcrnrt | 128G |
| 12013711 | flab_polyreact_jain_cic | 128G |
| 12013712 | flab_aggregation_jain_hicrt | 128G |
| 12013713 | flab_thermo_jain | 128G |
| 12013714 | abcov_kd | 128G |
| 12013715 | flab_aggregation_jain | 128G |
| 12013716 | flab_immunogenicity_marks | 128G |
| 12013717 | flab_polyreact_jain_bvp | 128G |
| 12013718 | ddg_combined | 128G |
| 12013719 | flab_binding_koenig | 128G |
| 12013720 | flab_expression_koenig | 128G |
| 12013721 | flab_binding_adams | 128G |
| 12013722 | flab_expression_adams | 128G |
| 12013723 | flab_binding_phillips_h3 | 128G |
| 12013724 | abagym_Ang2 | 128G |
| 12013725 | abagym_EGFR | 128G |
| 12013726 | abagym_HER2 | 128G |
| 12013727 | abagym_lysozyme | 128G |
| 12013728 | abagym_VEGF | 128G |
| 12013729 | tz_her2_binary | 128G |
| 12013730 | humanness_therapeutic | 128G |
| 12013731 | ginkgo_acsins | 128G |
| 12013732 | ginkgo_colloidal | 128G |
| 12013733 | ginkgo_heparin | 128G |
| 12013734 | ginkgo_hic | 128G |
| 12013735 | ginkgo_pr_cho | 128G |
| 12013736 | ginkgo_titer | 128G |
| 12013737 | buchwald_hartwig | 128G |
| 12013738 | bh_doyle_plates | 128G |
| 12013739 | tcga_luad_vs_lusc | 128G |
| 12013740 | tcga_brca_pam50 | 128G |

### Status checks

```bash
sacct -j 12013701-12013740 --format=State --noheader -X | sort | uniq -c
sacct -j 12013701-12013740 --format=JobID,JobName%30,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"
squeue -u $USER -o '%.10i %.8P %.30j %.2t %.10M %.10L' | grep cx_
```

### Per-dataset gate behaviour (expected)

`aggressive_pruning` overrides apply when the dataset has `n_samples >= 1000`.
Datasets where the gate kicks them off and they fall back to single-stage
0.15 only:
- flab_pk_jain_fcrnrt (n=132)
- flab_thermo_jain (n=137)
- flab_immunogenicity_marks (~250)
- abcov_kd (~141)
- ginkgo_* (n=94–242)
- abagym_* (~120–300)
- humanness_therapeutic (clf, small)
- tz_her2_binary (~5K subsample, gate active → overrides applied)
- caco2 (n=2158, gate active)
- logd, herg_ki, qm9, adme_*, ddg_combined, buchwald_hartwig, tcga_* (gate active)

Each per-dataset SLURM log prints a "Pruning mode: aggressive_pruning"
line followed by either "overrides active for: [...]" or "overrides
gated off (n=N < 1000)" so behaviour is auditable per-dataset.

---

## 2026-04-27 — Similarity experiment suite (40 jobs, aggressive_pruning + reverse order)

**Submitted:** 2026-04-27 via
`KIRBY_PRECOMPUTE_REVERSE_ORDER=1 PRUNING_MODE=aggressive_pruning bash tests/slurm_scripts/submit_all_datasets.sh similarity`
(prefix `sim_`).

Reverse condition order set so similarity precompute work staggers
naturally against the still-running complexity wave (12013701–12013740)
on overlapping datasets — one wave starts at general/none while the
other starts at top/capped, hitting different cache files until the
crossover point. Inner hybrid_eval cache helps reduce duplicate compute
in the middle.

**Partition:** long | **Wall:** 7-00:00:00 | **CPUs:** 8 | **Mem:** 128G default,
128G override on listed datasets per `get_mem_override` (same table as
complexity wave).

**Pruning:** `aggressive_pruning` — single-stage 0.15 strategy prune
globally; multi-stage `td=0.5 at ckpt=[50]` applied to gp/svm/mlp ONLY
when n_samples >= 1000. Smaller datasets fall back to single-stage
default automatically.

**ID range:** 12014438–12014477.

| Job ID | Dataset | Mem |
|--------|---------|-----|
| 12014438 | herg_ki | 128G |
| 12014439 | caco2 | 128G |
| 12014440 | logd | 128G |
| 12014441 | qm9 | 128G |
| 12014442 | adme_hlm | 128G |
| 12014443 | adme_rlm | 128G |
| 12014444 | adme_mdr1 | 128G |
| 12014445 | flab_thermo_tresanco | 128G |
| 12014446 | flab_pk_jain_heprt | 128G |
| 12014447 | flab_pk_jain_fcrnrt | 128G |
| 12014448 | flab_polyreact_jain_cic | 128G |
| 12014449 | flab_aggregation_jain_hicrt | 128G |
| 12014450 | flab_thermo_jain | 128G |
| 12014451 | abcov_kd | 128G |
| 12014452 | flab_aggregation_jain | 128G |
| 12014453 | flab_immunogenicity_marks | 128G |
| 12014454 | flab_polyreact_jain_bvp | 128G |
| 12014455 | ddg_combined | 128G |
| 12014456 | flab_binding_koenig | 128G |
| 12014457 | flab_expression_koenig | 128G |
| 12014458 | flab_binding_adams | 128G |
| 12014459 | flab_expression_adams | 128G |
| 12014460 | flab_binding_phillips_h3 | 128G |
| 12014461 | abagym_Ang2 | 128G |
| 12014462 | abagym_EGFR | 128G |
| 12014463 | abagym_HER2 | 128G |
| 12014464 | abagym_lysozyme | 128G |
| 12014465 | abagym_VEGF | 128G |
| 12014466 | tz_her2_binary | 128G |
| 12014467 | humanness_therapeutic | 128G |
| 12014468 | ginkgo_acsins | 128G |
| 12014469 | ginkgo_colloidal | 128G |
| 12014470 | ginkgo_heparin | 128G |
| 12014471 | ginkgo_hic | 128G |
| 12014472 | ginkgo_pr_cho | 128G |
| 12014473 | ginkgo_titer | 128G |
| 12014474 | buchwald_hartwig | 128G |
| 12014475 | bh_doyle_plates | 128G |
| 12014476 | tcga_luad_vs_lusc | 128G |
| 12014477 | tcga_brca_pam50 | 128G |

### Status checks

```bash
sacct -j 12014438-12014477 --format=State --noheader -X | sort | uniq -c
sacct -j 12014438-12014477 --format=JobID,JobName%30,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"
squeue -u $USER -o '%.10i %.8P %.30j %.2t %.10M %.10L' | grep sim_
```

### Verification of reverse-order behaviour

Each per-dataset SLURM log should show:
```
KIRBY_PRECOMPUTE_REVERSE_ORDER=1 → iterating aug_conditions in reverse: ['capped', 'uncapped', 'none']
```

Compare against any complexity job (12013701–12013740) log — those should
show the default forward order (no such line printed).

### Pending later (after precompute settles)

Noise, federated, and OPIG submissions queued for after both complexity
and similarity have populated their precompute caches. Those will run
with default forward order (no env var) and benefit from cache hits
across most conditions.

---

## 2026-04-29 — Missing-nucleotide gap-fill wave (34 jobs)

Reason: 16 nucleotide datasets had no submission in either the cx or sim
waves; plus two stragglers (sim_gb_demo_coding waiting on cx race, and
sim_herg_ki). Submitted alongside the 2026-04-27 waves still running.

Cancellations done same day (wedged jobs with frozen logs, manually
killed): 12013711, 12013716, 12013717, 12013726.

### Complexity (forward order) — 12044245–12044260

| JobID | Dataset | Mem |
|-------|---------|-----|
| 12044245 | gb_demo_human_or_worm | 128G |
| 12044246 | gb_human_enhancers_cohn | 128G |
| 12044247 | gb_human_nontata_promoters | 128G |
| 12044248 | nt_H2AFZ | 128G |
| 12044249 | nt_H3K27ac | 128G |
| 12044250 | nt_H3K27me3 | 128G |
| 12044251 | nt_H3K36me3 | 128G |
| 12044252 | nt_H3K4me1 | 128G |
| 12044253 | nt_H3K4me2 | 128G |
| 12044254 | nt_H3K4me3 | 128G |
| 12044255 | nt_H3K9ac | 128G |
| 12044256 | nt_H3K9me3 | 128G |
| 12044257 | nt_H4K20me1 | 128G |
| 12044258 | nt_promoter_all | 128G |
| 12044259 | nt_promoter_no_tata | 128G |
| 12044260 | nt_promoter_tata | 128G |

### Similarity (reverse order, KIRBY_PRECOMPUTE_REVERSE_ORDER=1) — 12044261–12044276

| JobID | Dataset | Mem |
|-------|---------|-----|
| 12044261 | gb_demo_human_or_worm | 128G |
| 12044262 | gb_human_enhancers_cohn | 128G |
| 12044263 | gb_human_nontata_promoters | 128G |
| 12044264 | nt_H2AFZ | 128G |
| 12044265 | nt_H3K27ac | 128G |
| 12044266 | nt_H3K27me3 | 128G |
| 12044267 | nt_H3K36me3 | 128G |
| 12044268 | nt_H3K4me1 | 128G |
| 12044269 | nt_H3K4me2 | 128G |
| 12044270 | nt_H3K4me3 | 128G |
| 12044271 | nt_H3K9ac | 128G |
| 12044272 | nt_H3K9me3 | 128G |
| 12044273 | nt_H4K20me1 | 128G |
| 12044274 | nt_promoter_all | 128G |
| 12044275 | nt_promoter_no_tata | 128G |
| 12044276 | nt_promoter_tata | 128G |

### Stragglers (sim, reverse order)

| JobID | Dataset | Mem |
|-------|---------|-----|
| 12044278 | gb_demo_coding_vs_intergenomic_seqs | 128G |
| 12044279 | herg_ki | default |

### Status checks

```bash
sacct -j 12044245-12044279 --format=State --noheader -X | sort | uniq -c
sacct -j 12044245-12044279 --format=JobID,JobName%30,State,Elapsed,ExitCode -X \
  | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"
```
