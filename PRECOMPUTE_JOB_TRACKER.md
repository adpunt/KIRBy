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
| 7483718 | dta_bindingdb_kd (post-c18342f, per-rep checkpoint) | **PD** (2026-04-15, medium+GPU+256G, 48h) |
| 7467954 | dta_davis (resubmit, --skip-existing) | **TIMEOUT** (2026-04-15 01:40, 48h wall, died at ~66% through contact graph features; no checkpointed output — pre-a81b0eb code) |
| 7483719 | dta_davis (post-c18342f heredoc + a81b0eb checkpointing) | **PD** (2026-04-15, long+GPU+256G, 96h wall) |
| 7457314 | dta_kiba_fill (--skip-existing) | **FAILED** (ExitCode 126:0, 6min — sbatch --wrap line-splitting bug) |
| 7467954 | dta_davis (resubmit, --skip-existing) | **RUNNING** (1d22h as of 2026-04-15, ongoing) |
| 7483635 | dta_kiba_fill (pre-c18342f heredoc bug) | **CANCELLED** (2026-04-15, bad heredoc) |
| 7483717 | dta_kiba_fill (post-c18342f) | **PD** (2026-04-15, long+GPU+256G, 24h) |

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
| 7483713 | abclf_tz_her2_binary (post-c18342f) | **PD** (2026-04-15, long+GPU+256G, 24h) |
| 7483714 | abclf_tz_her2_multiclass (post-c18342f) | **PD** |
| 7483715 | abclf_humanness_species (post-c18342f) | **PD** (long+GPU+128G, 24h) |
| 7483716 | abclf_humanness_therapeutic (post-c18342f) | **PD** |
| 7483636 | abclf_tz_her2_binary (DUPLICATE) | **CANCELLED** (2026-04-15) |
| 7483637 | abclf_tz_her2_multiclass (DUPLICATE) | **CANCELLED** |
| 7483638 | abclf_humanness_species (DUPLICATE) | **CANCELLED** |
| 7483639 | abclf_humanness_therapeutic (DUPLICATE) | **CANCELLED** |
| 7483640 | dta_kiba_fill (DUPLICATE) | **CANCELLED** |

### SchNet (molecular)

| Job ID | Name | Status |
|--------|------|--------|
| 7440325 | schnet_logd | **FAILED** (SLURM exit 0:0 but experiment broken, 4h17m, 2026-04-12) — baselines ran but all 3 FT reps (ft_schnet, lora_chemberta, lora_molformer) emitted `FT FACTORY ERROR: No data left in file` from `np.load(cache_path)` on every fold. Cache files are truncated/empty .npz. Whole point of job = FT reps. **Needs: clear corrupt FT cache, investigate why atomic write in `_base.py::load_or_compute` produced empty files, rerun.** |
| 7440326 | schnet_herg_ki | **FAILED** (1h50m, 2026-04-12) — same FT cache corruption |
| 7440327 | schnet_caco2 | **FAILED** (1h13m, 2026-04-12) — same FT cache corruption |

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
| 7483707 | nuc_nt_enhancers (heredoc resubmit 4, post-c18342f, inline python cmd) | **PD** (2026-04-15) |
| 7483708 | nuc_nt_enhancers_types (post-c18342f) | **PD** |
| 7483709 | nuc_nt_splice_sites_acceptors (post-c18342f) | **PD** |
| 7483710 | nuc_nt_splice_sites_donors (post-c18342f) | **PD** |
| 7440309 | nuc_nt_H2AFZ | DONE (18/25 reps, winner: mistral_dna+ridge=0.4323 mcc) |
| 7440310 | nuc_nt_H3K27ac | DONE (winner: gena_lm+ridge=0.4569 mcc) |
| 7440311 | nuc_nt_H3K27me3 | DONE (winner: gena_lm+ridge=0.5522 mcc) |
| 7440312 | nuc_nt_H3K36me3 | DONE (winner: gena_lm+ridge=0.5715 mcc) |
| 7440313 | nuc_nt_H3K4me1 | DONE (winner: gena_lm+ridge=0.4405 mcc) |
| 7440314 | nuc_nt_H3K4me2 | **DONE** (1d14h, 2026-04-12) |
| 7440315 | nuc_nt_H3K4me3 | **DONE** (1d3h, 2026-04-12) |
| 7440316 | nuc_nt_H3K9ac | **DONE** (1d11h, 2026-04-13) |
| 7440317 | nuc_nt_H3K9me3 | **DONE** (1d20h, 2026-04-13) |
| 7440318 | nuc_nt_H4K20me1 | **DONE** (22h, 2026-04-12) |
| 7440319 | nuc_gb_human_nontata_promoters | **DONE** (10h38m, 2026-04-12) |
| 7440320 | nuc_gb_human_enhancers_cohn | **DONE** (16h35m, 2026-04-12) |
| 7440321 | nuc_gb_demo_coding_vs_intergenomic | **OUT_OF_MEMORY** (1h44m, 2026-04-12) — 128G insufficient, resubmit with 256G |
| 7440322 | nuc_gb_demo_human_or_worm | **OUT_OF_MEMORY** (1h46m, 2026-04-12) — 128G insufficient, resubmit with 256G |
| 7483645 | nuc_gb_demo_coding_vs_intergenomic (pre-c18342f) | **FAILED** (same heredoc bug) |
| 7483646 | nuc_gb_demo_human_or_worm (pre-c18342f) | **FAILED** (same heredoc bug) |
| 7483711 | nuc_gb_demo_coding_vs_intergenomic (post-c18342f, OOM bump 256G) | **PD** (2026-04-15, medium+256G, 48h) |
| 7483712 | nuc_gb_demo_human_or_worm (post-c18342f, OOM bump 256G) | **PD** (2026-04-15, medium+256G, 48h) |

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
| 7483446 | cgr_pretrain | pretrain only | **PD/R** (2026-04-14, HTC + GPU) — resubmit with 2c367d6 manual loop + 4bdede7 pretrain_htc mode (`--gres=gpu:1 --partition=long --time=23:59:00`). Uses existing corpus.txt (40M, 392324 lines, unchanged from 11749515). Expected wall ~1-2h on A100/V100. Unblocks reaction re-survey with cgr_smiles_lm. |

### Reaction re-survey (blocked on 7467971)

Re-survey with 41 reps (38 original + 3 CGR). Submit after pretraining checkpoint exists.

| Job ID | Name | Datasets | Status |
|--------|------|----------|--------|
| — | rxn_baselines | buchwald_hartwig, bh_doyle_plates, suzuki | BLOCKED — waiting on 11749515 |
