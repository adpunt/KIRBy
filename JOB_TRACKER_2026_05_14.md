# Job tracker — 2026-05-14 resubmit (post-cache-fix)

103 jobs submitted at 2026-05-14 after cancel of the 103 pre-cache-fix jobs (commit `58e431b`). All jobs use:

- GP cap (n ≤ 1000)
- Nucleotide ridge-skip
- Dataset-specific uncapped skip rules (flab_thermo_tresanco, tcga_luad_vs_lusc, ddg_combined)
- `KIRBY_CACHE_DIR`-aware rep + dataset caches (server: `/data/stat-ecr/scat9264/.cache/kirby`)

Wall: 7 days (long partition).
Expected first-cache-write: ~24–48h from submission for small datasets, 5–7 days for the heaviest.

## How to update

```bash
# On server: pull state into one column-friendly file
sacct -j $(awk '{print $1}' JOB_TRACKER_2026_05_14.md | grep -E '^121[0-9]+$' | paste -sd,) \
      --format=JobID,State,Elapsed,ExitCode -X | head -200
```

When jobs finish, replace `RUNNING` → `COMPLETED` / `FAILED` / `TIMEOUT` in the table below.

---

## Complexity (57 jobs)

| Job ID | Dataset | Status |
|---|---|---|
| 12140220 | abagym_Ang2 | RUNNING |
| 12140221 | abagym_EGFR | RUNNING |
| 12140222 | abagym_VEGF | RUNNING |
| 12140223 | adme_hlm | RUNNING |
| 12140224 | adme_mdr1 | RUNNING |
| 12140225 | adme_rlm | RUNNING |
| 12140226 | bh_doyle_plates | RUNNING |
| 12140227 | bindingdb_kd | RUNNING |
| 12140228 | buchwald_hartwig | RUNNING |
| 12140229 | caco2 | RUNNING |
| 12140231 | ddg_combined | RUNNING |
| 12140232 | flab_aggregation_jain | RUNNING |
| 12140233 | flab_aggregation_jain_hicrt | RUNNING |
| 12140234 | flab_binding_adams | RUNNING |
| 12140235 | flab_binding_koenig | RUNNING |
| 12140236 | flab_binding_phillips_h3 | RUNNING |
| 12140237 | flab_expression_adams | RUNNING |
| 12140238 | flab_expression_koenig | RUNNING |
| 12140239 | flab_pk_jain_fcrnrt | RUNNING |
| 12140240 | flab_pk_jain_heprt | RUNNING |
| 12140241 | flab_thermo_jain | RUNNING |
| 12140243 | flab_thermo_tresanco | RUNNING |
| 12140244 | gb_demo_human_or_worm | RUNNING |
| 12140245 | gb_human_enhancers_cohn | RUNNING |
| 12140246 | gb_human_nontata_promoters | RUNNING |
| 12140247 | ginkgo_acsins | RUNNING |
| 12140248 | ginkgo_colloidal | RUNNING |
| 12140249 | ginkgo_heparin | RUNNING |
| 12140250 | ginkgo_hic | RUNNING |
| 12140251 | ginkgo_pr_cho | RUNNING |
| 12140252 | ginkgo_titer | RUNNING |
| 12140253 | herg | RUNNING |
| 12140254 | herg_ki | RUNNING |
| 12140255 | humanness_therapeutic | RUNNING |
| 12140256 | kiba | RUNNING |
| 12140257 | logd | RUNNING |
| 12140258 | nt_H2AFZ | RUNNING |
| 12140259 | nt_H3K27ac | RUNNING |
| 12140260 | nt_H3K27me3 | RUNNING |
| 12140261 | nt_H3K36me3 | RUNNING |
| 12140262 | nt_H3K4me1 | RUNNING |
| 12140263 | nt_H3K4me2 | RUNNING |
| 12140264 | nt_H3K4me3 | RUNNING |
| 12140265 | nt_H3K9ac | RUNNING |
| 12140266 | nt_H3K9me3 | RUNNING |
| 12140267 | nt_H4K20me1 | RUNNING |
| 12140268 | nt_enhancers | RUNNING |
| 12140269 | nt_enhancers_types | RUNNING |
| 12140270 | nt_promoter_all | RUNNING |
| 12140271 | nt_promoter_no_tata | RUNNING |
| 12140272 | nt_promoter_tata | RUNNING |
| 12140273 | nt_splice_sites_acceptors | RUNNING |
| 12140274 | nt_splice_sites_donors | RUNNING |
| 12140275 | qm9 | RUNNING |
| 12140276 | tcga_brca_pam50 | RUNNING |
| 12140277 | tcga_luad_vs_lusc | RUNNING |
| 12140278 | tz_her2_multiclass | RUNNING |

## Similarity (46 jobs)

| Job ID | Dataset | Status |
|---|---|---|
| 12140279 | abagym_lysozyme | RUNNING |
| 12140282 | bindingdb_kd | RUNNING |
| 12140283 | buchwald_hartwig | RUNNING |
| 12140316 | ddg_combined | RUNNING |
| 12140317 | flab_aggregation_jain | RUNNING |
| 12140345 | flab_aggregation_jain_hicrt | RUNNING |
| 12140346 | flab_binding_adams | RUNNING |
| 12140347 | flab_binding_koenig | RUNNING |
| 12140348 | flab_binding_phillips_h3 | RUNNING |
| 12140349 | flab_expression_adams | RUNNING |
| 12140350 | flab_expression_koenig | RUNNING |
| 12140351 | flab_pk_jain_fcrnrt | RUNNING |
| 12140353 | flab_pk_jain_heprt | RUNNING |
| 12140354 | flab_thermo_jain | RUNNING |
| 12140355 | flab_thermo_tresanco | RUNNING |
| 12140356 | gb_demo_human_or_worm | RUNNING |
| 12140357 | gb_human_enhancers_cohn | RUNNING |
| 12140358 | gb_human_nontata_promoters | RUNNING |
| 12140359 | ginkgo_acsins | RUNNING |
| 12140360 | ginkgo_colloidal | RUNNING |
| 12140361 | ginkgo_heparin | RUNNING |
| 12140362 | ginkgo_hic | RUNNING |
| 12140363 | ginkgo_pr_cho | RUNNING |
| 12140364 | ginkgo_titer | RUNNING |
| 12140365 | herg_ki | RUNNING |
| 12140366 | kiba | RUNNING |
| 12140367 | nt_H2AFZ | RUNNING |
| 12140368 | nt_H3K27ac | RUNNING |
| 12140369 | nt_H3K27me3 | RUNNING |
| 12140370 | nt_H3K36me3 | RUNNING |
| 12140371 | nt_H3K4me1 | RUNNING |
| 12140372 | nt_H3K4me2 | RUNNING |
| 12140373 | nt_H3K4me3 | RUNNING |
| 12140374 | nt_H3K9ac | RUNNING |
| 12140375 | nt_H3K9me3 | RUNNING |
| 12140376 | nt_H4K20me1 | RUNNING |
| 12140377 | nt_enhancers | RUNNING |
| 12140378 | nt_enhancers_types | RUNNING |
| 12140379 | nt_promoter_all | RUNNING |
| 12140380 | nt_promoter_no_tata | RUNNING |
| 12140381 | nt_promoter_tata | RUNNING |
| 12140382 | nt_splice_sites_acceptors | RUNNING |
| 12140383 | nt_splice_sites_donors | RUNNING |
| 12140384 | tcga_brca_pam50 | RUNNING |
| 12140385 | tcga_luad_vs_lusc | RUNNING |
| 12140386 | tz_her2_multiclass | RUNNING |

---

## Reverse-lookup index (paste a job ID, get the dataset)

For fast `grep 12140xxx` lookup when squeue gives you a number:

```
12140220 cx abagym_Ang2
12140221 cx abagym_EGFR
12140222 cx abagym_VEGF
12140223 cx adme_hlm
12140224 cx adme_mdr1
12140225 cx adme_rlm
12140226 cx bh_doyle_plates
12140227 cx bindingdb_kd
12140228 cx buchwald_hartwig
12140229 cx caco2
12140231 cx ddg_combined
12140232 cx flab_aggregation_jain
12140233 cx flab_aggregation_jain_hicrt
12140234 cx flab_binding_adams
12140235 cx flab_binding_koenig
12140236 cx flab_binding_phillips_h3
12140237 cx flab_expression_adams
12140238 cx flab_expression_koenig
12140239 cx flab_pk_jain_fcrnrt
12140240 cx flab_pk_jain_heprt
12140241 cx flab_thermo_jain
12140243 cx flab_thermo_tresanco
12140244 cx gb_demo_human_or_worm
12140245 cx gb_human_enhancers_cohn
12140246 cx gb_human_nontata_promoters
12140247 cx ginkgo_acsins
12140248 cx ginkgo_colloidal
12140249 cx ginkgo_heparin
12140250 cx ginkgo_hic
12140251 cx ginkgo_pr_cho
12140252 cx ginkgo_titer
12140253 cx herg
12140254 cx herg_ki
12140255 cx humanness_therapeutic
12140256 cx kiba
12140257 cx logd
12140258 cx nt_H2AFZ
12140259 cx nt_H3K27ac
12140260 cx nt_H3K27me3
12140261 cx nt_H3K36me3
12140262 cx nt_H3K4me1
12140263 cx nt_H3K4me2
12140264 cx nt_H3K4me3
12140265 cx nt_H3K9ac
12140266 cx nt_H3K9me3
12140267 cx nt_H4K20me1
12140268 cx nt_enhancers
12140269 cx nt_enhancers_types
12140270 cx nt_promoter_all
12140271 cx nt_promoter_no_tata
12140272 cx nt_promoter_tata
12140273 cx nt_splice_sites_acceptors
12140274 cx nt_splice_sites_donors
12140275 cx qm9
12140276 cx tcga_brca_pam50
12140277 cx tcga_luad_vs_lusc
12140278 cx tz_her2_multiclass
12140279 sim abagym_lysozyme
12140282 sim bindingdb_kd
12140283 sim buchwald_hartwig
12140316 sim ddg_combined
12140317 sim flab_aggregation_jain
12140345 sim flab_aggregation_jain_hicrt
12140346 sim flab_binding_adams
12140347 sim flab_binding_koenig
12140348 sim flab_binding_phillips_h3
12140349 sim flab_expression_adams
12140350 sim flab_expression_koenig
12140351 sim flab_pk_jain_fcrnrt
12140353 sim flab_pk_jain_heprt
12140354 sim flab_thermo_jain
12140355 sim flab_thermo_tresanco
12140356 sim gb_demo_human_or_worm
12140357 sim gb_human_enhancers_cohn
12140358 sim gb_human_nontata_promoters
12140359 sim ginkgo_acsins
12140360 sim ginkgo_colloidal
12140361 sim ginkgo_heparin
12140362 sim ginkgo_hic
12140363 sim ginkgo_pr_cho
12140364 sim ginkgo_titer
12140365 sim herg_ki
12140366 sim kiba
12140367 sim nt_H2AFZ
12140368 sim nt_H3K27ac
12140369 sim nt_H3K27me3
12140370 sim nt_H3K36me3
12140371 sim nt_H3K4me1
12140372 sim nt_H3K4me2
12140373 sim nt_H3K4me3
12140374 sim nt_H3K9ac
12140375 sim nt_H3K9me3
12140376 sim nt_H4K20me1
12140377 sim nt_enhancers
12140378 sim nt_enhancers_types
12140379 sim nt_promoter_all
12140380 sim nt_promoter_no_tata
12140381 sim nt_promoter_tata
12140382 sim nt_splice_sites_acceptors
12140383 sim nt_splice_sites_donors
12140384 sim tcga_brca_pam50
12140385 sim tcga_luad_vs_lusc
12140386 sim tz_her2_multiclass
```

---

## Known constraints / quirks

- **gp/mlp skipped on `flab_thermo_tresanco` uncapped** (per `_DATASET_STRATEGY_SKIPS`)
- **gp skipped on `tcga_luad_vs_lusc` uncapped, `ddg_combined` uncapped**
- **ridge skipped on all nucleotide datasets** (no signal in tests)
- **gp skipped on any dataset with n > 1000** (universal cap)

## Datasets already DONE before this wave (rapid completion expected)

These have valid pre-cache-fix search caches on stat-ecr — new jobs will cache-hit and just run remaining experiment scripts:

- abagym_Ang2, abagym_EGFR, abagym_HER2, abagym_VEGF, abcov_kd
- flab_aggregation_jain, flab_immunogenicity_marks, flab_pk_jain_heprt
- flab_polyreact_jain_bvp, flab_polyreact_jain_cic
- ginkgo_heparin, humanness_therapeutic, tz_her2_binary
