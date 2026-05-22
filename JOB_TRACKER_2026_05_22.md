# Job tracker — 2026-05-22 done-dataset gap fill

61 jobs submitted 2026-05-22 covering all missing complexity / similarity /
noise / OPIG experiments for the 23 precompute-DONE datasets identified by
`tests/check_precompute_status.py --done`. Driven by
`tests/audit_done_dataset_gaps.py` output, manually corrected to:

- Drop c23 from the 3 classification datasets (`humanness_therapeutic`,
  `tz_her2_binary`, `tz_her2_multiclass`) per the deprioritization memory.
- Add both abagym cx jobs after `scontrol show job 12158437` revealed the
  blocking job was `abagym_lysozyme` (not in done set), so the truncation
  match was a false positive.
- Keep c23 for regression antibody and reaction datasets.

All jobs use commit `1b2089a` (ngboost removed from `DOWNSTREAM_MODELS_NOISE`,
audit script added).

## Mapping

| job_id  | name                                | category   | dataset                       | experiments |
|---------|-------------------------------------|------------|-------------------------------|-------------|
| 12191288 | cx_abagym_HER2                     | complexity | abagym_HER2                   | c23 c24 |
| 12191289 | cx_abagym_EGFR                     | complexity | abagym_EGFR                   | c17 c18 c19 c21 c22 c23 c24 |
| 12191290 | sim_abagym_Ang2                    | similarity | abagym_Ang2                   | s03 |
| 12191291 | nz_abagym_Ang2                     | noise      | abagym_Ang2                   | n01–n05, n07–n13 |
| 12191292 | nz_abagym_EGFR                     | noise      | abagym_EGFR                   | n01–n05, n07–n13 |
| 12191293 | nz_abagym_HER2                     | noise      | abagym_HER2                   | n01–n05, n07–n13 |
| 12191294 | nz_abagym_VEGF                     | noise      | abagym_VEGF                   | n01–n05, n07–n13 |
| 12191295 | cx_abcov_kd                        | complexity | abcov_kd                      | c23 c24 |
| 12191296 | nz_abcov_kd                        | noise      | abcov_kd                      | n01–n05, n07–n13 |
| 12191297 | cx_buchwald_hartwig                | complexity | buchwald_hartwig              | c01 c02 c09 c10 c11 c13 c15–c19 c21–c24 (15 exps) |
| 12191298 | sim_buchwald_hartwig               | similarity | buchwald_hartwig              | s03 s11 |
| 12191299 | nz_buchwald_hartwig                | noise      | buchwald_hartwig              | n01–n05, n07–n13 |
| 12191300 | cx_flab_aggregation_jain           | complexity | flab_aggregation_jain         | c06–c11 c13 c15–c19 c21–c24 (16 exps) |
| 12191301 | sim_flab_aggregation_jain          | similarity | flab_aggregation_jain         | s06 |
| 12191302 | nz_flab_aggregation_jain           | noise      | flab_aggregation_jain         | n01–n05, n07–n13 |
| 12191303 | cx_flab_aggregation_jain_hicrt     | complexity | flab_aggregation_jain_hicrt   | c05 c06–c11 c13 c15–c19 c21–c24 (17 exps) |
| 12191304 | sim_flab_aggregation_jain_hicrt    | similarity | flab_aggregation_jain_hicrt   | s06 |
| 12191305 | nz_flab_aggregation_jain_hicrt     | noise      | flab_aggregation_jain_hicrt   | n01–n05, n07–n13 |
| 12191306 | cx_flab_immunogenicity_marks       | complexity | flab_immunogenicity_marks     | c17 c18 c19 c21 c22 c23 c24 |
| 12191307 | nz_flab_immunogenicity_marks       | noise      | flab_immunogenicity_marks     | n01–n05, n07–n13 |
| 12191308 | cx_flab_pk_jain_fcrnrt             | complexity | flab_pk_jain_fcrnrt           | c06–c11 c13 c15–c19 c21–c24 (16 exps) |
| 12191309 | sim_flab_pk_jain_fcrnrt            | similarity | flab_pk_jain_fcrnrt           | s06 |
| 12191310 | nz_flab_pk_jain_fcrnrt             | noise      | flab_pk_jain_fcrnrt           | n01–n05, n07–n13 |
| 12191311 | cx_flab_pk_jain_heprt              | complexity | flab_pk_jain_heprt            | c06–c11 c13 c15–c19 c21–c24 (16 exps) |
| 12191312 | sim_flab_pk_jain_heprt             | similarity | flab_pk_jain_heprt            | s06 |
| 12191313 | nz_flab_pk_jain_heprt              | noise      | flab_pk_jain_heprt            | n01–n05, n07–n13 |
| 12191314 | cx_flab_polyreact_jain_bvp         | complexity | flab_polyreact_jain_bvp       | c23 c24 |
| 12191315 | nz_flab_polyreact_jain_bvp         | noise      | flab_polyreact_jain_bvp       | n01–n05, n07–n13 |
| 12191316 | cx_flab_polyreact_jain_cic         | complexity | flab_polyreact_jain_cic       | c23 c24 |
| 12191317 | nz_flab_polyreact_jain_cic         | noise      | flab_polyreact_jain_cic       | n01–n05, n07–n13 |
| 12191318 | cx_flab_thermo_jain                | complexity | flab_thermo_jain              | c05 c06–c11 c13 c15–c19 c21–c24 (17 exps) |
| 12191319 | sim_flab_thermo_jain               | similarity | flab_thermo_jain              | s06 s08 s10 s11 s12 s13 |
| 12191320 | nz_flab_thermo_jain                | noise      | flab_thermo_jain              | n01–n05, n07–n13 |
| 12191321 | cx_ginkgo_acsins                   | complexity | ginkgo_acsins                 | c05 c06–c11 c13 c15–c19 c21–c24 (17 exps) |
| 12191322 | sim_ginkgo_acsins                  | similarity | ginkgo_acsins                 | s06 s08 s10 s11 s12 s13 |
| 12191323 | nz_ginkgo_acsins                   | noise      | ginkgo_acsins                 | n01–n05, n07–n13 |
| 12191324 | cx_ginkgo_colloidal                | complexity | ginkgo_colloidal              | c05 c06–c11 c13 c15–c19 c21–c24 (17 exps) |
| 12191325 | sim_ginkgo_colloidal               | similarity | ginkgo_colloidal              | s06 s08 s10 s11 s12 s13 |
| 12191326 | nz_ginkgo_colloidal                | noise      | ginkgo_colloidal              | n01–n05, n07–n13 |
| 12191327 | cx_ginkgo_heparin                  | complexity | ginkgo_heparin                | c09–c11 c13 c15–c19 c21–c24 (13 exps) |
| 12191328 | nz_ginkgo_heparin                  | noise      | ginkgo_heparin                | n01–n05, n07–n13 |
| 12191329 | cx_ginkgo_hic                      | complexity | ginkgo_hic                    | c05 c06–c11 c13 c15–c19 c21–c24 (17 exps) |
| 12191330 | sim_ginkgo_hic                     | similarity | ginkgo_hic                    | s06 s08 s10 s11 s12 s13 |
| 12191331 | nz_ginkgo_hic                      | noise      | ginkgo_hic                    | n01–n05, n07–n13 |
| 12191332 | cx_ginkgo_pr_cho                   | complexity | ginkgo_pr_cho                 | c05 c06–c11 c13 c15–c19 c21–c24 (17 exps) |
| 12191333 | sim_ginkgo_pr_cho                  | similarity | ginkgo_pr_cho                 | s06 s08 s10 s11 s12 s13 |
| 12191334 | nz_ginkgo_pr_cho                   | noise      | ginkgo_pr_cho                 | n01–n05, n07–n13 |
| 12191335 | cx_ginkgo_titer                    | complexity | ginkgo_titer                  | c05 c06–c11 c13 c15–c19 c21–c24 (17 exps) |
| 12191336 | sim_ginkgo_titer                   | similarity | ginkgo_titer                  | s06 s07 s08 s10 s11 s12 s13 |
| 12191337 | nz_ginkgo_titer                    | noise      | ginkgo_titer                  | n01–n05, n07–n13 |
| 12191338 | cx_humanness_therapeutic           | complexity | humanness_therapeutic         | c22 c24 (c23 dropped — clf) |
| 12191339 | nz_humanness_therapeutic           | noise      | humanness_therapeutic         | n01–n05, n07–n13 |
| 12191340 | cx_tz_her2_binary                  | complexity | tz_her2_binary                | c01–c11 c13 c15–c19 c21 c22 c24 (19 exps, c23 dropped) |
| 12191341 | sim_tz_her2_binary                 | similarity | tz_her2_binary                | s01–s08 s10–s13 (12 exps) |
| 12191342 | nz_tz_her2_binary                  | noise      | tz_her2_binary                | n01–n05, n07–n13 |
| 12191343 | cx_tz_her2_multiclass              | complexity | tz_her2_multiclass            | c05–c11 c13 c15–c19 c21 c22 c24 (17 exps, c23 dropped) |
| 12191344 | nz_tz_her2_multiclass              | noise      | tz_her2_multiclass            | n01–n05, n07–n13 |
| 12191345 | opig_humanness                     | opig       | humanness_species + humanness_therapeutic | full opig_humanness |
| 12191346 | opig_trastuzumab                   | opig       | tz_her2_*                     | full opig_trastuzumab_enrichment |
| 12191347 | opig_ginkgo                        | opig       | all ginkgo_*                  | full opig_ginkgo_developability |
| 12191348 | opig_kingsmith                     | opig       | buchwald_hartwig              | full opig_kingsmith_component_splits |

**Total: 61 jobs.** Range 12191288–12191348.

## Per-category counts

- complexity: 21 jobs (1 with 19 exps, 4 with 16–17, 1 with 15, 13 fill-ins)
- similarity: 13 jobs (1 with 12, 5 with 6–7, 7 small fill-ins)
- noise:      23 jobs (all 12-experiment n01–n13 minus n06)
- opig:        4 jobs (humanness, trastuzumab, ginkgo, kingsmith)

## Status check (run on server)

```bash
# In-flight count
squeue -u $USER | awk 'NR>1 && $1+0 >= 12191288 && $1+0 <= 12191348 {print $5}' | sort | uniq -c

# JSONs written since submission
find tests/results -name '*.json' -mmin -1440 | wc -l

# Re-run gap audit to see which datasets cleared
python tests/audit_done_dataset_gaps.py --no-commands
```

## Known caveats

- 12158437 (`cx_abagym_lysozyme`) is still running from an earlier wave — not in
  the 23 done set, won't conflict with these jobs.
- Capped strategy restrictions (`_DATASET_STRATEGY_SKIPS` in
  `find_best_hybrid.py`) only apply to large-n datasets; none of the 23 done
  datasets here are large enough to trigger them.
- ngboost is no longer trained as a downstream model in any of the 23 nz_*
  jobs (commit `1b2089a`).
- OPIG ginkgo job runs all 7 ginkgo datasets sequentially in one job; if it
  walls out at 24h, per-dataset re-submission may be needed.
