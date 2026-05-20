# Job tracker — 2026-05-20 cancel + resubmit (capped model-set restriction)

66 jobs submitted 2026-05-20 05:07 BST after cancel of 37 STUCK_CAPPED_F0 jobs from the May 14 + May 17 waves. All new jobs use:

- **Commit `7434b19`** — capped restricted via `_DATASET_STRATEGY_SKIPS` + `_DOMAIN_STRATEGY_SKIPS` so cache keys match between precompute and experiment-time calls (no on-the-fly recomputation).
- GP cap (n ≤ 1000), nucleotide ridge/knn/xgboost domain skip preserved.
- `KIRBY_CACHE_DIR` = `/data/stat-ecr/scat9264/.cache/kirby`.

Wall: 7 days (long partition).

## What still runs from prior waves (NOT cancelled)

- **Old wave 12140xxx in IN_EXPERIMENTS state** — writing JSONs to `tests/results/`. Wall hits ~2026-05-20 ~18:00 BST.
- **New wave 12158xxx in IN_EXPERIMENTS state** — mostly nucleotide cx_nt_*. Some may be doing precompute fold-searches (classifier couldn't disambiguate). These may overlap with new submissions below; cleanup commands at end of this file.
- **DTA cx jobs 12155652 (cx_kiba), 12155653 (cx_bindingdb_kd)** — IN_EXPERIMENTS, untouched.

## Cancelled in this wave (37 STUCK_CAPPED_F0 jobs)

Old wave (13): 12140223, 12140224, 12140225, 12140229, 12140234, 12140237, 12140254, 12140256, 12140346, 12140349, 12140356, 12140357, 12140365

New wave (24): 12158429, 12158430, 12158431, 12158432, 12158433, 12158434, 12158436, 12158440, 12158441, 12158450, 12158452, 12158454, 12158456, 12158458, 12158460, 12158462, 12158468, 12158470, 12158472, 12158474, 12158477, 12158478, 12158480, 12158482

## New wave job IDs (12179348–12179414)

| dataset | cx | sim |
|---|---|---|
| adme_hlm | 12179348 | 12179349 |
| adme_mdr1 | 12179350 | 12179351 |
| adme_rlm | 12179352 | 12179353 |
| caco2 | 12179354 | 12179355 |
| herg | 12179356 | 12179357 |
| herg_ki | 12179358 | 12179359 |
| logd | 12179360 | 12179361 |
| qm9 | 12179362 | 12179363 |
| bh_doyle_plates | 12179364 | 12179365 |
| tcga_brca_pam50 | 12179366 | 12179367 |
| bindingdb_kd | 12179368 | 12179369 |
| kiba | 12179370 | 12179371 |
| gb_demo_coding_vs_intergenomic_seqs | 12179372 | 12179373 |
| gb_demo_human_or_worm | 12179374 | 12179375 |
| gb_human_enhancers_cohn | 12179376 | 12179377 |
| gb_human_nontata_promoters | 12179378 | 12179379 |
| nt_H2AFZ | 12179380 | 12179381 |
| nt_H3K27ac | 12179382 | 12179383 |
| nt_H3K27me3 | 12179384 | 12179385 |
| nt_H3K36me3 | 12179386 | 12179387 |
| nt_H3K4me1 | 12179388 | 12179389 |
| nt_H3K4me2 | 12179390 | 12179391 |
| nt_H3K4me3 | 12179392 | 12179393 |
| nt_H3K9ac | 12179394 | 12179395 |
| nt_H3K9me3 | 12179396 | 12179397 |
| nt_H4K20me1 | 12179398 | 12179399 |
| nt_enhancers | 12179400 | 12179401 |
| nt_enhancers_types | 12179402 | 12179403 |
| nt_promoter_all | 12179404 | 12179406 |
| nt_promoter_no_tata | 12179407 | 12179408 |
| nt_promoter_tata | 12179409 | 12179410 |
| nt_splice_sites_acceptors | 12179411 | 12179412 |
| nt_splice_sites_donors | 12179413 | 12179414 |

(33 datasets × 2 directions = 66 jobs. Note one ID gap at 12179405 — that's the sim sbatch that ran where 12179405 = cx_nt_promoter_all's pair, gap from a queued race.)

## Capped model-set restrictions in effect (from `_DATASET_STRATEGY_SKIPS`)

| dataset | strategy | skipped models | active models after skip+_max_samples |
|---|---|---|---|
| adme_hlm, adme_mdr1, adme_rlm, caco2, logd, bh_doyle_plates | capped | mlp, svm, knn | rf, xgboost, ridge, lightgbm |
| herg, qm9 | capped | mlp, svm, knn, rf | xgboost, ridge, lightgbm |
| bindingdb_kd, kiba | capped | mlp, rf, knn | xgboost, ridge, lightgbm (svm auto-skip at n>10k) |
| all nt_*, gb_* | capped | mlp (+ domain skips ridge, knn, xgboost) | rf, lightgbm |
| tcga_brca_pam50 | all | gp | rf, xgboost, ridge, knn, lightgbm, mlp, svm |
| herg_ki | (none specific) | — | full default minus _max_samples auto-skips |

## Cleanup: identify and cancel 12158xxx duplicates

The previous duplicate-detection awk was flawed (matched the new wave by job name, which is what we just submitted). Corrected version — finds 12158xxx jobs whose dataset overlaps our new submission:

```bash
DATASETS="adme_hlm adme_mdr1 adme_rlm caco2 herg herg_ki logd qm9 bh_doyle_plates \
tcga_brca_pam50 bindingdb_kd kiba gb_demo_coding_vs_intergenomic_seqs \
gb_demo_human_or_worm gb_human_enhancers_cohn gb_human_nontata_promoters \
nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 \
nt_H3K9ac nt_H3K9me3 nt_H4K20me1 nt_enhancers nt_enhancers_types \
nt_promoter_all nt_promoter_no_tata nt_promoter_tata \
nt_splice_sites_acceptors nt_splice_sites_donors"

# For each dataset, find 12158xxx (NEW wave from May 17) and 12140xxx (OLD wave from May 14)
# jobs still running. The 12140xxx are mostly job-name=exp_%j (not findable by name)
# but the 12158xxx have descriptive names — those are the real duplicate risk.
DUPES=()
for ds in $DATASETS; do
    for direction in cx sim; do
        # Check squeue for 12158xxx jobs with matching name (truncated by SLURM to 8 chars)
        truncated="${direction}_${ds}"
        truncated="${truncated:0:8}"
        old=$(squeue -u $USER -h -o '%i %j' \
            | awk -v t="$truncated" '$1+0 >= 12158000 && $1+0 < 12179000 && index($2, t)==1 {print $1}' \
            | head -1)
        if [[ -n "$old" ]]; then
            full_name=$(squeue -h -j $old -o '%j')
            echo "DUPLICATE: $old name=$full_name overlaps new ${direction}_${ds}"
            DUPES+=($old)
        fi
    done
done

echo
echo "Total duplicates: ${#DUPES[@]}"
echo "To cancel: scancel ${DUPES[@]}"
```

The 12158xxx jobs in IN_EXPERIMENTS state are doing fold-searches whose contents we can't tell from the log (could be precompute or experiment). If they're in precompute and you cancel, the new 12179xxx jobs do the work with the restricted-capped model set. If they were in actual experiment scripts, you lose whatever incremental experiment JSON they were producing.

Lower-stakes alternative: don't cancel duplicates, let SLURM scheduler manage. The new jobs will mark stale caches and recompute capped; the old jobs will hit the same stale-detection on their next cache read and either die cleanly or recompute too. Some redundancy but no work is destroyed.

## Snapshot of completed experiments at submission time

See `snapshots/snapshot_20260520_*.txt`.

Headline counts at 2026-05-20 05:07 BST:

- complexity: 211 JSONs across c01–c24 (heavy weight on antibody/abagym/flab/ginkgo)
- similarity: 219 JSONs across s01–s13
- noise: 0
- federated: 0

Many of these are SMALL antibody datasets (abagym_*, flab_*, ginkgo_*) — the publication's heavyweight molecular/DTA/nucleotide work depends on this 2026-05-20 wave producing JSONs over the next 5–7 days.
