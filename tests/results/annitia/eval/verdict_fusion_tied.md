# Submission verdict

Tag: `fusion_tied`

Recipe: ensemble (rank-mean) of all hold-out-tied methods per endpoint.
Tie band: `1.0* SE`.

## hepatic

Components: `F09_trimmed_mean`, `F03_hodges_lehmann`, `anchor_rank_mean`, `F04_bates_granger`, `F05_stacking_l2`

## death

Components: `F01_caruana_bagged`, `F09_trimmed_mean`, `F03_hodges_lehmann`

## Why this is safer than cherry-pick

The cherry-pick failure on 2026-05-08 (`anomaly_blend_meta` × `weighted_anomaly_per_patient`) had OOF 0.9323 / LB 0.898 (gap +0.034) because both components fit weights on OOF. Here, every method was fit on `train_idx` and evaluated on `holdout_idx` — none saw the validation labels during fitting. The ensemble of hold-out-tied methods further dampens any remaining selection bias.