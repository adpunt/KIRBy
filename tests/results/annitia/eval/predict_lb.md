# Calibrated LB prediction

Conservative bands using empirical OOF→LB gap heuristic:
- ensemble: bias = 0.005
- single_holdout: bias = 0.015
- meta_meta: bias = 0.030 (cherry_pick observed)

| method | hold LB | ±SE | recipe | pred LB low | point | high |
|---|---|---|---|---|---|---|
| F09_trimmed_mean | 0.9258 | 0.0219 | ensemble | 0.8769 | 0.9208 | 0.9647 |
| F03_hodges_lehmann | 0.9252 | 0.0241 | ensemble | 0.8721 | 0.9202 | 0.9684 |
| anchor_rank_mean | 0.9229 | 0.0227 | ensemble | 0.8726 | 0.9179 | 0.9632 |
| F04_bates_granger | 0.9228 | 0.0232 | single_holdout | 0.8613 | 0.9078 | 0.9542 |
| F05_stacking_l2 | 0.9177 | 0.0229 | single_holdout | 0.8570 | 0.9027 | 0.9484 |
| F01_caruana_bagged | 0.8999 | 0.0231 | single_holdout | 0.8388 | 0.8849 | 0.9310 |
| F06_diversity_pen | 0.8992 | 0.0279 | single_holdout | 0.8284 | 0.8842 | 0.9400 |
| F02_geomean_rank | 0.7694 | 0.0321 | ensemble | 0.7002 | 0.7644 | 0.8285 |

## Anchor reference
- Existing LB best `meta_avg_aware_weighted` = 0.9302
- A new submission ONLY makes sense if its `pred LB low` exceeds 0.9302.
- Otherwise the expected outcome is regression or noise — keep the slot.