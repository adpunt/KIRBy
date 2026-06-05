# AUDIT_REDUNDANCY

Variants analyzed: 70
Val/test patients: 423
Disagreement threshold (|delta normalized rank|): 0.1
Cluster threshold (rho >=): 0.99
Effective clusters: hep=38, death=18

## Cluster summary (hepatic)

| cluster | size | representative | members |
|--------:|-----:|----------------|---------|
| 32 | 13 | `meta_avg_all` | BEST_BET, FINAL, meta_avg, meta_avg_all, meta_avg_aware_weighted, meta_avg_grand_all, ... (+7) |
| 22 | 8 | `honest` | honest, new_binwise_corrected, new_bma, noise_censoraware, noise_classbalance, noise_huber, noise_multiseed, round1b |
| 17 | 7 | `meta_avg_new_variants_only` | bootstrap_dropout, holdout_pick, meta_avg_aware_weighted_no_nnls, meta_avg_aware_weighted_v2, meta_avg_grand_all_v2, meta_avg_new_variants_only, weighted_anomaly_per_patient |
| 19 | 3 | `failure_fp_suppress` | failure_fn_recovery, failure_fp_suppress, new_inference_dropout |
| 25 | 3 | `best_per_endpoint` | best_per_endpoint, rank_mean, rank_trimmed |
| 24 | 2 | `rank_median` | best_combined, rank_median |
| 31 | 2 | `subgroup_routed` | subgroup_hep_only, subgroup_routed |
| 33 | 2 | `unc_bootstrap_var` | unc_bootstrap_var, unc_conformal |
| 0 | 1 | `HIER` | HIER |
| 1 | 1 | `LEARNED_STACK` | LEARNED_STACK |
| 2 | 1 | `anomaly_blend` | anomaly_blend |
| 3 | 1 | `anomaly_blend_meta` | anomaly_blend_meta |
| 4 | 1 | `anomaly_router_meta` | anomaly_router_meta |
| 5 | 1 | `best_combined_v2` | best_combined_v2 |
| 6 | 1 | `cluster_route` | cluster_route |
| 7 | 1 | `cluster_router_meta` | cluster_router_meta |
| 8 | 1 | `coxnet_stack_with_uncertainty` | coxnet_stack_with_uncertainty |
| 9 | 1 | `disagreement_route` | disagreement_route |
| 10 | 1 | `disagreement_router_meta` | disagreement_router_meta |
| 11 | 1 | `diverse` | diverse |
| 12 | 1 | `failure_hard_specialist` | failure_hard_specialist |
| 13 | 1 | `failure_robust` | failure_robust |
| 14 | 1 | `failure_xref_aware` | failure_xref_aware |
| 15 | 1 | `isotonic_avg` | isotonic_avg |
| 16 | 1 | `meta_avg_aware_top5` | meta_avg_aware_top5 |
| 18 | 1 | `new_cohort_split` | new_cohort_split |
| 20 | 1 | `new_per_endpoint` | new_per_endpoint |
| 21 | 1 | `nnls_stack` | nnls_stack |
| 23 | 1 | `noise_ipcw` | noise_ipcw |
| 26 | 1 | `residual_greedy` | residual_greedy |
| 27 | 1 | `round3` | round3 |
| 28 | 1 | `soft_blend_disagreement` | soft_blend_disagreement |
| 29 | 1 | `specialist_override` | specialist_override |
| 30 | 1 | `subgroup_blend` | subgroup_blend |
| 34 | 1 | `unc_disagree_soft` | unc_disagree_soft |
| 35 | 1 | `unc_mahal_blend` | unc_mahal_blend |
| 36 | 1 | `unc_specialist` | unc_specialist |
| 37 | 1 | `uncertainty_route` | uncertainty_route |

## Cluster summary (death)

| cluster | size | representative | members |
|--------:|-----:|----------------|---------|
| 10 | 43 | `diverse` | BEST_BET, FINAL, anomaly_blend, anomaly_blend_meta, anomaly_router_meta, best_combined, ... (+37) |
| 11 | 6 | `new_binwise_corrected` | new_binwise_corrected, noise_censoraware, noise_classbalance, noise_huber, noise_multiseed, round1b |
| 6 | 5 | `HIER` | HIER, failure_fn_recovery, failure_fp_suppress, failure_robust, new_inference_dropout |
| 14 | 2 | `unc_bootstrap_var` | unc_bootstrap_var, unc_conformal |
| 0 | 1 | `LEARNED_STACK` | LEARNED_STACK |
| 1 | 1 | `coxnet_stack_with_uncertainty` | coxnet_stack_with_uncertainty |
| 2 | 1 | `failure_hard_specialist` | failure_hard_specialist |
| 3 | 1 | `failure_xref_aware` | failure_xref_aware |
| 4 | 1 | `isotonic_avg` | isotonic_avg |
| 5 | 1 | `meta_avg_new_variants_only` | meta_avg_new_variants_only |
| 7 | 1 | `nnls_stack` | nnls_stack |
| 8 | 1 | `noise_ipcw` | noise_ipcw |
| 9 | 1 | `rank_median` | rank_median |
| 12 | 1 | `round3` | round3 |
| 13 | 1 | `subgroup_routed` | subgroup_routed |
| 15 | 1 | `unc_disagree_soft` | unc_disagree_soft |
| 16 | 1 | `unc_mahal_blend` | unc_mahal_blend |
| 17 | 1 | `unc_specialist` | unc_specialist |

## Top-10 most-unique variants

| variant | uniqueness_score | uniq_hep | uniq_death |
|---------|-----------------:|---------:|-----------:|
| `coxnet_stack_with_uncertainty` | 0.7113 | 0.7144 | 0.7041 |
| `LEARNED_STACK` | 0.2313 | 0.3239 | 0.0154 |
| `round3` | 0.1892 | 0.2398 | 0.0710 |
| `subgroup_routed` | 0.1483 | 0.0000 | 0.4943 |
| `disagreement_route` | 0.1241 | 0.1766 | 0.0015 |
| `nnls_stack` | 0.1236 | 0.0739 | 0.2398 |
| `uncertainty_route` | 0.1158 | 0.1654 | 0.0000 |
| `new_cohort_split` | 0.0857 | 0.1225 | 0.0000 |
| `subgroup_blend` | 0.0760 | 0.1086 | 0.0000 |
| `cluster_route` | 0.0741 | 0.1048 | 0.0024 |

## Top-10 most-redundant pairs (avg rho hep/death)

| variant_i | variant_j | rho_avg | rho_hep | rho_death |
|-----------|-----------|--------:|--------:|----------:|
| `rank_mean` | `rank_trimmed` | 1.0000 | 1.0000 | 1.0000 |
| `proven_meta` | `proven_meta_weighted` | 1.0000 | 1.0000 | 1.0000 |
| `meta_avg_top5` | `meta_avg_weighted` | 1.0000 | 1.0000 | 1.0000 |
| `ultra_meta` | `ultra_meta_weighted` | 0.9999 | 0.9999 | 1.0000 |
| `BEST_BET` | `ultra_meta_weighted` | 0.9998 | 0.9997 | 1.0000 |
| `BEST_BET` | `ultra_meta` | 0.9998 | 0.9997 | 1.0000 |
| `meta_avg_aware_weighted_no_nnls` | `meta_avg_aware_weighted_v2` | 0.9998 | 0.9999 | 0.9997 |
| `BEST_BET` | `meta_avg_grand_all` | 0.9997 | 0.9996 | 0.9999 |
| `bootstrap_dropout` | `meta_avg_aware_weighted_no_nnls` | 0.9997 | 0.9994 | 0.9999 |
| `BEST_BET` | `meta_avg_aware_weighted` | 0.9996 | 0.9993 | 1.0000 |

## ASCII heatmap (first 20 variants, hepatic)

Legend: `###` rho>=0.99, `## ` rho>=0.90, ` # ` rho>=0.80, ` . ` rho>=0.50.

```
         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
BEST_B  .  ###  #   .   #  ##  ##  ##  ##  ##  ##   .  ##       .   #  ##   #   #  ## 
FINAL  ###  .   #   .   #  ##  ##  ##  ##  ##  ##   .  ##       .   #  ##   #   #  ## 
HIER    #   #   .   .   #   .   #   #   #   #   #   .   #       .   .   #  ##  ##   # 
LEARNE  .   .   .   .   .       .   .   .   .   .       .               .   .   .   . 
anomal  #   #   #   .   .   #   #   #   #   #   #   .   #       .   .   #   #   #   # 
anomal ##  ##   .       #   .  ##   #   #  ##  ##   #   #       .   #   #   .   .  ## 
anomal ##  ##   #   .   #  ##   .   #  ##  ##  ##   .   #       .   .   #   #   #  ## 
best_c ##  ##   #   .   #   #   #   .  ##   #  ##   .  ##       .   #   #   #   #  ## 
best_c ##  ##   #   .   #   #  ##  ##   .   #  ##   .   #       .   .   #   #   #  ## 
best_p ##  ##   #   .   #  ##  ##   #   #   .  ##   .  ##       .   #  ##   #   #  ## 
bootst ##  ##   #   .   #  ##  ##  ##  ##  ##   .   .  ##       .   #  ##   #   #  ## 
cluste  .   .   .       .   #   .   .   .   .   .   .   .       .   .   .   .   .   . 
cluste ##  ##   #   .   #   #   #  ##   #  ##  ##   .   .       .   #   #   #   #  ## 
coxnet                                                      .                         
disagr  .   .   .       .   .   .   .   .   .   .   .   .       .   .   .   .   .   . 
disagr  #   #   .       .   #   .   #   .   #   #   .   #       .   .   #   .   .   # 
divers ##  ##   #   .   #   #   #   #   #  ##  ##   .   #       .   #   .   #   #  ## 
failur  #   #  ##   .   #   .   #   #   #   #   #   .   #       .   .   #   .  ###  # 
failur  #   #  ##   .   #   .   #   #   #   #   #   .   #       .   .   #  ###  .   # 
failur ##  ##   #   .   #  ##  ##  ##  ##  ##  ##   .  ##       .   #  ##   #   #   . 
```

## ASCII heatmap (first 20 variants, death)

```
         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
BEST_B  .  ###  #  ##  ### ### ### ### ### ### ### ### ###     ### ### ##   #   #  ## 
FINAL  ###  .   #  ##  ### ### ### ### ### ### ### ### ###     ### ### ###  #   #  ## 
HIER    #   #   .   #   #   #   #   #   #   #   #   #   #       #   #  ##  ### ###  # 
LEARNE ##  ##   #   .  ##  ##  ##  ##  ##  ##  ##  ##  ##      ##  ##  ##   #   #  ## 
anomal ### ###  #  ##   .  ### ### ##  ##  ##  ### ### ##      ### ### ##   #   #  ## 
anomal ### ###  #  ##  ###  .  ### ### ### ### ### ### ##      ### ### ##   #   #  ## 
anomal ### ###  #  ##  ### ###  .  ### ### ### ### ### ##      ### ### ##   #   #  ## 
best_c ### ###  #  ##  ##  ### ###  .  ### ### ### ##  ##      ##  ##  ###  #   #  ## 
best_c ### ###  #  ##  ##  ### ### ###  .  ### ### ##  ##      ##  ##  ###  #   #  ## 
best_p ### ###  #  ##  ##  ### ### ### ###  .  ### ##  ##      ##  ##  ###  #   #  ## 
bootst ### ###  #  ##  ### ### ### ### ### ###  .  ### ###     ### ### ###  #   #  ## 
cluste ### ###  #  ##  ### ### ### ##  ##  ##  ###  .  ##      ### ### ##   #   #  ## 
cluste ### ###  #  ##  ##  ##  ##  ##  ##  ##  ### ##   .      ### ##  ##   #   #  ## 
coxnet                                                      .                         
disagr ### ###  #  ##  ### ### ### ##  ##  ##  ### ### ###      .  ### ##   #   #  ## 
disagr ### ###  #  ##  ### ### ### ##  ##  ##  ### ### ##      ###  .  ##   #   #  ## 
divers ##  ### ##  ##  ##  ##  ##  ### ### ### ### ##  ##      ##  ##   .   #   #  ## 
failur  #   #  ###  #   #   #   #   #   #   #   #   #   #       #   #   #   .  ###  # 
failur  #   #  ###  #   #   #   #   #   #   #   #   #   #       #   #   #  ###  .   # 
failur ##  ##   #  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##      ##  ##  ##   #   #   . 
```

Full heatmaps saved as PNG: `variant_correlation_hep.png`, `variant_correlation_death.png`.

## Files

- /Users/apunt/repos/KIRBy/tests/results/annitia/full/variant_correlation_hep.csv
- /Users/apunt/repos/KIRBy/tests/results/annitia/full/variant_correlation_death.csv
- /Users/apunt/repos/KIRBy/tests/results/annitia/full/variant_clusters.csv
- /Users/apunt/repos/KIRBy/tests/results/annitia/full/contested_patients_per_pair.csv
- /Users/apunt/repos/KIRBy/tests/results/annitia/full/variant_correlation_hep.png
- /Users/apunt/repos/KIRBy/tests/results/annitia/full/variant_correlation_death.png
