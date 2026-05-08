# validate_top — rigorous candidate validation
Goal: decide which candidates are worth submission slots vs OOF noise.

## hepatic

### Point OOF C

| Candidate | C |
|---|---|
| anomaly_blend_meta | 0.9142 |
| rank_median | 0.9078 |
| specialist_override | 0.9061 |
| weighted_anomaly_per_patient | 0.9057 |
| cluster_router_meta | 0.9061 |
| rank_mean | 0.9034 |
| meta_avg_aware_weighted_v4 | 0.9077 |
| cherry_pick_hep_anomBlend_death_wapp | 0.9142 |

### Bootstrap (B=1000, stratified by event)

| Candidate | Mean C | Lo 95% | Hi 95% | Std |
|---|---|---|---|---|
| anomaly_blend_meta | 0.9129 | 0.8688 | 0.9515 | 0.0215 |
| rank_median | 0.9065 | 0.8559 | 0.9504 | 0.0247 |
| specialist_override | 0.9044 | 0.8594 | 0.9411 | 0.0211 |
| weighted_anomaly_per_patient | 0.9042 | 0.8536 | 0.9474 | 0.0247 |
| cluster_router_meta | 0.9045 | 0.8598 | 0.9413 | 0.0210 |
| rank_mean | 0.9020 | 0.8540 | 0.9416 | 0.0229 |
| meta_avg_aware_weighted_v4 | 0.9062 | 0.8570 | 0.9481 | 0.0238 |
| cherry_pick_hep_anomBlend_death_wapp | 0.9129 | 0.8688 | 0.9515 | 0.0215 |

### Win-rate vs current LB best (`meta_avg_aware_weighted`)

| Candidate | P(beats anchor) | Δmean |
|---|---|---|
| anomaly_blend_meta | 0.980 | +0.0067 |
| cherry_pick_hep_anomBlend_death_wapp | 0.980 | +0.0067 |
| rank_median | 0.527 | +0.0003 |
| cluster_router_meta | 0.390 | -0.0017 |
| specialist_override | 0.382 | -0.0018 |
| rank_mean | 0.268 | -0.0041 |
| weighted_anomaly_per_patient | 0.072 | -0.0020 |

### Per-cohort C-index breakdown

| Candidate |anomQ1|anomQ2|anomQ3|anomQ4|ageQ1|ageQ2|ageQ3|ageQ4|
|---|---|---|---|---|---|---|---|---|
| anomaly_blend_meta | 0.914 | 0.925 | 0.911 | 0.900 | 0.978 | 0.859 | 0.898 | 0.957 |
| rank_median | 0.873 | 0.933 | 0.925 | 0.892 | 0.985 | 0.847 | 0.883 | 0.952 |
| specialist_override | 0.921 | 0.908 | 0.880 | 0.897 | 0.957 | 0.845 | 0.906 | 0.946 |
| weighted_anomaly_per_patient | 0.886 | 0.917 | 0.903 | 0.902 | 0.970 | 0.836 | 0.902 | 0.953 |
| cluster_router_meta | 0.921 | 0.908 | 0.880 | 0.893 | 0.957 | 0.845 | 0.907 | 0.943 |
| rank_mean | 0.921 | 0.908 | 0.880 | 0.876 | 0.965 | 0.855 | 0.882 | 0.938 |
| meta_avg_aware_weighted_v4 | 0.896 | 0.921 | 0.903 | 0.902 | 0.974 | 0.844 | 0.900 | 0.952 |
| cherry_pick_hep_anomBlend_death_wapp | 0.914 | 0.925 | 0.911 | 0.900 | 0.978 | 0.859 | 0.898 | 0.957 |

### Train-OOF vs val-prediction distribution (KS p-value)

Lower p = larger distribution shift between OOF and val. Big shift → variant likely overfits.

| Candidate | KS p-value |
|---|---|
| anomaly_blend_meta | 1.000 |
| rank_median | 1.000 |
| specialist_override | 1.000 |
| weighted_anomaly_per_patient | 1.000 |
| cluster_router_meta | 1.000 |
| rank_mean | 1.000 |
| meta_avg_aware_weighted_v4 | 1.000 |
| cherry_pick_hep_anomBlend_death_wapp | 1.000 |

## death

### Point OOF C

| Candidate | C |
|---|---|
| anomaly_blend_meta | 0.9699 |
| rank_median | 0.9743 |
| specialist_override | 0.9745 |
| weighted_anomaly_per_patient | 0.9750 |
| cluster_router_meta | 0.9697 |
| rank_mean | 0.9738 |
| meta_avg_aware_weighted_v4 | 0.9732 |
| cherry_pick_hep_anomBlend_death_wapp | 0.9750 |

### Bootstrap (B=1000, stratified by event)

| Candidate | Mean C | Lo 95% | Hi 95% | Std |
|---|---|---|---|---|
| anomaly_blend_meta | 0.9699 | 0.9563 | 0.9809 | 0.0063 |
| rank_median | 0.9745 | 0.9648 | 0.9824 | 0.0044 |
| specialist_override | 0.9745 | 0.9653 | 0.9823 | 0.0045 |
| weighted_anomaly_per_patient | 0.9752 | 0.9651 | 0.9833 | 0.0045 |
| cluster_router_meta | 0.9697 | 0.9557 | 0.9810 | 0.0065 |
| rank_mean | 0.9739 | 0.9645 | 0.9817 | 0.0045 |
| meta_avg_aware_weighted_v4 | 0.9733 | 0.9628 | 0.9824 | 0.0050 |
| cherry_pick_hep_anomBlend_death_wapp | 0.9752 | 0.9651 | 0.9833 | 0.0045 |

### Win-rate vs current LB best (`meta_avg_aware_weighted`)

| Candidate | P(beats anchor) | Δmean |
|---|---|---|
| weighted_anomaly_per_patient | 0.954 | +0.0019 |
| cherry_pick_hep_anomBlend_death_wapp | 0.954 | +0.0019 |
| specialist_override | 0.657 | +0.0012 |
| rank_median | 0.639 | +0.0012 |
| rank_mean | 0.550 | +0.0006 |
| cluster_router_meta | 0.077 | -0.0036 |
| anomaly_blend_meta | 0.023 | -0.0034 |

### Per-cohort C-index breakdown

| Candidate |anomQ1|anomQ2|anomQ3|anomQ4|ageQ1|ageQ2|ageQ3|ageQ4|
|---|---|---|---|---|---|---|---|---|
| anomaly_blend_meta | 0.955 | 0.966 | 0.969 | 0.973 | 0.960 | 0.986 | 0.982 | 0.939 |
| rank_median | 0.983 | 0.962 | 0.974 | 0.969 | 0.972 | 0.991 | 0.977 | 0.949 |
| specialist_override | 0.986 | 0.964 | 0.970 | 0.975 | 0.960 | 0.979 | 0.976 | 0.955 |
| weighted_anomaly_per_patient | 0.984 | 0.965 | 0.970 | 0.971 | 0.969 | 0.989 | 0.982 | 0.947 |
| cluster_router_meta | 0.956 | 0.963 | 0.972 | 0.974 | 0.963 | 0.986 | 0.978 | 0.943 |
| rank_mean | 0.986 | 0.964 | 0.970 | 0.971 | 0.957 | 0.983 | 0.974 | 0.951 |
| meta_avg_aware_weighted_v4 | 0.978 | 0.964 | 0.970 | 0.972 | 0.966 | 0.988 | 0.982 | 0.944 |
| cherry_pick_hep_anomBlend_death_wapp | 0.984 | 0.965 | 0.970 | 0.971 | 0.969 | 0.989 | 0.982 | 0.947 |

### Train-OOF vs val-prediction distribution (KS p-value)

Lower p = larger distribution shift between OOF and val. Big shift → variant likely overfits.

| Candidate | KS p-value |
|---|---|
| anomaly_blend_meta | 1.000 |
| rank_median | 0.997 |
| specialist_override | 0.996 |
| weighted_anomaly_per_patient | 1.000 |
| cluster_router_meta | 0.730 |
| rank_mean | 1.000 |
| meta_avg_aware_weighted_v4 | 1.000 |
| cherry_pick_hep_anomBlend_death_wapp | 1.000 |

## Combined LB (0.7*C_hep + 0.3*C_death) bootstrap

### Combined-LB bootstrap (B=1000)

| Candidate | Mean LB | Lo 95% | Hi 95% | Std |
|---|---|---|---|---|
| cherry_pick_hep_anomBlend_death_wapp | 0.9323 | 0.9025 | 0.9569 | 0.0142 |
| anomaly_blend_meta | 0.9308 | 0.9002 | 0.9551 | 0.0143 |
| rank_median | 0.9276 | 0.8925 | 0.9556 | 0.0163 |
| meta_avg_aware_weighted_v4 | 0.9273 | 0.8943 | 0.9550 | 0.0158 |
| weighted_anomaly_per_patient | 0.9264 | 0.8912 | 0.9551 | 0.0163 |
| specialist_override | 0.9260 | 0.8960 | 0.9503 | 0.0143 |
| cluster_router_meta | 0.9247 | 0.8945 | 0.9493 | 0.0143 |
| rank_mean | 0.9247 | 0.8939 | 0.9514 | 0.0148 |

### Pairwise: P(candidate combined LB > anchor combined LB)

| Candidate | P(beat anchor) | Δmean LB |
|---|---|---|
| cherry_pick_hep_anomBlend_death_wapp | 0.990 | +0.0050 |
| anomaly_blend_meta | 0.940 | +0.0035 |
| rank_median | 0.560 | +0.0004 |
| specialist_override | 0.389 | -0.0012 |
| rank_mean | 0.296 | -0.0026 |
| cluster_router_meta | 0.282 | -0.0026 |
| weighted_anomaly_per_patient | 0.226 | -0.0009 |

