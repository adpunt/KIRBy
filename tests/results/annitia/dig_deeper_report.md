# dig_deeper report

Date: 2026-05-08, total candidates scanned: 108


## Top 25 by LB_combined (0.7*C_hep + 0.3*C_death)

| name                          |    C_hep |   C_death |   LB_combined |   event_quality_hep |   event_quality_death |   anchor_corr_combined |
|:------------------------------|---------:|----------:|--------------:|--------------------:|----------------------:|-----------------------:|
| anomaly_blend_meta            |   0.9142 |    0.9699 |        0.9309 |                 nan |                   nan |                 0.9455 |
| rank_median                   |   0.9078 |    0.9743 |        0.9277 |                 nan |                   nan |                 0.9149 |
| specialist_override           |   0.9061 |    0.9745 |        0.9266 |                 nan |                   nan |                 0.9262 |
| weighted_anomaly_per_patient  |   0.9057 |    0.9750 |        0.9265 |                 nan |                   nan |                 0.9576 |
| cluster_router_meta           |   0.9061 |    0.9697 |        0.9252 |                 nan |                   nan |                 0.9285 |
| rank_mean                     |   0.9034 |    0.9738 |        0.9245 |                 nan |                   nan |                 0.9324 |
| rank_trimmed                  |   0.9025 |    0.9738 |        0.9239 |                 nan |                   nan |                 0.9325 |
| anomaly_router_meta           |   0.9033 |    0.9698 |        0.9233 |                 nan |                   nan |                 0.9356 |
| disagreement_router_meta      |   0.9019 |    0.9701 |        0.9223 |                 nan |                   nan |                 0.9298 |
| uncertainty_route             |   0.8919 |    0.9697 |        0.9153 |                 nan |                   nan |                 0.8623 |
| anomaly_blend                 |   0.8859 |    0.9697 |        0.9111 |                 nan |                   nan |                 0.8763 |
| residual_greedy               |   0.8819 |    0.9686 |        0.9079 |                 nan |                   nan |                 0.8870 |
| cluster_route                 |   0.8811 |    0.9696 |        0.9077 |                 nan |                   nan |                 0.8342 |
| soft_blend_disagreement       |   0.8631 |    0.9748 |        0.8966 |                 nan |                   nan |                 0.9437 |
| disagreement_route            |   0.8639 |    0.9697 |        0.8956 |                 nan |                   nan |                 0.8638 |
| coxnet_stack_with_uncertainty |   0.5620 |    0.5415 |        0.5558 |                 nan |                   nan |                 0.2276 |
| BEST_BET                      | nan      |  nan      |      nan      |                 nan |                   nan |                 0.9995 |
| FINAL                         | nan      |  nan      |      nan      |                 nan |                   nan |                 0.9981 |
| HIER                          | nan      |  nan      |      nan      |                 nan |                   nan |                 0.8740 |
| LEARNED_STACK                 | nan      |  nan      |      nan      |                 nan |                   nan |                 0.6927 |
| aggressive_extreme            | nan      |  nan      |      nan      |                 nan |                   nan |                 0.9358 |
| aggressive_routed_blend       | nan      |  nan      |      nan      |                 nan |                   nan |                 0.9680 |
| aggressive_specialist_only    | nan      |  nan      |      nan      |                 nan |                   nan |                 0.8624 |
| aggressive_topk_only          | nan      |  nan      |      nan      |                 nan |                   nan |                 0.8688 |
| aggressive_xstack             | nan      |  nan      |      nan      |                 nan |                   nan |                 0.9414 |


## New fusion mechanisms


| name              |   C_hep |   C_death |   LB_combined |   event_quality_hep |   event_quality_death |   anchor_corr_combined |
|:------------------|--------:|----------:|--------------:|--------------------:|----------------------:|-----------------------:|
| dig_endpoint_pick |     nan |       nan |           nan |                 nan |                   nan |                 0.9483 |
| dig_pareto        |     nan |       nan |           nan |                 nan |                   nan |                 0.9547 |


## v4 vs v3 anchor


| name                       |   C_hep |   C_death |   LB_combined |   event_quality_hep |   event_quality_death |   anchor_corr_combined |
|:---------------------------|--------:|----------:|--------------:|--------------------:|----------------------:|-----------------------:|
| meta_avg_aware_weighted    |     nan |       nan |           nan |                 nan |                   nan |                 1.0000 |
| meta_avg_aware_weighted_v3 |     nan |       nan |           nan |                 nan |                   nan |                 0.9675 |
| meta_avg_aware_weighted_v4 |     nan |       nan |           nan |                 nan |                   nan |                 0.9569 |