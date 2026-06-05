# Local Validation Audit (Bootstrap OOF + Holdout 10%)

- Total variants: **85**
- Bootstrap-scored: **68**, holdout-scored: **73**
- Unique bootstrap mean LB values (4 dp): **45** / 68
- Unique holdout LB values       (4 dp): **3** / 73

**Note:** 17 variants lack bootstrap scores (bootstrap CSV was generated incrementally; resume the script `tests/experiments/annitia/bootstrap_candidate_ci.py` to fill in remaining rows).

## Spearman correlations

- ρ(bootstrap_mean_LB, holdout_LB) = **-0.303** (p=0.0231, n=56)
- ρ(bootstrap_mean_LB, known_LB)   = **0.100** (n=9)
- ρ(holdout_LB,        known_LB)   = **-0.548** (n=9)

## Differentiation diagnosis

**WARNING:** at least one method assigns <=5 unique values across all 70 variants. The variant_proxy_oof fallback collapses non-deterministic fusion variants (NNLS, isotonic, anomaly routing, cluster routing) onto a single mean-of-base-OOFs prediction. Local validation is therefore **fundamentally unable** to differentiate those candidates — the distinguishing fusion logic is invisible at the proxy level.

## Top-10 by bootstrap mean LB

| Variant | mean LB | std | 95% CI | hold LB | known LB |
|---|---|---|---|---|---|
| `failure_robust` | 0.8680 | 0.0401 | [0.7854, 0.9397] | 0.9426 | - |
| `failure_fp_suppress` | 0.8680 | 0.0397 | [0.7829, 0.9377] | 0.9426 | - |
| `method_pseudolabel` | 0.8673 | 0.0397 | [0.7832, 0.9385] | - | - |
| `new_binwise_corrected` | 0.8668 | 0.0397 | [0.7851, 0.9376] | 0.9426 | - |
| `deepfn_targeted` | 0.8667 | 0.0383 | [0.7935, 0.9362] | - | - |
| `blend_top3` | 0.8664 | 0.0391 | [0.7894, 0.9356] | - | - |
| `deepfn_isolation` | 0.8663 | 0.0385 | [0.7824, 0.9374] | - | - |
| `blend_top5` | 0.8663 | 0.0405 | [0.7818, 0.9379] | - | - |
| `method_conformal` | 0.8662 | 0.0407 | [0.7809, 0.9409] | - | - |
| `deepfn_specialist` | 0.8661 | 0.0390 | [0.7815, 0.9341] | - | - |

## Top-10 by holdout LB

| Variant | holdout LB | hep | death | bootstrap mean | known LB |
|---|---|---|---|---|---|
| `rank_median` | 0.9554 | 0.9593 | 0.9465 | 0.8563 | 0.9112 |
| `rank_trimmed` | 0.9489 | 0.9457 | 0.9565 | 0.8550 | - |
| `BEST_BET` | 0.9426 | 0.9367 | 0.9565 | 0.8564 | - |
| `new_bma` | 0.9426 | 0.9367 | 0.9565 | - | - |
| `repdisc_aware` | 0.9426 | 0.9367 | 0.9565 | - | - |
| `rank_mean` | 0.9426 | 0.9367 | 0.9565 | 0.8577 | - |
| `proven_meta_weighted` | 0.9426 | 0.9367 | 0.9565 | 0.8581 | - |
| `proven_meta` | 0.9426 | 0.9367 | 0.9565 | 0.8587 | - |
| `proven_median` | 0.9426 | 0.9367 | 0.9565 | 0.8585 | - |
| `noise_multiseed` | 0.9426 | 0.9367 | 0.9565 | - | - |

## Side-by-side (all variants, sorted by bootstrap mean LB)

| Variant | boot LB ± std | 95% CI | hold LB | known LB |
|---|---|---|---|---|
| `failure_robust` | 0.8680 ± 0.0401 | [0.7854, 0.9397] | 0.9426 | - |
| `failure_fp_suppress` | 0.8680 ± 0.0397 | [0.7829, 0.9377] | 0.9426 | - |
| `method_pseudolabel` | 0.8673 ± 0.0397 | [0.7832, 0.9385] | - | - |
| `new_binwise_corrected` | 0.8668 ± 0.0397 | [0.7851, 0.9376] | 0.9426 | - |
| `deepfn_targeted` | 0.8667 ± 0.0383 | [0.7935, 0.9362] | - | - |
| `blend_top3` | 0.8664 ± 0.0391 | [0.7894, 0.9356] | - | - |
| `deepfn_isolation` | 0.8663 ± 0.0385 | [0.7824, 0.9374] | - | - |
| `blend_top5` | 0.8663 ± 0.0405 | [0.7818, 0.9379] | - | - |
| `method_conformal` | 0.8662 ± 0.0407 | [0.7809, 0.9409] | - | - |
| `deepfn_specialist` | 0.8661 ± 0.0390 | [0.7815, 0.9341] | - | - |
| `failure_xref_aware` | 0.8659 ± 0.0395 | [0.7824, 0.9364] | 0.9426 | - |
| `deepfn_targeted_death` | 0.8659 ± 0.0396 | [0.7887, 0.9357] | - | - |
| `failure_hard_specialist` | 0.8652 ± 0.0400 | [0.7808, 0.9350] | 0.9426 | - |
| `blend_top1` | 0.8651 ± 0.0405 | [0.7831, 0.9382] | - | - |
| `blend_top4` | 0.8650 ± 0.0413 | [0.7747, 0.9375] | - | - |
| `failure_fn_recovery` | 0.8649 ± 0.0394 | [0.7845, 0.9365] | 0.9426 | - |
| `blend_top2` | 0.8644 ± 0.0399 | [0.7808, 0.9376] | - | - |
| `method_mixup` | 0.8643 ± 0.0386 | [0.7834, 0.9354] | - | - |
| `meta_avg_weighted` | 0.8615 ± 0.0414 | [0.7784, 0.9385] | 0.9426 | 0.9284 |
| `residual_greedy` | 0.8611 ± 0.0421 | [0.7717, 0.9370] | 0.9426 | - |
| `specialist_override` | 0.8605 ± 0.0421 | [0.7701, 0.9324] | 0.9426 | - |
| `meta_avg_all` | 0.8604 ± 0.0426 | [0.7706, 0.9354] | 0.9426 | 0.9294 |
| `bootstrap_dropout` | 0.8599 ± 0.0423 | [0.7683, 0.9352] | 0.9426 | - |
| `cluster_router_meta` | 0.8598 ± 0.0427 | [0.7717, 0.9342] | 0.9426 | - |
| `uncertainty_route` | 0.8597 ± 0.0405 | [0.7797, 0.9350] | 0.9426 | - |
| `weighted_anomaly_per_patient` | 0.8597 ± 0.0408 | [0.7734, 0.9326] | 0.9426 | - |
| `subgroup_blend` | 0.8593 ± 0.0414 | [0.7702, 0.9343] | 0.9426 | - |
| `subgroup_routed` | 0.8593 ± 0.0427 | [0.7706, 0.9386] | 0.9426 | - |
| `diverse` | 0.8588 ± 0.0423 | [0.7710, 0.9354] | 0.9426 | - |
| `ultra_meta_weighted` | 0.8587 ± 0.0405 | [0.7720, 0.9300] | 0.9426 | - |
| `proven_meta` | 0.8587 ± 0.0412 | [0.7790, 0.9362] | 0.9426 | - |
| `meta_avg_top5` | 0.8586 ± 0.0405 | [0.7781, 0.9330] | 0.9426 | 0.9278 |
| `meta_avg_aware_weighted_no_nnls` | 0.8585 ± 0.0410 | [0.7709, 0.9323] | 0.9426 | - |
| `proven_median` | 0.8585 ± 0.0423 | [0.7709, 0.9338] | 0.9426 | - |
| `round3` | 0.8585 ± 0.0426 | [0.7671, 0.9352] | 0.9426 | - |
| `nnls_stack` | 0.8584 ± 0.0423 | [0.7721, 0.9319] | 0.9426 | - |
| `isotonic_avg` | 0.8584 ± 0.0432 | [0.7679, 0.9352] | 0.9426 | - |
| `best_combined` | 0.8583 ± 0.0409 | [0.7777, 0.9310] | 0.9426 | 0.9123 |
| `holdout_pick` | 0.8583 ± 0.0422 | [0.7723, 0.9355] | 0.9426 | - |
| `meta_avg_aware_weighted_v2` | 0.8581 ± 0.0420 | [0.7696, 0.9334] | 0.9426 | - |
| `meta_avg_grand_all_v2` | 0.8581 ± 0.0412 | [0.7731, 0.9322] | 0.9426 | - |
| `HIER` | 0.8581 ± 0.0417 | [0.7743, 0.9337] | 0.9426 | - |
| `proven_meta_weighted` | 0.8581 ± 0.0413 | [0.7734, 0.9369] | 0.9426 | - |
| `anomaly_blend_meta` | 0.8580 ± 0.0410 | [0.7714, 0.9342] | 0.9426 | - |
| `subgroup_hep_only` | 0.8578 ± 0.0411 | [0.7713, 0.9347] | 0.9426 | - |
| `disagreement_route` | 0.8578 ± 0.0435 | [0.7665, 0.9393] | 0.9426 | - |
| `anomaly_blend` | 0.8578 ± 0.0431 | [0.7697, 0.9349] | 0.9426 | - |
| `rank_mean` | 0.8577 ± 0.0415 | [0.7727, 0.9353] | 0.9426 | - |
| `ultra_meta` | 0.8577 ± 0.0410 | [0.7718, 0.9285] | 0.9426 | - |
| `anomaly_router_meta` | 0.8576 ± 0.0418 | [0.7678, 0.9304] | 0.9426 | - |
| `meta_avg_new_variants_only` | 0.8575 ± 0.0414 | [0.7739, 0.9325] | 0.9426 | - |
| `coxnet_stack_with_uncertainty` | 0.8575 ± 0.0422 | [0.7674, 0.9322] | 0.9426 | - |
| `best_combined_v2` | 0.8573 ± 0.0423 | [0.7719, 0.9327] | 0.9426 | - |
| `honest` | 0.8573 ± 0.0405 | [0.7788, 0.9291] | 0.9426 | - |
| `round1b` | 0.8571 ± 0.0419 | [0.7746, 0.9340] | 0.9426 | - |
| `soft_blend_disagreement` | 0.8567 ± 0.0418 | [0.7702, 0.9323] | 0.9426 | - |
| `LEARNED_STACK` | 0.8567 ± 0.0441 | [0.7629, 0.9333] | 0.9426 | - |
| `cluster_route` | 0.8566 ± 0.0427 | [0.7665, 0.9312] | 0.9426 | - |
| `BEST_BET` | 0.8564 ± 0.0434 | [0.7652, 0.9363] | 0.9426 | - |
| `meta_avg` | 0.8564 ± 0.0411 | [0.7725, 0.9301] | 0.9426 | 0.9269 |
| `meta_avg_grand_all` | 0.8564 ± 0.0406 | [0.7727, 0.9276] | 0.9426 | 0.9294 |
| `meta_avg_aware_top5` | 0.8563 ± 0.0416 | [0.7748, 0.9314] | 0.9426 | - |
| `best_per_endpoint` | 0.8563 ± 0.0423 | [0.7687, 0.9326] | 0.9426 | - |
| `rank_median` | 0.8563 ± 0.0409 | [0.7687, 0.9295] | 0.9554 | 0.9112 |
| `FINAL` | 0.8558 ± 0.0425 | [0.7668, 0.9333] | 0.9426 | 0.9268 |
| `disagreement_router_meta` | 0.8557 ± 0.0412 | [0.7692, 0.9316] | 0.9426 | - |
| `meta_avg_aware_weighted` | 0.8551 ± 0.0415 | [0.7720, 0.9314] | 0.9426 | 0.9302 |
| `rank_trimmed` | 0.8550 ± 0.0425 | [0.7675, 0.9310] | 0.9489 | - |
| `new_bma` | - | - | 0.9426 | - |
| `new_cohort_split` | - | - | 0.9426 | - |
| `new_inference_dropout` | - | - | 0.9426 | - |
| `new_per_endpoint` | - | - | 0.9426 | - |
| `noise_censoraware` | - | - | 0.9426 | - |
| `noise_classbalance` | - | - | 0.9426 | - |
| `noise_huber` | - | - | 0.9426 | - |
| `noise_ipcw` | - | - | 0.9426 | - |
| `noise_multiseed` | - | - | 0.9426 | - |
| `repdisc_aware` | - | - | 0.9426 | - |
| `repdisc_blend` | - | - | 0.9426 | - |
| `repdisc_specialized` | - | - | 0.9426 | - |
| `unc_bootstrap_var` | - | - | 0.9426 | - |
| `unc_conformal` | - | - | 0.9426 | - |
| `unc_disagree_soft` | - | - | 0.9426 | - |
| `unc_mahal_blend` | - | - | 0.9426 | - |
| `unc_specialist` | - | - | 0.9426 | - |

## Recommendation

Bootstrap and holdout disagree (ρ<0.3). Treat both as weak local signal and lean on the LB-constraint reverse-engineering instead.
