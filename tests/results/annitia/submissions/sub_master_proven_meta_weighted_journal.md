# ANNITIA MASLD survival — `proven_meta_weighted` submission

**Author:** Adelaide Punt (St Catherine's College, Oxford)
**Submission CSV:** `sub_master_proven_meta_weighted.csv`
**Pipeline:** KIRBy `tests/experiments/annitia/` (multi-layered ensemble)

## Honest fusion picks

| Endpoint | Method | OOF-C |
|----------|--------|-------|
| Hep      | `proven_meta_weighted` | **0.9078** |
| Death    | `proven_meta_weighted` | **0.9730** |
| **LB-weighted (0.7·hep + 0.3·death)** |  | **0.9273** |

## Methodology

Five processing layers feed a single fusion stage:

1. **BE**  algorithm (Cox / Coxnet / RSF / GBSA / LGBM-AFT / NGBoost
   / CatBoost-Cox / CatBoost-AFT) × view-set (`snapshot`,
   `+clinical_pro`, `+fpca`, `+gp_smooth`, `+visit_density+clinical_slope`,
   `+meld_na+visit_density`).
2. **CD** Phase-C target framings (pseudo-observation, landmark binary,
   pchazard, logistic-hazard) and Phase-D training tricks
   (IPCW-weighted, bagged, robust-trimmed, snapshot, self-trained,
   XGB-pairwise, Fine-Gray, D.9 cross-pollination).
3. **HB** hep-only boost models.
4. **BRANCHES** DSM, TSFM-FFT, TabPFN landmark, TabPFN-2.5 disc, TabSurv,
   synthetic augmentation, NHANES-pretrained MLP, cross-endpoint
   pretrain/fine-tune.
5. **EXTERNAL** NHANES T2.x recipes (deferred where applicable).

Fusion strategy: **proven_meta_weighted** — see ensemble members below.

All model fits use 5-fold StratifiedKFold (random_state=42) on the
per-endpoint event flag. Hep and death use independent fold partitions
because their valid masks differ (hep n=1106, death n=966). Each
member's OOF prediction is from a model that did not see that row
during training. The fusion's OOF C-index is therefore honest
out-of-sample.


## Ensemble members

### Hep
- `hparam__hparam-rsf__snapshot+clinical_pro+fpca+gp_smooth`
- `hparam__hparam-catboost_aft__snapshot+clinical_pro+fpca+gp_smooth`
- `be__snapshot+clinical_pro__catboost_cox`
- `be__snapshot+clinical_pro+fpca__rsf`
- `be__snapshot+clinical_pro+summary+nit__cox`
- `branches__tabsurv-K10__snapshot+clinical_pro+fpca+gp_smooth`
- `branches__pretrained-mlp__nhanes_proj`

### Death
- `be__snapshot+clinical_pro+meld_na+visit_density__lgbm_aft`
- `hparam__hparam-rsf__snapshot+clinical_pro+meld_na+visit_density`
- `be__snapshot+clinical_pro+fpca+visit_density+clinical_slope__cox`
- `be__snapshot+clinical_pro+fpca+visit_density+clinical_slope__lgbm_aft`
- `be__snapshot+clinical_pro+meld_na+visit_density__coxnet`
