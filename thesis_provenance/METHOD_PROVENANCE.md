# Noise methods: what each one came from (2026-10-08)

Written by Claude on 2026-10-08; moved into the repo 2026-10-10. Not reviewed by the author. The bucket
on each line is Claude's judgement, not a measured result.

How the buckets were decided. Every line was first read from the code (file and line given). Most
method families were then searched in the literature for earlier use on molecular data; those results,
with queries and citations, are in LITERATURE_SEARCH_2026-10-08.md next to this file. A "c" bucket
means no source was named in the code AND none was found in that search. Three of the four methods
Chapter 8 names as new were not searched by their own form; the search file says which.

Buckets: (a) faithful implementation of a published method · (b) published method plus a stated
change · (c) no published precedent found · (d) provenance unclear or rests on a non-publication
(an AI-written report, a ported code base, or a citation that does not describe the code).

Changes that apply to almost every method are listed once here, not on every line:
- G1. Predictions used for scoring come out of fold, with inner folds that keep scaffold groups
  together. Engine: `NoiseEstimator._oof_apply`, NE:141-197; `oof_predict`, noisy_labels/core.py:135-186.
- G2. Neighbour methods find neighbours in a space the search builds (a base representation, an
  augmented space, a hybrid, ECFP4 Tanimoto, or forest proximity). k is swept over 5/10/20 and the
  counting scheme is rank or 1/d². The setting is picked by predicting a held-out fifth of the
  training molecules (n07:1906-1981, n21:597-666).
- G3. Regression adaptation of a method published for classification.

Paths: NE = src/kirby/utils/noise_estimation.py · L = src/kirby/utils/lnl_methods.py ·
M = src/kirby/utils/noise_mitigation_methods.py · nXX = tests/experiments/noise/nXX_*.py ·
helpers = tests/experiments/noise/noise_helpers.py

## Detectors in noise_estimation.py

| Method | Where | Bucket | What sets it apart from the source |
|---|---|---|---|
| neighbor_residual | NE:361-422 | b (G2, G3) | Classification branch = Wilson 1972 edited nearest neighbour as a score. Regression branch, \|y − neighbour mean\|/sd(y), has no single source. No citation in code. |
| local_variance_ratio | NE:425-497 | b | \|y − neighbour mean\| / neighbour sd, with floors. Same statistic as the meteorological buddy check (Båserud et al. 2020, TITAN) and RegENN-like; not used on molecular labels before (searched 2026-10-10). No citation in code. |
| distance_weighted | NE:500-554 | b (G3) | Dudani 1976 1/d-weighted kNN used as a residual. Ignores the swept scheme. No citation in code. |
| activity_cliff_aware | NE:557-628 | c | Score = \|y − neighbour mean\| / (sd(y) + neighbour sd). Formula not found (searched 2026-10-10); the damping idea is in SLOM (Chawla & Sun 2006), on feature distances, not labels. The docstring formula (:565) differs from the code (:621). Framing follows Maggiora 2006. |
| cross_val | NE:635-725 | b (G1) | Regression: Brodley & Friedl 1999 cross-validated filtering, as a residual. Classification: the confident-learning self-confidence score without per-class thresholds. No citation in code. |
| ensemble | NE:728-777 | c | Spread across seeds + residual, summed. No source found. |
| loo | NE:780-843 | b | Breiman 2001 out-of-bag residual. Named "leave-one-out". No citation in code. |
| hybrid | NE:850-907 | c | Mean of three min-max-scaled scores. |
| gaussian_process | NE:914-1080 | b (G1) | Standard GP; the score is posterior sd + \|residual\|, which is not from a source. Docstring :922 promises a second GP for the variance; the class has none. |
| heteroscedastic_nn | NE:1083-1201 | a (+G1) | β-NLL, Seitzer et al. 2022, cited :1096, in refs.bib as Seitzer2022. Dropout and weight decay added. |
| conformal | NE:1204-1295 | b | Normalised split conformal. No finite-sample (n+1) quantile correction; calibration rows are a contiguous prefix; the score is the interval width. Cites "ACS Omega 2024" (:1213) with no author: d for the citation. |
| deep_ensemble | NE:1298-1424 | b | Lakshminarayanan 2017, cited :1309, without the adversarial term. |
| bayesian_rf | NE:1427-1518 | d | Cites "Wager-style" (:1496) and "Jackknife+" (:1434). It computes neither: tree sd + 0.5·\|OOF residual\|. |
| quantile_forest | NE:1521-1598 | b | Meinshausen 2006 QRF (not cited in code); the score is the distance outside the 10–90% range, which is new. |
| quantile_regression | NE:1601-1701 | b | Koenker & Bassett quantile loss on GBM or a network; the score is the width. No citation in code. |
| ngboost | NE:1704-1834 | a (+G1) | Duan et al. 2020, cited :1716. 50 stages instead of 500. |
| hetero_gp | NE:1838-2021 | d | "PORTED, NOT INVENTED" from qsar_qm_models (:1850). No paper named. Closest is Kersting 2007, but the noise never feeds back into the GP. |
| smooth_by_scores | NE:2084-2135 | c | Four rules that move labels toward a target by score. No source. |
| clip_to_range | NE:2138-2175 | c | Per-molecule winsorising to a QRF range. No source found. |
| weights_from_scores | NE:2178-2206 | c | 1/(1+s), exp(−s), 1 − s². Generic; no source. |
| forest proximity | NE:2234-2277 | b | Breiman 2001 proximity, in-sample, min_samples_leaf 5. No citation in code. |

## Cleanups in n07_remediation.py

| Method | Where | Bucket | What sets it apart from the source |
|---|---|---|---|
| remove | n07:957-964 | b (G1) | Brodley & Friedl 1999 filtering. Flag share swept at 5/10/20%. Tested in QSAR before by Zhao, Wang, Sedykh & Zhu 2017 (ACS Omega). No citation in code. |
| downweight | n07:966-971 | c | 1/(1+score), one pass. No source. |
| huber_weight | n07:973-998 | b, d | Huber 1964 weights from a random-forest residual, one pass, no iteration. Comment :974 calls Huber "redescending"; it is not. Added from the "compass review" (:187-189), an AI report. |
| invvar_weight | n07:1000-1047 | a | Hedges & Olkin 1985 weight ∝ count. Not in refs.bib. A molecule measured k times counts k² times (:755-760). |
| js_shrink | n07:1049-1101 | d | Cites James & Stein 1961 and Efron & Morris 1975. tau² cancels, so shrink = 0.5s/(1+0.5s), a fixed function of the detector score. Not an empirical-Bayes estimate. From the compass report. |
| label_smooth | n07:1103-1137 | b (G2) | kNN / Nadaraya–Watson smoothing toward neighbours. Classification = kNN relabelling. No citation in code. |
| knn_smooth | n07:1139-1168 | b (G2, G3) | Wilson-style editing, but the flagged 5% are replaced by the neighbour mean. No citation in code. |
| prox_smooth | n07:1170-1237 | c | Forest proximity used as the neighbour kernel for smoothing. No source found. Compass report suggested trying proximity. |
| oof_smooth | n07:1247-1264 | b, d | Closest is Reed et al. 2014 target blending. Step scaled by score. Ported from noisy_housing. |
| gp_smooth | n07:1266-1343 | b | GP denoising toward the OOF posterior mean. No citation in code. |
| constant_lift | n07:1345-1381 | b | Tobit imputation, but uses limit + sd·mills, which is larger than the truncated-normal mean by (limit − p). The code says so itself (n30:91-97). One constant for all censored rows. |
| quantile_clip | n07:1383-1400 | c | QRF range + clipping. Same as clip_to_range. |
| screen_settings | n07:1906-1981 | c | Picks the neighbour space by inner hold-out. This is G2 itself. |

## Learning-with-noisy-labels survey (lnl_methods.py, run in n09)

The header says these were "adapted from the confirmation report LNL survey" (L:3). No hyperparameter is swept except for distance_selection (N09:1112).

| Method | Where | Bucket | What sets it apart from the source |
|---|---|---|---|
| small_loss | L:568-656 | b (G3) | Cites Han 2018 co-teaching. Uses one network and one selection, so it is not co-teaching. Closer to plain small-loss filtering. |
| confident_learning | L:659-745 | b (G3) | Cites Northcutt 2021. In-sample probabilities, no confident joint, fixed percentile cuts. The source would not call this confident learning. |
| distance_selection | L:748-830, n09:253-305 | b (G2, G3) | Wilson ENN-like, with a fixed 15% removal. No citation in code. |
| multistage_cleaning | L:837-922 | d | Cites "Zhao et al. (2025)", which is refs.bib zhao2025 (MANTRA). MANTRA fits a mixture to loss trajectories and removes once; it does not do three rounds of top-10% removal. |
| early_learning | L:925-1071 | d | Cites "Zhang et al. (2025)", which is refs.bib zhang2025. That paper uses an additive, similarity-driven update with a clean subset. The code is ×0.95 on above-median loss. Not Liu 2020 ELR either. |
| uncertainty_curriculum | L:1074-1231 | b | Cites Bengio 2009, but ranks by own loss, which is Kumar 2010 self-paced learning. Not ranked by uncertainty despite the name. |
| dividemix | L:1238-1433 | b (G3) | Li 2020 without the semi-supervised half. The source would recognise only the loss-mixture split. |
| meta_weight_net | L:1436-1624 | b | Shu 2019, with the meta set carved from noisy labels (the source needs clean ones). The docstring says first-order; the code uses create_graph=True. |
| mentornet | L:1627-1832 | b | Jiang 2018 predefined-curriculum variant. No LSTM; adds a neighbour-distance input. |
| contrast_to_divide (renamed josrc_weighting 2026-10-30) | L:1835-2005 | d | Cites Zheltonozhskii 2021, but the code does perturbation consistency, which is not in that paper. Renamed by the author 2026-10-30: the rule is Jo-SRC's (Yao 2021) in-distribution-noisy test used as a weight; code unchanged. |
| heteroscedastic_gp | L:2012-2145 | d | Cites Liu, Ong, Cai 2020. The code is a two-step residual fit plus a weighted network, which is not that paper. Closest is Kersting 2007. |
| evidential_kernel | L:2148-2302 | a | Amini 2020 deep evidential regression. "Kernel" in the name is a misnomer. |

## Removal filters (noise_mitigation_methods.py). None cites a paper with author and year.

| Method | Where | Bucket | Note |
|---|---|---|---|
| knn_consistency | M:441-527 | b / c | Two-class = Wilson ENN. The regression statistic ignores the molecule's own label, which no source does. |
| lof | M:530-590 | b | Breunig 2000 LOF on features, used to remove labels. |
| distance_weighted filter | M:593-654 | b | RegENN-like (Kordos & Blachnik 2012). |
| neighborhood_consensus | M:657-739 | c | Keeps activity cliffs. No source found. |
| activity_cliffs (removal) | M:742-829 | c | Removes cliff molecules as noise. The cliff definition follows Maggiora 2006 and Stumpfe & Bajorath. |
| mahalanobis | M:832-879 | a | Classical. |
| cv_disagreement | M:886-944 | b | Brodley & Friedl. |
| bootstrap_ensemble | M:947-1023 | b | Bagging filter. |
| co_teaching (forests) | M:1030-1085 | b | Han 2018 on forests. The counter threshold is half of what the comment says. |
| dividemix filter | M:1088-1126 | b | Li 2020 mixture split only. |
| zscore, prediction_error | M:1133-1188 | a | Generic. |

## n02, n11–n27

| Method | Where | Bucket | Note |
|---|---|---|---|
| reliability weight | n02:255-362 | c (G2) | Neighbour deviation → weight; features scaled by √importance. No source. |
| cliff-aware weight | n02:222-252 | c | Weighted form of neighborhood_consensus. |
| estimator × repair grid | n11:508-769 | d / c | "Full grid from the confirmation report" (:11). The combinations themselves have no source. |
| importance-weighted distance | n12:291-352, n21:348-400 | b | Wettschereck, Aha & Mohri 1997 feature-weighted kNN. Importance computed on the damaged labels. |
| cleanlab_suspicion | helpers:762-779 | a | Zhou, Mueller et al. 2024, called through the cleanlab package. Internal folds are random, not scaffold-grouped. Not in refs.bib. |
| gamma_test | helpers:782-815 | a | Stefánsson, Končar & Jones 1997. Not in refs.bib. |
| count / spread / precision weights | n19:496-597 | a | Inverse-variance weighting (Hedges & Olkin). |
| shrink_toward | n19:600-621 | b | James–Stein form with an ML prediction as the prior. Its tau² includes the noise. |
| fit_noise_variance (1/k regression) | n19:624-661, n26:264-321 | d → c | Regresses squared OOF error on 1/(number of measurements). From an AI report, which itself says it "found no paper" (n26:81). No precedent found in any field. |
| pooled_spread_variance | n19:664-689 | b, d | Smyth 2004 limma-style moderation with a fixed prior of 3. Not cited; from an AI report. |
| gp_per_molecule_noise | n19:1449-1462 | a | GP with known per-molecule noise (Rasmussen & Williams). |
| xgboost_interval_aft | n19:451-493 | b, d | Barnwal 2022 AFT on replicate min–max intervals. From a "second research report" (:136). |
| clean-before-search | n20 | c | Cleaning before the representation search. Searched 2026-10-10: nearest is Shanab et al. 2012 (same split of ranking data vs fitting data, but for class-imbalance sampling, classification). |
| cliff exemption | n25:868-944 | c | Puts cliff molecules back after any cleanup. None found after two searches (2026-10-08, 2026-10-10); nearest exempt a whole class (SPIDER, one-sided selection). |
| replicate noise diagnostics | n26:220-401 | a | Within mean square, split-half with Spearman–Brown, ICC(1). From an AI report (:11); the methods themselves are textbook. |
| propagation ratio | n27:207-210 | c | Change in held-out MSE ÷ squared label change. |

## n29–n41

| Method | Where | Bucket | Note |
|---|---|---|---|
| per-document offsets | n29:340-490 | b, d | Additive annotator bias (Raykar 2010) fitted by ridge on OOF residuals, documents as annotators. From an AI report (:13). |
| per_record_random_effect | n29:414-490 | b | BLUP-style shrinkage; moments, not REML. MERF / GPBoost named. |
| within_document_rank | n29:554-583 | b, d | Pairwise learning to rank per document. Done before in ChEMBL by Matsumoto 2021 and ActFound 2024. No citation in code. |
| document_quantile_map | n29:530-551 | b | Bolstad 2003 quantile normalisation. No citation in code. |
| censored_em | n30:371-407 | b | Buckley–James / EM-Tobit with a normal error, any ML model. |
| winsor_c | n30:410-425 | b, d | Huberised residuals around an OOF prediction. From AI report item B5. |
| xgboost_aft | n30:428-456 | b | Barnwal 2022 with a fixed scale. |
| bayes_shrink / break-even | n31:392-454 | d | The weight is "the report's H4". No paper. |
| knn_shapley | n32:995-1075 | b (G2) | Jia et al. 2019. The utility is the mean of per-neighbour squared gaps. Jia et al. already give an exact regression recursion (Theorem 6), so the docstring's reason (:1012-1015) is wrong. |
| itlm | n32:1214-1241 | b | Shen & Sanghavi 2019. The trimming learner is LightGBM, not the downstream model. |
| two_learner_exchange | n32:1244-1270 | b, d | Co-teaching on whole-set refits, intersection of the kept sets. From AI report item H6. |
| shrink_flagged | n32:1273-1302 | d | AI report item H4. |
| robust losses (Huber, fair, L1, Tukey) | n33:328-425 | a / b | Textbook losses with the threshold swept in MAD units. Tukey cites Beaton & Tukey 1974. |
| expectile, ngboost_t, Yeo–Johnson, target-noise bagging, Huber teacher | n33:352-529 | b, d | Each a textbook method. Chosen from AI report items A4, D, H and G. |
| c_mixup | n33:464-495 | b | Yao 2022 with a fixed bandwidth and one augmentation pass. |
| random_intercept | n34:245-300 | b | MERF (Hajjem 2014) with OOF LightGBM residuals and τ² by moments. |
| close-pair noise estimate | n35:262-332 | b, d | Delta test (Pi & Peterson 1994) under swept distances. From AI report item D5. |
| downweight_pairs | n35:368-383 | d | AI report item A7; weight 0.25 from a findings file. |
| remeasure priority | n36:327-358 | b, d | Sheng 2008 adapted to regression. The "leverage" score is invented in code (:53-58). |
| gamma test per setting | n37:297-344 | a (G2) | Stefánsson 1997. |
| SIMEX | n37:241-294 | b | Cook & Stefanski 1994 on the response, read at −(Gamma estimate). The source would not recognise this use. |
| tail diagnostics | n37:183-238 | a | Hill 1975, Breusch–Pagan regression. |
| detection theory | n38:280-339 | d | Formulas from AI report item B2. |
| source offsets + ρ-aware weights | n39:315-416 | b, d | Mixed model (REML) and the Kish design effect. No paper named; from AI report items A8 and D4. |
| model-side arms (retune, seeds, stack, GP nugget from Gamma, regression kriging) | n40 | b, d | Each textbook (Bergstra 2012, Breiman 1996, Mallows 1973, Hengl 2007). Chosen from an AI report. Setting the GP nugget from a Gamma-test estimate has no source found. |
| conformal p-values + BH | n41:248-360 | b | Bates 2023 + Benjamini–Hochberg. Calibration and test molecules are scored by models that saw each other, so the exchangeability assumption is broken. |
| trusted calibration | n41:322-348 | c | Calibrate on molecules with 3+ consistent replicates. No source found. |
| isolation_forest, lof on [features, residual] | n41:388-544 | b | Liu 2008, Breunig 2000, with an appended scaled residual. |
