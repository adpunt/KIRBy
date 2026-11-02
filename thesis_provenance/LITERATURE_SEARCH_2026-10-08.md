# Literature search behind the noise-method provenance (2026-10-08)

Written by Claude on 2026-10-10, from the search reports of the 2026-10-08 session plus four direct searches run 2026-10-10. Not reviewed
by the author. "None found" means none after the queries listed. It does not mean none exist,
because these were web, Crossref, Europe PMC and arXiv searches, not a systematic review. Each
citation below was checked against Crossref, arXiv or the publisher's page unless marked otherwise.

The question for each method was: has it, or its source method, been used on molecular property
or bioactivity data before?

## The four methods Chapter 8 names as new (searched directly 2026-10-10)

| Method (code) | Verdict | Closest published work |
|---|---|---|
| Putting cliff molecules back after cleaning (n25:868-944) | None found, after two passes. | Chemistry either removes cliffs or flags them as outliers (Cruz-Monteagudo et al. 2016, Curr. Pharm. Des., doi 10.2174/1381612822666160509124337; Grigoreva et al. 2019, doi 10.3103/S0027131419010036, abstract only), or leaves each cliff to a person (Fourches, Muratov & Tropsha 2016, JCIM, doi 10.1021/acs.jcim.6b00129, full text: "the modeler should decide whether to keep or discard any activity cliff-forming compounds"). Dablander et al. 2023 (doi 10.1186/s13321-023-00708-w) warns that removing cliffs loses information, without a method. Outside chemistry, some cleaning methods exempt a whole class decided in advance (one-sided selection, Kubat & Matwin 1997; neighbourhood cleaning rule, Laurikkala 2001, doi 10.1007/3-540-48229-6_9; SPIDER, Stefanowski & Wilk 2008, doi 10.1007/978-3-540-85836-2_27). Brodley & Friedl 1999 (doi 10.1613/jair.606, sec. 3.4) name the problem of discarding correct exceptions and leave it open. None puts examples back after cleaning, and none uses a random put-back control. |
| Cliff-aware neighbour gap (noise_estimation.py:557-628; regression score simplifies to \|y - neighbour mean\| / (sd(y) + sd of the neighbours' labels)) | Related but different. No paper with this formula, and none using it to separate cliffs from wrong labels. | The damping idea, an outlier score lowered where the neighbourhood is unstable, is in SLOM: Chawla & Sun 2006, Knowl. Inf. Syst. 9:412, doi 10.1007/s10115-005-0200-2 (formula read from the ELKI implementation, not the paper). SLOM scores feature-space distances, not label gaps, and is not a label-noise filter. RegENN (Kordos & Blachnik 2012, doi 10.1007/978-3-642-33266-1_33; threshold from secondary sources) scales a residual by local label spread, with no global term and no cliff argument. The classification form's disagreement fraction is kDN (Smith, Martinez & Giraud-Carrier 2014, doi 10.1007/s10994-013-5422-z); weighting it by neighbour purity: not found. |
| Neighbour gap divided by the neighbours' own spread (noise_estimation.py:425-497) | Same statistic published, outside machine learning and chemistry. | The meteorological "buddy check": \|x - mean of neighbours\| / max(neighbours' sd, floor), flag above a threshold. TITAN, Båserud et al. 2020, Adv. Sci. Res. 17:153, doi 10.5194/asr-17-153-2020 (code read: sd inflated by sqrt(1+1/n), fixed floor). Robust versions: SpotSweeper (Totty, Hicks & Guo 2025, Nat. Methods, doi 10.1038/s41592-025-02713-3; median and MAD) and the Hampel filter (Pearson 2002, doi 10.1109/87.974338). As a regression label-noise filter, RegENN thresholds a model residual at alpha x local sd (unverified against the primary paper). Spatial-outlier z-tests (Shekhar, Lu & Zhang 2003, doi 10.1023/A:1023455925009; Lu, Chen & Kou 2003) divide by one dataset-wide sd, not the local one. Martin, Saez & Corchado 2021's kNN filters (correct DOI 10.1109/ACCESS.2021.3123151) use a fixed threshold with no local spread. In cheminformatics: none found. Classification form (disagreement / max(1 - purity, 1/k)): none found, searched lightly. |
| Cleaning before the feature search (n20) | Related but different. No study crosses features chosen on cleaned vs as-recorded labels with a model fitted on cleaned vs as-recorded labels. | Same split, different second step: Shanab, Khoshgoftaar, Wald & Van Hulse 2012, Int. J. Bus. Intell. Data Min. 7:116, doi 10.1504/ijbidm.2012.048730 (ranks features on sampled data and trains on unsampled data; the step is class-imbalance sampling, not noise cleaning; classification). Label noise destabilises feature rankings: Altidor, Khoshgoftaar & Napolitano 2011, doi 10.1109/iri.2011.6009553; Pau et al. 2023, Information 14:438, doi 10.3390/info14080438 (classification, no cleaning). Selection and noise handling in one model: Frénay, Doquire & Verleysen 2014, doi 10.1016/j.csda.2013.05.001; Cappozzo et al. 2021, Anal. Chim. Acta, doi 10.1016/j.aca.2021.338245 (spectroscopy). Regression, QSAR, or several representations: none found. |

## By family

### Robust statistics
- Huber weights / Huber loss on bioactivity labels: none found (3 searches). Nearest: robust
  multiple linear regression for QSAR, Konovalov, Llewellyn, Vander Heyden & Coomans 2008, JCIM
  48:2081, doi 10.1021/ci800209k (abstract; estimator not named).
- Tukey biweight, Student-t likelihood on bioactivity labels: none found (5 searches, Europe PMC).
- Trimmed loss (ITLM, least trimmed squares) in QSAR: none found (5 searches).
- Removing training molecules with large out-of-fold residuals: DONE BEFORE. Zhao, Wang, Sedykh &
  Zhu 2017, ACS Omega 2:2805, doi 10.1021/acsomega.7b00274: corrupted labels in 8 datasets; molecules
  with large cross-validation error were enriched for the corrupted ones; removing them did not
  improve prediction of external molecules. Also Cao et al. 2010, J. Comput. Chem. 31:592,
  doi 10.1002/jcc.21351 (outliers from Monte Carlo cross-validation error).

### Empirical Bayes
- James–Stein / empirical-Bayes shrinkage of activity labels: none found (2 searches).
- limma-style moderation of replicate variances (Smyth 2004) on bioactivity: none found (2 searches;
  genomics only).

### Meta-analysis
- Weighting molecules by measurement count or inverse variance in QSAR: none found (3+3 searches).
  Related, not the same: Landrum & Riniker 2024, JCIM 64:1560, doi 10.1021/acs.jcim.4c00049
  (measures disagreement between sources).

### Econometrics and survival analysis
- Tobit / censored likelihood on censored activity labels: DONE BEFORE. Svensson, Friesacher,
  Winiwarter, Mervin, Arany & Engkvist 2025, Artif. Intell. Life Sci. 7:100128,
  doi 10.1016/j.ailsci.2025.100128. SparseChem (Arany et al. 2022, arXiv 2203.04676) uses a one-sided
  squared loss, not a Tobit likelihood.
- Interval-censored AFT (xgboost survival:aft) on activity data: none found (2 searches).
- SIMEX in QSAR: none found (4 searches; epidemiology only).

### Multiple testing and conformal prediction
- Conformal QSAR exists (Xu, Liaw, Sheridan & Svetnik 2024, ACS Omega 9:29478,
  doi 10.1021/acsomega.4c02017; this is the "ACS Omega 2024" named in noise_estimation.py:1213).
- Conformal outlier p-values with Benjamini–Hochberg to flag wrong labels in molecular data: none
  found (6 searches). Nearest: Zhan et al. 2025, PLoS Comput. Biol. 21:e1012803 (conformal label
  correction on text, radiomics and RNA-seq, no FDR control).

### Cooperative game theory (data valuation)
- KNN-Shapley / data valuation on molecules: DONE BEFORE for screening classification. Hesse, Boldini
  & Sieber 2024, JCIM 64:8142, doi 10.1021/acs.jcim.4c01547. Regression or QSAR use: none found.
- Jia et al. 2019 already give an exact KNN-Shapley recursion for regression (Theorem 6, appendix
  E.1, arXiv 1908.08619v4, read 2026-10-08). The comment at n32:1012-1015 saying it needs a sum is
  wrong.

### Learning with noisy labels (computer vision)
- Co-teaching, DivideMix, Meta-Weight-Net, MentorNet on molecular property data: none found
  (4 searches). Nearest: self-paced learning in a QSAR classifier, Xia & Wang 2018 (arXiv 1804.08615).
- Confident learning / cleanlab on molecules: only a preprint on physical properties, Marimuthu &
  McGuire 2025 (arXiv 2505.08688). None on bioactivity.
- Bootstrapping / soft target blending (Reed et al.) on molecules: none found (2 searches).
- C-Mixup / mixup for molecular regression: DONE BEFORE. Yao et al. 2022 includes a drug-target
  dataset; MixingDTA (Kim et al. 2025, Bioinformatics 41:i105, doi 10.1093/bioinformatics/btaf238).

### Uncertainty-based detection
- Deep evidential regression on molecules: Soleimany et al. 2021, ACS Cent. Sci. 7:1356,
  doi 10.1021/acscentsci.1c00546.
- Predicted aleatoric uncertainty used to find bad labels: detection only, Ryu, Kwon & Kim 2019,
  Chem. Sci. 10:8438, doi 10.1039/c9sc01992h. Filtering or relabelling by it and retraining: none
  found.
- Gaussian-process denoising of activity labels: none found (2 searches).
- Heteroscedastic GP in QSAR: none found.

### Neighbour methods
- kNN label smoothing / correction / edited nearest neighbour on bioactivity labels: none found
  (3 searches). Nearest: MODI (Golbraikh et al., JCIM, doi 10.1021/ci400572x; snippet only).
- Random-forest proximity as a molecular similarity: used for clustering, Li, Fedorowicz, Singh &
  Soderholm 2005, JCIM 45:952, doi 10.1021/ci050049u. Not for neighbour smoothing.
- Quantile regression forests in QSAR: for prediction intervals, Sheridan, Liaw & Tudor 2021
  (arXiv 2105.08626). For clipping labels: none found.
- Isolation forest on QSAR data: Chen et al. 2016, Anal. Methods 8:7225, doi 10.1039/c6ay01574c.

### Crowdsourcing and multi-source data
- Per-document or per-assay offsets / random effects on ChEMBL-type data: none found (6 searches).
  Related: Landrum & Riniker 2024 (measure, do not model); Schoenmaker et al. 2025, JCIM,
  doi 10.1021/acs.jcim.5c00603 (assay groups as features; summary only).
- Ranking within an assay to cancel assay offsets: DONE BEFORE. Matsumoto, Miyao & Funatsu 2021,
  ACS Omega 6:11964, doi 10.1021/acsomega.1c00463; ActFound (Feng et al. 2024, Nat. Mach. Intell.
  6:962, doi 10.1038/s42256-024-00876-w).
- Choosing compounds to re-measure: Bellamy, Rehim, Orhobor & King 2022, JCIM 62:3970,
  doi 10.1021/acs.jcim.2c00602 (retest policy). Sheng et al.'s repeated labelling applied directly:
  none found.
- Quantile normalisation across assays: none found (3 searches).

### Psychometrics and noise estimation
- Ceilings on model performance from assay error: DONE BEFORE. Brown, Muchmore & Hajduk 2009
  (Drug Discov. Today, doi 10.1016/j.drudis.2009.01.012); Kramer et al. 2012; Kolmar & Grulke 2021;
  Crusius, Cipcigan & Biggin 2025. ICC, Spearman–Brown or correction for attenuation by name: not
  found.
- Gamma test as a noise estimate on molecules: none found. Used once for variable selection,
  Marvuglia et al. 2014 (doi 10.1007/978-3-319-09150-1_47).
- Noise from near-identical molecule pairs: matched-pair noise studies exist, Nelen et al. 2025
  (J. Cheminform. 17:8, doi 10.1186/s13321-025-00956-y) and Kramer et al. 2014 (J. Med. Chem.
  57:3786, doi 10.1021/jm500317a).
- Regressing squared out-of-fold error on 1/(number of measurements): none found in any field.
  Measured on herg_ic50 (the only dataset with repeats that ran, 3 of 5 folds, file
  kirby_results_2026-10-08b/noise/n26_noise_variance_from_repeats/herg_ic50.json): it recovers
  added noise, but its slope on the real labels is negative in all 8 configurations.

## Citations in the code that point to the wrong paper
- lnl_methods.py:840 "Zhao et al. (2025)" = refs.bib zhao2025 (MANTRA, code models, one-off
  removal by loss-trajectory mixture). The code removes the worst 10% in three rounds.
- lnl_methods.py:929 "Zhang et al. (2025)" = refs.bib zhang2025 (additive, similarity-driven
  update with a clean subset). The code multiplies above-median-loss weights by 0.95 per epoch.
- lnl_methods.py:1842 Zheltonozhskii 2021: the code does perturbation consistency, not in that paper.
- lnl_methods.py:2019 Liu, Ong & Cai 2020: the code fits a GP, then a network to its residuals.
- noise_estimation.py:1434, 1496 "Jackknife+", "Wager-style": neither is computed.
