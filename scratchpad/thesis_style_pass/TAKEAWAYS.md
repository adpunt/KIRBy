# Thesis takeaways (for the abstract, Chapter 1 and Chapter 8)

Every number is copied from thesis.tex as of 2026-10-10.

## 1. The thesis in five sentences

1. Label noise in QSAR data is not Gaussian, and what it does to a model depends on the noise, the model, the representation and the dataset.
2. Model family decides robustness to noise more than representation does, and random forests were the most consistent model across four datasets.
3. Published methods for learning with noisy labels failed on molecular data. Those that compared molecules by distance failed worst, because most features of a standard representation carry nothing for the task.
4. We built the adaptive hybrid representation, which selects features from several representations for each dataset and model. It predicts better, gives better neighbors and keeps its lead when noise is handled.
5. With the hybrid and with methods we built or changed, we handled label noise, and which method works depends on the task, data, model and noise type.

## 2. Part one (Chapters 1 to 3): what noise does

- **Noise is not just Gaussian.** The shape of the distribution barely matters, but systematic noise does. Censoring lowered \aucnorm{} by 0.12 to 0.18 relative to Gaussian noise for the same models.
- **Model family matters more than representation for robustness.** Model family explains 47.2% of the variance in \aucnorm{}, representation 8.1% (QM9, Gaussian).
- **Representation acts through the model.** The pairing of model and representation explains 19.9% of the variance in \aucnorm{}, more than twice the 8.1% of representation alone. So choose them together.
- **Predicting well on clean labels does not mean robust.** On Caco-2, LightGBM keeps 26% of its R² at noise level 1.5, against 63% for RF.
- **No one-size-fits-all, but RF is solid.** RF ranks between first and eighth for \aucnorm{} on every dataset, representation and noise condition.
- **Conditional choices (Table tab:condition_choice).** Under Laplace, Student-t and outlier noise, prefer RF, then SVM. Under grouped-wider noise, prefer QRF. Under grouped-shifted noise, RF loses least of the models that predict well, and the GP loses most. NGBoost is the specialist under Gaussian noise on the HOMO–LUMO gap only.
- **Bayesian networks are not the fix.** They raised \aucnorm{} only by lowering clean R².
- **Uncertainty rises with noise but does not find the noisy labels.** Under censoring it points the wrong way. On QM9 with half the labels clipped, NGBoost's correlation runs from −0.84 to −0.80.

## 3. Part two (Chapters 4 to 7): narrowing, then widening

1. **LNL methods did not work well (Ch4).** All 20 configurations with the lowest retention used a distance. Neighbor-based weights gave negative mean recovery on every representation, −3.0% and −7.3% on ECFP4.
2. **Molecular similarity is why (Ch4).** The Spearman correlation between pair distance and label difference was below 0.10 in 44 of 70 combinations of dataset and representation. (A count; Ch6's pair is 0.0729 on hERG Ki.)
3. **The features are not right for similarity (Ch4, Ch5).** On hERG Ki, 1,815 of 2,048 ECFP4 features take one value in at least 95% of molecules.
4. **Fix the features with the hybrid (Ch5).** LogD with the SVM goes from 0.4930 to 0.4302 RMSE. The hybrid also beats the representation best in hindsight: Caco-2 with the DNN, 0.2980 to 0.2775.
5. **Similarity works better in the hybrid (Ch6).** On hERG Ki, distance against activity difference goes from 0.0729 to 0.2625 Spearman. Prediction from five neighbors goes from 0.6280 to 0.5822 RMSE.
6. **Noise handling works better (Ch7).** On LogD with outliers, the cleaned hybrid scores 0.5560 RMSE against 0.5915 for the cleaned single representation. Neighbors found in the hybrid took LogD under a scaffold-family shift to 0.6626, against 0.7535 with neighbors in the model's own space.

## 4. Noise handling: what we can do, and where

Noise was handled. The conditions say where, not whether.

- **Censoring (flag known):** censored EM took LogD from 0.6354 to 0.4795 RMSE, against 0.4314 before clipping. That is 76% of the lost RMSE.
- **Scaffold-family offsets (flag known):** on Caco-2, 0.3214 to 0.2796, against 0.2385 on clean labels.
- **Label flips (detector needed):** hERG, 0.8028 to 0.8375 AUROC, against 0.8179 for random deletion, on three folds.
- **Labels as delivered:** hERG Ki, 0.6380 to 0.5775 RMSE, against 0.6419 or above with shuffled scores. SMAC colloidal stability, 0.2740 to 0.4034 Spearman, against at most 0.2782 shuffled.

What each result depends on:
- **Noise type.** The forest residual detector falls below chance under censoring (LogD 0.1957 against 0.2369). A flag beats a detector.
- **Data.** Subtracting offsets that are not there hurts: Caco-2 as delivered, 0.2333 to 0.2669. On clean QM9, deletion hurt, 0.3550 to 0.3931.
- **Model.** Small-loss selection helped in 1 of 31 small-molecule conditions and 5 of 5 on Buchwald–Hartwig.
- **Task.** Detectors that predict a spread from features alone cannot see a flipped class.
- **Feature space.** The best method on the hybrid is rarely the best on the single representations.

## 5. Hybrids

- **Basic prediction:** better than the best single representation on every small-molecule dataset with both models, on every fold of the six regression datasets.
- **Noise:** the hybrid starts ahead and stays ahead through every cleaning on small molecules. Cleaning before the search helped: expression titer 0.3899 to 0.5619 Spearman.
- **Where it does not help:** small antibody sets (thermostability 0.6899 against 0.6901), where the search overstates itself by up to 0.1571 Spearman. Ensembles often win on antibodies (0.7854 against 0.7348). Recursive elimination beat it on hERG Ki with the DNN (0.5794 against 0.6112). Reactions were mixed. For detection alone, importance-weighted single features did better than the hybrid's.
- **Forward compatible:** a new embedding is one more representation searched. Its selected features beat random features in the same places for every embedding on Caco-2 and hERG Ki.

## 6. Original contributions

Noise:
1. NoiseInject, an open-source package for benchmarking noise robustness and uncertainty.
2. Seven noise conditions at a matched amount of noise, parameters taken from published measurement errors, including scaffold-grouped noise and censoring.
3. The variance decomposition of performance and robustness between model family, representation and their pairing. Chapter 1 states no earlier work has compared these for noise robustness.
4. Methods with no published source found (METHOD_PROVENANCE.md, bucket c): the cliff-aware gap; the gap over neighbor spread; cleaning labels before the feature search; putting activity-cliff molecules back after a cleanup.
5. Published methods with our own change: censored EM with any model; a random intercept per scaffold family on out-of-fold residuals; a searched neighbor space; scaffold-grouped out-of-fold scoring; regression versions of LNL methods.
6. Controls: QM9 with no noise added; shuffled detector scores; random deletion of the same size; the same action on undamaged molecules; shuffled per-molecule lifts.
7. The finding that a noise method's ranking depends on the feature space it runs in.

Hybrid:
8. The adaptive hybrid representation: features ranked within each representation, then a budget split among them.
9. A search over the selection settings for each dataset and model, saved as a blueprint that traces each feature to its representation.
10. Hybrids built for distance, at separate sites, and open to new embeddings.
11. Antibody representations built for this thesis: CDR and region readouts, mutation features, the change in embedding from the parent.

Caution: METHOD_PROVENANCE.md buckets are Claude's reading of the code. Several bucket-c methods came from AI-written reports, so claim only the four named in Chapter 8.

## 7. Research questions for Chapter 1, with answers

1. *How much of predictive performance and of noise robustness is explained by representation, and how much by model?* Model family explains 47.2% of robustness against 8.1% for representation. For R² under noise they matter about equally (26.6% and 29.0%).
2. *Does the type of noise matter, or only its amount?* The type matters. Shape barely does, but scaffold-shared offsets and censoring do the most damage.
3. *Do probabilistic models help under noise, and does their uncertainty find the noisy labels?* No to both. Mean uncertainty rises, but no per-molecule uncertainty finds corrupted labels outside censoring.
4. *Why do published noise-handling methods fail on QSAR data?* Those using a distance fail worst, because the distance is dominated by features irrelevant to the task.
5. *Can features be selected per dataset and model from several representations to make a better input?* Yes. The hybrid beats the best single representation on every small-molecule dataset.
6. *Does distance in a hybrid track activity better, and do neighbor-based methods improve?* Yes on all five small-molecule datasets. On antibodies and reactions it depends on the dataset.
7. *Can noisy labels be found and fixed, and does the hybrid help?* Yes, conditionally on noise type, data and model. The hybrid keeps its lead, and the method ranking changes with the feature space.

## For the author

- RF rank: Ch3 Conclusions say "first and eighth", Ch3 results "first and seventh" (Gaussian only). Check.
