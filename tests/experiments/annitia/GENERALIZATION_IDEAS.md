# Generalization mechanisms for ANNITIA fusion (running brainstorm)

The empirical pattern observed: **more variants → better generalization** (0.9112 → 0.9269 → 0.9278 → 0.9285 → 0.9294 → 0.9302). Each level added independent aggregation perspectives. We should keep pulling on this lever AND introduce mechanisms that improve generalization at each level individually.

## Categories of generalization mechanisms

### A. Aggregation-level robustness
1. **Rank-transform before averaging** ✓ done
2. **Median instead of mean** ✓ done (rank_median variant)
3. **Trimmed mean** ✓ done (drop top/bottom 20%)
4. **Reciprocal-rank fusion (RRF)** ✓ done
5. **Bootstrap aggregation of fusion**: fit fusion on N bootstrap samples of training, average across — reduces variance in the fusion weights
6. **Random-subset dropout fusion**: at each "inference", randomly drop k of the N variants and average — repeat M times, average the M predictions. Forces robustness to any single variant
7. **Bayesian model averaging**: weight each model by posterior probability given training data
8. **Constrained-positive meta-learner**: NNLS instead of unconstrained Coxnet; can't assign negative weights that overfit

### B. Variant-level diversity
9. **Diversity selection**: greedy pick low-correlation members ✓ done
10. **Source-stratified fusion**: per-source pre-fusion → inter-source fusion ✓ done (`refuse_hierarchical.py`)
11. **Quality-stratified weighting**: weight sources by their best CV C ✓ done
12. **Subjective gating on pretraining sources**: drop epic fails, boost epic wins ✓ done
13. **Disagreement-based dropout**: when ensemble disagrees a lot on a patient, downweight that prediction
14. **Cross-endpoint stacking** ✓ done

### C. Patient-level routing
15. **Anomaly-bucket routing** ✓ done (under-performed)
16. **Disagreement-bucket routing** ✓ done (under-performed)
17. **Cluster-based routing** ✓ done
18. **Soft anomaly blend** ✓ done
19. **Specialist override** for top-X% anomalous ✓ done — gave a small win
20. **Soft routing via gating network**: train a small network to predict per-patient model weights based on raw features (more powerful than discrete buckets but harder to validate)
21. **Per-time-bin routing**: different model for different event-time horizons (early-event vs late-event patients have different feature → outcome relationships)
22. **Cohort-specific specialists**: (e.g., diabetes vs no-diabetes patients use different models)

### D. Calibration / regularization
23. **Mid-alpha Coxnet** instead of smallest-alpha (which is least regularized) ✓ done in refuse_honest
24. **Strong L2 on stack_meta**: increases robustness when many features
25. **Conformal prediction wrappers**: produce calibrated risk intervals from any base model
26. **Isotonic regression calibration**: reshape predictions to be probabilistically meaningful (concordance is rank-based but calibration helps stacking)
27. **Bin-wise discordance correction**: post-hoc adjust each variant by its discordance per bin

### E. Data augmentation / training-side
28. **Synthetic positive augmentation** ✓ done (synaug, GMM)
29. **Mixup on patient features** (interpolate between similar patients)
30. **Pseudo-labeling**: use val_test predictions as soft labels and do self-distillation
31. **Snapshot ensembling at training time**: use multiple training checkpoints
32. **Adversarial training**: oversample the patients models get wrong

### F. Cross-validation methodology
33. **Repeated K-fold across multiple seeds**: average per-seed OOFs to reduce variance estimate
34. **Outer/inner nested CV** for any learned weights ✓ done (nested 5-fold for stack_meta)
35. **Group-K-fold by patient cluster**: each fold has similar patient mix
36. **Hold-out fusion-test set**: 10% of training reserved for fusion-method-selection (closest analog to real test conditions)
37. **Bootstrap C-index confidence intervals**: report uncertainty on each variant's OOF C
38. **Per-fold C distribution**: surface high-variance fusion variants

### G. Knowledge distillation / smaller models
39. **Distillation**: train a smaller faster model on the ensemble's predictions (could improve test stability)
40. **Pseudo-label smoothing**: train student on soft labels from teacher ensemble

### H. Ensemble diversity tricks
41. **Bagging at the variant-creation level**: train each variant on a bootstrap sample of training rows
42. **Feature-bagging**: each variant uses a random subset of features
43. **Algorithm-bagging**: same view set, different algorithms, ensembled (already implicit in BE)
44. **Hyperparameter-bagging**: same algorithm, different hparams, ensembled (Round 5 will produce this)

## Top picks worth implementing tonight (autonomous)
- (E.31) **Multi-seed K-fold**: re-run base models with seed=43, 44, 45 and average OOF/val across seeds — improves stability *for free* if compute allows
- (D.26) **Isotonic calibration**: post-process each variant via isotonic regression to align prediction distributions before averaging
- (A.6) **Random-subset dropout fusion**: cheap to add, gives meta-meta robustness
- (F.36) **Hold-out fusion-test set**: 10% holdout for honest fusion comparison

## Low-hanging fruit for tomorrow
- (B.10/H.41-44) Bagged-fusion: M=10 bootstrap samples × full pipeline, average across
- (C.20) Learned per-patient gating with raw features
- (D.27) Bin-wise discordance correction
- (G.39) Distillation onto a smaller model

## Actions to run autonomously tonight
- [ ] Add refuse_hierarchical.py to auto_chain
- [ ] Add bootstrap-fusion variant to auto_chain
- [ ] Add isotonic-calibration variant to auto_chain
- [ ] Add hold-out-fusion-test methodology to auto_chain
